//! `arena` — run a gauntlet from the command line.
//!
//! ```text
//! cargo run --release -p virus-arena --bin arena -- \
//!     --a ab-enhanced --a-budget nodes:60000 \
//!     --b ab-plain    --b-budget nodes:60000 \
//!     --games 400 --seed 11 --threads 4
//! ```
//!
//! A net-vs-net run names an artifact per side:
//!
//! ```text
//! cargo run --release -p virus-arena --bin arena -- \
//!     --a mcts --a-net artifacts/champions/gen-7.json \
//!     --b mcts --b-net artifacts/champions/gen-6.json \
//!     --budget nodes:400 --games 400 --seed 20260814 --threads 4
//! ```
//!
//! Arguments are parsed by hand rather than with a CLI crate. The workspace has
//! no argument-parsing dependency and this binary has seventeen flags; adding one
//! (and its transitive tree) to a crate whose whole job is to be a trustworthy
//! measuring instrument is a poor trade.
//!
//! Every flag is listed by `--help`. An unknown flag is an error, never a
//! silent default: a typo that quietly halves the node budget would produce a
//! plausible-looking number that means nothing, which is precisely the failure
//! mode this crate exists to prevent.

use std::process::ExitCode;
use virus_arena::engine::{Budget, SideSpec, SpecError, DEFAULT_NODE_LIMIT};
use virus_arena::gauntlet::{
    run_with_nets, GauntletConfig, SideNets, Termination, DEFAULT_EPSILON, DEFAULT_EXPLORE_TURNS,
    DEFAULT_MAX_TURNS,
};
use virus_mcts::PolicyValueNet;

const USAGE: &str = "\
arena — the virus-game gauntlet harness

USAGE:
    arena [OPTIONS]

SIDES:
    --a <SPEC>            side A engine [default: ab-enhanced]
    --b <SPEC>            side B engine [default: ab-plain]
    --a-budget <BUDGET>   side A per-action budget [default: --budget]
    --b-budget <BUDGET>   side B per-action budget [default: --budget]
    --budget <BUDGET>     budget for both sides [default: nodes:60000]

    SPEC   is greedy | ab-plain | ab-enhanced | mcts | mcts:<artifact.json>
    BUDGET is nodes:<n> | depth:<d> | ms:<milliseconds>
           nodes: is deterministic and is the only mode the determinism gate
           accepts.  ms: is wall-clock and reproduces nothing by construction.
           For an MCTS side, nodes:<n> means n simulations.

RUN:
    --games <N>           games to play, rounded up to even [default: 8]
    --seed <N>            base seed [default: 1]
    --rows <N> --cols <N> board size [default: 12x12]
    --threads <N>         worker threads [default: 1]
    --max-turns <N>       turn cap; a capped game is a draw [default: 100]
    --eps <P>             opening randomisation probability [default: 0.15]
    --explore-turns <N>   opening window, in turns [default: 8]
    --net <PATH>          net artifact for both MCTS sides
                          [default: artifacts/mcts_champion.json]
    --a-net <PATH>        side A's net artifact [default: --net]
    --b-net <PATH>        side B's net artifact [default: --net]

    Two different artifacts in one run is the net-vs-net gauntlet:

        arena --a mcts --a-net artifacts/champions/gen-7.json \\
              --b mcts --b-net artifacts/champions/gen-6.json \\
              --budget nodes:400 --games 400 --seed 20260814 --threads 4

    `--a mcts:<path>` is the same thing spelled inside the SPEC.  Giving both
    for one side is an error unless they name the same file: a run whose two
    flags disagree about which artifact played is a number nobody can read.

OUTPUT:
    --per-game            print one line per game before the summary
    -h, --help            this text

Pooling two runs?  Space their --seed values far apart.  See the `rng` module:
runs seeded 1 and 2 used to replay each other's openings and their 'independent'
results were correlated (nnue-trainer-riy).
";

fn main() -> ExitCode {
    match real_main() {
        Ok(code) => code,
        Err(error) => {
            eprintln!("arena: {error}");
            ExitCode::FAILURE
        }
    }
}

fn real_main() -> Result<ExitCode, SpecError> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|arg| arg == "-h" || arg == "--help") {
        print!("{USAGE}");
        return Ok(ExitCode::SUCCESS);
    }
    let options = Options::parse(&args)?;

    let (config, paths) = options.to_config()?;

    // One `PolicyValueNet` per *distinct* path, never per side. When both arms
    // name the same artifact — which every single-net run does — side B borrows
    // side A's, so the memory profile is the one loaded net it has always been.
    let net_a = match &paths.a {
        Some(path) => Some(load_net(path)?),
        None => None,
    };
    let net_b_owned = match &paths.b {
        Some(path) if paths.a.as_deref() != Some(path.as_str()) => Some(load_net(path)?),
        _ => None,
    };
    let nets = SideNets::new(
        net_a.as_ref(),
        match (&paths.b, &net_b_owned) {
            (None, _) => None,
            (Some(_), Some(net)) => Some(net),
            // Same path as side A: borrow the one already loaded.
            (Some(_), None) => net_a.as_ref(),
        },
    );
    if let (Some(a), Some(b)) = (&paths.a, &paths.b) {
        if a != b {
            // The side names carry only the file *stem*, so two artifacts in
            // different directories can print the same name. Say the full paths
            // once, or a net-vs-net report is unreproducible.
            eprintln!("arena: side A net {a}");
            eprintln!("arena: side B net {b}");
        }
    }

    eprintln!(
        "arena: {} vs {}, {} games, seed {}, {}x{}, {} thread(s)",
        config.side_a.name(),
        config.side_b.name(),
        config.even_games(),
        config.seed,
        config.rows,
        config.cols,
        config.threads,
    );
    let result = run_with_nets(&config, nets)?;

    if options.per_game {
        for game in &result.games {
            println!(
                "game {:>4} seat_a={} winner={} turns={:>3} plies={:>3} {:?} work_a={} work_b={}",
                game.index,
                if game.a_is_p1 { 1 } else { 2 },
                game.winner,
                game.turns,
                game.plies,
                game.termination,
                game.work_a,
                game.work_b,
            );
        }
    }
    println!("{}", result.summary);

    // A run full of stalls has produced a clean-looking tally out of broken
    // games. Say so loudly rather than letting the number stand.
    let stalled = result
        .games
        .iter()
        .filter(|game| game.termination == Termination::Stalled)
        .count();
    if stalled > 0 {
        eprintln!("arena: WARNING — {stalled} game(s) stalled; this tally is not trustworthy");
        return Ok(ExitCode::FAILURE);
    }
    Ok(ExitCode::SUCCESS)
}

/// Raw flags, before they become a [`GauntletConfig`].
#[derive(Debug)]
struct Options {
    side_a: String,
    side_b: String,
    budget: String,
    budget_a: Option<String>,
    budget_b: Option<String>,
    games: u32,
    seed: u64,
    rows: usize,
    cols: usize,
    threads: usize,
    max_turns: u32,
    epsilon: f64,
    explore_turns: u32,
    net: String,
    net_a: Option<String>,
    net_b: Option<String>,
    per_game: bool,
}

/// The artifact each side will actually play, after `mcts:<path>`, `--a-net`
/// and `--net` have been reconciled.
///
/// `None` means the side's engine needs no artifact — not "use the default".
/// Keeping those two apart is what stops a `greedy`-vs-`ab` run from loading
/// 700 KB of net it will never consult.
#[derive(Debug)]
struct NetPaths {
    a: Option<String>,
    b: Option<String>,
}

/// Loads one artifact, naming the file in the error.
fn load_net(path: &str) -> Result<PolicyValueNet, SpecError> {
    PolicyValueNet::load(path)
        .map_err(|error| SpecError(format!("could not load the net artifact {path}: {error}")))
}

/// Reconciles the two ways a side can name its artifact.
///
/// `mcts:<path>` inside the SPEC and `--a-net` are both explicit, so when they
/// disagree there is no defensible winner — the run is refused. Silently
/// preferring one would let a stale `--net` in a script override the path the
/// operator just typed, and the resulting tally would carry the wrong artifact's
/// name.
fn resolve_net(
    side: &SideSpec,
    which: &str,
    flag: Option<&String>,
    default: &str,
) -> Result<Option<String>, SpecError> {
    if !side.needs_net() {
        return Ok(None);
    }
    match (side.net.as_deref(), flag) {
        (Some(spec), Some(flag)) if spec != flag => Err(SpecError(format!(
            "side {} names two different artifacts: --{which} mcts:{spec} and --{which}-net \
             {flag}; give one or the other",
            which.to_uppercase()
        ))),
        (Some(spec), _) => Ok(Some(spec.to_owned())),
        (None, Some(flag)) => Ok(Some(flag.clone())),
        (None, None) => Ok(Some(default.to_owned())),
    }
}

impl Default for Options {
    fn default() -> Options {
        Options {
            side_a: "ab-enhanced".to_owned(),
            side_b: "ab-plain".to_owned(),
            budget: format!("nodes:{DEFAULT_NODE_LIMIT}"),
            budget_a: None,
            budget_b: None,
            games: 8,
            seed: 1,
            rows: 12,
            cols: 12,
            threads: 1,
            max_turns: DEFAULT_MAX_TURNS,
            epsilon: DEFAULT_EPSILON,
            explore_turns: DEFAULT_EXPLORE_TURNS,
            net: "artifacts/mcts_champion.json".to_owned(),
            net_a: None,
            net_b: None,
            per_game: false,
        }
    }
}

impl Options {
    fn parse(args: &[String]) -> Result<Options, SpecError> {
        let mut options = Options::default();
        let mut index = 0;
        while index < args.len() {
            let flag = args[index].clone();
            // Flags that take a value consume the next argument; `value`
            // advances `index` past it and errors rather than defaulting when
            // the value is missing.
            let mut value = || -> Result<String, SpecError> {
                index += 1;
                args.get(index)
                    .cloned()
                    .ok_or_else(|| SpecError(format!("{flag} needs a value")))
            };
            match flag.as_str() {
                "--a" => options.side_a = value()?,
                "--b" => options.side_b = value()?,
                "--budget" => options.budget = value()?,
                "--a-budget" => options.budget_a = Some(value()?),
                "--b-budget" => options.budget_b = Some(value()?),
                "--games" => options.games = number(&flag, &value()?)?,
                "--seed" => options.seed = number(&flag, &value()?)?,
                "--rows" => options.rows = number(&flag, &value()?)?,
                "--cols" => options.cols = number(&flag, &value()?)?,
                "--threads" => options.threads = number(&flag, &value()?)?,
                "--max-turns" => options.max_turns = number(&flag, &value()?)?,
                "--eps" => options.epsilon = number(&flag, &value()?)?,
                "--explore-turns" => options.explore_turns = number(&flag, &value()?)?,
                "--net" => options.net = value()?,
                "--a-net" => options.net_a = Some(value()?),
                "--b-net" => options.net_b = Some(value()?),
                "--per-game" => options.per_game = true,
                other => {
                    return Err(SpecError(format!(
                        "unknown flag {other:?}; run with --help"
                    )))
                }
            }
            index += 1;
        }
        Ok(options)
    }

    fn to_config(&self) -> Result<(GauntletConfig, NetPaths), SpecError> {
        let shared = Budget::parse(&self.budget)?;
        let budget_a = match &self.budget_a {
            Some(text) => Budget::parse(text)?,
            None => shared,
        };
        let budget_b = match &self.budget_b {
            Some(text) => Budget::parse(text)?,
            None => shared,
        };
        let mut config = GauntletConfig {
            side_a: SideSpec::parse(&self.side_a, budget_a)?,
            side_b: SideSpec::parse(&self.side_b, budget_b)?,
            games: self.games,
            seed: self.seed,
            rows: self.rows,
            cols: self.cols,
            max_turns: self.max_turns,
            epsilon: self.epsilon,
            explore_turns: self.explore_turns,
            threads: self.threads,
        };
        config.validate()?;
        // Per side: an explicit `mcts:<path>`, else `--a-net`/`--b-net`, else
        // `--net`. Two *different* artifacts is the net-vs-net run and is now
        // supported outright — `SideNets` carries one loaded net per arm.
        let paths = NetPaths {
            a: resolve_net(&config.side_a, "a", self.net_a.as_ref(), &self.net)?,
            b: resolve_net(&config.side_b, "b", self.net_b.as_ref(), &self.net)?,
        };
        // Write the resolved paths back into the specs so the report names the
        // artifacts that actually played. `SideSpec::name` renders `mcts[stem]`
        // from this field, and a net-vs-net summary reading "mcts vs mcts" is a
        // row nobody can reproduce a year later — which is the whole reason the
        // stem is in the name at all.
        config.side_a.net = paths.a.clone();
        config.side_b.net = paths.b.clone();
        Ok((config, paths))
    }
}

fn number<T: std::str::FromStr>(flag: &str, raw: &str) -> Result<T, SpecError> {
    raw.parse()
        .map_err(|_| SpecError(format!("{flag}: cannot parse {raw:?}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<(GauntletConfig, NetPaths), SpecError> {
        let args: Vec<String> = args.iter().map(|arg| (*arg).to_owned()).collect();
        Options::parse(&args)?.to_config()
    }

    /// The old default: one artifact, named once, used by whichever sides need
    /// it. This is the path every existing script takes and it must not have
    /// moved.
    #[test]
    fn one_net_flag_serves_both_mcts_sides() {
        let (config, paths) = parse(&["--a", "mcts", "--b", "mcts", "--budget", "nodes:8"])
            .expect("two mcts sides on the default artifact");
        assert_eq!(paths.a.as_deref(), Some("artifacts/mcts_champion.json"));
        assert_eq!(paths.b, paths.a);
        assert_eq!(config.side_a.name(), "mcts[mcts_champion]:n8");

        let (_, paths) = parse(&["--a", "mcts", "--b", "mcts", "--net", "x.json"]).expect("--net");
        assert_eq!(paths.a.as_deref(), Some("x.json"));
        assert_eq!(paths.b.as_deref(), Some("x.json"));
    }

    /// A side that needs no artifact must not be handed one: `None` here is
    /// what stops a greedy-vs-alpha-beta run loading 700 KB it never reads.
    #[test]
    fn a_side_that_needs_no_net_resolves_to_none() {
        let (_, paths) = parse(&["--a", "greedy", "--b", "ab-enhanced"]).expect("no nets");
        assert_eq!(paths.a, None);
        assert_eq!(paths.b, None);

        let (_, paths) =
            parse(&["--a", "mcts", "--b", "greedy", "--budget", "nodes:8"]).expect("one net side");
        assert_eq!(paths.a.as_deref(), Some("artifacts/mcts_champion.json"));
        assert_eq!(paths.b, None);
    }

    /// The bead: two different artifacts in one run, which used to be refused.
    #[test]
    fn two_different_artifacts_resolve_per_side() {
        let (config, paths) = parse(&[
            "--a",
            "mcts",
            "--a-net",
            "gen7.json",
            "--b",
            "mcts",
            "--b-net",
            "gen6.json",
            "--budget",
            "nodes:8",
        ])
        .expect("net-vs-net");
        assert_eq!(paths.a.as_deref(), Some("gen7.json"));
        assert_eq!(paths.b.as_deref(), Some("gen6.json"));
        // The report has to name both, or the row cannot be reproduced.
        assert_eq!(config.side_a.name(), "mcts[gen7]:n8");
        assert_eq!(config.side_b.name(), "mcts[gen6]:n8");
    }

    /// `--a-net` and `--b-net` are per side, so one of them alone leaves the
    /// other on `--net`.
    #[test]
    fn a_per_side_flag_leaves_the_other_side_on_the_shared_default() {
        let (_, paths) = parse(&[
            "--a",
            "mcts",
            "--a-net",
            "candidate.json",
            "--b",
            "mcts",
            "--net",
            "champion.json",
            "--budget",
            "nodes:8",
        ])
        .expect("mixed");
        assert_eq!(paths.a.as_deref(), Some("candidate.json"));
        assert_eq!(paths.b.as_deref(), Some("champion.json"));
    }

    /// `mcts:<path>` is the same statement spelled inside the SPEC, and it wins
    /// over the shared `--net` exactly as it did before.
    #[test]
    fn a_spec_path_overrides_the_shared_net_flag() {
        let (_, paths) = parse(&[
            "--a",
            "mcts:inline.json",
            "--b",
            "mcts",
            "--net",
            "shared.json",
            "--budget",
            "nodes:8",
        ])
        .expect("inline path");
        assert_eq!(paths.a.as_deref(), Some("inline.json"));
        assert_eq!(paths.b.as_deref(), Some("shared.json"));

        // Agreeing with `--a-net` is fine; there is nothing to disambiguate.
        let (_, paths) = parse(&[
            "--a",
            "mcts:same.json",
            "--a-net",
            "same.json",
            "--b",
            "greedy",
            "--budget",
            "nodes:8",
        ])
        .expect("agreeing paths");
        assert_eq!(paths.a.as_deref(), Some("same.json"));
    }

    /// Two explicit, disagreeing spellings for one side have no defensible
    /// winner. Picking one silently is how a stale `--a-net` in a script
    /// overrides the path the operator just typed and the tally comes back
    /// under the wrong artifact's name.
    #[test]
    fn a_side_named_twice_with_two_paths_is_refused() {
        let error = parse(&[
            "--a",
            "mcts:one.json",
            "--a-net",
            "other.json",
            "--b",
            "greedy",
            "--budget",
            "nodes:8",
        ])
        .expect_err("conflicting paths for side A");
        assert!(error.0.contains("one.json"), "{error}");
        assert!(error.0.contains("other.json"), "{error}");

        let error = parse(&[
            "--a",
            "greedy",
            "--b",
            "mcts:one.json",
            "--b-net",
            "other.json",
            "--budget",
            "nodes:8",
        ])
        .expect_err("conflicting paths for side B");
        assert!(error.0.contains("--b-net"), "{error}");
    }

    /// An unknown flag is an error, never a silent default — the new flags do
    /// not change that.
    #[test]
    fn the_new_flags_still_need_values() {
        assert!(parse(&["--a-net"]).is_err());
        assert!(parse(&["--b-net"]).is_err());
        assert!(parse(&["--a-nets", "x.json"]).is_err());
    }
}
