//! `arena` — run a gauntlet from the command line.
//!
//! ```text
//! cargo run --release -p virus-arena --bin arena -- \
//!     --a ab-enhanced --a-budget nodes:60000 \
//!     --b ab-plain    --b-budget nodes:60000 \
//!     --games 400 --seed 11 --threads 4
//! ```
//!
//! Arguments are parsed by hand rather than with a CLI crate. The workspace has
//! no argument-parsing dependency and this binary has eleven flags; adding one
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
    run, GauntletConfig, Termination, DEFAULT_EPSILON, DEFAULT_EXPLORE_TURNS, DEFAULT_MAX_TURNS,
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
    --net <PATH>          net artifact for MCTS sides
                          [default: artifacts/mcts_champion.json]

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

    let (config, net_path) = options.to_config()?;
    let net = if config.side_a.needs_net() || config.side_b.needs_net() {
        Some(PolicyValueNet::load(&net_path).map_err(|error| {
            SpecError(format!(
                "could not load the net artifact {net_path}: {error}"
            ))
        })?)
    } else {
        None
    };

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
    let result = run(&config, net.as_ref())?;

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
    per_game: bool,
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

    fn to_config(&self) -> Result<(GauntletConfig, String), SpecError> {
        let shared = Budget::parse(&self.budget)?;
        let budget_a = match &self.budget_a {
            Some(text) => Budget::parse(text)?,
            None => shared,
        };
        let budget_b = match &self.budget_b {
            Some(text) => Budget::parse(text)?,
            None => shared,
        };
        let config = GauntletConfig {
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
        // An explicit `mcts:<path>` on either side overrides `--net`.
        //
        // Two *different* artifacts in one run is refused rather than
        // approximated. The harness shares a single loaded net across every
        // game and thread (loading the 700 KB champion per game dominated the
        // first version's run time), so honouring two paths needs a second
        // loaded net threaded through `engine::build` — real work, not a flag.
        // Silently playing one artifact against itself would report a tidy
        // 50/50 for a comparison that never happened, which is exactly the
        // class of quiet wrongness this crate exists to prevent.
        let net = match (&config.side_a.net, &config.side_b.net) {
            (Some(a), Some(b)) if a != b => {
                return Err(SpecError(format!(
                    "two different net artifacts in one run is not supported yet ({a} vs {b}); \
                     net-vs-net gauntlets need a second loaded artifact threaded through the \
                     sides"
                )))
            }
            (Some(path), _) | (_, Some(path)) => path.clone(),
            (None, None) => self.net.clone(),
        };
        Ok((config, net))
    }
}

fn number<T: std::str::FromStr>(flag: &str, raw: &str) -> Result<T, SpecError> {
    raw.parse()
        .map_err(|_| SpecError(format!("{flag}: cannot parse {raw:?}")))
}
