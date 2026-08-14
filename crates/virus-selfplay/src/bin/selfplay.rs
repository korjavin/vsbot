//! `selfplay` — generate AlphaZero training rows from MCTS self-play.
//!
//! ```text
//! cargo run --release -p virus-selfplay --bin selfplay -- \
//!     --net artifacts/mcts_champion.json --out gen1/selfplay_shard_0.jsonl \
//!     --games 24 --sims 192 --shard 0 --shards 4 --seed 11011
//! ```
//!
//! The flag names are the ones `trainer/README.md`'s generation recipe already
//! writes, so that recipe runs against this binary unchanged.
//!
//! # Sharding is across processes, not threads
//!
//! There is no `--threads`. A shard is one process playing one game at a time,
//! and parallelism comes from running several shards at once:
//!
//! ```text
//! for i in 0 1; do
//!   selfplay --shard "$i" --shards 2 --seed 11011 --games 24 \
//!            --out "gen1/selfplay_shard_$i.jsonl" &
//! done; wait
//! cat gen1/selfplay_shard_*.jsonl > gen1/selfplay.jsonl
//! ```
//!
//! That is Java's arrangement and it is what makes the determinism criterion
//! checkable: with one game per process at a time, a shard's output is a pure
//! function of `(seed, shard_idx, shard_count)` with no interleaving to order.
//! An in-process thread pool would have to reorder rows back into game order to
//! stay byte-identical, which is the same file for more machinery — and the
//! searcher is single-threaded by construction anyway.
//!
//! Arguments are parsed by hand: the workspace carries no argument-parsing
//! dependency, and an unknown flag is an error rather than a silent default,
//! for `virus-arena`'s reason — a typo that quietly halved `--sims` would
//! produce a plausible-looking generation that trains a worse net.

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use virus_mcts::{PolicyValueNet, DEFAULT_GUMBEL_M};
use virus_selfplay::{
    generate, GameConfig, GumbelOptions, GumbelPv, Options, ResignConfig, DEFAULT_MAX_TURNS,
    DEFAULT_SIMS,
};

const USAGE: &str = "\
selfplay — MCTS self-play generator emitting SelfPlayMcts training rows

USAGE:
    selfplay [OPTIONS]

RUN:
    --games <N>          global game count; this shard plays g where
                         g % shards == shard [default: 8]
    --sims <N>           simulations per action [default: 192]
    --seed <N>           base seed; the game id embeds it [default: 11]
    --shard <I>          this shard's index [default: 0]
    --shards <N>         how many shards the run is split across [default: 1]
    --max-turns <N>      turn cap; a capped game keeps its territory
                         verdict, it is NOT a draw [default: 100]

NET:
    --net <PATH>         policy/value artifact
                         [default: artifacts/mcts_champion.json]
    --no-net             search with the hand-tuned leaf instead of a net.
                         For smoke tests only: the rows are legal but the
                         play is not the champion's, so a net trained on
                         them is not on the ladder.
    --cpuct <F>          PUCT exploration constant [default: 1.5]
    --value-scale <F>    hand-tuned leaf tanh divisor [default: 12000]
    --batch <N>          leaves per batched net evaluation [default: 8]

RESIGN (off unless --resign-threshold / RESIGN_THRESHOLD is given):
    --resign-threshold <F>     resign a side once the absolute-frame root
                               value has been beyond F against it for
                               --resign-plies of its own plies. This flag is
                               the on switch [suggested: 0.95]
    --resign-plies <N>         consecutive hopeless plies BY THE RESIGNING
                               SIDE; the opponent's plies in between do not
                               reset the count [default: 8]
    --resign-control-frac <F>  share of games that ignore the rule and play
                               out, so the false-resign rate is measurable
                               [default: 0.1]

    Each has an env equivalent (RESIGN_THRESHOLD, RESIGN_PLIES,
    RESIGN_CONTROL_FRAC); the flag wins when both are set. The other two are
    an error without a threshold rather than a silent no-op — a generation
    that ran RESIGN_PLIES=4 and quietly resigned nothing is the failure this
    binary refuses unknown flags to avoid.

GUMBEL ROOT SELECTION (off unless --gumbel / GUMBEL=1 is given):
    --gumbel                   replace PUCT + Dirichlet at the root with
                               Gumbel top-m + sequential halving. This flag
                               is the on switch
    --gumbel-m <N>             candidate-set size for the top-m draw
                               [default: 16]
    --gumbel-pv <raw|improved> what the emitted rows put in `pv`
                               [default: raw]

      raw       the literal root visit counts. Sequential halving spends the
                budget on m candidates and then on the finalists, so this is
                a SPARSE target (~m non-zero of ~34) and it discards every
                completed-Q value the recipe computed.
      improved  the completed-Q improved policy, quantised to integer
                pseudo-counts summing to the search's visit total. Dense and
                schema-valid, but NOT literal visit counts.

    Neither is lossless -- the row schema has one target field and Gumbel
    produces two different things. See the crate docs before choosing.

    Env equivalents: GUMBEL=1, GUMBEL_M, GUMBEL_PV; the flag wins when both
    are set. --gumbel-m/--gumbel-pv without the on switch are an error, not a
    silent no-op, for the same reason the resign flags are.

OUTPUT:
    --out <PATH>         JSONL destination [default: - for stdout]
    -h, --help           this text

In the default (PUCT) arm, Dirichlet root noise is always on and temperature-1
visit sampling always covers the first 21 plies: this binary only generates
self-play, and both are what makes self-play data worth training on. The Gumbel
arm replaces both -- its own draw is the exploration.

Resignation is OFF by default and must stay off until a measurement run shows
a false-resign rate under 5 %: it trades a correct label for a shorter game,
and the control arm is the only thing that says which way that trade went.

Gumbel is OFF by default for the mirror reason: it changes the training target,
and it ships only behind a fixed-sims gauntlet plus a policy-target measurement.
";

struct Args {
    options: Options,
    net: Option<PathBuf>,
    out: Option<PathBuf>,
}

/// Parses a flag's value **into the field's own type**, so a value too large
/// for that type is an error rather than a wrap.
///
/// Going through `u64` and casting with `as` would make `--sims 4294967297`
/// mean `--sims 1`: a generation that looks like it ran at the requested
/// strength, took the expected wall clock for one simulation per action, and
/// produced rows that pass every contract check while being worthless. That is
/// exactly the class of silent misconfiguration this binary refuses unknown
/// flags to avoid, so it must not reintroduce it in the numbers themselves.
fn number<T>(text: &str, what: &str) -> Result<T, String>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    text.parse::<T>()
        .map_err(|error| format!("{what}: {error} (got {text:?})"))
}

/// A resign setting from the environment, or `None` when the variable is unset
/// or empty.
///
/// An empty value reads as unset so that `RESIGN_THRESHOLD= selfplay ...` turns
/// the feature off rather than failing to parse — the shape a generation script
/// writes when it wants to clear an inherited setting.
fn env_number<T>(name: &str) -> Result<Option<T>, String>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    match std::env::var(name) {
        Ok(text) if !text.trim().is_empty() => number(text.trim(), name).map(Some),
        _ => Ok(None),
    }
}

/// A boolean from the environment: `1`/`true`/`on`/`yes` and their negatives.
///
/// Unset or empty reads as `None` so `GUMBEL= selfplay ...` clears an inherited
/// setting, the same shape [`env_number`] accepts.
fn env_bool(name: &str) -> Result<Option<bool>, String> {
    let Ok(text) = std::env::var(name) else {
        return Ok(None);
    };
    match text.trim() {
        "" => Ok(None),
        "1" | "true" | "on" | "yes" => Ok(Some(true)),
        "0" | "false" | "off" | "no" => Ok(Some(false)),
        other => Err(format!("{name} wants a boolean, got {other:?}")),
    }
}

/// Gumbel settings as the command line and environment left them: `on` is the
/// switch, the other two only refine it.
#[derive(Default)]
struct GumbelArgs {
    on: Option<bool>,
    m: Option<u16>,
    pv: Option<GumbelPv>,
}

impl GumbelArgs {
    /// Environment first, so a flag of the same name can overwrite it.
    fn from_env() -> Result<GumbelArgs, String> {
        Ok(GumbelArgs {
            on: env_bool("GUMBEL")?,
            m: env_number("GUMBEL_M")?,
            pv: match std::env::var("GUMBEL_PV") {
                Ok(text) if !text.trim().is_empty() => {
                    Some(GumbelPv::parse(&text).map_err(|error| format!("GUMBEL_PV: {error}"))?)
                }
                _ => None,
            },
        })
    }

    fn resolve(self) -> Result<Option<GumbelOptions>, String> {
        if self.on != Some(true) {
            if self.m.is_some() || self.pv.is_some() {
                return Err("--gumbel-m/--gumbel-pv (or GUMBEL_M/GUMBEL_PV) were set \
                            without --gumbel, so Gumbel root selection would have \
                            stayed off. Pass --gumbel to turn it on."
                    .to_owned());
            }
            return Ok(None);
        }
        let options = GumbelOptions {
            m: self.m.unwrap_or(DEFAULT_GUMBEL_M),
            pv: self.pv.unwrap_or_default(),
        };
        if options.m < 2 {
            return Err(format!(
                "--gumbel-m must be at least 2; {} candidates is not a choice",
                options.m
            ));
        }
        Ok(Some(options))
    }
}

/// Resign settings as the command line and environment left them: `threshold`
/// present is the on switch, the other two only refine it.
#[derive(Default)]
struct ResignArgs {
    threshold: Option<f64>,
    plies: Option<u32>,
    control_frac: Option<f64>,
}

impl ResignArgs {
    /// Environment first, so a flag of the same name can overwrite it.
    fn from_env() -> Result<ResignArgs, String> {
        Ok(ResignArgs {
            threshold: env_number("RESIGN_THRESHOLD")?,
            plies: env_number("RESIGN_PLIES")?,
            control_frac: env_number("RESIGN_CONTROL_FRAC")?,
        })
    }

    fn resolve(self) -> Result<Option<ResignConfig>, String> {
        let Some(threshold) = self.threshold else {
            if self.plies.is_some() || self.control_frac.is_some() {
                return Err("--resign-plies/--resign-control-frac (or RESIGN_PLIES/\
                            RESIGN_CONTROL_FRAC) were set without a resign threshold, \
                            so resignation would have stayed off. Pass \
                            --resign-threshold 0.95 to turn it on."
                    .to_owned());
            }
            return Ok(None);
        };
        let defaults = ResignConfig::default();
        let config = ResignConfig {
            threshold,
            plies: self.plies.unwrap_or(defaults.plies),
            control_frac: self.control_frac.unwrap_or(defaults.control_frac),
        };
        if !(0.0..=1.0).contains(&config.threshold) || config.threshold == 0.0 {
            return Err(format!(
                "--resign-threshold must be in (0, 1], got {}",
                config.threshold
            ));
        }
        if config.plies == 0 {
            return Err("--resign-plies must be at least 1".to_owned());
        }
        if !(0.0..=1.0).contains(&config.control_frac) {
            return Err(format!(
                "--resign-control-frac must be in [0, 1], got {}",
                config.control_frac
            ));
        }
        Ok(Some(config))
    }
}

fn parse() -> Result<Option<Args>, String> {
    let mut args = Args {
        options: Options::default(),
        net: Some(PathBuf::from("artifacts/mcts_champion.json")),
        out: None,
    };
    args.options.game = GameConfig {
        sims: DEFAULT_SIMS,
        max_turns: DEFAULT_MAX_TURNS,
        ..GameConfig::default()
    };
    let mut resign = ResignArgs::from_env()?;
    let mut gumbel = GumbelArgs::from_env()?;

    let mut raw = std::env::args().skip(1);
    while let Some(flag) = raw.next() {
        let mut value = || raw.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "-h" | "--help" => {
                print!("{USAGE}");
                return Ok(None);
            }
            "--games" => args.options.games = number(&value()?, "--games")?,
            "--seed" => args.options.seed = number(&value()?, "--seed")?,
            "--shard" => args.options.shard_idx = number(&value()?, "--shard")?,
            "--shards" => args.options.shard_count = number(&value()?, "--shards")?,
            "--sims" => args.options.game.sims = number(&value()?, "--sims")?,
            "--max-turns" => args.options.game.max_turns = number(&value()?, "--max-turns")?,
            "--batch" => args.options.game.batch_size = number(&value()?, "--batch")?,
            "--cpuct" => args.options.game.cpuct = number(&value()?, "--cpuct")?,
            "--value-scale" => {
                args.options.game.value_scale = number(&value()?, "--value-scale")?;
            }
            "--resign-threshold" => {
                resign.threshold = Some(number(&value()?, "--resign-threshold")?);
            }
            "--resign-plies" => resign.plies = Some(number(&value()?, "--resign-plies")?),
            "--resign-control-frac" => {
                resign.control_frac = Some(number(&value()?, "--resign-control-frac")?);
            }
            "--gumbel" => gumbel.on = Some(true),
            "--gumbel-m" => gumbel.m = Some(number(&value()?, "--gumbel-m")?),
            "--gumbel-pv" => {
                gumbel.pv = Some(
                    GumbelPv::parse(&value()?).map_err(|error| format!("--gumbel-pv: {error}"))?,
                );
            }
            "--net" => args.net = Some(PathBuf::from(value()?)),
            "--no-net" => args.net = None,
            "--out" => {
                let path = value()?;
                args.out = if path == "-" {
                    None
                } else {
                    Some(PathBuf::from(path))
                };
            }
            other => return Err(format!("unknown flag {other}\n\n{USAGE}")),
        }
    }

    if args.options.shard_count == 0 {
        return Err("--shards must be at least 1".to_owned());
    }
    if args.options.shard_idx >= args.options.shard_count {
        return Err(format!(
            "--shard {} does not exist in a {}-shard run",
            args.options.shard_idx, args.options.shard_count
        ));
    }
    if args.options.game.sims == 0 {
        return Err("--sims must be at least 1".to_owned());
    }
    args.options.game.resign = resign.resolve()?;
    args.options.game.gumbel = gumbel.resolve()?;
    Ok(Some(args))
}

fn run(args: Args) -> Result<(), String> {
    let net = match &args.net {
        Some(path) => Some(
            PolicyValueNet::load(path)
                .map_err(|error| format!("loading {}: {error}", path.display()))?,
        ),
        None => None,
    };

    // The banner goes to stderr so `--out -` stays a clean JSONL stream that a
    // caller can pipe straight into the validator.
    eprintln!(
        "shard {}/{}: games {}, sims {}, seed {}, net {}, resign {}, root {}",
        args.options.shard_idx,
        args.options.shard_count,
        args.options.games,
        args.options.game.sims,
        args.options.seed,
        args.net
            .as_ref()
            .map_or_else(|| "hand-tuned".to_owned(), |p| p.display().to_string()),
        args.options.game.resign.map_or_else(
            || "off".to_owned(),
            |r| format!(
                "|v|>{} for {} plies, {:.0}% control",
                r.threshold,
                r.plies,
                r.control_frac * 100.0
            )
        ),
        // Printed on every run, not only the Gumbel ones: which root rule and
        // which `pv` a generation was written with is the first thing anyone
        // reading its rows six weeks later needs, and "PUCT + Dirichlet" is a
        // fact about the file just as much as the other arm is.
        args.options.game.gumbel.map_or_else(
            || "PUCT + Dirichlet, pv = raw visits".to_owned(),
            |g| format!(
                "Gumbel top-{} + sequential halving, pv = {}",
                g.m,
                g.pv.label()
            ),
        ),
    );

    let stats = match &args.out {
        Some(path) => {
            let file = File::create(path)
                .map_err(|error| format!("creating {}: {error}", path.display()))?;
            let mut writer = BufWriter::new(file);
            let stats = generate(&args.options, net.as_ref(), &mut writer)
                .map_err(|error| format!("writing {}: {error}", path.display()))?;
            writer
                .flush()
                .map_err(|error| format!("flushing {}: {error}", path.display()))?;
            stats
        }
        None => {
            let stdout = io::stdout();
            let mut writer = BufWriter::new(stdout.lock());
            let stats = generate(&args.options, net.as_ref(), &mut writer)
                .map_err(|error| format!("writing stdout: {error}"))?;
            writer
                .flush()
                .map_err(|error| format!("flushing stdout: {error}"))?;
            stats
        }
    };

    let per_game = if stats.games == 0 {
        0.0
    } else {
        stats.seconds / stats.games as f64
    };
    eprintln!(
        // `searches` is the run's cost in the only load-independent unit there
        // is. Two runs over the same games are comparable on it; they are not
        // comparable on wall clock, which on a shared box mostly measures the
        // neighbours.
        "shard {}/{}: games {}, rows {}, searches {}, capped {}, outcomes p1={} draw={} p2={}, {:.1}s ({:.1} s/game, {:.1} games/hour)",
        args.options.shard_idx,
        args.options.shard_count,
        stats.games,
        stats.rows,
        stats.searches,
        stats.capped,
        stats.outcomes[2],
        stats.outcomes[1],
        stats.outcomes[0],
        stats.seconds,
        per_game,
        if per_game > 0.0 { 3600.0 / per_game } else { 0.0 },
    );

    // The resign report is the deliverable of a measurement run, so it is
    // printed whether or not the rule fired: "0 resigned, 0 would have" is a
    // result, and a run that printed nothing would be indistinguishable from
    // one where the flag never reached the config.
    if args.options.game.resign.is_some() {
        let percent = |part: u64, whole: u64| {
            if whole == 0 {
                f64::NAN
            } else {
                100.0 * part as f64 / whole as f64
            }
        };
        let resign_arm = stats.games - stats.control;
        eprintln!(
            "shard {}/{}: resign arm {} games, {} resigned ({:.1}%); control arm {} games, \
             {} would have resigned ({:.1}%), {} of those were WRONG -> false-resign {:.1}% \
             (bar: <5%); mean plies saved {:.1}, searches saved {}/{} = {:.1}% projected \
             generation throughput gain {:.1}%",
            args.options.shard_idx,
            args.options.shard_count,
            resign_arm,
            stats.resigned,
            percent(stats.resigned, resign_arm),
            stats.control,
            stats.control_would_resign,
            percent(stats.control_would_resign, stats.control),
            stats.control_false_resign,
            stats.false_resign_rate().map_or(f64::NAN, |r| r * 100.0),
            if stats.control_would_resign == 0 {
                0.0
            } else {
                stats.control_plies_saved as f64 / stats.control_would_resign as f64
            },
            stats.control_searches_saved,
            stats.control_searches,
            stats
                .projected_search_saving()
                .map_or(f64::NAN, |r| r * 100.0),
            // Saving s of the work means the same budget buys 1/(1-s) games.
            stats
                .projected_search_saving()
                .map_or(f64::NAN, |s| (1.0 / (1.0 - s) - 1.0) * 100.0),
        );
    }
    Ok(())
}

fn main() -> ExitCode {
    match parse() {
        Ok(None) => ExitCode::SUCCESS,
        Ok(Some(args)) => match run(args) {
            Ok(()) => ExitCode::SUCCESS,
            Err(message) => {
                eprintln!("selfplay: {message}");
                ExitCode::FAILURE
            }
        },
        Err(message) => {
            eprintln!("selfplay: {message}");
            ExitCode::FAILURE
        }
    }
}
