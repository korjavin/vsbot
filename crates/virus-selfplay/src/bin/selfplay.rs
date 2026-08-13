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

use virus_mcts::PolicyValueNet;
use virus_selfplay::{generate, GameConfig, Options, DEFAULT_MAX_TURNS, DEFAULT_SIMS};

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

OUTPUT:
    --out <PATH>         JSONL destination [default: - for stdout]
    -h, --help           this text

Dirichlet root noise is always on and temperature-1 visit sampling always
covers the first 21 plies: this binary only generates self-play, and both are
what makes self-play data worth training on.
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
        "shard {}/{}: games {}, sims {}, seed {}, net {}",
        args.options.shard_idx,
        args.options.shard_count,
        args.options.games,
        args.options.game.sims,
        args.options.seed,
        args.net
            .as_ref()
            .map_or_else(|| "hand-tuned".to_owned(), |p| p.display().to_string()),
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
        "shard {}/{}: games {}, rows {}, capped {}, outcomes p1={} draw={} p2={}, {:.1}s ({:.1} s/game, {:.1} games/hour)",
        args.options.shard_idx,
        args.options.shard_count,
        stats.games,
        stats.rows,
        stats.capped,
        stats.outcomes[2],
        stats.outcomes[1],
        stats.outcomes[0],
        stats.seconds,
        per_game,
        if per_game > 0.0 { 3600.0 / per_game } else { 0.0 },
    );
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
