//! `probe` — the `PlaceNeutrals` regression probe (bd `vsbot-07x`).
//!
//! ```text
//! # build the fixture (once per refresh; see fixtures/probes/README.md)
//! python3 fixtures/probes/tools/dump_games.py --db games.db \
//!     --out games-dump.jsonl --only-neutral
//! cargo run --release -p virus-arena --bin probe -- mine-db \
//!     --games games-dump.jsonl --out fixtures/probes/neutrals-v1.jsonl
//! cargo run --release -p virus-arena --bin probe -- mine-play \
//!     --out selfplay.jsonl
//!
//! # run a net over it
//! cargo run --release -p virus-arena --bin probe -- run \
//!     --set fixtures/probes/neutrals-v1.jsonl --sims 192 --sims 1000
//! ```
//!
//! **This tool is informational and is never a gate.** It always exits 0 on a
//! completed run, whatever the numbers say, and prints
//! [`virus_arena::probes::INFORMATIONAL`] on every invocation. ARCHITECTURE.md
//! invariant 7 is the reason: seven offline metrics were each believed to
//! predict strength and each one was wrong, so a probe that could fail a build
//! would be a strength gate wearing a diagnostic's clothes.
//!
//! Arguments are parsed by hand, like `arena`'s: the workspace has no
//! argument-parsing dependency and a measuring instrument is a poor place to
//! grow one. An unknown flag is an error, never a silent default.

use std::fmt;
use std::path::PathBuf;
use std::process::ExitCode;

use virus_arena::probes::{
    mine_games, mine_self_play, parse_set, render_set, render_table, run_probe, summarise,
    DumpGame, MineConfig, ProbeRecord, SelfPlayConfig, INFORMATIONAL,
};
use virus_mcts::PolicyValueNet;

const USAGE: &str = "\
probe — the PlaceNeutrals regression probe (bd vsbot-07x)

USAGE:
    probe run       [OPTIONS]     report what a net says about the probe set
    probe mine-db   [OPTIONS]     mine probe positions out of a games.db dump
    probe mine-play [OPTIONS]     mine positions the champion answers with a neutral

run:
    --set <PATH>       probe set JSONL [default: fixtures/probes/neutrals-v1.jsonl]
    --net <PATH>       net artifact [default: artifacts/mcts_champion.json]
    --sims <N>         simulation count to report; repeatable [default: 192, 1000]
    --jsonl <PATH>     also write one JSON report per position here
    --limit <N>        only the first N positions (for a smoke run)

mine-db:
    --games <PATH>     JSONL from fixtures/probes/tools/dump_games.py (required)
    --out <PATH>       probe set to write (required)
    --horizon <N>      turns of the mover's own to measure the swing over [default: 4]
    --min-swing <N>    smallest |swing| in cells that earns a label [default: 4]
    --max-suspect <N>  cap on lost-advantage positions [default: 30]
    --max-control <N>  cap on kept-advantage control positions [default: 8]

mine-play:
    --net <PATH>       net artifact [default: artifacts/mcts_champion.json]
    --out <PATH>       probe set to write (required)
    --games <N>        self-play games [default: 6]
    --seed <N>         base seed [default: 24301]
    --sims <N>         simulations per action [default: 800]
    --max <N>          cap on positions kept [default: 16]

    -h, --help         this text

The probe is a DIAGNOSTIC. It never gates anything; strength claims come only
from >=400-game gauntlets (ARCHITECTURE.md invariant 7).
";

/// Anything that stops a run.
#[derive(Debug)]
struct Failure(String);

impl fmt::Display for Failure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args.iter().any(|arg| arg == "-h" || arg == "--help") {
        print!("{USAGE}");
        return ExitCode::SUCCESS;
    }
    let result = match args[0].as_str() {
        "run" => run(&args[1..]),
        "mine-db" => mine_db(&args[1..]),
        "mine-play" => mine_play(&args[1..]),
        other => Err(Failure(format!("unknown subcommand {other}; try --help"))),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("probe: {error}");
            ExitCode::FAILURE
        }
    }
}

/// A hand-rolled flag reader: `--flag value`, unknown flags rejected.
struct Flags {
    pairs: Vec<(String, String)>,
}

impl Flags {
    fn parse(args: &[String], known: &[&str]) -> Result<Flags, Failure> {
        let mut pairs = Vec::new();
        let mut index = 0;
        while index < args.len() {
            let flag = &args[index];
            if !known.contains(&flag.as_str()) {
                return Err(Failure(format!("unknown flag {flag}; try --help")));
            }
            let value = args
                .get(index + 1)
                .ok_or_else(|| Failure(format!("{flag} needs a value")))?;
            pairs.push((flag.clone(), value.clone()));
            index += 2;
        }
        Ok(Flags { pairs })
    }

    fn last(&self, flag: &str) -> Option<&str> {
        self.pairs
            .iter()
            .rev()
            .find(|(name, _)| name == flag)
            .map(|(_, value)| value.as_str())
    }

    fn all(&self, flag: &str) -> Vec<&str> {
        self.pairs
            .iter()
            .filter(|(name, _)| name == flag)
            .map(|(_, value)| value.as_str())
            .collect()
    }

    fn path(&self, flag: &str, default: &str) -> PathBuf {
        PathBuf::from(self.last(flag).unwrap_or(default))
    }

    fn required(&self, flag: &str) -> Result<&str, Failure> {
        self.last(flag)
            .ok_or_else(|| Failure(format!("{flag} is required")))
    }

    fn number<T: std::str::FromStr>(&self, flag: &str, default: T) -> Result<T, Failure> {
        match self.last(flag) {
            None => Ok(default),
            Some(text) => text
                .parse()
                .map_err(|_| Failure(format!("{flag} wants a number, got {text}"))),
        }
    }
}

fn load_net(path: &PathBuf) -> Result<PolicyValueNet, Failure> {
    PolicyValueNet::load(path)
        .map_err(|error| Failure(format!("could not load {}: {error}", path.display())))
}

fn read(path: &PathBuf) -> Result<String, Failure> {
    std::fs::read_to_string(path)
        .map_err(|error| Failure(format!("could not read {}: {error}", path.display())))
}

fn write(path: &PathBuf, text: &str) -> Result<(), Failure> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|error| {
                Failure(format!("could not create {}: {error}", parent.display()))
            })?;
        }
    }
    std::fs::write(path, text)
        .map_err(|error| Failure(format!("could not write {}: {error}", path.display())))
}

// ------------------------------------------------------------------- run

fn run(args: &[String]) -> Result<(), Failure> {
    let flags = Flags::parse(args, &["--set", "--net", "--sims", "--jsonl", "--limit"])?;
    let set_path = flags.path("--set", "fixtures/probes/neutrals-v1.jsonl");
    let net_path = flags.path("--net", "artifacts/mcts_champion.json");

    let mut sims: Vec<u32> = Vec::new();
    for text in flags.all("--sims") {
        let count: u32 = text
            .parse()
            .map_err(|_| Failure(format!("--sims wants a number, got {text}")))?;
        // Reading the root prior costs one simulation, so a tree can never be
        // observed at zero. Reporting a `sims: 0` row off that one-simulation
        // tree would put a number in the table under a budget nobody spent —
        // exactly the kind of plausible-looking meaningless figure this crate
        // exists to prevent.
        if count == 0 {
            return Err(Failure(
                "--sims 0 measures nothing: reading the root prior already \
                 costs one simulation"
                    .to_owned(),
            ));
        }
        sims.push(count);
    }
    if sims.is_empty() {
        sims = vec![192, 1000];
    }
    sims.sort_unstable();
    sims.dedup();

    let mut records: Vec<ProbeRecord> =
        parse_set(&read(&set_path)?).map_err(|error| Failure(error.to_string()))?;
    if let Some(limit) = flags.last("--limit") {
        let limit: usize = limit
            .parse()
            .map_err(|_| Failure(format!("--limit wants a number, got {limit}")))?;
        records.truncate(limit);
    }
    if records.is_empty() {
        return Err(Failure(format!(
            "{} holds no positions",
            set_path.display()
        )));
    }
    let net = load_net(&net_path)?;

    println!("{INFORMATIONAL}");
    println!();
    println!(
        "net={} arch={} channels={} layers={} pair_bias={:+.4} value_head={}",
        net_path.display(),
        net.arch(),
        net.channels(),
        net.layers(),
        net.pair_bias(),
        net.has_value_head()
    );
    println!("set={} positions={}", set_path.display(), records.len());
    println!();

    let mut reports = Vec::with_capacity(records.len());
    for record in &records {
        reports.push(run_probe(&net, record, &sims).map_err(|error| Failure(error.to_string()))?);
    }

    print!("{}", render_table(&reports, &sims));
    println!();
    let summary = summarise(&reports);
    println!("positions              : {}", summary.positions);
    for (class, mass) in &summary.neutral_prior_mass_by_class {
        println!("mean p(neutral) [{class:<22}]: {mass:.4}");
    }
    for (count, share) in &summary.chose_neutrals_by_sims {
        println!(
            "chose PlaceNeutrals @{count:<5} : {:.1}% of positions",
            share * 100.0
        );
    }
    println!(
        "mean dV(neutral - before): {:+.4}",
        summary.mean_neutral_value_delta
    );
    println!(
        "mean dV(neutral - move)  : {:+.4}",
        summary.mean_neutral_minus_move
    );
    println!(
        "net's single favourite action is a neutral at {} of {} positions",
        summary.best_action_is_neutral, summary.positions
    );
    println!();
    println!("{INFORMATIONAL}");

    if let Some(path) = flags.last("--jsonl") {
        let mut text = String::new();
        for report in &reports {
            let line = serde_json::to_string(report)
                .map_err(|error| Failure(format!("could not render a report: {error}")))?;
            text.push_str(&line);
            text.push('\n');
        }
        write(&PathBuf::from(path), &text)?;
        eprintln!("wrote {} reports to {path}", reports.len());
    }
    Ok(())
}

// --------------------------------------------------------------- mine-db

fn mine_db(args: &[String]) -> Result<(), Failure> {
    let flags = Flags::parse(
        args,
        &[
            "--games",
            "--out",
            "--horizon",
            "--min-swing",
            "--max-suspect",
            "--max-control",
        ],
    )?;
    let games_path = PathBuf::from(flags.required("--games")?);
    let out_path = PathBuf::from(flags.required("--out")?);
    let horizon = flags.number("--horizon", 4u32)?;
    // `mine_games` clamps this defensively, but a typo that silently became a
    // different measurement is exactly the failure mode this crate exists to
    // prevent: say so instead.
    if horizon == 0 {
        return Err(Failure(
            "--horizon 0 measures nothing; the swing needs at least one \
             follow-up turn of the mover's own"
                .to_owned(),
        ));
    }
    // `--min-swing` is a magnitude: the class test is `swing <= -min_swing`,
    // so a negative one inverts it and files placements that *gained* up to
    // `|min_swing|` cells under `LostAdvantage`.
    let min_swing = flags.number("--min-swing", 4i64)?;
    if min_swing < 0 {
        return Err(Failure(format!(
            "--min-swing is a magnitude in cells and cannot be negative, got {min_swing}"
        )));
    }
    let config = MineConfig {
        horizon,
        min_swing,
        max_suspect: flags.number("--max-suspect", 30usize)?,
        max_control: flags.number("--max-control", 8usize)?,
    };

    let text = read(&games_path)?;
    let mut origin = format!("prod games.db dump {}", games_path.display());
    let mut games: Vec<DumpGame> = Vec::new();
    for (number, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        // The dump's first line is its `meta` header, not a game.
        if line.contains("\"meta\"") && !line.contains("\"turns\"") {
            let meta: serde_json::Value = serde_json::from_str(line)
                .map_err(|error| Failure(format!("line {}: {error}", number + 1)))?;
            origin = format!(
                "{} as of {}",
                meta["meta"]["source"].as_str().unwrap_or("games.db"),
                meta["meta"]["as_of"].as_str().unwrap_or("unknown")
            );
            continue;
        }
        games.push(
            serde_json::from_str(line)
                .map_err(|error| Failure(format!("line {}: {error}", number + 1)))?,
        );
    }

    let (records, stats) = mine_games(&games, config, &origin);
    write(
        &out_path,
        &render_set(&records).map_err(|error| Failure(error.to_string()))?,
    )?;

    eprintln!("{INFORMATIONAL}");
    eprintln!("origin: {origin}");
    eprintln!("{stats:#?}");
    eprintln!(
        "wrote {} positions to {}",
        records.len(),
        out_path.display()
    );
    Ok(())
}

// ------------------------------------------------------------- mine-play

fn mine_play(args: &[String]) -> Result<(), Failure> {
    let flags = Flags::parse(
        args,
        &["--net", "--out", "--games", "--seed", "--sims", "--max"],
    )?;
    let net_path = flags.path("--net", "artifacts/mcts_champion.json");
    let out_path = PathBuf::from(flags.required("--out")?);
    let defaults = SelfPlayConfig::default();
    let config = SelfPlayConfig {
        games: flags.number("--games", defaults.games)?,
        seed: flags.number("--seed", defaults.seed)?,
        sims: flags.number("--sims", defaults.sims)?,
        max_positions: flags.number("--max", defaults.max_positions)?,
        ..defaults
    };
    let net = load_net(&net_path)?;
    let origin = format!(
        "self-play on the ponderrepro trajectory generator: net={}, seed={}, sims={}, \
         random_plies={}",
        net_path.display(),
        config.seed,
        config.sims,
        config.random_plies
    );
    let records = mine_self_play(&net, config, &origin);
    write(
        &out_path,
        &render_set(&records).map_err(|error| Failure(error.to_string()))?,
    )?;
    eprintln!("{INFORMATIONAL}");
    eprintln!(
        "wrote {} positions to {}",
        records.len(),
        out_path.display()
    );
    Ok(())
}
