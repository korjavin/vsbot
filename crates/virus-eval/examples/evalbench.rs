//! Static-evaluation throughput microbenchmark.
//!
//! ```bash
//! cargo run --release -p virus-eval --example evalbench
//! ```
//!
//! Reference points: the Java `HandTunedEval` runs ~10-18 us/eval and sustains
//! 55-65k NPS inside its search; GoBot is in the same band. The number that
//! matters for this crate is the **reused-workspace** row — that is the
//! in-search cost. The fresh-workspace row is there to price the allocation tax
//! the workspace exists to remove.
//!
//! Dependency-free on purpose, matching `virus-core`'s microbench: three tight
//! loops over fixed inputs do not justify pulling a benchmark harness into the
//! build.
//!
//! Positions come from the parity fixture, so the board occupancy, component
//! shapes and threat density are the ones the engine actually meets in play.

use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

use virus_core::fixture::{parse_jsonl, SearchParityRecord};
use virus_core::State;
use virus_eval::{evaluate, evaluate_all, EvalParams, EvalWorkspace};

fn load_positions() -> Vec<State> {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/gobot_search_parity.jsonl");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("reading {}: {error}", path.display()));
    parse_jsonl::<SearchParityRecord>(&text)
        .unwrap_or_else(|(line, error)| panic!("line {line}: {error}"))
        .iter()
        .map(|record| record.to_state().expect("record decodes"))
        .collect()
}

fn report(label: &str, iterations: u64, elapsed: std::time::Duration) {
    let seconds = elapsed.as_secs_f64();
    let per = seconds * 1e6 / iterations as f64;
    let rate = iterations as f64 / seconds;
    println!("{label:<40} {per:>8.3} us/eval  {rate:>12.0} evals/s");
}

fn main() {
    let positions = load_positions();
    let params = EvalParams::default();
    println!(
        "virus-eval evalbench — {} fixture positions, 12x12, 2 players\n",
        positions.len()
    );

    let rounds = 200u64;
    let iterations = rounds * positions.len() as u64;

    // 1. The in-search path: one workspace, reused for every leaf.
    let mut workspace = EvalWorkspace::new();
    for state in &positions {
        black_box(evaluate_all(state, &params, &mut workspace));
    }
    let start = Instant::now();
    let mut checksum = 0i64;
    for _ in 0..rounds {
        for state in &positions {
            checksum = checksum.wrapping_add(black_box(evaluate(
                state,
                state.current_player(),
                &params,
                &mut workspace,
            )));
        }
    }
    report("evaluate (workspace reused)", iterations, start.elapsed());

    // 2. All four seats in one pass — the same work, four utilities out.
    let start = Instant::now();
    for _ in 0..rounds {
        for state in &positions {
            checksum =
                checksum.wrapping_add(black_box(evaluate_all(state, &params, &mut workspace))[0]);
        }
    }
    report(
        "evaluate_all (workspace reused)",
        iterations,
        start.elapsed(),
    );

    // 3. The allocation tax the workspace removes: a fresh workspace per leaf
    //    is what the Go/Java evaluators effectively do at every node.
    let rounds = 20u64;
    let iterations = rounds * positions.len() as u64;
    let start = Instant::now();
    for _ in 0..rounds {
        for state in &positions {
            let mut fresh = EvalWorkspace::new();
            checksum =
                checksum.wrapping_add(black_box(evaluate_all(state, &params, &mut fresh))[0]);
        }
    }
    report(
        "evaluate_all (fresh workspace)",
        iterations,
        start.elapsed(),
    );

    println!("\nchecksum {checksum:#x}");
}
