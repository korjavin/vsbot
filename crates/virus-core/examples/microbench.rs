//! Movegen / apply microbenchmark.
//!
//! ```bash
//! cargo run --release -p virus-core --example microbench
//! ```
//!
//! Deliberately dependency-free rather than Criterion: this measures three
//! tight loops on fixed inputs, and pulling a benchmark harness (and its
//! compile time) into CI buys nothing at this stage. The numbers exist to
//! anchor the "Rust removes the per-node allocation tax" claim in
//! ARCHITECTURE.md — re-run them before and after any hot-path change.
//!
//! Positions are sampled from the real parity fixture, so the branching factors
//! are the ones the engine actually meets in play.

use std::hint::black_box;
use std::time::Instant;
use virus_core::fixture::{parse_jsonl, SearchParityRecord};
use virus_core::{Position, Scratch, State};

fn load_positions() -> Vec<State> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/gobot_search_parity.jsonl");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("reading {}: {error}", path.display()));
    parse_jsonl::<SearchParityRecord>(&text)
        .unwrap_or_else(|(line, error)| panic!("line {line}: {error}"))
        .iter()
        .map(|record| record.to_state().expect("record decodes"))
        .collect()
}

fn report(label: &str, iterations: u64, elapsed: std::time::Duration) {
    let per = elapsed.as_secs_f64() * 1e9 / iterations as f64;
    println!("{label:<44} {per:>9.1} ns/op   ({iterations} ops)");
}

fn main() {
    let positions = load_positions();
    let mut scratch = Scratch::new();
    println!(
        "virus-core microbench — {} fixture positions, 12x12, 2 players\n",
        positions.len()
    );

    // 1. Movegen: connectivity flood-fill + frontier scan, thread-local scratch.
    let rounds = 200u64;
    let start = Instant::now();
    let mut count = 0usize;
    for _ in 0..rounds {
        for state in &positions {
            count += black_box(state.legal_actions()).len();
        }
    }
    report(
        "State::legal_actions (exact, incl. pairs)",
        rounds * positions.len() as u64,
        start.elapsed(),
    );
    let mean_branch = count as f64 / (rounds * positions.len() as u64) as f64;

    // 2. Position construction: the search's per-node entry point — one shared
    //    flood-fill plus, above the branch threshold, the Tarjan curation.
    let start = Instant::now();
    for _ in 0..rounds {
        for state in &positions {
            let position = Position::new_with(state.clone(), &mut scratch);
            black_box(&position);
        }
    }
    report(
        "Position::new_with (movegen + curation)",
        rounds * positions.len() as u64,
        start.elapsed(),
    );

    // 3. The search hot path: enumerate children and apply each one.
    let mut applied = 0u64;
    let start = Instant::now();
    for _ in 0..rounds {
        for state in &positions {
            let position = Position::new_with(state.clone(), &mut scratch);
            position.for_each_search_action(|action| {
                black_box(state.apply_generated(action));
                applied += 1;
                true
            });
        }
    }
    let elapsed = start.elapsed();
    report("apply_generated (copy + elimination)", applied, elapsed);

    // 4. Full node cost: enumerate, apply, and re-analyse every child.
    let mut nodes = 0u64;
    let start = Instant::now();
    for state in &positions {
        for _ in 0..20 {
            let position = Position::new(state.clone());
            position.for_each_search_action(|action| {
                black_box(position.apply_search(action).search_actions());
                nodes += 1;
                true
            });
        }
    }
    report("child node (apply + re-enumerate)", nodes, start.elapsed());

    println!("\nmean branching factor: {mean_branch:.1} actions/position");
    let single = Instant::now();
    let mut hashes = 0u64;
    for _ in 0..1000 {
        for state in &positions {
            hashes ^= black_box(state.hash());
            hashes ^= black_box(state.state_hash());
        }
    }
    report(
        "hash() + state_hash() (Zobrist + FNV)",
        1000 * positions.len() as u64,
        single.elapsed(),
    );
    black_box(hashes);
}
