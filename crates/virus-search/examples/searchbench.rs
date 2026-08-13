//! Search throughput benchmark: nodes per second, in-search.
//!
//! ```bash
//! cargo run --release -p virus-search --example searchbench
//! ```
//!
//! Reference points. The Java `GoBotSearcher` sustains **55-65k NPS** in-search
//! with the same hand-tuned leaf; GoBot is in the same band. `virus-eval`'s own
//! `evalbench` measures 162-178k evals/s standalone, which is the ceiling this
//! search can approach once per-node overhead (grid copy, elimination flood
//! fills, child materialisation) is paid.
//!
//! Reported per mode:
//!
//! * **NPS** — interior nodes per second. This is the number comparable to the
//!   Java figure above.
//! * **evals/s** — leaves per second, comparable to `evalbench`.
//! * **nodes/depth** — how much tree each mode needs to reach the same depth.
//!   Fewer is better and is what the ordering work actually buys; NPS alone can
//!   be *improved* by a worse search.
//!
//! Positions come from the parity fixture, so occupancy, component shape and
//! threat density are what the engine meets in play. Dependency-free on purpose,
//! matching `virus-core`'s microbench and `virus-eval`'s evalbench.

use std::hint::black_box;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use virus_core::fixture::{parse_jsonl, SearchParityRecord};
use virus_core::State;
use virus_search::{SearchOptions, Searcher};

fn load_positions(limit: usize) -> Vec<State> {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/gobot_search_parity.jsonl");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("reading {}: {error}", path.display()));
    let records: Vec<SearchParityRecord> =
        parse_jsonl(&text).unwrap_or_else(|(line, error)| panic!("line {line}: {error}"));
    let stride = (records.len() / limit.max(1)).max(1);
    records
        .iter()
        .step_by(stride)
        .take(limit)
        .map(|record| record.to_state().expect("record decodes"))
        .collect()
}

struct Totals {
    nodes: u64,
    evaluations: u64,
    elapsed: Duration,
}

fn run(positions: &[State], options: SearchOptions, depth: i32) -> Totals {
    let mut totals = Totals {
        nodes: 0,
        evaluations: 0,
        elapsed: Duration::ZERO,
    };
    for state in positions {
        let mut searcher = Searcher::new(state, options);
        let start = Instant::now();
        let result = searcher
            .search_to_depth(state, depth)
            .expect("every fixture position has a legal action");
        totals.elapsed += start.elapsed();
        black_box(result.score);
        totals.nodes += result.nodes;
        totals.evaluations += result.evaluations;
    }
    totals
}

fn report(label: &str, depth: i32, positions: usize, totals: &Totals) {
    let seconds = totals.elapsed.as_secs_f64();
    let nps = totals.nodes as f64 / seconds;
    let eps = totals.evaluations as f64 / seconds;
    let per_position = totals.nodes as f64 / positions as f64;
    println!(
        "{label:<28} d{depth}  {nps:>10.0} NPS  {eps:>10.0} evals/s  \
         {per_position:>10.0} nodes/position  {seconds:>6.2} s",
    );
}

fn main() {
    let positions = load_positions(40);
    println!(
        "virus-search searchbench — {} fixture positions, 12x12, 2 players",
        positions.len()
    );
    println!("Java GoBotSearcher reference: 55-65k NPS in-search\n");

    let deterministic = SearchOptions {
        smp_threads: 0,
        ..SearchOptions::default()
    };
    let plain = SearchOptions::plain();

    // Warm the caches and the branch predictor before anything is timed.
    run(&positions, deterministic, 3);

    for depth in [4, 5, 6] {
        let enhanced = run(&positions, deterministic, depth);
        report("enhanced", depth, positions.len(), &enhanced);
        let oracle = run(&positions, plain, depth);
        report("plain (GoBot oracle)", depth, positions.len(), &oracle);
        let ratio = enhanced.nodes as f64 / oracle.nodes as f64;
        println!(
            "  -> enhanced needs {:.0}% of the oracle's tree to reach the same depth\n",
            ratio * 100.0
        );
    }

    // Lazy SMP is a mode, not a default: it is NOT reproducible run to run, so
    // it is reported separately and never gates anything. The only meaningful
    // SMP metric is depth reached in a fixed wall-clock budget — helper nodes
    // never enter the main thread's counter, so its NPS is unchanged by design.
    let budget = Duration::from_millis(250);
    for threads in [0, 4] {
        let options = SearchOptions {
            smp_threads: threads,
            ..SearchOptions::default()
        };
        let mut depth_sum = 0i64;
        let mut nodes = 0u64;
        let mut elapsed = Duration::ZERO;
        for state in &positions {
            let mut searcher = Searcher::new(state, options);
            let start = Instant::now();
            let result = searcher
                .search(state, budget)
                .expect("every fixture position has a legal action");
            elapsed += start.elapsed();
            depth_sum += i64::from(result.depth);
            nodes += result.nodes;
        }
        let mean_depth = depth_sum as f64 / positions.len() as f64;
        let nps = nodes as f64 / elapsed.as_secs_f64();
        println!(
            "{:<28} {:>4} ms budget  mean depth {mean_depth:>4.2}  {nps:>10.0} NPS (main thread)",
            if threads == 0 {
                "enhanced, timed"
            } else {
                "enhanced + lazy SMP (4)"
            },
            budget.as_millis(),
        );
    }
}
