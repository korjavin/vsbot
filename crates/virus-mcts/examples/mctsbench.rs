//! MCTS and net-inference throughput microbenchmark.
//!
//! ```bash
//! cargo run --release -p virus-mcts --example mctsbench
//! ```
//!
//! Three numbers, in the order they matter:
//!
//! 1. **net forward** — one fused trunk pass serving both heads and the value.
//!    This is the cost of expanding a node. The Java reference
//!    (`PolicyNetPrior`, `f64`, 32ch x 4 layers) sits at roughly 0.5-2 ms per
//!    evaluation, and pays it *twice* per expanded node in net-value mode
//!    because `priors` and `valueMover` each run the whole trunk.
//! 2. **sims/s, net value** — the self-play and play-mode configuration.
//! 3. **sims/s, hand-tuned value** — the fallback path, where `virus-eval`
//!    rather than the net dominates.
//!
//! Dependency-free on purpose, matching `virus-eval`'s `evalbench`: fixed
//! inputs in tight loops do not justify a benchmark harness in the build.

use std::hint::black_box;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use virus_core::{Cell, CellKind, State};
use virus_mcts::net::Encoded;
use virus_mcts::{Config, MctsSearcher, PolicyValueNet, ValueSource, CELLS};

fn champion() -> PolicyValueNet {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../artifacts/mcts_champion.json");
    PolicyValueNet::load(&path)
        .unwrap_or_else(|error| panic!("loading {}: {error}", path.display()))
}

/// A developed midgame: realistic branching, neutral pairs still available.
fn midgame() -> State {
    let mut cells = vec![Cell::EMPTY; CELLS];
    cells[0] = Cell::new(1, CellKind::Base);
    cells[CELLS - 1] = Cell::new(2, CellKind::Base);
    for index in [1, 12, 13, 14, 25, 26, 27, 38] {
        cells[index] = Cell::new(1, CellKind::Normal);
    }
    for index in [
        CELLS - 2,
        CELLS - 13,
        CELLS - 14,
        CELLS - 15,
        CELLS - 26,
        CELLS - 27,
    ] {
        cells[index] = Cell::new(2, CellKind::Normal);
    }
    State::from_grid(12, 12, 2, &cells, 1, 3, &[false, false]).expect("legal midgame")
}

fn per_op(label: &str, ops: u64, elapsed: Duration) {
    let secs = elapsed.as_secs_f64();
    println!(
        "{label:<34} {ops:>8} ops in {secs:>7.3}s  {:>10.3} ms/op  {:>10.0} ops/s",
        secs * 1e3 / ops as f64,
        ops as f64 / secs
    );
}

fn main() {
    let net = champion();
    println!(
        "net: arch={} channels={} layers={} value_head={}",
        net.arch(),
        net.channels(),
        net.layers(),
        net.has_value_head()
    );
    let state = midgame();
    println!(
        "position: {} legal actions at the root\n",
        state.legal_actions().len()
    );

    // --- 1. standalone fused forward, both convolution paths ---
    let encoded = Encoded::from_state(&state);
    let mut scalar = net.clone();
    scalar.force_scalar();
    for (label, net) in [
        ("net forward, avx2", &net),
        ("net forward, portable", &scalar),
    ] {
        if label.contains("avx2") && !net.simd() {
            println!("net forward, avx2                  (unavailable on this CPU)");
            continue;
        }
        let mut scratch = net.scratch();
        for _ in 0..50 {
            black_box(net.forward(&encoded, &mut scratch));
        }
        let iterations = 2_000u64;
        let start = Instant::now();
        for _ in 0..iterations {
            black_box(net.forward(black_box(&encoded), &mut scratch));
        }
        per_op(label, iterations, start.elapsed());
    }

    // --- 2 & 3. full search ---
    for (label, source, net) in [
        ("mcts sims, net value", ValueSource::Net, Some(&net)),
        ("mcts sims, hand-tuned value", ValueSource::HandTuned, None),
    ] {
        let config = Config {
            value_source: source,
            ..Config::play()
        };
        // Warm up caches and the allocator.
        MctsSearcher::new(state.clone(), config, net).run_sims(100);
        let sims = if net.is_some() { 2_000 } else { 5_000 };
        let mut searcher = MctsSearcher::new(state.clone(), config, net);
        let start = Instant::now();
        searcher.run_sims(sims);
        let elapsed = start.elapsed();
        black_box(searcher.best_action());
        per_op(label, u64::from(sims), elapsed);
    }
}
