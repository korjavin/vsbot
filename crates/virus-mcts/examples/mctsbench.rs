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
use virus_mcts::{Config, MctsSearcher, ParallelMcts, PolicyValueNet, ValueSource, CELLS};

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

    // --- 1b. batched forward ---
    {
        let mut scratch = net.batch_scratch();
        let mut heads = Vec::with_capacity(64);
        for batch in [8usize, 16, 32] {
            let inputs = vec![encoded.clone(); batch];
            for _ in 0..10 {
                heads.clear();
                net.forward_batch(&inputs, &mut scratch, &mut heads);
                black_box(&heads);
            }
            let groups = 4_000u64 / batch as u64;
            let start = Instant::now();
            for _ in 0..groups {
                heads.clear();
                net.forward_batch(black_box(&inputs), &mut scratch, &mut heads);
                black_box(&heads);
            }
            per_op(
                &format!("net forward, batch {batch}"),
                groups * batch as u64,
                start.elapsed(),
            );
        }
    }
    println!();

    // --- 2 & 3. full search ---
    for (label, source, net) in [
        ("mcts sims, net value", ValueSource::Net, Some(&net)),
        ("mcts sims, hand-tuned value", ValueSource::HandTuned, None),
    ] {
        let config = Config {
            value_source: source,
            batch_size: 1,
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
        per_op(&format!("{label} (serial)"), u64::from(sims), elapsed);
    }
    println!();

    // --- 4. batch-size sweep, the S3-T1 acceptance table ---
    //
    // `REPEATS` short runs per cell, reported as the median: this devbox runs
    // gauntlets alongside development, so a single timing is worth very little
    // and a mean is worth less than a median.
    const REPEATS: usize = 5;
    let sims = 1_500u32;
    let mut serial = 0.0f64;
    for batch in [1u16, 2, 4, 8, 16, 24, 32, 48] {
        let config = Config {
            value_source: ValueSource::Net,
            batch_size: batch,
            ..Config::play()
        };
        MctsSearcher::new(state.clone(), config, Some(&net)).run_sims(100);
        let mut rates = Vec::with_capacity(REPEATS);
        for _ in 0..REPEATS {
            let mut searcher = MctsSearcher::new(state.clone(), config, Some(&net));
            let start = Instant::now();
            searcher.run_sims(sims);
            let elapsed = start.elapsed();
            black_box(searcher.best_action());
            rates.push(f64::from(sims) / elapsed.as_secs_f64());
        }
        rates.sort_by(f64::total_cmp);
        let median = rates[REPEATS / 2];
        if batch == 1 {
            serial = median;
        }
        println!(
            "batched search, B={batch:<3}         {median:>10.0} sims/s   {:>5.2}x serial   (min {:.0}, max {:.0})",
            median / serial,
            rates[0],
            rates[REPEATS - 1]
        );
    }
    println!();

    // --- 4b. DAG transpositions, the S3-T2 sim-savings table ---
    //
    // Two different savings, and they are not the same number:
    //
    // * **duplicates** — expansions the plain tree spent on a position it had
    //   already evaluated elsewhere in the tree. The DAG's arena is
    //   duplicate-free by construction, so this column is the sim saving. At
    //   `batch_size == 1` both arms hold `sims + 1` nodes; the difference is
    //   what those nodes *are*.
    // * **nodes** — under a batch, a merged node reached twice inside one round
    //   is `pending` the second time and evaluated once, so the DAG's arena is
    //   genuinely smaller as well.
    //
    // sims/s is reported alongside because the index is not free: it costs a
    // key lookup per child creation, against whatever the reuse saves.
    println!(
        "{:<28} {:>9} {:>9} {:>9} {:>8} {:>11}",
        "DAG sweep", "nodes", "distinct", "dup", "merges", "sims/s"
    );
    for batch in [1u16, 8] {
        for dag in [false, true] {
            let config = Config {
                value_source: ValueSource::Net,
                batch_size: batch,
                dag,
                ..Config::play()
            };
            MctsSearcher::new(state.clone(), config, Some(&net)).run_sims(100);
            let mut rates = Vec::with_capacity(REPEATS);
            let mut last = None;
            for _ in 0..REPEATS {
                let mut searcher = MctsSearcher::new(state.clone(), config, Some(&net));
                let start = Instant::now();
                searcher.run_sims(sims);
                let elapsed = start.elapsed();
                black_box(searcher.best_action());
                rates.push(f64::from(sims) / elapsed.as_secs_f64());
                last = Some(searcher);
            }
            rates.sort_by(f64::total_cmp);
            let searcher = last.expect("REPEATS is non-zero");
            let distinct = searcher.distinct_positions();
            println!(
                "  B={batch:<3} DAG {:<3}              {:>9} {distinct:>9} {:>9} {:>8} {:>11.0}",
                if dag { "on" } else { "off" },
                searcher.node_count(),
                searcher.node_count() - distinct,
                searcher.merges(),
                rates[REPEATS / 2],
            );
        }
    }
    println!();

    // --- 5. threads, over the shared tree ---
    let available = std::thread::available_parallelism().map_or(1, |n| n.get());
    for threads in [1usize, 2, 3, 4, 6, 8] {
        if threads > available * 2 {
            continue;
        }
        let config = Config {
            value_source: ValueSource::Net,
            batch_size: 16,
            threads,
            ..Config::play()
        };
        ParallelMcts::new(state.clone(), config, Some(&net)).run_sims(200);
        let mut rates = Vec::with_capacity(REPEATS);
        for _ in 0..REPEATS {
            let mut searcher = ParallelMcts::new(state.clone(), config, Some(&net));
            let start = Instant::now();
            searcher.run_sims(sims);
            let elapsed = start.elapsed();
            black_box(searcher.best_action());
            rates.push(f64::from(sims) / elapsed.as_secs_f64());
        }
        rates.sort_by(f64::total_cmp);
        let median = rates[REPEATS / 2];
        println!(
            "parallel B=16, {threads} thread(s)     {median:>10.0} sims/s   {:>5.2}x serial   (min {:.0}, max {:.0})",
            median / serial,
            rates[0],
            rates[REPEATS - 1]
        );
    }
    println!("\n({available} logical CPUs; run this with the box otherwise idle)");
}
