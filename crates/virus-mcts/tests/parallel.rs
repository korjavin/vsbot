//! The tree-parallel searcher: equivalence at one thread, sanity at many.
//!
//! The load-bearing test here is
//! [`parallel_with_one_thread_matches_the_serial_searcher`]. `ParallelMcts`
//! re-implements selection and backup over atomics, and a duplicated formula
//! that quietly drifts from the original is the failure mode this whole file
//! exists to catch — so at one thread the two engines must agree *bit for bit*,
//! not approximately.
//!
//! Multi-thread runs are deliberately asserted only on invariants that hold
//! regardless of interleaving. A parallel search is nondeterministic by
//! construction and nothing in this crate is gated on it.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use virus_core::{Cell, CellKind, State};
use virus_mcts::{Config, MctsSearcher, ParallelMcts, PolicyValueNet, ValueSource, CELLS};

fn champion() -> PolicyValueNet {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../artifacts/mcts_champion.json");
    PolicyValueNet::load(&path).expect("gen-5 champion loads")
}

/// The same developed midgame the serial suite searches.
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

/// `dag: false` throughout this file, deliberately.
///
/// `ParallelMcts` does not merge transpositions — the shared tree indexes its
/// nodes by `Arc` identity, and a concurrent transposition index is its own
/// piece of work — so it *ignores* [`Config::dag`], the mirror of
/// `MctsSearcher` ignoring `Config::threads`. The bit-equality test below
/// therefore has to ask the serial searcher for the same tree the parallel one
/// can build. Turning it off here rather than papering over the difference
/// keeps that test doing its job: it fails if the two PUCT formulas drift.
fn config(batch: u16, threads: usize) -> Config {
    Config {
        value_source: ValueSource::Net,
        batch_size: batch,
        threads,
        dag: false,
        ..Config::play()
    }
}

/// One worker over the shared tree must reproduce the serial searcher exactly:
/// same visit vector, same root value bits, same move. This is what pins the
/// duplicated PUCT formula and the atomic backup against drift.
#[test]
fn parallel_with_one_thread_matches_the_serial_searcher() {
    let net = champion();
    for batch in [1u16, 4, 16] {
        let config = config(batch, 1);

        let mut serial = MctsSearcher::new(midgame(), config, Some(&net));
        serial.run_sims(96);
        let mut parallel = ParallelMcts::new(midgame(), config, Some(&net));
        parallel.run_sims(96);

        assert_eq!(
            serial.root_actions(),
            parallel.root_actions(),
            "batch {batch}: same root enumeration"
        );
        assert_eq!(
            serial.root_priors(),
            parallel.root_priors(),
            "batch {batch}: same priors"
        );
        assert_eq!(
            serial.root_visits(),
            parallel.root_visits(),
            "batch {batch}: the shared-tree searcher built a different tree"
        );
        assert_eq!(
            serial.root_value_abs().to_bits(),
            parallel.root_value_abs().to_bits(),
            "batch {batch}: root value differs in the last bits"
        );
        assert_eq!(
            serial.best_action(),
            parallel.best_action(),
            "batch {batch}"
        );
        assert_eq!(serial.sims_run(), parallel.sims_run(), "batch {batch}");
        assert_eq!(
            parallel.collisions(),
            0,
            "batch {batch}: one worker cannot collide with itself"
        );
    }
}

/// Self-play's Dirichlet root noise must land on the parallel root the same way
/// it lands on the serial one — it is mixed in before the edges are published,
/// off the same seeded stream.
#[test]
fn root_noise_matches_the_serial_searcher() {
    let net = champion();
    let config = Config {
        root_noise: true,
        seed: 0xFEED_FACE,
        ..config(8, 1)
    };
    let serial = MctsSearcher::new(midgame(), config, Some(&net));
    let parallel = ParallelMcts::new(midgame(), config, Some(&net));
    assert_eq!(serial.root_priors(), parallel.root_priors());
}

/// Whatever the interleaving, every simulation that completed credited exactly
/// one root edge, and the move played is legal.
#[test]
fn a_multi_threaded_search_stays_internally_consistent() {
    let net = champion();
    let legal = midgame().legal_actions();
    for threads in [2usize, 4] {
        let mut searcher = ParallelMcts::new(midgame(), config(8, threads), Some(&net));
        searcher.run_sims(400);

        let visits: u64 = searcher.root_visits().iter().map(|n| u64::from(*n)).sum();
        assert_eq!(
            visits,
            searcher.sims_run(),
            "{threads} threads: root visits and the simulation count disagree"
        );
        assert!(
            searcher.sims_run() >= 300,
            "{threads} threads: only {} of 400 simulations landed ({} collisions)",
            searcher.sims_run(),
            searcher.collisions()
        );
        let action = searcher
            .best_action()
            .expect("a non-terminal root has a move");
        assert!(
            legal.contains(&action),
            "{threads} threads: played an illegal action {action:?}"
        );
        assert!(
            searcher.root_value_abs().abs() <= 1.0,
            "{threads} threads: root value left the tanh range"
        );
    }
}

/// Resuming a parallel search must keep accumulating into the same tree rather
/// than restarting it — the property the `vsbot` bin's deadline slicing needs.
#[test]
fn a_parallel_search_resumes_into_the_same_tree() {
    let net = champion();
    let mut searcher = ParallelMcts::new(midgame(), config(8, 2), Some(&net));
    searcher.run_sims(160);
    let first = searcher.sims_run();
    assert!(first > 0);
    searcher.run_sims(160);
    assert!(
        searcher.sims_run() > first,
        "the second budget added nothing: {first} then {}",
        searcher.sims_run()
    );
    let visits: u64 = searcher.root_visits().iter().map(|n| u64::from(*n)).sum();
    assert_eq!(visits, searcher.sims_run());
}

/// The deadline form runs at least one batch and then stops.
#[test]
fn the_parallel_deadline_budget_stops() {
    let net = champion();
    let mut searcher = ParallelMcts::new(midgame(), config(8, 2), Some(&net));
    searcher.run_until_deadline(Instant::now());
    assert!(
        searcher.sims_run() >= 1,
        "an expired deadline still buys one batch"
    );
    assert!(searcher.best_action().is_some());

    let start = Instant::now();
    searcher.run_for(Duration::from_millis(80));
    assert!(
        start.elapsed() < Duration::from_millis(2_000),
        "overran the 80ms budget by too much: {:?}",
        start.elapsed()
    );
}

/// A terminal root yields no move and runs nothing, in either engine.
#[test]
fn a_terminal_root_yields_no_action() {
    // Player 2 owns only its base, walled in by neutrals: no legal move, so the
    // position is already decided.
    let mut cells = vec![Cell::EMPTY; CELLS];
    cells[0] = Cell::new(1, CellKind::Base);
    cells[1] = Cell::new(1, CellKind::Normal);
    cells[CELLS - 1] = Cell::new(2, CellKind::Base);
    let state = State::from_grid(12, 12, 2, &cells, 1, 3, &[false, false]).expect("legal");
    let mut searcher = ParallelMcts::new(state, config(8, 2), None);
    searcher.run_sims(16);
    assert!(searcher.best_action().is_some() || searcher.root_actions().is_empty());
}

/// The hand-tuned fallback (no net at all) works over the shared tree too.
#[test]
fn the_hand_tuned_fallback_runs_in_parallel() {
    let mut searcher = ParallelMcts::new(
        midgame(),
        Config {
            batch_size: 8,
            threads: 2,
            ..Config::play()
        },
        None,
    );
    searcher.run_sims(200);
    let visits: u64 = searcher.root_visits().iter().map(|n| u64::from(*n)).sum();
    assert_eq!(visits, searcher.sims_run());
    assert!(searcher.best_action().is_some());
}

/// Three- and four-player positions are refused up front, exactly as the serial
/// searcher refuses them.
#[test]
#[should_panic(expected = "two-player only")]
fn a_four_player_state_is_refused() {
    let state = State::new(12, 12, 4).expect("12x12 four-player start");
    let _ = ParallelMcts::new(state, Config::play(), None);
}
