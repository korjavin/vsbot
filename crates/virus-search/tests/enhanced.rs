//! The enhanced (production) search stack.
//!
//! The parity fixtures pin the *oracle*; these pin the strength path — the
//! features that only exist behind [`SearchOptions::enhanced`]. Two of them are
//! not conveniences but ARCHITECTURE.md invariants with a documented body count:
//! the turn-ending LMR tripwire and the deadline/fixed-depth consistency gate.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use virus_core::fixture::{parse_jsonl, SearchParityRecord};
use virus_core::{Action, State};
use virus_search::{SearchOptions, Searcher};

/// A handful of real mid-game positions from the parity fixture, so the enhanced
/// path is exercised on the same shapes the oracle was recorded from rather than
/// on a hand-built toy.
fn positions(count: usize) -> Vec<State> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures")
        .join("gobot_search_parity.jsonl");
    let text = std::fs::read_to_string(&path).expect("fixture is checked in");
    let records: Vec<SearchParityRecord> = parse_jsonl(&text).expect("fixture parses");
    records
        .iter()
        .step_by(records.len() / count.max(1))
        .take(count)
        .map(|record| record.to_state().expect("fixture record is legal"))
        .collect()
}

fn wedged() -> State {
    let mut state = State::new(12, 12, 2).expect("12x12 two-player board");
    for action in [
        Action::mv(1, 1),
        Action::mv(2, 1),
        Action::mv(2, 2),
        Action::mv(10, 10),
        Action::mv(9, 10),
        Action::mv(9, 9),
    ] {
        state = state.apply(action).expect("wedge line is legal");
    }
    state
}

/// Deterministic options: lazy SMP off, everything else on. Every gate in this
/// file must be reproducible run to run, and helper TT entries steer the main
/// tree — so SMP is a *mode*, never a default.
fn deterministic() -> SearchOptions {
    SearchOptions {
        smp_threads: 0,
        ..SearchOptions::default()
    }
}

// ---------------------------------------------------------------- LMR

/// The tripwire. A turn-ending action must never be reduced: the plan flags the
/// turn boundary as where tempo-swing evaluation error concentrates, and a
/// reduced leaf on the far side lands in the *other* side's turn fragment with a
/// systematically different tempo term.
#[test]
fn late_move_reduction_never_touches_a_turn_ending_action() {
    let mut reduced = 0;
    for state in positions(8) {
        let mut searcher = Searcher::new(&state, deterministic());
        searcher
            .search_to_depth(&state, 5)
            .expect("a legal action exists");
        assert_eq!(
            searcher.stats().lmr_turn_ending_reductions,
            0,
            "reduced a turn-ending action"
        );
        reduced += searcher.stats().lmr_reductions;
    }
    assert!(
        reduced > 0,
        "no reductions happened at all, so the tripwire proves nothing"
    );
}

#[test]
fn late_move_reduction_can_be_switched_off() {
    let state = wedged();
    let mut on = Searcher::new(&state, deterministic());
    on.search_to_depth(&state, 5).expect("legal");
    let mut off = Searcher::new(
        &state,
        SearchOptions {
            lmr: false,
            ..deterministic()
        },
    );
    off.search_to_depth(&state, 5).expect("legal");
    assert!(on.stats().lmr_reductions > 0);
    assert_eq!(off.stats().lmr_reductions, 0);
}

// ---------------------------------------------------------------- invariant 3

/// **ARCHITECTURE.md invariant 3.** A budget-aborted search must return exactly
/// what the same iterative deepening returns at the depth it actually completed.
/// The Go engine's wall-clock `choose()` violated this and turned a
/// parity-perfect engine into an 0-10 live record.
///
/// The node-budget path is the *deterministic* form of the same abort mechanism
/// — same `at_depth`, same unwind, same discard of the partial iteration — so it
/// pins the invariant without a clock in the loop.
#[test]
fn a_budget_aborted_search_returns_the_completed_iteration() {
    for state in positions(6) {
        for limit in [2_000, 20_000, 100_000] {
            let mut budgeted = Searcher::new(&state, deterministic());
            let result = budgeted
                .search_node_budget(&state, limit)
                .expect("a legal action exists");
            if result.depth < 1 {
                continue; // not even depth 1 completed: the fallback, by design
            }
            assert!(!result.salvaged, "the node-budget path never salvages");

            let mut fixed = Searcher::new(&state, deterministic());
            let oracle = fixed
                .search_to_depth(&state, result.depth)
                .expect("a legal action exists");
            assert_eq!(
                (result.action, result.score),
                (oracle.action, oracle.score),
                "limit {limit} reported depth {} but returned a different move",
                result.depth,
            );
        }
    }
}

/// The wall-clock form of the same invariant. A salvaged result is the one
/// sanctioned exception (and carries its own flag), so it is checked separately;
/// everything else must equal the fixed-depth answer at the reported depth.
#[test]
fn a_deadline_search_returns_the_completed_iteration() {
    for state in positions(4) {
        for millis in [5, 25, 120] {
            let mut timed = Searcher::new(&state, deterministic());
            let result = timed
                .search(&state, Duration::from_millis(millis))
                .expect("a legal action exists");
            if result.depth < 1 || result.salvaged {
                continue;
            }
            let mut fixed = Searcher::new(&state, deterministic());
            let oracle = fixed
                .search_to_depth(&state, result.depth)
                .expect("a legal action exists");
            assert_eq!(
                (result.action, result.score),
                (oracle.action, oracle.score),
                "{millis} ms reported depth {} but returned a different move",
                result.depth,
            );
        }
    }
}

/// The soft deadline exists so a doomed iteration does not eat the second half
/// of the budget; the hard deadline still bounds an iteration already underway.
#[test]
fn a_deadline_search_respects_its_budget() {
    let state = wedged();
    let budget = Duration::from_millis(120);
    let mut searcher = Searcher::new(&state, deterministic());
    let start = Instant::now();
    searcher
        .search(&state, budget)
        .expect("a legal action exists");
    assert!(
        start.elapsed() < budget * 4,
        "overran the budget by more than the clock-check granularity allows"
    );
}

// ---------------------------------------------------------------- transposition

#[test]
fn the_packed_table_survives_between_moves() {
    let state = wedged();
    let mut searcher = Searcher::enhanced(&state);
    searcher
        .search_node_budget(&state, 30_000)
        .expect("a legal action exists");
    assert!(searcher.tt_has_entry(&state), "the root was stored");

    // Same seat, next action of the same turn: the table must carry over.
    let action = Action::mv(3, 3);
    let next = state.apply(action).expect("legal");
    assert_eq!(next.current_player(), state.current_player());
    searcher
        .search_node_budget(&next, 30_000)
        .expect("a legal action exists");
    assert!(
        searcher.tt_has_entry(&state),
        "the previous move's principal subtree must still be probeable"
    );
}

#[test]
fn the_staged_fast_path_actually_cuts() {
    let state = wedged();
    let mut searcher = Searcher::enhanced(&state);
    searcher.search_to_depth(&state, 6).expect("legal");
    let stats = searcher.stats();
    assert!(stats.tt_hits > 0, "no TT hits at all");
    assert!(
        stats.fast_path_cuts > 0,
        "no cut-node was resolved by the TT move alone"
    );
}

#[test]
fn the_oracle_keeps_no_packed_table() {
    let state = wedged();
    let mut searcher = Searcher::plain(&state);
    searcher.search_to_depth(&state, 4).expect("legal");
    assert!(!searcher.tt_has_entry(&state));
    assert_eq!(searcher.stats().tt_probes, 0);
    assert_eq!(searcher.stats().lmr_reductions, 0);
    assert_eq!(searcher.stats().fast_path_cuts, 0);
}

// ---------------------------------------------------------------- heuristics

#[test]
fn history_is_recorded_and_then_aged_between_searches() {
    let state = wedged();
    let mut searcher = Searcher::enhanced(&state);
    searcher.search_to_depth(&state, 5).expect("legal");
    let peak = searcher.history_peak(state.current_player());
    assert!(peak > 0, "no quiet cutoff was ever recorded");

    // A second search halves what the first learned instead of clearing it.
    searcher.search_to_depth(&state, 1).expect("legal");
    let aged = searcher.history_peak(state.current_player());
    assert!(
        aged > 0 && aged <= peak.div_euclid(2) + 1,
        "history should fade, not vanish or dominate: {peak} -> {aged}"
    );
}

// ---------------------------------------------------------------- aspiration

/// Progressive widening ends at the full window, so an aspirated iteration must
/// return the same *score* an un-aspirated one does. (The chosen action can
/// differ between equal-scoring siblings, which is why only the score is pinned.)
#[test]
fn aspiration_windows_do_not_change_the_score() {
    for state in positions(4) {
        let mut with = Searcher::new(&state, deterministic());
        let mut without = Searcher::new(
            &state,
            SearchOptions {
                aspiration: false,
                ..deterministic()
            },
        );
        let a = with.search_to_depth(&state, 5).expect("legal");
        let b = without.search_to_depth(&state, 5).expect("legal");
        assert_eq!(a.score, b.score, "aspiration changed the exact score");
    }
    // And it must actually have been exercised somewhere.
    let state = wedged();
    let mut searcher = Searcher::new(&state, deterministic());
    searcher.search_to_depth(&state, 6).expect("legal");
    let stats = searcher.stats();
    assert!(
        stats.aspiration_fail_lows + stats.aspiration_fail_highs > 0,
        "no aspirated iteration ever failed its window"
    );
}

// ---------------------------------------------------------------- lazy SMP

/// SMP is not deterministic by construction, so this only pins that it runs, is
/// bounded, and still returns a legal move.
#[test]
fn lazy_smp_helpers_keep_the_search_sound() {
    let state = wedged();
    let mut searcher = Searcher::new(
        &state,
        SearchOptions {
            smp_threads: 4,
            ..SearchOptions::default()
        },
    );
    let result = searcher
        .search(&state, Duration::from_millis(80))
        .expect("a legal action exists");
    let action = result.action.expect("an action was chosen");
    assert!(
        state.legal_actions().contains(&action),
        "SMP returned an illegal action"
    );
}

#[test]
fn smp_is_off_by_default() {
    assert_eq!(SearchOptions::default().smp_threads, 0);
    assert_eq!(SearchOptions::plain().smp_threads, 0);
}

// ---------------------------------------------------------------- determinism

#[test]
fn the_enhanced_search_is_deterministic_with_smp_off() {
    for state in positions(4) {
        let first = Searcher::new(&state, deterministic())
            .search_node_budget(&state, 50_000)
            .expect("legal");
        let second = Searcher::new(&state, deterministic())
            .search_node_budget(&state, 50_000)
            .expect("legal");
        assert_eq!(first, second);
    }
}

// ---------------------------------------------------------------- multiplayer

/// ARCHITECTURE.md invariant 1 in its purest form: with three seats there is no
/// single opponent to negate against, so the search carries the whole utility
/// vector and the mover maximises its own component.
#[test]
fn three_player_games_run_max_n() {
    let state = State::new(9, 9, 3).expect("9x9 three-player board");
    for options in [deterministic(), SearchOptions::plain()] {
        let mut searcher = Searcher::new(&state, options);
        let result = searcher
            .search_node_budget(&state, 20_000)
            .expect("a legal action exists");
        let action = result.action.expect("an action was chosen");
        assert!(state.legal_actions().contains(&action));
        // maxN never runs the alpha-beta heuristics, in either mode.
        assert_eq!(searcher.stats().lmr_reductions, 0);
    }
}

// ---------------------------------------------------------------- opening book

#[test]
fn the_live_paths_play_the_book_and_label_it() {
    let state = State::new(12, 12, 2).expect("12x12 two-player board");
    let result = virus_search::choose_node_budget(&state, 50_000).expect("legal");
    assert_eq!(result.action, Some(Action::mv(1, 1)));
    assert!(result.book && result.depth == 0 && result.nodes == 0);

    let timed = virus_search::choose(&state, Duration::from_millis(50)).expect("legal");
    assert_eq!(timed.action, Some(Action::mv(1, 1)));
    assert!(timed.book);
}

#[test]
fn the_oracle_skips_the_book() {
    let state = State::new(12, 12, 2).expect("12x12 two-player board");
    let result = virus_search::choose_depth(&state, 3).expect("legal");
    assert!(!result.book && result.nodes > 0);
}

// ---------------------------------------------------------------- contracts

/// Root-relative scores are only meaningful while the root player is the mover,
/// so a persistent searcher handed the other seat is a programming error, not a
/// recoverable condition. (`wedged()` has both blocks complete, so the opening
/// book is void for both seats and cannot short-circuit ahead of the check.)
#[test]
#[should_panic(expected = "searcher rooted at player")]
fn a_searcher_refuses_to_move_for_the_wrong_seat() {
    let state = wedged();
    let mut searcher = Searcher::enhanced(&state);
    let mut other = state;
    for action in [Action::mv(3, 3), Action::mv(3, 2), Action::mv(3, 1)] {
        other = other.apply(action).expect("legal");
    }
    assert_eq!(other.current_player(), 2);
    let _ = searcher.search_node_budget(&other, 1_000);
}

#[test]
fn a_zero_budget_has_no_answer() {
    let state = wedged();
    assert!(Searcher::enhanced(&state)
        .search_node_budget(&state, 0)
        .is_none());
    assert!(virus_search::choose_depth(&state, 0).is_none());
    assert!(virus_search::choose_depth(&state, 65).is_none());
}
