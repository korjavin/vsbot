//! The determinism gate.
//!
//! A gauntlet is a measuring instrument, and an instrument that gives a
//! different answer on the same input is not measuring anything. Both
//! predecessors learned this the expensive way: Go's arena documents that its
//! xorshift baseline agent mutates closure state across calls and therefore
//! "must run at workers=1", a footnote that existed because a parallel run had
//! already produced a verdict nobody could reproduce.
//!
//! So this file asserts the property directly, three ways:
//!
//! 1. The same config run twice gives byte-identical W-L-D **and** byte-identical
//!    per-game detail.
//! 2. Thread count changes nothing but wall time.
//! 3. A gauntlet of a configuration against *itself* reads exactly 50%, which is
//!    the pairing property that makes the tally interpretable at all.
//!
//! Everything here uses [`Budget::Nodes`]. Fixed-time mode is deliberately not
//! tested for reproducibility: a wall clock cannot be reproduced, and a test
//! that pretended otherwise would fail on a loaded CI box and teach everyone to
//! ignore it.

use virus_arena::engine::{Budget, Engine, SideSpec};
use virus_arena::gauntlet::{run, GauntletConfig, GauntletResult, Termination};

/// A full 12x12 run is minutes; the determinism property is board-size
/// independent, so the gate runs on a board that finishes in seconds.
fn config(side_a: Engine, side_b: Engine, games: u32) -> GauntletConfig {
    GauntletConfig {
        side_a: SideSpec {
            engine: side_a,
            budget: Budget::Nodes(1_500),
            net: None,
        },
        side_b: SideSpec {
            engine: side_b,
            budget: Budget::Nodes(1_500),
            net: None,
        },
        games,
        seed: 20_260_813,
        rows: 8,
        cols: 8,
        max_turns: 60,
        threads: 1,
        ..GauntletConfig::default()
    }
}

/// The full per-game fingerprint, not just the tally. Two runs that agree on
/// W-L-D by coincidence while disagreeing on which games were won would pass a
/// tally-only check and still be broken.
fn fingerprint(result: &GauntletResult) -> Vec<String> {
    result
        .games
        .iter()
        .map(|game| {
            format!(
                "{} a_p1={} winner={} turns={} plies={} term={:?} territory={} work_a={} work_b={}",
                game.index,
                game.a_is_p1,
                game.winner,
                game.turns,
                game.plies,
                game.termination,
                game.territory_winner,
                game.work_a,
                game.work_b,
            )
        })
        .collect()
}

#[test]
fn a_fixed_seed_node_budget_gauntlet_reproduces_byte_identically() {
    let config = config(
        Engine::AlphaBeta { enhanced: true },
        Engine::AlphaBeta { enhanced: false },
        12,
    );
    let first = run(&config, None).expect("first run");
    let second = run(&config, None).expect("second run");

    assert_eq!(first.record, second.record, "W-L-D must reproduce");
    assert_eq!(
        fingerprint(&first),
        fingerprint(&second),
        "every game must replay identically, not just the tally"
    );
    // Not a self-gauntlet: a run where nothing ever happens would satisfy the
    // assertions above without proving anything.
    assert!(first.record.games() == 12);
    assert!(
        first.games.iter().all(|game| game.plies > 5),
        "the games must actually be played"
    );
}

/// Games are independent, so spreading them over workers must change only the
/// wall time. This is the property that lets the nightly ladder use every core
/// without anyone having to re-derive whether the number still means the same
/// thing.
#[test]
fn the_tally_does_not_depend_on_the_worker_count() {
    let base = config(Engine::AlphaBeta { enhanced: true }, Engine::Greedy, 12);
    let serial = run(&base, None).expect("serial");
    for threads in [2, 3, 4, 8] {
        let parallel = run(
            &GauntletConfig {
                threads,
                ..base.clone()
            },
            None,
        )
        .expect("parallel");
        assert_eq!(
            serial.record, parallel.record,
            "W-L-D changed at {threads} threads"
        );
        assert_eq!(
            fingerprint(&serial),
            fingerprint(&parallel),
            "game detail changed at {threads} threads"
        );
    }
}

/// The pairing property. Both colours of a pair replay one opening with the
/// seats swapped, so a configuration played against itself splits every pair
/// 1-1 and the run reads exactly 50%. If this ever fails, state is leaking
/// between games and *every* number the harness has ever printed is suspect.
#[test]
fn a_self_gauntlet_reads_exactly_fifty_percent() {
    for engine in [
        Engine::AlphaBeta { enhanced: true },
        Engine::AlphaBeta { enhanced: false },
        Engine::Greedy,
    ] {
        let result = run(&config(engine, engine, 10), None).expect("self-gauntlet");
        assert_eq!(
            result.record.wins, result.record.losses,
            "{engine:?} against itself: {:?}",
            result.record
        );
        for pair in 0..5 {
            let even = &result.games[pair * 2];
            let odd = &result.games[pair * 2 + 1];
            assert_eq!(
                (even.winner, even.plies),
                (odd.winner, odd.plies),
                "{engine:?} pair {pair} did not replay one game from both chairs"
            );
        }
    }
}

/// A run whose games all ended by stalling would report a tidy tally built out
/// of nothing. The harness distinguishes the terminations; the gate checks that
/// real games are what is being counted.
#[test]
fn gate_runs_end_in_decided_games() {
    let result = run(
        &config(
            Engine::AlphaBeta { enhanced: true },
            Engine::AlphaBeta { enhanced: false },
            8,
        ),
        None,
    )
    .expect("run");
    for game in &result.games {
        assert_eq!(
            game.termination,
            Termination::Decided,
            "game {} did not finish on the board: {game:?}",
            game.index
        );
        assert_eq!(
            game.max_overrun,
            std::time::Duration::ZERO,
            "a node-budget game must never consult a clock"
        );
    }
}
