//! End-to-end play against the real Go server.
//!
//! This is the acceptance gate for `virus-proto`: unit tests prove the state
//! machine in isolation, but only a live server exercises the *ordering* the
//! invariants are about — `move_made` before `game_state`, the
//! `neutrals_placed` ack before `turn_change`, `users_update` bursts around
//! every game end.
//!
//! # Running it
//!
//! Opt-in, because it needs a Go toolchain and a checkout of the server:
//!
//! ```text
//! VSBOT_ITEST=1 cargo test -p vsbot --test live_games -- --nocapture
//! ```
//!
//! Knobs:
//!
//! | Variable                 | Default                                     |
//! |--------------------------|---------------------------------------------|
//! | `VSBOT_ITEST`            | unset — the tests skip                      |
//! | `VSBOT_ITEST_BACKEND`    | `$HOME/Project/virusgame/backend`           |
//! | `VSBOT_ITEST_GAMES`      | `3` (protocol run), `1` (MCTS run)          |
//! | `VSBOT_ITEST_MCTS_MS`    | `150` — MCTS move budget                    |
//! | `VSBOT_ITEST_SOAK`       | unset — the ponder soak skips               |
//! | `VSBOT_ITEST_SOAK_GAMES` | `20`                                        |
//! | `VSBOT_ITEST_SOAK_TURN_MS` | `240` — turn budget for the soak          |
//! | `VSBOT_ITEST_SOAK_OPPONENT` | `greedy` \| `mcts`                         |
//! | `VSBOT_ITEST_GAUNTLET`   | unset — the ponder gauntlet skips           |
//! | `VSBOT_ITEST_GAUNTLET_GAMES` | `100` — split across both directions    |
//! | `VSBOT_ITEST_GAUNTLET_TURN_MS` | `600` — turn budget for the gauntlet  |
//! | `VSBOT_ITEST_AB_MS`      | `200` — alpha-beta move budget              |
//! | `VSBOT_ITEST_MP_PLAYERS` | `3` — seats in the multiplayer run (3 or 4) |
//!
//! Seven scenarios live here. The protocol run pits two instant engines against
//! each other and is about *ordering*; the MCTS run puts the real champion on
//! one side and is about the engine adapter — that a searched move survives the
//! round trip and that the domain guard never lets an out-of-domain position
//! reach the searcher's asserts.
//!
//! Three are the acceptance for bd `vsbot-3ss`, and they are about the games the
//! champion **cannot** play. `SEARCH=ALPHABETA` plays full games on 12x12 and on
//! a board the champion cannot encode; `SEARCH=MCTS` on a 16x16 board must warn
//! loudly and then play alpha-beta rather than the greedy reference engine, with
//! the warning line asserted verbatim; and a three-player lobby game exercises
//! the max^n path live, which no other scenario here can reach at all.
//!
//! ```text
//! VSBOT_ITEST=1 cargo test -p vsbot --test live_games --release \
//!   alphabeta -- --nocapture
//! VSBOT_ITEST=1 cargo test -p vsbot --test live_games --release \
//!   mcts_falls_back -- --nocapture
//! ```
//!
//! The **ponder soak** is the acceptance gate for
//! S2's T3: twenty-plus games with `VSBOT_PONDER=true` on one side, asserting
//! zero forfeits, zero illegal moves, and — the point of the exercise — zero
//! out-of-turn emissions. The **ponder gauntlet** is the acceptance gate for
//! `vsbot-gei`: the same engine with pondering on and off, both seat orders,
//! scored on the server's own verdicts. They each start their own server, so
//! [`SERIAL`] keeps them from racing for a port.
//!
//! ```text
//! VSBOT_ITEST=1 VSBOT_ITEST_SOAK=1 cargo test -p vsbot --test live_games \
//!   --release ponder_soak -- --nocapture
//!
//! VSBOT_ITEST=1 VSBOT_ITEST_GAUNTLET=1 cargo test -p vsbot --test live_games \
//!   --release ponder_on_is_not_weaker -- --nocapture
//! ```
//!
//! `VSBOT_PONDER_TRACE=1` adds one line per answered action (see
//! `vsbot::mcts`), which is how the re-root and early-stop behaviour of a run
//! is read afterwards.
//!
//! The server hard-codes `:8080` in `main.go`, which is no good for a test that
//! must not fight whatever else the developer has running. It is therefore
//! built through `go build -overlay`: a patched `main.go` that honours
//! `VSBOT_ITEST_PORT` is substituted **at build time only**, so the server
//! checkout is never modified. If the anchor line ever moves, the build falls
//! back to the stock binary on port 8080.
//!
//! Not wired into CI: GitHub Actions has no checkout of the Go server, so the
//! job would have to vendor a second repository to build it. Run it locally
//! before touching this crate.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use virus_core::{Action, State};
use virus_proto::{
    Bot, BotConfig, Diagnostics, EngineKind, GreedyEngine, Inbound, Outgoing, SearchBudget,
    SearchEngine, SearchOutcome,
};
use vsbot::{build_engine, build_mcts, MctsSettings};

const OVERALL_TIMEOUT: Duration = Duration::from_secs(180);

/// Each scenario starts its own Go server, and the fallback path when the port
/// patch does not apply is the hard-coded `:8080`. Serialising the scenarios
/// keeps that from turning into a flake — cargo runs the tests in one binary on
/// several threads by default.
static SERIAL: Mutex<()> = Mutex::new(());

/// Plays greedily, but spends its neutral placement at the first legal
/// opportunity of every game.
///
/// That is the whole point: our own `neutrals_placed` ack arrives with a
/// snapshot still showing us as mover with `movesLeft == 3`, and acting on it
/// is what forfeited two live games on 2026-08-08 (ARCHITECTURE.md invariant
/// 2). This engine forces that exact sequence into every game of the run.
#[derive(Debug, Default)]
struct NeutralOpeningEngine {
    placements: AtomicUsize,
}

impl SearchEngine for NeutralOpeningEngine {
    fn choose(&self, state: &State, budget: &SearchBudget) -> Option<SearchOutcome> {
        if state.can_place_neutrals() {
            if let Some(action) = state
                .legal_actions()
                .into_iter()
                .find(|action| matches!(action, Action::PlaceNeutrals { .. }))
            {
                self.placements.fetch_add(1, Ordering::SeqCst);
                return Some(SearchOutcome::new(action));
            }
        }
        GreedyEngine.choose(state, budget)
    }

    fn name(&self) -> &'static str {
        "neutral-opening"
    }
}

// The serialisation guard is deliberately held across the run: that is the
// whole point of it, and a blocking lock is correct here because the two test
// futures live on different runtimes and never contend from within one.
#[allow(clippy::await_holding_lock)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn two_bots_play_full_games_without_a_single_illegal_move() {
    let Some(_guard) = enabled("protocol") else {
        return;
    };
    let target_games: u64 = std::env::var("VSBOT_ITEST_GAMES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(3);

    let neutral_engine = Arc::new(NeutralOpeningEngine::default());
    let report = run_scenario(Scenario {
        label: "protocol",
        target_games,
        budget: Budget::PerAction(Duration::from_millis(50)),
        board: CHAMPION_BOARD,
        ponder: PonderSide::Neither,
        timeout: OVERALL_TIMEOUT,
        challenger: Arc::new(GreedyEngine),
        acceptor: neutral_engine.clone(),
    })
    .await;

    report.assert_clean(target_games);
    let placements = neutral_engine.placements.load(Ordering::SeqCst);
    assert!(
        placements > 0,
        "no neutral placement happened; the neutrals_placed-ack path was not exercised"
    );

    eprintln!(
        "OK: {} games, challenger sent {} actions, acceptor sent {} actions, \
         {placements} neutral placements, 0 illegal moves, 0 server errors",
        report.challenger.games_finished,
        report.challenger.actions_sent,
        report.acceptor.actions_sent,
    );
}

/// The acceptance gate for `SEARCH=MCTS`: the champion, built through the same
/// [`build_engine`] the binary calls, plays a full game against the reference
/// engine on a real server.
///
/// This is the run that catches the failure the unit tests cannot: a searched
/// action that is legal in the *searcher's* copy of the position but not in the
/// one the server holds. The server forfeits an illegal move instantly, so a
/// completed game with a clean hub log is the proof.
#[allow(clippy::await_holding_lock)] // see the note on the scenario above
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_mcts_champion_plays_a_full_game_without_a_single_illegal_move() {
    let Some(_guard) = enabled("mcts") else {
        return;
    };
    let target_games: u64 = std::env::var("VSBOT_ITEST_GAMES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1);
    let move_millis: u64 = std::env::var("VSBOT_ITEST_MCTS_MS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(150);

    // Exactly what `main` does, artifact included — a test that hand-built an
    // `MctsEngine` would prove nothing about the deployed wiring.
    let setup = build_engine(
        EngineKind::Mcts,
        &MctsSettings {
            artifact: Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../artifacts/mcts_champion.json"),
            seed: 1,
            ponder_trace: tracing(),
        },
    )
    .expect("the in-repo champion loads and validates");
    assert_eq!(setup.engine.name(), "mcts");
    eprintln!("vsbot: {}", setup.description);

    let report = run_scenario(Scenario {
        label: "mcts",
        target_games,
        budget: Budget::PerAction(Duration::from_millis(move_millis)),
        board: CHAMPION_BOARD,
        ponder: PonderSide::Neither,
        timeout: OVERALL_TIMEOUT,
        challenger: setup.engine,
        acceptor: Arc::new(GreedyEngine),
    })
    .await;

    report.assert_clean(target_games);
    eprintln!(
        "OK (MCTS): {} games at {move_millis}ms/move, mcts challenger sent {} actions, \
         greedy acceptor sent {} actions, 0 illegal moves, 0 server errors",
        report.challenger.games_finished,
        report.challenger.actions_sent,
        report.acceptor.actions_sent,
    );
}

/// The acceptance gate for `SEARCH=ALPHABETA` (bd `vsbot-3ss`): the enhanced
/// alpha-beta engine, built through the same [`build_engine`] the binary calls,
/// plays complete games on the champion's board **and** on a board the champion
/// cannot encode.
///
/// The off-12x12 half is the one that could not be run before this bead. The
/// server takes its board size from the `challenge` frame, so putting the run on
/// 16x16 is a two-field change — but until `virus-search` was wired there was no
/// engine that could answer such a game with anything but "first capture I see".
///
/// The proof is the same as every other scenario's: the hub forfeits an illegal
/// action instantly and logs it, so a completed game with a clean hub log is
/// evidence that every action the searcher produced was legal in the *server's*
/// copy of the position, on both board sizes.
#[allow(clippy::await_holding_lock)] // see the note on the scenario above
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_alphabeta_engine_plays_full_games_on_and_off_the_champion_board() {
    let Some(_guard) = enabled("alphabeta") else {
        return;
    };
    let target_games: u64 = std::env::var("VSBOT_ITEST_GAMES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1);
    let move_millis: u64 = std::env::var("VSBOT_ITEST_AB_MS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(200);

    for (label, board) in [
        ("alphabeta-12x12", CHAMPION_BOARD),
        // Deliberately not square and not 12x12: a size assumption that survived
        // 16x16 by symmetry would still fall over here.
        ("alphabeta-off-board", (16, 14)),
    ] {
        // Exactly what `main` does for `SEARCH=ALPHABETA`. Note what it does
        // *not* do: no artifact is loaded, because this engine needs none.
        let setup = build_engine(EngineKind::AlphaBeta, &MctsSettings::default())
            .expect("SEARCH=ALPHABETA needs no artifact");
        assert_eq!(setup.engine.name(), "alphabeta");
        eprintln!("vsbot: {}", setup.description);

        let report = run_scenario(Scenario {
            label,
            target_games,
            budget: Budget::PerAction(Duration::from_millis(move_millis)),
            board,
            ponder: PonderSide::Neither,
            timeout: OVERALL_TIMEOUT,
            challenger: setup.engine,
            acceptor: Arc::new(GreedyEngine),
        })
        .await;

        report.assert_clean(target_games);
        assert_board(&report, board);
        eprintln!(
            "OK (ALPHABETA {}x{}): {} games at {move_millis}ms/move, alphabeta challenger sent \
             {} actions, greedy acceptor sent {} actions, 0 illegal moves, 0 server errors",
            board.0,
            board.1,
            report.challenger.games_finished,
            report.challenger.actions_sent,
            report.acceptor.actions_sent,
        );
    }
}

/// The other half of bd `vsbot-3ss`: `SEARCH=MCTS` offered a game it cannot
/// play must **warn loudly and then play alpha-beta**, live, for a whole game.
///
/// Three separate things are asserted, because two of them used to hold while
/// the third silently did not:
///
/// * the game completes with no illegal action and no forfeit — the champion's
///   domain guard still keeps the 12x12-only searcher's asserts off the
///   blocking worker on a 16x16 board;
/// * the engine recorded exactly one degradation and its warning line names the
///   alpha-beta engine — the Java `unwiredEvalWarning` post-mortem is that a
///   quiet fallback let a run report the wrong engine's results, so the loud
///   line *is* part of the deliverable and is asserted verbatim rather than
///   assumed;
/// * every action came from the search rather than the client's pre-selected
///   fallback (`fallback_actions == 0`) and carried a real completed depth, so
///   the degraded game was genuinely played by a searcher.
#[allow(clippy::await_holding_lock)] // see the note on the scenario above
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn mcts_falls_back_to_alphabeta_on_a_board_it_cannot_encode() {
    let Some(_guard) = enabled("mcts-fallback") else {
        return;
    };
    let target_games: u64 = std::env::var("VSBOT_ITEST_GAMES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1);
    let move_millis: u64 = std::env::var("VSBOT_ITEST_AB_MS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(200);
    const BOARD: (usize, usize) = (16, 16);

    // `build_mcts` is the arm `build_engine` calls for `SEARCH=MCTS`; going
    // through it rather than assembling an engine by hand keeps this a test of
    // the deployed wiring, while keeping the concrete type so the degradation
    // bookkeeping can be read afterwards.
    let engine = build_mcts(&MctsSettings {
        artifact: Path::new(env!("CARGO_MANIFEST_DIR")).join("../../artifacts/mcts_champion.json"),
        seed: 1,
        ponder_trace: tracing(),
    })
    .expect("the in-repo champion loads and validates");
    assert_eq!(engine.degradations(), 0, "nothing has degraded yet");
    assert_eq!(engine.name(), "mcts");

    let report = run_scenario(Scenario {
        label: "mcts-fallback",
        target_games,
        budget: Budget::PerAction(Duration::from_millis(move_millis)),
        board: BOARD,
        ponder: PonderSide::Neither,
        timeout: OVERALL_TIMEOUT,
        challenger: engine.clone(),
        acceptor: Arc::new(GreedyEngine),
    })
    .await;

    report.assert_clean(target_games);
    assert_board(&report, BOARD);

    // The warning, verbatim — what an operator would have seen on stderr.
    let warning = engine
        .last_warning()
        .expect("a 16x16 game must have warned that the champion cannot play it");
    eprintln!("asserted WARNING line: {warning}");
    assert!(
        warning.starts_with("WARNING: SEARCH=MCTS cannot play this game:"),
        "{warning}"
    );
    assert!(warning.contains("2 players on a 16x16 board"), "{warning}");
    assert!(
        warning.contains("FALLING BACK TO THE ALPHA-BETA ENGINE (alphabeta)"),
        "the warning must name the engine that is actually playing: {warning}"
    );
    assert!(
        warning.contains("NOT the champion's"),
        "the warning must say the moves are not the champion's: {warning}"
    );
    assert!(
        !warning.contains("GREEDY"),
        "the greedy fallback is what this bead replaced: {warning}"
    );
    assert_eq!(
        engine.degradations(),
        1,
        "one board shape, one degradation — the warning is per transition, not per move"
    );

    // And the degraded game was played by a searcher, not by the client's
    // pre-selected fallback action.
    assert_eq!(
        report.challenger.fallback_actions, 0,
        "the fallback engine overran its budget {} times on a 16x16 board",
        report.challenger.fallback_actions
    );
    assert!(report.challenger.actions_sent > 0);

    eprintln!(
        "OK (MCTS -> ALPHABETA on {}x{}): {} games at {move_millis}ms/move, {} actions sent, \
         {} degradation(s), 0 fallback actions, 0 illegal moves, 0 server errors",
        BOARD.0,
        BOARD.1,
        report.challenger.games_finished,
        report.challenger.actions_sent,
        engine.degradations(),
    );
}

/// The acceptance gate for pondering (bd `vsbot-dgv` T3).
///
/// Twenty-plus complete games with the challenger pondering through every
/// opponent turn, against the real Go server. Pondering means a search is
/// running on positions the bot may not act in, driven off message types the
/// turn-driver whitelist excludes — so the failure it risks is precisely the one
/// that forfeited two live games on 2026-08-08 (ARCHITECTURE.md invariant 2),
/// and only a live server produces the message *ordering* that would trigger it.
///
/// Three things are asserted, and each has a distinct evidence source:
///
/// * **zero illegal moves** — the hub logs `made illegal move` before it
///   forfeits (`hub.go:210`); the server log is grepped for it;
/// * **zero out-of-turn emissions** — `handleMove` answers an off-turn action
///   with a bare `error` frame and *no* log line (`hub.go:1016`), so the
///   client's `errors` counter is the only place it can appear. Zero there is
///   the proof;
/// * **zero forfeits** — every game reaches `game_end` on both sides.
///
/// The opponent is the **instant** reference engine by default, which is the
/// harsher choice and not the lazier one: a fast opponent packs `move_made`,
/// `game_state` and `turn_change` into the tightest possible window, which is
/// exactly the ordering that would race a pondering session into emitting off
/// its own turn. `VSBOT_ITEST_SOAK_OPPONENT=mcts` puts the champion on both
/// sides instead — a slower, more realistic opponent that gives the ponder tree
/// real thinking time, at the cost of a second search burning CPU. (Strength
/// under ponder is not this test's job; that is the deferred 400-game arena.)
///
/// The turn budget is scaled down so the run finishes in minutes; the allocator
/// is proportional, so the code paths are the deployed ones.
#[allow(clippy::await_holding_lock)] // see the note on the scenario above
#[tokio::test(flavor = "multi_thread", worker_threads = 6)]
async fn ponder_soak_plays_twenty_games_without_a_forfeit_or_an_out_of_turn_action() {
    if std::env::var("VSBOT_ITEST_SOAK").as_deref() != Ok("1") {
        eprintln!("skipping the ponder soak: set VSBOT_ITEST_SOAK=1 (it takes minutes)");
        return;
    }
    let Some(_guard) = enabled("ponder-soak") else {
        return;
    };
    let target_games: u64 = std::env::var("VSBOT_ITEST_SOAK_GAMES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(20);
    let turn_millis: u64 = std::env::var("VSBOT_ITEST_SOAK_TURN_MS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(240);

    let artifact = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../artifacts/mcts_champion.json");
    let engine = |seed: u64| {
        build_engine(
            EngineKind::Mcts,
            &MctsSettings {
                artifact: artifact.clone(),
                seed,
                ponder_trace: tracing(),
            },
        )
        .expect("the in-repo champion loads and validates")
        .engine
    };

    let slow_opponent = std::env::var("VSBOT_ITEST_SOAK_OPPONENT").as_deref() == Ok("mcts");
    let acceptor: Arc<dyn SearchEngine> = if slow_opponent {
        engine(2)
    } else {
        Arc::new(GreedyEngine)
    };

    let report = run_scenario(Scenario {
        label: "ponder-soak",
        target_games,
        budget: Budget::PerTurn(Duration::from_millis(turn_millis)),
        board: CHAMPION_BOARD,
        ponder: PonderSide::Challenger,
        timeout: Duration::from_secs(1800),
        challenger: engine(1),
        acceptor,
    })
    .await;

    report.assert_clean(target_games);
    assert!(
        report.challenger.ponder_steps > 0,
        "the pondering side never received a ponder step; the soak proved nothing"
    );
    assert!(
        report.challenger.ponder_answers > 0,
        "no turn was ever answered out of the ponder tree"
    );
    assert_eq!(
        report.acceptor.ponder_steps, 0,
        "the control side must not have pondered"
    );
    assert_eq!(
        report.challenger.fallback_actions, 0,
        "the pondering side had to answer with its fallback {} times — the time manager \
         did not hold under ponder",
        report.challenger.fallback_actions
    );

    eprintln!(
        "OK (ponder soak): {} games at {turn_millis}ms/turn vs {}, ponderer sent {} actions \
         ({} ponder steps, {} pondered answers, {} fallbacks), control sent {} actions, \
         0 illegal moves, 0 server errors, 0 out-of-turn actions",
        report.challenger.games_finished,
        if slow_opponent { "mcts" } else { "greedy" },
        report.challenger.actions_sent,
        report.challenger.ponder_steps,
        report.challenger.ponder_answers,
        report.challenger.fallback_actions,
        report.acceptor.actions_sent,
    );
}

/// The strength gate for pondering (bd `vsbot-gei`): ponder-on against
/// ponder-off, same engine, same budget, refereed by the real server.
///
/// The soak above proves pondering is *safe*; this proves it is not a
/// downgrade, which is the thing the owner's canary actually failed on. Free
/// compute that loses games is worse than no compute at all.
///
/// # Why both directions
///
/// The server seats the challenger at P1 and P1 moves first on an empty board,
/// so a single-direction run would fold first-mover advantage into the number.
/// The run is therefore split in half: the ponderer challenges for the first
/// half and accepts for the second, and the two halves are pooled.
///
/// # What is asserted, and what is only reported
///
/// The clean-run properties are hard assertions. The score is reported with a
/// Wilson 95% interval and asserted only *one-sided* — the interval's upper
/// bound must reach 0.50, i.e. the run must not have demonstrated that
/// pondering is weaker. Asserting the point estimate itself would make a
/// hundred-game test a coin flip, and ARCHITECTURE.md invariant 7 is explicit
/// that strength claims want ≥400 games; the ≥0.50 acceptance number lives in
/// the bead and is read off this test's printed output, not enforced by it.
#[allow(clippy::await_holding_lock)] // see the note on the scenario above
#[tokio::test(flavor = "multi_thread", worker_threads = 6)]
async fn ponder_on_is_not_weaker_than_ponder_off_over_a_local_gauntlet() {
    if std::env::var("VSBOT_ITEST_GAUNTLET").as_deref() != Ok("1") {
        eprintln!("skipping the ponder gauntlet: set VSBOT_ITEST_GAUNTLET=1 (it takes an hour)");
        return;
    }
    let Some(_guard) = enabled("ponder-gauntlet") else {
        return;
    };
    let total_games: u64 = std::env::var("VSBOT_ITEST_GAUNTLET_GAMES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(100);
    let turn_millis: u64 = std::env::var("VSBOT_ITEST_GAUNTLET_TURN_MS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(600);
    let per_direction = total_games.div_ceil(2);

    let artifact = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../artifacts/mcts_champion.json");
    let engine = |seed: u64| {
        build_engine(
            EngineKind::Mcts,
            &MctsSettings {
                artifact: artifact.clone(),
                seed,
                ponder_trace: tracing(),
            },
        )
        .expect("the in-repo champion loads and validates")
        .engine
    };

    // Both directions, pooled. The ponderer is the challenger (P1) in the first
    // block and the acceptor (P2) in the second.
    let mut ponderer = (0.0, 0u64);
    let mut lines = Vec::new();
    for (label, side) in [
        ("ponder-gauntlet-p1", PonderSide::Challenger),
        ("ponder-gauntlet-p2", PonderSide::Acceptor),
    ] {
        let report = run_scenario(Scenario {
            label,
            target_games: per_direction,
            budget: Budget::PerTurn(Duration::from_millis(turn_millis)),
            board: CHAMPION_BOARD,
            ponder: side,
            timeout: Duration::from_secs(7_200),
            challenger: engine(1),
            acceptor: engine(2),
        })
        .await;
        report.assert_clean(per_direction);
        let (on, off) = if side == PonderSide::Challenger {
            (&report.challenger, &report.acceptor)
        } else {
            (&report.acceptor, &report.challenger)
        };
        assert!(
            on.ponder_answers > 0,
            "{label}: no turn was answered out of the ponder tree; the block proved nothing"
        );
        assert_eq!(
            off.ponder_steps, 0,
            "{label}: the control side must not have pondered"
        );
        assert_eq!(
            on.fallback_actions, 0,
            "{label}: the pondering side fell back {} times — the time manager did not hold",
            on.fallback_actions
        );
        ponderer.0 += on.score();
        ponderer.1 += on.decided();
        lines.push(format!(
            "  {label}: ponder-on {}W/{}L/{}D ({} pondered answers), ponder-off {}W/{}L/{}D",
            on.games_won,
            on.games_lost,
            on.games_drawn,
            on.ponder_answers,
            off.games_won,
            off.games_lost,
            off.games_drawn,
        ));
    }

    let (score, games) = ponderer;
    assert!(games > 0, "the gauntlet decided no games");
    let rate = score / games as f64;
    let (low, high) = wilson95(score, games);
    eprintln!(
        "OK (ponder gauntlet): ponder-on scored {score}/{games} = {rate:.3} \
         [Wilson95 {low:.3}, {high:.3}] at {turn_millis}ms/turn, both directions pooled\n{}",
        lines.join("\n")
    );
    assert!(
        high >= 0.50,
        "ponder-on scored {rate:.3} over {games} games [Wilson95 {low:.3}, {high:.3}] — the \
         interval is entirely below 0.50, so pondering is measurably WEAKER than not \
         pondering. Free compute that loses games is a regression, not a feature."
    );
}

/// Wilson 95% interval for `score` successes (halves allowed) out of `games`.
///
/// The same interval `virus_arena::stats` reports, reimplemented in four lines
/// rather than depending on the arena crate from `vsbot`'s tests: the dependency
/// direction in CLAUDE.md puts `arena` alongside `vsbot`, not beneath it.
fn wilson95(score: f64, games: u64) -> (f64, f64) {
    const Z: f64 = 1.959_963_984_540_054;
    let n = games as f64;
    let p = score / n;
    let denominator = 1.0 + Z * Z / n;
    let centre = (p + Z * Z / (2.0 * n)) / denominator;
    let spread = Z * ((p * (1.0 - p) / n + Z * Z / (4.0 * n * n)).sqrt()) / denominator;
    ((centre - spread).max(0.0), (centre + spread).min(1.0))
}

/// Whether these runs should print the per-action ponder trace.
///
/// The binary reads `VSBOT_PONDER_TRACE` in `Settings::from`; these tests build
/// their engines directly, so they read the same variable themselves rather than
/// inventing a second spelling for it.
fn tracing() -> bool {
    std::env::var("VSBOT_PONDER_TRACE").as_deref() == Ok("1")
}

/// The `VSBOT_ITEST` gate plus the cross-scenario lock, or `None` to skip.
fn enabled(label: &str) -> Option<std::sync::MutexGuard<'static, ()>> {
    if std::env::var("VSBOT_ITEST").as_deref() != Ok("1") {
        eprintln!("skipping {label}: set VSBOT_ITEST=1 to run the live-server integration tests");
        return None;
    }
    // A panicking scenario poisons the lock; the next one is still perfectly
    // able to run, so take the guard regardless.
    Some(
        SERIAL
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()),
    )
}

/// How a scenario pays for its thinking.
#[derive(Clone, Copy, Debug)]
enum Budget {
    /// A fixed per-action budget; the intra-turn allocator is off.
    PerAction(Duration),
    /// A whole-turn budget the allocator splits — what the deployment runs.
    PerTurn(Duration),
}

/// Which side of a scenario ponders.
///
/// The server always seats the **challenger at P1** (see the seat-imbalance note
/// in `crossplay.py`), so which side ponders is also which colour ponders. A
/// gauntlet that only ever pondered as the challenger would be measuring
/// first-mover advantage as much as pondering; running both directions and
/// pooling is what cancels it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum PonderSide {
    Neither,
    Challenger,
    Acceptor,
}

impl PonderSide {
    fn challenger(self) -> bool {
        self == PonderSide::Challenger
    }

    fn acceptor(self) -> bool {
        self == PonderSide::Acceptor
    }
}

/// One end-to-end run: which engines, for how many games, at what budget.
struct Scenario {
    /// Names the scenario's workdir, so concurrent runs never share a server.
    label: &'static str,
    target_games: u64,
    budget: Budget,
    /// Board the challenger asks for, as `(rows, cols)`.
    ///
    /// The server takes the size from the `challenge` frame and accepts
    /// anything from 5x5 to 50x50, so this is the whole of what it takes to put
    /// a run on a board the champion cannot encode — which is exactly the
    /// condition the alpha-beta fallback exists for, and the reason it could not
    /// be tested live before.
    board: (usize, usize),
    /// Which side ponders. The other side is the control, so one run carries
    /// both the new behaviour and its baseline.
    ponder: PonderSide,
    /// How long the whole run may take.
    timeout: Duration,
    challenger: Arc<dyn SearchEngine>,
    acceptor: Arc<dyn SearchEngine>,
}

/// The champion's board. Every scenario that is not about board size uses it.
const CHAMPION_BOARD: (usize, usize) = (12, 12);

impl Budget {
    fn apply(self, config: &mut BotConfig) {
        match self {
            Budget::PerAction(budget) => config.move_budget = Some(budget),
            Budget::PerTurn(budget) => {
                config.move_budget = None;
                config.turn_budget = budget;
            }
        }
    }
}

struct Side {
    name: &'static str,
    errors: u64,
    illegal_moves: u64,
    actions_sent: u64,
    games_finished: u64,
    games_won: u64,
    games_lost: u64,
    games_drawn: u64,
    fallback_actions: u64,
    ponder_steps: u64,
    ponder_answers: u64,
    last_error: Option<String>,
}

impl Side {
    /// Score from this side's point of view: a win is 1, a draw is a half, and
    /// draws land in the denominator (the `virus_arena::stats` convention).
    fn score(&self) -> f64 {
        self.games_won as f64 + self.games_drawn as f64 / 2.0
    }

    fn decided(&self) -> u64 {
        self.games_won + self.games_lost + self.games_drawn
    }
}

struct Report {
    challenger: Side,
    acceptor: Side,
    server_output: String,
}

impl Report {
    /// Both sides finished their games, neither was forfeited, and the hub
    /// agrees.
    fn assert_clean(&self, target_games: u64) {
        // Independent, server-side confirmation. The hub logs every rejected
        // action before it forfeits the offender, so a clean log is proof the
        // clean client counters are not just a client-side blind spot.
        assert!(
            !self.server_output.contains("made illegal move"),
            "the server logged an illegal move:\n{}",
            tail(&self.server_output)
        );
        for side in [&self.challenger, &self.acceptor] {
            assert_eq!(
                side.illegal_moves, 0,
                "{} was forfeited for an illegal move: {:?}",
                side.name, side.last_error
            );
            // An out-of-turn action is *not* an illegal move server-side: the
            // hub answers `handleMove`'s "It is not this player's turn" with a
            // plain `error` frame and no log line (hub.go:1016, :344-349). So
            // the client's error counter is the only place it can show up, and
            // a zero here is what proves no action was emitted off-turn.
            assert_eq!(
                side.errors, 0,
                "{} received {} server errors, last: {:?}",
                side.name, side.errors, side.last_error
            );
            assert!(
                side.games_finished >= target_games,
                "{} finished {} games, wanted {target_games}",
                side.name,
                side.games_finished
            );
            assert!(side.actions_sent > 0, "{} never acted", side.name);
        }
    }
}

/// Independent, server-side confirmation that the run really happened on the
/// board it asked for.
///
/// Without this a scenario could silently be measuring 12x12 forever: the hub
/// replaces any dimension outside 5..=50 with its own `defaultBoardSize` of 12
/// and says nothing to the client about having done so, so a typo in
/// `challenge_rows` would turn the whole off-board half of this bead's evidence
/// into a second 12x12 run that passes. The hub logs the accepted size on the
/// challenge it created (`hub.go:854`), which is the one place the client cannot
/// influence.
fn assert_board(report: &Report, board: (usize, usize)) {
    let expected = format!("({}x{})", board.0, board.1);
    assert!(
        report.server_output.contains(&expected),
        "the server never logged a challenge on {expected} — the run was not played on the \
         board it asked for\n--- server log tail ---\n{}",
        tail(&report.server_output)
    );
}

/// A built-and-running Go server, plus where it is writing its log.
struct RunningServer {
    child: tokio::process::Child,
    port: u16,
    log: PathBuf,
}

impl RunningServer {
    /// Kills the server and returns everything it logged.
    async fn stop(mut self) -> String {
        let _ = self.child.kill().await;
        std::fs::read_to_string(&self.log).unwrap_or_default()
    }
}

/// Builds the server from the checkout and starts it on a free port.
fn start_server(label: &str) -> RunningServer {
    let workdir = std::env::temp_dir().join(format!("vsbot-itest-{}-{label}", std::process::id()));
    std::fs::create_dir_all(&workdir).expect("temp workdir");

    let server = build_server(&workdir);
    let port = if server.port_is_configurable {
        free_port()
    } else {
        assert!(
            port_is_free(8080),
            "the stock server hard-codes :8080 and it is busy — free it, or restore the \
             `http.ListenAndServe(\":8080\", nil)` line the overlay patch anchors on"
        );
        8080
    };

    let log = workdir.join("server.log");
    let log_handle = std::fs::File::create(&log).expect("server log");
    let child = tokio::process::Command::new(&server.binary)
        .current_dir(&workdir)
        .env("VSBOT_ITEST_PORT", port.to_string())
        .stdout(Stdio::from(log_handle.try_clone().expect("clone log")))
        .stderr(Stdio::from(log_handle))
        .stdin(Stdio::null())
        .kill_on_drop(true)
        .spawn()
        .unwrap_or_else(|error| panic!("could not start {}: {error}", server.binary.display()));

    RunningServer { child, port, log }
}

/// Builds and starts a server, plays the scenario, and stops the server.
async fn run_scenario(scenario: Scenario) -> Report {
    let server = start_server(scenario.label);
    let port = server.port;

    let target_games = scenario.target_games;
    let timeout = scenario.timeout;
    let outcome = tokio::time::timeout(timeout, play(port, scenario)).await;

    let server_output = server.stop().await;

    let Ok((challenger, acceptor)) = outcome else {
        panic!(
            "the run did not finish {target_games} games within {timeout:?}\n\
             --- server log tail ---\n{}",
            tail(&server_output)
        );
    };
    Report {
        challenger,
        acceptor,
        server_output,
    }
}

async fn play(port: u16, scenario: Scenario) -> (Side, Side) {
    wait_for_port(port).await;

    let url = format!("ws://127.0.0.1:{port}/ws");

    // These runs are scaled down to hundreds of milliseconds a turn and the
    // developer box is shared, so a scheduler hiccup can be a large fraction of
    // an action's ceiling. A generous grace keeps that from reading as an engine
    // overrun; the fallback's *timing* is pinned precisely by
    // `virus-proto/tests/time_manager.rs`, where nothing else is competing.
    const ITEST_FALLBACK_GRACE: Duration = Duration::from_secs(2);

    // The challenger's timer is the sole send driver; a short interval only
    // makes the run quick, it does not change the mechanism.
    let mut challenger_config = BotConfig {
        backend_url: url.clone(),
        name_prefix: "ITestChallenger".to_owned(),
        challenger: true,
        challenge_interval: Duration::from_secs(2),
        challenge_rows: scenario.board.0,
        challenge_cols: scenario.board.1,
        rng_seed: Some(0x5EED),
        // Exactly one side ponders, so one run carries both the new behaviour
        // and its control.
        ponder: scenario.ponder.challenger(),
        ponder_budget: Duration::from_secs(5),
        fallback_grace: ITEST_FALLBACK_GRACE,
        ..BotConfig::default()
    };
    scenario.budget.apply(&mut challenger_config);
    let (challenger, mut challenger_inbox) =
        Bot::new(Arc::new(challenger_config), scenario.challenger);

    let mut acceptor_config = BotConfig {
        backend_url: url,
        name_prefix: "ITestAcceptor".to_owned(),
        ponder: scenario.ponder.acceptor(),
        ponder_budget: Duration::from_secs(5),
        fallback_grace: ITEST_FALLBACK_GRACE,
        ..BotConfig::default()
    };
    scenario.budget.apply(&mut acceptor_config);
    let (acceptor, mut acceptor_inbox) = Bot::new(Arc::new(acceptor_config), scenario.acceptor);
    let target_games = scenario.target_games;
    let timeout = scenario.timeout;

    let acceptor_task = {
        let bot = acceptor.clone();
        tokio::spawn(async move { virus_proto::run_forever(&bot, &mut acceptor_inbox).await })
    };
    // Let the acceptor register before the challenger's first tick, so its very
    // first users_update already lists a peer.
    tokio::time::sleep(Duration::from_millis(300)).await;
    let challenger_task = {
        let bot = challenger.clone();
        tokio::spawn(async move { virus_proto::run_forever(&bot, &mut challenger_inbox).await })
    };

    let started = Instant::now();
    loop {
        let done = challenger.core().counters.games_finished >= target_games
            && acceptor.core().counters.games_finished >= target_games;
        let broken = challenger.core().counters.errors > 0 || acceptor.core().counters.errors > 0;
        if done || broken {
            break;
        }
        if started.elapsed() > timeout {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    challenger_task.abort();
    acceptor_task.abort();

    (side("challenger", &challenger), side("acceptor", &acceptor))
}

fn side(name: &'static str, bot: &Bot) -> Side {
    let core = bot.core();
    Side {
        name,
        errors: core.counters.errors,
        illegal_moves: core.counters.illegal_moves,
        actions_sent: core.counters.actions_sent,
        games_finished: core.counters.games_finished,
        games_won: core.counters.games_won,
        games_lost: core.counters.games_lost,
        games_drawn: core.counters.games_drawn,
        fallback_actions: core.counters.fallback_actions,
        ponder_steps: core.counters.ponder_steps,
        ponder_answers: core.counters.ponder_answers,
        last_error: core.last_error.clone(),
    }
}

struct Server {
    binary: PathBuf,
    /// True when the overlay patch applied, so `VSBOT_ITEST_PORT` is honoured.
    port_is_configurable: bool,
}

/// The stock `main.go` line the port patch anchors on.
const LISTEN_ANCHOR: &str = r#"err := http.ListenAndServe(":8080", nil)"#;

fn build_server(workdir: &Path) -> Server {
    let backend = std::env::var("VSBOT_ITEST_BACKEND")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            Path::new(&std::env::var("HOME").expect("HOME")).join("Project/virusgame/backend")
        });
    let main_go = backend.join("main.go");
    assert!(
        main_go.exists(),
        "no Go server source at {} — set VSBOT_ITEST_BACKEND",
        backend.display()
    );

    // Substitute a port-configurable `main.go` for this build only. `-overlay`
    // is a read-only redirection inside the Go toolchain: the checkout on disk
    // is never written to.
    let source = std::fs::read_to_string(&main_go).expect("read main.go");
    let port_is_configurable = source.contains(LISTEN_ANCHOR);
    let mut command = std::process::Command::new("go");
    command.current_dir(&backend);
    if port_is_configurable {
        let patched = workdir.join("main_patched.go");
        std::fs::write(
            &patched,
            source.replace(
                LISTEN_ANCHOR,
                "addr := \":8080\"\n\tif p := os.Getenv(\"VSBOT_ITEST_PORT\"); p != \"\" {\n\
                 \t\taddr = \":\" + p\n\t}\n\terr := http.ListenAndServe(addr, nil)",
            ),
        )
        .expect("write patched main.go");
        let overlay = workdir.join("overlay.json");
        std::fs::write(
            &overlay,
            serde_json::json!({ "Replace": { main_go.to_str().expect("utf-8 path"): patched } })
                .to_string(),
        )
        .expect("write overlay.json");
        command.arg("build").arg("-overlay").arg(&overlay);
    } else {
        eprintln!("warning: main.go no longer matches the port patch anchor; using :8080");
        command.arg("build");
    }

    let binary = workdir.join("vs-server");
    let status = command
        .arg("-o")
        .arg(&binary)
        .arg(".")
        .status()
        .expect("`go build` — is the Go toolchain installed?");
    assert!(status.success(), "go build failed in {}", backend.display());
    Server {
        binary,
        port_is_configurable,
    }
}

fn port_is_free(port: u16) -> bool {
    std::net::TcpListener::bind(("127.0.0.1", port)).is_ok()
}

/// A port the kernel just handed out and immediately released. Racy in theory;
/// in a single-test process it is the standard trick and good enough.
fn free_port() -> u16 {
    std::net::TcpListener::bind(("127.0.0.1", 0))
        .expect("bind an ephemeral port")
        .local_addr()
        .expect("local address")
        .port()
}

async fn wait_for_port(port: u16) {
    let deadline = Instant::now() + Duration::from_secs(30);
    while Instant::now() < deadline {
        if tokio::net::TcpStream::connect(("127.0.0.1", port))
            .await
            .is_ok()
        {
            return;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("the server never accepted a connection on :{port}");
}

fn tail(text: &str) -> String {
    let lines: Vec<&str> = text.lines().collect();
    lines[lines.len().saturating_sub(40)..].join("\n")
}

// ---------------------------------------------------------------- multiplayer
//
// The 3-4 player half of bd `vsbot-3ss`. It needs a different shape of harness
// from the scenarios above: a 1v1 game starts from a `challenge`, which
// `virus-proto` sends, but a multiplayer game starts from a *lobby*, and hosting
// one is not something a bot does. `create_lobby`, `add_bot` and
// `start_multiplayer_game` are host frames; putting them in `virus-proto` would
// add a whole outbound vocabulary to a crate whose entire job is to play the
// games it is invited to. So the host is a raw socket in this file, and the bots
// under test join it exactly the way the deployment does — through the server's
// `bot_wanted` broadcast, which `virus-proto` already answers.

/// What each bot seat of a multiplayer run reported.
struct MultiplayerReport {
    seats: Vec<Side>,
    server_output: String,
    /// Board the host asked the lobby for.
    board: (usize, usize),
    /// Seats the server seated, host included.
    players: usize,
}

/// A three- or four-player live game, hosted by a raw socket and played out by
/// `players - 1` bots running `engine`.
///
/// The host occupies seat 1 and plays instantly: it is a referee that has to
/// hold a chair, not an opponent worth measuring. What is under test is that the
/// bots' engine produces legal actions in a max^n game against a real server —
/// the hub forfeits an illegal action instantly and logs it, and a seat that
/// stops moving is resigned, so a game that reaches `game_end` with a clean log
/// is the evidence.
async fn run_multiplayer(
    label: &'static str,
    board: (usize, usize),
    players: usize,
    move_millis: u64,
    engine: impl Fn() -> Arc<dyn SearchEngine>,
    timeout: Duration,
) -> MultiplayerReport {
    assert!(
        (3..=4).contains(&players),
        "this harness exists for the 3-4 player games; {players} is a 1v1 scenario"
    );
    let server = start_server(label);
    let port = server.port;
    let outcome = tokio::time::timeout(
        timeout,
        play_multiplayer(port, board, players, move_millis, engine),
    )
    .await;
    let server_output = server.stop().await;

    let Ok(seats) = outcome else {
        panic!(
            "the {players}-player run did not finish within {timeout:?}\n\
             --- server log tail ---\n{}",
            tail(&server_output)
        );
    };
    MultiplayerReport {
        seats,
        server_output,
        board,
        players,
    }
}

async fn play_multiplayer(
    port: u16,
    board: (usize, usize),
    players: usize,
    move_millis: u64,
    engine: impl Fn() -> Arc<dyn SearchEngine>,
) -> Vec<Side> {
    wait_for_port(port).await;
    let url = format!("ws://127.0.0.1:{port}/ws");

    // Pure acceptors: nothing here challenges anybody, and the lobby is what
    // puts them in a game. `bot_wanted` is only answered from the idle phase, so
    // they must be connected and registered before the host starts adding bots
    // — hence the settle below.
    let bots: Vec<Bot> = (0..players - 1)
        .map(|index| {
            let config = BotConfig {
                backend_url: url.clone(),
                name_prefix: format!("ITestSeat{}", index + 2),
                move_budget: Some(Duration::from_millis(move_millis)),
                fallback_grace: Duration::from_secs(2),
                ..BotConfig::default()
            };
            let (bot, mut inbox) = Bot::new(Arc::new(config), engine());
            let driver = bot.clone();
            tokio::spawn(async move { virus_proto::run_forever(&driver, &mut inbox).await });
            bot
        })
        .collect();
    tokio::time::sleep(Duration::from_millis(500)).await;

    let host = tokio::spawn(host_a_lobby(url, board, players));

    loop {
        if bots
            .iter()
            .all(|bot| bot.core().counters.games_finished >= 1)
        {
            break;
        }
        if host.is_finished() {
            // The host returns on its own `game_end`; give the bots a moment to
            // see the same message before their counters are read.
            tokio::time::sleep(Duration::from_millis(500)).await;
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    host.abort();

    bots.iter()
        .enumerate()
        // `Side::name` is a `&'static str` because the 1v1 scenarios have two
        // fixed sides; a multiplayer run has at most three bot seats, so the
        // three names it can ever need are spelled out.
        .map(|(index, bot)| side(["seat-2", "seat-3", "seat-4"][index], bot))
        .collect()
}

/// The lobby host: create, fill with bots, start, then play seat 1 instantly.
///
/// Deliberately dumb — first legal action, no search, no time budget. It exists
/// so the game has a fourth wall; a thinking host would only make the run
/// longer and would measure nothing.
async fn host_a_lobby(url: String, board: (usize, usize), players: usize) {
    use futures_util::{SinkExt, StreamExt};
    use tokio_tungstenite::tungstenite::Message as Frame;

    let (socket, _) = tokio_tungstenite::connect_async(&url)
        .await
        .expect("the host connects");
    let (mut sink, mut stream) = socket.split();

    let mut seat = 0i32;
    let mut game_id = String::new();
    let mut bots_added = 0usize;
    let mut started = false;
    // Every frame the host acts on carries a snapshot, and several frames carry
    // the *same* snapshot. Acting twice on one would send two actions for a
    // single slot of the turn, and the second would come back as the out-of-turn
    // error the 1v1 scenarios assert against. The position hash is the cheapest
    // way to say "this is the board I already answered".
    let mut answered: Option<u64> = None;

    while let Some(Ok(frame)) = stream.next().await {
        let Frame::Text(text) = frame else {
            continue;
        };
        let Ok(raw) = serde_json::from_str::<serde_json::Value>(&text) else {
            continue;
        };
        let kind = raw["type"].as_str().unwrap_or_default().to_owned();
        let outgoing: serde_json::Value = match kind.as_str() {
            "welcome" => serde_json::json!({
                "type": "create_lobby",
                "rows": board.0,
                "cols": board.1,
            }),

            // Both carry the lobby's slot occupancy. The host fills one slot at
            // a time and waits to see it filled: asking for both bots at once
            // would broadcast two `bot_wanted`s into a field of idle bots and
            // let them race for the same request id, and the server fulfils a
            // request exactly once and ignores the loser — which would strand a
            // slot and hang the run.
            "lobby_created" | "lobby_update" => {
                if started {
                    continue;
                }
                let occupied = raw["lobby"]["players"]
                    .as_array()
                    .map(|slots| {
                        slots
                            .iter()
                            .filter(|slot| slot["isEmpty"] != serde_json::Value::Bool(true))
                            .count()
                    })
                    .unwrap_or(0);
                if occupied >= players {
                    started = true;
                    serde_json::json!({ "type": "start_multiplayer_game" })
                } else if occupied == bots_added + 1 && bots_added < players - 1 {
                    bots_added += 1;
                    serde_json::json!({ "type": "add_bot" })
                } else {
                    continue;
                }
            }

            "multiplayer_game_start" | "game_state" | "turn_change" => {
                let Ok(message) = serde_json::from_value::<Inbound>(raw) else {
                    continue;
                };
                if kind == "multiplayer_game_start" {
                    seat = message.your_player;
                    game_id = message.game_id.clone();
                }
                let Some(state) = message
                    .snapshot
                    .as_ref()
                    .and_then(|snapshot| snapshot.decode().ok())
                else {
                    continue;
                };
                if i32::from(state.current_player()) != seat
                    || !state.can_act()
                    || answered == Some(state.hash())
                {
                    continue;
                }
                let Some(action) = state.legal_actions().first().copied() else {
                    continue;
                };
                answered = Some(state.hash());
                let message = Outgoing::action(
                    &game_id,
                    &uuid::Uuid::new_v4().to_string(),
                    action,
                    Diagnostics::default(),
                );
                serde_json::to_value(&message).expect("an outbound frame serialises")
            }

            "game_end" => return,
            _ => continue,
        };
        if sink
            .send(Frame::Text(outgoing.to_string().into()))
            .await
            .is_err()
        {
            return;
        }
    }
}

impl MultiplayerReport {
    /// The multiplayer analogue of [`Report::assert_clean`].
    fn assert_clean(&self) {
        assert!(
            !self.server_output.contains("made illegal move"),
            "the server logged an illegal move:\n{}",
            tail(&self.server_output)
        );
        // Independent, server-side confirmation that the game really was
        // multiplayer and really was on the board the host asked for. The hub
        // replaces an out-of-range size with its own 12x12 default silently, and
        // a lobby that started short would be a 1v1 run wearing this test's
        // name — either would turn the whole multiplayer acceptance into a
        // pass that proved nothing. `hub.go` logs both facts itself.
        let lobby = format!("{}x{})", self.board.0, self.board.1);
        assert!(
            self.server_output
                .lines()
                .any(|line| line.contains("Lobby created") && line.ends_with(&lobby)),
            "the server never logged a lobby on {}x{}\n--- server log tail ---\n{}",
            self.board.0,
            self.board.1,
            tail(&self.server_output)
        );
        let seated = format!("with {} active players", self.players);
        assert!(
            self.server_output
                .lines()
                .any(|line| line.contains("Multiplayer game created") && line.contains(&seated)),
            "the server never created a game {seated}\n--- server log tail ---\n{}",
            tail(&self.server_output)
        );
        for seat in &self.seats {
            assert_eq!(
                seat.illegal_moves, 0,
                "{} was forfeited for an illegal move: {:?}",
                seat.name, seat.last_error
            );
            assert_eq!(
                seat.errors, 0,
                "{} received {} server errors, last: {:?}",
                seat.name, seat.errors, seat.last_error
            );
            assert!(seat.actions_sent > 0, "{} never acted", seat.name);
            assert_eq!(
                seat.fallback_actions, 0,
                "{} answered with its pre-selected fallback {} times — the max^n search did \
                 not hold its budget",
                seat.name, seat.fallback_actions
            );
        }
        assert!(
            self.seats.iter().any(|seat| seat.games_finished > 0),
            "no seat ever saw the game end"
        );
    }
}

/// The live half of the multiplayer acceptance for bd `vsbot-3ss`.
///
/// `SEARCH=ALPHABETA` is the only engine in this repository that can play a
/// three- or four-player game at all — the champion's absolute-frame backup has
/// nowhere to put a third seat's win, and `MctsSearcher::new` asserts two
/// players. So this run is at once the acceptance for the standalone engine in
/// multiplayer and for the fallback the champion now uses there: either way the
/// actions come out of the same `AlphaBetaEngine::choose` and the same max^n
/// search.
///
/// The unit test `alphabeta::tests::multiplayer_positions_go_through_the_max_n_path`
/// proves the max^n *branch* is the one taken (it is the only branch in the
/// crate that can score a four-element vector); this proves the moves it returns
/// survive a real server, in a real three-player game, for a whole game.
///
/// The board is 16x16 on purpose: multiplayer *and* off the champion's board, so
/// the fallback is exercised on the hardest shape the server can offer.
#[allow(clippy::await_holding_lock)] // see the note on the scenarios above
#[tokio::test(flavor = "multi_thread", worker_threads = 6)]
async fn the_alphabeta_engine_plays_a_full_multiplayer_game() {
    let Some(_guard) = enabled("alphabeta-multiplayer") else {
        return;
    };
    let move_millis: u64 = std::env::var("VSBOT_ITEST_AB_MS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(200);
    let players: usize = std::env::var("VSBOT_ITEST_MP_PLAYERS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(3);
    const BOARD: (usize, usize) = (16, 16);

    let report = run_multiplayer(
        "alphabeta-multiplayer",
        BOARD,
        players,
        move_millis,
        || {
            build_engine(EngineKind::AlphaBeta, &MctsSettings::default())
                .expect("SEARCH=ALPHABETA needs no artifact")
                .engine
        },
        Duration::from_secs(900),
    )
    .await;

    report.assert_clean();
    eprintln!(
        "OK (ALPHABETA multiplayer): {} seats on {}x{} at {move_millis}ms/move, {} bot seats \
         saw the game end, {} actions sent, 0 illegal moves, 0 server errors, 0 fallbacks",
        report.players,
        report.board.0,
        report.board.1,
        report
            .seats
            .iter()
            .filter(|seat| seat.games_finished > 0)
            .count(),
        report
            .seats
            .iter()
            .map(|seat| seat.actions_sent)
            .sum::<u64>(),
    );
}
