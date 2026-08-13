//! The invariants that cost the predecessors live games.
//!
//! Ported from Go's `cmd/bot-hoster/bot_search_test.go` (staleness,
//! cancellation, double-send) and extended with the Java post-mortem's
//! `neutrals_placed`-ack scenario, which the Go tests never covered.
//!
//! Every test drives the bot through **raw JSON frames**, so the wire
//! tolerance, the state machine and the send-time guards are all exercised on
//! the same path production takes.

use serde_json::{json, Value};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc as blocking_mpsc;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::mpsc::UnboundedReceiver;
use virus_core::cell::{Cell, CellKind};
use virus_core::{Action, Pos, Snapshot, State};
use virus_proto::bot::{ActionGuard, Outbound};
use virus_proto::{Bot, BotConfig, SearchBudget, SearchEngine, SearchOutcome};

// ------------------------------------------------------------------ engines

/// Counts calls and plays the first legal action.
#[derive(Debug, Default)]
struct CountingEngine {
    calls: AtomicUsize,
}

impl SearchEngine for CountingEngine {
    fn choose(&self, state: &State, _budget: &SearchBudget) -> Option<SearchOutcome> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        state
            .legal_actions()
            .first()
            .copied()
            .map(SearchOutcome::new)
    }
}

/// Counts `on_game_start` hooks as well as searches.
///
/// The hook is the only per-game signal the engine seam carries, and the
/// cross-play exploration wrapper reseeds its opening stream from it (bd
/// `vsbot-t3q.2`). It must fire once per *accepted* game start and never for a
/// rejected one, or a run's openings drift out of the schedule its seed names.
#[derive(Debug, Default)]
struct GameCountingEngine {
    games: AtomicUsize,
}

impl SearchEngine for GameCountingEngine {
    fn choose(&self, state: &State, _budget: &SearchBudget) -> Option<SearchOutcome> {
        state
            .legal_actions()
            .first()
            .copied()
            .map(SearchOutcome::new)
    }

    fn on_game_start(&self) {
        self.games.fetch_add(1, Ordering::SeqCst);
    }
}

/// Blocks until released, so a newer snapshot can overtake an in-flight search.
#[derive(Debug)]
struct BlockingEngine {
    started: Mutex<Option<blocking_mpsc::Sender<()>>>,
    release: Mutex<Option<blocking_mpsc::Receiver<()>>>,
}

impl BlockingEngine {
    fn new() -> (
        Arc<BlockingEngine>,
        blocking_mpsc::Receiver<()>,
        blocking_mpsc::Sender<()>,
    ) {
        let (started_tx, started_rx) = blocking_mpsc::channel();
        let (release_tx, release_rx) = blocking_mpsc::channel();
        let engine = Arc::new(BlockingEngine {
            started: Mutex::new(Some(started_tx)),
            release: Mutex::new(Some(release_rx)),
        });
        (engine, started_rx, release_tx)
    }
}

impl SearchEngine for BlockingEngine {
    fn choose(&self, state: &State, _budget: &SearchBudget) -> Option<SearchOutcome> {
        if let Some(started) = self.started.lock().unwrap().take() {
            let _ = started.send(());
        }
        if let Some(release) = self.release.lock().unwrap().take() {
            let _ = release.recv();
        }
        state
            .legal_actions()
            .first()
            .copied()
            .map(SearchOutcome::new)
    }
}

/// Always plays one specific action, so the wire conversion can be pinned.
#[derive(Debug)]
struct FixedEngine(Action);

impl SearchEngine for FixedEngine {
    fn choose(&self, _state: &State, _budget: &SearchBudget) -> Option<SearchOutcome> {
        Some(SearchOutcome {
            action: self.0,
            score: 12.5,
            depth: 7,
            nodes: 1234,
        })
    }
}

// ------------------------------------------------------------------ harness

fn config() -> Arc<BotConfig> {
    Arc::new(BotConfig {
        move_budget: Some(Duration::from_millis(50)),
        ..BotConfig::default()
    })
}

fn build(engine: Arc<dyn SearchEngine>) -> (Bot, UnboundedReceiver<Outbound>) {
    Bot::new(config(), engine)
}

fn frame(kind: &str, snapshot: &Snapshot) -> Value {
    json!({ "type": kind, "gameId": "g", "snapshot": snapshot })
}

fn game_start(seat: i32, snapshot: &Snapshot) -> Value {
    json!({
        "type": "game_start",
        "gameId": "g",
        "yourPlayer": seat,
        "opponentUsername": "peer",
        "snapshot": snapshot,
    })
}

fn feed(bot: &Bot, message: &Value) {
    bot.handle_text(&serde_json::to_string(message).expect("serialises"));
}

/// The next queued frame, or a failure if nothing arrives.
async fn next_frame(inbox: &mut UnboundedReceiver<Outbound>) -> (Value, Option<ActionGuard>) {
    let item = tokio::time::timeout(Duration::from_secs(5), inbox.recv())
        .await
        .expect("a frame should have been queued")
        .expect("outbox is open");
    match item {
        Outbound::Text { data, guard } => (
            serde_json::from_str(&data).expect("queued frame is JSON"),
            guard,
        ),
        Outbound::Pong(_) => panic!("unexpected pong"),
    }
}

/// Asserts nothing is queued within a grace window.
async fn assert_silent(inbox: &mut UnboundedReceiver<Outbound>) {
    match tokio::time::timeout(Duration::from_millis(300), inbox.recv()).await {
        Err(_) => {}
        Ok(Some(Outbound::Text { data, .. })) => panic!("unexpected frame queued: {data}"),
        Ok(Some(Outbound::Pong(_))) => panic!("unexpected pong queued"),
        Ok(None) => panic!("outbox closed"),
    }
}

/// A 6x6 two-player opening. Seat 1 to move, three actions left.
fn opening() -> State {
    State::new(6, 6, 2).expect("6x6 two-player board is valid")
}

/// The opening with two extra `Normal` cells for seat 1, so neutral placement
/// has something to consume.
fn opening_with_normals() -> Snapshot {
    let mut snapshot = opening().snapshot();
    snapshot.board[0][1] = Cell::new(1, CellKind::Normal);
    snapshot.board[1][0] = Cell::new(1, CellKind::Normal);
    snapshot
}

// -------------------------------------------------------------------- tests

/// Port of Go's `TestSameTurnSnapshotsDriveOneSequentialActionEach`.
///
/// A turn is three actions and arrives as three snapshots. The server sends
/// `move_made` *and then* `game_state` for the same mid-turn position — the
/// pair must produce exactly one action, and a verbatim repeat must produce
/// none.
#[tokio::test(flavor = "multi_thread")]
async fn same_turn_snapshots_drive_one_action_each() {
    let engine = Arc::new(CountingEngine::default());
    let (bot, mut inbox) = build(engine.clone());

    let start = opening();
    feed(&bot, &game_start(1, &start.snapshot()));
    let (first, _) = next_frame(&mut inbox).await;
    assert_eq!(first["type"], "move");

    let played = Action::mv(
        first["row"].as_i64().expect("row") as i32,
        first["col"].as_i64().expect("col") as i32,
    );
    let next = start.apply(played).expect("the chosen action is legal");
    let next_snapshot = next.snapshot();

    // `move_made` refreshes the position but must not act on its own.
    feed(
        &bot,
        &json!({
            "type": "move_made", "gameId": "g", "player": 1,
            "row": first["row"], "col": first["col"],
            "movesLeft": 2, "snapshot": next_snapshot,
        }),
    );
    assert_silent(&mut inbox).await;

    // `game_state` is the mid-turn nudge that does.
    feed(&bot, &frame("game_state", &next_snapshot));
    let (second, _) = next_frame(&mut inbox).await;
    assert_eq!(second["type"], "move");
    assert_eq!(engine.calls.load(Ordering::SeqCst), 2);

    // A verbatim repeat is a no-op: one snapshot, one action.
    feed(&bot, &frame("game_state", &next_snapshot));
    assert_silent(&mut inbox).await;
    assert_eq!(engine.calls.load(Ordering::SeqCst), 2);
}

/// Port of Go's `TestNewSnapshotCancelsStaleSearchAndPreventsDoubleSend`.
///
/// A search that finishes after its position was superseded must be dropped,
/// not sent — this is the exact shape of the stale-move forfeit.
#[tokio::test(flavor = "multi_thread")]
async fn a_late_search_result_is_dropped() {
    let (engine, started, release) = BlockingEngine::new();
    let (bot, mut inbox) = build(engine);

    let start = opening();
    feed(&bot, &game_start(1, &start.snapshot()));
    started.recv().expect("the search starts");

    let advanced = start
        .apply(start.legal_actions()[0])
        .expect("first legal action applies");
    feed(
        &bot,
        &json!({
            "type": "move_made", "gameId": "g", "player": 1,
            "row": 0, "col": 1, "snapshot": advanced.snapshot(),
        }),
    );

    release.send(()).expect("release the search");
    assert_silent(&mut inbox).await;
    assert_eq!(bot.core().counters.stale_results_dropped, 1);
    assert_eq!(bot.core().counters.actions_sent, 0);
}

/// Port of Go's `TestGameEndAndGameChangeInvalidateSearch`.
#[tokio::test(flavor = "multi_thread")]
async fn game_end_invalidates_an_in_flight_search() {
    let (engine, started, release) = BlockingEngine::new();
    let (bot, mut inbox) = build(engine);

    feed(&bot, &game_start(1, &opening().snapshot()));
    started.recv().expect("the search starts");
    feed(
        &bot,
        &json!({"type": "game_end", "gameId": "g", "winner": 2}),
    );
    release.send(()).expect("release the search");

    assert_silent(&mut inbox).await;
    assert_eq!(bot.core().counters.games_finished, 1);
    assert_eq!(bot.core().phase, virus_proto::Phase::Idle);
}

/// The per-game hook fires exactly once per accepted game start, and never for
/// a start the client rejected. bd `vsbot-t3q.2` rides on this: the exploration
/// wrapper derives one seeded opening stream per hook, so a hook that fired for
/// a rejected start would shift every later game onto the wrong seed.
#[tokio::test(flavor = "multi_thread")]
async fn the_per_game_hook_tracks_accepted_game_starts_only() {
    let engine = Arc::new(GameCountingEngine::default());
    let (bot, mut inbox) = build(engine.clone());
    assert_eq!(
        engine.games.load(Ordering::SeqCst),
        0,
        "nothing before a game"
    );

    // Seat 2 in the opening: accepted, but not our turn, so nothing is queued.
    feed(&bot, &game_start(2, &opening().snapshot()));
    assert_silent(&mut inbox).await;
    assert_eq!(engine.games.load(Ordering::SeqCst), 1);

    // Rejected starts: no snapshot, a seat outside the board, and no gameId.
    feed(
        &bot,
        &json!({"type": "game_start", "gameId": "g", "yourPlayer": 1}),
    );
    feed(&bot, &game_start(9, &opening().snapshot()));
    let mut no_id = game_start(1, &opening().snapshot());
    no_id["gameId"] = json!("");
    feed(&bot, &no_id);
    assert_eq!(
        engine.games.load(Ordering::SeqCst),
        1,
        "a rejected game start must not advance the per-game stream"
    );

    // A second real game does.
    feed(&bot, &game_start(2, &opening().snapshot()));
    assert_silent(&mut inbox).await;
    assert_eq!(engine.games.load(Ordering::SeqCst), 2);
}

/// Port of Go's `TestOldGameEndCannotCancelNewGame`.
#[tokio::test(flavor = "multi_thread")]
async fn a_stale_game_end_cannot_disturb_the_current_game() {
    let (bot, mut inbox) = build(Arc::new(CountingEngine::default()));

    // Seat 2 in the opening: not our turn, so nothing is queued.
    feed(&bot, &game_start(2, &opening().snapshot()));
    assert_silent(&mut inbox).await;

    feed(
        &bot,
        &json!({"type": "game_end", "gameId": "previous", "winner": 1}),
    );
    assert_eq!(bot.core().phase, virus_proto::Phase::InGame);
    assert_eq!(bot.core().current_game.as_deref(), Some("g"));
    assert_eq!(bot.core().counters.games_finished, 0);
}

/// **ARCHITECTURE.md invariant 2.** Two live forfeits on 2026-08-08.
///
/// Our own `neutrals_placed` ack carries the pre-`endTurn` snapshot, which
/// still shows us as the mover with `movesLeft == 3`. Acting on it fires a move
/// the server has not asked for. Nothing may be emitted until `turn_change`.
#[tokio::test(flavor = "multi_thread")]
async fn our_own_neutrals_ack_never_emits_an_action() {
    let engine = Arc::new(CountingEngine::default());
    let (bot, mut inbox) = build(engine.clone());

    // Join with the opponent to move, so the game start itself is quiet.
    let mut waiting = opening_with_normals();
    waiting.current_player = 2;
    feed(&bot, &game_start(1, &waiting));
    assert_silent(&mut inbox).await;

    // The ack: our two `Normal` cells are now `Neutral`, our neutral is spent —
    // and the server has *not* yet rotated the turn.
    let mut ack = opening_with_normals();
    ack.board[0][1] = Cell::NEUTRAL;
    ack.board[1][0] = Cell::NEUTRAL;
    ack.neutral_used[0] = true;
    ack.current_player = 1;
    ack.moves_left = 3;
    assert_eq!(ack.decode().expect("ack decodes").current_player(), 1);

    feed(
        &bot,
        &json!({
            "type": "neutrals_placed", "gameId": "g", "player": 1,
            "cells": [{"row": 0, "col": 1}, {"row": 1, "col": 0}],
            "snapshot": ack,
        }),
    );
    assert_silent(&mut inbox).await;
    assert_eq!(
        engine.calls.load(Ordering::SeqCst),
        0,
        "the ack must not even start a search"
    );
    // It was still absorbed: the position is authoritative, just not actionable.
    assert!(bot.core().position().is_some_and(|p| p.neutral_used(1)));

    // `turn_change` is the authoritative turn driver — now, and only now.
    feed(&bot, &frame("turn_change", &ack));
    let (action, _) = next_frame(&mut inbox).await;
    assert_eq!(action["type"], "move");
    assert_eq!(engine.calls.load(Ordering::SeqCst), 1);
}

/// The same hazard class: `action_ack` is our own message coming back, carrying
/// a snapshot that can show us as the mover. It refreshes, it never drives.
#[tokio::test(flavor = "multi_thread")]
async fn our_own_action_ack_never_emits_an_action() {
    let engine = Arc::new(CountingEngine::default());
    let (bot, mut inbox) = build(engine.clone());

    let mut waiting = opening().snapshot();
    waiting.current_player = 2;
    feed(&bot, &game_start(1, &waiting));
    assert_silent(&mut inbox).await;

    let mut ours = opening().snapshot();
    ours.current_player = 1;
    feed(
        &bot,
        &json!({
            "type": "action_ack", "gameId": "g",
            "requestId": "1c9a1b6e-0000-4000-8000-000000000000",
            "movesLeft": 3, "player": 1, "snapshot": ours,
        }),
    );
    assert_silent(&mut inbox).await;
    assert_eq!(engine.calls.load(Ordering::SeqCst), 0);
}

/// The writer's last gate. Between the search task's check and the wire, a
/// snapshot can land; the guard must reject the action rather than let a move
/// for a dead position through.
#[tokio::test(flavor = "multi_thread")]
async fn the_send_time_guard_rejects_a_superseded_action() {
    let (bot, mut inbox) = build(Arc::new(CountingEngine::default()));

    let start = opening();
    feed(&bot, &game_start(1, &start.snapshot()));
    let (_, guard) = next_frame(&mut inbox).await;
    let guard = guard.expect("actions carry a send-time guard");

    // Still valid the instant it was queued.
    assert!(bot.core().action_still_valid(&guard));

    let advanced = start
        .apply(start.legal_actions()[0])
        .expect("first legal action applies");
    feed(
        &bot,
        &json!({
            "type": "move_made", "gameId": "g", "player": 1,
            "row": 0, "col": 1, "snapshot": advanced.snapshot(),
        }),
    );
    assert!(
        !bot.core().action_still_valid(&guard),
        "a newer snapshot must invalidate the queued action"
    );

    // ...and so must losing the game, the seat, or the socket.
    bot.on_disconnected();
    assert!(!bot.core().action_still_valid(&guard));
}

/// Wire conversion, mirroring Go's `TestActionMessageConversion`: a move sends
/// `{row,col}`, a neutral placement sends two cells — and every action carries
/// a fresh UUID `requestId`, which neither predecessor sent.
#[tokio::test(flavor = "multi_thread")]
async fn actions_convert_to_the_documented_frames() {
    // Move.
    let (bot, mut inbox) = build(Arc::new(FixedEngine(Action::mv(0, 1))));
    feed(&bot, &game_start(1, &opening().snapshot()));
    let (message, _) = next_frame(&mut inbox).await;
    assert_eq!(message["type"], "move");
    assert_eq!(message["gameId"], "g");
    assert_eq!(message["row"], 0);
    assert_eq!(message["col"], 1);
    assert_eq!(message["score"], 12.5);
    assert_eq!(message["depth"], 7);
    assert_eq!(message["nodesEvaluated"], 1234);
    assert!(message["timeMs"].is_i64());
    let first_id = message["requestId"].as_str().expect("requestId").to_owned();
    assert_eq!(first_id.len(), 36, "a hyphenated UUID: {first_id}");

    // Neutral placement.
    let cells = [Pos::new(0, 1), Pos::new(1, 0)];
    let (bot, mut inbox) = build(Arc::new(FixedEngine(Action::neutrals(cells[0], cells[1]))));
    feed(&bot, &game_start(1, &opening_with_normals()));
    let (message, _) = next_frame(&mut inbox).await;
    assert_eq!(message["type"], "neutrals");
    assert_eq!(message["gameId"], "g");
    assert_eq!(
        message["cells"],
        json!([{"row": 0, "col": 1}, {"row": 1, "col": 0}])
    );
    assert!(
        message["row"].is_null(),
        "a neutrals frame carries no target"
    );
    let second_id = message["requestId"].as_str().expect("requestId").to_owned();
    assert_ne!(first_id, second_id, "request ids are never recycled");
}

/// An action the engine offers but the live position rejects is never sent —
/// the server forfeits the game for an illegal move, so this is the last line
/// of defence against an engine bug.
#[tokio::test(flavor = "multi_thread")]
async fn an_illegal_engine_choice_is_never_queued() {
    // (5,5) is seat 2's base corner: unreachable from seat 1's component and
    // not an empty cell, so it can never be a legal seat-1 move.
    let (bot, mut inbox) = build(Arc::new(FixedEngine(Action::mv(5, 5))));
    feed(&bot, &game_start(1, &opening().snapshot()));
    assert_silent(&mut inbox).await;
    assert_eq!(bot.core().counters.stale_results_dropped, 1);
}

/// Both wire dialects decode to the same position and produce the same action:
/// the Go server writes PascalCase keys with numeric kinds, the fixtures write
/// lowercase keys with named kinds, and unknown fields are ignored so a
/// server-side addition cannot break a live game.
#[tokio::test(flavor = "multi_thread")]
async fn tolerant_snapshot_dialects_agree() {
    let canonical = {
        let (bot, mut inbox) = build(Arc::new(CountingEngine::default()));
        feed(&bot, &game_start(1, &opening().snapshot()));
        let (message, _) = next_frame(&mut inbox).await;
        (message["row"].clone(), message["col"].clone())
    };

    for dialect in [pascal_case_opening(), lowercase_opening_with_junk()] {
        let (bot, mut inbox) = build(Arc::new(CountingEngine::default()));
        bot.handle_text(&dialect);
        let (message, _) = next_frame(&mut inbox).await;
        assert_eq!(message["type"], "move");
        assert_eq!((message["row"].clone(), message["col"].clone()), canonical);
    }
}

/// A snapshot the rules engine rejects must drop the position (never leave a
/// stale one in place to be played from) and ask the server to resync.
#[tokio::test(flavor = "multi_thread")]
async fn a_rejected_snapshot_drops_the_position_and_resyncs() {
    let (bot, mut inbox) = build(Arc::new(CountingEngine::default()));

    let mut waiting = opening().snapshot();
    waiting.current_player = 2;
    feed(&bot, &game_start(1, &waiting));
    assert_silent(&mut inbox).await;
    assert!(bot.core().position().is_some());

    // `movesLeft` above the per-turn cap: rejected by `Snapshot::decode`.
    let mut broken = opening().snapshot();
    broken.moves_left = 9;
    feed(&bot, &frame("turn_change", &broken));

    let (message, guard) = next_frame(&mut inbox).await;
    assert_eq!(message["type"], "resync");
    assert_eq!(message["gameId"], "g");
    assert!(guard.is_none(), "a resync is not a game action");
    assert!(bot.core().position().is_none(), "the position is dropped");
    assert_eq!(bot.core().counters.rejected_snapshots, 1);
    assert!(!bot.core().may_act());
}

/// Challenger mode is timer-driven and nothing else. Reacting to
/// `users_update` is what spammed the lobby in the Java predecessor.
#[tokio::test(flavor = "multi_thread")]
async fn users_update_never_sends_a_challenge() {
    let config = Arc::new(BotConfig {
        challenger: true,
        challenge_interval: Duration::from_secs(3600),
        rng_seed: Some(1),
        ..BotConfig::default()
    });
    let (bot, mut inbox) = Bot::new(config, Arc::new(CountingEngine::default()));

    feed(
        &bot,
        &json!({"type": "welcome", "userId": "me", "username": "Bot A"}),
    );
    for _ in 0..5 {
        feed(
            &bot,
            &json!({"type": "users_update", "users": [
                {"userId": "peer", "username": "Bot B", "inGame": false, "inLobby": false},
                {"userId": "me", "username": "Bot A", "inGame": false, "inLobby": false},
            ]}),
        );
    }
    assert_silent(&mut inbox).await;
    assert_eq!(bot.core().counters.challenges_sent, 0);
    assert_eq!(bot.core().peers.len(), 2);
}

/// A busy bot declines rather than double-booking itself.
#[tokio::test(flavor = "multi_thread")]
async fn a_busy_bot_declines_a_challenge() {
    let (bot, mut inbox) = build(Arc::new(CountingEngine::default()));

    let mut waiting = opening().snapshot();
    waiting.current_player = 2;
    feed(&bot, &game_start(1, &waiting));
    feed(
        &bot,
        &json!({"type": "challenge_received", "challengeId": "c1", "fromUsername": "peer"}),
    );

    let (message, _) = next_frame(&mut inbox).await;
    assert_eq!(message["type"], "decline_challenge");
    assert_eq!(message["challengeId"], "c1");
}

/// A bot that has accepted a challenge is busy *before* `game_start` arrives.
///
/// The server's `handleAcceptChallenge` has no in-game guard: a second accept
/// creates a second game and overwrites our `gameId`, abandoning the first to
/// time out. So the accept itself must claim the bot.
#[tokio::test(flavor = "multi_thread")]
async fn a_second_challenge_before_game_start_is_declined() {
    let (bot, mut inbox) = build(Arc::new(CountingEngine::default()));
    feed(
        &bot,
        &json!({"type": "welcome", "userId": "me", "username": "Bot A"}),
    );

    feed(
        &bot,
        &json!({"type": "challenge_received", "challengeId": "first"}),
    );
    let (accepted, _) = next_frame(&mut inbox).await;
    assert_eq!(accepted["type"], "accept_challenge");
    assert_eq!(accepted["challengeId"], "first");

    // No `game_start` yet — and a second challenge lands.
    feed(
        &bot,
        &json!({"type": "challenge_received", "challengeId": "second"}),
    );
    let (declined, _) = next_frame(&mut inbox).await;
    assert_eq!(declined["type"], "decline_challenge");
    assert_eq!(declined["challengeId"], "second");
}

/// ...but an accept the server never honours must not wedge the bot out of the
/// pool for the life of the process.
#[tokio::test(flavor = "multi_thread")]
async fn an_accepted_game_that_never_starts_heals_back_to_idle() {
    let config = Arc::new(BotConfig {
        pending_game_grace: Duration::from_millis(50),
        ..BotConfig::default()
    });
    let (bot, mut inbox) = Bot::new(config, Arc::new(CountingEngine::default()));
    feed(
        &bot,
        &json!({"type": "welcome", "userId": "me", "username": "Bot A"}),
    );
    feed(
        &bot,
        &json!({"type": "challenge_received", "challengeId": "dropped"}),
    );
    let (accepted, _) = next_frame(&mut inbox).await;
    assert_eq!(accepted["type"], "accept_challenge");
    assert_eq!(bot.core().phase, virus_proto::Phase::InGame);

    tokio::time::sleep(Duration::from_millis(120)).await;
    // Any subsequent traffic heals the optimistic flag.
    feed(&bot, &json!({"type": "users_update", "users": []}));
    assert_eq!(bot.core().phase, virus_proto::Phase::Idle);

    // And the bot is available again.
    feed(
        &bot,
        &json!({"type": "challenge_received", "challengeId": "next"}),
    );
    let (message, _) = next_frame(&mut inbox).await;
    assert_eq!(message["type"], "accept_challenge");
}

/// An idle bot accepts a challenge.
#[tokio::test(flavor = "multi_thread")]
async fn an_idle_bot_accepts_a_challenge() {
    let (bot, mut inbox) = build(Arc::new(CountingEngine::default()));
    feed(
        &bot,
        &json!({"type": "welcome", "userId": "me", "username": "Bot A"}),
    );
    feed(
        &bot,
        &json!({"type": "challenge_received", "challengeId": "c1"}),
    );
    let (message, _) = next_frame(&mut inbox).await;
    assert_eq!(message["type"], "accept_challenge");
    assert_eq!(message["challengeId"], "c1");
}

/// An idle bot answers `bot_wanted` by joining the named lobby, echoing the
/// server's correlation id.
#[tokio::test(flavor = "multi_thread")]
async fn an_idle_bot_joins_a_wanted_lobby() {
    let (bot, mut inbox) = build(Arc::new(CountingEngine::default()));
    feed(
        &bot,
        &json!({"type": "welcome", "userId": "me", "username": "Bot A"}),
    );
    feed(
        &bot,
        &json!({"type": "bot_wanted", "lobbyId": "L", "requestId": "r1"}),
    );
    let (message, _) = next_frame(&mut inbox).await;
    assert_eq!(message["type"], "join_lobby");
    assert_eq!(message["lobbyId"], "L");
    assert_eq!(message["requestId"], "r1");

    // Once in the lobby it is no longer available for a challenge.
    feed(
        &bot,
        &json!({"type": "lobby_joined", "lobby": {"lobbyId": "L"}}),
    );
    assert_eq!(bot.core().phase, virus_proto::Phase::InLobby);
    feed(
        &bot,
        &json!({"type": "challenge_received", "challengeId": "c1"}),
    );
    let (message, _) = next_frame(&mut inbox).await;
    assert_eq!(message["type"], "decline_challenge");
}

/// Unknown message types and undecodable frames are survivable: one bad frame
/// must never take a bot out of a live game.
#[tokio::test(flavor = "multi_thread")]
async fn unknown_and_malformed_frames_are_survivable() {
    let (bot, mut inbox) = build(Arc::new(CountingEngine::default()));

    let mut waiting = opening().snapshot();
    waiting.current_player = 2;
    feed(&bot, &game_start(1, &waiting));

    bot.handle_text("{ this is not json");
    bot.handle_text(r#"{"type":"chat_message","content":"hello"}"#);
    bot.handle_text(r#"{"type":"turn_change","gameId":"g"}"#); // snapshot missing
    assert_silent(&mut inbox).await;
    assert_eq!(bot.core().phase, virus_proto::Phase::InGame);
    assert!(bot.core().position().is_some());
}

// ------------------------------------------------------------ wire dialects

/// The Go server's dialect: PascalCase keys, numeric cell kinds.
fn pascal_case_opening() -> String {
    let state = opening();
    let mut rows = Vec::new();
    for row in 0..state.rows() {
        let cells: Vec<String> = (0..state.cols())
            .map(|col| {
                let cell = state.at(Pos::new(row as i32, col as i32));
                format!(
                    r#"{{"Owner":{},"Kind":{}}}"#,
                    cell.owner(),
                    cell.kind().as_u8()
                )
            })
            .collect();
        rows.push(format!("[{}]", cells.join(",")));
    }
    format!(
        r#"{{"type":"game_start","gameId":"g","yourPlayer":1,"snapshot":{{
             "Rows":6,"Cols":6,"Board":[{}],
             "Bases":[{{"Row":0,"Col":0}},{{"Row":5,"Col":5}}],
             "Active":[true,true],"NeutralUsed":[false,false],
             "CurrentPlayer":1,"MovesLeft":3,"GameOver":false,"Winner":0}}}}"#,
        rows.join(",")
    )
}

/// The fixture dialect: lowercase keys, named kinds — plus fields neither
/// producer emits today, which must be ignored rather than rejected.
fn lowercase_opening_with_junk() -> String {
    let snapshot = serde_json::to_value(opening().snapshot()).expect("serialises");
    let mut message = json!({
        "type": "game_start",
        "gameId": "g",
        "yourPlayer": 1,
        "snapshot": snapshot,
        "someFutureField": {"nested": [1, 2, 3]},
    });
    message["snapshot"]["turnCount"] = json!(17);
    message["snapshot"]["board"][0][0]["unknown"] = json!("ignored");
    message.to_string()
}
