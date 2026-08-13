//! The guards that make pondering safe.
//!
//! Pondering means a search is running on positions we are **not** allowed to
//! act in, driven by message types the turn-driver whitelist deliberately
//! excludes. That is a direct assault on ARCHITECTURE.md invariants 2 and 5, so
//! each guard gets a test that fails loudly if it is ever weakened:
//!
//! 1. a pondering session can never emit an action — only the authoritative turn
//!    driver, with `current == me`, ever asks it for one (invariant 2, the
//!    2026-08-08 double-forfeit);
//! 2. every accepted snapshot cancels whatever the session is thinking about
//!    (invariant 5);
//! 3. the session never holds the read loop, so the server's ping deadline is
//!    met while it thinks.

use serde_json::{json, Value};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::mpsc::UnboundedReceiver;
use virus_core::{Action, Player, State};
use virus_proto::bot::Outbound;
use virus_proto::ponder::{PonderInbox, PonderStep};
use virus_proto::{Bot, BotConfig, SearchBudget, SearchEngine, SearchOutcome};

/// One step the session was handed, as the session saw it.
#[derive(Clone, Debug)]
struct Observed {
    /// `"think"` or `"answer"`.
    kind: &'static str,
    /// Whose move it was in the position we were handed.
    mover: Player,
    /// Actions left in that position.
    moves_left: u8,
    /// The step's budget, kept so the test can check the token afterwards.
    budget: SearchBudget,
}

/// A pondering engine that records everything and thinks for a fixed span.
#[derive(Debug)]
struct PonderProbe {
    steps: Mutex<Vec<Observed>>,
    /// How long one step occupies the session's blocking worker.
    hold: Duration,
    /// Whether to declare pondering at all.
    can_ponder: bool,
    /// Whether the session should exit immediately instead of looping, so the
    /// "the session died" path can be exercised.
    quit_at_once: bool,
}

impl PonderProbe {
    fn new(hold: Duration) -> Arc<PonderProbe> {
        Arc::new(PonderProbe {
            steps: Mutex::new(Vec::new()),
            hold,
            can_ponder: true,
            quit_at_once: false,
        })
    }

    fn observed(&self) -> Vec<Observed> {
        self.steps.lock().expect("probe lock").clone()
    }

    fn of_kind(&self, kind: &str) -> Vec<Observed> {
        self.observed()
            .into_iter()
            .filter(|step| step.kind == kind)
            .collect()
    }
}

impl SearchEngine for PonderProbe {
    fn choose(&self, state: &State, _budget: &SearchBudget) -> Option<SearchOutcome> {
        state
            .legal_actions()
            .first()
            .copied()
            .map(SearchOutcome::new)
    }

    fn can_ponder(&self) -> bool {
        self.can_ponder
    }

    fn ponder(&self, inbox: &PonderInbox) {
        if self.quit_at_once {
            return;
        }
        let mut pending = inbox.next();
        while let Some(step) = pending.take() {
            let (kind, state, budget, reply) = match step {
                PonderStep::Think { state, budget } => ("think", state, budget, None),
                PonderStep::Answer {
                    state,
                    budget,
                    reply,
                } => ("answer", state, budget, Some(reply)),
            };
            self.steps.lock().expect("probe lock").push(Observed {
                kind,
                mover: state.current_player(),
                moves_left: state.moves_left(),
                budget: budget.clone(),
            });

            // Stand in for simulating: hold the worker, but poll for a newer
            // step exactly as the real session does, so a queued `Answer` is
            // never stuck behind a stale `Think`.
            let until = Instant::now() + self.hold;
            let mut interrupt = None;
            while Instant::now() < until && !budget.is_cancelled() && interrupt.is_none() {
                interrupt = inbox.try_next();
                std::thread::sleep(Duration::from_millis(2));
            }

            if let Some(reply) = reply {
                let _ = reply.send(
                    state
                        .legal_actions()
                        .first()
                        .copied()
                        .map(SearchOutcome::new),
                );
            }
            pending = interrupt.or_else(|| inbox.next());
        }
    }

    fn name(&self) -> &'static str {
        "ponder-probe"
    }
}

// ------------------------------------------------------------------ harness

fn config(ponder: bool) -> Arc<BotConfig> {
    Arc::new(BotConfig {
        turn_budget: Duration::from_millis(300),
        move_budget: Some(Duration::from_millis(30)),
        ponder,
        ponder_budget: Duration::from_secs(5),
        fallback_grace: Duration::from_millis(200),
        ..BotConfig::default()
    })
}

fn opening() -> State {
    State::new(6, 6, 2).expect("6x6 two-player board is valid")
}

/// The opening with seat 2 to move, so the bot (seat 1) is the ponderer.
fn opponent_to_move() -> State {
    let mut state = opening();
    for _ in 0..3 {
        let action = state
            .legal_actions()
            .first()
            .copied()
            .expect("seat 1 has a move");
        state = state.apply(action).expect("legal");
    }
    assert_eq!(state.current_player(), 2, "seat 2 should be to move");
    state
}

fn feed(bot: &Bot, message: &Value) {
    bot.handle_text(&serde_json::to_string(message).expect("serialises"));
}

fn snapshot_frame(kind: &str, state: &State) -> Value {
    json!({ "type": kind, "gameId": "g", "snapshot": state.snapshot() })
}

fn game_start(seat: i32, state: &State) -> Value {
    json!({
        "type": "game_start",
        "gameId": "g",
        "yourPlayer": seat,
        "snapshot": state.snapshot(),
    })
}

async fn assert_silent(inbox: &mut UnboundedReceiver<Outbound>) {
    match tokio::time::timeout(Duration::from_millis(400), inbox.recv()).await {
        Err(_) => {}
        Ok(Some(Outbound::Text { data, .. })) => panic!("a frame was queued: {data}"),
        Ok(Some(Outbound::Pong(_))) => panic!("unexpected pong"),
        Ok(None) => panic!("outbox closed"),
    }
}

async fn next_action(inbox: &mut UnboundedReceiver<Outbound>) -> Value {
    let item = tokio::time::timeout(Duration::from_secs(5), inbox.recv())
        .await
        .expect("an action should have been queued")
        .expect("outbox is open");
    match item {
        Outbound::Text { data, .. } => serde_json::from_str(&data).expect("queued frame is JSON"),
        Outbound::Pong(_) => panic!("unexpected pong"),
    }
}

/// Waits for the session to have recorded at least `count` steps.
async fn wait_for_steps(probe: &PonderProbe, count: usize) {
    let deadline = Instant::now() + Duration::from_secs(3);
    while Instant::now() < deadline {
        if probe.observed().len() >= count {
            return;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    panic!(
        "the session recorded {} steps, wanted {count}",
        probe.observed().len()
    );
}

// -------------------------------------------------------------------- tests

/// **Guard 1, the headline.** A whole opponent turn goes by — `turn_change`,
/// three `move_made`s — with the session thinking the entire time, and not one
/// frame reaches the outbox.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_pondering_session_never_emits_an_action() {
    let probe = PonderProbe::new(Duration::from_millis(30));
    let (bot, mut inbox) = Bot::new(config(true), probe.clone());

    let mut state = opponent_to_move();
    feed(&bot, &game_start(1, &state));
    feed(&bot, &snapshot_frame("turn_change", &state));

    for _ in 0..3 {
        let action = state
            .legal_actions()
            .first()
            .copied()
            .expect("the opponent has a move");
        state = state.apply(action).expect("legal");
        if state.current_player() != 2 {
            break;
        }
        feed(&bot, &snapshot_frame("move_made", &state));
    }

    assert_silent(&mut inbox).await;
    assert_eq!(bot.core().counters.actions_sent, 0);
    assert_eq!(
        bot.core().counters.ponder_answers,
        0,
        "no answer was ever requested, so none may have been produced"
    );
    assert!(
        !probe.observed().is_empty(),
        "the session must actually have run, or this test proves nothing"
    );
    assert!(
        probe.of_kind("answer").is_empty(),
        "a Think step must never turn into an Answer"
    );
}

/// **Guard 1, the specific hazard.** Our own `neutrals_placed` ack carries a
/// snapshot showing us as mover with `movesLeft > 0` — acting on it forfeited
/// two live games on 2026-08-08. Adding pondering must not open that door: the
/// ack drives neither a search nor a ponder step.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn our_own_neutrals_ack_drives_neither_a_search_nor_a_ponder_step() {
    let probe = PonderProbe::new(Duration::from_millis(10));
    let (bot, mut inbox) = Bot::new(config(true), probe.clone());

    let opponent = opponent_to_move();
    feed(&bot, &game_start(1, &opponent));
    wait_for_steps(&probe, 1).await;
    let before = probe.observed().len();

    // The ack: us as mover, three actions left — exactly what the server sends
    // back after our own placement.
    let ours = opening();
    assert_eq!(ours.current_player(), 1);
    assert_eq!(ours.moves_left(), 3);
    feed(&bot, &snapshot_frame("neutrals_placed", &ours));

    assert_silent(&mut inbox).await;
    assert_eq!(bot.core().counters.actions_sent, 0);
    assert_eq!(
        probe.observed().len(),
        before,
        "a position where we are the mover is not a position to ponder: {:?}",
        probe.observed()
    );
    assert!(probe.of_kind("answer").is_empty());
}

/// **Guard 1, the positive half.** An `Answer` step exists, but only ever for a
/// position the authoritative turn driver handed over with `current == me`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_answer_is_only_ever_requested_on_our_own_turn() {
    let probe = PonderProbe::new(Duration::from_millis(10));
    let (bot, mut inbox) = Bot::new(config(true), probe.clone());

    let opponent = opponent_to_move();
    feed(&bot, &game_start(1, &opponent));
    wait_for_steps(&probe, 1).await;

    // The opponent finishes their turn and it comes back to us.
    let mut ours = opponent.clone();
    while ours.current_player() == 2 {
        let action = ours
            .legal_actions()
            .first()
            .copied()
            .expect("the opponent has a move");
        ours = ours.apply(action).expect("legal");
    }
    feed(&bot, &snapshot_frame("turn_change", &ours));

    let message = next_action(&mut inbox).await;
    assert_eq!(message["type"], "move");
    assert_eq!(bot.core().counters.ponder_answers, 1);

    let answers = probe.of_kind("answer");
    assert_eq!(answers.len(), 1, "{answers:?}");
    for answer in &answers {
        assert_eq!(
            answer.mover, 1,
            "an Answer must only ever be asked for a position we are the mover in"
        );
        assert!(answer.moves_left > 0, "{answer:?}");
    }
    for think in probe.of_kind("think") {
        assert_ne!(
            think.mover, 1,
            "a Think step must only ever be a position somebody else moves in"
        );
    }
}

/// **Guard 2 (ARCHITECTURE.md invariant 5).** Installing a snapshot cancels
/// whatever the session is thinking about. Nothing simulates past a snapshot
/// without a fresh, explicitly re-tokened instruction.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn every_accepted_snapshot_cancels_the_step_in_flight() {
    let probe = PonderProbe::new(Duration::from_secs(2));
    let (bot, mut inbox) = Bot::new(config(true), probe.clone());

    let first = opponent_to_move();
    feed(&bot, &game_start(1, &first));
    wait_for_steps(&probe, 1).await;

    let opening_step = probe.observed()[0].clone();
    assert!(
        !opening_step.budget.is_cancelled(),
        "the session's first step starts live"
    );

    // The opponent acts: a new snapshot, a new version, and the old step is
    // worthless the instant it lands.
    let action = first
        .legal_actions()
        .first()
        .copied()
        .expect("the opponent has a move");
    let next = first.apply(action).expect("legal");
    feed(&bot, &snapshot_frame("move_made", &next));

    assert!(
        opening_step.budget.is_cancelled(),
        "the superseded step must be cancelled by the snapshot that superseded it"
    );
    wait_for_steps(&probe, 2).await;
    let steps = probe.observed();
    assert!(
        !steps[1].budget.is_cancelled(),
        "the replacement step carries a fresh token, not the cancelled one"
    );
    assert_silent(&mut inbox).await;
}

/// **Guard 2, the teardown half.** A game that ends stops the session outright.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_game_ending_cancels_and_tears_down_the_session() {
    let probe = PonderProbe::new(Duration::from_secs(2));
    let (bot, _inbox) = Bot::new(config(true), probe.clone());

    feed(&bot, &game_start(1, &opponent_to_move()));
    wait_for_steps(&probe, 1).await;
    assert!(bot.core().is_pondering());
    let step = probe.observed()[0].clone();

    feed(
        &bot,
        &json!({"type": "game_end", "gameId": "g", "winner": 2}),
    );
    assert!(!bot.core().is_pondering(), "the session must be dropped");
    assert!(
        step.budget.is_cancelled(),
        "and whatever it was thinking about cancelled"
    );
}

/// **Guard 3.** The session runs on a blocking worker, so the read loop keeps
/// decoding frames and answering pings while it thinks. The Java predecessor
/// searched inline and starved its pong deadline.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn pondering_never_holds_the_read_loop() {
    // Each step occupies the session's worker for two seconds.
    let probe = PonderProbe::new(Duration::from_secs(2));
    let (bot, mut inbox) = Bot::new(config(true), probe.clone());

    let mut state = opponent_to_move();
    feed(&bot, &game_start(1, &state));
    wait_for_steps(&probe, 1).await;

    // Now hammer the read loop while the session is stuck mid-"simulation".
    let started = Instant::now();
    for _ in 0..25 {
        let Some(action) = state.legal_actions().first().copied() else {
            break;
        };
        let Ok(next) = state.apply(action) else { break };
        state = next;
        feed(&bot, &snapshot_frame("move_made", &state));
        feed(&bot, &json!({"type": "users_update", "users": []}));
    }
    let elapsed = started.elapsed();
    assert!(
        elapsed < Duration::from_millis(500),
        "the read loop took {elapsed:?} while the session was thinking — it is being held"
    );

    // And a ping is still answered promptly.
    bot.handle_text(r#"{"type":"users_update","users":[]}"#);
    assert_silent(&mut inbox).await;
}

/// Pondering is off unless it is switched on: no session, no steps, and the bot
/// plays exactly as it did before S2. This is the canary-first default
/// (superiority.md Gate C).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn pondering_off_means_no_session_at_all() {
    let probe = PonderProbe::new(Duration::from_millis(10));
    let (bot, mut inbox) = Bot::new(config(false), probe.clone());

    let mut state = opponent_to_move();
    feed(&bot, &game_start(1, &state));
    for _ in 0..3 {
        let Some(action) = state.legal_actions().first().copied() else {
            break;
        };
        state = state.apply(action).expect("legal");
        feed(&bot, &snapshot_frame("move_made", &state));
    }

    assert_silent(&mut inbox).await;
    assert!(!bot.core().is_pondering());
    assert_eq!(bot.core().counters.ponder_steps, 0);
    assert!(probe.observed().is_empty());

    // ...and the turn still gets played, out of a fresh search.
    let mut ours = state;
    while ours.current_player() == 2 {
        let action = ours.legal_actions().first().copied().expect("a move");
        ours = ours.apply(action).expect("legal");
    }
    feed(&bot, &snapshot_frame("turn_change", &ours));
    assert_eq!(next_action(&mut inbox).await["type"], "move");
    assert_eq!(bot.core().counters.ponder_answers, 0);
}

/// An engine that does not declare `can_ponder` never gets a session, even with
/// `VSBOT_PONDER=true`. Half-wiring it would leave the client waiting out its
/// fallback timer once per turn.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_engine_that_cannot_ponder_is_never_given_a_session() {
    let probe = Arc::new(PonderProbe {
        steps: Mutex::new(Vec::new()),
        hold: Duration::from_millis(10),
        can_ponder: false,
        quit_at_once: false,
    });
    let (bot, mut inbox) = Bot::new(config(true), probe.clone());

    feed(&bot, &game_start(1, &opponent_to_move()));
    assert_silent(&mut inbox).await;
    assert!(!bot.core().is_pondering());
    assert!(probe.observed().is_empty());
}

/// A session that dies is a degraded mode, not a lost game: the turn falls back
/// to a fresh search.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_dead_session_still_lets_the_turn_be_played() {
    let probe = Arc::new(PonderProbe {
        steps: Mutex::new(Vec::new()),
        hold: Duration::from_millis(10),
        can_ponder: true,
        quit_at_once: true,
    });
    let (bot, mut inbox) = Bot::new(config(true), probe.clone());

    let opponent = opponent_to_move();
    feed(&bot, &game_start(1, &opponent));
    tokio::time::sleep(Duration::from_millis(100)).await;

    let mut ours = opponent;
    while ours.current_player() == 2 {
        let action = ours.legal_actions().first().copied().expect("a move");
        ours = ours.apply(action).expect("legal");
    }
    feed(&bot, &snapshot_frame("turn_change", &ours));

    let message = next_action(&mut inbox).await;
    assert_eq!(message["type"], "move");
    let played = Action::mv(
        message["row"].as_i64().expect("row") as i32,
        message["col"].as_i64().expect("col") as i32,
    );
    assert!(ours.legal_actions().contains(&played));
    assert_eq!(bot.core().counters.illegal_moves, 0);
}
