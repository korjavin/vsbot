//! The intra-turn time manager, driven through the real state machine.
//!
//! `virus_proto::clock`'s own unit tests pin the *rules* — how a turn divides,
//! when a search stops early, when it extends. These pin the *wiring*: that the
//! numbers the allocator produces are the numbers the engine is handed, that
//! spending is recorded per action, and that an engine which blows through its
//! budget answers with the fallback chosen before the search rather than losing
//! the game on the server's timer.
//!
//! Everything here runs on a scaled-down turn budget (600 ms rather than the
//! deployed 12 s) so the suite stays fast. The allocator is proportional, so the
//! ratios under test are the deployed ones.

use serde_json::{json, Value};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::mpsc::UnboundedReceiver;
use virus_core::{Action, State};
use virus_proto::bot::Outbound;
use virus_proto::{Bot, BotConfig, SearchBudget, SearchEngine, SearchOutcome};

/// The scaled-down turn budget. 600 ms divides as 300 / 180 / 120.
const TURN: Duration = Duration::from_millis(600);

/// What one action's budget looked like from inside the engine.
#[derive(Clone, Copy, Debug)]
struct Seen {
    /// `deadline - now`, i.e. what is left of the target when the engine starts.
    target: Duration,
    /// `ceiling - deadline`: the extension room the allocator granted.
    extension: Duration,
}

/// Records the budget it is given, then spends exactly `spend_ratio` of it.
///
/// A ratio of 1.0 models an engine that uses its whole share; 0.0 models a root
/// that settled immediately and hands its remainder back.
#[derive(Debug)]
struct BudgetProbe {
    seen: Mutex<Vec<Seen>>,
    spend_whole_target: bool,
}

impl BudgetProbe {
    fn new(spend_whole_target: bool) -> Arc<BudgetProbe> {
        Arc::new(BudgetProbe {
            seen: Mutex::new(Vec::new()),
            spend_whole_target,
        })
    }

    fn seen(&self) -> Vec<Seen> {
        self.seen.lock().expect("probe lock").clone()
    }
}

impl SearchEngine for BudgetProbe {
    fn choose(&self, state: &State, budget: &SearchBudget) -> Option<SearchOutcome> {
        let now = Instant::now();
        self.seen.lock().expect("probe lock").push(Seen {
            target: budget.deadline.saturating_duration_since(now),
            extension: budget.ceiling.saturating_duration_since(budget.deadline),
        });
        if self.spend_whole_target {
            // A blocking worker, so sleeping here is exactly what a real search
            // burning its budget looks like to the client.
            std::thread::sleep(budget.deadline.saturating_duration_since(Instant::now()));
        }
        state
            .legal_actions()
            .first()
            .copied()
            .map(SearchOutcome::new)
    }

    fn name(&self) -> &'static str {
        "budget-probe"
    }
}

/// Never answers, so the client's hard deadline is the only thing that can end
/// the action. Its `fallback` is a specific legal action the test can identify.
#[derive(Debug)]
struct StuckEngine {
    fallback: Action,
}

impl SearchEngine for StuckEngine {
    fn choose(&self, _state: &State, _budget: &SearchBudget) -> Option<SearchOutcome> {
        // Deliberately ignores both the deadline and the cancellation token:
        // this is the engine bug the fallback exists to survive.
        std::thread::sleep(Duration::from_secs(2));
        None
    }

    fn fallback(&self, _state: &State) -> Option<Action> {
        Some(self.fallback)
    }

    fn name(&self) -> &'static str {
        "stuck"
    }
}

// ------------------------------------------------------------------ harness

fn opening() -> State {
    State::new(6, 6, 2).expect("6x6 two-player board is valid")
}

fn feed(bot: &Bot, message: &Value) {
    bot.handle_text(&serde_json::to_string(message).expect("serialises"));
}

fn game_start(seat: i32, state: &State) -> Value {
    json!({
        "type": "game_start",
        "gameId": "g",
        "yourPlayer": seat,
        "snapshot": state.snapshot(),
    })
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

/// Reads the queued action, applies it, and pushes the resulting position back
/// as the server's mid-turn `game_state` nudge.
async fn play_one(bot: &Bot, inbox: &mut UnboundedReceiver<Outbound>, state: State) -> State {
    let message = next_action(inbox).await;
    assert_eq!(message["type"], "move", "expected a move, got {message}");
    let action = Action::mv(
        message["row"].as_i64().expect("row") as i32,
        message["col"].as_i64().expect("col") as i32,
    );
    let next = state.apply(action).expect("the chosen action is legal");
    feed(
        bot,
        &json!({ "type": "game_state", "gameId": "g", "snapshot": next.snapshot() }),
    );
    next
}

// -------------------------------------------------------------------- tests

/// The whole turn is divided, largest share first, and every action's budget is
/// smaller than the last — the shape superiority.md §2b asks for.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_turn_is_divided_across_its_three_actions_largest_first() {
    let probe = BudgetProbe::new(true);
    let config = Arc::new(BotConfig {
        turn_budget: TURN,
        move_budget: None,
        ..BotConfig::default()
    });
    let (bot, mut inbox) = Bot::new(config, probe.clone());

    let mut state = opening();
    feed(&bot, &game_start(1, &state));
    for _ in 0..3 {
        state = play_one(&bot, &mut inbox, state).await;
    }

    let seen = probe.seen();
    assert!(seen.len() >= 3, "three actions, three budgets: {seen:?}");
    let [first, second, third] = [seen[0], seen[1], seen[2]];

    // 50 / 30 / 20 of 600 ms, minus the sliver spent selecting the fallback.
    assert!(
        (250..=300).contains(&(first.target.as_millis() as u64)),
        "action 1 should get ~300ms of a 600ms turn, got {first:?}"
    );
    assert!(
        (140..=200).contains(&(second.target.as_millis() as u64)),
        "action 2 should get ~180ms, got {second:?}"
    );
    assert!(
        (80..=140).contains(&(third.target.as_millis() as u64)),
        "action 3 should get ~120ms, got {third:?}"
    );
    assert!(
        first.target > second.target && second.target > third.target,
        "the split must be uneven and front-loaded: {seen:?}"
    );

    // Extension room exists, and it shrinks to nothing by the last action —
    // there is no rest of the turn left to borrow from.
    assert!(first.extension > Duration::ZERO, "{first:?}");
    assert!(
        third.extension < first.extension,
        "the last action has nothing left to borrow: {seen:?}"
    );
}

/// A root that settles instantly hands its remainder to the next action instead
/// of forfeiting it. That is the entire reason the allocator banks rather than
/// dividing up front.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_stable_root_releases_its_remainder_to_the_next_action() {
    let probe = BudgetProbe::new(false);
    let config = Arc::new(BotConfig {
        turn_budget: TURN,
        move_budget: None,
        ..BotConfig::default()
    });
    let (bot, mut inbox) = Bot::new(config, probe.clone());

    let mut state = opening();
    feed(&bot, &game_start(1, &state));
    for _ in 0..2 {
        state = play_one(&bot, &mut inbox, state).await;
    }

    let seen = probe.seen();
    assert!(seen.len() >= 2, "{seen:?}");
    assert!(
        seen[1].target > seen[0].target,
        "action 1 spent almost nothing, so action 2 must get more than action 1 did, \
         not the 180ms a fixed split would give it: {seen:?}"
    );
    assert!(
        (300..=400).contains(&(seen[1].target.as_millis() as u64)),
        "action 2 should get ~360ms (60% of the ~600ms still banked): {seen:?}"
    );
}

/// `VSBOT_MOVE_MILLIS` is an override, not a hint: every action gets exactly it,
/// and nothing is banked, released or extended.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_per_action_override_gives_every_action_the_same_budget() {
    let probe = BudgetProbe::new(true);
    let config = Arc::new(BotConfig {
        turn_budget: TURN,
        move_budget: Some(Duration::from_millis(100)),
        ..BotConfig::default()
    });
    let (bot, mut inbox) = Bot::new(config, probe.clone());

    let mut state = opening();
    feed(&bot, &game_start(1, &state));
    for _ in 0..3 {
        state = play_one(&bot, &mut inbox, state).await;
    }

    for seen in probe.seen().iter().take(3) {
        assert!(
            (60..=100).contains(&(seen.target.as_millis() as u64)),
            "the override is exact, got {seen:?}"
        );
        assert_eq!(
            seen.extension,
            Duration::ZERO,
            "an override has no extension room: {seen:?}"
        );
    }
}

/// A new turn reopens the bank. Without this the second turn would be played on
/// whatever the first one left behind.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_next_turn_starts_from_a_full_budget() {
    let probe = BudgetProbe::new(true);
    let config = Arc::new(BotConfig {
        turn_budget: TURN,
        move_budget: None,
        ..BotConfig::default()
    });
    let (bot, mut inbox) = Bot::new(config, probe.clone());

    let mut state = opening();
    feed(&bot, &game_start(1, &state));
    for _ in 0..3 {
        state = play_one(&bot, &mut inbox, state).await;
    }
    assert_ne!(
        state.current_player(),
        1,
        "three actions must have ended our turn"
    );

    // The opponent plays a full turn, then it is ours again.
    for _ in 0..3 {
        let action = state
            .legal_actions()
            .first()
            .copied()
            .expect("the opponent has a move");
        state = state.apply(action).expect("legal");
        feed(
            &bot,
            &json!({ "type": "move_made", "gameId": "g", "snapshot": state.snapshot() }),
        );
    }
    assert_eq!(state.current_player(), 1, "the turn came back to us");
    feed(
        &bot,
        &json!({ "type": "turn_change", "gameId": "g", "snapshot": state.snapshot() }),
    );
    let _ = next_action(&mut inbox).await;

    let seen = probe.seen();
    assert!(seen.len() >= 4, "{seen:?}");
    assert!(
        (250..=300).contains(&(seen[3].target.as_millis() as u64)),
        "the second turn's first action must get a full 50% share again, got {:?}",
        seen[3]
    );
}

/// Fallback-first discipline (superiority.md §2b, the MCTS analogue of
/// ARCHITECTURE.md invariant 3).
///
/// An engine that ignores its deadline **and** its cancellation token is the
/// worst case the rule exists for: without a fallback the bot simply stops
/// moving and loses on the server's 120 s timer. With one, it plays the action
/// it picked before the search started.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_engine_that_overruns_answers_with_the_fallback_it_picked_first() {
    let state = opening();
    let chosen = *state
        .legal_actions()
        .last()
        .expect("the opening has legal moves");
    let Action::Move { target } = chosen else {
        panic!("the opening's last legal action should be a move");
    };

    let config = Arc::new(BotConfig {
        turn_budget: Duration::from_millis(120),
        move_budget: None,
        fallback_grace: Duration::from_millis(80),
        ..BotConfig::default()
    });
    let (bot, mut inbox) = Bot::new(config, Arc::new(StuckEngine { fallback: chosen }));

    let started = Instant::now();
    feed(&bot, &game_start(1, &state));
    let message = next_action(&mut inbox).await;

    assert_eq!(message["type"], "move");
    assert_eq!(message["row"], i64::from(target.row));
    assert_eq!(message["col"], i64::from(target.col));
    assert!(
        started.elapsed() < Duration::from_secs(2),
        "the client waited for the stuck engine instead of answering with the fallback"
    );

    let counters = bot.core().counters;
    assert_eq!(counters.actions_sent, 1);
    assert_eq!(
        counters.fallback_actions, 1,
        "the overrun must be counted, so a deployment can see the time manager slipping"
    );
    assert_eq!(counters.illegal_moves, 0);
}

/// The fallback is legal, which is the only property that makes it usable.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_default_fallback_is_a_legal_action() {
    #[derive(Debug)]
    struct Silent;
    impl SearchEngine for Silent {
        fn choose(&self, _state: &State, _budget: &SearchBudget) -> Option<SearchOutcome> {
            std::thread::sleep(Duration::from_secs(2));
            None
        }
    }

    let state = opening();
    let config = Arc::new(BotConfig {
        turn_budget: Duration::from_millis(120),
        move_budget: None,
        fallback_grace: Duration::from_millis(80),
        ..BotConfig::default()
    });
    let (bot, mut inbox) = Bot::new(config, Arc::new(Silent));

    feed(&bot, &game_start(1, &state));
    let message = next_action(&mut inbox).await;
    let played = Action::mv(
        message["row"].as_i64().expect("row") as i32,
        message["col"].as_i64().expect("col") as i32,
    );
    assert!(
        state.legal_actions().contains(&played),
        "the default fallback must be legal in the position it was chosen for"
    );
}
