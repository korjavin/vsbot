//! Seeded eps-greedy opening exploration for the live bot.
//!
//! # Why the deployed bot needs an RNG at all
//!
//! `virus_arena::gauntlet` randomises a gauntlet's openings on purpose: two
//! deterministic engines replay one game forever, so 400 repetitions of it are
//! one sample and not 400. Cross-play — `crates/virus-arena/crossplay` — drives
//! two *deployed* bots through the real server instead, and neither of them has
//! that lever. The Rust MCTS runs `mode=play` (no Dirichlet, argmax visits) and
//! the Java live path hard-codes root noise off, so a cross-play run replays a
//! small set of games and every interval computed from its game count is far
//! too narrow. Measured (bd `vsbot-t3q.2`): the 400-game S1 run contained **65
//! distinct games**, and the 50-game greedy-vs-GoBot run behind the `49-1`
//! contained **5**.
//!
//! This module is the missing lever, on the one side this repository controls.
//! [`ExploringEngine`] wraps whatever [`SearchEngine`] the binary built and,
//! inside an opening window, plays a uniformly random legal action instead of
//! the searched one with probability [`ExploreSettings::epsilon`].
//!
//! # It is off unless asked for
//!
//! CLAUDE.md: *production play paths take no RNG unless explicitly configured
//! (exploration)*. `VSBOT_EXPLORE_EPS` defaults to `0`, at which the wrapper is
//! not installed at all — the deployed bot is byte-for-byte the engine it was
//! before this module existed. Exploration makes the bot **weaker**; it buys
//! measurement validity, not strength, and belongs only in a harness run.
//!
//! # Why the window is counted in *our* turns
//!
//! The arena counts game turns and both of its sides explore, so an 8-turn
//! window at `eps = 0.15` spends about 24 coin flips per game, 12 per side. A
//! cross-play client sees only the turns it acts in — the opponent's plies never
//! reach the engine seam — and only one side of the match can be made to
//! explore at all. So [`ExploreSettings::turns`] counts **our own turns**, and
//! its default of `8` gives ~24 flips: the same expected opening noise per game
//! as an arena run, all of it on our side. The bias that creates is against
//! *us*, which is the safe direction for a "vsbot is stronger" claim.
//!
//! # Reproducibility
//!
//! Each game gets its own SplitMix64 stream, seeded
//! `mix64(seed ^ GOLDEN_GAMMA * (game_index + 1))` — the arena's derivation
//! minus its pair folding, which cross-play has no use for because it cannot
//! pair colours. Seeding a stream with `seed + k` instead is what caused
//! `nnue-trainer-riy`: two runs launched at nearby base seeds replayed
//! overlapping openings and their "independent" results were correlated. A
//! full-avalanche mixer on a golden-ratio-strided index makes seeds 1 and 2 as
//! unrelated as any other pair, so shards and instances can be handed
//! consecutive seeds safely.
//!
//! Given a seed, the *exploration schedule* — which of our plies are overridden,
//! and by which legal action — is a pure function of `(seed, game index, ply)`
//! and consults no clock. The games themselves are not reproducible, and cannot
//! be: a fixed-time MCTS run against a live opponent is a wall-clock
//! measurement. Reproducible noise on an unreproducible run is exactly what the
//! arena has too.

use std::sync::{Arc, Mutex};

use virus_arena::rng::{mix64, Rng, GOLDEN_GAMMA};
use virus_core::{Action, State};
use virus_proto::{SearchBudget, SearchEngine, SearchOutcome};

/// How much opening noise to inject, and from which stream.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExploreSettings {
    /// Probability that one of our actions inside the window is replaced by a
    /// uniformly random legal one. `0` disables exploration entirely.
    pub epsilon: f64,
    /// How many of **our own** turns the window covers, counted from the first
    /// turn we act in. See the module docs for why it is our turns and not the
    /// game's.
    pub turns: u32,
    /// Base seed for the per-game stream derivation.
    pub seed: u64,
}

impl Default for ExploreSettings {
    fn default() -> ExploreSettings {
        ExploreSettings {
            // Off. The deployed bot must not randomise its play because a
            // harness needed it to.
            epsilon: 0.0,
            turns: DEFAULT_EXPLORE_TURNS,
            seed: 1,
        }
    }
}

/// The default window, in our own turns. See the module docs.
pub const DEFAULT_EXPLORE_TURNS: u32 = 8;

impl ExploreSettings {
    /// Whether these settings would ever override an action.
    ///
    /// A zero window is as off as a zero epsilon, and both must skip the
    /// wrapper rather than install a decorator that provably does nothing.
    pub fn is_on(&self) -> bool {
        self.epsilon > 0.0 && self.turns > 0
    }
}

/// The seed for one game's opening stream.
///
/// The arena's `derive_game_seed` folds `game / 2` because its games come in
/// colour-swapped pairs that must share an opening. Cross-play has no pairs —
/// the server seats the challenger at P1 and a phase plays one chair — so this
/// strides by the game index itself. Everything else is the arena's, constant
/// for constant: golden-ratio stride, then the SplitMix64 finalizer. `+ 1`
/// keeps game 0 from multiplying by zero, which would hand the raw base seed
/// straight to the mixer for every run's first game.
pub fn derive_game_seed(seed: u64, game: u64) -> u64 {
    mix64(seed ^ GOLDEN_GAMMA.wrapping_mul(game.wrapping_add(1)))
}

/// Everything that resets at `game_start`, plus the last position's verdict.
#[derive(Debug)]
struct GameState {
    /// Index of the game being played, for the seed derivation.
    index: u64,
    rng: Rng,
    /// How many of our turns have opened, 1-based. `0` before our first.
    turn: u32,
    /// `moves_left` of the last position we answered, to spot a turn boundary.
    last_moves_left: Option<u8>,
    /// The last position's Zobrist key and what we decided for it.
    ///
    /// The server sends `move_made` and then `game_state` for the same mid-turn
    /// position, and a resync can repeat one verbatim. `Bot::maybe_search`'s
    /// `searched_version` gate stops most of that, but nothing guarantees the
    /// engine is asked exactly once per position — and a coin re-flipped for a
    /// position already decided would both advance the stream twice and let the
    /// same position answer differently on a retry. Memoising the last verdict
    /// makes the decorator idempotent per position.
    decided: Option<(u64, Option<Action>)>,
}

impl GameState {
    fn opening(index: u64, seed: u64) -> GameState {
        GameState {
            index,
            rng: Rng::new(derive_game_seed(seed, index)),
            turn: 0,
            last_moves_left: None,
            decided: None,
        }
    }
}

/// Counts an operator (and the tests) can read back.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ExploreCounters {
    /// Games started since the process began.
    pub games: u64,
    /// Our actions the window covered and the coin was flipped for.
    pub coins_flipped: u64,
    /// Actions actually replaced by a random legal one.
    pub explored: u64,
    /// Actions answered by the wrapped engine because the window had closed.
    pub past_window: u64,
}

/// A [`SearchEngine`] that plays random legal openings some of the time.
///
/// Wraps, never replaces: the inner engine is asked for its answer on **every**
/// ply, including the ones the coin overrides, and only the returned action
/// changes. That is the arena's rule and it is load-bearing for the same reason
/// — a searcher accumulates state across its calls (a tree, a table, killers),
/// and skipping a search would make an explored game differ from an unexplored
/// one in a second, uncontrolled way. It also keeps the wall-clock profile the
/// server sees identical, which matters when the opponent is on a live clock.
pub struct ExploringEngine {
    inner: Arc<dyn SearchEngine>,
    settings: ExploreSettings,
    /// One mutex for both, because a verdict and the stream it came from must
    /// not be observed apart. `choose` may be called concurrently for different
    /// positions (the trait says so), and the lock is held only across a few
    /// arithmetic operations — never across the inner search.
    state: Mutex<(GameState, ExploreCounters)>,
}

impl std::fmt::Debug for ExploringEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExploringEngine")
            .field("inner", &self.inner.name())
            .field("settings", &self.settings)
            .field("counters", &self.counters())
            .finish()
    }
}

impl ExploringEngine {
    /// Wraps `inner`. Call only when [`ExploreSettings::is_on`].
    pub fn new(inner: Arc<dyn SearchEngine>, settings: ExploreSettings) -> ExploringEngine {
        ExploringEngine {
            inner,
            settings,
            // Game 0's stream is live before any `game_start` arrives, so a
            // client that somehow searched first would still draw from a seeded
            // stream rather than from nothing.
            state: Mutex::new((
                GameState::opening(0, settings.seed),
                ExploreCounters::default(),
            )),
        }
    }

    /// The settings this wrapper was built with.
    pub fn settings(&self) -> ExploreSettings {
        self.settings
    }

    /// A snapshot of the counters.
    pub fn counters(&self) -> ExploreCounters {
        self.state.lock().expect("explore state").1
    }

    /// One line of provenance for the startup banner.
    pub fn describe(&self) -> String {
        format!(
            "exploration=ON eps={} window={} of our turns seed={} (opening noise: this bot plays \
             WEAKER on purpose, for cross-play diversity — bd vsbot-t3q.2)",
            self.settings.epsilon, self.settings.turns, self.settings.seed
        )
    }

    /// The exploration verdict for `state`: `Some(action)` to override with,
    /// `None` to keep the engine's answer.
    ///
    /// Split out from [`SearchEngine::choose`] so the decision is testable
    /// without a budget, a clock or an engine.
    fn verdict(&self, state: &State) -> Option<Action> {
        let mut guard = self.state.lock().expect("explore state");
        let (game, counters) = &mut *guard;
        let key = state.hash();
        if let Some((seen, decision)) = game.decided {
            if seen == key {
                return decision;
            }
        }

        // A turn of ours opens whenever `moves_left` fails to decrease. It
        // counts down 3-2-1 inside a turn; a turn spent on `PlaceNeutrals`
        // consumes the whole turn in one action and ends at 3, so two of our
        // turns in a row can both open at 3 — which is why this is `>=` and not
        // `>`, and why it is guarded by the position key above.
        let opens_a_turn = match game.last_moves_left {
            None => true,
            Some(previous) => state.moves_left() >= previous,
        };
        if opens_a_turn {
            game.turn += 1;
        }
        game.last_moves_left = Some(state.moves_left());

        let decision = if game.turn > self.settings.turns {
            // The window is shut. No draw is taken here, so the stream position
            // stays a function of the window alone.
            counters.past_window += 1;
            None
        } else {
            counters.coins_flipped += 1;
            if game.rng.next_f64() < self.settings.epsilon {
                let legal = state.legal_actions();
                // One draw whether or not it is used, so a position with no
                // legal action cannot desynchronise the stream.
                let pick = game.rng.below(legal.len());
                match pick {
                    Some(index) => {
                        counters.explored += 1;
                        Some(legal[index])
                    }
                    None => None,
                }
            } else {
                None
            }
        };
        game.decided = Some((key, decision));
        decision
    }
}

impl SearchEngine for ExploringEngine {
    fn choose(&self, state: &State, budget: &SearchBudget) -> Option<SearchOutcome> {
        let searched = self.inner.choose(state, budget);
        let Some(action) = self.verdict(state) else {
            return searched;
        };
        // Only override an answer the engine actually produced. `None` means
        // the position was superseded or has no legal move, and inventing an
        // action for a superseded position would change the client's
        // cancellation semantics for the sake of a coin flip.
        let searched = searched?;
        eprintln!(
            "vsbot: exploring — playing a random legal action instead of the searched one \
             (eps={}, window={} of our turns)",
            self.settings.epsilon, self.settings.turns
        );
        Some(SearchOutcome {
            action,
            // The searched score and depth describe the move that was *not*
            // played; reporting them against a random action would put a
            // fiction in the spectator view and in `games.db`. Nodes are the
            // one honest field — the search really did run.
            score: 0.0,
            depth: 0,
            nodes: searched.nodes,
        })
    }

    fn name(&self) -> &'static str {
        self.inner.name()
    }

    fn fallback(&self, state: &State) -> Option<Action> {
        // The overrun fallback is a safety net, not a sample. Randomising it
        // would put opening noise on exactly the plies where the bot is already
        // in trouble.
        self.inner.fallback(state)
    }

    fn can_ponder(&self) -> bool {
        // A pondering session answers actions without ever calling `choose`, so
        // a pondering explorer would silently explore nothing. `Settings::from`
        // rejects the combination outright; this is the belt to that braces.
        false
    }

    fn on_game_start(&self) {
        let mut guard = self.state.lock().expect("explore state");
        let (game, counters) = &mut *guard;
        counters.games += 1;
        *game = GameState::opening(game.index.wrapping_add(1), self.settings.seed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};
    use tokio_util::sync::CancellationToken;
    use virus_core::State;

    /// An engine that always answers with the *last* legal action, so "the
    /// engine's answer" and "a random answer" are trivially distinguishable.
    #[derive(Debug)]
    struct Last;

    impl SearchEngine for Last {
        fn choose(&self, state: &State, _budget: &SearchBudget) -> Option<SearchOutcome> {
            let actions = state.legal_actions();
            Some(SearchOutcome {
                nodes: 7,
                ..SearchOutcome::new(*actions.last()?)
            })
        }

        fn name(&self) -> &'static str {
            "last"
        }

        fn can_ponder(&self) -> bool {
            true
        }
    }

    fn budget() -> SearchBudget {
        SearchBudget::new(
            Instant::now() + Duration::from_secs(60),
            CancellationToken::new(),
        )
    }

    fn wrapped(settings: ExploreSettings) -> ExploringEngine {
        ExploringEngine::new(Arc::new(Last), settings)
    }

    fn board() -> State {
        State::new(12, 12, 2).expect("a 12x12 two-player board")
    }

    /// Plays `plies` of our own actions, always taking the engine's answer, and
    /// returns which of them were explored.
    fn play(engine: &ExploringEngine, plies: usize) -> Vec<bool> {
        let mut state = board();
        let mut explored = Vec::new();
        for _ in 0..plies {
            if state.game_over() || state.legal_actions().is_empty() {
                break;
            }
            let expected = *state.legal_actions().last().expect("legal actions");
            let outcome = engine.choose(&state, &budget()).expect("an action");
            explored.push(outcome.action != expected);
            state = state.apply(outcome.action).expect("a legal action");
            // Skip the opponent: this fixture only exercises our own turns, and
            // the decorator only ever sees ours.
            while !state.game_over() && state.current_player() != 1 {
                let reply = *state.legal_actions().first().expect("a reply");
                state = state.apply(reply).expect("a legal reply");
            }
        }
        explored
    }

    #[test]
    fn epsilon_zero_is_off_and_the_wrapper_is_never_installed() {
        assert!(
            !ExploreSettings::default().is_on(),
            "the default must be off"
        );
        assert!(!ExploreSettings {
            epsilon: 0.15,
            turns: 0,
            seed: 1
        }
        .is_on());
        assert!(ExploreSettings {
            epsilon: 0.15,
            turns: 8,
            seed: 1
        }
        .is_on());
    }

    /// The acceptance criterion of bd `vsbot-t3q.2`: the window shuts, and it
    /// shuts *completely*. At `eps = 1` every ply inside it is random and every
    /// ply after it is the engine's, with no tail.
    #[test]
    fn the_window_ends_cleanly() {
        let engine = wrapped(ExploreSettings {
            epsilon: 1.0,
            turns: 2,
            seed: 11,
        });
        engine.on_game_start();
        let explored = play(&engine, 12);
        assert!(explored.len() >= 9, "fixture should reach past the window");
        // Two of our turns is six of our actions, unless a `PlaceNeutrals`
        // ended one early; the boundary is where `past_window` starts counting.
        let counters = engine.counters();
        assert_eq!(
            counters.coins_flipped + counters.past_window,
            explored.len() as u64
        );
        assert_eq!(
            counters.explored, counters.coins_flipped,
            "eps=1 explores every covered ply"
        );
        let boundary = counters.coins_flipped as usize;
        assert!(
            explored[..boundary].iter().all(|hit| *hit),
            "every ply inside the window must be random: {explored:?}"
        );
        assert!(
            explored[boundary..].iter().all(|hit| !*hit),
            "no exploration may leak past the window: {explored:?}"
        );
    }

    #[test]
    fn the_window_is_counted_in_our_turns_not_our_plies() {
        let engine = wrapped(ExploreSettings {
            epsilon: 1.0,
            turns: 3,
            seed: 5,
        });
        engine.on_game_start();
        play(&engine, 15);
        // Three turns of three actions. `PlaceNeutrals` can end a turn in one
        // action, so this is an upper bound the opening cannot exceed.
        let flipped = engine.counters().coins_flipped;
        assert!(
            (1..=9).contains(&flipped),
            "flipped {flipped} coins in 3 turns"
        );
    }

    #[test]
    fn the_same_seed_replays_the_same_schedule_and_a_different_one_does_not() {
        let schedule = |seed: u64| {
            let engine = wrapped(ExploreSettings {
                epsilon: 0.5,
                turns: 8,
                seed,
            });
            engine.on_game_start();
            play(&engine, 20)
        };
        assert_eq!(schedule(2026), schedule(2026), "same seed, same schedule");
        assert_ne!(
            schedule(2026),
            schedule(2027),
            "adjacent seeds must not replay each other (nnue-trainer-riy)"
        );
    }

    /// Consecutive games must not replay each other either — that is the whole
    /// bug. The per-game derivation is what makes game 1 unrelated to game 0.
    #[test]
    fn consecutive_games_get_unrelated_streams() {
        let engine = wrapped(ExploreSettings {
            epsilon: 0.5,
            turns: 8,
            seed: 99,
        });
        let mut schedules = Vec::new();
        for _ in 0..6 {
            engine.on_game_start();
            schedules.push(play(&engine, 20));
        }
        let distinct: std::collections::HashSet<_> = schedules.iter().collect();
        assert!(
            distinct.len() >= 5,
            "six games produced {} distinct schedules: {schedules:?}",
            distinct.len()
        );
        assert_eq!(engine.counters().games, 6);
    }

    /// A repeated snapshot must not re-flip the coin: the server sends
    /// `move_made` and then `game_state` for one mid-turn position.
    #[test]
    fn a_repeated_position_is_decided_once() {
        let engine = wrapped(ExploreSettings {
            epsilon: 0.5,
            turns: 8,
            seed: 3,
        });
        engine.on_game_start();
        let state = board();
        let first = engine.choose(&state, &budget()).expect("an action");
        let counters = engine.counters();
        for _ in 0..5 {
            let again = engine.choose(&state, &budget()).expect("an action");
            assert_eq!(
                again.action, first.action,
                "the same position must decide the same way"
            );
        }
        assert_eq!(
            engine.counters().coins_flipped,
            counters.coins_flipped,
            "a repeat must not advance the stream"
        );
    }

    #[test]
    fn an_explored_action_is_legal_and_carries_no_fictional_score() {
        let engine = wrapped(ExploreSettings {
            epsilon: 1.0,
            turns: 8,
            seed: 17,
        });
        engine.on_game_start();
        let state = board();
        let outcome = engine.choose(&state, &budget()).expect("an action");
        assert!(
            state.legal_actions().contains(&outcome.action),
            "an explored action must be legal"
        );
        assert_eq!(outcome.score, 0.0, "a random move has no searched score");
        assert_eq!(outcome.depth, 0, "a random move has no searched depth");
        assert_eq!(
            outcome.nodes, 7,
            "the search really did run, and is reported"
        );
    }

    #[test]
    fn the_wrapper_delegates_name_and_fallback_and_refuses_to_ponder() {
        let engine = wrapped(ExploreSettings {
            epsilon: 0.15,
            turns: 8,
            seed: 1,
        });
        assert_eq!(engine.name(), "last");
        assert_eq!(engine.fallback(&board()), Last.fallback(&board()));
        assert!(
            !engine.can_ponder(),
            "a ponder session bypasses `choose` and would explore nothing"
        );
        assert!(
            Last.can_ponder(),
            "the inner engine can, which is the point"
        );
    }

    #[test]
    fn the_seed_derivation_avalanches() {
        assert_ne!(derive_game_seed(1, 0), derive_game_seed(2, 0));
        assert_ne!(derive_game_seed(1, 0), derive_game_seed(1, 1));
        // Adjacent base seeds must not produce overlapping game seeds. `seed +
        // k` did, and that is `nnue-trainer-riy`.
        let one: std::collections::HashSet<u64> =
            (0..64).map(|game| derive_game_seed(1, game)).collect();
        let two: std::collections::HashSet<u64> =
            (0..64).map(|game| derive_game_seed(2, game)).collect();
        assert!(one.is_disjoint(&two), "adjacent seeds shared a game seed");
        assert_eq!(one.len(), 64, "a base seed repeated a game seed");
    }

    #[test]
    fn the_stream_is_uniform_enough_to_be_a_coin() {
        let mut rng = Rng::new(derive_game_seed(7, 0));
        let hits = (0..10_000).filter(|_| rng.next_f64() < 0.15).count();
        assert!(
            (1200..1800).contains(&hits),
            "eps=0.15 fired {hits}/10000 times"
        );
        let mut rng = Rng::new(1);
        assert!((0..10_000).all(|_| rng.below(5).expect("non-empty") < 5));
        assert_eq!(Rng::new(1).below(0), None);
    }
}
