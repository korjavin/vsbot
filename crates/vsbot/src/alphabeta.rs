//! `virus-search` behind the [`SearchEngine`] seam.
//!
//! This is the engine with **no domain restriction**: the enhanced alpha-beta
//! searcher works on any board size and on two, three or four players (max^n
//! over a `[Score; 4]` vector for the multiplayer seats), exactly like the Go
//! bot it is a port of. That is why it — and not the greedy reference engine —
//! is what [`crate::MctsEngine`] falls back to when a game is outside the
//! champion's absolute-frame 12x12 two-player domain.
//!
//! # The two things this adapter is responsible for
//!
//! 1. **Driving the deadline entry point, and only that one.**
//!    [`Searcher::search_with_deadline`] is the entry whose contract is
//!    "the move of the deepest fully completed iteration", and `virus-search`
//!    has a test pinning it to what [`Searcher::search_to_depth`] returns at the
//!    depth it reports (ARCHITECTURE.md invariant 3 — the bug that made a
//!    parity-perfect Go engine lose 0-10 live). Slicing the budget into several
//!    shorter searches, or reaching for the node-budget entry to fake
//!    interruptibility, would break that consistency for nothing, so the
//!    allocator's deadline is passed straight through.
//! 2. **Never handing the searcher a position it would panic on.** The library
//!    asserts its root player, and a bot that panics its search worker mid-game
//!    forfeits on the server's 120 s timer.
//!
//! # Cancellation: what this engine can and cannot honour
//!
//! [`SearchBudget`] asks an engine to stop at its deadline **and** to poll
//! [`SearchBudget::is_cancelled`]. This one polls at entry and stops at the
//! deadline, but it cannot abort a search already inside `virus-search`: the
//! searcher's stop flag is private and reserved for its lazy-SMP helpers, and
//! there is no public hook to raise it. So a superseded search runs on to its
//! own deadline.
//!
//! The two obvious workarounds are both worse than the problem. Slicing the
//! budget into short `search_with_deadline` calls restarts iterative deepening
//! every slice, and — because a new iteration is not *started* past
//! [`SearchOptions::soft_deadline_percent`] of the budget it was given — a
//! sliced search stalls at a fixed shallow depth instead of deepening. The
//! node-budget entry is deterministic rather than wall-clock and would break the
//! deadline/depth consistency `virus-search` pins.
//!
//! What is bounded here instead is the *cost* of the overlap:
//!
//! * the deadline handed over is [`SearchBudget::deadline`], never the later
//!   `ceiling`, so a stale search dies at the target rather than the extension;
//! * a search that finds the cache lock held knows another search is still
//!   running, and takes a [`CONTENDED_TT_LOG2`]-sized table instead of the full
//!   one, so concurrent searches cannot multiply the engine's 32 MiB working set
//!   inside the deployment's 512 MB container.
//!
//! Removing the residual needs a public stop flag on `virus-search::Searcher`,
//! which is a change to that crate rather than to this adapter: bd `vsbot-tz7`.

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use virus_core::{Action, CellKind, Player, State, MAX_PLAYERS};
use virus_proto::{SearchBudget, SearchEngine, SearchOutcome};
use virus_search::{SearchOptions, SearchResult, Searcher};

/// `log2` of the transposition table a *contended* search gets: 2^18 entries,
/// 4 MiB against the cached searcher's 32 MiB.
///
/// Contention means a previous search is still running on this engine — the
/// client cancelled it, but `virus-search` cannot be interrupted, so it holds
/// its table until its own deadline. Giving every such overlap a second full
/// table is how a 512 MB container runs out of memory during a burst of
/// snapshots. A contended search starts on a cold table whatever its size, so
/// the strength it gives up for the bound is small and it gives it up only on
/// the rare path.
const CONTENDED_TT_LOG2: u32 = 18;

/// Enhanced iterative-deepening alpha-beta with the hand-tuned leaf evaluation.
///
/// One instance plays every game the process is offered; the cached searcher
/// below is what carries a warm transposition table from one move to the next.
pub struct AlphaBetaEngine {
    options: SearchOptions,
    /// The searcher from the previous move, kept so its packed transposition
    /// table starts the next move warm on the principal subtree it just spent a
    /// whole budget building.
    ///
    /// Behind a `Mutex` because [`SearchEngine::choose`] takes `&self` and the
    /// client may have two searches in flight — cancellation is cooperative, so
    /// a superseded search runs on until its own deadline. The lock is therefore
    /// only ever *tried*: a contended call builds its own searcher and searches
    /// with a cold table rather than queueing behind a dead position's deadline,
    /// because a search that answers late is a forfeit and a search that answers
    /// with a cold table is merely a slightly worse move.
    ///
    /// Paired with the shape it was built for, because that is the one thing a
    /// `Searcher` cannot be asked about after the fact.
    cached: Mutex<Option<(SearcherShape, Searcher)>>,
    /// Searchers constructed so far — see [`AlphaBetaEngine::searchers_built`].
    built: AtomicU64,
}

impl fmt::Debug for AlphaBetaEngine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AlphaBetaEngine")
            .field("enhanced", &self.options.enhanced)
            .field("smp_threads", &self.options.smp_threads)
            .finish()
    }
}

impl Default for AlphaBetaEngine {
    fn default() -> AlphaBetaEngine {
        AlphaBetaEngine::new()
    }
}

impl AlphaBetaEngine {
    /// The production configuration: the full enhanced stack, SMP off.
    ///
    /// SMP is off in [`SearchOptions::default`] and this adapter does not turn
    /// it on. Helper threads write the shared table, so a search with SMP on
    /// returns a different move run to run — and CLAUDE.md's "all engine
    /// randomness is seeded and deterministic; production play paths take no RNG
    /// unless explicitly configured" makes an irreproducible production move
    /// exactly the thing not to ship. It also shares this box with an arena
    /// cell and a trainer window; extra search threads would be taken from them.
    pub fn new() -> AlphaBetaEngine {
        let options = SearchOptions::default();
        debug_assert!(options.enhanced, "the bot plays the enhanced stack");
        debug_assert_eq!(options.smp_threads, 0, "production play is single-threaded");
        AlphaBetaEngine {
            options,
            cached: Mutex::new(None),
            built: AtomicU64::new(0),
        }
    }

    /// The startup banner line: what this engine is and what it evaluates with.
    pub fn describe(&self) -> String {
        format!(
            "enhanced={} aspiration={} lmr={} smp_threads={} tt=2^{} entries \
             soft_deadline={}% eval=hand-tuned domain=any-board-size,2-4-players",
            self.options.enhanced,
            self.options.aspiration,
            self.options.lmr,
            self.options.smp_threads,
            self.options.tt_log2,
            self.options.soft_deadline_percent,
        )
    }

    /// How many searchers this engine has constructed.
    ///
    /// Diagnostic, and the only way to test the cache at all: reuse is invisible
    /// in the move a search returns, so a cache that silently never hit — or one
    /// that silently never *missed*, which is the dangerous direction — would
    /// look exactly like a working one from the outside.
    pub fn searchers_built(&self) -> u64 {
        self.built.load(Ordering::SeqCst)
    }

    /// Runs one search, reusing the cached searcher when it fits this position.
    ///
    /// "Fits" is [`SearcherShape`]: same seat, same board, same set of live
    /// players. A `Searcher` fixes its root player and its multiplayer flag at
    /// construction and sizes its history tables to the board, so handing it a
    /// position of another shape is a programming error rather than a weaker
    /// search — `search_with_deadline` asserts the root outright, and the max^n
    /// branch is chosen once, from the position the searcher was built with.
    fn search(&self, state: &State, deadline: Instant) -> Option<SearchResult> {
        let shape = SearcherShape::of(state);
        // `try_lock`, never `lock` — see the field comment. A poisoned lock is
        // treated as an empty cache for the same reason: the panic that poisoned
        // it says nothing about this position, and refusing to move would be a
        // forfeit.
        let mut guard = match self.cached.try_lock() {
            Ok(guard) => guard,
            Err(std::sync::TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
            Err(std::sync::TryLockError::WouldBlock) => {
                // A held lock *is* the signal that a superseded search is still
                // running: it cannot be interrupted, so it keeps its table until
                // its own deadline. This search gets a small one so the two
                // cannot add up to two full working sets, and it is never
                // cached — the searcher that owns the lock is the one that
                // should stay warm.
                let options = SearchOptions {
                    tt_log2: CONTENDED_TT_LOG2,
                    ..self.options
                };
                self.built.fetch_add(1, Ordering::SeqCst);
                return Searcher::new(state, options).search_with_deadline(state, deadline);
            }
        };
        // The cached shape, not the position's own: comparing the position with
        // itself is trivially true and would reuse a 12x12 two-player searcher
        // for a 16x16 four-player game.
        if guard.as_ref().map(|(cached, _)| *cached) != Some(shape) {
            *guard = Some((shape, self.fresh(state)));
        }
        guard
            .as_mut()
            .expect("just built")
            .1
            .search_with_deadline(state, deadline)
    }

    fn fresh(&self, state: &State) -> Searcher {
        self.built.fetch_add(1, Ordering::SeqCst);
        Searcher::new(state, self.options)
    }
}

/// Everything a cached [`Searcher`] must agree with a position about.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct SearcherShape {
    /// The seat the searcher scores for. Fixed at construction, and
    /// `search_with_deadline` asserts it: root-relative scores are meaningless
    /// from the other chair, and an action played for the wrong seat forfeits.
    root: Player,
    rows: usize,
    cols: usize,
    /// Which seats are still alive, as a bitmask.
    ///
    /// Not merely the count: `Searcher` picks the max^n path from the number of
    /// *active* players at construction, so a four-player game that has just
    /// eliminated a seat must not keep searching as though it had not.
    active: u8,
}

impl SearcherShape {
    fn of(state: &State) -> SearcherShape {
        let mut active = 0u8;
        for player in 1..=MAX_PLAYERS as Player {
            if state.active(player) {
                active |= 1 << player;
            }
        }
        SearcherShape {
            root: state.current_player(),
            rows: state.rows(),
            cols: state.cols(),
            active,
        }
    }
}

impl SearchEngine for AlphaBetaEngine {
    fn choose(&self, state: &State, budget: &SearchBudget) -> Option<SearchOutcome> {
        if budget.is_cancelled() {
            // The documented `None` case. Checked first because building a
            // searcher allocates a 32 MiB packed table, and the client discards
            // a superseded answer at send time regardless.
            return None;
        }
        // Defence in depth, and the reason it is spelled out rather than left to
        // the library: the server really does publish a transient with a live
        // mover and `movesLeft == 0` (the `move_made` echo of an opponent's
        // third action, before the `turn_change` that rotates the turn). The
        // searcher's own guards happen to reject it today — the opening book
        // requires `placed + movesLeft == 3` and the root fallback goes through
        // `legal_actions`, which filters on `can_act` — but "happens to" is not
        // a contract, and the failure mode is a panicked search worker.
        if !state.can_act() {
            return None;
        }

        // The clock starts before the searcher is built, because building one
        // allocates and zeroes the packed table and that time comes out of the
        // same allocation the move is judged against.
        //
        // `budget.deadline` and not `budget.ceiling`: the extension room is
        // spent on the visit-based stop rules in `virus_proto::clock`, which
        // read a PUCT root's visit distribution. Alpha-beta has no such root, so
        // it stops at the target — which is what `SearchBudget`'s own docs call
        // the correct behaviour for an engine that does not implement them.
        let result = self.search(state, budget.deadline)?;
        let action = result.action?;
        Some(SearchOutcome {
            action,
            // Integral and in the mover's frame already: the searcher is rooted
            // at the side to move, so `SearchResult::score` needs no sign flip.
            // It is a hand-tuned material/mobility number, not a probability —
            // the diagnostics panel relays it verbatim and never compares it
            // with the champion's [-1, 1] value.
            score: result.score as f64,
            depth: result.depth,
            nodes: result.nodes as i64,
        })
    }

    /// First capture, else a move that keeps us alive, else the first legal
    /// action.
    ///
    /// The client holds this *before* the long search starts (fallback-first,
    /// superiority.md §2b), so it must cost approximately nothing — it is
    /// emphatically not a shallow search. One pass over `legal_actions` is the
    /// whole of it.
    fn fallback(&self, state: &State) -> Option<Action> {
        let actions = state.legal_actions();
        let mover = state.current_player();
        actions
            .iter()
            .copied()
            .find(|action| match *action {
                Action::Move { target } => {
                    let cell = state.at(target);
                    cell.kind() == CellKind::Normal && cell.owner() != mover
                }
                Action::PlaceNeutrals { .. } => false,
            })
            .or_else(|| actions.first().copied())
    }

    fn name(&self) -> &'static str {
        "alphabeta"
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::time::Duration;
    use tokio_util::sync::CancellationToken;
    use virus_proto::EngineKind;

    fn budget(millis: u64) -> SearchBudget {
        SearchBudget::new(
            Instant::now() + Duration::from_millis(millis),
            CancellationToken::new(),
        )
    }

    /// A position past every seat's opening turn.
    ///
    /// The wedge opening book answers the first turn of each seat by fiat —
    /// `depth == 0`, `nodes == 0`, no search at all — which is correct and is
    /// what the parity fixtures pin. Any assertion about a *search* therefore
    /// has to start after it, and the book voids itself once
    /// `placed + moves_left` stops equalling one turn's worth of actions.
    pub(crate) fn past_the_book(rows: usize, cols: usize, players: usize) -> State {
        let mut state = State::new(rows, cols, players)
            .unwrap_or_else(|error| panic!("{rows}x{cols} {players}p: {error}"));
        for _ in 0..players * virus_core::ACTIONS_PER_TURN as usize {
            let action = state.legal_actions()[0];
            state = state.apply(action).expect("legal");
        }
        state
    }

    /// Every action the engine returns must be legal in the position it was
    /// given — an illegal action is an instant server-side forfeit.
    fn plays_legally(rows: usize, cols: usize, players: usize, turns: usize) -> usize {
        let engine = AlphaBetaEngine::new();
        let mut state = State::new(rows, cols, players)
            .unwrap_or_else(|error| panic!("{rows}x{cols} {players}p: {error}"));
        let mut played = 0;
        for _ in 0..turns {
            if !state.can_act() {
                break;
            }
            let outcome = engine
                .choose(&state, &budget(60))
                .unwrap_or_else(|| panic!("no action on a {rows}x{cols} {players}-player board"));
            state = state.apply(outcome.action).unwrap_or_else(|error| {
                panic!("ILLEGAL action on a {rows}x{cols} {players}-player board: {error}")
            });
            played += 1;
        }
        played
    }

    #[test]
    fn it_plays_legally_on_the_champion_board() {
        assert!(plays_legally(12, 12, 2, 12) > 0);
    }

    /// The whole point of wiring this engine: the boards the champion's
    /// absolute-frame encoder has no representation for.
    #[test]
    fn it_plays_legally_off_the_twelve_by_twelve_board() {
        for (rows, cols) in [(8, 8), (10, 10), (16, 16), (20, 20), (11, 17)] {
            assert!(
                plays_legally(rows, cols, 2, 9) > 0,
                "{rows}x{cols} produced no action"
            );
        }
    }

    /// The max^n path — the deferred half of this bead's live acceptance.
    ///
    /// `Searcher` selects max^n from the number of *active* players at
    /// construction, so a three- or four-player root can only have been searched
    /// by it: there is no other branch in the crate that scores a `[Score; 4]`
    /// vector, and the 1v1 negamax-shaped path is unreachable with three seats
    /// alive. A completed depth past the book and a non-zero node count are
    /// therefore proof the multiplayer search ran, rather than the position
    /// falling out to the root fallback or the opening book.
    #[test]
    fn multiplayer_positions_go_through_the_max_n_path() {
        for (rows, cols, players) in [(12, 12, 3), (12, 12, 4), (16, 16, 3), (20, 20, 4)] {
            let engine = AlphaBetaEngine::new();
            let state = past_the_book(rows, cols, players);
            assert_eq!(state.players(), players);
            assert!(
                (1..=players as Player).all(|seat| state.active(seat)),
                "{players} players: a seat was eliminated before the assertion"
            );
            let searched = engine
                .choose(&state, &budget(400))
                .unwrap_or_else(|| panic!("{rows}x{cols} {players}p: no action"));
            assert!(
                searched.depth >= 1 && searched.nodes > 0,
                "{rows}x{cols} {players}p: depth {} nodes {} — the max^n search never ran",
                searched.depth,
                searched.nodes
            );
            assert!(
                state.apply(searched.action).is_ok(),
                "{rows}x{cols} {players}p: illegal searched action {:?}",
                searched.action
            );
        }
        // And a whole run of multiplayer turns stays legal, on and off 12x12.
        assert!(plays_legally(20, 20, 4, 12) > 0);
        assert!(plays_legally(16, 16, 3, 12) > 0);
    }

    /// ARCHITECTURE.md invariant 5: a superseded position's answer is worthless,
    /// and the expensive part must not even start.
    #[test]
    fn a_cancelled_budget_is_answered_with_nothing_before_any_work() {
        let engine = AlphaBetaEngine::new();
        let state = State::new(12, 12, 2).expect("12x12");
        let cancel = CancellationToken::new();
        cancel.cancel();
        let budget = SearchBudget::new(Instant::now() + Duration::from_secs(30), cancel);
        let started = Instant::now();
        assert_eq!(engine.choose(&state, &budget).map(|out| out.action), None);
        assert!(
            started.elapsed() < Duration::from_millis(50),
            "a cancelled search still did work: {:?}",
            started.elapsed()
        );
    }

    /// The `movesLeft == 0` transient the live soak found. It must produce no
    /// action rather than a panicked search worker.
    #[test]
    fn a_spent_turn_is_answered_with_nothing() {
        let engine = AlphaBetaEngine::new();
        let mut state = State::new(12, 12, 2).expect("12x12");
        for _ in 0..3 {
            let action = state.legal_actions()[0];
            state = state.apply(action).expect("legal");
        }
        // Rebuild the transient the server publishes: the same mover, no actions
        // left, before the turn rotates.
        let mut snapshot = state.snapshot();
        snapshot.moves_left = 0;
        snapshot.current_player = 1;
        let spent = snapshot.decode().expect("the transient decodes");
        assert!(!spent.can_act());
        assert!(engine.choose(&spent, &budget(50)).is_none());
    }

    /// The fallback is held before the search starts, so it may not search.
    #[test]
    fn the_fallback_is_instant_and_legal() {
        let engine = AlphaBetaEngine::new();
        for (rows, cols, players) in [(12, 12, 2), (16, 16, 2), (20, 20, 4)] {
            let state = State::new(rows, cols, players).expect("board");
            let started = Instant::now();
            let action = engine.fallback(&state).expect("a legal action exists");
            assert!(
                started.elapsed() < Duration::from_millis(20),
                "{rows}x{cols}: the fallback took {:?} — it must not be searching",
                started.elapsed()
            );
            assert!(
                state.apply(action).is_ok(),
                "{rows}x{cols}: illegal fallback"
            );
        }
    }

    /// The cached searcher must never be handed a position of another shape.
    /// Alternating seats, board sizes and seat counts through one engine is the
    /// cheap way to prove the guard, because a mismatch is an assert inside
    /// `virus-search` rather than a subtly worse move.
    #[test]
    fn one_engine_survives_alternating_seats_boards_and_player_counts() {
        let engine = AlphaBetaEngine::new();
        for (rows, cols, players) in [
            (12, 12, 2),
            (9, 9, 2),
            (12, 12, 3),
            (14, 14, 4),
            (12, 12, 2),
        ] {
            let mut state = State::new(rows, cols, players).expect("board");
            for _ in 0..8 {
                if !state.can_act() {
                    break;
                }
                let outcome = engine.choose(&state, &budget(30)).expect("an action");
                state = state.apply(outcome.action).expect("legal");
            }
        }
    }

    /// The cache must miss on every change of shape, and on none of the
    /// positions that share one.
    ///
    /// This is a regression test for a "does the cached searcher fit" check that
    /// compared the *position* with itself rather than with the cached
    /// searcher's shape. It was invisible from the outside — every board still
    /// produced legal moves — while a 12x12 two-player searcher was quietly
    /// being reused for a four-player game on another board, keeping the max^n
    /// flag and the history sizing it was constructed with. Reuse is only ever
    /// observable through the construction count, so that is what this asserts.
    #[test]
    fn the_cached_searcher_is_rebuilt_for_every_change_of_shape() {
        let engine = AlphaBetaEngine::new();
        let mut expected = 0;
        for (rows, cols, players) in [(12, 12, 2), (12, 12, 3), (16, 16, 2), (12, 12, 2)] {
            let state = past_the_book(rows, cols, players);
            engine.choose(&state, &budget(40)).expect("an action");
            expected += 1;
            assert_eq!(
                engine.searchers_built(),
                expected,
                "{rows}x{cols} {players}p reused a searcher built for another shape"
            );

            // The same shape again, moved on by one action of the same turn:
            // same seat, same board, same live seats, so the warm transposition
            // table must be kept rather than thrown away and rebuilt.
            let same_seat = state
                .apply(state.legal_actions()[0])
                .expect("legal within the turn");
            assert_eq!(same_seat.current_player(), state.current_player());
            engine.choose(&same_seat, &budget(40)).expect("an action");
            assert_eq!(
                engine.searchers_built(),
                expected,
                "{rows}x{cols} {players}p threw its warm table away mid-turn"
            );
        }
    }

    /// Two searches at once must both answer, and neither may wait for the
    /// other.
    ///
    /// This is the liveness half of the cancellation story. The client cancels a
    /// superseded search and immediately dispatches a new one, but
    /// `virus-search` cannot be interrupted, so the old search keeps running —
    /// and it holds the cache lock while it does. If the new search *queued* on
    /// that lock it would inherit the dead position's remaining budget, answer
    /// late, and the client would play its pre-selected fallback instead. A
    /// forfeit-adjacent failure, so the lock is only ever tried.
    #[test]
    fn a_second_concurrent_search_answers_without_waiting_for_the_first() {
        use std::sync::Arc;

        let engine = Arc::new(AlphaBetaEngine::new());
        let state = past_the_book(12, 12, 2);

        // The slow one holds the lock for most of a second.
        let long = {
            let engine = Arc::clone(&engine);
            let state = state.clone();
            std::thread::spawn(move || engine.choose(&state, &budget(800)).map(|out| out.action))
        };
        // Give it time to take the lock, then race a short search past it.
        std::thread::sleep(Duration::from_millis(120));
        let started = Instant::now();
        let quick = engine
            .choose(&state, &budget(100))
            .expect("the contended search still answers");
        let waited = started.elapsed();

        assert!(
            state.apply(quick.action).is_ok(),
            "the contended search returned an illegal action"
        );
        assert!(
            waited < Duration::from_millis(500),
            "the second search waited {waited:?} for the first — it queued on the lock \
             instead of building its own searcher"
        );
        let slow = long.join().expect("the first search survived");
        assert!(slow.is_some_and(|action| state.apply(action).is_ok()));
        assert_eq!(
            engine.searchers_built(),
            2,
            "the contended search must build its own searcher, not share one"
        );
    }

    /// The other seat of the same board is a different searcher: root-relative
    /// scores are only meaningful from the chair they were built for, and
    /// `virus-search` asserts that rather than returning a worse move.
    #[test]
    fn the_other_seat_gets_its_own_searcher() {
        let engine = AlphaBetaEngine::new();
        let ours = past_the_book(12, 12, 2);
        engine.choose(&ours, &budget(40)).expect("an action");
        assert_eq!(engine.searchers_built(), 1);

        // Play out the rest of the turn so the mover rotates.
        let mut theirs = ours.clone();
        while theirs.current_player() == ours.current_player() {
            theirs = theirs
                .apply(theirs.legal_actions()[0])
                .expect("legal within the turn");
        }
        engine.choose(&theirs, &budget(40)).expect("an action");
        assert_eq!(
            engine.searchers_built(),
            2,
            "the opponent's seat reused a searcher rooted at ours"
        );
    }

    /// `SEARCH=ALPHABETA` must name this engine, and the banner must say what it
    /// is playing with — an operator has to be able to confirm the engine from
    /// the log alone.
    #[test]
    fn it_reports_itself_for_the_banner() {
        let engine = AlphaBetaEngine::new();
        assert_eq!(engine.name(), "alphabeta");
        assert_eq!(EngineKind::AlphaBeta.as_str(), "ALPHABETA");
        let description = engine.describe();
        assert!(description.contains("enhanced=true"), "{description}");
        assert!(description.contains("smp_threads=0"), "{description}");
        assert!(description.contains("eval=hand-tuned"), "{description}");
    }
}
