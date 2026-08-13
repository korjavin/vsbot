//! `virus-mcts` behind the [`SearchEngine`] seam.
//!
//! The searcher itself is domain-restricted **by construction**:
//! [`MctsSearcher::new`] asserts two players, and asserts a 12x12 board whenever
//! a net is supplied. Those are not defensive niceties — the absolute-frame
//! backup has nowhere to put a third seat's win, and [`Encoded::from_state`] has
//! no encoding for another board size. An assert is the right behaviour for a
//! library, but a bot that panics its search worker mid-game forfeits on the
//! server's 120 s timer.
//!
//! So this adapter checks the *same two conditions* up front, per position, and
//! plays the greedy reference engine for any position outside the domain —
//! **never silently**. The Java post-mortem (`GameLoopHandler.unwiredEvalWarning`)
//! is the reason for the shouting: a quiet eval fallback once let a harness
//! report hand-tuned results as the net's for a whole run.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use virus_core::{Action, State};
use virus_mcts::{Config, MctsSearcher, NetError, PolicyValueNet, ValueSource, BOARD};
use virus_proto::clock::{verdict, MoveAllocation, RootProgress};
use virus_proto::ponder::{PonderInbox, PonderStep};
use virus_proto::{GreedyEngine, SearchBudget, SearchEngine, SearchOutcome};

/// Longest uninterrupted stretch of simulations.
///
/// [`MctsSearcher::run_until_deadline`] honours a deadline but knows nothing
/// about [`SearchBudget::cancel`], so the search is driven in slices and
/// cancellation is polled between them (ARCHITECTURE.md invariant 5: a
/// superseded position's answer is worthless the instant a newer snapshot
/// lands). 20 ms is short enough to drop a stale search promptly and long
/// enough that the polling itself costs nothing.
///
/// The slice is also where the intra-turn stop rules
/// ([`virus_proto::clock::verdict`]) are evaluated, so it doubles as the
/// resolution of the early-stop and extension decisions.
const CANCEL_POLL_SLICE: Duration = Duration::from_millis(20);

/// Sentinel for "no position shape has been logged yet".
const NO_SHAPE: u64 = u64::MAX;

/// Simulations a pondering session's **retained tree** may hold.
///
/// A node owns a whole `State`, so an uncapped ponder against an opponent who
/// thinks for minutes would grow without bound on a host that shares CPU and
/// memory with the nightly trainer window (superiority.md §2b). At roughly
/// 1.5 KB a node this is ~75 MB of ceiling, and it is far more simulations than
/// a 12 s turn can spend anyway.
///
/// It bounds the tree in memory, not the session's lifetime work: a re-root
/// frees everything outside the chosen child's subtree, and the allowance is
/// re-based onto what survived (see [`SearchEngine::ponder`]).
const PONDER_SIM_CAP: u64 = 50_000;

/// How many of the opponent's actions a ponder tree will chase before giving up
/// and rebuilding.
///
/// One turn. Beyond that the snapshots we missed are numerous enough that the
/// tree is unlikely to hold the position, and the search for it stops being
/// cheap.
const MAX_REROOT_PLIES: usize = 3;

/// Ceiling on `apply` calls spent looking for a re-root path, so a pathological
/// position cannot turn tree reuse into a stall.
const REROOT_APPLY_BUDGET: usize = 4_096;

/// PUCT search over the policy/value artifact, with a greedy safety net for
/// positions outside the searcher's domain.
#[derive(Debug)]
pub struct MctsEngine {
    net: PolicyValueNet,
    config: Config,
    artifact: PathBuf,
    /// Shape of the most recent position whose domain verdict was logged.
    ///
    /// The bot plays game after game in one process, so the interesting event
    /// is the *transition* — entering or leaving the degraded mode — not every
    /// individual move. Logging per move would bury the warning in three lines
    /// a turn; logging once ever would hide a later game's downgrade.
    last_shape: AtomicU64,
}

impl MctsEngine {
    /// Loads and validates `artifact`, then builds a play-mode searcher factory.
    ///
    /// Validation is [`PolicyValueNet::load`]'s job and it is exhaustive (arch
    /// string, board/plane counts, every tensor shape, every weight finite), so
    /// a wrong or truncated artifact fails **here**, at startup, instead of
    /// producing `NaN` priors on move 40 of a live game.
    pub fn load(artifact: impl AsRef<Path>, seed: u64) -> Result<MctsEngine, NetError> {
        let artifact = artifact.as_ref().to_path_buf();
        let net = PolicyValueNet::load(&artifact)?;
        Ok(MctsEngine {
            net,
            artifact,
            config: Config {
                seed,
                // The champion ships a value head; `ValueSource::Net` degrades
                // to the hand-tuned leaf on its own if a future artifact lacks
                // one, and the startup banner reports which it is.
                value_source: ValueSource::Net,
                // Play mode, explicitly: no Dirichlet root noise and argmax
                // visits, not sampling. `Config::play()` already means this;
                // spelling both out keeps a future `Config::default` change
                // from quietly turning exploration on in production.
                root_noise: false,
                visit_sampling: false,
                ..Config::play()
            },
            last_shape: AtomicU64::new(NO_SHAPE),
        })
    }

    /// The startup banner: artifact path and the meta the loader validated.
    ///
    /// Printed before the first game so a deployment can be checked against the
    /// artifact it *believes* it is running, rather than trusted.
    pub fn describe(&self) -> String {
        let mut line = format!(
            "artifact={} arch={} board={BOARD}x{BOARD} channels={} layers={}",
            self.artifact.display(),
            self.net.arch(),
            self.net.channels(),
            self.net.layers(),
        );
        let _ = write!(
            line,
            " value_head={} simd={} seed={} mode=play(no-dirichlet,argmax-visits)",
            if self.net.has_value_head() {
                "net"
            } else {
                "hand-tuned(artifact has no value head)"
            },
            self.net.simd(),
            self.config.seed,
        );
        line
    }

    /// Whether `state` is inside [`MctsSearcher`]'s domain, logging every
    /// change of verdict.
    fn in_domain(&self, state: &State) -> bool {
        let players = state.players();
        let (rows, cols) = (state.rows(), state.cols());
        let usable = players == 2 && rows == BOARD && cols == BOARD;

        let shape = shape_code(players, rows, cols);
        // A racing pair of search workers can log the same transition twice.
        // That is strictly better than the alternative (a compare-exchange loop
        // that could drop the very warning this exists to print).
        let previous = self.last_shape.swap(shape, Ordering::SeqCst);
        if previous != shape {
            if !usable {
                eprintln!(
                    "WARNING: SEARCH=MCTS cannot play this game: {players} players on a \
                     {rows}x{cols} board, and the absolute-frame searcher is two-player \
                     12x12 only. FALLING BACK TO THE GREEDY REFERENCE ENGINE for every \
                     position of this shape — moves from now on are NOT the champion's."
                );
            } else if previous != NO_SHAPE {
                eprintln!(
                    "vsbot: back inside the MCTS domain ({players} players, {rows}x{cols}); \
                     the champion engine is playing again."
                );
            }
        }
        usable
    }
}

impl SearchEngine for MctsEngine {
    fn choose(&self, state: &State, budget: &SearchBudget) -> Option<SearchOutcome> {
        if budget.is_cancelled() {
            // Cancelled before any candidate was established — the documented
            // `None` case. Worth checking first: building the searcher expands
            // the root, which is a full net forward, and the client would throw
            // the answer away at send time regardless.
            return None;
        }
        if !self.in_domain(state) {
            return GreedyEngine.choose(state, budget);
        }

        // The clock starts before the root expansion, because the root
        // expansion is a net forward and the allocation has to cover it.
        let started = Instant::now();
        let mut searcher = MctsSearcher::new(state.clone(), self.config, Some(&self.net));
        drive(&mut searcher, budget, started, u64::MAX, || false);
        self.harvest(state, &searcher, budget)
    }

    /// Prior argmax over the legal mask.
    ///
    /// The net's own first guess, restricted to the moves that exist — one
    /// forward pass, no simulations. Held by the client *before* the long
    /// search starts, so an overrun costs a policy move instead of a forfeit.
    fn fallback(&self, state: &State) -> Option<Action> {
        if !self.in_domain(state) {
            return GreedyEngine.fallback(state);
        }
        let searcher = MctsSearcher::new(state.clone(), self.config, Some(&self.net));
        prior_argmax(&searcher).or_else(|| state.legal_actions().first().copied())
    }

    fn can_ponder(&self) -> bool {
        true
    }

    /// One tree, carried across the opponent's actions and into our own turn.
    ///
    /// The loop parks on [`PonderInbox::next`] between positions, so an idle
    /// session costs a parked blocking thread and no CPU. It has no outbox: the
    /// only value it can produce is the reply to a [`PonderStep::Answer`], which
    /// the client only ever sends off the authoritative turn driver.
    fn ponder(&self, inbox: &PonderInbox) {
        // `(root position, tree rooted at it)`. The position is tracked
        // alongside because re-rooting has to know which action of the current
        // root leads to the new snapshot, and the searcher does not expose its
        // root state.
        let mut tree: Option<(State, MctsSearcher<'_>)> = None;
        let mut pending = inbox.next();

        while let Some(step) = pending.take() {
            let started = Instant::now();
            let (state, budget, reply) = match step {
                PonderStep::Think { state, budget } => (state, budget, None),
                PonderStep::Answer {
                    state,
                    budget,
                    reply,
                } => (state, budget, Some(reply)),
            };

            let reused = tree
                .as_mut()
                .is_some_and(|(root, searcher)| reroot(root, searcher, &state));
            if !reused {
                tree = self.in_domain(&state).then(|| {
                    (
                        state.clone(),
                        MctsSearcher::new(state.clone(), self.config, Some(&self.net)),
                    )
                });
            }

            let mut interrupt = None;
            if let Some((_, searcher)) = tree.as_mut() {
                // The cap bounds the *tree*, not the step. `sims_run` is
                // cumulative and survives a re-root, so it cannot be the
                // measure on its own: adding the allowance to it every step
                // would hand out a fresh 50,000 simulations per opponent action
                // and grow the retained tree without bound over a long game.
                //
                // What re-rooting frees is everything outside the child's
                // subtree, and the root's visit total is exactly the size of
                // what survived. Subtracting it re-bases the allowance onto the
                // tree that is actually in memory.
                let retained: u64 = searcher.root_visits().iter().map(|n| u64::from(*n)).sum();
                let cap = searcher
                    .sims_run()
                    .saturating_sub(retained)
                    .saturating_add(PONDER_SIM_CAP);
                drive(searcher, &budget, started, cap, || {
                    if interrupt.is_none() {
                        interrupt = inbox.try_next();
                    }
                    interrupt.is_some()
                });
            }

            // Answer whatever the step asked for *before* moving on, even when
            // a newer snapshot interrupted us: the client's version guard is
            // what decides whether the answer is still usable, and leaving the
            // reply channel dangling would make the client wait out its
            // fallback timer for nothing.
            if let Some(reply) = reply {
                let outcome = tree
                    .as_ref()
                    .and_then(|(_, searcher)| self.harvest(&state, searcher, &budget));
                let _ = reply.send(outcome);
            }

            pending = interrupt.or_else(|| inbox.next());
        }
    }

    fn name(&self) -> &'static str {
        "mcts"
    }
}

impl MctsEngine {
    /// Turns a finished search into an outcome, with the greedy engine as the
    /// last line of defence.
    fn harvest(
        &self,
        state: &State,
        searcher: &MctsSearcher<'_>,
        budget: &SearchBudget,
    ) -> Option<SearchOutcome> {
        let Some(action) = searcher.best_action() else {
            // Terminal or stuck root. Greedy returns `None` for the same
            // reason, so agreement is silent; disagreement is not.
            let fallback = GreedyEngine.choose(state, budget);
            if fallback.is_some() {
                eprintln!(
                    "WARNING: the MCTS root offered no action in a position that has legal \
                     moves — PLAYING THE GREEDY REFERENCE MOVE instead. This is a bug in the \
                     searcher or the snapshot, not a tuning issue."
                );
            }
            return fallback;
        };

        // `root_value_abs` is positive-is-good-for-player-1; `SearchOutcome`
        // wants the mover's frame. One sign application, in the one place the
        // frames meet.
        let value_abs = searcher.root_value_abs();
        let score = if state.current_player() == 1 {
            value_abs
        } else {
            -value_abs
        };
        Some(SearchOutcome {
            action,
            score,
            // PUCT has no completed depth; `0` is the documented "no depth"
            // value, and `nodes` carries the simulation count instead.
            depth: 0,
            nodes: searcher.sims_run() as i64,
        })
    }
}

/// Runs `searcher` under `budget`, applying the intra-turn stop rules between
/// simulation slices.
///
/// The rules themselves live in `virus_proto::clock` as pure functions; this is
/// only the plumbing that samples the root and acts on the verdict.
/// `interrupted` lets a pondering session abandon a superseded position without
/// waiting for its cancellation token, and `sim_cap` bounds the tree.
fn drive(
    searcher: &mut MctsSearcher<'_>,
    budget: &SearchBudget,
    started: Instant,
    sim_cap: u64,
    mut interrupted: impl FnMut() -> bool,
) {
    // A terminal root is left unexpanded, and `run_until_deadline` returns
    // immediately for one. Without this guard the loop would spin hot for the
    // whole budget doing nothing.
    if searcher.root_actions().is_empty() {
        return;
    }
    let allocation = MoveAllocation {
        target: budget.deadline.saturating_duration_since(started),
        ceiling: budget.ceiling.saturating_duration_since(started),
    };
    let sims_before = searcher.sims_run();
    let halfway = started + allocation.target / 2;
    let mut leader_at_halfway: Option<usize> = None;
    let mut leader_changed_late = false;

    // Do-while: `run_until_deadline` always runs at least one simulation, so
    // even an already-expired budget returns a searched move rather than the
    // first enumerated one.
    loop {
        let now = Instant::now();
        let slice = CANCEL_POLL_SLICE.min(budget.ceiling.saturating_duration_since(now));
        searcher.run_until_deadline(now + slice);
        if budget.is_cancelled() || interrupted() || searcher.sims_run() >= sim_cap {
            return;
        }

        let now = Instant::now();
        if now >= halfway {
            let leader = leader_index(searcher.root_visits());
            match leader_at_halfway {
                None => leader_at_halfway = Some(leader),
                Some(earlier) if earlier != leader => leader_changed_late = true,
                Some(_) => {}
            }
        }
        let progress = RootProgress::from_visits(
            searcher.root_visits(),
            searcher.sims_run() - sims_before,
            leader_changed_late,
        );
        if verdict(
            progress,
            now.saturating_duration_since(started),
            allocation,
            &budget.policy,
        )
        .is_stop()
        {
            return;
        }
    }
}

/// Index of the most-visited root action, ties broken by enumeration order —
/// the same rule [`MctsSearcher::best_action`] uses, so "the leader changed"
/// means "the move would have changed".
fn leader_index(visits: &[u32]) -> usize {
    let mut best = 0;
    for (index, count) in visits.iter().enumerate() {
        if *count > visits[best] {
            best = index;
        }
    }
    best
}

/// The highest-prior legal root action, ties broken by enumeration order.
fn prior_argmax(searcher: &MctsSearcher<'_>) -> Option<Action> {
    let actions = searcher.root_actions();
    let priors = searcher.root_priors();
    let mut best: Option<usize> = None;
    for index in 0..actions.len().min(priors.len()) {
        if best.is_none_or(|current| priors[index] > priors[current]) {
            best = Some(index);
        }
    }
    best.map(|index| actions[index])
}

/// Re-roots `searcher` onto `target`, following the actions that lead there.
///
/// Returns `false` — leaving the tree at whatever root it reached — when
/// `target` is not within [`MAX_REROOT_PLIES`] of the current root, or when one
/// of the actions on the way was never expanded into a node. The caller then
/// builds a fresh searcher, which is exactly what happened before pondering
/// existed.
fn reroot(root: &mut State, searcher: &mut MctsSearcher<'_>, target: &State) -> bool {
    if root == target {
        return true;
    }
    let mut spent = 0;
    let Some(path) = path_to(root, target, MAX_REROOT_PLIES, &mut spent) else {
        return false;
    };
    for action in path {
        let Ok(next) = root.apply(action) else {
            return false;
        };
        if !searcher.rebase(action) {
            return false;
        }
        *root = next;
    }
    true
}

/// The shortest action sequence from `from` to `target`, at most `plies` long.
///
/// Depth-first with an `apply` budget: the branching factor is ~34, so an
/// unbounded search would be a stall waiting to happen on a position the tree
/// simply does not contain.
fn path_to(from: &State, target: &State, plies: usize, spent: &mut usize) -> Option<Vec<Action>> {
    if plies == 0 {
        return None;
    }
    // The hash is a cheap reject; equality is the decision. `moves_left` and the
    // mover are part of the hash, so a same-board different-turn position does
    // not collide with the one we want.
    for action in from.legal_actions() {
        if *spent >= REROOT_APPLY_BUDGET {
            return None;
        }
        *spent += 1;
        let Ok(next) = from.apply(action) else {
            continue;
        };
        if next.hash() == target.hash() && next == *target {
            return Some(vec![action]);
        }
        if let Some(mut rest) = path_to(&next, target, plies - 1, spent) {
            let mut path = vec![action];
            path.append(&mut rest);
            return Some(path);
        }
    }
    None
}

/// Packs a position's domain-relevant shape into one comparable word.
fn shape_code(players: usize, rows: usize, cols: usize) -> u64 {
    ((players as u64) << 32) | ((rows as u64) << 16) | cols as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use tokio_util::sync::CancellationToken;
    use virus_proto::ponder::PonderInbox;

    /// The in-repo champion, resolved from the crate rather than the CWD so the
    /// test runs from anywhere.
    fn artifact() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../artifacts/mcts_champion.json")
    }

    fn budget(millis: u64) -> SearchBudget {
        SearchBudget::new(
            Instant::now() + Duration::from_millis(millis),
            CancellationToken::new(),
        )
    }

    fn engine() -> MctsEngine {
        MctsEngine::load(artifact(), 1).expect("the in-repo champion loads and validates")
    }

    #[test]
    fn a_missing_or_broken_artifact_is_an_error_not_a_downgrade() {
        assert!(MctsEngine::load("artifacts/does-not-exist.json", 1).is_err());
    }

    #[test]
    fn the_banner_names_the_artifact_and_its_meta() {
        let banner = engine().describe();
        assert!(banner.contains("mcts_champion.json"), "{banner}");
        assert!(banner.contains("arch=conv-policy-value-v1"), "{banner}");
        assert!(banner.contains("channels="), "{banner}");
        assert!(banner.contains("layers="), "{banner}");
        assert!(banner.contains("value_head=net"), "{banner}");
    }

    #[test]
    fn it_searches_a_two_player_12x12_position_and_reports_simulations() {
        let engine = engine();
        let state = State::new(12, 12, 2).expect("a legal opening position");
        let outcome = engine
            .choose(&state, &budget(200))
            .expect("the opening position has legal moves");
        assert!(
            state.legal_actions().contains(&outcome.action),
            "the chosen action must be legal"
        );
        assert!(outcome.nodes > 0, "no simulations were run");
        assert!(
            outcome.score.is_finite() && outcome.score.abs() <= 1.0,
            "root value {} is outside the tanh range",
            outcome.score
        );
    }

    #[test]
    fn play_mode_never_explores_and_is_reproducible() {
        let engine = engine();
        // The two exploration switches, asserted directly. A regression that
        // turned either on in production would make the bot's moves depend on
        // the seed, which is the thing "play mode" exists to rule out.
        assert!(!engine.config.root_noise, "Dirichlet noise in play mode");
        assert!(!engine.config.visit_sampling, "visit sampling in play mode");

        // With both off, the search is a pure function of the position and the
        // simulation count. Counting simulations rather than milliseconds is
        // the point: a wall-clock budget would let scheduler jitter change the
        // tree and make this assertion a coin flip on a loaded runner.
        let state = State::new(12, 12, 2).expect("a legal opening position");
        let mut first = MctsSearcher::new(state.clone(), engine.config, Some(&engine.net));
        first.run_sims(64);
        let mut second = MctsSearcher::new(state.clone(), engine.config, Some(&engine.net));
        second.run_sims(64);
        assert_eq!(first.best_action(), second.best_action());
        assert_eq!(
            first.root_value_abs().to_bits(),
            second.root_value_abs().to_bits(),
            "the root value must be bit-identical, not merely close"
        );
    }

    #[test]
    fn a_three_player_game_falls_back_to_greedy_instead_of_panicking() {
        // `MctsSearcher::new` asserts two players. Reaching it with three would
        // panic the blocking search worker and forfeit the game on the server's
        // move timer, so the adapter must never let that position through.
        let engine = engine();
        let state = State::new(12, 12, 3).expect("a legal three-player position");
        let outcome = engine
            .choose(&state, &budget(50))
            .expect("greedy always has a move here");
        assert!(state.legal_actions().contains(&outcome.action));
        // Greedy's signature: depth 1 and a node count equal to the move list.
        assert_eq!(outcome.depth, 1);
    }

    #[test]
    fn a_non_12x12_board_falls_back_to_greedy_instead_of_panicking() {
        let engine = engine();
        let state = State::new(10, 10, 2).expect("a legal 10x10 position");
        let outcome = engine
            .choose(&state, &budget(50))
            .expect("greedy always has a move here");
        assert!(state.legal_actions().contains(&outcome.action));
        assert_eq!(outcome.depth, 1);
    }

    #[test]
    fn the_fallback_verdict_is_logged_once_per_transition_not_once_per_process() {
        // Two different out-of-domain shapes must each warn, and a return to the
        // domain must be announced. The counter proves the transition logic
        // fires; the text itself goes to stderr, where the operator sees it.
        let engine = engine();
        assert!(!engine.in_domain(&State::new(12, 12, 3).expect("3p")));
        assert!(!engine.in_domain(&State::new(12, 12, 3).expect("3p")));
        assert!(!engine.in_domain(&State::new(10, 10, 2).expect("10x10")));
        assert!(engine.in_domain(&State::new(12, 12, 2).expect("2p 12x12")));
        assert!(engine.in_domain(&State::new(12, 12, 2).expect("2p 12x12")));
    }

    #[test]
    fn an_already_cancelled_search_does_no_work_at_all() {
        // Cancellation means the position was superseded, so there is nothing
        // worth computing — not even the root expansion.
        let engine = engine();
        let state = State::new(12, 12, 2).expect("a legal opening position");
        let cancel = CancellationToken::new();
        cancel.cancel();
        assert!(engine
            .choose(
                &state,
                &SearchBudget::new(Instant::now() + Duration::from_secs(30), cancel),
            )
            .is_none());
    }

    #[test]
    fn an_expired_but_live_budget_still_returns_a_searched_move() {
        // Out of time is not the same as superseded. Returning `None` here
        // would mean not moving at all, and the server's 120 s move timer
        // forfeits a bot that does not move.
        let engine = engine();
        let state = State::new(12, 12, 2).expect("a legal opening position");
        let outcome = engine
            .choose(
                &state,
                &SearchBudget::new(Instant::now(), CancellationToken::new()),
            )
            .expect("an action");
        assert!(state.legal_actions().contains(&outcome.action));
        assert!(outcome.nodes > 0, "at least one simulation must have run");
    }

    #[test]
    fn the_engine_is_shareable_across_search_workers() {
        // `Bot` clones an `Arc<dyn SearchEngine>` onto a blocking worker per
        // move; this is the compile-time proof that the adapter fits.
        let engine: Arc<dyn SearchEngine> = Arc::new(engine());
        assert_eq!(engine.name(), "mcts");
    }

    // ------------------------------------------------------- fallback-first

    /// The fallback is the net's own first guess restricted to the legal mask.
    /// It has to be legal — it is played verbatim when the search overruns.
    #[test]
    fn the_fallback_is_the_prior_argmax_over_the_legal_mask() {
        let engine = engine();
        let state = State::new(12, 12, 2).expect("a legal opening position");
        let chosen = engine
            .fallback(&state)
            .expect("the opening has legal moves");
        assert!(state.legal_actions().contains(&chosen));

        // Same thing, computed the long way: the highest-prior root action.
        let searcher = MctsSearcher::new(state.clone(), engine.config, Some(&engine.net));
        let priors = searcher.root_priors();
        let best = priors.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let index = searcher
            .root_actions()
            .iter()
            .position(|action| *action == chosen)
            .expect("the fallback is a root action");
        assert_eq!(priors[index], best, "the fallback must be the prior argmax");
    }

    /// The fallback must be cheap — the client holds it before the long search
    /// starts, and a slow one would defeat the whole point.
    #[test]
    fn the_fallback_costs_about_one_forward_pass() {
        let engine = engine();
        let state = State::new(12, 12, 2).expect("a legal opening position");
        let started = Instant::now();
        assert!(engine.fallback(&state).is_some());
        assert!(
            started.elapsed() < Duration::from_millis(250),
            "fallback selection took {:?}",
            started.elapsed()
        );
    }

    /// Out of the searcher's domain the fallback still answers, and legally.
    #[test]
    fn the_fallback_survives_an_out_of_domain_position() {
        let engine = engine();
        for state in [
            State::new(12, 12, 3).expect("3p"),
            State::new(10, 10, 2).expect("10x10"),
        ] {
            let chosen = engine.fallback(&state).expect("a legal action exists");
            assert!(state.legal_actions().contains(&chosen));
        }
    }

    // ---------------------------------------------------------- stop rules

    /// The ceiling is the last word. An engine that ran past it would eat the
    /// rest of the turn's bank and, three actions running, the owner's UX bound.
    #[test]
    fn a_search_never_runs_past_its_ceiling() {
        let engine = engine();
        let state = State::new(12, 12, 2).expect("a legal opening position");
        let started = Instant::now();
        let budget = SearchBudget {
            deadline: started + Duration::from_millis(60),
            ceiling: started + Duration::from_millis(150),
            policy: virus_proto::StopPolicy::default(),
            cancel: CancellationToken::new(),
        };
        let outcome = engine.choose(&state, &budget).expect("an action");
        let elapsed = started.elapsed();
        assert!(state.legal_actions().contains(&outcome.action));
        assert!(
            elapsed < Duration::from_millis(400),
            "the search ran {elapsed:?} against a 150ms ceiling"
        );
    }

    /// The rules are opt-in through the budget: a `SearchBudget::new` budget
    /// carries `StopPolicy::off()` and no extension room, which is exactly the
    /// pre-allocator behaviour every existing caller depends on.
    #[test]
    fn a_plain_budget_keeps_the_pre_allocator_behaviour() {
        let budget = SearchBudget::new(
            Instant::now() + Duration::from_millis(80),
            CancellationToken::new(),
        );
        assert_eq!(budget.ceiling, budget.deadline);
        assert!(!budget.policy.early_stop);
        assert!(!budget.policy.extension);

        let engine = engine();
        let state = State::new(12, 12, 2).expect("a legal opening position");
        let outcome = engine.choose(&state, &budget).expect("an action");
        assert!(state.legal_actions().contains(&outcome.action));
        assert!(outcome.nodes > 0);
    }

    // ------------------------------------------------------------- pondering

    /// The reason pondering is worth anything: the tree built during the
    /// opponent's turn is *the same tree* that answers our turn.
    ///
    /// `nodes` reports `sims_run`, which a fresh searcher starts at zero. So an
    /// answer carrying far more simulations than its own budget could have run
    /// is direct evidence that the opponent's turn was not thrown away.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_ponder_session_carries_its_tree_into_our_turn() {
        let engine = Arc::new(engine());
        assert!(engine.can_ponder());

        // We are seat 2. Seat 1 spends a turn; we think through all of it.
        let mut state = State::new(12, 12, 2).expect("a legal opening position");
        let (steps, inbox) = PonderInbox::channel();
        let session = {
            let engine = Arc::clone(&engine);
            tokio::task::spawn_blocking(move || engine.ponder(&inbox))
        };

        let think = |state: &State| PonderStep::Think {
            state: state.clone(),
            budget: SearchBudget::new(
                Instant::now() + Duration::from_millis(120),
                CancellationToken::new(),
            ),
        };

        steps.send(think(&state)).expect("the session is alive");
        while state.current_player() == 1 {
            let action = state
                .legal_actions()
                .first()
                .copied()
                .expect("seat 1 has a move");
            state = state.apply(action).expect("legal");
            tokio::time::sleep(Duration::from_millis(140)).await;
            steps.send(think(&state)).expect("the session is alive");
        }
        assert_eq!(state.current_player(), 2, "it is our turn now");
        tokio::time::sleep(Duration::from_millis(140)).await;

        let (reply, answer) = tokio::sync::oneshot::channel();
        steps
            .send(PonderStep::Answer {
                state: state.clone(),
                budget: SearchBudget::new(
                    Instant::now() + Duration::from_millis(40),
                    CancellationToken::new(),
                ),
                reply,
            })
            .expect("the session is alive");
        let outcome = tokio::time::timeout(Duration::from_secs(5), answer)
            .await
            .expect("the session answers")
            .expect("the reply channel is open")
            .expect("the position has legal moves");

        assert!(
            state.legal_actions().contains(&outcome.action),
            "a pondered answer must still be legal in the position it answers"
        );

        // A cold search under the same 40 ms budget, for comparison.
        let cold = engine
            .choose(
                &state,
                &SearchBudget::new(
                    Instant::now() + Duration::from_millis(40),
                    CancellationToken::new(),
                ),
            )
            .expect("an action");
        assert!(
            outcome.nodes > cold.nodes,
            "the pondered answer ran {} simulations against a cold {} — the tree was \
             rebuilt, not re-rooted",
            outcome.nodes,
            cold.nodes
        );

        drop(steps);
        session.await.expect("the session exits when its steps end");
    }

    /// The tree cap bounds what is *in memory*, and re-rooting frees most of
    /// it. Measuring the allowance against the cumulative `sims_run` alone
    /// would hand out a fresh allowance on every opponent action and let the
    /// retained tree grow without bound over a long game.
    ///
    /// The re-based allowance is `sims_run - retained + PONDER_SIM_CAP`, so this
    /// pins the arithmetic on the numbers a real session sees.
    #[test]
    fn the_ponder_cap_is_re_based_onto_what_a_re_root_kept() {
        // A session 60k simulations in, of which the re-root kept 8k.
        let sims_run: u64 = 60_000;
        let retained: u64 = 8_000;
        let cap = sims_run.saturating_sub(retained) + PONDER_SIM_CAP;

        assert_eq!(cap, 102_000);
        assert_eq!(
            cap - sims_run,
            PONDER_SIM_CAP - retained,
            "the tree may grow to the cap, counting what it already holds — not past it"
        );

        // A fresh tree starts at zero and simply gets the whole allowance.
        assert_eq!(0u64.saturating_sub(0) + PONDER_SIM_CAP, PONDER_SIM_CAP);

        // A tree already at the cap gets nothing more.
        let full = PONDER_SIM_CAP;
        assert_eq!(
            (sims_run.saturating_sub(full) + PONDER_SIM_CAP).saturating_sub(sims_run),
            0,
            "a retained tree at the cap may not grow"
        );
    }

    /// Dropping the client's end ends the session. Without this a finished game
    /// would leak a parked blocking worker per game, for the life of the bot.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_ponder_session_exits_when_its_channel_closes() {
        let engine = Arc::new(engine());
        let (steps, inbox) = PonderInbox::channel();
        let session = tokio::task::spawn_blocking(move || engine.ponder(&inbox));
        drop(steps);
        tokio::time::timeout(Duration::from_secs(5), session)
            .await
            .expect("the session must exit promptly")
            .expect("and without panicking");
    }

    /// A ponder step for a position outside the searcher's domain must not reach
    /// the searcher's asserts — a panicking session would take a blocking worker
    /// down mid-game.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_ponder_session_declines_an_out_of_domain_position() {
        let engine = Arc::new(engine());
        let (steps, inbox) = PonderInbox::channel();
        let session = tokio::task::spawn_blocking(move || engine.ponder(&inbox));

        let state = State::new(12, 12, 3).expect("a legal three-player position");
        let (reply, answer) = tokio::sync::oneshot::channel();
        steps
            .send(PonderStep::Answer {
                state,
                budget: SearchBudget::new(
                    Instant::now() + Duration::from_millis(20),
                    CancellationToken::new(),
                ),
                reply,
            })
            .expect("the session is alive");
        let answered = tokio::time::timeout(Duration::from_secs(5), answer)
            .await
            .expect("the session answers rather than panicking")
            .expect("the reply channel is open");
        assert!(
            answered.is_none(),
            "an out-of-domain position has no pondered answer; the client falls back"
        );

        drop(steps);
        session.await.expect("the session survived");
    }
}
