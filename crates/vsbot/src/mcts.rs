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

use virus_core::State;
use virus_mcts::{Config, MctsSearcher, NetError, PolicyValueNet, ValueSource, BOARD};
use virus_proto::{GreedyEngine, SearchBudget, SearchEngine, SearchOutcome};

/// Longest uninterrupted stretch of simulations.
///
/// [`MctsSearcher::run_until_deadline`] honours a deadline but knows nothing
/// about [`SearchBudget::cancel`], so the search is driven in slices and
/// cancellation is polled between them (ARCHITECTURE.md invariant 5: a
/// superseded position's answer is worthless the instant a newer snapshot
/// lands). 20 ms is short enough to drop a stale search promptly and long
/// enough that the polling itself costs nothing.
const CANCEL_POLL_SLICE: Duration = Duration::from_millis(20);

/// Sentinel for "no position shape has been logged yet".
const NO_SHAPE: u64 = u64::MAX;

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
        if !self.in_domain(state) {
            return GreedyEngine.choose(state, budget);
        }

        let mut searcher = MctsSearcher::new(state.clone(), self.config, Some(&self.net));
        // A terminal root is left unexpanded, and `run_until_deadline` returns
        // immediately for one. Without this guard the loop below would spin hot
        // for the whole move budget doing nothing.
        if !searcher.root_actions().is_empty() {
            // Do-while: `run_until_deadline` always runs at least one
            // simulation, so even an already-expired budget returns a searched
            // move rather than the first enumerated one.
            loop {
                let now = Instant::now();
                let slice = CANCEL_POLL_SLICE.min(budget.deadline.saturating_duration_since(now));
                searcher.run_until_deadline(now + slice);
                if budget.is_cancelled() || Instant::now() >= budget.deadline {
                    break;
                }
            }
        }

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

    fn name(&self) -> &'static str {
        "mcts"
    }
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
    fn a_cancelled_search_still_returns_a_legal_move() {
        let engine = engine();
        let state = State::new(12, 12, 2).expect("a legal opening position");
        let cancel = CancellationToken::new();
        cancel.cancel();
        let outcome = engine
            .choose(
                &state,
                &SearchBudget::new(Instant::now() + Duration::from_secs(30), cancel),
            )
            .expect("an action");
        assert!(state.legal_actions().contains(&outcome.action));
    }

    #[test]
    fn the_engine_is_shareable_across_search_workers() {
        // `Bot` clones an `Arc<dyn SearchEngine>` onto a blocking worker per
        // move; this is the compile-time proof that the adapter fits.
        let engine: Arc<dyn SearchEngine> = Arc::new(engine());
        assert_eq!(engine.name(), "mcts");
    }
}
