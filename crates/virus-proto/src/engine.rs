//! The engine seam.
//!
//! The client owns turn discipline, snapshot validation and cancellation; the
//! engine owns move choice and nothing else. Everything strength-related plugs
//! in through [`SearchEngine`], so the alpha-beta and MCTS crates can land
//! later without touching a line of protocol code.

use std::fmt;
use std::time::Instant;
use tokio_util::sync::CancellationToken;
use virus_core::{Action, CellKind, State};

/// What a search is allowed to spend.
///
/// A search must stop at [`SearchBudget::deadline`] *and* poll
/// [`SearchBudget::is_cancelled`]. Cancellation fires the instant a newer
/// snapshot is accepted (ARCHITECTURE.md invariant 5): the answer is already
/// worthless, and the client will discard it at send time regardless — polling
/// just stops the machine burning cycles on it.
#[derive(Clone, Debug)]
pub struct SearchBudget {
    /// Wall-clock instant the search must return by.
    pub deadline: Instant,
    /// Fires when the position this search started from is superseded.
    pub cancel: CancellationToken,
}

impl SearchBudget {
    /// A budget of `millis` from now, tied to `cancel`.
    pub fn new(deadline: Instant, cancel: CancellationToken) -> SearchBudget {
        SearchBudget { deadline, cancel }
    }

    /// Whether the position has been superseded.
    pub fn is_cancelled(&self) -> bool {
        self.cancel.is_cancelled()
    }

    /// Whether the search must stop now — deadline reached or cancelled.
    pub fn is_expired(&self) -> bool {
        self.is_cancelled() || Instant::now() >= self.deadline
    }
}

/// A chosen action plus the diagnostics the server relays to spectators.
#[derive(Clone, Copy, Debug)]
pub struct SearchOutcome {
    /// The action to play. Must be legal in the state it was chosen from.
    pub action: Action,
    /// Root score in the mover's frame.
    pub score: f64,
    /// Completed depth, or `0` for engines without one.
    pub depth: i32,
    /// Nodes or simulations spent.
    pub nodes: i64,
}

impl SearchOutcome {
    /// An outcome carrying only an action.
    pub fn new(action: Action) -> SearchOutcome {
        SearchOutcome {
            action,
            score: 0.0,
            depth: 0,
            nodes: 0,
        }
    }
}

/// Chooses an action for the side to move.
///
/// Implementations run on a blocking worker, never on the WebSocket read loop —
/// the Java predecessor searched inline and starved its pong deadline. They may
/// be called concurrently for different positions, so `&self` state must be
/// internally synchronised.
///
/// Returning `None` means "no action" (no legal move, or cancelled before any
/// candidate was established); the client then simply waits for the next
/// authoritative snapshot.
pub trait SearchEngine: Send + Sync + 'static {
    /// Picks an action for `state`, respecting `budget`.
    fn choose(&self, state: &State, budget: &SearchBudget) -> Option<SearchOutcome>;

    /// Short name for logs.
    fn name(&self) -> &'static str {
        "engine"
    }
}

/// The reference engine: take the first capture, else the first legal move.
///
/// Deliberately weak and instant. It exists so the protocol layer can be
/// exercised end-to-end against a real server before the real engines land, and
/// so protocol regressions are never masked by engine noise.
#[derive(Clone, Copy, Debug, Default)]
pub struct GreedyEngine;

impl SearchEngine for GreedyEngine {
    fn choose(&self, state: &State, _budget: &SearchBudget) -> Option<SearchOutcome> {
        let actions = state.legal_actions();
        let mover = state.current_player();
        let capture = actions.iter().copied().find(|action| match *action {
            Action::Move { target } => {
                let cell = state.at(target);
                cell.kind() == CellKind::Normal && cell.owner() != mover
            }
            Action::PlaceNeutrals { .. } => false,
        });
        // `legal_actions` enumerates every move before any neutral pair, so the
        // fallback is a move whenever one exists.
        let action = capture.or_else(|| actions.first().copied())?;
        Some(SearchOutcome {
            action,
            score: if capture.is_some() { 1.0 } else { 0.0 },
            depth: 1,
            nodes: actions.len() as i64,
        })
    }

    fn name(&self) -> &'static str {
        "greedy"
    }
}

/// Which engine the binary should build.
///
/// Only [`EngineKind::Greedy`] is wired today; the other two are the documented
/// plug-in points for `virus-search` and `virus-mcts`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum EngineKind {
    /// The reference engine in this crate.
    #[default]
    Greedy,
    /// Enhanced iterative-deepening alpha-beta (`virus-search`). Not merged.
    AlphaBeta,
    /// PUCT + policy/value net (`virus-mcts`). Not merged.
    Mcts,
}

impl EngineKind {
    /// Parses `SEARCH`. Case-insensitive; unknown values are an error rather
    /// than a silent fallback, because a typo that quietly downgrades the
    /// engine is exactly how a harness ends up reporting the wrong engine's
    /// results (see the Java `unwiredEvalWarning` post-mortem).
    pub fn parse(value: &str) -> Result<EngineKind, UnknownEngine> {
        match value.trim().to_ascii_uppercase().as_str() {
            "" | "GREEDY" => Ok(EngineKind::Greedy),
            "ALPHABETA" | "ALPHA_BETA" => Ok(EngineKind::AlphaBeta),
            "MCTS" => Ok(EngineKind::Mcts),
            _ => Err(UnknownEngine(value.to_owned())),
        }
    }

    /// The `SEARCH` spelling of this engine.
    pub fn as_str(self) -> &'static str {
        match self {
            EngineKind::Greedy => "GREEDY",
            EngineKind::AlphaBeta => "ALPHABETA",
            EngineKind::Mcts => "MCTS",
        }
    }
}

/// `SEARCH` named an engine that does not exist.
#[derive(Clone, Debug)]
pub struct UnknownEngine(pub String);

impl fmt::Display for UnknownEngine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "unknown SEARCH={:?}; expected GREEDY, ALPHABETA or MCTS",
            self.0
        )
    }
}

impl std::error::Error for UnknownEngine {}
