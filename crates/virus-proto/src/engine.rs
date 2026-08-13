//! The engine seam.
//!
//! The client owns turn discipline, snapshot validation and cancellation; the
//! engine owns move choice and nothing else. Everything strength-related plugs
//! in through [`SearchEngine`], so the alpha-beta and MCTS crates can land
//! later without touching a line of protocol code.

use crate::clock::{MoveAllocation, StopPolicy};
use crate::ponder::PonderInbox;
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
///
/// [`SearchBudget::ceiling`] is the intra-turn allocator's extension room: an
/// engine that implements the visit-based stop rules
/// ([`crate::clock::verdict`]) may run past `deadline` toward `ceiling` while
/// its root is unstable, and must stop at `ceiling` unconditionally. An engine
/// that ignores it and stops at `deadline` is still correct — that is what
/// `ceiling == deadline` means, and it is what [`SearchBudget::new`] builds.
#[derive(Clone, Debug)]
pub struct SearchBudget {
    /// Wall-clock instant the search aims to return by.
    pub deadline: Instant,
    /// Wall-clock instant the search must return by, come what may. Never
    /// earlier than [`SearchBudget::deadline`].
    pub ceiling: Instant,
    /// Which visit-based stop rules apply.
    pub policy: StopPolicy,
    /// Fires when the position this search started from is superseded.
    pub cancel: CancellationToken,
}

impl SearchBudget {
    /// A budget that ends at `deadline`, with no extension room.
    pub fn new(deadline: Instant, cancel: CancellationToken) -> SearchBudget {
        SearchBudget {
            deadline,
            ceiling: deadline,
            policy: StopPolicy::off(),
            cancel,
        }
    }

    /// The budget for an allocated action of a turn, measured from `started`.
    pub fn allocated(
        started: Instant,
        allocation: MoveAllocation,
        policy: StopPolicy,
        cancel: CancellationToken,
    ) -> SearchBudget {
        SearchBudget {
            deadline: allocation.target_deadline(started),
            ceiling: allocation.ceiling_deadline(started),
            policy,
            cancel,
        }
    }

    /// Whether the position has been superseded.
    pub fn is_cancelled(&self) -> bool {
        self.cancel.is_cancelled()
    }

    /// Whether the search must stop now — deadline reached or cancelled.
    pub fn is_expired(&self) -> bool {
        self.is_cancelled() || Instant::now() >= self.deadline
    }

    /// Whether the search must stop now even counting its extension room.
    pub fn is_exhausted(&self) -> bool {
        self.is_cancelled() || Instant::now() >= self.ceiling
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

    /// A legal action to answer with if the search overruns its deadline.
    ///
    /// **Fallback-first discipline** (superiority.md §2b, the MCTS analogue of
    /// ARCHITECTURE.md invariant 3): the client asks for this *before* it starts
    /// the long search, and plays it if the search has not answered by the hard
    /// deadline. A bot that does not move loses on the server's timer, so
    /// "something legal, chosen cheaply" always beats "the best move, too late".
    ///
    /// Implementations must be **fast** — one net forward at most — and must
    /// return a legal action whenever one exists. The default is the first legal
    /// action, which is instant and always legal.
    fn fallback(&self, state: &State) -> Option<Action> {
        state.legal_actions().first().copied()
    }

    /// Whether this engine can run a pondering session.
    ///
    /// Default `false`: the client then never opens a session and pondering is
    /// simply off for this engine, rather than half-wired.
    fn can_ponder(&self) -> bool {
        false
    }

    /// Called once per accepted `game_start`, before the opening search.
    ///
    /// The engine seam is otherwise stateless across games by design, and every
    /// engine in this repository ignores this hook. It exists for the one thing
    /// a `choose`-only seam genuinely cannot express: a **per-game** random
    /// stream. Cross-play measured 400 games that contained 65 distinct ones
    /// (bd `vsbot-t3q.2`) because two deterministic bots replay one opening
    /// forever, and the fix — seeded eps-greedy openings, the same discipline
    /// `virus_arena::gauntlet` applies — needs to know where one game ends and
    /// the next begins. A decorator cannot infer that from positions alone: a
    /// rematch can open on a board identical to the last game's.
    ///
    /// Implementations must be **fast and non-blocking**; this runs on the
    /// WebSocket read loop. The default does nothing.
    fn on_game_start(&self) {}

    /// Runs a pondering session until the client tears it down.
    ///
    /// Called once per game on a blocking worker. The implementation loops on
    /// [`PonderInbox::next`], keeping one search tree alive across steps and
    /// re-rooting it into each new position, and returns as soon as `next`
    /// yields `None`.
    ///
    /// It must never produce an action other than as the reply to a
    /// [`crate::ponder::PonderStep::Answer`]; it has no other way to, and the
    /// client would reject one anyway.
    fn ponder(&self, inbox: &PonderInbox) {
        crate::ponder::decline(inbox);
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
/// [`EngineKind::Mcts`] is the production default and [`EngineKind::Greedy`] is
/// the reference engine in this crate; [`EngineKind::AlphaBeta`] is the
/// documented plug-in point for `virus-search`, which is not merged yet.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum EngineKind {
    /// PUCT + policy/value net (`virus-mcts`). The engine the bot plays with.
    #[default]
    Mcts,
    /// The reference engine in this crate: deliberately weak and instant.
    Greedy,
    /// Enhanced iterative-deepening alpha-beta (`virus-search`). Not merged.
    AlphaBeta,
}

impl EngineKind {
    /// Parses `SEARCH`. Case-insensitive; unknown values are an error rather
    /// than a silent fallback, because a typo that quietly downgrades the
    /// engine is exactly how a harness ends up reporting the wrong engine's
    /// results (see the Java `unwiredEvalWarning` post-mortem).
    ///
    /// An empty value means "unset", which resolves to [`EngineKind::default`]
    /// — the same answer the binary reaches when `SEARCH` is absent, so the two
    /// spellings of "I did not choose" can never disagree.
    pub fn parse(value: &str) -> Result<EngineKind, UnknownEngine> {
        match value.trim().to_ascii_uppercase().as_str() {
            "" => Ok(EngineKind::default()),
            "GREEDY" => Ok(EngineKind::Greedy),
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
            "unknown SEARCH={:?}; expected MCTS, GREEDY or ALPHABETA",
            self.0
        )
    }
}

impl std::error::Error for UnknownEngine {}
