//! Pondering: thinking on positions we are not allowed to act on.
//!
//! The server streams a snapshot on *every* action, including the opponent's,
//! so a bot can search the opponent-to-move position while they think and
//! re-root into the matched child each time they act (superiority.md §2b). With
//! per-action nodes, that re-rooting is free: the child of the action they
//! played is already in the tree.
//!
//! # Why a channel and not a shared tree
//!
//! A `virus_mcts::MctsSearcher` borrows the net it searches with, so it cannot
//! be stored next to the net it borrows from, and it cannot be handed between
//! tasks. Instead the whole session lives inside **one** blocking call —
//! [`SearchEngine::ponder`](crate::engine::SearchEngine::ponder) — with the tree
//! as a local variable, and the client steers it through [`PonderStep`]s. The
//! tree therefore survives an opponent action, and survives into our own turn,
//! without ever being shared.
//!
//! # The three hard guards
//!
//! 1. **A ponder session can never emit an action.** It has no outbox: the only
//!    value it can produce is the reply to a [`PonderStep::Answer`], and an
//!    `Answer` is only ever sent by the authoritative turn driver with
//!    `current == me`. The reply then goes through the same
//!    `ActionGuard` revalidation — in the search task and again in the writer —
//!    as any other action.
//! 2. **Version-gated cancellation on every accepted snapshot.** Each step
//!    carries its own [`crate::SearchBudget`] with a fresh
//!    `CancellationToken`; installing a snapshot cancels whatever token is
//!    current. A session therefore never simulates past a snapshot without an
//!    explicit, freshly-tokened re-authorisation from the client.
//! 3. **A ponder session never touches the read loop.** It runs on a blocking
//!    worker; the client only ever does a non-blocking
//!    [`std::sync::mpsc::Sender::send`] into it.

use crate::engine::{SearchBudget, SearchOutcome};
use std::sync::mpsc;
use virus_core::State;

/// One instruction for a pondering session, in arrival order.
#[derive(Debug)]
pub enum PonderStep {
    /// Think about `state` under `budget` and produce nothing.
    ///
    /// `state` is a position we are **not** allowed to act on. A session that
    /// already holds a tree containing it should re-root into it rather than
    /// starting over; that carry-over is the entire point of pondering.
    Think {
        /// The position to think about.
        state: State,
        /// How long, and the token that cuts it short.
        budget: SearchBudget,
    },
    /// Think about `state` under `budget`, then answer.
    ///
    /// Only the authoritative turn driver sends this, and only with
    /// `state.current_player() == our seat`. The reply is advisory: the client
    /// revalidates it against the live position before it reaches the wire, and
    /// answers with its pre-selected fallback if the reply never arrives.
    Answer {
        /// The position to act in.
        state: State,
        /// How long, and the token that cuts it short.
        budget: SearchBudget,
        /// Where the chosen action goes. Dropping it is a legal answer of
        /// "nothing"; the client falls back.
        reply: tokio::sync::oneshot::Sender<Option<SearchOutcome>>,
    },
}

impl PonderStep {
    /// The position this step is about.
    pub fn state(&self) -> &State {
        match self {
            PonderStep::Think { state, .. } | PonderStep::Answer { state, .. } => state,
        }
    }

    /// The budget this step must respect.
    pub fn budget(&self) -> &SearchBudget {
        match self {
            PonderStep::Think { budget, .. } | PonderStep::Answer { budget, .. } => budget,
        }
    }
}

/// A pondering session's end of the step channel.
///
/// Blocking by design: the session runs on a blocking worker and parks on
/// [`PonderInbox::next`] between positions, so an idle session costs a parked
/// thread and no CPU.
#[derive(Debug)]
pub struct PonderInbox {
    steps: mpsc::Receiver<PonderStep>,
}

impl PonderInbox {
    /// Builds a session channel: the client's sender and the session's inbox.
    pub fn channel() -> (mpsc::Sender<PonderStep>, PonderInbox) {
        let (sender, steps) = mpsc::channel();
        (sender, PonderInbox { steps })
    }

    /// Blocks for the next step.
    ///
    /// `None` means the client dropped the session — the game ended, the socket
    /// died, or pondering was torn down. The engine must return promptly.
    pub fn next(&self) -> Option<PonderStep> {
        self.steps.recv().ok()
    }

    /// The next step if one is already queued, without blocking.
    ///
    /// An engine polls this between simulation slices: a queued step means the
    /// position it is thinking about has been superseded, and continuing to
    /// simulate on it is wasted work.
    pub fn try_next(&self) -> Option<PonderStep> {
        self.steps.try_recv().ok()
    }
}

/// Drains a session's steps without thinking, answering every request with
/// "nothing".
///
/// The default [`SearchEngine::ponder`](crate::engine::SearchEngine::ponder)
/// body. An engine that declares
/// [`can_ponder`](crate::engine::SearchEngine::can_ponder) never sees it, but
/// leaving the default as a *silent* no-op would hang the client on the first
/// `Answer` until its fallback timer fired, once per turn, for the life of the
/// process.
pub fn decline(inbox: &PonderInbox) {
    while let Some(step) = inbox.next() {
        if let PonderStep::Answer { reply, .. } = step {
            let _ = reply.send(None);
        }
    }
}
