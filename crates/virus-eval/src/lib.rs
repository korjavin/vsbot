//! Integer-exact port of the hand-tuned evaluator.
//!
//! **Stub.** Implemented by bead `vsbot-q67`; this crate exists so the
//! workspace builds and the dependency direction (`core <- eval <-
//! search/mcts`) is fixed from day one.
//!
//! When it lands it must be integer-exact against
//! `fixtures/gobot_staticeval_parity.jsonl` — CLAUDE.md treats a parity break
//! as a hard failure, never "close enough".

#![deny(missing_docs)]

/// Score type: centipawn-like integers, never floats. Float evaluation would
/// make byte-exact parity with the Go/Java oracles impossible.
pub type Score = i32;

/// Score returned for a position the side to move has won.
pub const WIN_SCORE: Score = 1_000_000;

/// Placeholder so the crate has a public surface and downstream wiring can
/// compile before the evaluator exists.
pub fn placeholder_score(state: &virus_core::State) -> Score {
    let _ = state;
    0
}
