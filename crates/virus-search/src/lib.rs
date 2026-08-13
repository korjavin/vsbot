//! Enhanced iterative-deepening alpha-beta searcher.
//!
//! **Stub.** Implemented by a later bead; this crate exists so the workspace
//! builds and the dependency direction is fixed.
//!
//! Two invariants are already load-bearing for whoever writes it:
//!
//! * ARCHITECTURE.md invariant 1 — the mover does **not** alternate per ply, so
//!   this must use `maximizing = state.current_player() == root_player`. Naive
//!   negamax negation is wrong here.
//! * ARCHITECTURE.md invariant 3 — a deadline search must return the same move
//!   the fixed-depth search returns at the depth it completed; partial-iteration
//!   salvage only when `best_score > alpha_orig`.

#![deny(missing_docs)]

/// Placeholder so the crate has a public surface.
pub fn placeholder() {}
