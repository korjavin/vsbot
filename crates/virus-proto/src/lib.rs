//! WebSocket bot client.
//!
//! **Stub.** Implemented by bead `vsbot-kw5`; this crate exists so the
//! workspace builds and the dependency direction is fixed.
//!
//! Two invariants govern this crate before a line of it is written:
//!
//! * ARCHITECTURE.md invariant 2 — **never act on your own `neutrals_placed`
//!   ack.** Its snapshot still shows you as mover with `movesLeft > 0`; acting
//!   on it caused two live forfeits on 2026-08-08. `turn_change` is the
//!   authoritative turn driver.
//! * ARCHITECTURE.md invariant 5 — the snapshot is the only board source, every
//!   snapshot is re-validated through [`virus_core::Snapshot::decode`], and a
//!   new snapshot cancels the in-flight search via a version counter.

#![deny(missing_docs)]

/// Placeholder so the crate has a public surface.
pub fn placeholder() {}
