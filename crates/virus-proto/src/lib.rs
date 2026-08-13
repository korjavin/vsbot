//! WebSocket bot client.
//!
//! A port of the Go bot client (`virusgame/backend/cmd/bot-hoster/bot_client.go`),
//! hardened with the fixes the Java client (`nnue-trainer` `GameLoopHandler`)
//! had to make live.
//!
//! # The shape of it
//!
//! * [`message`] — the wire catalog, tolerant on the way in, exact on the way out.
//! * [`engine`] — the [`SearchEngine`] seam. Everything strength-related plugs
//!   in here; this crate ships only [`GreedyEngine`], deliberately weak, so the
//!   protocol can be exercised end-to-end before the real engines land.
//! * [`clock`] — the intra-turn time allocator and the visit-based stop rules.
//!   Pure functions, so the time manager is testable without a wall clock.
//! * [`ponder`] — the seam for thinking on the opponent's positions. A session
//!   has no outbox and cannot emit an action; see the module docs.
//! * [`bot`] — the state machine. Transport-free, and where every invariant is
//!   enforced and tested.
//! * [`client`] — tokio-tungstenite transport: read loop, guarded writer,
//!   challenger timer, reconnect with backoff.
//! * [`config`] — plain configuration data. **Nothing here reads the
//!   environment**; the `vsbot` binary does that in one place (CLAUDE.md).
//!
//! # The two invariants that cost live games
//!
//! * ARCHITECTURE.md invariant 2 — **never act on your own `neutrals_placed`
//!   ack.** Its snapshot still shows us as mover with `movesLeft > 0`; acting
//!   on it caused two forfeits on 2026-08-08. `turn_change` is the
//!   authoritative turn driver. Enforced by [`bot::Driver`]: only `game_start`,
//!   `game_state` and `turn_change` may start a search.
//! * ARCHITECTURE.md invariant 5 — the snapshot is the only board source, every
//!   snapshot is re-validated through [`virus_core::Snapshot::decode`], and a
//!   new snapshot cancels the in-flight search via a version counter. A result
//!   is re-checked against `{game, version, seat, movesLeft, legality}` in the
//!   search task *and* again in the writer.
//!
//! # Example
//!
//! ```no_run
//! use std::sync::Arc;
//! use virus_proto::{Bot, BotConfig, GreedyEngine};
//!
//! # async fn run() {
//! let config = Arc::new(BotConfig::default());
//! let (bot, mut inbox) = Bot::new(config, Arc::new(GreedyEngine));
//! virus_proto::run_forever(&bot, &mut inbox).await;
//! # }
//! ```

#![deny(missing_docs)]
#![deny(missing_debug_implementations)]

pub mod bot;
pub mod client;
pub mod clock;
pub mod config;
pub mod engine;
pub mod message;
pub mod ponder;

pub use bot::{ActionGuard, Bot, BotCore, Counters, Driver, Outbound, Phase};
pub use client::{run_forever, run_session, ProtoError};
pub use clock::{MoveAllocation, RootProgress, StopPolicy, TurnAllocator, Verdict};
pub use config::{connect_url, BotConfig, Rng};
pub use engine::{
    EngineKind, GreedyEngine, SearchBudget, SearchEngine, SearchOutcome, UnknownEngine,
};
pub use message::{CellPos, Diagnostics, Inbound, Outgoing, UserInfo};
pub use ponder::{PonderInbox, PonderStep};
