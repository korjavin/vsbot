//! The gauntlet harness: how every strength claim in this project gets made.
//!
//! ARCHITECTURE.md invariant 7 is the reason this crate exists. Seven separate
//! offline metrics — holdout top-1, value MAE, policy agreement, and four more —
//! were each believed to predict playing strength, and each one was wrong. The
//! only thing that ever predicted strength was playing games. So CLAUDE.md's
//! rule is "never gate strength claims on offline metrics; gauntlets only
//! (>=400 games)", and this crate is the gauntlet.
//!
//! # What it does that the predecessors did not
//!
//! It ports the *discipline* from Java's `train/GauntletMatch.java` and Go's
//! `backend/arena/`, and closes three gaps those two left:
//!
//! * **Fixed-time mode.** Java's gauntlet could only budget nodes and depth. A
//!   node is not a comparable unit across engine families — an MCTS simulation
//!   and an alpha-beta node cost wildly different amounts — so the moment the
//!   project needed "MCTS vs alpha-beta at equal compute" the harness had no
//!   answer, and `docs/plans/superiority.md` S0 records that this cost the
//!   search-strength work its first step. [`engine::Budget::Millis`] is here
//!   from the start, deadline-enforced per action, with the worst overrun
//!   reported so a side that ignores the clock is visible in the output.
//! * **Sample-size discipline in the type system.** Both predecessors kept
//!   "≥400 games, never report a 24-game cell" in prose. [`stats::Verdict`]
//!   makes the harness say `INFORMATIONAL ONLY` itself.
//! * **Wilson intervals on every report.** Go had them; Java did not.
//!   [`stats::wilson95`] is Go's, constant for constant.
//!
//! # The shape of a run
//!
//! ```no_run
//! use virus_arena::engine::{Budget, Engine, SideSpec};
//! use virus_arena::gauntlet::{run, GauntletConfig};
//!
//! let config = GauntletConfig {
//!     side_a: SideSpec::parse("ab-enhanced", Budget::Nodes(60_000)).expect("side"),
//!     side_b: SideSpec::parse("ab-plain", Budget::Nodes(60_000)).expect("side"),
//!     games: 400,
//!     seed: 11,
//!     ..GauntletConfig::default()
//! };
//! let result = run(&config, None).expect("gauntlet");
//! println!("{}", result.summary);
//! ```
//!
//! Games `2k` and `2k+1` share an opening seed with the colours swapped, so
//! first-mover advantage cancels within the pair; see [`gauntlet`] for why that
//! matters and [`rng`] for why the seed derivation is not `seed + k`.
//!
//! # Net vs net
//!
//! Two *different* artifacts in one run — the RL loop's candidate-vs-champion
//! gate — is [`gauntlet::run_with_nets`] with a [`gauntlet::SideNets`] per arm,
//! or `--a-net`/`--b-net` on the `arena` binary. [`run`] is the single-artifact
//! case and is exactly `run_with_nets` with both arms borrowing one loaded net,
//! so a one-net run still loads one net.
//!
//! A net follows its *side*, never its seat: the arms swap chairs between the
//! two games of a pair and their artifacts swap with them. `tests/net_vs_net.rs`
//! is the tripwire for that, and it is a tripwire that has been checked to fire.
//!
//! # Cross-play
//!
//! Rust-vs-Rust settles internal A/Bs, but the bar this project must clear is
//! the *deployed* Go bot, which only exists behind a WebSocket server. That
//! harness is `crossplay/crossplay.py` — it boots the Go server, the Go
//! bot-hoster and the `vsbot` binary, and reads the result off the server's own
//! `games.db`. It is a script rather than a subcommand because it orchestrates
//! three external processes and an SQLite file that this crate would otherwise
//! need a C dependency to read. See `README.md`.
//!
//! # The `PlaceNeutrals` probe
//!
//! [`probes`] is the other measuring instrument here, and it measures something
//! the gauntlet cannot: *what a net believes* about one named position class
//! (bd `vsbot-07x`'s unmotivated neutral placements). It is deliberately not a
//! gauntlet and deliberately not a gate — invariant 7 again, from the other
//! direction. Its output carries [`probes::INFORMATIONAL`] so no number out of
//! it can be quoted as a strength claim.

#![deny(missing_docs)]
#![deny(missing_debug_implementations)]

pub mod engine;
pub mod gauntlet;
pub mod probes;
pub mod rng;
pub mod stats;

pub use engine::{Budget, Engine, SideSpec};
pub use gauntlet::{run, run_with_nets, GauntletConfig, GauntletResult, SideNets};
pub use stats::{wilson95, Interval, Record, Summary, Verdict, GATE_MIN_GAMES, VERDICT_MIN_GAMES};
