//! PUCT searcher and convolutional policy/value net inference.
//!
//! A port of the production champion in `nnue-trainer/.../mcts/`: the
//! `MctsSearcher` and the `PolicyNetPrior` conv net, running the gen-5
//! `mcts_champion.json` artifact.
//!
//! ```no_run
//! use std::time::Duration;
//! use virus_core::State;
//! use virus_mcts::{Config, MctsSearcher, PolicyValueNet, ValueSource};
//!
//! let net = PolicyValueNet::load("artifacts/mcts_champion.json")?;
//! let config = Config {
//!     value_source: ValueSource::Net,
//!     ..Config::play()
//! };
//! let mut searcher = MctsSearcher::new(State::new(12, 12, 2)?, config, Some(&net));
//! searcher.run_for(Duration::from_millis(200));
//! let action = searcher.best_action();
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # The two things this crate gets right that the Java original does not
//!
//! 1. **One trunk forward per expansion.** Java runs the trunk once for the
//!    priors and again for the value head; [`PolicyValueNet::forward`] returns
//!    both from a single pass, which is a free ~2x on net-value search.
//! 2. **Load-time validation.** Every shape and every weight is checked in
//!    [`PolicyValueNet::load`], so a wrong-shape artifact fails at startup
//!    rather than producing `NaN` priors mid-search.
//!
//! # ARCHITECTURE.md invariant 1
//!
//! The mover does not alternate per ply — a turn is three actions, so ~53% of
//! edges keep the mover. The searcher therefore uses **absolute-frame backup**:
//! one sign application at the leaf, none on the way up, and `sign(node)`
//! applied at selection. See [`search`] for the full argument.

#![deny(missing_docs)]
#![deny(missing_debug_implementations)]

pub mod action_id;
pub mod gumbel;
pub mod net;
pub mod parallel;
pub mod rng;
pub mod search;

pub use action_id::{action_from_id, action_id, ACTION_ID_COUNT};
pub use gumbel::{GumbelConfig, DEFAULT_GUMBEL_C_SCALE, DEFAULT_GUMBEL_C_VISIT, DEFAULT_GUMBEL_M};
pub use net::{
    BatchScratch, Encoded, Heads, NetError, NetScratch, PolicyValueNet, BATCH_LANES, BOARD, CELLS,
    PLANES,
};
pub use parallel::ParallelMcts;
pub use rng::Rng;
pub use search::{
    terminal_value_abs, Config, MctsSearcher, ValueSource, DEFAULT_BATCH_SIZE, DEFAULT_CPUCT,
    DEFAULT_DAG, DEFAULT_VALUE_SCALE, DEFAULT_VIRTUAL_LOSS, TEMPERATURE_PLIES,
};
