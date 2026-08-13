//! The Virus rules engine.
//!
//! A port of the Go engine (`virusgame/backend/game/`), cross-checked against
//! the Java port (`nnue-trainer/.../search/gobot/`). Everything that follows is
//! rules and representation only — no evaluation, no search.
//!
//! # The rules in one screen
//!
//! * Rectangular `rows x cols` board (server default 12x12), 2-4 players.
//!   Bases sit in fixed corners in seat order: P1 `(0,0)`,
//!   P2 `(rows-1, cols-1)`, P3 `(0, cols-1)`, P4 `(rows-1, 0)`.
//! * Cells are `Empty`, `Normal(owner)`, `Base(owner)`, `Fortified(owner)` or
//!   `Neutral`. `Base` and `Fortified` are invulnerable; `Neutral` is dead
//!   space.
//! * **Three actions per turn.** A [`Action::Move`] plays an empty cell or
//!   captures an enemy `Normal`; the target must be 8-adjacent to the mover's
//!   base-connected component.
//! * [`Action::PlaceNeutrals`] converts two of your own `Normal` cells to
//!   `Neutral`. Once per game per player, only at turn start
//!   (`moves_left == ACTIONS_PER_TURN`), and it consumes the whole turn.
//! * After every action, any active player with no legal move is eliminated.
//!   Their cells stay on the board and stay capturable. Last active player
//!   wins; turn-capped games are decided by [`State::outcome_winner`].
//!
//! # The four rules that bit the predecessors
//!
//! 1. **The mover does not alternate per ply.** Roughly 47% of legal children
//!    flip the mover and 53% do not, because a turn is three actions. Any
//!    consumer must branch on `state.current_player()`, never on parity.
//! 2. **Capture fortifies.** Taking an enemy `Normal` yields *your*
//!    `Fortified`, not your `Normal`.
//! 3. **Elimination leaves the cells.** Only the active flag flips.
//! 4. **Neutral placement costs the turn**, not one action.
//!
//! # Enumeration order
//!
//! Move order is part of the engine's contract — see the `ENUMERATION ORDER`
//! block in [`state`] and the module docs of [`position`]. Search parity with
//! the Go/Java oracles depends on identical child ordering.

#![deny(missing_docs)]
#![deny(missing_debug_implementations)]

pub mod action;
pub mod cell;
pub mod fixture;
pub mod position;
pub mod scratch;
pub mod snapshot;
pub mod state;
mod zobrist;

pub use action::{Action, Pos};
pub use cell::{Cell, CellKind, Player, MAX_PLAYERS};
pub use position::{Position, EXACT_BRANCH_LIMIT, MAX_STRATEGIC_PAIRS};
pub use scratch::Scratch;
pub use snapshot::{Snapshot, SnapshotError};
pub use state::{RuleError, State, ACTIONS_PER_TURN, MAX_CELLS, MAX_DIM};
