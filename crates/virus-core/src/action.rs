//! Board coordinates and the two action kinds.

use std::fmt;

/// A board coordinate. Signed so neighbourhood scans can compute `row - 1`
/// without underflow, exactly like the Go/Java originals.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Pos {
    /// Row, `0..rows`.
    pub row: i32,
    /// Column, `0..cols`.
    pub col: i32,
}

impl Pos {
    /// Builds a coordinate.
    pub const fn new(row: i32, col: i32) -> Pos {
        Pos { row, col }
    }
}

impl fmt::Debug for Pos {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({},{})", self.row, self.col)
    }
}

/// A single action. One turn is [`crate::ACTIONS_PER_TURN`] actions, except for
/// `PlaceNeutrals`, which consumes the whole turn.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Action {
    /// Place on an empty cell, or capture an enemy `Normal`.
    ///
    /// Capturing turns the cell into **your `Fortified`**, not your `Normal`
    /// (ARCHITECTURE.md invariant 4).
    Move {
        /// Cell being played.
        target: Pos,
    },
    /// Convert two of your own `Normal` cells to `Neutral`. Once per game per
    /// player, only at `moves_left == ACTIONS_PER_TURN`, consumes the turn.
    PlaceNeutrals {
        /// The two cells, in enumeration order (not necessarily sorted).
        cells: [Pos; 2],
    },
}

impl Action {
    /// Convenience constructor for a move.
    pub const fn mv(row: i32, col: i32) -> Action {
        Action::Move {
            target: Pos::new(row, col),
        }
    }

    /// Convenience constructor for a neutral placement.
    pub const fn neutrals(a: Pos, b: Pos) -> Action {
        Action::PlaceNeutrals { cells: [a, b] }
    }

    /// True when both actions denote the same game transition. `PlaceNeutrals`
    /// compares as an unordered pair, matching Java's
    /// `PlaceNeutralsAction.equals` and the parity fixtures' contract.
    pub fn same_transition(self, other: Action) -> bool {
        match (self, other) {
            (Action::Move { target: a }, Action::Move { target: b }) => a == b,
            (Action::PlaceNeutrals { cells: a }, Action::PlaceNeutrals { cells: b }) => {
                (a[0] == b[0] && a[1] == b[1]) || (a[0] == b[1] && a[1] == b[0])
            }
            _ => false,
        }
    }
}
