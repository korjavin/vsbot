//! Flat action ids — the trainer's policy-target space.
//!
//! `python/mcts/train_policy.py` and `SelfPlayMcts.flatIndex` agree on one
//! numbering, and self-play rows are written in it, so the Rust generator must
//! reproduce it exactly or every emitted policy target lands on the wrong
//! output:
//!
//! * a [`Action::Move`] is its target's row-major cell index, `0..144`;
//! * a [`Action::PlaceNeutrals`] on cells `i < j` is `144 + i * 144 + j`.
//!
//! The space is [`ACTION_ID_COUNT`] wide and is 12x12-only, like the net.

use virus_core::{Action, Pos};

use crate::net::{BOARD, CELLS};

/// Size of the flat action space: 144 moves plus 144x144 ordered pair slots.
pub const ACTION_ID_COUNT: usize = CELLS + CELLS * CELLS;

/// Row-major cell index of a 12x12 position.
///
/// Both coordinates are bounds-checked *before* they are folded together.
/// Checking only the folded index would admit `(0, 12)` as cell 12, aliasing
/// `(1, 0)` — two different actions sharing one policy target, which in an
/// emitted self-play row is a silently mislabelled training example rather than
/// a crash. Negative coordinates alias just as badly once cast to `usize`.
///
/// # Panics
/// Panics on any coordinate outside `0..12`.
fn cell(pos: Pos) -> usize {
    assert!(
        (0..BOARD as i32).contains(&pos.row) && (0..BOARD as i32).contains(&pos.col),
        "{pos:?} is off the {BOARD}x{BOARD} board"
    );
    pos.row as usize * BOARD + pos.col as usize
}

/// The flat id of an action on a 12x12 board.
///
/// # Panics
/// Panics on coordinates outside the 12x12 board, or on a `PlaceNeutrals` whose
/// two cells coincide — neither is producible by `virus-core`'s enumerator.
pub fn action_id(action: Action) -> usize {
    match action {
        Action::Move { target } => cell(target),
        Action::PlaceNeutrals { cells } => {
            let (a, b) = (cell(cells[0]), cell(cells[1]));
            assert!(a != b, "neutral pair {cells:?} repeats a cell");
            CELLS + a.min(b) * CELLS + a.max(b)
        }
    }
}

/// Inverse of [`action_id`]; `None` for an id outside the space or one that
/// encodes a degenerate `i >= j` pair.
pub fn action_from_id(id: usize) -> Option<Action> {
    let pos = |index: usize| Pos::new((index / BOARD) as i32, (index % BOARD) as i32);
    if id < CELLS {
        return Some(Action::Move { target: pos(id) });
    }
    let rest = id.checked_sub(CELLS)?;
    if rest >= CELLS * CELLS {
        return None;
    }
    let (i, j) = (rest / CELLS, rest % CELLS);
    if i >= j {
        return None;
    }
    Some(Action::neutrals(pos(i), pos(j)))
}
