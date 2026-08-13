//! The wedge opening book.
//!
//! Port of `virusgame/backend/search/opening_book.go`, cross-checked against
//! `nnue-trainer/.../search/gobot/GoOpeningBook.java`.
//!
//! The plain search reply to an empty board is a width-1 diagonal tendril: every
//! cell of it is an articulation point, so one enemy cut forfeits the whole
//! distal chain (Go production losses b543fe02 / bbfc5e0c / 82d29155). Sweeps
//! proved that shape is genuinely eval-optimal from empty, so no eval term
//! dislodges it at a safe weight — the reply is placed by fiat instead.
//!
//! The line is the **wedge**: one base-adjacent diagonal anchor, then a width-2
//! advancing pair, oriented inward. It has the minimum possible base halo —
//! exactly one capturable base-adjacent cell (connectivity needs at least one,
//! and captured cells become enemy fortified footholds). Deterministic, no data
//! files, no tuning.
//!
//! Only the live entry points consult it. [`crate::choose_depth`] — the parity
//! oracle — deliberately skips it, which is why the checked-in fixtures are pure
//! search.

use virus_core::{Action, CellKind, Player, Pos, State, ACTIONS_PER_TURN};

/// The canonical thick first-turn placement for the side to move, or `None` to
/// defer to search.
///
/// Fires only while the mover owns exactly its base plus a prefix of the wedge
/// block — its genuine first turn, spread over that turn's three calls. Any own
/// cell outside the block (mid-game, or a seeded position), or a block cell that
/// is not a legal empty placement (a tiny board where the block collides with
/// another base), voids the book.
pub fn opening_book_move(state: &State) -> Option<Action> {
    if state.game_over() {
        return None;
    }
    let player = state.current_player();
    if !state.active(player) {
        return None;
    }
    let base = find_base(state, player)?;

    // Orient inward toward the board centre. Starting bases are corners, so each
    // delta resolves to +1 or -1; the comparison keeps it right for odd sizes.
    let dr = if base.row * 2 < state.rows() as i32 - 1 {
        1
    } else {
        -1
    };
    let dc = if base.col * 2 < state.cols() as i32 - 1 {
        1
    } else {
        -1
    };

    // Order is load-bearing: later cells connect through earlier ones, so they
    // must be placed in array order.
    let block = [
        Pos::new(base.row + dr, base.col + dc), // base-adjacent diagonal anchor
        Pos::new(base.row + 2 * dr, base.col + dc), // advancing pair, straight
        Pos::new(base.row + 2 * dr, base.col + 2 * dc), // advancing pair, diagonal
    ];

    // Any own non-base cell outside the block means this is not a fresh opening
    // turn — defer to search.
    for index in 0..state.cell_count() {
        let cell = state.cell_at(index);
        let pos = state.pos_of(index);
        if cell.owner() == player && cell.kind() != CellKind::Base && !block.contains(&pos) {
            return None;
        }
    }

    // Every wedge cell must be reachable: already ours from an earlier book move,
    // or a legal empty placement. A collision — out of bounds, or another
    // player's cell on a tiny board — voids the book. The first still-empty cell
    // in array order is the next move.
    let mut next = None;
    let mut placed = 0u8;
    for pos in block {
        // Go's `state.At` reports `ok == false` out of bounds, which voids the book.
        let cell = state.at_checked(pos)?;
        if cell.kind() == CellKind::Empty {
            if next.is_none() {
                next = Some(pos);
            }
        } else if cell.owner() == player {
            placed += 1; // placed by an earlier book move this turn
        } else {
            return None;
        }
    }
    let next = next?; // block already complete: the opening is over

    // Only fire on the mover's genuine first turn. Across that turn's three
    // calls `placed + moves_left` is invariant at 3 (0+3, 1+2, 2+1); any later
    // turn — or a player captured down to a block-cell prefix mid-game, as some
    // fixed-depth goldens are — fails this and defers to search.
    if placed + state.moves_left() != ACTIONS_PER_TURN {
        return None;
    }
    Some(Action::Move { target: next })
}

fn find_base(state: &State, player: Player) -> Option<Pos> {
    (0..state.cell_count()).find_map(|index| {
        let cell = state.cell_at(index);
        (cell.owner() == player && cell.kind() == CellKind::Base).then(|| state.pos_of(index))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plays_the_whole_wedge_from_a_fresh_board() {
        let mut state = State::new(12, 12, 2).expect("12x12");
        let expected = [Action::mv(1, 1), Action::mv(2, 1), Action::mv(2, 2)];
        for action in expected {
            assert_eq!(opening_book_move(&state), Some(action));
            state = state.apply(action).expect("book move is legal");
        }
        // Turn spent; the mover flipped and P2 gets its own mirrored wedge.
        assert_eq!(state.current_player(), 2);
        assert_eq!(opening_book_move(&state), Some(Action::mv(10, 10)));
    }

    #[test]
    fn mid_turn_state_keeps_the_placed_plus_moves_left_invariant() {
        let state = State::new(12, 12, 2).expect("12x12");
        // 0 placed + 3 left.
        assert!(opening_book_move(&state).is_some());
        let state = state.apply(Action::mv(1, 1)).expect("legal");
        // 1 placed + 2 left.
        assert_eq!(opening_book_move(&state), Some(Action::mv(2, 1)));
    }

    #[test]
    fn a_later_turn_voids_the_book() {
        // A player captured down to a wedge prefix mid-game must not re-enter
        // the book: placed + moves_left would be 1 + 3, not 3.
        let mut state = State::new(12, 12, 2).expect("12x12");
        state = state.apply(Action::mv(1, 1)).expect("legal");
        state = state.apply(Action::mv(2, 1)).expect("legal");
        state = state.apply(Action::mv(2, 2)).expect("legal");
        // P2 plays out its own turn so P1 comes back around.
        for action in [Action::mv(10, 10), Action::mv(9, 10), Action::mv(9, 9)] {
            state = state.apply(action).expect("legal");
        }
        assert_eq!(state.current_player(), 1);
        assert_eq!(state.moves_left(), ACTIONS_PER_TURN);
        assert_eq!(opening_book_move(&state), None);
    }

    #[test]
    fn an_own_cell_outside_the_block_voids_the_book() {
        let mut state = State::new(12, 12, 2).expect("12x12");
        state = state.apply(Action::mv(0, 1)).expect("legal");
        assert_eq!(opening_book_move(&state), None);
    }

    #[test]
    fn a_collision_on_a_tiny_board_voids_the_book() {
        // 3x3: P1's wedge runs into P3's corner base at (0,2).
        let state = State::new(3, 3, 4).expect("3x3 four-player board");
        assert_eq!(opening_book_move(&state), None);
    }
}
