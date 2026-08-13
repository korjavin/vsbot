//! Rule-by-rule unit tests.
//!
//! Each of the first four has a production bug behind it (ARCHITECTURE.md
//! "Non-negotiable invariants"), so they are written as regression tests, not
//! as coverage filler.

use virus_core::cell::{Cell, CellKind};
use virus_core::{Action, Pos, Position, RuleError, State, ACTIONS_PER_TURN};

/// Builds a board from ASCII art, matching `State`'s `Debug` glyphs:
/// `.` empty, `#` neutral, `1`-`4` a normal, `A`-`D` a base, `a`-`d` a
/// fortified cell.
fn cells(rows: &[&str]) -> Vec<Cell> {
    rows.iter()
        .flat_map(|row| row.chars())
        .map(|glyph| match glyph {
            '.' => Cell::EMPTY,
            '#' => Cell::NEUTRAL,
            '1'..='4' => Cell::new(glyph as u8 - b'0', CellKind::Normal),
            'A'..='D' => Cell::new(glyph as u8 - b'A' + 1, CellKind::Base),
            'a'..='d' => Cell::new(glyph as u8 - b'a' + 1, CellKind::Fortified),
            other => panic!("unknown board glyph {other:?}"),
        })
        .collect()
}

fn state(rows: &[&str], players: usize, current: u8, moves_left: u8) -> State {
    state_with(rows, players, current, moves_left, &[false; 4])
}

fn state_with(
    rows: &[&str],
    players: usize,
    current: u8,
    moves_left: u8,
    neutral_used: &[bool],
) -> State {
    let height = rows.len();
    let width = rows[0].len();
    assert!(rows.iter().all(|row| row.len() == width), "ragged board");
    State::from_grid(
        height,
        width,
        players,
        &cells(rows),
        current,
        moves_left,
        neutral_used,
    )
    .expect("test board is well formed")
}

// ---------------------------------------------------------------- invariant 4

/// ARCHITECTURE.md invariant 4. Getting this backwards silently corrupted a
/// whole training run.
#[test]
fn capturing_an_enemy_normal_makes_it_your_fortified() {
    let before = state(&["A2.", "...", "..B"], 2, 1, 3);
    let after = before.apply(Action::mv(0, 1)).expect("capture is legal");
    let captured = after.at(Pos::new(0, 1));
    assert_eq!(captured.owner(), 1, "the capturer owns the cell");
    assert_eq!(
        captured.kind(),
        CellKind::Fortified,
        "a capture fortifies; it must NOT become a plain Normal"
    );
    // Playing an empty cell still produces a plain Normal.
    let plain = before.apply(Action::mv(1, 1)).expect("empty cell is legal");
    assert_eq!(plain.at(Pos::new(1, 1)).kind(), CellKind::Normal);
}

#[test]
fn fortified_bases_and_neutrals_are_immune() {
    let board = state(&["A2b.", "..#.", "....", "...B"], 2, 1, 3);
    // The enemy Normal is takeable...
    assert!(board.legal_move(1, Pos::new(0, 1)));
    // ...but a Fortified, a Base and a Neutral never are, even when adjacent.
    let fortified = state(&["Ab..", "....", "....", "...B"], 2, 1, 3);
    assert!(!fortified.legal_move(1, Pos::new(0, 1)));
    let neutral = state(&["A#..", "....", "....", "...B"], 2, 1, 3);
    assert!(!neutral.legal_move(1, Pos::new(0, 1)));
    let enemy_base = state(&["A...", "....", "....", "..1B"], 2, 1, 3);
    assert!(!enemy_base.legal_move(1, Pos::new(3, 3)));
    // Your own cells are not targets either.
    assert!(!board.legal_move(1, Pos::new(0, 0)));
}

// ---------------------------------------------------------------- invariant 1

/// ARCHITECTURE.md invariant 1: a turn is three actions, so the mover does not
/// alternate per ply. Consumers must branch on `current_player()`.
#[test]
fn the_mover_does_not_alternate_per_ply() {
    let mut position = State::new(12, 12, 2).expect("valid board");
    assert_eq!(position.current_player(), 1);
    assert_eq!(position.moves_left(), 3);

    position = position.apply(Action::mv(0, 1)).expect("legal");
    assert_eq!(position.current_player(), 1, "still player 1's turn");
    assert_eq!(position.moves_left(), 2);

    position = position.apply(Action::mv(0, 2)).expect("legal");
    assert_eq!(position.current_player(), 1);
    assert_eq!(position.moves_left(), 1);

    position = position.apply(Action::mv(0, 3)).expect("legal");
    assert_eq!(
        position.current_player(),
        2,
        "turn passes after three actions"
    );
    assert_eq!(position.moves_left(), 3);
}

// ---------------------------------------------------------------- neutrals

#[test]
fn placing_neutrals_consumes_the_whole_turn_and_only_happens_once() {
    // Two own normals next to the base, plenty of room for both players.
    let start = state(&["A11.", "....", "....", "...B"], 2, 1, 3);
    assert_eq!(start.moves_left(), ACTIONS_PER_TURN);

    let pair = Action::neutrals(Pos::new(0, 1), Pos::new(0, 2));
    let after = start.apply(pair).expect("placement is legal at turn start");

    assert_eq!(after.at(Pos::new(0, 1)), Cell::NEUTRAL);
    assert_eq!(after.at(Pos::new(0, 2)), Cell::NEUTRAL);
    assert!(after.neutral_used(1), "the placement is spent");
    assert_eq!(
        after.current_player(),
        2,
        "the placement consumes the whole turn, not one action"
    );
    assert_eq!(after.moves_left(), ACTIONS_PER_TURN);

    // Once per game.
    let spent = state_with(
        &["A11.", "....", "....", "...B"],
        2,
        1,
        3,
        &[true, false, false, false],
    );
    assert_eq!(spent.apply(pair), Err(RuleError::InvalidAction));

    // Only at turn start.
    let midturn = state(&["A11.", "....", "....", "...B"], 2, 1, 2);
    assert_eq!(midturn.apply(pair), Err(RuleError::InvalidAction));
    assert!(
        !midturn
            .legal_actions()
            .iter()
            .any(|action| matches!(action, Action::PlaceNeutrals { .. })),
        "no pairs are enumerated mid-turn"
    );
}

#[test]
fn neutral_pairs_must_be_two_distinct_own_normals() {
    let board = state(&["A1b2", "....", "....", "...B"], 2, 1, 3);
    let own = Pos::new(0, 1);
    // Same cell twice.
    assert_eq!(
        board.apply(Action::neutrals(own, own)),
        Err(RuleError::InvalidAction)
    );
    // Own base and own fortified are not Normals.
    assert_eq!(
        board.apply(Action::neutrals(own, Pos::new(0, 0))),
        Err(RuleError::InvalidAction)
    );
    assert_eq!(
        board.apply(Action::neutrals(own, Pos::new(0, 2))),
        Err(RuleError::InvalidAction)
    );
    // An enemy Normal is not yours.
    assert_eq!(
        board.apply(Action::neutrals(own, Pos::new(0, 3))),
        Err(RuleError::InvalidAction)
    );
    // Only one own Normal exists, so no pair is enumerated.
    assert!(!board
        .legal_actions()
        .iter()
        .any(|action| matches!(action, Action::PlaceNeutrals { .. })));
}

// ---------------------------------------------------------------- elimination

/// The eliminated player's cells stay on the board and stay capturable — only
/// the active flag flips. Erasing them was a real production bug.
#[test]
fn eliminated_players_keep_their_cells_and_stay_capturable() {
    // P1 walls player 2 into the bottom-right corner with invulnerable
    // Fortified cells. P3 keeps the game alive so it does not end on the spot.
    let board = [
        "A1..C", //
        ".1..3", ".1..3", "..aaa", "..a2B",
    ];
    let before = state(&board, 3, 1, 3);
    assert!(before.has_move(1) && before.has_move(3));
    assert!(!before.has_move(2), "player 2 is already walled in");

    // Any action re-runs elimination.
    let after = before.apply(Action::mv(0, 2)).expect("legal");
    assert!(!after.active(2), "player 2 is eliminated");
    assert!(after.active(1) && after.active(3));
    assert!(!after.game_over(), "two players are still in");

    // The cells stayed.
    assert_eq!(after.at(Pos::new(4, 4)).owner(), 2);
    assert_eq!(after.at(Pos::new(4, 4)).kind(), CellKind::Base);
    assert_eq!(after.at(Pos::new(4, 3)).owner(), 2);
    assert_eq!(after.at(Pos::new(4, 3)).kind(), CellKind::Normal);

    // And the eliminated player's Normal is still a legal capture.
    assert!(
        after.legal_move(1, Pos::new(4, 3)),
        "a dead player's Normal remains capturable"
    );
    let captured = after.apply(Action::mv(4, 3)).expect("capture is legal");
    assert_eq!(captured.at(Pos::new(4, 3)).owner(), 1);
    assert_eq!(captured.at(Pos::new(4, 3)).kind(), CellKind::Fortified);
}

#[test]
fn the_last_active_player_wins() {
    // Player 2 is walled in with only player 1 left.
    let before = state(&["A1..", ".1..", "..aa", "..aB"], 2, 1, 3);
    assert!(!before.has_move(2));
    let after = before.apply(Action::mv(0, 2)).expect("legal");
    assert!(after.game_over());
    assert_eq!(after.winner(), 1);
    assert_eq!(after.moves_left(), 0);
    assert!(after.legal_actions().is_empty());
    assert_eq!(after.apply(Action::mv(0, 3)), Err(RuleError::GameOver));
}

#[test]
fn a_skipped_seat_does_not_get_the_turn() {
    // Player 2 is walled in; the turn must jump from player 1 to player 3.
    let board = [
        "A1..C", //
        ".1..3", ".1..3", "..aaa", "..a2B",
    ];
    let mut position = state(&board, 3, 1, 1);
    position = position.apply(Action::mv(0, 2)).expect("legal");
    assert!(!position.active(2));
    assert_eq!(
        position.current_player(),
        3,
        "the eliminated seat is skipped"
    );
}

// ---------------------------------------------------------------- legality

#[test]
fn adjacency_is_eight_way() {
    let board = state(&["A...", "....", "....", "...B"], 2, 1, 3);
    // Orthogonal and diagonal neighbours of the base are all legal.
    assert!(board.legal_move(1, Pos::new(0, 1)));
    assert!(board.legal_move(1, Pos::new(1, 0)));
    assert!(board.legal_move(1, Pos::new(1, 1)), "diagonals count");
    // A knight's move away does not.
    assert!(!board.legal_move(1, Pos::new(2, 1)));
    assert_eq!(
        board.legal_actions().len(),
        3,
        "a corner base has exactly three targets"
    );
}

#[test]
fn targets_must_touch_the_base_connected_component() {
    // The stray Normal at (2,2) is not reachable from player 1's base, so it
    // confers no adjacency — this is the whole point of rooting connectivity at
    // the base rather than flooding all owned cells.
    let board = state(&["A...", "....", "..1.", "...B"], 2, 1, 3);
    assert!(!board.legal_move(1, Pos::new(2, 3)));
    assert!(!board.legal_move(1, Pos::new(3, 2)));
    assert!(board.legal_move(1, Pos::new(1, 1)));

    // Bridge it to the base and the same cells become legal.
    let bridged = state(&["A...", ".1..", "..1.", "...B"], 2, 1, 3);
    assert!(bridged.legal_move(1, Pos::new(2, 3)));
    assert!(bridged.legal_move(1, Pos::new(3, 2)));
}

#[test]
fn a_neutral_split_can_sever_a_component() {
    // (1,1) is the only link between the base and the far group.
    let joined = state(&["A...", ".1..", "..1.", "..1B"], 2, 1, 3);
    assert!(joined.legal_move(1, Pos::new(2, 3)));
    // Neutralising the articulation cell strands everything behind it.
    let severed = state(&["A...", ".#..", "..1.", "..1B"], 2, 1, 3);
    assert!(!severed.legal_move(1, Pos::new(2, 3)));
    assert!(
        !severed.legal_move(1, Pos::new(1, 1)),
        "neutral is dead space"
    );
}

#[test]
fn out_of_bounds_and_wrong_shaped_actions_are_rejected() {
    let board = state(&["A...", "....", "....", "...B"], 2, 1, 3);
    assert!(!board.legal_move(1, Pos::new(-1, 0)));
    assert!(!board.legal_move(1, Pos::new(0, 4)));
    assert_eq!(
        board.apply(Action::mv(-1, 0)),
        Err(RuleError::InvalidAction)
    );
    assert_eq!(board.apply(Action::mv(9, 9)), Err(RuleError::InvalidAction));
    assert_eq!(
        board.apply(Action::neutrals(Pos::new(0, 1), Pos::new(-1, 0))),
        Err(RuleError::InvalidAction)
    );
    assert!(State::new(1, 12, 2).is_err(), "boards must be at least 2x2");
    assert!(State::new(12, 12, 5).is_err(), "at most four seats");
    assert!(State::new(12, 12, 1).is_err(), "at least two seats");
    assert!(State::new(51, 12, 2).is_err(), "board edge is capped");
}

// ---------------------------------------------------------------- outcome

#[test]
fn territory_decides_a_turn_capped_game() {
    // Both sides can still move, so the tiebreak is owned cells.
    let leading = state(&["A11.", "....", "....", "..2B"], 2, 1, 3);
    assert_eq!(leading.owned_cells(1), 3);
    assert_eq!(leading.owned_cells(2), 2);
    assert_eq!(leading.outcome_winner(), 1);

    let equal = state(&["A1..", "....", "....", "..2B"], 2, 1, 3);
    assert_eq!(equal.owned_cells(1), 2);
    assert_eq!(equal.owned_cells(2), 2);
    assert_eq!(equal.outcome_winner(), 0, "an exact tie is a draw");
}

#[test]
fn the_last_player_able_to_move_wins_regardless_of_territory() {
    // Player 2 is sealed into a neutral-walled pocket but owns six cells to
    // player 1's one. Survival beats territory.
    let board = state(&["A....", ".....", ".####", ".#222", ".#22B"], 2, 1, 3);
    assert!(board.has_move(1));
    assert!(!board.has_move(2));
    assert!(board.owned_cells(2) > board.owned_cells(1));
    assert_eq!(board.outcome_winner(), 1);
}

#[test]
fn an_eliminated_player_cannot_win_the_territory_tiebreak() {
    // Player 2 is sealed in and owns the most cells; players 1 and 3 can both
    // still move, so the game is decided on territory. Eliminated seats keep
    // their cells, so player 2 must be excluded from the tiebreak or it would
    // be handed a win it cannot play for.
    let board = state(&["A...C", "11..3", ".####", ".#222", ".#22B"], 3, 1, 3);
    let eliminated = board.apply(Action::mv(0, 1)).expect("legal");
    assert!(!eliminated.active(2), "player 2 is eliminated");
    assert!(eliminated.active(1) && eliminated.active(3));
    assert!(!eliminated.game_over());
    assert!(eliminated.owned_cells(2) > eliminated.owned_cells(1));
    assert!(eliminated.owned_cells(1) > eliminated.owned_cells(3));
    assert_eq!(eliminated.outcome_winner(), 1);
}

// ---------------------------------------------------------------- hashing

#[test]
fn the_zobrist_key_covers_the_hidden_state() {
    let board = ["A11.", "....", "....", "...B"];
    let base = state(&board, 2, 1, 3);

    // ARCHITECTURE.md invariant 6: the key must separate positions that differ
    // only in moves_left, neutral_used or the side to move.
    assert_ne!(base.hash(), state(&board, 2, 1, 2).hash(), "moves_left");
    assert_ne!(base.hash(), state(&board, 2, 2, 3).hash(), "side to move");
    assert_ne!(
        base.hash(),
        state_with(&board, 2, 1, 3, &[true, false, false, false]).hash(),
        "neutral_used"
    );
    // …and of course the grid.
    assert_ne!(
        base.hash(),
        state(&["A1..", "....", "....", "...B"], 2, 1, 3).hash()
    );
    // Equal positions hash equally.
    assert_eq!(base.hash(), state(&board, 2, 1, 3).hash());
}

#[test]
fn the_incremental_key_tracks_a_full_game() {
    let mut position = State::new(8, 8, 2).expect("valid board");
    let mut rng = 0x1234_5678_9abc_def0u64;
    for _ in 0..200 {
        if position.game_over() {
            break;
        }
        let actions = position.legal_actions();
        if actions.is_empty() {
            break;
        }
        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let action = actions[(rng >> 33) as usize % actions.len()];
        position = position.apply(action).expect("enumerated action is legal");
        assert_eq!(
            position.hash(),
            position.recomputed_hash(),
            "incremental key drifted from the recomputed one after {action:?}"
        );
    }
}

// ---------------------------------------------------------------- enumeration

#[test]
fn moves_are_enumerated_in_row_major_target_order() {
    let board = state(&["A...", ".1..", "....", "...B"], 2, 1, 3);
    let moves: Vec<Pos> = board
        .legal_actions()
        .into_iter()
        .filter_map(|action| match action {
            Action::Move { target } => Some(target),
            Action::PlaceNeutrals { .. } => None,
        })
        .collect();
    let mut sorted = moves.clone();
    sorted.sort_by_key(|pos| (pos.row, pos.col));
    assert_eq!(moves, sorted, "targets come out in ascending board index");
    assert!(moves.contains(&Pos::new(0, 1)));
    assert!(moves.contains(&Pos::new(2, 2)));
}

#[test]
fn neutral_pairs_follow_the_moves_and_are_ordered_by_board_index() {
    let board = state(&["A11.", "...1", "....", "...B"], 2, 1, 3);
    let actions = board.legal_actions();
    let first_pair = actions
        .iter()
        .position(|action| matches!(action, Action::PlaceNeutrals { .. }))
        .expect("pairs exist at turn start");
    assert!(
        actions[..first_pair]
            .iter()
            .all(|action| matches!(action, Action::Move { .. })),
        "every move precedes every pair"
    );
    let pairs: Vec<[Pos; 2]> = actions[first_pair..]
        .iter()
        .map(|action| match action {
            Action::PlaceNeutrals { cells } => *cells,
            Action::Move { .. } => panic!("moves must not follow pairs"),
        })
        .collect();
    // Three own normals -> C(3,2) = 3 pairs, in (i, j) board order.
    assert_eq!(
        pairs,
        vec![
            [Pos::new(0, 1), Pos::new(0, 2)],
            [Pos::new(0, 1), Pos::new(1, 3)],
            [Pos::new(0, 2), Pos::new(1, 3)],
        ]
    );
}

#[test]
fn the_curated_pair_set_is_bounded_and_legal() {
    // A wide-open 12x12 midgame: C(owned, 2) alone is far past the threshold,
    // so the curation must kick in and stay within its cap.
    let mut position = State::new(12, 12, 2).expect("valid board");
    let mut rng = 0xdead_beef_cafe_f00du64;
    for _ in 0..30 {
        let actions: Vec<Action> = position
            .legal_actions()
            .into_iter()
            .filter(|action| matches!(action, Action::Move { .. }))
            .collect();
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        position = position
            .apply(actions[(rng >> 33) as usize % actions.len()])
            .expect("legal");
    }
    // Advance to a turn start so neutrals are available.
    while position.moves_left() != 3 || position.neutral_used(position.current_player()) {
        let actions: Vec<Action> = position
            .legal_actions()
            .into_iter()
            .filter(|action| matches!(action, Action::Move { .. }))
            .collect();
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        position = position
            .apply(actions[(rng >> 33) as usize % actions.len()])
            .expect("legal");
    }

    let view = Position::new(position.clone());
    let search = view.search_actions();
    let exact = view.legal_actions();
    let curated: Vec<Action> = search
        .iter()
        .copied()
        .filter(|action| matches!(action, Action::PlaceNeutrals { .. }))
        .collect();
    assert!(
        curated.len() <= virus_core::MAX_STRATEGIC_PAIRS,
        "curated set must respect its cap"
    );
    assert!(
        search.len() < exact.len(),
        "curation should prune ({} search vs {} exact)",
        search.len(),
        exact.len()
    );
    // Every curated action must still be genuinely legal, and unique.
    for action in &curated {
        assert!(
            position.apply(*action).is_ok(),
            "curated {action:?} is not legal"
        );
    }
    for (index, action) in search.iter().enumerate() {
        assert!(
            !search[..index].iter().any(|other| other == action),
            "duplicate action {action:?}"
        );
    }
    // Moves are never pruned — only pairs are.
    let exact_moves: Vec<&Action> = exact
        .iter()
        .filter(|action| matches!(action, Action::Move { .. }))
        .collect();
    let search_moves: Vec<&Action> = search
        .iter()
        .filter(|action| matches!(action, Action::Move { .. }))
        .collect();
    assert_eq!(exact_moves, search_moves);
}

#[test]
fn a_finished_or_inactive_position_enumerates_nothing() {
    let finished = state(&["A1..", ".1..", "..aa", "..aB"], 2, 1, 3)
        .apply(Action::mv(0, 2))
        .expect("legal");
    assert!(finished.game_over());
    assert!(finished.legal_actions().is_empty());
    assert!(Position::new(finished).search_actions().is_empty());
}
