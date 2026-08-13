//! Snapshot decoding: wire tolerance, validation, and the defensive activity
//! recomputation.
//!
//! ARCHITECTURE.md invariant 5: the snapshot is the only board source, it is
//! re-validated every time, and the server's `Active[]` is not trusted.

use virus_core::cell::CellKind;
use virus_core::{Pos, Snapshot, State};

fn decode(json: &str) -> Result<State, virus_core::SnapshotError> {
    serde_json::from_str::<Snapshot>(json)
        .expect("snapshot parses")
        .decode()
}

/// A 4x4 two-player snapshot, rendered in whichever wire dialect is asked for.
fn wire(pascal_case_numeric_kinds: bool, active: [bool; 2]) -> String {
    // Row 0: player 1 base + a normal; row 3: player 2 base.
    let grid = ["A1..", "....", "....", "...B"];
    let mut rows = Vec::new();
    for row in grid {
        let mut cells = Vec::new();
        for glyph in row.chars() {
            let (owner, kind) = match glyph {
                '.' => (0, CellKind::Empty),
                '#' => (0, CellKind::Neutral),
                '1'..='4' => (glyph as u8 - b'0', CellKind::Normal),
                'A'..='D' => (glyph as u8 - b'A' + 1, CellKind::Base),
                'a'..='d' => (glyph as u8 - b'a' + 1, CellKind::Fortified),
                other => panic!("bad glyph {other:?}"),
            };
            cells.push(if pascal_case_numeric_kinds {
                format!(r#"{{"Owner":{owner},"Kind":{}}}"#, kind.as_u8())
            } else {
                format!(r#"{{"owner":{owner},"kind":"{}"}}"#, kind.name())
            });
        }
        rows.push(format!("[{}]", cells.join(",")));
    }
    let board = format!("[{}]", rows.join(","));
    let active = format!("[{},{}]", active[0], active[1]);
    if pascal_case_numeric_kinds {
        format!(
            r#"{{"Rows":4,"Cols":4,"Board":{board},
                 "Bases":[{{"Row":0,"Col":0}},{{"Row":3,"Col":3}}],
                 "Active":{active},"NeutralUsed":[false,false],
                 "CurrentPlayer":1,"MovesLeft":3,"GameOver":false,"Winner":0}}"#
        )
    } else {
        format!(
            r#"{{"rows":4,"cols":4,"board":{board},
                 "bases":[{{"row":0,"col":0}},{{"row":3,"col":3}}],
                 "active":{active},"neutralUsed":[false,false],
                 "currentPlayer":1,"movesLeft":3,"gameOver":false,"winner":0}}"#
        )
    }
}

/// The Go server emits PascalCase keys and numeric kinds; the fixtures emit
/// lowercase keys and named kinds. Both must decode to the same position.
#[test]
fn both_wire_dialects_decode_identically() {
    let go_style = decode(&wire(true, [true, true])).expect("go-style snapshot decodes");
    let fixture_style = decode(&wire(false, [true, true])).expect("fixture-style snapshot decodes");
    assert_eq!(go_style.hash(), fixture_style.hash());
    assert_eq!(go_style.snapshot(), fixture_style.snapshot());
    assert_eq!(go_style.at(Pos::new(0, 0)).kind(), CellKind::Base);
    assert_eq!(go_style.at(Pos::new(0, 1)).kind(), CellKind::Normal);
    assert_eq!(go_style.current_player(), 1);
    assert_eq!(go_style.moves_left(), 3);
}

/// The literal bytes the Go server produces, captured by marshalling
/// `game.New(3, 3, 2).Snapshot()` with the real backend.
///
/// It is a *hybrid* dialect and that is easy to get wrong: `game.Snapshot` has
/// explicit `json:"…"` tags, so its own keys are camelCase — but the nested
/// `game.Cell` and `game.Pos` have no tags at all, so they fall back to Go
/// field names and emit `Owner`/`Kind`/`Row`/`Col`, with `Kind` as a number.
/// Pinning the exact bytes here means a protocol drift fails loudly instead of
/// at the first live game.
const GO_SERVER_SNAPSHOT: &str = r#"{"rows":3,"cols":3,"board":[[{"Owner":1,"Kind":2},{"Owner":0,"Kind":0},{"Owner":0,"Kind":0}],[{"Owner":0,"Kind":0},{"Owner":0,"Kind":0},{"Owner":0,"Kind":0}],[{"Owner":0,"Kind":0},{"Owner":0,"Kind":0},{"Owner":2,"Kind":2}]],"bases":[{"Row":0,"Col":0},{"Row":2,"Col":2}],"active":[true,true],"neutralUsed":[false,false],"currentPlayer":1,"movesLeft":3,"gameOver":false,"winner":0}"#;

#[test]
fn decodes_the_real_go_server_wire_format() {
    let state = decode(GO_SERVER_SNAPSHOT).expect("the live server format decodes");
    let expected = State::new(3, 3, 2).expect("valid board");
    assert_eq!(state.snapshot(), expected.snapshot());
    assert_eq!(state.hash(), expected.hash());
    assert_eq!(state.at(Pos::new(0, 0)).kind(), CellKind::Base);
    assert_eq!(state.at(Pos::new(2, 2)).owner(), 2);
    assert_eq!(state.current_player(), 1);
    assert_eq!(state.moves_left(), 3);
    assert!(state.active(1) && state.active(2));
}

#[test]
fn unknown_fields_are_ignored() {
    let json = r#"{"rows":4,"cols":4,
        "board":[
          [{"owner":1,"kind":"BASE","hp":9},{"owner":0,"kind":"EMPTY"},{"owner":0,"kind":"EMPTY"},{"owner":0,"kind":"EMPTY"}],
          [{"owner":0,"kind":"EMPTY"},{"owner":0,"kind":"EMPTY"},{"owner":0,"kind":"EMPTY"},{"owner":0,"kind":"EMPTY"}],
          [{"owner":0,"kind":"EMPTY"},{"owner":0,"kind":"EMPTY"},{"owner":0,"kind":"EMPTY"},{"owner":0,"kind":"EMPTY"}],
          [{"owner":0,"kind":"EMPTY"},{"owner":0,"kind":"EMPTY"},{"owner":0,"kind":"EMPTY"},{"owner":2,"kind":"BASE"}]],
        "bases":[{"row":0,"col":0,"z":1},{"row":3,"col":3}],
        "active":[true,true],"neutralUsed":[false,false],
        "currentPlayer":1,"movesLeft":3,"serverBuild":"abc"}"#;
    let state = decode(json).expect("extra fields must not break decoding");
    assert_eq!(state.players(), 2);
}

/// The documented server bug: an eliminated seat that still owns cells is
/// reported as active. Trusting it would make the engine search a dead
/// player's branches and mis-score every terminal.
#[test]
fn a_stuck_seat_reported_active_is_recomputed_as_dead() {
    // Player 2 is sealed into a neutral pocket: no legal move anywhere.
    let json = r#"{"rows":5,"cols":5,
        "board":[
          [{"owner":1,"kind":"BASE"},{"owner":0,"kind":"EMPTY"},{"owner":0,"kind":"EMPTY"},{"owner":0,"kind":"EMPTY"},{"owner":0,"kind":"EMPTY"}],
          [{"owner":0,"kind":"EMPTY"},{"owner":0,"kind":"EMPTY"},{"owner":0,"kind":"EMPTY"},{"owner":0,"kind":"EMPTY"},{"owner":0,"kind":"EMPTY"}],
          [{"owner":0,"kind":"EMPTY"},{"owner":0,"kind":"NEUTRAL"},{"owner":0,"kind":"NEUTRAL"},{"owner":0,"kind":"NEUTRAL"},{"owner":0,"kind":"NEUTRAL"}],
          [{"owner":0,"kind":"EMPTY"},{"owner":0,"kind":"NEUTRAL"},{"owner":2,"kind":"NORMAL"},{"owner":2,"kind":"NORMAL"},{"owner":2,"kind":"NORMAL"}],
          [{"owner":0,"kind":"EMPTY"},{"owner":0,"kind":"NEUTRAL"},{"owner":2,"kind":"NORMAL"},{"owner":2,"kind":"NORMAL"},{"owner":2,"kind":"BASE"}]],
        "bases":[{"row":0,"col":0},{"row":4,"col":4}],
        "active":[true,true],
        "neutralUsed":[false,false],
        "currentPlayer":1,"movesLeft":3,"gameOver":false,"winner":0}"#;
    let state = decode(json).expect("snapshot decodes");
    assert!(
        !state.active(2),
        "the server claimed seat 2 was active; it has no legal move"
    );
    assert!(state.active(1));
    // Its cells are still there — elimination never erases them.
    assert_eq!(state.at(Pos::new(4, 4)).owner(), 2);
    assert_eq!(state.owned_cells(2), 6);
    // And with only one seat left the position is imported as decided rather
    // than rejected.
    assert!(state.game_over());
    assert_eq!(state.winner(), 1);
}

#[test]
fn an_honest_snapshot_keeps_both_seats_active() {
    let state = decode(&wire(false, [true, true])).expect("decodes");
    assert!(state.active(1) && state.active(2));
    assert!(!state.game_over());
}

/// The derivation only ever clears the flag: a seat the server called dead
/// stays dead, because elimination is sticky in the real rules.
#[test]
fn a_seat_the_server_calls_dead_is_never_revived() {
    let state = decode(&wire(false, [true, false])).expect("decodes");
    assert!(!state.active(2), "seat 2 is not promoted back to active");
    assert!(state.game_over(), "only one live seat remains");
    assert_eq!(state.winner(), 1);
}

#[test]
fn round_trips_through_the_engine() {
    let original = State::new(12, 12, 4).expect("valid board");
    let restored = original.snapshot().decode().expect("round-trips");
    assert_eq!(original.snapshot(), restored.snapshot());
    assert_eq!(original.hash(), restored.hash());
    assert_eq!(original.state_hash(), restored.state_hash());
}

#[test]
fn round_trips_a_mid_game_position() {
    let mut position = State::new(9, 9, 3).expect("valid board");
    let mut rng = 0x0bad_c0de_dead_10ccu64;
    for _ in 0..40 {
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
        position = position
            .apply(actions[(rng >> 33) as usize % actions.len()])
            .expect("legal");
        let restored = position.snapshot().decode().expect("round-trips");
        assert_eq!(position.snapshot(), restored.snapshot());
        assert_eq!(position.hash(), restored.hash());
    }
}

// ---------------------------------------------------------------- rejections

fn rejection_reason(json: &str) -> &'static str {
    decode(json)
        .expect_err("snapshot should be rejected")
        .reason()
}

#[test]
fn malformed_snapshots_are_rejected_with_a_reason() {
    // An owned Empty cell.
    let bad_cell = wire(false, [true, true]).replace(
        r#"{"owner":0,"kind":"EMPTY"}"#,
        r#"{"owner":1,"kind":"EMPTY"}"#,
    );
    assert_eq!(rejection_reason(&bad_cell), "malformed cell");

    // A base cell that is not where `bases` says it is.
    let stray_base = wire(false, [true, true]).replace(
        r#"[{"owner":0,"kind":"EMPTY"},{"owner":0,"kind":"EMPTY"},{"owner":0,"kind":"EMPTY"},{"owner":0,"kind":"EMPTY"}]"#,
        r#"[{"owner":1,"kind":"BASE"},{"owner":0,"kind":"EMPTY"},{"owner":0,"kind":"EMPTY"},{"owner":0,"kind":"EMPTY"}]"#,
    );
    assert_eq!(
        rejection_reason(&stray_base),
        "base cell not at the declared base"
    );

    // Wrong row count.
    let short = wire(false, [true, true]).replace(r#""rows":4"#, r#""rows":3"#);
    assert_eq!(rejection_reason(&short), "board row count mismatch");

    // Out-of-range mover.
    let bad_mover =
        wire(false, [true, true]).replace(r#""currentPlayer":1"#, r#""currentPlayer":3"#);
    assert_eq!(rejection_reason(&bad_mover), "currentPlayer out of range");

    // Out-of-range movesLeft.
    let bad_moves = wire(false, [true, true]).replace(r#""movesLeft":3"#, r#""movesLeft":4"#);
    assert_eq!(rejection_reason(&bad_moves), "movesLeft out of range");

    // Per-seat vectors that disagree with the seat count.
    let bad_active =
        wire(false, [true, true]).replace(r#""active":[true,true]"#, r#""active":[true]"#);
    assert_eq!(
        rejection_reason(&bad_active),
        "per-seat vector length mismatch"
    );

    // A seat declared active whose base is gone.
    let no_base = wire(false, [true, true]).replace(
        r#"{"owner":2,"kind":"BASE"}"#,
        r#"{"owner":2,"kind":"NORMAL"}"#,
    );
    assert_eq!(rejection_reason(&no_base), "active seat has no intact base");

    // A winner declared while the game is still running.
    let early_winner = wire(false, [true, true]).replace(r#""winner":0"#, r#""winner":2"#);
    assert_eq!(
        rejection_reason(&early_winner),
        "winner declared while the game runs"
    );
}

#[test]
fn an_unknown_cell_kind_fails_to_parse() {
    let json = wire(false, [true, true]).replace(r#""kind":"EMPTY""#, r#""kind":"LAVA""#);
    assert!(serde_json::from_str::<Snapshot>(&json).is_err());
}
