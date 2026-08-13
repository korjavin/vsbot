//! Integer-exact parity against GoBot's `StaticEval` over every recorded
//! position.
//!
//! `fixtures/gobot_staticeval_parity.jsonl` is the oracle: 419 records, each a
//! board plus the hidden state (`player`, `movesLeft`, `neutralUsed`) and the
//! score GoBot's `search.StaticEval` returned for that player.
//!
//! CLAUDE.md: parity is a hard gate. One divergent record fails the suite, and
//! the fix is always in the port — never a widened tolerance.

use serde::Deserialize;
use std::path::PathBuf;
use virus_core::fixture::parse_jsonl;
use virus_core::{Cell, State};
use virus_eval::{evaluate, evaluate_all, EvalParams, EvalWorkspace, MATE_SCORE};

/// One `search.StaticEval` result from the GoBot oracle.
///
/// The same board encoding as `SearchParityRecord`, minus `depth`/`action` and
/// with `score` meaning the leaf evaluation rather than a search result — hence
/// a record type of its own rather than a reuse of virus-core's.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct StaticEvalRecord {
    /// Row-major grid.
    board: Vec<Vec<Cell>>,
    /// Whose utility was recorded; also the mover, so tempo terms line up.
    player: u8,
    /// `search.StaticEval(state, player)`.
    score: i64,
    /// Actions remaining this turn. Hidden state; feeds the tempo terms.
    moves_left: u8,
    /// Per-seat neutral-placement flags, index `player - 1`. Hidden state.
    neutral_used: Vec<bool>,
}

impl StaticEvalRecord {
    fn to_state(&self) -> State {
        let rows = self.board.len();
        let cols = self.board[0].len();
        let cells: Vec<Cell> = self.board.iter().flatten().copied().collect();
        State::from_grid(
            rows,
            cols,
            2,
            &cells,
            self.player,
            self.moves_left,
            &self.neutral_used,
        )
        .expect("fixture record is a legal position")
    }
}

fn records() -> Vec<StaticEvalRecord> {
    // CARGO_MANIFEST_DIR is crates/virus-eval.
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures")
        .join("gobot_staticeval_parity.jsonl");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("reading {}: {error}", path.display()));
    parse_jsonl::<StaticEvalRecord>(&text).unwrap_or_else(|(line, error)| {
        panic!("gobot_staticeval_parity.jsonl line {line}: {error}")
    })
}

#[test]
fn every_record_matches_gobot_exactly() {
    let records = records();
    assert_eq!(records.len(), 419, "fixture size changed");

    let params = EvalParams::default();
    let mut workspace = EvalWorkspace::new();
    let mut mismatches = Vec::new();
    for (line, record) in records.iter().enumerate() {
        let state = record.to_state();
        let got = evaluate(&state, record.player, &params, &mut workspace);
        if got != record.score {
            mismatches.push(format!(
                "line {}: player {} movesLeft {} — want {}, got {} (delta {})",
                line + 1,
                record.player,
                record.moves_left,
                record.score,
                got,
                got - record.score
            ));
        }
    }
    assert!(
        mismatches.is_empty(),
        "{} of {} records diverged from GoBot:\n{}",
        mismatches.len(),
        records.len(),
        mismatches
            .iter()
            .take(20)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}

/// A shared workspace must give bit-identical results to a fresh one — the
/// whole point of the reuse pattern is that nothing leaks between evaluations.
#[test]
fn shared_workspace_matches_fresh_workspace() {
    let params = EvalParams::default();
    let mut shared = EvalWorkspace::new();
    for record in records() {
        let state = record.to_state();
        let with_shared = evaluate_all(&state, &params, &mut shared);
        let with_fresh = evaluate_all(&state, &params, &mut EvalWorkspace::new());
        assert_eq!(with_shared, with_fresh, "workspace state leaked");
    }
}

/// Seats 3 and 4 do not exist in these two-player records, so they must carry
/// the eliminated-seat sentinel rather than a computed score.
#[test]
fn absent_seats_are_pinned_to_half_a_mate() {
    let params = EvalParams::default();
    let mut workspace = EvalWorkspace::new();
    for record in records() {
        let state = record.to_state();
        let all = evaluate_all(&state, &params, &mut workspace);
        assert_eq!(all[2], -MATE_SCORE / 2);
        assert_eq!(all[3], -MATE_SCORE / 2);
        // Two active seats: each seat's utility is its raw minus the other's,
        // so the pair sums to zero.
        assert_eq!(all[0] + all[1], 0);
    }
}
