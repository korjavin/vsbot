//! Parity-fixture record types.
//!
//! `fixtures/gobot_search_parity.jsonl` (412 records) and its node-budget
//! companion are the search-parity oracle; `gobot_staticeval_parity.jsonl` uses
//! the same board encoding. The schema is documented in
//! `fixtures/gobot_search_parity.README.md`.
//!
//! The hidden state matters: a record's board grid alone does **not** determine
//! the search result. `player`, `moves_left` and `neutral_used` feed both the
//! evaluation's tempo terms and the transposition key, so a position must be
//! rebuilt from all four (README, "Hidden State the port MUST reproduce").
//!
//! This lives in `virus-core` rather than in a test file so the eval, search and
//! MCTS crates can all replay the same records.

use crate::action::{Action, Pos};
use crate::cell::Cell;
use crate::state::{State, ACTIONS_PER_TURN};
use serde::{Deserialize, Serialize};

/// A recorded action, in the fixtures' encoding.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum FixtureAction {
    /// `{"type":"MOVE","target":{"row":r,"col":c}}`
    #[serde(rename = "MOVE")]
    Move {
        /// Cell played.
        target: Pos,
    },
    /// `{"type":"PLACE_NEUTRALS","pos1":{…},"pos2":{…}}`
    #[serde(rename = "PLACE_NEUTRALS")]
    PlaceNeutrals {
        /// Go's `Action.Neutrals[0]`.
        pos1: Pos,
        /// Go's `Action.Neutrals[1]`.
        pos2: Pos,
    },
}

impl From<FixtureAction> for Action {
    fn from(action: FixtureAction) -> Action {
        match action {
            FixtureAction::Move { target } => Action::Move { target },
            FixtureAction::PlaceNeutrals { pos1, pos2 } => Action::neutrals(pos1, pos2),
        }
    }
}

impl From<Action> for FixtureAction {
    fn from(action: Action) -> FixtureAction {
        match action {
            Action::Move { target } => FixtureAction::Move { target },
            Action::PlaceNeutrals { cells } => FixtureAction::PlaceNeutrals {
                pos1: cells[0],
                pos2: cells[1],
            },
        }
    }
}

/// One `search.ChooseDepth` result from the GoBot oracle.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SearchParityRecord {
    /// Row-major grid.
    pub board: Vec<Vec<Cell>>,
    /// The mover, i.e. the search root.
    pub player: u8,
    /// Fixed search depth (3 or 5 in this fixture).
    pub depth: i32,
    /// `Result.Score` from `ChooseDepth`.
    pub score: i64,
    /// The chosen action.
    pub action: FixtureAction,
    /// Actions remaining this turn. Hidden state; feeds tempo terms and the
    /// state hash.
    #[serde(default = "default_moves_left")]
    pub moves_left: u8,
    /// Per-seat neutral-placement flags, index `player - 1`. Hidden state.
    #[serde(default)]
    pub neutral_used: Vec<bool>,
}

fn default_moves_left() -> u8 {
    ACTIONS_PER_TURN
}

impl SearchParityRecord {
    /// Rebuilds the position, hidden state included.
    ///
    /// The fixtures are two-player games; seats are derived the way Java's
    /// `GoState.fromBoard` does (bases at the default corners, a seat is active
    /// iff its base is intact).
    pub fn to_state(&self) -> Result<State, crate::RuleError> {
        let rows = self.board.len();
        let cols = self.board.first().map(Vec::len).unwrap_or(0);
        if rows == 0 || cols == 0 || self.board.iter().any(|row| row.len() != cols) {
            return Err(crate::RuleError::InvalidAction);
        }
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
    }
}

/// Parses a JSONL fixture into records, reporting the 1-based line number of
/// the first failure.
pub fn parse_jsonl<T: serde::de::DeserializeOwned>(
    text: &str,
) -> Result<Vec<T>, (usize, serde_json::Error)> {
    let mut records = Vec::new();
    for (line_index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<T>(line) {
            Ok(record) => records.push(record),
            Err(error) => return Err((line_index + 1, error)),
        }
    }
    Ok(records)
}
