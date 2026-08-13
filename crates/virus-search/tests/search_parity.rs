//! Byte-exact parity with the GoBot search oracle.
//!
//! `fixtures/gobot_search_parity.jsonl` (412 records) is one deterministic
//! `search.ChooseDepth` result each; `fixtures/gobot_nodebudget_parity.jsonl`
//! (158 records) is the `search.ChooseNodeBudget` companion, which unlike
//! `ChooseDepth` *does* consult the opening book.
//!
//! Both are hard gates (CLAUDE.md): a divergent record fails the suite and the
//! fix is always in the port. virus-core's enumeration order is separately
//! golden-pinned against the real Go engine (`gobot_movegen_order.jsonl`), so a
//! failure here is a search bug, never a movegen one.

use std::path::PathBuf;

use serde::Deserialize;
use virus_core::fixture::{parse_jsonl, FixtureAction, SearchParityRecord};
use virus_core::{Action, Cell, State};
use virus_search::{choose_depth, choose_node_budget_plain};

fn fixture(name: &str) -> String {
    // CARGO_MANIFEST_DIR is crates/virus-search.
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures")
        .join(name);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("read {}: {error}", path.display()))
}

/// One `search.ChooseNodeBudget` result from the GoBot oracle. Same encoding as
/// [`SearchParityRecord`] with `nodeLimit` in place of `depth`.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct NodeBudgetRecord {
    board: Vec<Vec<Cell>>,
    player: u8,
    node_limit: u64,
    score: i64,
    action: FixtureAction,
    moves_left: u8,
    neutral_used: Vec<bool>,
}

impl NodeBudgetRecord {
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

/// `PlaceNeutrals` compares as an unordered pair, matching Java's
/// `PlaceNeutralsAction.equals` and the fixtures' documented contract.
fn matches(chosen: Option<Action>, expected: &FixtureAction) -> bool {
    chosen.is_some_and(|action| action.same_transition(Action::from(expected.clone())))
}

#[test]
fn choose_depth_matches_the_gobot_oracle() {
    let text = fixture("gobot_search_parity.jsonl");
    let records: Vec<SearchParityRecord> =
        parse_jsonl(&text).unwrap_or_else(|(line, error)| panic!("line {line}: {error}"));
    assert_eq!(records.len(), 412, "fixture size changed unexpectedly");

    let mut failures = Vec::new();
    for (index, record) in records.iter().enumerate() {
        let state = record
            .to_state()
            .expect("fixture record is a legal position");
        let result = choose_depth(&state, record.depth)
            .unwrap_or_else(|| panic!("record {index}: search returned no result"));
        if !matches(result.action, &record.action) || result.score != record.score {
            failures.push(format!(
                "record {index} (player {}, depth {}, movesLeft {}): \
                 got {:?} score {}, want {:?} score {}",
                record.player,
                record.depth,
                record.moves_left,
                result.action,
                result.score,
                record.action,
                record.score,
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} records diverged:\n{}",
        failures.len(),
        records.len(),
        failures
            .iter()
            .take(10)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n"),
    );
}

#[test]
fn choose_node_budget_matches_the_gobot_oracle() {
    let text = fixture("gobot_nodebudget_parity.jsonl");
    let records: Vec<NodeBudgetRecord> =
        parse_jsonl(&text).unwrap_or_else(|(line, error)| panic!("line {line}: {error}"));
    assert_eq!(records.len(), 158, "fixture size changed unexpectedly");

    let mut failures = Vec::new();
    for (index, record) in records.iter().enumerate() {
        let state = record.to_state();
        let result = choose_node_budget_plain(&state, record.node_limit)
            .unwrap_or_else(|| panic!("record {index}: search returned no result"));
        if !matches(result.action, &record.action) || result.score != record.score {
            failures.push(format!(
                "record {index} (player {}, limit {}, movesLeft {}): \
                 got {:?} score {}, want {:?} score {}",
                record.player,
                record.node_limit,
                record.moves_left,
                result.action,
                result.score,
                record.action,
                record.score,
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} records diverged:\n{}",
        failures.len(),
        records.len(),
        failures
            .iter()
            .take(10)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n"),
    );
}

/// The oracle must stay an oracle: the opening book is a live-path feature and
/// `choose_depth` deliberately skips it, which is what makes every fixture
/// record pure search.
#[test]
fn choose_depth_never_consults_the_opening_book() {
    let state = State::new(12, 12, 2).expect("12x12 two-player board");
    assert_eq!(
        virus_search::book::opening_book_move(&state),
        Some(Action::mv(1, 1)),
        "the book does fire on this position"
    );
    let result = choose_depth(&state, 3).expect("a legal action exists");
    assert!(result.depth == 3 && result.nodes > 0, "search actually ran");
}
