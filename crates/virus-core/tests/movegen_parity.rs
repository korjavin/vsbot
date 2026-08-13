//! Parity against the Go engine over all 412 recorded positions.
//!
//! `fixtures/gobot_search_parity.jsonl` supplies the positions;
//! `fixtures/gobot_movegen_order.jsonl` supplies what GoBot's own
//! `game.Position.ForEachSearchAction` enumerated for each of them, plus the
//! FNV-1a state hash before and after applying the recorded action. See
//! `fixtures/gobot_movegen_order.README.md`.
//!
//! CLAUDE.md: parity is a hard gate. A single divergent record fails the suite.

use serde::Deserialize;
use std::path::PathBuf;
use virus_core::fixture::{parse_jsonl, SearchParityRecord};
use virus_core::{Action, Pos, Position};

fn fixtures_dir() -> PathBuf {
    // CARGO_MANIFEST_DIR is crates/virus-core.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("fixtures")
}

fn read(name: &str) -> String {
    let path = fixtures_dir().join(name);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("reading {}: {error}", path.display()))
}

fn search_records() -> Vec<SearchParityRecord> {
    parse_jsonl::<SearchParityRecord>(&read("gobot_search_parity.jsonl"))
        .unwrap_or_else(|(line, error)| panic!("gobot_search_parity.jsonl line {line}: {error}"))
}

/// One record of the enumeration-order golden file.
#[derive(Debug, Deserialize)]
struct OrderRecord {
    index: usize,
    #[serde(rename = "legalCount")]
    legal_count: usize,
    strategic: bool,
    #[serde(rename = "stateHash")]
    state_hash: u64,
    #[serde(rename = "afterHash")]
    after_hash: u64,
    #[serde(rename = "searchActions")]
    search_actions: Vec<GoAction>,
}

/// `["M", row, col]` or `["N", r1, c1, r2, c2]`.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum GoAction {
    Move(String, i32, i32),
    Neutrals(String, i32, i32, i32, i32),
}

impl From<&GoAction> for Action {
    fn from(action: &GoAction) -> Action {
        match action {
            GoAction::Move(tag, row, col) => {
                assert_eq!(tag, "M");
                Action::mv(*row, *col)
            }
            GoAction::Neutrals(tag, r1, c1, r2, c2) => {
                assert_eq!(tag, "N");
                Action::neutrals(Pos::new(*r1, *c1), Pos::new(*r2, *c2))
            }
        }
    }
}

fn order_records() -> Vec<OrderRecord> {
    parse_jsonl::<OrderRecord>(&read("gobot_movegen_order.jsonl"))
        .unwrap_or_else(|(line, error)| panic!("gobot_movegen_order.jsonl line {line}: {error}"))
}

/// Acceptance: every one of the 412 records decodes into a valid position.
#[test]
fn decodes_every_search_parity_record() {
    let records = search_records();
    assert_eq!(
        records.len(),
        412,
        "fixture size changed; parity coverage must be re-justified"
    );
    for (index, record) in records.iter().enumerate() {
        let state = record
            .to_state()
            .unwrap_or_else(|error| panic!("record {index}: {error}"));
        assert_eq!(state.rows(), 12, "record {index}");
        assert_eq!(state.cols(), 12, "record {index}");
        assert_eq!(state.current_player(), record.player, "record {index}");
        assert_eq!(state.moves_left(), record.moves_left, "record {index}");
        assert!(
            (3..=5).contains(&record.depth) && record.depth != 4,
            "record {index}: unexpected depth {}",
            record.depth
        );
    }
}

/// The FNV-1a key the Go/Java transposition tables use must match exactly, or
/// a shared-TT parity run in the search bead is meaningless.
#[test]
fn state_hash_matches_gobot() {
    for (record, golden) in search_records().iter().zip(order_records()) {
        let state = record.to_state().expect("record decodes");
        assert_eq!(
            state.state_hash(),
            golden.state_hash,
            "record {}: state hash divergence",
            golden.index
        );
    }
}

/// The whole point of the golden file: identical child ordering.
#[test]
fn search_action_order_matches_gobot() {
    let search = search_records();
    let golden = order_records();
    assert_eq!(search.len(), golden.len());
    let mut strategic_records = 0;
    for (record, golden) in search.iter().zip(golden.iter()) {
        let state = record.to_state().expect("record decodes");
        let position = Position::new(state);
        let ours = position.search_actions();
        let theirs: Vec<Action> = golden.search_actions.iter().map(Action::from).collect();
        assert_eq!(
            ours.len(),
            theirs.len(),
            "record {}: enumerated {} actions, GoBot enumerated {}",
            golden.index,
            ours.len(),
            theirs.len()
        );
        for (slot, (ours, theirs)) in ours.iter().zip(theirs.iter()).enumerate() {
            assert_eq!(
                ours, theirs,
                "record {}: action {slot} diverges",
                golden.index
            );
        }
        assert_eq!(
            position.legal_actions().len(),
            golden.legal_count,
            "record {}: legal action count diverges",
            golden.index
        );
        if golden.strategic {
            strategic_records += 1;
            assert!(
                ours.len()
                    > theirs
                        .iter()
                        .filter(|a| matches!(a, Action::Move { .. }))
                        .count(),
                "record {}: strategic record enumerated no curated pairs",
                golden.index
            );
        }
    }
    assert!(
        strategic_records >= 100,
        "expected the curated-pair branch to be well covered, got {strategic_records}"
    );
}

/// Applying the action GoBot chose must land on the state GoBot landed on.
/// One hash covers cells, mover, movesLeft, activity and neutral flags, so this
/// pins capture-fortifies, elimination and turn advance across 412 positions.
#[test]
fn apply_matches_gobot_transition() {
    for (record, golden) in search_records().iter().zip(order_records()) {
        let state = record.to_state().expect("record decodes");
        let action: Action = record.action.clone().into();
        let next = state
            .apply(action)
            .unwrap_or_else(|error| panic!("record {}: {error}", golden.index));
        assert_eq!(
            next.state_hash(),
            golden.after_hash,
            "record {}: successor divergence after {action:?}",
            golden.index
        );
        // The fast path must agree with the legality-checked one.
        assert_eq!(
            state.apply_generated(action).state_hash(),
            golden.after_hash,
            "record {}: apply_generated diverges from apply",
            golden.index
        );
    }
}

/// The action GoBot picked must be one our enumerator actually offers —
/// including curated `PlaceNeutrals` pairs, which is the sharpest available
/// check on the neutral-pair curation.
#[test]
fn recorded_choice_is_enumerated() {
    let mut neutral_choices = 0;
    for (index, record) in search_records().iter().enumerate() {
        let state = record.to_state().expect("record decodes");
        let chosen: Action = record.action.clone().into();
        let position = Position::new(state);
        let actions = position.search_actions();
        assert!(
            actions.iter().any(|a| a.same_transition(chosen)),
            "record {index}: GoBot chose {chosen:?}, which we never enumerated"
        );
        if matches!(chosen, Action::PlaceNeutrals { .. }) {
            neutral_choices += 1;
        }
    }
    assert!(
        neutral_choices > 0,
        "fixture should contain PlaceNeutrals choices"
    );
}

/// `Position` must be a pure cache over `State`: same actions, same successors.
#[test]
fn position_agrees_with_the_authoritative_state() {
    for (index, record) in search_records().iter().enumerate().take(60) {
        let state = record.to_state().expect("record decodes");
        let position = Position::new(state.clone());
        assert_eq!(
            position.legal_actions(),
            state.legal_actions(),
            "record {index}: Position::legal_actions diverges from State::legal_actions"
        );
        for action in position.search_actions() {
            let authoritative = state.apply(action).unwrap_or_else(|error| {
                panic!("record {index}: generated illegal {action:?}: {error}")
            });
            let fast = position.apply_search(action);
            assert_eq!(
                fast.state().snapshot(),
                authoritative.snapshot(),
                "record {index}: apply_search diverges for {action:?}"
            );
        }
    }
}

/// An unanalysed `Position` (the one `apply_search` returns) must enumerate the
/// same actions as a freshly analysed one — the lazy path is easy to get wrong.
#[test]
fn unanalyzed_position_enumerates_identically() {
    for record in search_records().iter().take(40) {
        let state = record.to_state().expect("record decodes");
        let analyzed = Position::new(state.clone());
        for action in analyzed.search_actions().into_iter().take(4) {
            let lazy = analyzed.apply_search(action);
            assert!(!lazy.analyzed());
            let eager = Position::new(lazy.state().clone());
            assert_eq!(
                lazy.search_actions(),
                eager.search_actions(),
                "lazy/eager enumeration divergence after {action:?}"
            );
        }
    }
}

/// The incremental Zobrist key must equal a from-scratch recomputation after
/// every transition, or the transposition table silently corrupts.
#[test]
fn zobrist_stays_incrementally_correct() {
    for record in search_records().iter().take(80) {
        let state = record.to_state().expect("record decodes");
        let position = Position::new(state.clone());
        for action in position.search_actions().into_iter().take(12) {
            let next = state.apply(action).expect("enumerated action is legal");
            assert_eq!(
                next.hash(),
                next.recomputed_hash(),
                "incremental hash diverges after {action:?}"
            );
        }
    }
}
