//! The committed probe set must stay loadable and stay *about* what it says.
//!
//! These are fixture-integrity checks, not a strength gate. Nothing here
//! asserts anything about what a net says at these positions — that is the
//! probe's own output and ARCHITECTURE.md invariant 7 forbids gating on it.
//! What they do assert is that every committed record still decodes under the
//! live snapshot validator, still hosts a real neutral decision, and still
//! carries the provenance that lets someone find it again. A fixture that
//! quietly rotted into 48 positions where `PlaceNeutrals` is illegal would
//! report a perfect score forever.

use std::collections::BTreeSet;
use std::path::PathBuf;

use virus_arena::probes::{parse_set, ProbeClass, ProbeRecord, ProbeSource};

/// The committed set, read from the workspace root.
fn probe_set() -> Vec<ProbeRecord> {
    // CARGO_MANIFEST_DIR is crates/virus-arena.
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/probes/neutrals-v1.jsonl");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("reading {}: {error}", path.display()));
    parse_set(&text).expect("the committed probe set parses")
}

#[test]
fn the_committed_set_is_the_documented_size() {
    let records = probe_set();
    // bd vsbot-07x asks for 20-60 positions; `fixtures/probes/README.md`
    // documents the split. A set that drifted out of that band is either
    // truncated or was regenerated without updating the docs.
    assert!(
        (20..=60).contains(&records.len()),
        "the probe set holds {} positions, outside the documented 20..=60",
        records.len()
    );
}

#[test]
fn every_position_is_a_real_neutral_decision() {
    for record in probe_set() {
        // `state()` re-validates through the live snapshot decoder and refuses
        // any position that cannot host a `PlaceNeutrals`.
        let state = record
            .state()
            .unwrap_or_else(|error| panic!("{}: {error}", record.id));
        assert_eq!(state.moves_left(), 3, "{}", record.id);
        assert!(state.can_place_neutrals(), "{}", record.id);
        let mover = state.current_player();
        assert!(
            !state.neutral_used(mover),
            "{}: the mover has already spent its placement",
            record.id
        );
        // Two own `Normal` cells are what a pair is made of; without them the
        // action is legal-looking but unreachable.
        assert!(
            state.owned_normals(mover).len() >= 2,
            "{}: the mover has fewer than two normals to convert",
            record.id
        );
        // The probe compares the pair class against the move class, so a
        // position with no legal move has nothing to compare and `run_probe`
        // refuses it.
        assert!(
            !state.move_targets(mover).is_empty(),
            "{}: the mover has no legal move, so there is no move class",
            record.id
        );
    }
}

#[test]
fn ids_and_positions_are_unique() {
    let records = probe_set();
    let mut ids = BTreeSet::new();
    let mut hashes = BTreeSet::new();
    for record in &records {
        assert!(ids.insert(record.id.clone()), "duplicate id {}", record.id);
        let state = record.state().expect("a valid position");
        assert!(
            hashes.insert(state.state_hash()),
            "{} repeats a position already in the set",
            record.id
        );
    }
}

#[test]
fn every_record_carries_its_provenance() {
    for record in probe_set() {
        assert!(
            !record.provenance.origin.is_empty(),
            "{}: no origin recorded",
            record.id
        );
        match record.source {
            ProbeSource::GamesDb => {
                assert!(
                    record.provenance.game_id.is_some(),
                    "{}: a mined position must name its game",
                    record.id
                );
                assert!(
                    record.provenance.turn.is_some(),
                    "{}: a mined position must name its turn",
                    record.id
                );
                assert!(
                    record.labels.placer_won.is_some(),
                    "{}: the class leans on the outcome, so it must be recorded",
                    record.id
                );
            }
            ProbeSource::PonderRepro => {
                assert!(
                    record.provenance.seed.is_some(),
                    "{}: a self-play position must name the seed that reproduces it",
                    record.id
                );
                assert_eq!(
                    record.class,
                    ProbeClass::ChampionChoseNeutral,
                    "{}",
                    record.id
                );
            }
            ProbeSource::LiveOwnerGame => {}
        }
        // Every source records the pair that was actually played there; that is
        // the decision the probe is about.
        assert!(
            record.provenance.played_neutrals.is_some(),
            "{}: no played pair recorded",
            record.id
        );
    }
}

#[test]
fn both_mined_classes_are_represented() {
    let records = probe_set();
    for class in [
        ProbeClass::LostAdvantage,
        ProbeClass::KeptAdvantage,
        ProbeClass::ChampionChoseNeutral,
    ] {
        assert!(
            records.iter().any(|record| record.class == class),
            "the set has no {class} positions; without the control class the \
             suspect numbers have nothing to be compared against"
        );
    }
}

#[test]
fn the_played_pair_was_legal_at_the_position() {
    for record in probe_set() {
        let state = record.state().expect("a valid position");
        let Some([a, b]) = record.provenance.played_neutrals else {
            continue;
        };
        let action = virus_core::Action::neutrals(
            virus_core::Pos::new(a[0], a[1]),
            virus_core::Pos::new(b[0], b[1]),
        );
        assert!(
            state.apply(action).is_ok(),
            "{}: the recorded pair {a:?}+{b:?} is not legal at the recorded position",
            record.id
        );
    }
}
