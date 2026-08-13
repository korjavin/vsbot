//! Searcher behaviour: determinism, legality, budgets, and the self-play knobs.
//!
//! The determinism tests are the load-bearing ones. CLAUDE.md requires every
//! engine path to be reproducible, and MCTS is where that is easiest to lose —
//! a stray `HashMap` iteration or an unseeded shuffle would show up here as a
//! byte-unstable visit vector long before it showed up as an unreproducible
//! gauntlet.
//!
//! Sim budgets are deliberately small. CI runs `cargo test` unoptimised, where
//! a net forward costs ~30x its release time, so these are sized to prove the
//! property rather than to search well; `examples/mctsbench` is where the real
//! numbers come from.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use virus_core::{Action, Cell, CellKind, State};
use virus_mcts::{
    action_from_id, action_id, Config, MctsSearcher, PolicyValueNet, ValueSource, ACTION_ID_COUNT,
    CELLS,
};

fn champion() -> PolicyValueNet {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../artifacts/mcts_champion.json");
    PolicyValueNet::load(&path).expect("gen-5 champion loads")
}

fn fresh() -> State {
    State::new(12, 12, 2).expect("12x12 two-player start")
}

/// A developed midgame position: both sides have material, the branching factor
/// is realistic, and neutral placement is still available so the pair head is
/// exercised.
fn midgame() -> State {
    let mut cells = vec![Cell::EMPTY; CELLS];
    cells[0] = Cell::new(1, CellKind::Base);
    cells[CELLS - 1] = Cell::new(2, CellKind::Base);
    for index in [1, 12, 13, 14, 25, 26, 27, 38] {
        cells[index] = Cell::new(1, CellKind::Normal);
    }
    for index in [
        CELLS - 2,
        CELLS - 13,
        CELLS - 14,
        CELLS - 15,
        CELLS - 26,
        CELLS - 27,
    ] {
        cells[index] = Cell::new(2, CellKind::Normal);
    }
    State::from_grid(12, 12, 2, &cells, 1, 3, &[false, false]).expect("legal midgame")
}

fn run(state: State, config: Config, net: Option<&PolicyValueNet>, sims: u32) -> MctsSearcher<'_> {
    let mut searcher = MctsSearcher::new(state, config, net);
    searcher.run_sims(sims);
    searcher
}

// ---------------------------------------------------------------- determinism

#[test]
fn run_sims_is_byte_stable_for_a_fixed_seed() {
    let net = champion();
    let config = Config {
        value_source: ValueSource::Net,
        ..Config::play()
    };
    let a = run(midgame(), config, Some(&net), 150);
    let b = run(midgame(), config, Some(&net), 150);
    assert_eq!(a.best_action(), b.best_action(), "same seed, same move");
    assert_eq!(a.root_visits(), b.root_visits(), "same seed, same tree");
    assert_eq!(a.root_actions(), b.root_actions());
    assert_eq!(
        a.root_value_abs().to_bits(),
        b.root_value_abs().to_bits(),
        "root value is bit-identical, not merely close"
    );
}

/// Play mode must consume no randomness at all, so the seed cannot reach the
/// result. If a stochastic tie-break ever creeps in, this fails.
#[test]
fn play_mode_ignores_the_seed_entirely() {
    let net = champion();
    let base = Config {
        value_source: ValueSource::Net,
        ..Config::play()
    };
    let a = run(midgame(), Config { seed: 1, ..base }, Some(&net), 120);
    let b = run(
        midgame(),
        Config {
            seed: 0xDEAD_BEEF_CAFE_F00D,
            ..base
        },
        Some(&net),
        120,
    );
    assert_eq!(a.root_visits(), b.root_visits(), "no RNG in the play path");
    assert_eq!(a.best_action(), b.best_action());
}

/// Splitting one budget into two calls must build the same tree as spending it
/// in one go — the property `run_until_deadline` relies on to be sane.
#[test]
fn simulations_are_resumable() {
    let net = champion();
    let config = Config {
        value_source: ValueSource::Net,
        ..Config::play()
    };
    let one_shot = run(midgame(), config, Some(&net), 120);
    let mut split = MctsSearcher::new(midgame(), config, Some(&net));
    split.run_sims(50);
    split.run_sims(70);
    assert_eq!(one_shot.root_visits(), split.root_visits());
    assert_eq!(one_shot.sims_run(), split.sims_run());
}

#[test]
fn the_hand_tuned_fallback_is_deterministic_too() {
    let a = run(midgame(), Config::play(), None, 200);
    let b = run(midgame(), Config::play(), None, 200);
    assert_eq!(a.root_visits(), b.root_visits());
    assert_eq!(a.best_action(), b.best_action());
}

// ---------------------------------------------------------------- correctness

#[test]
fn the_search_always_picks_a_legal_action() {
    let net = champion();
    for (label, state) in [("fresh", fresh()), ("midgame", midgame())] {
        for (source, net) in [
            (ValueSource::HandTuned, None),
            (ValueSource::Net, Some(&net)),
        ] {
            let legal = state.legal_actions();
            let searcher = run(
                state.clone(),
                Config {
                    value_source: source,
                    ..Config::play()
                },
                net,
                60,
            );
            let action = searcher
                .best_action()
                .expect("a non-terminal root has a move");
            assert!(
                legal.contains(&action),
                "{label}/{source:?} played an illegal action {action:?}"
            );
        }
    }
}

#[test]
fn priors_are_a_distribution_over_the_legal_actions() {
    let net = champion();
    let state = midgame();
    let searcher = MctsSearcher::new(state.clone(), Config::play(), Some(&net));
    let priors = searcher.root_priors();
    assert_eq!(priors.len(), state.legal_actions().len());
    assert!(
        state
            .legal_actions()
            .iter()
            .any(|a| matches!(a, Action::PlaceNeutrals { .. })),
        "the test position must offer neutral pairs, or the pair head is untested"
    );
    for prior in priors {
        assert!(
            *prior > 0.0 && prior.is_finite(),
            "every legal action gets positive prior mass, got {prior}"
        );
    }
    let sum: f32 = priors.iter().sum();
    assert!((sum - 1.0).abs() < 1e-4, "softmax normalises, got {sum}");
}

#[test]
fn the_root_value_stays_in_the_tanh_range() {
    let net = champion();
    let searcher = run(
        midgame(),
        Config {
            value_source: ValueSource::Net,
            ..Config::play()
        },
        Some(&net),
        100,
    );
    let value = searcher.root_value_abs();
    assert!(
        (-1.0..=1.0).contains(&value),
        "root value {value} escaped [-1, 1]"
    );
}

#[test]
fn every_simulation_lands_somewhere() {
    let searcher = run(midgame(), Config::play(), None, 137);
    assert_eq!(searcher.sims_run(), 137);
    let total: u32 = searcher.root_visits().iter().sum();
    // Every simulation credits exactly one root edge, except those that stop at
    // the root itself (only possible on the very first, expanding, sim).
    assert!(
        total == 137 || total == 136,
        "root edge visits {total} do not account for 137 sims"
    );
}

#[test]
fn a_terminal_root_yields_no_action() {
    // Both bases intact but the whole board is neutral: nobody can move.
    let mut cells = vec![Cell::NEUTRAL; CELLS];
    cells[0] = Cell::new(1, CellKind::Base);
    cells[CELLS - 1] = Cell::new(2, CellKind::Base);
    let state = State::from_grid(12, 12, 2, &cells, 1, 3, &[false, false]).expect("legal");
    assert!(state.legal_actions().is_empty(), "the position is stuck");

    let mut searcher = MctsSearcher::new(state, Config::play(), None);
    searcher.run_sims(50);
    assert_eq!(searcher.best_action(), None);
    assert!((-1.0..=1.0).contains(&searcher.root_value_abs()));
}

// ---------------------------------------------------------------- budgets

#[test]
fn the_deadline_budget_runs_at_least_one_sim_and_then_stops() {
    let net = champion();
    let mut searcher = MctsSearcher::new(
        midgame(),
        Config {
            value_source: ValueSource::Net,
            ..Config::play()
        },
        Some(&net),
    );
    // An already-expired deadline still buys one simulation, so the caller
    // always has a move to play.
    searcher.run_until_deadline(Instant::now());
    assert_eq!(searcher.sims_run(), 1);
    assert!(searcher.best_action().is_some());

    let start = Instant::now();
    searcher.run_for(Duration::from_millis(120));
    let elapsed = start.elapsed();
    assert!(
        searcher.sims_run() > 1,
        "the budget bought more simulations"
    );
    assert!(
        elapsed < Duration::from_millis(600),
        "overran the 120ms budget by too much: {elapsed:?}"
    );
}

// ---------------------------------------------------------------- self-play

#[test]
fn root_noise_perturbs_the_prior_and_is_seeded() {
    let net = champion();
    let plain = MctsSearcher::new(midgame(), Config::play(), Some(&net));
    let clean: Vec<f32> = plain.root_priors().to_vec();

    let noisy = |seed: u64| {
        let config = Config {
            root_noise: true,
            ..Config::self_play(seed, 0)
        };
        MctsSearcher::new(midgame(), config, Some(&net))
            .root_priors()
            .to_vec()
    };

    let a = noisy(7);
    let b = noisy(7);
    let c = noisy(8);
    assert_eq!(a, b, "root noise is a pure function of the seed");
    assert_ne!(a, c, "different seeds give different noise");
    assert_ne!(a, clean, "noise actually moved the prior");
    let sum: f32 = a.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-3,
        "the mixed prior stays a distribution, got {sum}"
    );
    for (mixed, base) in a.iter().zip(&clean) {
        // (1 - eps) * p <= mixed <= (1 - eps) * p + eps
        assert!(
            *mixed >= 0.75 * base - 1e-6 && *mixed <= 0.75 * base + 0.25 + 1e-6,
            "mixed prior {mixed} is not a 0.75/0.25 blend of {base}"
        );
    }
}

#[test]
fn temperature_sampling_is_on_only_for_the_opening_plies() {
    assert!(Config::self_play(1, 0).visit_sampling);
    assert!(Config::self_play(1, 20).visit_sampling);
    assert!(
        !Config::self_play(1, 21).visit_sampling,
        "tau drops at ply 21"
    );
    assert!(!Config::play().visit_sampling);
    assert!(!Config::play().root_noise, "play mode never takes noise");
}

#[test]
fn visit_sampling_picks_legal_actions_and_follows_the_seed() {
    let net = champion();
    let state = midgame();
    let legal = state.legal_actions();
    let sample = |seed: u64| {
        let mut searcher = MctsSearcher::new(state.clone(), Config::self_play(seed, 0), Some(&net));
        searcher.run_sims(40);
        searcher.chosen_action().expect("a move")
    };
    let mut seen = std::collections::BTreeSet::new();
    for seed in 0..8u64 {
        let action = sample(seed);
        assert!(legal.contains(&action), "sampled an illegal {action:?}");
        seen.insert(action_id(action));
    }
    assert!(
        seen.len() > 1,
        "temperature-1 sampling collapsed onto a single move"
    );
    assert_eq!(sample(3), sample(3), "sampling is reproducible per seed");
}

// ---------------------------------------------------------------- action ids

#[test]
fn flat_action_ids_round_trip() {
    for index in 0..CELLS {
        let action = Action::mv((index / 12) as i32, (index % 12) as i32);
        assert_eq!(action_id(action), index);
        assert_eq!(action_from_id(index), Some(action));
    }
    for (i, j) in [(0usize, 1usize), (5, 100), (0, 143), (142, 143)] {
        let action = virus_core::Action::neutrals(
            virus_core::Pos::new((i / 12) as i32, (i % 12) as i32),
            virus_core::Pos::new((j / 12) as i32, (j % 12) as i32),
        );
        let id = action_id(action);
        assert_eq!(id, CELLS + i * CELLS + j, "matches the trainer's numbering");
        assert!(id < ACTION_ID_COUNT);
        assert_eq!(action_from_id(id), Some(action));
    }
    assert_eq!(action_from_id(ACTION_ID_COUNT), None);
    // i >= j is not a pair the enumerator can produce.
    assert_eq!(action_from_id(CELLS + 5 * CELLS + 5), None);
}

/// Off-board coordinates must panic rather than wrap into a valid cell.
///
/// `(0, 12)` folds to row-major index 12, which is the perfectly valid cell
/// `(1, 0)`. Checking only the folded index would let the two share one policy
/// target — a mislabelled self-play row, not a crash.
#[test]
#[should_panic(expected = "off the 12x12 board")]
fn a_column_past_the_edge_does_not_alias_the_next_row() {
    action_id(Action::mv(0, 12));
}

#[test]
#[should_panic(expected = "off the 12x12 board")]
fn a_negative_coordinate_is_rejected() {
    action_id(Action::mv(-1, 0));
}

#[test]
#[should_panic(expected = "off the 12x12 board")]
fn an_off_board_neutral_pair_is_rejected() {
    action_id(virus_core::Action::neutrals(
        virus_core::Pos::new(0, 0),
        virus_core::Pos::new(12, 0),
    ));
}

/// The nets are 12x12 two-player artifacts. A wider game has no encoding — the
/// symbol alphabet has one "opponent" class and one opponent neutral-used
/// plane — so it must be refused up front, not searched on nonsense priors.
#[test]
#[should_panic(expected = "two-player only")]
fn a_four_player_state_is_refused_when_a_net_is_supplied() {
    let net = champion();
    let state = State::new(12, 12, 4).expect("12x12 four-player start");
    let _ = MctsSearcher::new(state, Config::play(), Some(&net));
}

/// ...but the hand-tuned fallback has no such limit.
#[test]
fn a_four_player_state_searches_fine_without_a_net() {
    let state = State::new(12, 12, 4).expect("12x12 four-player start");
    let legal = state.legal_actions();
    let searcher = run(state, Config::play(), None, 60);
    let action = searcher.best_action().expect("a move");
    assert!(
        legal.contains(&action),
        "illegal {action:?} in a 4-player game"
    );
}

/// Ordering must not matter: the trainer keys pairs by `min`/`max`.
#[test]
fn pair_ids_are_order_independent() {
    let a = virus_core::Pos::new(0, 5);
    let b = virus_core::Pos::new(3, 2);
    assert_eq!(
        action_id(virus_core::Action::neutrals(a, b)),
        action_id(virus_core::Action::neutrals(b, a))
    );
}

/// Every action the engine can actually generate has a distinct flat id.
#[test]
fn generated_actions_map_to_distinct_ids() {
    let state = midgame();
    let mut ids = std::collections::BTreeSet::new();
    for action in state.legal_actions() {
        let id = action_id(action);
        assert!(id < ACTION_ID_COUNT);
        assert_eq!(action_from_id(id), Some(action), "{action:?} round-trips");
        assert!(ids.insert(id), "{action:?} collided on id {id}");
    }
    assert!(!ids.is_empty());
}
