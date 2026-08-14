//! Gumbel / sequential-halving root selection (superiority.md §2d item 3).
//!
//! Three properties carry the feature and each has tests here:
//!
//! * **Play mode cannot reach it.** Same discipline as Dirichlet root noise:
//!   the exploration knobs are off in `Config::play()` explicitly, and asking
//!   for Gumbel *and* Dirichlet is a panic rather than a silent stacking.
//! * **It is deterministic.** Same seed, same schedule, same move, same target
//!   — batched or serial, DAG or tree.
//! * **The frames are right.** ARCHITECTURE.md invariant 1: the tree stores `W`
//!   absolute, and every `q` the Gumbel machinery reads is the root mover's.
//!   The test for it runs a player-2 root and checks the answer flips.
//!
//! Sim budgets are small for the same reason `tests/search.rs` keeps them
//! small: CI runs unoptimised. `examples/gumbelgauntlet` and
//! `examples/gumbelprobe` are where the real numbers come from.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use virus_core::{Cell, CellKind, State};
use virus_mcts::{
    Config, GumbelConfig, MctsSearcher, ParallelMcts, PolicyValueNet, ValueSource, CELLS,
};

fn champion() -> PolicyValueNet {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../artifacts/mcts_champion.json");
    PolicyValueNet::load(&path).expect("gen-5 champion loads")
}

/// The same developed midgame `tests/search.rs` uses, with `mover` a parameter:
/// the frame tests need a player-2 root and the rest do not care.
fn midgame(mover: virus_core::Player) -> State {
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
    State::from_grid(12, 12, 2, &cells, mover, 3, &[false, false]).expect("legal midgame")
}

fn gumbel(seed: u64, sims: u32, m: u16) -> Config {
    Config {
        value_source: ValueSource::Net,
        ..Config::self_play_gumbel(seed, sims, m)
    }
}

fn run<'net>(
    state: State,
    config: Config,
    net: Option<&'net PolicyValueNet>,
    sims: u32,
) -> MctsSearcher<'net> {
    let mut searcher = MctsSearcher::new(state, config, net);
    searcher.run_sims(sims);
    searcher
}

fn nonzero(counts: &[u32]) -> usize {
    counts.iter().filter(|n| **n > 0).count()
}

// ------------------------------------------------------- play-mode isolation

/// The structural half of "self-play only": every configuration a production
/// path can name has all three exploration knobs off.
#[test]
fn play_mode_takes_no_exploration_knob() {
    for config in [Config::play(), Config::default()] {
        assert!(config.gumbel.is_none(), "play mode must not run Gumbel");
        assert!(!config.root_noise);
        assert!(!config.visit_sampling);
    }
    // And the production bin's own shape — `Config { .., ..Config::play() }`,
    // `vsbot/src/mcts.rs` — cannot pick it up from a default.
    let bin_shaped = Config {
        seed: 42,
        value_source: ValueSource::Net,
        root_noise: false,
        visit_sampling: false,
        ..Config::play()
    };
    assert!(bin_shaped.gumbel.is_none());
}

/// The Dirichlet self-play config is untouched by this work: T3 adds an
/// alternative, it does not change the arm every generation to date ran.
#[test]
fn the_dirichlet_self_play_config_is_unchanged() {
    let config = Config::self_play(7, 0);
    assert!(config.gumbel.is_none());
    assert!(config.root_noise);
    assert!(config.visit_sampling);
}

#[test]
fn the_gumbel_self_play_config_replaces_dirichlet_rather_than_stacking() {
    let config = Config::self_play_gumbel(7, 192, 16);
    let asked = config.gumbel.expect("gumbel is on");
    assert_eq!(asked.m, 16);
    assert_eq!(asked.sims, 192);
    assert!(!config.root_noise, "Gumbel replaces Dirichlet");
    assert!(
        !config.visit_sampling,
        "the Gumbel draw is the exploration; sampling on top discards it"
    );
    assert_eq!(config.value_source, ValueSource::Net);
}

/// The shared-tree engine has no schedule, and quietly running PUCT instead
/// would produce rows a caller would file as a Gumbel generation.
#[test]
#[should_panic(expected = "no Gumbel root schedule")]
fn the_parallel_engine_refuses_a_gumbel_config() {
    let _ = ParallelMcts::new(midgame(1), Config::self_play_gumbel(1, 64, 8), None);
}

#[test]
#[should_panic(expected = "alternatives")]
fn asking_for_both_noise_and_gumbel_is_refused() {
    let config = Config {
        root_noise: true,
        gumbel: Some(GumbelConfig::default()),
        ..Config::default()
    };
    let _ = MctsSearcher::new(midgame(1), config, None);
}

// ------------------------------------------------------------- the schedule

/// Sequential halving spends the budget on the candidate set and nowhere else,
/// and it spends all of it.
#[test]
fn the_budget_goes_to_the_candidate_set_and_is_spent_in_full() {
    let net = champion();
    let searcher = run(midgame(1), gumbel(11, 64, 8), Some(&net), 64);
    let visits = searcher.root_visits();
    assert!(
        searcher.root_actions().len() > 8,
        "the position must offer more actions than m for this to mean anything"
    );
    assert_eq!(visits.iter().sum::<u32>(), 64, "every simulation lands");
    assert_eq!(
        nonzero(visits),
        8,
        "exactly the top-m candidates get visits"
    );
}

/// `m` above the legal action count degrades to "every action is a candidate"
/// rather than indexing off the end.
#[test]
fn an_m_larger_than_the_action_count_is_clamped() {
    let net = champion();
    let searcher = run(midgame(1), gumbel(11, 48, 4096), Some(&net), 48);
    let actions = searcher.root_actions().len();
    assert_eq!(nonzero(searcher.root_visits()), actions.min(48));
}

/// The played action is the schedule's argmax, which is always one of the
/// candidates the schedule actually measured — never an action cut in round
/// one, and never an unvisited one.
#[test]
fn the_played_action_is_a_measured_candidate() {
    let net = champion();
    let searcher = run(midgame(1), gumbel(5, 64, 8), Some(&net), 64);
    let action = searcher.best_action().expect("a non-terminal root answers");
    let slot = searcher
        .root_actions()
        .iter()
        .position(|a| *a == action)
        .expect("the answer is a root action");
    assert!(
        searcher.root_visits()[slot] > 0,
        "the Gumbel answer must be an action the search measured"
    );
    assert!(searcher.is_gumbel());
}

/// A Gumbel search must survive the budgets a caller can actually give it:
/// fewer simulations than the schedule was planned for (halvings never
/// finish), and more (the surplus goes to the survivor).
#[test]
fn a_budget_that_misses_the_plan_still_produces_a_legal_answer() {
    let net = champion();
    let legal = midgame(1).legal_actions();
    for (planned, actual) in [(192u32, 24u32), (24, 96)] {
        let searcher = run(midgame(1), gumbel(3, planned, 8), Some(&net), actual);
        let action = searcher.best_action().expect("an answer");
        assert!(
            legal.contains(&action),
            "{planned}/{actual} played {action:?}"
        );
        assert_eq!(searcher.root_visits().iter().sum::<u32>(), actual);
    }
}

/// A deadline budget is not the Gumbel use case (self-play is sims-budgeted),
/// but it must not deadlock or panic on a phase boundary.
#[test]
fn a_deadline_budget_terminates_under_a_gumbel_schedule() {
    let net = champion();
    let mut searcher = MctsSearcher::new(midgame(1), gumbel(9, 192, 16), Some(&net));
    searcher.run_until_deadline(Instant::now() + Duration::from_millis(30));
    assert!(searcher.sims_run() >= 1);
    assert!(searcher.best_action().is_some());
}

// ------------------------------------------------------------- determinism

#[test]
fn a_gumbel_search_is_byte_stable_for_a_fixed_seed() {
    let net = champion();
    for batch in [1u16, 8] {
        let config = Config {
            batch_size: batch,
            ..gumbel(20_260_814, 96, 8)
        };
        let a = run(midgame(1), config, Some(&net), 96);
        let b = run(midgame(1), config, Some(&net), 96);
        assert_eq!(a.root_visits(), b.root_visits(), "batch {batch}");
        assert_eq!(a.best_action(), b.best_action(), "batch {batch}");
        let (pa, pb) = (a.root_improved_policy(), b.root_improved_policy());
        assert_eq!(
            pa.iter().map(|p| p.to_bits()).collect::<Vec<_>>(),
            pb.iter().map(|p| p.to_bits()).collect::<Vec<_>>(),
            "the improved policy is bit-identical, not merely close"
        );
    }
}

/// The seed has to reach the answer — that is what makes this exploration
/// rather than a deterministic re-ranking of the prior. Checked over several
/// seeds because any two draws can coincide.
#[test]
fn the_seed_changes_the_draw_and_therefore_the_search() {
    let net = champion();
    let base = run(midgame(1), gumbel(1, 64, 8), Some(&net), 64);
    let differing = (2..8u64)
        .map(|seed| run(midgame(1), gumbel(seed, 64, 8), Some(&net), 64))
        .filter(|other| other.root_visits() != base.root_visits())
        .count();
    assert!(
        differing >= 4,
        "the Gumbel draw barely moved across six seeds — is it seeded at all?"
    );
}

/// The DAG is orthogonal: it merges positions below the root, and the root
/// schedule neither reads nor writes the index.
#[test]
fn gumbel_is_deterministic_with_and_without_the_dag() {
    let net = champion();
    for dag in [true, false] {
        let config = Config {
            dag,
            ..gumbel(77, 64, 8)
        };
        let a = run(midgame(1), config, Some(&net), 64);
        let b = run(midgame(1), config, Some(&net), 64);
        assert_eq!(a.root_visits(), b.root_visits(), "dag {dag}");
        assert_eq!(a.best_action(), b.best_action(), "dag {dag}");
    }
}

// -------------------------------------------------------- the policy target

/// The improved policy is a distribution over **every** legal action, not over
/// the candidate set: that is exactly the information the raw visit counts
/// throw away at these budgets.
#[test]
fn the_improved_policy_covers_every_legal_action() {
    let net = champion();
    let searcher = run(midgame(1), gumbel(13, 64, 8), Some(&net), 64);
    let policy = searcher.root_improved_policy();
    assert_eq!(policy.len(), searcher.root_actions().len());
    assert!((policy.iter().sum::<f32>() - 1.0).abs() < 1e-4);
    assert!(policy.iter().all(|p| *p >= 0.0));
    assert!(
        policy.iter().filter(|p| **p > 0.0).count() > nonzero(searcher.root_visits()),
        "the completed-Q target must be denser than the sequential-halving visits"
    );
}

/// Defined for a PUCT search too, so the two arms are comparable on one
/// quantity — which is what the S3-T3 policy-target measurement needs.
#[test]
fn the_improved_policy_is_defined_for_a_puct_search_as_well() {
    let net = champion();
    let config = Config {
        value_source: ValueSource::Net,
        ..Config::play()
    };
    let searcher = run(midgame(1), config, Some(&net), 64);
    assert!(!searcher.is_gumbel());
    let policy = searcher.root_improved_policy();
    assert_eq!(policy.len(), searcher.root_actions().len());
    assert!((policy.iter().sum::<f32>() - 1.0).abs() < 1e-4);
}

/// A terminal root has no target and must say so rather than divide by zero.
#[test]
fn a_terminal_root_has_no_improved_policy() {
    let mut cells = vec![Cell::EMPTY; CELLS];
    cells[0] = Cell::new(1, CellKind::Base);
    cells[CELLS - 1] = Cell::new(2, CellKind::Base);
    for cell in cells.iter_mut().skip(1).take(CELLS - 2) {
        *cell = Cell::new(0, CellKind::Neutral);
    }
    let state = State::from_grid(12, 12, 2, &cells, 1, 3, &[true, true]).expect("legal");
    let searcher = MctsSearcher::new(state, gumbel(1, 16, 4), None);
    assert!(searcher.root_improved_policy().is_empty());
    assert!(
        !searcher.is_gumbel(),
        "no schedule at a root with no choice"
    );
}

/// ARCHITECTURE.md invariant 1, applied to the completed-Q target.
///
/// With no net the priors are uniform, so every logit is equal and the improved
/// policy's ordering is **purely** the mover-frame `q`. That makes the frame
/// checkable from outside: for a player-1 root the target's argmax is the
/// largest absolute-frame `W/N`, and for a player-2 root it is the *smallest* —
/// the same value, read from the other chair. An implementation that skipped
/// the `sign(root)` multiply would pass the first half and fail the second.
#[test]
fn the_improved_policy_reads_q_from_the_root_movers_chair() {
    for mover in [1u8, 2u8] {
        let searcher = run(midgame(mover), gumbel(21, 96, 8), None, 96);
        let policy = searcher.root_improved_policy();
        let values = searcher.root_action_values_abs();
        let visits = searcher.root_visits();

        let top = (0..policy.len())
            .filter(|a| visits[*a] > 0)
            .max_by(|x, y| policy[*x].total_cmp(&policy[*y]))
            .expect("something was visited");
        let expected = (0..values.len())
            .filter(|a| visits[*a] > 0)
            .max_by(|x, y| {
                let sign = if mover == 1 { 1.0 } else { -1.0 };
                (sign * values[*x]).total_cmp(&(sign * values[*y]))
            })
            .expect("something was visited");
        assert_eq!(
            top, expected,
            "mover {mover}: the target's best visited action is not the one \
             best for the mover"
        );

        // And the two chairs genuinely disagree on this position, so the test
        // is not passing on a degenerate tree where every ordering coincides.
        let visited: Vec<usize> = (0..values.len()).filter(|a| visits[*a] > 0).collect();
        let lo = visited
            .iter()
            .copied()
            .min_by(|x, y| values[*x].total_cmp(&values[*y]))
            .expect("something was visited");
        let hi = visited
            .iter()
            .copied()
            .max_by(|x, y| values[*x].total_cmp(&values[*y]))
            .expect("something was visited");
        assert_ne!(lo, hi, "mover {mover}: the visited values are all equal");
    }
}
