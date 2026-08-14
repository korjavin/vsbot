//! Net-vs-net: two *different* artifacts in one gauntlet run.
//!
//! Until bd `vsbot-0zr` this was refused outright — the harness shared one
//! loaded `PolicyValueNet` across every game and thread, and `arena` returned an
//! error rather than silently playing one artifact against itself. That refusal
//! was right (a fabricated 50/50 for a comparison that never happened is worse
//! than no number), and this file is the evidence that the replacement is real
//! rather than a flag that changes the report and nothing else.
//!
//! Four properties, in the order they can go wrong:
//!
//! 1. **The two artifacts really are different.** A "net-vs-net" test that
//!    loaded the same file twice would pass every assertion below while proving
//!    nothing, so the metadata difference is asserted first.
//! 2. **Side B's net is actually consulted.** A vs B must not replay A vs A.
//!    This is the assertion that would have failed against the old
//!    single-net plumbing.
//! 3. **A net follows its side, not its seat.** Swapping the two arms must
//!    mirror the run exactly — game `2k` of the swapped run is game `2k+1` of
//!    the original, because both games of a pair share an opening seed. A net
//!    pinned to a seat would leave the run *unchanged* instead of mirrored,
//!    which is the same first-mover bias the colour pairing exists to cancel.
//! 4. **The single-net path did not move.** `run` and
//!    `run_with_nets(SideNets::shared(..))` must be byte-identical, and a net
//!    against itself must still read exactly 50%.
//!
//! Everything here is [`Budget::Nodes`] (simulations, for MCTS) so the runs are
//! clock-free and reproducible; see `tests/determinism.rs`.

use std::path::PathBuf;
use virus_arena::engine::{self, Budget, Engine, SideSpec};
use virus_arena::gauntlet::{run, run_with_nets, GauntletConfig, GauntletResult, SideNets};
use virus_mcts::PolicyValueNet;

/// The two artifacts the repo ships. They differ in more than their weights:
/// the champion has a value head and `mcts_policy` does not, so a run that
/// confused them would also be running a different `ValueSource` path.
const CHAMPION: &str = "artifacts/mcts_champion.json";
const POLICY: &str = "artifacts/mcts_policy.json";

fn load(relative: &str) -> PolicyValueNet {
    // CARGO_MANIFEST_DIR is crates/virus-arena.
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(relative);
    PolicyValueNet::load(&path)
        .unwrap_or_else(|error| panic!("loading {}: {error}", path.display()))
}

/// An MCTS side at a fixed simulation count.
///
/// The `net` field is display metadata only — `engine::build` takes the loaded
/// artifact as an argument — but it is filled in so a failure message names the
/// artifact that was supposed to be playing.
fn mcts(artifact: &str, sims: u64) -> SideSpec {
    SideSpec {
        engine: Engine::Mcts,
        budget: Budget::Nodes(sims),
        net: Some(artifact.to_owned()),
    }
}

/// A deliberately tiny run: MCTS on the 12x12 board the net encoding requires,
/// at a handful of simulations, capped early. The properties under test are
/// about *which artifact each seat holds*, and none of them get truer with a
/// bigger budget — they only get slower.
fn config(side_a: SideSpec, side_b: SideSpec) -> GauntletConfig {
    GauntletConfig {
        side_a,
        side_b,
        games: 4,
        seed: 20_260_814,
        // The net's input encoding has no representation for another size.
        rows: 12,
        cols: 12,
        max_turns: 12,
        threads: 1,
        ..GauntletConfig::default()
    }
}

/// The per-game detail, not just the tally — two runs agreeing on W-L-D while
/// disagreeing about which games were won would pass a tally-only check.
fn fingerprint(result: &GauntletResult) -> Vec<String> {
    result
        .games
        .iter()
        .map(|game| {
            format!(
                "{} a_p1={} winner={} turns={} plies={} territory={} work_a={} work_b={}",
                game.index,
                game.a_is_p1,
                game.winner,
                game.turns,
                game.plies,
                game.territory_winner,
                game.work_a,
                game.work_b,
            )
        })
        .collect()
}

/// Property 1. If this fails, every other test in the file is vacuous.
#[test]
fn the_two_test_artifacts_are_genuinely_different_nets() {
    let champion = load(CHAMPION);
    let policy = load(POLICY);
    assert_ne!(
        champion.arch(),
        policy.arch(),
        "the two artifacts must not be the same export"
    );
    assert!(
        champion.has_value_head(),
        "{CHAMPION} is the value-head artifact"
    );
    assert!(
        !policy.has_value_head(),
        "{POLICY} is the policy-only artifact"
    );
    assert_ne!(champion.pair_bias(), policy.pair_bias());
}

/// Property 1, at the level the harness sees it: two arms, two artifacts, two
/// different searchers. Same positions, same seat, same simulation count — only
/// the net differs, and the moves differ with it.
///
/// A *line* rather than the opening position alone: the first move of a fresh
/// 12x12 board is close to forced and both artifacts pick it, which says
/// nothing about either. Walking a few plies in gets to positions where the two
/// policies actually disagree.
#[test]
fn two_arms_on_two_artifacts_search_differently() {
    let champion = load(CHAMPION);
    let policy = load(POLICY);
    let spec = mcts(CHAMPION, 64);

    let mut state = virus_core::State::new(12, 12, 2).expect("12x12 two-player");
    let mut champion_line = Vec::new();
    let mut policy_line = Vec::new();
    for _ in 0..9 {
        let seat = state.current_player();
        // A fresh side per position: `MctsSide` roots at the seat it is built
        // for and asserts it is not asked to move out of turn.
        let (champion_action, _) = engine::build(&spec, seat, Some(&champion))
            .expect("champion side")
            .choose(&state);
        let (policy_action, _) = engine::build(&spec, seat, Some(&policy))
            .expect("policy side")
            .choose(&state);
        champion_line.push(champion_action);
        policy_line.push(policy_action);
        // Advance down a fixed line so both nets are asked about the same
        // positions; which line does not matter, only that it is the same one.
        state = state
            .apply(state.legal_actions()[0])
            .expect("legal_actions returns legal actions");
    }

    assert!(champion_line.iter().all(Option::is_some));
    assert_ne!(
        champion_line, policy_line,
        "two different artifacts chose identically over nine positions; either the net \
         argument is being ignored or these artifacts are the same export — check the \
         first test in this file before believing the latter"
    );
}

/// Property 2: the one the old plumbing could not satisfy. A gauntlet of the
/// champion against the policy-only net must not be the champion against
/// itself.
#[test]
fn a_net_vs_net_run_is_not_one_artifact_against_itself() {
    let champion = load(CHAMPION);
    let policy = load(POLICY);
    let config = config(mcts(CHAMPION, 8), mcts(POLICY, 8));

    let net_vs_net = run_with_nets(&config, SideNets::new(Some(&champion), Some(&policy)))
        .expect("net-vs-net run");
    let self_play =
        run_with_nets(&config, SideNets::shared(Some(&champion))).expect("champion self-gauntlet");

    assert_ne!(
        fingerprint(&net_vs_net),
        fingerprint(&self_play),
        "side B's artifact was not consulted: A-vs-B replayed A-vs-A"
    );
    // And the report has to name both artifacts, or the row is unreproducible.
    assert_eq!(net_vs_net.summary.side_a, "mcts[mcts_champion]:n8");
    assert_eq!(net_vs_net.summary.side_b, "mcts[mcts_policy]:n8");
}

/// Property 3: a net travels with its arm.
///
/// Both games of a pair share an opening seed, so swapping the arms swaps the
/// two games *within* each pair: game `2k` of the swapped run is game `2k+1` of
/// the original, played from the other chair. The tally therefore mirrors
/// exactly. If a net were pinned to a seat instead, the swapped run would be
/// the *same* run and the tally would come back unmirrored.
#[test]
fn a_net_follows_its_side_and_not_its_seat() {
    let champion = load(CHAMPION);
    let policy = load(POLICY);

    let forward = run_with_nets(
        &config(mcts(CHAMPION, 8), mcts(POLICY, 8)),
        SideNets::new(Some(&champion), Some(&policy)),
    )
    .expect("forward");
    let swapped = run_with_nets(
        &config(mcts(POLICY, 8), mcts(CHAMPION, 8)),
        SideNets::new(Some(&policy), Some(&champion)),
    )
    .expect("swapped");

    assert_eq!(forward.record.wins, swapped.record.losses);
    assert_eq!(forward.record.losses, swapped.record.wins);
    assert_eq!(forward.record.draws, swapped.record.draws);

    for pair in 0..(forward.games.len() / 2) {
        let original = &forward.games[pair * 2 + 1];
        let mirrored = &swapped.games[pair * 2];
        assert_eq!(
            (
                original.winner,
                original.plies,
                original.turns,
                original.territory_winner
            ),
            (
                mirrored.winner,
                mirrored.plies,
                mirrored.turns,
                mirrored.territory_winner
            ),
            "pair {pair}: swapping the arms must replay the pair's other game"
        );
        // Work is reported per *side*, so it swaps with the arms.
        assert_eq!(original.work_a, mirrored.work_b);
        assert_eq!(original.work_b, mirrored.work_a);
    }

    // A run of empty games would satisfy the mirror trivially.
    assert!(
        forward.games.iter().all(|game| game.plies > 10),
        "the games must actually be played: {:?}",
        fingerprint(&forward)
    );
}

/// Property 4a: `run` is exactly `run_with_nets` with a shared artifact. The
/// single-net path must not have moved a byte — one loaded net, one set of
/// games, one tally.
#[test]
fn the_shared_net_path_is_unchanged() {
    let champion = load(CHAMPION);
    let config = config(
        mcts(CHAMPION, 8),
        SideSpec::parse("greedy", Budget::Nodes(1)).expect("greedy"),
    );

    let via_run = run(&config, Some(&champion)).expect("run");
    let via_nets =
        run_with_nets(&config, SideNets::shared(Some(&champion))).expect("run_with_nets");

    assert_eq!(via_run.record, via_nets.record);
    assert_eq!(fingerprint(&via_run), fingerprint(&via_nets));
}

/// Property 4b: the pairing property, on the path that now goes through
/// `SideNets`. One artifact against itself still splits every pair 1-1.
#[test]
fn one_artifact_against_itself_still_cancels_exactly() {
    let champion = load(CHAMPION);
    let config = config(mcts(CHAMPION, 8), mcts(CHAMPION, 8));
    let result = run_with_nets(&config, SideNets::shared(Some(&champion))).expect("self-gauntlet");

    assert_eq!(
        result.record.wins, result.record.losses,
        "a net against itself must read 50%: {:?}",
        result.record
    );
    for pair in 0..(result.games.len() / 2) {
        let even = &result.games[pair * 2];
        let odd = &result.games[pair * 2 + 1];
        assert_eq!(
            (even.winner, even.plies),
            (odd.winner, odd.plies),
            "pair {pair} did not replay one game from both chairs"
        );
    }
}
