//! Offline reproduction of the pondering canary failure (bd `vsbot-gei`).
//!
//! ```bash
//! cargo run --release -p vsbot --example ponderrepro -- \
//!     --games 6 --ponder-sims 6000 --answer-sims 800
//! ```
//!
//! # Why offline, and why simulation counts instead of milliseconds
//!
//! The live soak is the acceptance evidence, but it is hours long and every
//! number it produces is entangled with the box's load. The *mechanism* the
//! bead's four hypotheses are about is not: it is what a re-rooted tree looks
//! like when our turn arrives, and what the stop rules then do with it. Both are
//! pure functions of the tree, so this harness drives them on simulation counts
//! and reports the same quantities the live trace does — deterministically, in
//! minutes.
//!
//! One `MctsSearcher` per game plays the part of the pondering session: it is
//! rooted at the opponent-to-move position, simulated while they "think", and
//! re-rooted through their action exactly as `MctsEngine::ponder` does. When our
//! turn arrives it holds an inherited visit count, and the harness asks three
//! questions of that moment:
//!
//! * **would the early-stop rule fire immediately?** `virus_proto::clock::verdict`
//!   is the deployed rule, called here on the tree's real visit counts and a
//!   representative first sample;
//! * **does the answer change if it does not fire?** the warm tree is searched
//!   for the action's full simulation allowance and the argmax compared;
//! * **does the warm answer match a cold search of the same position** under the
//!   same allowance?
//!
//! A tree whose inherited lead is large enough to survive the whole allowance
//! answers the first question "yes" and the second "no" — which is hypothesis 3
//! of the bead, stated as an experiment rather than an argument.
//!
//! The state-hash assertion is checked on every re-root too, so a run that
//! prints `mismatches=0` is direct evidence against hypothesis 2.

use std::path::PathBuf;
use std::time::Duration;

use virus_core::{Action, Player, State};
use virus_mcts::{Config, MctsSearcher, PolicyValueNet, ValueSource};
use virus_proto::clock::{verdict, MoveAllocation, RootProgress, StopPolicy, Verdict};

/// SplitMix64's finalizer, `virus_arena::rng::mix64`.
fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

const GOLDEN_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Rng {
        Rng { state: seed }
    }

    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(GOLDEN_GAMMA);
        mix64(self.state)
    }

    fn below(&mut self, bound: usize) -> usize {
        if bound == 0 {
            0
        } else {
            (self.next() % bound as u64) as usize
        }
    }
}

struct Options {
    artifact: PathBuf,
    games: u64,
    seed: u64,
    /// Simulations the "opponent is thinking" phase adds per opponent action.
    ponder_sims: u32,
    /// Simulations one of our actions may add on top of what it inherited.
    answer_sims: u32,
    /// Simulations the opponent's own (cold) search runs per action.
    opponent_sims: u32,
    /// Simulations for the deep reference search, or `0` to skip it.
    deep_sims: u32,
    /// Plies of random opening play, so two deterministic engines do not
    /// replay one game forever.
    random_plies: u32,
    turn_cap: u32,
}

fn main() {
    let options = parse();
    let net = PolicyValueNet::load(&options.artifact).expect("the champion artifact loads");
    let config = Config {
        value_source: ValueSource::Net,
        ..Config::play()
    };

    let mut tally = Tally::default();
    for game in 0..options.games {
        play(
            &net,
            config,
            &options,
            mix64(options.seed ^ (game + 1)),
            &mut tally,
        );
        eprintln!("game {} done: {tally}", game + 1);
    }
    println!("{tally}");
    println!("{}", tally.verdict_line());
}

#[derive(Default)]
struct Tally {
    /// Our actions observed.
    actions: u64,
    /// Actions whose tree carried inherited visits in.
    warm: u64,
    /// Sum of inherited visits over warm actions, for the mean.
    inherited_total: u64,
    /// Warm actions where the deployed early-stop rule fires on the first
    /// sample of the action — i.e. the action is over before it began.
    early_stop_immediately: u64,
    /// Warm actions where searching the full allowance moved the argmax.
    answer_moved_with_more_search: u64,
    /// The intersection that matters: actions the rule stopped on the first
    /// sample *and* whose answer the full allowance would have changed. Every
    /// one of these is a move the deployment played that its own search, given
    /// the time it had already been allocated, would have rejected.
    stopped_early_and_wrong: u64,
    /// Warm actions where the warm answer differed from a cold search's.
    warm_differs_from_cold: u64,
    /// Warm actions where the immediately-stopped answer differed from a cold
    /// search's — the move the deployment would actually have played.
    stopped_differs_from_cold: u64,
    /// `PlaceNeutrals` chosen, warm-immediate vs cold vs the deep reference.
    neutrals_warm: u64,
    neutrals_cold: u64,
    neutrals_deep: u64,
    /// Warm actions compared against a much deeper cold search of the same
    /// position (`--deep-sims`), which is the closest thing to ground truth an
    /// offline harness can offer. Agreement is a *diagnostic*, never a strength
    /// claim (ARCHITECTURE.md invariant 7) — but if the answer the deployment
    /// plays agrees with the deep search *less often* than a plain cold search
    /// does, the reused tree is not simply "a deeper search".
    deep_comparisons: u64,
    stopped_matches_deep: u64,
    full_matches_deep: u64,
    cold_matches_deep: u64,
    /// Re-roots whose tree root disagreed with the position it should hold.
    mismatches: u64,
    /// Re-roots attempted, and the ones the tree could not serve.
    reroots: u64,
    reroot_misses: u64,
}

impl std::fmt::Display for Tally {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mean = self.inherited_total.checked_div(self.warm).unwrap_or(0);
        write!(
            f,
            "actions={} warm={} mean_inherited={mean} early_stop_immediately={} \
             answer_moved_with_more_search={} stopped_early_and_wrong={} \
             warm_differs_from_cold={} \
             stopped_differs_from_cold={} neutrals(warm/cold/deep)={}/{}/{} \
             deep_agreement(stopped/full/cold)={}/{}/{} of {} \
             reroots={} misses={} mismatches={}",
            self.actions,
            self.warm,
            self.early_stop_immediately,
            self.answer_moved_with_more_search,
            self.stopped_early_and_wrong,
            self.warm_differs_from_cold,
            self.stopped_differs_from_cold,
            self.neutrals_warm,
            self.neutrals_cold,
            self.neutrals_deep,
            self.stopped_matches_deep,
            self.full_matches_deep,
            self.cold_matches_deep,
            self.deep_comparisons,
            self.reroots,
            self.reroot_misses,
            self.mismatches,
        )
    }
}

impl Tally {
    fn verdict_line(&self) -> String {
        if self.warm == 0 {
            return "no warm actions were observed; the run proved nothing".to_owned();
        }
        let pct = |n: u64| 100.0 * n as f64 / self.warm as f64;
        let of_stopped = if self.early_stop_immediately == 0 {
            0.0
        } else {
            100.0 * self.stopped_early_and_wrong as f64 / self.early_stop_immediately as f64
        };
        format!(
            "of {} warm actions: {:.1}% stopped on the first sample, {:.1}% would have \
             changed their answer given the full allowance, {:.1}% of the played answers \
             differ from a cold search of the same position. Of the actions the rule \
             stopped on the first sample, {of_stopped:.1}% would have answered differently \
             had they been allowed to spend the budget they already had.",
            self.warm,
            pct(self.early_stop_immediately),
            pct(self.answer_moved_with_more_search),
            pct(self.stopped_differs_from_cold),
        )
    }
}

/// One game, with seat 2 playing the part of the pondering deployment.
fn play(net: &PolicyValueNet, config: Config, options: &Options, seed: u64, tally: &mut Tally) {
    const US: Player = 2;
    let mut rng = Rng::new(seed);
    let mut state = State::new(12, 12, 2).expect("a legal opening position");
    // `(root position, tree)`, exactly the pair `MctsEngine::ponder` keeps.
    let mut tree: Option<(State, MctsSearcher<'_>)> = None;
    let mut ply = 0u32;
    let mut turns = 0u32;

    while !state.game_over() && turns < options.turn_cap {
        if state.moves_left() == 3 {
            turns += 1;
        }
        let random = ply < options.random_plies;
        ply += 1;

        if state.current_player() != US {
            // The opponent is thinking; so are we, on their position.
            ensure_tree(&mut tree, &state, config, net, options, tally);
            if let Some((_, searcher)) = tree.as_mut() {
                searcher.run_sims(options.ponder_sims);
            }
            let action = if random {
                let legal = state.legal_actions();
                legal[rng.below(legal.len())]
            } else {
                let mut cold = MctsSearcher::new(state.clone(), config, Some(net));
                cold.run_sims(options.opponent_sims);
                cold.best_action().expect("a legal action")
            };
            state = state.apply(action).expect("a legal action applies");
            continue;
        }

        // Our turn: re-root onto it and answer.
        ensure_tree(&mut tree, &state, config, net, options, tally);
        let Some((_, searcher)) = tree.as_mut() else {
            break;
        };
        let inherited = searcher.root_visit_total();
        tally.actions += 1;

        let action = if inherited > 0 {
            tally.warm += 1;
            tally.inherited_total += inherited;
            let stopped = searcher.best_action().expect("a legal action");
            let fired = fires_immediately(searcher, options);
            if fired {
                tally.early_stop_immediately += 1;
            }
            searcher.run_sims(options.answer_sims);
            let full = searcher.best_action().expect("a legal action");
            if full != stopped {
                tally.answer_moved_with_more_search += 1;
                if fired {
                    tally.stopped_early_and_wrong += 1;
                }
            }
            let mut cold = MctsSearcher::new(state.clone(), config, Some(net));
            cold.run_sims(options.answer_sims);
            let cold = cold.best_action().expect("a legal action");
            if full != cold {
                tally.warm_differs_from_cold += 1;
            }
            if stopped != cold {
                tally.stopped_differs_from_cold += 1;
            }
            if matches!(stopped, Action::PlaceNeutrals { .. }) {
                tally.neutrals_warm += 1;
            }
            if matches!(cold, Action::PlaceNeutrals { .. }) {
                tally.neutrals_cold += 1;
            }
            if options.deep_sims > 0 {
                let mut deep = MctsSearcher::new(state.clone(), config, Some(net));
                deep.run_sims(options.deep_sims);
                let deep = deep.best_action().expect("a legal action");
                tally.deep_comparisons += 1;
                tally.stopped_matches_deep += u64::from(stopped == deep);
                tally.full_matches_deep += u64::from(full == deep);
                tally.cold_matches_deep += u64::from(cold == deep);
                if matches!(deep, Action::PlaceNeutrals { .. }) {
                    tally.neutrals_deep += 1;
                }
            }
            // The deployment plays the answer the stop rules produced. With the
            // rule firing on the first sample that is `stopped`; the game must
            // follow the ponder-on trajectory or the later positions are not
            // the ones the deployment would have reached.
            stopped
        } else {
            searcher.run_sims(options.answer_sims);
            searcher.best_action().expect("a legal action")
        };
        state = state.apply(action).expect("a legal action applies");
    }
}

/// Re-roots the tree onto `state`, or rebuilds it, tallying both.
fn ensure_tree<'net>(
    tree: &mut Option<(State, MctsSearcher<'net>)>,
    state: &State,
    config: Config,
    net: &'net PolicyValueNet,
    _options: &Options,
    tally: &mut Tally,
) {
    if let Some((root, searcher)) = tree.as_mut() {
        tally.reroots += 1;
        if reroot(root, searcher, state) {
            // The permanent assertion, checked here too so an offline run is
            // evidence for or against hypothesis 2 on its own.
            if searcher.root_state() != state {
                tally.mismatches += 1;
            } else {
                return;
            }
        }
        tally.reroot_misses += 1;
    }
    *tree = Some((
        state.clone(),
        MctsSearcher::new(state.clone(), config, Some(net)),
    ));
}

/// `vsbot::mcts::reroot`, one ply at a time — the harness always steps a single
/// action, so the multi-ply path search is not needed here.
fn reroot(root: &mut State, searcher: &mut MctsSearcher<'_>, target: &State) -> bool {
    if root == target {
        return true;
    }
    for action in root.legal_actions() {
        let Ok(next) = root.apply(action) else {
            continue;
        };
        if next.hash() == target.hash() && next == *target {
            if !searcher.rebase(action) {
                return false;
            }
            *root = next;
            return true;
        }
    }
    false
}

/// Whether the deployed early-stop rule fires on this action's *first* sample.
///
/// The first sample is one [`CANCEL_POLL_SLICE`](vsbot) worth of simulations
/// against the tree's cumulative visit counts, which is exactly the pairing
/// `vsbot::mcts::drive` hands to [`verdict`]. A representative slice is one
/// twenty-fifth of the action's allowance — 20 ms of a 500 ms action — so the
/// rate estimate here is the generous one: a real first slice is contaminated by
/// the re-root's own cost and estimates the rate *lower*, which makes the rule
/// fire more often, not less.
fn fires_immediately(searcher: &MctsSearcher<'_>, options: &Options) -> bool {
    const SLICE: Duration = Duration::from_millis(20);
    const TARGET: Duration = Duration::from_millis(500);
    let allocation = MoveAllocation {
        target: TARGET,
        ceiling: TARGET.mul_f64(1.5),
    };
    // The allowance, mapped onto the same clock: `answer_sims` simulations fill
    // the target, so one slice is that fraction of them.
    let slice_sims = (u64::from(options.answer_sims) * SLICE.as_millis() as u64)
        .div_ceil(TARGET.as_millis() as u64)
        .max(1);
    let progress = RootProgress::from_visits(searcher.root_visits(), slice_sims, false);
    verdict(progress, SLICE, allocation, &StopPolicy::default()) == Verdict::StopUncatchable
}

fn parse() -> Options {
    let mut options = Options {
        artifact: PathBuf::from("artifacts/mcts_champion.json"),
        games: 4,
        seed: 0x5EED,
        ponder_sims: 4_000,
        answer_sims: 800,
        opponent_sims: 800,
        deep_sims: 0,
        random_plies: 8,
        turn_cap: 200,
    };
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let mut value = || args.next().expect("a value follows every flag");
        match flag.as_str() {
            "--artifact" => options.artifact = PathBuf::from(value()),
            "--games" => options.games = value().parse().expect("a game count"),
            "--seed" => options.seed = value().parse().expect("a seed"),
            "--ponder-sims" => options.ponder_sims = value().parse().expect("a count"),
            "--answer-sims" => options.answer_sims = value().parse().expect("a count"),
            "--opponent-sims" => options.opponent_sims = value().parse().expect("a count"),
            "--deep-sims" => options.deep_sims = value().parse().expect("a count"),
            "--random-plies" => options.random_plies = value().parse().expect("a count"),
            "--turn-cap" => options.turn_cap = value().parse().expect("a count"),
            other => panic!("unknown flag {other}"),
        }
    }
    options
}
