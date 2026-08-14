//! Policy-target quality: what the two arms actually hand the trainer.
//!
//! ```bash
//! cargo run --release -p virus-mcts --example gumbelprobe -- \
//!     --games 12 --sims 192 --m 16 --every 5
//! ```
//!
//! S3-T3's third acceptance item. A gauntlet answers "which arm plays better";
//! this answers the other half, **"which arm writes a better training row"** —
//! and it has to be asked separately, because the row is what a generation is
//! for and the two questions have gone in opposite directions before.
//!
//! # What is measured
//!
//! Positions are sampled from real PUCT self-play games (every `--every`-th
//! ply), so the distribution is the one a generation would actually see, not a
//! set of hand-built midgames. On each sampled position both arms search at the
//! **same** simulation budget, and four targets are compared over the root's
//! full legal action set:
//!
//! | target | what it is |
//! |---|---|
//! | `gumbel pi'` | the completed-Q improved policy of the Gumbel search |
//! | `gumbel pv` | that same search's raw visit counts, normalised |
//! | `puct pv` | the PUCT+Dirichlet search's raw visit counts, normalised — **today's target** |
//! | `puct pi'` | the completed-Q policy of the PUCT search, for reference |
//!
//! Reported per target: mean **entropy** in nats (how much of the action space
//! the target actually teaches), mean **support** (non-zero entries, out of the
//! legal action count), and **top-1 agreement** against the incumbent `puct pv`
//! and against `gumbel pv`.
//!
//! Entropy is the number to read first. A policy target that has collapsed onto
//! one action teaches the net to imitate a single search decision and nothing
//! about the shape of the position; at the other extreme a target that has not
//! concentrated at all teaches nothing but the prior. `virus-selfplay`'s `pv`
//! column can carry exactly one of these, which is why the choice is a
//! documented decision there rather than a default.

use std::time::Instant;

use virus_core::{State, ACTIONS_PER_TURN};
use virus_mcts::{Config, MctsSearcher, PolicyValueNet, ValueSource};

fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

const GOLDEN_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

/// `virus_selfplay::derive_game_seed`.
fn derive_game_seed(seed: u64, game: u64) -> u64 {
    mix64(seed ^ GOLDEN_GAMMA.wrapping_mul(game.wrapping_add(1)))
}

// ---------------------------------------------------------------- statistics

/// Shannon entropy in nats. `0` for a one-hot target, `ln(k)` for a uniform one
/// over `k` actions.
fn entropy(policy: &[f64]) -> f64 {
    -policy
        .iter()
        .filter(|p| **p > 0.0)
        .map(|p| p * p.ln())
        .sum::<f64>()
}

fn argmax(policy: &[f64]) -> usize {
    (0..policy.len())
        .max_by(|x, y| policy[*x].total_cmp(&policy[*y]))
        .unwrap_or(0)
}

fn normalise<T: Copy + Into<f64>>(counts: &[T]) -> Vec<f64> {
    let total: f64 = counts.iter().map(|c| (*c).into()).sum();
    if total <= 0.0 {
        let uniform = 1.0 / counts.len().max(1) as f64;
        return vec![uniform; counts.len()];
    }
    counts.iter().map(|c| (*c).into() / total).collect()
}

/// One target's running aggregate over the sampled positions.
#[derive(Clone, Debug, Default)]
struct Aggregate {
    label: &'static str,
    entropy: f64,
    support: f64,
    /// Positions where this target's argmax matches the incumbent `puct pv`.
    agree_puct_pv: u32,
    /// Positions where this target's argmax matches `gumbel pv`.
    agree_gumbel_pv: u32,
    samples: u32,
}

impl Aggregate {
    fn new(label: &'static str) -> Aggregate {
        Aggregate {
            label,
            ..Aggregate::default()
        }
    }

    fn observe(&mut self, policy: &[f64], puct_pv_top: usize, gumbel_pv_top: usize) {
        let top = argmax(policy);
        self.entropy += entropy(policy);
        self.support += policy.iter().filter(|p| **p > 0.0).count() as f64;
        self.agree_puct_pv += u32::from(top == puct_pv_top);
        self.agree_gumbel_pv += u32::from(top == gumbel_pv_top);
        self.samples += 1;
    }

    fn report(&self, actions: f64) {
        let n = f64::from(self.samples.max(1));
        println!(
            "  {:<12} entropy {:>6.3} nats ({:>5.1}% of ln(legal))   support {:>5.1}/{:.1}   \
             top-1 = puct pv {:>5.1}%   = gumbel pv {:>5.1}%",
            self.label,
            self.entropy / n,
            100.0 * (self.entropy / n) / actions.ln().max(1e-9),
            self.support / n,
            actions,
            100.0 * f64::from(self.agree_puct_pv) / n,
            100.0 * f64::from(self.agree_gumbel_pv) / n,
        );
    }
}

// ---------------------------------------------------------------- cli

struct Args {
    games: u64,
    sims: u32,
    m: u16,
    every: u32,
    max_turns: u32,
    seed: u64,
    net: String,
}

fn parse() -> Args {
    let mut args = Args {
        games: 12,
        sims: 192,
        m: virus_mcts::DEFAULT_GUMBEL_M,
        every: 5,
        max_turns: 100,
        seed: 20_260_814,
        net: "artifacts/mcts_champion.json".to_owned(),
    };
    let mut raw = std::env::args().skip(1);
    while let Some(flag) = raw.next() {
        let mut value = || raw.next().unwrap_or_else(|| panic!("{flag} needs a value"));
        match flag.as_str() {
            "--games" => args.games = value().parse().expect("--games"),
            "--sims" => args.sims = value().parse().expect("--sims"),
            "--m" => args.m = value().parse().expect("--m"),
            "--every" => args.every = value().parse().expect("--every"),
            "--max-turns" => args.max_turns = value().parse().expect("--max-turns"),
            "--seed" => args.seed = value().parse().expect("--seed"),
            "--net" => args.net = value(),
            other => panic!("unknown flag {other}"),
        }
    }
    args
}

fn main() {
    let args = parse();
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let net = PolicyValueNet::load(root.join(&args.net)).expect("net loads");

    println!(
        "policy-target probe: {} games, every {}th ply sampled, {} sims/action, \
         Gumbel m={}, seed {}, net {}\n",
        args.games, args.every, args.sims, args.m, args.seed, args.net
    );

    let mut targets = [
        Aggregate::new("gumbel pi'"),
        Aggregate::new("gumbel pv"),
        Aggregate::new("puct pv"),
        Aggregate::new("puct pi'"),
    ];
    let mut actions_total = 0.0f64;
    let mut samples = 0u32;
    let started = Instant::now();

    for game in 0..args.games {
        let game_seed = derive_game_seed(args.seed, game);
        let mut state = State::new(12, 12, 2).expect("12x12 two-player start");
        let ply_ceiling = args.max_turns * u32::from(ACTIONS_PER_TURN);
        let mut ply = 0u32;

        while ply < ply_ceiling && !state.game_over() {
            let seed = mix64(game_seed ^ u64::from(ply + 1));
            let base = Config {
                value_source: ValueSource::Net,
                ..Config::self_play(seed, ply)
            };
            // The PUCT search is both the sampler (it plays the game, so the
            // positions are on-distribution for a real generation) and one of
            // the two arms being measured. Running it once for both jobs is
            // what keeps the probe affordable.
            let mut puct = MctsSearcher::new(state.clone(), base, Some(&net));
            puct.run_sims(args.sims);
            if puct.root_actions().is_empty() {
                break;
            }

            if ply % args.every == 0 && puct.root_actions().len() > 1 {
                let gumbel_config = Config {
                    value_source: ValueSource::Net,
                    ..Config::self_play_gumbel(seed, args.sims, args.m)
                };
                let mut gumbel = MctsSearcher::new(state.clone(), gumbel_config, Some(&net));
                gumbel.run_sims(args.sims);

                let gumbel_pi: Vec<f64> = gumbel
                    .root_improved_policy()
                    .iter()
                    .map(|p| f64::from(*p))
                    .collect();
                let gumbel_pv = normalise(gumbel.root_visits());
                let puct_pv = normalise(puct.root_visits());
                let puct_pi: Vec<f64> = puct
                    .root_improved_policy()
                    .iter()
                    .map(|p| f64::from(*p))
                    .collect();

                let puct_top = argmax(&puct_pv);
                let gumbel_top = argmax(&gumbel_pv);
                for (aggregate, policy) in targets
                    .iter_mut()
                    .zip([&gumbel_pi, &gumbel_pv, &puct_pv, &puct_pi])
                {
                    aggregate.observe(policy, puct_top, gumbel_top);
                }
                actions_total += puct.root_actions().len() as f64;
                samples += 1;
            }

            let Some(action) = puct.chosen_action() else {
                break;
            };
            state = state.apply_generated(action);
            ply += 1;
        }
        eprint!("\rgame {}/{}, {samples} positions", game + 1, args.games);
    }
    eprintln!();

    if samples == 0 {
        println!("no multi-choice positions sampled — raise --games or lower --every");
        return;
    }
    let actions = actions_total / f64::from(samples);
    println!(
        "{samples} positions, mean {actions:.1} legal actions, {:.1}s\n",
        started.elapsed().as_secs_f64()
    );
    for aggregate in &targets {
        aggregate.report(actions);
    }
    println!(
        "\n`puct pv` is the incumbent target (what every generation to date trained on).\n\
         `gumbel pv` is what a GUMBEL_PV=raw run would write; `gumbel pi'` is what\n\
         GUMBEL_PV=improved writes, quantised to integer counts at 1/{} resolution.",
        args.sims
    );
}
