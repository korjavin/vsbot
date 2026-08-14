//! Self-play-mode gauntlet: Gumbel root selection vs PUCT + Dirichlet.
//!
//! ```bash
//! cargo run --release -p virus-mcts --example gumbelgauntlet -- \
//!     --games 100 --sims 192 --a-m 16 --jobs 2
//! ```
//!
//! # Why this is not `batchgauntlet`
//!
//! `batchgauntlet` compares two **play-mode** configurations: no noise, no
//! sampling, argmax visits, and an epsilon-random opening bolted on because two
//! deterministic engines would otherwise replay one game forever. That is the
//! right harness for the DAG and for batching, which change how a search
//! *thinks* and are supposed to be invisible to exploration.
//!
//! Gumbel is not that kind of change. It **is** the exploration: it replaces
//! Dirichlet noise and temperature sampling with a draw of its own, and the
//! acceptance question for S3-T3 is "which arm generates better self-play",
//! not "which arm plays better with exploration switched off". So both seats
//! here run their real self-play configuration —
//! [`Config::self_play_gumbel`] on one side, [`Config::self_play`] (Dirichlet
//! plus temperature-1 sampling for the opening 21 plies) on the other — and
//! play [`MctsSearcher::chosen_action`], exactly as `virus-selfplay` does.
//! Neither seat needs an epsilon-random opening because neither seat is
//! deterministic.
//!
//! Everything else follows the project's gauntlet rules, verbatim from
//! `virus_arena` and `batchgauntlet`:
//!
//! * **Colour-paired games.** Games `2k` and `2k+1` share
//!   `derive_game_seed(seed, game) = mix64(seed ^ GOLDEN * (game/2 + 1))` and
//!   swap seats, so first-mover advantage cancels inside the pair. Game counts
//!   round up to even.
//! * **Per-ply seeds are `virus-selfplay`'s**, `mix64(game_seed ^ (ply + 1))`,
//!   so a game here is drawn from the same stream a real generation would use.
//! * **Turn cap is a draw** — a game neither side could finish is not evidence
//!   either way. (`virus-selfplay` scores it by territory instead, because that
//!   is a *training label*, not a comparison.)
//! * **Pooled score `(W + 0.5·D)/N`** plus a Wilson 95% interval.
//!
//! # Fixed simulations, always
//!
//! There is no `--millis`. Sequential halving plans its phases against a
//! simulation budget, so a wall-clock arm would be halving on boundaries that
//! move with the box's load — the comparison would not even be well defined,
//! let alone reproducible. Fixed sims is also load-tolerant: contention makes
//! this slower, never different, so a run on a busy box is the same run.

use std::time::Instant;

use virus_core::{Action, Player, State, ACTIONS_PER_TURN};
use virus_mcts::{Config, MctsSearcher, PolicyValueNet, ValueSource};

// ---------------------------------------------------------------- pairing rng

/// SplitMix64's finalizer, `virus_arena::rng::mix64`.
fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

const GOLDEN_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

/// `virus_arena::rng::derive_game_seed`: both colours of a pair share it.
fn derive_game_seed(seed: u64, game: u64) -> u64 {
    mix64(seed ^ GOLDEN_GAMMA.wrapping_mul((game / 2) + 1))
}

// ---------------------------------------------------------------- the sides

#[derive(Clone, Copy, Debug)]
struct Arm {
    /// `None` is PUCT + Dirichlet; `Some(m)` is Gumbel top-`m`.
    gumbel: Option<u16>,
    batch: u16,
    dag: bool,
}

impl Arm {
    fn label(&self) -> String {
        match self.gumbel {
            Some(m) => format!("Gumbel top-{m} + sequential halving"),
            None => "PUCT + Dirichlet (+ temperature-1 opening)".to_owned(),
        }
    }

    /// The searcher config for one ply, seeded exactly as `virus-selfplay`
    /// seeds it.
    fn config(&self, game_seed: u64, ply: u32, sims: u32) -> Config {
        let seed = mix64(game_seed ^ u64::from(ply + 1));
        let template = match self.gumbel {
            Some(m) => Config::self_play_gumbel(seed, sims, m),
            None => Config::self_play(seed, ply),
        };
        Config {
            value_source: ValueSource::Net,
            batch_size: self.batch,
            dag: self.dag,
            ..template
        }
    }

    fn choose(
        &self,
        state: &State,
        net: &PolicyValueNet,
        game_seed: u64,
        ply: u32,
        sims: u32,
    ) -> Option<Action> {
        let mut searcher =
            MctsSearcher::new(state.clone(), self.config(game_seed, ply, sims), Some(net));
        searcher.run_sims(sims);
        searcher.chosen_action()
    }
}

// ---------------------------------------------------------------- one game

const MAX_TURNS: u32 = 100;

#[derive(Clone, Copy, Debug)]
struct GameOutcome {
    winner: Player,
    a_is_p1: bool,
    capped: bool,
}

impl GameOutcome {
    /// `1` win, `0` loss, `0.5` draw, from side A's chair.
    fn score_for_a(&self) -> f64 {
        if self.winner == 0 {
            return 0.5;
        }
        if (self.winner == 1) == self.a_is_p1 {
            1.0
        } else {
            0.0
        }
    }
}

fn play_game(
    a: Arm,
    b: Arm,
    net: &PolicyValueNet,
    seed: u64,
    index: u32,
    sims: u32,
) -> GameOutcome {
    let a_is_p1 = index % 2 == 0;
    let game_seed = derive_game_seed(seed, u64::from(index));
    let (p1, p2) = if a_is_p1 { (a, b) } else { (b, a) };

    let mut state = State::new(12, 12, 2).expect("12x12 two-player start");
    let ply_ceiling = MAX_TURNS * u32::from(ACTIONS_PER_TURN);
    let mut ply = 0u32;
    let mut capped = true;

    while ply < ply_ceiling {
        if state.game_over() {
            capped = false;
            break;
        }
        let arm = if state.current_player() == 1 { p1 } else { p2 };
        let Some(action) = arm.choose(&state, net, game_seed, ply, sims) else {
            capped = false;
            break;
        };
        let Ok(next) = state.apply(action) else {
            capped = false;
            break;
        };
        state = next;
        ply += 1;
    }

    GameOutcome {
        winner: if capped { 0 } else { state.outcome_winner() },
        a_is_p1,
        capped,
    }
}

// ---------------------------------------------------------------- reporting

/// `virus_arena::stats::wilson95`, in percent.
fn wilson95(wins: f64, games: u32) -> (f64, f64) {
    const Z: f64 = 1.959_963_984_540_054;
    if games == 0 {
        return (0.0, 0.0);
    }
    let n = f64::from(games);
    let p = wins / n;
    let denominator = 1.0 + Z * Z / n;
    let center = (p + Z * Z / (2.0 * n)) / denominator;
    let margin = Z * ((p * (1.0 - p) + Z * Z / (4.0 * n)) / n).sqrt() / denominator;
    (100.0 * (center - margin), 100.0 * (center + margin))
}

// ---------------------------------------------------------------- cli

struct Args {
    games: u32,
    sims: u32,
    seed: u64,
    jobs: usize,
    a: Arm,
    b: Arm,
    net: String,
}

fn parse_bool(flag: &str, value: &str) -> bool {
    match value {
        "1" | "true" | "on" | "yes" => true,
        "0" | "false" | "off" | "no" => false,
        other => panic!("{flag} wants a boolean, got {other}"),
    }
}

fn parse() -> Args {
    let mut args = Args {
        games: 100,
        sims: 192,
        seed: 20_260_814,
        jobs: 2,
        a: Arm {
            gumbel: Some(virus_mcts::DEFAULT_GUMBEL_M),
            batch: virus_mcts::DEFAULT_BATCH_SIZE,
            dag: virus_mcts::DEFAULT_DAG,
        },
        b: Arm {
            gumbel: None,
            batch: virus_mcts::DEFAULT_BATCH_SIZE,
            dag: virus_mcts::DEFAULT_DAG,
        },
        net: "artifacts/mcts_champion.json".to_owned(),
    };
    let mut raw = std::env::args().skip(1);
    while let Some(flag) = raw.next() {
        let mut value = || raw.next().unwrap_or_else(|| panic!("{flag} needs a value"));
        match flag.as_str() {
            "--games" => args.games = value().parse().expect("--games"),
            "--sims" => args.sims = value().parse().expect("--sims"),
            "--seed" => args.seed = value().parse().expect("--seed"),
            "--jobs" => args.jobs = value().parse().expect("--jobs"),
            "--a-m" => args.a.gumbel = Some(value().parse().expect("--a-m")),
            "--b-m" => args.b.gumbel = Some(value().parse().expect("--b-m")),
            "--a-gumbel" => {
                if !parse_bool("--a-gumbel", &value()) {
                    args.a.gumbel = None;
                }
            }
            "--b-gumbel" => {
                if parse_bool("--b-gumbel", &value()) {
                    args.b.gumbel = Some(virus_mcts::DEFAULT_GUMBEL_M);
                } else {
                    args.b.gumbel = None;
                }
            }
            "--batch" => {
                let batch = value().parse().expect("--batch");
                args.a.batch = batch;
                args.b.batch = batch;
            }
            "--dag" => {
                let dag = parse_bool("--dag", &value());
                args.a.dag = dag;
                args.b.dag = dag;
            }
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
    let games = args.games.div_ceil(2) * 2;

    println!(
        "A = {}\nB = {}\n{} games, {} sims/action (fixed — load-tolerant), seed {}, \
         {} concurrent game(s), batch {}, DAG {}, net {}\n",
        args.a.label(),
        args.b.label(),
        games,
        args.sims,
        args.seed,
        args.jobs,
        args.a.batch,
        if args.a.dag { "on" } else { "off" },
        args.net
    );

    let next = std::sync::atomic::AtomicU32::new(0);
    let started = Instant::now();
    let outcomes: Vec<GameOutcome> = std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for _ in 0..args.jobs.max(1) {
            let next = &next;
            let net = &net;
            let args = &args;
            handles.push(scope.spawn(move || {
                let mut mine = Vec::new();
                loop {
                    let index = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if index >= games {
                        return mine;
                    }
                    mine.push(play_game(args.a, args.b, net, args.seed, index, args.sims));
                }
            }));
        }
        handles
            .into_iter()
            .flat_map(|handle| handle.join().expect("a gauntlet worker panicked"))
            .collect()
    });

    let (mut wins, mut losses, mut draws, mut capped) = (0u32, 0u32, 0u32, 0u32);
    // Seat split, because a stochastic-both-sides gauntlet is exactly where a
    // broken colour pairing hides: a big gap between the two chairs means the
    // pairing is not cancelling first-mover advantage and the pooled number is
    // not measuring the arms.
    let mut as_p1 = (0u32, 0u32);
    for game in &outcomes {
        let score = game.score_for_a();
        match score {
            s if s > 0.75 => wins += 1,
            s if s < 0.25 => losses += 1,
            _ => draws += 1,
        }
        capped += u32::from(game.capped);
        if game.a_is_p1 {
            as_p1.0 += u32::from(score > 0.75);
            as_p1.1 += 1;
        }
    }
    let n = wins + losses + draws;
    let pooled = (f64::from(wins) + 0.5 * f64::from(draws)) / f64::from(n.max(1));
    let (low, high) = wilson95(f64::from(wins), n);

    println!("A: {wins}W {losses}L {draws}D over {n} games ({capped} hit the turn cap)");
    println!(
        "  win rate {:.1}% (draws not half-wins)  wilson95 [{low:.1}%, {high:.1}%]",
        100.0 * f64::from(wins) / f64::from(n.max(1))
    );
    println!("  pooled score {pooled:.4} (W+0.5D)/N   S3-T3 bar is >= 0.5500");
    println!(
        "  seat split: A won {}/{} as player 1, {}/{} as player 2",
        as_p1.0,
        as_p1.1,
        wins - as_p1.0,
        n - as_p1.1
    );
    println!("  elapsed {:.1}s", started.elapsed().as_secs_f64());
}
