//! Fixed-time self-gauntlet between two MCTS configurations of *this* crate.
//!
//! ```bash
//! cargo run --release -p virus-mcts --example batchgauntlet -- \
//!     --games 100 --millis 200 --a-batch 16 --b-batch 1
//! ```
//!
//! # Why this exists and is not in `virus-arena`
//!
//! The arena is the project's gauntlet harness and stays the authority for
//! engine-vs-engine work. It builds its MCTS side from `Config::play()` though,
//! so both of its seats necessarily share one `batch_size` — which makes the
//! one comparison S3-T1 has to make (batched searcher vs the serial one, same
//! net, same wall clock) unreachable from its CLI. Rather than reach across
//! into another crate's file, this example runs the pairing rules itself:
//!
//! * **Colour-paired games.** Games `2k` and `2k+1` are the same opening from
//!   both chairs, so first-mover advantage cancels inside the pair. Game counts
//!   are rounded up to even.
//! * **Seeded, diverse openings.** `derive(seed, game) = mix64(seed ^ GOLDEN *
//!   (game/2 + 1))`, verbatim from `virus_arena::rng`, with 15% uniformly
//!   random actions over the first 8 turns. Two deterministic engines replay
//!   one game forever otherwise.
//! * **Turn cap is a draw**, not a territory decision, matching the arena.
//! * **Pooled score `(W + 0.5·D)/N`** — the Gate A quantity — plus a Wilson 95%
//!   interval on the headline win rate, the same formula `virus_arena::stats`
//!   uses.
//!
//! # Reading the result honestly
//!
//! A wall-clock gauntlet is not reproducible, and `--jobs` above 1 puts games
//! in competition for cores. Both seats of a game are affected identically and
//! the pairing cancels seat effects, but a run taken on a loaded box measures
//! that box. Take the lock and say what else was running.

use std::time::{Duration, Instant};

use virus_core::{Action, Player, State, ACTIONS_PER_TURN};
use virus_mcts::{Config, MctsSearcher, ParallelMcts, PolicyValueNet, ValueSource};

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

struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Rng {
        Rng { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(GOLDEN_GAMMA);
        mix64(self.state)
    }
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
    fn below(&mut self, len: usize) -> Option<usize> {
        if len == 0 {
            return None;
        }
        Some(((u128::from(self.next_u64()) * len as u128) >> 64) as usize)
    }
}

// ---------------------------------------------------------------- the sides

#[derive(Clone, Copy, Debug)]
struct Arm {
    batch: u16,
    threads: usize,
}

impl Arm {
    fn label(&self) -> String {
        match (self.batch, self.threads) {
            (b, 1) if b <= 1 => "serial".to_owned(),
            (b, 1) => format!("batched B={b}"),
            (b, t) => format!("batched B={b} x{t} threads"),
        }
    }

    fn config(&self) -> Config {
        Config {
            value_source: ValueSource::Net,
            batch_size: self.batch,
            threads: self.threads,
            ..Config::play()
        }
    }

    /// One move under a wall-clock budget, plus the simulations it bought.
    fn choose(&self, state: &State, net: &PolicyValueNet, millis: u64) -> (Option<Action>, u64) {
        let deadline = Instant::now() + Duration::from_millis(millis);
        if self.threads > 1 {
            let mut searcher = ParallelMcts::new(state.clone(), self.config(), Some(net));
            searcher.run_until_deadline(deadline);
            (searcher.best_action(), searcher.sims_run())
        } else {
            let mut searcher = MctsSearcher::new(state.clone(), self.config(), Some(net));
            searcher.run_until_deadline(deadline);
            (searcher.best_action(), searcher.sims_run())
        }
    }
}

// ---------------------------------------------------------------- one game

const MAX_TURNS: u32 = 100;
const EPSILON: f64 = 0.15;
const EXPLORE_TURNS: u32 = 8;

#[derive(Clone, Copy, Debug)]
struct GameOutcome {
    winner: Player,
    a_is_p1: bool,
    capped: bool,
    sims_a: u64,
    sims_b: u64,
    moves_a: u64,
    moves_b: u64,
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
    millis: u64,
) -> GameOutcome {
    let a_is_p1 = index % 2 == 0;
    let mut rng = Rng::new(derive_game_seed(seed, u64::from(index)));
    let (p1, p2) = if a_is_p1 { (a, b) } else { (b, a) };

    let mut state = State::new(12, 12, 2).expect("12x12 two-player start");
    let ply_ceiling = MAX_TURNS * u32::from(ACTIONS_PER_TURN);
    let mut plies = 0u32;
    let mut turns = 0u32;
    let mut sims = [0u64; 2];
    let mut moves = [0u64; 2];
    let mut capped = true;

    while turns < MAX_TURNS && plies < ply_ceiling {
        if state.game_over() {
            capped = false;
            break;
        }
        let legal = state.legal_actions();
        if legal.is_empty() {
            capped = false;
            break;
        }
        let mover = state.current_player();
        let arm = if mover == 1 { p1 } else { p2 };
        // Search on every ply, including the ones the coin overrides: both
        // colours of a pair must see the same sequence of search calls.
        let (searched, ran) = arm.choose(&state, net, millis);
        let a_moved = (mover == 1) == a_is_p1;
        sims[usize::from(!a_moved)] += ran;
        moves[usize::from(!a_moved)] += 1;

        let chosen = if turns < EXPLORE_TURNS && rng.next_f64() < EPSILON {
            legal[rng.below(legal.len()).expect("legal is non-empty")]
        } else {
            match searched {
                Some(action) => action,
                None => {
                    capped = false;
                    break;
                }
            }
        };
        let action = if legal.contains(&chosen) {
            chosen
        } else {
            legal[0]
        };
        let Ok(next) = state.apply(action) else {
            capped = false;
            break;
        };
        if next.current_player() != mover {
            turns += 1;
        }
        state = next;
        plies += 1;
    }

    GameOutcome {
        winner: if capped { 0 } else { state.outcome_winner() },
        a_is_p1,
        capped,
        sims_a: sims[0],
        sims_b: sims[1],
        moves_a: moves[0],
        moves_b: moves[1],
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
    millis: u64,
    seed: u64,
    jobs: usize,
    a: Arm,
    b: Arm,
    net: String,
}

fn parse() -> Args {
    let mut args = Args {
        games: 100,
        millis: 200,
        seed: 20_260_813,
        jobs: 2,
        a: Arm {
            batch: virus_mcts::DEFAULT_BATCH_SIZE,
            threads: 1,
        },
        b: Arm {
            batch: 1,
            threads: 1,
        },
        net: "artifacts/mcts_champion.json".to_owned(),
    };
    let mut raw = std::env::args().skip(1);
    while let Some(flag) = raw.next() {
        let mut value = || raw.next().unwrap_or_else(|| panic!("{flag} needs a value"));
        match flag.as_str() {
            "--games" => args.games = value().parse().expect("--games"),
            "--millis" => args.millis = value().parse().expect("--millis"),
            "--seed" => args.seed = value().parse().expect("--seed"),
            "--jobs" => args.jobs = value().parse().expect("--jobs"),
            "--a-batch" => args.a.batch = value().parse().expect("--a-batch"),
            "--b-batch" => args.b.batch = value().parse().expect("--b-batch"),
            "--a-threads" => args.a.threads = value().parse().expect("--a-threads"),
            "--b-threads" => args.b.threads = value().parse().expect("--b-threads"),
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
        "A = {}   B = {}\n{} games, {} ms/action, seed {}, {} concurrent game(s), net {}\n",
        args.a.label(),
        args.b.label(),
        games,
        args.millis,
        args.seed,
        args.jobs,
        args.net
    );

    let next = std::sync::atomic::AtomicU32::new(0);
    let started = Instant::now();
    let mut outcomes: Vec<GameOutcome> = std::thread::scope(|scope| {
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
                    mine.push(play_game(
                        args.a,
                        args.b,
                        net,
                        args.seed,
                        index,
                        args.millis,
                    ));
                }
            }));
        }
        handles
            .into_iter()
            .flat_map(|handle| handle.join().expect("a gauntlet worker panicked"))
            .collect()
    });
    outcomes.sort_by(|x, y| x.score_for_a().total_cmp(&y.score_for_a()));

    let (mut wins, mut losses, mut draws, mut capped) = (0u32, 0u32, 0u32, 0u32);
    let (mut sims_a, mut sims_b, mut moves_a, mut moves_b) = (0u64, 0u64, 0u64, 0u64);
    for game in &outcomes {
        match game.score_for_a() {
            s if s > 0.75 => wins += 1,
            s if s < 0.25 => losses += 1,
            _ => draws += 1,
        }
        capped += u32::from(game.capped);
        sims_a += game.sims_a;
        sims_b += game.sims_b;
        moves_a += game.moves_a;
        moves_b += game.moves_b;
    }
    let n = wins + losses + draws;
    let pooled = (f64::from(wins) + 0.5 * f64::from(draws)) / f64::from(n.max(1));
    let (low, high) = wilson95(f64::from(wins), n);

    println!("A: {wins}W {losses}L {draws}D over {n} games ({capped} hit the turn cap)");
    println!(
        "  win rate {:.1}% (draws not half-wins)  wilson95 [{low:.1}%, {high:.1}%]",
        100.0 * f64::from(wins) / f64::from(n.max(1))
    );
    println!("  pooled score {pooled:.4} (W+0.5D)/N   gate A needs >= 0.5500");
    println!(
        "  sims/action: A {:.0}, B {:.0}   ({:.2}x)",
        sims_a as f64 / moves_a.max(1) as f64,
        sims_b as f64 / moves_b.max(1) as f64,
        (sims_a as f64 / moves_a.max(1) as f64) / (sims_b as f64 / moves_b.max(1) as f64).max(1e-9)
    );
    println!("  elapsed {:.1}s", started.elapsed().as_secs_f64());
}
