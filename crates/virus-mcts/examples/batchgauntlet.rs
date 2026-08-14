//! Self-gauntlet between two MCTS configurations of *this* crate.
//!
//! ```bash
//! # fixed time — measures throughput work (S3-T1)
//! cargo run --release -p virus-mcts --example batchgauntlet -- \
//!     --games 100 --millis 200 --a-batch 16 --b-batch 1
//!
//! # fixed simulations — measures search-quality work (S3-T2)
//! cargo run --release -p virus-mcts --example batchgauntlet -- \
//!     --games 100 --sims 800 --a-dag 1 --b-dag 0
//! ```
//!
//! # Which budget to use
//!
//! `--millis` asks "which arm is stronger per second", and is the right
//! question for anything that buys throughput. It is also **wall-clock**, so it
//! measures the box as much as the change and is worth nothing on a loaded one.
//!
//! `--sims` asks "which arm is stronger per simulation", and is the right
//! question for anything that changes what a simulation *learns* — which is
//! what a DAG does. It is also **load-tolerant**: contention makes it slower,
//! never different, so the result on a busy box is the same result. That is why
//! S3-T2's acceptance gauntlet is a fixed-sims one.
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

/// What one action's search is allowed to spend.
#[derive(Clone, Copy, Debug)]
enum Budget {
    /// Wall clock. Not reproducible; measures the box too.
    Millis(u64),
    /// Simulations. Reproducible and load-tolerant.
    Sims(u32),
}

#[derive(Clone, Copy, Debug)]
struct Arm {
    batch: u16,
    threads: usize,
    dag: bool,
}

/// What one action's search cost and covered.
#[derive(Clone, Copy, Debug, Default)]
struct Spent {
    sims: u64,
    /// Nodes in the arena when the search stopped — expansions, hence net
    /// forwards.
    nodes: u64,
    /// Child links the DAG resolved onto an existing node. Always 0 for
    /// `ParallelMcts`, which ignores the flag.
    merges: u64,
}

impl Arm {
    fn label(&self) -> String {
        let base = match (self.batch, self.threads) {
            (b, 1) if b <= 1 => "serial".to_owned(),
            (b, 1) => format!("batched B={b}"),
            (b, t) => format!("batched B={b} x{t} threads"),
        };
        format!("{base}, DAG {}", if self.dag { "on" } else { "off" })
    }

    fn config(&self) -> Config {
        Config {
            value_source: ValueSource::Net,
            batch_size: self.batch,
            threads: self.threads,
            dag: self.dag,
            ..Config::play()
        }
    }

    /// One move under `budget`, plus what the search spent getting there.
    fn choose(
        &self,
        state: &State,
        net: &PolicyValueNet,
        budget: Budget,
    ) -> (Option<Action>, Spent) {
        if self.threads > 1 {
            // The shared-tree engine ignores `Config::dag` and exposes no node
            // count; the `--sims`/`--a-dag` comparison is a serial one.
            let mut searcher = ParallelMcts::new(state.clone(), self.config(), Some(net));
            match budget {
                Budget::Millis(ms) => {
                    searcher.run_until_deadline(Instant::now() + Duration::from_millis(ms));
                }
                Budget::Sims(n) => searcher.run_sims(n),
            }
            let spent = Spent {
                sims: searcher.sims_run(),
                ..Spent::default()
            };
            (searcher.best_action(), spent)
        } else {
            let mut searcher = MctsSearcher::new(state.clone(), self.config(), Some(net));
            match budget {
                Budget::Millis(ms) => {
                    searcher.run_until_deadline(Instant::now() + Duration::from_millis(ms));
                }
                Budget::Sims(n) => searcher.run_sims(n),
            }
            let spent = Spent {
                sims: searcher.sims_run(),
                nodes: searcher.node_count() as u64,
                merges: searcher.merges(),
            };
            (searcher.best_action(), spent)
        }
    }
}

// ---------------------------------------------------------------- one game

const MAX_TURNS: u32 = 100;
const EPSILON: f64 = 0.15;
const EXPLORE_TURNS: u32 = 8;

#[derive(Clone, Debug)]
struct GameOutcome {
    winner: Player,
    a_is_p1: bool,
    capped: bool,
    /// What each of side A's actions spent, and each of side B's.
    ///
    /// Kept per action rather than summed because the **mean is worthless
    /// here**: a near-terminal endgame position has a small, fully expanded
    /// tree whose leaves are all terminal, so its simulations need no net at
    /// all and run three orders of magnitude faster than a midgame one. A
    /// handful of such actions per game swamp any total. The median is the
    /// number that describes a typical action.
    per_action_a: Vec<Spent>,
    per_action_b: Vec<Spent>,
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
    budget: Budget,
) -> GameOutcome {
    let a_is_p1 = index % 2 == 0;
    let mut rng = Rng::new(derive_game_seed(seed, u64::from(index)));
    let (p1, p2) = if a_is_p1 { (a, b) } else { (b, a) };

    let mut state = State::new(12, 12, 2).expect("12x12 two-player start");
    let ply_ceiling = MAX_TURNS * u32::from(ACTIONS_PER_TURN);
    let mut plies = 0u32;
    let mut turns = 0u32;
    let mut sims: [Vec<Spent>; 2] = [Vec::new(), Vec::new()];
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
        let (searched, spent) = arm.choose(&state, net, budget);
        let a_moved = (mover == 1) == a_is_p1;
        sims[usize::from(!a_moved)].push(spent);

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

    let [per_action_a, per_action_b] = sims;
    GameOutcome {
        winner: if capped { 0 } else { state.outcome_winner() },
        a_is_p1,
        capped,
        per_action_a,
        per_action_b,
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
    budget: Budget,
    seed: u64,
    jobs: usize,
    a: Arm,
    b: Arm,
    net: String,
}

/// `1`/`0`, `true`/`false`, `on`/`off`.
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
        budget: Budget::Millis(200),
        seed: 20_260_813,
        jobs: 2,
        a: Arm {
            batch: virus_mcts::DEFAULT_BATCH_SIZE,
            threads: 1,
            dag: virus_mcts::DEFAULT_DAG,
        },
        b: Arm {
            batch: 1,
            threads: 1,
            dag: virus_mcts::DEFAULT_DAG,
        },
        net: "artifacts/mcts_champion.json".to_owned(),
    };
    let mut raw = std::env::args().skip(1);
    while let Some(flag) = raw.next() {
        let mut value = || raw.next().unwrap_or_else(|| panic!("{flag} needs a value"));
        match flag.as_str() {
            "--games" => args.games = value().parse().expect("--games"),
            // Last budget flag on the command line wins.
            "--millis" => args.budget = Budget::Millis(value().parse().expect("--millis")),
            "--sims" => args.budget = Budget::Sims(value().parse().expect("--sims")),
            "--seed" => args.seed = value().parse().expect("--seed"),
            "--jobs" => args.jobs = value().parse().expect("--jobs"),
            "--a-batch" => args.a.batch = value().parse().expect("--a-batch"),
            "--b-batch" => args.b.batch = value().parse().expect("--b-batch"),
            "--a-threads" => args.a.threads = value().parse().expect("--a-threads"),
            "--b-threads" => args.b.threads = value().parse().expect("--b-threads"),
            "--a-dag" => args.a.dag = parse_bool("--a-dag", &value()),
            "--b-dag" => args.b.dag = parse_bool("--b-dag", &value()),
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

    let budget_label = match args.budget {
        Budget::Millis(ms) => format!("{ms} ms/action (wall clock — load-sensitive)"),
        Budget::Sims(n) => format!("{n} sims/action (fixed — load-tolerant)"),
    };
    println!(
        "A = {}   B = {}\n{} games, {}, seed {}, {} concurrent game(s), net {}\n",
        args.a.label(),
        args.b.label(),
        games,
        budget_label,
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
                        args.budget,
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
    let (mut spent_a, mut spent_b): (Vec<Spent>, Vec<Spent>) = (Vec::new(), Vec::new());
    for game in &outcomes {
        match game.score_for_a() {
            s if s > 0.75 => wins += 1,
            s if s < 0.25 => losses += 1,
            _ => draws += 1,
        }
        capped += u32::from(game.capped);
        spent_a.extend_from_slice(&game.per_action_a);
        spent_b.extend_from_slice(&game.per_action_b);
    }
    let merges_a: u64 = spent_a.iter().map(|s| s.merges).sum();
    let merges_b: u64 = spent_b.iter().map(|s| s.merges).sum();
    let nodes_a: u64 = spent_a.iter().map(|s| s.nodes).sum();
    let nodes_b: u64 = spent_b.iter().map(|s| s.nodes).sum();
    let total_sims_a: u64 = spent_a.iter().map(|s| s.sims).sum();
    let total_sims_b: u64 = spent_b.iter().map(|s| s.sims).sum();
    let mut sims_a: Vec<u64> = spent_a.iter().map(|s| s.sims).collect();
    let mut sims_b: Vec<u64> = spent_b.iter().map(|s| s.sims).collect();
    sims_a.sort_unstable();
    sims_b.sort_unstable();
    let n = wins + losses + draws;
    let pooled = (f64::from(wins) + 0.5 * f64::from(draws)) / f64::from(n.max(1));
    let (low, high) = wilson95(f64::from(wins), n);

    println!("A: {wins}W {losses}L {draws}D over {n} games ({capped} hit the turn cap)");
    println!(
        "  win rate {:.1}% (draws not half-wins)  wilson95 [{low:.1}%, {high:.1}%]",
        100.0 * f64::from(wins) / f64::from(n.max(1))
    );
    println!("  pooled score {pooled:.4} (W+0.5D)/N   gate A needs >= 0.5500");
    // Quartiles, not means: see `GameOutcome::per_action_a`. The upper quartile
    // is where the net-bound positions live and is the number the throughput
    // work is about; the lower one is the near-terminal endgame, where a
    // simulation touches no net at all and both arms are identical.
    let quantile = |sorted: &[u64], q: f64| -> u64 {
        if sorted.is_empty() {
            return 0;
        }
        sorted[(((sorted.len() - 1) as f64) * q) as usize]
    };
    for (label, q) in [("p25", 0.25), ("median", 0.5), ("p75", 0.75)] {
        let a = quantile(&sims_a, q);
        let b = quantile(&sims_b, q);
        println!(
            "  sims/action {label:>6}: A {a:>9}, B {b:>9}   ({:.2}x)",
            a as f64 / (b.max(1)) as f64
        );
    }
    println!("  actions searched: A {}, B {}", sims_a.len(), sims_b.len());
    // The DAG's own numbers, over every action of the run. `merges` counts the
    // child links that landed on a node the search already had — the duplicate
    // subtree roots, and the statistics both action orders now share instead of
    // splitting. `nodes` is the arena at the end of each action, i.e. the net
    // forwards paid for.
    if merges_a > 0 || merges_b > 0 {
        println!(
            "  DAG merges:   A {merges_a:>10}, B {merges_b:>10}   \
             ({:.1}% and {:.1}% of simulations)",
            100.0 * merges_a as f64 / total_sims_a.max(1) as f64,
            100.0 * merges_b as f64 / total_sims_b.max(1) as f64,
        );
        println!(
            "  expansions:   A {nodes_a:>10}, B {nodes_b:>10}   \
             ({:+.1}% net forwards for A)",
            100.0 * (nodes_a as f64 - nodes_b as f64) / nodes_b.max(1) as f64,
        );
    }
    println!("  elapsed {:.1}s", started.elapsed().as_secs_f64());
}
