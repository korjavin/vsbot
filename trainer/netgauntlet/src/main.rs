//! `netgauntlet` — one net against another, at fixed simulations.
//!
//! ```text
//! netgauntlet --a-net work/gen6/candidate.json --b-net work/gen6/champion.json \
//!             --games 400 --sims 192 --seed 61000 --jobs 2
//! ```
//!
//! Side A is the **candidate**, side B the **champion**, and every number is
//! printed from A's chair. The last line is machine-readable so a shell stage
//! can pool several instances without parsing prose:
//!
//! ```text
//! RESULT w=<W> l=<L> d=<D> n=<N> pooled=<(W+0.5D)/N> capped=<C> stalled=<S>
//! ```
//!
//! # What this reuses, and why that matters
//!
//! Everything except the game loop is `virus-arena`'s, imported rather than
//! copied, so a gauntlet run here and a gauntlet run by `arena` differ in one
//! respect only — the number of loaded nets:
//!
//! * `virus_arena::rng::derive_game_seed` — the colour-pairing seed. Games
//!   `2k` and `2k+1` are the same opening from both chairs, so first-mover
//!   advantage cancels *inside* a pair.
//! * `virus_arena::engine::build` — the side itself. That is arena's
//!   `MctsSide`: `ValueSource::Net`, `Config::play()` (no Dirichlet, no visit
//!   sampling, no RNG), `Budget::Nodes(sims)` meaning simulations. A
//!   reimplementation here could quietly differ in any of those and would
//!   report the difference as a strength change.
//! * `virus_arena::stats` — `Record`, `pooled_score`, `wilson95`, `verdict`.
//!   The gate arithmetic has exactly one implementation in this repo.
//!
//! # Fixed sims, not fixed time
//!
//! `--sims` is a node budget, which is deterministic and load-tolerant: this
//! box runs arena cells and other executors concurrently, and a wall-clock
//! budget would measure the box's spare capacity as if it were the net's
//! strength. The batched-search work (`batchgauntlet`) set that precedent. It
//! does mean the two sides are compared per *simulation*, not per second —
//! fair here because both sides are the same architecture at the same
//! geometry, and unfair the moment they are not.
//!
//! # Turn cap is a draw
//!
//! Matching `virus_arena::gauntlet`, and unlike `virus-selfplay` (whose capped
//! games keep a territory verdict, because a training row needs a target). A
//! capped game is one the engines failed to decide; scoring it by territory
//! would let a turtling net bank a win it never played out. `capped=` in the
//! RESULT line exists so a run that is mostly caps is visible as such.

use std::process::ExitCode;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Instant;

use virus_arena::engine::{self, Budget, SideSpec, SpecError};
use virus_arena::rng::{derive_game_seed, Rng};
use virus_arena::stats::{Outcome, Record, GATE_MIN_GAMES, PROMOTION_THRESHOLD};
use virus_core::{Action, Player, State, ACTIONS_PER_TURN};
use virus_mcts::PolicyValueNet;

const USAGE: &str = "\
netgauntlet — candidate vs champion at fixed sims, colour-paired and seeded

USAGE:
    netgauntlet --a-net <PATH> --b-net <PATH> [OPTIONS]

SIDES:
    --a-net <PATH>       side A's artifact (the candidate) [required]
    --b-net <PATH>       side B's artifact (the champion)  [required]

RUN:
    --games <N>          games, rounded up to even [default: 100]
    --sims <N>           simulations per action, both sides [default: 192]
    --seed <N>           base seed; space pooled instances >=1000 apart
                         [default: 11]
    --jobs <N>           concurrent games [default: 1]
    --rows <N> --cols <N>  board size [default: 12x12]
    --max-turns <N>      turn cap; a capped game is a DRAW [default: 100]
    --eps <P>            opening randomisation probability [default: 0.15]
    --explore-turns <N>  opening window, in turns [default: 8]

OUTPUT:
    -h, --help           this text

Two different artifacts is the whole point; `arena` refuses that pairing
because it shares one loaded net across all games. See this package's
Cargo.toml for why that refusal is right and why this exists anyway.
";

// Arena's own defaults, named here so a drift between the two harnesses would
// be a compile-time import error rather than a silent difference in numbers.
use virus_arena::gauntlet::{
    GauntletConfig, DEFAULT_EPSILON, DEFAULT_EXPLORE_TURNS, DEFAULT_MAX_TURNS,
};

struct Args {
    a_net: String,
    b_net: String,
    games: u32,
    sims: u64,
    seed: u64,
    jobs: usize,
    rows: usize,
    cols: usize,
    max_turns: u32,
    epsilon: f64,
    explore_turns: u32,
}

/// Parses into the field's own type: `--sims 4294967297` must be an error, not
/// a wrap to one simulation per action. `virus-selfplay`'s CLI documents the
/// same reasoning at length — a misconfiguration that still produces a
/// plausible-looking gauntlet is the expensive kind.
fn number<T>(text: &str, what: &str) -> Result<T, String>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    text.parse::<T>()
        .map_err(|error| format!("{what}: {error} (got {text:?})"))
}

fn parse() -> Result<Option<Args>, String> {
    let mut args = Args {
        a_net: String::new(),
        b_net: String::new(),
        games: 100,
        sims: 192,
        seed: 11,
        jobs: 1,
        rows: 12,
        cols: 12,
        max_turns: DEFAULT_MAX_TURNS,
        epsilon: DEFAULT_EPSILON,
        explore_turns: DEFAULT_EXPLORE_TURNS,
    };
    let mut raw = std::env::args().skip(1);
    while let Some(flag) = raw.next() {
        let mut value = || raw.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "-h" | "--help" => {
                print!("{USAGE}");
                return Ok(None);
            }
            "--a-net" => args.a_net = value()?,
            "--b-net" => args.b_net = value()?,
            "--games" => args.games = number(&value()?, "--games")?,
            "--sims" => args.sims = number(&value()?, "--sims")?,
            "--seed" => args.seed = number(&value()?, "--seed")?,
            "--jobs" => args.jobs = number(&value()?, "--jobs")?,
            "--rows" => args.rows = number(&value()?, "--rows")?,
            "--cols" => args.cols = number(&value()?, "--cols")?,
            "--max-turns" => args.max_turns = number(&value()?, "--max-turns")?,
            "--eps" => args.epsilon = number(&value()?, "--eps")?,
            "--explore-turns" => args.explore_turns = number(&value()?, "--explore-turns")?,
            other => return Err(format!("unknown flag {other:?}\n\n{USAGE}")),
        }
    }
    if args.a_net.is_empty() || args.b_net.is_empty() {
        return Err("--a-net and --b-net are both required".to_owned());
    }
    if args.a_net == args.b_net {
        // Playing an artifact against itself is a 50% machine. It is never what
        // a gate wants and it is easy to reach by a copy-paste, so it stops the
        // run instead of producing a tidy meaningless number.
        return Err(format!(
            "--a-net and --b-net are the same file ({}); that measures nothing",
            args.a_net
        ));
    }
    if args.sims == 0 {
        return Err("--sims 0: a side that never searches is not an engine".to_owned());
    }
    Ok(Some(args))
}

/// Runs every check `arena` runs, by building the config `arena` would build
/// and asking it.
///
/// Reused rather than reimplemented for the same reason `engine::build` is:
/// `GauntletConfig::validate` rejects zero games, an out-of-range epsilon, a
/// zero turn cap and — the one most likely to bite here — a board that is not
/// 12x12 when a net is playing, which would otherwise panic inside a worker
/// and surface only as "a worker panicked". `--games 0` in particular would
/// otherwise exit 0 having printed `RESULT ... n=0`, and the generation script
/// would stamp its gauntlet stage complete on a tally that does not exist.
fn validate(args: &Args, spec: &SideSpec) -> Result<(), SpecError> {
    GauntletConfig {
        side_a: spec.clone(),
        side_b: spec.clone(),
        games: args.games,
        seed: args.seed,
        rows: args.rows,
        cols: args.cols,
        max_turns: args.max_turns,
        epsilon: args.epsilon,
        explore_turns: args.explore_turns,
        threads: args.jobs,
    }
    .validate()
}

/// How a game ended. Kept separate from the winner so a report can say whether
/// the *engines* decided it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Termination {
    Decided,
    TurnCap,
    Stalled,
}

struct GameOutcome {
    index: u32,
    winner: Player,
    a_is_p1: bool,
    termination: Termination,
}

impl GameOutcome {
    fn outcome_for_a(&self) -> Outcome {
        if self.winner == 0 {
            return Outcome::Draw;
        }
        if (self.winner == 1) == self.a_is_p1 {
            Outcome::Win
        } else {
            Outcome::Loss
        }
    }
}

/// One game. A transcription of `virus_arena::gauntlet::play_game` with one
/// change: each seat is built against **its own** net.
fn play_game(
    args: &Args,
    index: u32,
    spec: &SideSpec,
    net_a: &PolicyValueNet,
    net_b: &PolicyValueNet,
) -> Result<GameOutcome, SpecError> {
    let a_is_p1 = index % 2 == 0;
    let mut rng = Rng::new(derive_game_seed(args.seed, u64::from(index)));

    let (net_p1, net_p2) = if a_is_p1 {
        (net_a, net_b)
    } else {
        (net_b, net_a)
    };
    let mut seat1 = engine::build(spec, 1, Some(net_p1))?;
    let mut seat2 = engine::build(spec, 2, Some(net_p2))?;

    let mut state = State::new(args.rows, args.cols, 2)
        .map_err(|error| SpecError(format!("{}x{} board: {error}", args.rows, args.cols)))?;

    let ply_ceiling = args.max_turns.saturating_mul(u32::from(ACTIONS_PER_TURN));
    let mut termination = Termination::TurnCap;
    let mut plies = 0u32;
    let mut turns = 0u32;

    while turns < args.max_turns && plies < ply_ceiling {
        if state.game_over() {
            termination = Termination::Decided;
            break;
        }
        let legal = state.legal_actions();
        if legal.is_empty() {
            termination = Termination::Stalled;
            break;
        }
        let mover = state.current_player();
        let side: &mut dyn engine::Side = if mover == 1 {
            seat1.as_mut()
        } else {
            seat2.as_mut()
        };
        // Search on every ply, including the ones the coin overrides: both
        // colours of a pair must see the same sequence of search calls.
        let (searched, _stats) = side.choose(&state);

        let chosen: Action = if turns < args.explore_turns && rng.next_f64() < args.epsilon {
            legal[rng.below(legal.len()).expect("legal is non-empty")]
        } else {
            match searched {
                Some(action) => action,
                None => {
                    termination = Termination::Stalled;
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
            termination = Termination::Stalled;
            break;
        };
        if next.current_player() != mover {
            turns += 1;
        }
        state = next;
        plies += 1;
    }
    if state.game_over() {
        termination = Termination::Decided;
    }

    Ok(GameOutcome {
        index,
        winner: if termination == Termination::Decided {
            state.winner()
        } else {
            0
        },
        a_is_p1,
        termination,
    })
}

fn main() -> ExitCode {
    match real_main() {
        Ok(code) => code,
        Err(error) => {
            eprintln!("netgauntlet: {error}");
            ExitCode::FAILURE
        }
    }
}

fn real_main() -> Result<ExitCode, String> {
    let Some(args) = parse()? else {
        return Ok(ExitCode::SUCCESS);
    };

    let net_a =
        PolicyValueNet::load(&args.a_net).map_err(|e| format!("--a-net {}: {e}", args.a_net))?;
    let net_b =
        PolicyValueNet::load(&args.b_net).map_err(|e| format!("--b-net {}: {e}", args.b_net))?;
    // `mcts` with no `:path`: the artifact comes from `engine::build`'s `net`
    // argument, which is where the two sides diverge.
    let spec = SideSpec::parse("mcts", Budget::Nodes(args.sims)).map_err(|e| e.0)?;
    validate(&args, &spec).map_err(|e| e.0)?;

    let games = args.games.div_ceil(2) * 2;
    eprintln!(
        "netgauntlet: A={} vs B={}, {games} games, {} sims/action, seed {}, {}x{}, {} concurrent",
        args.a_net, args.b_net, args.sims, args.seed, args.rows, args.cols, args.jobs,
    );

    let next = AtomicU32::new(0);
    let started = Instant::now();
    let mut collected: Vec<GameOutcome> = Vec::with_capacity(games as usize);
    let threads = args.jobs.max(1).min(games as usize);
    let failure: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);

    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(threads);
        for _ in 0..threads {
            let (next, args, spec, net_a, net_b, failure) =
                (&next, &args, &spec, &net_a, &net_b, &failure);
            handles.push(scope.spawn(move || {
                let mut mine = Vec::new();
                loop {
                    let index = next.fetch_add(1, Ordering::Relaxed);
                    if index >= games {
                        return mine;
                    }
                    match play_game(args, index, spec, net_a, net_b) {
                        Ok(outcome) => mine.push(outcome),
                        Err(error) => {
                            *failure.lock().expect("failure mutex") = Some(error.0);
                            return mine;
                        }
                    }
                }
            }));
        }
        for handle in handles {
            match handle.join() {
                Ok(mut games) => collected.append(&mut games),
                Err(_) => {
                    *failure.lock().expect("failure mutex") =
                        Some("a worker panicked; see the panic message above".to_owned())
                }
            }
        }
    });
    if let Some(error) = failure.lock().expect("failure mutex").take() {
        return Err(error);
    }

    // Fold in index order, never completion order: the tally must not depend on
    // which worker finished first.
    collected.sort_by_key(|game| game.index);
    let mut record = Record::default();
    let mut capped = 0u32;
    let mut stalled = 0u32;
    for game in &collected {
        record.add(game.outcome_for_a());
        match game.termination {
            Termination::TurnCap => capped += 1,
            Termination::Stalled => stalled += 1,
            Termination::Decided => {}
        }
    }

    let n = record.games();
    let pooled = record.pooled_score();
    let interval = record.wilson95();
    let verdict = record.verdict();
    println!(
        "candidate {} vs champion {}\n\
         W-L-D {}-{}-{} of {n}   win rate {:.1}% Wilson95 {interval}\n\
         pooled (W+0.5D)/N = {pooled:.4}   gate {PROMOTION_THRESHOLD} -> {}\n\
         {} capped, {} stalled, {:.1}s   [{}]",
        args.a_net,
        args.b_net,
        record.wins,
        record.losses,
        record.draws,
        record.win_rate(),
        if pooled >= PROMOTION_THRESHOLD {
            "PASS"
        } else {
            "FAIL"
        },
        capped,
        stalled,
        started.elapsed().as_secs_f64(),
        verdict.label(),
    );
    if let Some(caveat) = verdict.caveat() {
        println!("NOTE: {caveat}");
    }
    if n < GATE_MIN_GAMES {
        println!(
            "NOTE: a single instance under {GATE_MIN_GAMES} games cannot promote on its own; \
             pool instances and re-apply the gate to the pooled tally."
        );
    }
    // Machine-readable, last, one line: the report stage pools these.
    println!(
        "RESULT w={} l={} d={} n={n} pooled={pooled:.6} capped={capped} stalled={stalled}",
        record.wins, record.losses, record.draws,
    );

    if stalled > 0 {
        eprintln!(
            "netgauntlet: WARNING — {stalled} game(s) stalled; this tally is not trustworthy"
        );
        return Ok(ExitCode::FAILURE);
    }
    Ok(ExitCode::SUCCESS)
}
