//! The match loop: colour-paired games, seeded openings, and the tally.
//!
//! # The three things that make a gauntlet mean something
//!
//! **1. Colours are paired.** Games `2k` and `2k+1` are the same opening seed
//! with the seats swapped, so first-mover advantage cancels within the pair
//! instead of being assumed away. [`GauntletConfig::games`] is rounded *up* to
//! even for exactly this reason: an odd count leaves one game where side A
//! always moves first, a deterministic bias that no amount of sampling removes.
//!
//! **2. Identical sides cancel exactly.** If A and B are the same
//! configuration, the two games of a pair are the *same game* — same seed, same
//! deterministic engines in both seats — so the same seat wins both, one point
//! each way. A self-gauntlet reading anything other than 50/50 (in a
//! deterministic budget mode) means the harness is leaking state between games,
//! and [`self_play_cancels_exactly`] in the test suite is the tripwire.
//!
//! **3. Openings are seeded and diverse.** Two deterministic engines replay one
//! game forever, so 400 repetitions of it are one sample, not 400. Epsilon-greedy
//! opening noise (`eps = 0.15` over the first 8 turns, ported from Java's
//! `GauntletMatch`) buys the diversity, and it is drawn from a per-pair seed so
//! the diversity is reproducible.
//!
//! # The subtlety in the epsilon loop
//!
//! The search runs on **every** ply, including plies where the coin says to
//! play a random move and the search result is thrown away. This looks wasteful
//! and is load-bearing: an enhanced searcher accumulates a transposition table,
//! killers and history across its moves, and skipping the search on random
//! plies would leave the two colours of a pair with different accumulated
//! state. Java's `GauntletMatch` does the same thing for the same reason.
//!
//! # Draws
//!
//! A game that reaches the turn cap is a **draw**, not a territory decision.
//! That is Java's tally rule and it is the conservative one: territory at the
//! cap measures who was ahead in a game neither side finished, which is a
//! different question from who won. The territory verdict is still computed and
//! carried on [`GameOutcome::territory_winner`] for anyone recording a corpus,
//! and the cap-hit count is reported separately so a run decided by the cap
//! cannot be mistaken for a run decided by the engines.

use crate::engine::{self, MoveStats, SideSpec};
use crate::rng::{derive_game_seed, Rng};
use crate::stats::{Outcome, Record, Summary};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use virus_core::{Action, Player, State, ACTIONS_PER_TURN};
use virus_mcts::{PolicyValueNet, BOARD as NET_BOARD};

/// Java's `GauntletMatch.Config.maxTurns`, in whole turns.
pub const DEFAULT_MAX_TURNS: u32 = 100;

/// Java's `GauntletMatch.Config.epsilon`.
pub const DEFAULT_EPSILON: f64 = 0.15;

/// Java's `GauntletMatch.Config.exploreTurns`, in whole turns.
pub const DEFAULT_EXPLORE_TURNS: u32 = 8;

/// Everything a gauntlet needs to run.
#[derive(Clone, Debug)]
pub struct GauntletConfig {
    /// Side A's configuration.
    pub side_a: SideSpec,
    /// Side B's configuration.
    pub side_b: SideSpec,
    /// Games to play. Rounded up to even — see the module docs.
    pub games: u32,
    /// Base seed. Runs whose results are pooled must use base seeds spaced far
    /// apart; see [`crate::rng`] for why nearby seeds used to overlap.
    pub seed: u64,
    /// Board rows.
    pub rows: usize,
    /// Board columns.
    pub cols: usize,
    /// Turn cap; a game reaching it is a draw.
    pub max_turns: u32,
    /// Probability of playing a uniformly random legal action during the
    /// opening window.
    pub epsilon: f64,
    /// Length of the opening window, in whole turns.
    pub explore_turns: u32,
    /// Worker threads. Games are independent, so this never changes the tally
    /// in a deterministic budget mode — only how long it takes.
    pub threads: usize,
}

impl Default for GauntletConfig {
    fn default() -> GauntletConfig {
        GauntletConfig {
            side_a: SideSpec {
                engine: engine::Engine::AlphaBeta { enhanced: true },
                budget: engine::Budget::Nodes(engine::DEFAULT_NODE_LIMIT),
                net: None,
            },
            side_b: SideSpec {
                engine: engine::Engine::AlphaBeta { enhanced: false },
                budget: engine::Budget::Nodes(engine::DEFAULT_NODE_LIMIT),
                net: None,
            },
            games: 8,
            seed: 1,
            rows: 12,
            cols: 12,
            max_turns: DEFAULT_MAX_TURNS,
            epsilon: DEFAULT_EPSILON,
            explore_turns: DEFAULT_EXPLORE_TURNS,
            threads: 1,
        }
    }
}

impl GauntletConfig {
    /// The effective game count: `games` rounded up to even.
    pub fn even_games(&self) -> u32 {
        self.games.div_ceil(2) * 2
    }

    /// A hard ply ceiling, used only as a runaway guard.
    ///
    /// A turn is *at most* [`ACTIONS_PER_TURN`] actions, so `max_turns` turns
    /// can never exceed this many plies. The real cap is counted in turns —
    /// see [`play_game`] — and this exists so a bug in that counting shows up
    /// as a finished game rather than a hung worker.
    pub fn ply_ceiling(&self) -> u32 {
        self.max_turns * u32::from(ACTIONS_PER_TURN)
    }

    /// Rejects a configuration that cannot produce a meaningful result.
    pub fn validate(&self) -> Result<(), engine::SpecError> {
        self.side_a.validate()?;
        self.side_b.validate()?;
        if self.games == 0 {
            return Err(engine::SpecError(
                "a gauntlet needs at least one game".to_owned(),
            ));
        }
        if !(0.0..=1.0).contains(&self.epsilon) {
            return Err(engine::SpecError(format!(
                "epsilon {} is not a probability",
                self.epsilon
            )));
        }
        if self.max_turns == 0 {
            return Err(engine::SpecError("max_turns must be at least 1".to_owned()));
        }
        // The net's input encoding has no representation for another board
        // size, so `MctsSearcher::new` asserts 12x12 when a net is supplied.
        // Catching it here turns a panic inside a worker — which the harness
        // can only report as "a worker panicked" — into a message that names
        // the actual mistake, before any game runs.
        if (self.side_a.needs_net() || self.side_b.needs_net())
            && (self.rows != NET_BOARD || self.cols != NET_BOARD)
        {
            return Err(engine::SpecError(format!(
                "an MCTS side needs a {NET_BOARD}x{NET_BOARD} board, got {}x{}",
                self.rows, self.cols
            )));
        }
        // Two-player only: MCTS's absolute frame is a one-axis construct and
        // pairing is a two-colour idea. Both would need redesign, not a
        // relaxed check.
        Ok(())
    }
}

/// Why a game stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Termination {
    /// A player was eliminated or ran out of moves; the state is terminal.
    Decided,
    /// The turn cap was reached. Scored as a draw.
    TurnCap,
    /// A side returned no action while legal actions existed. This should never
    /// happen and is surfaced rather than swallowed.
    Stalled,
}

/// The result of one game.
#[derive(Clone, Copy, Debug)]
pub struct GameOutcome {
    /// Index in the run, so a parallel run can be folded in a fixed order.
    pub index: u32,
    /// The winning seat, or 0 for none.
    pub winner: Player,
    /// Whether side A held seat 1.
    pub a_is_p1: bool,
    /// Why the game stopped.
    pub termination: Termination,
    /// Actions played.
    pub plies: u32,
    /// Whole turns completed. Not `plies / 3` — a `PlaceNeutrals` spends a
    /// turn in one action, so the two counts diverge in any game where a seat
    /// used its placement.
    pub turns: u32,
    /// Territory verdict — the corpus label. **Not** the tally.
    pub territory_winner: Player,
    /// Worst per-move deadline overrun in this game.
    pub max_overrun: Duration,
    /// Total search work by side A (nodes or sims).
    pub work_a: u64,
    /// Total search work by side B.
    pub work_b: u64,
}

impl GameOutcome {
    /// The result from side A's perspective.
    pub fn outcome_for_a(&self) -> Outcome {
        if self.winner == 0 {
            return Outcome::Draw;
        }
        let a_won = (self.winner == 1) == self.a_is_p1;
        if a_won {
            Outcome::Win
        } else {
            Outcome::Loss
        }
    }
}

/// A finished gauntlet.
#[derive(Clone, Debug)]
pub struct GauntletResult {
    /// The tally from side A's perspective.
    pub record: Record,
    /// Per-game outcomes in index order.
    pub games: Vec<GameOutcome>,
    /// The rendered summary.
    pub summary: Summary,
}

/// Runs a gauntlet, returning the tally from side A's perspective.
///
/// `net` is loaded once by the caller and shared by every game and thread.
///
/// # Determinism
///
/// In a node- or depth-budget mode the result is a pure function of the config:
/// every game is independent, seeded from `(seed, game index)`, and no engine
/// consults a clock. `threads` therefore changes only the wall time. In
/// [`engine::Budget::Millis`] mode nothing is reproducible by construction —
/// that is what a wall clock means — and `threads > 1` additionally makes each
/// side's effective compute depend on machine load.
pub fn run(
    config: &GauntletConfig,
    net: Option<&PolicyValueNet>,
) -> Result<GauntletResult, engine::SpecError> {
    config.validate()?;
    let total = config.even_games();
    let started = Instant::now();

    let next = AtomicUsize::new(0);
    let threads = config.threads.max(1).min(total as usize);
    let mut collected: Vec<GameOutcome> = Vec::with_capacity(total as usize);

    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(threads);
        for _ in 0..threads {
            let next = &next;
            handles.push(
                scope.spawn(move || -> Result<Vec<GameOutcome>, engine::SpecError> {
                    let mut mine = Vec::new();
                    loop {
                        let index = next.fetch_add(1, Ordering::Relaxed) as u32;
                        if index >= total {
                            return Ok(mine);
                        }
                        mine.push(play_game(config, index, net)?);
                    }
                }),
            );
        }
        for handle in handles {
            match handle.join() {
                Ok(Ok(mut games)) => collected.append(&mut games),
                Ok(Err(error)) => return Err(error),
                Err(_) => {
                    return Err(engine::SpecError(
                        "a gauntlet worker panicked; see the panic message above".to_owned(),
                    ))
                }
            }
        }
        Ok(())
    })?;

    // Fold in index order, never completion order: the tally must not depend on
    // which worker finished first.
    collected.sort_by_key(|game| game.index);

    let mut record = Record::default();
    let mut capped = 0;
    let mut max_overrun = Duration::ZERO;
    for game in &collected {
        record.add(game.outcome_for_a());
        if game.termination == Termination::TurnCap {
            capped += 1;
        }
        max_overrun = max_overrun.max(game.max_overrun);
    }

    let summary = Summary {
        side_a: config.side_a.name(),
        side_b: config.side_b.name(),
        record,
        capped,
        max_overrun_ms: max_overrun.as_millis() as u64,
        elapsed_secs: started.elapsed().as_secs_f64(),
    };
    Ok(GauntletResult {
        record,
        games: collected,
        summary,
    })
}

/// Plays one game of the run.
pub fn play_game(
    config: &GauntletConfig,
    index: u32,
    net: Option<&PolicyValueNet>,
) -> Result<GameOutcome, engine::SpecError> {
    // Even indices give side A seat 1. Both games of a pair share a seed, so
    // the pair is one opening played from both chairs.
    let a_is_p1 = index % 2 == 0;
    let seed = derive_game_seed(config.seed, u64::from(index));
    let mut rng = Rng::new(seed);

    let (spec_p1, spec_p2) = if a_is_p1 {
        (&config.side_a, &config.side_b)
    } else {
        (&config.side_b, &config.side_a)
    };
    let mut seat1 = engine::build(spec_p1, 1, net)?;
    let mut seat2 = engine::build(spec_p2, 2, net)?;

    let mut state = State::new(config.rows, config.cols, 2).map_err(|error| {
        engine::SpecError(format!("{}x{} board: {error}", config.rows, config.cols))
    })?;

    // Both limits are counted in **turns**, from the actual seat changes, not
    // derived from a ply count.
    //
    // Java derives them as `turns * ACTIONS_PER_TURN` and that is wrong here
    // for a rule its own engine has: `PlaceNeutrals` consumes a whole turn in a
    // single action. A game where both players spend their once-per-game
    // placement therefore fits more than `max_turns` turns inside
    // `max_turns * 3` plies, and the epsilon window runs past the turn it was
    // asked for. The drift is small — at most one extra turn per seat per game
    // — but the flags say "turns" and there is no reason for them to mean
    // something else.
    let ply_ceiling = config.ply_ceiling();
    let mut termination = Termination::TurnCap;
    let mut plies = 0u32;
    let mut turns = 0u32;
    let mut max_overrun = Duration::ZERO;
    let mut work = [0u64; 2];

    while turns < config.max_turns && plies < ply_ceiling {
        if state.game_over() {
            termination = Termination::Decided;
            break;
        }
        let legal = state.legal_actions();
        if legal.is_empty() {
            // `virus-core` eliminates a moveless player, so a non-terminal
            // position with no legal action should be unreachable. Treat it as
            // a stall rather than asserting: a gauntlet that dies mid-run
            // reports nothing at all.
            termination = Termination::Stalled;
            break;
        }

        let mover = state.current_player();
        let side: &mut dyn engine::Side = if mover == 1 {
            seat1.as_mut()
        } else {
            seat2.as_mut()
        };
        // Always search, even on a ply the coin will override: the enhanced
        // searcher's table, killers and history evolve with every call, and a
        // skipped search would leave a pair's two colours in different states.
        let (searched, stats): (Option<Action>, MoveStats) = side.choose(&state);
        max_overrun = max_overrun.max(stats.overrun);
        let a_moved = (mover == 1) == a_is_p1;
        work[usize::from(!a_moved)] += stats.work;

        let chosen = if turns < config.explore_turns && rng.next_f64() < config.epsilon {
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

        // `legal_actions` is the authority on legality; anything else is a bug
        // in a side, and playing it would corrupt the game rather than the
        // tally. Fall back the way Java does.
        let action = if legal.contains(&chosen) {
            chosen
        } else {
            legal[0]
        };
        let next = match state.apply(action) {
            Ok(next) => next,
            Err(_) => {
                termination = Termination::Stalled;
                break;
            }
        };
        // A turn ends when the seat on move changes — which is after three
        // actions, or after a single `PlaceNeutrals`, or immediately when the
        // action eliminated somebody and `advance` skipped past them.
        if next.current_player() != mover {
            turns += 1;
        }
        state = next;
        plies += 1;
    }
    if state.game_over() {
        termination = Termination::Decided;
    }

    let winner = if termination == Termination::Decided {
        state.winner()
    } else {
        0
    };
    Ok(GameOutcome {
        index,
        winner,
        a_is_p1,
        termination,
        plies,
        turns,
        territory_winner: state.outcome_winner(),
        max_overrun,
        work_a: work[0],
        work_b: work[1],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Budget, Engine};

    fn spec(engine: Engine, budget: Budget) -> SideSpec {
        SideSpec {
            engine,
            budget,
            net: None,
        }
    }

    fn small(side_a: SideSpec, side_b: SideSpec, games: u32) -> GauntletConfig {
        GauntletConfig {
            side_a,
            side_b,
            games,
            seed: 4242,
            // A small board finishes fast; the pairing and tally logic under
            // test is board-size independent.
            rows: 7,
            cols: 7,
            max_turns: 40,
            ..GauntletConfig::default()
        }
    }

    /// The pairing property that makes the whole harness trustworthy: with the
    /// same configuration on both sides, every pair splits 1-1, so the run
    /// reads exactly 50% no matter how strong the engine is.
    #[test]
    fn self_play_cancels_exactly() {
        let side = spec(Engine::AlphaBeta { enhanced: true }, Budget::Nodes(400));
        let config = small(side.clone(), side, 8);
        let result = run(&config, None).expect("run");
        assert_eq!(
            result.record.wins, result.record.losses,
            "{:?}",
            result.record
        );
        for pair in 0..4 {
            let even = &result.games[pair * 2];
            let odd = &result.games[pair * 2 + 1];
            assert_eq!(
                even.winner, odd.winner,
                "pair {pair} must replay one game from both chairs"
            );
            assert_eq!(even.plies, odd.plies, "pair {pair} must replay one game");
        }
    }

    /// The determinism gate in miniature: same seeds, same node budget, same
    /// tally — and the same tally again with the work spread over threads.
    #[test]
    fn a_node_budget_run_is_reproducible_and_thread_invariant() {
        let config = small(
            spec(Engine::AlphaBeta { enhanced: true }, Budget::Nodes(300)),
            spec(Engine::Greedy, Budget::Nodes(1)),
            6,
        );
        let first = run(&config, None).expect("first");
        let second = run(&config, None).expect("second");
        assert_eq!(first.record, second.record);

        let threaded = GauntletConfig {
            threads: 4,
            ..config
        };
        let third = run(&threaded, None).expect("threaded");
        assert_eq!(first.record, third.record);
        let winners = |result: &GauntletResult| -> Vec<(u32, Player)> {
            result
                .games
                .iter()
                .map(|game| (game.index, game.winner))
                .collect()
        };
        assert_eq!(winners(&first), winners(&third));
    }

    /// A different base seed must produce different games, or the opening
    /// diversity is decorative.
    #[test]
    fn a_different_seed_produces_different_games() {
        let config = small(
            spec(Engine::AlphaBeta { enhanced: true }, Budget::Nodes(300)),
            spec(Engine::Greedy, Budget::Nodes(1)),
            4,
        );
        let first = run(&config, None).expect("first");
        let other = run(
            &GauntletConfig {
                seed: 999_983,
                ..config
            },
            None,
        )
        .expect("other");
        let plies = |result: &GauntletResult| -> Vec<u32> {
            result.games.iter().map(|game| game.plies).collect()
        };
        assert_ne!(plies(&first), plies(&other));
    }

    /// The harness has to be able to tell strong from weak, or a 50% reading
    /// proves nothing. Alpha-beta must crush the greedy floor.
    #[test]
    fn alpha_beta_beats_the_greedy_floor() {
        let config = small(
            spec(Engine::AlphaBeta { enhanced: true }, Budget::Nodes(2_000)),
            spec(Engine::Greedy, Budget::Nodes(1)),
            8,
        );
        let result = run(&config, None).expect("run");
        assert!(
            result.record.wins >= 7,
            "alpha-beta should dominate greedy: {:?}",
            result.record
        );
    }

    #[test]
    fn odd_game_counts_are_rounded_up_to_even() {
        let config = small(
            spec(Engine::Greedy, Budget::Nodes(1)),
            spec(Engine::Greedy, Budget::Nodes(1)),
            5,
        );
        assert_eq!(config.even_games(), 6);
        let result = run(&config, None).expect("run");
        assert_eq!(result.record.games(), 6);
        assert_eq!(result.games.len(), 6);
    }

    #[test]
    fn seats_alternate_by_game_parity() {
        let config = small(
            spec(Engine::Greedy, Budget::Nodes(1)),
            spec(Engine::AlphaBeta { enhanced: true }, Budget::Nodes(300)),
            6,
        );
        let result = run(&config, None).expect("run");
        for game in &result.games {
            assert_eq!(game.a_is_p1, game.index % 2 == 0);
        }
    }

    /// Every game must finish for a real reason. A run full of stalls is a
    /// broken harness reporting a clean-looking tally.
    #[test]
    fn games_terminate_decisively() {
        let config = small(
            spec(Engine::AlphaBeta { enhanced: true }, Budget::Nodes(500)),
            spec(Engine::Greedy, Budget::Nodes(1)),
            4,
        );
        let result = run(&config, None).expect("run");
        for game in &result.games {
            assert_eq!(game.termination, Termination::Decided, "{game:?}");
            assert!(game.plies > 10, "{game:?}");
            assert_eq!(game.max_overrun, Duration::ZERO, "node mode is clock-free");
        }
    }

    #[test]
    fn outcomes_are_read_from_side_as_chair() {
        let win_as_p1 = GameOutcome {
            index: 0,
            winner: 1,
            a_is_p1: true,
            termination: Termination::Decided,
            plies: 10,
            turns: 3,
            territory_winner: 1,
            max_overrun: Duration::ZERO,
            work_a: 0,
            work_b: 0,
        };
        assert_eq!(win_as_p1.outcome_for_a(), Outcome::Win);
        assert_eq!(
            GameOutcome {
                a_is_p1: false,
                ..win_as_p1
            }
            .outcome_for_a(),
            Outcome::Loss
        );
        assert_eq!(
            GameOutcome {
                winner: 0,
                ..win_as_p1
            }
            .outcome_for_a(),
            Outcome::Draw
        );
    }

    /// The turn cap must bite in turns, not in `turns * 3` plies.
    ///
    /// A `PlaceNeutrals` spends a whole turn in a single action, so deriving
    /// the cap from a ply count lets a game with placements run past the turn
    /// limit it was given. This pins the counting to the actual seat changes.
    #[test]
    fn the_turn_cap_is_counted_in_turns() {
        let config = GauntletConfig {
            max_turns: 4,
            // Pure random play, so neutral placements actually occur.
            epsilon: 1.0,
            explore_turns: 99,
            ..small(
                spec(Engine::Greedy, Budget::Nodes(1)),
                spec(Engine::Greedy, Budget::Nodes(1)),
                8,
            )
        };
        let result = run(&config, None).expect("run");
        let mut saw_short_turn = false;
        for game in &result.games {
            assert!(
                game.turns <= 4,
                "game {} ran {} turns past a 4-turn cap",
                game.index,
                game.turns
            );
            assert!(
                game.plies <= config.ply_ceiling(),
                "game {} exceeded the ply ceiling",
                game.index
            );
            if game.termination == Termination::TurnCap {
                assert_eq!(game.turns, 4, "a capped game stops exactly at the cap");
            }
            // The whole point of the fix: at least one game must have a turn
            // shorter than three actions, or the test proves nothing about
            // neutral placement.
            if game.plies < game.turns * u32::from(ACTIONS_PER_TURN) {
                saw_short_turn = true;
            }
        }
        assert!(
            saw_short_turn,
            "no game contained a short turn, so this test did not exercise the rule \
             it exists for"
        );
    }

    #[test]
    fn a_bad_configuration_is_refused() {
        let base = small(
            spec(Engine::Greedy, Budget::Nodes(1)),
            spec(Engine::Greedy, Budget::Nodes(1)),
            2,
        );
        assert!(GauntletConfig {
            games: 0,
            ..base.clone()
        }
        .validate()
        .is_err());
        assert!(GauntletConfig {
            epsilon: 1.5,
            ..base.clone()
        }
        .validate()
        .is_err());
        assert!(GauntletConfig {
            max_turns: 0,
            ..base.clone()
        }
        .validate()
        .is_err());
        assert!(base.validate().is_ok());

        // The 7x7 test board plus an MCTS side is the combination that would
        // otherwise panic inside a worker thread.
        let mcts = GauntletConfig {
            side_a: spec(Engine::Mcts, Budget::Nodes(8)),
            ..base.clone()
        };
        let error = mcts.validate().expect_err("7x7 MCTS must be refused");
        assert!(error.0.contains("12x12"), "{error}");
        assert!(GauntletConfig {
            rows: 12,
            cols: 12,
            ..mcts
        }
        .validate()
        .is_ok());
    }

    /// Epsilon-zero means the opening RNG never fires, which is the mode a
    /// pure "who plays this position better" comparison wants.
    #[test]
    fn epsilon_zero_replays_one_opening() {
        let config = GauntletConfig {
            epsilon: 0.0,
            ..small(
                spec(Engine::AlphaBeta { enhanced: true }, Budget::Nodes(300)),
                spec(Engine::Greedy, Budget::Nodes(1)),
                4,
            )
        };
        let result = run(&config, None).expect("run");
        // Both pairs play the identical game, because nothing differs between
        // them once the opening noise is off.
        assert_eq!(result.games[0].plies, result.games[2].plies);
        assert_eq!(result.games[0].winner, result.games[2].winner);
    }
}
