//! The polymorphic side: what a gauntlet arm actually is.
//!
//! A gauntlet compares two *configurations*, not two crates. The comparisons
//! this project needs are all of the form "same everything except one knob":
//! enhanced alpha-beta vs plain alpha-beta at an equal node budget; MCTS vs
//! alpha-beta at equal wall clock; anything at all vs the greedy floor to prove
//! the harness can tell strong from weak. So a side is
//! [`SideSpec`] = engine choice + [`Budget`], and both are per-side.
//!
//! # Per-side budgets are not a nicety
//!
//! Java's `GauntletMatch` has one `Config` shared by both arms, which makes
//! "MCTS at 1 s vs alpha-beta at 1 s" expressible and "MCTS at 400 sims vs
//! alpha-beta at 60 k nodes" not. The second shape is the one that runs
//! overnight without a clock, so it is the one the nightly ladder wants. Here
//! each side carries its own budget and the harness never picks one for you.
//!
//! # Fixed time is a first-class mode
//!
//! `docs/plans/superiority.md` S0: the Java gauntlet lacked a wall-clock mode
//! and that gap cost the search-strength work its first step — every number it
//! produced was in nodes, and nodes are not comparable across engines with
//! different node costs. [`Budget::Millis`] is therefore not an afterthought:
//! every engine here enforces it as a hard per-move deadline and reports its
//! worst overrun, so a side that quietly ignores the clock shows up in the
//! report instead of silently winning on time.

use std::time::{Duration, Instant};
use virus_core::{Action, CellKind, Player, State};
use virus_mcts::{Config as MctsConfig, MctsSearcher, PolicyValueNet, ValueSource};
use virus_search::{SearchOptions, Searcher};

/// How much a side may spend on one action.
///
/// One action, not one turn: a turn is three actions and the server arms its
/// timer per action, so the per-action budget is the unit both predecessors
/// used and the unit that compares to them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Budget {
    /// A search-node ceiling. Deterministic: no clock is consulted anywhere, so
    /// two runs at the same seeds produce byte-identical games. This is the
    /// only mode the determinism gate accepts.
    ///
    /// For [`Engine::Mcts`] the unit is simulations rather than alpha-beta
    /// nodes — those are the two engines' respective atoms of work, and there
    /// is no honest conversion between them. Never read a node-mode gauntlet
    /// across engine families as an equal-compute comparison; use
    /// [`Budget::Millis`] for that.
    Nodes(u64),
    /// A fixed nominal search depth. Alpha-beta only; MCTS has no depth knob
    /// and [`SideSpec::validate`] rejects the combination up front rather than
    /// letting it surface as a wrong move mid-match.
    Depth(i32),
    /// A wall-clock ceiling per action, enforced as a deadline.
    ///
    /// **Not deterministic** — a machine under load searches fewer nodes. That
    /// is inherent to the mode, not a defect, but it means a fixed-time result
    /// cannot be reproduced byte for byte and must never be used as the
    /// determinism gate.
    Millis(u64),
}

impl Budget {
    /// The wall-clock ceiling, when this is a timed budget.
    pub fn deadline(self) -> Option<Duration> {
        match self {
            Budget::Millis(ms) => Some(Duration::from_millis(ms)),
            _ => None,
        }
    }

    /// The short tag used in side names, matching Java's `n60000` / `d4` /
    /// `1000ms` provenance strings.
    pub fn tag(self) -> String {
        match self {
            Budget::Nodes(n) => format!("n{n}"),
            Budget::Depth(d) => format!("d{d}"),
            Budget::Millis(ms) => format!("{ms}ms"),
        }
    }

    /// Parses `nodes:60000`, `depth:4` or `ms:1000`.
    pub fn parse(text: &str) -> Result<Budget, SpecError> {
        let (kind, value) = text
            .split_once(':')
            .ok_or_else(|| SpecError(format!("budget {text:?} must look like nodes:60000")))?;
        let bad = |_| SpecError(format!("budget {text:?} has a non-numeric amount"));
        match kind {
            "nodes" | "n" => Ok(Budget::Nodes(value.parse().map_err(bad)?)),
            "depth" | "d" => Ok(Budget::Depth(value.parse().map_err(bad)?)),
            "ms" | "millis" => Ok(Budget::Millis(value.parse().map_err(bad)?)),
            other => Err(SpecError(format!(
                "unknown budget kind {other:?}; expected nodes, depth or ms"
            ))),
        }
    }
}

/// Java's `GauntletMatch.Config.nodeLimit` default, kept so a node-mode run
/// here is directly comparable to a recorded Java one.
pub const DEFAULT_NODE_LIMIT: u64 = 60_000;

/// Which searcher a side runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Engine {
    /// First capture, else first legal action. The floor: an arm that cannot
    /// beat this convincingly is broken, not merely weak.
    Greedy,
    /// Alpha-beta over the hand-tuned eval. `enhanced == false` is the
    /// byte-exact GoBot oracle; `true` is the full strength stack.
    AlphaBeta {
        /// Whether to enable the enhanced stack (TT, killers, history, LMR,
        /// aspiration windows).
        enhanced: bool,
    },
    /// PUCT + the conv policy/value net.
    Mcts,
}

/// A complete side: which engine, how much it may spend, and — for MCTS —
/// which artifact it plays.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SideSpec {
    /// The searcher.
    pub engine: Engine,
    /// Its per-action budget.
    pub budget: Budget,
    /// Net artifact path. Required by [`Engine::Mcts`], ignored otherwise.
    pub net: Option<String>,
}

/// A malformed side or budget specification.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpecError(pub String);

impl std::fmt::Display for SpecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for SpecError {}

impl SideSpec {
    /// Parses `greedy`, `ab-plain`, `ab-enhanced`, `mcts` or `mcts:<path>`.
    pub fn parse(text: &str, budget: Budget) -> Result<SideSpec, SpecError> {
        let (engine, net) = match text.split_once(':') {
            Some(("mcts", path)) => (Engine::Mcts, Some(path.to_owned())),
            _ => match text {
                "greedy" => (Engine::Greedy, None),
                "ab-plain" => (Engine::AlphaBeta { enhanced: false }, None),
                "ab-enhanced" | "ab" => (Engine::AlphaBeta { enhanced: true }, None),
                "mcts" => (Engine::Mcts, None),
                other => {
                    return Err(SpecError(format!(
                        "unknown side {other:?}; expected greedy, ab-plain, ab-enhanced or \
                         mcts[:path]"
                    )))
                }
            },
        };
        let spec = SideSpec {
            engine,
            budget,
            net,
        };
        spec.validate()?;
        Ok(spec)
    }

    /// Rejects combinations that have no meaning, before any game is played.
    ///
    /// Up front rather than at the first move: a gauntlet that dies 40 minutes
    /// in with "MCTS has no depth" has already burned the run.
    pub fn validate(&self) -> Result<(), SpecError> {
        match (self.engine, self.budget) {
            (Engine::Mcts, Budget::Depth(_)) => Err(SpecError(
                "MCTS has no fixed-depth mode; use nodes: (simulations) or ms:".to_owned(),
            )),
            (_, Budget::Nodes(0)) => {
                Err(SpecError("a node budget of 0 searches nothing".to_owned()))
            }
            (_, Budget::Millis(0)) => Err(SpecError(
                "a time budget of 0 ms searches nothing".to_owned(),
            )),
            (_, Budget::Depth(d)) if d <= 0 => {
                Err(SpecError(format!("depth {d} must be at least 1")))
            }
            _ => Ok(()),
        }
    }

    /// Whether this side needs a loaded net artifact.
    pub fn needs_net(&self) -> bool {
        self.engine == Engine::Mcts
    }

    /// The provenance name, in Java's `gauntlet:<eval>:<budget>` shape.
    ///
    /// An MCTS side carries its artifact's file stem, because "mcts" alone
    /// answers the wrong question: every generation of the ladder is "mcts",
    /// and a benchmarks table that does not say *which* net played is a table
    /// nobody can reproduce a year later.
    pub fn name(&self) -> String {
        let engine = match self.engine {
            Engine::Greedy => "greedy".to_owned(),
            Engine::AlphaBeta { enhanced: true } => "ab-enhanced".to_owned(),
            Engine::AlphaBeta { enhanced: false } => "ab-plain".to_owned(),
            Engine::Mcts => match self.net.as_deref().map(artifact_stem) {
                Some(stem) => format!("mcts[{stem}]"),
                None => "mcts".to_owned(),
            },
        };
        format!("{engine}:{}", self.budget.tag())
    }
}

/// The file stem of an artifact path, for use in a side's name.
///
/// Hand-rolled rather than `Path::file_stem` so the result is always valid
/// UTF-8 and never an `OsStr` dance; a name is display text.
fn artifact_stem(path: &str) -> &str {
    let file = path.rsplit(['/', '\\']).next().unwrap_or(path);
    file.strip_suffix(".json").unwrap_or(file)
}

/// Per-move telemetry a side reports back to the harness.
#[derive(Clone, Copy, Debug, Default)]
pub struct MoveStats {
    /// Search nodes (alpha-beta) or simulations (MCTS) spent on this action.
    pub work: u64,
    /// How far past its deadline the move ran. Zero outside fixed-time mode
    /// and, ideally, in it too.
    pub overrun: Duration,
}

/// One arm of a gauntlet, playing one seat of one game.
///
/// A fresh instance per game per seat: alpha-beta's transposition table is
/// reused *within* a seat's moves (which is the enhanced stack's cross-move
/// reuse feature and must be exercised) but never across games, because a table
/// carried between games would make game `n`'s result depend on game `n-1` and
/// destroy the harness's game-level independence — and with it any
/// parallelism-invariance claim.
pub trait Side: Send {
    /// Picks an action for `state`, whose mover is this side's seat.
    ///
    /// `None` means the position has no legal action; the harness ends the game
    /// rather than guessing.
    fn choose(&mut self, state: &State) -> (Option<Action>, MoveStats);
}

/// Builds a side for one seat of one game.
///
/// `net` is borrowed rather than owned so a whole gauntlet shares one loaded
/// artifact across every game and thread — loading the 700 KB champion per game
/// dominated the run time of the first version of this.
pub fn build<'net>(
    spec: &SideSpec,
    seat: Player,
    net: Option<&'net PolicyValueNet>,
) -> Result<Box<dyn Side + 'net>, SpecError> {
    spec.validate()?;
    Ok(match spec.engine {
        Engine::Greedy => Box::new(GreedySide),
        Engine::AlphaBeta { enhanced } => Box::new(AlphaBetaSide::new(enhanced, spec.budget)),
        Engine::Mcts => {
            let net = net
                .ok_or_else(|| SpecError("an MCTS side needs a loaded net artifact".to_owned()))?;
            Box::new(MctsSide {
                budget: spec.budget,
                net,
                seat,
            })
        }
    })
}

/// The reference floor: take the first capture, else the first legal action.
///
/// Reimplemented rather than borrowed from `virus-proto`: CLAUDE.md fixes the
/// dependency direction as "proto and arena depend on the engine crates, never
/// the reverse", and arena depending on proto would make the harness drag in a
/// WebSocket client and a tokio runtime to run an offline match. The behaviour
/// is pinned to proto's by a test.
#[derive(Debug)]
struct GreedySide;

impl Side for GreedySide {
    fn choose(&mut self, state: &State) -> (Option<Action>, MoveStats) {
        let actions = state.legal_actions();
        let mover = state.current_player();
        let capture = actions.iter().copied().find(|action| match *action {
            Action::Move { target } => {
                let cell = state.at(target);
                cell.kind() == CellKind::Normal && cell.owner() != mover
            }
            Action::PlaceNeutrals { .. } => false,
        });
        let action = capture.or_else(|| actions.first().copied());
        (
            action,
            MoveStats {
                work: actions.len() as u64,
                overrun: Duration::ZERO,
            },
        )
    }
}

/// An alpha-beta arm holding one persistent searcher for its seat.
struct AlphaBetaSide {
    /// Built on the first move rather than up front. A `Searcher` fixes its
    /// root player from the state it is constructed with, and at game start the
    /// mover is always seat 1 — building both sides there would hand seat 2 a
    /// searcher scoring every position from seat 1's chair, which is not a
    /// weaker engine but a *wrong* one, and it would have been invisible in the
    /// tally.
    searcher: Option<Searcher>,
    options: SearchOptions,
    budget: Budget,
    /// `false` for the plain oracle, which Java rebuilds per move; keeping the
    /// distinction means a plain-vs-enhanced A/B measures the enhancements
    /// including cross-move table reuse, which is one of them.
    persistent: bool,
}

impl AlphaBetaSide {
    fn new(enhanced: bool, budget: Budget) -> AlphaBetaSide {
        let options = if enhanced {
            // SMP is off in `SearchOptions::default`, and the harness never
            // turns it on: helper threads write the shared table, so a search
            // with SMP on returns a different move run to run and the
            // determinism gate would be measuring the scheduler.
            SearchOptions::default()
        } else {
            SearchOptions::plain()
        };
        debug_assert_eq!(options.smp_threads, 0, "the arena requires SMP off");
        AlphaBetaSide {
            searcher: None,
            options,
            budget,
            persistent: enhanced,
        }
    }
}

impl Side for AlphaBetaSide {
    fn choose(&mut self, state: &State) -> (Option<Action>, MoveStats) {
        // The clock starts *before* the searcher is built, not after.
        //
        // `Searcher::new` is not free: the enhanced arm allocates a 32 MiB
        // packed transposition table on its first move of a game, and the plain
        // arm rebuilds its searcher on every single move. Starting the timer
        // afterwards would hand alpha-beta that work outside the budget it is
        // being compared under, and hide it from the overrun telemetry too — so
        // the one arm whose setup cost is unmetered would be the one this
        // harness is trying to measure against MCTS at equal wall clock.
        let started = Instant::now();
        // The oracle is stateless by construction: Java's non-enhanced arm
        // builds a fresh searcher per move, and a plain arm carrying a table
        // between moves would not be the oracle any more. Plain mode allocates
        // no packed table, so this is cheap.
        let reuse = self.persistent
            && self
                .searcher
                .as_ref()
                .is_some_and(|searcher| searcher.root_player() == state.current_player());
        if !reuse {
            self.searcher = Some(Searcher::new(state, self.options));
        }
        let searcher = self.searcher.as_mut().expect("just built");
        let result = match self.budget {
            Budget::Nodes(limit) => searcher.search_node_budget(state, limit),
            Budget::Depth(depth) => searcher.search_to_depth(state, depth),
            Budget::Millis(ms) => {
                let deadline = started + Duration::from_millis(ms);
                searcher.search_with_deadline(state, deadline)
            }
        };
        let elapsed = started.elapsed();
        let overrun = match self.budget.deadline() {
            Some(limit) => elapsed.saturating_sub(limit),
            None => Duration::ZERO,
        };
        match result {
            Some(result) => (
                result.action,
                MoveStats {
                    work: result.nodes,
                    overrun,
                },
            ),
            None => (None, MoveStats::default()),
        }
    }
}

/// A PUCT arm. A fresh tree per move: `MctsSearcher` roots at a fixed position,
/// and tree reuse across moves is an engine feature, not a harness one.
struct MctsSide<'net> {
    budget: Budget,
    net: &'net PolicyValueNet,
    seat: Player,
}

impl Side for MctsSide<'_> {
    fn choose(&mut self, state: &State) -> (Option<Action>, MoveStats) {
        debug_assert_eq!(
            state.current_player(),
            self.seat,
            "asked to move out of turn"
        );
        let started = Instant::now();
        // Play mode: no Dirichlet noise, no visit sampling, no RNG draws at
        // all. The seed is inert here and the searcher is a pure function of
        // the position — which is what lets a node-mode gauntlet reproduce.
        //
        // `ValueSource::Net` is not a tuning knob. `Config::play()` defaults the
        // leaf value to the hand-tuned eval, which would leave the artifact
        // supplying only the priors — an engine that is neither the champion nor
        // the hand-tuned bar, reported under the champion's name. It is also
        // several times cheaper per simulation, so the mistake surfaces as a
        // *flattering* sims/s figure rather than an obviously broken one.
        // `virus-mcts`'s own crate-level example loads the gen-5 champion
        // exactly this way, and `ValueSource::Net` falls back to the hand-tuned
        // value by itself when an artifact has no value head.
        let config = MctsConfig {
            value_source: ValueSource::Net,
            ..MctsConfig::play()
        };
        let mut searcher = MctsSearcher::new(state.clone(), config, Some(self.net));
        match self.budget {
            Budget::Nodes(sims) => searcher.run_sims(sims.min(u64::from(u32::MAX)) as u32),
            Budget::Millis(ms) => {
                searcher.run_until_deadline(started + Duration::from_millis(ms));
            }
            // Rejected by `SideSpec::validate` before any game starts.
            Budget::Depth(_) => unreachable!("MCTS depth budgets are rejected at validation"),
        }
        let elapsed = started.elapsed();
        let overrun = match self.budget.deadline() {
            Some(limit) => elapsed.saturating_sub(limit),
            None => Duration::ZERO,
        };
        let action = searcher.best_action();
        (
            action,
            MoveStats {
                work: searcher.sims_run(),
                overrun,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fresh() -> State {
        State::new(12, 12, 2).expect("12x12 two-player")
    }

    #[test]
    fn budgets_parse_from_the_cli_form() {
        assert_eq!(Budget::parse("nodes:60000"), Ok(Budget::Nodes(60_000)));
        assert_eq!(Budget::parse("n:60000"), Ok(Budget::Nodes(60_000)));
        assert_eq!(Budget::parse("depth:4"), Ok(Budget::Depth(4)));
        assert_eq!(Budget::parse("d:4"), Ok(Budget::Depth(4)));
        assert_eq!(Budget::parse("ms:1000"), Ok(Budget::Millis(1000)));
        assert_eq!(Budget::parse("millis:1000"), Ok(Budget::Millis(1000)));
        assert!(Budget::parse("nodes").is_err());
        assert!(Budget::parse("weeks:3").is_err());
        assert!(Budget::parse("nodes:many").is_err());
    }

    /// `tag` is the display form Java's provenance strings use
    /// (`gauntlet:<eval>:n60000`), deliberately not the CLI form — it goes into
    /// report headers and recorded player names, where `nodes:60000` reads
    /// worse and, in the games.db case, is ambiguous against the `:` separator.
    #[test]
    fn tags_match_the_java_provenance_spelling() {
        assert_eq!(Budget::Nodes(60_000).tag(), "n60000");
        assert_eq!(Budget::Depth(4).tag(), "d4");
        assert_eq!(Budget::Millis(1000).tag(), "1000ms");
    }

    #[test]
    fn side_specs_parse_to_the_expected_engines() {
        let nodes = Budget::Nodes(100);
        assert_eq!(
            SideSpec::parse("ab-plain", nodes).expect("plain").engine,
            Engine::AlphaBeta { enhanced: false }
        );
        assert_eq!(
            SideSpec::parse("ab-enhanced", nodes)
                .expect("enhanced")
                .engine,
            Engine::AlphaBeta { enhanced: true }
        );
        assert_eq!(
            SideSpec::parse("ab", nodes).expect("ab").engine,
            Engine::AlphaBeta { enhanced: true }
        );
        assert_eq!(
            SideSpec::parse("greedy", nodes).expect("greedy").engine,
            Engine::Greedy
        );
        let mcts = SideSpec::parse("mcts:artifacts/x.json", nodes).expect("mcts");
        assert_eq!(mcts.engine, Engine::Mcts);
        assert_eq!(mcts.net.as_deref(), Some("artifacts/x.json"));
        assert!(SideSpec::parse("stockfish", nodes).is_err());
    }

    /// The whole point of validating up front: none of these may be discovered
    /// forty minutes into a run.
    #[test]
    fn meaningless_combinations_are_refused_before_any_game() {
        assert!(SideSpec::parse("mcts", Budget::Depth(4)).is_err());
        assert!(SideSpec::parse("ab-enhanced", Budget::Nodes(0)).is_err());
        assert!(SideSpec::parse("ab-enhanced", Budget::Millis(0)).is_err());
        assert!(SideSpec::parse("ab-enhanced", Budget::Depth(0)).is_err());
        assert!(SideSpec::parse("ab-enhanced", Budget::Depth(-1)).is_err());
        // Depth is fine for alpha-beta.
        assert!(SideSpec::parse("ab-enhanced", Budget::Depth(2)).is_ok());
    }

    #[test]
    fn names_carry_engine_and_budget() {
        let spec = SideSpec::parse("ab-enhanced", Budget::Nodes(60_000)).expect("spec");
        assert_eq!(spec.name(), "ab-enhanced:n60000");
        let spec = SideSpec::parse("mcts", Budget::Millis(1000)).expect("spec");
        assert_eq!(spec.name(), "mcts:1000ms");
    }

    /// "mcts" alone does not identify a side: every generation of the ladder is
    /// "mcts". A report has to name the artifact that actually played.
    #[test]
    fn an_mcts_side_names_its_artifact() {
        let spec = SideSpec::parse("mcts:artifacts/mcts_champion.json", Budget::Millis(1000))
            .expect("spec");
        assert_eq!(spec.name(), "mcts[mcts_champion]:1000ms");

        assert_eq!(
            artifact_stem("artifacts/mcts_champion.json"),
            "mcts_champion"
        );
        assert_eq!(artifact_stem("/a/b/gen7.json"), "gen7");
        assert_eq!(artifact_stem("bare"), "bare");
        assert_eq!(artifact_stem("net.json"), "net");
    }

    #[test]
    fn an_mcts_side_without_a_net_is_an_error_not_a_panic() {
        let spec = SideSpec::parse("mcts", Budget::Nodes(8)).expect("spec");
        assert!(build(&spec, 1, None).is_err());
    }

    /// The arena's greedy floor must be the same engine `virus-proto` exposes,
    /// or "greedy" means two different things in two different reports.
    #[test]
    fn the_greedy_floor_matches_the_documented_rule() {
        let state = fresh();
        let mut side = GreedySide;
        let (action, stats) = side.choose(&state);
        let expected = {
            let actions = state.legal_actions();
            let mover = state.current_player();
            actions
                .iter()
                .copied()
                .find(|action| match *action {
                    Action::Move { target } => {
                        let cell = state.at(target);
                        cell.kind() == CellKind::Normal && cell.owner() != mover
                    }
                    Action::PlaceNeutrals { .. } => false,
                })
                .or_else(|| actions.first().copied())
        };
        assert_eq!(action, expected);
        assert_eq!(stats.work, state.legal_actions().len() as u64);
    }

    /// SMP writes the shared table from helper threads, so a search with it on
    /// is not reproducible. Every arena arm must have it off.
    #[test]
    fn alpha_beta_arms_never_enable_smp() {
        for enhanced in [true, false] {
            let side = AlphaBetaSide::new(enhanced, Budget::Nodes(100));
            assert_eq!(side.options.smp_threads, 0);
        }
    }

    #[test]
    fn a_node_budgeted_alpha_beta_arm_returns_a_legal_action() {
        let state = fresh();
        let legal = state.legal_actions();
        for enhanced in [true, false] {
            let mut side = AlphaBetaSide::new(enhanced, Budget::Nodes(2_000));
            let (action, stats) = side.choose(&state);
            let action = action.expect("a legal action exists");
            assert!(legal.contains(&action), "{action:?}");
            assert_eq!(stats.overrun, Duration::ZERO, "node mode consults no clock");
        }
    }

    #[test]
    fn a_depth_budgeted_arm_returns_a_legal_action() {
        let state = fresh();
        let mut side = AlphaBetaSide::new(false, Budget::Depth(2));
        let (action, _) = side.choose(&state);
        assert!(state.legal_actions().contains(&action.expect("action")));
    }

    /// A searcher takes its root player from the state it is built with, so a
    /// side that never moves first must not be built at game start. This is the
    /// regression test for that: seat 2's searcher must be rooted at seat 2.
    #[test]
    fn a_second_seat_arm_roots_at_its_own_seat() {
        let state = fresh();
        // Advance to seat 2's turn: three actions is one whole turn.
        let mut state = state;
        for _ in 0..3 {
            let action = state.legal_actions()[0];
            state = state.apply(action).expect("legal");
        }
        assert_eq!(state.current_player(), 2, "seat 2 should be on move");
        let mut side = AlphaBetaSide::new(true, Budget::Nodes(2_000));
        let (action, _) = side.choose(&state);
        assert!(state.legal_actions().contains(&action.expect("action")));
        assert_eq!(
            side.searcher.as_ref().expect("built").root_player(),
            2,
            "seat 2's searcher must score from seat 2's chair"
        );
    }
}
