//! The searcher itself: two modes behind one implementation.
//!
//! * **Plain** ([`SearchOptions::plain`]) is a literal port of Go's
//!   `search.go` — unbounded map transposition table, ply-exact probing, no
//!   heuristics. It is the *oracle*: [`crate::choose_depth`] and a plain
//!   [`Searcher::search_node_budget`] reproduce
//!   `fixtures/gobot_search_parity.jsonl` and
//!   `fixtures/gobot_nodebudget_parity.jsonl` move- and score-exact.
//! * **Enhanced** (the default) is a port of Java's `GoBotSearcher` — the
//!   measured strength stack: staged movegen, the packed lockless TT, killers,
//!   history, turn-aware LMR, aspiration windows, lazy SMP, soft deadlines and
//!   partial-iteration salvage.
//!
//! Every enhancement is gated on [`SearchOptions::enhanced`], so the oracle
//! stays reachable forever and a strength regression can always be bisected
//! against it.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::sync::Arc;
use std::time::Instant;

use virus_core::{Action, CellKind, Player, Position, Scratch, State};
use virus_eval::{EvalParams, EvalWorkspace, Score, MATE_SCORE};

use crate::book::opening_book_move;
use crate::tt::{self, TranspositionTable};

/// Deepest iteration the iterative-deepening loops will attempt.
pub const MAX_DEPTH: i32 = 64;

/// The alpha-beta window bound. Go's `infScore`; deliberately far outside the
/// mate band so a window bound can never be mistaken for a score.
pub const INF_SCORE: Score = 1 << 60;

/// Don't *start* an iteration once this share of the time budget is gone.
pub const SOFT_DEADLINE_PERCENT: u32 = 55;

/// Aspiration half-window, in hand-tuned eval units.
///
/// The measured median best-vs-second-best root score gap is 1299, so +/-1500
/// brackets a typical between-iteration swing: most iterations complete inside
/// the window and pay a narrow search instead of a full-window one. Java's sweep
/// found 1500 the local optimum at production depth (-8.8% nodes at depth 6);
/// 750 and 3000 were both worse.
pub const ASPIRATION_DELTA: Score = 1500;

/// Hard cap on lazy-SMP threads. The machine also runs gauntlets.
pub const SMP_THREAD_CAP: usize = 4;

const FLAG_EXACT: u8 = 0;
const FLAG_LOWER: u8 = 1;
const FLAG_UPPER: u8 = 2;

// Move-ordering tiers. The gaps are load-bearing: the LMR guard below keys off
// "order < ORDER_CAPTURE" meaning "a genuinely quiet move", which is only exact
// because history is capped below the capture tier.
const ORDER_TT: i32 = 10_000_000;
const ORDER_KILLER: i32 = 5_000_000;
const ORDER_WIN: i32 = 1_000_000;
const ORDER_ELIMINATION: i32 = 100_000;
const ORDER_CAPTURE: i32 = 10_000;
const ORDER_HISTORY_CAP: i32 = 9_000;
const ORDER_TURN_CONTINUATION: i32 = 100;

// The tiers must stay strictly ordered and history must stay strictly below the
// capture tier: the LMR guard reads `child.order < ORDER_CAPTURE` as "this is a
// genuinely quiet move", which is only exact while that holds. Checked at
// compile time so a retune cannot quietly invalidate the guard.
const _: () = assert!(ORDER_TURN_CONTINUATION < ORDER_HISTORY_CAP);
const _: () = assert!(ORDER_HISTORY_CAP < ORDER_CAPTURE);
const _: () = assert!(ORDER_CAPTURE < ORDER_ELIMINATION);
const _: () = assert!(ORDER_ELIMINATION < ORDER_WIN);
const _: () = assert!(ORDER_WIN < ORDER_KILLER);
const _: () = assert!(ORDER_KILLER < ORDER_TT);

/// Ceiling on a stored history counter (the *ordering* contribution is capped at
/// [`ORDER_HISTORY_CAP`]; this only stops the accumulator overflowing).
const HISTORY_STORE_CAP: i32 = 1 << 28;

/// A scout is only reduced once it is this late among the children searched.
const LMR_LATE_AFTER: i32 = 4;

/// Interior nodes shallower than this are never reduced.
const LMR_MIN_DEPTH: i32 = 3;

/// Aspiration is off below this depth: shallow scores swing too much to bracket.
const ASPIRATION_MIN_DEPTH: i32 = 3;

const MAX_ALTERNATIVES: usize = 4;

/// How many nodes between wall-clock reads. `Instant::now` costs ~20 ns, which
/// at the node rates this search reaches would be a double-digit percentage of
/// the whole search; the resulting deadline granularity is far below the
/// server's 120 s move timer. Deterministic paths never read the clock at all.
const CLOCK_CHECK_INTERVAL: u32 = 1024;

/// Signals that the running budget (node limit, deadline, or an SMP abort) was
/// exhausted mid-search. Only the iterative-deepening loops catch it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Incomplete;

type Searched<T> = Result<T, Incomplete>;

/// A root candidate action with the score the search gave it.
///
/// Diagnostics only — populating this never changes the chosen action, score,
/// node count or depth.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RootMove {
    /// The candidate.
    pub action: Action,
    /// Its score. `exact == false` means this is a scout *bound*, not a value.
    pub score: Score,
    /// Whether `score` is an exact value rather than a fail-low bound.
    pub exact: bool,
}

/// The outcome of a search.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SearchResult {
    /// The chosen action. `None` only for a position with no legal action.
    pub action: Option<Action>,
    /// Root-relative score for the searcher's root player.
    pub score: Score,
    /// Depth of the deepest fully completed iteration (`0` = none, or a book move).
    pub depth: i32,
    /// Interior nodes visited.
    pub nodes: u64,
    /// Leaf evaluations performed.
    pub evaluations: u64,
    /// Whether a node budget was reached.
    pub budget_exhausted: bool,
    /// Whether the search ran out of depth rather than out of budget.
    pub search_complete: bool,
    /// Whether this came from a partially completed iteration (see
    /// [`Searcher::search_with_deadline`]).
    pub salvaged: bool,
    /// Whether the opening book supplied the move.
    pub book: bool,
    /// Next-best root candidates, best first. Diagnostics only.
    pub alternatives: Vec<RootMove>,
}

impl SearchResult {
    fn empty() -> SearchResult {
        SearchResult {
            action: None,
            score: 0,
            depth: 0,
            nodes: 0,
            evaluations: 0,
            budget_exhausted: false,
            search_complete: false,
            salvaged: false,
            book: false,
            alternatives: Vec::new(),
        }
    }

    fn with_action(action: Action) -> SearchResult {
        SearchResult {
            action: Some(action),
            ..SearchResult::empty()
        }
    }
}

/// Diagnostic counters. None of these feed a search decision.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SearchStats {
    /// Cut-nodes resolved by the TT move alone, without materialising siblings.
    pub fast_path_cuts: u64,
    /// Enhanced-TT probes.
    pub tt_probes: u64,
    /// Enhanced-TT probes that found an entry (at any depth).
    pub tt_hits: u64,
    /// Interior-node beta cutoffs.
    pub cutoffs: u64,
    /// Sum of the searched-child index at each cutoff; the mean is an ordering-
    /// quality readout.
    pub cutoff_index_sum: u64,
    /// Aspirated iterations that failed low and were re-searched.
    pub aspiration_fail_lows: u64,
    /// Aspirated iterations that failed high and were re-searched.
    pub aspiration_fail_highs: u64,
    /// Scouts searched one action shallower.
    pub lmr_reductions: u64,
    /// Reduced scouts that failed high and were re-run at full depth.
    pub lmr_re_searches: u64,
    /// **Tripwire.** Reductions applied to a turn-*ending* action. Must stay 0:
    /// a reduced leaf on the far side of a turn boundary lands in the other
    /// side's turn fragment with a systematically different tempo term, which is
    /// exactly where eval error concentrates.
    pub lmr_turn_ending_reductions: u64,
}

/// Everything that distinguishes the oracle from the production searcher.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SearchOptions {
    /// Master switch for the whole strength stack. `false` is the byte-exact
    /// GoBot oracle.
    pub enhanced: bool,
    /// Aspiration windows (enhanced only).
    pub aspiration: bool,
    /// Turn-aware late-move reduction (enhanced only).
    pub lmr: bool,
    /// Lazy-SMP helper count *including* the main thread. `0` or `1` is off.
    ///
    /// Defaults to off: helper TT entries steer the main tree, so a search with
    /// SMP on is not reproducible run to run. Every deterministic gate — both
    /// parity fixtures, the deadline-consistency test, the fixed-node gauntlets
    /// — must keep it off.
    pub smp_threads: usize,
    /// `log2` of the packed table size. 21 is 2^21 entries (32 MiB).
    pub tt_log2: u32,
    /// Don't start a new iteration past this share of the time budget.
    pub soft_deadline_percent: u32,
    /// Aspiration half-window.
    pub aspiration_delta: Score,
    /// Leaf evaluation weights.
    pub params: EvalParams,
}

impl Default for SearchOptions {
    fn default() -> SearchOptions {
        SearchOptions {
            enhanced: true,
            aspiration: true,
            lmr: true,
            smp_threads: 0,
            tt_log2: tt::DEFAULT_LOG2_SIZE,
            soft_deadline_percent: SOFT_DEADLINE_PERCENT,
            aspiration_delta: ASPIRATION_DELTA,
            params: EvalParams::default(),
        }
    }
}

impl SearchOptions {
    /// The parity oracle: GoBot's `search.go` and nothing else.
    pub fn plain() -> SearchOptions {
        SearchOptions {
            enhanced: false,
            aspiration: false,
            lmr: false,
            smp_threads: 0,
            ..SearchOptions::default()
        }
    }
}

/// One materialised child: the action, the resulting state, and its ordering key.
#[derive(Clone, Debug)]
struct Child {
    action: Action,
    state: State,
    order: i32,
}

/// A per-ply child buffer. States are big, so the sort permutes a `u32` index
/// vector rather than moving `Child`s around, and both vectors are pooled by ply
/// so an interior node allocates nothing.
#[derive(Debug, Default)]
struct ChildBuf {
    children: Vec<Child>,
    order: Vec<u32>,
}

impl ChildBuf {
    fn is_empty(&self) -> bool {
        self.order.is_empty()
    }
}

/// A plain-mode transposition entry: GoBot's `tableEntry`, verbatim.
#[derive(Clone, Copy, Debug)]
struct MapEntry {
    depth: i32,
    ply: i32,
    flag: u8,
    best_action: Option<Action>,
    values: [Score; 4],
}

/// The searcher.
///
/// An enhanced searcher is meant to live for a whole game: the packed table
/// survives between moves (aged, not cleared), so each move starts warm on the
/// previous move's principal subtree. One instance per seat — root-relative
/// scores are only meaningful while the root player is the mover.
pub struct Searcher {
    root: Player,
    multi: bool,
    options: SearchOptions,

    /// Plain-mode table, and the `max_n` table in every mode. `max_n` only runs
    /// in 3-4 player games, which the 1v1 strength paths never reach; enhancing
    /// it would be untested dead weight.
    map_table: HashMap<u64, MapEntry>,
    tt: Option<Arc<TranspositionTable>>,

    scratch: Box<Scratch>,
    eval: EvalWorkspace,
    bufs: Vec<ChildBuf>,

    nodes: u64,
    evaluations: u64,
    node_limit: u64,
    deadline: Option<Instant>,
    clock_countdown: u32,
    timed_out: bool,
    stop: Option<Arc<AtomicBool>>,

    killers: Vec<[Option<Action>; 2]>,
    /// `[mover - 1][cell index]`; only the two 1v1 seats are tracked.
    history: [Vec<i32>; 2],

    stats: SearchStats,
    partial_root: Option<SearchResult>,
}

impl std::fmt::Debug for Searcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Searcher")
            .field("root", &self.root)
            .field("multi", &self.multi)
            .field("enhanced", &self.options.enhanced)
            .field("nodes", &self.nodes)
            .finish()
    }
}

impl Searcher {
    /// A searcher rooted at `state`'s mover.
    pub fn new(state: &State, options: SearchOptions) -> Searcher {
        Searcher::with_table(state, options, None)
    }

    /// A searcher rooted at `state`'s mover with the full strength stack.
    pub fn enhanced(state: &State) -> Searcher {
        Searcher::new(state, SearchOptions::default())
    }

    /// The GoBot oracle: no book, no heuristics, GoBot's exact table semantics.
    pub fn plain(state: &State) -> Searcher {
        Searcher::new(state, SearchOptions::plain())
    }

    fn with_table(
        state: &State,
        options: SearchOptions,
        shared: Option<Arc<TranspositionTable>>,
    ) -> Searcher {
        let active = active_count(state);
        Searcher {
            root: state.current_player(),
            multi: active > 2,
            tt: if options.enhanced {
                Some(shared.unwrap_or_else(|| Arc::new(TranspositionTable::new(options.tt_log2))))
            } else {
                None
            },
            options,
            map_table: HashMap::new(),
            scratch: Scratch::new(),
            eval: EvalWorkspace::new(),
            bufs: (0..=MAX_DEPTH as usize + 1)
                .map(|_| ChildBuf::default())
                .collect(),
            nodes: 0,
            evaluations: 0,
            node_limit: 0,
            deadline: None,
            clock_countdown: 0,
            timed_out: false,
            stop: None,
            killers: vec![[None, None]; MAX_DEPTH as usize + 1],
            history: [Vec::new(), Vec::new()],
            stats: SearchStats::default(),
            partial_root: None,
        }
    }

    /// The seat this searcher evaluates for, fixed at construction.
    pub fn root_player(&self) -> Player {
        self.root
    }

    /// The configuration in force.
    pub fn options(&self) -> &SearchOptions {
        &self.options
    }

    /// Diagnostic counters from the searches run so far.
    pub fn stats(&self) -> &SearchStats {
        &self.stats
    }

    /// Whether the packed table already holds an entry for `state`.
    pub fn tt_has_entry(&self, state: &State) -> bool {
        self.tt
            .as_ref()
            .is_some_and(|table| table.probe(state.hash()) != 0)
    }

    /// The move order the search would see at `ply`, best first.
    ///
    /// Diagnostics and tests only — it materialises every child, which is
    /// exactly what the staged fast path exists to avoid.
    pub fn ordered_actions(
        &mut self,
        state: &State,
        tt_move: Option<Action>,
        ply: i32,
    ) -> Vec<Action> {
        let Ok(buf) = self.ordered_children(state, tt_move, tt_move.is_some(), ply) else {
            return Vec::new();
        };
        let actions = buf
            .order
            .iter()
            .map(|&index| buf.children[index as usize].action)
            .collect();
        self.release(ply, buf);
        actions
    }

    /// Seeds a killer slot. Diagnostics and tests only.
    pub fn set_killer(&mut self, ply: i32, action: Action) {
        let slot = &mut self.killers[ply as usize];
        slot[1] = slot[0];
        slot[0] = Some(action);
    }

    /// Seeds the history counter for a cell. Diagnostics and tests only.
    pub fn set_history(&mut self, mover: Player, index: usize, value: i32) {
        let cells = self.history[mover as usize - 1].len();
        if index < cells {
            self.history[mover as usize - 1][index] = value;
        }
    }

    /// The history counter for a cell, or `0` when out of range.
    pub fn history_value(&self, mover: Player, index: usize) -> i32 {
        self.history
            .get(mover as usize - 1)
            .and_then(|side| side.get(index))
            .copied()
            .unwrap_or(0)
    }

    /// The largest history counter recorded for `mover`. Diagnostics only.
    pub fn history_peak(&self, mover: Player) -> i32 {
        self.history
            .get(mover as usize - 1)
            .and_then(|side| side.iter().copied().max())
            .unwrap_or(0)
    }

    /// Sizes the killer and history tables for `state` without running a
    /// search. Diagnostics and tests only.
    pub fn prepare_heuristics(&mut self, state: &State) {
        let cells = state.cell_count();
        for side in &mut self.history {
            side.clear();
            side.resize(cells, 0);
        }
    }

    // ---------------------------------------------------------------- entry points

    /// The enhanced iterative-deepening loop run to exactly `max_depth` with no
    /// budget.
    ///
    /// This is the deterministic oracle for what [`Searcher::search_with_deadline`]
    /// and [`Searcher::search_node_budget`] must return at the depth they report
    /// (ARCHITECTURE.md invariant 3). Skips the opening book, like `choose_depth`.
    ///
    /// # Panics
    /// Panics when `state`'s mover is not this searcher's root player.
    pub fn search_to_depth(&mut self, state: &State, max_depth: i32) -> Option<SearchResult> {
        self.begin_search(state, None, 0);
        let mut best: Option<SearchResult> = None;
        let mut previous = None;
        for depth in 1..=max_depth.min(MAX_DEPTH) {
            // No budget, so `Incomplete` is unreachable here.
            let Ok(mut result) = self.at_depth_aspirated(state, depth, previous) else {
                break;
            };
            result.depth = depth;
            result.nodes = self.nodes;
            result.evaluations = self.evaluations;
            previous = Some(result.score);
            best = Some(result);
        }
        best
    }

    /// Deterministic iterative deepening bounded by a node limit rather than a
    /// clock — Go's `ChooseNodeBudget`.
    ///
    /// Consults the opening book. Returns `None` when the position has no legal
    /// action or `limit == 0`. An aborted iteration is discarded outright: the
    /// answer is always the deepest fully completed one.
    ///
    /// # Panics
    /// Panics when `state`'s mover is not this searcher's root player.
    pub fn search_node_budget(&mut self, state: &State, limit: u64) -> Option<SearchResult> {
        self.assert_root(state);
        if let Some(book) = self.book_result(state) {
            return Some(book);
        }
        let fallback = preserving_fallback(state)?;
        if limit == 0 {
            return None;
        }
        self.begin_search(state, None, limit);
        let mut best = SearchResult::with_action(fallback);
        let stop = Arc::new(AtomicBool::new(false));
        let helper_count = self.helper_count();
        std::thread::scope(|scope| {
            self.spawn_helpers(scope, state, &stop, helper_count);
            let mut previous = None;
            let mut depth = 1;
            while depth <= MAX_DEPTH && self.nodes < limit {
                let Ok(mut result) = self.at_depth_aspirated(state, depth, previous) else {
                    break;
                };
                result.depth = depth;
                previous = Some(result.score);
                best = result;
                depth += 1;
            }
            stop.store(true, AtomicOrdering::Relaxed);
        });
        best.nodes = self.nodes;
        best.evaluations = self.evaluations;
        best.budget_exhausted = self.nodes >= limit;
        best.search_complete = best.depth == MAX_DEPTH;
        Some(best)
    }

    /// Iterative deepening bounded by a wall-clock deadline — Go's `Choose`.
    ///
    /// Consults the opening book. Returns `None` only when the position has no
    /// legal action.
    ///
    /// # Time management
    ///
    /// A new iteration costs roughly the effective branching factor times the
    /// last one, so past [`SearchOptions::soft_deadline_percent`] of the budget
    /// it would almost surely be cut — it is not started, and the time is kept.
    ///
    /// # The invariant that cost the Java bot a 0-10 live run
    ///
    /// ARCHITECTURE.md invariant 3: the returned move is the move of the deepest
    /// **fully completed** iteration. An abort mid-iteration unwinds out of the
    /// root loop before anything is committed, so a partially searched iteration
    /// can never override a complete one. The single exception is *salvage*, and
    /// it is guarded by `best_score > alpha_orig`: child 0 is the previous
    /// iteration's principal move (TT-ordered first), root best-move updates only
    /// happen on exact re-searched scores, and under an aspiration window an
    /// all-fail-low prefix is bounds only and refuses to salvage. A salvaged
    /// result is flagged [`SearchResult::salvaged`] and keeps the *completed*
    /// iteration's depth label.
    ///
    /// # Panics
    /// Panics when `state`'s mover is not this searcher's root player.
    pub fn search_with_deadline(
        &mut self,
        state: &State,
        deadline: Instant,
    ) -> Option<SearchResult> {
        self.assert_root(state);
        if let Some(book) = self.book_result(state) {
            return Some(book);
        }
        let fallback = preserving_fallback(state)?;
        let start = Instant::now();
        let budget = deadline.saturating_duration_since(start).as_nanos();
        self.begin_search(state, Some(deadline), 0);
        let mut best = SearchResult::with_action(fallback);
        let stop = Arc::new(AtomicBool::new(false));
        let helper_count = self.helper_count();
        let soft = u128::from(self.options.soft_deadline_percent);
        std::thread::scope(|scope| {
            self.spawn_helpers(scope, state, &stop, helper_count);
            let mut previous = None;
            for depth in 1..=MAX_DEPTH {
                if self.options.enhanced
                    && depth > 1
                    && budget > 0
                    && start.elapsed().as_nanos() * 100 > budget * soft
                {
                    break;
                }
                match self.at_depth_aspirated(state, depth, previous) {
                    Ok(mut result) => {
                        result.depth = depth;
                        result.nodes = self.nodes;
                        result.evaluations = self.evaluations;
                        previous = Some(result.score);
                        best = result;
                    }
                    Err(_) => {
                        if let Some(mut partial) = self.partial_root.take() {
                            // The label stays the last COMPLETED iteration.
                            partial.depth = best.depth;
                            partial.nodes = self.nodes;
                            partial.evaluations = self.evaluations;
                            best = partial;
                        }
                        break;
                    }
                }
            }
            stop.store(true, AtomicOrdering::Relaxed);
        });
        Some(best)
    }

    /// Iterative deepening for `budget` of wall-clock time.
    ///
    /// # Panics
    /// Panics when `state`'s mover is not this searcher's root player.
    pub fn search(&mut self, state: &State, budget: std::time::Duration) -> Option<SearchResult> {
        self.search_with_deadline(state, Instant::now() + budget)
    }

    // ---------------------------------------------------------------- lifecycle

    fn book_result(&mut self, state: &State) -> Option<SearchResult> {
        let action = opening_book_move(state)?;
        let player = state.current_player();
        // The book plays by fiat, but a real static eval of the resulting
        // position gives the diagnostics UI a meaningful number instead of zero.
        let score = match state.apply(action) {
            Ok(next) => virus_eval::evaluate(&next, player, &self.options.params, &mut self.eval),
            Err(_) => 0,
        };
        let mut result = SearchResult::with_action(action);
        result.score = score;
        result.search_complete = true;
        result.book = true;
        Some(result)
    }

    /// A persistent searcher's scores are root-relative, so handing it the other
    /// seat is a programming error rather than a recoverable condition.
    ///
    /// Checked *before* the opening book, not only inside [`Searcher::begin_search`]:
    /// the book fires on any fresh opening turn regardless of seat, so a
    /// book-eligible position would otherwise hand a P1-rooted searcher P2's
    /// wedge move and skip the check entirely. Java has the same short-circuit
    /// order and the same latent hole; an action played for the wrong seat is
    /// exactly the class of bug that forfeits a live game.
    fn assert_root(&self, state: &State) {
        assert_eq!(
            state.current_player(),
            self.root,
            "searcher rooted at player {} asked to move for {}",
            self.root,
            state.current_player()
        );
    }

    /// Per-call reset. Budgets and counters are per move; the packed table
    /// persists across moves, aged by one generation.
    fn begin_search(&mut self, state: &State, deadline: Option<Instant>, limit: u64) {
        self.assert_root(state);
        self.deadline = deadline;
        self.node_limit = limit;
        self.nodes = 0;
        self.evaluations = 0;
        self.clock_countdown = 0;
        self.timed_out = false;
        self.partial_root = None;
        if !self.options.enhanced {
            return;
        }
        if let Some(table) = &self.tt {
            table.bump_generation();
        }
        for slot in &mut self.killers {
            *slot = [None, None];
        }
        let cells = state.cell_count();
        for side in &mut self.history {
            if side.len() != cells {
                side.clear();
                side.resize(cells, 0);
            } else {
                // Age: last move's refutations fade rather than dominating forever.
                for value in side.iter_mut() {
                    *value >>= 1;
                }
            }
        }
    }

    fn helper_count(&self) -> usize {
        if !self.options.enhanced || self.multi {
            return 0;
        }
        let available = std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(1);
        let threads = self.options.smp_threads.min(SMP_THREAD_CAP).min(available);
        threads.saturating_sub(1)
    }

    /// Starts the lazy-SMP helpers.
    ///
    /// Each helper is its own searcher sharing only the XOR-verified packed
    /// table, running the same iterative deepening from the same root at a
    /// staggered start depth (the classic 2/3 skew). Helpers keep the FULL
    /// window: their scores are discarded, so there is no trusted centre to
    /// aspirate around, and full-window bounds stay valid for whatever window
    /// the main thread happens to be searching. Their only output is the table
    /// entries they leave behind.
    fn spawn_helpers<'scope, 'env: 'scope>(
        &self,
        scope: &'scope std::thread::Scope<'scope, 'env>,
        state: &'env State,
        stop: &'env Arc<AtomicBool>,
        count: usize,
    ) {
        let Some(table) = self.tt.clone() else {
            return;
        };
        let options = self.options;
        for index in 0..count {
            let table = table.clone();
            let stop = stop.clone();
            let start_depth = 2 + (index as i32 & 1);
            scope.spawn(move || {
                let mut helper = Searcher::with_table(state, options, Some(table));
                helper.stop = Some(stop);
                helper.history = [vec![0; state.cell_count()], vec![0; state.cell_count()]];
                for depth in start_depth..=MAX_DEPTH {
                    if helper
                        .at_depth(state, depth, -INF_SCORE, INF_SCORE)
                        .is_err()
                    {
                        break;
                    }
                }
            });
        }
    }

    fn running(&mut self) -> bool {
        if let Some(stop) = &self.stop {
            if stop.load(AtomicOrdering::Relaxed) {
                return false;
            }
        }
        if self.node_limit > 0 && self.nodes >= self.node_limit {
            return false;
        }
        let Some(deadline) = self.deadline else {
            return true;
        };
        if self.clock_countdown == 0 {
            self.clock_countdown = CLOCK_CHECK_INTERVAL;
            self.timed_out = Instant::now() >= deadline;
        } else {
            self.clock_countdown -= 1;
        }
        !self.timed_out
    }

    // ---------------------------------------------------------------- root

    /// One iteration, opened with an aspiration window around the previous
    /// iteration's score.
    ///
    /// On a fail the failing side widens to 4 delta, then to the full window —
    /// standard progressive widening, so the returned result is always exact.
    /// Mate-band scores bypass aspiration entirely: a window around a mate score
    /// is meaningless re-search churn.
    fn at_depth_aspirated(
        &mut self,
        state: &State,
        depth: i32,
        previous: Option<Score>,
    ) -> Searched<SearchResult> {
        let delta = self.options.aspiration_delta;
        let centre = match previous {
            Some(score)
                if self.options.enhanced
                    && !self.multi
                    && self.options.aspiration
                    && depth >= ASPIRATION_MIN_DEPTH
                    && score.abs() < tt::MATE_BAND =>
            {
                score
            }
            _ => return self.at_depth(state, depth, -INF_SCORE, INF_SCORE),
        };
        let mut alpha = centre - delta;
        let mut beta = centre + delta;
        let mut widened_low = false;
        let mut widened_high = false;
        loop {
            let result = self.at_depth(state, depth, alpha, beta)?;
            if result.score <= alpha {
                self.stats.aspiration_fail_lows += 1;
                alpha = if widened_low {
                    -INF_SCORE
                } else {
                    centre - 4 * delta
                };
                widened_low = true;
            } else if result.score >= beta {
                self.stats.aspiration_fail_highs += 1;
                beta = if widened_high {
                    INF_SCORE
                } else {
                    centre + 4 * delta
                };
                widened_high = true;
            } else {
                return Ok(result);
            }
        }
    }

    fn at_depth(
        &mut self,
        state: &State,
        depth: i32,
        alpha_orig: Score,
        beta_orig: Score,
    ) -> Searched<SearchResult> {
        self.partial_root = None;
        let key = state.hash();
        let (has_root, root_tt_move) = self.probe_tt_move(state, key);
        let mut buf = self.ordered_children(state, root_tt_move, has_root, 0)?;
        if buf.is_empty() {
            self.release(0, buf);
            return Ok(SearchResult::empty());
        }
        preserve_actor(&mut buf, self.root);

        let mut best_action = buf.children[buf.order[0] as usize].action;
        let mut best_score = -INF_SCORE;
        let mut roots: Vec<RootMove> = Vec::with_capacity(buf.order.len());
        let mut alpha = alpha_orig;
        let beta = beta_orig;
        let mut aborted = None;

        for (index, &slot) in buf.order.iter().enumerate() {
            let child = &buf.children[slot as usize];
            // A scout that fails low yields a bound, not a value; flagged so
            // consumers that rank or sample over the scores can skip it.
            let mut exact = true;
            let outcome = if self.multi {
                self.max_n(&child.state, depth - 1, 1)
                    .map(|values| values[self.root as usize - 1])
            } else if index == 0 {
                self.minimax(&child.state, depth - 1, alpha, beta, 1)
            } else {
                // Null-window scout; re-search full window on a fail inside.
                match self.minimax(&child.state, depth - 1, alpha, alpha + 1, 1) {
                    Ok(score) if score > alpha && score < beta => {
                        self.minimax(&child.state, depth - 1, alpha, beta, 1)
                    }
                    Ok(score) => {
                        exact = false;
                        Ok(score)
                    }
                    Err(error) => Err(error),
                }
            };
            let score = match outcome {
                Ok(score) => score,
                Err(error) => {
                    aborted = Some(error);
                    break;
                }
            };
            roots.push(RootMove {
                action: child.action,
                score,
                exact,
            });
            if score > best_score {
                best_action = child.action;
                best_score = score;
            }
            if !self.multi && score > alpha {
                alpha = score;
            }
        }
        self.release(0, buf);

        if let Some(error) = aborted {
            if self.options.enhanced && !roots.is_empty() && best_score > alpha_orig {
                let mut partial = SearchResult::with_action(best_action);
                partial.score = best_score;
                partial.salvaged = true;
                partial.alternatives = top_alternatives(&roots, best_action);
                self.partial_root = Some(partial);
            }
            return Err(error);
        }

        // A full-window iteration always stores EXACT; an aspirated iteration
        // that failed its window stores the true bound, so the re-search orders
        // off it.
        let flag = if best_score <= alpha_orig {
            FLAG_UPPER
        } else if best_score >= beta_orig {
            FLAG_LOWER
        } else {
            FLAG_EXACT
        };
        self.store_entry(state, key, depth, 0, flag, Some(best_action), best_score);

        let mut result = SearchResult::with_action(best_action);
        result.score = best_score;
        result.alternatives = top_alternatives(&roots, best_action);
        Ok(result)
    }

    // ---------------------------------------------------------------- interior

    fn minimax(
        &mut self,
        state: &State,
        depth: i32,
        mut alpha: Score,
        mut beta: Score,
        ply: i32,
    ) -> Searched<Score> {
        if !self.running() {
            return Err(Incomplete);
        }
        self.nodes += 1;
        if state.game_over() {
            return Ok(terminal_score(state, self.root, ply));
        }
        if depth == 0 {
            self.evaluations += 1;
            return Ok(self.leaf_eval(state));
        }
        let key = state.hash();
        let (hit, tt_move, cutoff) = self.probe(state, key, depth, ply, &mut alpha, &mut beta);
        if let Some(score) = cutoff {
            return Ok(score);
        }
        let alpha_orig = alpha;
        let beta_orig = beta;
        let maximizing = state.current_player() == self.root;
        let mut best = if maximizing { -INF_SCORE } else { INF_SCORE };
        let mut best_action: Option<Action> = None;

        // Staged move generation, stage A: with a TT best move in hand, apply
        // and search ONLY that child before materialising the ~30 siblings. Each
        // sibling costs a grid copy plus a flood fill per active player in
        // elimination detection, and at a cut-node all of that is thrown away.
        //
        // A full 64-bit key match means the stored action was generated for this
        // exact state, so it is legal; `tt_move_target_plausible` only shields a
        // genuine hash collision from corrupting the subtree. PlaceNeutrals TT
        // moves skip the fast path entirely (ARCHITECTURE.md invariant 6): the
        // search enumerates only a curated SUBSET of legal neutral pairs, so
        // legality alone would not prove the move is in the unstaged child list.
        let mut searched_tt_first = false;
        if self.options.enhanced {
            if let Some(action @ Action::Move { .. }) = tt_move {
                if tt_move_target_plausible(state, action) {
                    searched_tt_first = true;
                    let child = state.apply_generated_with(action, &mut self.scratch);
                    best = self.minimax(&child, depth - 1, alpha, beta, ply + 1)?;
                    best_action = Some(action);
                    if maximizing {
                        alpha = alpha.max(best);
                    } else {
                        beta = beta.min(best);
                    }
                    if alpha >= beta {
                        self.stats.fast_path_cuts += 1;
                        self.stats.cutoffs += 1; // fail-high at searched index 0
                        self.record_cutoff(state, action, depth, ply);
                        let flag = bound_flag(best, alpha_orig, beta_orig);
                        self.store_entry(state, key, depth, ply, flag, best_action, best);
                        return Ok(best);
                    }
                }
            }
        }

        let buf = self.ordered_children(state, tt_move, hit, ply)?;
        if buf.is_empty() && !searched_tt_first {
            self.release(ply, buf);
            self.evaluations += 1;
            return Ok(self.leaf_eval(state));
        }
        let mut searched = i32::from(searched_tt_first);
        let lmr_node =
            self.options.enhanced && self.options.lmr && depth >= LMR_MIN_DEPTH && !self.multi;
        let mut aborted = None;

        for (index, &slot) in buf.order.iter().enumerate() {
            let child = &buf.children[slot as usize];
            if searched_tt_first && Some(child.action) == tt_move {
                continue; // already searched full-window above
            }
            searched += 1;

            // Turn-aware late-move reduction: scout late QUIET SAME-TURN moves
            // one action shallower; a fail-high re-searches at full depth before
            // the normal PVS widening.
            //
            // The ordering tiers make the exclusions exact — TT, killer, win,
            // elimination and capture all land at order >= 10_000, while a quiet
            // move tops out at history (9_000) plus turn continuation (100).
            //
            // Turn-ENDING actions are never reduced. A turn boundary is where
            // tempo-swing eval error concentrates, and a reduced leaf on the far
            // side lands in the other side's turn fragment with a systematically
            // different tempo term. `lmr_turn_ending_reductions` is the tripwire.
            let mut reduction = 0;
            if lmr_node
                && searched > LMR_LATE_AFTER
                && child.order < ORDER_CAPTURE
                && matches!(child.action, Action::Move { .. })
                && child.state.current_player() == state.current_player()
            {
                reduction = 1;
                self.stats.lmr_reductions += 1;
            }
            // Independent tripwire, re-derived from the child rather than from
            // the guard above, so a future edit to the guard cannot silently
            // start reducing across a turn boundary. Asserted zero in the tests.
            if reduction != 0 && child.state.current_player() != state.current_player() {
                self.stats.lmr_turn_ending_reductions += 1;
            }

            let outcome = if !searched_tt_first && index == 0 {
                self.minimax(&child.state, depth - 1, alpha, beta, ply + 1)
            } else if maximizing {
                self.scout_max(child, depth, alpha, beta, ply, reduction)
            } else {
                self.scout_min(child, depth, alpha, beta, ply, reduction)
            };
            let score = match outcome {
                Ok(score) => score,
                Err(error) => {
                    aborted = Some(error);
                    break;
                }
            };

            if maximizing {
                if score > best {
                    best = score;
                    best_action = Some(child.action);
                }
                alpha = alpha.max(best);
            } else {
                if score < best {
                    best = score;
                    best_action = Some(child.action);
                }
                beta = beta.min(best);
            }
            if alpha >= beta {
                self.stats.cutoffs += 1;
                self.stats.cutoff_index_sum += (searched - 1) as u64;
                if let Some(action) = best_action {
                    self.record_cutoff(state, action, depth, ply);
                }
                break;
            }
        }
        self.release(ply, buf);
        if let Some(error) = aborted {
            return Err(error);
        }

        let flag = bound_flag(best, alpha_orig, beta_orig);
        self.store_entry(state, key, depth, ply, flag, best_action, best);
        Ok(best)
    }

    /// Null-window scout for the maximizing side, with the LMR re-search ladder.
    fn scout_max(
        &mut self,
        child: &Child,
        depth: i32,
        alpha: Score,
        beta: Score,
        ply: i32,
        reduction: i32,
    ) -> Searched<Score> {
        let mut score = self.minimax(
            &child.state,
            depth - 1 - reduction,
            alpha,
            alpha + 1,
            ply + 1,
        )?;
        if reduction != 0 && score > alpha {
            self.stats.lmr_re_searches += 1; // reduced scout failed high: verify
            score = self.minimax(&child.state, depth - 1, alpha, alpha + 1, ply + 1)?;
        }
        if score > alpha && score < beta {
            score = self.minimax(&child.state, depth - 1, alpha, beta, ply + 1)?;
        }
        Ok(score)
    }

    /// Null-window scout for the minimizing side.
    fn scout_min(
        &mut self,
        child: &Child,
        depth: i32,
        alpha: Score,
        beta: Score,
        ply: i32,
        reduction: i32,
    ) -> Searched<Score> {
        let mut score =
            self.minimax(&child.state, depth - 1 - reduction, beta - 1, beta, ply + 1)?;
        if reduction != 0 && score < beta {
            self.stats.lmr_re_searches += 1;
            score = self.minimax(&child.state, depth - 1, beta - 1, beta, ply + 1)?;
        }
        if score < beta && score > alpha {
            score = self.minimax(&child.state, depth - 1, alpha, beta, ply + 1)?;
        }
        Ok(score)
    }

    /// max^n for 3-4 player games.
    ///
    /// ARCHITECTURE.md invariant 1 in its purest form: there is no single
    /// opponent to negate against, so every node carries the whole utility
    /// vector and the mover maximises its own component. The only pruning that
    /// is sound without a constant-sum assumption is the exact mate cutoff — a
    /// child that already delivers an immediate terminal win for the mover
    /// cannot be beaten by a sibling.
    ///
    /// Keeps the map table and ply-exact probing in both modes: this only runs
    /// in 3-4 player games, which the 1v1 strength paths never reach.
    fn max_n(&mut self, state: &State, depth: i32, ply: i32) -> Searched<[Score; 4]> {
        if !self.running() {
            return Err(Incomplete);
        }
        self.nodes += 1;
        if state.game_over() {
            return Ok(terminal_scores(state, ply));
        }
        if depth == 0 {
            self.evaluations += 1;
            return Ok(self.leaf_eval_all(state));
        }
        let key = state.hash();
        let entry = self.map_table.get(&key).copied();
        let hit = entry.is_some();
        if let Some(entry) = entry {
            if entry.depth >= depth && entry.ply == ply {
                return Ok(entry.values);
            }
        }
        let tt_move = entry.and_then(|entry| entry.best_action);
        let buf = self.ordered_children(state, tt_move, hit, ply)?;
        if buf.is_empty() {
            self.release(ply, buf);
            self.evaluations += 1;
            return Ok(self.leaf_eval_all(state));
        }

        let player = state.current_player() as usize - 1;
        let max_bound = MATE_SCORE - (ply + 1) as Score;
        let mut best = [0 as Score; 4];
        best[player] = -INF_SCORE;
        let mut best_action = None;
        let mut aborted = None;
        for &slot in &buf.order {
            let child = &buf.children[slot as usize];
            match self.max_n(&child.state, depth - 1, ply + 1) {
                Ok(values) => {
                    if values[player] > best[player] {
                        best = values;
                        best_action = Some(child.action);
                        if best[player] >= max_bound {
                            break;
                        }
                    }
                }
                Err(error) => {
                    aborted = Some(error);
                    break;
                }
            }
        }
        self.release(ply, buf);
        if let Some(error) = aborted {
            return Err(error);
        }
        self.map_table.insert(
            key,
            MapEntry {
                depth,
                ply,
                flag: FLAG_EXACT,
                best_action,
                values: best,
            },
        );
        Ok(best)
    }

    // ---------------------------------------------------------------- table access

    /// The TT move for a root probe, and whether there was an entry at all.
    fn probe_tt_move(&mut self, state: &State, key: u64) -> (bool, Option<Action>) {
        if self.options.enhanced {
            let entry = self.tt.as_ref().map_or(0, |table| table.probe(key));
            if entry == 0 {
                return (false, None);
            }
            (
                true,
                tt::decode_action(tt::action_bits_of(entry), state.cols()),
            )
        } else {
            match self.map_table.get(&key) {
                Some(entry) => (true, entry.best_action),
                None => (false, None),
            }
        }
    }

    /// Full interior probe: returns `(hit, tt_move, early_return)` and may
    /// tighten `alpha` / `beta`.
    ///
    /// Enhanced probing is depth-sufficient and **ply-free**: a transposition
    /// reached at a different distance from the root — or persisted from a
    /// previous move's search — is usable, with mate scores rebased on the way
    /// out. Plain probing is ply-exact, which is what GoBot does and what the
    /// fixtures pin.
    #[allow(clippy::too_many_arguments)]
    fn probe(
        &mut self,
        state: &State,
        key: u64,
        depth: i32,
        ply: i32,
        alpha: &mut Score,
        beta: &mut Score,
    ) -> (bool, Option<Action>, Option<Score>) {
        if self.options.enhanced {
            self.stats.tt_probes += 1;
            let entry = self.tt.as_ref().map_or(0, |table| table.probe(key));
            if entry == 0 {
                return (false, None, None);
            }
            self.stats.tt_hits += 1;
            let tt_move = tt::decode_action(tt::action_bits_of(entry), state.cols());
            if tt::depth_of(entry) < depth {
                return (true, tt_move, None);
            }
            let value = tt::from_stored_score(tt::score_of(entry), ply);
            let cutoff = apply_bound(tt::flag_of(entry), value, alpha, beta);
            (true, tt_move, cutoff)
        } else {
            let Some(entry) = self.map_table.get(&key).copied() else {
                return (false, None, None);
            };
            let tt_move = entry.best_action;
            if entry.depth < depth || entry.ply != ply {
                return (true, tt_move, None);
            }
            let cutoff = apply_bound(entry.flag, entry.values[0], alpha, beta);
            (true, tt_move, cutoff)
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn store_entry(
        &mut self,
        state: &State,
        key: u64,
        depth: i32,
        ply: i32,
        flag: u8,
        best_action: Option<Action>,
        best: Score,
    ) {
        if self.options.enhanced {
            let bits = best_action.map_or(0, |action| {
                tt::encode_action(action, state.cols(), state.cell_count())
            });
            if let Some(table) = &self.tt {
                table.store(
                    key,
                    depth.min(63),
                    flag,
                    tt::to_stored_score(best, ply),
                    bits,
                );
            }
        } else {
            self.map_table.insert(
                key,
                MapEntry {
                    depth,
                    ply,
                    flag,
                    best_action,
                    values: [best, 0, 0, 0],
                },
            );
        }
    }

    // ---------------------------------------------------------------- move ordering

    fn ordered_children(
        &mut self,
        state: &State,
        tt_move: Option<Action>,
        has_tt: bool,
        ply: i32,
    ) -> Searched<ChildBuf> {
        let mut buf = std::mem::take(&mut self.bufs[ply.clamp(0, MAX_DEPTH + 1) as usize]);
        buf.children.clear();
        buf.order.clear();

        let actor = state.current_player();
        let before_active = active_count(state);
        let cols = state.cols();
        let position = Position::new_with(state.clone(), &mut self.scratch);
        let heuristics = self.options.enhanced && ply <= MAX_DEPTH;
        // The history table only tracks the two 1v1 seats.
        let use_history = heuristics && (actor as usize) <= self.history.len();
        let killers = if heuristics {
            self.killers[ply as usize]
        } else {
            [None, None]
        };

        let mut stopped = false;
        position.for_each_search_action(|action| {
            if !self.running() {
                stopped = true;
                return false;
            }
            let next = state.apply_generated_with(action, &mut self.scratch);
            let mut order = 0;
            if has_tt && Some(action) == tt_move {
                order += ORDER_TT;
            }
            // Killers rank right after the TT move, ahead of every static tier.
            if heuristics && (Some(action) == killers[0] || Some(action) == killers[1]) {
                order += ORDER_KILLER;
            }
            if next.game_over() && next.winner() == actor {
                order += ORDER_WIN;
            }
            order += (before_active - active_count(&next)) * ORDER_ELIMINATION;
            if let Action::Move { target } = action {
                let cell = state.at(target);
                if cell.kind() == CellKind::Normal && cell.owner() != actor {
                    order += ORDER_CAPTURE;
                }
                if use_history {
                    // History sits below the capture bonus and is capped, so it
                    // biases quiet-move order and never reorders captures.
                    let index = target.row as usize * cols + target.col as usize;
                    order += self.history[actor as usize - 1][index].min(ORDER_HISTORY_CAP);
                }
            }
            if next.current_player() == actor {
                order += ORDER_TURN_CONTINUATION;
            }
            buf.children.push(Child {
                action,
                state: next,
                order,
            });
            true
        });
        if stopped {
            self.release(ply, buf);
            return Err(Incomplete);
        }

        buf.order.extend(0..buf.children.len() as u32);
        let ChildBuf { children, order } = &mut buf;
        // Stable descending sort, exactly Go's `sort.SliceStable`: equal-order
        // siblings keep board order, which is what makes "first wins" tie-breaks
        // reproduce the oracle.
        order.sort_by_key(|&index| std::cmp::Reverse(children[index as usize].order));
        Ok(buf)
    }

    fn release(&mut self, ply: i32, buf: ChildBuf) {
        self.bufs[ply.clamp(0, MAX_DEPTH + 1) as usize] = buf;
    }

    /// On a fail-high caused by a QUIET move, remember it as a killer for this
    /// ply and bump its cell's history — the same refutation usually works at
    /// sibling nodes. Captures already order high statically, so they are
    /// excluded on purpose.
    fn record_cutoff(&mut self, state: &State, action: Action, depth: i32, ply: i32) {
        if !self.options.enhanced || ply > MAX_DEPTH {
            return;
        }
        let Action::Move { target } = action else {
            return;
        };
        let actor = state.current_player();
        let cell = state.at(target);
        if cell.kind() == CellKind::Normal && cell.owner() != actor {
            return;
        }
        let slot = &mut self.killers[ply as usize];
        if slot[0] != Some(action) {
            slot[1] = slot[0];
            slot[0] = Some(action);
        }
        if (actor as usize) <= self.history.len() {
            let index = target.row as usize * state.cols() + target.col as usize;
            let side = &mut self.history[actor as usize - 1];
            if index < side.len() {
                side[index] = (side[index] + depth * depth).min(HISTORY_STORE_CAP);
            }
        }
    }

    // ---------------------------------------------------------------- leaves

    fn leaf_eval(&mut self, state: &State) -> Score {
        virus_eval::evaluate(state, self.root, &self.options.params, &mut self.eval)
    }

    fn leaf_eval_all(&mut self, state: &State) -> [Score; 4] {
        virus_eval::evaluate_all(state, &self.options.params, &mut self.eval)
    }
}

// ---------------------------------------------------------------- free helpers

/// Applies a stored bound to the window, returning a score when it cuts.
fn apply_bound(flag: u8, value: Score, alpha: &mut Score, beta: &mut Score) -> Option<Score> {
    match flag {
        FLAG_EXACT => return Some(value),
        FLAG_LOWER => {
            if value >= *beta {
                return Some(value);
            }
            *alpha = (*alpha).max(value);
        }
        FLAG_UPPER => {
            if value <= *alpha {
                return Some(value);
            }
            *beta = (*beta).min(value);
        }
        _ => {}
    }
    if *alpha >= *beta {
        return Some(value);
    }
    None
}

fn bound_flag(best: Score, alpha_orig: Score, beta_orig: Score) -> u8 {
    if best <= alpha_orig {
        FLAG_UPPER
    } else if best >= beta_orig {
        FLAG_LOWER
    } else {
        FLAG_EXACT
    }
}

/// Cheap sanity check on a TT move before the staged fast path applies it.
///
/// A full 64-bit key match already implies legality and connectivity; this only
/// stops a genuine hash collision from applying an out-of-bounds or nonsensical
/// target.
fn tt_move_target_plausible(state: &State, action: Action) -> bool {
    let Action::Move { target } = action else {
        return false;
    };
    let Some(cell) = state.at_checked(target) else {
        return false;
    };
    cell.kind() == CellKind::Empty
        || (cell.kind() == CellKind::Normal && cell.owner() != state.current_player())
}

/// Drops root children that immediately eliminate the actor, unless every child
/// does.
fn preserve_actor(buf: &mut ChildBuf, actor: Player) {
    let children = &buf.children;
    if !buf
        .order
        .iter()
        .any(|&index| children[index as usize].state.active(actor))
    {
        return;
    }
    buf.order
        .retain(|&index| children[index as usize].state.active(actor));
}

/// A legal action that does not immediately eliminate the actor if one exists,
/// else the first legal action.
///
/// Deliberately independent of any search budget: even an already-cancelled
/// caller gets a self-preserving legal action. An illegal move sent to the
/// server is an instant forfeit, so this never returns a generated-but-
/// unvalidated action.
fn preserving_fallback(state: &State) -> Option<Action> {
    let actions = state.legal_actions();
    let first = *actions.first()?;
    let actor = state.current_player();
    for &action in &actions {
        if state.apply(action).is_ok_and(|next| next.active(actor)) {
            return Some(action);
        }
    }
    Some(first)
}

fn active_count(state: &State) -> i32 {
    (1..=virus_core::MAX_PLAYERS as Player)
        .filter(|&player| state.active(player))
        .count() as i32
}

fn terminal_score(state: &State, player: Player, ply: i32) -> Score {
    if state.winner() == player {
        MATE_SCORE - ply as Score
    } else {
        -MATE_SCORE + ply as Score
    }
}

fn terminal_scores(state: &State, ply: i32) -> [Score; 4] {
    let mut scores = [0 as Score; 4];
    for (seat, score) in scores.iter_mut().enumerate() {
        *score = terminal_score(state, seat as Player + 1, ply);
    }
    scores
}

/// The best next-best root moves (excluding the chosen one), best first.
fn top_alternatives(roots: &[RootMove], chosen: Action) -> Vec<RootMove> {
    if roots.len() <= 1 {
        return Vec::new();
    }
    let mut sorted = roots.to_vec();
    sorted.sort_by_key(|root| std::cmp::Reverse(root.score));
    sorted.retain(|root| root.action != chosen);
    sorted.truncate(MAX_ALTERNATIVES);
    sorted
}

// ---------------------------------------------------------------- entry points

/// One deterministic, fully completed fixed-depth search — Go's `ChooseDepth`.
///
/// **This is the parity oracle.** It runs in plain mode and deliberately skips
/// the opening book, so every record in `fixtures/gobot_search_parity.jsonl` is
/// pure search. Returns `None` when `depth` is out of range or the position has
/// no legal action.
pub fn choose_depth(state: &State, depth: i32) -> Option<SearchResult> {
    choose_depth_with(state, depth, &EvalParams::default())
}

/// [`choose_depth`] with explicit evaluation weights.
pub fn choose_depth_with(state: &State, depth: i32, params: &EvalParams) -> Option<SearchResult> {
    if !(1..=MAX_DEPTH).contains(&depth) {
        return None;
    }
    // Go computes the fallback up front and only uses it on an incomplete
    // search; a fixed-depth search has no budget so it can never be incomplete,
    // leaving this purely as the "does a legal action exist" gate it also is
    // there.
    preserving_fallback(state)?;
    let options = SearchOptions {
        params: *params,
        ..SearchOptions::plain()
    };
    let mut searcher = Searcher::new(state, options);
    searcher.begin_search(state, None, 0);
    let mut result = searcher
        .at_depth(state, depth, -INF_SCORE, INF_SCORE)
        .ok()?;
    result.depth = depth;
    result.nodes = searcher.nodes;
    result.evaluations = searcher.evaluations;
    Some(result)
}

/// Deterministic node-budget search with the full strength stack — the live
/// gauntlet entry point.
pub fn choose_node_budget(state: &State, limit: u64) -> Option<SearchResult> {
    Searcher::enhanced(state).search_node_budget(state, limit)
}

/// The plain-mode node-budget oracle, matching
/// `fixtures/gobot_nodebudget_parity.jsonl`. Consults the opening book, exactly
/// as Go's `ChooseNodeBudget` does.
pub fn choose_node_budget_plain(state: &State, limit: u64) -> Option<SearchResult> {
    Searcher::plain(state).search_node_budget(state, limit)
}

/// Production move choice: enhanced iterative deepening for `budget`.
pub fn choose(state: &State, budget: std::time::Duration) -> Option<SearchResult> {
    Searcher::enhanced(state).search(state, budget)
}

#[cfg(test)]
mod tests {
    use super::*;
    use virus_core::Pos;

    fn opening(actions: &[Action]) -> State {
        let mut state = State::new(12, 12, 2).expect("12x12 two-player board");
        for &action in actions {
            state = state.apply(action).expect("test line is legal");
        }
        state
    }

    /// Both wedges played out: P1 is back on move at `moves_left == 3` with own
    /// `Normal` cells, so both action kinds and every ordering tier are live.
    fn wedged() -> State {
        opening(&[
            Action::mv(1, 1),
            Action::mv(2, 1),
            Action::mv(2, 2),
            Action::mv(10, 10),
            Action::mv(9, 10),
            Action::mv(9, 9),
        ])
    }

    /// The `n`th-from-last `Move` in an ordering (`PlaceNeutrals` actions end a
    /// turn, so they always sort last and are never useful as a quiet probe).
    fn late_move(ordering: &[Action], back: usize) -> Action {
        *ordering
            .iter()
            .rev()
            .filter(|action| matches!(action, Action::Move { .. }))
            .nth(back)
            .expect("enough moves in the ordering")
    }

    /// ARCHITECTURE.md invariant 6. The search enumerates only a curated SUBSET
    /// of legal neutral pairs, so a `PlaceNeutrals` TT move is never provably in
    /// the unstaged child list — it must not take the staged fast path, however
    /// legal it looks.
    #[test]
    fn place_neutrals_tt_moves_never_take_the_fast_path() {
        let state = wedged();
        let pair = Action::neutrals(Pos::new(1, 1), Pos::new(2, 2));
        assert!(
            state
                .legal_actions()
                .iter()
                .any(|action| action.same_transition(pair)),
            "the pair is genuinely legal here, so only the kind check can reject it"
        );
        assert!(!tt_move_target_plausible(&state, pair));
    }

    #[test]
    fn tt_move_plausibility_rejects_impossible_targets() {
        let state = opening(&[]);
        assert!(!tt_move_target_plausible(&state, Action::mv(99, 99)));
        assert!(!tt_move_target_plausible(&state, Action::mv(-1, 0)));
        assert!(
            !tt_move_target_plausible(&state, Action::mv(0, 0)),
            "own base"
        );
        assert!(tt_move_target_plausible(&state, Action::mv(5, 5)), "empty");
        let state = opening(&[Action::mv(1, 1)]);
        assert!(
            !tt_move_target_plausible(&state, Action::mv(1, 1)),
            "own Normal"
        );
        let state = opening(&[Action::mv(1, 1), Action::mv(1, 2), Action::mv(2, 2)]);
        assert_eq!(state.current_player(), 2);
        assert!(
            tt_move_target_plausible(&state, Action::mv(1, 1)),
            "enemy Normal"
        );
    }

    #[test]
    fn the_fallback_prefers_a_self_preserving_action() {
        let state = opening(&[]);
        let action = preserving_fallback(&state).expect("a legal action exists");
        let next = state.apply(action).expect("the fallback is legal");
        assert!(next.active(state.current_player()));
    }

    #[test]
    fn bound_flags_follow_fail_soft_alpha_beta() {
        assert_eq!(bound_flag(10, 10, 20), FLAG_UPPER);
        assert_eq!(bound_flag(20, 10, 20), FLAG_LOWER);
        assert_eq!(bound_flag(15, 10, 20), FLAG_EXACT);
    }

    #[test]
    fn stored_bounds_tighten_the_window_before_cutting() {
        let (mut alpha, mut beta) = (0, 100);
        assert_eq!(apply_bound(FLAG_EXACT, 42, &mut alpha, &mut beta), Some(42));

        let (mut alpha, mut beta) = (0, 100);
        assert_eq!(
            apply_bound(FLAG_LOWER, 150, &mut alpha, &mut beta),
            Some(150)
        );

        let (mut alpha, mut beta) = (0, 100);
        assert_eq!(apply_bound(FLAG_LOWER, 40, &mut alpha, &mut beta), None);
        assert_eq!((alpha, beta), (40, 100));

        let (mut alpha, mut beta) = (0, 100);
        assert_eq!(apply_bound(FLAG_UPPER, -5, &mut alpha, &mut beta), Some(-5));

        let (mut alpha, mut beta) = (0, 100);
        assert_eq!(apply_bound(FLAG_UPPER, 60, &mut alpha, &mut beta), None);
        assert_eq!((alpha, beta), (0, 60));
    }

    /// Order tiers, highest first: TT, killer, win, elimination, capture,
    /// history, turn continuation.
    #[test]
    fn move_ordering_puts_the_tt_move_first_then_the_killer() {
        let state = wedged();
        let mut searcher = Searcher::enhanced(&state);
        searcher.prepare_heuristics(&state);

        let baseline = searcher.ordered_actions(&state, None, 1);
        assert!(baseline.len() > 3);

        let tt_move = late_move(&baseline, 0);
        let killer = late_move(&baseline, 1);
        searcher.set_killer(1, killer);
        let ordered = searcher.ordered_actions(&state, Some(tt_move), 1);
        assert_eq!(ordered[0], tt_move, "TT move outranks every other tier");
        assert_eq!(
            ordered[1], killer,
            "killer ranks directly below the TT move"
        );
    }

    #[test]
    fn history_biases_quiet_moves_but_stays_below_the_capture_tier() {
        let state = wedged();
        let mut searcher = Searcher::enhanced(&state);
        searcher.prepare_heuristics(&state);
        let baseline = searcher.ordered_actions(&state, None, 1);
        let last = late_move(&baseline, 0);
        let Action::Move { target } = last else {
            panic!("expected a move");
        };
        searcher.set_history(
            state.current_player(),
            target.row as usize * state.cols() + target.col as usize,
            1 << 20,
        );
        let ordered = searcher.ordered_actions(&state, None, 1);
        assert_eq!(
            ordered[0], last,
            "a saturated history counter promotes a quiet move to the front"
        );
    }

    /// ARCHITECTURE.md invariant 3, at its sharpest point. Salvaging a partially
    /// searched iteration is only sound when the running best beat the window we
    /// opened with: under an aspiration window an all-children-fail-low prefix is
    /// *bounds only*, and promoting a bound over the previous iteration's real
    /// move is exactly the class of bug that made the Go engine lose 0-10 live.
    #[test]
    fn salvage_refuses_a_partial_iteration_that_only_produced_bounds() {
        let state = wedged();

        // Establish the unbudgeted cost of one depth-5 iteration, so the aborts
        // below land at known fractions of it rather than at a magic number.
        let mut reference = Searcher::enhanced(&state);
        reference.begin_search(&state, None, 0);
        reference
            .at_depth(&state, 5, -INF_SCORE, INF_SCORE)
            .expect("an unbudgeted iteration always completes");
        let total = reference.nodes;
        assert!(total > 100, "the reference iteration is too small to slice");

        // An alpha above anything the position can score: every completed child
        // fails low, so the whole prefix is bounds and nothing is salvageable.
        let unreachable_alpha = MATE_SCORE / 2;

        let mut salvaged_somewhere = false;
        for tenths in 1..10 {
            let limit = total * tenths / 10;

            let mut full_window = Searcher::enhanced(&state);
            full_window.begin_search(&state, None, limit);
            if full_window
                .at_depth(&state, 5, -INF_SCORE, INF_SCORE)
                .is_err()
                && full_window.partial_root.is_some()
            {
                salvaged_somewhere = true;
            }

            let mut aspirated = Searcher::enhanced(&state);
            aspirated.begin_search(&state, None, limit);
            let _ = aspirated.at_depth(&state, 5, unreachable_alpha, INF_SCORE);
            assert!(
                aspirated.partial_root.is_none(),
                "a fail-low prefix is a bound, not a move (limit {limit})"
            );
        }
        assert!(
            salvaged_somewhere,
            "no abort ever produced a salvageable partial iteration"
        );
    }

    #[test]
    fn plain_mode_ignores_killers_and_history() {
        let state = wedged();
        let mut searcher = Searcher::plain(&state);
        searcher.prepare_heuristics(&state);
        let baseline = searcher.ordered_actions(&state, None, 1);
        let last = late_move(&baseline, 0);
        searcher.set_killer(1, last);
        assert_eq!(
            searcher.ordered_actions(&state, None, 1),
            baseline,
            "the oracle must not see a heuristic"
        );
    }
}
