//! The PUCT searcher.
//!
//! A port of `nnue-trainer/.../mcts/MctsSearcher.java`, which is the current
//! production champion's search.
//!
//! # Per-action nodes
//!
//! A node is a whole [`State`] and an edge is a **single** action, not a whole
//! turn. Branching stays around 34 instead of exploding into turn triples — at
//! the cost of the one property every textbook MCTS takes for granted, which
//! the next section is about.
//!
//! # Absolute-frame backup (ARCHITECTURE.md invariant 1)
//!
//! A turn is three actions, so ~53% of edges keep the mover and only ~47% flip
//! it. Negamax-style "negate on every backup" is therefore wrong on the
//! majority of edges. Instead:
//!
//! * the leaf value is converted **once**, at the leaf, into a fixed absolute
//!   frame where positive is good for player 1;
//! * `W` accumulates in that frame with **no negation anywhere** on the way up;
//! * the mover enters only at selection, as `sign(node) * Q_abs + U` with
//!   `sign(node) = +1` iff the node's mover is player 1.
//!
//! That single axis is also the searcher's hard domain limit: it is two-player
//! zero-sum by construction, so [`MctsSearcher::new`] refuses three- and
//! four-player positions outright rather than scoring a third seat's win as a
//! draw. The same is true of the Java original, whose `GoState` is two-player
//! throughout.
//!
//! The `tests` module at the bottom of this file pins the invariant on a
//! hand-built tree whose movers do not alternate — the case a per-edge negation
//! gets wrong — and on the mirrored-position identity for the leaf flip.
//!
//! # Leaf batching and virtual loss
//!
//! A simulation's cost is dominated by the one net forward its expansion pays
//! for, and a single 13x12x12 forward leaves most of the machine idle (see
//! [`PolicyValueNet::forward_batch`]). So the searcher does not run one
//! simulation at a time: it descends [`Config::batch_size`] times in a row,
//! collecting that many leaves, evaluates them as **one** batched forward, and
//! only then backs the results up.
//!
//! Repeating selection against an unchanged tree would pick the same leaf every
//! time, so each descent applies a **virtual loss** to the edges it walks: the
//! edge's effective visit count goes up by one and its effective `Q` is dragged
//! towards a loss *for the node's own mover*, which is what decorrelates the
//! batch. Virtual loss is tracked in separate `vl`/`vl_visits` counters rather
//! than folded into `n`/`w`, so removing it after backup is an exact
//! decrement — no floating-point residue — and so a search with
//! `batch_size == 1` is bit-for-bit the serial searcher: nothing is ever in
//! flight when the only descent of a round selects.
//!
//! # DAG transpositions (superiority.md §2d item 2)
//!
//! A turn is three actions and the actions of a turn largely commute: playing
//! `A` then `B` reaches exactly the same position as `B` then `A`. A tree keeps
//! those as two separate subtrees and expands both, paying a net forward per
//! duplicate; a **DAG** keeps one node and lets both parents point at it. The
//! Java transposition-table audit already proved these collisions are real and
//! that the full state key catches them
//! (`20260807-search-strength.md:110-112`).
//!
//! [`Config::dag`] turns it on. The key is [`State::hash`] — the incremental
//! Zobrist key, which by ARCHITECTURE.md invariant 6 already covers the grid,
//! the side to move, `moves_left`, `neutral_used` **and** `active`. Nothing is
//! added to it here; a merge is additionally confirmed by a full `State`
//! equality check, so a 64-bit key collision costs a duplicated node and never
//! a wrong merge.
//!
//! Four things make this safe, in the order they would otherwise bite:
//!
//! * **Value merging is frame-safe.** Backup is absolute-frame (invariant 1),
//!   so a node's `W` means the same thing no matter which parent a visit
//!   arrived through. In a negamax tree, whose sign depends on ply parity,
//!   merging two parents of *different* movers would silently invert half the
//!   statistics. Here it is a commutative add, exactly as it is for
//!   [`crate::parallel::ParallelMcts`].
//! * **The DAG is acyclic by the rules, not by luck.** Every action strictly
//!   decreases the lexicographic tuple `(empty cells, -fortified cells,
//!   -neutral cells)`: a move onto an empty cell spends an empty cell, a
//!   capture turns an enemy `Normal` into a `Fortified`, and `PlaceNeutrals`
//!   turns two `Normal`s into `Neutral`s. No action ever reverses any of the
//!   three, so no position can reach itself and a descent cannot loop. The same
//!   argument says two paths of *different* lengths cannot meet: the tuple is
//!   strictly monotone along a path, so a shared position is always at a shared
//!   depth. Transpositions are within-turn permutations, which is precisely the
//!   case this is for.
//! * **The root is never a merge target.** Only nodes created as *children* are
//!   entered in the index, so nothing can ever be redirected onto node 0 — and
//!   therefore Dirichlet root noise, which lives in the root's `prior` and
//!   nowhere else, can never leak into an interior node. (It could not happen
//!   anyway, by the acyclicity argument, but the index makes it structural
//!   rather than a proof someone has to re-derive.) A [`MctsSearcher::rebase`]
//!   re-establishes the same property for the new root.
//! * **Virtual loss is per *edge*, so merging nodes does not disturb it.** A
//!   descent increments `vl` on the `(parent, edge)` pairs it walks and backup
//!   decrements exactly those; two descents reaching one merged node through
//!   different parents perturb different counters. What *does* change is that
//!   the merged node is far more often already `pending` in the same batch,
//!   which the existing leaf-reuse path handles and which is where the sim
//!   savings show up.
//!
//! ## What it saves, and what it does not
//!
//! Be precise here, because the plan (superiority.md §2d.2) says "sim savings +
//! memory" and only one of those is a node-count reduction.
//!
//! A simulation expands exactly one leaf, so at `batch_size == 1` **both arms
//! hold `sims + 1` nodes**. The DAG does not make the arena smaller at equal
//! simulations. What it changes is what those expansions were spent on: a plain
//! tree re-expands positions it has already evaluated — a second net forward, a
//! second subtree, a second set of statistics that never learns from the
//! first — while the DAG's arena is duplicate-free by construction, so every
//! forward buys a position it has never seen and every visit lands on
//! statistics the other order can read. That is the sim saving, and it is
//! measured as the plain tree's duplicate count.
//!
//! Measured at 1500 simulations from a developed midgame
//! (`examples/mctsbench`): a plain tree's 1501 nodes cover **890** distinct
//! positions, the DAG's 1501 nodes cover **1501**. Two of every five
//! expansions the tree paid for were a position it already had. Note how
//! little it takes — **92** merges produced that difference, because a merge
//! near the top of the tree stops an entire duplicate subtree from forming.
//!
//! Batching adds a second, smaller saving that *is* a node-count reduction: a
//! merged node reached twice inside one batch is already `pending` the second
//! time, so the round runs fewer forwards than it had descents.
//!
//! The memory claim is therefore about the ponder regime rather than about any
//! one search: at a fixed simulation budget the arena is the same size, but a
//! long ponder is bounded by *distinct reachable positions*, and the DAG is
//! what stops a tree from spending that bound on copies. Reclamation itself is
//! unchanged and lives in [`MctsSearcher::rebase`] — the index is rebuilt from
//! the survivors on every re-root, so neither an unreachable node nor a stale
//! key outlives it.
//!
//! # Randomness
//!
//! Play mode draws nothing: `run_sims` and `run_until_deadline` are pure
//! functions of the position, the config and the net, batched or not. Dirichlet
//! root noise and temperature sampling are self-play only, and both run off the
//! seeded [`Rng`], so even those are reproducible.
//!
//! Thread parallelism lives in [`crate::parallel::ParallelMcts`], a separate
//! opt-in type: this searcher is single-threaded and deterministic, full stop.

use std::collections::HashMap;
use std::hash::{BuildHasher, Hasher};
use std::time::{Duration, Instant};

use virus_core::{Action, Player, Scratch, State};
use virus_eval::{evaluate, EvalParams, EvalWorkspace};

use crate::gumbel::{
    logits_from_prior, rescale, softmax, GumbelConfig, GumbelPlan, DEFAULT_GUMBEL_C_SCALE,
    DEFAULT_GUMBEL_C_VISIT,
};
use crate::net::{BatchScratch, Encoded, Heads, NetScratch, PolicyValueNet, BOARD};
use crate::rng::Rng;

/// Sentinel for "this edge has no child node yet".
const NO_NODE: u32 = u32::MAX;

/// Default exploration constant.
pub const DEFAULT_CPUCT: f64 = 1.5;

/// Default value squash: `v = tanh(hand_tuned / VALUE_SCALE)`.
///
/// 12000 maps a typical decisive mid-game evaluation (~13k) to ~0.8, and
/// terminal-adjacent evaluations (~5e8) saturate to +-1.
pub const DEFAULT_VALUE_SCALE: f64 = 12_000.0;

/// Plies of temperature-1 visit sampling at the start of a self-play game:
/// 21 plies = 7 turns.
pub const TEMPERATURE_PLIES: u32 = 21;

/// Default leaves per batched net evaluation.
///
/// **Eight, because that is where throughput plateaus and nothing above it buys
/// anything.** Measured on this box with `examples/mctsbench`: 8 through 48 all
/// land within measurement noise of each other at 2.1-2.5x the serial searcher,
/// so the whole gain is already collected at 8 — which is no coincidence, it is
/// [`crate::net::BATCH_LANES`], the width the batched trunk fills.
///
/// Below the lane width a batch is an outright *loss*: the net rounds a batch up
/// to whole lane groups, so a round of 2 leaves pays for a group of 8 and
/// measured **0.6x** serial. Anything from 2 to 7 is worse than both ends.
///
/// Above the lane width the extra size costs tree freshness — a round selects
/// `batch_size` times against a tree that only virtual loss is perturbing — and
/// buys no throughput. The 100-game fixed-time self-gauntlets against the serial
/// searcher (see the S3-T1 PR) scored 0.55 pooled at 8 and 0.52 at 16; those two
/// samples do not separate on their own, but they point the same way as the
/// argument, so the smallest batch on the plateau is the default.
///
/// Set it to `1` for the serial searcher, bit for bit.
pub const DEFAULT_BATCH_SIZE: u16 = 8;

/// Default virtual-loss weight, in leaf-value units.
///
/// One whole loss, the AlphaGo setting. Leaf values live in `[-1, 1]`, so an
/// edge with one descent in flight and no real visits scores `Q = -1` — bad
/// enough that the next descent in the batch takes a genuinely different
/// branch, and transient, because it is removed the moment the batch backs up.
pub const DEFAULT_VIRTUAL_LOSS: f64 = 1.0;

/// Whether [`Config::dag`] merges transpositions by default.
///
/// **On, because the position coverage it buys is large and the gauntlet went
/// the right way.** At 1500 simulations from a developed midgame
/// (`examples/mctsbench`), a plain tree spends **~40%** of its expansions on
/// positions it has already evaluated elsewhere — 1501 nodes covering 890
/// distinct positions — while the DAG's 1501 nodes are 1501 distinct
/// positions. That is a 1.69x widening of what a fixed simulation budget sees,
/// and it comes from only 92 merges: a merge near the top of the tree stops a
/// whole duplicate subtree from ever forming.
///
/// The fixed-simulation self-gauntlet against `dag: false`, both arms at the
/// default batch size, scored **0.5575 pooled over 400 games** — 223W 177L 0D,
/// wilson95 [50.9%, 60.5%], four 100-game blocks on well-separated seeds
/// scoring 0.56/0.51/0.54/0.62. S3-T2 only asks for no regression (>= 0.50);
/// with the interval's lower bound above 50% this is a real gain rather than
/// merely a safe change. See the S3-T2 PR for the load caveats.
///
/// Set it to `false` for the plain tree searcher, node for node.
pub const DEFAULT_DAG: bool = true;

/// A `HashMap` hasher for keys that are *already* hashes.
///
/// The transposition index is keyed by [`State::hash`], an incremental Zobrist
/// key that is uniformly distributed by construction, so running it through
/// SipHash buys nothing. Two reasons this is more than a micro-optimisation:
///
/// * `std`'s `RandomState` is seeded per process, so iteration order differs
///   run to run. Nothing here iterates the index — it is only ever `get` and
///   `insert`, both order-independent — but "this searcher is deterministic" is
///   a contract the crate keeps *by construction*, and a fixed hasher keeps it
///   that way even if someone later adds an iteration.
/// * It is one multiply instead of a SipHash round on the child-creation path.
#[derive(Clone, Copy, Debug, Default)]
struct ZobristHasher(u64);

impl Hasher for ZobristHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    /// Never called for a `u64` key, and a footgun if it silently were, so it
    /// is a real (FNV-1a) hash rather than a `unimplemented!` or a no-op.
    fn write(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.0 = (self.0 ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3);
        }
    }

    fn write_u64(&mut self, value: u64) {
        // Fibonacci mixing. A Zobrist key is already uniform in every bit, so
        // this is belt and braces against a future key that is not.
        self.0 = value.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }
}

/// [`ZobristHasher`]'s `BuildHasher`; stateless, hence seedless, hence stable.
#[derive(Clone, Copy, Debug, Default)]
struct BuildZobristHasher;

impl BuildHasher for BuildZobristHasher {
    type Hasher = ZobristHasher;

    fn build_hasher(&self) -> ZobristHasher {
        ZobristHasher(0)
    }
}

/// Position key -> node index, for the nodes eligible to be merged into.
type Transpositions = HashMap<u64, u32, BuildZobristHasher>;

/// Where a leaf's value comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ValueSource {
    /// `tanh(hand_tuned_eval / value_scale)` from `virus-eval`.
    #[default]
    HandTuned,
    /// The artifact's value head, falling back to [`ValueSource::HandTuned`]
    /// when the loaded net has none.
    Net,
}

/// Searcher tuning. [`Config::default`] is the play-mode configuration.
#[derive(Clone, Copy, Debug)]
pub struct Config {
    /// PUCT exploration constant.
    pub cpuct: f64,
    /// Seed for every stochastic decision in this search.
    pub seed: u64,
    /// Divisor in the hand-tuned leaf's `tanh` squash.
    pub value_scale: f64,
    /// Which leaf value to use.
    pub value_source: ValueSource,
    /// Dirichlet root noise. **Self-play only** — leave off in play mode.
    pub root_noise: bool,
    /// Dirichlet concentration.
    pub noise_alpha: f64,
    /// Weight of the noise in the mixed prior.
    pub noise_epsilon: f64,
    /// Sample the root action from the visit counts (temperature 1) instead of
    /// taking the argmax. **Self-play only.**
    pub visit_sampling: bool,
    /// Leaves collected per batched net evaluation. `0` and `1` both mean the
    /// serial searcher; see the module docs and [`DEFAULT_BATCH_SIZE`]. Values
    /// between 2 and [`crate::net::BATCH_LANES`] are worse than either end —
    /// the net rounds a batch up to whole lane groups.
    pub batch_size: u16,
    /// Virtual-loss weight applied to an edge with a descent in flight.
    /// Inert at `batch_size <= 1` in a single-threaded search.
    pub virtual_loss: f64,
    /// Worker threads for [`crate::parallel::ParallelMcts`]. **Ignored by
    /// [`MctsSearcher`]**, which is single-threaded by construction; it lives
    /// here so a caller can carry one configuration through both.
    pub threads: usize,
    /// Merge transpositions: two action orders reaching the same position share
    /// one node, and the tree becomes a DAG. See the module's "DAG
    /// transpositions" section for the safety argument and
    /// [`MctsSearcher::rebase`] for the memory story.
    ///
    /// **Ignored by [`crate::parallel::ParallelMcts`]**, the mirror of
    /// [`Config::threads`] being ignored here: the shared-tree engine indexes
    /// its nodes by `Arc` identity and merging them needs a concurrent index,
    /// which is its own piece of work.
    pub dag: bool,
    /// Gumbel / sequential-halving root selection. **Self-play only** — the
    /// same rule, and the same reason, as [`Config::root_noise`]: it is an
    /// exploration mechanism that deliberately does not play the strongest
    /// move it knows.
    ///
    /// `None` is ordinary PUCT at the root and is what [`Config::default`] and
    /// [`Config::play`] give. Mutually exclusive with [`Config::root_noise`]:
    /// Gumbel *replaces* Dirichlet rather than stacking on it, and
    /// [`MctsSearcher::new`] refuses a configuration that asks for both.
    ///
    /// **Refused by [`crate::parallel::ParallelMcts`]** — *not* ignored, which
    /// is where it parts company with [`Config::dag`]. Silently dropping the
    /// DAG costs simulations; silently dropping this would run PUCT with
    /// argmax-visit selection and write a different training target under a
    /// configuration that says Gumbel. Self-play is single-threaded and one
    /// game per process, so nothing needs the combination.
    pub gumbel: Option<GumbelConfig>,
}

impl Default for Config {
    fn default() -> Config {
        Config {
            cpuct: DEFAULT_CPUCT,
            seed: 1,
            value_scale: DEFAULT_VALUE_SCALE,
            value_source: ValueSource::HandTuned,
            root_noise: false,
            noise_alpha: 0.3,
            noise_epsilon: 0.25,
            visit_sampling: false,
            batch_size: DEFAULT_BATCH_SIZE,
            virtual_loss: DEFAULT_VIRTUAL_LOSS,
            threads: 1,
            dag: DEFAULT_DAG,
            gumbel: None,
        }
    }
}

impl Config {
    /// The play-mode configuration: no noise, no Gumbel, no sampling, no RNG
    /// draws at all.
    ///
    /// The three exploration knobs are spelled out rather than inherited so
    /// that a future change to [`Config::default`] cannot turn any of them on
    /// in production. `play_mode_takes_no_exploration_knob` pins it.
    pub fn play() -> Config {
        Config {
            root_noise: false,
            visit_sampling: false,
            gumbel: None,
            ..Config::default()
        }
    }

    /// The self-play configuration for `ply`: Dirichlet root noise always, and
    /// temperature-1 visit sampling for the first [`TEMPERATURE_PLIES`] plies.
    pub fn self_play(seed: u64, ply: u32) -> Config {
        Config {
            seed,
            value_source: ValueSource::Net,
            root_noise: true,
            visit_sampling: ply < TEMPERATURE_PLIES,
            ..Config::default()
        }
    }

    /// The **Gumbel** self-play configuration: root-only Gumbel top-`m` with
    /// sequential halving over a `sims` budget.
    ///
    /// Neither Dirichlet noise nor temperature sampling, and that is the point
    /// rather than an omission. Both exist to stop self-play from replaying one
    /// game; the Gumbel draw already does that, and it does it as a *policy
    /// improvement* — the selected action is the argmax of `g + logit +
    /// sigma(q)`, which the paper shows never scores below a draw from the
    /// prior. Stacking Dirichlet on top would perturb the very logits the
    /// top-`m` draw is a sample from, and stacking visit sampling on top would
    /// throw away the argmax the schedule spent its whole budget establishing.
    ///
    /// There is no `ply` argument for the same reason: the Gumbel variate is
    /// redrawn at every root, so exploration does not need a decaying window.
    pub fn self_play_gumbel(seed: u64, sims: u32, m: u16) -> Config {
        Config {
            seed,
            value_source: ValueSource::Net,
            root_noise: false,
            visit_sampling: false,
            gumbel: Some(GumbelConfig {
                m,
                sims,
                ..GumbelConfig::default()
            }),
            ..Config::default()
        }
    }
}

/// One tree node: a position, its legal actions, and the per-edge statistics.
#[derive(Debug)]
struct Node {
    state: State,
    /// `state.current_player()`, cached because selection reads it every visit.
    mover: Player,
    terminal: bool,
    /// Absolute-frame terminal value; meaningful only when `terminal`.
    terminal_value_abs: f64,
    /// This node's **own** leaf evaluation in the absolute frame — the value
    /// its expansion backed up, or its terminal value.
    ///
    /// Distinct from the visit-weighted average of its children, which is what
    /// [`MctsSearcher::root_value_abs`] reports, and the completed-Q `v_mix`
    /// needs *this* one: `v_mix` interpolates the node's own estimate with its
    /// children's, so feeding it the children's average again would collapse
    /// the interpolation. Stored per node rather than beside the root so that
    /// [`MctsSearcher::rebase`] promotes a child without having to re-derive or
    /// re-evaluate anything.
    leaf_value_abs: f64,
    expanded: bool,
    /// Set while this node is a collected-but-unevaluated leaf of the current
    /// batch, so a second descent in the same batch reuses its pending
    /// evaluation instead of queueing a duplicate forward.
    pending: bool,
    actions: Vec<Action>,
    prior: Vec<f32>,
    children: Vec<u32>,
    n: Vec<u32>,
    /// Absolute-frame value sums — positive is good for player 1.
    w: Vec<f64>,
    /// Descents currently in flight through each edge — the virtual loss.
    /// Always back to all-zero once a batch has backed up.
    vl: Vec<u32>,
    /// Sum of the edge visits below this node.
    visits: u32,
    /// `vl` summed over the edges, kept alongside so selection does not have to
    /// re-add it.
    vl_visits: u32,
}

impl Node {
    fn new(state: State) -> Node {
        let mover = state.current_player();
        let (terminal, terminal_value_abs) = if state.game_over() {
            (true, terminal_value_abs(&state))
        } else {
            (false, 0.0)
        };
        Node {
            state,
            mover,
            terminal,
            terminal_value_abs,
            leaf_value_abs: terminal_value_abs,
            expanded: false,
            pending: false,
            actions: Vec::new(),
            prior: Vec::new(),
            children: Vec::new(),
            n: Vec::new(),
            w: Vec::new(),
            vl: Vec::new(),
            visits: 0,
            vl_visits: 0,
        }
    }
}

/// One collected leaf of the current batch.
#[derive(Clone, Copy, Debug)]
struct Leaf {
    node: u32,
    /// `Some` for a terminal leaf, whose value needs no net; `None` until the
    /// batched forward fills it in.
    value: Option<f64>,
}

/// One collected descent, as a slice of the shared path buffer plus the leaf it
/// ended at. Several paths may share a leaf.
#[derive(Clone, Copy, Debug)]
struct Descent {
    start: usize,
    end: usize,
    leaf: usize,
}

/// Terminal value in the absolute frame, from the single labelling rule
/// (including the territory tiebreak).
///
/// Two-player by construction: the frame is one axis, `+1` for player 1 and
/// `-1` for player 2, so there is nowhere to put a win for a third seat. That
/// is why [`MctsSearcher::new`] refuses anything but a two-player position
/// rather than letting such a win score as a draw here.
pub fn terminal_value_abs(state: &State) -> f64 {
    match state.outcome_winner() {
        1 => 1.0,
        2 => -1.0,
        _ => 0.0,
    }
}

/// Hand-tuned leaf value in the absolute frame.
///
/// Queried from the leaf's **own** mover — its natural, in-distribution frame,
/// the same rationale as the alpha-beta searcher's leaf evaluation — then
/// squashed and flipped to positive-is-good-for-player-1. That flip is the
/// single sign application the whole design turns on.
pub(crate) fn hand_tuned_value_abs(
    state: &State,
    value_scale: f64,
    params: &EvalParams,
    workspace: &mut EvalWorkspace,
) -> f64 {
    let mover = state.current_player();
    let v = (evaluate(state, mover, params, workspace) as f64 / value_scale).tanh();
    if mover == 1 {
        v
    } else {
        -v
    }
}

/// A PUCT search over one position.
///
/// The searcher borrows the net rather than owning it, so a gauntlet can share
/// one loaded artifact across every game and thread.
#[derive(Debug)]
pub struct MctsSearcher<'net> {
    config: Config,
    net: Option<&'net PolicyValueNet>,
    nodes: Vec<Node>,
    /// [`State::hash`] -> node index, for [`Config::dag`]. Empty and untouched
    /// when the flag is off. **Node 0 is never in it**, so the root — the only
    /// node that can carry Dirichlet noise — is never a merge target.
    transpositions: Transpositions,
    /// Child links resolved to an existing node instead of allocating one: the
    /// duplicate expansions, and the net forwards they would have cost, that
    /// the DAG saved.
    merges: u64,
    /// Zobrist keys that matched but whose positions did not. Expected to stay
    /// at zero over any realistic search; a non-zero count is not a correctness
    /// problem (the position check catches it and the node is duplicated) but
    /// it is the number to look at if the merge rate ever looks wrong.
    key_collisions: u64,
    /// The root's sequential-halving schedule, when [`Config::gumbel`] is on.
    /// Redrawn by [`MctsSearcher::rebase`], because a schedule belongs to one
    /// root and its edge indices mean nothing at the next one.
    gumbel: Option<GumbelPlan>,
    rng: Rng,
    sims: u64,
    /// Reusable batch buffers, so a round of simulations allocates nothing
    /// beyond new tree nodes.
    path: Vec<(u32, u32)>,
    descents: Vec<Descent>,
    leaves: Vec<Leaf>,
    eval_index: Vec<usize>,
    encoded: Vec<Encoded>,
    heads: Vec<Heads>,
    scratch: Box<Scratch>,
    net_scratch: Option<NetScratch>,
    /// Allocated on the first batched forward, not up front: it is ~0.5 MB and
    /// a `batch_size <= 1` searcher never touches it.
    batch_scratch: Option<BatchScratch>,
    eval_params: EvalParams,
    eval_workspace: EvalWorkspace,
}

impl<'net> MctsSearcher<'net> {
    /// Builds a searcher and expands the root (plus root noise, if configured).
    ///
    /// # Panics
    ///
    /// Panics on a position outside this searcher's domain. Two separate
    /// limits, both checked up front so a mismatch can never surface as a
    /// quietly wrong move halfway through a search:
    ///
    /// * **Two players, always.** The absolute frame is a two-player zero-sum
    ///   construct end to end — [`terminal_value_abs`] labels the outcome on a
    ///   single `+1`/`-1` axis, and `select` reads any mover that is not
    ///   player 1 as "the opponent of player 1". Under three or four seats that
    ///   silently allies seats 2-4 and scores a win for seat 3 or 4 as a draw.
    ///   Supporting them needs a per-seat value vector, which is a different
    ///   design, not a relaxed assertion.
    /// * **12x12, when a net is supplied.** [`Encoded::from_state`] has no
    ///   representation for another board size.
    pub fn new(state: State, config: Config, net: Option<&'net PolicyValueNet>) -> Self {
        assert_eq!(
            state.players(),
            2,
            "the absolute-frame searcher is two-player only"
        );
        assert!(
            net.is_none() || (state.rows() == BOARD && state.cols() == BOARD),
            "policy net is {BOARD}x{BOARD} only, got {}x{}",
            state.rows(),
            state.cols()
        );
        assert!(
            !(config.root_noise && config.gumbel.is_some()),
            "Dirichlet root noise and Gumbel root selection are alternatives, \
             not layers: Gumbel draws its top-m from the prior the noise would \
             have perturbed"
        );
        let net_scratch = net.map(|net| net.scratch());
        let mut searcher = MctsSearcher {
            config,
            net,
            nodes: vec![Node::new(state)],
            transpositions: Transpositions::default(),
            merges: 0,
            key_collisions: 0,
            gumbel: None,
            rng: Rng::new(config.seed),
            sims: 0,
            path: Vec::with_capacity(1024),
            descents: Vec::with_capacity(64),
            leaves: Vec::with_capacity(64),
            eval_index: Vec::with_capacity(64),
            encoded: Vec::with_capacity(64),
            heads: Vec::with_capacity(64),
            scratch: Scratch::new(),
            net_scratch,
            batch_scratch: None,
            eval_params: EvalParams::default(),
            eval_workspace: EvalWorkspace::new(),
        };
        if !searcher.nodes[0].terminal {
            // `expand` records the leaf value on the node itself, so nothing
            // here has to hold on to it.
            searcher.expand(0);
            if !searcher.nodes[0].terminal {
                if searcher.config.root_noise {
                    searcher.apply_root_noise();
                }
                // After the noise branch, not before, purely so the two are
                // read as the alternatives they are; the assert above means at
                // most one of them ever runs.
                searcher.plan_gumbel();
            }
        }
        searcher
    }

    /// Draws this root's Gumbel schedule, if [`Config::gumbel`] asks for one.
    fn plan_gumbel(&mut self) {
        let Some(config) = self.config.gumbel else {
            return;
        };
        let root = &self.nodes[0];
        if root.terminal || root.actions.is_empty() {
            self.gumbel = None;
            return;
        }
        let prior = root.prior.clone();
        self.gumbel = Some(GumbelPlan::new(&prior, &config, &mut self.rng));
    }

    /// Runs exactly `count` further simulations.
    ///
    /// Deterministic: for a given position, config and net this always builds
    /// the same tree and returns the same move. Batching does not change that —
    /// the leaves of a round are collected, evaluated and backed up in a fixed
    /// order — but a different [`Config::batch_size`] does build a different
    /// (equally valid) tree, exactly as a different `cpuct` would.
    pub fn run_sims(&mut self, count: u32) {
        if self.nodes[0].terminal {
            return;
        }
        let batch = u32::from(self.config.batch_size.max(1));
        let mut done = 0;
        while done < count {
            done += self.simulate_round(batch.min(count - done));
        }
    }

    /// Simulates until `deadline`, always running at least one simulation.
    ///
    /// The deadline is checked between batches, so a budget can overshoot by up
    /// to one batch — at the tuned [`DEFAULT_BATCH_SIZE`] one to two
    /// milliseconds, against the hundreds a real move budget allows. Callers
    /// that slice a turn into deadlines (the `vsbot` bin does) still land
    /// inside their fallback discipline; a caller that needs the old
    /// simulation-granular check can set `batch_size` to 1.
    pub fn run_until_deadline(&mut self, deadline: Instant) {
        if self.nodes[0].terminal {
            return;
        }
        let batch = u32::from(self.config.batch_size.max(1));
        loop {
            self.simulate_round(batch);
            if Instant::now() >= deadline {
                return;
            }
        }
    }

    /// [`MctsSearcher::run_until_deadline`] with a relative budget.
    pub fn run_for(&mut self, budget: Duration) {
        self.run_until_deadline(Instant::now() + budget);
    }

    /// Simulations run so far.
    pub fn sims_run(&self) -> u64 {
        self.sims
    }

    /// The configuration this searcher was built with.
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Nodes currently in the arena, root included.
    ///
    /// A simulation expands exactly one leaf, so at `batch_size == 1` this is
    /// `sims + 1` whether or not [`Config::dag`] is on — the DAG's saving is
    /// that none of those nodes is a duplicate position, not that there are
    /// fewer of them. Under a batch it *is* also smaller, because a merged node
    /// reached twice in one round is evaluated once. See the module's "What it
    /// saves, and what it does not".
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Distinct positions in the arena.
    ///
    /// **The sim-savings measurement.** Equal to [`MctsSearcher::node_count`]
    /// exactly when the arena holds no duplicate, which is what
    /// [`Config::dag`] gives whenever [`MctsSearcher::key_collisions`] is zero
    /// — a rejected key collision leaves the loser position without an index
    /// entry, so further arrivals at it duplicate. The shortfall in a plain
    /// tree is the number of expansions — and net forwards — it spent on a
    /// position it had already evaluated somewhere else. Comparing the two arms
    /// at equal simulations is what the S3-T2 numbers are.
    ///
    /// `O(nodes)` and allocating, so it is a diagnostic for benches and tests,
    /// not something to call inside a search.
    pub fn distinct_positions(&self) -> usize {
        self.nodes
            .iter()
            .map(|node| node.state.hash())
            .collect::<std::collections::HashSet<_>>()
            .len()
    }

    /// Child links that resolved to an existing node instead of allocating a
    /// new one. Always `0` when [`Config::dag`] is off.
    pub fn merges(&self) -> u64 {
        self.merges
    }

    /// Zobrist keys that matched a node holding a *different* position.
    ///
    /// Such a match is rejected — the merge is confirmed by full `State`
    /// equality, so a collision costs a duplicated node and never a wrong
    /// merge — and counted here so the rate is observable rather than assumed.
    pub fn key_collisions(&self) -> u64 {
        self.key_collisions
    }

    /// The root's legal actions, in `virus-core` enumeration order. Empty at a
    /// terminal root.
    pub fn root_actions(&self) -> &[Action] {
        &self.nodes[0].actions
    }

    /// The position the tree is currently rooted at.
    ///
    /// The only way to check, from outside, that a re-rooted tree still
    /// describes the position its owner thinks it does. A pondering client
    /// re-roots off snapshots it decoded itself, so the tree's root and the
    /// authoritative snapshot are two independent derivations of the same
    /// position — and ARCHITECTURE.md invariant 5 says the snapshot wins.
    /// Comparing them is what turns "we searched the wrong tree" from a silent
    /// strength regression into a loud, droppable event.
    pub fn root_state(&self) -> &State {
        &self.nodes[0].state
    }

    /// Total visits below the root: the size, in simulations, of the tree that
    /// a re-root kept.
    ///
    /// Distinct from [`MctsSearcher::sims_run`], which is cumulative over the
    /// searcher's whole life and never goes down.
    pub fn root_visit_total(&self) -> u64 {
        u64::from(self.nodes[0].visits)
    }

    /// Per-root-action visit counts, parallel to [`MctsSearcher::root_actions`].
    /// This is the self-play policy target.
    pub fn root_visits(&self) -> &[u32] {
        &self.nodes[0].n
    }

    /// Root priors after any Dirichlet noise, parallel to
    /// [`MctsSearcher::root_actions`].
    pub fn root_priors(&self) -> &[f32] {
        &self.nodes[0].prior
    }

    /// Per-root-action mean value in the **absolute** frame (`W/N`), parallel
    /// to [`MctsSearcher::root_actions`]; `0.0` for an edge with no visits.
    ///
    /// Absolute, like everything the tree stores: multiply by `+1` for a
    /// player-1 root and `-1` for a player-2 one to read it from the mover's
    /// chair. Exposed for the analysis tools (`examples/gumbelprobe`) and for
    /// tests that need to check a frame conversion from outside the crate —
    /// the searcher's own selection never goes through it.
    pub fn root_action_values_abs(&self) -> Vec<f64> {
        let root = &self.nodes[0];
        (0..root.n.len())
            .map(|a| {
                if root.n[a] > 0 {
                    root.w[a] / f64::from(root.n[a])
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Whether this search is running a Gumbel root schedule.
    pub fn is_gumbel(&self) -> bool {
        self.gumbel.is_some()
    }

    /// The **completed-Q improved policy** over the root actions, parallel to
    /// [`MctsSearcher::root_actions`] and summing to 1.
    ///
    /// `softmax(logit(a) + sigma(completedQ(a)))`, the Gumbel MuZero training
    /// target. `completedQ(a)` is the measured `q(a)` for a visited action and
    /// `v_mix` for an unvisited one, where
    ///
    /// ```text
    /// v_mix = (v(root) + N * sum_{n(b)>0} pi(b) q(b) / sum_{n(b)>0} pi(b))
    ///         / (1 + N)
    /// ```
    ///
    /// interpolates the root's own value estimate with the prior-weighted
    /// average of what its visited children measured. That completion is the
    /// whole reason the target is better than visit counts at small budgets:
    /// an action the schedule never visited is scored by what the position is
    /// worth rather than by the zero visits it happens to have.
    ///
    /// Everything is in the **root mover's** frame; the tree's absolute-frame
    /// `w` is converted by the single `sign(root)` multiply, once.
    ///
    /// Defined for a PUCT search too — the formula needs only priors, visits,
    /// `q` and the root value, all of which any search has — so the two arms
    /// can be compared on the same quantity. Empty at a terminal root.
    ///
    /// **Not the emitted `pv`.** See [`crate::gumbel`] and
    /// `virus-selfplay`'s `GumbelPv` for what the row schema can and cannot
    /// carry.
    pub fn root_improved_policy(&self) -> Vec<f32> {
        let root = &self.nodes[0];
        if root.terminal || root.actions.is_empty() {
            return Vec::new();
        }
        // The plan's logits when there is one (they are the pre-noise prior's,
        // which is what the top-m draw sampled from); otherwise the root's own.
        let mut scored = match &self.gumbel {
            Some(plan) => plan.logits().to_vec(),
            None => logits_from_prior(&root.prior),
        };
        for (logit, sigma) in scored.iter_mut().zip(self.completed_q_sigma()) {
            *logit += sigma;
        }
        softmax(&scored)
    }

    /// Root value estimate in the absolute frame.
    pub fn root_value_abs(&self) -> f64 {
        let root = &self.nodes[0];
        if root.terminal {
            return root.terminal_value_abs;
        }
        if root.visits == 0 {
            return 0.0;
        }
        root.w.iter().sum::<f64>() / f64::from(root.visits)
    }

    /// The root's **own** leaf evaluation in the absolute frame: the net's
    /// `v(s_root)`, or the hand-tuned squash when the artifact has no value
    /// head, or the terminal value at a finished position.
    ///
    /// Deliberately not the same number as [`MctsSearcher::root_value_abs`],
    /// which averages the *children*. The completed-Q `v_mix` interpolates the
    /// two, so confusing them collapses the interpolation onto one side — and
    /// it does so silently, in a direction that looks plausible. This is the
    /// accessor that lets a test say which one a re-rooted tree kept.
    pub fn root_leaf_value_abs(&self) -> f64 {
        self.nodes[0].leaf_value_abs
    }

    /// Most-visited root action, ties broken by enumeration order. `None` at a
    /// terminal or stuck root.
    ///
    /// Under a Gumbel schedule this is instead the schedule's own answer — the
    /// argmax of `g + logit + sigma(q)` over the surviving candidates. Visit
    /// counts are the wrong reading there: sequential halving allocates visits
    /// by phase, so the count only ranks the finalists and says nothing about
    /// an action cut in round one.
    pub fn best_action(&self) -> Option<Action> {
        let root = &self.nodes[0];
        if root.terminal || root.actions.is_empty() {
            return None;
        }
        if let Some(plan) = &self.gumbel {
            return Some(root.actions[plan.choice(&self.gumbel_scores())]);
        }
        let mut best = 0;
        for a in 1..root.actions.len() {
            if root.n[a] > root.n[best] {
                best = a;
            }
        }
        Some(root.actions[best])
    }

    /// The action to play: [`MctsSearcher::best_action`] in play mode, or a
    /// draw proportional to the root visit counts when
    /// [`Config::visit_sampling`] is on.
    ///
    /// A Gumbel search answers with [`MctsSearcher::best_action`] whatever
    /// `visit_sampling` says: [`Config::self_play_gumbel`] never sets it, and
    /// the Gumbel draw is already the exploration a sampler would have added.
    pub fn chosen_action(&mut self) -> Option<Action> {
        if !self.config.visit_sampling || self.gumbel.is_some() {
            return self.best_action();
        }
        let root = &self.nodes[0];
        if root.terminal || root.actions.is_empty() {
            return None;
        }
        let total: u64 = root.n.iter().map(|n| u64::from(*n)).sum();
        if total == 0 {
            return Some(root.actions[0]);
        }
        let target = (self.rng.next_f64() * total as f64) as u64;
        let root = &self.nodes[0];
        let mut cumulative = 0u64;
        for (a, n) in root.n.iter().enumerate() {
            cumulative += u64::from(*n);
            if target < cumulative {
                return Some(root.actions[a]);
            }
        }
        Some(root.actions[root.actions.len() - 1])
    }

    /// Re-roots the tree at the child reached by `action`, keeping its subtree
    /// and discarding everything else.
    ///
    /// This is what makes pondering worth anything (superiority.md §2b): the
    /// client searches the opponent-to-move position while they think, and when
    /// they act, the position they moved into is *already in the tree* with its
    /// visit counts intact. Per-action nodes make that a pure book-keeping
    /// operation — an edge is one action, so the opponent's action is exactly
    /// one re-root, and our own turn arrives after three of them.
    ///
    /// Returns `false` and leaves the tree untouched when `action` is not a root
    /// action or has never been expanded into a node; the caller then builds a
    /// fresh searcher. Never partially applies.
    ///
    /// # Memory, and why a DAG does not leak here
    ///
    /// This is the arena's only reclamation point, and it is a *tracing* one:
    /// it walks what is reachable from the new root and moves those nodes into
    /// a fresh `Vec`, so everything else — the discarded siblings and their
    /// whole subtrees — is dropped, not merely orphaned. That is unchanged by
    /// [`Config::dag`]; the breadth-first walk already guards on `mapping`, so
    /// a node reachable through several parents is discovered once, renumbered
    /// once and moved once, and a diamond is not a double free or a double
    /// count.
    ///
    /// What the DAG adds is the **index**, which is the part that would leak if
    /// it were left alone — and would be worse than a leak: a key still
    /// pointing at a discarded node's old slot would merge a future child onto
    /// whatever now occupies that index. So the index is rebuilt from the
    /// survivors, in their new numbering, every time. It is rebuilt rather than
    /// patched because a re-root is already `O(nodes)`, so a second linear pass
    /// costs nothing measurable and cannot go stale the way a patch can. The
    /// rebuild skips node 0, which re-establishes for the new root the property
    /// [`MctsSearcher::new`] gives the original one: the root is never a merge
    /// target, so noised priors are never shared.
    ///
    /// The index's own cost is small enough to state and forget: one
    /// `(u64, u32)` entry per node, against a [`Node`] that owns a whole
    /// `State` plus six per-edge vectors over a ~34-wide action list — order of
    /// a few percent. It is also bounded by the arena rather than by the search
    /// history, precisely because this rebuilds it.
    ///
    /// The overall story is therefore: **nodes are bounded by what a ponder
    /// budget can reach from the current root, and every opponent action
    /// collapses that bound.** The DAG changes nothing about the shape of that
    /// reclamation; what it changes is that the bound is spent on distinct
    /// positions instead of on copies of them.
    pub fn rebase(&mut self, action: Action) -> bool {
        if self.nodes[0].terminal {
            return false;
        }
        let Some(edge) = self.nodes[0].actions.iter().position(|a| *a == action) else {
            return false;
        };
        let child = self.nodes[0].children[edge];
        if child == NO_NODE {
            return false;
        }

        // Breadth-first over the reachable subtree, assigning each node its new
        // index as it is discovered. `order[i]` is the old index of what will
        // become node `i`, so the mapping and the ordering are the same fact.
        let mut mapping = vec![NO_NODE; self.nodes.len()];
        let mut order: Vec<u32> = vec![child];
        mapping[child as usize] = 0;
        let mut head = 0;
        while head < order.len() {
            let id = order[head] as usize;
            head += 1;
            for slot in 0..self.nodes[id].children.len() {
                let next = self.nodes[id].children[slot];
                if next != NO_NODE && mapping[next as usize] == NO_NODE {
                    mapping[next as usize] = order.len() as u32;
                    order.push(next);
                }
            }
        }

        // Move the survivors into a fresh arena rather than copying them: a
        // node owns a whole `State` and two vectors, and this runs on every
        // opponent action.
        let mut slots: Vec<Option<Node>> = std::mem::take(&mut self.nodes)
            .into_iter()
            .map(Some)
            .collect();
        let mut kept = Vec::with_capacity(order.len());
        for old in &order {
            let mut node = slots[*old as usize]
                .take()
                .expect("breadth-first order visits every reachable node exactly once");
            for slot in node.children.iter_mut() {
                if *slot != NO_NODE {
                    // Every child of a reachable node is itself reachable, so
                    // the mapping is always assigned here.
                    *slot = mapping[*slot as usize];
                }
            }
            kept.push(node);
        }
        self.nodes = kept;
        self.rebuild_transpositions();
        // A schedule belongs to one root: its edge indices, its Gumbel draw and
        // its spent budget all describe the position that has just been thrown
        // away. Redrawing is the only defensible answer — carrying it over
        // would rank the new root's edges by the old root's variates, and
        // dropping it would silently turn a Gumbel search into a PUCT one.
        // Self-play (the only caller that sets `gumbel`) builds a fresh
        // searcher per ply and never reaches this, so it is a well-definedness
        // guarantee rather than a hot path.
        //
        // The promoted child carries its own `leaf_value_abs` from the
        // expansion that created it, so `v_mix` is the same quantity a fresh
        // searcher on this position would compute — no re-evaluation, and in
        // particular *not* the child-visit average, which would double-count
        // the children `v_mix` is interpolating against.
        self.gumbel = None;
        self.plan_gumbel();
        true
    }

    /// Re-derives the transposition index from the surviving arena.
    ///
    /// See [`MctsSearcher::rebase`] for why this is a rebuild and not a patch.
    /// Node 0 is skipped so the new root cannot be merged onto.
    fn rebuild_transpositions(&mut self) {
        self.transpositions.clear();
        if !self.config.dag {
            return;
        }
        self.transpositions.reserve(self.nodes.len());
        for id in 1..self.nodes.len() {
            // First writer wins, matching `link_child`: with the survivors in
            // breadth-first order this is the shallowest node holding the key,
            // and it is a deterministic choice either way.
            self.transpositions
                .entry(self.nodes[id].state.hash())
                .or_insert(id as u32);
        }
    }

    // ---------------------------------------------------------------- core

    /// Collects up to `target` descents, evaluates their leaves in one batched
    /// forward, backs the values up and removes the virtual loss.
    ///
    /// Returns the number of simulations actually run, which is `target` unless
    /// the root is terminal.
    fn simulate_round(&mut self, target: u32) -> u32 {
        let target = self.gumbel_prepare(target);
        self.path.clear();
        self.descents.clear();
        self.leaves.clear();
        for _ in 0..target {
            self.descend();
        }
        self.evaluate_leaves();
        self.backup();
        let ran = self.descents.len() as u32;
        self.sims += u64::from(ran);
        ran
    }

    /// Walks from the root to a leaf, applying virtual loss on the way down and
    /// recording the path and the leaf it reached.
    fn descend(&mut self) {
        let start = self.path.len();
        let mut id = 0u32;
        let leaf = loop {
            let (terminal, terminal_value, pending, expanded) = {
                let node = &self.nodes[id as usize];
                (
                    node.terminal,
                    node.terminal_value_abs,
                    node.pending,
                    node.expanded,
                )
            };
            if terminal {
                break self.record_leaf(id, Some(terminal_value));
            }
            if pending {
                // Already queued by an earlier descent in this batch: reuse its
                // forward rather than evaluating the same position twice.
                break self.reuse_leaf(id);
            }
            if !expanded {
                if !self.begin_expand(id as usize) {
                    let value = self.nodes[id as usize].terminal_value_abs;
                    break self.record_leaf(id, Some(value));
                }
                self.nodes[id as usize].pending = true;
                break self.record_leaf(id, None);
            }

            let a = match (id, self.gumbel.as_mut()) {
                // Root under a Gumbel schedule: the edge is dictated, not
                // chosen. Everything below the root keeps ordinary PUCT — the
                // schedule is a *root* bandit and says nothing about the tree.
                (0, Some(plan)) => plan.take(),
                _ => self.select(id as usize),
            };
            let mut child = self.nodes[id as usize].children[a];
            if child == NO_NODE {
                let next = {
                    let node = &self.nodes[id as usize];
                    node.state
                        .apply_generated_with(node.actions[a], &mut self.scratch)
                };
                child = self.link_child(id as usize, a, next);
            }
            self.path.push((id, a as u32));
            let node = &mut self.nodes[id as usize];
            node.vl[a] += 1;
            node.vl_visits += 1;
            id = child;
        };
        self.descents.push(Descent {
            start,
            end: self.path.len(),
            leaf,
        });
    }

    /// Points `parent`'s `edge` at the node for `next`, creating that node only
    /// if the DAG has no node for the position already.
    ///
    /// With [`Config::dag`] off this is exactly the old "push a node, store its
    /// index" — the tree the searcher has always built, allocation for
    /// allocation.
    ///
    /// With it on, a hit in the index is confirmed against the full `State`
    /// before it is used: [`State::hash`] is 64 bits, and a birthday collision
    /// inside a big tree is unlikely but not impossible. A collision loses the
    /// merge (a duplicate node is created, and the incumbent keeps the key)
    /// rather than silently welding two different positions together — the one
    /// failure mode a transposition table must not have.
    fn link_child(&mut self, parent: usize, edge: usize, next: State) -> u32 {
        if self.config.dag {
            if let Some(&existing) = self.transpositions.get(&next.hash()) {
                if self.nodes[existing as usize].state == next {
                    self.merges += 1;
                    self.nodes[parent].children[edge] = existing;
                    return existing;
                }
                self.key_collisions += 1;
            }
        }
        let key = next.hash();
        let child = self.nodes.len() as u32;
        self.nodes.push(Node::new(next));
        self.nodes[parent].children[edge] = child;
        if self.config.dag {
            // Only ever a fresh key: a hit that was confirmed returned above,
            // and a hit that collided leaves the incumbent in place, so the
            // index stays a function of node *creation* order and therefore
            // deterministic.
            self.transpositions.entry(key).or_insert(child);
        }
        child
    }

    fn record_leaf(&mut self, node: u32, value: Option<f64>) -> usize {
        self.leaves.push(Leaf { node, value });
        self.leaves.len() - 1
    }

    /// The batch slot already holding `node`'s pending evaluation.
    fn reuse_leaf(&self, node: u32) -> usize {
        self.leaves
            .iter()
            .position(|leaf| leaf.node == node && leaf.value.is_none())
            .expect("a pending node was recorded as a leaf of this batch")
    }

    /// Runs the net over every leaf still awaiting a value and finishes those
    /// nodes' expansions.
    fn evaluate_leaves(&mut self) {
        self.eval_index.clear();
        for (i, leaf) in self.leaves.iter().enumerate() {
            if leaf.value.is_none() {
                self.eval_index.push(i);
            }
        }
        if self.eval_index.is_empty() {
            return;
        }
        // Buffers are moved out and back so the `&mut self` expansion calls
        // below do not collide with the borrows they would otherwise hold.
        let index = std::mem::take(&mut self.eval_index);

        if index.len() == 1 || self.net.is_none() {
            // One leaf, or no net at all: batching has nothing to amortise, and
            // taking the single-position path keeps `batch_size == 1` exactly
            // the serial searcher.
            for &i in &index {
                let id = self.leaves[i].node as usize;
                let value = self.finish_expand_single(id);
                self.leaves[i].value = Some(value);
                self.nodes[id].pending = false;
            }
            self.eval_index = index;
            return;
        }

        let net = self.net.expect("checked above");
        let mut encoded = std::mem::take(&mut self.encoded);
        let mut heads = std::mem::take(&mut self.heads);
        encoded.clear();
        heads.clear();
        for &i in &index {
            let id = self.leaves[i].node as usize;
            encoded.push(Encoded::from_state(&self.nodes[id].state));
        }
        let scratch = self
            .batch_scratch
            .get_or_insert_with(|| net.batch_scratch());
        net.forward_batch(&encoded, scratch, &mut heads);
        debug_assert_eq!(heads.len(), index.len());
        for (slot, &i) in index.iter().enumerate() {
            let id = self.leaves[i].node as usize;
            let value = self.finish_expand_with(id, Some(&heads[slot]));
            self.leaves[i].value = Some(value);
            self.nodes[id].pending = false;
        }
        self.eval_index = index;
        self.encoded = encoded;
        self.heads = heads;
    }

    /// Credits every edge of every collected descent with its leaf value, and
    /// removes the virtual loss the descent applied.
    ///
    /// `v_abs` is added **as is** at every level. There is no negation here and
    /// there must never be one: the value is already in the absolute frame, and
    /// the mover is reapplied at selection instead. On the 53% of edges that do
    /// not flip the mover, a negamax-style flip here would invert the child's
    /// meaning relative to its parent.
    fn backup(&mut self) {
        for d in 0..self.descents.len() {
            let descent = self.descents[d];
            let v_abs = self.leaves[descent.leaf]
                .value
                .expect("every leaf is valued before backup");
            for step in descent.start..descent.end {
                let (parent, edge) = self.path[step];
                let node = &mut self.nodes[parent as usize];
                let edge = edge as usize;
                node.visits += 1;
                node.n[edge] += 1;
                node.w[edge] += v_abs;
                // Exactly undoes the descent's virtual loss: integer counters,
                // so nothing is left behind.
                node.vl[edge] -= 1;
                node.vl_visits -= 1;
            }
        }
    }

    /// Generates `id`'s legal actions and allocates its edge arrays, returning
    /// `false` when the position has none — in which case the node is marked
    /// terminal and carries its outcome value.
    fn begin_expand(&mut self, id: usize) -> bool {
        let actions = {
            let state = &self.nodes[id].state;
            state.legal_actions_with(&mut self.scratch)
        };
        if actions.is_empty() {
            // Stuck without `game_over` (a snapshot root can be): score it by
            // the real outcome rule rather than guessing.
            let node = &mut self.nodes[id];
            node.terminal = true;
            node.terminal_value_abs = terminal_value_abs(&node.state);
            node.leaf_value_abs = node.terminal_value_abs;
            return false;
        }
        let node = &mut self.nodes[id];
        node.children = vec![NO_NODE; actions.len()];
        node.n = vec![0; actions.len()];
        node.w = vec![0.0; actions.len()];
        node.vl = vec![0; actions.len()];
        node.actions = actions;
        true
    }

    /// Turns one already-[`begin_expand`](Self::begin_expand)ed node's net
    /// outputs into its prior, and returns the leaf value in the absolute
    /// frame.
    ///
    /// Expansion and leaf evaluation are fused on purpose: with a net both come
    /// out of **one** trunk pass. The Java original calls `priors` and then
    /// `valueMover`, paying for the trunk twice per expanded node.
    fn finish_expand_with(&mut self, id: usize, heads: Option<&Heads>) -> f64 {
        let (prior, value_abs) = {
            let node = &self.nodes[id];
            let state = &node.state;
            let mover = node.mover;
            match (self.net, heads) {
                (Some(net), Some(heads)) => {
                    let prior = softmax_over(&node.actions, heads, net.pair_bias(), state.cols());
                    let value = match (self.config.value_source, heads.value) {
                        (ValueSource::Net, Some(v)) => {
                            let v = f64::from(v);
                            Some(if mover == 1 { v } else { -v })
                        }
                        _ => None,
                    };
                    (prior, value)
                }
                _ => (
                    vec![1.0 / node.actions.len() as f32; node.actions.len()],
                    None,
                ),
            }
        };
        let value_abs = value_abs.unwrap_or_else(|| {
            let state = &self.nodes[id].state;
            hand_tuned_value_abs(
                state,
                self.config.value_scale,
                &self.eval_params,
                &mut self.eval_workspace,
            )
        });
        let node = &mut self.nodes[id];
        node.prior = prior;
        node.expanded = true;
        node.leaf_value_abs = value_abs;
        value_abs
    }

    /// [`finish_expand_with`](Self::finish_expand_with) driving its own
    /// single-position forward.
    fn finish_expand_single(&mut self, id: usize) -> f64 {
        let heads = match self.net {
            Some(net) => {
                let encoded = Encoded::from_state(&self.nodes[id].state);
                let scratch = self
                    .net_scratch
                    .as_mut()
                    .expect("a net always brings its scratch");
                Some(net.forward(&encoded, scratch))
            }
            None => None,
        };
        self.finish_expand_with(id, heads.as_ref())
    }

    /// Expands `id` end to end and returns its leaf value. Used for the root,
    /// which is expanded before any batch exists.
    fn expand(&mut self, id: usize) -> f64 {
        if !self.begin_expand(id) {
            return self.nodes[id].terminal_value_abs;
        }
        self.finish_expand_single(id)
    }

    /// PUCT selection: `argmax_a sign(node) * Q_abs(a) + cpuct * P(a) *
    /// sqrt(N + 1) / (1 + n(a))`, over visit counts that include the descents
    /// currently in flight.
    ///
    /// The sign is the *only* place the mover enters. Converting the
    /// absolute-frame `Q` here, instead of negating on the way up, is what
    /// survives the 53% of edges that keep the mover.
    ///
    /// Virtual loss enters as `-virtual_loss` per in-flight descent, *after*
    /// the sign — a loss for whoever is to move at this node, which is what
    /// makes it repel the next descent of the batch regardless of seat. With no
    /// descent in flight every `vl` term is zero and this is the plain PUCT
    /// formula, unchanged.
    fn select(&self, id: usize) -> usize {
        let node = &self.nodes[id];
        let sign = if node.mover == 1 { 1.0 } else { -1.0 };
        let sqrt_n = f64::from(node.visits + node.vl_visits + 1).sqrt();
        let virtual_loss = self.config.virtual_loss;
        let mut best = 0;
        let mut best_score = f64::NEG_INFINITY;
        for a in 0..node.actions.len() {
            let n = node.n[a] + node.vl[a];
            let q = if n > 0 {
                (sign * node.w[a] - virtual_loss * f64::from(node.vl[a])) / f64::from(n)
            } else {
                0.0
            };
            let u = self.config.cpuct * f64::from(node.prior[a]) * sqrt_n / f64::from(1 + n);
            let score = q + u;
            if score > best_score {
                best_score = score;
                best = a;
            }
        }
        best
    }

    /// Settles any due halving and shortens the coming round so the next one
    /// does not straddle a phase boundary.
    ///
    /// Both halves matter. A halving must see statistics that have **backed
    /// up**: a batch has `batch_size` descents in flight with only virtual loss
    /// standing in for their results, so cutting the candidate set mid-batch
    /// would rank several candidates on visits whose values had not arrived.
    /// Clamping the round to [`GumbelPlan::phase_remaining`] makes every phase
    /// boundary a batch boundary; the cost is a short final round per phase
    /// (three of them per search at `m = 16`), which is nothing against the
    /// batch throughput the rest of the schedule keeps.
    fn gumbel_prepare(&mut self, target: u32) -> u32 {
        let Some(plan) = &self.gumbel else {
            return target;
        };
        if plan.needs_halving() {
            let scores = self.gumbel_scores();
            self.gumbel
                .as_mut()
                .expect("checked just above")
                .halve(&scores);
        }
        let remaining = self
            .gumbel
            .as_ref()
            .expect("checked just above")
            .phase_remaining();
        target.min(remaining).max(1)
    }

    /// `sigma(completedQ(a))` for every root edge, in the **root mover's**
    /// frame — `mctx`'s `qtransform_completed_by_mix_value`.
    ///
    /// One function, used by both the sequential-halving ranking and the
    /// improved policy, because in the reference implementation they are the
    /// same transform and letting them drift apart would mean the target no
    /// longer describes the action the search played.
    ///
    /// Three steps, and each is a documented failure if skipped:
    ///
    /// 1. **Complete.** A visited edge contributes its measured `q`; an
    ///    unvisited one contributes `v_mix`, the `(1 + N)`-weighted blend of the
    ///    root's own value with the prior-weighted average of what its visited
    ///    children found. Scoring an unvisited action as `0` instead would rank
    ///    it above every action in a lost position and below every action in a
    ///    won one.
    /// 2. **Rescale** min-max onto `[0, 1]`, so `c_scale` does not have to
    ///    absorb this engine's leaf-value spread.
    /// 3. **Scale** by `(c_visit + max_b N(b)) * c_scale`.
    ///
    /// The frame conversion — absolute `W` to the root mover's chair — is the
    /// single `sign` multiply, applied here and nowhere else in the Gumbel
    /// path (ARCHITECTURE.md invariant 1).
    fn completed_q_sigma(&self) -> Vec<f64> {
        let root = &self.nodes[0];
        let sign = if root.mover == 1 { 1.0 } else { -1.0 };
        let (c_visit, c_scale) = self
            .config
            .gumbel
            .map_or((DEFAULT_GUMBEL_C_VISIT, DEFAULT_GUMBEL_C_SCALE), |gumbel| {
                (gumbel.c_visit, gumbel.c_scale)
            });
        let edges = root.actions.len();

        let q = |a: usize| sign * root.w[a] / f64::from(root.n[a]);
        let visits: u32 = root.n.iter().sum();
        let mut prior_visited = 0.0f64;
        let mut prior_weighted_q = 0.0f64;
        for a in 0..edges {
            if root.n[a] > 0 {
                let prior = f64::from(root.prior[a]);
                prior_visited += prior;
                prior_weighted_q += prior * q(a);
            }
        }
        let v_root = sign * root.leaf_value_abs;
        let v_mix = if visits == 0 || prior_visited <= 0.0 {
            v_root
        } else {
            (v_root + f64::from(visits) * (prior_weighted_q / prior_visited))
                / (1.0 + f64::from(visits))
        };

        let mut completed: Vec<f64> = (0..edges)
            .map(|a| if root.n[a] > 0 { q(a) } else { v_mix })
            .collect();
        rescale(&mut completed);
        let scale = (c_visit + f64::from(root.n.iter().copied().max().unwrap_or(0))) * c_scale;
        for value in completed.iter_mut() {
            *value *= scale;
        }
        completed
    }

    /// `g(a) + logit(a) + sigma(completedQ(a))` for every root edge — the
    /// sequential-halving ranking and the final answer.
    fn gumbel_scores(&self) -> Vec<f64> {
        let plan = self
            .gumbel
            .as_ref()
            .expect("only called on a Gumbel search");
        self.completed_q_sigma()
            .into_iter()
            .enumerate()
            .map(|(a, sigma)| plan.gumbel_logit(a) + sigma)
            .collect()
    }

    /// Mixes Dirichlet noise into the root prior. Self-play exploration only.
    fn apply_root_noise(&mut self) {
        let k = self.nodes[0].prior.len();
        if k == 0 {
            return;
        }
        let mut g = Vec::with_capacity(k);
        let mut sum = 0.0;
        for _ in 0..k {
            let value = self.rng.gamma(self.config.noise_alpha);
            sum += value;
            g.push(value);
        }
        if sum <= 0.0 {
            return;
        }
        let epsilon = self.config.noise_epsilon;
        for (prior, value) in self.nodes[0].prior.iter_mut().zip(&g) {
            *prior = ((1.0 - epsilon) * f64::from(*prior) + epsilon * (value / sum)) as f32;
        }
    }
}

/// Softmax of the net's logits over the node's legal actions only.
///
/// Masked, not full-space: the trainer's 20 880-wide action space is mostly
/// illegal at any given node, and the prior must be a distribution over what
/// the searcher can actually play.
pub(crate) fn softmax_over(
    actions: &[Action],
    heads: &crate::net::Heads,
    pair_bias: f32,
    cols: usize,
) -> Vec<f32> {
    let mut logits: Vec<f32> = actions
        .iter()
        .map(|action| match action {
            Action::Move { target } => heads.move_logits[cell(*target, cols)],
            Action::PlaceNeutrals { cells } => {
                heads.pair_u[cell(cells[0], cols)] + heads.pair_u[cell(cells[1], cols)] + pair_bias
            }
        })
        .collect();
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for logit in logits.iter_mut() {
        *logit = (*logit - max).exp();
        sum += *logit;
    }
    for logit in logits.iter_mut() {
        *logit /= sum;
    }
    logits
}

fn cell(pos: virus_core::Pos, cols: usize) -> usize {
    pos.row as usize * cols + pos.col as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use virus_core::{Cell, CellKind};

    const CELLS_12: usize = 144;

    /// A legal 12x12 two-player position with the mover and `moves_left` we
    /// want. Only the mover matters for the backup/selection unit tests, but
    /// building real states keeps the nodes honest.
    fn state_with(mover: Player, moves_left: u8) -> State {
        let mut cells = vec![Cell::EMPTY; CELLS_12];
        cells[0] = Cell::new(1, CellKind::Base);
        cells[CELLS_12 - 1] = Cell::new(2, CellKind::Base);
        cells[1] = Cell::new(1, CellKind::Normal);
        cells[CELLS_12 - 2] = Cell::new(2, CellKind::Normal);
        State::from_grid(12, 12, 2, &cells, mover, moves_left, &[false, false])
            .expect("hand-built position is legal")
    }

    /// A node with `edges` unvisited children, built by hand rather than by
    /// expansion so the tests below can pick any mover sequence they like.
    fn node_with(mover: Player, moves_left: u8, edges: usize) -> Node {
        let mut node = Node::new(state_with(mover, moves_left));
        node.mover = mover;
        node.expanded = true;
        node.actions = (0..edges).map(|i| Action::mv(0, i as i32)).collect();
        node.prior = vec![1.0 / edges as f32; edges];
        node.children = vec![NO_NODE; edges];
        node.n = vec![0; edges];
        node.w = vec![0.0; edges];
        node.vl = vec![0; edges];
        node
    }

    /// Drives [`MctsSearcher::backup`] over a single hand-built path, the way
    /// one descent would have left the buffers.
    fn backup_path(searcher: &mut MctsSearcher<'_>, path: &[(u32, u32)], v_abs: f64) {
        searcher.path.clear();
        searcher.path.extend_from_slice(path);
        // `backup` removes the virtual loss each step applied, so put it there.
        for (node, edge) in path {
            let node = &mut searcher.nodes[*node as usize];
            node.vl[*edge as usize] += 1;
            node.vl_visits += 1;
        }
        searcher.leaves.clear();
        searcher.leaves.push(Leaf {
            node: 0,
            value: Some(v_abs),
        });
        searcher.descents.clear();
        searcher.descents.push(Descent {
            start: 0,
            end: path.len(),
            leaf: 0,
        });
        searcher.backup();
    }

    /// A searcher whose tree is replaced wholesale, so the unit tests exercise
    /// backup and selection in isolation from expansion.
    fn harness(nodes: Vec<Node>) -> MctsSearcher<'static> {
        let mut searcher = MctsSearcher::new(state_with(1, 3), Config::play(), None);
        searcher.nodes = nodes;
        searcher
    }

    /// ARCHITECTURE.md invariant 1, the 53% case.
    ///
    /// Hand-built 3-node chain over the mover sequence **1 -> 1 -> 2**: the
    /// first edge keeps the mover (a `Move` with `moves_left` 3 -> 2), the
    /// second flips it. A negamax backup negates on both edges and so gets the
    /// first one exactly backwards. Absolute-frame backup adds the same value
    /// at every level.
    #[test]
    fn absolute_frame_backup_never_negates_on_a_non_alternating_path() {
        // root(mover 1, ml 3) -e0-> mid(mover 1, ml 2) -e1-> leaf(mover 2, ml 3)
        let mut searcher = harness(vec![
            node_with(1, 3, 2),
            node_with(1, 2, 2),
            node_with(2, 3, 2),
        ]);
        searcher.nodes[0].children[0] = 1;
        searcher.nodes[1].children[1] = 2;

        // The leaf is a win for player 1, so v_abs = +1 by definition.
        let v_abs = 1.0;
        backup_path(&mut searcher, &[(0, 0), (1, 1)], v_abs);

        assert_eq!(searcher.nodes[0].w[0], v_abs, "root edge keeps the sign");
        assert_eq!(
            searcher.nodes[1].w[1], v_abs,
            "the mover-preserving edge is credited with the SAME value, not its negation"
        );
        assert_eq!(searcher.nodes[0].n[0], 1);
        assert_eq!(searcher.nodes[1].n[1], 1);
        assert_eq!(searcher.nodes[0].visits, 1);
        assert_eq!(searcher.nodes[1].visits, 1);
        // Virtual loss is gone again, exactly.
        assert_eq!(searcher.nodes[0].vl, vec![0, 0]);
        assert_eq!(searcher.nodes[1].vl, vec![0, 0]);
        assert_eq!(searcher.nodes[0].vl_visits, 0);
        assert_eq!(searcher.nodes[1].vl_visits, 0);
        // Untouched edges stay clean.
        assert_eq!(searcher.nodes[0].w[1], 0.0);
        assert_eq!(searcher.nodes[1].w[0], 0.0);

        // The mover re-enters only here. Both nodes hold the identical W, and
        // that identical W means opposite things to opposite movers.
        assert_eq!(searcher.select(0), 0, "player 1 prefers the +1 edge");
        let mut mirrored = searcher;
        mirrored.nodes[0].mover = 2;
        assert_eq!(
            mirrored.select(0),
            1,
            "the same tree, read by player 2, prefers the other edge"
        );
    }

    /// The complementary half: a negating backup produces the mirrored `W` on a
    /// mover-preserving edge, and that tree selects differently. Pinning the
    /// difference stops a future "simplification" to negamax from passing the
    /// test above by accident.
    #[test]
    fn a_negamax_backup_would_pick_the_other_move() {
        let mut searcher = harness(vec![node_with(1, 2, 2), node_with(1, 1, 2)]);
        searcher.nodes[0].children[0] = 1;

        // Absolute frame: the leaf is +1 for player 1 and the child's mover is
        // ALSO player 1, so the edge is worth +1 to the parent.
        backup_path(&mut searcher, &[(0, 0)], 1.0);
        // Give the sibling a modest positive value so the choice is a real one.
        searcher.nodes[0].n[1] = 1;
        searcher.nodes[0].w[1] = 0.5;
        searcher.nodes[0].visits = 2;
        assert_eq!(searcher.select(0), 0, "absolute frame: +1 beats +0.5");

        // What negamax would have stored on that edge instead.
        searcher.nodes[0].w[0] = -1.0;
        assert_eq!(
            searcher.select(0),
            1,
            "a negated backup flips the decision — the bug this frame prevents"
        );
    }

    /// Selection's sign is the *node's* mover, never the path's parity.
    #[test]
    fn selection_sign_follows_the_node_mover() {
        let mut searcher = harness(vec![node_with(2, 3, 2)]);
        searcher.nodes[0].n = vec![1, 1];
        searcher.nodes[0].w = vec![0.9, -0.9];
        searcher.nodes[0].visits = 2;
        assert_eq!(
            searcher.select(0),
            1,
            "player 2 maximises -Q_abs, so the player-1-favourable edge is the worse one"
        );
        searcher.nodes[0].mover = 1;
        assert_eq!(searcher.select(0), 0);
    }

    /// The hand-tuned leaf is flipped exactly once, at the leaf.
    ///
    /// Rotating the board 180 degrees and swapping the two seats maps every
    /// position onto its mirror; the bases land on each other's corners, so the
    /// mirror is a legal position with the roles exchanged. The absolute-frame
    /// value of a position and of its mirror must therefore be exact
    /// negations — which is only true if the mover sign is applied once, at the
    /// leaf.
    #[test]
    fn the_hand_tuned_leaf_is_flipped_once_by_the_mover() {
        let params = EvalParams::default();
        let mut workspace = EvalWorkspace::new();

        let mut cells = vec![Cell::EMPTY; CELLS_12];
        cells[0] = Cell::new(1, CellKind::Base);
        cells[CELLS_12 - 1] = Cell::new(2, CellKind::Base);
        // Deliberately lopsided, so the value is nowhere near zero.
        for index in [1, 12, 13, 14, 25] {
            cells[index] = Cell::new(1, CellKind::Normal);
        }
        cells[CELLS_12 - 2] = Cell::new(2, CellKind::Normal);

        let swap = |cell: Cell| match cell.kind() {
            CellKind::Empty | CellKind::Neutral => cell,
            kind => Cell::new(if cell.owner() == 1 { 2 } else { 1 }, kind),
        };
        let mirrored: Vec<Cell> = (0..CELLS_12).rev().map(|i| swap(cells[i])).collect();

        let direct = State::from_grid(12, 12, 2, &cells, 1, 3, &[false, false]).expect("legal");
        let mirror = State::from_grid(12, 12, 2, &mirrored, 2, 3, &[false, false]).expect("legal");

        let v1 = hand_tuned_value_abs(&direct, DEFAULT_VALUE_SCALE, &params, &mut workspace);
        let v2 = hand_tuned_value_abs(&mirror, DEFAULT_VALUE_SCALE, &params, &mut workspace);
        assert!(
            v1.abs() <= 1.0 && v2.abs() <= 1.0,
            "tanh keeps values bounded"
        );
        assert!(
            v1.abs() > 1e-6,
            "the test position must be decisive, got {v1}"
        );
        assert!(
            (v1 + v2).abs() < 1e-12,
            "mirror image must invert the absolute value: {v1} vs {v2}"
        );
    }

    // ------------------------------------------------------------- re-rooting

    #[test]
    fn rebasing_keeps_the_subtree_and_its_statistics() {
        let mut searcher = MctsSearcher::new(state_with(1, 3), Config::play(), None);
        searcher.run_sims(400);
        let before = searcher.nodes.len();

        let action = searcher
            .best_action()
            .expect("a searched root has a best action");
        let edge = searcher.nodes[0]
            .actions
            .iter()
            .position(|a| *a == action)
            .expect("the best action is a root action");
        let child = searcher.nodes[0].children[edge] as usize;
        let expected_state = searcher.nodes[child].state.clone();
        let expected_visits = searcher.nodes[child].visits;
        let expected_n = searcher.nodes[child].n.clone();
        let expected_w = searcher.nodes[child].w.clone();
        assert!(expected_visits > 0, "the leader must have been searched");

        assert!(searcher.rebase(action));

        assert_eq!(searcher.nodes[0].state, expected_state);
        assert_eq!(
            searcher.nodes[0].visits, expected_visits,
            "the whole point is that the work already done survives"
        );
        assert_eq!(searcher.nodes[0].n, expected_n);
        assert_eq!(searcher.nodes[0].w, expected_w);
        assert!(
            searcher.nodes.len() < before,
            "the discarded siblings must be freed, not merely orphaned"
        );
    }

    /// Every child index has to be rewritten into the compacted arena. A stale
    /// index would make the next simulation read a sibling's statistics.
    #[test]
    fn rebasing_rewrites_every_child_index() {
        let mut searcher = MctsSearcher::new(state_with(1, 3), Config::play(), None);
        searcher.run_sims(600);
        let action = searcher.best_action().expect("a best action");
        assert!(searcher.rebase(action));

        let count = searcher.nodes.len() as u32;
        for (id, node) in searcher.nodes.iter().enumerate() {
            for child in &node.children {
                if *child != NO_NODE {
                    assert!(
                        *child < count && *child as usize != id,
                        "node {id} points at {child}, outside a {count}-node arena"
                    );
                }
            }
        }
        // And the tree is still usable: more simulations must not panic and
        // must keep accumulating.
        let before = searcher.sims_run();
        searcher.run_sims(200);
        assert_eq!(searcher.sims_run(), before + 200);
        assert!(searcher.best_action().is_some());
    }

    /// A re-rooted tree must reach the same conclusion as one grown from the
    /// same position: re-rooting is book-keeping, not a different search.
    #[test]
    fn rebasing_reaches_the_same_position_as_applying_the_action() {
        let root = state_with(1, 3);
        let mut searcher = MctsSearcher::new(root.clone(), Config::play(), None);
        searcher.run_sims(300);
        let action = searcher.best_action().expect("a best action");
        let applied = root.apply(action).expect("the searched action is legal");

        assert!(searcher.rebase(action));
        assert_eq!(searcher.nodes[0].state, applied);
        assert_eq!(searcher.nodes[0].mover, applied.current_player());
    }

    /// Batching's virtual loss must be gone by the time a re-root happens.
    ///
    /// Hypothesis 1 of bd `vsbot-gei`: a re-root taken while `vl`/`vl_visits`
    /// were non-zero would orphan those adjustments into the surviving subtree,
    /// leaving edges that look permanently worse than they are — a systematic
    /// `Q` corruption that would read exactly like "confident bad moves". The
    /// structural claim that rules it out is that a round is atomic
    /// (descend/evaluate/back up), so this asserts the whole arena is clean at
    /// every point a caller could possibly re-root from.
    #[test]
    fn a_batched_search_leaves_no_virtual_loss_behind_for_a_re_root_to_inherit() {
        for batch_size in [1, 2, 8, 16] {
            let config = Config {
                batch_size,
                ..Config::play()
            };
            let mut searcher = MctsSearcher::new(state_with(1, 3), config, None);
            // Deliberately not a multiple of the batch size, so the last round
            // is a partial one.
            searcher.run_sims(453);
            assert_clean(&searcher, batch_size, "after simulating");

            let action = searcher.best_action().expect("a best action");
            assert!(searcher.rebase(action));
            assert_clean(&searcher, batch_size, "after re-rooting");

            searcher.run_sims(211);
            assert_clean(&searcher, batch_size, "after simulating a re-rooted tree");
        }
    }

    /// Every node's book-keeping is self-consistent and free of in-flight state.
    fn assert_clean(searcher: &MctsSearcher<'_>, batch_size: u16, when: &str) {
        for (id, node) in searcher.nodes.iter().enumerate() {
            assert!(
                node.vl.iter().all(|v| *v == 0),
                "batch {batch_size}: node {id} still carries virtual loss {:?} {when}",
                node.vl
            );
            assert_eq!(
                node.vl_visits, 0,
                "batch {batch_size}: node {id} still counts in-flight descents {when}"
            );
            assert!(
                !node.pending,
                "batch {batch_size}: node {id} is still marked pending {when}"
            );
            assert_eq!(
                node.visits,
                node.n.iter().sum::<u32>(),
                "batch {batch_size}: node {id}'s visit total disagrees with its edges {when}"
            );
            if node.expanded {
                assert_eq!(
                    node.prior.len(),
                    node.actions.len(),
                    "batch {batch_size}: node {id} has a prior of the wrong width {when}"
                );
            }
        }
    }

    /// The tree's root position is what a caller re-rooting off snapshots has to
    /// check itself against (bd `vsbot-gei`'s permanent assertion).
    #[test]
    fn the_root_state_and_visit_total_describe_the_tree_that_survived_a_re_root() {
        let root = state_with(1, 3);
        let mut searcher = MctsSearcher::new(root.clone(), Config::play(), None);
        assert_eq!(searcher.root_state(), &root);
        assert_eq!(
            searcher.root_visit_total(),
            0,
            "an unsearched root is empty"
        );

        searcher.run_sims(400);
        assert_eq!(
            searcher.root_state(),
            &root,
            "searching does not move the root"
        );
        assert_eq!(searcher.root_visit_total(), 400);

        let action = searcher.best_action().expect("a best action");
        let applied = root.apply(action).expect("the searched action is legal");
        let inherited = {
            let edge = searcher.nodes[0]
                .actions
                .iter()
                .position(|a| *a == action)
                .expect("a root action");
            u64::from(searcher.nodes[searcher.nodes[0].children[edge] as usize].visits)
        };
        assert!(searcher.rebase(action));
        assert_eq!(
            searcher.root_state(),
            &applied,
            "the tree must be rooted at the position the action reaches"
        );
        assert_eq!(
            searcher.root_visit_total(),
            inherited,
            "the visit total must describe what the re-root kept, not the whole run"
        );
        assert!(
            searcher.sims_run() > searcher.root_visit_total(),
            "the cumulative simulation count must survive the re-root that the visit \
             total does not"
        );
    }

    // ------------------------------------------------------ DAG transpositions

    /// Every `(parent, edge)` pointing at each node, i.e. the DAG read upwards.
    fn parents_of(searcher: &MctsSearcher<'_>) -> Vec<Vec<(u32, usize)>> {
        let mut parents = vec![Vec::new(); searcher.nodes.len()];
        for (id, node) in searcher.nodes.iter().enumerate() {
            for (edge, child) in node.children.iter().enumerate() {
                if *child != NO_NODE {
                    parents[*child as usize].push((id as u32, edge));
                }
            }
        }
        parents
    }

    /// A position whose turn has all three actions left, searched hard enough
    /// that the top of the tree is fully built out.
    fn searched(dag: bool, batch_size: u16, sims: u32) -> MctsSearcher<'static> {
        let config = Config {
            dag,
            batch_size,
            ..Config::play()
        };
        let mut searcher = MctsSearcher::new(state_with(1, 3), config, None);
        searcher.run_sims(sims);
        searcher
    }

    /// The acceptance case, stated directly: two orders of the same two actions
    /// reach one node.
    ///
    /// The pairs are taken from the tree the search actually built rather than
    /// hand-picked, and then *re-derived independently* through
    /// `State::apply` — so the assertion is not "the searcher agrees with
    /// itself" but "the node both orders share really is the position the rules
    /// say both orders reach".
    #[test]
    fn a_within_turn_permutation_reaches_exactly_one_node() {
        let searcher = searched(true, 1, 4_000);
        let root = &searcher.nodes[0];
        let mut diamonds = 0;

        for (ea, ca) in root.children.iter().enumerate() {
            for (eb, cb) in root.children.iter().enumerate() {
                if ea >= eb || *ca == NO_NODE || *cb == NO_NODE {
                    continue;
                }
                let (a, b) = (root.actions[ea], root.actions[eb]);
                let (ca, cb) = (*ca as usize, *cb as usize);
                // The grandchild reached as a-then-b, and as b-then-a.
                let ab = searcher.nodes[ca]
                    .actions
                    .iter()
                    .position(|x| *x == b)
                    .map(|e| searcher.nodes[ca].children[e]);
                let ba = searcher.nodes[cb]
                    .actions
                    .iter()
                    .position(|x| *x == a)
                    .map(|e| searcher.nodes[cb].children[e]);
                let (Some(ab), Some(ba)) = (ab, ba) else {
                    continue;
                };
                if ab == NO_NODE || ba == NO_NODE {
                    continue;
                }

                // Independent derivation, straight from the rules.
                let root_state = &searcher.nodes[0].state;
                let via_ab = root_state.apply(a).and_then(|s| s.apply(b));
                let via_ba = root_state.apply(b).and_then(|s| s.apply(a));
                let (Ok(via_ab), Ok(via_ba)) = (via_ab, via_ba) else {
                    continue;
                };
                assert_eq!(
                    via_ab, via_ba,
                    "the two orders of {a:?} and {b:?} do not commute after all"
                );

                assert_eq!(
                    ab, ba,
                    "{a:?} then {b:?} and {b:?} then {a:?} reach the same position but \
                     landed on nodes {ab} and {ba}"
                );
                assert_eq!(
                    searcher.nodes[ab as usize].state, via_ab,
                    "the merged node holds a different position than the rules reach"
                );
                assert_eq!(
                    searcher.nodes[ab as usize].state.hash(),
                    via_ab.hash(),
                    "invariant 6: the key must agree with the position"
                );
                diamonds += 1;
            }
        }
        assert!(
            diamonds > 0,
            "no permutation pair was explored — the test proved nothing"
        );
    }

    /// The same tree built without [`Config::dag`] keeps those permutations
    /// apart, which is what makes the test above about the DAG and not about
    /// the position.
    #[test]
    fn without_the_dag_the_same_permutation_is_two_nodes() {
        let searcher = searched(false, 1, 4_000);
        let root = &searcher.nodes[0];
        let mut split = 0;
        for (ea, ca) in root.children.iter().enumerate() {
            for (eb, cb) in root.children.iter().enumerate() {
                if ea >= eb || *ca == NO_NODE || *cb == NO_NODE {
                    continue;
                }
                let (a, b) = (root.actions[ea], root.actions[eb]);
                let (ca, cb) = (*ca as usize, *cb as usize);
                let child_of = |node: usize, action: Action| {
                    searcher.nodes[node]
                        .actions
                        .iter()
                        .position(|x| *x == action)
                        .map(|e| searcher.nodes[node].children[e])
                        .filter(|c| *c != NO_NODE)
                };
                let (Some(ab), Some(ba)) = (child_of(ca, b), child_of(cb, a)) else {
                    continue;
                };
                if searcher.nodes[ab as usize].state == searcher.nodes[ba as usize].state {
                    assert_ne!(ab, ba, "a plain tree must not share nodes");
                    split += 1;
                }
            }
        }
        assert!(split > 0, "the comparison arm explored no permutation pair");
        assert_eq!(searcher.merges(), 0, "the flag is off");
        assert!(searcher.transpositions.is_empty(), "the index is not built");
    }

    /// Merged statistics, as an exact identity rather than an inequality.
    ///
    /// At `batch_size == 1` nothing is ever in flight, so a descent that
    /// reaches a node either **stops** there — which happens exactly once per
    /// node, on the visit that expands it — or **passes through**, crediting
    /// one edge and one `visits`. So for every non-root, non-terminal node the
    /// visits arriving from *all* its parents must equal its own visit total
    /// plus that single expansion stop. With the DAG on, "all its parents" is
    /// genuinely plural, and this identity is what says the node pools their
    /// work instead of double-counting or losing it.
    #[test]
    fn a_merged_node_pools_the_visits_of_every_parent() {
        let searcher = searched(true, 1, 4_000);
        let parents = parents_of(&searcher);
        let mut shared = 0;

        for (id, node) in searcher.nodes.iter().enumerate().skip(1) {
            let arrivals: u32 = parents[id]
                .iter()
                .map(|(parent, edge)| searcher.nodes[*parent as usize].n[*edge])
                .sum();
            if node.terminal {
                assert_eq!(node.visits, 0, "node {id} is terminal but was walked past");
                continue;
            }
            if !node.expanded {
                assert_eq!(arrivals, 0, "node {id} was visited without being expanded");
                continue;
            }
            assert_eq!(
                arrivals,
                node.visits + 1,
                "node {id}: {} parents delivered {arrivals} visits but the node counts {}",
                parents[id].len(),
                node.visits
            );
            if parents[id].len() > 1 {
                shared += 1;
                assert!(
                    parents[id]
                        .iter()
                        .all(|(p, e)| searcher.nodes[*p as usize].n[*e] > 0),
                    "node {id} has a parent edge that never carried a visit"
                );
            }
        }
        assert!(
            shared > 0,
            "nothing was merged — the identity proved nothing"
        );
    }

    /// **What the DAG actually saves, stated exactly.**
    ///
    /// Not nodes per simulation: a simulation expands exactly one leaf either
    /// way, so both arms hold `sims + 1` nodes at `batch_size == 1`. What
    /// changes is *what those expansions were spent on*. A plain tree re-expands
    /// positions it has already evaluated — a second net forward, a second
    /// subtree, a second set of statistics that never learns from the first. The
    /// DAG's arena is duplicate-free by construction, so every forward it pays
    /// for buys a position it has never seen.
    ///
    /// The saving is therefore the plain tree's duplicate count, and that is
    /// what this measures. (Batching adds a second, smaller saving on top: a
    /// merged node reached twice inside one batch is `pending` the second time,
    /// so the round runs fewer forwards than it had descents. That one *does*
    /// show up as nodes, and is asserted below.)
    #[test]
    fn the_dag_spends_every_expansion_on_a_position_it_has_not_seen() {
        let distinct = |searcher: &MctsSearcher<'_>| {
            searcher
                .nodes
                .iter()
                .map(|node| node.state.hash())
                .collect::<std::collections::HashSet<_>>()
                .len()
        };

        let with = searched(true, 1, 4_000);
        let without = searched(false, 1, 4_000);
        assert_eq!(with.sims_run(), without.sims_run());

        assert_eq!(
            distinct(&with),
            with.node_count(),
            "the DAG's arena must hold no duplicate position at all"
        );
        let duplicates = without.node_count() - distinct(&without);
        assert!(
            duplicates > 0,
            "the plain tree expanded no duplicate — nothing to save"
        );
        assert!(with.merges() > 0);
        assert_eq!(
            with.key_collisions(),
            0,
            "a 64-bit key should not collide in a {}-node arena",
            with.node_count()
        );

        // With a batch, leaf reuse turns some of those merges into forwards the
        // round never runs, which is visible as a smaller arena.
        let with = searched(true, 8, 4_000);
        let without = searched(false, 8, 4_000);
        assert!(
            with.node_count() < without.node_count(),
            "batched DAG expanded {} nodes, batched tree {}",
            with.node_count(),
            without.node_count()
        );
    }

    /// Dirichlet root noise lives in node 0's `prior` and nowhere else, so the
    /// one thing that would leak it is an interior edge pointing back at the
    /// root. The index makes that structurally impossible; this checks it.
    #[test]
    fn the_root_is_never_a_merge_target_so_its_noised_prior_cannot_leak() {
        let config = Config {
            dag: true,
            root_noise: true,
            seed: 0xD1CE,
            ..Config::play()
        };
        let mut searcher = MctsSearcher::new(state_with(1, 3), config, None);
        let noised = searcher.nodes[0].prior.clone();
        searcher.run_sims(2_000);

        assert!(
            !searcher.transpositions.values().any(|id| *id == 0),
            "the root is in the transposition index"
        );
        for (id, node) in searcher.nodes.iter().enumerate() {
            assert!(
                node.children.iter().all(|c| *c != 0),
                "node {id} points back at the root"
            );
        }
        assert_eq!(
            searcher.nodes[0].prior, noised,
            "the search overwrote the noised root prior"
        );
        // And the noise really was there to leak.
        let plain = MctsSearcher::new(
            state_with(1, 3),
            Config {
                dag: true,
                ..Config::play()
            },
            None,
        );
        assert_ne!(
            plain.nodes[0].prior, noised,
            "the root prior was not actually noised"
        );
    }

    /// A re-root must leave the index describing the arena it now has: every
    /// key resolving to a live node that really holds that position, nothing
    /// pointing at the new root, and nothing left over from the discarded
    /// subtrees.
    #[test]
    fn re_rooting_rebuilds_the_index_and_reclaims_the_unreachable_nodes() {
        for batch_size in [1u16, 8] {
            let mut searcher = searched(true, batch_size, 3_000);
            let before_nodes = searcher.node_count();
            let before_keys = searcher.transpositions.len();
            assert!(before_keys > 0);

            let action = searcher.best_action().expect("a best action");
            assert!(searcher.rebase(action));

            assert!(
                searcher.node_count() < before_nodes,
                "batch {batch_size}: nothing was reclaimed"
            );
            assert!(
                searcher.transpositions.len() < before_keys,
                "batch {batch_size}: the index still holds discarded keys"
            );
            assert_index_sound(&searcher, batch_size, "after re-rooting");

            // Reachability: the arena must be exactly what the new root reaches.
            let mut seen = vec![false; searcher.node_count()];
            let mut queue = vec![0u32];
            seen[0] = true;
            while let Some(id) = queue.pop() {
                for child in &searcher.nodes[id as usize].children {
                    if *child != NO_NODE && !seen[*child as usize] {
                        seen[*child as usize] = true;
                        queue.push(*child);
                    }
                }
            }
            assert!(
                seen.iter().all(|s| *s),
                "batch {batch_size}: the arena kept a node the root cannot reach"
            );

            searcher.run_sims(1_000);
            assert_clean(&searcher, batch_size, "after searching a re-rooted DAG");
            assert_index_sound(&searcher, batch_size, "after searching a re-rooted DAG");
            assert!(searcher.best_action().is_some());
        }
    }

    /// Every index entry names a live node that holds exactly that key, and the
    /// root is not among them.
    fn assert_index_sound(searcher: &MctsSearcher<'_>, batch_size: u16, when: &str) {
        for (key, id) in &searcher.transpositions {
            assert!(
                (*id as usize) < searcher.nodes.len(),
                "batch {batch_size}: key {key:#018x} points outside the arena {when}"
            );
            assert_ne!(
                *id, 0,
                "batch {batch_size}: key {key:#018x} points at the root {when}"
            );
            assert_eq!(
                searcher.nodes[*id as usize].state.hash(),
                *key,
                "batch {batch_size}: node {id} does not hold key {key:#018x} {when}"
            );
        }
        assert!(
            searcher.transpositions.len() < searcher.nodes.len().max(1),
            "batch {batch_size}: the index cannot have an entry per node plus the root {when}"
        );
    }

    /// Batching's leaf reuse and the DAG compound: a merged node reached twice
    /// in one batch is `pending` the second time. The arena must still come out
    /// of every round clean.
    #[test]
    fn a_batched_dag_search_leaves_nothing_in_flight() {
        for batch_size in [1u16, 2, 8, 16] {
            let mut searcher = searched(true, batch_size, 1_453);
            assert_clean(&searcher, batch_size, "after simulating a DAG");
            assert_index_sound(&searcher, batch_size, "after simulating a DAG");
            assert!(searcher.merges() > 0, "batch {batch_size}: nothing merged");
            let action = searcher.best_action().expect("a best action");
            assert!(searcher.rebase(action));
            assert_clean(&searcher, batch_size, "after re-rooting a DAG");
        }
    }

    #[test]
    fn rebasing_refuses_an_unreachable_action_without_touching_the_tree() {
        let mut searcher = MctsSearcher::new(state_with(1, 3), Config::play(), None);
        searcher.run_sims(50);
        let before = searcher.nodes.len();

        // Not a root action at all.
        assert!(!searcher.rebase(Action::mv(11, 5)));
        assert_eq!(searcher.nodes.len(), before);

        // A root action that 50 simulations never expanded into a node.
        let unvisited = searcher.nodes[0]
            .actions
            .iter()
            .zip(&searcher.nodes[0].children)
            .find(|(_, child)| **child == NO_NODE)
            .map(|(action, _)| *action);
        if let Some(action) = unvisited {
            assert!(!searcher.rebase(action));
            assert_eq!(searcher.nodes.len(), before);
        }
    }
}
