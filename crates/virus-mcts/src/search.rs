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
//! # Randomness
//!
//! Play mode draws nothing: `run_sims` and `run_until_deadline` are pure
//! functions of the position, the config and the net, batched or not. Dirichlet
//! root noise and temperature sampling are self-play only, and both run off the
//! seeded [`Rng`], so even those are reproducible.
//!
//! Thread parallelism lives in [`crate::parallel::ParallelMcts`], a separate
//! opt-in type: this searcher is single-threaded and deterministic, full stop.

use std::time::{Duration, Instant};

use virus_core::{Action, Player, Scratch, State};
use virus_eval::{evaluate, EvalParams, EvalWorkspace};

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
/// Measured on this box with `examples/mctsbench`: throughput plateaus at
/// **8**, where the batched trunk first fills its vector width, and 8 through
/// 48 all land within measurement noise of each other at 2.1-2.5x the serial
/// searcher. Below 8 the batch is a *loss* — [`crate::net::BATCH_LANES`] is the
/// group size, so a round of 2 leaves pays for a group of 8 and measured 0.6x
/// serial. Sixteen is the default because it is the smallest multiple of the
/// lane width that keeps a round full even when a few of its descents land on
/// terminal nodes or on a leaf another descent already claimed, and it holds
/// the staleness a batch introduces to two lane groups' worth.
///
/// Set it to `1` for the serial searcher; anything between 2 and 7 is strictly
/// worse than either.
pub const DEFAULT_BATCH_SIZE: u16 = 16;

/// Default virtual-loss weight, in leaf-value units.
///
/// One whole loss, the AlphaGo setting. Leaf values live in `[-1, 1]`, so an
/// edge with one descent in flight and no real visits scores `Q = -1` — bad
/// enough that the next descent in the batch takes a genuinely different
/// branch, and transient, because it is removed the moment the batch backs up.
pub const DEFAULT_VIRTUAL_LOSS: f64 = 1.0;

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
        }
    }
}

impl Config {
    /// The play-mode configuration: no noise, no sampling, no RNG draws at all.
    pub fn play() -> Config {
        Config::default()
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
        let net_scratch = net.map(|net| net.scratch());
        let mut searcher = MctsSearcher {
            config,
            net,
            nodes: vec![Node::new(state)],
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
            searcher.expand(0);
            if !searcher.nodes[0].terminal && searcher.config.root_noise {
                searcher.apply_root_noise();
            }
        }
        searcher
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
    /// to one batch — at the tuned [`DEFAULT_BATCH_SIZE`] a couple of
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

    /// The root's legal actions, in `virus-core` enumeration order. Empty at a
    /// terminal root.
    pub fn root_actions(&self) -> &[Action] {
        &self.nodes[0].actions
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

    /// Most-visited root action, ties broken by enumeration order. `None` at a
    /// terminal or stuck root.
    pub fn best_action(&self) -> Option<Action> {
        let root = &self.nodes[0];
        if root.terminal || root.actions.is_empty() {
            return None;
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
    pub fn chosen_action(&mut self) -> Option<Action> {
        if !self.config.visit_sampling {
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

    // ---------------------------------------------------------------- core

    /// Collects up to `target` descents, evaluates their leaves in one batched
    /// forward, backs the values up and removes the virtual loss.
    ///
    /// Returns the number of simulations actually run, which is `target` unless
    /// the root is terminal.
    fn simulate_round(&mut self, target: u32) -> u32 {
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

            let a = self.select(id as usize);
            let mut child = self.nodes[id as usize].children[a];
            if child == NO_NODE {
                let next = {
                    let node = &self.nodes[id as usize];
                    node.state
                        .apply_generated_with(node.actions[a], &mut self.scratch)
                };
                child = self.nodes.len() as u32;
                self.nodes.push(Node::new(next));
                self.nodes[id as usize].children[a] = child;
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
}
