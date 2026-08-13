//! Tree-parallel PUCT: worker threads descending one shared tree.
//!
//! This is the **secondary** throughput axis and it is opt-in. The primary one
//! is leaf batching, which lives in [`crate::search`] and needs no threads at
//! all; PR #8 measured shared-table lazy SMP in the alpha-beta stack gaining
//! nothing on this 4-vCPU box, and the honest expectation here was the same.
//! See the PR body for what it actually measured.
//!
//! # Why the shared tree is safe to write to concurrently
//!
//! ARCHITECTURE.md invariant 1 does most of the work. Backup is
//! *absolute-frame*: `W` accumulates in one fixed frame with no negation
//! anywhere on the way up, and the mover enters only at selection. So a backup
//! is a plain commutative add into a per-edge accumulator — it does not depend
//! on the path it came from, on the order relative to other backups, or on any
//! node's state at the time it lands. Two workers backing up through the same
//! edge in either order produce the same `W`. A negamax tree, whose backup sign
//! depends on ply parity, would need far more care.
//!
//! What is left to synchronise is therefore only *structure*:
//!
//! * **Node statistics** (`n`, `w`, `vl`, `visits`) are per-edge atomics,
//!   `Relaxed`. They are heuristics that steer selection; a read that misses a
//!   concurrent increment picks a slightly stale-but-valid branch, which is
//!   exactly what virtual loss is already perturbing on purpose. `w` is an
//!   `AtomicU64` of `f64` bits updated by a compare-exchange loop, so a
//!   single-threaded run adds in `f64` exactly like [`crate::MctsSearcher`]
//!   does — which is what lets `parallel_with_one_thread_matches_the_serial_searcher`
//!   assert bit equality rather than a tolerance.
//! * **Expansion and child creation** go through `OnceLock`, which carries the
//!   release/acquire edge that publishes the `Edges` a worker wrote. Whoever
//!   gets there first wins; there is no global lock and no lock at all on the
//!   read path once a node is expanded.
//!
//! There is no `unsafe` here and no big mutex. The only blocking primitive is
//! `OnceLock::get_or_init` on a single child slot, held for the duration of one
//! `apply` — microseconds, on one edge.
//!
//! # Expansion collisions
//!
//! Batching defers a leaf's net forward to the end of the round, so a worker
//! cannot expand a node inline; it *claims* the node with an `expanding` flag
//! and evaluates it with the rest of its batch. A second worker that reaches a
//! node already claimed by someone else has no value to back up and no way to
//! wait for one without stalling its whole batch, so it **abandons** that
//! descent: it removes its virtual loss and takes no simulation credit. The
//! abandoned descents are counted and reported by
//! [`ParallelMcts::collisions`], because a collision rate that is not small is
//! the first thing to look at when threads fail to scale.
//!
//! # Determinism
//!
//! With `threads == 1` this engine is deterministic and identical to
//! [`crate::MctsSearcher`] at the same batch size, edge for edge. With more
//! than one thread it is **not** deterministic and cannot be made so: workers
//! interleave. Nothing is gated on it — the parity fixtures, the determinism
//! tests and the play path all run on the serial searcher — and it is reached
//! only by explicitly asking for `threads > 1`, the same opt-in discipline the
//! alpha-beta SMP work used.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use virus_core::{Action, Player, Scratch, State};
use virus_eval::{EvalParams, EvalWorkspace};

use crate::net::{Encoded, Heads, PolicyValueNet, BOARD};
use crate::rng::Rng;
use crate::search::{hand_tuned_value_abs, softmax_over, terminal_value_abs, Config, ValueSource};

/// A node's edges: fixed once expansion publishes them, apart from the
/// statistics, which are atomics.
#[derive(Debug)]
struct Edges {
    actions: Vec<Action>,
    prior: Vec<f32>,
    n: Vec<AtomicU32>,
    /// Absolute-frame value sums, as `f64` bits.
    w: Vec<AtomicU64>,
    /// Descents currently in flight through each edge.
    vl: Vec<AtomicU32>,
    children: Vec<OnceLock<Arc<SharedNode>>>,
    /// The leaf value the expanding worker computed, so a worker that lost the
    /// race can back up the same number instead of recomputing it.
    leaf_value_abs: f64,
}

/// One node of the shared tree.
#[derive(Debug)]
struct SharedNode {
    state: State,
    mover: Player,
    /// Set exactly when this node is terminal, to its absolute-frame value.
    /// Populated at construction for a `game_over` position and on discovery
    /// for a position that turns out to have no legal action.
    outcome: OnceLock<f64>,
    edges: OnceLock<Edges>,
    /// Claimed by the worker that will evaluate this node in its batch.
    expanding: AtomicBool,
    visits: AtomicU32,
    vl_visits: AtomicU32,
}

impl SharedNode {
    fn new(state: State) -> Arc<SharedNode> {
        let mover = state.current_player();
        let outcome = OnceLock::new();
        if state.game_over() {
            outcome
                .set(terminal_value_abs(&state))
                .expect("a fresh OnceLock is empty");
        }
        Arc::new(SharedNode {
            state,
            mover,
            outcome,
            edges: OnceLock::new(),
            expanding: AtomicBool::new(false),
            visits: AtomicU32::new(0),
            vl_visits: AtomicU32::new(0),
        })
    }

    /// Marks a node that generated no legal actions as terminal.
    fn mark_stuck(&self) -> f64 {
        *self.outcome.get_or_init(|| terminal_value_abs(&self.state))
    }
}

/// Adds `delta` to an `f64` stored as atomic bits.
///
/// A compare-exchange loop rather than a fixed-point `fetch_add`: the point is
/// that a single-threaded run performs exactly the `f64` additions the serial
/// searcher performs, in the same order, so the two trees are bit-identical.
fn add_f64(slot: &AtomicU64, delta: f64) {
    let mut current = slot.load(Ordering::Relaxed);
    loop {
        let updated = (f64::from_bits(current) + delta).to_bits();
        match slot.compare_exchange_weak(current, updated, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => return,
            Err(actual) => current = actual,
        }
    }
}

/// PUCT selection over a shared node, with in-flight descents included.
///
/// The formula is [`crate::MctsSearcher`]'s, term for term — the duplication is
/// deliberate and pinned by `parallel_with_one_thread_matches_the_serial_searcher`,
/// which fails the moment the two drift apart.
fn select(node: &SharedNode, edges: &Edges, config: &Config) -> usize {
    let sign = if node.mover == 1 { 1.0 } else { -1.0 };
    let visits = node.visits.load(Ordering::Relaxed) + node.vl_visits.load(Ordering::Relaxed);
    let sqrt_n = f64::from(visits + 1).sqrt();
    let virtual_loss = config.virtual_loss;
    let mut best = 0;
    let mut best_score = f64::NEG_INFINITY;
    for a in 0..edges.actions.len() {
        let vl = edges.vl[a].load(Ordering::Relaxed);
        let n = edges.n[a].load(Ordering::Relaxed) + vl;
        let q = if n > 0 {
            let w = f64::from_bits(edges.w[a].load(Ordering::Relaxed));
            (sign * w - virtual_loss * f64::from(vl)) / f64::from(n)
        } else {
            0.0
        };
        let u = config.cpuct * f64::from(edges.prior[a]) * sqrt_n / f64::from(1 + n);
        let score = q + u;
        if score > best_score {
            best_score = score;
            best = a;
        }
    }
    best
}

/// Everything one worker owns: its scratch buffers and its in-flight batch.
struct Worker<'net> {
    config: Config,
    net: Option<&'net PolicyValueNet>,
    scratch: Box<Scratch>,
    net_scratch: Option<crate::net::NetScratch>,
    batch_scratch: Option<crate::net::BatchScratch>,
    eval_params: EvalParams,
    eval_workspace: EvalWorkspace,
    /// `(node, edge)` steps of every descent in the current round.
    path: Vec<(Arc<SharedNode>, usize)>,
    descents: Vec<(usize, usize, usize)>,
    /// Leaves of the current round: the node, and its value once known.
    leaves: Vec<(Arc<SharedNode>, Option<f64>)>,
    encoded: Vec<Encoded>,
    heads: Vec<Heads>,
    collisions: u64,
}

/// What a descent ended in.
enum Stop {
    /// Reached a leaf, recorded at this index in `leaves`.
    Leaf(usize),
    /// Ran into a node another worker had already claimed for expansion.
    Collision,
}

impl<'net> Worker<'net> {
    fn new(config: Config, net: Option<&'net PolicyValueNet>) -> Worker<'net> {
        Worker {
            config,
            net,
            scratch: Scratch::new(),
            net_scratch: net.map(PolicyValueNet::scratch),
            batch_scratch: None,
            eval_params: EvalParams::default(),
            eval_workspace: EvalWorkspace::new(),
            path: Vec::with_capacity(1024),
            descents: Vec::with_capacity(64),
            leaves: Vec::with_capacity(64),
            encoded: Vec::with_capacity(64),
            heads: Vec::with_capacity(64),
            collisions: 0,
        }
    }

    /// Collects up to `target` descents, evaluates their leaves in one batched
    /// forward, and backs the values up. Returns the simulations actually run —
    /// less than `target` when descents collided.
    fn round(&mut self, root: &Arc<SharedNode>, target: u32) -> u32 {
        self.path.clear();
        self.descents.clear();
        self.leaves.clear();
        for _ in 0..target {
            self.descend(root);
        }
        self.evaluate_leaves();
        self.backup();
        self.descents.len() as u32
    }

    fn descend(&mut self, root: &Arc<SharedNode>) {
        let start = self.path.len();
        let mut node = Arc::clone(root);
        let stop = loop {
            if let Some(value) = node.outcome.get() {
                break Stop::Leaf(self.record_leaf(&node, Some(*value)));
            }
            let Some(edges) = node.edges.get() else {
                // Unexpanded. Either this worker's own pending claim from
                // earlier in the batch, a fresh claim, or somebody else's.
                if let Some(index) = self.pending_slot(&node) {
                    break Stop::Leaf(index);
                }
                if node
                    .expanding
                    .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                    .is_err()
                {
                    break Stop::Collision;
                }
                break Stop::Leaf(self.record_leaf(&node, None));
            };

            let a = select(&node, edges, &self.config);
            let child = edges.children[a].get_or_init(|| {
                SharedNode::new(
                    node.state
                        .apply_generated_with(edges.actions[a], &mut self.scratch),
                )
            });
            let child = Arc::clone(child);
            edges.vl[a].fetch_add(1, Ordering::Relaxed);
            node.vl_visits.fetch_add(1, Ordering::Relaxed);
            self.path.push((node, a));
            node = child;
        };
        match stop {
            Stop::Leaf(leaf) => self.descents.push((start, self.path.len(), leaf)),
            Stop::Collision => {
                // Undo this descent's virtual loss and take no credit for it.
                for (parent, edge) in self.path.drain(start..) {
                    let edges = parent
                        .edges
                        .get()
                        .expect("a node on a path was expanded before it was walked");
                    edges.vl[edge].fetch_sub(1, Ordering::Relaxed);
                    parent.vl_visits.fetch_sub(1, Ordering::Relaxed);
                }
                self.collisions += 1;
            }
        }
    }

    fn record_leaf(&mut self, node: &Arc<SharedNode>, value: Option<f64>) -> usize {
        self.leaves.push((Arc::clone(node), value));
        self.leaves.len() - 1
    }

    /// This worker's own pending claim on `node`, if it made one this round.
    fn pending_slot(&self, node: &Arc<SharedNode>) -> Option<usize> {
        self.leaves
            .iter()
            .position(|(other, value)| value.is_none() && Arc::ptr_eq(other, node))
    }

    fn evaluate_leaves(&mut self) {
        let pending: Vec<usize> = self
            .leaves
            .iter()
            .enumerate()
            .filter(|(_, (_, value))| value.is_none())
            .map(|(i, _)| i)
            .collect();
        if pending.is_empty() {
            return;
        }

        // Generate every claimed leaf's actions first: a node with none is
        // terminal after all and drops out of the batch.
        let mut evaluate = Vec::with_capacity(pending.len());
        for &i in &pending {
            let node = Arc::clone(&self.leaves[i].0);
            let actions = node.state.legal_actions_with(&mut self.scratch);
            if actions.is_empty() {
                let value = node.mark_stuck();
                self.leaves[i].1 = Some(value);
                node.expanding.store(false, Ordering::Release);
            } else {
                evaluate.push((i, actions));
            }
        }
        if evaluate.is_empty() {
            return;
        }

        let mut heads = std::mem::take(&mut self.heads);
        heads.clear();
        match (self.net, evaluate.len()) {
            (Some(net), 1) => {
                let encoded = Encoded::from_state(&self.leaves[evaluate[0].0].0.state);
                let scratch = self
                    .net_scratch
                    .as_mut()
                    .expect("a net always brings its scratch");
                heads.push(net.forward(&encoded, scratch));
            }
            (Some(net), _) => {
                let mut encoded = std::mem::take(&mut self.encoded);
                encoded.clear();
                for (i, _) in &evaluate {
                    encoded.push(Encoded::from_state(&self.leaves[*i].0.state));
                }
                let scratch = self
                    .batch_scratch
                    .get_or_insert_with(|| net.batch_scratch());
                net.forward_batch(&encoded, scratch, &mut heads);
                self.encoded = encoded;
            }
            (None, _) => {}
        }

        for (slot, (i, actions)) in evaluate.into_iter().enumerate() {
            let node = Arc::clone(&self.leaves[i].0);
            let value = self.publish_edges(&node, actions, heads.get(slot));
            self.leaves[i].1 = Some(value);
            node.expanding.store(false, Ordering::Release);
        }
        self.heads = heads;
    }

    /// Builds `node`'s [`Edges`] from its net outputs and publishes them,
    /// returning the leaf value in the absolute frame.
    fn publish_edges(
        &mut self,
        node: &SharedNode,
        actions: Vec<Action>,
        heads: Option<&Heads>,
    ) -> f64 {
        let (edges, value) = build_edges(
            node,
            actions,
            heads,
            &self.config,
            self.net,
            &self.eval_params,
            &mut self.eval_workspace,
        );
        // A racing worker may have published first; its edges are equally
        // valid and its `leaf_value_abs` is the same number, so take whichever
        // won and back that up.
        match node.edges.set(edges) {
            Ok(()) => value,
            Err(_) => {
                node.edges
                    .get()
                    .expect("set failed because a value is present")
                    .leaf_value_abs
            }
        }
    }

    fn backup(&mut self) {
        for &(start, end, leaf) in &self.descents {
            let v_abs = self.leaves[leaf]
                .1
                .expect("every leaf is valued before backup");
            for (parent, edge) in &self.path[start..end] {
                let edges = parent
                    .edges
                    .get()
                    .expect("a node on a path was expanded before it was walked");
                parent.visits.fetch_add(1, Ordering::Relaxed);
                edges.n[*edge].fetch_add(1, Ordering::Relaxed);
                add_f64(&edges.w[*edge], v_abs);
                edges.vl[*edge].fetch_sub(1, Ordering::Relaxed);
                parent.vl_visits.fetch_sub(1, Ordering::Relaxed);
            }
        }
    }
}

/// Turns one node's legal actions and net outputs into its [`Edges`] plus the
/// leaf value, mirroring `MctsSearcher::finish_expand_with`.
#[allow(clippy::too_many_arguments)]
fn build_edges(
    node: &SharedNode,
    actions: Vec<Action>,
    heads: Option<&Heads>,
    config: &Config,
    net: Option<&PolicyValueNet>,
    eval_params: &EvalParams,
    eval_workspace: &mut EvalWorkspace,
) -> (Edges, f64) {
    let (prior, value_abs) = match (net, heads) {
        (Some(net), Some(heads)) => {
            let prior = softmax_over(&actions, heads, net.pair_bias(), node.state.cols());
            let value = match (config.value_source, heads.value) {
                (ValueSource::Net, Some(v)) => {
                    let v = f64::from(v);
                    Some(if node.mover == 1 { v } else { -v })
                }
                _ => None,
            };
            (prior, value)
        }
        _ => (vec![1.0 / actions.len() as f32; actions.len()], None),
    };
    let value_abs = value_abs.unwrap_or_else(|| {
        hand_tuned_value_abs(&node.state, config.value_scale, eval_params, eval_workspace)
    });
    (new_edges(actions, prior, value_abs), value_abs)
}

fn new_edges(actions: Vec<Action>, prior: Vec<f32>, leaf_value_abs: f64) -> Edges {
    let k = actions.len();
    Edges {
        actions,
        prior,
        n: (0..k).map(|_| AtomicU32::new(0)).collect(),
        w: (0..k).map(|_| AtomicU64::new(0.0f64.to_bits())).collect(),
        vl: (0..k).map(|_| AtomicU32::new(0)).collect(),
        children: (0..k).map(|_| OnceLock::new()).collect(),
        leaf_value_abs,
    }
}

/// A tree-parallel PUCT search over one position.
///
/// The API mirrors [`crate::MctsSearcher`] method for method, so a caller can
/// swap one for the other behind its own configuration. It is a distinct type
/// rather than a mode of the serial searcher on purpose: the serial searcher's
/// determinism is a contract that the parity fixtures, the arena's node-budget
/// reproducibility and the play path all rely on, and a mode flag that can
/// silently remove it is exactly the sort of thing that gets switched on by
/// accident.
#[derive(Debug)]
pub struct ParallelMcts<'net> {
    config: Config,
    net: Option<&'net PolicyValueNet>,
    root: Arc<SharedNode>,
    threads: usize,
    sims: u64,
    collisions: u64,
    rng: Rng,
    /// Refreshed from the shared tree at the end of every run, so the accessors
    /// can hand out a plain slice.
    root_visits: Vec<u32>,
}

impl<'net> ParallelMcts<'net> {
    /// Builds a searcher and expands the root (plus root noise, if configured).
    ///
    /// # Panics
    ///
    /// Same domain limits as [`crate::MctsSearcher::new`]: two players always,
    /// and a 12x12 board whenever a net is supplied.
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
        let root = SharedNode::new(state);
        let mut searcher = ParallelMcts {
            config,
            net,
            root,
            threads: config.threads.max(1),
            sims: 0,
            collisions: 0,
            rng: Rng::new(config.seed),
            root_visits: Vec::new(),
        };
        searcher.expand_root();
        searcher
    }

    /// Expands the root on this thread, before any worker exists, so Dirichlet
    /// noise can be mixed into the prior *before* the edges are published — an
    /// `Edges` is immutable once it is in the `OnceLock`.
    fn expand_root(&mut self) {
        if self.root.outcome.get().is_some() {
            return;
        }
        let mut worker = Worker::new(self.config, self.net);
        let actions = self.root.state.legal_actions_with(&mut worker.scratch);
        if actions.is_empty() {
            self.root.mark_stuck();
            return;
        }
        let heads = self.net.map(|net| {
            let encoded = Encoded::from_state(&self.root.state);
            let scratch = worker
                .net_scratch
                .as_mut()
                .expect("a net always brings its scratch");
            net.forward(&encoded, scratch)
        });
        let (mut edges, _) = build_edges(
            &self.root,
            actions,
            heads.as_ref(),
            &self.config,
            self.net,
            &worker.eval_params,
            &mut worker.eval_workspace,
        );
        if self.config.root_noise {
            apply_root_noise(&mut edges.prior, &self.config, &mut self.rng);
        }
        self.root
            .edges
            .set(edges)
            .expect("the root is expanded once, single-threaded");
        self.refresh_root();
    }

    /// Runs at least `count` further simulations, spread over the workers.
    ///
    /// Exactly `count` when nothing collides; a collision costs its descent, so
    /// a heavily contended search can fall a few short of the request. The
    /// shortfall is visible as [`ParallelMcts::sims_run`] and the cause as
    /// [`ParallelMcts::collisions`].
    pub fn run_sims(&mut self, count: u32) {
        if self.is_terminal() {
            return;
        }
        let budget = AtomicU32::new(count);
        self.drive(|worker, root, batch| {
            loop {
                let claim = claim_from(&budget, batch);
                if claim == 0 {
                    return;
                }
                let ran = worker.round(root, claim);
                if ran < claim {
                    // Hand the collided descents back so another pass can spend
                    // them rather than silently shrinking the budget.
                    budget.fetch_add(claim - ran, Ordering::Relaxed);
                }
            }
        });
    }

    /// Simulates until `deadline`, always running at least one batch.
    pub fn run_until_deadline(&mut self, deadline: Instant) {
        if self.is_terminal() {
            return;
        }
        self.drive(|worker, root, batch| loop {
            worker.round(root, batch);
            if Instant::now() >= deadline {
                return;
            }
        });
    }

    /// [`ParallelMcts::run_until_deadline`] with a relative budget.
    pub fn run_for(&mut self, budget: Duration) {
        self.run_until_deadline(Instant::now() + budget);
    }

    /// Spawns the workers, runs `body` on each, and folds the results back.
    ///
    /// A panicking worker is **re-raised on this thread**, not swallowed. It
    /// would otherwise surface as a search that silently ran a fraction of its
    /// budget and still handed back a move, which is the shape of bug that
    /// invariant 3 (`ARCHITECTURE.md`) exists about: a searcher must never
    /// quietly return something other than the result its budget bought.
    ///
    /// # Panics
    ///
    /// Propagates any worker panic, after every worker has been joined.
    fn drive<F>(&mut self, body: F)
    where
        F: Fn(&mut Worker<'net>, &Arc<SharedNode>, u32) + Sync,
    {
        let batch = u32::from(self.config.batch_size.max(1));
        let config = self.config;
        let net = self.net;
        let root = &self.root;
        let body = &body;
        let collisions: u64 = std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(self.threads);
            for _ in 0..self.threads {
                handles.push(scope.spawn(move || {
                    let mut worker = Worker::new(config, net);
                    body(&mut worker, root, batch);
                    worker.collisions
                }));
            }
            // Join every worker before re-raising, so a panic cannot leave a
            // sibling still walking the tree this searcher is about to read.
            let mut collisions = 0u64;
            let mut panic = None;
            for handle in handles {
                match handle.join() {
                    Ok(count) => collisions += count,
                    Err(payload) => panic = panic.or(Some(payload)),
                }
            }
            if let Some(payload) = panic {
                std::panic::resume_unwind(payload);
            }
            collisions
        });
        self.collisions += collisions;
        self.refresh_root();
    }

    /// Snapshots the root's edge visits out of the atomics.
    fn refresh_root(&mut self) {
        self.root_visits.clear();
        if let Some(edges) = self.root.edges.get() {
            self.root_visits
                .extend(edges.n.iter().map(|n| n.load(Ordering::Relaxed)));
        }
        self.sims = u64::from(self.root.visits.load(Ordering::Relaxed));
    }

    fn is_terminal(&self) -> bool {
        self.root.outcome.get().is_some()
    }

    /// Simulations run so far.
    pub fn sims_run(&self) -> u64 {
        self.sims
    }

    /// Descents abandoned because another worker had already claimed the node
    /// they reached. Zero by construction at `threads == 1`.
    pub fn collisions(&self) -> u64 {
        self.collisions
    }

    /// Worker threads this search runs on.
    pub fn threads(&self) -> usize {
        self.threads
    }

    /// The root's legal actions, in `virus-core` enumeration order. Empty at a
    /// terminal root.
    pub fn root_actions(&self) -> &[Action] {
        match self.root.edges.get() {
            Some(edges) => &edges.actions,
            None => &[],
        }
    }

    /// Per-root-action visit counts, parallel to
    /// [`ParallelMcts::root_actions`], as of the last completed run.
    pub fn root_visits(&self) -> &[u32] {
        &self.root_visits
    }

    /// Root priors after any Dirichlet noise, parallel to
    /// [`ParallelMcts::root_actions`].
    pub fn root_priors(&self) -> &[f32] {
        match self.root.edges.get() {
            Some(edges) => &edges.prior,
            None => &[],
        }
    }

    /// Root value estimate in the absolute frame.
    pub fn root_value_abs(&self) -> f64 {
        if let Some(value) = self.root.outcome.get() {
            return *value;
        }
        let visits = self.root.visits.load(Ordering::Relaxed);
        if visits == 0 {
            return 0.0;
        }
        let Some(edges) = self.root.edges.get() else {
            return 0.0;
        };
        let sum: f64 = edges
            .w
            .iter()
            .map(|w| f64::from_bits(w.load(Ordering::Relaxed)))
            .sum();
        sum / f64::from(visits)
    }

    /// Most-visited root action, ties broken by enumeration order. `None` at a
    /// terminal or stuck root.
    pub fn best_action(&self) -> Option<Action> {
        let actions = self.root_actions();
        if actions.is_empty() {
            return None;
        }
        let visits = self.root_visits();
        let mut best = 0;
        for a in 1..actions.len() {
            if visits[a] > visits[best] {
                best = a;
            }
        }
        Some(actions[best])
    }

    /// The action to play: [`ParallelMcts::best_action`] in play mode, or a
    /// draw proportional to the root visit counts when
    /// [`Config::visit_sampling`] is on.
    pub fn chosen_action(&mut self) -> Option<Action> {
        if !self.config.visit_sampling {
            return self.best_action();
        }
        if self.root_actions().is_empty() {
            return None;
        }
        let total: u64 = self.root_visits.iter().map(|n| u64::from(*n)).sum();
        if total == 0 {
            return Some(self.root_actions()[0]);
        }
        let target = (self.rng.next_f64() * total as f64) as u64;
        let mut cumulative = 0u64;
        for (a, n) in self.root_visits.iter().enumerate() {
            cumulative += u64::from(*n);
            if target < cumulative {
                return Some(self.root_actions()[a]);
            }
        }
        Some(self.root_actions()[self.root_actions().len() - 1])
    }
}

/// Takes up to `batch` simulations off the shared budget, returning what it
/// actually got.
fn claim_from(budget: &AtomicU32, batch: u32) -> u32 {
    let mut left = budget.load(Ordering::Relaxed);
    loop {
        if left == 0 {
            return 0;
        }
        let take = batch.min(left);
        match budget.compare_exchange_weak(left, left - take, Ordering::Relaxed, Ordering::Relaxed)
        {
            Ok(_) => return take,
            Err(actual) => left = actual,
        }
    }
}

/// Mixes Dirichlet noise into a root prior. Self-play exploration only.
///
/// Draw for draw the same as `MctsSearcher::apply_root_noise`, off the same
/// seeded stream, so the two searchers perturb an identical root identically.
fn apply_root_noise(prior: &mut [f32], config: &Config, rng: &mut Rng) {
    let k = prior.len();
    if k == 0 {
        return;
    }
    let mut g = Vec::with_capacity(k);
    let mut sum = 0.0;
    for _ in 0..k {
        let value = rng.gamma(config.noise_alpha);
        sum += value;
        g.push(value);
    }
    if sum <= 0.0 {
        return;
    }
    let epsilon = config.noise_epsilon;
    for (prior, value) in prior.iter_mut().zip(&g) {
        *prior = ((1.0 - epsilon) * f64::from(*prior) + epsilon * (value / sum)) as f32;
    }
}
