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
//! The `tests` module at the bottom of this file pins the invariant on a
//! hand-built tree whose movers do not alternate — the case a per-edge negation
//! gets wrong — and on the mirrored-position identity for the leaf flip.
//!
//! # Randomness
//!
//! Play mode draws nothing: `run_sims` and `run_until_deadline` are pure
//! functions of the position, the config and the net. Dirichlet root noise and
//! temperature sampling are self-play only, and both run off the seeded
//! [`Rng`], so even those are reproducible.

use std::time::{Duration, Instant};

use virus_core::{Action, Player, Scratch, State};
use virus_eval::{evaluate, EvalParams, EvalWorkspace};

use crate::net::{Encoded, NetScratch, PolicyValueNet, BOARD};
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
    actions: Vec<Action>,
    prior: Vec<f32>,
    children: Vec<u32>,
    n: Vec<u32>,
    /// Absolute-frame value sums — positive is good for player 1.
    w: Vec<f64>,
    /// Sum of the edge visits below this node.
    visits: u32,
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
            actions: Vec::new(),
            prior: Vec::new(),
            children: Vec::new(),
            n: Vec::new(),
            w: Vec::new(),
            visits: 0,
        }
    }
}

/// Terminal value in the absolute frame, from the single labelling rule
/// (including the territory tiebreak).
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
fn hand_tuned_value_abs(
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
    /// Reusable path buffers, so a simulation allocates nothing.
    path_nodes: Vec<u32>,
    path_edges: Vec<u32>,
    scratch: Box<Scratch>,
    net_scratch: Option<NetScratch>,
    eval_params: EvalParams,
    eval_workspace: EvalWorkspace,
}

impl<'net> MctsSearcher<'net> {
    /// Builds a searcher and expands the root (plus root noise, if configured).
    ///
    /// # Panics
    /// Panics when a net is supplied for a position it cannot encode: the
    /// artifacts are 12x12 two-player nets, and [`Encoded::from_state`] has no
    /// representation for a wider board or a third seat. Checking here rather
    /// than at the first expansion keeps the "a shape mismatch fails before the
    /// search, never during it" rule that [`PolicyValueNet::load`] starts —
    /// otherwise a four-player game would search happily on priors derived from
    /// an encoding that silently collapsed three opponents into one.
    ///
    /// The hand-tuned value source has no such limit: with `net` set to `None`
    /// the searcher runs on any board `virus-core` accepts.
    pub fn new(state: State, config: Config, net: Option<&'net PolicyValueNet>) -> Self {
        assert!(
            net.is_none()
                || (state.rows() == BOARD && state.cols() == BOARD && state.players() == 2),
            "policy net is {BOARD}x{BOARD} two-player only, got {}x{} with {} players",
            state.rows(),
            state.cols(),
            state.players()
        );
        let net_scratch = net.map(|net| net.scratch());
        let mut searcher = MctsSearcher {
            config,
            net,
            nodes: vec![Node::new(state)],
            rng: Rng::new(config.seed),
            sims: 0,
            path_nodes: Vec::with_capacity(64),
            path_edges: Vec::with_capacity(64),
            scratch: Scratch::new(),
            net_scratch,
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
    /// the same tree and returns the same move.
    pub fn run_sims(&mut self, count: u32) {
        if self.nodes[0].terminal {
            return;
        }
        for _ in 0..count {
            self.simulate_once();
        }
    }

    /// Simulates until `deadline`, always running at least one simulation.
    pub fn run_until_deadline(&mut self, deadline: Instant) {
        if self.nodes[0].terminal {
            return;
        }
        loop {
            self.simulate_once();
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

    fn simulate_once(&mut self) {
        self.path_nodes.clear();
        self.path_edges.clear();
        let mut id = 0u32;
        let v_abs = loop {
            let node = &self.nodes[id as usize];
            if node.terminal {
                break node.terminal_value_abs;
            }
            if !node.expanded {
                break self.expand(id as usize);
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
            self.path_nodes.push(id);
            self.path_edges.push(a as u32);
            id = child;
        };
        self.backup(v_abs);
        self.sims += 1;
    }

    /// Credits every edge on the current path with the leaf value.
    ///
    /// `v_abs` is added **as is** at every level. There is no negation here and
    /// there must never be one: the value is already in the absolute frame, and
    /// the mover is reapplied at selection instead. On the 53% of edges that do
    /// not flip the mover, a negamax-style flip here would invert the child's
    /// meaning relative to its parent.
    fn backup(&mut self, v_abs: f64) {
        for (parent, edge) in self.path_nodes.iter().zip(&self.path_edges) {
            let node = &mut self.nodes[*parent as usize];
            let edge = *edge as usize;
            node.visits += 1;
            node.n[edge] += 1;
            node.w[edge] += v_abs;
        }
    }

    /// Expands `id` and returns the leaf value in the absolute frame.
    ///
    /// Expansion and leaf evaluation are fused on purpose: with a net both come
    /// out of **one** trunk pass. The Java original calls `priors` and then
    /// `valueMover`, paying for the trunk twice per expanded node.
    fn expand(&mut self, id: usize) -> f64 {
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
            return node.terminal_value_abs;
        }

        let (prior, value_abs) = {
            let state = &self.nodes[id].state;
            let mover = state.current_player();
            match (self.net, self.net_scratch.as_mut()) {
                (Some(net), Some(net_scratch)) => {
                    let heads = net.forward(&Encoded::from_state(state), net_scratch);
                    let prior = softmax_over(&actions, &heads, net.pair_bias(), state.cols());
                    let value = match (self.config.value_source, heads.value) {
                        (ValueSource::Net, Some(v)) => {
                            let v = f64::from(v);
                            Some(if mover == 1 { v } else { -v })
                        }
                        _ => None,
                    };
                    (prior, value)
                }
                _ => (vec![1.0 / actions.len() as f32; actions.len()], None),
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
        node.children = vec![NO_NODE; actions.len()];
        node.n = vec![0; actions.len()];
        node.w = vec![0.0; actions.len()];
        node.prior = prior;
        node.actions = actions;
        node.expanded = true;
        value_abs
    }

    /// PUCT selection: `argmax_a sign(node) * Q_abs(a) + cpuct * P(a) *
    /// sqrt(N + 1) / (1 + n(a))`.
    ///
    /// The sign is the *only* place the mover enters. Converting the
    /// absolute-frame `Q` here, instead of negating on the way up, is what
    /// survives the 53% of edges that keep the mover.
    fn select(&self, id: usize) -> usize {
        let node = &self.nodes[id];
        let sign = if node.mover == 1 { 1.0 } else { -1.0 };
        let sqrt_n = f64::from(node.visits + 1).sqrt();
        let mut best = 0;
        let mut best_score = f64::NEG_INFINITY;
        for a in 0..node.actions.len() {
            let q = if node.n[a] > 0 {
                sign * node.w[a] / f64::from(node.n[a])
            } else {
                0.0
            };
            let u =
                self.config.cpuct * f64::from(node.prior[a]) * sqrt_n / f64::from(1 + node.n[a]);
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
fn softmax_over(
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
        node
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
        searcher.path_nodes = vec![0, 1];
        searcher.path_edges = vec![0, 1];
        searcher.backup(v_abs);

        assert_eq!(searcher.nodes[0].w[0], v_abs, "root edge keeps the sign");
        assert_eq!(
            searcher.nodes[1].w[1], v_abs,
            "the mover-preserving edge is credited with the SAME value, not its negation"
        );
        assert_eq!(searcher.nodes[0].n[0], 1);
        assert_eq!(searcher.nodes[1].n[1], 1);
        assert_eq!(searcher.nodes[0].visits, 1);
        assert_eq!(searcher.nodes[1].visits, 1);
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
        searcher.path_nodes = vec![0];
        searcher.path_edges = vec![0];
        searcher.backup(1.0);
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
