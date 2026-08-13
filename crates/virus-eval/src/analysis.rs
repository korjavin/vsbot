//! Per-seat structural analysis: connectivity, Tarjan cut-loss, Voronoi space
//! race, and the metric counters the weight vector is applied to.
//!
//! Every function here is a literal transcription of its counterpart in
//! `virusgame/backend/search/evaluate.go`. Scan orders, tie-breaks and integer
//! division points are part of the contract; see the crate docs.

use crate::workspace::{Frame, Scratch};
use virus_core::{CellKind, Player, State, MAX_PLAYERS};

/// Raw per-seat counters (Go's `playerMetrics`), before any weight is applied.
///
/// The slice-valued fields of the Go struct (`articulation`, `cutLoss`,
/// `connectedCells`) live in the workspace instead and are addressed by seat,
/// which keeps this `Copy` and keeps the borrow checker out of the way.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Metrics {
    /// Own cells reachable from the base.
    pub(crate) connected: i64,
    /// Own cells severed from the base.
    pub(crate) disconnected: i64,
    /// Own `Normal` cells.
    pub(crate) normal: i64,
    /// Own `Fortified` cells.
    pub(crate) fortified: i64,
    /// Distinct legal targets from the connected component.
    pub(crate) mobility: i64,
    /// Of those, enemy `Normal` cells.
    pub(crate) captures: i64,
    /// Own connected cells adjacent to the base.
    pub(crate) base_exits: i64,
    /// Empty or enemy-`Normal` cells adjacent to the base.
    pub(crate) base_openings: i64,
    /// Own `Fortified` cells adjacent to the base.
    pub(crate) base_anchors: i64,
    /// Enemy `Normal` cells beside a base that is itself under threat.
    pub(crate) base_threat: i64,
    /// Own connected `Normal` cells an active opponent can reach.
    pub(crate) threatened: i64,
    /// Summed `cut_loss` of the threatened cells that are articulation points.
    pub(crate) threatened_loss: i64,
    /// Urgency multiplier; see [`threat_tempo`].
    pub(crate) threat_tempo: i64,
}

/// The mover and its remaining actions — the only two inputs to the tempo
/// terms, bundled so [`analyze`] stays inside clippy's argument budget.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Tempo {
    /// GoBot's `state.CurrentPlayer()`. Distinct from the score player.
    pub(crate) current: Player,
    /// Actions left in the current turn, `0..=3`.
    pub(crate) moves_left: i64,
}

/// In-bounds 8-neighbours of `index`, in Go's scan order: rows ascending,
/// columns ascending within a row, the cell itself omitted.
///
/// Order matters beyond aesthetics — the Tarjan DFS visits children in exactly
/// this order, and a different order yields a different (still valid) set of
/// low-links and therefore different `cut_loss` values.
#[inline]
pub(crate) fn neighbors(rows: usize, cols: usize, index: usize, out: &mut [usize; 8]) -> usize {
    let row = index / cols;
    let col = index % cols;
    let row0 = row.saturating_sub(1);
    let row1 = (row + 1).min(rows - 1);
    let col0 = col.saturating_sub(1);
    let col1 = (col + 1).min(cols - 1);
    let mut count = 0;
    for r in row0..=row1 {
        for c in col0..=col1 {
            let neighbour = r * cols + c;
            if neighbour != index {
                out[count] = neighbour;
                count += 1;
            }
        }
    }
    count
}

/// `value * 1000 / denominator`, guarded (Go's `ratio`).
#[inline]
pub(crate) fn ratio(value: i64, denominator: i64) -> i64 {
    if value <= 0 || denominator <= 0 {
        return 0;
    }
    value * 1000 / denominator
}

/// Whether any 8-neighbour of `index` is in `connected`.
#[inline]
pub(crate) fn adjacent_connected(
    rows: usize,
    cols: usize,
    index: usize,
    connected: &[bool],
) -> bool {
    let mut nearby = [0usize; 8];
    let count = neighbors(rows, cols, index, &mut nearby);
    nearby[..count].iter().any(|&n| connected[n])
}

/// Flood-fills `player`'s base-connected component into `seen`.
///
/// Rooted at the fixed corner base and expands to any cell the player owns,
/// regardless of kind — `Fortified` cells conduct connectivity just like
/// `Normal` ones. A missing or captured base yields an empty component.
pub(crate) fn connected_into(state: &State, player: Player, queue: &mut [u32], seen: &mut [bool]) {
    seen.fill(false);
    let rows = state.rows();
    let cols = state.cols();
    let base_index = state.index(state.base(player));
    let cell = state.cell_at(base_index);
    if cell.owner() != player || cell.kind() != CellKind::Base {
        return;
    }
    seen[base_index] = true;
    queue[0] = base_index as u32;
    let (mut head, mut tail) = (0usize, 1usize);
    let mut nearby = [0usize; 8];
    while head < tail {
        let index = queue[head] as usize;
        head += 1;
        let count = neighbors(rows, cols, index, &mut nearby);
        for &neighbour in &nearby[..count] {
            if !seen[neighbour] && state.cell_at(neighbour).owner() == player {
                seen[neighbour] = true;
                queue[tail] = neighbour as u32;
                tail += 1;
            }
        }
    }
}

/// Multi-source BFS over empty cells, seeded from every active seat's connected
/// territory; returns how many open cells each seat reaches *strictly* first.
///
/// This is the Tron space-partition counter to strangulation: many plies ahead
/// it sees which open region an opponent is walling off, which immediate
/// mobility cannot.
///
/// Deliberately **side-to-move independent**: both sides' first-reach counts
/// come from the same shared BFS on the same position, so the ±40-50 cell phase
/// swing of a raw own-minus-opponent differential cancels in same-ply sibling
/// comparisons and no per-side tempo baseline is needed (vs-ai2.34 post-mortem).
///
/// Plain cell distance, not `ceil(dist/3)` turn distance: the nearest owner is
/// identical under that monotone rescale — only same-turn ties differ — so this
/// captures the signal at lower cost.
///
/// One Go quirk is reproduced deliberately: the popped cell's owner is read at
/// *pop* time, so a cell that became contested (`-2`) after being enqueued
/// propagates contested-ness to its descendants. Changing that changes scores.
pub(crate) fn space_race(
    state: &State,
    connected: &[Vec<bool>; MAX_PLAYERS],
    scratch: &mut Scratch,
    size: usize,
) -> [i64; MAX_PLAYERS] {
    let rows = state.rows();
    let cols = state.cols();
    let dist = &mut scratch.space_dist[..size];
    let owner = &mut scratch.space_owner[..size];
    let queue = &mut scratch.queue[..size];
    dist.fill(-1);
    owner.fill(-1);

    let mut tail = 0usize;
    for (seat, mask) in connected.iter().enumerate() {
        for (index, &live) in mask[..size].iter().enumerate() {
            if live {
                dist[index] = 0;
                owner[index] = seat as i8;
                queue[tail] = index as u32;
                tail += 1;
            }
        }
    }

    let mut nearby = [0usize; 8];
    let mut head = 0usize;
    while head < tail {
        let index = queue[head] as usize;
        head += 1;
        let (d, o) = (dist[index], owner[index]);
        let count = neighbors(rows, cols, index, &mut nearby);
        for &neighbour in &nearby[..count] {
            if state.cell_at(neighbour).kind() != CellKind::Empty {
                continue;
            }
            if dist[neighbour] == -1 {
                dist[neighbour] = d + 1;
                owner[neighbour] = o;
                queue[tail] = neighbour as u32;
                tail += 1;
            } else if dist[neighbour] == d + 1 && owner[neighbour] != o && owner[neighbour] != -2 {
                // Reached at equal distance by another source: nobody owns it.
                owner[neighbour] = -2;
            }
        }
    }

    let mut counts = [0i64; MAX_PLAYERS];
    for index in 0..size {
        if state.cell_at(index).kind() == CellKind::Empty && owner[index] >= 0 {
            counts[owner[index] as usize] += 1;
        }
    }
    counts
}

/// Whether any active opponent's connected component touches `index`.
fn threatened_by_connected(
    state: &State,
    index: usize,
    player: Player,
    connected: &[Vec<bool>; MAX_PLAYERS],
) -> bool {
    let mut nearby = [0usize; 8];
    let count = neighbors(state.rows(), state.cols(), index, &mut nearby);
    for (seat, mask) in connected.iter().enumerate() {
        let opponent = seat as Player + 1;
        if opponent == player || !state.active(opponent) {
            continue;
        }
        if nearby[..count].iter().any(|&n| mask[n]) {
            return true;
        }
    }
    false
}

/// Urgency multiplier for the threat terms.
///
/// An unresolved attack grows more urgent as the defender spends its turn, and
/// stays fully urgent while an opponent still has actions in hand.
fn threat_tempo(player: Player, tempo: Tempo) -> i64 {
    if tempo.current == player {
        (4 - tempo.moves_left).max(1)
    } else {
        tempo.moves_left.max(1)
    }
}

/// Tarjan articulation points of `player`'s connected component, with the
/// subtree size each cut would sever.
///
/// `cut_loss[i]` is what capturing cell `i` costs its owner: every downstream
/// component separated from the base, plus the cell itself. Non-`Normal` cells
/// are filtered out afterwards — a `Base` or `Fortified` articulation point is
/// invulnerable, so its cut is not a real threat.
fn articulation_into(
    state: &State,
    player: Player,
    connected: &[bool],
    scratch: &mut Scratch,
    result: &mut [bool],
    cut_loss: &mut [u16],
) {
    let size = connected.len();
    let Scratch {
        discovery,
        low,
        parent,
        subtree,
        stack,
        ..
    } = scratch;
    let mut tarjan = Tarjan {
        rows: state.rows(),
        cols: state.cols(),
        connected,
        discovery: &mut discovery[..size],
        low: &mut low[..size],
        parent: &mut parent[..size],
        subtree: &mut subtree[..size],
        stack,
        result,
        cut_loss,
        time: 0,
    };

    // Root the first DFS at the base so the tree matches Go's, then sweep for
    // any component the base cannot reach (there is none in practice, but the
    // Go code sweeps and a silent divergence here would be very hard to find).
    let base_index = state.index(state.base(player));
    if base_index < size && connected[base_index] {
        tarjan.visit(base_index);
    }
    for (index, &live) in connected.iter().enumerate() {
        if live && tarjan.discovery[index] == 0 {
            tarjan.visit(index);
        }
    }

    for index in 0..size {
        if !result[index] {
            continue;
        }
        let cell = state.cell_at(index);
        if cell.kind() != CellKind::Normal || cell.owner() != player {
            result[index] = false;
            cut_loss[index] = 0;
        } else {
            // Capturing the cut cell loses the cell itself as well as every
            // downstream component separated from the base.
            cut_loss[index] += 1;
        }
    }
}

/// The recursive Go `visit` closure, unrolled onto an explicit stack.
///
/// Bundling the eight buffers into one struct is not decoration: the traversal
/// needs all of them live at once, and the alternative is a twelve-argument
/// function.
struct Tarjan<'a> {
    rows: usize,
    cols: usize,
    connected: &'a [bool],
    discovery: &'a mut [u16],
    low: &'a mut [u16],
    parent: &'a mut [i32],
    subtree: &'a mut [u16],
    stack: &'a mut Vec<Frame>,
    result: &'a mut [bool],
    cut_loss: &'a mut [u16],
    time: u16,
}

impl Tarjan<'_> {
    /// One DFS tree rooted at `root`.
    ///
    /// A child's post-visit fold — subtree sum, low-link propagation, cut test —
    /// runs when that child pops, which is exactly where the recursive version
    /// runs it on return. The child counter is read from the parent frame at
    /// that moment, so the root's `children > 1` test sees the same value Go's
    /// does.
    fn visit(&mut self, root: usize) {
        self.time += 1;
        self.discovery[root] = self.time;
        self.low[root] = self.time;
        self.subtree[root] = 1;
        self.stack.clear();
        self.stack.push(Frame {
            index: root as u32,
            cursor: 0,
            children: 0,
        });
        while let Some(top) = self.stack.last().copied() {
            let index = top.index as usize;
            let row = index / self.cols;
            let col = index % self.cols;
            let row0 = row.saturating_sub(1);
            let row1 = (row + 1).min(self.rows - 1);
            let col0 = col.saturating_sub(1);
            let col1 = (col + 1).min(self.cols - 1);
            let width = col1 - col0 + 1;
            // The scan walks the whole 3x3 rectangle and skips the cell itself,
            // which reproduces Go's `neighbors` order exactly.
            let total = ((row1 - row0 + 1) * width) as u8;
            if top.cursor >= total {
                self.stack.pop();
                if let Some(parent_frame) = self.stack.last().copied() {
                    let p = parent_frame.index as usize;
                    self.subtree[p] += self.subtree[index];
                    if self.low[index] < self.low[p] {
                        self.low[p] = self.low[index];
                    }
                    if (self.parent[p] == -1 && parent_frame.children > 1)
                        || (self.parent[p] != -1 && self.low[index] >= self.discovery[p])
                    {
                        self.result[p] = true;
                        self.cut_loss[p] += self.subtree[index];
                    }
                }
                continue;
            }
            let step = top.cursor as usize;
            let last = self.stack.len() - 1;
            self.stack[last].cursor += 1;
            let neighbour = (row0 + step / width) * self.cols + (col0 + step % width);
            if neighbour == index || !self.connected[neighbour] {
                continue;
            }
            if self.discovery[neighbour] == 0 {
                self.stack[last].children += 1;
                self.parent[neighbour] = index as i32;
                self.time += 1;
                self.discovery[neighbour] = self.time;
                self.low[neighbour] = self.time;
                self.subtree[neighbour] = 1;
                self.stack.push(Frame {
                    index: neighbour as u32,
                    cursor: 0,
                    children: 0,
                });
            } else if self.parent[index] != neighbour as i32
                && self.discovery[neighbour] < self.low[index]
            {
                self.low[index] = self.discovery[neighbour];
            }
        }
    }
}

/// Counts every structural metric for one seat.
///
/// Mirrors Go's `analyzeWithConnectivity`: one board sweep for material,
/// connectivity, threats and mobility, then a base-neighbourhood pass for the
/// exit/opening/anchor/threat counters.
pub(crate) fn analyze(
    state: &State,
    player: Player,
    connected: &[Vec<bool>; MAX_PLAYERS],
    articulation: &mut [bool],
    cut_loss: &mut [u16],
    scratch: &mut Scratch,
    tempo: Tempo,
) -> Metrics {
    let rows = state.rows();
    let cols = state.cols();
    let size = rows * cols;
    let own = &connected[player as usize - 1][..size];

    scratch.targets[..size].fill(false);
    scratch.discovery[..size].fill(0);
    scratch.low[..size].fill(0);
    scratch.subtree[..size].fill(0);
    scratch.parent[..size].fill(-1);
    articulation.fill(false);
    cut_loss.fill(0);

    articulation_into(state, player, own, scratch, articulation, cut_loss);

    let mut m = Metrics::default();
    let mut nearby = [0usize; 8];
    for index in 0..size {
        let cell = state.cell_at(index);
        if cell.owner() == player {
            match cell.kind() {
                CellKind::Normal => m.normal += 1,
                CellKind::Fortified => m.fortified += 1,
                _ => {}
            }
            if own[index] {
                m.connected += 1;
            } else {
                m.disconnected += 1;
            }
        }
        if own[index]
            && cell.kind() == CellKind::Normal
            && threatened_by_connected(state, index, player, connected)
        {
            m.threatened += 1;
            if articulation[index] {
                m.threatened_loss += cut_loss[index] as i64;
            }
        }
        if !own[index] {
            continue;
        }
        let count = neighbors(rows, cols, index, &mut nearby);
        for &target_index in &nearby[..count] {
            let target = state.cell_at(target_index);
            if !scratch.targets[target_index]
                && (target.kind() == CellKind::Empty
                    || (target.kind() == CellKind::Normal && target.owner() != player))
            {
                scratch.targets[target_index] = true;
                m.mobility += 1;
                if target.kind() == CellKind::Normal {
                    m.captures += 1;
                }
            }
        }
    }

    let base_index = state.index(state.base(player));
    let count = neighbors(rows, cols, base_index, &mut nearby);
    for &index in &nearby[..count] {
        let cell = state.cell_at(index);
        if cell.owner() == player && own[index] {
            m.base_exits += 1;
            if cell.kind() == CellKind::Fortified {
                m.base_anchors += 1;
            }
        } else if cell.kind() == CellKind::Empty {
            m.base_openings += 1;
        } else if cell.kind() == CellKind::Normal && cell.owner() != player {
            // An enemy normal is a legal capture from the base, but is a
            // contested opening rather than owned escape structure.
            m.base_openings += 1;
            if threatened_by_connected(state, base_index, player, connected) {
                m.base_threat += 1;
            }
        }
    }
    m.threat_tempo = threat_tempo(player, tempo);
    m
}
