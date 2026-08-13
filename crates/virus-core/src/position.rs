//! The search-facing move enumerator.
//!
//! Port of `virusgame/backend/game/position.go`, cross-checked against
//! `nnue-trainer/.../search/gobot/GoPosition.java`. [`State`] remains the sole
//! rules implementation; this layer only caches connectivity and curates the
//! neutral-placement branch set.
//!
//! **Move order here IS the search move order.** Equal-scoring siblings are
//! resolved "first wins", so any reordering silently changes the move the
//! engine plays and breaks parity with the Go/Java oracles. See the
//! `ENUMERATION ORDER` block in [`crate::state`] for the canonical order.

use crate::action::{Action, Pos};
use crate::cell::{CellKind, Player};
use crate::scratch::{with_thread_scratch, ArtScratch, Scratch};
use crate::state::State;

/// Branch-count ceiling at or below which every neutral pair is enumerated
/// exactly. Above it the Tarjan-based curation runs instead.
///
/// The compute site ([`Position::new`]) and the consume site
/// ([`Position::for_each_search_action`]) must agree, so the decision lives in
/// one predicate.
pub const EXACT_BRANCH_LIMIT: usize = 32;

/// Hard cap on curated neutral pairs.
pub const MAX_STRATEGIC_PAIRS: usize = 48;

/// Cap on the defensive-cell shortlist the curation pairs up.
const MAX_DEFENSIVE: usize = 12;

/// Number of "robust filler" partners kept for defensive cells.
const FILLER_LIMIT: usize = 2;

/// Whether the curated pair set replaces exact enumeration.
pub fn uses_strategic_pairs(moves: usize, owned: usize) -> bool {
    moves + owned * owned.saturating_sub(1) / 2 > EXACT_BRANCH_LIMIT
}

/// An allocation-conscious view of a [`State`] with cached connectivity and
/// frontier.
#[derive(Clone, Debug)]
pub struct Position {
    state: State,
    /// Cached move frontier; `None` when the position was built by
    /// [`Position::apply_search`] and has not been analysed.
    moves: Option<Vec<Pos>>,
    /// Cached owned-`Normal` list; only populated when neutrals are available.
    owned: Option<Vec<Pos>>,
    /// Curated pairs; only populated above [`EXACT_BRANCH_LIMIT`].
    search_pairs: Option<Vec<[Pos; 2]>>,
    analyzed: bool,
}

impl Position {
    /// Analyses a state: one flood-fill shared by the frontier and the
    /// neutral-pair curation.
    pub fn new(state: State) -> Position {
        with_thread_scratch(|scratch| Position::new_with(state, scratch))
    }

    /// [`Position::new`] using caller-supplied scratch space.
    pub fn new_with(state: State, scratch: &mut Scratch) -> Position {
        if !state.can_act() {
            return Position {
                state,
                moves: Some(Vec::new()),
                owned: None,
                search_pairs: None,
                analyzed: true,
            };
        }
        let player = state.current_player();
        let mut moves = Vec::new();
        {
            let Scratch { bfs, connected, .. } = &mut *scratch;
            state.connected_mask(player, connected, bfs);
            state.frontier_from(player, connected, bfs, &mut moves);
        }
        let mut owned = None;
        let mut search_pairs = None;
        if state.can_place_neutrals() {
            let list = state.owned_normals(player);
            // Below the threshold the pairs are enumerated exactly at consume
            // time, so the Tarjan analysis would be thrown away — skip it.
            if uses_strategic_pairs(moves.len(), list.len()) {
                search_pairs = Some(strategic_neutral_pairs(&state, &list, scratch));
            }
            owned = Some(list);
        }
        Position {
            state,
            moves: Some(moves),
            owned,
            search_pairs,
            analyzed: true,
        }
    }

    /// The underlying state.
    pub fn state(&self) -> &State {
        &self.state
    }

    /// Consumes the position, yielding its state.
    pub fn into_state(self) -> State {
        self.state
    }

    fn move_list(&self) -> Vec<Pos> {
        match &self.moves {
            Some(moves) => moves.clone(),
            None => self.state.move_targets(self.state.current_player()),
        }
    }

    fn owned_normals(&self) -> Vec<Pos> {
        match &self.owned {
            Some(owned) => owned.clone(),
            None => self.state.owned_normals(self.state.current_player()),
        }
    }

    /// Exactly [`State::legal_actions`], reusing the cached frontier.
    pub fn legal_actions(&self) -> Vec<Action> {
        let mut actions = Vec::new();
        self.for_each_legal_action(|action| {
            actions.push(action);
            true
        });
        actions
    }

    /// Enumerates the authoritative action set. Returning `false` stops
    /// enumeration.
    pub fn for_each_legal_action(&self, mut yield_action: impl FnMut(Action) -> bool) {
        if !self.state.can_act() {
            return;
        }
        for target in self.move_list() {
            if !yield_action(Action::Move { target }) {
                return;
            }
        }
        if !self.state.can_place_neutrals() {
            return;
        }
        let owned = self.owned_normals();
        for_each_neutral_pair(&owned, yield_action);
    }

    /// Enumerates the *search* action set: every move in board order, then
    /// either every neutral pair (small positions, kept exact so tie-breaking
    /// stays identical to the authoritative order) or the bounded curated set.
    ///
    /// Returning `false` stops enumeration.
    pub fn for_each_search_action(&self, mut yield_action: impl FnMut(Action) -> bool) {
        if !self.state.can_act() {
            return;
        }
        let moves = self.move_list();
        for target in moves.iter().copied() {
            if !yield_action(Action::Move { target }) {
                return;
            }
        }
        if !self.state.can_place_neutrals() {
            return;
        }
        let owned = self.owned_normals();
        if !uses_strategic_pairs(moves.len(), owned.len()) {
            for_each_neutral_pair(&owned, yield_action);
            return;
        }
        match &self.search_pairs {
            Some(pairs) => {
                for pair in pairs {
                    if !yield_action(Action::PlaceNeutrals { cells: *pair }) {
                        return;
                    }
                }
            }
            None => {
                let pairs = with_thread_scratch(|scratch| {
                    let Scratch { bfs, connected, .. } = &mut *scratch;
                    self.state
                        .connected_mask(self.state.current_player(), connected, bfs);
                    strategic_neutral_pairs(&self.state, &owned, scratch)
                });
                for pair in pairs {
                    if !yield_action(Action::PlaceNeutrals { cells: pair }) {
                        return;
                    }
                }
            }
        }
    }

    /// The search action set, materialised in enumeration order.
    pub fn search_actions(&self) -> Vec<Action> {
        let mut actions = Vec::new();
        self.for_each_search_action(|action| {
            actions.push(action);
            true
        });
        actions
    }

    /// Legality-checked successor.
    pub fn apply(&self, action: Action) -> Result<Position, crate::RuleError> {
        Ok(Position::new(self.state.apply(action)?))
    }

    /// Successor for an action already emitted by
    /// [`Position::for_each_search_action`], without repeating legality work.
    ///
    /// The returned position is unanalysed: its caches are filled lazily, which
    /// matters because a searcher discards most children after one evaluation.
    ///
    /// # Panics
    /// Panics on out-of-bounds coordinates. Never pass arbitrary input —
    /// [`State::apply`] is the boundary oracle.
    pub fn apply_search(&self, action: Action) -> Position {
        Position {
            state: self.state.apply_generated(action),
            moves: None,
            owned: None,
            search_pairs: None,
            analyzed: false,
        }
    }

    /// Whether this position's caches were populated up front.
    pub fn analyzed(&self) -> bool {
        self.analyzed
    }
}

fn for_each_neutral_pair(cells: &[Pos], mut yield_action: impl FnMut(Action) -> bool) {
    for i in 0..cells.len() {
        for j in (i + 1)..cells.len() {
            if !yield_action(Action::neutrals(cells[i], cells[j])) {
                return;
            }
        }
    }
}

/// A deliberately bounded defensive branch set for neutral placement.
///
/// Neutralising one important cell with every possible filler is strategically
/// redundant and catastrophically expensive (`C(owned, 2)` can be thousands of
/// branches). Instead: shortlist at most [`MAX_DEFENSIVE`] highest-priority
/// defensive cells, pair them with robust fillers, and keep general (including
/// non-adjacent) two-vertex separators involving those cells. Pair classes get
/// reserved representation before the remaining capacity is distributed, so a
/// large defensive class cannot starve fillers or separators.
///
/// ARCHITECTURE.md invariant 6: because the search only ever sees this subset,
/// a transposition-table `PlaceNeutrals` move must be re-validated against it
/// before being trusted.
///
/// Requires `scratch.connected` to already hold the mover's component.
fn strategic_neutral_pairs(state: &State, owned: &[Pos], scratch: &mut Scratch) -> Vec<[Pos; 2]> {
    let player = state.current_player();
    if !state.can_place_neutrals() {
        return Vec::new();
    }
    let count = state.cell_count();

    // Articulation cells of the mover's component, in the full graph.
    {
        let Scratch {
            connected,
            art,
            cuts,
            ..
        } = &mut *scratch;
        articulation_cells(state, connected, None, art);
        cuts[..count].copy_from_slice(&art.cuts[..count]);
    }

    // Own `Normal` cells an active opponent can capture right now. This needs a
    // second connectivity mask so the mover's is not clobbered.
    {
        let Scratch {
            bfs,
            alt_connected,
            threatened,
            ..
        } = &mut *scratch;
        threatened[..count].fill(false);
        let mut targets = Vec::new();
        for opponent in 1..=state.players() as Player {
            if opponent == player || !state.active(opponent) {
                continue;
            }
            state.connected_mask(opponent, alt_connected, bfs);
            state.frontier_from(opponent, alt_connected, bfs, &mut targets);
            for target in &targets {
                let index = state.index(*target);
                let cell = state.cell_at(index);
                if cell.owner() == player && cell.kind() == CellKind::Normal {
                    threatened[index] = true;
                }
            }
        }
    }

    let base = state.base(player);
    let mut base_defense = Vec::new();
    let mut threat_defense = Vec::new();
    let mut cut_defense = Vec::new();
    for pos in owned {
        let index = state.index(*pos);
        if adjacent(*pos, base) {
            base_defense.push(*pos);
        }
        if scratch.threatened[index] {
            threat_defense.push(*pos);
        }
        if scratch.cuts[index] {
            cut_defense.push(*pos);
        }
    }

    // Seed every available defensive class, then fill in survival priority.
    // Stable board order breaks ties deterministically.
    let mut defensive: Vec<Pos> = Vec::with_capacity(MAX_DEFENSIVE);
    let classes = [&base_defense, &threat_defense, &cut_defense];
    for class in classes {
        if let Some(first) = class.first() {
            add_defensive(&mut defensive, *first);
        }
    }
    for class in classes {
        for pos in class.iter() {
            add_defensive(&mut defensive, *pos);
        }
    }

    let fillers = robust_fillers(state, owned, scratch, &defensive, FILLER_LIMIT);

    let normalize = |a: Pos, b: Pos| -> Option<[Pos; 2]> {
        let (ia, ib) = (state.index(a), state.index(b));
        if ia == ib {
            None
        } else if ia < ib {
            Some([a, b])
        } else {
            Some([b, a])
        }
    };

    let mut defensive_filler: Vec<[Pos; 2]> = Vec::new();
    for cell in &defensive {
        for filler in &fillers {
            if let Some(pair) = normalize(*cell, *filler) {
                append_unique(&mut defensive_filler, pair, MAX_STRATEGIC_PAIRS);
            }
        }
    }
    let mut defensive_pairs: Vec<[Pos; 2]> = Vec::new();
    for i in 0..defensive.len() {
        for j in (i + 1)..defensive.len() {
            if let Some(pair) = normalize(defensive[i], defensive[j]) {
                append_unique(&mut defensive_pairs, pair, MAX_STRATEGIC_PAIRS);
            }
        }
    }

    // Tarjan in `G - u` finds every partner `v` of a general two-vertex
    // separator. Scratch is reused and `defensive` is capped, bounding both
    // time and allocation.
    let mut separators: Vec<[Pos; 2]> = Vec::new();
    for u in &defensive {
        let u_index = state.index(*u);
        if scratch.cuts[u_index] {
            continue;
        }
        {
            let Scratch { connected, art, .. } = &mut *scratch;
            articulation_cells(state, connected, Some(u_index), art);
        }
        for v in owned {
            if scratch.art.cuts[state.index(*v)] {
                if let Some(pair) = normalize(*u, *v) {
                    append_unique(&mut separators, pair, MAX_STRATEGIC_PAIRS);
                }
            }
        }
    }

    let mut pairs: Vec<[Pos; 2]> = Vec::with_capacity(MAX_STRATEGIC_PAIRS);
    // Reserve one true separator and one pair per defensive cell before
    // distributing the rest. Fillers are only safe partners *for* a tactical
    // cell; a standalone filler pair would be destructive self-cleanup.
    if let Some(first) = separators.first() {
        append_unique(&mut pairs, *first, MAX_STRATEGIC_PAIRS);
    }
    for i in 0..defensive.len() {
        if !fillers.is_empty() {
            if let Some(pair) = normalize(defensive[i], fillers[i % fillers.len()]) {
                append_unique(&mut pairs, pair, MAX_STRATEGIC_PAIRS);
            }
        } else if defensive.len() > 1 {
            if let Some(pair) = normalize(defensive[i], defensive[(i + 1) % defensive.len()]) {
                append_unique(&mut pairs, pair, MAX_STRATEGIC_PAIRS);
            }
        }
    }
    let distribute = [&separators, &defensive_filler, &defensive_pairs];
    let maximum = distribute
        .iter()
        .map(|class| class.len())
        .max()
        .unwrap_or(0);
    for index in 0..maximum {
        if pairs.len() >= MAX_STRATEGIC_PAIRS {
            break;
        }
        for class in distribute {
            if let Some(pair) = class.get(index) {
                append_unique(&mut pairs, *pair, MAX_STRATEGIC_PAIRS);
            }
        }
    }
    pairs
}

fn add_defensive(defensive: &mut Vec<Pos>, pos: Pos) {
    if defensive.contains(&pos) {
        return;
    }
    if defensive.len() < MAX_DEFENSIVE {
        defensive.push(pos);
    }
}

fn append_unique(pairs: &mut Vec<[Pos; 2]>, pair: [Pos; 2], limit: usize) {
    if pairs.contains(&pair) {
        return;
    }
    if pairs.len() < limit {
        pairs.push(pair);
    }
}

/// Own cells that are safe to sacrifice: not a cut vertex, not under threat,
/// not already shortlisted — preferring the ones furthest from the base.
fn robust_fillers(
    state: &State,
    owned: &[Pos],
    scratch: &Scratch,
    defensive: &[Pos],
    limit: usize,
) -> Vec<Pos> {
    let base = state.base(state.current_player());
    let mut result: Vec<Pos> = Vec::with_capacity(limit);
    while result.len() < limit {
        let mut best = Pos::default();
        let mut best_score = -1i32;
        let mut found = false;
        for pos in owned {
            let index = state.index(*pos);
            if scratch.cuts[index]
                || scratch.threatened[index]
                || defensive.contains(pos)
                || result.contains(pos)
            {
                continue;
            }
            let score = (pos.row - base.row).abs() + (pos.col - base.col).abs();
            if !found || score > best_score {
                best = *pos;
                best_score = score;
                found = true;
            }
        }
        if !found {
            break;
        }
        result.push(best);
    }
    result
}

fn adjacent(a: Pos, b: Pos) -> bool {
    (a.row - b.row).abs() <= 1 && (a.col - b.col).abs() <= 1 && a != b
}

/// Tarjan articulation points of the mover's connected component, optionally
/// with one vertex removed from the graph.
///
/// Iterative rather than recursive: a 50x50 component is 2500 vertices deep in
/// the worst case, and this runs inside the search. The traversal order and the
/// cut conditions are identical to the Go/Java recursive version — children are
/// visited in the same 8-neighbourhood scan order, and the cut tests fire at the
/// same points relative to each child's return.
fn articulation_cells(
    state: &State,
    connected: &[bool],
    excluded: Option<usize>,
    art: &mut ArtScratch,
) {
    let count = state.cell_count();
    art.discovery[..count].fill(0);
    art.low[..count].fill(0);
    art.cuts[..count].fill(false);
    art.parent[..count].fill(-1);

    let base_index = state.index(state.base(state.current_player()));
    if Some(base_index) == excluded || !connected[base_index] {
        return;
    }

    let cols = state.cols();
    let mut time: u32 = 1;
    art.discovery[base_index] = time;
    art.low[base_index] = time;
    // (vertex, cursor into its 3x3 scan, children discovered so far)
    let mut stack: Vec<(usize, u8, u32)> = Vec::with_capacity(64);
    stack.push((base_index, 0, 0));

    while let Some(&(index, cursor, _)) = stack.last() {
        let (row0, row1, col0, col1) = state.neighbourhood(index);
        let width = col1 - col0 + 1;
        let total = ((row1 - row0 + 1) * width) as u8;
        if cursor >= total {
            // Done with this vertex: fold its low-link into its parent and run
            // the parent's cut tests, at exactly the point the recursive
            // version runs them on return.
            stack.pop();
            if let Some(&(parent, _, parent_children)) = stack.last() {
                if art.low[index] < art.low[parent] {
                    art.low[parent] = art.low[index];
                }
                if art.parent[parent] == -1 && parent_children > 1 {
                    art.cuts[parent] = true;
                }
                if art.parent[parent] != -1 && art.low[index] >= art.discovery[parent] {
                    art.cuts[parent] = true;
                }
            }
            continue;
        }
        let top = stack.len() - 1;
        stack[top].1 += 1;
        let step = cursor as usize;
        let neighbour = (row0 + step / width) * cols + (col0 + step % width);
        if neighbour == index || Some(neighbour) == excluded || !connected[neighbour] {
            continue;
        }
        if art.discovery[neighbour] == 0 {
            art.parent[neighbour] = index as i32;
            stack[top].2 += 1;
            time += 1;
            art.discovery[neighbour] = time;
            art.low[neighbour] = time;
            stack.push((neighbour, 0, 0));
        } else if art.parent[index] != neighbour as i32 && art.discovery[neighbour] < art.low[index]
        {
            art.low[index] = art.discovery[neighbour];
        }
    }
}
