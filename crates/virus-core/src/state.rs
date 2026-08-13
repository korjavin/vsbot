//! The authoritative rules engine.
//!
//! A direct port of `virusgame/backend/game/state.go`, cross-checked against
//! `nnue-trainer/.../search/gobot/GoState.java`. Every rule here has a
//! predecessor bug attached to it; see the module comments and
//! ARCHITECTURE.md's "Non-negotiable invariants".

use crate::action::{Action, Pos};
use crate::cell::{Cell, CellKind, Player, MAX_PLAYERS};
use crate::scratch::{with_thread_scratch, BfsScratch, Scratch};
use crate::zobrist;
use std::fmt;

/// Actions available at the start of a turn.
pub const ACTIONS_PER_TURN: u8 = 3;

/// Largest supported board edge (the server plays 12x12; the Go engine's own
/// tests go to 50x50).
pub const MAX_DIM: usize = 50;
/// Largest supported board area.
pub const MAX_CELLS: usize = MAX_DIM * MAX_DIM;

/// Why a transition was refused.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RuleError {
    /// The game is already decided; no further actions exist.
    GameOver,
    /// The action is not legal in this position (or the arguments to a
    /// constructor were out of range).
    InvalidAction,
}

impl fmt::Display for RuleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RuleError::GameOver => f.write_str("game is over"),
            RuleError::InvalidAction => f.write_str("invalid action"),
        }
    }
}

impl std::error::Error for RuleError {}

/// A complete game position: the grid plus the hidden state the search and its
/// transposition key depend on.
///
/// Value semantics, like Go's `game.State`: [`State::apply`] copies before
/// mutating, so a parent state stays safe to retain in a search tree. The board
/// is a flat byte array (144 used bytes at 12x12), so the copy is a single
/// memcpy — cheaper and far simpler than make/unmake, and it keeps every node
/// independently usable by a parallel searcher.
#[derive(Clone, PartialEq, Eq)]
pub struct State {
    /// Row-major packed cells; only `rows * cols` entries are meaningful.
    cells: [u8; MAX_CELLS],
    rows: u8,
    cols: u8,
    players: u8,
    bases: [Pos; MAX_PLAYERS],
    active: [bool; MAX_PLAYERS],
    neutral_used: [bool; MAX_PLAYERS],
    current: Player,
    moves_left: u8,
    winner: Player,
    over: bool,
    /// Incrementally maintained Zobrist key (see [`crate::zobrist`]).
    hash: u64,
}

impl State {
    // ---------------------------------------------------------------- construction

    /// Creates a fresh game. Bases are the fixed corners in seat order:
    /// P1 `(0,0)`, P2 `(rows-1, cols-1)`, P3 `(0, cols-1)`, P4 `(rows-1, 0)`.
    pub fn new(rows: usize, cols: usize, players: usize) -> Result<State, RuleError> {
        if !(2..=MAX_DIM).contains(&rows)
            || !(2..=MAX_DIM).contains(&cols)
            || !(2..=MAX_PLAYERS).contains(&players)
        {
            return Err(RuleError::InvalidAction);
        }
        let mut state = State {
            cells: [Cell::EMPTY.packed(); MAX_CELLS],
            rows: rows as u8,
            cols: cols as u8,
            players: players as u8,
            bases: default_bases(rows, cols),
            active: [false; MAX_PLAYERS],
            neutral_used: [false; MAX_PLAYERS],
            current: 1,
            moves_left: ACTIONS_PER_TURN,
            winner: 0,
            over: false,
            hash: 0,
        };
        for seat in 0..players {
            state.active[seat] = true;
            let base = state.bases[seat];
            let index = state.index(base);
            state.cells[index] = Cell::new(seat as Player + 1, CellKind::Base).packed();
        }
        state.recompute_hash();
        Ok(state)
    }

    /// Builds a mid-game position from a bare grid plus the hidden state, the
    /// way the parity fixtures encode it (and the way Java's
    /// `GoState.fromBoard` does).
    ///
    /// Bases are the default corners and `active[p]` is derived from an intact
    /// base — GoBot flips the active flag only on elimination, which for a
    /// non-terminal snapshot coincides with "base intact".
    ///
    /// `cells` is row-major and must be `rows * cols` long.
    pub fn from_grid(
        rows: usize,
        cols: usize,
        players: usize,
        cells: &[Cell],
        current: Player,
        moves_left: u8,
        neutral_used: &[bool],
    ) -> Result<State, RuleError> {
        let mut state = State::new(rows, cols, players)?;
        if cells.len() != rows * cols
            || moves_left > ACTIONS_PER_TURN
            || current < 1
            || current as usize > players
        {
            return Err(RuleError::InvalidAction);
        }
        for (index, cell) in cells.iter().enumerate() {
            if !cell.is_well_formed(players) {
                return Err(RuleError::InvalidAction);
            }
            state.cells[index] = cell.packed();
        }
        for seat in 0..players {
            let base = state.bases[seat];
            let cell = state.at(base);
            state.active[seat] =
                cell.owner() == seat as Player + 1 && cell.kind() == CellKind::Base;
            state.neutral_used[seat] = neutral_used.get(seat).copied().unwrap_or(false);
        }
        state.current = current;
        state.moves_left = moves_left;
        state.recompute_hash();
        Ok(state)
    }

    // ---------------------------------------------------------------- accessors

    /// Board height.
    pub fn rows(&self) -> usize {
        self.rows as usize
    }
    /// Board width.
    pub fn cols(&self) -> usize {
        self.cols as usize
    }
    /// Number of seats.
    pub fn players(&self) -> usize {
        self.players as usize
    }
    /// Seat to move.
    pub fn current_player(&self) -> Player {
        self.current
    }
    /// Actions remaining in the current turn.
    pub fn moves_left(&self) -> u8 {
        self.moves_left
    }
    /// Whether the game is decided.
    pub fn game_over(&self) -> bool {
        self.over
    }
    /// The winner, or `0` while the game runs / on a draw.
    pub fn winner(&self) -> Player {
        self.winner
    }
    /// The incremental Zobrist key.
    pub fn hash(&self) -> u64 {
        self.hash
    }
    /// Base cell of a seat.
    pub fn base(&self, player: Player) -> Pos {
        self.bases[player as usize - 1]
    }

    /// Whether a seat is still in the game.
    pub fn active(&self, player: Player) -> bool {
        self.valid_player(player) && self.active[player as usize - 1]
    }

    /// Whether the side to move may play an action at all: the game is running,
    /// the mover is active, and the turn still has actions in it.
    ///
    /// The `moves_left > 0` clause is load-bearing, not defensive dressing. No
    /// rules transition can produce a running position with a live mover and
    /// `moves_left == 0` ([`State::mutate`] rotates the turn the moment the
    /// budget hits zero), but an *imported* one can: the server publishes that
    /// exact transient when the `move_made` echo of a turn's last action arrives
    /// before the `turn_change` that rotates the seat. Enumerating actions there
    /// hands the caller moves whose `apply` computes `0 - 1`, and the wrapped
    /// `255` indexes the Zobrist `moves_left` table out of bounds — a panicked
    /// search worker, observed live. Every enumeration entry point gates on this
    /// predicate so the bug class is unreachable at the source.
    #[inline]
    pub fn can_act(&self) -> bool {
        !self.over && self.active(self.current) && self.moves_left > 0
    }

    /// Whether a seat has already spent its once-per-game neutral placement.
    pub fn neutral_used(&self, player: Player) -> bool {
        self.valid_player(player) && self.neutral_used[player as usize - 1]
    }

    /// Cell at a coordinate, or `None` when out of bounds.
    pub fn at_checked(&self, pos: Pos) -> Option<Cell> {
        if self.in_bounds(pos) {
            Some(self.at(pos))
        } else {
            None
        }
    }

    /// Cell at an in-bounds coordinate.
    ///
    /// # Panics
    /// Panics when `pos` is out of bounds.
    pub fn at(&self, pos: Pos) -> Cell {
        self.cell_at(self.index(pos))
    }

    /// Cell at a row-major board index. Hot path: the stored bytes are only
    /// ever written from validated [`Cell`]s, so no re-validation is needed.
    #[inline]
    pub fn cell_at(&self, index: usize) -> Cell {
        Cell::from_packed_unchecked(self.cells[index])
    }

    /// The raw row-major grid, `rows * cols` packed bytes.
    #[inline]
    pub fn grid(&self) -> &[u8] {
        &self.cells[..self.cell_count()]
    }

    /// Number of meaningful cells (`rows * cols`).
    #[inline]
    pub fn cell_count(&self) -> usize {
        self.rows as usize * self.cols as usize
    }

    /// Row-major index of a coordinate.
    #[inline]
    pub fn index(&self, pos: Pos) -> usize {
        pos.row as usize * self.cols as usize + pos.col as usize
    }

    /// Coordinate of a row-major index.
    #[inline]
    pub fn pos_of(&self, index: usize) -> Pos {
        Pos::new(
            (index / self.cols as usize) as i32,
            (index % self.cols as usize) as i32,
        )
    }

    /// Whether a coordinate lies on the board.
    #[inline]
    pub fn in_bounds(&self, pos: Pos) -> bool {
        pos.row >= 0 && pos.row < self.rows as i32 && pos.col >= 0 && pos.col < self.cols as i32
    }

    #[inline]
    fn valid_player(&self, player: Player) -> bool {
        player >= 1 && player <= self.players
    }

    /// How many cells a seat owns (any kind). Used by the territory tiebreak.
    pub fn owned_cells(&self, player: Player) -> usize {
        self.grid()
            .iter()
            .filter(|&&packed| packed >> 3 == player)
            .count()
    }

    // ---------------------------------------------------------------- hashing

    /// Recomputes the Zobrist key from scratch.
    ///
    /// [`State::hash`] is maintained incrementally through every mutation; this
    /// is the independent recomputation to check it against. A divergence means
    /// some mutation path forgot to update the key, which would silently
    /// corrupt the transposition table.
    pub fn recomputed_hash(&self) -> u64 {
        let mut copy = self.clone();
        copy.recompute_hash();
        copy.hash
    }

    /// Recomputes the Zobrist key from scratch. Only constructors need this;
    /// every mutation path maintains the key incrementally.
    fn recompute_hash(&mut self) {
        let table = zobrist::table();
        let mut hash = table.current(self.current) ^ table.moves_left(self.moves_left);
        for seat in 0..self.players() {
            let player = seat as Player + 1;
            if self.active[seat] {
                hash ^= table.active(player);
            }
            if self.neutral_used[seat] {
                hash ^= table.neutral_used(player);
            }
        }
        for index in 0..self.cell_count() {
            hash ^= table.cell(index, self.cell_at(index));
        }
        self.hash = hash;
    }

    /// The FNV-1a position hash used by GoBot's and the Java port's
    /// transposition tables (`search.go: stateHash`).
    ///
    /// Kept byte-for-byte identical to theirs so the search-parity work in the
    /// next bead can key a shared TT the same way. The engine's own TT uses
    /// [`State::hash`] (Zobrist, incremental) instead — this is a parity aid.
    pub fn state_hash(&self) -> u64 {
        const PRIME: u64 = 1_099_511_628_211;
        let mut hash: u64 = 1_469_598_103_934_665_603;
        let mut add = |value: u8| {
            hash ^= value as u64;
            hash = hash.wrapping_mul(PRIME);
        };
        add(self.rows);
        add(self.cols);
        add(self.current);
        add(self.moves_left);
        for player in 1..=MAX_PLAYERS as Player {
            if self.active(player) {
                add(player | 0x10);
            }
            if self.neutral_used(player) {
                add(player | 0x20);
            }
        }
        for index in 0..self.cell_count() {
            add(self.cells[index]);
        }
        hash
    }

    // ---------------------------------------------------------------- mutators

    #[inline]
    fn set_cell(&mut self, index: usize, cell: Cell) {
        let table = zobrist::table();
        self.hash ^= table.cell(index, self.cell_at(index));
        self.cells[index] = cell.packed();
        self.hash ^= table.cell(index, cell);
    }

    #[inline]
    fn set_current(&mut self, player: Player) {
        let table = zobrist::table();
        self.hash ^= table.current(self.current) ^ table.current(player);
        self.current = player;
    }

    #[inline]
    fn set_moves_left(&mut self, moves_left: u8) {
        let table = zobrist::table();
        self.hash ^= table.moves_left(self.moves_left) ^ table.moves_left(moves_left);
        self.moves_left = moves_left;
    }

    #[inline]
    fn set_active(&mut self, player: Player, value: bool) {
        let seat = player as usize - 1;
        if self.active[seat] != value {
            self.hash ^= zobrist::table().active(player);
            self.active[seat] = value;
        }
    }

    #[inline]
    fn set_neutral_used(&mut self, player: Player, value: bool) {
        let seat = player as usize - 1;
        if self.neutral_used[seat] != value {
            self.hash ^= zobrist::table().neutral_used(player);
            self.neutral_used[seat] = value;
        }
    }

    // ---------------------------------------------------------------- movegen
    //
    // ENUMERATION ORDER — DO NOT CHANGE.
    //
    // Search parity with the Go and Java engines depends on identical child
    // ordering (equal-scoring siblings are resolved by "first wins", so a
    // different order silently picks a different move). The order is:
    //
    //   1. `Move` actions, ascending row-major board index of the *target*.
    //      Note this is the order of the frontier bitmap scan, NOT the order
    //      the connectivity BFS discovered cells in.
    //   2. `PlaceNeutrals` actions, only when `moves_left == ACTIONS_PER_TURN`
    //      and the mover has not used its placement: all pairs `(i, j)`,
    //      `i < j`, over the mover's own `Normal` cells listed in ascending
    //      row-major board index.
    //
    // `Position::for_each_search_action` keeps step 1 identical and replaces
    // step 2 with a curated subset above the branch threshold; see
    // `position.rs`.
    //
    // Every entry point first gates on `State::can_act`, so a position the
    // mover cannot legally act in — finished, mover eliminated, or the turn
    // budget already spent — enumerates nothing at all.

    /// Every legal action for the side to move, in the canonical enumeration
    /// order documented above.
    pub fn legal_actions(&self) -> Vec<Action> {
        with_thread_scratch(|scratch| self.legal_actions_with(scratch))
    }

    /// [`State::legal_actions`] using caller-supplied scratch space.
    pub fn legal_actions_with(&self, scratch: &mut Scratch) -> Vec<Action> {
        if !self.can_act() {
            return Vec::new();
        }
        let mut targets = Vec::new();
        self.move_targets_into(self.current, scratch, &mut targets);
        let mut actions: Vec<Action> = targets
            .into_iter()
            .map(|target| Action::Move { target })
            .collect();
        if self.can_place_neutrals() {
            let owned = self.owned_normals(self.current);
            for i in 0..owned.len() {
                for j in (i + 1)..owned.len() {
                    actions.push(Action::neutrals(owned[i], owned[j]));
                }
            }
        }
        actions
    }

    /// Whether the side to move may spend its neutral placement right now.
    #[inline]
    pub fn can_place_neutrals(&self) -> bool {
        self.moves_left == ACTIONS_PER_TURN && !self.neutral_used(self.current)
    }

    /// The mover's own `Normal` cells in ascending board index — the source
    /// list for neutral pairs.
    pub fn owned_normals(&self, player: Player) -> Vec<Pos> {
        let mut cells = Vec::new();
        for index in 0..self.cell_count() {
            let cell = self.cell_at(index);
            if cell.owner() == player && cell.kind() == CellKind::Normal {
                cells.push(self.pos_of(index));
            }
        }
        cells
    }

    /// Legal move targets for `player`, ascending board index.
    pub fn move_targets(&self, player: Player) -> Vec<Pos> {
        with_thread_scratch(|scratch| {
            let mut out = Vec::new();
            self.move_targets_into(player, scratch, &mut out);
            out
        })
    }

    /// Floods `player`'s base-connected component into `scratch.connected`,
    /// then writes its legal frontier into `out` (ascending board index).
    ///
    /// Computing connectivity once and deriving the whole frontier from it is
    /// what makes movegen O(cells) instead of O(cells * BFS).
    pub fn move_targets_into(&self, player: Player, scratch: &mut Scratch, out: &mut Vec<Pos>) {
        let Scratch { bfs, connected, .. } = scratch;
        self.connected_mask(player, connected, bfs);
        self.frontier_from(player, connected, bfs, out);
    }

    /// Derives the legal frontier from an already-computed connectivity mask,
    /// so callers holding that mask (the search [`crate::Position`]) never
    /// repeat the flood-fill.
    pub(crate) fn frontier_from(
        &self,
        player: Player,
        connected: &[bool],
        bfs: &mut BfsScratch,
        out: &mut Vec<Pos>,
    ) {
        let count = self.cell_count();
        let frontier = &mut bfs.frontier;
        frontier[..count].fill(false);
        for (index, &is_connected) in connected[..count].iter().enumerate() {
            if !is_connected {
                continue;
            }
            self.for_each_neighbour_inclusive(index, |neighbour| {
                if self.cell_at(neighbour).is_capturable_by(player) {
                    frontier[neighbour] = true;
                }
            });
        }
        out.clear();
        for (index, &legal) in frontier[..count].iter().enumerate() {
            if legal {
                out.push(self.pos_of(index));
            }
        }
    }

    /// Floods `player`'s base-connected component into `mask`.
    ///
    /// A component is grown from the player's *base* over 8-neighbours of cells
    /// the player owns. A player whose base is gone has an empty component —
    /// which is how base destruction eliminates them.
    pub(crate) fn connected_mask(&self, player: Player, mask: &mut [bool], bfs: &mut BfsScratch) {
        let count = self.cell_count();
        mask[..count].fill(false);
        if !self.valid_player(player) {
            return;
        }
        let base = self.bases[player as usize - 1];
        let base_cell = self.at(base);
        if base_cell.owner() != player || base_cell.kind() != CellKind::Base {
            return;
        }
        let base_index = self.index(base);
        mask[base_index] = true;
        bfs.queue[0] = base_index as u16;
        let (mut head, mut tail) = (0usize, 1usize);
        while head < tail {
            let current = bfs.queue[head] as usize;
            head += 1;
            let (row0, row1, col0, col1) = self.neighbourhood(current);
            for row in row0..=row1 {
                for col in col0..=col1 {
                    let neighbour = row * self.cols as usize + col;
                    if neighbour == current || mask[neighbour] {
                        continue;
                    }
                    if self.cells[neighbour] >> 3 == player {
                        mask[neighbour] = true;
                        bfs.queue[tail] = neighbour as u16;
                        tail += 1;
                    }
                }
            }
        }
    }

    /// Whether `player` has at least one legal move.
    ///
    /// Fused flood-fill + frontier probe with an early exit, so the common case
    /// (a player with plenty of room) stops after a handful of cells. Touches
    /// only [`BfsScratch`], leaving the caller's connectivity mask intact.
    pub(crate) fn has_move_bfs(&self, player: Player, bfs: &mut BfsScratch) -> bool {
        let count = self.cell_count();
        if !self.valid_player(player) {
            return false;
        }
        let base = self.bases[player as usize - 1];
        let base_cell = self.at(base);
        if base_cell.owner() != player || base_cell.kind() != CellKind::Base {
            return false;
        }
        bfs.seen[..count].fill(false);
        let base_index = self.index(base);
        bfs.seen[base_index] = true;
        bfs.queue[0] = base_index as u16;
        let (mut head, mut tail) = (0usize, 1usize);
        while head < tail {
            let current = bfs.queue[head] as usize;
            head += 1;
            let (row0, row1, col0, col1) = self.neighbourhood(current);
            for row in row0..=row1 {
                for col in col0..=col1 {
                    let neighbour = row * self.cols as usize + col;
                    let cell = self.cell_at(neighbour);
                    if cell.is_capturable_by(player) {
                        return true;
                    }
                    if !bfs.seen[neighbour] && cell.owner() == player {
                        bfs.seen[neighbour] = true;
                        bfs.queue[tail] = neighbour as u16;
                        tail += 1;
                    }
                }
            }
        }
        false
    }

    /// Whether `player` has at least one legal move.
    pub fn has_move(&self, player: Player) -> bool {
        with_thread_scratch(|scratch| self.has_move_bfs(player, &mut scratch.bfs))
    }

    /// Inclusive 8-neighbourhood bounds `(row0, row1, col0, col1)` of a board
    /// index, clamped to the board. The centre is included; call sites that
    /// must exclude it compare indices.
    #[inline]
    pub(crate) fn neighbourhood(&self, index: usize) -> (usize, usize, usize, usize) {
        let (rows, cols) = (self.rows as usize, self.cols as usize);
        let (row, col) = (index / cols, index % cols);
        (
            row.saturating_sub(1),
            (row + 1).min(rows - 1),
            col.saturating_sub(1),
            (col + 1).min(cols - 1),
        )
    }

    /// Visits the 8-neighbourhood of `index` **including** `index` itself.
    ///
    /// The Go original writes the frontier this way (`moveTargetsFrom` does not
    /// skip the centre). It is harmless — an own cell is never a legal target —
    /// but reproducing it keeps the code a line-for-line port.
    #[inline]
    fn for_each_neighbour_inclusive(&self, index: usize, mut visit: impl FnMut(usize)) {
        let (row0, row1, col0, col1) = self.neighbourhood(index);
        for row in row0..=row1 {
            for col in col0..=col1 {
                visit(row * self.cols as usize + col);
            }
        }
    }

    // ---------------------------------------------------------------- legality

    fn legal_action_with(&self, action: Action, scratch: &mut Scratch) -> bool {
        // `can_act` rather than `active`: a mover with no actions left has no
        // legal action, so `apply` rejects it instead of underflowing the turn
        // budget. See [`State::can_act`].
        if !self.can_act() {
            return false;
        }
        match action {
            Action::Move { target } => self.legal_move_with(self.current, target, scratch),
            Action::PlaceNeutrals { cells } => {
                if !self.can_place_neutrals() || cells[0] == cells[1] {
                    return false;
                }
                cells.iter().all(|&pos| {
                    self.at_checked(pos).is_some_and(|cell| {
                        cell.owner() == self.current && cell.kind() == CellKind::Normal
                    })
                })
            }
        }
    }

    /// Whether `player` may play `target`: the cell must be empty or an enemy
    /// `Normal`, and 8-adjacent to `player`'s base-connected component.
    pub fn legal_move(&self, player: Player, target: Pos) -> bool {
        with_thread_scratch(|scratch| self.legal_move_with(player, target, scratch))
    }

    fn legal_move_with(&self, player: Player, target: Pos, scratch: &mut Scratch) -> bool {
        let Some(cell) = self.at_checked(target) else {
            return false;
        };
        if !cell.is_capturable_by(player) {
            return false;
        }
        let Scratch { bfs, connected, .. } = scratch;
        self.connected_mask(player, connected, bfs);
        let target_index = self.index(target);
        let (row0, row1, col0, col1) = self.neighbourhood(target_index);
        for row in row0..=row1 {
            for col in col0..=col1 {
                let neighbour = row * self.cols as usize + col;
                if neighbour != target_index && connected[neighbour] {
                    return true;
                }
            }
        }
        false
    }

    // ---------------------------------------------------------------- transitions

    /// Legality-checked successor. On error the receiver is unchanged.
    pub fn apply(&self, action: Action) -> Result<State, RuleError> {
        with_thread_scratch(|scratch| self.apply_with(action, scratch))
    }

    /// [`State::apply`] using caller-supplied scratch space.
    pub fn apply_with(&self, action: Action, scratch: &mut Scratch) -> Result<State, RuleError> {
        if self.over {
            return Err(RuleError::GameOver);
        }
        if !self.legal_action_with(action, scratch) {
            return Err(RuleError::InvalidAction);
        }
        Ok(self.mutate(action, scratch))
    }

    /// Search hot-path successor for an action already emitted by
    /// [`crate::Position`]. Skips the legality traversal but shares every other
    /// rule with [`State::apply`].
    ///
    /// # Panics
    /// Panics on an out-of-bounds coordinate. Never hand it unvalidated input —
    /// an illegal move sent to the server is an instant forfeit.
    pub fn apply_generated(&self, action: Action) -> State {
        with_thread_scratch(|scratch| self.apply_generated_with(action, scratch))
    }

    /// [`State::apply_generated`] using caller-supplied scratch space.
    pub fn apply_generated_with(&self, action: Action, scratch: &mut Scratch) -> State {
        self.mutate(action, scratch)
    }

    fn mutate(&self, action: Action, scratch: &mut Scratch) -> State {
        let mut next = self.clone();
        let player = self.current;
        match action {
            Action::PlaceNeutrals { cells } => {
                // Consumes the entire turn, not one action — and it is once per
                // game per player.
                let first = next.index(cells[0]);
                let second = next.index(cells[1]);
                next.set_cell(first, Cell::NEUTRAL);
                next.set_cell(second, Cell::NEUTRAL);
                next.set_neutral_used(player, true);
                next.set_moves_left(0);
            }
            Action::Move { target } => {
                // ARCHITECTURE.md invariant 4: capturing an enemy `Normal`
                // yields *your* `Fortified`, never your `Normal`. Getting this
                // backwards silently corrupted a whole training run.
                let index = next.index(target);
                let kind = if self.cell_at(index).kind() == CellKind::Normal {
                    CellKind::Fortified
                } else {
                    CellKind::Normal
                };
                next.set_cell(index, Cell::new(player, kind));
                next.set_moves_left(next.moves_left - 1);
            }
        }
        next.eliminate_stuck_players(&mut scratch.bfs);
        if next.finish_if_terminal() {
            return next;
        }
        // The mover keeps the turn while it still has actions — this is why the
        // side to move does NOT alternate per ply (invariant 1).
        if !next.active(player) || next.moves_left == 0 {
            next.advance(player);
        }
        next
    }

    /// Deactivates every active player with no legal move.
    ///
    /// Their cells **stay on the board and remain capturable** — only the flag
    /// flips (GoBot vs-ai2.45). Erasing them was a real production bug.
    fn eliminate_stuck_players(&mut self, bfs: &mut BfsScratch) {
        for player in 1..=self.players {
            if self.active(player) && !self.has_move_bfs(player, bfs) {
                self.set_active(player, false);
            }
        }
    }

    fn finish_if_terminal(&mut self) -> bool {
        let mut active = 0;
        let mut winner = 0;
        for player in 1..=self.players {
            if self.active(player) {
                active += 1;
                winner = player;
            }
        }
        if active > 1 {
            return false;
        }
        self.over = true;
        self.winner = winner;
        self.set_moves_left(0);
        true
    }

    fn advance(&mut self, after: Player) {
        for offset in 1..=self.players {
            let player = (after - 1 + offset) % self.players + 1;
            if self.active(player) {
                self.set_current(player);
                self.set_moves_left(ACTIONS_PER_TURN);
                return;
            }
        }
    }

    // ---------------------------------------------------------------- snapshot hooks
    //
    // Only `snapshot.rs` uses these: importing an untrusted position needs to
    // install a server-declared base layout and a defensively recomputed
    // activity set, neither of which any rules transition may do.

    pub(crate) fn override_bases(&mut self, bases: [Pos; MAX_PLAYERS]) {
        self.bases = bases;
    }

    pub(crate) fn override_active(&mut self, player: Player, value: bool) {
        self.set_active(player, value);
    }

    pub(crate) fn override_terminal(&mut self, winner: Player) {
        self.over = true;
        self.winner = winner;
        self.set_moves_left(0);
    }

    // ---------------------------------------------------------------- outcome

    /// Size-general terminal outcome, including turn-capped and board-filled
    /// games. Port of Java `GoState.outcomeWinner`.
    ///
    /// The real rule is "the last player still able to move wins". When nobody
    /// can move (simultaneous fill) or everybody still can (a turn cap), the
    /// tiebreak is territory: most owned cells wins, an exact tie returns `0`.
    ///
    /// Eliminated players keep their cells, so a base-destroyed player could
    /// otherwise out-territory a live-but-stuck opponent and be handed the win.
    /// Only players still flagged active share the tiebreak — unless nobody is,
    /// in which case all seats are compared.
    pub fn outcome_winner(&self) -> Player {
        with_thread_scratch(|scratch| {
            let mut survivors = 0;
            let mut survivor = 0;
            for player in 1..=self.players {
                if self.active(player) && self.has_move_bfs(player, &mut scratch.bfs) {
                    survivors += 1;
                    survivor = player;
                }
            }
            if survivors == 1 {
                return survivor;
            }
            let any_active = (1..=self.players).any(|player| self.active(player));
            let mut best = 0;
            let mut best_owned = -1i32;
            let mut tie = false;
            for player in 1..=self.players {
                if any_active && !self.active(player) {
                    continue;
                }
                let owned = self.owned_cells(player) as i32;
                if owned > best_owned {
                    best_owned = owned;
                    best = player;
                    tie = false;
                } else if owned == best_owned {
                    tie = true;
                }
            }
            if tie {
                0
            } else {
                best
            }
        })
    }
}

/// Base corners in seat order: P1 top-left, P2 bottom-right, P3 top-right,
/// P4 bottom-left.
pub(crate) fn default_bases(rows: usize, cols: usize) -> [Pos; MAX_PLAYERS] {
    [
        Pos::new(0, 0),
        Pos::new(rows as i32 - 1, cols as i32 - 1),
        Pos::new(0, cols as i32 - 1),
        Pos::new(rows as i32 - 1, 0),
    ]
}

impl fmt::Debug for State {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "State {}x{} players={} current={} movesLeft={} over={} winner={}",
            self.rows,
            self.cols,
            self.players,
            self.current,
            self.moves_left,
            self.over,
            self.winner
        )?;
        for row in 0..self.rows() {
            for col in 0..self.cols() {
                let cell = self.at(Pos::new(row as i32, col as i32));
                let glyph = match cell.kind() {
                    CellKind::Empty => '.',
                    CellKind::Neutral => '#',
                    CellKind::Normal => (b'0' + cell.owner()) as char,
                    CellKind::Base => (b'A' + cell.owner() - 1) as char,
                    CellKind::Fortified => (b'a' + cell.owner() - 1) as char,
                };
                write!(f, "{glyph}")?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}
