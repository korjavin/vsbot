//! Incremental Zobrist hashing.
//!
//! ARCHITECTURE.md invariant 6: the transposition key **must** include
//! `movesLeft`, `neutralUsed` and the side to move, not just the grid. Two
//! positions with identical boards but different `movesLeft` have completely
//! different values (a 3-action turn is worth roughly three tempi), and the Go
//! engine's TT bugs all traced back to an under-specified key.
//!
//! `active` is hashed too, even though the bead only requires the four fields
//! above. Elimination is *sticky*: `eliminate_stuck_players` only ever clears
//! the flag, and a stuck player can regain a legal target when an opponent
//! plays a fresh `Normal` next to them. So `active` is genuinely not a function
//! of the grid, and omitting it would let two different positions collide.
//!
//! The tables are derived from SplitMix64 with a fixed seed, so the keys are
//! identical on every run and every machine (CLAUDE.md: all engine randomness
//! is seeded and deterministic).

use crate::cell::{Cell, CellKind, MAX_PLAYERS};
use crate::{ACTIONS_PER_TURN, MAX_CELLS};
use std::sync::OnceLock;

/// Distinct `(kind, owner)` codes a cell can take.
const CELL_CODES: usize = CellKind::COUNT * (MAX_PLAYERS + 1);

pub(crate) struct Zobrist {
    /// `cells[index][code]`, `MAX_CELLS` rows. The code for a plain empty cell
    /// is forced to zero so a fresh board hashes to `0` for its grid
    /// contribution and `set_cell` stays a pure two-XOR update. Heap-allocated
    /// directly (rather than as a boxed array literal) to keep the 500 KiB
    /// table off the stack during construction.
    cells: Vec<[u64; CELL_CODES]>,
    /// Keyed by seat, index `player`. Index 0 is unused.
    current: [u64; MAX_PLAYERS + 1],
    /// Keyed by remaining actions, `0..=ACTIONS_PER_TURN`.
    moves_left: [u64; ACTIONS_PER_TURN as usize + 1],
    /// Keyed by seat index (`player - 1`).
    active: [u64; MAX_PLAYERS],
    /// Keyed by seat index (`player - 1`).
    neutral_used: [u64; MAX_PLAYERS],
}

/// SplitMix64 — a small, well-distributed, fully deterministic generator.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// Seed for the Zobrist tables. Changing it changes every hash; it must stay
/// fixed so persisted books/tables remain valid.
const ZOBRIST_SEED: u64 = 0x5653_424F_5401_2026;

fn build() -> Box<Zobrist> {
    let mut rng = SplitMix64(ZOBRIST_SEED);
    let mut table = Box::new(Zobrist {
        cells: Vec::with_capacity(MAX_CELLS),
        current: [0; MAX_PLAYERS + 1],
        moves_left: [0; ACTIONS_PER_TURN as usize + 1],
        active: [0; MAX_PLAYERS],
        neutral_used: [0; MAX_PLAYERS],
    });
    let empty_code = Cell::EMPTY.zobrist_code();
    for _ in 0..MAX_CELLS {
        let mut row = [0u64; CELL_CODES];
        for (code, key) in row.iter_mut().enumerate() {
            // Draw unconditionally so the generator stream stays independent of
            // which code is zeroed, keeping the table stable if
            // `Cell::zobrist_code` is ever re-ordered.
            let drawn = rng.next();
            *key = if code == empty_code { 0 } else { drawn };
        }
        table.cells.push(row);
    }
    for key in table.current.iter_mut() {
        *key = rng.next();
    }
    for key in table.moves_left.iter_mut() {
        *key = rng.next();
    }
    for key in table.active.iter_mut() {
        *key = rng.next();
    }
    for key in table.neutral_used.iter_mut() {
        *key = rng.next();
    }
    table
}

pub(crate) fn table() -> &'static Zobrist {
    static TABLE: OnceLock<Box<Zobrist>> = OnceLock::new();
    TABLE.get_or_init(build)
}

impl Zobrist {
    #[inline]
    pub(crate) fn cell(&self, index: usize, cell: Cell) -> u64 {
        self.cells[index][cell.zobrist_code()]
    }

    #[inline]
    pub(crate) fn current(&self, player: u8) -> u64 {
        self.current[player as usize]
    }

    #[inline]
    pub(crate) fn moves_left(&self, moves_left: u8) -> u64 {
        self.moves_left[moves_left as usize]
    }

    #[inline]
    pub(crate) fn active(&self, player: u8) -> u64 {
        self.active[player as usize - 1]
    }

    #[inline]
    pub(crate) fn neutral_used(&self, player: u8) -> u64 {
        self.neutral_used[player as usize - 1]
    }
}
