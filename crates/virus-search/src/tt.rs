//! The packed, lockless transposition table used by the enhanced search.
//!
//! Port of `nnue-trainer/.../search/gobot/GoTranspositionTable.java`. Fixed-size,
//! power-of-two, allocation-free probe/store, depth-preferred and
//! generation-aged replacement. The plain (parity-oracle) path keeps GoBot's
//! unbounded map instead — see [`crate::Searcher`].
//!
//! # Entry packing (64 bits)
//!
//! `score 32 | depth 6 | flag 2 | generation 6 | action 18`
//!
//! Scores are root-relative and ply-independent **except** mate scores, which
//! encode distance-to-root. Those are rebased to node-relative distance on store
//! ([`to_stored_score`]) and back to the probing node's ply on probe
//! ([`from_stored_score`]) — the standard mate-in-TT correction, and
//! ARCHITECTURE.md invariant 6.
//!
//! # Thread safety
//!
//! Lazy SMP shares one table between threads, so the classic XOR trick applies:
//! `keys[i] = key ^ data[i]`, and a probe only trusts `data[i]` when
//! `keys[i] ^ data[i]` reproduces the probed key. Key and payload live in two
//! separate slots, so without the XOR a reader could pair a stale key with a new
//! entry's payload and attribute a wrong score or move to the position. With it,
//! any mismatched pair fails the verify with probability `~1 - 2^-64` and reads
//! as a miss.
//!
//! Java gets away with plain `long[]` because its races are benign-by-JLS; Rust
//! needs the accesses to be atomic to be defined at all, so both arrays are
//! [`AtomicU64`] read and written with [`Ordering::Relaxed`]. Relaxed is exactly
//! the guarantee the XOR verify needs (per-location atomicity, no ordering
//! between the two stores) and compiles to the same plain loads and stores on
//! every target we ship on — no fences on the hottest read in the search.
//!
//! # Action packing (18 bits)
//!
//! Bit 17 = present, bit 16 = kind (`0` move, `1` neutral pair). A move stores a
//! 16-bit cell index; a neutral pair stores two 8-bit cell indices, normalized
//! ascending because `PlaceNeutrals` compares unordered. Anything that does not
//! fit (a board above 256 cells for a pair) stores as absent, which only loses
//! an ordering hint.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use virus_core::{Action, Pos};
use virus_eval::{Score, MATE_SCORE};

/// Default table size: `2^21` entries (32 MiB across the two arrays).
pub const DEFAULT_LOG2_SIZE: u32 = 21;

/// Scores beyond this magnitude are mate scores and carry a ply distance.
pub const MATE_BAND: Score = MATE_SCORE - 1000;

const ACTION_PRESENT: u32 = 1 << 17;
const ACTION_NEUTRAL: u32 = 1 << 16;
const ACTION_MASK: u32 = 0x3FFFF;

/// A raw packed entry. `0` means "no entry".
pub type PackedEntry = u64;

/// The packed transposition table.
pub struct TranspositionTable {
    keys: Vec<AtomicU64>,
    data: Vec<AtomicU64>,
    mask: usize,
    generation: AtomicU32,
}

impl std::fmt::Debug for TranspositionTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TranspositionTable")
            .field("entries", &self.data.len())
            .field("generation", &self.generation.load(Ordering::Relaxed))
            .finish()
    }
}

impl Default for TranspositionTable {
    fn default() -> TranspositionTable {
        TranspositionTable::new(DEFAULT_LOG2_SIZE)
    }
}

impl TranspositionTable {
    /// Allocates a table of `2^log2_size` entries.
    ///
    /// # Panics
    /// Panics when `log2_size` is 0 or above 31.
    pub fn new(log2_size: u32) -> TranspositionTable {
        assert!(
            (1..=31).contains(&log2_size),
            "transposition table log2 size must be 1..=31, got {log2_size}"
        );
        let size = 1usize << log2_size;
        TranspositionTable {
            keys: (0..size).map(|_| AtomicU64::new(0)).collect(),
            data: (0..size).map(|_| AtomicU64::new(0)).collect(),
            mask: size - 1,
            generation: AtomicU32::new(0),
        }
    }

    /// Number of slots.
    pub fn capacity(&self) -> usize {
        self.data.len()
    }

    /// The current generation (0..64).
    pub fn generation(&self) -> u32 {
        self.generation.load(Ordering::Relaxed)
    }

    /// Starts a new search: everything currently stored ages by one generation.
    ///
    /// Called once per move, never per iteration — the table deliberately
    /// survives between moves so each search starts warm on the previous move's
    /// principal subtree.
    pub fn bump_generation(&self) {
        let next = (self.generation.load(Ordering::Relaxed) + 1) & 63;
        self.generation.store(next, Ordering::Relaxed);
    }

    /// Wipes the table. Only tests and a seat change need this.
    pub fn clear(&self) {
        for slot in &self.keys {
            slot.store(0, Ordering::Relaxed);
        }
        for slot in &self.data {
            slot.store(0, Ordering::Relaxed);
        }
    }

    /// The packed entry for `key`, or `0` on a miss.
    ///
    /// Stored entries always carry depth >= 1, so a zero payload is
    /// unambiguously "empty".
    pub fn probe(&self, key: u64) -> PackedEntry {
        let index = (key as usize) & self.mask;
        let data = self.data[index].load(Ordering::Relaxed);
        if self.keys[index].load(Ordering::Relaxed) ^ data == key {
            data
        } else {
            0
        }
    }

    /// Depth-preferred, generation-aged replacement.
    ///
    /// Always replaces an empty slot, the same key, or anything from an older
    /// generation; within the current generation the deeper entry keeps the
    /// slot. Under SMP the inspection of the old entry is racy, and a wrong
    /// replacement decision is benign.
    pub fn store(&self, key: u64, depth: i32, flag: u8, score: i32, action_bits: u32) {
        let index = (key as usize) & self.mask;
        let old = self.data[index].load(Ordering::Relaxed);
        let same_key = self.keys[index].load(Ordering::Relaxed) ^ old == key;
        let generation = self.generation.load(Ordering::Relaxed);
        if old != 0 && !same_key && generation_of(old) == generation && depth_of(old) > depth {
            return;
        }
        let packed = (score as u32 as u64)
            | (((depth as u64) & 63) << 32)
            | (((flag as u64) & 3) << 38)
            | ((generation as u64) << 40)
            | (((action_bits as u64) & ACTION_MASK as u64) << 46);
        self.data[index].store(packed, Ordering::Relaxed);
        self.keys[index].store(key ^ packed, Ordering::Relaxed);
    }
}

/// The stored (possibly mate-rebased) score of a packed entry.
#[inline]
pub fn score_of(entry: PackedEntry) -> i32 {
    entry as u32 as i32
}

/// The stored depth of a packed entry.
#[inline]
pub fn depth_of(entry: PackedEntry) -> i32 {
    ((entry >> 32) & 63) as i32
}

/// The stored bound flag of a packed entry.
#[inline]
pub fn flag_of(entry: PackedEntry) -> u8 {
    ((entry >> 38) & 3) as u8
}

/// The generation a packed entry was written in.
#[inline]
pub fn generation_of(entry: PackedEntry) -> u32 {
    ((entry >> 40) & 63) as u32
}

/// The packed action bits of an entry.
#[inline]
pub fn action_bits_of(entry: PackedEntry) -> u32 {
    ((entry >> 46) & ACTION_MASK as u64) as u32
}

/// Root-relative score at `ply` to its stored (node-relative for mates) form.
#[inline]
pub fn to_stored_score(score: Score, ply: i32) -> i32 {
    if score > MATE_BAND {
        (score + ply as Score) as i32
    } else if score < -MATE_BAND {
        (score - ply as Score) as i32
    } else {
        score as i32
    }
}

/// Stored score back to root-relative at the probing node's `ply`.
#[inline]
pub fn from_stored_score(stored: i32, ply: i32) -> Score {
    let stored = stored as Score;
    if stored > MATE_BAND {
        stored - ply as Score
    } else if stored < -MATE_BAND {
        stored + ply as Score
    } else {
        stored
    }
}

/// 18-bit encoding of `action` on a `cols`-wide, `cells`-cell board, or `0` when
/// it does not fit (which reads back as "absent").
pub fn encode_action(action: Action, cols: usize, cells: usize) -> u32 {
    match action {
        Action::Move { target } => {
            let index = target.row as isize * cols as isize + target.col as isize;
            if index < 0 || index as usize >= cells || index > 0xFFFF {
                return 0;
            }
            ACTION_PRESENT | index as u32
        }
        Action::PlaceNeutrals { cells: pair } => {
            let a = pair[0].row as isize * cols as isize + pair[0].col as isize;
            let b = pair[1].row as isize * cols as isize + pair[1].col as isize;
            let (low, high) = (a.min(b), a.max(b));
            if low < 0 || high as usize >= cells || high > 0xFF {
                return 0;
            }
            ACTION_PRESENT | ACTION_NEUTRAL | ((high as u32) << 8) | low as u32
        }
    }
}

/// Inverse of [`encode_action`]; `None` when the entry carries no action.
pub fn decode_action(bits: u32, cols: usize) -> Option<Action> {
    if bits & ACTION_PRESENT == 0 {
        return None;
    }
    let cols = cols as i32;
    if bits & ACTION_NEUTRAL == 0 {
        let index = (bits & 0xFFFF) as i32;
        return Some(Action::Move {
            target: Pos::new(index / cols, index % cols),
        });
    }
    let low = (bits & 0xFF) as i32;
    let high = ((bits >> 8) & 0xFF) as i32;
    Some(Action::neutrals(
        Pos::new(low / cols, low % cols),
        Pos::new(high / cols, high % cols),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_a_stored_entry() {
        let table = TranspositionTable::new(8);
        table.store(
            0x1234_5678_9abc_def0,
            7,
            1,
            -4321,
            encode_action(Action::mv(3, 4), 12, 144),
        );
        let entry = table.probe(0x1234_5678_9abc_def0);
        assert_ne!(entry, 0);
        assert_eq!(score_of(entry), -4321);
        assert_eq!(depth_of(entry), 7);
        assert_eq!(flag_of(entry), 1);
        assert_eq!(
            decode_action(action_bits_of(entry), 12),
            Some(Action::mv(3, 4))
        );
    }

    #[test]
    fn a_corrupted_key_slot_reads_as_a_miss() {
        let table = TranspositionTable::new(8);
        table.store(99, 3, 0, 12, 0);
        assert_ne!(table.probe(99), 0);
        // A different key landing on the same slot must not be trusted.
        let colliding = 99 + (1 << 20);
        assert_eq!(table.probe(colliding), 0);
    }

    #[test]
    fn deeper_same_generation_entries_keep_the_slot() {
        let table = TranspositionTable::new(4);
        let mask = table.capacity() as u64 - 1;
        let a = 0x1000 & mask;
        let b = a + (1 << 12);
        table.store(a, 9, 0, 1, 0);
        table.store(b, 3, 0, 2, 0);
        assert_ne!(
            table.probe(a),
            0,
            "shallower entry must not evict the deeper one"
        );
        assert_eq!(table.probe(b), 0);
        // A new generation ages the deep entry out of the way.
        table.bump_generation();
        table.store(b, 3, 0, 2, 0);
        assert_ne!(table.probe(b), 0);
        assert_eq!(table.probe(a), 0);
    }

    #[test]
    fn mate_scores_rebase_symmetrically() {
        let mate = MATE_SCORE - 4;
        assert_eq!(from_stored_score(to_stored_score(mate, 6), 6), mate);
        assert_eq!(from_stored_score(to_stored_score(-mate, 6), 6), -mate);
        // A mate stored 6 plies down is 6 plies closer when probed at the root.
        let stored = to_stored_score(mate, 6);
        assert_eq!(from_stored_score(stored, 0), mate + 6);
        // Ordinary scores pass through untouched.
        assert_eq!(to_stored_score(1234, 9), 1234);
        assert_eq!(from_stored_score(1234, 9), 1234);
    }

    #[test]
    fn neutral_pairs_encode_unordered() {
        let forward = encode_action(Action::neutrals(Pos::new(1, 2), Pos::new(3, 4)), 12, 144);
        let backward = encode_action(Action::neutrals(Pos::new(3, 4), Pos::new(1, 2)), 12, 144);
        assert_eq!(forward, backward);
        let decoded = decode_action(forward, 12).expect("present");
        assert!(decoded.same_transition(Action::neutrals(Pos::new(1, 2), Pos::new(3, 4))));
    }

    #[test]
    fn oversized_neutral_pairs_store_as_absent() {
        // 20x20 = 400 cells: the second index no longer fits in 8 bits.
        assert_eq!(
            encode_action(Action::neutrals(Pos::new(19, 19), Pos::new(0, 0)), 20, 400),
            0
        );
        assert_eq!(decode_action(0, 20), None);
    }
}
