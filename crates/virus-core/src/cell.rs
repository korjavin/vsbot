//! Board cell primitives.
//!
//! A [`Cell`] is packed into a single byte as `owner << 3 | kind`, which is the
//! exact byte GoBot's `stateHash` (`virusgame/backend/search/search.go`) and the
//! Java `GoState.hash()` feed into FNV-1a. Keeping the same packing means
//! [`crate::State::state_hash`] can be a literal transcription of theirs.

use std::fmt;

/// Seat number. `0` means "nobody"; real players are `1..=4`.
pub type Player = u8;

/// Maximum number of seats the rules support.
pub const MAX_PLAYERS: usize = 4;

/// What occupies a cell.
///
/// Discriminants match Go's `game.CellKind` iota order exactly — they are part
/// of the wire format and of the hash byte, so they must not be reordered.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
#[repr(u8)]
pub enum CellKind {
    /// Unoccupied and playable.
    #[default]
    Empty = 0,
    /// A normal piece. The only capturable kind.
    Normal = 1,
    /// A player's home base. Invulnerable; the root of connectivity.
    Base = 2,
    /// A captured piece. Invulnerable.
    Fortified = 3,
    /// Dead space created by `PlaceNeutrals`. Belongs to nobody, never playable.
    Neutral = 4,
}

impl CellKind {
    /// Number of distinct kinds; used to size the Zobrist table.
    pub const COUNT: usize = 5;

    /// Decodes a numeric kind, rejecting out-of-range values.
    pub const fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(CellKind::Empty),
            1 => Some(CellKind::Normal),
            2 => Some(CellKind::Base),
            3 => Some(CellKind::Fortified),
            4 => Some(CellKind::Neutral),
            _ => None,
        }
    }

    /// The numeric discriminant.
    pub const fn as_u8(self) -> u8 {
        self as u8
    }

    /// The upper-case wire name used by the JSON fixtures.
    pub const fn name(self) -> &'static str {
        match self {
            CellKind::Empty => "EMPTY",
            CellKind::Normal => "NORMAL",
            CellKind::Base => "BASE",
            CellKind::Fortified => "FORTIFIED",
            CellKind::Neutral => "NEUTRAL",
        }
    }

    /// Parses a wire name. Accepts any letter case.
    pub fn parse(text: &str) -> Option<Self> {
        match text.to_ascii_uppercase().as_str() {
            "EMPTY" => Some(CellKind::Empty),
            "NORMAL" => Some(CellKind::Normal),
            "BASE" => Some(CellKind::Base),
            "FORTIFIED" => Some(CellKind::Fortified),
            "NEUTRAL" => Some(CellKind::Neutral),
            _ => None,
        }
    }
}

impl fmt::Display for CellKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// An owner + kind pair packed into one byte (`owner << 3 | kind`).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Cell(u8);

impl Cell {
    /// The empty cell (owner 0, kind [`CellKind::Empty`]).
    pub const EMPTY: Cell = Cell(0);
    /// A neutral (dead) cell.
    pub const NEUTRAL: Cell = Cell(CellKind::Neutral as u8);

    /// Builds a cell. Owners above [`MAX_PLAYERS`] are rejected by
    /// [`Cell::is_well_formed`], not here, so decoders can inspect them.
    pub const fn new(owner: Player, kind: CellKind) -> Cell {
        Cell((owner << 3) | kind as u8)
    }

    /// The owning seat, or `0`.
    pub const fn owner(self) -> Player {
        self.0 >> 3
    }

    /// What occupies the cell.
    pub const fn kind(self) -> CellKind {
        // The low three bits can only hold 0..=7; `new` never builds anything
        // above 4, and `from_packed` rejects the rest.
        match CellKind::from_u8(self.0 & 0b111) {
            Some(kind) => kind,
            None => CellKind::Empty,
        }
    }

    /// The raw hash byte (`owner << 3 | kind`).
    pub const fn packed(self) -> u8 {
        self.0
    }

    /// Decodes a raw hash byte, rejecting invalid kinds and owners.
    pub const fn from_packed(byte: u8) -> Option<Cell> {
        if byte >> 3 > MAX_PLAYERS as u8 {
            return None;
        }
        match CellKind::from_u8(byte & 0b111) {
            Some(_) => Some(Cell(byte)),
            None => None,
        }
    }

    /// Wraps a byte that is already known to be a valid packing (every byte the
    /// engine stores came from [`Cell::new`]). An invalid low nibble degrades
    /// to [`CellKind::Empty`] rather than panicking.
    #[inline]
    pub(crate) const fn from_packed_unchecked(byte: u8) -> Cell {
        Cell(byte)
    }

    /// Dense index in `0..CellKind::COUNT * (MAX_PLAYERS + 1)`, used to index
    /// the Zobrist table without wasting the sparse `owner << 3` gaps.
    pub(crate) const fn zobrist_code(self) -> usize {
        (self.kind() as usize) * (MAX_PLAYERS + 1) + self.owner() as usize
    }

    /// True when owner and kind agree: only `Empty`/`Neutral` may be ownerless,
    /// and only `Normal`/`Base`/`Fortified` may be owned.
    pub fn is_well_formed(self, players: usize) -> bool {
        if self.owner() as usize > players {
            return false;
        }
        match self.kind() {
            CellKind::Empty | CellKind::Neutral => self.owner() == 0,
            CellKind::Normal | CellKind::Base | CellKind::Fortified => self.owner() >= 1,
        }
    }

    /// True when this cell is a legal move target for `player`: an empty cell,
    /// or an *enemy* `Normal`. `Base`, `Fortified` and `Neutral` are immune.
    #[inline]
    pub const fn is_capturable_by(self, player: Player) -> bool {
        matches!(self.kind(), CellKind::Empty)
            || (matches!(self.kind(), CellKind::Normal) && self.owner() != player)
    }
}

impl fmt::Debug for Cell {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Cell({}, {})", self.owner(), self.kind())
    }
}
