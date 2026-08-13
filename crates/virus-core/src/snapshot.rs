//! Wire snapshot decoding and validation.
//!
//! ARCHITECTURE.md invariant 5: **the snapshot is the only board source.** The
//! board is never reconstructed from move deltas, and every snapshot is
//! re-validated before the engine acts on it — an illegal move sent to the
//! server is an instant forfeit.
//!
//! Port of `virusgame/backend/game/snapshot.go: FromSnapshot`, plus one
//! deliberate deviation documented on [`Snapshot::decode`]: the server's
//! `Active[]` is not trusted.
//!
//! # Decoding tolerance
//!
//! Two producers emit these structures with different conventions:
//!
//! * The Go server marshals `game.Pos` and `game.Cell` with **no** JSON tags,
//!   so it emits `{"Row":…,"Col":…}` and `{"Owner":1,"Kind":2}` — PascalCase
//!   keys and *numeric* kinds.
//! * The parity fixtures use `{"row":…,"col":…}` and
//!   `{"owner":1,"kind":"BASE"}` — lowercase keys and *string* kinds.
//!
//! Both are accepted everywhere. Unknown keys are ignored so a server-side
//! field addition cannot break the bot mid-game.

use crate::action::Pos;
use crate::cell::{Cell, CellKind, Player, MAX_PLAYERS};
use crate::state::{default_bases, State, ACTIONS_PER_TURN, MAX_DIM};
use serde::de::{self, MapAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;

/// Why a snapshot was rejected.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct SnapshotError(&'static str);

impl SnapshotError {
    /// The reason, for logs.
    pub fn reason(self) -> &'static str {
        self.0
    }
}

impl fmt::Display for SnapshotError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid snapshot: {}", self.0)
    }
}

impl std::error::Error for SnapshotError {}

/// The wire representation of a complete game position. Seats and
/// `current_player` are 1-based; the vectors are ordered by seat.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Snapshot {
    /// Board height.
    #[serde(alias = "Rows")]
    pub rows: usize,
    /// Board width.
    #[serde(alias = "Cols")]
    pub cols: usize,
    /// Row-major grid, `rows` rows of `cols` cells.
    #[serde(alias = "Board")]
    pub board: Vec<Vec<Cell>>,
    /// Base coordinate per seat. Its length defines the number of players.
    #[serde(alias = "Bases")]
    pub bases: Vec<Pos>,
    /// Server-reported activity per seat. **Not trusted** — see
    /// [`Snapshot::decode`].
    #[serde(alias = "Active")]
    pub active: Vec<bool>,
    /// Whether each seat has spent its neutral placement.
    #[serde(alias = "NeutralUsed", alias = "neutral_used")]
    pub neutral_used: Vec<bool>,
    /// Seat to move.
    #[serde(alias = "CurrentPlayer", alias = "current", alias = "player")]
    pub current_player: Player,
    /// Actions remaining in the current turn.
    #[serde(alias = "MovesLeft", alias = "moves_left")]
    pub moves_left: u8,
    /// Whether the server considers the game finished.
    #[serde(default, alias = "GameOver", alias = "game_over")]
    pub game_over: bool,
    /// Winning seat, or `0`.
    #[serde(default, alias = "Winner")]
    pub winner: Player,
}

impl Snapshot {
    /// Validates and imports an untrusted wire snapshot.
    ///
    /// Faithful to Go's `FromSnapshot` except for one deliberate hardening.
    ///
    /// # The `Active[]` server bug
    ///
    /// ARCHITECTURE.md invariant 5 records a live server quirk: seats that have
    /// been eliminated but still own cells are sometimes reported as active.
    /// Trusting that flag makes the engine search branches for a player who
    /// cannot move, and mis-scores every terminal.
    ///
    /// So activity is **recomputed**: a seat is active only if the server says
    /// so *and* its base is intact *and* it has a legal move. The derivation
    /// only ever clears the flag — it never promotes a seat the server called
    /// dead, because elimination is sticky in the real rules and a stuck player
    /// can technically regain a target later.
    ///
    /// If the recomputation leaves at most one live seat, the game really is
    /// over and the snapshot is imported as terminal rather than rejected.
    pub fn decode(&self) -> Result<State, SnapshotError> {
        let players = self.bases.len();
        if self.rows < 2 || self.rows > MAX_DIM || self.cols < 2 || self.cols > MAX_DIM {
            return Err(SnapshotError("board dimensions out of range"));
        }
        if !(2..=MAX_PLAYERS).contains(&players) {
            return Err(SnapshotError("player count out of range"));
        }
        if self.active.len() != players || self.neutral_used.len() != players {
            return Err(SnapshotError("per-seat vector length mismatch"));
        }
        if self.board.len() != self.rows {
            return Err(SnapshotError("board row count mismatch"));
        }
        if self.moves_left > ACTIONS_PER_TURN {
            return Err(SnapshotError("movesLeft out of range"));
        }
        if self.current_player < 1 || self.current_player as usize > players {
            return Err(SnapshotError("currentPlayer out of range"));
        }
        if self.winner != 0 && self.winner as usize > players {
            return Err(SnapshotError("winner out of range"));
        }

        let mut cells = Vec::with_capacity(self.rows * self.cols);
        let mut has_pieces = [false; MAX_PLAYERS];
        for (row_index, row) in self.board.iter().enumerate() {
            if row.len() != self.cols {
                return Err(SnapshotError("board column count mismatch"));
            }
            for (col_index, cell) in row.iter().enumerate() {
                if !cell.is_well_formed(players) {
                    return Err(SnapshotError("malformed cell"));
                }
                if cell.owner() > 0 {
                    has_pieces[cell.owner() as usize - 1] = true;
                }
                if cell.kind() == CellKind::Base
                    && self.bases[cell.owner() as usize - 1]
                        != Pos::new(row_index as i32, col_index as i32)
                {
                    return Err(SnapshotError("base cell not at the declared base"));
                }
                cells.push(*cell);
            }
        }

        // Build with the declared bases before deriving activity: connectivity
        // (and therefore "has a legal move") is rooted at the base.
        let mut state = State::from_grid(
            self.rows,
            self.cols,
            players,
            &cells,
            self.current_player,
            self.moves_left,
            &self.neutral_used,
        )
        .map_err(|_| SnapshotError("grid rejected by the rules engine"))?;

        // The declared bases override the default corners: the server owns the
        // layout, and connectivity is rooted at whatever it says.
        let mut bases = default_bases(self.rows, self.cols);
        for (seat, slot) in bases.iter_mut().enumerate().take(players) {
            let base = self.bases[seat];
            if !state.in_bounds(base) {
                return Err(SnapshotError("base out of bounds"));
            }
            if self.bases[..seat].contains(&base) {
                return Err(SnapshotError("duplicate base"));
            }
            *slot = base;
        }
        state.override_bases(bases);

        for (seat, &owns_pieces) in has_pieces.iter().enumerate().take(players) {
            let player = seat as Player + 1;
            // Eliminated players keep their cells (invariant: cells stay and
            // remain capturable), so an inactive seat MAY still own pieces.
            // Only the forward direction holds: an active seat must own pieces
            // and hold an intact base.
            let base_cell = state.at(self.bases[seat]);
            let base_intact = base_cell.owner() == player && base_cell.kind() == CellKind::Base;
            if self.active[seat] && !owns_pieces {
                return Err(SnapshotError("active seat owns no pieces"));
            }
            if self.active[seat] && !base_intact {
                return Err(SnapshotError("active seat has no intact base"));
            }
            let derived = self.active[seat] && base_intact && state.has_move(player);
            state.override_active(player, derived);
        }

        if self.game_over {
            state.override_terminal(self.winner);
            return Ok(state);
        }
        let live: Vec<Player> = (1..=players as Player)
            .filter(|&player| state.active(player))
            .collect();
        if live.len() <= 1 {
            // The defensive recomputation just discovered the game is actually
            // decided. Import as terminal instead of rejecting the snapshot.
            state.override_terminal(live.first().copied().unwrap_or(0));
            return Ok(state);
        }
        if self.winner != 0 {
            return Err(SnapshotError("winner declared while the game runs"));
        }
        if !state.active(self.current_player) {
            // The side to move is never an eliminated player. Reaching here
            // means the server handed us a turn we cannot legally play.
            return Err(SnapshotError("side to move is not active"));
        }
        Ok(state)
    }
}

impl State {
    /// Detached wire snapshot of this state.
    pub fn snapshot(&self) -> Snapshot {
        let players = self.players();
        let mut board = Vec::with_capacity(self.rows());
        for row in 0..self.rows() {
            let mut cells = Vec::with_capacity(self.cols());
            for col in 0..self.cols() {
                cells.push(self.at(Pos::new(row as i32, col as i32)));
            }
            board.push(cells);
        }
        Snapshot {
            rows: self.rows(),
            cols: self.cols(),
            board,
            bases: (1..=players as Player).map(|p| self.base(p)).collect(),
            active: (1..=players as Player).map(|p| self.active(p)).collect(),
            neutral_used: (1..=players as Player)
                .map(|p| self.neutral_used(p))
                .collect(),
            current_player: self.current_player(),
            moves_left: self.moves_left(),
            game_over: self.game_over(),
            winner: self.winner(),
        }
    }
}

// ---------------------------------------------------------------- serde glue

impl Serialize for Pos {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("Pos", 2)?;
        state.serialize_field("row", &self.row)?;
        state.serialize_field("col", &self.col)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for Pos {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Pos, D::Error> {
        struct PosVisitor;

        impl<'de> Visitor<'de> for PosVisitor {
            type Value = Pos;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a position object with row/col (any letter case)")
            }

            fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> Result<Pos, M::Error> {
                let mut row = None;
                let mut col = None;
                while let Some(key) = map.next_key::<String>()? {
                    match key.to_ascii_lowercase().as_str() {
                        "row" => row = Some(map.next_value::<i32>()?),
                        "col" => col = Some(map.next_value::<i32>()?),
                        _ => {
                            map.next_value::<de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(Pos::new(
                    row.ok_or_else(|| de::Error::missing_field("row"))?,
                    col.ok_or_else(|| de::Error::missing_field("col"))?,
                ))
            }
        }

        deserializer.deserialize_map(PosVisitor)
    }
}

impl Serialize for Cell {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("Cell", 2)?;
        state.serialize_field("owner", &self.owner())?;
        state.serialize_field("kind", self.kind().name())?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for Cell {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Cell, D::Error> {
        struct CellVisitor;

        impl<'de> Visitor<'de> for CellVisitor {
            type Value = Cell;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a cell object with owner/kind (kind may be a name or a number)")
            }

            fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> Result<Cell, M::Error> {
                let mut owner: Option<u8> = None;
                let mut kind: Option<CellKind> = None;
                while let Some(key) = map.next_key::<String>()? {
                    match key.to_ascii_lowercase().as_str() {
                        "owner" => owner = Some(map.next_value::<u8>()?),
                        "kind" => kind = Some(map.next_value::<KindField>()?.0),
                        _ => {
                            map.next_value::<de::IgnoredAny>()?;
                        }
                    }
                }
                let owner = owner.unwrap_or(0);
                let kind = kind.unwrap_or(CellKind::Empty);
                if owner as usize > MAX_PLAYERS {
                    return Err(de::Error::custom(format!("owner {owner} out of range")));
                }
                Ok(Cell::new(owner, kind))
            }
        }

        deserializer.deserialize_map(CellVisitor)
    }
}

/// A [`CellKind`] written either as its upper-case name or as its numeric
/// discriminant.
struct KindField(CellKind);

impl<'de> Deserialize<'de> for KindField {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<KindField, D::Error> {
        struct KindVisitor;

        impl Visitor<'_> for KindVisitor {
            type Value = KindField;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a cell kind name or its numeric discriminant")
            }

            fn visit_str<E: de::Error>(self, value: &str) -> Result<KindField, E> {
                CellKind::parse(value)
                    .map(KindField)
                    .ok_or_else(|| E::custom(format!("unknown cell kind {value:?}")))
            }

            fn visit_u64<E: de::Error>(self, value: u64) -> Result<KindField, E> {
                u8::try_from(value)
                    .ok()
                    .and_then(CellKind::from_u8)
                    .map(KindField)
                    .ok_or_else(|| E::custom(format!("unknown cell kind {value}")))
            }

            fn visit_i64<E: de::Error>(self, value: i64) -> Result<KindField, E> {
                u8::try_from(value)
                    .ok()
                    .and_then(CellKind::from_u8)
                    .map(KindField)
                    .ok_or_else(|| E::custom(format!("unknown cell kind {value}")))
            }
        }

        deserializer.deserialize_any(KindVisitor)
    }
}
