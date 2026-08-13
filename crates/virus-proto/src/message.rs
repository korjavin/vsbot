//! The wire message catalog.
//!
//! Ported from the Go bot client (`virusgame/backend/cmd/bot-hoster/bot_client.go`)
//! and cross-checked against the server's own `Message` struct
//! (`virusgame/backend/types.go`) and the Java `GameLoopHandler`.
//!
//! # Tolerance
//!
//! [`Inbound`] is one flat struct with every field optional, exactly like the
//! server's own `Message`: the server marshals a single type with `omitempty`,
//! so an integer field that happens to be `0` is simply absent from the JSON.
//! Every field therefore carries `#[serde(default)]`, and unknown fields are
//! ignored so a server-side addition can never break a live game.
//!
//! Snapshots decode through [`virus_core::Snapshot`], which already accepts
//! both wire dialects (`{"Row":…}`/numeric kinds from the Go server,
//! `{"row":…}`/named kinds from the fixtures).

use serde::{Deserialize, Serialize};
use virus_core::{Action, Pos, Snapshot};

/// A board coordinate as the protocol writes it: `{"row":…,"col":…}`.
///
/// [`virus_core::Pos`] deserialises tolerantly but serialises to the same shape;
/// this mirror exists so `cells` round-trips as plain data without dragging the
/// engine's coordinate type into the wire layer.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub struct CellPos {
    /// Row, 0-based.
    pub row: i32,
    /// Column, 0-based.
    pub col: i32,
}

impl From<Pos> for CellPos {
    fn from(pos: Pos) -> CellPos {
        CellPos {
            row: pos.row,
            col: pos.col,
        }
    }
}

impl From<CellPos> for Pos {
    fn from(cell: CellPos) -> Pos {
        Pos::new(cell.row, cell.col)
    }
}

/// One entry of the lobby broadcast. Only the fields the bot reads are kept.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UserInfo {
    /// Server-assigned user id — the handle a `challenge` targets.
    #[serde(default)]
    pub user_id: String,
    /// Display name.
    #[serde(default)]
    pub username: String,
    /// Whether the peer is mid-game.
    #[serde(default)]
    pub in_game: bool,
    /// Whether the peer is sitting in a lobby.
    #[serde(default)]
    pub in_lobby: bool,
}

impl UserInfo {
    /// A peer that could accept a challenge right now.
    pub fn is_idle(&self) -> bool {
        !self.user_id.is_empty() && !self.in_game && !self.in_lobby
    }
}

/// Per-seat metadata attached to `multiplayer_game_start`.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GamePlayerInfo {
    /// 1-based seat.
    #[serde(default)]
    pub player_index: i32,
    /// Display name.
    #[serde(default)]
    pub username: String,
    /// Whether the seat is a bot.
    #[serde(default)]
    pub is_bot: bool,
    /// Server-reported activity. Not trusted for rules purposes — the snapshot
    /// decode recomputes it (ARCHITECTURE.md invariant 5).
    #[serde(default)]
    pub is_active: bool,
}

/// The lobby descriptor carried by `lobby_joined`.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LobbyInfo {
    /// Lobby id.
    #[serde(default)]
    pub lobby_id: String,
}

/// Every server message the bot understands, in one tolerant struct.
///
/// The variants the bot acts on are named by [`Inbound::kind`]; anything else
/// decodes fine and is ignored.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Inbound {
    /// Message discriminator.
    #[serde(default, rename = "type")]
    pub kind: String,
    /// Our server-assigned id (`welcome`).
    #[serde(default)]
    pub user_id: String,
    /// On `welcome`, our display name. On `error`, the server reuses this field
    /// for the human-readable reason — see `Hub.rejectAction`.
    #[serde(default)]
    pub username: String,
    /// Correlation id echoed back on acks, errors and `bot_wanted`.
    #[serde(default)]
    pub request_id: String,
    /// Lobby id (`bot_wanted`).
    #[serde(default)]
    pub lobby_id: String,
    /// Lobby descriptor (`lobby_joined`).
    #[serde(default)]
    pub lobby: Option<LobbyInfo>,
    /// Game id.
    #[serde(default)]
    pub game_id: String,
    /// Our seat, 1-based (`game_start`, `multiplayer_game_start`).
    #[serde(default)]
    pub your_player: i32,
    /// The seat a `move_made` / `neutrals_placed` / `turn_change` refers to.
    #[serde(default)]
    pub player: i32,
    /// Move row.
    #[serde(default)]
    pub row: Option<i32>,
    /// Move column.
    #[serde(default)]
    pub col: Option<i32>,
    /// The two cells of a `neutrals_placed`.
    #[serde(default)]
    pub cells: Vec<CellPos>,
    /// Actions left in the current turn, as the server counts them.
    #[serde(default)]
    pub moves_left: i32,
    /// Winning seat on `game_end`, else absent.
    #[serde(default)]
    pub winner: i32,
    /// Seat removed by `player_eliminated`.
    #[serde(default)]
    pub eliminated_player: i32,
    /// Challenge handle (`challenge_received`).
    #[serde(default)]
    pub challenge_id: String,
    /// Challenger's display name.
    #[serde(default)]
    pub from_username: String,
    /// Seat metadata (`multiplayer_game_start`).
    #[serde(default)]
    pub game_players: Vec<GamePlayerInfo>,
    /// Lobby population (`users_update`). Recorded, never acted on directly —
    /// the challenger's timer is the sole send driver.
    #[serde(default)]
    pub users: Vec<UserInfo>,
    /// The authoritative position. Present on every state-bearing message.
    #[serde(default)]
    pub snapshot: Option<Snapshot>,
}

/// Search diagnostics forwarded with an action. Display-only metadata: the
/// server relays them to spectators and never validates them.
#[derive(Clone, Copy, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Diagnostics {
    /// Root score in the mover's frame.
    pub score: f64,
    /// Completed search depth, or `0` where the engine has no depth analogue.
    pub depth: i32,
    /// Nodes or simulations spent.
    pub nodes_evaluated: i64,
    /// Wall-clock milliseconds spent choosing.
    pub time_ms: i64,
}

/// Every message the bot sends.
///
/// Action messages (`move`, `neutrals`) always carry a fresh UUID `requestId`.
/// The server keys idempotent replay on it (`Game.actionRequestReplay`): a
/// re-delivered identical action is acknowledged rather than re-applied, and a
/// reused id with different content is rejected — so ids are never recycled.
#[derive(Clone, Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Outgoing {
    /// Play a cell. The server infers grow-vs-attack from the board.
    #[serde(rename = "move", rename_all = "camelCase")]
    Move {
        /// Game the action belongs to.
        game_id: String,
        /// Fresh UUID.
        request_id: String,
        /// Target row.
        row: i32,
        /// Target column.
        col: i32,
        /// Search diagnostics.
        #[serde(flatten)]
        diagnostics: Diagnostics,
    },
    /// Convert two of our own `Normal` cells to `Neutral`. Consumes the turn.
    #[serde(rename = "neutrals", rename_all = "camelCase")]
    Neutrals {
        /// Game the action belongs to.
        game_id: String,
        /// Fresh UUID.
        request_id: String,
        /// Exactly two distinct cells.
        cells: Vec<CellPos>,
        /// Search diagnostics.
        #[serde(flatten)]
        diagnostics: Diagnostics,
    },
    /// Answer a `bot_wanted` broadcast.
    #[serde(rename = "join_lobby", rename_all = "camelCase")]
    JoinLobby {
        /// Lobby to join.
        lobby_id: String,
        /// The id echoed from `bot_wanted`.
        request_id: String,
    },
    /// Accept a 1v1 challenge.
    #[serde(rename = "accept_challenge", rename_all = "camelCase")]
    AcceptChallenge {
        /// Challenge handle.
        challenge_id: String,
    },
    /// Refuse a 1v1 challenge (we are busy).
    #[serde(rename = "decline_challenge", rename_all = "camelCase")]
    DeclineChallenge {
        /// Challenge handle.
        challenge_id: String,
    },
    /// Open a 1v1 challenge against a peer (challenger mode).
    #[serde(rename = "challenge", rename_all = "camelCase")]
    Challenge {
        /// Peer to challenge.
        target_user_id: String,
        /// Board height.
        rows: usize,
        /// Board width.
        cols: usize,
    },
    /// Ask the server to re-send the authoritative snapshot.
    ///
    /// Sent when a snapshot fails validation: rather than keep playing off a
    /// position we no longer trust, we drop it and wait for a clean one.
    #[serde(rename = "resync", rename_all = "camelCase")]
    Resync {
        /// Game to resynchronise.
        game_id: String,
    },
}

impl Outgoing {
    /// Builds the action message for a chosen [`Action`], mirroring Go's
    /// `actionMessage` and Java's `writeAction` — the single translation point
    /// from engine action to wire action.
    pub fn action(
        game_id: &str,
        request_id: &str,
        action: Action,
        diagnostics: Diagnostics,
    ) -> Outgoing {
        match action {
            Action::Move { target } => Outgoing::Move {
                game_id: game_id.to_owned(),
                request_id: request_id.to_owned(),
                row: target.row,
                col: target.col,
                diagnostics,
            },
            Action::PlaceNeutrals { cells } => Outgoing::Neutrals {
                game_id: game_id.to_owned(),
                request_id: request_id.to_owned(),
                cells: vec![cells[0].into(), cells[1].into()],
                diagnostics,
            },
        }
    }
}
