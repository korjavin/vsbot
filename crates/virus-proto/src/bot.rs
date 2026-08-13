//! The bot state machine: transport-independent, and the place every
//! non-negotiable invariant is enforced.
//!
//! # Why this type knows nothing about sockets
//!
//! [`Bot`] consumes decoded [`Inbound`] messages and produces [`Outbound`]
//! items on a channel. `crate::client` bolts a WebSocket onto both ends. That
//! split is what lets the invariants below be tested exhaustively without a
//! server, which is how the Go predecessor's `bot_search_test.go` caught the
//! staleness bugs in the first place.
//!
//! # The invariants, and where they live
//!
//! 1. **Never act on our own `neutrals_placed` ack.** Only four message types
//!    are allowed to start a search — see [`Driver`]. `neutrals_placed` is not
//!    one of them, for anybody: our own ack still shows us as mover with
//!    `movesLeft > 0` (two live forfeits, 2026-08-08), and an opponent's
//!    placement is immediately followed by the authoritative `turn_change`.
//! 2. **Snapshot-authoritative.** [`Bot::absorb`] is the only writer of
//!    [`BotCore::position`], every snapshot goes through
//!    [`virus_core::Snapshot::decode`], and nothing is ever reconstructed from
//!    a move delta. A snapshot that fails validation *drops* the position
//!    rather than leaving a stale one in place, and asks the server to resync.
//! 3. **Version-gated cancellation.** Each accepted snapshot bumps
//!    [`BotCore::position_version`] and cancels the in-flight search. A result
//!    is re-validated against `{game, version, seat, movesLeft, legality}`
//!    twice: in the search task before it queues, and in the writer before it
//!    hits the wire.
//! 4. **Act only when** `!game_over && current == me && moves_left > 0`.
//! 5. **The search never runs on the read loop** — it is a `spawn_blocking`
//!    task, so server pings are answered while thinking. The Java predecessor
//!    searched inline and starved its pong deadline.

use crate::clock::{MoveAllocation, TurnAllocator};
use crate::config::BotConfig;
use crate::engine::{SearchBudget, SearchEngine, SearchOutcome};
use crate::message::{Diagnostics, Inbound, Outgoing, UserInfo};
use crate::ponder::{PonderInbox, PonderStep};
use std::sync::mpsc as blocking_mpsc;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;
use virus_core::{Action, CellKind, Player, Pos, State};

/// Minimum gap between `resync` requests, so a burst of server errors cannot
/// turn into a request storm.
const RESYNC_COOLDOWN: Duration = Duration::from_secs(1);

/// How long the client waits for its pre-selected fallback before giving up on
/// it too. A `fallback` implementation is contractually one net forward.
const FALLBACK_SELECTION_TIMEOUT: Duration = Duration::from_millis(500);

/// Where the bot is in its lifecycle.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Phase {
    /// No live socket.
    Disconnected,
    /// Connected and available.
    Idle,
    /// Sitting in a lobby, waiting for the game to start.
    InLobby,
    /// Playing.
    InGame,
}

/// The message types allowed to start a search.
///
/// This is a whitelist, not a hint. `neutrals_placed`, `action_ack`, `error`
/// and `player_eliminated` all carry a snapshot that can show us as the mover
/// with actions left, and none of them means it is our turn.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Driver {
    /// `game_start` / `multiplayer_game_start` — seat 1 must open the game.
    GameStart,
    /// `game_state` — the server's mid-turn "you may act again" nudge.
    GameState,
    /// `turn_change` — the authoritative turn driver.
    TurnChange,
}

/// Everything an action must still be true about at send time.
#[derive(Clone, Debug)]
pub struct ActionGuard {
    /// Game the action belongs to.
    pub game_id: String,
    /// [`BotCore::position_version`] the search ran against.
    pub version: u64,
    /// Our seat.
    pub seat: Player,
    /// Actions remaining in the searched position.
    pub moves_left: u8,
    /// The action itself, re-checked for legality against the live position.
    pub action: Action,
}

/// An item queued for the socket writer.
#[derive(Debug)]
pub enum Outbound {
    /// A JSON frame. `guard`, when present, must still validate at write time.
    Text {
        /// Serialised JSON.
        data: String,
        /// Send-time revalidation for game actions.
        guard: Option<ActionGuard>,
    },
    /// A pong for a server ping, routed through the writer so it is flushed
    /// promptly even while a search is running.
    Pong(Vec<u8>),
}

/// Observability counters. Cheap, and the integration test asserts on them.
#[derive(Clone, Copy, Debug, Default)]
pub struct Counters {
    /// `error` messages received.
    pub errors: u64,
    /// The subset of those that were illegal-move forfeits. The server phrases
    /// them as `"Defeated by illegal move: …"` (`Hub.handleIllegalAction`);
    /// this counter must be zero for the lifetime of a deployment.
    pub illegal_moves: u64,
    /// Snapshots rejected by [`virus_core::Snapshot::decode`].
    pub rejected_snapshots: u64,
    /// Actions that reached the outbox.
    pub actions_sent: u64,
    /// Search results discarded because the position had moved on.
    pub stale_results_dropped: u64,
    /// `game_end` messages for our game.
    pub games_finished: u64,
    /// Challenges we initiated.
    pub challenges_sent: u64,
    /// Actions answered with the pre-selected fallback because the engine
    /// overran its ceiling. Must stay at zero in a healthy deployment; a
    /// non-zero value means the time manager is not holding.
    pub fallback_actions: u64,
    /// Ponder steps handed to a live session.
    pub ponder_steps: u64,
    /// Turns answered out of a pondering session's tree rather than a fresh
    /// search.
    pub ponder_answers: u64,
}

/// The mutable client state. Guarded by one mutex; no `await` is ever held
/// across it.
#[derive(Debug)]
pub struct BotCore {
    /// Lifecycle phase.
    pub phase: Phase,
    /// Server-assigned id, from `welcome`.
    pub user_id: String,
    /// Server-assigned display name.
    pub username: String,
    /// The game we are playing, if any.
    pub current_game: Option<String>,
    /// The lobby we are sitting in, if any.
    pub current_lobby: Option<String>,
    /// Our seat, 1-based. Meaningless outside [`Phase::InGame`].
    pub seat: Player,
    /// The authoritative position. `None` means "not trusted, do not act".
    pub position: Option<State>,
    /// Bumped by every accepted snapshot and every invalidation.
    pub position_version: u64,
    /// Last `users_update`. Read by the challenger timer only.
    pub peers: Vec<UserInfo>,
    /// Counters.
    pub counters: Counters,
    /// The most recent server `error` text, for diagnostics.
    pub last_error: Option<String>,

    /// Identity of the current position: Zobrist key plus terminal status. Used
    /// to suppress no-op snapshot re-sends without a deep structural compare.
    position_key: Option<(u64, bool, Player)>,
    /// The version a search has already been started for.
    searched_version: u64,
    /// The version a ponder step has already been queued for. The mirror of
    /// [`BotCore::searched_version`], and needed for the same reason: the
    /// server sends `move_made` and then `game_state` for one position, and a
    /// resync can repeat a frame verbatim. One snapshot, one step.
    pondered_version: u64,
    /// Cancels the in-flight search.
    cancel: Option<CancellationToken>,
    /// When the optimistic challenger busy-flag was set.
    pending_game_since: Option<Instant>,
    /// Last resync request, for the cooldown.
    last_resync: Option<Instant>,
    /// Splits the turn budget across the turn's actions.
    allocator: TurnAllocator,
    /// The live pondering session, if any.
    ponder: Option<PonderSlot>,
}

/// The client's end of a pondering session.
#[derive(Debug)]
struct PonderSlot {
    /// The game the session is thinking about. A session never outlives it.
    game_id: String,
    /// Steps go in here; dropping it ends the session.
    steps: blocking_mpsc::Sender<PonderStep>,
    /// Cancels the step currently in flight.
    cancel: CancellationToken,
}

impl Default for BotCore {
    fn default() -> BotCore {
        BotCore {
            phase: Phase::Disconnected,
            user_id: String::new(),
            username: String::new(),
            current_game: None,
            current_lobby: None,
            seat: 0,
            position: None,
            position_version: 0,
            peers: Vec::new(),
            counters: Counters::default(),
            last_error: None,
            position_key: None,
            searched_version: 0,
            pondered_version: 0,
            cancel: None,
            pending_game_since: None,
            last_resync: None,
            allocator: TurnAllocator::default(),
            ponder: None,
        }
    }
}

impl BotCore {
    /// Cancels the in-flight search, if any.
    fn cancel_search(&mut self) {
        if let Some(token) = self.cancel.take() {
            token.cancel();
        }
        // ARCHITECTURE.md invariant 5, extended to the pondering session: no
        // simulation may continue past a snapshot without a fresh, explicitly
        // re-tokened instruction from the client.
        if let Some(slot) = self.ponder.as_ref() {
            slot.cancel.cancel();
        }
    }

    /// Ends the pondering session outright. The engine's `next()` returns
    /// `None` as soon as the sender is dropped.
    fn stop_ponder(&mut self) {
        if let Some(slot) = self.ponder.take() {
            slot.cancel.cancel();
        }
    }

    /// Cancels the search and invalidates every in-flight result by moving the
    /// version on. Used whenever the position stops being meaningful.
    fn invalidate(&mut self) {
        self.cancel_search();
        self.position_version = self.position_version.wrapping_add(1);
    }

    /// Drops the position entirely: nothing may be played until a fresh
    /// authoritative snapshot arrives.
    fn drop_position(&mut self) {
        self.invalidate();
        self.position = None;
        self.position_key = None;
    }

    /// Installs a validated position. Returns `false` when the snapshot is
    /// identical to the one already held, in which case nothing changes.
    fn install(&mut self, state: State) -> bool {
        let key = (state.hash(), state.game_over(), state.winner());
        if self.position_key == Some(key) {
            return false;
        }
        self.invalidate();
        // The one boundary `movesLeft` cannot show on its own: a turn spent on
        // `PlaceNeutrals` ends at `movesLeft == 3`, so two of our turns in a row
        // can both open at 3. Any position that is not ours ends the turn.
        if state.game_over() || state.current_player() != self.seat {
            self.allocator.end_turn();
        }
        self.position_key = Some(key);
        self.position = Some(state);
        true
    }

    /// Returns to the idle pool.
    fn go_idle(&mut self) {
        self.invalidate();
        self.stop_ponder();
        self.phase = Phase::Idle;
        self.current_game = None;
        self.current_lobby = None;
        self.position = None;
        self.position_key = None;
        self.seat = 0;
        self.pending_game_since = None;
    }

    /// Turn discipline (ARCHITECTURE.md invariant 2): the four conditions that
    /// must hold before any action is even considered.
    pub fn may_act(&self) -> bool {
        self.phase == Phase::InGame
            && self.current_game.is_some()
            && self.position.as_ref().is_some_and(|position| {
                !position.game_over()
                    && position.current_player() == self.seat
                    && position.moves_left() > 0
            })
    }

    /// Send-time revalidation. Called in the search task *and* again in the
    /// writer, because between those two points a snapshot may have landed.
    pub fn action_still_valid(&self, guard: &ActionGuard) -> bool {
        if self.phase != Phase::InGame
            || self.current_game.as_deref() != Some(guard.game_id.as_str())
            || self.position_version != guard.version
            || self.seat != guard.seat
        {
            return false;
        }
        let Some(position) = self.position.as_ref() else {
            return false;
        };
        position.moves_left() == guard.moves_left
            && self.may_act()
            && action_is_legal(position, guard.action)
    }

    /// The current position, for tests and diagnostics.
    pub fn position(&self) -> Option<&State> {
        self.position.as_ref()
    }

    /// The version a search has already been dispatched for.
    pub fn searched_version(&self) -> u64 {
        self.searched_version
    }

    /// The intra-turn time allocator, for diagnostics and tests.
    pub fn allocator(&self) -> &TurnAllocator {
        &self.allocator
    }

    /// Whether a pondering session may be given this position.
    ///
    /// The mirror image of [`BotCore::may_act`], and deliberately *narrower*
    /// than "not our turn": we ponder only positions where somebody else is the
    /// mover. A position where we are the mover with no actions left is a
    /// transient the next snapshot resolves, and a game that is over is not
    /// worth a simulation.
    pub fn may_ponder(&self) -> bool {
        self.phase == Phase::InGame
            && self.current_game.is_some()
            && self.position.as_ref().is_some_and(|position| {
                !position.game_over() && position.current_player() != self.seat
            })
    }

    /// Whether a pondering session is live.
    pub fn is_pondering(&self) -> bool {
        self.ponder.is_some()
    }
}

/// Re-checks an action against a live position. The version gate already
/// implies the position is byte-identical to the searched one, so this can only
/// fail on an engine bug — which is exactly why it is checked: the server
/// forfeits the game for an illegal move.
fn action_is_legal(position: &State, action: Action) -> bool {
    match action {
        Action::Move { target } => position.legal_move(position.current_player(), target),
        Action::PlaceNeutrals { cells } => {
            position.can_place_neutrals()
                && cells[0] != cells[1]
                && cells.iter().all(|&cell| is_own_normal(position, cell))
        }
    }
}

fn is_own_normal(position: &State, pos: Pos) -> bool {
    position.at_checked(pos).is_some_and(|cell| {
        cell.owner() == position.current_player() && cell.kind() == CellKind::Normal
    })
}

/// The client state machine.
///
/// Cheap to clone — every clone shares one core, one outbox and one engine.
#[derive(Clone)]
pub struct Bot {
    core: Arc<Mutex<BotCore>>,
    outbox: mpsc::UnboundedSender<Outbound>,
    engine: Arc<dyn SearchEngine>,
    config: Arc<BotConfig>,
}

impl std::fmt::Debug for Bot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Bot")
            .field("engine", &self.engine.name())
            .field("core", &self.core)
            .finish_non_exhaustive()
    }
}

impl Bot {
    /// Builds a bot and the receiving end of its outbox.
    ///
    /// `crate::client` drives the receiver into a socket; tests read it
    /// directly.
    pub fn new(
        config: Arc<BotConfig>,
        engine: Arc<dyn SearchEngine>,
    ) -> (Bot, mpsc::UnboundedReceiver<Outbound>) {
        let (outbox, inbox) = mpsc::unbounded_channel();
        let core = BotCore {
            allocator: config.allocator(),
            ..BotCore::default()
        };
        let bot = Bot {
            core: Arc::new(Mutex::new(core)),
            outbox,
            engine,
            config,
        };
        (bot, inbox)
    }

    /// Locks the core. Never held across an `await`.
    pub fn core(&self) -> MutexGuard<'_, BotCore> {
        self.core
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
    }

    /// The configuration this bot was built with.
    pub fn config(&self) -> &BotConfig {
        &self.config
    }

    /// Parses and dispatches one raw server frame. Malformed JSON is logged and
    /// dropped: one bad frame must never take the bot down mid-game.
    pub fn handle_text(&self, raw: &str) {
        match serde_json::from_str::<Inbound>(raw) {
            Ok(message) => self.handle(message),
            Err(error) => self.log(&format!("undecodable frame ({error}): {raw:.240}")),
        }
    }

    /// Dispatches one decoded server message.
    pub fn handle(&self, message: Inbound) {
        self.heal_pending_game();
        match message.kind.as_str() {
            "welcome" => self.on_welcome(&message),
            "users_update" => self.core().peers = message.users,

            "bot_wanted" => self.on_bot_wanted(&message),
            "lobby_joined" => self.on_lobby_joined(&message),
            "lobby_closed" => self.on_lobby_closed(),
            "challenge_received" => self.on_challenge_received(&message),

            "game_start" | "multiplayer_game_start" => self.on_game_start(&message),

            // Drives a search: the server's mid-turn nudge and the
            // authoritative turn driver.
            "game_state" | "turn_change" => {
                let driver = if message.kind == "turn_change" {
                    Driver::TurnChange
                } else {
                    Driver::GameState
                };
                if self.absorb(&message) {
                    self.maybe_search(driver);
                    self.maybe_ponder();
                }
            }

            // Refresh only. `move_made` is echoed to every participant and is
            // always followed by `game_state` (mid-turn) or `turn_change` (turn
            // over); acting here would double up on that.
            //
            // It *is* the moment a pondering session re-roots, though: this is
            // the opponent's action arriving, and the child it leads to is
            // already in the tree.
            "move_made" | "player_eliminated" => {
                if self.absorb(&message) {
                    self.maybe_ponder();
                }
            }

            // ARCHITECTURE.md invariant 2. The snapshot on our OWN ack still
            // shows us as mover with movesLeft > 0 — acting on it forfeited two
            // live games on 2026-08-08. An opponent's placement ends their turn
            // server-side, so the `turn_change` that follows is what wakes us.
            // Either way: absorb, never search.
            //
            // `maybe_ponder` is safe to reach from here precisely because it can
            // only ever produce a `PonderStep`, never an action, and only for a
            // position somebody else is the mover in — which our own ack, still
            // showing us as mover, is not.
            "neutrals_placed" => {
                if self.absorb(&message) {
                    self.maybe_ponder();
                }
            }

            // Our own idempotent-replay acknowledgement. Same hazard class as
            // `neutrals_placed`: it is our message coming back, not a turn.
            "action_ack" => {
                if self.absorb(&message) {
                    self.maybe_ponder();
                }
            }

            "game_end" => self.on_game_end(&message),
            "error" => self.on_error(&message),

            _ => {}
        }
    }

    /// Called by the transport when the socket dies. Everything in flight is
    /// invalidated; nothing queued before the drop may be written after it.
    pub fn on_disconnected(&self) {
        let mut core = self.core();
        core.invalidate();
        core.stop_ponder();
        core.phase = Phase::Disconnected;
        core.position = None;
        core.position_key = None;
        core.current_game = None;
        core.current_lobby = None;
        core.user_id.clear();
        core.peers.clear();
        core.pending_game_since = None;
    }

    // ------------------------------------------------------------- handlers

    fn on_welcome(&self, message: &Inbound) {
        let mut core = self.core();
        core.user_id = message.user_id.clone();
        core.username = message.username.clone();
        core.phase = Phase::Idle;
        let name = core.username.clone();
        let id = core.user_id.clone();
        drop(core);
        self.log(&format!("registered as {name} ({id})"));
    }

    fn on_bot_wanted(&self, message: &Inbound) {
        if self.core().phase != Phase::Idle {
            return;
        }
        self.send(Outgoing::JoinLobby {
            lobby_id: message.lobby_id.clone(),
            request_id: message.request_id.clone(),
        });
    }

    fn on_lobby_joined(&self, message: &Inbound) {
        let lobby = message
            .lobby
            .as_ref()
            .map(|lobby| lobby.lobby_id.clone())
            .unwrap_or_else(|| message.lobby_id.clone());
        let mut core = self.core();
        core.phase = Phase::InLobby;
        core.current_lobby = Some(lobby);
    }

    fn on_lobby_closed(&self) {
        let mut core = self.core();
        core.current_lobby = None;
        // A lobby closing never ends a game in progress — the Go client reset
        // to idle unconditionally here, which would have abandoned a live game
        // had the two ever overlapped.
        if core.phase != Phase::InGame {
            core.go_idle();
        }
    }

    fn on_challenge_received(&self, message: &Inbound) {
        let mut core = self.core();
        if core.phase != Phase::Idle {
            drop(core);
            self.send(Outgoing::DeclineChallenge {
                challenge_id: message.challenge_id.clone(),
            });
            return;
        }
        // Flip busy *before* accepting, so a second challenge arriving in the
        // window between our accept and the server's `game_start` is declined
        // rather than double-accepted.
        //
        // This matters for every bot, not just self-sparring ones: the server's
        // `handleAcceptChallenge` has no in-game guard, so a second accept
        // creates a *second* game and overwrites our `gameId` — abandoning the
        // first game to time out. The Go client only armed this in challenger
        // mode and carried the hole everywhere else.
        //
        // The flag is only optimistic; `heal_pending_game` returns us to the
        // pool if the game never materialises.
        core.phase = Phase::InGame;
        core.pending_game_since = Some(Instant::now());
        drop(core);
        self.send(Outgoing::AcceptChallenge {
            challenge_id: message.challenge_id.clone(),
        });
    }

    fn on_game_start(&self, message: &Inbound) {
        let Some(snapshot) = message.snapshot.as_ref() else {
            self.log("game start without a snapshot — ignored");
            return;
        };
        let state = match snapshot.decode() {
            Ok(state) => state,
            Err(error) => {
                self.core().counters.rejected_snapshots += 1;
                self.log(&format!("rejected game-start snapshot: {error}"));
                return;
            }
        };
        let players = state.players();
        if message.your_player < 1 || message.your_player as usize > players {
            self.log(&format!(
                "yourPlayer {} outside 1..={players} — ignoring game start",
                message.your_player
            ));
            return;
        }
        if message.kind == "multiplayer_game_start" && !seats_are_sane(message, players) {
            self.log("multiplayer_game_start carried inconsistent gamePlayers — ignored");
            return;
        }
        if message.game_id.is_empty() {
            self.log("game start without a gameId — ignored");
            return;
        }

        {
            let mut core = self.core();
            core.cancel_search();
            // A session belongs to one game: its tree is that game's positions.
            core.stop_ponder();
            core.phase = Phase::InGame;
            core.current_game = Some(message.game_id.clone());
            core.seat = message.your_player as Player;
            core.searched_version = 0;
            core.pondered_version = 0;
            core.pending_game_since = None;
            core.allocator = self.config.allocator();
            // Clearing the key forces the install even when a rematch happens
            // to open on a position identical to the last game's opening.
            core.position_key = None;
            core.install(state);
        }
        self.log(&format!(
            "game {} started as seat {}",
            message.game_id, message.your_player
        ));
        self.maybe_search(Driver::GameStart);
        self.maybe_ponder();
    }

    fn on_game_end(&self, message: &Inbound) {
        let mut core = self.core();
        // A `game_end` for a game we already left must not disturb the new one.
        if !message.game_id.is_empty() && core.current_game.as_deref() != Some(&message.game_id) {
            return;
        }
        core.counters.games_finished += 1;
        core.go_idle();
        drop(core);
        self.log(&format!("game ended, winner seat {}", message.winner));
    }

    fn on_error(&self, message: &Inbound) {
        {
            let mut core = self.core();
            core.counters.errors += 1;
            // `Hub.handleIllegalAction` phrases a forfeit this way. It is the
            // one server error that means the game is already lost.
            if message.username.starts_with("Defeated by illegal move") {
                core.counters.illegal_moves += 1;
            }
            core.last_error = Some(message.username.clone());
        }
        self.log(&format!("server error: {}", message.username));
        // The position we hold may be exactly the one the server disagreed
        // with. Drop it and ask for the authoritative one instead of guessing.
        if !message.game_id.is_empty()
            && self.core().current_game.as_deref() == Some(&message.game_id)
        {
            self.core().drop_position();
            self.request_resync(&message.game_id);
        }
    }

    // -------------------------------------------------------------- helpers

    /// Validates and installs the snapshot carried by `message`.
    ///
    /// Returns `true` when the position is trustworthy afterwards — the *only*
    /// condition under which a caller may go on to start a search.
    fn absorb(&self, message: &Inbound) -> bool {
        let Some(snapshot) = message.snapshot.as_ref() else {
            self.log(&format!("{} carried no snapshot — ignored", message.kind));
            return false;
        };
        {
            let core = self.core();
            if core.phase != Phase::InGame {
                return false;
            }
            match (core.current_game.as_deref(), message.game_id.as_str()) {
                (Some(current), incoming) if !incoming.is_empty() && incoming != current => {
                    return false;
                }
                (None, _) => return false,
                _ => {}
            }
        }
        match snapshot.decode() {
            Ok(state) => {
                let mut core = self.core();
                core.install(state);
                true
            }
            Err(error) => {
                let game_id = {
                    let mut core = self.core();
                    core.counters.rejected_snapshots += 1;
                    core.drop_position();
                    core.current_game.clone()
                };
                self.log(&format!(
                    "rejected {} snapshot: {error} — dropping the position and resyncing",
                    message.kind
                ));
                if let Some(game_id) = game_id {
                    self.request_resync(&game_id);
                }
                false
            }
        }
    }

    fn request_resync(&self, game_id: &str) {
        {
            let mut core = self.core();
            let now = Instant::now();
            if core
                .last_resync
                .is_some_and(|last| now.duration_since(last) < RESYNC_COOLDOWN)
            {
                return;
            }
            core.last_resync = Some(now);
        }
        self.send(Outgoing::Resync {
            game_id: game_id.to_owned(),
        });
    }

    /// Undoes the optimistic busy flag if the accepted game never arrived.
    ///
    /// Without this, one accept the server dropped on the floor (admission
    /// refused, challenger disconnected) would wedge the bot out of the pool
    /// for the life of the process.
    fn heal_pending_game(&self) {
        let grace = self.config.pending_game_grace;
        let mut core = self.core();
        let stuck = core.phase == Phase::InGame
            && core.current_game.is_none()
            && core
                .pending_game_since
                .is_some_and(|since| since.elapsed() > grace);
        if stuck {
            core.go_idle();
        }
    }

    /// Starts a search if — and only if — turn discipline allows it and no
    /// search has already been dispatched for this snapshot.
    ///
    /// The `searched_version` gate is the double-send guard: the server sends
    /// `move_made` and then `game_state` for the same mid-turn position, and a
    /// resync can repeat a `game_state` verbatim. One snapshot, one action.
    fn maybe_search(&self, driver: Driver) {
        let dispatch = {
            let mut core = self.core();
            if !core.may_act() || core.searched_version == core.position_version {
                return;
            }
            let (Some(game_id), Some(position)) =
                (core.current_game.clone(), core.position.clone())
            else {
                return;
            };
            core.cancel_search();
            let cancel = CancellationToken::new();
            core.cancel = Some(cancel.clone());
            core.searched_version = core.position_version;
            // The allocator is the only thing that decides how long an action
            // may take. `movesLeft` is what tells it which action of the turn
            // this is.
            let allocation = core.allocator.allocate(position.moves_left());
            Dispatch {
                position,
                game_id,
                version: core.position_version,
                seat: core.seat,
                cancel,
                allocation,
            }
        };
        let bot = self.clone();
        tokio::spawn(async move { bot.run_search(dispatch, driver).await });
    }

    /// Starts, or feeds, the pondering session.
    ///
    /// **This function cannot emit an action and must never learn how to.** It
    /// is reachable from message types the turn-driver whitelist deliberately
    /// excludes (`move_made`, `neutrals_placed`, `action_ack`), and the only
    /// reason that is safe is that a [`PonderStep::Think`] has no reply channel.
    fn maybe_ponder(&self) {
        if !self.config.ponder || !self.engine.can_ponder() {
            return;
        }
        let step = {
            let mut core = self.core();
            if !core.may_ponder() || core.pondered_version == core.position_version {
                return;
            }
            let (Some(game_id), Some(position)) =
                (core.current_game.clone(), core.position.clone())
            else {
                return;
            };
            self.ensure_ponder_session(&mut core, &game_id);
            let cancel = CancellationToken::new();
            core.pondered_version = core.position_version;
            let Some(slot) = core.ponder.as_mut() else {
                return;
            };
            slot.cancel = cancel.clone();
            core.counters.ponder_steps += 1;
            PonderStep::Think {
                state: position,
                budget: SearchBudget::new(Instant::now() + self.config.ponder_budget, cancel),
            }
        };
        self.send_ponder_step(step);
    }

    /// Ensures a session exists for `game_id`, spawning one if needed.
    fn ensure_ponder_session(&self, core: &mut BotCore, game_id: &str) {
        if core
            .ponder
            .as_ref()
            .is_some_and(|slot| slot.game_id == game_id)
        {
            return;
        }
        core.stop_ponder();
        let (steps, inbox) = PonderInbox::channel();
        let engine = Arc::clone(&self.engine);
        // A blocking worker, exactly like the search: the read loop must stay
        // free to answer the server's pings while the session thinks.
        tokio::task::spawn_blocking(move || engine.ponder(&inbox));
        core.ponder = Some(PonderSlot {
            game_id: game_id.to_owned(),
            steps,
            cancel: CancellationToken::new(),
        });
    }

    /// Queues a step for the session, tearing it down if it has gone away.
    ///
    /// Non-blocking: `std::sync::mpsc` is unbounded, so this never parks the
    /// read loop no matter how far behind the session is.
    fn send_ponder_step(&self, step: PonderStep) {
        let mut core = self.core();
        let Some(slot) = core.ponder.as_ref() else {
            return;
        };
        if slot.steps.send(step).is_err() {
            core.ponder = None;
            drop(core);
            self.log("the pondering session ended; continuing without it");
        }
    }

    /// The search task: off the read loop, cancellable, and revalidated before
    /// anything is queued.
    async fn run_search(self, dispatch: Dispatch, driver: Driver) {
        let started = Instant::now();
        let budget = SearchBudget::allocated(
            started,
            dispatch.allocation,
            self.config.stop_policy,
            dispatch.cancel.clone(),
        );
        let moves_left = dispatch.position.moves_left();

        // Fallback-first (superiority.md §2b): a legal answer is in hand
        // *before* the long search starts, so a search that overruns costs a
        // weaker move rather than a forfeit.
        let fallback = self.select_fallback(&dispatch.position).await;

        let hard_wait = dispatch.allocation.ceiling + self.config.fallback_grace;
        let searched = self.run_engine(&dispatch, &budget, hard_wait).await;
        self.core().allocator.spent(started.elapsed());

        let (outcome, from_fallback) = match searched {
            EngineAnswer::Chose(outcome) => (outcome, false),
            EngineAnswer::Overran => {
                // Stop the runaway search: cancelling the token does not move
                // the version, so the fallback still passes the guard below.
                dispatch.cancel.cancel();
                let Some(action) = fallback else {
                    self.log(&format!(
                        "{driver:?}: the engine overran {hard_wait:?} and no fallback was \
                         available — not moving"
                    ));
                    return;
                };
                self.core().counters.fallback_actions += 1;
                self.log(&format!(
                    "{driver:?}: the engine overran {hard_wait:?} — PLAYING THE PRE-SELECTED \
                     FALLBACK. The time manager is not holding; this must not happen in a \
                     healthy deployment."
                ));
                (SearchOutcome::new(action), true)
            }
            EngineAnswer::Nothing => {
                if dispatch.cancel.is_cancelled() {
                    // The documented `None` case: superseded before a candidate
                    // was established. The next snapshot drives.
                    return;
                }
                let Some(action) = fallback else {
                    self.log(&format!("{driver:?}: engine offered no action"));
                    return;
                };
                self.core().counters.fallback_actions += 1;
                self.log(&format!(
                    "{driver:?}: the engine offered no action in a position that has legal \
                     moves — PLAYING THE PRE-SELECTED FALLBACK."
                ));
                (SearchOutcome::new(action), true)
            }
            EngineAnswer::Failed => return,
        };

        let guard = ActionGuard {
            game_id: dispatch.game_id.clone(),
            version: dispatch.version,
            seat: dispatch.seat,
            moves_left,
            action: outcome.action,
        };
        {
            let mut core = self.core();
            if !core.action_still_valid(&guard) {
                core.counters.stale_results_dropped += 1;
                drop(core);
                self.log("dropped a stale search result");
                return;
            }
            core.counters.actions_sent += 1;
        }

        let diagnostics = Diagnostics {
            score: outcome.score,
            depth: outcome.depth,
            // A fallback move is not a searched move, and the spectator view
            // should not claim it is.
            nodes_evaluated: if from_fallback { 0 } else { outcome.nodes },
            time_ms: started.elapsed().as_millis() as i64,
        };
        // A fresh UUID per action: the server keys idempotent replay on it, and
        // reusing one with different content is a hard rejection.
        let request_id = Uuid::new_v4().to_string();
        let message = Outgoing::action(&dispatch.game_id, &request_id, outcome.action, diagnostics);
        self.enqueue(message, Some(guard));
    }

    /// Asks the engine for its overrun answer, before any long search runs.
    ///
    /// Bounded, because a `fallback` that blocked would defeat the whole point
    /// of having one. The first legal action is the backstop for the backstop.
    async fn select_fallback(&self, position: &State) -> Option<Action> {
        let engine = Arc::clone(&self.engine);
        let state = position.clone();
        let chosen = tokio::time::timeout(
            FALLBACK_SELECTION_TIMEOUT,
            tokio::task::spawn_blocking(move || engine.fallback(&state)),
        )
        .await;
        match chosen {
            Ok(Ok(Some(action))) => Some(action),
            Ok(Ok(None)) => None,
            _ => {
                self.log(
                    "the engine's fallback selection did not answer in time; using the first \
                     legal action",
                );
                position.legal_actions().first().copied()
            }
        }
    }

    /// Runs the search — through the pondering session when one is live for
    /// this game, otherwise as a fresh blocking search.
    async fn run_engine(
        &self,
        dispatch: &Dispatch,
        budget: &SearchBudget,
        hard_wait: Duration,
    ) -> EngineAnswer {
        if let Some(reply) = self.hand_to_ponder(dispatch, budget) {
            return match tokio::time::timeout(hard_wait, reply).await {
                Ok(Ok(Some(outcome))) => EngineAnswer::Chose(outcome),
                Ok(Ok(None)) => EngineAnswer::Nothing,
                // The session died mid-answer. Not fatal: the fallback covers
                // it, and the next turn opens a fresh session.
                Ok(Err(_)) => {
                    self.core().ponder = None;
                    EngineAnswer::Nothing
                }
                Err(_) => EngineAnswer::Overran,
            };
        }

        let engine = Arc::clone(&self.engine);
        let position = dispatch.position.clone();
        let budget = budget.clone();
        let joined = tokio::task::spawn_blocking(move || engine.choose(&position, &budget));
        match tokio::time::timeout(hard_wait, joined).await {
            Ok(Ok(Some(outcome))) => EngineAnswer::Chose(outcome),
            Ok(Ok(None)) => EngineAnswer::Nothing,
            Ok(Err(error)) => {
                self.log(&format!("search task failed: {error}"));
                EngineAnswer::Failed
            }
            Err(_) => EngineAnswer::Overran,
        }
    }

    /// Hands this turn to the live pondering session, so the tree it built
    /// during the opponent's turn is continued rather than thrown away.
    ///
    /// **The only place a `PonderStep::Answer` is ever created.** Its caller is
    /// [`Bot::run_search`], which only ever runs off a [`Driver`] message with
    /// [`BotCore::may_act`] true — that is the whole of ARCHITECTURE.md
    /// invariant 2 as it applies to pondering.
    fn hand_to_ponder(
        &self,
        dispatch: &Dispatch,
        budget: &SearchBudget,
    ) -> Option<tokio::sync::oneshot::Receiver<Option<SearchOutcome>>> {
        let mut core = self.core();
        let slot = core.ponder.as_ref()?;
        if slot.game_id != dispatch.game_id {
            return None;
        }
        // Belt and braces around the invariant this whole path turns on. The
        // caller has already checked it; checking again here costs nothing and
        // makes the guarantee local to the function that could break it.
        if !core.may_act() || core.position_version != dispatch.version {
            return None;
        }
        let (reply, receiver) = tokio::sync::oneshot::channel();
        let step = PonderStep::Answer {
            state: dispatch.position.clone(),
            budget: budget.clone(),
            reply,
        };
        let slot = core.ponder.as_mut()?;
        slot.cancel = budget.cancel.clone();
        if slot.steps.send(step).is_err() {
            core.ponder = None;
            return None;
        }
        core.counters.ponder_answers += 1;
        Some(receiver)
    }

    /// Queues a non-game message (no send-time guard).
    pub(crate) fn send(&self, message: Outgoing) {
        self.enqueue(message, None);
    }

    fn enqueue(&self, message: Outgoing, guard: Option<ActionGuard>) {
        match serde_json::to_string(&message) {
            Ok(data) => {
                let _ = self.outbox.send(Outbound::Text { data, guard });
            }
            Err(error) => self.log(&format!("failed to serialise {message:?}: {error}")),
        }
    }

    /// Queues a pong. Routed through the writer so it is flushed even while a
    /// search occupies a blocking worker.
    pub(crate) fn pong(&self, payload: Vec<u8>) {
        let _ = self.outbox.send(Outbound::Pong(payload));
    }

    /// Challenger tick: the sole driver of outbound challenges.
    ///
    /// Reactive `users_update` handling is what spammed the lobby in the Java
    /// predecessor, so the user list is only ever *read* here — never acted on
    /// as it arrives.
    pub(crate) fn challenge_tick(&self, rng: &mut crate::config::Rng) {
        // A challenge we accepted but never saw start would otherwise keep us
        // out of the pool forever; the timer is the one thing guaranteed to
        // keep running when no messages arrive.
        self.heal_pending_game();
        let target = {
            let core = self.core();
            if core.phase != Phase::Idle || core.user_id.is_empty() {
                return;
            }
            let candidates: Vec<&str> = core
                .peers
                .iter()
                .filter(|peer| peer.is_idle() && peer.user_id != core.user_id)
                .map(|peer| peer.user_id.as_str())
                .collect();
            let index = rng.below(candidates.len());
            match index {
                Some(index) => candidates[index].to_owned(),
                None => return,
            }
        };
        self.core().counters.challenges_sent += 1;
        self.log(&format!("challenging {target}"));
        self.send(Outgoing::Challenge {
            target_user_id: target,
            rows: self.config.challenge_rows,
            cols: self.config.challenge_cols,
        });
    }

    pub(crate) fn log(&self, line: &str) {
        let name = {
            let core = self.core();
            if core.username.is_empty() {
                "vsbot".to_owned()
            } else {
                core.username.clone()
            }
        };
        eprintln!("[{name}] {line}");
    }
}

struct Dispatch {
    position: State,
    game_id: String,
    version: u64,
    seat: Player,
    cancel: CancellationToken,
    allocation: MoveAllocation,
}

/// What came back from the engine for one action.
#[derive(Debug)]
enum EngineAnswer {
    /// A searched action.
    Chose(SearchOutcome),
    /// The engine declined: a terminal root, or cancellation before a candidate
    /// existed.
    Nothing,
    /// The engine blew through its ceiling plus grace. The fallback answers.
    Overran,
    /// The blocking task itself failed (panic). Nothing to send.
    Failed,
}

/// Mirrors Go's `decodeSnapshot` seat validation for `multiplayer_game_start`.
fn seats_are_sane(message: &Inbound, players: usize) -> bool {
    if message.game_players.len() != players {
        return false;
    }
    let mut seen = vec![false; players];
    for player in &message.game_players {
        let index = player.player_index;
        if index < 1 || index as usize > players || seen[index as usize - 1] {
            return false;
        }
        seen[index as usize - 1] = true;
    }
    true
}
