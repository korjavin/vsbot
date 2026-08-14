//! The `PlaceNeutrals` regression probe: a fixed position set, and what a net
//! artifact says about it.
//!
//! # What this is for
//!
//! bd `vsbot-07x` records a known gen-5 weakness: at certain positions the
//! champion genuinely *prefers* an unmotivated `PlaceNeutrals` — cold search,
//! warm search and a 10 000-simulation reference all pick it, so it is the
//! net's judgement rather than a search bug, and the owner independently
//! flagged it in live play. A weakness nobody can measure cannot be trained
//! away, so this module turns it into a number: a committed set of
//! neutral-decision positions ([`ProbeRecord`]) and a per-generation report
//! ([`ProbeReport`]) of what a net does with them.
//!
//! # This is informational, never a gate
//!
//! ARCHITECTURE.md invariant 7 — seven separate offline metrics were each
//! believed to predict playing strength and each one was wrong. A probe run is
//! a *diagnostic*: it says what the net's policy and value do at a class of
//! positions, and nothing at all about whether one net is stronger than
//! another. Only a >=400-game gauntlet says that. [`INFORMATIONAL`] is printed
//! on every run so the number can never be quoted without the caveat.
//!
//! # The three sources
//!
//! [`ProbeSource`] records where each position came from, because they are
//! evidence of different things:
//!
//! * [`ProbeSource::GamesDb`] — mined from the published prod `games.db` by
//!   [`mine_games`]: turns where a player really did place neutrals and then
//!   lost material advantage. Real games, real opponents, a *behavioural*
//!   label that owes nothing to our net.
//! * [`ProbeSource::PonderRepro`] — positions the champion itself chose
//!   `PlaceNeutrals` at, produced by [`mine_self_play`] on the trajectory
//!   generator `vsbot/examples/ponderrepro` uses. These are the bead's repro
//!   material: the net's own judgement, caught in the act.
//! * [`ProbeSource::LiveOwnerGame`] — the owner's live game. See
//!   `docs/probes.md` for why v1 carries none of these.
//!
//! # The mining heuristic (`GamesDb`), stated exactly
//!
//! A game is replayed turn by turn through `virus-core`'s real rules (the same
//! reconstruction Java's `GamesDbReplay` performs: the board is rebuilt at each
//! recorded turn with `moves_left = 3` and the threaded per-seat neutral
//! budget, and every recorded action goes through the legality-checked
//! [`State::apply`]). For each turn whose *first* recorded action is a
//! `PlaceNeutrals`, with mover `m` and opponent `o`:
//!
//! ```text
//! advantage(s) = s.owned_cells(m) - s.owned_cells(o)
//! before       = advantage(position at that turn's start)
//! after        = advantage(position at the start of m's h-th later turn)
//! swing        = after - before
//! ```
//!
//! `h` is [`MineConfig::horizon`] turns of the *mover's own* turns, clamped to
//! however many the game actually had; a turn with fewer than
//! [`MIN_HORIZON_TURNS`] of follow-up is dropped, because there is then nothing
//! to measure a swing against.
//!
//! # Why the swing alone is not the label
//!
//! `swing` on its own does **not** discriminate, and the first cut of this
//! miner proved it: every one of the 149 labellable neutral placements in the
//! 2026-08-09 corpus came out "lost advantage", none came out "kept". The
//! reason is mechanical rather than strategic. `PlaceNeutrals` converts two of
//! your own cells to dead space *and* consumes the whole turn, so the mover
//! forfeits three placements and hands the opponent an uncontested turn: the
//! immediate cost is around five cells before any judgement enters. A label
//! that fires on a near-deterministic consequence of the action's own rules
//! labels the rules, not the decision.
//!
//! So [`Labels::immediate_cost`] records that mechanical price separately —
//! `advantage(T+1) - advantage(T)`, the first turn's worth — and the class
//! leans on the one signal in a recorded game that is *not* implied by the
//! action: who won.
//!
//! ```text
//! LostAdvantage : swing <= -min_swing  AND the placer went on to lose
//! KeptAdvantage : the placer went on to win
//! (dropped)     : anything else
//! ```
//!
//! In the 2026-08-09 corpus the placer lost 338 of 431 recorded neutral
//! placements (78%), so both classes are populated and the control group is
//! real.
//!
//! The heuristic's limits are the point of naming it: this is
//! *correlational*, over a corpus of games between bots of varying strength.
//! A neutral placement that walls off a losing race is a deliberate sacrifice
//! and will be labelled `LostAdvantage` here; a bad neutral placement by a
//! player who was winning by miles anyway will be labelled `KeptAdvantage`.
//! That is why the probe reports numbers rather than verdicts, and why the
//! `KeptAdvantage` positions are kept in the set at all — they are the control
//! group. A net that puts the same policy mass on neutrals in both classes has
//! learned nothing about when a neutral is good.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use serde::{Deserialize, Serialize};
use virus_core::{Action, Cell, CellKind, Player, Pos, Snapshot, State};
use virus_mcts::{Config, Encoded, MctsSearcher, PolicyValueNet, ValueSource};

use crate::rng::{mix64, Rng};

/// The line every probe run prints, so a number from it can never be quoted as
/// a strength claim. See ARCHITECTURE.md invariant 7.
pub const INFORMATIONAL: &str = "INFORMATIONAL ONLY — per ARCHITECTURE.md invariant 7 this probe \
     is a diagnostic and NEVER a gate. It measures what a net says about one \
     position class; it says nothing about playing strength. Strength claims \
     come only from >=400-game gauntlets with colour-paired seeds.";

/// Board edge the net, and therefore this probe, is defined on.
pub const BOARD: usize = 12;

/// Fewest follow-up turns of the mover's own a mined position must have before
/// its material swing means anything.
///
/// One is enough because the first follow-up turn is what
/// [`Labels::immediate_cost`] is measured over; 61% of the corpus's neutral
/// placements have no second one, and dropping them would throw away the very
/// end-of-game cluster the class is about.
pub const MIN_HORIZON_TURNS: u32 = 1;

// ------------------------------------------------------------------ records

/// Where a probe position came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ProbeSource {
    /// Mined from the published prod `games.db` by [`mine_games`].
    GamesDb,
    /// Produced by [`mine_self_play`] on the `ponderrepro` trajectory
    /// generator: a position the champion itself answered with a neutral.
    PonderRepro,
    /// A position taken from a live game the owner played or watched — in
    /// practice a human seat against a `SuperiorBot` seat, mined by
    /// [`mine_games`] out of a dump narrowed to those games.
    LiveOwnerGame,
}

impl ProbeSource {
    /// The `id` prefix records from this source carry.
    ///
    /// Ids are the handle everything else uses — the report table, the
    /// per-position JSONL, a bead quoting one position — so the source has to be
    /// legible in the id itself. Two positions from the same game id under two
    /// different sources would otherwise be indistinguishable in a table.
    pub fn id_prefix(self) -> &'static str {
        match self {
            ProbeSource::GamesDb => "gamesdb",
            ProbeSource::PonderRepro => "selfplay",
            ProbeSource::LiveOwnerGame => "live",
        }
    }

    /// Parses the wire spelling, which is the kebab-case `serde` form.
    pub fn parse(text: &str) -> Result<ProbeSource, ProbeError> {
        match text {
            "games-db" => Ok(ProbeSource::GamesDb),
            "ponder-repro" => Ok(ProbeSource::PonderRepro),
            "live-owner-game" => Ok(ProbeSource::LiveOwnerGame),
            other => Err(ProbeError(format!(
                "unknown probe source {other:?}; expected games-db, ponder-repro or \
                 live-owner-game"
            ))),
        }
    }
}

impl fmt::Display for ProbeSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            ProbeSource::GamesDb => "games-db",
            ProbeSource::PonderRepro => "ponder-repro",
            ProbeSource::LiveOwnerGame => "live-owner-game",
        })
    }
}

/// What the mining heuristic said about the neutral placement at a position.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ProbeClass {
    /// The mover's material advantage fell by at least `min_swing` over the
    /// horizon *and* the mover went on to lose the game. The suspect class.
    LostAdvantage,
    /// The mover went on to win the game. The control class — a neutral
    /// placement that (correlationally) did not cost its player the game.
    KeptAdvantage,
    /// The champion chose `PlaceNeutrals` here in self-play. No material label:
    /// this class is about the net's judgement, not a game's outcome.
    ChampionChoseNeutral,
}

impl fmt::Display for ProbeClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            ProbeClass::LostAdvantage => "lost-advantage",
            ProbeClass::KeptAdvantage => "kept-advantage",
            ProbeClass::ChampionChoseNeutral => "champion-chose-neutral",
        })
    }
}

/// Where one probe position came from, in enough detail to find it again.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Provenance {
    /// Human-readable source of truth, including the snapshot's "as of".
    pub origin: String,
    /// `games.db` game id, for a mined position.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub game_id: Option<String>,
    /// The game's recorded start time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub started_at: Option<String>,
    /// Seat names, seat order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub players: Vec<String>,
    /// Recorded turn number the position starts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn: Option<u32>,
    /// Seed that reproduces a self-play position.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// The neutral pair actually played here, `[[r,c],[r,c]]`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub played_neutrals: Option<[[i32; 2]; 2]>,
    /// Anything a reader needs that the fields above do not carry.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub note: String,
}

/// The mining heuristic's measurements for one position.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Labels {
    /// Seat to move at the probe position.
    pub mover: Player,
    /// The mover's `Normal` cells — the source list for neutral pairs, so
    /// `legalPairs == ownNormals * (ownNormals - 1) / 2`.
    pub own_normals: usize,
    /// Legal `Move` actions at the position.
    pub legal_moves: usize,
    /// Legal `PlaceNeutrals` actions at the position.
    pub legal_pairs: usize,
    /// `owned_cells(mover) - owned_cells(opponent)` at the position.
    pub advantage_before: i64,
    /// The same, `horizonTurns` of the mover's own turns later.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub advantage_after: Option<i64>,
    /// `advantageAfter - advantageBefore`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub advantage_swing: Option<i64>,
    /// The mechanical price of the action itself: the advantage change over the
    /// mover's *first* follow-up turn. `PlaceNeutrals` gives up two cells and
    /// the turn's three placements, so this is near-deterministic and is
    /// deliberately kept out of the class label.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub immediate_cost: Option<i64>,
    /// The same advantage at the last replayed position of the game.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub advantage_final: Option<i64>,
    /// How many of the mover's own turns the swing was measured over.
    #[serde(default)]
    pub horizon_turns: u32,
    /// Whether the seat that placed the neutrals went on to win the recorded
    /// game. `None` for a position with no recorded outcome.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub placer_won: Option<bool>,
}

/// One position in the probe set.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProbeRecord {
    /// Stable identifier, unique within the set.
    pub id: String,
    /// Where the position came from.
    pub source: ProbeSource,
    /// What the heuristic said about it.
    pub class: ProbeClass,
    /// The position itself, in the wire form `virus-core` decodes.
    pub snapshot: Snapshot,
    /// Where to find it again.
    pub provenance: Provenance,
    /// The heuristic's measurements.
    pub labels: Labels,
}

impl ProbeRecord {
    /// Rebuilds the position, re-validating the snapshot on the way in.
    ///
    /// ARCHITECTURE.md invariant 5: a snapshot is the only board source and is
    /// re-validated every time, including one we wrote ourselves.
    pub fn state(&self) -> Result<State, ProbeError> {
        let state = self
            .snapshot
            .decode()
            .map_err(|error| ProbeError(format!("{}: {error}", self.id)))?;
        if state.rows() != BOARD || state.cols() != BOARD || state.players() != 2 {
            return Err(ProbeError(format!(
                "{}: the net is {BOARD}x{BOARD} two-player only, got {}x{} with {} seats",
                self.id,
                state.rows(),
                state.cols(),
                state.players()
            )));
        }
        if !state.can_place_neutrals() {
            return Err(ProbeError(format!(
                "{}: not a neutral-decision position (movesLeft={}, neutralUsed={})",
                self.id,
                state.moves_left(),
                state.neutral_used(state.current_player())
            )));
        }
        Ok(state)
    }
}

/// Anything that can go wrong reading, mining or running a probe set.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProbeError(pub String);

impl fmt::Display for ProbeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ProbeError {}

/// Parses a probe set from JSONL, one [`ProbeRecord`] per non-blank line.
pub fn parse_set(text: &str) -> Result<Vec<ProbeRecord>, ProbeError> {
    let mut out = Vec::new();
    for (line_number, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let record: ProbeRecord = serde_json::from_str(line)
            .map_err(|error| ProbeError(format!("line {}: {error}", line_number + 1)))?;
        out.push(record);
    }
    Ok(out)
}

/// Renders a probe set as JSONL, one record per line.
pub fn render_set(records: &[ProbeRecord]) -> Result<String, ProbeError> {
    let mut out = String::new();
    for record in records {
        let line = serde_json::to_string(record)
            .map_err(|error| ProbeError(format!("{}: {error}", record.id)))?;
        out.push_str(&line);
        out.push('\n');
    }
    Ok(out)
}

// ------------------------------------------------------------- games.db mine

/// One recorded action, as `fixtures/probes/tools/dump_games.py` normalises it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DumpMove {
    /// `"move"` or `"neutrals"`.
    pub kind: String,
    /// One `[row, col]` for a move, two for a neutral pair.
    pub cells: Vec<[i32; 2]>,
}

impl DumpMove {
    /// The engine action this records, or `None` if the shape is wrong.
    fn action(&self) -> Option<Action> {
        match (self.kind.as_str(), self.cells.as_slice()) {
            ("move", [target]) => Some(Action::Move {
                target: Pos::new(target[0], target[1]),
            }),
            ("neutrals", [a, b]) => {
                Some(Action::neutrals(Pos::new(a[0], a[1]), Pos::new(b[0], b[1])))
            }
            _ => None,
        }
    }
}

/// One recorded turn from the dump.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DumpTurn {
    /// Recorded turn number.
    pub turn: u32,
    /// Seat that owns the turn.
    pub player: Player,
    /// Actions played, in order. Go's `omitempty` can drop the whole list.
    #[serde(default)]
    pub moves: Vec<DumpMove>,
}

/// One recorded game from the dump.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DumpGame {
    /// `games.id`.
    pub id: String,
    /// `games.started_at`, trimmed of Go's monotonic-clock suffix.
    #[serde(default)]
    pub started_at: String,
    /// Board height.
    pub rows: usize,
    /// Board width.
    pub cols: usize,
    /// Seat names, seat order.
    #[serde(default)]
    pub players: Vec<String>,
    /// `games.result`: the winning seat, or 0.
    #[serde(default)]
    pub result: i32,
    /// `games.termination`.
    #[serde(default)]
    pub termination: String,
    /// The replayable turn list.
    pub turns: Vec<DumpTurn>,
}

/// How [`mine_games`] selects positions.
#[derive(Clone, Copy, Debug)]
pub struct MineConfig {
    /// Turns of the mover's own to measure the material swing over.
    pub horizon: u32,
    /// Smallest `|swing|`, in cells, that earns a class label.
    pub min_swing: i64,
    /// Most `LostAdvantage` positions to keep.
    pub max_suspect: usize,
    /// Most `KeptAdvantage` control positions to keep.
    pub max_control: usize,
    /// Which source the mined records are filed under.
    ///
    /// The heuristic is the same whatever the corpus; what changes is *which
    /// games were fed in*, and that is provenance rather than method. A dump
    /// narrowed to the owner's own live games is mined identically and filed
    /// under [`ProbeSource::LiveOwnerGame`], so a reader can tell the two
    /// halves apart without re-deriving which game ids were which.
    pub source: ProbeSource,
}

impl Default for MineConfig {
    fn default() -> MineConfig {
        MineConfig {
            horizon: 4,
            min_swing: 4,
            max_suspect: 30,
            max_control: 8,
            source: ProbeSource::GamesDb,
        }
    }
}

/// Why a game yielded nothing.
///
/// Named individually rather than pooled into one bucket: a parser bug hiding
/// in an anonymous `replay_error` silently dropped games from the Java miner,
/// and `docs/probes.md` asks the reader to trust these counts.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MineStats {
    /// Games read.
    pub games: usize,
    /// Games replayed end to end.
    pub replayed: usize,
    /// Games with a turn owned by a seat outside 1..=2.
    pub skipped_multiplayer: usize,
    /// Games where the rules rejected a recorded action.
    pub skipped_illegal: usize,
    /// Games whose board is not the net's size.
    pub skipped_board_size: usize,
    /// Neutral placements seen in replayed games.
    pub neutral_turns: usize,
    /// Neutral placements dropped for want of follow-up turns.
    pub dropped_short_horizon: usize,
    /// Neutral placements the placer lost with a swing shallower than
    /// `min_swing`: too flat to call suspect, and not a control either.
    pub dropped_flat: usize,
    /// Neutral placements in a game whose recorded `result` names no winner:
    /// both classes are defined by the outcome, so there is nothing to label.
    pub dropped_no_winner: usize,
    /// Labelled placements dropped as a repeat of a position already kept.
    ///
    /// Not a rounding error: the corpus is mostly deterministic bots, which
    /// replay the same opening, so a single mid-game position can appear in
    /// dozens of games. Six copies of one board in a 38-position set would let
    /// that board write the aggregate on its own.
    pub dropped_duplicate: usize,
}

/// The position at the start of one replayed turn.
#[derive(Clone, Debug)]
struct TurnPosition {
    turn: u32,
    state: State,
}

/// Replays one dumped game into the position at the start of every turn.
///
/// Faithful to Java's `GamesDbReplay`: the board is rebuilt at each recorded
/// turn with `moves_left = 3` and the threaded per-seat neutral budget (the
/// PGN records no per-ply `movesLeft`), and each recorded action goes through
/// the legality-checked [`State::apply`] so an out-of-rules recording surfaces
/// as a rejection instead of fabricating a board.
fn replay(game: &DumpGame) -> Result<Vec<TurnPosition>, &'static str> {
    if game.rows != BOARD || game.cols != BOARD {
        return Err("board_size");
    }
    let mut board = initial_board(game.rows, game.cols);
    let mut neutral_used = [false, false];
    let mut positions = Vec::new();
    let mut over = false;

    for turn in &game.turns {
        if turn.player < 1 || turn.player > 2 {
            return Err("multiplayer");
        }
        if over {
            // Go's `omitempty` leaves a trailing `{"player":N}`; that is just
            // the tail. A turn carrying moves after the game ended cannot
            // exist under the rules.
            if turn.moves.is_empty() {
                continue;
            }
            return Err("illegal_move");
        }
        let mut state = State::from_grid(
            game.rows,
            game.cols,
            2,
            &board,
            turn.player,
            virus_core::ACTIONS_PER_TURN,
            &neutral_used,
        )
        .map_err(|_| "illegal_move")?;
        positions.push(TurnPosition {
            turn: turn.turn,
            state: state.clone(),
        });

        for recorded in &turn.moves {
            let action = recorded.action().ok_or("illegal_move")?;
            if state.game_over() || state.current_player() != turn.player {
                return Err("illegal_move");
            }
            state = state.apply(action).map_err(|_| "illegal_move")?;
        }
        board = (0..state.cell_count()).map(|i| state.cell_at(i)).collect();
        neutral_used = [state.neutral_used(1), state.neutral_used(2)];
        over = state.game_over();
    }
    Ok(positions)
}

/// The opening board: an empty grid with the two bases in opposite corners.
fn initial_board(rows: usize, cols: usize) -> Vec<Cell> {
    let mut board = vec![Cell::EMPTY; rows * cols];
    board[0] = Cell::new(1, CellKind::Base);
    board[rows * cols - 1] = Cell::new(2, CellKind::Base);
    board
}

/// Whether `mover` won a game whose recorded `result` names a real seat, or
/// `None` when the recording names no winner.
fn decided(result: i32, mover: Player) -> Option<bool> {
    match result {
        1 | 2 => Some(result == mover as i32),
        _ => None,
    }
}

/// `owned_cells(mover) - owned_cells(opponent)`.
fn advantage(state: &State, mover: Player) -> i64 {
    let opponent = if mover == 1 { 2 } else { 1 };
    state.owned_cells(mover) as i64 - state.owned_cells(opponent) as i64
}

/// The measurements the heuristic reads, computed for one position.
///
/// `after` is `(the position h own-turns later, h)`; `next` is the mover's very
/// next turn, which is what [`Labels::immediate_cost`] is measured over.
fn label(
    state: &State,
    next: Option<&State>,
    after: Option<(&State, u32)>,
    last: &State,
    placer_won: Option<bool>,
) -> Labels {
    let mover = state.current_player();
    let before = advantage(state, mover);
    let (advantage_after, horizon_turns) = match after {
        Some((state, turns)) => (Some(advantage(state, mover)), turns),
        None => (None, 0),
    };
    let normals = state.owned_normals(mover).len();
    Labels {
        mover,
        own_normals: normals,
        legal_moves: state.move_targets(mover).len(),
        legal_pairs: normals * normals.saturating_sub(1) / 2,
        advantage_before: before,
        advantage_after,
        advantage_swing: advantage_after.map(|after| after - before),
        immediate_cost: next.map(|next| advantage(next, mover) - before),
        advantage_final: Some(advantage(last, mover)),
        horizon_turns,
        placer_won,
    }
}

/// Mines neutral-decision positions out of replayed prod games.
///
/// The heuristic, its horizon and its limits are documented in the module
/// header. Selection is deterministic: suspects are taken in ascending swing
/// (the largest material collapse first) and controls in descending swing
/// (the placements that cost their winner least first), ties broken by id then
/// turn, so the same dump always produces the same set.
pub fn mine_games(
    games: &[DumpGame],
    config: MineConfig,
    origin: &str,
) -> (Vec<ProbeRecord>, MineStats) {
    let mut stats = MineStats {
        games: games.len(),
        ..MineStats::default()
    };
    let mut candidates: Vec<Candidate> = Vec::new();

    for game in games {
        let positions = match replay(game) {
            Ok(positions) => positions,
            Err("multiplayer") => {
                stats.skipped_multiplayer += 1;
                continue;
            }
            Err("board_size") => {
                stats.skipped_board_size += 1;
                continue;
            }
            Err(_) => {
                stats.skipped_illegal += 1;
                continue;
            }
        };
        stats.replayed += 1;
        let Some(last) = positions.last() else {
            continue;
        };

        for (index, position) in positions.iter().enumerate() {
            let turn = &game.turns[index];
            let Some(first) = turn.moves.first() else {
                continue;
            };
            let Some(Action::PlaceNeutrals { cells }) = first.action() else {
                continue;
            };
            if !position.state.can_place_neutrals() {
                continue;
            }
            stats.neutral_turns += 1;

            let mover = position.state.current_player();
            let later: Vec<&TurnPosition> = positions[index + 1..]
                .iter()
                .filter(|p| p.state.current_player() == mover)
                .collect();
            if (later.len() as u32) < MIN_HORIZON_TURNS {
                stats.dropped_short_horizon += 1;
                continue;
            }
            // Both classes are defined by who won, so a game with no winner
            // recorded — `games.result` is 0 for a draw, an abandoned game, or
            // one the recorder never resolved — cannot be labelled at all.
            // Reading `result != mover` as "the placer lost" would file every
            // such position under `LostAdvantage`, which is the one thing the
            // class is documented not to contain.
            let Some(placer_won) = decided(game.result, mover) else {
                stats.dropped_no_winner += 1;
                continue;
            };
            // `min(1)` on the horizon, not just on `later.len()`: `--horizon 0`
            // is a legal thing to type and must not index `later[-1]`.
            let horizon = (config.horizon.max(1) as usize).min(later.len());
            let after = later[horizon - 1];
            let labels = label(
                &position.state,
                Some(&later[0].state),
                Some((&after.state, horizon as u32)),
                &last.state,
                Some(placer_won),
            );
            let swing = labels.advantage_swing.unwrap_or(0);
            // The class leans on the game's outcome, not on the swing alone:
            // the swing is dominated by the action's own mechanical price. See
            // the module header.
            let class = if placer_won {
                ProbeClass::KeptAdvantage
            } else if swing <= -config.min_swing {
                ProbeClass::LostAdvantage
            } else {
                stats.dropped_flat += 1;
                continue;
            };

            let short = game.id.get(..8).unwrap_or(&game.id);
            candidates.push(Candidate {
                hash: position.state.state_hash(),
                swing,
                record: ProbeRecord {
                    id: format!("{}-{short}-t{}", config.source.id_prefix(), position.turn),
                    source: config.source,
                    class,
                    snapshot: position.state.snapshot(),
                    provenance: Provenance {
                        origin: origin.to_owned(),
                        game_id: Some(game.id.clone()),
                        started_at: Some(game.started_at.clone()),
                        players: game.players.clone(),
                        turn: Some(position.turn),
                        seed: None,
                        played_neutrals: Some([
                            [cells[0].row, cells[0].col],
                            [cells[1].row, cells[1].col],
                        ]),
                        note: format!(
                            "seat {mover} placed neutrals here; termination={}, winner={}",
                            game.termination, game.result
                        ),
                    },
                    labels,
                },
            });
        }
    }

    candidates.sort_by(|a, b| {
        a.swing
            .cmp(&b.swing)
            .then_with(|| a.record.id.cmp(&b.record.id))
            .then_with(|| a.record.provenance.turn.cmp(&b.record.provenance.turn))
    });

    let mut seen: BTreeSet<u64> = BTreeSet::new();
    let mut records = take(
        &candidates,
        ProbeClass::LostAdvantage,
        config.max_suspect,
        false,
        &mut seen,
        &mut stats,
    );
    records.extend(take(
        &candidates,
        ProbeClass::KeptAdvantage,
        config.max_control,
        true,
        &mut seen,
        &mut stats,
    ));
    (records, stats)
}

/// Takes up to `limit` distinct positions of one class off the sorted
/// candidate list, worst-first (`reversed == false`) or best-first.
fn take(
    candidates: &[Candidate],
    wanted: ProbeClass,
    limit: usize,
    reversed: bool,
    seen: &mut BTreeSet<u64>,
    stats: &mut MineStats,
) -> Vec<ProbeRecord> {
    let ordered: Box<dyn Iterator<Item = &Candidate>> = if reversed {
        Box::new(candidates.iter().rev())
    } else {
        Box::new(candidates.iter())
    };
    let mut kept = Vec::new();
    for candidate in ordered {
        // Checked before the push, not after: `--max-control 0` is how a
        // caller asks for no control group, and a post-push `== limit` would
        // never fire and hand back every candidate instead of none.
        if kept.len() >= limit {
            break;
        }
        if candidate.record.class != wanted {
            continue;
        }
        if !seen.insert(candidate.hash) {
            stats.dropped_duplicate += 1;
            continue;
        }
        kept.push(candidate.record.clone());
    }
    kept
}

/// A mined position awaiting selection.
#[derive(Clone, Debug)]
struct Candidate {
    /// `State::state_hash`, which folds in the mover, `movesLeft` and both
    /// neutral budgets — everything two "same board" positions could differ by.
    hash: u64,
    /// The material swing the selection sorts on.
    swing: i64,
    record: ProbeRecord,
}

// ------------------------------------------------------------ self-play mine

/// How [`mine_self_play`] generates trajectories.
///
/// The defaults are `vsbot/examples/ponderrepro`'s: the same 12x12 two-player
/// opening, the same `SplitMix64` seed derivation, the same eight plies of
/// random opening play, the same play-mode configuration with the net's value
/// head. What is deliberately *not* reproduced is the pondering session — the
/// warm tree and the early-stop rule belong to bd `vsbot-gei`, and this bead
/// records that cold search, warm search and a 10 000-simulation reference all
/// choose the same neutral. Cold search is therefore sufficient to catch the
/// positions, and is the one that reproduces from a seed alone.
#[derive(Clone, Copy, Debug)]
pub struct SelfPlayConfig {
    /// Games to play.
    pub games: u64,
    /// Base seed; game `g` runs on `derive_game_seed(seed, g)`.
    pub seed: u64,
    /// Simulations each side spends per action.
    pub sims: u32,
    /// Plies of random opening play, so a deterministic engine does not replay
    /// one game forever.
    pub random_plies: u32,
    /// Turn cap.
    pub turn_cap: u32,
    /// Most positions to keep.
    pub max_positions: usize,
}

impl Default for SelfPlayConfig {
    fn default() -> SelfPlayConfig {
        SelfPlayConfig {
            games: 6,
            seed: 0x5EED,
            sims: 800,
            random_plies: 8,
            turn_cap: 200,
            max_positions: 16,
        }
    }
}

/// Plays self-play games and keeps every position the net answered with a
/// `PlaceNeutrals`.
///
/// Deterministic for a given seed, simulation count and net.
pub fn mine_self_play(
    net: &PolicyValueNet,
    config: SelfPlayConfig,
    origin: &str,
) -> Vec<ProbeRecord> {
    let search = Config {
        value_source: ValueSource::Net,
        ..Config::play()
    };
    let mut out: Vec<ProbeRecord> = Vec::new();
    let mut seen: BTreeSet<u64> = BTreeSet::new();

    for game in 0..config.games {
        // `ponderrepro`'s derivation, deliberately *not* the gauntlet's
        // `rng::derive_game_seed`: that one hands games `2k` and `2k+1` the
        // same opening on purpose (colour-paired seeds), and with both seats
        // played by the same deterministic engine here, a pair would simply
        // replay one game twice and put two copies of every position in the
        // set.
        let seed = mix64(config.seed ^ (game + 1));
        let mut rng = Rng::new(seed);
        let mut state = State::new(BOARD, BOARD, 2).expect("a legal 12x12 opening");
        let mut ply = 0u32;
        let mut turns = 0u32;

        while !state.game_over() && turns < config.turn_cap && out.len() < config.max_positions {
            if state.moves_left() == virus_core::ACTIONS_PER_TURN {
                turns += 1;
            }
            let legal = state.legal_actions();
            if legal.is_empty() {
                break;
            }
            let action = if ply < config.random_plies {
                legal[rng.below(legal.len()).unwrap_or(0)]
            } else {
                let mut searcher = MctsSearcher::new(state.clone(), search, Some(net));
                searcher.run_sims(config.sims);
                match searcher.best_action() {
                    Some(action) => action,
                    None => break,
                }
            };
            ply += 1;

            if let Action::PlaceNeutrals { cells } = action {
                if ply > config.random_plies && seen.insert(state.state_hash()) {
                    let labels = label(&state, None, None, &state, None);
                    out.push(ProbeRecord {
                        id: format!("selfplay-g{game}-p{ply}"),
                        source: ProbeSource::PonderRepro,
                        class: ProbeClass::ChampionChoseNeutral,
                        snapshot: state.snapshot(),
                        provenance: Provenance {
                            origin: origin.to_owned(),
                            game_id: None,
                            started_at: None,
                            players: Vec::new(),
                            turn: Some(turns),
                            seed: Some(seed),
                            played_neutrals: Some([
                                [cells[0].row, cells[0].col],
                                [cells[1].row, cells[1].col],
                            ]),
                            note: format!(
                                "the champion chose this pair at {} sims (cold, play mode)",
                                config.sims
                            ),
                        },
                        labels,
                    });
                }
            }
            state = state.apply(action).expect("a legal action applies");
        }
        // A self-play game is minutes of work; a run with no sign of life is
        // indistinguishable from a hang.
        eprintln!(
            "probe mine-play: game {} of {} done after {turns} turns, {} positions kept",
            game + 1,
            config.games,
            out.len()
        );
    }
    out
}

// ------------------------------------------------------------------- report

/// What one net said about one probe position at one simulation count.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SearchOutcome {
    /// Simulations run.
    pub sims: u32,
    /// Whether the most-visited root action is a `PlaceNeutrals`.
    pub chose_neutrals: bool,
    /// The chosen action, rendered.
    pub chosen: String,
    /// Share of root visits that landed on `PlaceNeutrals` edges.
    pub neutral_visit_share: f64,
    /// Root value after the search, in the *mover's* frame.
    pub root_value: f64,
}

/// What one net said about one probe position.
///
/// Every field is a description of the net's output. None of them is a
/// strength measurement; see [`INFORMATIONAL`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProbeReport {
    /// The record this describes.
    pub id: String,
    /// Its source.
    pub source: ProbeSource,
    /// Its class.
    pub class: ProbeClass,
    /// Mover's own `Normal` cells — the pair head's support.
    pub own_normals: usize,
    /// Legal `Move` actions.
    pub legal_moves: usize,
    /// Legal `PlaceNeutrals` actions.
    pub legal_pairs: usize,
    /// Total prior mass the masked softmax puts on `PlaceNeutrals` actions.
    pub neutral_prior_mass: f64,
    /// The single largest `PlaceNeutrals` prior.
    pub top_neutral_prior: f64,
    /// The single largest `Move` prior.
    pub top_move_prior: f64,
    /// `logsumexp` over the pair logits: the unnormalised weight of the whole
    /// `PlaceNeutrals` class.
    pub pair_logsumexp: f64,
    /// `logsumexp` over the move logits.
    pub move_logsumexp: f64,
    /// `pairLogsumexp - moveLogsumexp`. Positive means the *class* of neutrals
    /// outweighs the class of moves. `neutralPriorMass == sigmoid(this)`.
    pub class_logit_gap: f64,
    /// `ln(legalPairs) - ln(legalMoves)`: the share of the class gap that the
    /// pair class gets purely for having more members than the move class.
    pub count_term: f64,
    /// `classLogitGap - countTerm`: what the net's actual logit *levels*
    /// contribute once the class sizes are accounted for. This is the term the
    /// net can steer and the count term is the one it cannot.
    pub level_term: f64,
    /// The best pair logit minus the best move logit. Positive means the net's
    /// single favourite action is a neutral placement.
    pub best_logit_gap: f64,
    /// The artifact's global `pair_bias`, repeated per row because it is the
    /// one term of the factored pair logit that cannot depend on the position.
    pub pair_bias: f64,
    /// `pair_bias + 2 * ln(sum(exp(u_i))) - ln 2` over the mover's normals.
    ///
    /// The factored head makes the pair class's weight a closed form:
    /// `sum_{i<j} exp(u_i + u_j + b) = exp(b) * (S^2 - Q) / 2` with
    /// `S = sum exp(u_i)` and `Q = sum exp(2 u_i)`. This field drops the `Q`
    /// correction, so it is an upper bound that sits within a few hundredths
    /// of [`ProbeReport::pair_logsumexp`] — and the reason it is reported is
    /// the `S^2`: the class's weight is *quadratic* in the mover's own cell
    /// count, whatever the trunk says.
    pub pair_logsumexp_closed_form: f64,
    /// Net value at the position, mover frame, before any action.
    pub value_before: f64,
    /// Net value after the net's favourite neutral pair, mover frame.
    pub value_after_top_neutral: f64,
    /// Net value after the net's favourite move, mover frame.
    pub value_after_top_move: f64,
    /// One outcome per requested simulation count, in ascending order.
    pub searches: Vec<SearchOutcome>,
}

impl ProbeReport {
    /// `value_after_top_neutral - value_before`: what the net thinks spending
    /// the whole turn on a neutral pair is worth.
    pub fn neutral_value_delta(&self) -> f64 {
        self.value_after_top_neutral - self.value_before
    }

    /// `value_after_top_neutral - value_after_top_move`: what the net thinks
    /// the neutral is worth *relative to just playing*.
    pub fn neutral_minus_move_value(&self) -> f64 {
        self.value_after_top_neutral - self.value_after_top_move
    }
}

fn render_action(action: Action) -> String {
    match action {
        Action::Move { target } => format!("move({},{})", target.row, target.col),
        Action::PlaceNeutrals { cells } => format!(
            "neutrals({},{})+({},{})",
            cells[0].row, cells[0].col, cells[1].row, cells[1].col
        ),
    }
}

/// `ln(sum(exp(values)))`, computed stably.
fn logsumexp(values: &[f64]) -> f64 {
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !max.is_finite() {
        return max;
    }
    max + values.iter().map(|v| (v - max).exp()).sum::<f64>().ln()
}

/// Net value of `state` expressed in `mover`'s frame.
///
/// The value head is mover-frame by construction, so a child whose mover
/// flipped needs a single sign application — the same absolute-frame discipline
/// ARCHITECTURE.md invariant 1 requires of the searcher.
fn value_in_frame(
    net: &PolicyValueNet,
    state: &State,
    mover: Player,
    scratch: &mut virus_mcts::NetScratch,
) -> f64 {
    let heads = net.forward(&Encoded::from_state(state), scratch);
    let value = f64::from(heads.value.unwrap_or(0.0));
    if state.current_player() == mover {
        value
    } else {
        -value
    }
}

/// Runs one net over one probe position at each of `sim_counts`.
///
/// The priors are read off a real [`MctsSearcher`] root rather than recomputed
/// here: the deployed masked softmax lives in `virus-mcts` and a second copy of
/// it in the measuring instrument would be free to drift from the thing it
/// measures.
pub fn run_probe(
    net: &PolicyValueNet,
    record: &ProbeRecord,
    sim_counts: &[u32],
) -> Result<ProbeReport, ProbeError> {
    let state = record.state()?;
    let mover = state.current_player();
    // Every reported quantity is a comparison of the pair class against the
    // move class. With no legal move there is no comparison: `logsumexp` over
    // the empty class is `-inf`, `ln(0)` poisons `countTerm`, and the row would
    // enter the aggregates as a silent NaN. Refuse it instead.
    if state.move_targets(mover).is_empty() {
        return Err(ProbeError(format!(
            "{}: the mover has no legal move, so there is no move class to compare against",
            record.id
        )));
    }

    let mut net_scratch = net.scratch();
    let heads = net.forward(&Encoded::from_state(&state), &mut net_scratch);
    let pair_bias = f64::from(net.pair_bias());

    let mut searcher = MctsSearcher::new(
        state.clone(),
        Config {
            value_source: ValueSource::Net,
            ..Config::play()
        },
        Some(net),
    );
    // One simulation is what turns the root's expansion into a prior; the
    // searcher computes it exactly as a real search would.
    searcher.run_sims(1);
    let actions: Vec<Action> = searcher.root_actions().to_vec();
    let priors: Vec<f64> = searcher
        .root_priors()
        .iter()
        .map(|p| f64::from(*p))
        .collect();

    let cell = |pos: Pos| pos.row as usize * state.cols() + pos.col as usize;
    let mut neutral_prior_mass = 0.0;
    let mut top_neutral = (f64::NEG_INFINITY, None);
    let mut top_move = (f64::NEG_INFINITY, None);
    let mut pair_logits = Vec::new();
    let mut move_logits = Vec::new();

    for (index, action) in actions.iter().enumerate() {
        let prior = priors.get(index).copied().unwrap_or(0.0);
        match action {
            Action::Move { target } => {
                move_logits.push(f64::from(heads.move_logits[cell(*target)]));
                if prior > top_move.0 {
                    top_move = (prior, Some(*action));
                }
            }
            Action::PlaceNeutrals { cells } => {
                neutral_prior_mass += prior;
                pair_logits.push(
                    f64::from(heads.pair_u[cell(cells[0])])
                        + f64::from(heads.pair_u[cell(cells[1])])
                        + pair_bias,
                );
                if prior > top_neutral.0 {
                    top_neutral = (prior, Some(*action));
                }
            }
        }
    }

    let own_normals = state.owned_normals(mover);
    let sum_exp_u: f64 = own_normals
        .iter()
        .map(|pos| f64::from(heads.pair_u[cell(*pos)]).exp())
        .sum();
    let closed_form = if sum_exp_u > 0.0 {
        pair_bias + 2.0 * sum_exp_u.ln() - 2.0f64.ln()
    } else {
        f64::NEG_INFINITY
    };

    let value_before = f64::from(heads.value.unwrap_or(0.0));
    let mut after = |action: Option<Action>| -> f64 {
        match action.and_then(|action| state.apply(action).ok()) {
            Some(child) => value_in_frame(net, &child, mover, &mut net_scratch),
            None => f64::NAN,
        }
    };
    let value_after_top_neutral = after(top_neutral.1);
    let value_after_top_move = after(top_move.1);

    let mut sorted: Vec<u32> = sim_counts.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    let mut searches = Vec::new();
    // The prior read above already cost one simulation; counting from the
    // searcher's own tally rather than from zero keeps `sims: 192` meaning 192
    // simulations, not 193. Searches are cumulative on one tree: MCTS only
    // accumulates, so running to 192 and then 808 more is the same tree as
    // running to 1000 in one go.
    let mut run = u32::try_from(searcher.sims_run()).unwrap_or(u32::MAX);
    for target in sorted {
        if target > run {
            searcher.run_sims(target - run);
            run = target;
        }
        let visits = searcher.root_visits();
        let total: u64 = visits.iter().map(|v| u64::from(*v)).sum();
        let neutral: u64 = searcher
            .root_actions()
            .iter()
            .zip(visits)
            .filter(|(action, _)| matches!(action, Action::PlaceNeutrals { .. }))
            .map(|(_, v)| u64::from(*v))
            .sum();
        let chosen = searcher.best_action();
        let root_value_abs = searcher.root_value_abs();
        searches.push(SearchOutcome {
            sims: target,
            chose_neutrals: matches!(chosen, Some(Action::PlaceNeutrals { .. })),
            chosen: chosen
                .map(render_action)
                .unwrap_or_else(|| "none".to_owned()),
            neutral_visit_share: if total == 0 {
                0.0
            } else {
                neutral as f64 / total as f64
            },
            root_value: if mover == 1 {
                root_value_abs
            } else {
                -root_value_abs
            },
        });
    }

    Ok(ProbeReport {
        id: record.id.clone(),
        source: record.source,
        class: record.class,
        own_normals: own_normals.len(),
        legal_moves: move_logits.len(),
        legal_pairs: pair_logits.len(),
        neutral_prior_mass,
        top_neutral_prior: if top_neutral.0.is_finite() {
            top_neutral.0
        } else {
            0.0
        },
        top_move_prior: if top_move.0.is_finite() {
            top_move.0
        } else {
            0.0
        },
        pair_logsumexp: logsumexp(&pair_logits),
        move_logsumexp: logsumexp(&move_logits),
        class_logit_gap: logsumexp(&pair_logits) - logsumexp(&move_logits),
        count_term: (pair_logits.len() as f64).ln() - (move_logits.len() as f64).ln(),
        level_term: (logsumexp(&pair_logits) - logsumexp(&move_logits))
            - ((pair_logits.len() as f64).ln() - (move_logits.len() as f64).ln()),
        best_logit_gap: pair_logits
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
            - move_logits
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max),
        pair_bias,
        pair_logsumexp_closed_form: closed_form,
        value_before,
        value_after_top_neutral,
        value_after_top_move,
        searches,
    })
}

/// Mean of `values`, or 0 for an empty slice.
fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

/// The aggregate a run prints under the per-position table.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProbeSummary {
    /// Positions reported.
    pub positions: usize,
    /// Mean prior mass on `PlaceNeutrals`, by class.
    pub neutral_prior_mass_by_class: BTreeMap<String, f64>,
    /// Fraction of positions whose search chose a `PlaceNeutrals`, by
    /// simulation count.
    pub chose_neutrals_by_sims: BTreeMap<u32, f64>,
    /// Mean `value_after_top_neutral - value_before`.
    pub mean_neutral_value_delta: f64,
    /// Mean `value_after_top_neutral - value_after_top_move`.
    pub mean_neutral_minus_move: f64,
    /// Positions where the net's single favourite action is a neutral pair.
    pub best_action_is_neutral: usize,
}

/// Aggregates a run.
pub fn summarise(reports: &[ProbeReport]) -> ProbeSummary {
    let mut by_class: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut by_sims: BTreeMap<u32, (usize, usize)> = BTreeMap::new();
    for report in reports {
        by_class
            .entry(report.class.to_string())
            .or_default()
            .push(report.neutral_prior_mass);
        for search in &report.searches {
            let slot = by_sims.entry(search.sims).or_insert((0, 0));
            slot.1 += 1;
            slot.0 += usize::from(search.chose_neutrals);
        }
    }
    ProbeSummary {
        positions: reports.len(),
        neutral_prior_mass_by_class: by_class
            .into_iter()
            .map(|(class, values)| (class, mean(&values)))
            .collect(),
        chose_neutrals_by_sims: by_sims
            .into_iter()
            .map(|(sims, (hits, total))| (sims, hits as f64 / total.max(1) as f64))
            .collect(),
        mean_neutral_value_delta: mean(
            &reports
                .iter()
                .map(ProbeReport::neutral_value_delta)
                .collect::<Vec<_>>(),
        ),
        mean_neutral_minus_move: mean(
            &reports
                .iter()
                .map(ProbeReport::neutral_minus_move_value)
                .collect::<Vec<_>>(),
        ),
        best_action_is_neutral: reports
            .iter()
            .filter(|report| report.best_logit_gap > 0.0)
            .count(),
    }
}

/// Renders the per-position table a run prints.
pub fn render_table(reports: &[ProbeReport], sim_counts: &[u32]) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "{:<26} {:<22} {:>5} {:>6} {:>8} {:>8} {:>8} {:>8}",
        "id", "class", "norm", "pairs", "p(neut)", "top(n)", "top(mv)", "dV(n)"
    ));
    for sims in sim_counts {
        out.push_str(&format!(" {:>10}", format!("@{sims}")));
    }
    out.push('\n');
    for report in reports {
        out.push_str(&format!(
            "{:<26} {:<22} {:>5} {:>6} {:>8.4} {:>8.4} {:>8.4} {:>+8.4}",
            report.id,
            report.class.to_string(),
            report.own_normals,
            report.legal_pairs,
            report.neutral_prior_mass,
            report.top_neutral_prior,
            report.top_move_prior,
            report.neutral_value_delta(),
        ));
        for sims in sim_counts {
            let mark = report
                .searches
                .iter()
                .find(|search| search.sims == *sims)
                .map(|search| {
                    if search.chose_neutrals {
                        format!("NEUTRAL {:.0}%", search.neutral_visit_share * 100.0)
                    } else {
                        format!("move {:.0}%", search.neutral_visit_share * 100.0)
                    }
                })
                .unwrap_or_else(|| "-".to_owned());
            out.push_str(&format!(" {mark:>10}"));
        }
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The opening turn of a two-player 12x12 game, as the dump records it.
    fn opening_game() -> DumpGame {
        DumpGame {
            id: "test-game-0001".to_owned(),
            started_at: "2026-08-01 00:00:00".to_owned(),
            rows: 12,
            cols: 12,
            players: vec!["A".to_owned(), "B".to_owned()],
            result: 1,
            termination: "no_moves".to_owned(),
            turns: vec![
                DumpTurn {
                    turn: 1,
                    player: 1,
                    moves: vec![
                        DumpMove {
                            kind: "move".to_owned(),
                            cells: vec![[1, 1]],
                        },
                        DumpMove {
                            kind: "move".to_owned(),
                            cells: vec![[2, 2]],
                        },
                        DumpMove {
                            kind: "move".to_owned(),
                            cells: vec![[3, 3]],
                        },
                    ],
                },
                DumpTurn {
                    turn: 2,
                    player: 2,
                    moves: vec![DumpMove {
                        kind: "move".to_owned(),
                        cells: vec![[10, 10]],
                    }],
                },
            ],
        }
    }

    #[test]
    fn a_recorded_game_replays_through_the_real_rules() {
        let positions = replay(&opening_game()).expect("the recording is legal");
        assert_eq!(positions.len(), 2);
        assert_eq!(positions[0].state.current_player(), 1);
        assert_eq!(positions[0].state.moves_left(), 3);
        // Turn 2's position must show turn 1's three placements.
        assert_eq!(positions[1].state.owned_cells(1), 4);
        assert_eq!(positions[1].state.current_player(), 2);
    }

    #[test]
    fn an_out_of_rules_recording_is_rejected_not_fabricated() {
        let mut game = opening_game();
        // (6,6) is not adjacent to seat 1's base component on turn 1.
        game.turns[0].moves[0].cells = vec![[6, 6]];
        assert!(replay(&game).is_err());
    }

    #[test]
    fn a_seat_outside_the_two_player_rules_is_skipped_by_name() {
        let mut game = opening_game();
        game.turns[1].player = 3;
        assert_eq!(replay(&game).err(), Some("multiplayer"));
    }

    #[test]
    fn omitted_zero_coordinates_round_trip() {
        // Go's `omitempty` drops zeroes; the dump restores them, and the
        // action must come back out at (0, 0).
        let recorded: DumpMove =
            serde_json::from_str(r#"{"kind":"move","cells":[[0,1]]}"#).expect("parses");
        assert_eq!(
            recorded.action(),
            Some(Action::Move {
                target: Pos::new(0, 1)
            })
        );
    }

    #[test]
    fn a_probe_record_round_trips_through_jsonl() {
        let state = State::new(12, 12, 2).expect("a legal opening");
        let record = ProbeRecord {
            id: "unit-0".to_owned(),
            source: ProbeSource::GamesDb,
            class: ProbeClass::LostAdvantage,
            snapshot: state.snapshot(),
            provenance: Provenance {
                origin: "unit test".to_owned(),
                ..Provenance::default()
            },
            labels: label(&state, None, None, &state, None),
        };
        let text = render_set(std::slice::from_ref(&record)).expect("renders");
        let parsed = parse_set(&text).expect("parses");
        assert_eq!(parsed, vec![record]);
    }

    #[test]
    fn a_non_neutral_position_is_refused_by_state() {
        let state = State::new(12, 12, 2).expect("a legal opening");
        let after = state
            .apply(Action::Move {
                target: Pos::new(1, 1),
            })
            .expect("a legal move");
        let record = ProbeRecord {
            id: "unit-1".to_owned(),
            source: ProbeSource::GamesDb,
            class: ProbeClass::LostAdvantage,
            snapshot: after.snapshot(),
            provenance: Provenance::default(),
            labels: Labels::default(),
        };
        // `movesLeft` is 2, so the position cannot host a neutral decision and
        // must not silently enter a probe set.
        assert!(record.state().is_err());
    }

    /// One placed cell, as the dump records it.
    fn mv(row: i32, col: i32) -> DumpMove {
        DumpMove {
            kind: "move".to_owned(),
            cells: vec![[row, col]],
        }
    }

    /// A complete two-player game in which seat 2 places neutrals on turn 4
    /// and then keeps losing ground. Seat 2 has two further turns of its own,
    /// which is what the horizon needs.
    fn game_with_a_neutral(result: i32) -> DumpGame {
        let turn = |turn: u32, player: Player, moves: Vec<DumpMove>| DumpTurn {
            turn,
            player,
            moves,
        };
        DumpGame {
            id: "test-neutral-0001".to_owned(),
            started_at: "2026-08-01 00:00:00".to_owned(),
            rows: 12,
            cols: 12,
            players: vec!["A".to_owned(), "B".to_owned()],
            result,
            termination: "no_moves".to_owned(),
            turns: vec![
                turn(1, 1, vec![mv(1, 1), mv(2, 2), mv(3, 3)]),
                turn(2, 2, vec![mv(10, 10), mv(9, 9), mv(8, 8)]),
                turn(3, 1, vec![mv(4, 4), mv(5, 5), mv(6, 6)]),
                turn(
                    4,
                    2,
                    vec![DumpMove {
                        kind: "neutrals".to_owned(),
                        cells: vec![[10, 10], [9, 9]],
                    }],
                ),
                turn(5, 1, vec![mv(7, 7), mv(4, 3), mv(3, 4)]),
                turn(6, 2, vec![mv(10, 11), mv(11, 10), mv(9, 10)]),
                turn(7, 1, vec![mv(7, 6), mv(6, 7), mv(8, 7)]),
                turn(8, 2, vec![mv(11, 9), mv(10, 9), mv(11, 8)]),
            ],
        }
    }

    fn mine_config() -> MineConfig {
        MineConfig {
            horizon: 4,
            min_swing: 4,
            max_suspect: 10,
            max_control: 10,
            source: ProbeSource::GamesDb,
        }
    }

    /// The wire spelling is what the fixture and the `--source` flag both use,
    /// so `parse` and `Display` have to agree with `serde` and with each other.
    #[test]
    fn probe_sources_round_trip_their_wire_spelling() {
        for source in [
            ProbeSource::GamesDb,
            ProbeSource::PonderRepro,
            ProbeSource::LiveOwnerGame,
        ] {
            let text = source.to_string();
            assert_eq!(ProbeSource::parse(&text), Ok(source), "{text}");
            // `serde`'s kebab-case rename is the fixture's spelling; a `Display`
            // that drifted from it would make `--source` and the committed file
            // disagree about the same value.
            assert_eq!(
                serde_json::to_string(&source).expect("serialise"),
                format!("\"{text}\"")
            );
        }
        assert!(ProbeSource::parse("games_db").is_err());
        assert!(ProbeSource::parse("").is_err());
    }

    /// The corpus a dump was narrowed to is provenance, not method: the same
    /// heuristic files its output under whichever source the caller names, and
    /// the id carries it so a report table stays legible.
    #[test]
    fn the_configured_source_reaches_the_mined_records() {
        let config = MineConfig {
            source: ProbeSource::LiveOwnerGame,
            ..mine_config()
        };
        let (records, _) = mine_games(&[game_with_a_neutral(1)], config, "unit test");
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].source, ProbeSource::LiveOwnerGame);
        assert!(
            records[0].id.starts_with("live-"),
            "the id must name the source: {}",
            records[0].id
        );

        // Same corpus, same heuristic, different tag: everything except the
        // source and the id prefix must be identical, or the tag is quietly
        // changing the measurement.
        let (default, _) = mine_games(&[game_with_a_neutral(1)], mine_config(), "unit test");
        assert!(default[0].id.starts_with("gamesdb-"));
        assert_eq!(records[0].class, default[0].class);
        assert_eq!(records[0].labels, default[0].labels);
        assert_eq!(records[0].snapshot, default[0].snapshot);
        assert_eq!(records[0].provenance, default[0].provenance);
        assert_eq!(
            records[0].id.trim_start_matches("live-"),
            default[0].id.trim_start_matches("gamesdb-")
        );
    }

    #[test]
    fn a_lost_neutral_placement_is_mined_as_a_suspect() {
        let (records, stats) = mine_games(&[game_with_a_neutral(1)], mine_config(), "unit test");
        assert_eq!(stats.replayed, 1, "{stats:?}");
        assert_eq!(stats.neutral_turns, 1, "{stats:?}");
        assert_eq!(records.len(), 1, "{stats:?}");
        assert_eq!(records[0].class, ProbeClass::LostAdvantage);
        assert_eq!(records[0].labels.mover, 2);
        assert_eq!(records[0].labels.placer_won, Some(false));
        // The action's mechanical price is recorded separately from the swing.
        assert!(records[0].labels.immediate_cost.unwrap() < 0);
    }

    #[test]
    fn a_game_with_no_recorded_winner_is_not_labelled_a_loss() {
        // `games.result == 0` means no winning seat. Reading that as "the
        // placer lost" would file a draw under `LostAdvantage`, which is the
        // one thing that class is documented not to contain.
        let (records, stats) = mine_games(&[game_with_a_neutral(0)], mine_config(), "unit test");
        assert!(records.is_empty(), "{records:#?}");
        assert_eq!(stats.dropped_no_winner, 1, "{stats:?}");
        assert_eq!(stats.dropped_flat, 0, "{stats:?}");
    }

    #[test]
    fn a_zero_horizon_does_not_index_off_the_front() {
        // `--horizon 0` is a legal thing to type; it must clamp, not panic.
        let config = MineConfig {
            horizon: 0,
            ..mine_config()
        };
        let (records, _) = mine_games(&[game_with_a_neutral(1)], config, "unit test");
        let one = MineConfig {
            horizon: 1,
            ..mine_config()
        };
        let (expected, _) = mine_games(&[game_with_a_neutral(1)], one, "unit test");
        assert_eq!(records, expected);
    }

    #[test]
    fn a_zero_cap_takes_nothing() {
        let config = MineConfig {
            max_suspect: 0,
            max_control: 0,
            ..mine_config()
        };
        let (records, _) = mine_games(&[game_with_a_neutral(1)], config, "unit test");
        assert!(records.is_empty(), "{records:#?}");
    }

    #[test]
    fn logsumexp_is_stable_and_correct() {
        let values = [1.0, 2.0, 3.0];
        let naive = values.iter().map(|v: &f64| v.exp()).sum::<f64>().ln();
        assert!((logsumexp(&values) - naive).abs() < 1e-12);
        // A shifted copy must move by exactly the shift.
        let shifted: Vec<f64> = values.iter().map(|v| v + 700.0).collect();
        assert!((logsumexp(&shifted) - (logsumexp(&values) + 700.0)).abs() < 1e-9);
    }

    #[test]
    fn the_advantage_label_is_the_movers_cell_lead() {
        let state = State::new(12, 12, 2).expect("a legal opening");
        // Both seats own exactly their base at the opening.
        assert_eq!(advantage(&state, 1), 0);
        let after = state
            .apply(Action::Move {
                target: Pos::new(1, 1),
            })
            .expect("a legal move");
        assert_eq!(advantage(&after, 1), 1);
        assert_eq!(advantage(&after, 2), -1);
    }
}
