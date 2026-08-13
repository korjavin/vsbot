//! Integer-exact port of GoBot's hand-tuned static evaluation.
//!
//! Source of truth: `virusgame/backend/search/evaluate.go` (`StaticEval`),
//! cross-checked against the already-verified Java port
//! `nnue-trainer/.../search/eval/HandTunedEval.java`. This is a literal
//! translation — same structure, same operation order, same integer division
//! points — so scores are integer-identical to GoBot on
//! `fixtures/gobot_staticeval_parity.jsonl`. Do **not** "improve" the math:
//! every divergence is a bug, never a rounding difference. CLAUDE.md treats a
//! parity break as a hard failure.
//!
//! # Two roles that must not be conflated
//!
//! * `score_player` — whose utility comes back (GoBot's `s.root`).
//! * `current_player` — the mover, read by the tempo terms (GoBot's
//!   `state.CurrentPlayer()`), which here lives on the [`State`] itself.
//!
//! At an opponent-to-move leaf these differ. [`evaluate`] takes the score
//! player explicitly and always reads tempo from `state.current_player()`, so
//! the two can never be accidentally unified.
//!
//! # Weights are a parameter, not a global
//!
//! Go kept the active weights in a package-level `var activeEvalParams`, which
//! forced its SPSA tuner to run serially. Here [`EvalParams`] is a plain struct
//! passed by reference; a tuner can evaluate many perturbed vectors in
//! parallel.
//!
//! # FAILED IDEAS — do not re-attempt
//!
//! Verbatim from `evaluate.go` lines 7-43. Each of these was measured and
//! killed; re-deriving them from first principles is a guaranteed waste.
//!
//! `spaceRaceWeight` scales the Voronoi space-race term. Chosen by the vs-ai2.34
//! sweep: peak of the 2..48 curve, 69.5% vs the MobilityAttacker strangler at
//! n=200 (w95 [63,75]); 48 was already past the peak at 61.2%.
//!
//! vs-ai2.38 tried an ungated own-max-cutLoss fragility penalty here to stop the
//! width-1 opening tendril. The Task-3 sweep killed it: the opening only flips to
//! width-2 at weight >= 380, but the strength gate is already broken at weight 12
//! (legacy 62.5% < 85%) and the constructed width-2 invariant fails at every
//! weight > 0, and any nonzero penalty diverges from the frozen origin-main eval
//! oracle. No weight fixes the opening without wrecking strength — ship nothing.
//! See docs/plans/completed/20260717-vs-ai2.38-ungated-fragility.md Task 3 for the curve.
//!
//! vs-ai2.38 attempt 2 (ROBUST SPACE) also null: reseed spaceRace from each
//! player's post-worst-cut surviving component (drop the max-cutLoss articulation
//! cell) so a width-1 tendril claims ~no deep space. It changed the opening but
//! did NOT flip the constructed-cut width-1 gate (cutter still wins 27), broke the
//! width-2 invariant (cutter went 35-loss -> 69-win), and regressed 4 tactical
//! tests (defense/reconnect/capture) — discounting fragile space also makes the
//! bot abandon reconnecting its own cuts. Reverted. Abandon the "discount fragile
//! space" family; a structural opening constraint is the next avenue.
//! See docs/plans/completed/20260717-vs-ai2.38-robust-space.md.
//!
//! vs-ai2.47 (exchange-ratio blindness) also null for the static-term family: a
//! retaliation penalty on own capturable-next-turn Normals (the threatened/
//! threatenedLoss signal) cannot flip a constructed 1-for-2 without breaking a
//! favorable 2-for-1. Symmetric form is directionally WRONG (capturing into
//! contact exposes the opponent more, so the penalty rewards the capture: the
//! bad-trade score RISES with weight, -2730 at w=0 -> +14647 at w=2000).
//! Mover-only form is budget-fragile: it only fires at contact leaves, deeper
//! search resolves the exchange (threatened=0) and reverts to the capture at
//! 100k+ nodes even at w=3000, while production reaches depth 6-8. The
//! fully-resolved 1-for-2 already nets ~-388 in the material terms, so the
//! mispricing lives strictly at intermediate contact leaves — the standard cure
//! is quiescence in search, not an eval constant. Gates for the pattern live in
//! arena/exchange_evidence_test.go + arena/exchange_gate_test.go.
//! See docs/plans/20260717-vs-ai2.47-exchange-ratio.md Task 4 for the sweep data.

#![deny(missing_docs)]
#![deny(missing_debug_implementations)]

mod analysis;
mod workspace;

use virus_core::{Player, State, MAX_PLAYERS};

pub use workspace::EvalWorkspace;

use analysis::{adjacent_connected, ratio, Metrics};

/// Score type.
///
/// `i64` because Go's `int` is 64-bit on every platform GoBot ships on, and the
/// `ratio` intermediates (`value * 1000`) genuinely exceed 32 bits on large
/// boards. Never a float: float evaluation would make byte-exact parity with
/// the Go/Java oracles impossible.
pub type Score = i64;

/// Terminal-position magnitude (Go's `mateScore`). A won seat scores
/// `MATE_SCORE`, a lost one `-MATE_SCORE`, and an eliminated seat's *raw* score
/// is pinned to `-MATE_SCORE / 2`.
pub const MATE_SCORE: Score = 1_000_000_000;

/// Default weight of the Voronoi space-race term (Go's `spaceRaceWeight`).
/// See the FAILED IDEAS block: 32 is the measured peak of the 2..48 sweep.
pub const SPACE_RACE_WEIGHT: i64 = 32;

/// The flat vector of hand-set evaluation weights (Go's `EvalParams`).
///
/// [`EvalParams::default`] reproduces `defaultEvalParams()` exactly, so the
/// production path is byte-equivalent to the Go literals.
///
/// Deliberately **not** a global: Go's package-level `activeEvalParams` forced
/// its SPSA tuner to run one perturbation at a time. Pass this by reference.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EvalParams {
    /// Weight of base-connected owned cells (area-normalized).
    pub connected: i64,
    /// Weight of owned `Normal` cells (area-normalized).
    pub normal: i64,
    /// Weight of owned `Fortified` cells (area-normalized).
    pub fortified: i64,
    /// Weight of distinct legal targets (area-normalized).
    pub mobility: i64,
    /// Weight of capturable enemy `Normal` targets (area-normalized).
    pub captures: i64,
    /// Penalty weight for owned cells severed from the base
    /// (normalized by *owned*, not area).
    pub disconnected: i64,
    /// Raw bonus per own base-adjacent connected cell.
    pub base_exits: i64,
    /// Raw bonus per empty-or-enemy-normal cell adjacent to the base.
    pub base_openings: i64,
    /// Raw bonus per base-adjacent own `Fortified` cell.
    pub base_anchors: i64,
    /// Penalty per enemy `Normal` beside a threatened base, times tempo.
    pub base_threat: i64,
    /// Multiplier on the threatened-articulation subtree-loss ratio.
    pub threatened_loss_mult: i64,
    /// Multiplier on the threatened-cell-count ratio.
    pub threatened_mult: i64,
    /// Weight of first-reach empty cells in the Voronoi partition.
    pub space_race: i64,
    /// Flat penalty when the base has neither exit nor opening.
    pub sealed_base_penalty: i64,
    /// Flat bonus while the once-per-game neutral placement is unspent.
    pub neutral_unused_bonus: i64,
    /// Per remaining action, for the mover only.
    pub moves_left_tempo: i64,
    /// Flat bonus per opponent articulation cell we stand beside.
    pub predatory_cut_base: i64,
    /// Divisor applied to the predatory cut-loss ratio.
    pub predatory_cut_loss_div: i64,
}

impl Default for EvalParams {
    /// The hand-tuned literals from Go's `defaultEvalParams()`.
    fn default() -> EvalParams {
        EvalParams {
            connected: 10,
            normal: 30,
            fortified: 6,
            mobility: 1,
            captures: 1,
            disconnected: 1,
            base_exits: 180,
            base_openings: 80,
            base_anchors: 240,
            base_threat: 650,
            threatened_loss_mult: 1,
            threatened_mult: 1,
            space_race: SPACE_RACE_WEIGHT,
            sealed_base_penalty: 5000,
            neutral_unused_bonus: 20,
            moves_left_tempo: 12,
            predatory_cut_base: 150,
            predatory_cut_loss_div: 2,
        }
    }
}

/// Utility for `score_player`, higher = better.
///
/// `score_player` is the utility index (GoBot's `s.root`); the tempo terms read
/// `state.current_player()`, which at an opponent-to-move leaf is a *different*
/// seat. Keeping them separate is a hard requirement — conflating them was a
/// live parity bug in the Java port's first attempt.
///
/// # Panics
/// Panics when `score_player` is outside `1..=4`.
pub fn evaluate(
    state: &State,
    score_player: Player,
    params: &EvalParams,
    workspace: &mut EvalWorkspace,
) -> Score {
    assert!(
        (1..=MAX_PLAYERS as Player).contains(&score_player),
        "score_player must be 1..=4, got {score_player}"
    );
    evaluate_all(state, params, workspace)[score_player as usize - 1]
}

/// [`evaluate`] with the default weights and a throwaway workspace.
///
/// Convenience for tests and one-shot callers only. In a search loop, hold one
/// [`EvalWorkspace`] per searcher and call [`evaluate`] — this function
/// allocates on every call.
pub fn static_eval(state: &State, score_player: Player) -> Score {
    let mut workspace = EvalWorkspace::new();
    evaluate(state, score_player, &EvalParams::default(), &mut workspace)
}

/// Every seat's utility in one pass, indexed by `seat - 1`.
///
/// One pass rather than four: the connectivity masks, the Voronoi BFS and the
/// per-seat Tarjan analyses are shared, and the predatory-cut term needs every
/// seat's articulation set anyway.
///
/// Seats that do not exist on this board (a 2-player game has no seat 3 or 4)
/// and eliminated seats both count as inactive and score `-MATE_SCORE / 2`.
pub fn evaluate_all(
    state: &State,
    params: &EvalParams,
    workspace: &mut EvalWorkspace,
) -> [Score; MAX_PLAYERS] {
    let mut utility = [0 as Score; MAX_PLAYERS];
    if state.game_over() {
        for (seat, value) in utility.iter_mut().enumerate() {
            *value = if state.winner() == seat as Player + 1 {
                MATE_SCORE
            } else {
                -MATE_SCORE
            };
        }
        return utility;
    }

    let size = state.cell_count();
    workspace.ensure(size);
    let rows = state.rows();
    let cols = state.cols();

    // Destructured so the four buffer groups can be borrowed independently:
    // `analyze` reads every seat's connectivity while writing one seat's
    // articulation set.
    let EvalWorkspace {
        connected,
        articulation,
        cut_loss,
        scratch,
    } = workspace;

    // Connectivity for every active seat, then the shared space-race BFS. Both
    // must run before any per-seat analysis: `threatened` reads *opponents'*
    // masks, and the Voronoi partition is seeded from all seats at once.
    for (seat, mask) in connected.iter_mut().enumerate() {
        if state.active(seat as Player + 1) {
            analysis::connected_into(
                state,
                seat as Player + 1,
                &mut scratch.queue[..size],
                &mut mask[..size],
            );
        } else {
            mask[..size].fill(false);
        }
    }
    let space = analysis::space_race(state, connected, scratch, size);

    let mut metrics = [Metrics::default(); MAX_PLAYERS];
    let mut raw = [0 as Score; MAX_PLAYERS];
    let mut active_count: i64 = 0;
    let area = (rows * cols) as i64;

    for seat in 0..MAX_PLAYERS {
        let player = seat as Player + 1;
        if !state.active(player) {
            raw[seat] = -MATE_SCORE / 2;
            continue;
        }
        active_count += 1;
        let m = analysis::analyze(
            state,
            player,
            connected,
            &mut articulation[seat][..size],
            &mut cut_loss[seat][..size],
            scratch,
            analysis::Tempo {
                current: state.current_player(),
                moves_left: state.moves_left() as i64,
            },
        );
        metrics[seat] = m;
        let owned = m.normal + m.fortified + 1; // include the base
        let p = params;
        // Operation order is load-bearing: every `normalized`/`ratio` is a
        // separate truncating integer division, so regrouping the sum changes
        // the score. Mirrors evaluate.go:282-290 term for term.
        raw[seat] = normalized(m.connected, area, p.connected)
            + normalized(m.normal, area, p.normal)
            + normalized(m.fortified, area, p.fortified)
            + normalized(m.mobility, area, p.mobility)
            + normalized(m.captures, area, p.captures)
            - normalized(m.disconnected, owned, p.disconnected)
            + p.base_exits * m.base_exits
            + p.base_openings * m.base_openings
            + p.base_anchors * m.base_anchors
            - p.base_threat * m.base_threat * m.threat_tempo
            - m.threat_tempo
                * p.threatened_loss_mult
                * ratio(m.threatened_loss, m.connected.max(1))
            - m.threat_tempo * p.threatened_mult * ratio(m.threatened, m.connected.max(1))
            + normalized(space[seat], area, p.space_race);
        if m.base_exits + m.base_openings == 0 {
            raw[seat] -= p.sealed_base_penalty;
        }
        if !state.neutral_used(player) {
            raw[seat] += p.neutral_unused_bonus;
        }
        if state.current_player() == player {
            raw[seat] += state.moves_left() as i64 * p.moves_left_tempo;
        }
    }

    // Predatory cut: reward standing beside an opponent's articulation cell,
    // scaled by how much of its component the cut severs.
    for seat in 0..MAX_PLAYERS {
        if !state.active(seat as Player + 1) {
            continue;
        }
        for opponent in 0..MAX_PLAYERS {
            if opponent == seat || !state.active(opponent as Player + 1) {
                continue;
            }
            for index in 0..size {
                if articulation[opponent][index]
                    && adjacent_connected(rows, cols, index, &connected[seat])
                {
                    let loss = cut_loss[opponent][index] as i64;
                    raw[seat] += params.predatory_cut_base
                        + ratio(loss, metrics[opponent].connected.max(1))
                            / params.predatory_cut_loss_div;
                }
            }
        }
    }

    // Utility: own raw minus the mean of the *active* opponents' raw. Inactive
    // seats keep their raw `-MATE_SCORE / 2` unchanged.
    for seat in 0..MAX_PLAYERS {
        if !state.active(seat as Player + 1) {
            utility[seat] = raw[seat];
            continue;
        }
        let mut opponents: Score = 0;
        for (other, value) in raw.iter().enumerate() {
            if other != seat && state.active(other as Player + 1) {
                opponents += value;
            }
        }
        utility[seat] = if active_count > 1 {
            // Go and Rust integer division both truncate toward zero, so a
            // negative opponent sum rounds identically.
            raw[seat] - opponents / (active_count - 1)
        } else {
            raw[seat]
        };
    }
    utility
}

/// `value * weight * 1000 / denominator`, guarded (Go's `normalized`).
///
/// The multiplications happen *before* the division — the whole point of the
/// helper. Doing it the other way loses the fraction and breaks parity.
#[inline]
fn normalized(value: i64, denominator: i64, weight: i64) -> i64 {
    if value <= 0 || denominator <= 0 || weight <= 0 {
        return 0;
    }
    value * weight * 1000 / denominator
}

#[cfg(test)]
mod tests {
    use super::*;
    use analysis::neighbors;

    #[test]
    fn default_params_match_go_literals() {
        let p = EvalParams::default();
        assert_eq!(p.connected, 10);
        assert_eq!(p.normal, 30);
        assert_eq!(p.space_race, 32);
        assert_eq!(p.sealed_base_penalty, 5000);
        assert_eq!(p.predatory_cut_loss_div, 2);
    }

    #[test]
    fn normalized_multiplies_before_dividing() {
        // 7 * 30 * 1000 / 144 = 1458, whereas (7 / 144) * 30 * 1000 = 0.
        assert_eq!(normalized(7, 144, 30), 1458);
        assert_eq!(normalized(0, 144, 30), 0);
        assert_eq!(normalized(-1, 144, 30), 0);
        assert_eq!(normalized(7, 0, 30), 0);
        assert_eq!(normalized(7, 144, 0), 0);
    }

    #[test]
    fn inactive_seats_score_half_a_mate() {
        // A 2-player start position: seats 3 and 4 do not exist.
        let state = State::new(12, 12, 2).expect("12x12 two-player board");
        let scores = evaluate_all(&state, &EvalParams::default(), &mut EvalWorkspace::new());
        assert_eq!(scores[2], -MATE_SCORE / 2);
        assert_eq!(scores[3], -MATE_SCORE / 2);
        // Symmetric start: both live seats have the same utility.
        assert_eq!(scores[0] + scores[1], 0);
    }

    #[test]
    fn workspace_reuse_is_score_invariant() {
        let state = State::new(12, 12, 2).expect("12x12 two-player board");
        let params = EvalParams::default();
        let mut shared = EvalWorkspace::new();
        let first = evaluate_all(&state, &params, &mut shared);
        for _ in 0..8 {
            assert_eq!(evaluate_all(&state, &params, &mut shared), first);
        }
        assert_eq!(
            evaluate_all(&state, &params, &mut EvalWorkspace::new()),
            first
        );
    }

    #[test]
    fn neighbors_scan_order_matches_go() {
        // Go scans row-1..row+1 then col-1..col+1, skipping the cell itself.
        let mut out = [0usize; 8];
        let count = neighbors(3, 3, 4, &mut out); // centre of a 3x3 board
        assert_eq!(count, 8);
        assert_eq!(out, [0, 1, 2, 3, 5, 6, 7, 8]);

        let count = neighbors(3, 3, 0, &mut out); // top-left corner
        assert_eq!(count, 3);
        assert_eq!(&out[..3], &[1, 3, 4]);
    }
}
