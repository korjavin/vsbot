//! Intra-turn time allocation and the visit-based stop rules.
//!
//! # Why a turn budget and not a move budget
//!
//! The server arms a fresh 120 s auto-resign timer per *action*, so the wire
//! imposes no turn budget at all. The binding constraint is human: the owner
//! will not play a bot that takes longer than ~10-15 s over a full three-action
//! turn (owner directive, 2026-08-13). So the thing to divide is the **turn**,
//! and the per-action deadline is a consequence of that division, not an input.
//!
//! # The division
//!
//! A turn is up to three actions and they are not worth the same. Action 1 sets
//! the turn's direction with the widest choice; action 3 is usually a
//! consequence of the first two. [`TurnAllocator`] therefore splits a *bank* —
//! what is left of the turn — with the weights [`ACTION_WEIGHTS`], which give
//! 50% / 30% / 20% of a fully-spent turn.
//!
//! The bank is the mechanism behind "a stable root releases its remainder to the
//! next action": the allocator hands out a share of what is *left*, and
//! [`TurnAllocator::spent`] only deducts what the search actually used. An
//! action that stops early after 1 s of its 6 s share leaves 11 s in the bank,
//! and the next action's share is computed from 11 s rather than from 6 s.
//!
//! # The stop rules
//!
//! [`verdict`] is a pure function of the root's visit counts and the clock, so
//! the rules are unit-testable without a net, a searcher, or a wall clock that
//! has to cooperate:
//!
//! * **early stop** — when the runner-up cannot catch the leader even if it took
//!   *every* remaining simulation, more thinking cannot change the move. Stop
//!   and give the time back to the bank. Saves human-facing latency at zero
//!   strength cost — but only for a lead **this search produced**. A lead a
//!   re-root inherited from the opponent's turn is there from the first sample
//!   and says nothing about whether this action has settled; see [`verdict`].
//! * **extension** — when the target passes with an unstable root (a leader that
//!   changed late, or a top-2 gap too small to trust), keep going toward
//!   [`MoveAllocation::ceiling`]. The bank caps the ceiling, so an extension
//!   borrows from the rest of the turn rather than from the turn bound.

use std::time::{Duration, Instant};

/// Actions in a full turn. The server's `movesLeft` counts down from this.
pub const ACTIONS_PER_TURN: u8 = 3;

/// Relative weight of the 1st, 2nd and 3rd action of a turn.
///
/// Uneven on purpose (superiority.md §2b): action 1 picks the turn's direction
/// from the widest choice and deserves the most. Spent in full, these give
/// 50% / 30% / 20% of the turn budget.
pub const ACTION_WEIGHTS: [u32; ACTIONS_PER_TURN as usize] = [5, 3, 2];

/// Floor on a single action's target.
///
/// A turn whose bank is exhausted (a previous action overran) must still search
/// *something*: answering instantly with the fallback for the rest of the turn
/// would turn one slow action into three bad ones.
pub const MIN_ACTION_BUDGET: Duration = Duration::from_millis(50);

/// How far past its target an unstable root may run, as a percentage.
///
/// 150% of the target, and never more than the bank — so an extension is
/// borrowed from the rest of the turn, never added to the turn bound.
pub const EXTENSION_PERCENT: u32 = 150;

/// What one action of a turn is allowed to spend.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MoveAllocation {
    /// What the search aims to spend. A stable root stops here or earlier.
    pub target: Duration,
    /// The most an *unstable* root may spend. Always `>= target`.
    pub ceiling: Duration,
}

impl MoveAllocation {
    /// A fixed allocation with no room to extend — the `VSBOT_MOVE_MILLIS`
    /// per-action override, and the shape every pre-allocator caller gets.
    pub fn fixed(budget: Duration) -> MoveAllocation {
        MoveAllocation {
            target: budget,
            ceiling: budget,
        }
    }

    /// The target deadline for a search started at `started`.
    pub fn target_deadline(&self, started: Instant) -> Instant {
        started + self.target
    }

    /// The ceiling deadline for a search started at `started`.
    pub fn ceiling_deadline(&self, started: Instant) -> Instant {
        started + self.ceiling
    }
}

/// Splits a turn's wall clock across the turn's actions.
///
/// One allocator per bot; [`TurnAllocator::allocate`] is called once per action
/// the bot is about to search and [`TurnAllocator::spent`] once with what that
/// action actually took.
#[derive(Clone, Copy, Debug)]
pub struct TurnAllocator {
    turn: Duration,
    fixed: Option<Duration>,
    bank: Duration,
    last_moves_left: u8,
    /// Set when the position stopped being ours, so the next allocation is
    /// known to open a fresh turn even if `movesLeft` did not increase.
    ///
    /// It is not redundant with the `movesLeft` rule: `PlaceNeutrals` consumes a
    /// whole turn from `movesLeft == 3`, so two consecutive turns of ours can
    /// both open at 3 with nothing in between to notice.
    turn_ended: bool,
}

impl TurnAllocator {
    /// An allocator that splits `turn` across a turn's actions.
    pub fn new(turn: Duration) -> TurnAllocator {
        TurnAllocator {
            turn,
            fixed: None,
            bank: turn,
            last_moves_left: 0,
            turn_ended: true,
        }
    }

    /// An allocator disabled by a per-action override: every action gets
    /// exactly `budget` and nothing is banked.
    ///
    /// This is what `VSBOT_MOVE_MILLIS` selects. Keeping it as a *mode of the
    /// allocator* rather than a branch at the call site means the override is
    /// impossible to half-apply.
    pub fn fixed(budget: Duration) -> TurnAllocator {
        TurnAllocator {
            turn: budget.saturating_mul(u32::from(ACTIONS_PER_TURN)),
            fixed: Some(budget),
            bank: budget,
            last_moves_left: 0,
            turn_ended: true,
        }
    }

    /// Whether a per-action override is in force, i.e. the allocator is off.
    pub fn is_fixed(&self) -> bool {
        self.fixed.is_some()
    }

    /// The whole-turn budget this allocator divides.
    pub fn turn_budget(&self) -> Duration {
        self.turn
    }

    /// What is left of the current turn.
    pub fn bank(&self) -> Duration {
        self.bank
    }

    /// Records that the turn is over, so the next allocation opens a fresh bank.
    ///
    /// Called from the snapshot path whenever the installed position is not ours
    /// to move.
    pub fn end_turn(&mut self) {
        self.turn_ended = true;
    }

    /// Allocates for an action with `moves_left` actions remaining in the turn.
    ///
    /// `moves_left` is the server's `movesLeft` for the position about to be
    /// searched: 3 for the first action of a turn, 1 for the last.
    pub fn allocate(&mut self, moves_left: u8) -> MoveAllocation {
        if let Some(budget) = self.fixed {
            return MoveAllocation::fixed(budget);
        }
        // A fresh turn either announced itself (`end_turn`) or is implied by
        // `movesLeft` going *up*, which only a turn boundary can do.
        if self.turn_ended || moves_left > self.last_moves_left {
            self.bank = self.turn;
            self.turn_ended = false;
        }
        self.last_moves_left = moves_left;

        let index =
            usize::from(ACTIONS_PER_TURN.saturating_sub(moves_left.clamp(1, ACTIONS_PER_TURN)));
        let remaining_weight: u32 = ACTION_WEIGHTS[index..].iter().sum();
        let share = self
            .bank
            .mul_f64(f64::from(ACTION_WEIGHTS[index]) / f64::from(remaining_weight));

        let target = share.max(MIN_ACTION_BUDGET);
        // The ceiling may borrow from the rest of the turn but never from
        // outside it, so three extended actions still fit the turn bound.
        let ceiling = target
            .mul_f64(f64::from(EXTENSION_PERCENT) / 100.0)
            .min(self.bank)
            .max(target);
        MoveAllocation { target, ceiling }
    }

    /// Deducts what an action actually spent from the turn's bank.
    ///
    /// The remainder of an early-stopped action stays in the bank and is
    /// re-divided by the next [`TurnAllocator::allocate`] — the "stable root
    /// releases its remainder" rule.
    pub fn spent(&mut self, elapsed: Duration) {
        if self.fixed.is_some() {
            return;
        }
        self.bank = self.bank.saturating_sub(elapsed);
    }
}

impl Default for TurnAllocator {
    fn default() -> TurnAllocator {
        TurnAllocator::new(Duration::from_millis(12_000))
    }
}

/// Which visit-based stop rules a search should apply.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct StopPolicy {
    /// Stop as soon as the visit leader is mathematically uncatchable.
    pub early_stop: bool,
    /// Run past the target toward the ceiling while the root looks unstable.
    pub extension: bool,
    /// Top-2 visit gap, as a fraction of the leader's visits, below which the
    /// root counts as unstable.
    pub unstable_gap: f64,
}

impl Default for StopPolicy {
    fn default() -> StopPolicy {
        StopPolicy {
            early_stop: true,
            extension: true,
            unstable_gap: 0.20,
        }
    }
}

impl StopPolicy {
    /// Both rules off: spend exactly the target, every time.
    pub fn off() -> StopPolicy {
        StopPolicy {
            early_stop: false,
            extension: false,
            unstable_gap: 0.0,
        }
    }
}

/// Everything the stop rules read from a search in progress.
///
/// Sampled between simulation slices, never mid-simulation.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct RootProgress {
    /// Visits of the most-visited root action.
    pub leader_visits: u64,
    /// Visits of the second most-visited root action.
    pub runner_up_visits: u64,
    /// Simulations **this search** has run, for the rate estimate.
    ///
    /// Not the tree's cumulative total: on a re-rooted tree the visit counts
    /// above are inherited and this is not, and [`verdict`] depends on exactly
    /// that difference to tell "the search has settled" from "the tree is old".
    pub sims: u64,
    /// Whether the visit leader changed after the halfway mark of the target.
    ///
    /// An early leader change is normal noise; a late one means the search has
    /// not settled and the extension is worth paying for.
    pub leader_changed_late: bool,
}

impl RootProgress {
    /// Reads the top two visit counts out of a root's per-action visit vector.
    pub fn from_visits(visits: &[u32], sims: u64, leader_changed_late: bool) -> RootProgress {
        let mut leader = 0u64;
        let mut runner_up = 0u64;
        for &n in visits {
            let n = u64::from(n);
            if n > leader {
                runner_up = leader;
                leader = n;
            } else if n > runner_up {
                runner_up = n;
            }
        }
        RootProgress {
            leader_visits: leader,
            runner_up_visits: runner_up,
            sims,
            leader_changed_late,
        }
    }

    /// Whether the root is too unsettled to answer on.
    pub fn is_unstable(&self, policy: &StopPolicy) -> bool {
        if self.leader_changed_late {
            return true;
        }
        if self.leader_visits == 0 {
            return true;
        }
        let gap = (self.leader_visits - self.runner_up_visits) as f64 / self.leader_visits as f64;
        gap < policy.unstable_gap
    }
}

/// Why a search stopped, or that it should keep going.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Verdict {
    /// Keep simulating.
    Continue,
    /// The runner-up cannot overtake the leader in the time that is left.
    StopUncatchable,
    /// The target passed with a settled root.
    StopTarget,
    /// The ceiling passed. The last word, always.
    StopCeiling,
}

impl Verdict {
    /// Whether the search must stop.
    pub fn is_stop(self) -> bool {
        self != Verdict::Continue
    }
}

/// The stop rules, as a pure function of the clock and the root's visits.
///
/// `elapsed` is time since the search started. The horizon for the
/// uncatchability test is the **ceiling**, not the target: a search that could
/// still legitimately extend must not be cut short by a rule that assumed it
/// would stop at the target.
pub fn verdict(
    progress: RootProgress,
    elapsed: Duration,
    allocation: MoveAllocation,
    policy: &StopPolicy,
) -> Verdict {
    if elapsed >= allocation.ceiling {
        return Verdict::StopCeiling;
    }
    if policy.early_stop && progress.sims > 0 && !elapsed.is_zero() {
        let horizon = allocation.ceiling.saturating_sub(elapsed);
        let rate = progress.sims as f64 / elapsed.as_secs_f64();
        let still_available = rate * horizon.as_secs_f64();
        let lead = progress.leader_visits - progress.runner_up_visits;
        // Two conditions, and the second one is the fix for bd `vsbot-gei`.
        //
        // The first is the rule itself: the runner-up cannot overtake the
        // leader with the simulations that are left.
        //
        // The second is that the lead must be one *this* search could have
        // produced. Every simulation adds exactly one visit to one root edge,
        // so on a tree this search grew from nothing the lead can never exceed
        // the simulation count and the condition is free — the rule is
        // bit-for-bit what it always was. On a **re-rooted** tree it is not
        // free: the visit counts arrive inherited from the opponent's turn, the
        // lead is enormous from the very first sample, and without this the
        // rule fires before the action has simulated anything at all. That is
        // not "the search has settled", it is "the tree is old", and the two
        // are not the same claim. It ended one pondered action in five inside a
        // fifth of its allocated time on the live soak — the owner's "much
        // faster, seems more stupid" canary verdict, in one line of arithmetic.
        //
        // Requiring `lead <= sims` makes a warm action spend at least half its
        // ceiling before the rule may fire (if `sims >= lead > rate * horizon`
        // and `sims = rate * elapsed`, then `elapsed > horizon`), which is the
        // right shape: the skipped simulations are not wasted under tree reuse.
        // The session re-roots into the action we play, so they sharpen exactly
        // the subtree the next action of the turn inherits.
        if lead as f64 > still_available && lead <= progress.sims {
            return Verdict::StopUncatchable;
        }
    }
    if elapsed < allocation.target {
        return Verdict::Continue;
    }
    if policy.extension && progress.is_unstable(policy) {
        return Verdict::Continue;
    }
    Verdict::StopTarget
}

#[cfg(test)]
mod tests {
    use super::*;

    const TURN: Duration = Duration::from_millis(12_000);

    fn millis(value: u64) -> Duration {
        Duration::from_millis(value)
    }

    // ------------------------------------------------------------ allocation

    #[test]
    fn a_fully_spent_turn_splits_uneven_with_the_first_action_largest() {
        let mut clock = TurnAllocator::new(TURN);

        let first = clock.allocate(3);
        assert_eq!(first.target, millis(6_000));
        clock.spent(first.target);

        let second = clock.allocate(2);
        assert_eq!(second.target, millis(3_600));
        clock.spent(second.target);

        let third = clock.allocate(1);
        assert_eq!(third.target, millis(2_400));
        clock.spent(third.target);

        // Uneven, largest first, and the whole turn is spent — not a millisecond
        // more, which is the UX bound the owner set.
        assert!(first.target > second.target && second.target > third.target);
        assert_eq!(
            first.target + second.target + third.target,
            TURN,
            "the three shares must add up to the turn budget exactly"
        );
        assert_eq!(clock.bank(), Duration::ZERO);
    }

    #[test]
    fn a_stable_root_releases_its_remainder_to_the_next_action() {
        let mut clock = TurnAllocator::new(TURN);

        let first = clock.allocate(3);
        assert_eq!(first.target, millis(6_000));
        // Early stop after a second: 11 s is left, not 6 s.
        clock.spent(millis(1_000));
        assert_eq!(clock.bank(), millis(11_000));

        let second = clock.allocate(2);
        assert_eq!(
            second.target,
            millis(6_600),
            "the released remainder must be re-divided, not forfeited"
        );
        clock.spent(second.target);

        let third = clock.allocate(1);
        assert_eq!(third.target, millis(4_400));
        assert_eq!(
            millis(1_000) + second.target + third.target,
            TURN,
            "the turn bound still holds after a release"
        );
    }

    #[test]
    fn an_overrun_shrinks_the_rest_of_the_turn_instead_of_extending_it() {
        let mut clock = TurnAllocator::new(TURN);
        clock.allocate(3);
        clock.spent(millis(11_800));

        let second = clock.allocate(2);
        assert_eq!(second.target, millis(120));
        clock.spent(millis(120));

        let third = clock.allocate(1);
        assert_eq!(third.target, millis(80));
    }

    #[test]
    fn an_exhausted_bank_still_leaves_a_floor_to_search_with() {
        let mut clock = TurnAllocator::new(TURN);
        clock.allocate(3);
        clock.spent(millis(60_000));
        assert_eq!(clock.bank(), Duration::ZERO);

        let second = clock.allocate(2);
        assert_eq!(second.target, MIN_ACTION_BUDGET);
        assert_eq!(
            second.ceiling, MIN_ACTION_BUDGET,
            "an empty bank has nothing to lend an extension"
        );
    }

    #[test]
    fn the_ceiling_extends_past_the_target_but_never_past_the_bank() {
        let mut clock = TurnAllocator::new(TURN);
        let first = clock.allocate(3);
        assert_eq!(first.ceiling, millis(9_000));
        assert!(first.ceiling <= clock.bank());

        clock.spent(first.target);
        let second = clock.allocate(2);
        assert_eq!(second.ceiling, millis(5_400));

        clock.spent(second.target);
        let third = clock.allocate(1);
        // The last action's share *is* the bank, so there is nothing to borrow.
        assert_eq!(third.ceiling, third.target);
    }

    #[test]
    fn a_new_turn_reopens_the_bank() {
        let mut clock = TurnAllocator::new(TURN);
        clock.allocate(3);
        clock.spent(millis(6_000));
        clock.allocate(2);
        clock.spent(millis(3_600));

        // movesLeft going up is a turn boundary all by itself.
        let next_turn = clock.allocate(3);
        assert_eq!(next_turn.target, millis(6_000));
    }

    /// `PlaceNeutrals` consumes a whole turn from `movesLeft == 3`, so two of
    /// our turns in a row can both open at 3 with no decrement between them.
    /// Only [`TurnAllocator::end_turn`] can see that boundary.
    #[test]
    fn back_to_back_turns_at_moves_left_three_still_reopen_the_bank() {
        let mut clock = TurnAllocator::new(TURN);
        let neutral_turn = clock.allocate(3);
        clock.spent(neutral_turn.target);
        assert_eq!(clock.bank(), millis(6_000));

        clock.end_turn();
        let next_turn = clock.allocate(3);
        assert_eq!(
            next_turn.target,
            millis(6_000),
            "the opponent's turn in between must have reopened the bank"
        );
    }

    /// A repeated snapshot for the same action (a resync replaying
    /// `game_state`) must not hand out a second full turn.
    #[test]
    fn a_repeated_moves_left_does_not_reopen_the_bank() {
        let mut clock = TurnAllocator::new(TURN);
        let first = clock.allocate(3);
        clock.spent(first.target);
        let repeat = clock.allocate(3);
        assert_eq!(repeat.target, millis(3_000), "half of the 6 s that is left");
    }

    #[test]
    fn a_per_action_override_disables_the_allocator_entirely() {
        let mut clock = TurnAllocator::fixed(millis(1_000));
        assert!(clock.is_fixed());
        for moves_left in [3, 2, 1, 3, 2, 1] {
            let allocation = clock.allocate(moves_left);
            assert_eq!(allocation.target, millis(1_000));
            assert_eq!(
                allocation.ceiling,
                millis(1_000),
                "an override has no extension room: it is an exact per-action budget"
            );
            clock.spent(millis(5_000));
        }
    }

    // ------------------------------------------------------------ stop rules

    fn allocation(target: u64, ceiling: u64) -> MoveAllocation {
        MoveAllocation {
            target: millis(target),
            ceiling: millis(ceiling),
        }
    }

    #[test]
    fn top_two_visits_are_read_off_the_visit_vector() {
        let progress = RootProgress::from_visits(&[3, 40, 7, 40, 1], 91, false);
        assert_eq!(progress.leader_visits, 40);
        assert_eq!(progress.runner_up_visits, 40);
        assert_eq!(progress.sims, 91);

        let empty = RootProgress::from_visits(&[], 0, false);
        assert_eq!(empty.leader_visits, 0);
        assert_eq!(empty.runner_up_visits, 0);
    }

    #[test]
    fn an_uncatchable_leader_stops_the_search_early() {
        // 7000 sims in 3.5 s is 2000/s, so at most 5000 more before the 6 s
        // ceiling. A 6000-visit lead cannot be closed by 5000 simulations even
        // if every one of them went to the runner-up.
        let progress = RootProgress::from_visits(&[6_500, 500], 7_000, false);
        assert_eq!(
            verdict(
                progress,
                millis(3_500),
                allocation(4_000, 6_000),
                &StopPolicy::default()
            ),
            Verdict::StopUncatchable
        );
        // Switching the rule off spends the rest of the target anyway.
        assert_eq!(
            verdict(
                progress,
                millis(3_500),
                allocation(4_000, 6_000),
                &StopPolicy {
                    early_stop: false,
                    ..StopPolicy::default()
                }
            ),
            Verdict::Continue
        );
    }

    /// bd `vsbot-gei`: a lead this search did not produce is not a reason to
    /// stop it.
    ///
    /// The numbers are a real line from the instrumented soak — a pondered
    /// action that inherited 7326 visits and had run 128 simulations of its own
    /// when the rule fired, 37 ms into a 739 ms allocation.
    #[test]
    fn an_inherited_lead_does_not_end_an_action_before_it_has_searched() {
        let warm = RootProgress::from_visits(&[7_448, 3], 128, false);
        assert_eq!(
            verdict(
                warm,
                millis(37),
                allocation(739, 1_108),
                &StopPolicy::default()
            ),
            Verdict::Continue,
            "the 7445-visit lead came from the opponent's turn; 128 simulations of our own \
             is not evidence that this action has settled"
        );

        // The same tree once the action has actually produced the lead it is
        // invoking. `sims >= lead` and the runner-up cannot catch up, so the
        // rule is earned and fires.
        let earned = RootProgress::from_visits(&[7_448, 3], 8_000, false);
        assert_eq!(
            verdict(
                earned,
                millis(600),
                allocation(739, 1_108),
                &StopPolicy::default()
            ),
            Verdict::StopUncatchable
        );
    }

    /// The new condition must be *free* for a search that grew its own tree.
    ///
    /// Every simulation adds one visit to one root edge, so a cold search's
    /// lead can never exceed its simulation count and the guard can never
    /// change a verdict. A deployment that does not ponder must see exactly the
    /// behaviour it saw before.
    #[test]
    fn the_guard_cannot_change_a_verdict_for_a_search_that_grew_its_own_tree() {
        for (leader, runner_up, elapsed_ms) in [
            (6_500u32, 500u32, 3_500u64),
            (4_000, 2_000, 3_500),
            (900, 100, 4_000),
            (600, 500, 1_000),
            (1_050, 1_000, 4_000),
        ] {
            // A cold search's simulation count is at least the visits on the
            // tree; anything from there up leaves the guard satisfied.
            let sims = u64::from(leader) + u64::from(runner_up);
            let progress = RootProgress::from_visits(&[leader, runner_up], sims, false);
            let lead = u64::from(leader - runner_up);
            assert!(
                lead <= progress.sims,
                "a cold search cannot have a lead of {lead} after {sims} simulations"
            );
            // And the verdict is decided by the horizon arithmetic alone.
            let allocation = allocation(4_000, 6_000);
            let horizon = allocation.ceiling - millis(elapsed_ms);
            let rate = sims as f64 / (elapsed_ms as f64 / 1_000.0);
            let uncatchable = lead as f64 > rate * horizon.as_secs_f64();
            let ruling = verdict(
                progress,
                millis(elapsed_ms),
                allocation,
                &StopPolicy::default(),
            );
            assert_eq!(
                ruling == Verdict::StopUncatchable,
                uncatchable,
                "the guard changed the verdict for a cold root {leader}/{runner_up}"
            );
        }
    }

    #[test]
    fn a_catchable_runner_up_keeps_the_search_running() {
        // Same clock, but the lead is 100 visits and thousands of simulations
        // are still to come.
        let progress = RootProgress::from_visits(&[600, 500], 1_100, false);
        assert_eq!(
            verdict(
                progress,
                millis(1_000),
                allocation(4_000, 6_000),
                &StopPolicy::default()
            ),
            Verdict::Continue
        );
    }

    /// The uncatchability horizon is the ceiling, not the target. Measuring it
    /// against the target would cut short searches that were about to extend.
    #[test]
    fn the_uncatchable_horizon_reaches_to_the_ceiling() {
        // 6000 sims in 3.5 s is ~1714/s: 857 more before the 4 s target, 4285
        // before the 6 s ceiling. A 2000-visit lead is uncatchable within the
        // target and catchable within the ceiling — so the rule must not fire.
        let progress = RootProgress::from_visits(&[4_000, 2_000], 6_000, false);
        assert_eq!(
            verdict(
                progress,
                millis(3_500),
                allocation(4_000, 6_000),
                &StopPolicy::default()
            ),
            Verdict::Continue
        );
        // The same root against a search with no room to extend *is* settled.
        assert_eq!(
            verdict(
                progress,
                millis(3_500),
                allocation(4_000, 4_000),
                &StopPolicy::default()
            ),
            Verdict::StopUncatchable
        );
    }

    #[test]
    fn a_settled_root_stops_at_the_target() {
        // Fast enough that the runner-up could still theoretically catch up, so
        // the stop is the target rule and not the uncatchable one.
        let progress = RootProgress::from_visits(&[900, 100], 100_000, false);
        assert_eq!(
            verdict(
                progress,
                millis(4_000),
                allocation(4_000, 6_000),
                &StopPolicy::default()
            ),
            Verdict::StopTarget
        );
    }

    #[test]
    fn an_unstable_root_extends_past_the_target() {
        // Top-2 gap of 5%, well inside the 20% instability threshold.
        let close = RootProgress::from_visits(&[1_050, 1_000], 2_050, false);
        assert!(close.is_unstable(&StopPolicy::default()));
        assert_eq!(
            verdict(
                close,
                millis(4_000),
                allocation(4_000, 6_000),
                &StopPolicy::default()
            ),
            Verdict::Continue
        );

        // A late leader change is instability even with a comfortable gap.
        let flipped = RootProgress::from_visits(&[900, 100], 100_000, true);
        assert!(flipped.is_unstable(&StopPolicy::default()));
        assert_eq!(
            verdict(
                flipped,
                millis(4_000),
                allocation(4_000, 6_000),
                &StopPolicy::default()
            ),
            Verdict::Continue
        );
    }

    #[test]
    fn the_ceiling_is_the_last_word_even_for_an_unstable_root() {
        let unstable = RootProgress::from_visits(&[1_010, 1_000], 2_010, true);
        assert_eq!(
            verdict(
                unstable,
                millis(6_000),
                allocation(4_000, 6_000),
                &StopPolicy::default()
            ),
            Verdict::StopCeiling
        );
        assert_eq!(
            verdict(
                unstable,
                millis(9_999),
                allocation(4_000, 6_000),
                &StopPolicy::default()
            ),
            Verdict::StopCeiling
        );
    }

    #[test]
    fn the_extension_rule_can_be_switched_off() {
        let unstable = RootProgress::from_visits(&[1_010, 1_000], 2_010, true);
        assert_eq!(
            verdict(
                unstable,
                millis(4_000),
                allocation(4_000, 6_000),
                &StopPolicy::off()
            ),
            Verdict::StopTarget
        );
    }

    #[test]
    fn a_search_that_has_run_nothing_yet_never_stops_early() {
        let nothing = RootProgress::default();
        assert_eq!(
            verdict(
                nothing,
                Duration::ZERO,
                allocation(4_000, 6_000),
                &StopPolicy::default()
            ),
            Verdict::Continue
        );
        assert_eq!(
            verdict(
                nothing,
                millis(10),
                allocation(4_000, 6_000),
                &StopPolicy::default()
            ),
            Verdict::Continue
        );
    }

    /// The whole point of the exercise: whatever the rules do, three actions
    /// never add up to more than the turn bound.
    #[test]
    fn no_sequence_of_extensions_can_break_the_turn_bound() {
        let mut clock = TurnAllocator::new(TURN);
        let mut total = Duration::ZERO;
        for moves_left in [3, 2, 1] {
            let allocation = clock.allocate(moves_left);
            assert!(allocation.ceiling <= clock.bank());
            // Every action extends all the way to its ceiling.
            clock.spent(allocation.ceiling);
            total += allocation.ceiling;
        }
        assert!(
            total <= TURN,
            "three maximally-extended actions spent {total:?}, over the {TURN:?} bound"
        );
    }
}
