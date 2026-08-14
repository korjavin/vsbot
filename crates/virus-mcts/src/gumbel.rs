//! Gumbel / sequential-halving root selection (superiority.md §2d item 3).
//!
//! **Self-play only.** [`Config::play`](crate::Config::play) leaves
//! [`Config::gumbel`](crate::Config::gumbel) at `None` and the production bin
//! builds its configuration from it, exactly as it does for Dirichlet root
//! noise — the two exploration mechanisms are structurally unreachable from
//! play mode for the same reason and are pinned by the same tests.
//!
//! # What the recipe is
//!
//! At 192-256 simulations, PUCT spends a large share of its budget re-deciding
//! between the two or three actions the prior already liked, and the visit
//! counts it produces are a noisy target: at 192 sims over ~34 actions, the
//! difference between the second and third action is a handful of visits.
//! Gumbel MuZero (Danihelka et al., 2022) replaces the root's PUCT rule with a
//! bandit that spends the budget on purpose:
//!
//! 1. Draw one Gumbel variate `g(a)` per legal root action.
//! 2. Take the `m` actions with the largest `g(a) + logit(a)` — the standard
//!    Gumbel-top-k trick, which makes that set an exact sample of `m` actions
//!    drawn without replacement from the prior.
//! 3. Split the simulation budget into `ceil(log2(m))` phases. Every surviving
//!    candidate gets an equal share of the phase, then the worse half is cut,
//!    ranked by `g(a) + logit(a) + sigma(q(a))`.
//! 4. Play the last survivor. That action is the argmax of
//!    `g(a) + logit(a) + sigma(q(a))`, which is what makes the whole thing a
//!    *policy improvement*: the paper's theorem is that this selection never
//!    has a lower expected value than sampling the prior, for any number of
//!    simulations including one.
//!
//! `sigma` is the monotone transform
//!
//! ```text
//! sigma(q) = (c_visit + max_b N(b)) * c_scale * rescale(completedQ)
//! ```
//!
//! where `rescale` is min-max onto `[0, 1]` across the root's actions. Both
//! halves matter and the second is easy to drop:
//!
//! * The `max_b N(b)` factor grows with the search, so early and noisy `q`
//!   estimates cannot outvote the prior while late, well-measured ones can.
//! * The **min-max rescale** is what makes `c_scale` a constant rather than a
//!   per-project tuning knob. Without it `sigma` is proportional to however
//!   wide this engine's leaf values happen to spread, which here is set by an
//!   arbitrary `tanh` divisor ([`crate::DEFAULT_VALUE_SCALE`]) — and a wider
//!   spread would silently drive the improved policy to one-hot. Measured
//!   here: with raw `[-1, 1]` values and `c_scale = 1` the target *was*
//!   effectively one-hot.
//!
//! `c_visit = 50`, `c_scale = 0.1`, rescaling on: DeepMind's `mctx`
//! (`qtransform_completed_by_mix_value`), which is the reference
//! implementation of the paper. Deviating from it here would have meant
//! re-tuning two constants against a gauntlet, which is not what this bead is
//! for.
//!
//! Interior nodes are untouched: they keep ordinary PUCT. Only the root's
//! choice of edge and the root's final answer change.
//!
//! # Frames
//!
//! Every `q` here is in the **root mover's** frame. The tree stores `w` in the
//! absolute frame (ARCHITECTURE.md invariant 1), so the conversion is the same
//! single `sign(root)` multiplication that `select` applies, and it happens in
//! exactly one place per formula. A Gumbel implementation that forgot it would
//! rank player 2's candidates by how good they are for player 1 — and would
//! still produce plausible-looking visit counts.
//!
//! # Determinism
//!
//! The Gumbel variates are drawn once, at construction, from the searcher's
//! seeded [`Rng`](crate::Rng). Halving is a total order (score first, edge
//! index as tie-break) over a `Vec`, never a `HashMap`, so a run is
//! reproducible bit for bit from `(position, config, net)`.

use crate::rng::Rng;

/// Default candidate-set size.
///
/// Sixteen: the root of a developed 12x12 position offers ~34 actions, so `m =
/// 16` keeps roughly the top half of the prior in play, and `ceil(log2(16)) =
/// 4` phases divide a 192-simulation budget into 3 visits per candidate in the
/// first phase and 48 for the two finalists — which is the shape the paper's
/// experiments use (`m` around `n/12`).
pub const DEFAULT_GUMBEL_M: u16 = 16;

/// Default `c_visit` in `sigma` — `mctx`'s `maxvisit_init`.
pub const DEFAULT_GUMBEL_C_VISIT: f64 = 50.0;

/// Default `c_scale` in `sigma` — `mctx`'s `value_scale`.
///
/// `0.1` is the reference constant **and it is only meaningful alongside the
/// min-max rescale**: it multiplies a completed `Q` that has already been
/// mapped onto `[0, 1]`, so the whole `sigma` term spans `(c_visit + max N) *
/// 0.1` logits between the root's best and worst action, about 10-15 at a
/// 192-simulation budget. Applied to raw `[-1, 1]` values instead it would
/// span five to ten times that and the improved policy would be one-hot.
pub const DEFAULT_GUMBEL_C_SCALE: f64 = 0.1;

/// Denominator floor in the min-max rescale, `mctx`'s `epsilon`. Reached when
/// every completed `Q` is equal, which is exactly the state before the first
/// simulation backs up.
pub const GUMBEL_RESCALE_EPSILON: f64 = 1e-8;

/// Smallest prior a logit is taken of.
///
/// `ln(0)` is `-inf`, and `-inf + gumbel` is still `-inf`, which would make an
/// action unrankable rather than merely unlikely. A masked softmax can
/// underflow to exactly zero for an action the net hates, so the floor is not
/// hypothetical.
const MIN_PRIOR: f64 = 1e-30;

/// Root-only Gumbel selection settings. **Self-play only.**
///
/// `sims` is the budget the sequential-halving schedule is planned against, and
/// it has to be stated up front because the whole point of the schedule is to
/// decide *in advance* how the budget is split across phases. A search that
/// runs fewer simulations than `sims` simply stops mid-schedule (the answer is
/// still the argmax over whatever survived); one that runs more spends the
/// surplus on the final survivor.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GumbelConfig {
    /// Candidate-set size: how many of the legal root actions the Gumbel-top-k
    /// draw keeps. Clamped down to the number of legal actions.
    pub m: u16,
    /// The simulation budget the phase schedule is planned for.
    pub sims: u32,
    /// `c_visit` in `sigma`.
    pub c_visit: f64,
    /// `c_scale` in `sigma`.
    pub c_scale: f64,
}

impl Default for GumbelConfig {
    fn default() -> GumbelConfig {
        GumbelConfig {
            m: DEFAULT_GUMBEL_M,
            sims: 192,
            c_visit: DEFAULT_GUMBEL_C_VISIT,
            c_scale: DEFAULT_GUMBEL_C_SCALE,
        }
    }
}

/// The sequential-halving schedule for one root, plus the Gumbel draw it is
/// ranked by.
///
/// Owned by the searcher and consulted on every descent that starts at the
/// root. Interior descents never see it.
#[derive(Clone, Debug)]
pub(crate) struct GumbelPlan {
    /// One Gumbel(0,1) variate per **root edge index**, drawn once.
    gumbel: Vec<f64>,
    /// `ln(prior)` per root edge index. The prior is a softmax, so this is the
    /// net's logits up to a constant, and a constant is invisible to both the
    /// top-k draw and the improved policy's softmax.
    logits: Vec<f64>,
    /// Root edge indices still in contention, best-first.
    alive: Vec<usize>,
    /// Simulations each surviving candidate gets in the current phase.
    per_action: u32,
    /// Simulations handed out in the current phase.
    given: u32,
    /// Simulations handed out over the whole plan.
    scheduled: u32,
    /// Halvings done so far.
    phase: u32,
    /// Halvings the schedule was planned for, `ceil(log2(m))`.
    rounds: u32,
    /// Budget the schedule was planned against.
    budget: u32,
}

/// `ln(prior)`, floored at [`MIN_PRIOR`] so a zeroed prior stays rankable.
///
/// A softmax is shift-invariant, so this recovers the net's logits up to a
/// constant — and a constant is invisible to both the Gumbel-top-k draw and the
/// improved policy's own softmax.
pub(crate) fn logits_from_prior(prior: &[f32]) -> Vec<f64> {
    prior
        .iter()
        .map(|p| f64::from(*p).max(MIN_PRIOR).ln())
        .collect()
}

impl GumbelPlan {
    /// Draws the Gumbel variates and picks the top-`m` candidate set.
    ///
    /// `prior` is the root's prior over its legal actions, in edge order.
    pub(crate) fn new(prior: &[f32], config: &GumbelConfig, rng: &mut Rng) -> GumbelPlan {
        let k = prior.len();
        let logits = logits_from_prior(prior);
        // Drawn for **every** action, in edge order, so the stream a plan
        // consumes depends only on the action count — not on which actions the
        // top-k happens to keep.
        let gumbel: Vec<f64> = (0..k).map(|_| gumbel_variate(rng)).collect();

        let m = usize::from(config.m.max(1)).min(k.max(1));
        let mut alive: Vec<usize> = (0..k).collect();
        // Descending by `g + logit`, edge index breaking ties: the Gumbel-top-k
        // draw. Ties are astronomically unlikely with a continuous variate and
        // are ordered anyway, because "deterministic" is a contract here.
        alive.sort_by(|&x, &y| {
            let (sx, sy) = (gumbel[x] + logits[x], gumbel[y] + logits[y]);
            sy.total_cmp(&sx).then(x.cmp(&y))
        });
        alive.truncate(m);

        let rounds = rounds_for(m);
        let budget = config.sims.max(1);
        let mut plan = GumbelPlan {
            gumbel,
            logits,
            alive,
            per_action: 1,
            given: 0,
            scheduled: 0,
            phase: 0,
            rounds,
            budget,
        };
        plan.per_action = plan.phase_share();
        plan
    }

    /// Simulations per candidate in the current phase.
    ///
    /// The *remaining* budget over the *remaining* phases rather than a share
    /// fixed at construction: integer division truncates, and re-deriving it
    /// each phase is what stops the truncation from accumulating into a final
    /// phase that never runs.
    fn phase_share(&self) -> u32 {
        let alive = self.alive.len().max(1) as u32;
        let left = self.budget.saturating_sub(self.scheduled);
        let phases_left = self.rounds.saturating_sub(self.phase).max(1);
        (left / (phases_left * alive)).max(1)
    }

    /// Whether the current phase is used up and a halving is due.
    pub(crate) fn needs_halving(&self) -> bool {
        self.alive.len() > 1 && self.given >= self.per_action * self.alive.len() as u32
    }

    /// Simulations left before the next halving decision, or [`u32::MAX`] once
    /// a single candidate remains.
    ///
    /// The searcher shortens its batch to this, so a halving is always decided
    /// on statistics that have actually backed up rather than on a phase
    /// boundary that fell inside a batch.
    pub(crate) fn phase_remaining(&self) -> u32 {
        if self.alive.len() <= 1 {
            return u32::MAX;
        }
        (self.per_action * self.alive.len() as u32).saturating_sub(self.given)
    }

    /// The root edge the next descent must take.
    ///
    /// **Round-robin over the candidates, not `per_action` in a row each.** The
    /// two orders hand out identical totals and the difference looks cosmetic;
    /// it is not, because of leaf batching. A round collects
    /// [`crate::Config::batch_size`] descents before any of them backs up, and
    /// a descent that reaches an already-collected node reuses its pending
    /// evaluation instead of queueing a second one. So `per_action` consecutive
    /// descents into a *fresh* candidate all land on the same newly created
    /// child: the first expands it, the rest reuse it, and the phase spends
    /// `per_action` simulations to learn what one net forward already said.
    /// At the tuned batch size that wasted about a sixth of the whole budget in
    /// the first phase, where every candidate is fresh by definition.
    ///
    /// Round-robin puts `min(batch, |alive|)` *different* candidates in the
    /// first batch, so each pays for its own forward; by the time the cycle
    /// comes back around their children are expanded and the descents continue
    /// past them into genuinely new positions, decorrelated by the ordinary
    /// virtual loss. It is also what `mctx` does — it selects the considered
    /// action with the fewest visits, which with nothing in flight *is*
    /// round-robin.
    ///
    /// # Panics
    /// Panics when a halving is due; the searcher performs it first.
    pub(crate) fn take(&mut self) -> usize {
        assert!(
            !self.needs_halving(),
            "halve the plan before taking from it"
        );
        let slot = if self.alive.len() == 1 {
            0
        } else {
            self.given as usize % self.alive.len()
        };
        self.given += 1;
        self.scheduled += 1;
        self.alive[slot.min(self.alive.len() - 1)]
    }

    /// Cuts the worse half of the candidate set by `scores`, then re-plans the
    /// next phase.
    pub(crate) fn halve(&mut self, scores: &[f64]) {
        self.alive.sort_by(|&x, &y| {
            scores[y]
                .partial_cmp(&scores[x])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(x.cmp(&y))
        });
        self.alive.truncate(self.alive.len().div_ceil(2));
        self.phase += 1;
        self.given = 0;
        self.per_action = self.phase_share();
    }

    /// The action to play: the argmax of `g + logit + sigma(q)` over what is
    /// still alive.
    ///
    /// After a completed schedule that set is a single action and this is a
    /// lookup. It is an argmax rather than `alive[0]` so that a search stopped
    /// early — fewer simulations than `sims`, a deadline, a caller that never
    /// ran the plan out — still answers with the best candidate it measured
    /// instead of whichever one the last completed halving happened to leave in
    /// front.
    pub(crate) fn choice(&self, scores: &[f64]) -> usize {
        let mut best = self.alive[0];
        for &edge in &self.alive[1..] {
            if scores[edge] > scores[best] {
                best = edge;
            }
        }
        best
    }

    /// `g(a) + logit(a)` for a root edge — the ranking before any search.
    pub(crate) fn gumbel_logit(&self, edge: usize) -> f64 {
        self.gumbel[edge] + self.logits[edge]
    }

    /// `ln(prior)` for every root edge.
    pub(crate) fn logits(&self) -> &[f64] {
        &self.logits
    }

    /// Root edges still in contention, best-first.
    #[cfg(test)]
    pub(crate) fn alive(&self) -> &[usize] {
        &self.alive
    }
}

/// `ceil(log2(m))`, at least 1: the number of halvings that reduce `m`
/// candidates to one.
fn rounds_for(m: usize) -> u32 {
    if m <= 1 {
        return 1;
    }
    (usize::BITS - (m - 1).leading_zeros()).max(1)
}

/// One Gumbel(0,1) variate, `-ln(-ln U)`.
///
/// `U` is redrawn on an exact `0.0`, which [`Rng::next_f64`] can produce (it is
/// uniform on `[0, 1)`) and which would give `+inf`. One rejection in 2^53
/// draws costs nothing and removes an infinity from the ranking.
fn gumbel_variate(rng: &mut Rng) -> f64 {
    loop {
        let u = rng.next_f64();
        if u > 0.0 {
            return -(-u.ln()).ln();
        }
    }
}

/// Min-max rescale onto `[0, 1]`, `mctx`'s `_rescale_qvalues`.
///
/// An all-equal input (every action the same completed value, which is the
/// state of a root before its first backup) divides by
/// [`GUMBEL_RESCALE_EPSILON`] instead of by zero, and the result is all-zero —
/// `sigma` contributes nothing and the ranking is the prior's, which is
/// exactly right when nothing has been measured.
pub(crate) fn rescale(values: &mut [f64]) {
    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let span = (max - min).max(GUMBEL_RESCALE_EPSILON);
    for value in values.iter_mut() {
        *value = (*value - min) / span;
    }
}

/// Softmax over `logits`.
pub(crate) fn softmax(logits: &[f64]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut out: Vec<f64> = logits.iter().map(|l| (l - max).exp()).collect();
    let sum: f64 = out.iter().sum();
    if sum <= 0.0 || !sum.is_finite() {
        let uniform = 1.0 / out.len().max(1) as f32;
        return vec![uniform; out.len()];
    }
    for value in out.iter_mut() {
        *value /= sum;
    }
    out.iter().map(|value| *value as f32).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rounds_is_ceil_log2() {
        assert_eq!(rounds_for(0), 1);
        assert_eq!(rounds_for(1), 1);
        assert_eq!(rounds_for(2), 1);
        assert_eq!(rounds_for(3), 2);
        assert_eq!(rounds_for(4), 2);
        assert_eq!(rounds_for(5), 3);
        assert_eq!(rounds_for(16), 4);
        assert_eq!(rounds_for(17), 5);
    }

    fn plan(k: usize, m: u16, sims: u32, seed: u64) -> GumbelPlan {
        let prior = vec![1.0 / k as f32; k];
        let config = GumbelConfig {
            m,
            sims,
            ..GumbelConfig::default()
        };
        GumbelPlan::new(&prior, &config, &mut Rng::new(seed))
    }

    /// The whole point of the schedule: the budget is spent on the candidate
    /// set, split evenly inside a phase, and halved between phases.
    #[test]
    fn the_schedule_spends_the_budget_in_halving_phases() {
        let mut plan = plan(34, 16, 192, 7);
        let mut visits = [0u32; 34];
        let mut alive_sizes = vec![plan.alive().len()];
        // Rank by edge index so the halving is a fixed, checkable order.
        let scores: Vec<f64> = (0..34).map(|a| -(a as f64)).collect();
        for _ in 0..192 {
            if plan.needs_halving() {
                plan.halve(&scores);
                alive_sizes.push(plan.alive().len());
            }
            visits[plan.take()] += 1;
        }
        // Four phases of 48 simulations each, the candidate set halving between
        // them. The last cut is due exactly as the budget runs out, which is
        // what "the schedule is planned for `sims`" means.
        assert_eq!(alive_sizes, vec![16, 8, 4, 2]);
        assert!(plan.needs_halving(), "the budget ends on a phase boundary");
        plan.halve(&scores);
        assert_eq!(plan.alive().len(), 1);
        assert_eq!(visits.iter().sum::<u32>(), 192);
        // Only the candidate set is ever visited.
        assert_eq!(visits.iter().filter(|v| **v > 0).count(), 16);
    }

    /// Every candidate must be measured at least once before the first cut,
    /// however small the budget: a phase share of zero would halve on noise.
    #[test]
    fn a_starved_budget_still_gives_every_candidate_a_visit() {
        let mut plan = plan(34, 16, 4, 7);
        let scores: Vec<f64> = (0..34).map(|a| -(a as f64)).collect();
        let mut visits = [0u32; 34];
        for _ in 0..16 {
            if plan.needs_halving() {
                plan.halve(&scores);
            }
            visits[plan.take()] += 1;
        }
        assert_eq!(visits.iter().filter(|v| **v > 0).count(), 16);
    }

    #[test]
    fn a_surplus_budget_goes_to_the_last_survivor() {
        let mut plan = plan(8, 4, 16, 11);
        let scores: Vec<f64> = (0..8).map(|a| -(a as f64)).collect();
        let mut last = 0;
        let mut visits = [0u32; 8];
        for _ in 0..64 {
            if plan.needs_halving() {
                plan.halve(&scores);
            }
            last = plan.take();
            visits[last] += 1;
        }
        assert_eq!(plan.alive().len(), 1);
        assert_eq!(plan.alive()[0], last);
        assert!(visits[last] > 16, "the survivor absorbs the surplus");
        assert_eq!(plan.phase_remaining(), u32::MAX);
    }

    /// `m` above the action count is clamped, not an out-of-bounds candidate.
    #[test]
    fn m_is_clamped_to_the_legal_actions() {
        let plan = plan(3, 64, 32, 3);
        assert_eq!(plan.alive().len(), 3);
    }

    #[test]
    fn the_draw_follows_the_seed() {
        let a = plan(34, 16, 192, 99);
        let b = plan(34, 16, 192, 99);
        let c = plan(34, 16, 192, 100);
        assert_eq!(a.alive(), b.alive());
        assert_ne!(a.alive(), c.alive(), "a different seed is a different draw");
    }

    /// A degenerate prior is the case `ln(0)` would break.
    #[test]
    fn a_zero_prior_is_ranked_rather_than_infinite() {
        let prior = vec![0.0f32, 1.0, 0.0];
        let plan = GumbelPlan::new(&prior, &GumbelConfig::default(), &mut Rng::new(5));
        assert!(plan.logits().iter().all(|l| l.is_finite()));
        assert_eq!(plan.alive().len(), 3);
    }

    #[test]
    fn softmax_is_a_distribution_and_survives_a_flat_input() {
        let p = softmax(&[1.0, 2.0, 3.0]);
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(p[2] > p[1] && p[1] > p[0]);
        let flat = softmax(&[0.0; 4]);
        assert!(flat.iter().all(|v| (v - 0.25).abs() < 1e-6));
    }
}
