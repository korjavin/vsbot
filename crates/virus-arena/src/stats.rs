//! Win/loss/draw accounting, Wilson intervals, and the sample-size discipline.
//!
//! # Why this module refuses to answer some questions
//!
//! ARCHITECTURE.md invariant 7: offline metrics do not predict strength, and
//! seven documented attempts to shortcut a gauntlet all failed. The predecessors
//! then found the *second* failure mode — reporting a 24-game cell as if it were
//! evidence. `nnue-trainer`'s plan doc says it outright: "screens are for tuning
//! only, never reported" and "never report or gate a 24-game cell".
//!
//! Neither predecessor encoded that in code. Java's `GauntletMatch` has no
//! sample-size logic at all; Go's arena has Wilson but no floor. It was prose,
//! and prose loses to a deadline. So here it is a type: [`Record::verdict`]
//! returns [`Verdict::Informational`] below [`VERDICT_MIN_GAMES`] and the
//! renderer prints that word instead of a claim. You can still get the raw
//! numbers — you just cannot get the harness to call them a result.
//!
//! # Draws
//!
//! Draws are **not** half-wins in the headline, matching Go's `Wilson95`
//! comment: "the superior-engine gate is about demonstrated wins, and this
//! conservative definition cannot hide draws". A draw raises the denominator
//! and not the numerator, so a drawish engine's interval sags rather than
//! sitting at a comfortable 50%.
//!
//! The half-win *pooled score* — the number Java's promotion gate uses
//! (`(W + 0.5·D)/N ≥ 0.55`) — is still reported as [`Record::pooled_score`],
//! because Gate A in `docs/plans/superiority.md` is defined in those terms and
//! this crate has to be able to reproduce it. Two numbers, two purposes, both
//! labelled.

use std::fmt;

/// Games required before a result may gate anything.
///
/// At `p ≈ 0.5` the standard error is `0.5/sqrt(n)`, so 400 games gives
/// ±2.5 pts and the 0.55 promotion gate sits at ~2σ. CLAUDE.md's "gauntlets
/// only (≥400 games)" is this number.
pub const GATE_MIN_GAMES: u32 = 400;

/// Games required before the harness will state a direction at all.
///
/// Below this the interval is wider than any effect worth shipping, so the
/// honest output is the raw tally and the word "informational".
pub const VERDICT_MIN_GAMES: u32 = 100;

/// The pooled half-win score a candidate must reach to pass Gate A.
pub const PROMOTION_THRESHOLD: f64 = 0.55;

/// Two-sided 95% normal quantile, to Go's `arena.Wilson95` precision.
const Z95: f64 = 1.959963984540054;

/// A win/loss/draw tally from one side's perspective.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Record {
    /// Games the focus side won outright.
    pub wins: u32,
    /// Games the focus side lost outright.
    pub losses: u32,
    /// Games with no winner: a territory tie, or a game stopped at the ply cap.
    pub draws: u32,
}

impl Record {
    /// Folds one game outcome in, from the focus side's perspective.
    pub fn add(&mut self, outcome: Outcome) {
        match outcome {
            Outcome::Win => self.wins += 1,
            Outcome::Loss => self.losses += 1,
            Outcome::Draw => self.draws += 1,
        }
    }

    /// Merges another tally into this one.
    pub fn merge(&mut self, other: Record) {
        self.wins += other.wins;
        self.losses += other.losses;
        self.draws += other.draws;
    }

    /// Total games played.
    pub fn games(&self) -> u32 {
        self.wins + self.losses + self.draws
    }

    /// The same tally seen from the opponent's chair.
    pub fn flipped(&self) -> Record {
        Record {
            wins: self.losses,
            losses: self.wins,
            draws: self.draws,
        }
    }

    /// Headline win rate as a percentage: `wins / games`, draws in the
    /// denominator only. `0.0` for an empty record.
    pub fn win_rate(&self) -> f64 {
        if self.games() == 0 {
            return 0.0;
        }
        100.0 * f64::from(self.wins) / f64::from(self.games())
    }

    /// The half-win pooled score in `[0, 1]` — `(W + 0.5·D) / N`, the quantity
    /// Gate A compares against [`PROMOTION_THRESHOLD`]. `0.0` when empty.
    pub fn pooled_score(&self) -> f64 {
        if self.games() == 0 {
            return 0.0;
        }
        (f64::from(self.wins) + 0.5 * f64::from(self.draws)) / f64::from(self.games())
    }

    /// Wins minus losses — Java's `Result.margin()`.
    pub fn margin(&self) -> i64 {
        i64::from(self.wins) - i64::from(self.losses)
    }

    /// The Wilson 95% score interval on the headline win rate, in percent.
    pub fn wilson95(&self) -> Interval {
        wilson95(self.wins, self.games())
    }

    /// What this sample is allowed to be used for.
    pub fn verdict(&self) -> Verdict {
        let games = self.games();
        if games < VERDICT_MIN_GAMES {
            Verdict::Informational
        } else if games < GATE_MIN_GAMES {
            Verdict::Indicative
        } else {
            Verdict::Gateable
        }
    }
}

/// One game's result for the focus side.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// The focus side won.
    Win,
    /// The focus side lost.
    Loss,
    /// Nobody won.
    Draw,
}

/// A binomial confidence interval, in percent.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Interval {
    /// Lower bound, percent.
    pub low: f64,
    /// Upper bound, percent.
    pub high: f64,
}

impl fmt::Display for Interval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{:.1}%, {:.1}%]", self.low, self.high)
    }
}

/// The Wilson score interval for `wins` out of `games`, in percent.
///
/// A direct port of Go's `arena.Wilson95`. Wilson rather than the normal
/// approximation because the arena routinely reports cells near 0% and 100%,
/// where `p ± z·sqrt(p(1-p)/n)` produces bounds outside `[0, 100]` and a zero
/// width at exactly 0 wins — both of which have been read as "certainty" by a
/// reader in a hurry. Wilson stays inside the unit interval and keeps a
/// sensible width at the extremes.
///
/// `games == 0` yields `[0, 0]` rather than NaN, so a report of an aborted run
/// still renders.
pub fn wilson95(wins: u32, games: u32) -> Interval {
    if games == 0 {
        return Interval {
            low: 0.0,
            high: 0.0,
        };
    }
    let n = f64::from(games);
    let p = f64::from(wins) / n;
    let denominator = 1.0 + Z95 * Z95 / n;
    let center = (p + Z95 * Z95 / (2.0 * n)) / denominator;
    let margin = Z95 * ((p * (1.0 - p) + Z95 * Z95 / (4.0 * n)) / n).sqrt() / denominator;
    Interval {
        low: 100.0 * (center - margin),
        high: 100.0 * (center + margin),
    }
}

/// How much weight a sample is entitled to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// Under [`VERDICT_MIN_GAMES`]. Report the tally, claim nothing.
    Informational,
    /// At least [`VERDICT_MIN_GAMES`] but under [`GATE_MIN_GAMES`]. A direction
    /// may be stated with the interval attached; it may not gate a decision.
    Indicative,
    /// At least [`GATE_MIN_GAMES`]. May gate.
    Gateable,
}

impl Verdict {
    /// The label printed next to the headline.
    pub fn label(self) -> &'static str {
        match self {
            Verdict::Informational => "INFORMATIONAL ONLY",
            Verdict::Indicative => "indicative",
            Verdict::Gateable => "gate-eligible",
        }
    }

    /// Whether a result at this sample size may gate a promotion or a claim.
    pub fn may_gate(self) -> bool {
        matches!(self, Verdict::Gateable)
    }

    /// The sentence explaining the label, or `None` when the sample is large
    /// enough that no caveat is owed.
    pub fn caveat(self) -> Option<&'static str> {
        match self {
            Verdict::Informational => Some(concat!(
                "fewer than 100 games: the interval is wider than any effect worth ",
                "shipping. Do not quote this as a strength result.",
            )),
            Verdict::Indicative => Some(concat!(
                "fewer than 400 games: a direction may be read off the interval, but ",
                "this cannot gate a promotion (CLAUDE.md: gauntlets only, >=400 games).",
            )),
            Verdict::Gateable => None,
        }
    }
}

/// A rendered gauntlet result: the tally plus everything a reader needs to know
/// how much to believe it.
#[derive(Clone, Debug)]
pub struct Summary {
    /// Name of the focus side.
    pub side_a: String,
    /// Name of the opponent.
    pub side_b: String,
    /// The tally from `side_a`'s perspective.
    pub record: Record,
    /// Games that ended at the ply cap rather than terminally. Counted as
    /// draws; broken out because a high count means the cap, not the engines,
    /// decided the match.
    pub capped: u32,
    /// Worst per-move deadline overrun observed, in milliseconds. Always 0
    /// outside fixed-time mode.
    pub max_overrun_ms: u64,
    /// Wall-clock seconds the gauntlet took.
    pub elapsed_secs: f64,
}

impl fmt::Display for Summary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let record = self.record;
        let verdict = record.verdict();
        writeln!(f, "{} vs {}", self.side_a, self.side_b)?;
        writeln!(
            f,
            "  W-L-D {}-{}-{} over {} games ({})",
            record.wins,
            record.losses,
            record.draws,
            record.games(),
            verdict.label(),
        )?;
        writeln!(
            f,
            "  win rate {:.1}% (draws not half-wins)  wilson95 {}",
            record.win_rate(),
            record.wilson95(),
        )?;
        writeln!(
            f,
            "  pooled score {:.4} (W+0.5D)/N  margin {:+}",
            record.pooled_score(),
            record.margin(),
        )?;
        if self.capped > 0 {
            writeln!(f, "  turn-capped games: {} (scored as draws)", self.capped)?;
        }
        if self.max_overrun_ms > 0 {
            writeln!(
                f,
                "  worst move deadline overrun: {} ms",
                self.max_overrun_ms
            )?;
        }
        writeln!(f, "  elapsed {:.1}s", self.elapsed_secs)?;
        if let Some(caveat) = verdict.caveat() {
            write!(f, "  NOTE: {caveat}")?;
        } else {
            write!(
                f,
                "  gate A ({:.2} pooled): {}",
                PROMOTION_THRESHOLD,
                if record.pooled_score() >= PROMOTION_THRESHOLD {
                    "PASS"
                } else {
                    "fail"
                }
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reference values come straight from Go's `arena.Wilson95` formula;
    /// this pins the constant and the algebra together.
    #[test]
    fn wilson_matches_the_go_reference() {
        let interval = wilson95(50, 100);
        assert!((interval.low - 40.383).abs() < 0.01, "{interval}");
        assert!((interval.high - 59.617).abs() < 0.01, "{interval}");

        let interval = wilson95(220, 400);
        assert!((interval.low - 50.100).abs() < 0.01, "{interval}");
        assert!((interval.high - 59.805).abs() < 0.01, "{interval}");
    }

    /// The reason Wilson is here rather than the normal approximation: at the
    /// extremes the interval stays inside `[0, 100]` and keeps a real width.
    #[test]
    fn wilson_behaves_at_the_extremes() {
        let none = wilson95(0, 50);
        assert!(none.low.abs() < 1e-9, "{none}");
        assert!(none.high > 0.0 && none.high < 15.0, "{none}");

        let all = wilson95(50, 50);
        assert!(all.low > 85.0 && all.low < 100.0, "{all}");
        assert!((all.high - 100.0).abs() < 1e-6, "{all}");

        let empty = wilson95(0, 0);
        assert_eq!((empty.low, empty.high), (0.0, 0.0));
    }

    /// The headline must not launder draws into wins.
    #[test]
    fn draws_are_not_half_wins_in_the_headline() {
        let record = Record {
            wins: 40,
            losses: 40,
            draws: 20,
        };
        assert_eq!(record.win_rate(), 40.0);
        assert!((record.pooled_score() - 0.5).abs() < 1e-12);
        assert_eq!(record.margin(), 0);
        // The interval sits on the 40% headline, not the 50% pooled score.
        let interval = record.wilson95();
        assert!(interval.low < 40.0 && interval.high > 40.0, "{interval}");
    }

    #[test]
    fn the_sample_size_ladder_is_enforced() {
        let at = |games: u32| {
            Record {
                wins: games / 2,
                losses: games - games / 2,
                draws: 0,
            }
            .verdict()
        };
        assert_eq!(at(0), Verdict::Informational);
        assert_eq!(at(99), Verdict::Informational);
        assert_eq!(at(100), Verdict::Indicative);
        assert_eq!(at(399), Verdict::Indicative);
        assert_eq!(at(400), Verdict::Gateable);
        assert!(!at(399).may_gate());
        assert!(at(400).may_gate());
        assert!(at(399).caveat().is_some());
        assert!(at(400).caveat().is_none());
    }

    /// The word a reader scans for has to actually be in the output.
    #[test]
    fn a_small_sample_prints_informational_only() {
        let summary = Summary {
            side_a: "a".to_owned(),
            side_b: "b".to_owned(),
            record: Record {
                wins: 6,
                losses: 4,
                draws: 0,
            },
            capped: 0,
            max_overrun_ms: 0,
            elapsed_secs: 1.0,
        };
        let text = summary.to_string();
        assert!(text.contains("INFORMATIONAL ONLY"), "{text}");
        assert!(text.contains("Do not quote this"), "{text}");
        assert!(!text.contains("PASS"), "{text}");
    }

    #[test]
    fn flipping_a_record_swaps_wins_and_losses() {
        let record = Record {
            wins: 3,
            losses: 7,
            draws: 2,
        };
        assert_eq!(
            record.flipped(),
            Record {
                wins: 7,
                losses: 3,
                draws: 2
            }
        );
        assert_eq!(record.flipped().flipped(), record);
    }

    #[test]
    fn merging_adds_componentwise() {
        let mut left = Record {
            wins: 1,
            losses: 2,
            draws: 3,
        };
        left.merge(Record {
            wins: 10,
            losses: 20,
            draws: 30,
        });
        assert_eq!(
            left,
            Record {
                wins: 11,
                losses: 22,
                draws: 33
            }
        );
        assert_eq!(left.games(), 66);
    }

    #[test]
    fn adding_outcomes_tallies_them() {
        let mut record = Record::default();
        record.add(Outcome::Win);
        record.add(Outcome::Win);
        record.add(Outcome::Loss);
        record.add(Outcome::Draw);
        assert_eq!(
            record,
            Record {
                wins: 2,
                losses: 1,
                draws: 1
            }
        );
    }

    #[test]
    fn an_empty_record_reports_zero_not_nan() {
        let record = Record::default();
        assert_eq!(record.win_rate(), 0.0);
        assert_eq!(record.pooled_score(), 0.0);
        assert_eq!(record.games(), 0);
    }
}
