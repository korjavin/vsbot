//! Seed derivation and the opening-diversity RNG.
//!
//! # Why the derivation looks like this
//!
//! A gauntlet is a set of *pairs*: games `2k` and `2k+1` are the same opening
//! played with the colours swapped, so a side's first-mover advantage cancels
//! exactly. That only works if both games of a pair get the **same** seed, and
//! it only produces independent evidence if different pairs — and different
//! *runs* — get openings that do not overlap.
//!
//! Java's `GauntletMatch.deriveGameSeed` is ported verbatim:
//!
//! ```text
//! derive(seed, game) = mix64(seed ^ (0x9E3779B97F4A7C15 * (game / 2 + 1)))
//! ```
//!
//! Two details are load-bearing and neither is obvious:
//!
//! * `game / 2` is the **pair** index, so the two colours of a pair derive the
//!   same value. Using `game` would give each colour its own opening and throw
//!   away the whole point of pairing.
//! * The golden-ratio constant multiplies the pair index *before* the XOR
//!   (Java's precedence), then the whole thing goes through the SplitMix64
//!   finalizer. Seeding a stream with `seed + k` instead — the obvious thing —
//!   is what caused `nnue-trainer-riy`: two runs launched with nearby base
//!   seeds replayed overlapping openings and their "independent" results were
//!   correlated. A full-avalanche mixer on a golden-ratio-strided input makes
//!   base seeds 1 and 2 as unrelated as any other pair.
//!
//! # Deliberate deviation: the stream, not the derivation
//!
//! Java draws its epsilon coin from `java.util.Random` (a 48-bit LCG). This
//! crate uses SplitMix64 for the stream as well. Bit-identical replay of a Java
//! gauntlet is unreachable regardless — the engines differ, so the games differ
//! from the first non-random ply — and an unverified reimplementation of
//! `java.util.Random` (no JVM on the build hosts to check it against) would be
//! a liability with no payoff. What is actually load-bearing is preserved: one
//! stream per game seeded from the pair seed, draws taken in a fixed order, and
//! the same stream for both colours of a pair.

/// The SplitMix64 finalizer: a bijection on `u64` with full avalanche.
///
/// This is the mixing half of SplitMix64 with no state increment, matching
/// Java's `GauntletMatch.mix64`.
pub fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The golden-ratio odd constant SplitMix64 strides by.
pub const GOLDEN_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

/// The per-game opening seed, shared by both colours of a pair.
///
/// See the module docs: `game / 2` is the pair index, and the `+ 1` keeps pair
/// 0 from multiplying by zero (which would hand the raw base seed straight to
/// the mixer for every run's first pair).
pub fn derive_game_seed(seed: u64, game: u64) -> u64 {
    mix64(seed ^ GOLDEN_GAMMA.wrapping_mul((game / 2) + 1))
}

/// A SplitMix64 stream.
///
/// Deliberately duplicated rather than borrowed from `virus-mcts`: that one is
/// part of the *engine's* self-play contract and its draw sequence is pinned by
/// net-parity fixtures. The arena's opening stream must be free to change
/// without touching an engine gate.
#[derive(Clone, Copy, Debug)]
pub struct Rng {
    state: u64,
}

impl Rng {
    /// A stream seeded with `seed`.
    pub fn new(seed: u64) -> Rng {
        Rng { state: seed }
    }

    /// The next 64 bits.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(GOLDEN_GAMMA);
        mix64(self.state)
    }

    /// A uniform draw in `[0, 1)`, using the top 53 bits so every value is
    /// exactly representable.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// A uniform index below `len`, or `None` when `len == 0`.
    ///
    /// Lemire's multiply-shift. The modulo bias it replaces is irrelevant at
    /// these bounds, but the method is branch-free and needs no rejection loop,
    /// which keeps the number of draws per call fixed at one — and a fixed draw
    /// count per ply is what keeps a pair's two colours on the same stream
    /// position.
    pub fn below(&mut self, len: usize) -> Option<usize> {
        if len == 0 {
            return None;
        }
        let draw = u128::from(self.next_u64());
        Some(((draw * len as u128) >> 64) as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_colours_of_a_pair_share_a_seed() {
        for pair in 0..64u64 {
            let even = derive_game_seed(7, pair * 2);
            let odd = derive_game_seed(7, pair * 2 + 1);
            assert_eq!(even, odd, "pair {pair} must replay one opening");
        }
    }

    #[test]
    fn different_pairs_get_different_seeds() {
        let seeds: std::collections::HashSet<u64> = (0..512u64)
            .map(|pair| derive_game_seed(1, pair * 2))
            .collect();
        assert_eq!(seeds.len(), 512);
    }

    /// The `nnue-trainer-riy` regression: runs launched at base seeds 1, 2, 3
    /// must not replay each other's openings. With a `seed + k` scheme the
    /// first 511 pairs of run 1 would be pairs 0..510 of run 2.
    #[test]
    fn nearby_base_seeds_do_not_overlap() {
        let mut all = std::collections::HashSet::new();
        for base in 1..=8u64 {
            for pair in 0..256u64 {
                assert!(
                    all.insert(derive_game_seed(base, pair * 2)),
                    "base seed {base} pair {pair} collided with an earlier run"
                );
            }
        }
    }

    /// SplitMix64's finalizer is a bijection, so 0 is the only input that can
    /// produce 0 — a cheap guard against a typo'd shift or constant.
    #[test]
    fn mix64_is_a_bijection_on_a_sample() {
        assert_eq!(mix64(0), 0);
        let sample: std::collections::HashSet<u64> = (0..4096u64).map(mix64).collect();
        assert_eq!(sample.len(), 4096);
    }

    #[test]
    fn below_stays_in_range_and_covers_it() {
        let mut rng = Rng::new(12345);
        let mut seen = [0usize; 5];
        for _ in 0..10_000 {
            let index = rng.below(5).expect("non-empty");
            assert!(index < 5);
            seen[index] += 1;
        }
        assert!(seen.iter().all(|count| *count > 1500), "{seen:?}");
        assert_eq!(Rng::new(1).below(0), None);
    }

    #[test]
    fn next_f64_is_a_unit_interval_draw() {
        let mut rng = Rng::new(99);
        let mut below_eps = 0;
        for _ in 0..10_000 {
            let value = rng.next_f64();
            assert!((0.0..1.0).contains(&value), "{value}");
            if value < 0.15 {
                below_eps += 1;
            }
        }
        // 1500 expected, SD ~36; a 10-SD band catches a broken shift without
        // ever flaking.
        assert!((1140..=1860).contains(&below_eps), "{below_eps}");
    }

    #[test]
    fn a_stream_is_reproducible() {
        let first: Vec<u64> = (0..8)
            .scan(Rng::new(42), |rng, _| Some(rng.next_u64()))
            .collect();
        let second: Vec<u64> = (0..8)
            .scan(Rng::new(42), |rng, _| Some(rng.next_u64()))
            .collect();
        assert_eq!(first, second);
    }
}
