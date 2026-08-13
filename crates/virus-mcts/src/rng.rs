//! Seeded randomness for the searcher.
//!
//! CLAUDE.md: *all engine randomness is seeded and deterministic; production
//! play paths take no RNG unless explicitly configured*. Everything stochastic
//! in this crate — Dirichlet root noise and temperature sampling — runs through
//! this one generator, which is seeded from [`crate::Config::seed`] and never
//! from the clock or the OS.
//!
//! SplitMix64 rather than a bigger generator: the consumers here draw at most a
//! few hundred values per search, the algorithm is four lines with no state
//! beyond a `u64`, and `virus-arena` already derives its seeds with the same
//! mixer, so a run is reproducible end to end.

/// A seeded SplitMix64 generator.
#[derive(Clone, Debug)]
pub struct Rng {
    state: u64,
    /// The unused half of the last Marsaglia-polar pair.
    spare: Option<f64>,
}

impl Rng {
    /// Creates a generator from a seed. Equal seeds give equal streams.
    pub fn new(seed: u64) -> Rng {
        Rng {
            state: seed,
            spare: None,
        }
    }

    /// Next raw 64 bits.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`, using the top 53 bits — the same construction as
    /// Java's `Random.nextDouble`, so the value stream is comparable even
    /// though the underlying generator differs.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Standard normal, Marsaglia polar method with the usual cached spare.
    pub fn next_gaussian(&mut self) -> f64 {
        if let Some(value) = self.spare.take() {
            return value;
        }
        loop {
            let u = 2.0 * self.next_f64() - 1.0;
            let v = 2.0 * self.next_f64() - 1.0;
            let s = u * u + v * v;
            if s >= 1.0 || s == 0.0 {
                continue;
            }
            let factor = (-2.0 * s.ln() / s).sqrt();
            self.spare = Some(v * factor);
            return u * factor;
        }
    }

    /// Marsaglia-Tsang gamma variate, shape only.
    ///
    /// A Dirichlet draw normalises its gammas, so the scale cancels and only
    /// the shape matters — which is why the searcher can use this directly for
    /// root noise.
    ///
    /// # Panics
    /// Panics on a non-positive shape.
    pub fn gamma(&mut self, alpha: f64) -> f64 {
        assert!(alpha > 0.0, "gamma shape must be positive, got {alpha}");
        if alpha < 1.0 {
            // Boost: gamma(a) = gamma(a + 1) * U^(1/a).
            return self.gamma(alpha + 1.0) * self.next_f64().powf(1.0 / alpha);
        }
        let d = alpha - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let x = self.next_gaussian();
            let v = 1.0 + c * x;
            if v <= 0.0 {
                continue;
            }
            let v = v * v * v;
            let u = self.next_f64();
            if u < 1.0 - 0.0331 * x * x * x * x || u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
                return d * v;
            }
        }
    }
}
