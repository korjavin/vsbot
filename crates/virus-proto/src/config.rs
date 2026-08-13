//! Client configuration.
//!
//! Plain data, constructed by the `vsbot` binary from the environment. Nothing
//! in this crate reads `std::env` — CLAUDE.md keeps env parsing in exactly one
//! place so a deployment can be reasoned about from a single function.

use crate::clock::{StopPolicy, TurnAllocator};
use std::time::Duration;

/// Everything the client needs to run.
#[derive(Clone, Debug)]
pub struct BotConfig {
    /// Base WebSocket URL, e.g. `ws://localhost:8080/ws`. The `bot=true` and
    /// `namePrefix` query parameters are appended by [`crate::connect_url`].
    pub backend_url: String,
    /// Prefixed onto the server-assigned bot name. Empty means "no prefix".
    pub name_prefix: String,
    /// Wall-clock budget for a whole three-action turn, split across its actions
    /// by [`TurnAllocator`].
    ///
    /// This — not the per-action figure — is the number the owner's UX bound is
    /// expressed in (10-15 s per turn, owner directive 2026-08-13). The server's
    /// 120 s per-action timer is a failsafe, never a budget.
    pub turn_budget: Duration,
    /// Per-action override. `Some` **disables the allocator**: every action gets
    /// exactly this, and nothing is banked or released.
    ///
    /// `None` is the deployed default and means "allocate [`BotConfig::turn_budget`]".
    pub move_budget: Option<Duration>,
    /// Which visit-based stop rules the engine applies inside its allocation.
    pub stop_policy: StopPolicy,
    /// Whether to think on the opponent's positions during their turn.
    ///
    /// Off by default: pondering is a behaviour change the owner judges, so it
    /// ships canary-first (superiority.md Gate C).
    pub ponder: bool,
    /// How long one pondering step may run before the session parks.
    ///
    /// A cap on memory as much as on time: the tree grows with every simulation
    /// and the production host shares CPU with the nightly trainer window.
    pub ponder_budget: Duration,
    /// Grace on top of an action's ceiling before the client stops waiting for
    /// the engine and plays its pre-selected fallback.
    pub fallback_grace: Duration,
    /// Whether this instance initiates games (challenger mode).
    pub challenger: bool,
    /// How often the challenger's timer fires. The timer is the *sole* send
    /// driver: the Java predecessor challenged reactively off `users_update`
    /// and spammed the lobby.
    pub challenge_interval: Duration,
    /// Board size a challenge asks for.
    pub challenge_rows: usize,
    /// Board size a challenge asks for.
    pub challenge_cols: usize,
    /// First reconnect delay; doubles up to [`BotConfig::reconnect_max`].
    pub reconnect_min: Duration,
    /// Reconnect backoff ceiling.
    pub reconnect_max: Duration,
    /// How long a session must last before its failure is treated as a fresh
    /// fault rather than a continuation of the outage — i.e. before the
    /// reconnect backoff resets to [`BotConfig::reconnect_min`].
    pub stable_session: Duration,
    /// How long an accepted-but-unstarted challenge may hold the bot out of the
    /// pool before it heals back to idle.
    pub pending_game_grace: Duration,
    /// Seed for the challenger's target choice and initial jitter. `None` takes
    /// the seed from the clock. No engine path consumes this.
    pub rng_seed: Option<u64>,
}

impl Default for BotConfig {
    fn default() -> BotConfig {
        BotConfig {
            backend_url: "ws://localhost:8080/ws".to_owned(),
            name_prefix: String::new(),
            turn_budget: Duration::from_millis(12_000),
            move_budget: None,
            stop_policy: StopPolicy::default(),
            ponder: false,
            ponder_budget: Duration::from_secs(30),
            fallback_grace: Duration::from_millis(500),
            challenger: false,
            challenge_interval: Duration::from_secs(300),
            challenge_rows: 12,
            challenge_cols: 12,
            reconnect_min: Duration::from_secs(1),
            reconnect_max: Duration::from_secs(30),
            stable_session: Duration::from_secs(30),
            pending_game_grace: Duration::from_secs(15),
            rng_seed: None,
        }
    }
}

impl BotConfig {
    /// The allocator this configuration implies: a per-action override when
    /// [`BotConfig::move_budget`] is set, the turn splitter otherwise.
    ///
    /// One function so the override can never be half-applied.
    pub fn allocator(&self) -> TurnAllocator {
        match self.move_budget {
            Some(budget) => TurnAllocator::fixed(budget),
            None => TurnAllocator::new(self.turn_budget),
        }
    }

    /// One line describing the time budget, for the startup banner.
    pub fn budget_summary(&self) -> String {
        match self.move_budget {
            Some(budget) => format!(
                "budget=fixed {}ms/action (allocator disabled by the per-action override)",
                budget.as_millis()
            ),
            None => format!(
                "budget={}ms/turn split 50/30/20 across up to 3 actions (early-stop={}, extension={})",
                self.turn_budget.as_millis(),
                self.stop_policy.early_stop,
                self.stop_policy.extension,
            ),
        }
    }
}

/// Builds the dial URL: `backend_url` plus `bot=true` and an optional
/// `namePrefix`, joined with `?` or `&` as the base URL requires.
pub fn connect_url(config: &BotConfig) -> String {
    let separator = if config.backend_url.contains('?') {
        '&'
    } else {
        '?'
    };
    let mut url = format!("{}{separator}bot=true", config.backend_url);
    if !config.name_prefix.is_empty() {
        url.push_str("&namePrefix=");
        url.push_str(&percent_encode(&config.name_prefix));
    }
    url
}

/// Percent-encodes a query-string value (RFC 3986 unreserved set kept).
fn percent_encode(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for byte in value.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(byte as char);
            }
            _ => out.push_str(&format!("%{byte:02X}")),
        }
    }
    out
}

/// SplitMix64: the challenger's seeded, deterministic RNG.
///
/// Not an engine path — it only picks a challenge target and jitters the first
/// timer tick — but seeded anyway so a self-sparring run reproduces.
#[derive(Clone, Debug)]
pub struct Rng(u64);

impl Rng {
    /// Seeds the generator.
    pub fn new(seed: u64) -> Rng {
        Rng(seed)
    }

    /// The next 64 bits.
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A uniform index in `0..len`, or `None` when `len` is zero.
    pub fn below(&mut self, len: usize) -> Option<usize> {
        if len == 0 {
            return None;
        }
        Some((self.next_u64() % len as u64) as usize)
    }

    /// A uniform fraction of `span`, used for the challenger's start jitter.
    pub fn fraction_of(&mut self, span: Duration) -> Duration {
        let nanos = span.as_nanos().max(1) as u64;
        Duration::from_nanos(self.next_u64() % nanos)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn connect_url_appends_bot_flag_and_escaped_prefix() {
        let config = BotConfig {
            backend_url: "ws://host/ws".to_owned(),
            name_prefix: "Canary Bot".to_owned(),
            ..BotConfig::default()
        };
        assert_eq!(
            connect_url(&config),
            "ws://host/ws?bot=true&namePrefix=Canary%20Bot"
        );

        let with_query = BotConfig {
            backend_url: "ws://host/ws?x=1".to_owned(),
            name_prefix: String::new(),
            ..BotConfig::default()
        };
        assert_eq!(connect_url(&with_query), "ws://host/ws?x=1&bot=true");
    }

    #[test]
    fn rng_is_deterministic_for_a_seed() {
        let a: Vec<u64> = (0..4).map(|_| Rng::new(7).next_u64()).collect();
        assert!(a.windows(2).all(|w| w[0] == w[1]));
        let mut stream = Rng::new(7);
        assert_ne!(stream.next_u64(), stream.next_u64());
        assert_eq!(Rng::new(1).below(0), None);
        assert!(Rng::new(1).below(5).is_some_and(|index| index < 5));
    }
}
