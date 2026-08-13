//! Configuration and engine wiring for the bot binary.
//!
//! Per CLAUDE.md, **env-var config is read here and only here**, then passed
//! down as plain structs — no crate below this one touches `std::env`. That is
//! why [`Settings::from`] takes a lookup closure: the process environment is
//! read exactly once, in `main`, and everything else is testable.
//!
//! This is a library so the integration tests can build the very same engine
//! the binary builds. A test that reimplemented the wiring would prove nothing
//! about what gets deployed.
//!
//! # Environment
//!
//! | Variable                    | Default                        | Meaning |
//! |-----------------------------|--------------------------------|---------|
//! | `BACKEND_URL`               | `ws://localhost:8080/ws`       | WebSocket endpoint; `?bot=true` is appended. |
//! | `BOT_NAME_PREFIX`           | *(empty)*                      | Prefixes the server-assigned bot name. |
//! | `VSBOT_TURN_MILLIS`         | `12000`                        | Budget for a whole 3-action turn, split by the allocator. |
//! | `VSBOT_MOVE_MILLIS`         | *(unset)*                      | Per-action override. **Disables the allocator.** |
//! | `MOVE_MILLIS`               | *(unset)*                      | Legacy spelling of `VSBOT_MOVE_MILLIS`. |
//! | `VSBOT_EARLY_STOP`          | `true`                         | Stop once the visit leader is uncatchable. |
//! | `VSBOT_EXTENSION`           | `true`                         | Run past the target toward the ceiling on an unstable root. |
//! | `VSBOT_PONDER`              | `false`                        | Think on the opponent's positions during their turn. |
//! | `VSBOT_PONDER_SECS`         | `30`                           | Cap on one pondering step. |
//! | `SEARCH`                    | `MCTS`                         | `MCTS` \| `GREEDY` \| `ALPHABETA`. |
//! | `MCTS_ARTIFACT`             | `artifacts/mcts_champion.json` | Policy/value net for `SEARCH=MCTS`. |
//! | `MCTS_SEED`                 | `1`                            | Seed for the searcher's RNG. |
//! | `CHALLENGER`                | `false`                        | Initiate games on a timer. |
//! | `CHALLENGER_INTERVAL_SECS`  | `300`                          | Challenger period; the first tick is jittered. |
//! | `VSBOT_EXPLORE_EPS`         | `0` *(off)*                    | **Harness only.** Chance of a random legal action inside the opening window. |
//! | `VSBOT_EXPLORE_TURNS`       | `8`                            | Window length, in *our own* turns. |
//! | `VSBOT_EXPLORE_SEED`        | `1`                            | Base seed for the per-game opening stream. |
//!
//! `VSBOT_EXPLORE_EPS` is the one knob here that makes the bot **weaker on
//! purpose**, so it is off unless a harness sets it — see [`explore`] for what
//! it buys and why cross-play cannot be measured without it. It is refused
//! together with `VSBOT_PONDER`, because a pondering session answers actions
//! without ever calling `choose` and would silently explore nothing.
//!
//! Unset and empty are the same thing. An unparseable value is a startup
//! failure, never a silent fallback: a typo that quietly downgrades the engine
//! is how the Java harness ended up reporting the wrong engine's results.
//!
//! # Budget profiles
//!
//! | Profile              | Environment                                      | Why |
//! |----------------------|--------------------------------------------------|-----|
//! | **Deployed default** | *(nothing)*                                      | 12 s/turn — 6 s / 3.6 s / 2.4 s, inside the owner's 10-15 s UX bound. |
//! | **Owner canary**     | `VSBOT_TURN_MILLIS=15000 VSBOT_PONDER=true`      | The top of the bound plus ponder; a behaviour change the owner judges (superiority.md Gate C). |
//! | **Bot gauntlets**    | `VSBOT_MOVE_MILLIS=1000`                         | Fixed per action: gauntlets and the RL gate stay at fixed time (§4). |
//! | **Predecessor parity** | `VSBOT_MOVE_MILLIS=1000 VSBOT_EARLY_STOP=false VSBOT_EXTENSION=false` | Exactly what shipped before S2, for an A/B. |
//! | **Integration tests** | `VSBOT_MOVE_MILLIS=50`                          | Fast and deterministic in wall-clock terms. |

#![deny(missing_docs)]
#![deny(missing_debug_implementations)]

pub mod explore;
pub mod mcts;

use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use virus_proto::{BotConfig, EngineKind, GreedyEngine, SearchEngine, StopPolicy};

pub use explore::{ExploreSettings, ExploringEngine};
pub use mcts::MctsEngine;

/// Where the champion artifact lives in the repository and in the deploy image.
///
/// Relative to the working directory on purpose: the container sets its own
/// workdir, and a developer running from the repo root gets the in-repo copy.
/// `MCTS_ARTIFACT` overrides it for everyone else.
pub const DEFAULT_MCTS_ARTIFACT: &str = "artifacts/mcts_champion.json";

/// Everything the process needs, resolved from the environment exactly once.
#[derive(Clone, Debug)]
pub struct Settings {
    /// Protocol-layer configuration.
    pub bot: BotConfig,
    /// Which engine `SEARCH` selected.
    pub engine: EngineKind,
    /// Inputs the MCTS engine needs; ignored by the other engines.
    pub mcts: MctsSettings,
    /// Opening-exploration settings. Off by default; see [`explore`].
    pub explore: ExploreSettings,
}

/// `SEARCH=MCTS` inputs.
#[derive(Clone, Debug)]
pub struct MctsSettings {
    /// Path to the policy/value artifact.
    pub artifact: PathBuf,
    /// Seed for the searcher's RNG.
    ///
    /// Play mode draws no random numbers, so this changes nothing today; it is
    /// plumbed so that a debugging session which *does* enable exploration is
    /// reproducible rather than mysterious.
    pub seed: u64,
    /// One diagnostic line per answered action (`VSBOT_PONDER_TRACE`).
    ///
    /// Off by default. It is the instrument bd `vsbot-gei` was diagnosed with —
    /// re-root hit rate, inherited visits, simulations this action added, and
    /// which stop rule ended it — and it is kept because those are the numbers
    /// any future ponder regression will be argued about.
    pub ponder_trace: bool,
}

impl Default for MctsSettings {
    fn default() -> MctsSettings {
        MctsSettings {
            artifact: PathBuf::from(DEFAULT_MCTS_ARTIFACT),
            seed: 1,
            ponder_trace: false,
        }
    }
}

impl Settings {
    /// Resolves settings from a lookup closure. `main` passes `std::env::var`;
    /// tests pass a map, so no test ever mutates the process environment.
    pub fn from(lookup: impl Fn(&str) -> Option<String>) -> Result<Settings, ConfigError> {
        let read =
            |key: &str| -> Option<String> { lookup(key).filter(|value| !value.trim().is_empty()) };
        let defaults = BotConfig::default();
        let mcts_defaults = MctsSettings::default();

        let millis = |raw: &str| raw.trim().parse::<u64>().ok().filter(|value| *value > 0);
        let flag = |raw: &str| match raw.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Some(true),
            "0" | "false" | "no" | "off" => Some(false),
            _ => None,
        };

        let turn_millis = parse_field(
            read("VSBOT_TURN_MILLIS").as_deref(),
            "VSBOT_TURN_MILLIS",
            millis,
        )?
        .unwrap_or(defaults.turn_budget.as_millis() as u64);

        // `VSBOT_MOVE_MILLIS` is the documented spelling; `MOVE_MILLIS` is what
        // shipped before the allocator and is still honoured so an existing
        // deployment's compose file keeps meaning what it meant. Setting either
        // one disables the allocator — that *is* what a per-action override is.
        let move_millis = match read("VSBOT_MOVE_MILLIS") {
            Some(raw) => parse_field(Some(raw.as_str()), "VSBOT_MOVE_MILLIS", millis)?,
            None => parse_field(read("MOVE_MILLIS").as_deref(), "MOVE_MILLIS", millis)?,
        };

        let early_stop = parse_field(
            read("VSBOT_EARLY_STOP").as_deref(),
            "VSBOT_EARLY_STOP",
            flag,
        )?
        .unwrap_or(defaults.stop_policy.early_stop);
        let extension = parse_field(read("VSBOT_EXTENSION").as_deref(), "VSBOT_EXTENSION", flag)?
            .unwrap_or(defaults.stop_policy.extension);
        let ponder = parse_field(read("VSBOT_PONDER").as_deref(), "VSBOT_PONDER", flag)?
            .unwrap_or(defaults.ponder);
        let ponder_secs = parse_field(
            read("VSBOT_PONDER_SECS").as_deref(),
            "VSBOT_PONDER_SECS",
            |raw| raw.trim().parse::<u64>().ok().filter(|secs| *secs > 0),
        )?
        .unwrap_or(defaults.ponder_budget.as_secs());

        let challenger = parse_field(read("CHALLENGER").as_deref(), "CHALLENGER", flag)?
            .unwrap_or(defaults.challenger);

        let interval_secs = parse_field(
            read("CHALLENGER_INTERVAL_SECS").as_deref(),
            "CHALLENGER_INTERVAL_SECS",
            |raw| raw.trim().parse::<u64>().ok().filter(|secs| *secs > 0),
        )?
        .unwrap_or(defaults.challenge_interval.as_secs());

        let seed = parse_field(read("MCTS_SEED").as_deref(), "MCTS_SEED", |raw| {
            raw.trim().parse::<u64>().ok()
        })?
        .unwrap_or(mcts_defaults.seed);

        let ponder_trace = parse_field(
            read("VSBOT_PONDER_TRACE").as_deref(),
            "VSBOT_PONDER_TRACE",
            flag,
        )?
        .unwrap_or(mcts_defaults.ponder_trace);

        let engine = match read("SEARCH") {
            Some(raw) => {
                EngineKind::parse(&raw).map_err(|error| ConfigError("SEARCH", error.to_string()))?
            }
            None => EngineKind::default(),
        };

        let explore_defaults = ExploreSettings::default();
        let explore = ExploreSettings {
            epsilon: parse_field(
                read("VSBOT_EXPLORE_EPS").as_deref(),
                "VSBOT_EXPLORE_EPS",
                |raw| {
                    raw.trim()
                        .parse::<f64>()
                        .ok()
                        .filter(|value| (0.0..=1.0).contains(value))
                },
            )?
            .unwrap_or(explore_defaults.epsilon),
            turns: parse_field(
                read("VSBOT_EXPLORE_TURNS").as_deref(),
                "VSBOT_EXPLORE_TURNS",
                |raw| raw.trim().parse::<u32>().ok(),
            )?
            .unwrap_or(explore_defaults.turns),
            seed: parse_field(
                read("VSBOT_EXPLORE_SEED").as_deref(),
                "VSBOT_EXPLORE_SEED",
                |raw| raw.trim().parse::<u64>().ok(),
            )?
            .unwrap_or(explore_defaults.seed),
        };

        // Refused, not quietly reconciled. A pondering session answers actions
        // straight from its own tree and never calls `SearchEngine::choose`, so
        // a run with both on would explore nothing at all and report a
        // diversity number it did not earn — the exact class of silent
        // downgrade the rest of this function exists to prevent.
        if explore.is_on() && ponder {
            return Err(ConfigError(
                "VSBOT_EXPLORE_EPS",
                "cannot be combined with VSBOT_PONDER=true: a pondering session answers actions \
                 without consulting the exploration wrapper, so the openings would silently stop \
                 being randomised. Turn one of them off."
                    .to_owned(),
            ));
        }

        Ok(Settings {
            bot: BotConfig {
                backend_url: read("BACKEND_URL").unwrap_or(defaults.backend_url),
                name_prefix: read("BOT_NAME_PREFIX").unwrap_or_default(),
                turn_budget: Duration::from_millis(turn_millis),
                move_budget: move_millis.map(Duration::from_millis),
                stop_policy: StopPolicy {
                    early_stop,
                    extension,
                    ..StopPolicy::default()
                },
                ponder,
                ponder_budget: Duration::from_secs(ponder_secs),
                challenger,
                challenge_interval: Duration::from_secs(interval_secs),
                ..BotConfig::default()
            },
            engine,
            mcts: MctsSettings {
                artifact: read("MCTS_ARTIFACT")
                    .map(PathBuf::from)
                    .unwrap_or(mcts_defaults.artifact),
                seed,
                ponder_trace,
            },
            explore,
        })
    }
}

fn parse_field<T>(
    raw: Option<&str>,
    key: &'static str,
    parse: impl Fn(&str) -> Option<T>,
) -> Result<Option<T>, ConfigError> {
    match raw {
        None => Ok(None),
        Some(value) => match parse(value) {
            Some(parsed) => Ok(Some(parsed)),
            None => Err(ConfigError(key, format!("cannot parse {value:?}"))),
        },
    }
}

/// A rejected environment value.
#[derive(Clone, Debug)]
pub struct ConfigError(pub &'static str, pub String);

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.0, self.1)
    }
}

impl std::error::Error for ConfigError {}

/// A built engine plus the lines describing it.
///
/// The description is returned rather than printed so the caller owns the log
/// format — and so tests can assert on it.
#[derive(Clone)]
pub struct EngineSetup {
    /// The engine to hand to `Bot`.
    pub engine: Arc<dyn SearchEngine>,
    /// One line of provenance: which engine, and what it loaded.
    pub description: String,
}

impl EngineSetup {
    /// Wraps the engine in [`ExploringEngine`] when `settings` asks for it.
    ///
    /// A separate step rather than a [`build_engine`] argument on purpose:
    /// exploration is a *measurement harness* behaviour, not part of the engine
    /// the deployment runs, and keeping it out of `build_engine` means every
    /// caller that wants the deployed engine — the integration tests, the arena
    /// — gets exactly that with no flag to forget.
    ///
    /// Off is the identity: at `eps = 0` the setup is returned untouched, so
    /// nothing is interposed on the production path.
    pub fn with_exploration(self, settings: &ExploreSettings) -> EngineSetup {
        if !settings.is_on() {
            return self;
        }
        let explorer = ExploringEngine::new(self.engine, *settings);
        let description = format!("{} {}", self.description, explorer.describe());
        EngineSetup {
            engine: Arc::new(explorer),
            description,
        }
    }
}

impl fmt::Debug for EngineSetup {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EngineSetup")
            .field("engine", &self.engine.name())
            .field("description", &self.description)
            .finish()
    }
}

/// **Engine wiring point.**
///
/// `virus-mcts` is wired. `virus-search` (enhanced alpha-beta) has landed as a
/// crate but is deliberately **not** wired here — that is a separate bead — so
/// `SEARCH=ALPHABETA` is a hard startup failure rather than a downgrade.
/// Falling back to the greedy reference engine there would let a deployment
/// believe it is running a real searcher while it plays at the level of "first
/// capture I see".
///
/// A missing or invalid `MCTS_ARTIFACT` is a startup failure for the same
/// reason. The *per-game* domain checks (two players, 12x12) are different in
/// kind: they depend on a game the process has not been offered yet, so they
/// cannot be decided here and are handled — loudly — inside [`MctsEngine`].
pub fn build_engine(kind: EngineKind, mcts: &MctsSettings) -> Result<EngineSetup, String> {
    match kind {
        EngineKind::Mcts => {
            let engine = MctsEngine::load(&mcts.artifact, mcts.seed, mcts.ponder_trace).map_err(
                |error| {
                    format!(
                        "SEARCH=MCTS could not load {}: {error}. Set MCTS_ARTIFACT to a valid \
                         policy/value export, or run SEARCH=GREEDY deliberately — the bot will \
                         not quietly downgrade itself.",
                        mcts.artifact.display()
                    )
                },
            )?;
            let description = format!("engine=MCTS {}", engine.describe());
            Ok(EngineSetup {
                engine: Arc::new(engine),
                description,
            })
        }
        EngineKind::Greedy => Ok(EngineSetup {
            engine: Arc::new(GreedyEngine),
            description: "engine=GREEDY (reference engine: first capture, else first legal move)"
                .to_owned(),
        }),
        EngineKind::AlphaBeta => Err(
            "SEARCH=ALPHABETA is not wired into this binary yet (a follow-up bead extends the \
             chain once virus-search has a deadline-safe entry point); use SEARCH=MCTS"
                .to_owned(),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::path::Path;

    fn settings(pairs: &[(&str, &str)]) -> Result<Settings, ConfigError> {
        let map: HashMap<String, String> = pairs
            .iter()
            .map(|(key, value)| ((*key).to_owned(), (*value).to_owned()))
            .collect();
        Settings::from(|key| map.get(key).cloned())
    }

    fn champion() -> MctsSettings {
        MctsSettings {
            ponder_trace: false,
            artifact: Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../artifacts/mcts_champion.json"),
            seed: 1,
        }
    }

    #[test]
    fn defaults_need_no_environment() {
        let settings = settings(&[]).expect("defaults are valid");
        assert_eq!(settings.bot.backend_url, "ws://localhost:8080/ws");
        assert_eq!(settings.bot.name_prefix, "");
        assert_eq!(settings.bot.turn_budget, Duration::from_millis(12_000));
        assert_eq!(
            settings.bot.move_budget, None,
            "no override means the allocator runs; that is the S2 default"
        );
        assert_eq!(settings.bot.challenge_interval, Duration::from_secs(300));
        assert!(!settings.bot.challenger);
        assert_eq!(settings.mcts.artifact, Path::new(DEFAULT_MCTS_ARTIFACT));
        assert_eq!(settings.mcts.seed, 1);
    }

    /// Pondering is a behaviour change the owner judges, so it must be off
    /// until a canary deployment turns it on (superiority.md Gate C).
    #[test]
    fn pondering_is_off_unless_asked_for() {
        assert!(!settings(&[]).expect("defaults").bot.ponder);
        assert!(
            !settings(&[("VSBOT_PONDER", "false")])
                .expect("explicit off")
                .bot
                .ponder
        );
        assert!(
            settings(&[("VSBOT_PONDER", "true")])
                .expect("explicit on")
                .bot
                .ponder
        );
    }

    #[test]
    fn the_turn_allocator_is_on_by_default_and_the_override_switches_it_off() {
        let allocated = settings(&[("VSBOT_TURN_MILLIS", "15000")]).expect("valid");
        let clock = allocated.bot.allocator();
        assert!(!clock.is_fixed());
        assert_eq!(clock.turn_budget(), Duration::from_millis(15_000));

        for key in ["VSBOT_MOVE_MILLIS", "MOVE_MILLIS"] {
            let overridden = settings(&[(key, "800"), ("VSBOT_TURN_MILLIS", "15000")])
                .unwrap_or_else(|error| panic!("{key} should be valid: {error}"));
            assert_eq!(overridden.bot.move_budget, Some(Duration::from_millis(800)));
            let mut clock = overridden.bot.allocator();
            assert!(clock.is_fixed(), "{key} must disable the allocator");
            assert_eq!(clock.allocate(3).target, Duration::from_millis(800));
        }

        // The documented spelling wins when both are set, rather than the two
        // silently disagreeing.
        let both = settings(&[("VSBOT_MOVE_MILLIS", "800"), ("MOVE_MILLIS", "50")])
            .expect("both spellings are valid");
        assert_eq!(both.bot.move_budget, Some(Duration::from_millis(800)));
    }

    #[test]
    fn the_stop_rules_are_on_by_default_and_individually_switchable() {
        let defaults = settings(&[]).expect("defaults");
        assert!(defaults.bot.stop_policy.early_stop);
        assert!(defaults.bot.stop_policy.extension);

        let off = settings(&[("VSBOT_EARLY_STOP", "no"), ("VSBOT_EXTENSION", "off")])
            .expect("both switch off");
        assert!(!off.bot.stop_policy.early_stop);
        assert!(!off.bot.stop_policy.extension);
    }

    /// Exploration weakens the bot on purpose, so nothing but an explicit
    /// `VSBOT_EXPLORE_EPS` may switch it on — CLAUDE.md's "production play
    /// paths take no RNG unless explicitly configured".
    #[test]
    fn opening_exploration_is_off_unless_asked_for() {
        let defaults = settings(&[]).expect("defaults");
        assert_eq!(defaults.explore.epsilon, 0.0);
        assert_eq!(defaults.explore.turns, explore::DEFAULT_EXPLORE_TURNS);
        assert_eq!(defaults.explore.seed, 1);
        assert!(!defaults.explore.is_on());

        let asked = settings(&[
            ("VSBOT_EXPLORE_EPS", "0.15"),
            ("VSBOT_EXPLORE_TURNS", "6"),
            ("VSBOT_EXPLORE_SEED", "20260813"),
        ])
        .expect("valid exploration settings");
        assert_eq!(asked.explore.epsilon, 0.15);
        assert_eq!(asked.explore.turns, 6);
        assert_eq!(asked.explore.seed, 20_260_813);
        assert!(asked.explore.is_on());
    }

    /// Off must be the *identity* on the deployed path: not a wrapper that
    /// happens never to fire, but no wrapper at all.
    #[test]
    fn the_exploration_wrapper_is_only_installed_when_it_is_on() {
        let plain = build_engine(EngineKind::Greedy, &champion()).expect("greedy");
        let untouched = build_engine(EngineKind::Greedy, &champion())
            .expect("greedy")
            .with_exploration(&ExploreSettings::default());
        assert_eq!(untouched.description, plain.description);

        let explored = build_engine(EngineKind::Greedy, &champion())
            .expect("greedy")
            .with_exploration(&ExploreSettings {
                epsilon: 0.15,
                turns: 8,
                seed: 4,
            });
        assert!(
            explored.description.contains("exploration=ON"),
            "{explored:?}"
        );
        assert!(explored.description.contains("WEAKER"), "{explored:?}");
        // The wrapper must not disguise which engine is playing.
        assert_eq!(explored.engine.name(), plain.engine.name());
    }

    /// A pondering session never calls `choose`, so the two together would
    /// silently produce zero exploration. Refuse rather than reconcile.
    #[test]
    fn exploration_and_pondering_are_refused_together() {
        let error = settings(&[("VSBOT_EXPLORE_EPS", "0.15"), ("VSBOT_PONDER", "true")])
            .expect_err("the combination is rejected");
        assert_eq!(error.0, "VSBOT_EXPLORE_EPS");
        assert!(error.1.contains("VSBOT_PONDER"), "{error}");
        // Off is not a conflict, whatever the window and seed say.
        settings(&[
            ("VSBOT_EXPLORE_EPS", "0"),
            ("VSBOT_EXPLORE_TURNS", "8"),
            ("VSBOT_PONDER", "true"),
        ])
        .expect("eps=0 is not exploration");
    }

    #[test]
    fn mcts_is_the_default_engine() {
        assert_eq!(settings(&[]).expect("defaults").engine, EngineKind::Mcts);
        assert_eq!(
            settings(&[("SEARCH", "   ")])
                .expect("blank is unset")
                .engine,
            EngineKind::Mcts
        );
        assert_eq!(EngineKind::default(), EngineKind::Mcts);
    }

    #[test]
    fn every_field_is_read_from_the_environment() {
        let settings = settings(&[
            ("BACKEND_URL", "wss://vs.wandergeek.org/ws"),
            ("BOT_NAME_PREFIX", "Canary"),
            ("VSBOT_TURN_MILLIS", "15000"),
            ("VSBOT_MOVE_MILLIS", "250"),
            ("VSBOT_EARLY_STOP", "false"),
            ("VSBOT_EXTENSION", "0"),
            ("VSBOT_PONDER", "yes"),
            ("VSBOT_PONDER_SECS", "90"),
            ("SEARCH", "greedy"),
            ("MCTS_ARTIFACT", "/opt/nets/gen7.json"),
            ("MCTS_SEED", "424242"),
            ("CHALLENGER", "true"),
            ("CHALLENGER_INTERVAL_SECS", "45"),
        ])
        .expect("all values are valid");
        assert_eq!(settings.bot.backend_url, "wss://vs.wandergeek.org/ws");
        assert_eq!(settings.bot.name_prefix, "Canary");
        assert_eq!(settings.bot.turn_budget, Duration::from_millis(15_000));
        assert_eq!(settings.bot.move_budget, Some(Duration::from_millis(250)));
        assert!(!settings.bot.stop_policy.early_stop);
        assert!(!settings.bot.stop_policy.extension);
        assert!(settings.bot.ponder);
        assert_eq!(settings.bot.ponder_budget, Duration::from_secs(90));
        assert_eq!(settings.bot.challenge_interval, Duration::from_secs(45));
        assert!(settings.bot.challenger);
        assert_eq!(settings.engine, EngineKind::Greedy);
        assert_eq!(settings.mcts.artifact, Path::new("/opt/nets/gen7.json"));
        assert_eq!(settings.mcts.seed, 424_242);
    }

    #[test]
    fn empty_values_read_as_unset() {
        let settings = settings(&[
            ("BACKEND_URL", "   "),
            ("SEARCH", ""),
            ("VSBOT_TURN_MILLIS", ""),
            ("VSBOT_MOVE_MILLIS", " "),
            ("MOVE_MILLIS", ""),
            ("VSBOT_PONDER", "  "),
            ("MCTS_ARTIFACT", " "),
        ])
        .expect("blank values fall back to defaults");
        assert_eq!(settings.bot.backend_url, "ws://localhost:8080/ws");
        assert_eq!(settings.engine, EngineKind::Mcts);
        assert_eq!(settings.bot.turn_budget, Duration::from_millis(12_000));
        assert_eq!(
            settings.bot.move_budget, None,
            "a blank override is no override, not a zero one"
        );
        assert!(!settings.bot.ponder);
        assert_eq!(settings.mcts.artifact, Path::new(DEFAULT_MCTS_ARTIFACT));
    }

    #[test]
    fn bad_values_fail_startup_instead_of_falling_back() {
        for pair in [
            ("SEARCH", "gobot"),
            ("MOVE_MILLIS", "0"),
            ("MOVE_MILLIS", "soon"),
            ("VSBOT_MOVE_MILLIS", "0"),
            ("VSBOT_MOVE_MILLIS", "later"),
            ("VSBOT_TURN_MILLIS", "0"),
            ("VSBOT_TURN_MILLIS", "twelve"),
            ("VSBOT_PONDER", "sometimes"),
            ("VSBOT_PONDER_SECS", "0"),
            ("VSBOT_EARLY_STOP", "occasionally"),
            ("VSBOT_EXTENSION", "-1"),
            ("MCTS_SEED", "-1"),
            ("MCTS_SEED", "lucky"),
            // A probability outside [0, 1] is a typo, not a stronger request.
            ("VSBOT_EXPLORE_EPS", "1.5"),
            ("VSBOT_EXPLORE_EPS", "-0.1"),
            ("VSBOT_EXPLORE_EPS", "often"),
            ("VSBOT_EXPLORE_TURNS", "-1"),
            ("VSBOT_EXPLORE_TURNS", "eight"),
            ("VSBOT_EXPLORE_SEED", "-1"),
            ("CHALLENGER", "maybe"),
            ("CHALLENGER_INTERVAL_SECS", "-1"),
        ] {
            assert!(
                settings(&[pair]).is_err(),
                "{pair:?} should have been rejected"
            );
        }
    }

    #[test]
    fn mcts_and_greedy_are_wired_and_describe_themselves() {
        let mcts = build_engine(EngineKind::Mcts, &champion()).expect("the champion loads");
        assert_eq!(mcts.engine.name(), "mcts");
        assert!(mcts.description.starts_with("engine=MCTS "), "{mcts:?}");
        assert!(mcts.description.contains("mcts_champion.json"), "{mcts:?}");

        let greedy = build_engine(EngineKind::Greedy, &champion()).expect("greedy needs nothing");
        assert_eq!(greedy.engine.name(), "greedy");
        assert!(
            greedy.description.starts_with("engine=GREEDY"),
            "{greedy:?}"
        );
    }

    #[test]
    fn alphabeta_errors_rather_than_falling_back() {
        let error = build_engine(EngineKind::AlphaBeta, &champion())
            .expect_err("virus-search is not wired into the binary");
        assert!(error.contains("not wired into this binary"), "{error}");
    }

    #[test]
    fn a_bad_artifact_fails_startup_rather_than_downgrading_to_greedy() {
        let missing = MctsSettings {
            ponder_trace: false,
            artifact: PathBuf::from("/nonexistent/gen99.json"),
            seed: 1,
        };
        let error = build_engine(EngineKind::Mcts, &missing).expect_err("no such artifact");
        assert!(error.contains("could not load"), "{error}");
        assert!(error.contains("will not quietly downgrade"), "{error}");
    }
}
