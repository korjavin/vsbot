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
//! | `MOVE_MILLIS`               | `1000`                         | Wall-clock budget per action. |
//! | `SEARCH`                    | `MCTS`                         | `MCTS` \| `GREEDY` \| `ALPHABETA`. |
//! | `MCTS_ARTIFACT`             | `artifacts/mcts_champion.json` | Policy/value net for `SEARCH=MCTS`. |
//! | `MCTS_SEED`                 | `1`                            | Seed for the searcher's RNG. |
//! | `CHALLENGER`                | `false`                        | Initiate games on a timer. |
//! | `CHALLENGER_INTERVAL_SECS`  | `300`                          | Challenger period; the first tick is jittered. |
//!
//! Unset and empty are the same thing. An unparseable value is a startup
//! failure, never a silent fallback: a typo that quietly downgrades the engine
//! is how the Java harness ended up reporting the wrong engine's results.

#![deny(missing_docs)]
#![deny(missing_debug_implementations)]

pub mod mcts;

use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use virus_proto::{BotConfig, EngineKind, GreedyEngine, SearchEngine};

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
}

impl Default for MctsSettings {
    fn default() -> MctsSettings {
        MctsSettings {
            artifact: PathBuf::from(DEFAULT_MCTS_ARTIFACT),
            seed: 1,
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

        let move_millis = parse_field(read("MOVE_MILLIS").as_deref(), "MOVE_MILLIS", |raw| {
            raw.trim().parse::<u64>().ok().filter(|millis| *millis > 0)
        })?
        .unwrap_or(defaults.move_budget.as_millis() as u64);

        let challenger = parse_field(read("CHALLENGER").as_deref(), "CHALLENGER", |raw| match raw
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "1" | "true" | "yes" | "on" => Some(true),
            "0" | "false" | "no" | "off" => Some(false),
            _ => None,
        })?
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

        let engine = match read("SEARCH") {
            Some(raw) => {
                EngineKind::parse(&raw).map_err(|error| ConfigError("SEARCH", error.to_string()))?
            }
            None => EngineKind::default(),
        };

        Ok(Settings {
            bot: BotConfig {
                backend_url: read("BACKEND_URL").unwrap_or(defaults.backend_url),
                name_prefix: read("BOT_NAME_PREFIX").unwrap_or_default(),
                move_budget: Duration::from_millis(move_millis),
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
            },
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
            let engine = MctsEngine::load(&mcts.artifact, mcts.seed).map_err(|error| {
                format!(
                    "SEARCH=MCTS could not load {}: {error}. Set MCTS_ARTIFACT to a valid \
                     policy/value export, or run SEARCH=GREEDY deliberately — the bot will \
                     not quietly downgrade itself.",
                    mcts.artifact.display()
                )
            })?;
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
        assert_eq!(settings.bot.move_budget, Duration::from_millis(1000));
        assert_eq!(settings.bot.challenge_interval, Duration::from_secs(300));
        assert!(!settings.bot.challenger);
        assert_eq!(settings.mcts.artifact, Path::new(DEFAULT_MCTS_ARTIFACT));
        assert_eq!(settings.mcts.seed, 1);
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
            ("MOVE_MILLIS", "250"),
            ("SEARCH", "greedy"),
            ("MCTS_ARTIFACT", "/opt/nets/gen7.json"),
            ("MCTS_SEED", "424242"),
            ("CHALLENGER", "true"),
            ("CHALLENGER_INTERVAL_SECS", "45"),
        ])
        .expect("all values are valid");
        assert_eq!(settings.bot.backend_url, "wss://vs.wandergeek.org/ws");
        assert_eq!(settings.bot.name_prefix, "Canary");
        assert_eq!(settings.bot.move_budget, Duration::from_millis(250));
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
            ("MOVE_MILLIS", ""),
            ("MCTS_ARTIFACT", " "),
        ])
        .expect("blank values fall back to defaults");
        assert_eq!(settings.bot.backend_url, "ws://localhost:8080/ws");
        assert_eq!(settings.engine, EngineKind::Mcts);
        assert_eq!(settings.bot.move_budget, Duration::from_millis(1000));
        assert_eq!(settings.mcts.artifact, Path::new(DEFAULT_MCTS_ARTIFACT));
    }

    #[test]
    fn bad_values_fail_startup_instead_of_falling_back() {
        for pair in [
            ("SEARCH", "gobot"),
            ("MOVE_MILLIS", "0"),
            ("MOVE_MILLIS", "soon"),
            ("MCTS_SEED", "-1"),
            ("MCTS_SEED", "lucky"),
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
            artifact: PathBuf::from("/nonexistent/gen99.json"),
            seed: 1,
        };
        let error = build_engine(EngineKind::Mcts, &missing).expect_err("no such artifact");
        assert!(error.contains("could not load"), "{error}");
        assert!(error.contains("will not quietly downgrade"), "{error}");
    }
}
