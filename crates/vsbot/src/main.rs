//! The bot binary.
//!
//! Per CLAUDE.md, **env-var config is read here and only here**, then passed
//! down as plain structs — no crate below this one touches `std::env`. That is
//! why [`Settings::from`] takes a lookup closure: the process environment is
//! read exactly once, in [`main`], and everything else is testable.
//!
//! # Environment
//!
//! | Variable                    | Default                    | Meaning |
//! |-----------------------------|----------------------------|---------|
//! | `BACKEND_URL`               | `ws://localhost:8080/ws`   | WebSocket endpoint; `?bot=true` is appended. |
//! | `BOT_NAME_PREFIX`           | *(empty)*                  | Prefixes the server-assigned bot name. |
//! | `MOVE_MILLIS`               | `1000`                     | Wall-clock budget per action. |
//! | `SEARCH`                    | `GREEDY`                   | `GREEDY` \| `ALPHABETA` \| `MCTS`. |
//! | `CHALLENGER`                | `false`                    | Initiate games on a timer. |
//! | `CHALLENGER_INTERVAL_SECS`  | `300`                      | Challenger period; the first tick is jittered. |
//!
//! Unset and empty are the same thing. An unparseable value is a startup
//! failure, never a silent fallback: a typo that quietly downgrades the engine
//! is how the Java harness ended up reporting the wrong engine's results.

use std::fmt;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;
use virus_proto::{Bot, BotConfig, EngineKind, GreedyEngine, SearchEngine};

fn main() -> ExitCode {
    install_crypto_provider();

    let settings = match Settings::from(|key| std::env::var(key).ok()) {
        Ok(settings) => settings,
        Err(error) => {
            eprintln!("vsbot: {error}");
            return ExitCode::FAILURE;
        }
    };
    let engine = match build_engine(settings.engine) {
        Ok(engine) => engine,
        Err(error) => {
            eprintln!("vsbot: {error}");
            return ExitCode::FAILURE;
        }
    };

    eprintln!(
        "vsbot {} starting: url={} search={} move_budget={:?} challenger={}",
        env!("CARGO_PKG_VERSION"),
        settings.bot.backend_url,
        settings.engine.as_str(),
        settings.bot.move_budget,
        settings.bot.challenger,
    );

    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(error) => {
            eprintln!("vsbot: could not start the tokio runtime: {error}");
            return ExitCode::FAILURE;
        }
    };
    runtime.block_on(async move {
        let (bot, mut inbox) = Bot::new(Arc::new(settings.bot), engine);
        virus_proto::run_forever(&bot, &mut inbox).await
    })
}

/// Selects rustls' crypto provider before any TLS is attempted.
///
/// `tokio-tungstenite`'s `rustls-tls-webpki-roots` feature brings in rustls but
/// picks no provider, so rustls 0.23 cannot infer a process-level default and
/// **panics inside the connect future** the first time a `wss://` URL is
/// dialled. That panic is invisible until the bot is pointed at production —
/// exactly how it was found: the container came up, printed its banner, and
/// died on the first handshake against `wss://vs.wandergeek.org/ws`.
///
/// Doing it here, eagerly, converts a first-connection crash into a startup
/// crash. `install_default` returns `Err` only if a provider is already
/// installed, which cannot happen this early but is harmless either way.
fn install_crypto_provider() {
    let _ = rustls::crypto::ring::default_provider().install_default();
}

/// Everything the process needs, resolved from the environment exactly once.
#[derive(Clone, Debug)]
struct Settings {
    bot: BotConfig,
    engine: EngineKind,
}

impl Settings {
    /// Resolves settings from a lookup closure. `main` passes `std::env::var`;
    /// tests pass a map, so no test ever mutates the process environment.
    fn from(lookup: impl Fn(&str) -> Option<String>) -> Result<Settings, ConfigError> {
        let read =
            |key: &str| -> Option<String> { lookup(key).filter(|value| !value.trim().is_empty()) };
        let defaults = BotConfig::default();

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
struct ConfigError(&'static str, String);

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.0, self.1)
    }
}

/// **Engine wiring point.**
///
/// `virus-search` (enhanced alpha-beta) and `virus-mcts` (PUCT + policy/value
/// net) are still stubs; when they land, each arm becomes one line building the
/// real searcher behind [`SearchEngine`]. Nothing else in the binary or in
/// `virus-proto` changes — that is the whole point of the trait.
///
/// Until then an unmerged engine is a hard startup failure. Falling back to the
/// greedy reference engine would let a deployment believe it is running the
/// champion while it plays at random.
fn build_engine(kind: EngineKind) -> Result<Arc<dyn SearchEngine>, String> {
    match kind {
        EngineKind::Greedy => Ok(Arc::new(GreedyEngine)),
        EngineKind::AlphaBeta => Err(
            "SEARCH=ALPHABETA is not yet merged (virus-search is a stub); use SEARCH=GREEDY"
                .to_owned(),
        ),
        EngineKind::Mcts => Err(
            "SEARCH=MCTS is not yet merged (virus-mcts is a stub); use SEARCH=GREEDY".to_owned(),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn settings(pairs: &[(&str, &str)]) -> Result<Settings, ConfigError> {
        let map: HashMap<String, String> = pairs
            .iter()
            .map(|(key, value)| ((*key).to_owned(), (*value).to_owned()))
            .collect();
        Settings::from(|key| map.get(key).cloned())
    }

    #[test]
    fn defaults_need_no_environment() {
        let settings = settings(&[]).expect("defaults are valid");
        assert_eq!(settings.bot.backend_url, "ws://localhost:8080/ws");
        assert_eq!(settings.bot.name_prefix, "");
        assert_eq!(settings.bot.move_budget, Duration::from_millis(1000));
        assert_eq!(settings.bot.challenge_interval, Duration::from_secs(300));
        assert!(!settings.bot.challenger);
        assert_eq!(settings.engine, EngineKind::Greedy);
    }

    #[test]
    fn every_field_is_read_from_the_environment() {
        let settings = settings(&[
            ("BACKEND_URL", "wss://vs.wandergeek.org/ws"),
            ("BOT_NAME_PREFIX", "Canary"),
            ("MOVE_MILLIS", "250"),
            ("SEARCH", "mcts"),
            ("CHALLENGER", "true"),
            ("CHALLENGER_INTERVAL_SECS", "45"),
        ])
        .expect("all values are valid");
        assert_eq!(settings.bot.backend_url, "wss://vs.wandergeek.org/ws");
        assert_eq!(settings.bot.name_prefix, "Canary");
        assert_eq!(settings.bot.move_budget, Duration::from_millis(250));
        assert_eq!(settings.bot.challenge_interval, Duration::from_secs(45));
        assert!(settings.bot.challenger);
        assert_eq!(settings.engine, EngineKind::Mcts);
    }

    #[test]
    fn empty_values_read_as_unset() {
        let settings = settings(&[("BACKEND_URL", "   "), ("SEARCH", ""), ("MOVE_MILLIS", "")])
            .expect("blank values fall back to defaults");
        assert_eq!(settings.bot.backend_url, "ws://localhost:8080/ws");
        assert_eq!(settings.engine, EngineKind::Greedy);
        assert_eq!(settings.bot.move_budget, Duration::from_millis(1000));
    }

    #[test]
    fn bad_values_fail_startup_instead_of_falling_back() {
        for pair in [
            ("SEARCH", "gobot"),
            ("MOVE_MILLIS", "0"),
            ("MOVE_MILLIS", "soon"),
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
    fn only_the_greedy_engine_is_wired_today() {
        assert!(build_engine(EngineKind::Greedy).is_ok());
        for unmerged in [EngineKind::AlphaBeta, EngineKind::Mcts] {
            match build_engine(unmerged) {
                Err(error) => assert!(error.contains("not yet merged"), "{error}"),
                Ok(_) => panic!("{} should not be wired yet", unmerged.as_str()),
            }
        }
    }
}
