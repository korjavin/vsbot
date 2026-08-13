//! The bot binary.
//!
//! Deliberately thin: it reads the process environment, builds the engine, logs
//! exactly what it is about to play as, and hands both to `virus-proto`. The
//! logic lives in the `vsbot` library so the integration tests exercise the same
//! wiring the deployment runs — see that crate's docs for the environment table.

use std::process::ExitCode;
use std::sync::Arc;
use virus_proto::Bot;
use vsbot::{build_engine, Settings};

fn main() -> ExitCode {
    install_crypto_provider();

    let settings = match Settings::from(|key| std::env::var(key).ok()) {
        Ok(settings) => settings,
        Err(error) => {
            eprintln!("vsbot: {error}");
            return ExitCode::FAILURE;
        }
    };
    let setup = match build_engine(settings.engine, &settings.mcts) {
        Ok(setup) => setup,
        Err(error) => {
            eprintln!("vsbot: {error}");
            return ExitCode::FAILURE;
        }
    };

    // Two lines, printed before the first connection: what the process is, and
    // what it is playing with. An operator must be able to confirm the engine
    // and the artifact from the log alone — inferring them from move quality is
    // exactly the mistake the Java post-mortem records.
    eprintln!(
        "vsbot {} starting: url={} search={} challenger={}",
        env!("CARGO_PKG_VERSION"),
        settings.bot.backend_url,
        settings.engine.as_str(),
        settings.bot.challenger,
    );
    // The time budget is the thing the owner's UX bound is expressed in, and
    // pondering is a behaviour change that ships canary-first — both belong in
    // the banner so a deployment can be checked rather than trusted.
    eprintln!(
        "vsbot: {} ponder={}",
        settings.bot.budget_summary(),
        if settings.bot.ponder {
            "on (CANARY: search runs during the opponent's turn)"
        } else {
            "off"
        }
    );
    eprintln!("vsbot: {}", setup.description);

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
        let (bot, mut inbox) = Bot::new(Arc::new(settings.bot), setup.engine);
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
