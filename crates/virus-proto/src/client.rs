//! The WebSocket transport.
//!
//! Three tasks per connection, and the split matters:
//!
//! * the **read loop** decodes frames and calls [`Bot::handle`] — it must never
//!   block, so a server ping is always answered inside the 60 s pong deadline;
//! * the **writer** drains the outbox, re-checking each action's guard against
//!   the live position immediately before it hits the wire;
//! * the **challenger timer** (optional) is the sole driver of outbound
//!   challenges.
//!
//! Searches live on `spawn_blocking` workers owned by [`Bot`], not here.

use crate::bot::{Bot, Outbound};
use crate::config::{connect_url, Rng};
use futures_util::stream::StreamExt;
use futures_util::SinkExt;
use std::fmt;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite::Message as WsMessage;

/// A transport failure. Every variant is retryable by [`run_forever`].
#[derive(Debug)]
pub enum ProtoError {
    /// The dial or the WebSocket handshake failed.
    Connect(String),
    /// The socket faulted mid-session.
    Socket(String),
}

impl fmt::Display for ProtoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProtoError::Connect(detail) => write!(f, "connect failed: {detail}"),
            ProtoError::Socket(detail) => write!(f, "socket error: {detail}"),
        }
    }
}

impl std::error::Error for ProtoError {}

/// Connects, plays until the socket dies, and returns.
///
/// `inbox` is the receiver handed back by [`Bot::new`]. It is passed by mutable
/// reference so a reconnect keeps the same queue — anything queued for the dead
/// socket is still guard-checked against the (now invalidated) position and
/// dropped, rather than replayed into a new game.
pub async fn run_session(
    bot: &Bot,
    inbox: &mut mpsc::UnboundedReceiver<Outbound>,
) -> Result<(), ProtoError> {
    let url = connect_url(bot.config());
    let (socket, _response) = tokio_tungstenite::connect_async(url.as_str())
        .await
        .map_err(|error| ProtoError::Connect(error.to_string()))?;
    bot.log(&format!("connected to {url}"));

    let (mut sink, mut stream) = socket.split();
    let challenger = spawn_challenger(bot);

    let result = loop {
        tokio::select! {
            // Draining the outbox and reading are both cheap; neither can
            // starve the other because the search runs elsewhere.
            queued = inbox.recv() => {
                let Some(item) = queued else { break Ok(()) };
                if let Err(error) = write_item(bot, &mut sink, item).await {
                    break Err(error);
                }
            }
            frame = stream.next() => {
                match frame {
                    Some(Ok(WsMessage::Text(text))) => bot.handle_text(&text),
                    Some(Ok(WsMessage::Binary(bytes))) => {
                        match std::str::from_utf8(&bytes) {
                            Ok(text) => bot.handle_text(text),
                            Err(_) => bot.log("ignored a non-UTF-8 binary frame"),
                        }
                    }
                    // Answer through the writer so the pong is flushed now, not
                    // whenever the next application frame happens to go out.
                    Some(Ok(WsMessage::Ping(payload))) => bot.pong(payload.to_vec()),
                    Some(Ok(WsMessage::Pong(_))) | Some(Ok(WsMessage::Frame(_))) => {}
                    Some(Ok(WsMessage::Close(_))) | None => break Ok(()),
                    Some(Err(error)) => break Err(ProtoError::Socket(error.to_string())),
                }
            }
        }
    };

    challenger.abort();
    bot.on_disconnected();
    let _ = sink.close().await;
    result
}

/// Reconnect loop with exponential backoff, reset on every successful session.
pub async fn run_forever(bot: &Bot, inbox: &mut mpsc::UnboundedReceiver<Outbound>) -> ! {
    let mut backoff = bot.config().reconnect_min;
    loop {
        match run_session(bot, inbox).await {
            Ok(()) => bot.log("session closed; reconnecting"),
            Err(error) => bot.log(&format!("{error}; reconnecting")),
        }
        tokio::time::sleep(backoff).await;
        backoff = (backoff * 2).min(bot.config().reconnect_max);
        if bot.core().phase == crate::bot::Phase::Idle {
            backoff = bot.config().reconnect_min;
        }
    }
}

async fn write_item<S>(bot: &Bot, sink: &mut S, item: Outbound) -> Result<(), ProtoError>
where
    S: SinkExt<WsMessage> + Unpin,
    S::Error: fmt::Display,
{
    let frame = match item {
        Outbound::Pong(payload) => WsMessage::Pong(payload.into()),
        Outbound::Text { data, guard } => {
            // Last gate before the wire. A snapshot may have landed between the
            // search task's check and this moment; sending anyway is how a
            // stale move becomes an illegal-move forfeit.
            if let Some(guard) = guard {
                if !bot.core().action_still_valid(&guard) {
                    bot.core().counters.stale_results_dropped += 1;
                    bot.log("writer dropped a superseded action");
                    return Ok(());
                }
            }
            WsMessage::Text(data.into())
        }
    };
    sink.send(frame)
        .await
        .map_err(|error| ProtoError::Socket(error.to_string()))
}

/// Timer-driven challenger. Not spawned when `challenger` is off.
fn spawn_challenger(bot: &Bot) -> tokio::task::JoinHandle<()> {
    let bot = bot.clone();
    tokio::spawn(async move {
        if !bot.config().challenger {
            // Park forever; the caller aborts us when the session ends.
            std::future::pending::<()>().await;
            return;
        }
        let interval = bot.config().challenge_interval;
        let mut rng = Rng::new(bot.config().rng_seed.unwrap_or_else(clock_seed));
        // Jitter the first tick so a fleet started by one compose file does not
        // fire every challenge in the same instant.
        tokio::time::sleep(rng.fraction_of(interval)).await;
        loop {
            bot.challenge_tick(&mut rng);
            tokio::time::sleep(interval).await;
        }
    })
}

fn clock_seed() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_nanos() as u64
}
