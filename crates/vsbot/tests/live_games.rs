//! End-to-end play against the real Go server.
//!
//! This is the acceptance gate for `virus-proto`: unit tests prove the state
//! machine in isolation, but only a live server exercises the *ordering* the
//! invariants are about — `move_made` before `game_state`, the
//! `neutrals_placed` ack before `turn_change`, `users_update` bursts around
//! every game end.
//!
//! # Running it
//!
//! Opt-in, because it needs a Go toolchain and a checkout of the server:
//!
//! ```text
//! VSBOT_ITEST=1 cargo test -p vsbot --test live_games -- --nocapture
//! ```
//!
//! Knobs:
//!
//! | Variable                | Default                           |
//! |-------------------------|-----------------------------------|
//! | `VSBOT_ITEST`           | unset — the test skips            |
//! | `VSBOT_ITEST_BACKEND`   | `$HOME/Project/virusgame/backend` |
//! | `VSBOT_ITEST_GAMES`     | `3`                               |
//!
//! The server hard-codes `:8080` in `main.go`, which is no good for a test that
//! must not fight whatever else the developer has running. It is therefore
//! built through `go build -overlay`: a patched `main.go` that honours
//! `VSBOT_ITEST_PORT` is substituted **at build time only**, so the server
//! checkout is never modified. If the anchor line ever moves, the build falls
//! back to the stock binary on port 8080.
//!
//! Not wired into CI: GitHub Actions has no checkout of the Go server, so the
//! job would have to vendor a second repository to build it. Run it locally
//! before touching this crate.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use virus_core::{Action, State};
use virus_proto::{Bot, BotConfig, GreedyEngine, SearchBudget, SearchEngine, SearchOutcome};

const OVERALL_TIMEOUT: Duration = Duration::from_secs(180);

/// Plays greedily, but spends its neutral placement at the first legal
/// opportunity of every game.
///
/// That is the whole point: our own `neutrals_placed` ack arrives with a
/// snapshot still showing us as mover with `movesLeft == 3`, and acting on it
/// is what forfeited two live games on 2026-08-08 (ARCHITECTURE.md invariant
/// 2). This engine forces that exact sequence into every game of the run.
#[derive(Debug, Default)]
struct NeutralOpeningEngine {
    placements: AtomicUsize,
}

impl SearchEngine for NeutralOpeningEngine {
    fn choose(&self, state: &State, budget: &SearchBudget) -> Option<SearchOutcome> {
        if state.can_place_neutrals() {
            if let Some(action) = state
                .legal_actions()
                .into_iter()
                .find(|action| matches!(action, Action::PlaceNeutrals { .. }))
            {
                self.placements.fetch_add(1, Ordering::SeqCst);
                return Some(SearchOutcome::new(action));
            }
        }
        GreedyEngine.choose(state, budget)
    }

    fn name(&self) -> &'static str {
        "neutral-opening"
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn two_bots_play_full_games_without_a_single_illegal_move() {
    if std::env::var("VSBOT_ITEST").as_deref() != Ok("1") {
        eprintln!("skipping: set VSBOT_ITEST=1 to run the live-server integration test");
        return;
    }
    let target_games: u64 = std::env::var("VSBOT_ITEST_GAMES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(3);

    let workdir = std::env::temp_dir().join(format!("vsbot-itest-{}", std::process::id()));
    std::fs::create_dir_all(&workdir).expect("temp workdir");

    let server = build_server(&workdir);
    let port = if server.port_is_configurable {
        free_port()
    } else {
        assert!(
            port_is_free(8080),
            "the stock server hard-codes :8080 and it is busy — free it, or restore the \
             `http.ListenAndServe(\":8080\", nil)` line the overlay patch anchors on"
        );
        8080
    };

    let server_log = workdir.join("server.log");
    let log_handle = std::fs::File::create(&server_log).expect("server log");
    let mut child = tokio::process::Command::new(&server.binary)
        .current_dir(&workdir)
        .env("VSBOT_ITEST_PORT", port.to_string())
        .stdout(Stdio::from(log_handle.try_clone().expect("clone log")))
        .stderr(Stdio::from(log_handle))
        .stdin(Stdio::null())
        .kill_on_drop(true)
        .spawn()
        .unwrap_or_else(|error| panic!("could not start {}: {error}", server.binary.display()));

    let outcome = tokio::time::timeout(OVERALL_TIMEOUT, play(port, target_games)).await;

    let _ = child.kill().await;
    let server_output = std::fs::read_to_string(&server_log).unwrap_or_default();

    let Ok(report) = outcome else {
        panic!(
            "the run did not finish {target_games} games within {OVERALL_TIMEOUT:?}\n\
             --- server log tail ---\n{}",
            tail(&server_output)
        );
    };

    // Independent, server-side confirmation. The hub logs every rejected action
    // before it forfeits the offender, so a clean log is proof the clean client
    // counters are not just a client-side blind spot.
    assert!(
        !server_output.contains("made illegal move"),
        "the server logged an illegal move:\n{}",
        tail(&server_output)
    );

    for side in [&report.challenger, &report.acceptor] {
        assert_eq!(
            side.illegal_moves, 0,
            "{} was forfeited for an illegal move: {:?}",
            side.name, side.last_error
        );
        assert_eq!(
            side.errors, 0,
            "{} received {} server errors, last: {:?}",
            side.name, side.errors, side.last_error
        );
        assert!(
            side.games_finished >= target_games,
            "{} finished {} games, wanted {target_games}",
            side.name,
            side.games_finished
        );
        assert!(side.actions_sent > 0, "{} never acted", side.name);
    }
    assert!(
        report.neutral_placements > 0,
        "no neutral placement happened; the neutrals_placed-ack path was not exercised"
    );

    eprintln!(
        "OK: {} games, challenger sent {} actions, acceptor sent {} actions, \
         {} neutral placements, 0 illegal moves, 0 server errors",
        report.challenger.games_finished,
        report.challenger.actions_sent,
        report.acceptor.actions_sent,
        report.neutral_placements,
    );
}

struct Side {
    name: &'static str,
    errors: u64,
    illegal_moves: u64,
    actions_sent: u64,
    games_finished: u64,
    last_error: Option<String>,
}

struct Report {
    challenger: Side,
    acceptor: Side,
    neutral_placements: usize,
}

async fn play(port: u16, target_games: u64) -> Report {
    wait_for_port(port).await;

    let url = format!("ws://127.0.0.1:{port}/ws");
    let neutral_engine = Arc::new(NeutralOpeningEngine::default());

    // The challenger's timer is the sole send driver; a short interval only
    // makes the run quick, it does not change the mechanism.
    let (challenger, mut challenger_inbox) = Bot::new(
        Arc::new(BotConfig {
            backend_url: url.clone(),
            name_prefix: "ITestChallenger".to_owned(),
            move_budget: Duration::from_millis(50),
            challenger: true,
            challenge_interval: Duration::from_secs(2),
            rng_seed: Some(0x5EED),
            ..BotConfig::default()
        }),
        Arc::new(GreedyEngine),
    );
    let (acceptor, mut acceptor_inbox) = Bot::new(
        Arc::new(BotConfig {
            backend_url: url,
            name_prefix: "ITestAcceptor".to_owned(),
            move_budget: Duration::from_millis(50),
            ..BotConfig::default()
        }),
        neutral_engine.clone(),
    );

    let acceptor_task = {
        let bot = acceptor.clone();
        tokio::spawn(async move { virus_proto::run_forever(&bot, &mut acceptor_inbox).await })
    };
    // Let the acceptor register before the challenger's first tick, so its very
    // first users_update already lists a peer.
    tokio::time::sleep(Duration::from_millis(300)).await;
    let challenger_task = {
        let bot = challenger.clone();
        tokio::spawn(async move { virus_proto::run_forever(&bot, &mut challenger_inbox).await })
    };

    let started = Instant::now();
    loop {
        let done = challenger.core().counters.games_finished >= target_games
            && acceptor.core().counters.games_finished >= target_games;
        let broken = challenger.core().counters.errors > 0 || acceptor.core().counters.errors > 0;
        if done || broken {
            break;
        }
        if started.elapsed() > OVERALL_TIMEOUT {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    challenger_task.abort();
    acceptor_task.abort();

    Report {
        challenger: side("challenger", &challenger),
        acceptor: side("acceptor", &acceptor),
        neutral_placements: neutral_engine.placements.load(Ordering::SeqCst),
    }
}

fn side(name: &'static str, bot: &Bot) -> Side {
    let core = bot.core();
    Side {
        name,
        errors: core.counters.errors,
        illegal_moves: core.counters.illegal_moves,
        actions_sent: core.counters.actions_sent,
        games_finished: core.counters.games_finished,
        last_error: core.last_error.clone(),
    }
}

struct Server {
    binary: PathBuf,
    /// True when the overlay patch applied, so `VSBOT_ITEST_PORT` is honoured.
    port_is_configurable: bool,
}

/// The stock `main.go` line the port patch anchors on.
const LISTEN_ANCHOR: &str = r#"err := http.ListenAndServe(":8080", nil)"#;

fn build_server(workdir: &Path) -> Server {
    let backend = std::env::var("VSBOT_ITEST_BACKEND")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            Path::new(&std::env::var("HOME").expect("HOME")).join("Project/virusgame/backend")
        });
    let main_go = backend.join("main.go");
    assert!(
        main_go.exists(),
        "no Go server source at {} — set VSBOT_ITEST_BACKEND",
        backend.display()
    );

    // Substitute a port-configurable `main.go` for this build only. `-overlay`
    // is a read-only redirection inside the Go toolchain: the checkout on disk
    // is never written to.
    let source = std::fs::read_to_string(&main_go).expect("read main.go");
    let port_is_configurable = source.contains(LISTEN_ANCHOR);
    let mut command = std::process::Command::new("go");
    command.current_dir(&backend);
    if port_is_configurable {
        let patched = workdir.join("main_patched.go");
        std::fs::write(
            &patched,
            source.replace(
                LISTEN_ANCHOR,
                "addr := \":8080\"\n\tif p := os.Getenv(\"VSBOT_ITEST_PORT\"); p != \"\" {\n\
                 \t\taddr = \":\" + p\n\t}\n\terr := http.ListenAndServe(addr, nil)",
            ),
        )
        .expect("write patched main.go");
        let overlay = workdir.join("overlay.json");
        std::fs::write(
            &overlay,
            serde_json::json!({ "Replace": { main_go.to_str().expect("utf-8 path"): patched } })
                .to_string(),
        )
        .expect("write overlay.json");
        command.arg("build").arg("-overlay").arg(&overlay);
    } else {
        eprintln!("warning: main.go no longer matches the port patch anchor; using :8080");
        command.arg("build");
    }

    let binary = workdir.join("vs-server");
    let status = command
        .arg("-o")
        .arg(&binary)
        .arg(".")
        .status()
        .expect("`go build` — is the Go toolchain installed?");
    assert!(status.success(), "go build failed in {}", backend.display());
    Server {
        binary,
        port_is_configurable,
    }
}

fn port_is_free(port: u16) -> bool {
    std::net::TcpListener::bind(("127.0.0.1", port)).is_ok()
}

/// A port the kernel just handed out and immediately released. Racy in theory;
/// in a single-test process it is the standard trick and good enough.
fn free_port() -> u16 {
    std::net::TcpListener::bind(("127.0.0.1", 0))
        .expect("bind an ephemeral port")
        .local_addr()
        .expect("local address")
        .port()
}

async fn wait_for_port(port: u16) {
    let deadline = Instant::now() + Duration::from_secs(30);
    while Instant::now() < deadline {
        if tokio::net::TcpStream::connect(("127.0.0.1", port))
            .await
            .is_ok()
        {
            return;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("the server never accepted a connection on :{port}");
}

fn tail(text: &str) -> String {
    let lines: Vec<&str> = text.lines().collect();
    lines[lines.len().saturating_sub(40)..].join("\n")
}
