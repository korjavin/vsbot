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
//! | Variable                 | Default                                     |
//! |--------------------------|---------------------------------------------|
//! | `VSBOT_ITEST`            | unset — the tests skip                      |
//! | `VSBOT_ITEST_BACKEND`    | `$HOME/Project/virusgame/backend`           |
//! | `VSBOT_ITEST_GAMES`      | `3` (protocol run), `1` (MCTS run)          |
//! | `VSBOT_ITEST_MCTS_MS`    | `150` — MCTS move budget                    |
//! | `VSBOT_ITEST_SOAK`       | unset — the ponder soak skips               |
//! | `VSBOT_ITEST_SOAK_GAMES` | `20`                                        |
//! | `VSBOT_ITEST_SOAK_TURN_MS` | `240` — turn budget for the soak          |
//! | `VSBOT_ITEST_SOAK_OPPONENT` | `greedy` \| `mcts`                         |
//!
//! Three scenarios live here. The protocol run pits two instant engines against
//! each other and is about *ordering*; the MCTS run puts the real champion on
//! one side and is about the engine adapter — that a searched move survives the
//! round trip and that the domain guard never lets an out-of-domain position
//! reach the searcher's asserts. The **ponder soak** is the acceptance gate for
//! S2's T3: twenty-plus games with `VSBOT_PONDER=true` on one side, asserting
//! zero forfeits, zero illegal moves, and — the point of the exercise — zero
//! out-of-turn emissions. They each start their own server, so [`SERIAL`] keeps
//! them from racing for a port.
//!
//! ```text
//! VSBOT_ITEST=1 VSBOT_ITEST_SOAK=1 cargo test -p vsbot --test live_games \
//!   --release ponder_soak -- --nocapture
//! ```
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
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use virus_core::{Action, State};
use virus_proto::{
    Bot, BotConfig, EngineKind, GreedyEngine, SearchBudget, SearchEngine, SearchOutcome,
};
use vsbot::{build_engine, MctsSettings};

const OVERALL_TIMEOUT: Duration = Duration::from_secs(180);

/// Each scenario starts its own Go server, and the fallback path when the port
/// patch does not apply is the hard-coded `:8080`. Serialising the scenarios
/// keeps that from turning into a flake — cargo runs the tests in one binary on
/// several threads by default.
static SERIAL: Mutex<()> = Mutex::new(());

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

// The serialisation guard is deliberately held across the run: that is the
// whole point of it, and a blocking lock is correct here because the two test
// futures live on different runtimes and never contend from within one.
#[allow(clippy::await_holding_lock)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn two_bots_play_full_games_without_a_single_illegal_move() {
    let Some(_guard) = enabled("protocol") else {
        return;
    };
    let target_games: u64 = std::env::var("VSBOT_ITEST_GAMES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(3);

    let neutral_engine = Arc::new(NeutralOpeningEngine::default());
    let report = run_scenario(Scenario {
        label: "protocol",
        target_games,
        budget: Budget::PerAction(Duration::from_millis(50)),
        ponder: false,
        timeout: OVERALL_TIMEOUT,
        challenger: Arc::new(GreedyEngine),
        acceptor: neutral_engine.clone(),
    })
    .await;

    report.assert_clean(target_games);
    let placements = neutral_engine.placements.load(Ordering::SeqCst);
    assert!(
        placements > 0,
        "no neutral placement happened; the neutrals_placed-ack path was not exercised"
    );

    eprintln!(
        "OK: {} games, challenger sent {} actions, acceptor sent {} actions, \
         {placements} neutral placements, 0 illegal moves, 0 server errors",
        report.challenger.games_finished,
        report.challenger.actions_sent,
        report.acceptor.actions_sent,
    );
}

/// The acceptance gate for `SEARCH=MCTS`: the champion, built through the same
/// [`build_engine`] the binary calls, plays a full game against the reference
/// engine on a real server.
///
/// This is the run that catches the failure the unit tests cannot: a searched
/// action that is legal in the *searcher's* copy of the position but not in the
/// one the server holds. The server forfeits an illegal move instantly, so a
/// completed game with a clean hub log is the proof.
#[allow(clippy::await_holding_lock)] // see the note on the scenario above
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_mcts_champion_plays_a_full_game_without_a_single_illegal_move() {
    let Some(_guard) = enabled("mcts") else {
        return;
    };
    let target_games: u64 = std::env::var("VSBOT_ITEST_GAMES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1);
    let move_millis: u64 = std::env::var("VSBOT_ITEST_MCTS_MS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(150);

    // Exactly what `main` does, artifact included — a test that hand-built an
    // `MctsEngine` would prove nothing about the deployed wiring.
    let setup = build_engine(
        EngineKind::Mcts,
        &MctsSettings {
            artifact: Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../artifacts/mcts_champion.json"),
            seed: 1,
        },
    )
    .expect("the in-repo champion loads and validates");
    assert_eq!(setup.engine.name(), "mcts");
    eprintln!("vsbot: {}", setup.description);

    let report = run_scenario(Scenario {
        label: "mcts",
        target_games,
        budget: Budget::PerAction(Duration::from_millis(move_millis)),
        ponder: false,
        timeout: OVERALL_TIMEOUT,
        challenger: setup.engine,
        acceptor: Arc::new(GreedyEngine),
    })
    .await;

    report.assert_clean(target_games);
    eprintln!(
        "OK (MCTS): {} games at {move_millis}ms/move, mcts challenger sent {} actions, \
         greedy acceptor sent {} actions, 0 illegal moves, 0 server errors",
        report.challenger.games_finished,
        report.challenger.actions_sent,
        report.acceptor.actions_sent,
    );
}

/// The acceptance gate for pondering (bd `vsbot-dgv` T3).
///
/// Twenty-plus complete games with the challenger pondering through every
/// opponent turn, against the real Go server. Pondering means a search is
/// running on positions the bot may not act in, driven off message types the
/// turn-driver whitelist excludes — so the failure it risks is precisely the one
/// that forfeited two live games on 2026-08-08 (ARCHITECTURE.md invariant 2),
/// and only a live server produces the message *ordering* that would trigger it.
///
/// Three things are asserted, and each has a distinct evidence source:
///
/// * **zero illegal moves** — the hub logs `made illegal move` before it
///   forfeits (`hub.go:210`); the server log is grepped for it;
/// * **zero out-of-turn emissions** — `handleMove` answers an off-turn action
///   with a bare `error` frame and *no* log line (`hub.go:1016`), so the
///   client's `errors` counter is the only place it can appear. Zero there is
///   the proof;
/// * **zero forfeits** — every game reaches `game_end` on both sides.
///
/// The opponent is the **instant** reference engine by default, which is the
/// harsher choice and not the lazier one: a fast opponent packs `move_made`,
/// `game_state` and `turn_change` into the tightest possible window, which is
/// exactly the ordering that would race a pondering session into emitting off
/// its own turn. `VSBOT_ITEST_SOAK_OPPONENT=mcts` puts the champion on both
/// sides instead — a slower, more realistic opponent that gives the ponder tree
/// real thinking time, at the cost of a second search burning CPU. (Strength
/// under ponder is not this test's job; that is the deferred 400-game arena.)
///
/// The turn budget is scaled down so the run finishes in minutes; the allocator
/// is proportional, so the code paths are the deployed ones.
#[allow(clippy::await_holding_lock)] // see the note on the scenario above
#[tokio::test(flavor = "multi_thread", worker_threads = 6)]
async fn ponder_soak_plays_twenty_games_without_a_forfeit_or_an_out_of_turn_action() {
    if std::env::var("VSBOT_ITEST_SOAK").as_deref() != Ok("1") {
        eprintln!("skipping the ponder soak: set VSBOT_ITEST_SOAK=1 (it takes minutes)");
        return;
    }
    let Some(_guard) = enabled("ponder-soak") else {
        return;
    };
    let target_games: u64 = std::env::var("VSBOT_ITEST_SOAK_GAMES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(20);
    let turn_millis: u64 = std::env::var("VSBOT_ITEST_SOAK_TURN_MS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(240);

    let artifact = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../artifacts/mcts_champion.json");
    let engine = |seed: u64| {
        build_engine(
            EngineKind::Mcts,
            &MctsSettings {
                artifact: artifact.clone(),
                seed,
            },
        )
        .expect("the in-repo champion loads and validates")
        .engine
    };

    let slow_opponent = std::env::var("VSBOT_ITEST_SOAK_OPPONENT").as_deref() == Ok("mcts");
    let acceptor: Arc<dyn SearchEngine> = if slow_opponent {
        engine(2)
    } else {
        Arc::new(GreedyEngine)
    };

    let report = run_scenario(Scenario {
        label: "ponder-soak",
        target_games,
        budget: Budget::PerTurn(Duration::from_millis(turn_millis)),
        ponder: true,
        timeout: Duration::from_secs(1800),
        challenger: engine(1),
        acceptor,
    })
    .await;

    report.assert_clean(target_games);
    assert!(
        report.challenger.ponder_steps > 0,
        "the pondering side never received a ponder step; the soak proved nothing"
    );
    assert!(
        report.challenger.ponder_answers > 0,
        "no turn was ever answered out of the ponder tree"
    );
    assert_eq!(
        report.acceptor.ponder_steps, 0,
        "the control side must not have pondered"
    );
    assert_eq!(
        report.challenger.fallback_actions, 0,
        "the pondering side had to answer with its fallback {} times — the time manager \
         did not hold under ponder",
        report.challenger.fallback_actions
    );

    eprintln!(
        "OK (ponder soak): {} games at {turn_millis}ms/turn vs {}, ponderer sent {} actions \
         ({} ponder steps, {} pondered answers, {} fallbacks), control sent {} actions, \
         0 illegal moves, 0 server errors, 0 out-of-turn actions",
        report.challenger.games_finished,
        if slow_opponent { "mcts" } else { "greedy" },
        report.challenger.actions_sent,
        report.challenger.ponder_steps,
        report.challenger.ponder_answers,
        report.challenger.fallback_actions,
        report.acceptor.actions_sent,
    );
}

/// The `VSBOT_ITEST` gate plus the cross-scenario lock, or `None` to skip.
fn enabled(label: &str) -> Option<std::sync::MutexGuard<'static, ()>> {
    if std::env::var("VSBOT_ITEST").as_deref() != Ok("1") {
        eprintln!("skipping {label}: set VSBOT_ITEST=1 to run the live-server integration tests");
        return None;
    }
    // A panicking scenario poisons the lock; the next one is still perfectly
    // able to run, so take the guard regardless.
    Some(
        SERIAL
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()),
    )
}

/// How a scenario pays for its thinking.
#[derive(Clone, Copy, Debug)]
enum Budget {
    /// A fixed per-action budget; the intra-turn allocator is off.
    PerAction(Duration),
    /// A whole-turn budget the allocator splits — what the deployment runs.
    PerTurn(Duration),
}

/// One end-to-end run: which engines, for how many games, at what budget.
struct Scenario {
    /// Names the scenario's workdir, so concurrent runs never share a server.
    label: &'static str,
    target_games: u64,
    budget: Budget,
    /// Whether the *challenger* side ponders. The acceptor never does, so one
    /// run covers both the new behaviour and the control.
    ponder: bool,
    /// How long the whole run may take.
    timeout: Duration,
    challenger: Arc<dyn SearchEngine>,
    acceptor: Arc<dyn SearchEngine>,
}

impl Budget {
    fn apply(self, config: &mut BotConfig) {
        match self {
            Budget::PerAction(budget) => config.move_budget = Some(budget),
            Budget::PerTurn(budget) => {
                config.move_budget = None;
                config.turn_budget = budget;
            }
        }
    }
}

struct Side {
    name: &'static str,
    errors: u64,
    illegal_moves: u64,
    actions_sent: u64,
    games_finished: u64,
    fallback_actions: u64,
    ponder_steps: u64,
    ponder_answers: u64,
    last_error: Option<String>,
}

struct Report {
    challenger: Side,
    acceptor: Side,
    server_output: String,
}

impl Report {
    /// Both sides finished their games, neither was forfeited, and the hub
    /// agrees.
    fn assert_clean(&self, target_games: u64) {
        // Independent, server-side confirmation. The hub logs every rejected
        // action before it forfeits the offender, so a clean log is proof the
        // clean client counters are not just a client-side blind spot.
        assert!(
            !self.server_output.contains("made illegal move"),
            "the server logged an illegal move:\n{}",
            tail(&self.server_output)
        );
        for side in [&self.challenger, &self.acceptor] {
            assert_eq!(
                side.illegal_moves, 0,
                "{} was forfeited for an illegal move: {:?}",
                side.name, side.last_error
            );
            // An out-of-turn action is *not* an illegal move server-side: the
            // hub answers `handleMove`'s "It is not this player's turn" with a
            // plain `error` frame and no log line (hub.go:1016, :344-349). So
            // the client's error counter is the only place it can show up, and
            // a zero here is what proves no action was emitted off-turn.
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
    }
}

/// Builds and starts a server, plays the scenario, and stops the server.
async fn run_scenario(scenario: Scenario) -> Report {
    let workdir = std::env::temp_dir().join(format!(
        "vsbot-itest-{}-{}",
        std::process::id(),
        scenario.label
    ));
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

    let target_games = scenario.target_games;
    let timeout = scenario.timeout;
    let outcome = tokio::time::timeout(timeout, play(port, scenario)).await;

    let _ = child.kill().await;
    let server_output = std::fs::read_to_string(&server_log).unwrap_or_default();

    let Ok((challenger, acceptor)) = outcome else {
        panic!(
            "the run did not finish {target_games} games within {timeout:?}\n\
             --- server log tail ---\n{}",
            tail(&server_output)
        );
    };
    Report {
        challenger,
        acceptor,
        server_output,
    }
}

async fn play(port: u16, scenario: Scenario) -> (Side, Side) {
    wait_for_port(port).await;

    let url = format!("ws://127.0.0.1:{port}/ws");

    // The challenger's timer is the sole send driver; a short interval only
    // makes the run quick, it does not change the mechanism.
    let mut challenger_config = BotConfig {
        backend_url: url.clone(),
        name_prefix: "ITestChallenger".to_owned(),
        challenger: true,
        challenge_interval: Duration::from_secs(2),
        rng_seed: Some(0x5EED),
        // Only the challenger ponders, so one run carries both the new
        // behaviour and its control.
        ponder: scenario.ponder,
        ponder_budget: Duration::from_secs(5),
        ..BotConfig::default()
    };
    scenario.budget.apply(&mut challenger_config);
    let (challenger, mut challenger_inbox) =
        Bot::new(Arc::new(challenger_config), scenario.challenger);

    let mut acceptor_config = BotConfig {
        backend_url: url,
        name_prefix: "ITestAcceptor".to_owned(),
        ..BotConfig::default()
    };
    scenario.budget.apply(&mut acceptor_config);
    let (acceptor, mut acceptor_inbox) = Bot::new(Arc::new(acceptor_config), scenario.acceptor);
    let target_games = scenario.target_games;
    let timeout = scenario.timeout;

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
        if started.elapsed() > timeout {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    challenger_task.abort();
    acceptor_task.abort();

    (side("challenger", &challenger), side("acceptor", &acceptor))
}

fn side(name: &'static str, bot: &Bot) -> Side {
    let core = bot.core();
    Side {
        name,
        errors: core.counters.errors,
        illegal_moves: core.counters.illegal_moves,
        actions_sent: core.counters.actions_sent,
        games_finished: core.counters.games_finished,
        fallback_actions: core.counters.fallback_actions,
        ponder_steps: core.counters.ponder_steps,
        ponder_answers: core.counters.ponder_answers,
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
