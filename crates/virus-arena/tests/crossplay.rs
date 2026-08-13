//! Cross-play against the Go bot, driven through the real server.
//!
//! Opt-in, exactly like `crates/vsbot/tests/live_games.rs`: it needs a Go
//! toolchain, a checkout of the server, and a built `vsbot` binary, none of
//! which GitHub Actions has. Running it by default would either fail CI on
//! every machine or — worse — silently skip and look green.
//!
//! ```text
//! cargo build --release -p vsbot
//! VSBOT_CROSSPLAY=1 cargo test -p virus-arena --test crossplay -- --nocapture
//! ```
//!
//! | Variable                   | Default                            |
//! |----------------------------|------------------------------------|
//! | `VSBOT_CROSSPLAY`          | unset — the test skips             |
//! | `VSBOT_CROSSPLAY_GAMES`    | `50`                               |
//! | `VSBOT_CROSSPLAY_SEARCH`   | `GREEDY`                           |
//! | `VSBOT_CROSSPLAY_TIMEOUT`  | `1800` (seconds)                   |
//! | `VSBOT_ITEST_BACKEND`      | `$HOME/Project/virusgame/backend`  |
//!
//! The work is done by `crossplay/crossplay.py`, which boots the three
//! processes and reads the server's own `games.db`. This test is the thin
//! cargo-shaped wrapper around it, so the harness is reachable both ways: a
//! human runs the script, CI-style automation runs the test.

use std::path::PathBuf;
use std::process::Command;

/// The counting logic, checked in CI.
///
/// The harness itself cannot run on a GitHub runner — it needs a Go toolchain,
/// a checkout of the server and, for the Java arm, docker — so the test below
/// is opt-in and never runs there. But what a cross-play run *means* is decided
/// by pure functions over `games.db` rows: which seat we held, whether a
/// `disconnect` is a result, and how many of the games were actually different
/// games. Those need none of that infrastructure, and leaving them uncovered is
/// how the harness came to report `49-1` over what were five distinct games
/// (bd `vsbot-t3q.1`). `--self-test` exercises them against a synthetic
/// database in about a second.
#[test]
fn the_crossplay_tally_is_correct() {
    let script = repo_root().join("crates/virus-arena/crossplay/crossplay.py");
    assert!(script.exists(), "missing {}", script.display());
    let output = Command::new("python3")
        .arg(&script)
        .arg("--self-test")
        .output()
        .expect("python3 — is it on PATH? the harness needs it for sqlite3");
    assert!(
        output.status.success(),
        "crossplay --self-test failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn vsbot_plays_the_go_bot_through_the_real_server() {
    if std::env::var("VSBOT_CROSSPLAY").as_deref() != Ok("1") {
        eprintln!("skipping: set VSBOT_CROSSPLAY=1 to run the cross-play harness");
        return;
    }
    let games = env_or("VSBOT_CROSSPLAY_GAMES", "50");
    let search = env_or("VSBOT_CROSSPLAY_SEARCH", "GREEDY");
    let timeout = env_or("VSBOT_CROSSPLAY_TIMEOUT", "1800");

    let root = repo_root();
    let script = root.join("crates/virus-arena/crossplay/crossplay.py");
    assert!(script.exists(), "missing {}", script.display());

    // Release, not debug: the whole point of a wall-clock cross-play run is how
    // much search fits in a move, and a debug build measures the wrong engine.
    let vsbot = root.join("target/release/vsbot");
    assert!(
        vsbot.exists(),
        "no vsbot binary at {} — run `cargo build --release -p vsbot` first",
        vsbot.display()
    );

    let mut command = Command::new("python3");
    command
        .current_dir(&root)
        .arg(&script)
        .arg("--games")
        .arg(&games)
        .arg("--search")
        .arg(&search)
        .arg("--timeout")
        .arg(&timeout)
        .arg("--vsbot")
        .arg(&vsbot);
    if let Ok(backend) = std::env::var("VSBOT_ITEST_BACKEND") {
        command.arg("--backend").arg(backend);
    }

    let status = command
        .status()
        .expect("python3 — is it on PATH? the harness needs it for sqlite3");
    assert!(
        status.success(),
        "cross-play did not collect {games} games; see the log directory it printed"
    );
}

fn env_or(key: &str, fallback: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| fallback.to_owned())
}

/// The workspace root: this crate's manifest directory is `crates/virus-arena`.
fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|crates| crates.parent())
        .expect("crates/virus-arena has two ancestors")
        .to_path_buf()
}
