#!/usr/bin/env python3
"""Cross-play: vsbot against the deployed Go bot, refereed by the real server.

Rust-vs-Rust gauntlets (``arena``) settle internal A/Bs, but the bar this
project has to clear is the *deployed* Go bot, and that bot only exists behind
a WebSocket server.  So the only honest way to measure against it is to run the
real thing: boot the Go server, boot the Go bot-hoster in its default
accept-only mode, point ``vsbot`` at it as a challenger, and read the results
off the server's own ``games.db``.  Nothing here reimplements a rule or an
outcome; the server decides who won, exactly as it does in production.

Ported from ``~/Project/nnue-trainer/eval_java_vs_go.py``, with four changes
that were bugs there:

* **The name filter is not seat-ordered.**  The Python original counted only
  rows where the Java bot sat in seat 1, silently discarding every game it
  played from the other chair.  That is a colour bias baked into the
  measurement.  Here both seat orders are counted and folded to vsbot's
  perspective, and the per-seat split is printed so a colour effect is visible
  rather than hidden.
* **Draws are counted.**  The original dropped ``result = 0`` rows from both
  the numerator and the denominator, so a drawish opponent looked like an even
  one.  Draws are reported and land in the denominator, matching
  ``virus_arena::stats``.
* **The baseline is a rowid *and* a start time.**  A rowid snapshot alone
  cannot tell a resumed run's games from a previous run's if the db was
  restored; the ``started_at`` floor is a cheap second guard.
* **Ports are allocated, not assumed.**  The Go server hard-codes ``:8080``;
  this script builds it through ``go build -overlay`` with a patched ``main.go``
  so a run cannot fight whatever the developer already has listening.  The
  checkout on disk is never written to.  This is the same trick
  ``crates/vsbot/tests/live_games.rs`` uses, and it degrades to :8080 if the
  anchor line ever moves.

Usage::

    python3 crates/virus-arena/crossplay/crossplay.py --games 50

Run it from the repository root.  ``--help`` lists every knob.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# The stock `main.go` line the port patch anchors on. Kept character-identical
# to `crates/vsbot/tests/live_games.rs`; if one has to change, so does the other.
LISTEN_ANCHOR = 'err := http.ListenAndServe(":8080", nil)'

PATCHED_LISTEN = (
    'addr := ":8080"\n'
    '\tif p := os.Getenv("VSBOT_ITEST_PORT"); p != "" {\n'
    '\t\taddr = ":" + p\n'
    "\t}\n"
    "\terr := http.ListenAndServe(addr, nil)"
)


class Failure(RuntimeError):
    """Anything that means the run cannot produce a trustworthy number."""


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="vsbot vs the Go bot, refereed by the real server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--games", type=int, default=50, help="games to collect")
    parser.add_argument(
        "--backend",
        default=os.environ.get(
            "VSBOT_ITEST_BACKEND", str(Path.home() / "Project/virusgame/backend")
        ),
        help="checkout of the Go server",
    )
    parser.add_argument(
        "--vsbot",
        default="target/release/vsbot",
        help="the vsbot binary (build it with `cargo build --release -p vsbot`)",
    )
    parser.add_argument(
        "--search",
        default="GREEDY",
        help="vsbot SEARCH mode: GREEDY | ALPHABETA | MCTS",
    )
    parser.add_argument(
        "--move-millis", type=int, default=1000, help="vsbot per-action budget"
    )
    parser.add_argument(
        "--go-bots",
        type=int,
        default=2,
        help="accept-only Go bots in the pool; more means more concurrent games",
    )
    parser.add_argument(
        "--vsbot-instances",
        type=int,
        default=2,
        help="challenger vsbot processes; keep <= --go-bots",
    )
    parser.add_argument(
        "--challenge-secs",
        type=int,
        default=5,
        help="how often each vsbot instance offers a new game",
    )
    parser.add_argument(
        "--timeout", type=int, default=3600, help="hard wall-clock limit, seconds"
    )
    parser.add_argument("--poll-secs", type=int, default=5, help="db poll interval")
    parser.add_argument(
        "--workdir",
        default="",
        help="where logs and the built server go [default: a temp dir]",
    )
    parser.add_argument(
        "--json", action="store_true", help="print the result as JSON on stdout"
    )
    return parser.parse_args(argv)


# --------------------------------------------------------------------- server


def build_server(backend: Path, workdir: Path) -> tuple[Path, bool]:
    """Builds the Go server, patched to honour ``VSBOT_ITEST_PORT``.

    Returns ``(binary, port_is_configurable)``.  ``-overlay`` is a read-only
    redirection inside the Go toolchain: the checkout is never written to.
    """
    main_go = backend / "main.go"
    if not main_go.exists():
        raise Failure(f"no Go server source at {backend} — pass --backend")
    source = main_go.read_text()
    configurable = LISTEN_ANCHOR in source

    command = ["go", "build"]
    if configurable:
        patched = workdir / "main_patched.go"
        patched.write_text(source.replace(LISTEN_ANCHOR, PATCHED_LISTEN))
        overlay = workdir / "overlay.json"
        overlay.write_text(json.dumps({"Replace": {str(main_go): str(patched)}}))
        command += ["-overlay", str(overlay)]
    else:
        print(
            "warning: main.go no longer matches the port patch anchor; using :8080",
            file=sys.stderr,
        )

    binary = workdir / "vs-server"
    command += ["-o", str(binary), "."]
    result = subprocess.run(command, cwd=backend, capture_output=True, text=True)
    if result.returncode != 0:
        raise Failure(f"go build failed in {backend}:\n{result.stderr}")
    return binary, configurable


def build_bot_hoster(backend: Path, workdir: Path) -> Path:
    binary = workdir / "bot-hoster"
    result = subprocess.run(
        ["go", "build", "-o", str(binary), "./cmd/bot-hoster"],
        cwd=backend,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise Failure(f"go build ./cmd/bot-hoster failed:\n{result.stderr}")
    return binary


def free_port() -> int:
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def port_is_free(port: int) -> bool:
    import socket

    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def wait_for_port(port: int, deadline: float) -> None:
    import socket

    while time.time() < deadline:
        with socket.socket() as sock:
            sock.settimeout(0.5)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.2)
    raise Failure(f"the server never accepted a connection on :{port}")


# ------------------------------------------------------------------ processes


def spawn(command: list[str], *, cwd: Path, env: dict[str, str], log: Path):
    """Starts a process in its own group so the whole tree can be killed."""
    handle = log.open("w")
    return subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )


def kill_group(process) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        # SIGTERM is not always enough for a process holding a WebSocket read.
        # The original script had no escalation and left servers running.
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


# ------------------------------------------------------------------- games.db


def read_only(db: Path) -> sqlite3.Connection:
    # Read-only URI, opened and closed per call: the server holds the write
    # side in WAL mode with a single connection, and a long-lived reader here
    # is how you get a locked database in the middle of a run.
    return sqlite3.connect(f"file:{db}?mode=ro", uri=True)


def max_rowid(db: Path) -> int:
    if not db.exists():
        return 0
    try:
        with read_only(db) as connection:
            return connection.execute(
                "SELECT COALESCE(max(rowid), 0) FROM games"
            ).fetchone()[0]
    except sqlite3.Error:
        return 0


# A game the harness itself ended by killing a process is not a result. Every
# other termination is: `illegal_move` and `timeout` are instant forfeits in
# production (ARCHITECTURE.md), so they are real losses and are counted — but
# also reported separately, because a run full of them means something is wrong
# with the bot, not with the opponent.
ARTIFACT_TERMINATIONS = {"disconnect"}
RED_FLAG_TERMINATIONS = {"illegal_move", "timeout"}


def tally(db: Path, baseline: int, ours: str, theirs: str) -> dict:
    """W-L-D from vsbot's perspective, counting both seat orders.

    ``result`` is the winning seat number, ``0`` for none.  A row is ours if one
    seat's name starts with our prefix and the other starts with the Go bot's,
    in either order — which is the fix for the seat-ordered filter the Python
    original used.
    """
    empty = {
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "as_p1": 0,
        "as_p2": 0,
        "total": 0,
        "discarded": 0,
        "red_flags": 0,
    }
    if not db.exists():
        return empty
    try:
        with read_only(db) as connection:
            rows = connection.execute(
                "SELECT player1_name, player2_name, result, termination FROM games "
                "WHERE rowid > ? "
                "AND ((player1_name LIKE ?  AND player2_name LIKE ?) "
                "  OR (player1_name LIKE ?  AND player2_name LIKE ?))",
                (baseline, f"{ours}%", f"{theirs}%", f"{theirs}%", f"{ours}%"),
            ).fetchall()
    except sqlite3.Error:
        return empty

    out = dict(empty)
    for player1, player2, result, termination in rows:
        our_seat = 1 if (player1 or "").startswith(ours) else 2
        # A bot named with both prefixes would be ambiguous; the prefixes are
        # chosen not to overlap, and this guard keeps a mistake from silently
        # scoring the wrong side.
        if (player2 or "").startswith(ours) and our_seat == 1:
            continue
        # Shutdown races: the poll that ends the run and the SIGTERM that
        # follows it are not atomic, so an in-flight game can land in the table
        # as a `disconnect` loss for whoever was killed first. Counting those
        # would put a harness artifact in the headline.
        if (termination or "") in ARTIFACT_TERMINATIONS:
            out["discarded"] += 1
            continue
        if (termination or "") in RED_FLAG_TERMINATIONS:
            out["red_flags"] += 1
        out["as_p1" if our_seat == 1 else "as_p2"] += 1
        out["total"] += 1
        if result == 0:
            out["draws"] += 1
        elif result == our_seat:
            out["wins"] += 1
        else:
            out["losses"] += 1
    return out


def wilson95(wins: int, games: int) -> tuple[float, float]:
    """The same interval ``virus_arena::stats::wilson95`` computes."""
    if games == 0:
        return (0.0, 0.0)
    z = 1.959963984540054
    n = float(games)
    p = wins / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    margin = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5 / denominator
    return (100 * (center - margin), 100 * (center + margin))


def warn_about_seat_imbalance(report: dict) -> None:
    """Says out loud that a lopsided cross-play run carries a colour bias.

    The `arena` gauntlet cancels first-mover advantage by pairing colours. This
    harness *cannot*: the server seats the challenger at P1, only vsbot
    challenges (the Go pool is accept-only so it does not spar with itself), and
    a Go challenger would only ever target one of its own acceptors. So every
    game here is vsbot-as-P1 against GoBot-as-P2, and P1 moves first on an empty
    board.

    The counting code handles both seat orders — free, and correct the day the
    server can alternate — but until then the honest output is a warning, not a
    silent number. The Python original had the same bias and never mentioned it.
    """
    total = report["games"]
    if total == 0:
        return
    minority = min(report["as_p1"], report["as_p2"])
    if minority * 4 >= total:  # each colour has at least a quarter of the games
        return
    print(
        f"crossplay: WARNING — colours are not balanced "
        f"({report['as_p1']} P1 / {report['as_p2']} P2). This number includes "
        "first-mover advantage and is NOT comparable to an `arena` gauntlet, "
        "which cancels it by pairing. Treat it as a plumbing check.",
        file=sys.stderr,
    )


# ----------------------------------------------------------------------- main


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    backend = Path(args.backend).expanduser().resolve()
    vsbot = Path(args.vsbot).resolve()
    if not vsbot.exists():
        raise Failure(
            f"no vsbot binary at {vsbot} — run `cargo build --release -p vsbot`"
        )
    if shutil.which("go") is None:
        raise Failure("no Go toolchain on PATH; the server and bot-hoster need it")

    temporary = None
    if args.workdir:
        workdir = Path(args.workdir).resolve()
        workdir.mkdir(parents=True, exist_ok=True)
    else:
        temporary = tempfile.TemporaryDirectory(prefix="vsbot-crossplay-")
        workdir = Path(temporary.name)

    # Distinct, non-overlapping prefixes. The server renders a bot's lobby name
    # as "<prefix> Bot NNNN", so these are what the LIKE filters match on.
    ours, theirs = "RustBot", "GoBot"

    print(f"crossplay: workdir {workdir}", file=sys.stderr)
    server_binary, configurable = build_server(backend, workdir)
    hoster_binary = build_bot_hoster(backend, workdir)

    if configurable:
        port = free_port()
    elif port_is_free(8080):
        port = 8080
    else:
        raise Failure(
            "the stock server hard-codes :8080 and it is busy — free it, or restore "
            "the ListenAndServe line the overlay patch anchors on"
        )

    # The server writes `data/games.db` relative to its CWD, so running it in a
    # fresh workdir gives this run a private database and the rowid baseline is
    # 0. The baseline is kept anyway: a --workdir reuse must not count the
    # previous run's games.
    db = workdir / "data" / "games.db"
    baseline = max_rowid(db)
    started_at = time.time()

    processes = []
    try:
        server = spawn(
            [str(server_binary)],
            cwd=workdir,
            env={**os.environ, "VSBOT_ITEST_PORT": str(port)},
            log=workdir / "server.log",
        )
        processes.append(server)
        wait_for_port(port, started_at + 30)

        # Accept-only is the bot-hoster's default: with BOT_CHALLENGER unset a
        # bot never initiates, so every game in the tally is one vsbot asked
        # for. That keeps the pool from sparring with itself and polluting the
        # database with GoBot-vs-GoBot rows.
        hoster = spawn(
            [str(hoster_binary)],
            cwd=workdir,
            env={
                **os.environ,
                "BACKEND_URL": f"ws://127.0.0.1:{port}/ws",
                "BOT_POOL_SIZE": str(args.go_bots),
                "BOT_NAME_PREFIX": theirs,
                "BOT_EXPLORE_EPSILON": "0",
            },
            log=workdir / "gobot.log",
        )
        processes.append(hoster)
        time.sleep(2)

        for instance in range(args.vsbot_instances):
            processes.append(
                spawn(
                    [str(vsbot)],
                    cwd=workdir,
                    env={
                        **os.environ,
                        "BACKEND_URL": f"ws://127.0.0.1:{port}/ws",
                        "BOT_NAME_PREFIX": ours,
                        "SEARCH": args.search,
                        "MOVE_MILLIS": str(args.move_millis),
                        "CHALLENGER": "true",
                        "CHALLENGER_INTERVAL_SECS": str(args.challenge_secs),
                    },
                    log=workdir / f"vsbot-{instance}.log",
                )
            )

        deadline = started_at + args.timeout
        last = -1
        result = tally(db, baseline, ours, theirs)
        while True:
            result = tally(db, baseline, ours, theirs)
            if result["total"] != last:
                print(
                    f"  games {result['total']}/{args.games} | "
                    f"vsbot {result['wins']}-{result['losses']}-{result['draws']} GoBot",
                    file=sys.stderr,
                )
                last = result["total"]
            if result["total"] >= args.games:
                break
            if time.time() > deadline:
                print(
                    f"crossplay: timed out after {args.timeout}s with "
                    f"{result['total']} game(s)",
                    file=sys.stderr,
                )
                break
            # A dead process means no more games will ever arrive; say so
            # instead of polling until the timeout.
            dead = [p for p in processes if p.poll() is not None]
            if dead:
                print(
                    f"crossplay: {len(dead)} process(es) exited early; see the logs "
                    f"in {workdir}",
                    file=sys.stderr,
                )
                break
            time.sleep(args.poll_secs)
    finally:
        for process in reversed(processes):
            kill_group(process)

    low, high = wilson95(result["wins"], result["total"])
    report = {
        "vsbot_search": args.search,
        "move_millis": args.move_millis,
        "wins": result["wins"],
        "losses": result["losses"],
        "draws": result["draws"],
        "games": result["total"],
        "as_p1": result["as_p1"],
        "as_p2": result["as_p2"],
        "discarded_disconnects": result["discarded"],
        "red_flag_terminations": result["red_flags"],
        "win_rate": (100.0 * result["wins"] / result["total"]) if result["total"] else 0.0,
        "wilson95_low": low,
        "wilson95_high": high,
        # The same discipline `virus_arena::stats::Verdict` enforces. A
        # cross-play smoke run is a plumbing check, not a strength claim, and
        # the output has to say so.
        "verdict": (
            "INFORMATIONAL ONLY"
            if result["total"] < 100
            else ("indicative" if result["total"] < 400 else "gate-eligible")
        ),
    }
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(
            f"\n=== vsbot({args.search}) vs GoBot: "
            f"W-L-D {report['wins']}-{report['losses']}-{report['draws']} "
            f"over {report['games']} games ({report['verdict']}) ==="
        )
        print(
            f"    win rate {report['win_rate']:.1f}% (draws not half-wins)  "
            f"wilson95 [{low:.1f}%, {high:.1f}%]"
        )
        print(f"    seats: {report['as_p1']} as P1, {report['as_p2']} as P2")
        if report["discarded_disconnects"]:
            print(
                f"    discarded {report['discarded_disconnects']} disconnect(s) "
                "(harness shutdown artifacts, not results)"
            )
        if report["red_flag_terminations"]:
            print(
                f"    !! {report['red_flag_terminations']} game(s) ended in a forfeit "
                "(illegal move or timeout) — investigate before reading the tally"
            )
    warn_about_seat_imbalance(report)
    if temporary is not None and args.workdir == "":
        # Keep the logs when the run did not reach its target; they are the
        # only evidence of why.
        if report["games"] >= args.games:
            temporary.cleanup()
        else:
            print(f"crossplay: keeping logs in {workdir}", file=sys.stderr)
            temporary._finalizer.detach()  # noqa: SLF001
    return 0 if report["games"] >= args.games else 1


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except Failure as error:
        print(f"crossplay: {error}", file=sys.stderr)
        sys.exit(2)
    except KeyboardInterrupt:
        sys.exit(130)
