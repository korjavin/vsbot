#!/usr/bin/env python3
"""Cross-play: vsbot against a rival bot, refereed by the real server.

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

Seats, and why they used to be unbalanced
-----------------------------------------

The server seats the **challenger** at P1: ``handleAcceptChallenge`` builds the
game with ``Player1: challenge.FromUser``.  Challenge direction *is* the seat
assignment, so a run in which only one side ever challenges is a run played
entirely from one chair, first-mover advantage included.  That is what produced
the 98% in ``docs/benchmarks.md`` row 3 (bd ``vsbot-t3q.1``).

Which directions are available depends on the opponent, and the difference is
structural rather than a matter of configuration:

* ``--opponent go`` — **``ours`` only.**  The Go bot-hoster's challenger mode
  targets ``Manager.IsAcceptor(userID)``, which returns ``false`` for any id
  outside its own pool, so a Go challenger can only ever spar with its own
  acceptors.  It cannot challenge ``vsbot``, and therefore cannot seat ``vsbot``
  at P2.  ``--direction theirs``/``alternate`` are refused rather than silently
  producing another one-chair run.
* ``--opponent java`` — **all three.**  The Java bot's ``attemptChallenge``
  picks any eligible online user that is not itself and not in a game, so it
  will challenge ``vsbot``.  ``--direction alternate`` therefore runs the games
  in two phases — half with ``vsbot`` challenging (``vsbot`` at P1), half with
  the Java bot challenging (``vsbot`` at P2) — and the pooled number is
  colour-balanced the way an ``arena`` gauntlet is.

Usage::

    # vsbot vs the Go bot (one chair only; a plumbing check)
    python3 crates/virus-arena/crossplay/crossplay.py --games 50

    # vsbot vs the Java champion, colour-balanced (superiority.md S1)
    python3 crates/virus-arena/crossplay/crossplay.py \\
        --opponent java --direction alternate --search MCTS --games 400

Run it from the repository root.  ``--help`` lists every knob.
"""

from __future__ import annotations

import argparse
import hashlib
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
        description="vsbot vs a rival bot, refereed by the real server",
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
        "--mcts-artifact",
        default="artifacts/mcts_champion.json",
        help=(
            "net artifact for --search MCTS. Resolved to an absolute path before "
            "it reaches vsbot, which runs with its cwd set to the run's workdir "
            "and would otherwise never find a repo-relative one"
        ),
    )
    parser.add_argument(
        "--opponent",
        choices=("go", "java"),
        default="go",
        help="which rival to play: the Go bot-hoster, or the Java bot in docker",
    )
    parser.add_argument(
        "--direction",
        choices=("ours", "theirs", "alternate"),
        default="ours",
        help=(
            "who challenges, which is who sits at P1. 'alternate' splits the "
            "games into two equal phases and is the only colour-balanced mode. "
            "Only --opponent java supports anything but 'ours' -- see the "
            "module docstring"
        ),
    )
    parser.add_argument(
        "--go-bots",
        type=int,
        default=2,
        help="accept-only Go bots in the pool; more means more concurrent games",
    )
    parser.add_argument(
        "--java-image",
        default="ghcr.io/korjavin/nnue-trainer:latest",
        help="docker image for the Java bot (there is no JVM on this host)",
    )
    parser.add_argument(
        "--java-search",
        default="MCTS",
        help="Java bot SEARCH mode: MCTS | GOBOT",
    )
    parser.add_argument(
        "--java-mcts-value",
        default="net",
        help=(
            "Java MCTS_VALUE. 'net' uses the champion artifact's value head, "
            "which is what makes this a same-net implementation comparison"
        ),
    )
    parser.add_argument(
        "--vsbot-instances",
        type=int,
        default=2,
        help=(
            "challenger vsbot processes. More than one buys concurrency at the "
            "cost of throughput: vsbot's challenger picks any idle peer, "
            "including another vsbot, so roughly half the games at N>1 are "
            "vsbot-vs-vsbot. Those are excluded from the tally by the name "
            "filter, never miscounted -- they are only wasted compute"
        ),
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
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="check the counting logic against a synthetic db and exit; needs "
        "no server, no bots and no docker, so CI can run it",
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


def docker_available() -> None:
    """Fails early, with the reason, rather than mid-run with a dead process."""
    if shutil.which("docker") is None:
        raise Failure("no docker on PATH; --opponent java runs the Java bot in a container")
    probe = subprocess.run(
        ["docker", "info"], capture_output=True, text=True
    )
    if probe.returncode != 0:
        raise Failure(f"docker is installed but not usable:\n{probe.stderr.strip()}")


def spawn_java(
    *,
    image: str,
    name: str,
    port: int,
    prefix: str,
    search: str,
    mcts_value: str,
    move_millis: int,
    challenger: bool,
    challenge_secs: int,
    cwd: Path,
    log: Path,
):
    """Starts the Java bot in a container on the host network.

    ``--network host`` is what lets the container reach the server this script
    just booted on a loopback port; a bridge network would need the server bound
    to a routable address, which the port patch does not do.

    The container is named so cleanup can remove it explicitly: killing the
    ``docker run`` client does not necessarily stop the container, and a
    survivor would keep playing into the next phase's tally.
    """
    # The Java bot reads its name prefix from the URL, not from an env var.
    backend = f"ws://127.0.0.1:{port}/ws?bot=true&namePrefix={prefix}"
    command = [
        "docker", "run", "--rm", "--name", name, "--network", "host",
        "-e", f"BACKEND_URL={backend}",
        "-e", f"SEARCH={search}",
        "-e", f"MCTS_VALUE={mcts_value}",
        "-e", f"MCTS_MOVE_MILLIS={move_millis}",
        "-e", f"CHALLENGER_MODE={'true' if challenger else 'false'}",
        "-e", f"CHALLENGE_INTERVAL_SEC={challenge_secs}",
        image,
    ]
    return spawn(command, cwd=cwd, env=dict(os.environ), log=log)


def remove_container(name: str) -> None:
    subprocess.run(
        ["docker", "rm", "-f", name], capture_output=True, text=True
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


def game_fingerprint(pgn: str) -> str:
    """A digest of the *moves* of a game, ignoring how long each one took.

    Two deterministic engines replay one game forever, and the wall-clock
    `duration_cs` field is the one part of the record that always differs — so
    hashing the raw PGN would report 400 distinct games where there are three.
    Only `(turn, player, [(type, row, col)])` goes into the digest.
    """
    try:
        turns = json.loads(pgn or "[]")
    except (TypeError, ValueError):
        return ""
    moves = [
        (
            turn.get("turn"),
            turn.get("player"),
            tuple(
                (move.get("type"), move.get("row"), move.get("col"))
                for move in turn.get("moves", [])
            ),
        )
        for turn in turns
    ]
    return hashlib.md5(repr(moves).encode()).hexdigest()


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
        # Wins and draws split by our chair. A cross-play run cannot pair
        # colours the way `arena` does, so the only way the first-mover effect
        # becomes visible instead of baked-in is to report it per seat.
        "wins_p1": 0,
        "wins_p2": 0,
        "draws_p1": 0,
        "draws_p2": 0,
        "total": 0,
        "discarded": 0,
        "red_flags": 0,
        # How many of the counted games were actually *different* games. Both
        # engines play deterministically, so this can be far below `total`; see
        # `warn_about_low_diversity`.
        "distinct": 0,
        # The digests behind `distinct`. Carried so a multi-shard run can union
        # them: two shards that each played three distinct games may well have
        # played the *same* three, and summing `distinct` would hide that.
        "fingerprints": set(),
    }
    if not db.exists():
        return empty
    try:
        with read_only(db) as connection:
            rows = connection.execute(
                "SELECT player1_name, player2_name, result, termination, pgn_content "
                "FROM games "
                "WHERE rowid > ? "
                "AND ((player1_name LIKE ?  AND player2_name LIKE ?) "
                "  OR (player1_name LIKE ?  AND player2_name LIKE ?))",
                (baseline, f"{ours}%", f"{theirs}%", f"{theirs}%", f"{ours}%"),
            ).fetchall()
    except sqlite3.Error:
        return empty

    out = dict(empty)
    fingerprints: set[str] = set()
    out["fingerprints"] = fingerprints
    for player1, player2, result, termination, pgn in rows:
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
        seat = "p1" if our_seat == 1 else "p2"
        out[f"as_{seat}"] += 1
        out["total"] += 1
        fingerprints.add(game_fingerprint(pgn))
        if result == 0:
            out["draws"] += 1
            out[f"draws_{seat}"] += 1
        elif result == our_seat:
            out["wins"] += 1
            out[f"wins_{seat}"] += 1
        else:
            out["losses"] += 1
    out["distinct"] = len(fingerprints)
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
    """Says out loud when a lopsided cross-play run carries a colour bias.

    The server seats the challenger at P1, so challenge direction is seat
    assignment.  ``--direction alternate`` splits the games between the two
    directions and produces a balanced run; ``--direction ours`` — the only mode
    the Go bot-hoster can support, because its challenger only targets its own
    pool's acceptors — plays every game from one chair and carries a full
    first-mover advantage.

    Measured on this project's own board (`docs/benchmarks.md`), that advantage
    is not a rounding error, so a one-chair number is a plumbing check and not a
    strength result.  The warning is the honest output; the Python original had
    the same bias and never mentioned it.
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
        "which cancels it by pairing. Treat it as a plumbing check, and use "
        "--direction alternate (needs --opponent java) for a balanced number.",
        file=sys.stderr,
    )


def warn_about_low_diversity(report: dict) -> None:
    """The finding behind bd ``vsbot-t3q.1``, turned into a guard rail.

    A cross-play run has no opening randomisation.  ``arena`` injects some on
    purpose (eps 0.15 over the first 8 turns) precisely because two
    deterministic engines otherwise replay one game forever; this harness drives
    two *deployed* bots that both play argmax with no root noise, so the only
    thing that varies between games is wall-clock jitter changing how deep a
    search got.

    That is how row 3 of ``docs/benchmarks.md`` reported ``49-1`` with a Wilson
    interval of ``[89.5%, 99.6%]``: the interval is binomial and assumes 50
    independent games, but a 9-game re-run of the same configuration contained
    only **4** distinct games and an opening identical across all nine.  The
    number was two or three game outcomes, repeated.

    So the game count is not the sample size, and any interval computed from it
    is optimistic by however much this ratio says.  Report it rather than
    letting a confident-looking percentage stand on three games.
    """
    total = report["games"]
    distinct = report["distinct_games"]
    if total == 0 or distinct == 0:
        return
    if distinct * 2 >= total:  # at least half the games were different games
        return
    print(
        f"crossplay: WARNING — only {distinct} of {total} games were distinct "
        f"(the rest are replays: both bots play deterministically and nothing "
        f"randomises the opening). The Wilson interval above assumes {total} "
        f"independent games and is therefore far too narrow — the effective "
        f"sample is closer to {distinct}. Do not quote this as a strength "
        f"result. See bd vsbot-t3q.1.",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------- phases


def run_phase(
    *,
    args,
    workdir: Path,
    port: int,
    db: Path,
    baseline: int,
    ours: str,
    theirs: str,
    vsbot: Path,
    hoster_binary,
    we_challenge: bool,
    target: int,
    deadline: float,
    phase_index: int,
    server_processes: list,
) -> dict:
    """Plays one challenge direction until the cumulative ``target`` is reached.

    The server outlives a phase; only the bots are restarted, because challenge
    direction is fixed at process start on both sides (``CHALLENGER`` for vsbot,
    ``CHALLENGER_MODE`` for the Java bot).  Tearing the bots down between phases
    is also what keeps a phase-1 game from finishing inside phase 2 and being
    attributed to the wrong chair: every game in the table was started by
    whichever side was the challenger while it was running.
    """
    processes: list = []
    containers: list[str] = []
    tag = f"{phase_index}-{'ours' if we_challenge else 'theirs'}"
    result = tally(db, baseline, ours, theirs)
    try:
        if args.opponent == "go":
            # Accept-only unless asked otherwise; with BOT_CHALLENGER unset a Go
            # bot never initiates, so the pool cannot spar with itself and fill
            # games.db with GoBot-vs-GoBot rows the name filter would drop.
            processes.append(
                spawn(
                    [str(hoster_binary)],
                    cwd=workdir,
                    env={
                        **os.environ,
                        "BACKEND_URL": f"ws://127.0.0.1:{port}/ws",
                        "BOT_POOL_SIZE": str(args.go_bots),
                        "BOT_NAME_PREFIX": theirs,
                        "BOT_EXPLORE_EPSILON": "0",
                    },
                    log=workdir / f"gobot-{tag}.log",
                )
            )
        else:
            name = f"vsbot-crossplay-java-{os.getpid()}-{tag}"
            # A survivor from an interrupted run holds the name and would make
            # `docker run` fail; remove it before claiming the name.
            remove_container(name)
            containers.append(name)
            processes.append(
                spawn_java(
                    image=args.java_image,
                    name=name,
                    port=port,
                    prefix=theirs,
                    search=args.java_search,
                    mcts_value=args.java_mcts_value,
                    move_millis=args.move_millis,
                    challenger=not we_challenge,
                    challenge_secs=args.challenge_secs,
                    cwd=workdir,
                    log=workdir / f"java-{tag}.log",
                )
            )
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
                        "MCTS_ARTIFACT": str(args.mcts_artifact),
                        "MOVE_MILLIS": str(args.move_millis),
                        "CHALLENGER": "true" if we_challenge else "false",
                        "CHALLENGER_INTERVAL_SECS": str(args.challenge_secs),
                    },
                    log=workdir / f"vsbot-{tag}-{instance}.log",
                )
            )

        last = -1
        while True:
            result = tally(db, baseline, ours, theirs)
            if result["total"] != last:
                print(
                    f"  games {result['total']}/{target} | "
                    f"vsbot {result['wins']}-{result['losses']}-{result['draws']} {theirs}",
                    file=sys.stderr,
                )
                last = result["total"]
            if result["total"] >= target:
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
            dead = [p for p in processes if p.poll() is not None] + [
                p for p in server_processes if p.poll() is not None
            ]
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
        for name in containers:
            remove_container(name)
        # A killed bot's in-flight game lands as a `disconnect`, which `tally`
        # discards -- but only if we re-read after the server has written it.
        time.sleep(2)
        result = tally(db, baseline, ours, theirs)
    return result


# ------------------------------------------------------------------ self-test


def self_test() -> int:
    """Exercises the counting logic against a synthetic games.db.

    CI is Rust-only and the real harness needs a Go toolchain, a server
    checkout and (for the Java arm) docker, so none of it can run there. The
    parts that decide what a number *means* — seat folding, the disconnect
    discard, and the distinct-game fingerprint — are pure functions over rows,
    so they can be checked hermetically in a second. `tests/crossplay.rs` runs
    this, which is why the tally is covered even though the harness is not.
    """
    failures: list[str] = []

    def check(name: str, got, want) -> None:
        if got != want:
            failures.append(f"{name}: got {got!r}, want {want!r}")

    def pgn(moves: list[tuple[int, int]], duration: int) -> str:
        return json.dumps(
            [
                {
                    "turn": 1,
                    "player": 1,
                    "moves": [
                        {"type": "place", "row": r, "col": c, "duration_cs": duration}
                        for r, c in moves
                    ],
                }
            ]
        )

    # A replay differs only in how long each move took, and must not be counted
    # as a different game -- the bug that made 50 replays look like 50 samples.
    check(
        "fingerprint ignores duration",
        game_fingerprint(pgn([(0, 1), (0, 2)], 0)),
        game_fingerprint(pgn([(0, 1), (0, 2)], 99)),
    )
    if game_fingerprint(pgn([(0, 1)], 0)) == game_fingerprint(pgn([(0, 2)], 0)):
        failures.append("fingerprint: different moves collided")
    check("fingerprint tolerates junk", game_fingerprint("not json"), "")

    with tempfile.TemporaryDirectory(prefix="crossplay-selftest-") as directory:
        db = Path(directory) / "games.db"
        connection = sqlite3.connect(db)
        connection.execute(
            "CREATE TABLE games (player1_name TEXT, player2_name TEXT, "
            "result INTEGER, termination TEXT, pgn_content TEXT)"
        )
        replay = pgn([(0, 1)], 0)
        other = pgn([(5, 5)], 0)
        connection.executemany(
            "INSERT INTO games VALUES (?, ?, ?, ?, ?)",
            [
                # We are P1 and win.
                ("RustBot Bot 1", "GoBot Bot 2", 1, "no_moves", replay),
                # Same game replayed: still a win, but not a second sample.
                ("RustBot Bot 1", "GoBot Bot 2", 1, "no_moves", replay),
                # We are P2 and win -- the seat order the Python original dropped.
                ("GoBot Bot 2", "RustBot Bot 1", 2, "no_moves", other),
                # We are P2 and lose.
                ("GoBot Bot 2", "RustBot Bot 1", 1, "no_moves", other),
                # A draw.
                ("RustBot Bot 1", "GoBot Bot 2", 0, "no_moves", other),
                # A forfeit: a real loss, but flagged.
                ("RustBot Bot 1", "GoBot Bot 2", 2, "timeout", other),
                # A harness shutdown artifact: discarded, not a result.
                ("RustBot Bot 1", "GoBot Bot 2", 2, "disconnect", other),
                # Someone else's game entirely.
                ("GoBot Bot 2", "GoBot Bot 3", 1, "no_moves", other),
            ],
        )
        connection.commit()
        connection.close()

        counted = tally(db, 0, "RustBot", "GoBot")
        check("total", counted["total"], 6)
        check("wins", counted["wins"], 3)
        check("losses", counted["losses"], 2)
        check("draws", counted["draws"], 1)
        check("as_p1", counted["as_p1"], 4)
        check("as_p2", counted["as_p2"], 2)
        check("wins_p1", counted["wins_p1"], 2)
        check("wins_p2", counted["wins_p2"], 1)
        check("discarded", counted["discarded"], 1)
        check("red_flags", counted["red_flags"], 1)
        # Two `replay` rows plus four `other` rows counted = 2 distinct.
        check("distinct", counted["distinct"], 2)

        # The rowid baseline must exclude everything already in the table.
        check("baseline excludes old rows", tally(db, 8, "RustBot", "GoBot")["total"], 0)

    check("wilson95 of nothing", wilson95(0, 0), (0.0, 0.0))
    low, high = wilson95(5, 10)
    if not (low < 50.0 < high):
        failures.append(f"wilson95(5,10) should straddle 50%, got [{low}, {high}]")

    for failure in failures:
        print(f"self-test FAIL {failure}", file=sys.stderr)
    if failures:
        return 1
    print("crossplay self-test: ok", file=sys.stderr)
    return 0


# ----------------------------------------------------------------------- main


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.self_test:
        return self_test()
    backend = Path(args.backend).expanduser().resolve()
    vsbot = Path(args.vsbot).resolve()
    if not vsbot.exists():
        raise Failure(
            f"no vsbot binary at {vsbot} — run `cargo build --release -p vsbot`"
        )
    if shutil.which("go") is None:
        raise Failure("no Go toolchain on PATH; the server and bot-hoster need it")

    # Resolved here, against the invocation cwd, because the bots are started
    # with cwd=workdir. A relative artifact path silently missing is exactly the
    # failure that makes a run measure the wrong engine.
    args.mcts_artifact = Path(args.mcts_artifact).resolve()
    if args.search.upper() == "MCTS" and not args.mcts_artifact.exists():
        raise Failure(
            f"no MCTS artifact at {args.mcts_artifact} — pass --mcts-artifact"
        )

    temporary = None
    if args.workdir:
        workdir = Path(args.workdir).resolve()
        workdir.mkdir(parents=True, exist_ok=True)
    else:
        temporary = tempfile.TemporaryDirectory(prefix="vsbot-crossplay-")
        workdir = Path(temporary.name)

    # Distinct, non-overlapping prefixes. The server renders a bot's lobby name
    # as "<prefix> Bot NNNN", so these are what the LIKE filters match on.
    ours = "RustBot"
    theirs = "GoBot" if args.opponent == "go" else "JavaBot"

    # Refuse the impossible combination up front rather than running it and
    # reporting another single-chair number as if it were balanced. The Go
    # bot-hoster's challenger targets `Manager.IsAcceptor(userID)`, which is
    # false for every id outside its own pool, so it can never challenge vsbot.
    if args.opponent == "go" and args.direction != "ours":
        raise Failure(
            f"--direction {args.direction} needs an opponent that can challenge "
            "vsbot, and the Go bot-hoster cannot: its challenger mode only "
            "targets its own pool's acceptors (Manager.IsAcceptor). Use "
            "--opponent java for a colour-balanced run, or --direction ours and "
            "read the result as one-chair"
        )

    print(f"crossplay: workdir {workdir}", file=sys.stderr)
    server_binary, configurable = build_server(backend, workdir)
    hoster_binary = build_bot_hoster(backend, workdir) if args.opponent == "go" else None
    if args.opponent == "java":
        docker_available()

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

    # `alternate` splits the target evenly: the first half with vsbot
    # challenging (vsbot at P1), the second with the rival challenging (vsbot at
    # P2). The phase targets are cumulative because both phases write into the
    # same games.db, and the tally is always taken against the same baseline.
    if args.direction == "alternate":
        first_half = args.games // 2
        phases = [("ours", first_half), ("theirs", args.games)]
    else:
        phases = [(args.direction, args.games)]

    server_processes: list = []
    result = tally(db, baseline, ours, theirs)
    try:
        server = spawn(
            [str(server_binary)],
            cwd=workdir,
            env={**os.environ, "VSBOT_ITEST_PORT": str(port)},
            log=workdir / "server.log",
        )
        server_processes.append(server)
        wait_for_port(port, started_at + 30)

        deadline = started_at + args.timeout
        for phase_index, (direction, cumulative_target) in enumerate(phases):
            if result["total"] >= cumulative_target:
                continue
            we_challenge = direction == "ours"
            print(
                f"crossplay: phase {phase_index + 1}/{len(phases)} — "
                f"{'vsbot' if we_challenge else theirs} challenges, so vsbot sits at "
                f"P{'1' if we_challenge else '2'}; collecting to {cumulative_target} game(s)",
                file=sys.stderr,
            )
            result = run_phase(
                args=args,
                workdir=workdir,
                port=port,
                db=db,
                baseline=baseline,
                ours=ours,
                theirs=theirs,
                vsbot=vsbot,
                hoster_binary=hoster_binary,
                we_challenge=we_challenge,
                target=cumulative_target,
                deadline=deadline,
                phase_index=phase_index,
                server_processes=server_processes,
            )
            if result["total"] < cumulative_target:
                # Out of time or a dead process; the next phase cannot fix it
                # and running it would only add games from one more chair.
                break
    finally:
        for process in reversed(server_processes):
            kill_group(process)

    low, high = wilson95(result["wins"], result["total"])

    def seat_rate(wins: int, games: int) -> float:
        return (100.0 * wins / games) if games else 0.0

    report = {
        "opponent": args.opponent,
        "opponent_name": theirs,
        "direction": args.direction,
        "vsbot_search": args.search,
        "move_millis": args.move_millis,
        "wins": result["wins"],
        "losses": result["losses"],
        "draws": result["draws"],
        "games": result["total"],
        "as_p1": result["as_p1"],
        "as_p2": result["as_p2"],
        "wins_as_p1": result["wins_p1"],
        "wins_as_p2": result["wins_p2"],
        # The first-mover effect, quantified rather than assumed away. With
        # --direction alternate these two are the colour-split of a balanced
        # run; with `ours` one of them is vacuous and the warning below fires.
        "win_rate_as_p1": seat_rate(result["wins_p1"], result["as_p1"]),
        "win_rate_as_p2": seat_rate(result["wins_p2"], result["as_p2"]),
        # (W + D/2) / N -- the pooled score superiority.md's gates are written
        # against. `win_rate` below does not count draws as half, so the two
        # differ in any run with draws and the gate must read this one.
        "pooled_score": (
            (result["wins"] + 0.5 * result["draws"]) / result["total"]
            if result["total"]
            else 0.0
        ),
        "discarded_disconnects": result["discarded"],
        "red_flag_terminations": result["red_flags"],
        "distinct_games": result["distinct"],
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
            f"\n=== vsbot({args.search}) vs {theirs}: "
            f"W-L-D {report['wins']}-{report['losses']}-{report['draws']} "
            f"over {report['games']} games ({report['verdict']}) ==="
        )
        print(
            f"    win rate {report['win_rate']:.1f}% (draws not half-wins)  "
            f"wilson95 [{low:.1f}%, {high:.1f}%]"
        )
        print(f"    pooled score {report['pooled_score']:.4f} (W+0.5D)/N")
        print(
            f"    seats: {report['as_p1']} as P1 ({report['win_rate_as_p1']:.1f}% won), "
            f"{report['as_p2']} as P2 ({report['win_rate_as_p2']:.1f}% won)"
        )
        print(
            f"    distinct games: {report['distinct_games']}/{report['games']} "
            "(replays share an opening; see bd vsbot-t3q.1)"
        )
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
    warn_about_low_diversity(report)
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
