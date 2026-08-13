#!/usr/bin/env python3
"""Run several independent ``crossplay`` shards at once and pool the result.

Why shards rather than one big run with more bots
-------------------------------------------------

``crossplay.py --vsbot-instances N`` buys concurrency badly.  ``vsbot``'s
challenger picks *any* idle peer (``challenge_tick`` filters on
``peer.is_idle()`` and nothing else), so with several vsbots in one lobby a
large share of the games are vsbot-vs-vsbot.  Those are dropped by the name
filter rather than miscounted, but they are wasted compute — and at 1 s/move
the compute *is* the run.

A shard is instead a whole private stack: its own server on its own port, one
``vsbot``, one opponent.  Nothing else is idle in that lobby, so every game
started is a game that counts, and N shards use N cores with no waste.

Pooling is not just addition
----------------------------

Games are summed, but **distinct games are unioned**, not summed.  Two shards
running an identical configuration would otherwise be credited with twice the
diversity the run actually has -- and before bd ``vsbot-t3q.2`` they *did* run
an identical configuration, because nothing randomised a cross-play opening.
The pooled report therefore counts unique move-sequence digests across every
shard, and each shard is now handed its own ``--explore-seed`` derived from the
pool's, so shards no longer replay each other in the first place.

Usage::

    python3 crates/virus-arena/crossplay/crossplay_pool.py \\
        --shards 4 --games 400 --opponent java --direction alternate \\
        --search MCTS --workdir /tmp/s1

Run it from the repository root.  Any flag it does not recognise is passed
through to ``crossplay.py`` unchanged, so the shard configuration is exactly the
single-run configuration.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import crossplay  # noqa: E402


def parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="run N crossplay shards in parallel and pool the tally",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--shards", type=int, default=4, help="parallel shards")
    parser.add_argument("--games", type=int, default=400, help="TOTAL games to collect")
    parser.add_argument(
        "--workdir", required=True, help="root for the shard workdirs; kept, not cleaned"
    )
    parser.add_argument(
        "--opponent", choices=("go", "java"), default="java", help="rival bot"
    )
    parser.add_argument(
        "--poll-secs", type=int, default=30, help="how often to print pooled progress"
    )
    # Consumed here rather than passed through: every shard must get its own
    # exploration stream, so the pool derives one per shard from this base. The
    # epsilon is consumed only so the pooled report can record what produced its
    # distinct-game count; it is handed to the shards unchanged.
    parser.add_argument(
        "--explore-seed",
        type=int,
        default=crossplay.DEFAULT_EXPLORE_SEED,
        help="base exploration seed; shard k runs on derive_seed(base, k)",
    )
    parser.add_argument(
        "--explore-eps",
        type=float,
        default=crossplay.DEFAULT_EXPLORE_EPS,
        help="vsbot opening-exploration probability, passed to every shard",
    )
    return parser.parse_known_args(argv)


def main(argv: list[str]) -> int:
    args, passthrough = parse_args(argv)
    if args.shards < 1:
        raise crossplay.Failure("--shards must be at least 1")

    root = Path(args.workdir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve().parent / "crossplay.py"

    # The remainder goes to the first shards, so the pooled total is exactly
    # --games rather than a multiple of --shards.
    base, extra = divmod(args.games, args.shards)
    per_shard = [base + (1 if i < extra else 0) for i in range(args.shards)]

    ours = "RustBot"
    theirs = "GoBot" if args.opponent == "go" else "JavaBot"

    processes = []
    # (workdir, baseline). The baseline is each shard's max rowid *before* it
    # starts, exactly as `crossplay.py` takes its own: `--workdir` is documented
    # as kept rather than cleaned, so a re-run against the same root would
    # otherwise pool the previous run's games into this one's headline and could
    # even report success without playing a game.
    shards: list[tuple[Path, int]] = []
    for index, games in enumerate(per_shard):
        if games == 0:
            continue
        shard = root / f"shard-{index}"
        shard.mkdir(parents=True, exist_ok=True)
        shards.append((shard, crossplay.max_rowid(shard / "data" / "games.db")))
        command = [
            sys.executable, str(script),
            "--games", str(games),
            "--opponent", args.opponent,
            "--workdir", str(shard),
            # Disjoint per shard. Shards are otherwise identical, and identical
            # shards replay each other's games -- which the union in `pool`
            # would report honestly as diversity this run never had.
            "--explore-seed", str(crossplay.derive_seed(args.explore_seed, index)),
            "--explore-eps", str(args.explore_eps),
            *passthrough,
        ]
        log = (shard / "shard.log").open("w")
        print(f"pool: shard {index} -> {games} game(s) in {shard}", file=sys.stderr)
        processes.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT))
        # `crossplay.free_port` binds :0, reads the port and closes again, so two
        # shards starting in the same instant can be handed the same number and
        # the second server dies on bind. Staggering costs seconds and removes
        # the race.
        time.sleep(3)

    try:
        while True:
            pooled = pool(shards, ours, theirs)
            alive = [p for p in processes if p.poll() is None]
            print(
                f"  pooled {pooled['total']}/{args.games} | "
                f"vsbot {pooled['wins']}-{pooled['losses']}-{pooled['draws']} {theirs} | "
                f"{pooled['distinct']} distinct | {len(alive)} shard(s) running",
                file=sys.stderr,
            )
            if not alive:
                break
            time.sleep(args.poll_secs)
    finally:
        shutdown(processes)

    pooled = pool(shards, ours, theirs)
    report = render(pooled, args, theirs)
    print(json.dumps(report, indent=2))
    return 0 if pooled["total"] >= args.games else 1


def shutdown(processes: list) -> None:
    """Stops the shards and, through them, everything they started.

    A shard's server, bots and Java container are each in their own session (see
    `crossplay.spawn`), so signalling this process group would never reach them.
    What does reach them is the shard's own teardown, which is why `crossplay.py`
    turns SIGTERM into an orderly exit: terminate the shard, give it time to run
    its `finally`, and only then escalate. Skipping the wait is how a run leaves
    a Java container playing into the next run's database.
    """
    for process in processes:
        if process.poll() is None:
            process.terminate()
    deadline = time.time() + 30
    for process in processes:
        remaining = max(0.0, deadline - time.time())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            print(
                f"pool: shard pid {process.pid} ignored SIGTERM; killing it. Check "
                "for surviving `docker` containers and servers.",
                file=sys.stderr,
            )
            process.kill()


def pool(shards: list[tuple[Path, int]], ours: str, theirs: str) -> dict:
    """Sums every shard's tally, unioning the game fingerprints."""
    total = {
        key: 0
        for key in (
            "wins", "losses", "draws", "as_p1", "as_p2",
            "wins_p1", "wins_p2", "draws_p1", "draws_p2",
            "total", "discarded", "red_flags",
        )
    }
    fingerprints: set[str] = set()
    for shard, baseline in shards:
        one = crossplay.tally(shard / "data" / "games.db", baseline, ours, theirs)
        for key in total:
            total[key] += one[key]
        fingerprints |= one["fingerprints"]
    total["distinct"] = len(fingerprints)
    return total


def render(pooled: dict, args, theirs: str) -> dict:
    games = pooled["total"]
    low, high = crossplay.wilson95(pooled["wins"], games)

    def rate(wins: int, n: int) -> float:
        return (100.0 * wins / n) if n else 0.0

    report = {
        "shards": args.shards,
        "opponent": theirs,
        # `distinct_games` below only means something next to these.
        "explore_eps": args.explore_eps,
        "explore_seed": args.explore_seed,
        "wins": pooled["wins"],
        "losses": pooled["losses"],
        "draws": pooled["draws"],
        "games": games,
        "as_p1": pooled["as_p1"],
        "as_p2": pooled["as_p2"],
        "win_rate_as_p1": rate(pooled["wins_p1"], pooled["as_p1"]),
        "win_rate_as_p2": rate(pooled["wins_p2"], pooled["as_p2"]),
        # The gate quantity: superiority.md S1 wants >= 0.55 here.
        "pooled_score": (
            (pooled["wins"] + 0.5 * pooled["draws"]) / games if games else 0.0
        ),
        "win_rate": rate(pooled["wins"], games),
        "wilson95_low": low,
        "wilson95_high": high,
        "distinct_games": pooled["distinct"],
        "discarded_disconnects": pooled["discarded"],
        "red_flag_terminations": pooled["red_flags"],
        "verdict": (
            "INFORMATIONAL ONLY"
            if games < 100
            else ("indicative" if games < 400 else "gate-eligible")
        ),
    }
    crossplay.warn_about_seat_imbalance(report)
    crossplay.warn_about_low_diversity(report)
    return report


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except crossplay.Failure as error:
        print(f"pool: {error}", file=sys.stderr)
        sys.exit(2)
    except KeyboardInterrupt:
        sys.exit(130)
