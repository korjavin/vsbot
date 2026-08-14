#!/usr/bin/env python3
"""Dump prod `games.db` into the normalised JSONL the `probe mine-db` step reads.

Why a Python script and not a subcommand: `virus-arena`'s module docs already
record the reason the cross-play harness is a script — reading an SQLite file
from Rust needs a C dependency, and this crate's whole job is to be a
trustworthy measuring instrument, so it does not grow one for a step that runs
once per fixture refresh.

Provenance of the input:

    curl -o games.db https://vs.wandergeek.org/data/games.db

The published file is a periodic snapshot; its HTTP `Last-Modified` is the
authoritative "as of" timestamp for anything mined out of it and is recorded in
the dump's `meta` line.

Schema of `games.pgn_content` (Go's recorder, marshalled with `omitempty`):

    [{"turn":1,"player":1,"moves":[{"type":"place","row":1,"col":1,...}, ...]}, ...]

  * `type` is `place` | `attack` | `move` (all a `Move` action) or
    `neutral` | `neutrals` (a `PlaceNeutrals` action carrying `cells`);
  * `omitempty` DROPS zero-valued ints, so an absent `row`/`col` means 0 —
    reading them as missing discarded whole games in the Java miner
    (`GamesDbReplay.pos`), so they are defaulted here;
  * Go's default struct marshalling can capitalise the keys, so both cases are
    accepted.

This script only normalises and filters; it applies no game rules. Replay
happens in Rust, through `virus-core`, so the probe fixture is built by the
same rules engine the probe then runs against.

Usage:

    python3 fixtures/probes/tools/dump_games.py \
        --db games.db --out games-dump.jsonl [--rows 12] [--cols 12] \
        [--only-neutral]
"""

import argparse
import json
import sqlite3
import sys
from email.utils import parsedate_to_datetime


MOVE_TYPES = {"place", "attack", "move"}
NEUTRAL_TYPES = {"neutral", "neutrals"}


def field(node, name):
    """Stored field, tolerating the capitalised keys Go's default marshalling emits."""
    if name in node:
        return node[name]
    return node.get(name[0].upper() + name[1:])


def coord(node):
    """`(row, col)`, restoring the zeroes `omitempty` dropped."""
    row = field(node, "row")
    col = field(node, "col")
    return [int(row or 0), int(col or 0)]


def normalise_move(move):
    """One stored move node as `{"kind": ..., ...}`, or `None` if unparseable."""
    kind = field(move, "type")
    if kind is None:
        return None
    kind = str(kind).lower()
    if kind in MOVE_TYPES:
        return {"kind": "move", "cells": [coord(move)]}
    if kind in NEUTRAL_TYPES:
        cells = field(move, "cells")
        if not isinstance(cells, list) or len(cells) != 2:
            return None
        return {"kind": "neutrals", "cells": [coord(cells[0]), coord(cells[1])]}
    return None


def normalise_turn(turn):
    """One stored turn as `{"turn": n, "player": p, "moves": [...]}`, or `None`."""
    player = field(turn, "player")
    if player is None:
        return None
    moves = field(turn, "moves") or []
    if not isinstance(moves, list):
        return None
    out = []
    for move in moves:
        normalised = normalise_move(move)
        if normalised is None:
            return None
        out.append(normalised)
    return {
        "turn": int(field(turn, "turn") or 0),
        "player": int(player),
        "moves": out,
    }


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, help="path to games.db")
    parser.add_argument("--out", required=True, help="path to write JSONL to")
    parser.add_argument("--rows", type=int, default=12)
    parser.add_argument("--cols", type=int, default=12)
    parser.add_argument(
        "--last-modified",
        default="",
        help="the HTTP Last-Modified of the downloaded games.db, recorded in the meta line",
    )
    parser.add_argument(
        "--only-neutral",
        action="store_true",
        help="keep only games whose PGN contains a neutral placement",
    )
    args = parser.parse_args(argv)

    connection = sqlite3.connect(args.db)
    rows = connection.execute(
        "SELECT id, started_at, rows, cols, player1_name, player2_name,"
        " player3_name, player4_name, result, termination, pgn_content FROM games"
        " ORDER BY started_at"
    )

    kept = skipped = 0
    as_of = args.last_modified
    if as_of:
        try:
            as_of = parsedate_to_datetime(as_of).isoformat()
        except (TypeError, ValueError):
            pass

    with open(args.out, "w", encoding="utf-8") as out:
        out.write(
            json.dumps(
                {
                    "meta": {
                        "source": "https://vs.wandergeek.org/data/games.db",
                        "as_of": as_of,
                        "rows": args.rows,
                        "cols": args.cols,
                    }
                }
            )
            + "\n"
        )
        for (
            game_id,
            started_at,
            game_rows,
            game_cols,
            p1,
            p2,
            p3,
            p4,
            result,
            termination,
            pgn,
        ) in rows:
            if game_rows != args.rows or game_cols != args.cols:
                skipped += 1
                continue
            # The net and the absolute-frame searcher are two-player only.
            if p3 or p4:
                skipped += 1
                continue
            if not pgn:
                skipped += 1
                continue
            if args.only_neutral and "eutral" not in pgn:
                skipped += 1
                continue
            try:
                turns = json.loads(pgn)
            except json.JSONDecodeError:
                skipped += 1
                continue
            if not isinstance(turns, list):
                skipped += 1
                continue
            normalised = [normalise_turn(turn) for turn in turns]
            if any(turn is None for turn in normalised):
                skipped += 1
                continue
            out.write(
                json.dumps(
                    {
                        "id": game_id,
                        "startedAt": (started_at or "").split(" m=")[0],
                        "rows": game_rows,
                        "cols": game_cols,
                        "players": [p1 or "", p2 or ""],
                        "result": result if result is not None else 0,
                        "termination": termination or "",
                        "turns": normalised,
                    }
                )
                + "\n"
            )
            kept += 1

    print(f"wrote {kept} games to {args.out} ({skipped} skipped)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
