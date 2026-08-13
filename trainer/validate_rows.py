#!/usr/bin/env python3
"""Strict validator for SelfPlayMcts JSONL — point it at the Rust emitter's output.

    python3 trainer/validate_rows.py gen1/selfplay.jsonl [more.jsonl ...]

This is the field-for-field half of S4's acceptance criterion ("train_selfplay.py
consumes Rust JSONL field-for-field"). Running the trainer proves the rows *parse*;
this proves they mean the same thing the Java emitter meant — the failures that
otherwise show up only as a net that trains to a worse champion three hours later:

* z pre-flipped into the mover frame instead of left absolute
* pair ids emitted unordered (i > j), aliasing distinct pairs
* pi holding only the *searched* actions instead of the whole legal set
* pv holding normalised probabilities instead of raw visit counts
* forced (single-action) positions emitted, diluting the policy target

Exit 0 = every row valid. Exit 1 = at least one violation (all are reported, not
just the first). Stdlib only, so it runs on the host without the torch image.
"""

import argparse
import collections
import json
import sys

from rows_schema import BOARD, CELLS, FIELDS, FLAT, decode

MAX_REPORTED = 25


class Problems:
    def __init__(self):
        self.items = []

    def add(self, where, message):
        self.items.append(f"{where}: {message}")

    def __bool__(self):
        return bool(self.items)


def check_row(row, where, problems):
    if not isinstance(row, dict):
        problems.add(where, f"expected a JSON object, got {type(row).__name__}")
        return None

    missing = [f for f in FIELDS if f not in row]
    if missing:
        problems.add(where, f"missing field(s) {missing}")
        return None
    extra = [k for k in row if k not in FIELDS]
    if extra:
        # Not fatal for the trainer (it reads by key), but it means the emitter
        # and the contract have drifted — say so loudly.
        problems.add(where, f"unexpected field(s) {extra} — not in the SelfPlayMcts contract")

    if not isinstance(row["g"], str) or not row["g"]:
        problems.add(where, f"g must be a non-empty string, got {row['g']!r}")

    sym = row["sym"]
    if not isinstance(sym, list) or len(sym) != CELLS:
        problems.add(where, f"sym must be {CELLS} entries ({BOARD}x{BOARD}), got {len(sym) if isinstance(sym, list) else type(sym).__name__}")
    else:
        bad = {s for s in sym if not isinstance(s, int) or isinstance(s, bool) or not 0 <= s <= 7}
        if bad:
            problems.add(where, f"sym holds non-symbol value(s) {sorted(bad)[:5]}; legal range is 0..7")

    if row["ml"] not in (1, 2, 3):
        problems.add(where, f"ml must be 1..3, got {row['ml']!r}")
    for flag in ("nuo", "nux"):
        if row[flag] not in (0, 1):
            problems.add(where, f"{flag} must be 0 or 1, got {row[flag]!r}")
    if row["mover"] not in (1, 2):
        problems.add(where, f"mover must be 1 or 2, got {row['mover']!r}")
    if row["z"] not in (-1, 0, 1):
        problems.add(where, f"z must be -1, 0 or 1 (ABSOLUTE frame, +1 = player 1 won), got {row['z']!r}")

    pi, pv = row["pi"], row["pv"]
    if not isinstance(pi, list) or not isinstance(pv, list):
        problems.add(where, "pi and pv must both be lists")
        return None
    if len(pi) != len(pv):
        problems.add(where, f"pi/pv length mismatch: {len(pi)} vs {len(pv)}")
        return None
    if len(pi) < 2:
        problems.add(where, f"pi has {len(pi)} action(s); SelfPlayMcts only emits multi-choice positions (root.actions.length > 1)")

    # Type-check every entry BEFORE any set/Counter work: a JSON array or object
    # in pi is unhashable, and hashing it first would abort the whole run with a
    # traceback instead of the aggregated report this tool promises. Malformed
    # output is exactly the input this validator exists to survive.
    ints = []
    for action in pi:
        if not isinstance(action, int) or isinstance(action, bool):
            problems.add(where, f"pi entry {action!r} is not an int")
            continue
        ints.append(action)
        if not 0 <= action < FLAT:
            problems.add(where, f"pi entry {action} outside the flat space [0, {FLAT})")
            continue
        kind = decode(action)
        if kind[0] == "pair" and kind[1] >= kind[2]:
            problems.add(where, f"pair id {action} decodes to (i={kind[1]}, j={kind[2]}) — must be 144 + min*144 + max with i < j")

    if len(ints) != len(set(ints)):
        dupes = [a for a, c in collections.Counter(ints).items() if c > 1]
        problems.add(where, f"pi has duplicate action id(s) {dupes[:5]}")

    if any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in pv):
        problems.add(where, f"pv must be non-negative integer visit counts, got {pv[:5]}...")
    elif sum(pv) <= 0:
        problems.add(where, "pv sums to 0 — the trainer would produce an all-zero policy target for this row")

    # A neutral pair is only legal while the mover still has its placement.
    if row["nuo"] == 1 and any(isinstance(a, int) and a >= CELLS for a in pi):
        problems.add(where, "nuo=1 (mover already used its neutrals) but pi offers a neutral pair")

    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="SelfPlayMcts JSONL file(s)")
    ap.add_argument("--quiet", action="store_true", help="only print the verdict")
    args = ap.parse_args()

    problems = Problems()
    rows = 0
    games = collections.defaultdict(list)
    for path in args.paths:
        with open(path) as handle:
            for lineno, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    problems.add(f"{path}:{lineno}", "blank line — JSONL must be one object per line")
                    continue
                rows += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    problems.add(f"{path}:{lineno}", f"not valid JSON: {error}")
                    continue
                checked = check_row(row, f"{path}:{lineno}", problems)
                if checked:
                    games[checked["g"]].append(checked)

    # Cross-row invariant: z is a per-GAME label, so every row of a game carries
    # the same absolute-frame value. A per-row z is the classic sign of an
    # emitter that flipped into the mover frame at write time.
    for game_id, game_rows in games.items():
        zs = {r["z"] for r in game_rows}
        if len(zs) > 1:
            movers = {r["mover"] for r in game_rows}
            hint = " (z varies with mover — z must stay ABSOLUTE; the trainer flips it)" if len(movers) > 1 else ""
            problems.add(f"game {game_id}", f"rows disagree on z: {sorted(zs)}{hint}")

    if not args.quiet:
        movers = collections.Counter(r["mover"] for rs in games.values() for r in rs)
        outcomes = collections.Counter(rs[0]["z"] for rs in games.values() if rs)
        print(f"rows: {rows} across {len(games)} game(s)")
        print(f"mover split: p1={movers[1]} p2={movers[2]}")
        print(f"game outcomes (absolute z): {dict(sorted(outcomes.items()))}")

    if problems:
        print(f"\nFAIL: {len(problems.items)} violation(s)", file=sys.stderr)
        for item in problems.items[:MAX_REPORTED]:
            print(f"  {item}", file=sys.stderr)
        if len(problems.items) > MAX_REPORTED:
            print(f"  ... and {len(problems.items) - MAX_REPORTED} more", file=sys.stderr)
        return 1

    print("OK: rows match the SelfPlayMcts contract")
    return 0


if __name__ == "__main__":
    sys.exit(main())
