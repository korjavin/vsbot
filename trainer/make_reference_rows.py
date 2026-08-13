#!/usr/bin/env python3
"""Deterministic stand-in SelfPlayMcts rows, for exercising the pipeline before Rust emits any.

    python3 trainer/make_reference_rows.py OUT.jsonl [--games 12] [--seed 11]

**This is scaffolding, not training data.** The rows are contract-valid but the
positions are synthetic: no game rules are simulated, so a net trained on them
learns nothing useful. Its only job is to make every stage downstream of the
emitter — train_selfplay.py, the artifact export, the Rust loader — runnable and
provable *today*, so that when the Rust emitter lands the only new variable is
the emitter itself.

The moment `virus-mcts`' self-play binary exists, delete the call to this script
from trainer/roundtrip.sh and point the same commands at real
`gen<N>/selfplay.jsonl`. Nothing else in the pipeline changes — that is the
property this script is here to establish.

Why not `fixtures/mcts/mcts_selfplay_tiny.json`? Despite the name, that file is
a trained *net* artifact (`meta`/`conv`/`move_head`/`pair_head`/`value_head`,
channels 8 x layers 2) — the Java loader's parity fixture, byte-identical to
`nnue-trainer/src/test/resources/mcts/mcts_selfplay_tiny.json`. It is the right
reference for the export side of the round trip (see trainer/validate_artifact.py)
and no help at all as trainer input. No self-play *rows* are vendored in this
repo, by design: they are generated output, not fixtures.
"""

import argparse
import json
import random
import sys

from rows_schema import BOARD, CELLS, move_id, pair_id

# A game is a handful of multi-choice positions; SelfPlayMcts skips forced ones.
PLIES_PER_GAME = 6


def board(rng, mover, filled):
    """A plausible mover-relative symbol grid: two bases, some normals, a few neutrals.

    Mover-relative means own = 2/4/6 and enemy = 3/5/7 *from this row's mover*,
    so the same physical board yields swapped symbols on consecutive plies —
    exactly the encoding train_policy.planes() one-hots.
    """
    sym = [0] * CELLS
    own_base, enemy_base = (0, CELLS - 1) if mover == 1 else (CELLS - 1, 0)
    sym[own_base] = 2
    sym[enemy_base] = 3
    cells = [c for c in range(CELLS) if c not in (own_base, enemy_base)]
    rng.shuffle(cells)
    take = cells[: 2 * filled + 3]
    for index, cell in enumerate(take):
        if index % 5 == 4:
            sym[cell] = 1  # neutral
        elif index % 2 == 0:
            sym[cell] = 4 if rng.random() < 0.8 else 6  # own normal / fortified
        else:
            sym[cell] = 5 if rng.random() < 0.8 else 7  # enemy normal / fortified
    return sym


def actions(rng, sym, neutrals_available):
    """A legal-looking action set: moves onto empty cells, plus optional neutral pairs.

    `pi` is the whole legal set (the mask the trainer softmaxes over) and `pv`
    is raw root visit counts — not probabilities. Both properties are asserted
    by trainer/validate_rows.py.
    """
    empties = [c for c in range(CELLS) if sym[c] == 0]
    rng.shuffle(empties)
    ids = [move_id(c // BOARD, c % BOARD) for c in empties[: rng.randint(4, 12)]]
    if neutrals_available:
        for _ in range(rng.randint(1, 3)):
            a, b = rng.sample(empties[:20], 2)
            flat = pair_id(a, b)
            if flat not in ids:
                ids.append(flat)
    # Visit counts: one clear favourite plus a long tail, the shape MCTS produces.
    visits = [rng.randint(0, 6) for _ in ids]
    visits[rng.randrange(len(visits))] += rng.randint(20, 90)
    return ids, visits


def game(rng, game_id):
    """One game's rows. z is drawn once and stamped on every row in the ABSOLUTE frame."""
    z = rng.choice([1, -1, 1, -1, 0])
    rows = []
    for ply in range(PLIES_PER_GAME):
        mover = 1 + (ply % 2)
        # Neutral placement is a once-per-game resource; spend it mid-game.
        nuo = 1 if ply >= 4 else 0
        nux = 1 if ply >= 3 else 0
        sym = board(rng, mover, ply)
        ids, visits = actions(rng, sym, neutrals_available=(nuo == 0))
        rows.append(
            {
                "g": game_id,
                "sym": sym,
                "ml": 1 + (ply % 3),
                "nuo": nuo,
                "nux": nux,
                "mover": mover,
                "pi": ids,
                "pv": visits,
                "z": z,
            }
        )
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", help="output JSONL path")
    ap.add_argument("--games", type=int, default=12)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    written = 0
    with open(args.out, "w") as handle:
        for index in range(args.games):
            for row in game(rng, f"synth{args.seed}-{index}"):
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
                written += 1
    print(f"{written} synthetic rows across {args.games} games -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
