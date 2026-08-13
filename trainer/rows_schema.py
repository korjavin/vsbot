#!/usr/bin/env python3
"""The SelfPlayMcts JSONL row contract, in one place.

Authority: ``~/Project/nnue-trainer/src/main/java/com/engine/nnue_trainer/mcts/
SelfPlayMcts.java`` (``row()`` + ``flatIndex()``). The consumer is
``python/mcts/train_selfplay.py`` in that same checkout, UNCHANGED — the whole
point of S4 is that Rust replaces the Java *emitter*, not the trainer.

Row (one JSON object per line, no trailing commas, no pretty-printing)::

    {"g":  "sp11000-3",   game id; every row of one game shares it AND shares z
     "sym": [144 ints],   PatternContract symbols, row-major 12x12, MOVER-RELATIVE
     "ml":  1|2|3,        moves left in the current turn
     "nuo": 0|1,          mover has spent its neutral placement
     "nux": 0|1,          opponent has spent its neutral placement
     "mover": 1|2,        whose turn it is; the trainer's z flip point
     "pi":  [flat ids],   EVERY legal action at the root (this is the mask)
     "pv":  [visits],     root visit counts, same order/length as pi (the target)
     "z":   -1|0|1}       game outcome, ABSOLUTE frame (+1 = player 1 won)

Two things that are easy to get wrong and are therefore checked hard:

* **z is absolute, not mover-relative.** ``train_selfplay.mover_z`` flips it
  (``z if mover == 1 else -z``) because the features are mover-relative. An
  emitter that pre-flips z trains the value head backwards for player 2 and
  nothing downstream will complain.
* **Flat action ids.** ``cell = row*12 + col`` for a move; ``144 + min*144 +
  max`` for a neutral pair. Emitting an unordered pair (i > j) silently aliases
  two different pairs onto ids the trainer treats as distinct.

Rows are only emitted for **multi-choice** positions (``root.actions.length >
1``): a forced position carries no policy signal.
"""

BOARD = 12
CELLS = BOARD * BOARD  # 144
PAIR_OFFSET = CELLS
FLAT = CELLS + CELLS * CELLS  # 20880, matching train_policy.FLAT

#: Symbols PatternContract.getSymbol emits for an on-board cell, mover-relative.
#: 0 empty, 1 neutral, 2/3 own/enemy base, 4/5 own/enemy normal, 6/7 own/enemy
#: fortified. (The contract's 8th value is out-of-bounds and unreachable here.)
SYMBOLS = range(8)

FIELDS = ("g", "sym", "ml", "nuo", "nux", "mover", "pi", "pv", "z")


def move_id(row, col):
    """Flat id of a MoveAction targeting (row, col)."""
    return row * BOARD + col


def pair_id(cell_a, cell_b):
    """Flat id of a PlaceNeutralsAction on two cells; order-independent by construction."""
    if cell_a == cell_b:
        raise ValueError(f"a neutral pair needs two distinct cells, got {cell_a} twice")
    return PAIR_OFFSET + min(cell_a, cell_b) * CELLS + max(cell_a, cell_b)


def decode(flat):
    """Inverse of the above: ``("move", cell)`` or ``("pair", i, j)`` with i < j."""
    if not 0 <= flat < FLAT:
        raise ValueError(f"flat id {flat} outside [0, {FLAT})")
    if flat < CELLS:
        return ("move", flat)
    rest = flat - PAIR_OFFSET
    return ("pair", rest // CELLS, rest % CELLS)
