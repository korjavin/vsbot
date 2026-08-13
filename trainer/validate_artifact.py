#!/usr/bin/env python3
"""Structural check of a trained artifact against the schema our Rust loader accepts.

    python3 trainer/validate_artifact.py candidate.json [--reference artifacts/mcts_champion.json]

The authority is `crates/virus-mcts/src/net.rs` (`PolicyValueNet::from_raw`),
which validates declared-vs-actual shape and finiteness of every weight. This
script is the *cheap, no-toolchain* mirror of those checks, plus a field-by-field
diff of the artifact's shape-signature against a known-good reference (the
promoted champion by default).

It exists for two reasons:

1. It runs in a second, with no cargo build, so trainer/roundtrip.sh can fail
   fast on an obviously wrong export before paying for a Rust compile.
2. It is the documented fallback if the Rust load check ever cannot run (no
   toolchain on the box, workspace mid-refactor): the round trip still gets a
   stated, reproducible verdict rather than an untested artifact.

Passing here does NOT replace the Rust load — `PolicyValueNet::load` is the real
gate, and roundtrip.sh runs it. Passing here says "nothing structural is wrong".
"""

import argparse
import json
import math
import sys

BOARD = 12
PLANES = 13
SUPPORTED_ARCH = ("conv-policy-v1", "conv-policy-value-v1")


def fail(problems, message):
    problems.append(message)


def dims(value):
    """Nested-list shape, e.g. [[.,.],[.,.]] -> (2, 2); scalars -> ()."""
    shape = []
    while isinstance(value, list):
        shape.append(len(value))
        if not value:
            break
        value = value[0]
    return tuple(shape)


def finite(value):
    """Every leaf float is finite (the NaN-prior failure net.rs refuses to start with)."""
    if isinstance(value, list):
        return all(finite(v) for v in value)
    return isinstance(value, (int, float)) and math.isfinite(value)


def signature(net):
    """The artifact's shape fingerprint — what a reference comparison is actually about."""
    sig = {
        "arch": net.get("meta", {}).get("arch"),
        "board": net.get("meta", {}).get("board"),
        "planes": net.get("meta", {}).get("planes"),
        "conv": [dims(layer.get("w")) for layer in net.get("conv", [])],
        "move_head.w": dims(net.get("move_head", {}).get("w")),
        "pair_head.w": dims(net.get("pair_head", {}).get("w")),
    }
    if "value_head" in net:
        head = net["value_head"]
        sig["value_head.fc1_w"] = dims(head.get("fc1_w"))
        sig["value_head.fc2_w"] = dims(head.get("fc2_w"))
    return sig


def check(net, problems):
    meta = net.get("meta")
    if not isinstance(meta, dict):
        fail(problems, "missing `meta` object")
        return
    arch = meta.get("arch")
    if arch not in SUPPORTED_ARCH:
        fail(problems, f"meta.arch {arch!r} is not one of {SUPPORTED_ARCH} — net.rs SUPPORTED_ARCH refuses it")
    if meta.get("board") != BOARD:
        fail(problems, f"meta.board must be {BOARD}, got {meta.get('board')!r}")
    if meta.get("planes") != PLANES:
        fail(problems, f"meta.planes must be {PLANES}, got {meta.get('planes')!r}")

    channels = meta.get("channels")
    layers = meta.get("layers")
    conv = net.get("conv")
    if not isinstance(conv, list) or not conv:
        fail(problems, "missing or empty `conv` stack")
        return
    if len(conv) != layers:
        fail(problems, f"meta.layers says {layers} but `conv` holds {len(conv)} layer(s)")

    for index, layer in enumerate(conv):
        # [out][in][3][3], in = PLANES for layer 0 and `channels` afterwards.
        expected_in = PLANES if index == 0 else channels
        got = dims(layer.get("w"))
        want = (channels, expected_in, 3, 3)
        if got != want:
            fail(problems, f"conv[{index}].w shape {got} != expected {want}")
        if dims(layer.get("b")) != (channels,):
            fail(problems, f"conv[{index}].b shape {dims(layer.get('b'))} != ({channels},)")
        if not finite(layer.get("w")) or not finite(layer.get("b")):
            fail(problems, f"conv[{index}] holds a non-finite weight — net.rs rejects NaN/inf")

    for head in ("move_head", "pair_head"):
        block = net.get(head)
        if not isinstance(block, dict):
            fail(problems, f"missing `{head}`")
            continue
        # Exported as tolist(weight)[0]: the head is a 1x1 Conv2d with one output
        # channel, so its weight is [1][channels][1][1] and dropping the output
        # axis leaves [channels][1][1] — the trailing spatial dims survive. Both
        # artifacts/mcts_champion.json and fixtures/mcts/mcts_selfplay_tiny.json
        # are shaped this way; expecting a flat [channels] here is the mistake.
        if dims(block.get("w")) != (channels, 1, 1):
            fail(problems, f"{head}.w shape {dims(block.get('w'))} != ({channels}, 1, 1)")
        if not isinstance(block.get("b"), (int, float)) or not math.isfinite(block["b"]):
            fail(problems, f"{head}.b must be a finite scalar, got {block.get('b')!r}")
        if not finite(block.get("w")):
            fail(problems, f"{head}.w holds a non-finite weight")

    if not isinstance(net.get("pair_bias"), (int, float)) or not math.isfinite(net.get("pair_bias", float("nan"))):
        fail(problems, f"pair_bias must be a finite scalar, got {net.get('pair_bias')!r}")

    head = net.get("value_head")
    if arch == "conv-policy-value-v1" and not isinstance(head, dict):
        fail(problems, "arch declares conv-policy-value-v1 but there is no `value_head`")
    elif isinstance(head, dict):
        hidden = dims(head.get("fc1_w"))
        if len(hidden) != 2 or hidden[1] != channels:
            fail(problems, f"value_head.fc1_w shape {hidden} != (hidden, {channels})")
        elif dims(head.get("fc1_b")) != (hidden[0],):
            fail(problems, f"value_head.fc1_b shape {dims(head.get('fc1_b'))} != ({hidden[0]},)")
        elif dims(head.get("fc2_w")) != (hidden[0],):
            fail(problems, f"value_head.fc2_w shape {dims(head.get('fc2_w'))} != ({hidden[0]},)")
        if not isinstance(head.get("fc2_b"), (int, float)) or not math.isfinite(head.get("fc2_b", float("nan"))):
            fail(problems, f"value_head.fc2_b must be a finite scalar, got {head.get('fc2_b')!r}")
        for key in ("fc1_w", "fc1_b", "fc2_w"):
            if not finite(head.get(key)):
                fail(problems, f"value_head.{key} holds a non-finite weight")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("artifact")
    ap.add_argument("--reference", default=None, help="known-good artifact to compare field structure against")
    ap.add_argument(
        "--require-identical",
        action="store_true",
        help="fail unless the shape signature matches --reference exactly (use when the reference has the same channels/layers)",
    )
    args = ap.parse_args()

    with open(args.artifact) as handle:
        net = json.load(handle)

    problems = []
    check(net, problems)

    sig = signature(net)
    print(f"{args.artifact}:")
    for key, value in sig.items():
        print(f"  {key}: {value}")

    if args.reference:
        with open(args.reference) as handle:
            ref = json.load(handle)
        ref_sig = signature(ref)
        missing = sorted(set(ref_sig) - set(sig))
        extra = sorted(set(sig) - set(ref_sig))
        print(f"\nvs reference {args.reference}:")
        if missing:
            fail(problems, f"absent field group(s) the reference has: {missing}")
        if extra:
            print(f"  note: field group(s) the reference lacks: {extra}")
        # Channel/layer counts legitimately differ (a smoke net is smaller than
        # the champion); the *arch contract* must not.
        for key in ("arch", "board", "planes"):
            if sig.get(key) != ref_sig.get(key):
                fail(problems, f"{key} differs from reference: {sig.get(key)!r} vs {ref_sig.get(key)!r}")
        print(f"  same field groups: {not missing}")
        print(f"  same arch/board/planes: {all(sig.get(k) == ref_sig.get(k) for k in ('arch', 'board', 'planes'))}")
        # Informational by default: a smoke net is deliberately smaller than the
        # champion, so an identical signature is only *expected* when the
        # reference was trained at the same channels/layers. When the caller
        # knows it was (roundtrip.sh step 5b, against the 8x2 tiny fixture),
        # --require-identical turns that expectation into a gate — otherwise a
        # trainer that quietly ignored --channels/--layers, or changed an
        # exported tensor's shape, still goes green.
        identical = sig == ref_sig
        print(f"  identical shape signature: {identical}")
        if args.require_identical and not identical:
            differing = sorted(k for k in set(sig) | set(ref_sig) if sig.get(k) != ref_sig.get(k))
            fail(problems, f"--require-identical: shape signature differs from the reference in {differing}")
            for key in differing:
                fail(problems, f"    {key}: {sig.get(key)!r} vs reference {ref_sig.get(key)!r}")

    if problems:
        print(f"\nFAIL: {len(problems)} problem(s)", file=sys.stderr)
        for item in problems:
            print(f"  {item}", file=sys.stderr)
        return 1
    print("\nOK: artifact matches the conv-policy-value-v1 schema net.rs loads")
    return 0


if __name__ == "__main__":
    sys.exit(main())
