#!/usr/bin/env python3
"""Self-test for the row/artifact validators. Stdlib only — no torch, no docker, no cargo.

    python3 trainer/selftest.py

`validate_rows.py` is what the Rust emitter will be held to, so the thing that
actually needs proving is that it *rejects* the wrong rows. A validator that
only ever prints OK is worse than none: it converts an unchecked assumption into
a false assurance. Every case below is a mistake a schema port really makes.
"""

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent

sys.path.insert(0, str(HERE))

import make_reference_rows  # noqa: E402
import rows_schema  # noqa: E402


def run(script, *args):
    """Run one of the validators as a subprocess; returns (exit code, stdout+stderr)."""
    result = subprocess.run(
        [sys.executable, str(HERE / script), *args],
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stdout + result.stderr


def write_rows(rows, directory, name="rows.jsonl"):
    path = Path(directory) / name
    path.write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows))
    return str(path)


def good_rows():
    """Two games' worth of valid rows, from the same generator roundtrip.sh uses."""
    import random

    rng = random.Random(11)
    return make_reference_rows.game(rng, "g0") + make_reference_rows.game(rng, "g1")


class FlatIdCodecTest(unittest.TestCase):
    """The id space SelfPlayMcts.flatIndex defines: [0,144) moves, 144 + min*144 + max pairs."""

    def test_move_ids_are_row_major_cells(self):
        self.assertEqual(rows_schema.move_id(0, 0), 0)
        self.assertEqual(rows_schema.move_id(0, 11), 11)
        self.assertEqual(rows_schema.move_id(11, 11), 143)

    def test_pair_ids_are_order_independent(self):
        self.assertEqual(rows_schema.pair_id(7, 3), rows_schema.pair_id(3, 7))
        self.assertEqual(rows_schema.pair_id(3, 7), 144 + 3 * 144 + 7)

    def test_pair_needs_two_distinct_cells(self):
        with self.assertRaises(ValueError):
            rows_schema.pair_id(5, 5)

    def test_decode_inverts_both_encodings(self):
        self.assertEqual(rows_schema.decode(rows_schema.move_id(4, 2)), ("move", 50))
        self.assertEqual(rows_schema.decode(rows_schema.pair_id(3, 7)), ("pair", 3, 7))

    def test_flat_space_size_matches_the_trainer(self):
        # train_policy.FLAT = CELLS + CELLS*CELLS. A mismatch here silently
        # shifts every pair id relative to the net's output layer.
        self.assertEqual(rows_schema.FLAT, 144 + 144 * 144)


class ValidRowsTest(unittest.TestCase):
    def test_the_generator_produces_rows_the_validator_accepts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_rows(good_rows(), directory)
            code, out = run("validate_rows.py", path)
            self.assertEqual(code, 0, out)
            self.assertIn("OK:", out)

    def test_generation_is_deterministic_for_a_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "a.jsonl"
            second = Path(directory) / "b.jsonl"
            for target in (first, second):
                code, out = run("make_reference_rows.py", str(target), "--games", "3", "--seed", "5")
                self.assertEqual(code, 0, out)
            self.assertEqual(first.read_text(), second.read_text())


class RejectionTest(unittest.TestCase):
    """Each case is a real porting mistake; each must exit 1 with a pointed message."""

    def assert_rejected(self, mutate, expect):
        rows = good_rows()
        mutate(rows)
        with tempfile.TemporaryDirectory() as directory:
            path = write_rows(rows, directory)
            code, out = run("validate_rows.py", path)
        self.assertEqual(code, 1, f"expected rejection, got OK:\n{out}")
        self.assertIn(expect, out)

    def test_rejects_z_flipped_into_the_mover_frame(self):
        # The headline failure: an emitter that helpfully pre-flips z. The rows
        # stay individually well-formed, so only the per-game invariant catches it.
        def mutate(rows):
            for row in rows:
                if row["mover"] == 2:
                    row["z"] = -row["z"]

        self.assert_rejected(mutate, "z must stay ABSOLUTE")

    def test_rejects_unordered_pair_ids(self):
        def mutate(rows):
            rows[0]["pi"][0] = rows_schema.PAIR_OFFSET + 9 * rows_schema.CELLS + 2  # i=9 > j=2

        self.assert_rejected(mutate, "must be 144 + min*144 + max")

    def test_rejects_normalised_visit_counts(self):
        # pv is raw visits. Probabilities are ints-rounded-to-0 or floats; both wrong.
        def mutate(rows):
            total = sum(rows[0]["pv"])
            rows[0]["pv"] = [v / total for v in rows[0]["pv"]]

        self.assert_rejected(mutate, "non-negative integer visit counts")

    def test_rejects_forced_single_action_positions(self):
        def mutate(rows):
            rows[0]["pi"] = rows[0]["pi"][:1]
            rows[0]["pv"] = rows[0]["pv"][:1]

        self.assert_rejected(mutate, "multi-choice positions")

    def test_rejects_pi_pv_length_mismatch(self):
        def mutate(rows):
            rows[0]["pv"] = rows[0]["pv"][:-1]

        self.assert_rejected(mutate, "pi/pv length mismatch")

    def test_rejects_wrong_board_size(self):
        def mutate(rows):
            rows[0]["sym"] = rows[0]["sym"][:100]

        self.assert_rejected(mutate, "sym must be 144 entries")

    def test_rejects_out_of_range_symbols(self):
        def mutate(rows):
            rows[0]["sym"][0] = 8  # the out-of-bounds symbol; never valid on-board

        self.assert_rejected(mutate, "non-symbol value")

    def test_rejects_missing_mover(self):
        def mutate(rows):
            del rows[0]["mover"]

        self.assert_rejected(mutate, "missing field(s) ['mover']")

    def test_rejects_duplicate_action_ids(self):
        def mutate(rows):
            rows[0]["pi"][1] = rows[0]["pi"][0]

        self.assert_rejected(mutate, "duplicate action id")

    def test_rejects_neutral_pair_after_the_placement_is_spent(self):
        def mutate(rows):
            rows[0]["nuo"] = 1
            rows[0]["pi"].append(rows_schema.pair_id(4, 9))
            rows[0]["pv"].append(3)

        self.assert_rejected(mutate, "pi offers a neutral pair")

    def test_rejects_a_zero_policy_target(self):
        def mutate(rows):
            rows[0]["pv"] = [0] * len(rows[0]["pi"])

        self.assert_rejected(mutate, "pv sums to 0")

    def test_reports_rather_than_crashes_on_an_unhashable_action(self):
        # A nested array in pi is well-formed JSON but not an action id. The
        # duplicate-detection pass used to hash pi before type-checking it, so
        # this aborted the whole file with a TypeError traceback instead of
        # reporting the row. Malformed output is what this tool is *for*.
        def mutate(rows):
            rows[0]["pi"][0] = [1, 2]

        self.assert_rejected(mutate, "is not an int")

    def test_an_unhashable_action_does_not_hide_later_rows(self):
        # The aggregated report must survive the bad row and keep checking.
        rows = good_rows()
        rows[0]["pi"][0] = {"cell": 4}
        rows[-1]["mover"] = 3
        with tempfile.TemporaryDirectory() as directory:
            path = write_rows(rows, directory)
            code, out = run("validate_rows.py", path)
        self.assertEqual(code, 1, out)
        self.assertNotIn("Traceback", out)
        self.assertIn("is not an int", out)
        self.assertIn("mover must be 1 or 2", out)


class ArtifactValidatorTest(unittest.TestCase):
    """The vendored fixtures are known-good; mutations of them must be caught."""

    FIXTURE = HERE.parent / "fixtures" / "mcts" / "mcts_selfplay_tiny.json"
    CHAMPION = HERE.parent / "artifacts" / "mcts_champion.json"

    def test_accepts_the_vendored_tiny_net_and_the_champion(self):
        for path in (self.FIXTURE, self.CHAMPION):
            code, out = run("validate_artifact.py", str(path))
            self.assertEqual(code, 0, f"{path}:\n{out}")

    def test_the_tiny_fixture_and_champion_share_the_arch_contract(self):
        code, out = run("validate_artifact.py", str(self.FIXTURE), "--reference", str(self.CHAMPION))
        self.assertEqual(code, 0, out)
        self.assertIn("same arch/board/planes: True", out)

    def assert_artifact_rejected(self, mutate, expect):
        net = json.loads(self.FIXTURE.read_text())
        mutate(net)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "net.json"
            path.write_text(json.dumps(net))
            code, out = run("validate_artifact.py", str(path))
        self.assertEqual(code, 1, f"expected rejection, got OK:\n{out}")
        self.assertIn(expect, out)

    def test_rejects_an_unsupported_arch(self):
        self.assert_artifact_rejected(
            lambda net: net["meta"].update(arch="conv-policy-value-v2"),
            "is not one of",
        )

    def test_rejects_a_layer_count_that_disagrees_with_the_conv_stack(self):
        self.assert_artifact_rejected(
            lambda net: net["conv"].append(copy.deepcopy(net["conv"][-1])),
            "conv` holds",
        )

    def test_rejects_a_non_finite_weight(self):
        # net.rs refuses NaN at load rather than letting it poison a search.
        def mutate(net):
            net["conv"][0]["b"][0] = float("nan")

        self.assert_artifact_rejected(mutate, "non-finite weight")

    def test_rejects_a_value_head_stripped_from_a_value_arch(self):
        self.assert_artifact_rejected(lambda net: net.pop("value_head"), "there is no `value_head`")

    def test_require_identical_gates_a_geometry_mismatch(self):
        # roundtrip.sh step 5b asserts the 8x2 candidate matches the tiny
        # fixture exactly. Without --require-identical that was a printed
        # observation, not a gate: a trainer that ignored --channels/--layers
        # went green anyway.
        code, out = run(
            "validate_artifact.py",
            str(self.CHAMPION),  # 32x4 — deliberately not the fixture's geometry
            "--reference",
            str(self.FIXTURE),
            "--require-identical",
        )
        self.assertEqual(code, 1, f"expected the mismatch to gate:\n{out}")
        self.assertIn("shape signature differs", out)

    def test_require_identical_passes_for_a_matching_geometry(self):
        code, out = run(
            "validate_artifact.py",
            str(self.FIXTURE),
            "--reference",
            str(self.FIXTURE),
            "--require-identical",
        )
        self.assertEqual(code, 0, out)
        self.assertIn("identical shape signature: True", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
