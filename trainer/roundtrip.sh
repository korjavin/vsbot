#!/usr/bin/env bash
# roundtrip.sh — prove the S4 training pipeline end to end, in one command.
#
#   trainer/roundtrip.sh
#
# rows -> validate -> train (nnue-trainer, UNCHANGED, in docker) -> artifact
#      -> schema check -> load in our Rust inference
#
# The point is schema compatibility, not a good net: it trains a deliberately
# tiny net for a few epochs and throws it away. What it establishes is that
# every seam between the Rust side and the python trainer holds — so when the
# `virus-mcts` self-play emitter lands, proving it is one command:
#
#   SELFPLAY_JSONL=work/gen1/selfplay.jsonl trainer/roundtrip.sh
#
# With SELFPLAY_JSONL unset, trainer/make_reference_rows.py synthesises
# contract-valid stand-in rows so the pipeline is runnable *before* the emitter
# exists. That is the only difference between the two invocations.
#
# Requirements: docker, and ~/Project/nnue-trainer checked out (see TRAINER_HOME).
# The cargo step is skipped with a stated verdict if cargo is missing.
#
# Knobs (env):
#   TRAINER_HOME     nnue-trainer checkout            (default ~/Project/nnue-trainer)
#   SELFPLAY_JSONL   real rows to use instead of synthetic ones  (default: synthesise)
#   WORK             scratch dir                      (default target/roundtrip)
#   IMAGE            trainer image tag                (default vsbot-trainer:cpu)
#   REBUILD=1        force a docker build even if IMAGE exists
#   GAMES            synthetic games to emit          (default 12)
#   EPOCHS           training epochs                  (default 3)
#   CHANNELS/LAYERS  net size — small on purpose      (default 8 / 2)
#   SEED             trainer seed                     (default 7)
#   SKIP_RUST=1      skip the cargo load check (prints why, still exits 0)
#   JOBS             cargo build jobs                 (default 2 — the box is shared)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

: "${TRAINER_HOME:=$HOME/Project/nnue-trainer}"
: "${SELFPLAY_JSONL:=}"
: "${WORK:=$ROOT/target/roundtrip}"
: "${IMAGE:=vsbot-trainer:cpu}"
: "${REBUILD:=0}"
: "${GAMES:=12}"
: "${EPOCHS:=3}"
: "${CHANNELS:=8}"
: "${LAYERS:=2}"
: "${SEED:=7}"
: "${SKIP_RUST:=0}"
: "${JOBS:=2}"

REFERENCE="$ROOT/artifacts/mcts_champion.json"

step() { printf '\n=== %s ===\n' "$*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

# ---------------------------------------------------------------- 0. preflight
step "0/6 preflight"
command -v docker >/dev/null || die "docker not found — it is how torch runs on this box (see trainer/README.md)"
docker info >/dev/null 2>&1 || die "docker daemon not reachable"
TRAINER_SCRIPT="$TRAINER_HOME/python/mcts/train_selfplay.py"
[ -f "$TRAINER_SCRIPT" ] || die "no trainer at $TRAINER_SCRIPT — clone nnue-trainer or set TRAINER_HOME"
[ -f "$REFERENCE" ] || die "missing reference artifact $REFERENCE"
echo "trainer:   $TRAINER_HOME (mounted read-only; NOT vendored into this repo)"
echo "work:      $WORK"
# Check the checkers before trusting them: steps 3 and 5 are only worth running
# if they still reject the rows/artifacts they are supposed to reject.
python3 "$ROOT/trainer/selftest.py" 2>&1 | tail -3
mkdir -p "$WORK"
WORK="$(cd "$WORK" && pwd)"
rm -f "$WORK/candidate.json" "$WORK/parity.json"

# ---------------------------------------------------------------- 1. image
step "1/6 trainer image"
if [ "$REBUILD" = 1 ] || ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "building $IMAGE (first run pulls the CPU torch wheel — a few minutes)"
  docker build -t "$IMAGE" -f "$ROOT/trainer/Dockerfile" "$ROOT/trainer"
else
  echo "$IMAGE already built (REBUILD=1 to force)"
fi
docker run --rm "$IMAGE" python -c 'import torch; print("torch", torch.__version__)'

# ---------------------------------------------------------------- 2. rows
step "2/6 self-play rows"
if [ -n "$SELFPLAY_JSONL" ]; then
  [ -f "$SELFPLAY_JSONL" ] || die "SELFPLAY_JSONL=$SELFPLAY_JSONL does not exist"
  echo "using REAL rows: $SELFPLAY_JSONL"
  cp -f "$SELFPLAY_JSONL" "$WORK/selfplay.jsonl"
else
  echo "no SELFPLAY_JSONL set — synthesising stand-in rows (see trainer/make_reference_rows.py)"
  python3 "$ROOT/trainer/make_reference_rows.py" "$WORK/selfplay.jsonl" --games "$GAMES" --seed "$SEED"
fi

# ---------------------------------------------------------------- 3. contract
step "3/6 row contract (field-for-field vs SelfPlayMcts.java)"
python3 "$ROOT/trainer/validate_rows.py" "$WORK/selfplay.jsonl"

# ---------------------------------------------------------------- 4. train
step "4/6 train — nnue-trainer python/mcts/train_selfplay.py, unchanged"
# --user keeps the artifacts owned by the invoking user rather than root; --cpus
# is deliberate courtesy on a shared box (torch would otherwise take every core).
docker run --rm \
  --user "$(id -u):$(id -g)" \
  --cpus "$JOBS" \
  -v "$TRAINER_HOME":/trainer:ro \
  -v "$WORK":/work \
  "$IMAGE" \
  python python/mcts/train_selfplay.py /work/selfplay.jsonl \
    --out /work/candidate.json \
    --fixture /work/parity.json \
    --epochs "$EPOCHS" --channels "$CHANNELS" --layers "$LAYERS" --seed "$SEED"
[ -s "$WORK/candidate.json" ] || die "training exited 0 but wrote no candidate"
echo "candidate: $WORK/candidate.json ($(wc -c < "$WORK/candidate.json" | tr -d ' ') bytes)"

# ---------------------------------------------------------------- 5. schema
step "5/6 artifact schema vs the promoted champion"
python3 "$ROOT/trainer/validate_artifact.py" "$WORK/candidate.json" --reference "$REFERENCE"

# CHANNELS=8/LAYERS=2 is not an arbitrary "small": it is exactly the geometry of
# fixtures/mcts/mcts_selfplay_tiny.json, the vendored net the Java loader's
# parity test uses. At the defaults the freshly trained candidate must therefore
# have a byte-for-byte identical *shape signature* to that fixture — a sharper
# assertion than "matches the champion's arch", and it is free.
if [ "$CHANNELS" = 8 ] && [ "$LAYERS" = 2 ] && [ -f "$ROOT/fixtures/mcts/mcts_selfplay_tiny.json" ]; then
  step "5b/6 same geometry as the vendored tiny fixture"
  python3 "$ROOT/trainer/validate_artifact.py" "$WORK/candidate.json" \
    --reference "$ROOT/fixtures/mcts/mcts_selfplay_tiny.json" | tail -6
fi

# ---------------------------------------------------------------- 6. rust load
step "6/6 Rust inference load (crates/virus-mcts PolicyValueNet::load)"
if [ "$SKIP_RUST" = 1 ]; then
  echo "SKIPPED: SKIP_RUST=1. Verdict rests on step 5 (python schema check) alone."
elif ! command -v cargo >/dev/null; then
  echo "SKIPPED: cargo not on PATH (try: export PATH=\"\$HOME/.cargo/bin:\$PATH\")."
  echo "Verdict rests on step 5 (python schema check) alone."
else
  # batchgauntlet's --net takes any path and calls PolicyValueNet::load, which is
  # the exhaustive validator (arch name, declared-vs-actual shapes, finiteness).
  # Using an existing example rather than adding one keeps crates/** untouched.
  # Two games at a 1 ms budget: the load is the assertion, the games are proof
  # the weights actually drive a search.
  cargo run --jobs "$JOBS" -p virus-mcts --example batchgauntlet -- \
    --net "$WORK/candidate.json" --games 2 --millis 1 --jobs 1 --seed "$SEED"
fi

step "ROUND TRIP GREEN"
echo "rows -> train_selfplay.py (unchanged) -> $WORK/candidate.json -> loaded by virus-mcts"
