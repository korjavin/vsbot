#!/usr/bin/env bash
# generation.sh — ONE self-play RL generation, end to end, on this box.
#
#   trainer/generation.sh [--from <stage>] [--until <stage>]
#
# Stages, in order, each idempotent and stamped:
#
#   selfplay   $SHARDS `selfplay` processes play $SELFPLAY_GAMES games with the
#              CHAMPION net at $SIMS sims/action, deterministic per (seed,shard).
#              Shards are concatenated and put through trainer/validate_rows.py,
#              which is a HARD STOP: rows that fail the SelfPlayMcts contract
#              train a net on a lie, and the lie survives every later stage.
#   train      the unchanged nnue-trainer `python/mcts/train_selfplay.py`, in
#              docker, over a $WINDOW-generation sliding window, at the
#              CHAMPION'S geometry -> gen$GEN/candidate.json. Then the python
#              schema check, then a real `PolicyValueNet::load` from Rust.
#   gauntlet   candidate vs champion, net-vs-net, fixed sims, colour-paired,
#              per-instance seeds spaced 1000.
#   report     pools the instances, applies (W+0.5D)/N >= $GATE, writes
#              gen$GEN/report.md. PROMOTES NOTHING — see "no auto-promotion".
#
# Resuming: each finished stage drops $GDIR/<stage>.done and is skipped on a
# re-run. `--from train` clears that stamp and every later one and re-runs from
# there; `--until <stage>` stops after it. So a run interrupted in `train`
# resumes with `trainer/generation.sh` (self-play is not replayed), and a
# deliberate retrain is `trainer/generation.sh --from train`.
#
# Exit codes mirror the Java reference (scripts/mcts_selfplay_gen.sh):
#   0  the pooled score met $GATE at the sample size actually played
#   1  KEPT BACK — candidate did not meet the gate. A NORMAL outcome for a
#      first generation, not a script failure. The report is still written.
#   2  a stage failed. Nothing downstream is trustworthy.
#
# ## No auto-promotion, deliberately
#
# The Java script copies a passing candidate over champion.json. This one does
# not, and never should: docs/CANARY.md promotes on Gate A **plus** Gate B
# (no regression vs ab-enhanced and vs the Go bot) **plus** the Gate C live
# soak, and Gate A itself needs N >= 400. A local generation is one third of
# one of those three gates. It reports; the canary pipeline promotes.
#
# ## Knobs (env), defaults sized for THIS box (4 cores, shared)
#
#   GEN=6                 generation number; picks $WORK/gen$GEN
#   WORK=work             output root (gitignored)
#   CHAMPION=artifacts/mcts_champion.json      the net self-play plays with and
#                                              the candidate is gated against
#   SELFPLAY_GAMES=1000   games across all shards
#   SIMS=192              self-play sims/action
#   SHARDS=2              self-play processes. 2, not $(nproc): a T2 arena cell
#                         and other executors share this box.
#   SEED_BASE=11          self-play seed = SEED_BASE + GEN*1000
#   WINDOW=3              training window, in generations (see below)
#   EPOCHS=8              trainer epochs
#   CHANNELS=32 LAYERS=4  net geometry. These are the CHAMPION's, and the
#                         trainer's own defaults. roundtrip.sh's 8x2 is a
#                         schema test, not a ladder net — training the ladder at
#                         8x2 would produce a candidate that loads, gauntlets,
#                         and is structurally incapable of beating gen-5.
#   TRAIN_SEED=7          trainer seed
#   GATE=0.55             pooled-score threshold
#   GATE_GAMES=100        games per gauntlet instance
#   GATE_INSTANCES=1      gauntlet instances, played SEQUENTIALLY
#   GATE_SIMS=192         gauntlet sims/action (fixed sims: load-tolerant)
#   GATE_JOBS=2           concurrent games inside one instance
#   GATE_AUTOSCALE=1      if the arena lock is free AND no arena is running,
#                         take the lock and raise GATE_INSTANCES to
#                         $GATE_FULL_INSTANCES (=4, i.e. the full 400). Set 0 to
#                         pin GATE_INSTANCES exactly.
#   ARENA_LOCK=/tmp/vsbot-arena.lock
#   TRAINER_HOME=~/Project/nnue-trainer      mounted READ-ONLY at /trainer
#   IMAGE=vsbot-trainer:cpu
#   JOBS=2                docker --cpus and cargo --jobs
#
# ## The $WINDOW=3 sliding window, and what it means for gen 6
#
# Training consumes the current generation plus the previous two. gen 6 is the
# FIRST Rust generation: there is no work/gen4 or work/gen5 selfplay.jsonl on
# this box (gens 1-5 were produced by the Java emitter, in that repo's work
# tree), so the window here collapses to gen 6's own rows alone. That is a real
# handicap and it is stated in the report, not hidden: a net trained on one
# generation of its own predecessor's games is the exact overfitting case
# WINDOW=3 exists to damp. From gen 7 the window fills naturally — gen 7 pools
# gen6+gen7, gen 8 pools gen6+gen7+gen8, gen 9 drops gen6 — with no change to
# this script, because the window is "generations that HAVE a selfplay.jsonl",
# not "the last three directories".
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

STAGES="selfplay train gauntlet report"

usage() { sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 1; }

FROM=""
UNTIL=""
while [ $# -gt 0 ]; do
  case "$1" in
    --from)  FROM="$2";  shift 2 ;;
    --until) UNTIL="$2"; shift 2 ;;
    -h|--help) usage ;;
    *) echo "unknown argument: $1" >&2; usage ;;
  esac
done

: "${GEN:=6}"
: "${WORK:=work}"
: "${CHAMPION:=artifacts/mcts_champion.json}"
: "${SELFPLAY_GAMES:=1000}"
: "${SIMS:=192}"
: "${SHARDS:=2}"
: "${SEED_BASE:=11}"
: "${WINDOW:=3}"
: "${EPOCHS:=8}"
: "${CHANNELS:=32}"
: "${LAYERS:=4}"
: "${TRAIN_SEED:=7}"
: "${GATE:=0.55}"
: "${GATE_GAMES:=100}"
: "${GATE_INSTANCES:=1}"
: "${GATE_FULL_INSTANCES:=4}"
: "${GATE_SIMS:=192}"
: "${GATE_JOBS:=2}"
: "${GATE_AUTOSCALE:=1}"
: "${ARENA_LOCK:=/tmp/vsbot-arena.lock}"
: "${TRAINER_HOME:=$HOME/Project/nnue-trainer}"
: "${IMAGE:=vsbot-trainer:cpu}"
: "${JOBS:=2}"

die() { echo "ERROR: $*" >&2; exit 2; }
step() { printf '\n>> [gen %s / %s] === %s\n' "$GEN" "$1" "$(date '+%Y-%m-%d %H:%M:%S')"; }

mkdir -p "$WORK"
WORK="$(cd "$WORK" && pwd)"
GDIR="$WORK/gen$GEN"
mkdir -p "$GDIR/logs"
CAND="$GDIR/candidate.json"
ROWS="$GDIR/selfplay.jsonl"
REPORT="$GDIR/report.md"

# Per-generation champion snapshot. Taken once per generation, so a run
# interrupted after self-play still gates against exactly the net its games were
# played with even if `artifacts/` moved underneath it — and re-taken for the
# NEXT generation, so a champion promoted by the canary pipeline is picked up.
#
# A single `$WORK/champion.json` shared by every generation had the second half
# of that backwards: it was written once, ever, so after a promotion each later
# generation kept self-playing and gating against the first champion the box had
# ever seen, silently, with $CHAMPION pointing at the right file the whole time.
CHAMP="$GDIR/champion.json"
if [ ! -f "$CHAMP" ]; then
  [ -f "$CHAMPION" ] || die "no champion at $CHAMPION"
  cp "$CHAMPION" "$CHAMP"
elif ! cmp -s "$CHAMP" "$CHAMPION"; then
  # Not fatal: a resumed run MUST keep its snapshot. But say so, because the
  # alternative is a report that names a champion the generation never played.
  echo "NOTE: gen $GEN's champion snapshot differs from $CHAMPION — keeping the"
  echo "      snapshot this generation started with. Bump GEN for a new champion."
fi
CHAMP_SHA="$(sha256sum "$CHAMP" | cut -c1-12)"

SELFPLAY_BIN="$ROOT/target/release/selfplay"
NETGAUNTLET_BIN="$ROOT/trainer/netgauntlet/target/release/netgauntlet"
SELFPLAY_SEED=$((SEED_BASE + GEN * 1000))

# ---------------------------------------------------------------- stage machinery
stamp() { echo "$GDIR/$1.done"; }

for s in $FROM $UNTIL; do
  case " $STAGES " in *" $s "*) ;; *) die "'$s': not one of: $STAGES" ;; esac
done
if [ -n "$FROM" ]; then
  clear=0
  for s in $STAGES; do
    if [ "$s" = "$FROM" ]; then clear=1; fi
    if [ "$clear" = 1 ]; then rm -f "$(stamp "$s")"; fi
  done
fi

run_stage() {
  local s="$1"
  # `report` is cheap and is the thing a human re-reads, so it always re-runs.
  if [ "$s" != report ] && [ -f "$(stamp "$s")" ]; then
    echo ">> [gen $GEN / $s] already done — skipping (redo with --from $s)"
    return 0
  fi
  step "$s"
  # `|| return` and not a bare call: this function is invoked under `||`, which
  # switches `set -e` off for everything inside it, so a stage's failure has to
  # be propagated by hand. Without this the KEEP verdict (stage_report's
  # non-zero) was swallowed by the `touch` line below and the script exited 0.
  "stage_$s" || return $?
  [ "$s" = report ] || touch "$(stamp "$s")"
}

# Each stage records what IT actually did, so `--from report` after a re-run
# with different knobs cannot describe the wrong configuration. The report
# concatenates these rows rather than re-reading the environment.
ran() { printf '%s\n' "$2" > "$GDIR/ran_$1.md"; }

# ---------------------------------------------------------------- 1. self-play
stage_selfplay() {
  [ -x "$SELFPLAY_BIN" ] || die "no $SELFPLAY_BIN — cargo build --release -p virus-selfplay --jobs $JOBS"
  echo "$SELFPLAY_GAMES games, $SIMS sims/action, seed $SELFPLAY_SEED, $SHARDS shards, net $CHAMP"
  rm -f "$GDIR"/selfplay_shard_*.jsonl "$ROWS"
  local pids="" i
  for i in $(seq 0 $((SHARDS - 1))); do
    "$SELFPLAY_BIN" \
      --net "$CHAMP" \
      --out "$GDIR/selfplay_shard_$i.jsonl" \
      --games "$SELFPLAY_GAMES" --sims "$SIMS" \
      --shard "$i" --shards "$SHARDS" --seed "$SELFPLAY_SEED" \
      > "$GDIR/logs/selfplay_$i.log" 2>&1 &
    pids="$pids $!"
  done
  local p
  for p in $pids; do wait "$p" || die "a self-play shard failed — see $GDIR/logs/selfplay_*.log"; done
  # Shard order is fixed by the glob, so the concatenation is reproducible.
  cat "$GDIR"/selfplay_shard_*.jsonl > "$ROWS"
  echo "rows: $(wc -l < "$ROWS" | tr -d ' ')"

  # HARD STOP. Not a warning: every downstream stage would still "succeed".
  python3 "$ROOT/trainer/validate_rows.py" "$ROWS" \
    || die "row contract violated — the emitter and the trainer disagree; do NOT train on these"

  ran selfplay "| self-play | $SELFPLAY_GAMES games, $SIMS sims/action, $SHARDS shards, seed $SELFPLAY_SEED, champion \`$CHAMPION\` (sha256 $CHAMP_SHA), $(wc -l < "$ROWS" | tr -d ' ') rows, \`validate_rows.py\` clean |"
}

# ---------------------------------------------------------------- 2. train
stage_train() {
  [ -s "$ROWS" ] || die "no rows at $ROWS (run the selfplay stage first)"
  [ -f "$TRAINER_HOME/python/mcts/train_selfplay.py" ] \
    || die "no trainer at $TRAINER_HOME/python/mcts/train_selfplay.py — clone nnue-trainer or set TRAINER_HOME"
  command -v docker >/dev/null || die "docker not found — it is how torch runs on this box"
  docker image inspect "$IMAGE" >/dev/null 2>&1 \
    || die "no $IMAGE — build it with trainer/roundtrip.sh (step 1) or docker build -f trainer/Dockerfile trainer"

  # The window, over generations that HAVE rows — gaps are skipped, not fatal.
  local datasets="" g
  for g in $(seq $((GEN - WINDOW + 1)) "$GEN"); do
    if [ "$g" -ge 1 ] && [ -s "$WORK/gen$g/selfplay.jsonl" ]; then
      datasets="$datasets /work/gen$g/selfplay.jsonl"
      echo "window: gen$g ($(wc -l < "$WORK/gen$g/selfplay.jsonl" | tr -d ' ') rows)"
    fi
  done
  [ -n "$datasets" ] || die "no self-play datasets for generations <= $GEN"

  rm -f "$CAND"
  # shellcheck disable=SC2086
  # PYTHONUNBUFFERED because this stage runs for an hour on a real generation
  # and python block-buffers stdout when it is a pipe: without it the epoch
  # lines sit in the container's buffer until exit, so an interrupted run leaves
  # an EMPTY train.log and the holdout curves — the diagnosis for a kept-back
  # candidate — are gone. Learned the hard way on gen 6.
  docker run --rm \
    --user "$(id -u):$(id -g)" \
    --cpus "$JOBS" \
    -e PYTHONUNBUFFERED=1 \
    -v "$TRAINER_HOME":/trainer:ro \
    -v "$WORK":/work \
    "$IMAGE" \
    python python/mcts/train_selfplay.py $datasets \
      --out "/work/gen$GEN/candidate.json" \
      --epochs "$EPOCHS" --channels "$CHANNELS" --layers "$LAYERS" --seed "$TRAIN_SEED" \
    2>&1 | tee "$GDIR/logs/train.log"
  [ -s "$CAND" ] || die "training exited 0 but wrote no candidate at $CAND"

  # Structural check against the champion. --require-identical because the
  # geometry is deliberately the champion's: at matched CHANNELS/LAYERS the two
  # artifacts must have the same shape signature, so a trainer that ignored
  # --channels/--layers is caught here rather than in the gauntlet, where it
  # would look like a weak net instead of a wrong one.
  python3 "$ROOT/trainer/validate_artifact.py" "$CAND" --reference "$CHAMP" --require-identical

  # And the assertion that actually matters: our own Rust loader accepts it and
  # the weights drive a real search. batchgauntlet's --net calls
  # PolicyValueNet::load (arch name, declared-vs-actual shapes, finiteness of
  # every weight); two 1 ms games prove the load was not vacuous. Uses an
  # existing example, so crates/** stays untouched.
  if command -v cargo >/dev/null; then
    cargo run --release --jobs "$JOBS" -p virus-mcts --example batchgauntlet -- \
      --net "$CAND" --games 2 --millis 1 --jobs 1 --seed "$TRAIN_SEED" \
      2>&1 | tee "$GDIR/logs/rustload.log"
  else
    die "cargo not on PATH — the Rust load check is not optional here (export PATH=\"\$HOME/.cargo/bin:\$PATH\")"
  fi

  ran train "| train | window=$WINDOW ($(echo $datasets | wc -w | tr -d ' ') generation(s):$(echo "$datasets" | sed 's#/work/gen#gen#g; s#/selfplay.jsonl##g')), epochs=$EPOCHS, ${CHANNELS}ch x ${LAYERS} layers, seed $TRAIN_SEED, nnue-trainer \`train_selfplay.py\` UNCHANGED in \`$IMAGE\`; artifact schema identical to the champion's; loaded by \`PolicyValueNet::load\` |"
}

# ---------------------------------------------------------------- 3. gauntlet
#
# Sequential instances, not parallel ones. The Java script forks
# GATE_INSTANCES JVMs at once because it runs on a bigger machine; here 4
# concurrent instances x $GATE_JOBS games would oversubscribe 4 cores against
# whatever else is running. Fixed sims means sequencing costs wall clock and
# changes no number.
stage_gauntlet() {
  [ -x "$NETGAUNTLET_BIN" ] \
    || die "no $NETGAUNTLET_BIN — (cd trainer/netgauntlet && cargo build --release --jobs $JOBS)"
  [ -s "$CAND" ] || die "no candidate at $CAND (run the train stage first)"

  local instances="$GATE_INSTANCES" took_lock=0
  if [ "$GATE_AUTOSCALE" = 1 ] && [ "$instances" -lt "$GATE_FULL_INSTANCES" ]; then
    # mkdir is the atomic test-and-set. pgrep is the second condition because a
    # previous run can leave a stale lock, and because the thing being protected
    # is core contention, not the directory.
    if pgrep -f 'target/release/arena' >/dev/null 2>&1; then
      echo "autoscale: an arena run is live — staying at $instances instance(s)"
    elif mkdir "$ARENA_LOCK" 2>/dev/null; then
      took_lock=1
      instances="$GATE_FULL_INSTANCES"
      echo "autoscale: took $ARENA_LOCK — raising to $instances instances ($((instances * GATE_GAMES)) games)"
      # Release on ANY exit, including a stage failure, so the next user of the
      # box is not blocked by our crash.
      trap 'rmdir "$ARENA_LOCK" 2>/dev/null || true' EXIT
    else
      echo "autoscale: $ARENA_LOCK is held — staying at $instances instance(s)"
    fi
  fi
  echo "$instances x $GATE_GAMES games, candidate vs champion, $GATE_SIMS sims, seeds spaced 1000"

  # Stale logs from a run with more instances would be pooled by stage_report.
  rm -f "$GDIR"/logs/gauntlet_*.log
  local i seed
  for i in $(seq 0 $((instances - 1))); do
    seed=$((SEED_BASE + GEN * 10000 + i * 1000))
    echo "   instance $i: seed $seed"
    "$NETGAUNTLET_BIN" \
      --a-net "$CAND" --b-net "$CHAMP" \
      --games "$GATE_GAMES" --sims "$GATE_SIMS" --seed "$seed" --jobs "$GATE_JOBS" \
      > "$GDIR/logs/gauntlet_$i.log" 2>&1 \
      || die "gauntlet instance $i failed — see $GDIR/logs/gauntlet_$i.log"
    grep -h '^RESULT' "$GDIR/logs/gauntlet_$i.log"
  done
  if [ "$took_lock" = 1 ]; then
    rmdir "$ARENA_LOCK" 2>/dev/null || true
    trap - EXIT
  fi

  ran gauntlet "| gauntlet | $instances x $GATE_GAMES games = $((instances * GATE_GAMES)) at $GATE_SIMS FIXED sims, colour-paired, per-instance seeds from $((SEED_BASE + GEN * 10000)) spaced 1000, $GATE_JOBS concurrent games$( [ "$took_lock" = 1 ] && echo ", arena lock held" || echo ", arena lock NOT held") |"
}

# ---------------------------------------------------------------- 4. report
stage_report() {
  local results w l d n pooled capped stalled verdict lo hi
  results="$(cat "$GDIR"/logs/gauntlet_*.log 2>/dev/null | grep -h '^RESULT' || true)"
  [ -n "$results" ] || die "no RESULT lines in $GDIR/logs/gauntlet_*.log"

  # Pool FIRST, then divide. Applying the gate per instance and taking a
  # majority is a different, weaker test.
  read -r w l d capped stalled <<EOF
$(printf '%s\n' "$results" | awk '
    { for (i = 1; i <= NF; i++) { split($i, kv, "="); v[kv[1]] += kv[2] } }
    END { printf "%d %d %d %d %d", v["w"], v["l"], v["d"], v["capped"], v["stalled"] }')
EOF
  n=$((w + l + d))
  [ "$n" -gt 0 ] || die "pooled 0 games"

  read -r pooled verdict <<EOF
$(awk -v w="$w" -v d="$d" -v n="$n" -v t="$GATE" 'BEGIN {
    p = (w + 0.5 * d) / n; printf "%.4f %s", p, (p >= t ? "GATE-A-PASS" : "KEPT-BACK") }')
EOF
  # Wilson 95% on the HEADLINE win rate (draws in the denominator only) — the
  # same interval virus_arena::stats::wilson95 prints, recomputed here because
  # the pooled tally spans instances.
  read -r lo hi <<EOF
$(awk -v w="$w" -v n="$n" 'BEGIN {
    z = 1.959963984540054; p = w / n; den = 1 + z*z/n;
    c = (p + z*z/(2*n)) / den; m = z * sqrt((p*(1-p) + z*z/(4*n)) / n) / den;
    printf "%.1f %.1f", 100*(c-m), 100*(c+m) }')
EOF

  # Row diagnostics. When a candidate is kept back these are the useful half of
  # the output: a normalised visit entropy near 1.0 means the root visits were
  # near-uniform, i.e. the self-play produced a policy target with no signal in
  # it, and no amount of training fixes that — you raise --sims. Offline, and
  # therefore diagnostic only (ARCHITECTURE.md invariant 7).
  local rowstats
  rowstats="$(python3 - "$ROWS" <<'PY' 2>/dev/null || echo "(row diagnostics unavailable)"
import json, math, sys

path = sys.argv[1]
total = sum(1 for _ in open(path))
stride = max(1, total // 20000)          # cap the parse at ~20k rows
games, z_of, legal, entropy, sampled = set(), {}, 0.0, 0.0, 0
for index, line in enumerate(open(path)):
    if not line.strip():
        continue
    if index % stride:
        continue
    row = json.loads(line)
    games.add(row["g"])
    z_of[row["g"]] = row["z"]
    visits = row["pv"]
    k = len(visits)
    legal += k
    total_visits = sum(visits) or 1
    h = -sum((v / total_visits) * math.log(v / total_visits) for v in visits if v > 0)
    entropy += h / math.log(k) if k > 1 else 0.0
    sampled += 1

if sampled:
    print(f"rows {total} (diagnostics over {sampled}, every {stride}th)")
    print(f"mean legal actions per row   {legal / sampled:.1f}")
    print(f"mean normalised visit entropy {entropy / sampled:.3f}   (1.0 = uniform = no policy signal)")
    wins = [z for z in z_of.values()]
    print(f"outcomes over {len(games)} sampled games: p1 {wins.count(1)}  p2 {wins.count(-1)}  draw {wins.count(0)}")
PY
)"

  local rows_n games_n win_note
  rows_n="$( [ -s "$ROWS" ] && wc -l < "$ROWS" | tr -d ' ' || echo 0)"
  # `g` is the first field of every row (serde fixes the order), so the game id
  # is the line prefix up to the first comma — no JSON parse over a file that
  # can be hundreds of MB.
  games_n="$( [ -s "$ROWS" ] && python3 -c '
import sys
print(len({line.partition(",")[0] for line in open(sys.argv[1]) if line.strip()}))' "$ROWS" || echo 0)"
  if [ "$n" -ge 400 ]; then
    win_note="N=$n meets the Gate A minimum of 400."
  else
    win_note="N=$n is BELOW the Gate A minimum of 400: the interval is wide and a 400-game confirmation is required before any promotion claim."
  fi

  {
    echo "# gen $GEN report"
    echo
    echo "_$(date '+%Y-%m-%d %H:%M %Z') · $(hostname) · $(nproc) cores · load$(cut -d' ' -f1-3 /proc/loadavg | sed 's/^/ /')_"
    echo
    echo "## Verdict: **$verdict**"
    echo
    echo "| | |"
    echo "|---|---|"
    echo "| pooled score (W+0.5D)/N | **$pooled** (gate $GATE) |"
    echo "| candidate W-L-D | $w-$l-$d of $n |"
    echo "| headline win rate | $(awk -v w="$w" -v n="$n" 'BEGIN{printf "%.1f%%", 100*w/n}') Wilson95 [$lo%, $hi%] |"
    echo "| turn-capped (draws) | $capped |"
    echo "| stalled | $stalled |"
    echo
    echo "$win_note"
    echo
    echo "## What ran"
    echo
    echo "| stage | what it actually did |"
    echo "|---|---|"
    # Written by each stage as it ran — not re-derived from the environment,
    # which `--from report` could have changed since.
    cat "$GDIR"/ran_selfplay.md "$GDIR"/ran_train.md "$GDIR"/ran_gauntlet.md 2>/dev/null
    echo "| rows | $rows_n rows from $games_n games |"
    echo
    echo "Per-instance tallies:"
    echo
    echo '```'
    printf '%s\n' "$results"
    echo '```'
    echo
    echo "## Diagnostics"
    echo
    echo "Self-play rows:"
    echo
    echo '```'
    printf '%s\n' "$rowstats"
    echo '```'
    echo
    echo "Holdout metrics from the trainer (offline — ARCHITECTURE.md invariant 7:"
    echo "these do NOT gate anything, they are here to diagnose the gauntlet result):"
    echo
    echo '```'
    grep -E 'holdout|epoch|top1|MAE|loss' "$GDIR/logs/train.log" 2>/dev/null | tail -20 || echo "(no train log)"
    echo '```'
    echo
    echo "## Box-load caveat"
    echo
    echo "Fixed sims, not fixed time, so contention changes the wall clock and"
    echo "not the tally — but it is recorded anyway, because a stalled or capped"
    echo "count that moves with load would mean the opposite."
    echo
    echo '```'
    echo "load average: $(cut -d' ' -f1-3 /proc/loadavg) on $(nproc) cores"
    ps -eo pcpu,comm --sort=-pcpu --no-headers | head -6 | sed 's/^/top cpu: /'
    echo '```'
    echo
    echo "## Promotion recommendation"
    echo
    echo "**Nothing here promotes anything.** \`docs/CANARY.md\` requires all three:"
    echo
    echo "- **Gate A** — \`>=$GATE\` pooled over \`N >= 400\` vs the current champion. This run: $pooled over $n."
    echo "- **Gate B** — no regression vs \`ab-enhanced\` and vs the Go bot, 400 games each at 1 s/action. NOT RUN here."
    echo "- **Gate C** — the live canary soak, >= 20 completed games. NOT RUN here."
    echo
    if [ "$verdict" = GATE-A-PASS ] && [ "$n" -ge 400 ]; then
      echo "Recommendation: Gate A is met at full sample size. Schedule Gate B, then the Gate C soak."
    elif [ "$verdict" = GATE-A-PASS ]; then
      echo "Recommendation: Gate A's threshold is met but only over $n games. Re-run the gauntlet at 400 (\`GATE_INSTANCES=4 trainer/generation.sh --from gauntlet\`) before scheduling Gate B."
    else
      echo "Recommendation: KEEP the current champion. The candidate is kept back, not deleted — \`$CAND\`. A first generation failing its gate is an ordinary outcome; the diagnosis above (row count, holdout curves) is the useful output."
    fi
    echo
    echo "Candidate artifact: \`$CAND\`"
  } > "$REPORT"

  echo
  cat "$REPORT"
  echo "$(date '+%Y-%m-%d %H:%M') gen $GEN: $w-$l-$d of $n pooled=$pooled $verdict" >> "$WORK/history.log"
  [ "$verdict" = GATE-A-PASS ]
}

# ---------------------------------------------------------------- drive
for s in $STAGES; do
  run_stage "$s" || {
    # stage_report's non-zero is the KEEP verdict, not an error.
    [ "$s" = report ] && exit 1
    exit 2
  }
  if [ "$s" = "$UNTIL" ]; then
    echo ">> stopped after [$s] (--until); re-run without --until to continue"
    exit 0
  fi
done
