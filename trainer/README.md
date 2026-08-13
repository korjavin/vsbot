# trainer/ — running the nnue-trainer python trainers against Rust self-play

This directory is the **trainer side** of S4 (`docs/plans/superiority.md` §5, bead
`vsbot-ml6`). It contains no trainer code and never will.

The design constraint, stated once because everything here follows from it:

> **Rust replaces the Java *emitter*, not the trainer.**
> `python/mcts/train_selfplay.py` in `~/Project/nnue-trainer` stays byte-identical.
> It already consumes any JSONL in the `SelfPlayMcts` row schema, so a Rust
> generator that emits that schema plugs into the existing pipeline with zero
> trainer changes — and we keep the ability to diff a Rust generation against a
> Java one on the same trainer.

So this directory holds: a docker image that can *run* that trainer, validators
for the two contracts either side of it, and one script that proves the whole
chain end to end.

| file | what it is |
|---|---|
| `Dockerfile` | python 3.12 + CPU torch. **No trainer code inside** — the checkout is bind-mounted. |
| `roundtrip.sh` | rows → train → artifact → Rust load, in one command. The proof. |
| `rows_schema.py` | the `SelfPlayMcts` row contract + flat-action-id codec, in one place. |
| `validate_rows.py` | strict checker for emitter output. **Point the Rust emitter at this.** |
| `validate_artifact.py` | structural check of a trained artifact against what `virus-mcts` loads. |
| `make_reference_rows.py` | contract-valid *stand-in* rows, so the pipeline runs before the emitter exists. |

## Path assumption

Everything here assumes the trainer checkout lives at:

```
~/Project/nnue-trainer
```

Override with `TRAINER_HOME=/path/to/nnue-trainer`. It is mounted **read-only**
at `/trainer` inside the container, which is both a safety property (the trainer
cannot be mutated by a training run) and a statement of intent: if a change is
ever genuinely needed there, it gets made and committed in *that* repo, not
patched from this one.

Only two files in that checkout are load-bearing for us:

* `python/mcts/train_selfplay.py` — the trainer we run.
* `src/main/java/.../mcts/SelfPlayMcts.java` — the row-schema authority the Rust
  emitter must match (`row()` and `flatIndex()`).

## Prove the pipeline works: `roundtrip.sh`

```bash
trainer/roundtrip.sh
```

Each step fails loudly:

* **0 preflight** — docker reachable, trainer checkout present, and
  `selftest.py` green (see "checking the checkers" below).
* **1 image** — build `vsbot-trainer:cpu` if absent (first run pulls the CPU
  torch wheel; a few minutes and ~1.4 GB).
* **2 rows** — real rows if `SELFPLAY_JSONL` is set, otherwise synthesised ones.
* **3 contract** — `validate_rows.py` over those rows.
* **4 train** — `train_selfplay.py`, unchanged, in the container. A tiny net
  (8 channels × 2 layers), 3 epochs, thrown away.
* **5 / 5b schema** — `validate_artifact.py` against the champion, then against
  the vendored tiny fixture with `--require-identical`.
* **6 load** — the artifact is loaded by `virus-mcts`' real loader.

Step 4's geometry is not arbitrary: 8×2 is exactly the shape of the vendored
`fixtures/mcts/mcts_selfplay_tiny.json`, so at the defaults the fresh candidate
must come out with a shape signature *identical* to a known-good net — which is
what step 5b asserts. Step 6 runs
`cargo run -p virus-mcts --example batchgauntlet -- --net <candidate>`, which
calls `PolicyValueNet::load` — the exhaustive validator (arch name,
declared-vs-actual shapes, finiteness of every weight) — and then actually
searches with the weights. That example already accepts `--net`, so proving the
round trip requires **no change to `crates/`**.

### Checking the checkers

`python3 trainer/selftest.py` (stdlib only, ~1 s) feeds each validator the
mistakes a schema port really makes — `z` pre-flipped into the mover frame,
unordered pair ids, normalised `pv` instead of raw visits, forced positions
emitted, a `NaN` weight — and asserts each is rejected with a pointed message.
A validator that only ever prints OK is worse than no validator: it converts an
unchecked assumption into a false assurance. `.github/workflows/trainer.yml`
runs this on every PR that touches `trainer/` or the fixtures.

Useful knobs: `SELFPLAY_JSONL=` (real rows), `EPOCHS=`, `CHANNELS=`/`LAYERS=`,
`WORK=`, `SKIP_RUST=1`, `REBUILD=1`, `JOBS=` (cargo/​docker cores — default 2,
because this box is shared).

### Once the Rust emitter lands

The emitter executor's one-command proof of schema compatibility is:

```bash
SELFPLAY_JSONL=work/mcts-rl/gen1/selfplay.jsonl trainer/roundtrip.sh
```

Green means the Rust rows are field-for-field acceptable to the unchanged
trainer, and the net that comes out loads in our inference. That is S4's first
acceptance clause, mechanised.

At that point, delete the `make_reference_rows.py` branch from step 3 — it is
scaffolding, and its rows are synthetic (no game rules are simulated; a net
trained on them learns nothing). Its only purpose is to make every stage
downstream of the emitter runnable *today*, so the emitter is the sole new
variable when it arrives.

## A full local generation, once the emitter exists

The Java reference is `~/Project/nnue-trainer/scripts/mcts_selfplay_gen.sh`
(stages `selfplay curriculum train gauntlet report`, resumable `.done.<stage>`
stamps). The Rust equivalent should keep that contract stage for stage. Below is
what each stage's command line looks like locally at reduced size; `<selfplay>`
is whatever binary/example `virus-mcts` grows.

```bash
export WORK=work/mcts-rl
mkdir -p "$WORK/gen1/logs"
cp artifacts/mcts_champion.json "$WORK/champion.json"   # continue the ladder, don't restart it
```

**1. self-play** — sharded across cores, deterministic per `(seed, shard)`
regardless of shard count, `seed = SEED_BASE + gen*1000` (`SEED_BASE=11`):

```bash
for i in $(seq 0 3); do
  <selfplay> --net "$WORK/champion.json" --out "$WORK/gen1/selfplay_shard_$i.jsonl" \
             --games 24 --sims 128 --shard "$i" --shards 4 --seed 11011 \
             > "$WORK/gen1/logs/selfplay_$i.log" 2>&1 &
done; wait
cat "$WORK"/gen1/selfplay_shard_*.jsonl > "$WORK/gen1/selfplay.jsonl"
python3 trainer/validate_rows.py "$WORK/gen1/selfplay.jsonl"
```

**2. train** — the sliding window, see below:

```bash
docker run --rm --user "$(id -u):$(id -g)" --cpus 4 \
  -v "$HOME/Project/nnue-trainer":/trainer:ro \
  -v "$PWD/$WORK":/work \
  vsbot-trainer:cpu \
  python python/mcts/train_selfplay.py \
    /work/gen1/selfplay.jsonl \
    --out /work/gen1/candidate.json --epochs 8
python3 trainer/validate_artifact.py "$WORK/gen1/candidate.json" \
    --reference artifacts/mcts_champion.json
```

**3. gauntlet** — candidate vs champion at fixed sims, seeds spaced 1000 per
instance, `virus-arena`'s job.

**4. report** — pool the instances, apply the gate, promote or keep:

```bash
# PROMOTE -> cp gen1/candidate.json to champion.json (archive champion_gen1.json)
# KEEP    -> champion unchanged
# either way gen increments, so the next generation gets fresh seeds
```

Java defaults that produced gen-5, for reference: `GAMES=192`, `SIMS=256`,
`GATE_GAMES=100 × GATE_INSTANCES=4`, `GATE_SIMS=256`, `EPOCHS=8`, `WINDOW=3`,
`SEED_BASE=11`. Locally, shrink `--games`/`--sims`; keep everything else.

## The `WINDOW=3` sliding window

Training consumes **the current generation plus the previous two** — not just
the newest rows, and not the whole history:

```
gen 1:  gen1
gen 2:  gen1 gen2
gen 3:  gen1 gen2 gen3
gen 4:        gen2 gen3 gen4      <- gen1 falls out
gen 5:              gen3 gen4 gen5
```

`train_selfplay.py` takes a positional list of datasets and simply concatenates
them, so the window is expressed entirely at the call site:

```bash
DATASETS=""
for g in $(seq $((GEN - WINDOW + 1)) "$GEN"); do
  [ "$g" -ge 1 ] && [ -f "$WORK/gen$g/selfplay.jsonl" ] \
    && DATASETS="$DATASETS /work/gen$g/selfplay.jsonl"
done
# ... python python/mcts/train_selfplay.py $DATASETS --out /work/gen$GEN/candidate.json
```

Why 3: one generation's rows are too few and too correlated (a net trained only
on its own predecessor's games overfits that opponent), while the full history
drags in rows generated by champions several promotions stale. Three is the
Java pipeline's setting and the number the gen-5 ladder was actually produced
with, so changing it is a deliberate experiment, not a default to re-derive.

Two details that are easy to lose in a port:

* The window is over **generations that have a `selfplay.jsonl`**, skipping
  gaps — not over "the last three directories".
* Curriculum rows (if that stage is ever ported) are **current-generation only**
  and are oversampled by repeating the filename `CURRICULUM_REPEAT` times, since
  the trainer has no per-file weighting.

## The promotion gate

One number, one formula, three places it must stay identical:

```
pooled score = (W + 0.5·D) / N        promote iff >= 0.55,  N >= 400
```

`W`/`L`/`D` are the candidate's tally **pooled across gauntlet instances** —
pool first, then divide. Applying the gate per instance and taking a majority is
a different, weaker test.

Where it lives:

* **Rust, authoritative:** `crates/virus-arena/src/stats.rs` —
  `PROMOTION_THRESHOLD = 0.55`, `GATE_MIN_GAMES = 400`, and
  `Record::pooled_score()` computing `(W + 0.5·D)/N`. S4's "port the pooled-gate
  arithmetic into virus-arena" is about the *report/promote stage* consuming
  these; the arithmetic itself is already there.
* **Java reference:** `mcts_selfplay_gen.sh`'s `stage_report` (the `awk` pooling
  + verdict) and `GauntletMctsRun.promote`, which is JUnit-pinned.

`GATE_MIN_GAMES = 400` is not decoration: at `p ≈ 0.5` the standard error is
`0.5/√N`, so 400 games puts the 0.55 threshold at about 2σ. A 100-game gauntlet
that "passes 0.55" has a standard error of ±5 points and has demonstrated
nothing. Reduced-size *local* generations are for exercising the plumbing; a
promotion claim needs the full 400.

## Notes

* The image pins `torch==2.13.0` / `numpy==2.5.1`, matching
  `nnue-trainer/Dockerfile.trainer`, so a net trained here is the net the server
  sidecar would produce. That reference image also carries a JDK and a primed
  `~/.m2` because it runs the Java loops; ours does not, because that half is
  Rust's job now.
* Containers run with `--user $(id -u):$(id -g)` so artifacts land owned by you,
  not root.
* `--cpus` is set on every `docker run` here. The devbox runs arena measurements
  concurrently; torch will otherwise take every core.
