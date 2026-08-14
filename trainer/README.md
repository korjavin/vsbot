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
| `generation.sh` | **one RL generation**: selfplay → train → gauntlet → report, stamped and resumable. |
| `netgauntlet/` | a candidate-vs-champion (two-net) gauntlet, which `arena` cannot run. See below. |
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

## A full local generation: `generation.sh`

```bash
export PATH="$HOME/.cargo/bin:$PATH"
cargo build --release -p virus-selfplay --jobs 2
(cd trainer/netgauntlet && cargo build --release --jobs 2)

GEN=6 trainer/generation.sh                 # the whole thing
GEN=6 trainer/generation.sh --until train   # stop before the gauntlet
GEN=6 trainer/generation.sh --from train    # redo training and everything after
```

Four stages — `selfplay train gauntlet report` — ported in spirit from the Java
reference `~/Project/nnue-trainer/scripts/mcts_selfplay_gen.sh`, which the
header comment cites stage for stage. Everything lands in `work/gen<N>/`
(gitignored). Each finished stage drops `work/gen<N>/<stage>.done` and is
skipped on a re-run, so an interrupted run resumes by re-invoking the same
command and a deliberate redo is `--from <stage>`. `report` always re-runs; it
is the file a human re-reads.

Exit codes follow the Java script: **0** the pooled score met the gate at the
sample size played, **1** KEPT BACK, **2** a stage failed. `1` is an outcome,
not a bug.

Three differences from the Java script, all deliberate:

* **No `curriculum` stage.** It needs the prod `games.db`, which is not on this
  box. The window logic that would consume it is already in the trainer call
  site (see below), so adding the stage later changes one function.
* **No auto-promotion.** Java copies a passing candidate over `champion.json`.
  This does not, ever: `docs/CANARY.md` promotes on Gate A **and** Gate B **and**
  the Gate C live soak, and Gate A itself needs `N >= 400`. A local generation is
  one third of one of those. It reports; the canary pipeline promotes.
* **Gauntlet instances run sequentially.** Java forks four JVMs at once on a
  bigger machine. Here four concurrent instances would oversubscribe four
  shared cores; fixed sims means sequencing costs wall clock and changes no
  number.

Knobs are environment variables, defaulted for this box — `SELFPLAY_GAMES=1000`,
`SIMS=192`, `SHARDS=2`, `EPOCHS=8`, `CHANNELS=32`, `LAYERS=4`, `WINDOW=3`,
`GATE_GAMES=100`, `GATE_SIMS=192`, `GATE=0.55`, `SEED_BASE=11`, `JOBS=2`. Run
`trainer/generation.sh --help` for the full list and what each one costs.

`CHANNELS=32 LAYERS=4` is the load-bearing one: that is the **champion's**
geometry and the trainer's own default. `roundtrip.sh`'s `8x2` is a schema test
(it matches the vendored tiny fixture on purpose); training a ladder candidate
at `8x2` would produce a net that validates, loads, gauntlets, and is
structurally incapable of beating gen-5.

Java defaults that produced gen-5, for reference: `GAMES=192`, `SIMS=256`,
`GATE_GAMES=100 × GATE_INSTANCES=4`, `GATE_SIMS=256`, `EPOCHS=8`, `WINDOW=3`,
`SEED_BASE=11`.

### `netgauntlet/` — why a two-net gauntlet lives here

`arena` refuses `--a mcts:X --b mcts:Y` when `X != Y`, and that refusal is
correct: `virus_arena::gauntlet::run` shares one loaded `PolicyValueNet` across
every game and thread, so honouring two paths needs a second net threaded
through `engine::build`'s call sites. Silently playing one artifact against
itself would report a tidy 50/50 for a comparison that never happened.
`docs/CANARY.md` records the gap under "Harness status".

A generation's gate is exactly that comparison, so `trainer/netgauntlet` does
it — by *reusing* arena rather than reimplementing it. The pairing RNG
(`virus_arena::rng`), the side construction (`virus_arena::engine::build`, i.e.
arena's own `MctsSide` with `ValueSource::Net` and `Config::play()`) and the
pooled/Wilson arithmetic (`virus_arena::stats`) are all imported. Only the game
loop is local, and only because it holds two nets instead of one.

It is its own cargo workspace (`[workspace]` in its `Cargo.toml`), so it is not
built by `cargo build --workspace`, not linted by the `-D warnings` gate, and
not shipped. **The proper end state is `--a-net`/`--b-net` inside `virus-arena`,
at which point this deletes and `generation.sh` calls `arena`.** That is a
follow-up bead.

```bash
trainer/netgauntlet/target/release/netgauntlet \
    --a-net work/gen6/candidate.json --b-net work/champion.json \
    --games 100 --sims 192 --seed 60011 --jobs 2
```

Its last line is machine-readable, which is how `stage_report` pools instances:

```text
RESULT w=<W> l=<L> d=<D> n=<N> pooled=<(W+0.5D)/N> capped=<C> stalled=<S>
```

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

**gen 6 is a degenerate case, and says so in its report.** It is the first
*Rust* generation: gens 1–5 were emitted by the Java pipeline and their
`selfplay.jsonl` files live in that repo's work tree, not here. So the window
collapses to gen 6's own rows — the exact single-generation overfitting case
`WINDOW=3` exists to damp. From gen 7 it fills naturally (gen 7 pools 6+7, gen 8
pools 6+7+8, gen 9 drops gen 6) with no change to `generation.sh`, because the
window is over *generations that have a `selfplay.jsonl`*, not over the last
three directories. Pointing gen 7 at the Java rows (copy them to
`work/gen4/selfplay.jsonl` etc.) would fill it a generation earlier; whether the
two emitters' rows are close enough to pool is a question for the bead that
tries it, not an assumption to bake in here.

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
