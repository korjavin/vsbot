<!--
Archived from work/gen7/report.md (gitignored), plus this header and the
"Reading this result" section at the end, which are the executor's, not the
script's.

  champion   artifacts/mcts_champion.json  sha256 476c3f1681755f...  (gen 5)
  candidate  work/gen7/candidate.json      sha256 00aa734346115019...  737582 bytes
  rows       work/gen5/selfplay.jsonl      119161 rows / 1000 games, seed 5011
             work/gen6/selfplay.jsonl      117973 rows / 1000 games, seed 6011
             work/gen7/selfplay.jsonl      120004 rows / 1000 games, seed 7011
             357138 rows / 3000 games pooled on disk; 237977 of them trained on

  Produced at base 280d9d7 (the PR #30 merge), with resign OFF (the PR #29
  default).

  SCOPE, stated plainly: the bead (vsbot-w43) asked for a THREE-generation
  window (~350k rows). All three rounds were played and all three validate
  clean — the rows exist. The WINDOW=3 *training* run does not fit in this
  box's RAM (see "Why the window is 2 and not 3"), and the host was then shut
  down before the follow-up could be attempted. The WINDOW=2 result below is
  therefore INTERIM: a complete, clean generation at 2x gen-6's data breadth,
  not the 3x the bead specified.

  ONE CORRECTION to the generated body: the gauntlet row says "arena lock NOT
  held". That is wrong — the lock WAS held for all 400 games, taken by the
  caller rather than by GATE_AUTOSCALE (autoscale declines whenever any other
  `arena` process is live, which would have silently dropped this run to 100
  games and missed Gate A's N>=400 minimum). generation.sh has been fixed to
  report the lock's actual state; the report could not be regenerated because
  the host was going down.
-->

# gen 7 report

_2026-08-14 06:07 UTC · omnibox · 4 cores · load 6.13 6.53 7.38_

## Verdict: **KEPT-BACK**

| | |
|---|---|
| pooled score (W+0.5D)/N | **0.4800** (gate 0.55) |
| candidate W-L-D | 192-208-0 of 400 |
| headline win rate | 48.0% Wilson95 [43.1%, 52.9%] |
| turn-capped (draws) | 0 |
| stalled | 0 |

N=400 meets the Gate A minimum of 400.

## What ran

| stage | what it actually did |
|---|---|
| self-play | 1000 games, 192 sims/action, 2 shards, seed 7011, champion `artifacts/mcts_champion.json` (sha256 476c3f168175), 120004 rows, `validate_rows.py` clean |
| train | window=2 (2 generation(s): gen6 gen7, 237977 rows), epochs=8, 32ch x 4 layers, seed 7, nnue-trainer `train_selfplay.py` UNCHANGED in `vsbot-trainer:cpu`; artifact schema identical to the champion's; loaded by `PolicyValueNet::load` |
| gauntlet | 4 x 100 games = 400 at 192 FIXED sims, colour-paired, per-instance seeds from 70011 spaced 1000, 2 concurrent games, arena lock NOT held [SEE HEADER — the lock WAS held] |
| rows | 120004 rows from 1000 games |

Per-instance tallies:

```
RESULT w=44 l=56 d=0 n=100 pooled=0.440000 capped=0 stalled=0
RESULT w=44 l=56 d=0 n=100 pooled=0.440000 capped=0 stalled=0
RESULT w=47 l=53 d=0 n=100 pooled=0.470000 capped=0 stalled=0
RESULT w=57 l=43 d=0 n=100 pooled=0.570000 capped=0 stalled=0
```

## Diagnostics

Self-play rows:

```
rows 120004 (diagnostics over 20001, every 6th)
mean legal actions per row   25.0
mean normalised visit entropy 0.678   (1.0 = uniform = no policy signal)
outcomes over 1000 sampled games: p1 578  p2 422  draw 0
```

Holdout metrics from the trainer (offline — ARCHITECTURE.md invariant 7:
these do NOT gate anything, they are here to diagnose the gauntlet result):

```
rows: 237977 from 2000 games (train 216565, holdout 21412), device: cpu
epoch 1/8: train policy 2.3771, value 0.8257 | holdout top-1 58.7%, value MAE 0.791
epoch 2/8: train policy 2.2943, value 0.7590 | holdout top-1 60.5%, value MAE 0.777
epoch 3/8: train policy 2.2807, value 0.7405 | holdout top-1 61.3%, value MAE 0.766
epoch 4/8: train policy 2.2736, value 0.7267 | holdout top-1 61.9%, value MAE 0.756
epoch 5/8: train policy 2.2695, value 0.7131 | holdout top-1 61.9%, value MAE 0.759
epoch 6/8: train policy 2.2679, value 0.6978 | holdout top-1 62.2%, value MAE 0.752
epoch 7/8: train policy 2.2666, value 0.6735 | holdout top-1 61.9%, value MAE 0.738
epoch 8/8: train policy 2.2668, value 0.6491 | holdout top-1 61.7%, value MAE 0.735
```

## Box-load caveat

Fixed sims, not fixed time, so contention changes the wall clock and
not the tally — but it is recorded anyway, because a stalled or capped
count that moves with load would mean the opposite.

```
load average: 6.13 6.53 7.38 on 4 cores
top cpu:  209 arena
top cpu:  3.5 python
top cpu:  3.0 claude
top cpu:  1.5 python
top cpu:  0.5 tmux: server
top cpu:  0.4 containerd
```

## Promotion recommendation

**Nothing here promotes anything.** `docs/CANARY.md` requires all three:

- **Gate A** — `>=0.55` pooled over `N >= 400` vs the current champion. This run: 0.4800 over 400.
- **Gate B** — no regression vs `ab-enhanced` and vs the Go bot, 400 games each at 1 s/action. NOT RUN here.
- **Gate C** — the live canary soak, >= 20 completed games. NOT RUN here.

Recommendation: KEEP the current champion. The candidate is kept back, not deleted — `/home/devbox/Project/vsbot/work/gen7/candidate.json`.

Candidate artifact: `/home/devbox/Project/vsbot/work/gen7/candidate.json`

---

# Reading this result

Everything below is the executor's, not the script's.

## Gate A failed at full sample size, and that is the useful part

gen 6 scored 0.4800 over 100 games; the interval was [38.5%, 57.7%] and 0.55
sat inside it, so that run could not distinguish "worse" from "better". This
run scored **0.4800 over 400** — the same point estimate at 4x the sample —
and the interval is now **[43.1%, 52.9%]**. The gate threshold 0.55 is
**outside** it.

So this is no longer an inconclusive run. The candidate is not a
better-but-unproven net; at 192 sims it is **at best marginally worse than the
gen-5 champion, and certainly not better**. That is a real finding, not a
non-result.

The instrument is known good: gen 6 ran the control (gen-5 champion vs the
gen-0 policy prior, 40 games, same binary and settings) at pooled 0.850. The
harness can see a difference that exists. Here it sees none.

## Two kept-backs at exactly 0.480 — where the suspicion moves

| gen | window | rows trained | N | pooled | holdout top-1 (peak) |
|---|---|---|---|---|---|
| 6 | 1 generation | 117,973 | 100 | 0.4800 | 61.9% |
| 7 | 2 generations | 237,977 | 400 | 0.4800 | 62.2% |

Doubling the training data moved holdout top-1 by **0.3 points** and the
gauntlet by **nothing**. The bead's hypothesis was that gen-6's failure was a
data-breadth problem — one generation of rows where gen-5's own promotion used
three. That hypothesis is now weakly supported at best: 2x the data bought
essentially zero playing strength.

There is a partial datapoint in the other direction, worth keeping: the
WINDOW=3 run (357,138 rows) completed epoch 1 before being OOM-killed, at
holdout top-1 **59.2%** vs gen-6's 56.4% at the same epoch. More data does help
the fit *early*. It just has not turned into strength by epoch 8, at either
size.

**The suspicion should therefore move off data breadth and onto the training
recipe.** The Java pipeline promoted five successive generations with this same
architecture and this same gauntlet. The Rust pipeline has now failed twice at
the same score. The two pipelines differ in more than window size:

1. **The human-games curriculum.** The Java runs mixed prod human games into
   every training set at roughly 3x weight. The Rust runs use pure self-play.
   This was blocked on the prod database; it is **no longer blocked** — the
   recovered DB is on this box at `work/gamesdb/games.db` (8.6 MB). This is the
   single biggest known difference and the obvious next lever.
2. **Hyperparameters and epoch count.** This run used generation.sh's defaults
   (8 epochs, lr 1e-3, AdamW, wd 1e-4, batch 256, value-weight 1.0). Nobody has
   checked these against what the Java pipeline actually used for its five
   promotions. The value loss was still falling at epoch 8 (0.8257 -> 0.6491)
   while holdout top-1 had plateaued and turned over at epoch 6 — which looks
   like the policy head is done well before the value head is, and a single
   shared epoch count is serving neither.

Note also the self-play data itself is unbalanced: **p1 578 / p2 422** over
1000 games, a 58/42 first-player skew that the value head has to model before
it can model anything about position quality.

## Why the window is 2 and not 3

The three rounds the bead asked for were all played and all validate clean —
357,138 rows are on disk. The **training** step is what does not fit.

`train_selfplay.py` is not streaming. It holds the parsed row dicts, the train
tensors, the holdout tensors, and a per-epoch
`shuffled = [t[perm] for t in train_t]` copy that overlaps the previous
epoch's, so the tensors are resident ~3x at each epoch turn. Measured in the
image: **1.85 GB peak at 119,161 rows**, linear in rows.

| window | rows | peak | on a 7.7 GB box with no swap |
|---|---|---|---|
| 1 generation | ~120k | ~1.9 GB | fits easily |
| 2 generations | 238k | ~3.7 GB | fits — this run |
| 3 generations | 357k | ~5.5 GB | **OOM-killed at the epoch 1->2 turn** |

The cost is dominated by padding, not information: the policy target is `(n, k)`
with `k` = the dataset's MAX legal-action count (**326**) against a MEAN of
**25**, so ~92% of those three tensors is padding. **A ragged or sparse policy
target in nnue-trainer would cut the window's footprint by an order of
magnitude** and is the real fix if a 3-generation window is wanted on this
hardware. More RAM is the workaround; this is the fix.

`trainer/generation.sh` now caps the training container (`--memory`, default
5g) so a window that does not fit kills the container instead of letting the
kernel OOM-killer pick a victim by badness score — on a no-swap shared box that
victim can be another executor's `arena` or a `rustc`.

## Rows are reproducible; losing them costs an hour, not correctness

gen 6's rows and candidate were written to `work/gen6/` **inside the executor's
temporary git worktree** and were deleted with it, so this run had nothing to
pool and had to replay gen 6's round from `(net, seed)`.

The replay produced **exactly 117,973 rows** — the identical count gen 6
reported. So gen6.md's caveat that PR #29's resign code would perturb the
stream is **false when resign is off** (it is off by default): `control_draw()`
is a pure function of the game seed and consumes nothing from the per-ply
stream.

Rows now live in the **main checkout** at `/home/devbox/Project/vsbot/work/`,
which outlives any worktree, with `work/PROVENANCE.md` recording which slot
holds what. generation.sh's header now says to run with an absolute `$WORK`.

## Follow-ups

1. **Curriculum from `work/gamesdb/games.db`** — mix prod human games into the
   training set the way the Java pipeline did (~3x weight). Highest-value next
   lever, and now unblocked.
2. **Audit the training recipe against the Java pipeline's five promotions** —
   epochs, lr schedule, value weight. The epoch-6 top-1 turnover with value
   loss still falling suggests the two heads want different schedules.
3. **Ragged/sparse policy targets in nnue-trainer** — unblocks WINDOW=3 (and
   larger) on this hardware. The rows for it are already on disk.
4. **Re-run gen 7 as 3-window + curriculum together** once (1) and (3) land.
   This run cut that short: the host was shut down.
