<!--
Archived verbatim from work/gen6/report.md (gitignored), plus this header.

  champion   artifacts/mcts_champion.json  sha256 476c3f1681755f...  (gen 5)
  candidate  work/gen6/candidate.json      sha256 73522ca6fc2819...  739688 bytes
  rows       work/gen6/selfplay.jsonl      117973 rows / 1000 games, 116 MB

  Harness control, same binary and settings, run before this gauntlet:
  gen-5 champion vs the gen-0 policy prior (artifacts/mcts_policy.json),
  40 games at 192 sims, seed 424242 -> 34-6-0, pooled 0.850. The instrument
  can see a difference that exists.

  The absolute paths below are the executor worktree's; the relative ones
  above are the repo's.
-->

# gen 6 report

_2026-08-14 02:10 UTC · omnibox · 4 cores · load 13.81 15.04 16.40_

## Verdict: **KEPT-BACK**

| | |
|---|---|
| pooled score (W+0.5D)/N | **0.4800** (gate 0.55) |
| candidate W-L-D | 48-52-0 of 100 |
| headline win rate | 48.0% Wilson95 [38.5%, 57.7%] |
| turn-capped (draws) | 0 |
| stalled | 0 |

N=100 is BELOW the Gate A minimum of 400: the interval is wide and a 400-game confirmation is required before any promotion claim.

## What ran

| stage | what it actually did |
|---|---|
| self-play | 1000 games, 192 sims/action, 2 shards, seed 6011, net `champion.json`, 117973 rows, `validate_rows.py` clean |
| train | window=3 (1 generation(s): gen6), epochs=8, 32ch x 4 layers, seed 7, nnue-trainer `train_selfplay.py` UNCHANGED in `vsbot-trainer:cpu`; artifact schema identical to the champion's; loaded by `PolicyValueNet::load` |
| gauntlet | 1 x 100 games = 100 at 192 FIXED sims, colour-paired, per-instance seeds from 60011 spaced 1000, 2 concurrent games, arena lock NOT held |
| rows | 117973 rows from 1000 games |

Per-instance tallies:

```
RESULT w=48 l=52 d=0 n=100 pooled=0.480000 capped=0 stalled=0
```

## Diagnostics

Self-play rows:

```
rows 117973 (diagnostics over 23595, every 5th)
mean legal actions per row   25.0
mean normalised visit entropy 0.679   (1.0 = uniform = no policy signal)
outcomes over 1000 sampled games: p1 578  p2 422  draw 0
```

Holdout metrics from the trainer (offline — ARCHITECTURE.md invariant 7:
these do NOT gate anything, they are here to diagnose the gauntlet result):

```
rows: 117973 from 1000 games (train 108125, holdout 9848), device: cpu
epoch 1/8: train policy 2.4281, value 0.8527 | holdout top-1 56.4%, value MAE 0.810
epoch 2/8: train policy 2.3166, value 0.7709 | holdout top-1 59.1%, value MAE 0.813
epoch 3/8: train policy 2.2983, value 0.7550 | holdout top-1 60.7%, value MAE 0.802
epoch 4/8: train policy 2.2887, value 0.7412 | holdout top-1 60.7%, value MAE 0.789
epoch 5/8: train policy 2.2816, value 0.7299 | holdout top-1 61.2%, value MAE 0.776
epoch 6/8: train policy 2.2773, value 0.7189 | holdout top-1 61.9%, value MAE 0.779
epoch 7/8: train policy 2.2744, value 0.7114 | holdout top-1 60.8%, value MAE 0.783
epoch 8/8: train policy 2.2722, value 0.7014 | holdout top-1 61.5%, value MAE 0.770
```

## Box-load caveat

Fixed sims, not fixed time, so contention changes the wall clock and
not the tally — but it is recorded anyway, because a stalled or capped
count that moves with load would mean the opposite.

```
load average: 13.81 15.04 16.40 on 4 cores
top cpu: 69.5 arena
top cpu: 38.9 batchgauntlet
top cpu: 33.3 rustdoc
top cpu: 29.3 selfplay
top cpu: 29.0 selfplay
top cpu: 28.7 probe
```

## Promotion recommendation

**Nothing here promotes anything.** `docs/CANARY.md` requires all three:

- **Gate A** — `>=0.55` pooled over `N >= 400` vs the current champion. This run: 0.4800 over 100.
- **Gate B** — no regression vs `ab-enhanced` and vs the Go bot, 400 games each at 1 s/action. NOT RUN here.
- **Gate C** — the live canary soak, >= 20 completed games. NOT RUN here.

Recommendation: KEEP the current champion. The candidate is kept back, not deleted — `/home/devbox/Project/vsbot/.claude/worktrees/agent-a982e020ae9e98f24/work/gen6/candidate.json`. A first generation failing its gate is an ordinary outcome; the diagnosis above (row count, holdout curves) is the useful output.

Candidate artifact: `/home/devbox/Project/vsbot/.claude/worktrees/agent-a982e020ae9e98f24/work/gen6/candidate.json`
