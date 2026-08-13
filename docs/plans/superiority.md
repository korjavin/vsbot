# Superiority design — beating the gen-5 champion, the Go bot, and the owner

Bead `vsbot-12o` (parent epic `vsbot-t3q`). Status: **design — code lands in the follow-up
beads of §5.** Depends on `vsbot-tfa` (virus-mcts) and `vsbot-e7h` (virus-arena).

Convention used throughout: win-rate → Elo via `Elo = 400·log10(p/(1−p))`
(55% → +35, 55.5% → +38, 62% → +85, 65.75% → +113, 69% → +139). "Feasibility doc" =
`~/Project/nnue-trainer/docs/plans/20260807-mcts-az-feasibility.md`; "deep-labels doc" =
`~/Project/nnue-trainer/docs/nnue-v3-deep-labels.md`; "server doc" =
`~/Project/nnue-trainer/docs/server-training.md`; "canary doc" =
`~/Project/virusgame/docs/nnue-canary.md`; "gen script" =
`~/Project/nnue-trainer/scripts/mcts_selfplay_gen.sh`. Numbers without a citation do not
exist in this document.

## 0. Verdict

The path is **not** a new paradigm. It is: (1) take the proven gen-5 MCTS champion, (2)
multiply its compute by everything the predecessors left on the table — Rust throughput,
the 120 s/move server budget both predecessors ignored (they play a flat 1 s:
`GoBotSearcher.PRODUCTION_BUDGET_MILLIS`, cited at
`20260807-search-strength.md:26-27`; `MCTS_MOVE_MILLIS=1000` default, server doc:117),
pondering, and parallel/batched MCTS — and (3) keep turning the RL crank with bigger,
better-targeted generations feeding the **existing** python trainer unchanged. The owner
remains the only promotion judge (canary doc:9-16); every gauntlet gate below is a sanity
floor, not a strength proof.

The single most dangerous failure mode is inventing new offline metrics or new schemas.
Seven offline-metric/strength disconnects are documented (deep-labels doc:40-44;
`ARCHITECTURE.md:45` invariant 7), and the working trainer/schema is the product of five
consecutive promoted generations. We reuse both verbatim.

## 1. Baselines — what "strictly stronger" means

| opponent | what it is | current standing | source |
|---|---|---|---|
| gen-5 Java champion | PUCT + conv 4×32 policy/value net (`meta: planes 13, channels 32, layers 4`), 1 s/move, prod since 2026-08-09 | the bar; promoted 5 consecutive times at 55.5–69 %/gen | `artifacts/mcts_champion.json` meta; bd `nnue-trainer-1jh.3` notes (gen-5: 222-178 = 55.5 % vs gen-4); range per bd `vsbot-12o`; gen-1 = 65.75 %/400 (deep-labels doc:144-146) |
| hand-tuned enhanced alpha-beta ("the bar") | Java `GoBotSearcher` + H2 stack (staged movegen, packed TT, killers/history, lazy SMP, salvage) | 62.0 % ± 2.4 over its own pre-overhaul baseline at equal time/400 games; dethroned by the RL champion | bd `nnue-trainer-1jh.2` close reason; bd `nnue-trainer-1jh` notes |
| Go bot | `~/Project/virusgame/backend/search`, single-threaded alpha-beta, 1 s/move | historically beaten by the Java clone lineage (6-0 with a search-depth edge, `20260807-search-strength.md:5-7`); must never regress vs it | `ARCHITECTURE.md:3` |
| the owner (human) | plays on `vs.wandergeek.org`; explicit target opponent | beat the scripted-gate-dominant vs-ai2.52 3 straight by turtling — the Goodhart lesson | canary doc:9-12 |

Game/search constants everything below builds on: ~34 legal actions/position, 47.1 % of
edges flip the mover (feasibility doc:29-37); ~55 recorded actions/game (feasibility
doc:44-45); server timer = **120 s per move with auto-resign**
(`~/Project/virusgame/backend/hub.go:2590-2604`, `backend/types.go:206`); illegal move =
instant forfeit (`ARCHITECTURE.md:35`).

## 2. Ranked levers (expected Elo / effort)

Ranking is marginal-Elo per marginal-effort, grounded where possible in measured history.
Two caveats apply to every compute-shaped lever (a, b, d):

- **The sims→Elo curve for this game is unmeasured.** Literature puts self-play MCTS at
  ~50–120 Elo per compute doubling, decaying (literature estimate, not a repo number).
  Phase S2-T2 measures the real curve; downstream numbers re-derive from it.
- **Compute converts only when the evaluator isn't the ceiling.** The repo has both
  datapoints: a pure search edge went 6-0 (`20260807-search-strength.md:5-7`), but with a
  bad eval, 4× time bought exactly zero wins (bd `nnue-trainer-raz.2.5` close: 0-4 across
  depths 8-12), and the NNUE leaf's 5× speed converted to nothing (bd `nnue-trainer-1jh.1`
  close: 28.5 % at equal time, identical to fixed depth). The RL value net improves each
  generation precisely on search-visited positions (deep-labels doc:109-121), so for MCTS
  the compute and RL levers compound instead of cancelling — but the deep-labels wall
  (44.3 % at label depth → 9.0 % one ply deeper, deep-labels doc:66-71) is the standing
  warning to *measure* strength at the new budget, never assume it.

| rank | lever | expected Elo | effort | evidence anchor |
|---|---|---|---|---|
| 1 | (a) Rust throughput at 1 s/move | +75…+250 vs gen-5 Java (1.5–3.3 doublings if Rust lands 3–10×/core; measure in S0) | ~0 marginal (funded by `vsbot-tfa`) | 4.5 M MACs/eval, Java est. 0.5–2 ms/eval → 500–2000 sims/move (feasibility doc:135-139, 149-153) |
| 2 | (b) 120 s/move + time manager + ponder | +150…+400 vs 1 s self (up to 6.9 doublings; decaying curve, S2-T2 measures) | small | timer: `hub.go:2590-2604`; both predecessors flat 1 s (see §0); ponder feasibility §2b |
| 3 | (c) RL ladder in Rust | +38…+113 per promoted generation, compounding | medium | measured per-gen gains: 65.75 % gen-1 → 55.5 % gen-5 (deep-labels doc:144-146; bd `nnue-trainer-1jh.3`) |
| 4 | (d) MCTS engineering the Java v1 skipped | 2–4× more sims at equal time (≈ +100…+200 via the same curve) **and** bigger/better generations | medium-high | skipped list: feasibility doc:318-324; lazy-SMP analogue measured 1.6–2.5× (`20260807-search-strength.md:226-227`) |
| 5 | (f) human-games curriculum, aimed at the owner | unquantifiable by gauntlet **by design** — owner-canary judged | small (exists; extend) | gen-5 = first curriculum-trained gen, promoted (bd `nnue-trainer-1jh.3` notes); server doc:63-84 |
| 6 | (e) net scaling beyond 4×32 | unknown sign at fixed time (bigger net = fewer sims); gate-decided | medium | cost math §2e; trainer already parametric (`train_selfplay.py:195-196`) |
| — | parked: AB+MCTS hybrid/ensemble (bd `vsbot-12o` lever 4) | unknown | high | no measurement asks for it yet; `artifacts/ordering_policy.json` stays vendored for the AB stack (`vsbot-6me`), revisit only if MCTS stalls vs the AB bar |

### 2a. Raw Rust throughput at 1 s/move

The gen-5 net is ~40 k params, ~4.5 M MACs/eval; Java hand-rolled float inference was
estimated 0.5–2 ms/eval single-core, making 1 s/move worth ~500–2000 sims single-threaded
(feasibility doc:135-153 — estimates; no measured Java evals/s was ever published, which
is itself a reason S0 exists). A sim is net-dominated: path `apply` ≈ 2–3 µs (feasibility
doc:150-151). Rust with SIMD-friendly f32 conv (3×3 on 12×12, layout chosen for
autovectorization) plus leaf batching should land 3–10× Java per core — **working
assumption, not a number; S0 measures it and every downstream claim re-derives.**

This lever is free at the margin: `vsbot-tfa` builds the searcher and inference anyway,
and parity fixtures (`fixtures/mcts/mcts_policy_parity.json`, `mcts_value_parity.json`)
pin correctness. The first gauntlet that matters (S1) is Rust-gen-5 vs Java-gen-5, both at
1 s/move: same net, more sims — if that isn't ≥ 0.55 pooled over 400, the throughput
didn't materialize and ranks 2–4 all shrink.

### 2b. The 120 s budget: time management + pondering

**Clock model.** The server arms a fresh 120 s auto-resign timer per move
(`hub.go:2590-2604`) — there is **no banked clock**. So "time management" here is not
chess clock allocation; it is: pick a per-action budget ≤ a safety margin under 120 s,
spend it where it matters, and never time out or move illegally (both are instant losses:
`types.go:206`, `ARCHITECTURE.md:35`).

Design:

- `VSBOT_MOVE_MILLIS` env (read in the `vsbot` bin per `CLAUDE.md` config convention),
  default 1000 (parity with predecessors), raised deliberately per opponent profile. Hard
  ceiling ~100 s: leaves ≥ 20 s for WS RTT, snapshot re-validation, and salvage.
- **Fallback-first discipline**: select a legal fallback (prior argmax over the legal
  mask) before the long search starts; any deadline overrun answers with it. This is the
  MCTS analogue of invariant 3 (`ARCHITECTURE.md:41`).
- **Early stop / extension**: stop when the visit leader cannot be overtaken in the
  remaining budget (visit-lead stopping — saves human-facing latency at zero Elo cost);
  extend toward the ceiling when the root is unstable (leader changes, top-2 visit gap
  small). Both are standard MCTS time-manager rules; measured, not assumed, in S2.
- **Budget asymmetry is legitimate.** Gauntlets and the RL gate stay at fixed sims/time
  (protocol of §4); the *live* bot exploiting 120 s against opponents playing 1 s is the
  point of the exercise, not an experimental confound.

**Pondering is protocol-feasible today, no server cooperation needed.** The server
streams a snapshot on every action, including the opponent's: `move_made` carries a
snapshot and is handled even when `player != myPlayerIndex`
(`~/Project/nnue-trainer/.../protocol/GameLoopHandler.java:57-64`), as do opponent
`neutrals_placed` (:91-93) and `turn_change` (:94-102). A pondering MCTS searches the
current opponent-to-move state, re-roots into the matched child on each opponent action
(tree reuse — free with per-action nodes), and on our `turn_change` continues the same
tree for our budget. Two hard guards, both existing invariants: never emit a move except
off the authoritative turn driver with `currentPlayer == us` (invariant 2,
`ARCHITECTURE.md:40`; the 2026-08-08 double-forfeit at `GameLoopHandler.java:86-90` is
the cautionary tale), and version-gated cancellation on every new snapshot (invariant 5,
`ARCHITECTURE.md:43`).

Expected value: vs bots that answer in ~1–3 s, ponder ≈ ≤1 extra doubling. Vs the owner —
who thinks for tens of seconds to minutes — ponder multiplies effective compute severalfold
on exactly the target opponent. Cost: CPU on the prod host is shared with the nightly
trainer window (server doc:33, :129-138); ponder threads must be capped and optionally
disabled during the training window.

### 2c. Continue the RL ladder with Rust self-play

**Keep the schema and the trainer. Do not invent.** The row schema is
`{"g","sym"[144, mover-relative],"ml":1..3,"nuo":0|1,"nux":0|1,"mover":1|2,"pi":[flat
action ids],"pv":[root visit counts],"z":-1|0|1 in the ABSOLUTE frame}`
(`SelfPlayMcts.java:29-35`); the trainer performs the one mover flip itself
(`train_selfplay.py:52-54`, the pinned v3 lesson). `train_selfplay.py` consumes any JSONL
in this schema — a Rust generator that emits it plugs into the existing pipeline with
zero trainer changes.

The pipeline to preserve, stage for stage (gen script:138 `selfplay curriculum train
gauntlet report`, resumable stamps): self-play sharded across cores with the champion
artifact → optional human-games curriculum → sliding-window training (last `WINDOW=3`
generations, gen script:100, :220-232) → 4×100-game candidate-vs-champion gauntlet at
fixed `GATE_SIMS=256` with per-instance seeds spaced 1000 (gen script:95-98, :246-249) →
pooled `(W+0.5D)/N ≥ GATE=0.55` promotion arithmetic (gen script:273-278). Defaults that
produced gen-5: `GAMES=192`-per-shard-set locally / 1000 nightly, `SIMS=192-256`,
`EPOCHS=8`, `SEED_BASE=11` (gen script:93-101; server doc:25).

What Rust changes: **generation size and target quality at the same nightly window.** The
measured reference is 1000 games at 192 sims ≈ 3.5–5 h self-play inside the 5–7 h window
(server doc:129-138). A k× Rust speedup is spent on some mix of (i) more games/gen (less
overfit per window, more diversity), (ii) more sims/move in self-play (better `pv`
targets — the AZ-standard is ~800 sims vs our 192), and (iii) more gauntlet games
(tighter gates). The per-generation gain decayed from 65.75 % (gen-1) to 55.5 % (gen-5) at
constant compute; better targets and bigger gens are the standard remedy and the cheap
one, because they reuse everything.

Why the ladder keeps paying: the deep-labels post-mortem establishes that training must
happen on search-visited positions and that fixed-depth labels are a treadmill
(deep-labels doc:109-121). Self-play RL dissolves that by construction, and it is the
only lever in this project's history with five consecutive out-of-noise promotions.

### 2d. MCTS engineering the Java v1 deliberately skipped

The Java searcher shipped with an explicit deferral list: "Leaf-eval batching, virtual
loss / tree parallelism, DAG transpositions via `GoState.hash()`, Gumbel/
sequential-halving root selection, resign thresholds" (feasibility doc:318-324) — each
"a known upgrade with a trigger". The triggers have now fired (we want 120 s budgets and
bigger generations). In dependency order:

1. **Leaf batching + in-tree parallelism with virtual loss.** Run B simulations'
   selection phases concurrently (virtual loss decorrelates paths), evaluate leaves as
   one batched forward pass. Batching is what makes SIMD conv inference pay; it also
   feeds (2). Expected 2–4× sims/s combined with threading (the alpha-beta analogue,
   lazy SMP, measured 1.6–2.5× effective at 6-8 threads, `20260807-search-strength.md:226-227`;
   MCTS tree parallelism scales at least as well at these widths — literature, verify by
   gauntlet). Absolute-frame backup (invariant 1, `ARCHITECTURE.md:39`) is
   parallelism-safe by construction: `W` accumulates in one frame, no per-edge negation.
2. **DAG transpositions.** Within-turn action permutations reach identical states — the
   Java TT audit proved these collisions are real and already caught by the full state
   key (`20260807-search-strength.md:110-112`); an MCTS DAG merges them instead of
   duplicating subtrees. Key must include movesLeft/neutralUsed/side (invariant 6,
   `ARCHITECTURE.md:44`; `GoState.hash()` precedent, feasibility doc:41-43). Payoff:
   sim savings + memory (which binds at 120 s budgets — a 100 k-sim tree with ~34-wide
   nodes is ~10⁶ edges; at long ponder budgets node pooling + re-rooting + DAG are what
   keep RSS bounded).
3. **Gumbel root selection for self-play.** At 192–256 sims, Gumbel/sequential-halving
   produces better policy targets and stronger move selection than PUCT+Dirichlet at
   equal sims (literature — the exact regime we generate in). Self-play only at first;
   production stays PUCT until a fixed-sims gauntlet says otherwise.
4. **Resign threshold for self-play throughput.** Games average ~55 actions (feasibility
   doc:44-45) and self-play budgets 60–100 plies; resign when `v_abs` stays beyond a
   threshold (e.g. |v| > 0.95) for k consecutive plies, with a 10 % no-resign control to
   measure the false-resign rate (<5 % rule — AZ standard). Saves the played-out lost
   tails; pure generation-size lever.

### 2e. Net scaling — after throughput headroom, gate-decided

Gen-5 is 4 layers × 32 channels, ~31 k trunk params, ~4.5 M MACs (feasibility doc:135-136;
`artifacts/mcts_champion.json` meta). Conv cost scales ≈ layers × channels² (per-layer
MACs = 9·C²·144): 4×48 ≈ 2.25×, 6×64 ≈ 7× the eval cost — i.e. 6×64 must be worth ~2.8
sims-doublings of strength at fixed wall-clock to break even. That is why scaling is
gated at **fixed time, not fixed sims**: a bigger net that wins at equal sims but loses
at equal wall-clock is a regression in production. The trainer is already parametric
(`--channels/--layers`, `train_selfplay.py:195-196`); the artifact meta carries the
shape, so inference just reads it. Do this only after S0 establishes headroom and at most
one shape change per generation (confounds).

### 2f. Curriculum from the owner's games — the target-opponent lever

Already proven once: gen-5, the first curriculum-trained generation, promoted (bd
`nnue-trainer-1jh.3` notes). Mechanism to preserve: `HumanCurriculumEmitter --human-only`
replays every valid 12×12 game from the freshly fetched prod `games.db` with a human
player, runs deep MCTS with the **current champion** on every multi-choice position, and
emits rows in the exact self-play schema (`pv` = root visits, `z` = real game outcome,
absolute frame), oversampled ×`CURRICULUM_REPEAT=3` because ~200 human games meet ~1500
self-play games and the trainer has no per-file weighting (server doc:63-84; gen
script:186-216, :226-231).

Rust extensions (cheap, in expected-value order): raise `CURRICULUM_SIMS` with the Rust
speedup (deeper targets on exactly the positions the owner reaches); weight recent
owner games by recency-repeat; emit curriculum rows for *owner-won* games with extra
repeats (positions where the current champion's judgment demonstrably failed). All of it
stays inside the schema; none of it changes the trainer. The reason this ranks above net
scaling despite unquantifiable gauntlet Elo: the canary doc's core finding is that
gauntlet fitness did **not** transfer to the owner (canary doc:9-12); curriculum on
owner-reached positions is the only training-side lever aimed at the actual promotion
judge.

## 3. Compute budget reality

What throughput actually matters: (i) self-play sims/s → generation size × target
quality per nightly window; (ii) gauntlet games/h → gate latency (400 games/gate);
(iii) single-move compute at ≤ 120 s → live strength.

| resource | spec | role | source |
|---|---|---|---|
| owner's server, nightly trainer window | container `cpus: 4.0`, `mem_limit: 3g`, `nice 10`, window from 03:00; measured ~5–7 h/generation at 1000 games × 192 sims incl. 4×100-game gauntlet + minutes of training | **the sole RL crank** — the laptop is retired from training by owner constraint (bd `nnue-trainer-1jh.3` notes) | server doc:24-33, :129-138 |
| owner's server, bot process | shares the same host; 120 s/move available live | production play + ponder (capped, see §2b) | server doc; `hub.go:2590` |
| this devbox | 4 vCPU AMD EPYC-Genoa, 8 GB (measured 2026-08-13: `nproc`, `/proc/cpuinfo`, `free`) | development, benches, parity, small gauntlets — it is *smaller* than the 8-core reference box that did 1500 games/192 sims in 2.5–3.5 h (server doc:131), so full-size gauntlets belong on the server window too | measured |
| trainer (python/torch CPU) | minutes per generation at ≤100 k params | not a binding resource | feasibility doc:164-166; server doc:135 |

Planning consequence: **the nightly window is the budget.** Every self-play/gauntlet
throughput claim in §2 cashes out as "what fits between 03:00 and morning on 4 shared
cores." A 5× Rust speedup turns the measured window into roughly: 1000 games at ~800–1000
sims, or ~5000 games at 192 sims, or the gen-5 recipe finishing in ~1.5 h with the rest
of the window spent on a bigger gauntlet — the split per generation is an S5 knob, chosen
by what the gate trend says is binding (target quality vs data volume).

## 4. Promotion gates

Three gates, strictly ordered. A/B are automated sanity floors; C is the only promotion.

**Gate A — generation gate (automated, nightly).** Candidate vs current champion,
pooled `(W + 0.5·D)/N ≥ 0.55` with `N ≥ 400`, fixed sims (`GATE_SIMS`), color-paired
seeds, per-instance seed ranges spaced ≥1000 (gen script:246-249, :273-278; seed-overlap
bug bd `nnue-trainer-riy`; SE at 400 games ≈ ±2.5 pts so 0.55 ≈ 2σ, feasibility
doc:202-206). Identical arithmetic to the Java gate — the Rust arena port must reproduce
it and `ARCHITECTURE.md:24`'s Wilson95 reporting.

**Gate B — regression floor (before any prod candidate).** The candidate plays ≥400
games at 1 s/move fixed time against each of: (i) the Rust enhanced-AB hand-tuned bar
(`vsbot-6me`), (ii) the Go bot (cross-play per `vsbot-e7h`). Rule: **never regress** —
score ≥ the current champion's recorded score against the same opponent minus 2·SE
(≈5 pts at 400). Rationale: the RL gate is self-referential (net vs net); B anchors the
ladder to fixed external opponents so a cycle-exploiting candidate cannot climb.

**Gate C — owner canary (the promotion).** Per the Goodhart lesson: vs-ai2.52 dominated
every scripted gate and lost 3 straight to the owner by turtling; "eval changes ship
canary-first; the owner is the only promotion judge"; gates are "sanity floors, not
strength proof" (canary doc:9-16). Mechanics, matching the existing runbook: the
candidate never auto-ships (server doc:96-100); it deploys under a distinct canary bot
identity on `vs.wandergeek.org`, the owner plays it, and the owner's verdict — not any
number from A/B — promotes it to the default artifact (one candidate per canary, the
`canary/` branch discipline, canary doc:89-95). New live budgets (§2b) are canaried the
same way as new nets: a 120 s ponder bot is a behavior change the owner judges.

**Never gated on:** holdout top-1, value MAE, or any offline metric — logged for
debugging only (seven documented disconnects, deep-labels doc:40-44; invariant 7,
`ARCHITECTURE.md:45`; `CLAUDE.md` "gauntlets only").

## 5. Phased bead breakdown (file as beads verbatim)

Each phase is executor-sized; acceptance criteria are the close conditions. Dependency
spine: S0 → S1 → {S2, S3, S4} → S5 → S6; S7 applies to every prod candidate from S1 on.

### S0 — throughput ground truth + arena time mode
*Depends: vsbot-tfa, vsbot-e7h.*
Criterion benches in `virus-mcts`: net forward (single + batch 8/32), sims/s
single-thread with the gen-5 champion, `apply` cost; a fixed-time mode (ms/move) in
`virus-arena` alongside fixed-sims (the Java gauntlet's gap, `20260807-search-strength.md:39-44`,
cost the H2 work its first harness step — build it first here too).
**Acceptance:** parity fixtures green; measured evals/s (1× and batched), sims/s, and
the Rust:Java throughput ratio recorded in the bead + a `docs/benchmarks.md` table;
arena drives both sides by wall clock with per-move deadline enforcement.

### S1 — dethrone at 1 s: Rust gen-5 vs Java gen-5
*Depends: S0.*
Cross-play gauntlet (per `vsbot-e7h`): Rust MCTS with `artifacts/mcts_champion.json` vs
the Java prod champion, both 1 s/move, 400 games, color-paired, seed ranges spaced ≥1000.
**Acceptance:** pooled ≥ 0.55 → Rust champion becomes the working bar and goes to Gate
B + C. If < 0.55: file the measured sims ratio and loss analysis as a new bead;
re-rank §2 (this outcome means throughput did not materialize and S3 jumps the queue).

### S2 — time manager + pondering
*Depends: S1 (ships in virus-proto/vsbot).*
T1: `VSBOT_MOVE_MILLIS` + fallback-first deadline discipline + early-stop/extension
rules (§2b). T2: **sims→Elo curve**: self-gauntlet cells at 1 s vs 4 s, 4 s vs 16 s,
16 s vs 60 s, 400 games each, fixed-time arena — this is the number every §2 estimate
re-derives from. T3: ponder (search on opponent-to-move snapshots, tree re-root on
their actions, emit only on authoritative turn-driver, version-gated cancellation).
**Acceptance:** T2 curve recorded (Elo per doubling at three points); T3 shows no
regression in a 400-game arena with simulated opponent latency and a ≥20-game live soak
with zero forfeits/timeouts (invariants 2/3/5 each covered by a test); budget profiles
env-documented.

### S3 — MCTS engineering (one bead per item, in this order)
*Depends: S0; T2-T4 also gate through fixed-sims runs so they can proceed in parallel
with S2.*
T1 leaf batching + in-tree parallelism with virtual loss — **accept:** ≥2× sims/s at
equal cores (bench) AND fixed-time self-gauntlet ≥0.55/400 vs the serial searcher.
T2 DAG transpositions (key incl. movesLeft/neutralUsed/side per invariant 6) —
**accept:** correctness tests on within-turn permutation merges; fixed-sims
no-regression (≥0.45/400) and either fixed-time win ≥0.55/400 or measured ≥20 % node
memory reduction at 60 s budgets.
T3 Gumbel root for self-play — **accept:** fixed-sims (192) self-gauntlet Gumbel vs
PUCT+Dirichlet ≥0.55/400; `pi`/`pv` targets still schema-exact.
T4 self-play resign — **accept:** ≥20 % self-play wall-clock saving at measured
false-resign rate <5 % (10 % no-resign control games).

### S4 — Rust self-play generator + trainer round-trip
*Depends: S3-T1 (throughput), S0.*
Emit the exact `SelfPlayMcts` row schema (§2c) from Rust self-play; port the pooled-gate
arithmetic `(W+0.5D)/N ≥ 0.55` into virus-arena; wire the gen-script stage contract
(selfplay → train → gauntlet → report, resumable stamps, `WINDOW=3` sliding window).
**Acceptance:** `train_selfplay.py` consumes Rust JSONL **unchanged** (schema validated
field-for-field incl. absolute-frame `z` and flat action ids); one full generation runs
end-to-end on this devbox at reduced size; the first full-size Rust generation's
candidate passes Gate A vs the gen-5 champion (≥0.55 pooled/400 at 256 sims). A
promoted candidate then goes through Gates B + C like any champion.

### S5 — server nightly integration
*Depends: S4.*
Rust self-play + gauntlet inside the trainer container (or a `vsbot-trainer` sibling
image), preserving: 03:00 window semantics, resumable stamps, fresh-games
guardrail/watermark, curriculum stage (Rust port of the human-only emitter at raised
`CURRICULUM_SIMS`, ×3 repeat), candidate-lands-in-volume / never-auto-ships (server
doc:14-60, :87-100).
**Acceptance:** one unattended nightly generation completes inside the window on the
server delivering ≥2× the Java recipe's compute (games × sims vs 1000×192); `runs.log`/
`history.log` lines equivalent to today's; a second consecutive night resumes/advances
without intervention.

### S6 — net scaling sweep
*Depends: S5 (needs the nightly crank).*
One shape change per generation: 4×48 first, then 6×64 if 4×48 gates through. Trained on
the same sliding window; gated at **fixed wall-clock** (Gate A run in fixed-time mode at
the S2-measured production budget), plus standard Gates B + C.
**Acceptance:** verdict per shape recorded (promote/reject with pooled score); rejected
shapes documented in the bead so they are not re-tried blind.

### S7 — canary runbook (standing, applies to every candidate)
Deploy candidate under a distinct bot name → owner plays → owner verdict recorded in the
bead → on promote: artifact copied to `artifacts/mcts_champion.json` equivalent, tagged,
deployed as default (mirror of server doc:104-122's runbook, adapted to vsbot's deploy
bead `vsbot-1fm`).
**Acceptance per candidate:** owner verdict recorded; no candidate becomes default
without it; Gate B scores archived next to the artifact.

## 6. Non-goals

- **No NNUE distillation revival.** v1/v2/v3 all failed; labels were the binding
  constraint and fixed-depth labels are trapped at their depth (deep-labels doc:109-121;
  `ARCHITECTURE.md:12`). No supervised value-distillation of any evaluator.
- **No offline-metric gating.** Holdout top-1 / value MAE / any proxy: logged, never
  gating (seven disconnects, deep-labels doc:40-44; invariant 7).
- **No new row schema, no trainer rewrite.** The schema and `train_selfplay.py` are
  load-bearing, promotion-proven interfaces; Rust conforms to them (§2c).
- **No 4-player / non-12×12 net support.** Same fallback discipline as prod: MCTS is
  12×12 1v1; other games fall back to the AB stack (server doc:126-127).
- **No GPU dependency.** The nets are ≤100 k params; CPU self-play is the proven regime
  (feasibility doc:7-10) and the server has no GPU.
- **No AB+MCTS ensembling yet** — parked until a measurement asks for it (§2, last row).
