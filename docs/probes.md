# `PlaceNeutrals` probe — gen-5 baseline and the pair-head verdict

bd `vsbot-07x`. Everything here is **informational**. ARCHITECTURE.md
invariant 7 — seven offline metrics were each believed to predict playing
strength and each one was wrong — so nothing in this document is a strength
claim, nothing in it gates anything, and `probe run` prints that caveat on
every invocation and exits 0 whatever the numbers say. Strength claims come
only from >=400-game gauntlets.

- **Probe set**: `fixtures/probes/neutrals-v1.jsonl` (schema, provenance and
  the regeneration recipe: `fixtures/probes/README.md`).
- **Tool**: `cargo run --release -p virus-arena --bin probe -- run`.
- **Heuristic**: `crates/virus-arena/src/probes.rs` module header.

## The gen-5 baseline

`artifacts/mcts_champion.json` — `conv-policy-value-v1`, 32 channels, 4 layers,
`pair_bias = -0.0547`, value head present — over
`fixtures/probes/neutrals-v1.jsonl` (46 positions: 38 mined from prod
`games.db` as of 2026-08-09, 8 from champion self-play).

```
cargo run --release -p virus-arena --bin probe -- run \
    --set fixtures/probes/neutrals-v1.jsonl --sims 192 --sims 1000
```

| | all 46 | mined 38 | self-play 8 |
|---|---|---|---|
| mean prior mass on the `PlaceNeutrals` class | 0.299 | 0.271 | 0.431 |
| chose `PlaceNeutrals` @192 sims | **32.6%** | 23.7% | 75.0% |
| chose `PlaceNeutrals` @1000 sims | **43.5%** | 31.6% | **100%** |
| net's single favourite action is a pair | 2 of 46 | 1 of 38 | 1 of 8 |
| value head rates the post-neutral child above the position | 20 of 46 | 12 of 38 | **8 of 8** |
| mean `V(after neutral) - V(before)` | +0.024 | -0.069 | +0.428 |
| mean `V(after neutral) - V(after best move)` | +0.059 | -0.034 | +0.503 |

By mined class:

| class | n | mean class prior mass |
|---|---|---|
| `lost-advantage` (suspect) | 30 | 0.263 |
| `kept-advantage` (control) | 8 | 0.302 |
| `champion-chose-neutral` | 8 | 0.431 |

**The headline for the bead.** More search does not rescue the decision: the
share of positions answered with a neutral *rises* from 32.6% at 192
simulations to 43.5% at 1000, and on the self-play half it goes to 100%. That
is the bead's own observation (cold, warm and a 10 000-simulation reference
all agree) reproduced as a repeatable number.

**The control group barely separates.** The champion puts *more* prior mass on
neutrals at the control positions (0.302) than at the suspect ones (0.263).
The direction is defensible — those are the placements whose player went on to
win — but the 0.04 gap is not a net that has learned when a neutral is good.



## What the numbers say about the bead's hypothesis (a)

The bead asks: *does the factored pair head — `logit(PlaceNeutrals{i,j}) =
u[i] + u[j] + pair_bias` — structurally over-value neutrals in some phases?
Can it even express "no neutral is good here" relative to the move logits?*

The masked softmax the searcher actually uses gives an exact decomposition.
Writing `L_pair` and `L_move` for the log-sum-exp of each class's logits over
the node's legal actions:

```
neutralPriorMass = sigmoid(L_pair - L_move)
```

(verified on the set: `max |mass - sigmoid(gap)| = 1.2e-7`). And that gap
splits into two terms that mean different things:

```
classLogitGap = countTerm + levelTerm
countTerm     = ln(#pairs) - ln(#moves)      the class's size, which the net cannot change
levelTerm     = the rest                      the logit levels, which the net can
```

### Verdict 1 — the head *can* express "no neutral is good here". It is not the bottleneck.

`levelTerm` reaches **-5.61** on the mined half; the smallest observed prior
mass on the whole `PlaceNeutrals` class is **0.0026** (0.26%), at a position
with 21 legal pairs. Three of the 38 mined positions sit under 4% class mass.
The head is demonstrably capable of suppressing the entire class, and the net
demonstrably does so when it wants to.

`pair_bias` is a **single global scalar**, `-0.0547` in the gen-5 champion.
It is position-independent by construction, so it cannot be the *phase*-
dependent mechanism the bead hypothesised: it shifts every pair at every
position by the same 0.05 nats, which is nothing next to a `levelTerm` that
ranges over six nats. **The shared `pair_bias` is not the problem.**

So: **no head-architecture change is warranted by this evidence**, and no
S6-class training change is filed. What the analysis did find instead is
below, and it points at direction (b), not at the head.

### Verdict 2 — there *is* a structural inflation, but it is class-size, not the pair head, and it does not drive the choice

The pair class is enormous — every `C(n,2)` combination of the mover's own
normals is a separate legal action, up to 190 of them in the self-play
positions against a dozen or so moves. The factored head makes the class's
weight a closed form:

```
sum_{i<j} exp(u_i + u_j + b) = exp(b) * (S^2 - Q) / 2      S = sum exp(u_i), Q = sum exp(2 u_i)
```

— *quadratic in the mover's own cell count*, whatever the trunk says. The
reported `pairLogsumexpClosedForm` drops the `Q` correction and lands within
0.09–1.00 nats (mean 0.27) of the true `L_pair`, so the `S^2` picture is the
right one. Holding the class's share of the prior constant as the mover grows
therefore requires the trunk to push `u` down like `-ln(n)` — work the head
does not do for free.

That inflation is real and it is worst exactly where the mover is running out
of moves, which is the end-of-game phase where the corpus's neutral placements
cluster:

| phase (mover's legal moves) | n | mean class mass | mean countTerm | mean levelTerm | chose neutral @1000 |
|---|---|---|---|---|---|
| <= 5 (cornered) | 5 | 0.480 | **+2.336** | -2.434 | **0%** |
| 6–14 | 19 | 0.268 | +0.159 | -1.345 | 42% |
| >= 15 (open) | 14 | 0.201 | +0.067 | -1.973 | 29% |

Across the mined half `countTerm` averages **+0.41 nats** and peaks at
**+3.11** (a 22x odds gift, at a position with one legal move and ten legal
pairs). The net compensates: `levelTerm` averages **-1.72**.

**But the inflation is not what makes the champion play the neutral.** The
cornered bucket has the largest class mass *and* the largest count term, and
the searcher chose a neutral there **0% of the time**. Split the mined half by
what the search actually did at 1000 simulations:

| | chose NEUTRAL (12) | chose a move (26) |
|---|---|---|
| mean class mass | 0.351 | 0.234 |
| mean countTerm | **-0.044** | **+0.622** |
| mean levelTerm | -0.701 | -2.190 |
| mean top single pair prior | 0.085 | 0.026 |

The count term runs *the wrong way*: positions where the class is numerically
inflated are positions the searcher does **not** answer with a neutral. That
is what one should expect — PUCT spreads a class's prior over its members, so
a class that is big because it has many members hands each member a tiny
prior, and MCTS visits are driven by backed-up value, not by prior mass. The
single largest pair prior beats the single largest move prior at only **1 of
38** mined positions.

### Verdict 3 — the driver is the value head

Compare the net's value at the position with its value after the top-prior
neutral child and after the top-prior move child, all in the mover's frame:

| | chose NEUTRAL (12) | chose a move (26) |
|---|---|---|
| mean `V(after neutral) - V(before)` | **+0.181** | -0.185 |
| mean `V(after neutral) - V(after move)` | **+0.236** | -0.159 |

and as a classifier:

- value head rates the neutral child **above** the move child at 15 of 38
  positions → the search played a neutral at **10 of those 15 (67%)**;
- value head rates the move child above → the search played a neutral at
  **2 of 23 (9%)**.

The self-play half makes the same point without any hedging. Those eight
positions were selected purely by "the champion answered this with a neutral",
with no reference to values — and the value head rates the post-neutral child
above the position at **8 of 8** of them, by **+0.12 to +0.89** (mean +0.43).
Every single position where gen-5 volunteers a neutral is a position where its
value head believes the neutral gains it something.

At **12 of 38** mined positions the value head rates the post-neutral child
*above the position it came from* — by up to **+0.73** on a `tanh` scale of
±1. That is the anomaly. `PlaceNeutrals` converts two of your own cells to
dead space and forfeits the turn's three placements; measured over this same
corpus the mechanical price is **-5.5 cells of material advantage** on the
mover's very next turn (mean over 414 recorded placements, median -6). A
value head that reads that as a gain of two thirds of a win is not being
misled by the policy prior — it is wrong about the position.

**Conclusion for the bead.** Hypothesis (a) is *not supported*: the factored
pair head can express "no neutral is good here", uses a global `pair_bias` too
small and too position-independent to be a phase mechanism, and its one real
structural artefact (class-size inflation of the prior) is negatively
associated with the behaviour being investigated. The over-valuation lives in
the **value head**, which is a labelling and curriculum problem — the bead's
direction (b) — not a head-architecture problem. No S6-class change is filed
off this analysis.

## What this implies for direction (b), the curriculum signal

Not implemented here — this bead builds the instrument and reads it — but the
analysis narrows what the signal has to be, so recording it saves the next
bead the re-derivation:

- **Oversample on the value target, not the policy target.** The policy prior
  is not what picks the neutral (verdict 2); the value head is (verdict 3). A
  curriculum that reweights `PlaceNeutrals`-eligible positions in the *policy*
  loss would be aiming at the wrong head.
- **The label is cheap and already computed.** `Labels::immediate_cost` — the
  mover's advantage change over its own next turn — averages **-5.5 cells**
  over 414 recorded placements. Any self-play position whose post-neutral child
  the net values *above* the pre-neutral position is, mechanically, a
  mis-valued position; that predicate needs one extra forward pass per
  neutral-eligible node and needs no game outcome.
- **Where to find them.** 61% of the corpus's recorded neutral placements have
  at most one follow-up turn of their own, i.e. they cluster at the end of the
  game where terminal labels are least ambiguous — a cheap, dense source of
  the exact positions the value head is wrong about.
- **How to know it worked.** Re-run this probe on the new generation. The
  number to watch is `chose PlaceNeutrals @1000` (43.5% at gen-5) and the
  self-play half's `8 of 8`. Informationally, and never as a gate: whether the
  generation is *stronger* is still a >=400-game gauntlet question.

## What this does not show

- Nothing here is a strength measurement, and the probe must never become a
  gate (invariant 7). Two nets with different probe numbers are not thereby
  ranked.
- The mining heuristic is correlational over a corpus of bots of varying
  strength; `fixtures/probes/README.md` and the module header state its limits.
- The `champion-chose-neutral` half is selected *by* the gen-5 champion's own
  choices, so a future generation scoring differently on it is partly
  regression-to-the-mean. That is why the mined half — selected with no
  reference to any net — is reported separately in the table above.
- The owner's 2026-08-13 live game against `SuperiorBot Bot 1079` is **not**
  in the set: the published `games.db` snapshot stops at 2026-08-09 and names
  no such seat. See `fixtures/probes/README.md`.
