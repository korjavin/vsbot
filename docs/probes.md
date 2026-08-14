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
`fixtures/probes/neutrals-v1.jsonl` (**48 positions**: 38 mined from prod
`games.db` as of 2026-08-09, 8 from champion self-play, and **2 from the
owner's live games against `SuperiorBot`**, added 2026-08-14 from the
WAL-recovered prod corpus — see "The owner's live games" below).

```
cargo run --release -p virus-arena --bin probe -- run \
    --set fixtures/probes/neutrals-v1.jsonl --sims 192 --sims 1000
```

| | all 48 | mined 38 | self-play 8 | live-owner 2 |
|---|---|---|---|---|
| mean prior mass on the `PlaceNeutrals` class | 0.316 | 0.271 | 0.431 | **0.710** |
| chose `PlaceNeutrals` @192 sims | **31.2%** | 21.1% | 62.5% | **100%** |
| chose `PlaceNeutrals` @1000 sims | **43.8%** | 31.6% | 87.5% | **100%** |
| net's single favourite action is a pair | 2 of 48 | 1 of 38 | 1 of 8 | 0 of 2 |
| value head rates the post-neutral child above the position | 22 of 48 | 12 of 38 | **8 of 8** | **2 of 2** |
| mean `V(after neutral) - V(before)` | +0.043 | -0.069 | +0.467 | **+0.484** |
| mean `V(after neutral) - V(after best move)` | +0.078 | -0.034 | +0.501 | **+0.511** |

By class:

| class | n | mean class prior mass | chose `PlaceNeutrals` @1000 |
|---|---|---|---|
| `lost-advantage` (suspect) | 31 | 0.273 | 25.8% |
| `kept-advantage` (control) | 9 | 0.362 | 66.7% |
| `champion-chose-neutral` | 8 | 0.431 | 87.5% |

**The headline for the bead.** More search does not rescue the decision: the
share of positions answered with a neutral *rises* from 31.2% at 192
simulations to 43.8% at 1000, and on the self-play half it goes from 62.5% to
87.5%. That is the bead's own observation (cold, warm and a 10 000-simulation
reference all agree) reproduced as a repeatable number.

**The control group barely separates.** The champion puts *more* prior mass on
neutrals at the control positions (0.362) than at the suspect ones (0.273).
The direction is defensible — those are the placements whose player went on to
win — but the gap is not a net that has learned when a neutral is good.

> **Three cells from the 2026-08-13 table did not reproduce.** Re-running the
> committed set with the committed champion on 2026-08-14 gives `@192` = 13 of
> the original 46 positions where the first table recorded 15, and `@1000` = 19
> where it recorded 20 (the deltas are one mined position at 192, one self-play
> position at 192, and one self-play position at 1000). Everything computed from
> a plain forward pass reproduces **exactly** — mean class mass 0.299, favourite-
> is-a-pair 2 of 46, value-head-above 20 of 46, mean `dV` +0.024 — and only MCTS
> search *choices* moved, on three positions that were near the boundary. The
> run itself is byte-reproducible (two consecutive runs produce identical
> per-position JSONL), so this is a difference between builds or hosts, not
> run-to-run noise. The table above is the reproducing measurement. Separately,
> the old table's self-play `mean dV` cell (+0.428) was inconsistent with its own
> other two cells, which imply +0.466; the reproducing value is +0.467.
>
> Nothing downstream turns on the three cells — the verdicts below are all read
> off the mined half at 1000 simulations, which reproduces exactly (12 of 38) —
> but a number that does not reproduce must not be quoted as if it did.

## The owner's live games

bd `vsbot-07x` named the owner's 2026-08-13 game against a `SuperiorBot` seat
as the repro material for an *unmotivated* neutral placement, and v1 shipped
without it: the published `games.db` snapshot stopped at 2026-08-09. The prod
database was recovered by WAL replay on 2026-08-14 (2041 games, through
08:53), and the games are in it.

**There is no seat named `SuperiorBot Bot 1079` anywhere in the recovered
corpus.** The bead's identifier is wrong or misremembered. What the corpus does
contain is 30 games with a `SuperiorBot` seat, of which four pair one against a
non-bot-named seat *and* contain a neutral placement:

| game | started | seats | result | neutral |
|---|---|---|---|---|
| `4e7cd5c0` | 2026-08-13 16:04 | `WiseBuffalo50` vs `SuperiorBot Bot 1220` | seat 1 (`no_moves`) | turn 6, seat 2 |
| `1e86961a` | 2026-08-13 16:34 | `CunningFalcon20` vs `SuperiorBot Bot 1220` | seat 2 (`resignation`) | turn 6, seat 2 |
| `f870e183` | 2026-08-13 18:10 | `WildDragon26` vs `SuperiorBot Bot 1220` | seat 1 (`disconnect`) | turn 6, seat 2 |
| `cea648f5` | 2026-08-14 08:33 | `ModernMoose85` vs `SuperiorBot Bot 7420` | seat 2 (`resignation`) | turn 12, seat 2 |

### The unmotivated placement, identified

In **every one of the four**, `SuperiorBot` converted the same two of its own
cells: **`(8,8)` and `(11,10)`**. `(8,8)` is a cell it placed on turn 2 and
`(11,10)` one it placed on turn 4 — two cells of its own opening structure,
`(11,10)` orthogonally adjacent to its own base at `(11,11)`.

The first three games are the *same position*. The three opposing seats played
an identical opening (`(1,1) (2,2) (3,3)` / `(4,4) (5,3) (3,5)` / `(5,5) (6,4)
(4,6)`, the last turn differing only in move order), the bot answered
identically, and the miner's position hash confirms it: four neutral turns
collapse to **two distinct positions**, which is why the fixture gains two
records and not four. The bot's recorded think time for the placement is
`102 cs` in all three. This is a deterministic client reaching one position
repeatedly, not four independent observations — do not read the repetition as
four pieces of evidence.

At the first of those positions (`live-4e7cd5c0-t6`) the bot was **behind by 3
cells with 14 legal moves available**, and the placement cost it **5 more cells
on its very next turn** (`immediateCost = -5`); it finished the game 18 cells
down. That is the bead's "unmotivated neutral placement", in a named game, at a
named turn, with the exact pair recorded.

### gen-5 reproduces it

| position | class | own normals | legal pairs | class mass | `dV(neutral)` | @192 | @1000 |
|---|---|---|---|---|---|---|---|
| `live-4e7cd5c0-t6` | `lost-advantage` | 6 | 15 | 0.581 | **+0.277** | NEUTRAL 83% | NEUTRAL 73% |
| `live-cea648f5-t12` | `kept-advantage` | 15 | 105 | 0.840 | **+0.691** | NEUTRAL 90% | NEUTRAL 98% |

The gen-5 champion answers **both** with a neutral at both simulation counts —
the only source in the set at 100% — and at 1000 simulations it picks
**`(8,8)+(11,10)` at both positions**: the identical pair the deployed bot
played. The value head is the reason, exactly as verdict 3 below says: it rates
the post-neutral child **+0.28** and **+0.69** above the position it came from,
on a `tanh` scale of ±1, for an action that mechanically costs 5 cells.

These two are the sharpest positions in the whole set, and they are the ones the
owner actually saw. They are also only two positions, from one bot build against
one deterministic opening — a *demonstration*, not a measurement.

> **"Human" here means "non-bot-named seat".** The filter reads a seat as a bot
> iff its name contains "bot", which is the server's own naming convention. The
> three seats above carry web-player-style names and yet played byte-identical
> openings, so at least some of them are scripted rather than human. It does not
> change what the positions are — real recorded games in which the deployed
> `SuperiorBot` chose a neutral — but the corpus is not evidence about how the
> bot behaves against *people*.



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
  numbers to watch are `chose PlaceNeutrals @1000` (43.8% at gen-5), the
  self-play half's `8 of 8` on value-head-above, and — the sharpest one — the
  two `live-owner-game` positions, which gen-5 answers with a neutral at both
  simulation counts and at 1000 with the *identical pair the deployed bot
  played*. A generation that stops volunteering `(8,8)+(11,10)` at
  `live-4e7cd5c0-t6` has changed the behaviour the owner reported.
  Informationally, and never as a gate: whether the generation is *stronger* is
  still a >=400-game gauntlet question.

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
- The `live-owner-game` half is **two positions from one bot build against one
  deterministic opening**. It is the sharpest illustration in the set and the
  weakest sample in it; nothing about frequency, and nothing about how the bot
  behaves against varied opposition, follows from it. The mined 38 remain the
  half to read for anything aggregate.
- No seat named `SuperiorBot Bot 1079` exists in the recovered corpus, so the
  identifier in bd `vsbot-07x` does not resolve. The positions above are from
  `SuperiorBot Bot 1220` and `Bot 7420`; if the bead meant a different game, it
  is not in the 2026-08-14 recovery either.
