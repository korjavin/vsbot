# Benchmarks

Every number here came from `virus-arena`. Read the caveat column before quoting
any of them.

The house rule (CLAUDE.md, ARCHITECTURE.md invariant 7) is that only gauntlets
count, and only at ≥400 games. **Nothing on this page is at 400 games yet.**
Everything below is labelled `INFORMATIONAL ONLY` or `indicative` by the harness
itself, and none of it may gate a promotion. They are here because they are the
project's first real Rust-vs-Rust datapoints and because reproducing them is how
you check the harness still works.

Reproduce any row by pasting its command. Node-budget rows reproduce exactly;
fixed-time rows do not, by construction (see "Determinism" below).

Machine for every run in this file: 4 cores, 7 GB RAM, shared dev host, load
average 1–2 before each run.

---

## 1. Enhanced alpha-beta vs the plain oracle, equal node budget

The A/B the enhanced stack exists to justify: same eval, same budget, the only
difference is the enhancement set (TT + killers + history + LMR + aspiration
windows + cross-move table reuse).

```bash
cargo build --release -p virus-arena
./target/release/arena --a ab-enhanced --b ab-plain \
    --budget nodes:60000 --games 100 --seed 11 --threads 4
```

```
ab-enhanced:n60000 vs ab-plain:n60000
  W-L-D 53-47-0 over 100 games (indicative)
  win rate 53.0% (draws not half-wins)  wilson95 [43.3%, 62.5%]
  pooled score 0.5300 (W+0.5D)/N  margin +6
  elapsed 495.8s
  NOTE: fewer than 400 games: a direction may be read off the interval, but
        this cannot gate a promotion (CLAUDE.md: gauntlets only, >=400 games).
```

| | |
|---|---|
| W-L-D | **53-47-0** over 100 games |
| Wilson 95% | **[43.3%, 62.5%]** — spans 50% |
| Pooled `(W+0.5D)/N` | 0.5300 |
| Verdict | `indicative` — may not gate |

**Read this correctly: it is not evidence that the enhancement stack does
nothing.** The interval spans 50%, so at 100 games it cannot separate the two —
and it is measured at equal *nodes*, which is the wrong axis for this stack. The
enhancements (TT reuse, killers, history, LMR, aspiration windows) mostly buy
*better nodes per node* and better time-to-depth; Java's headline +64% figure was
measured at equal **time**, where the plain searcher's worse move ordering
translates directly into less depth. An equal-node comparison deliberately
removes most of that advantage. The value of this run is that it exercises the
whole harness against a real workload, and that the two arms are close enough
for the pairing machinery to be doing visible work.

## 2. MCTS gen-5 champion vs enhanced alpha-beta, 1 s/move

**The project's first Rust-vs-Rust strength datapoint across engine families.**
Both sides at 1000 ms per action, wall clock, deadline enforced.

Fixed time is the only honest way to compare these two: an MCTS simulation and
an alpha-beta node are different units of work and there is no conversion
between them. This is the mode `docs/plans/superiority.md` S0 called out as the
gap that cost the Java harness its first step.

```bash
./target/release/arena \
    --a mcts:artifacts/mcts_champion.json --b ab-enhanced \
    --budget ms:1000 --games 100 --seed 20260813 --threads 4
```

```
mcts[mcts_champion]:1000ms vs ab-enhanced:1000ms
  W-L-D 93-7-0 over 100 games (indicative)
  win rate 93.0% (draws not half-wins)  wilson95 [86.3%, 96.6%]
  pooled score 0.9300 (W+0.5D)/N  margin +86
  worst move deadline overrun: 11 ms
  elapsed 1437.4s
  NOTE: fewer than 400 games: a direction may be read off the interval, but
        this cannot gate a promotion (CLAUDE.md: gauntlets only, >=400 games).
```

| | |
|---|---|
| W-L-D | **93-7-0** over 100 games |
| Wilson 95% | **[86.3%, 96.6%]** |
| Pooled `(W+0.5D)/N` | 0.9300 |
| Worst deadline overrun | 11 ms of 1000 |
| Verdict | `indicative` — may not gate |

**The gen-5 MCTS champion dominates the hand-tuned enhanced alpha-beta bar at
equal wall clock.** The interval is nowhere near 50%, so even at 100 games the
direction is not in doubt — but it is still 100 games, not 400, so it labels
`indicative` and cannot gate anything. Reproduce at `--games 400` for a
gate-eligible number.

This is consistent with the predecessor's history rather than surprising: the
Java MCTS champion dethroned the Java hand-tuned enhanced searcher too
(`superiority.md` §0). What is new is that both sides are now Rust, so the
comparison is free of the cross-language confound.

**Two things to know before quoting the margin.**

*Alpha-beta does not spend its whole second.* `SearchOptions::soft_deadline_percent`
is 55: past 55% of the budget it refuses to start an iteration it would almost
certainly not finish. That is the engine's own production time management, not a
handicap the harness imposed — it is how the bar actually plays — but it means
"1 s/move" is a ceiling for that arm, not a spend.

*The `work_a` column is not sims per second.* It counts simulations, and a
simulation that selects down to an already-known terminal node returns a cached
value without touching the net or the eval (`virus-mcts/src/search.rs:403`). In
a decisively won endgame the tree saturates with those, so per-game sim totals
run far above what sustained net-value throughput would suggest. The honest
throughput figures come from `virus-mcts`'s own microbenchmark:

```
$ cargo run --release -p virus-mcts --example mctsbench
net forward, avx2             0.249 ms/op    4014 ops/s
net forward, portable         0.393 ms/op    2546 ops/s
mcts sims, net value          0.263 ms/op    3799 sims/s
mcts sims, hand-tuned value   0.011 ms/op   92062 sims/s
```

So ~3800 net-value sims per second per core, i.e. roughly 3800 sims in a 1 s
move from a fresh opening tree. (Recorded here for convenience; establishing the
Rust:Java throughput ratio is `superiority.md` S0's job, not this bead's.)

## 3. Cross-play: vsbot vs the Go bot, through the real server

Not an engine comparison — a **plumbing check**. The Go server refereed, the Go
bot-hoster ran its default accept-only pool, and `vsbot` challenged. W-L-D was
read off the server's own `games.db`.

```bash
cargo build --release -p vsbot
python3 crates/virus-arena/crossplay/crossplay.py --games 50 --search GREEDY
```

```
=== vsbot(GREEDY) vs GoBot: W-L-D 49-1-0 over 50 games (INFORMATIONAL ONLY) ===
    win rate 98.0% (draws not half-wins)  wilson95 [89.5%, 99.6%]
    seats: 50 as P1, 0 as P2
crossplay: WARNING — colours are not balanced (50 P1 / 0 P2). This number
includes first-mover advantage and is NOT comparable to an `arena` gauntlet,
which cancels it by pairing. Treat it as a plumbing check.
```

The plumbing works. 53 vsbot-vs-GoBot games were recorded, all terminating
`no_moves` (a real board ending, not a forfeit or a timeout), median 41 turns.
3 further rows were `disconnect`s from the shutdown race and were discarded, and
3 were vsbot-vs-vsbot and never matched the name filter. The Go bots really
searched: 2968 logged searches at depth 4–5, 5k–35k nodes each.

### 3a. Why that 98% was not a strength result (bd `vsbot-t3q.1`, resolved)

**The 98% was a harness artifact. The live Go bot is fine.** It disagreed with
the arena — which puts `greedy` at 25% against the byte-exact GoBot oracle — by
far more than a colour bias can explain, so it got its own bead. The answer is
two compounding defects in the cross-play harness, and neither is an engine
problem.

**First, the Go bot was searching properly**, which rules out the candidates the
bead led with (a diverged binary, a differently-configured search, CPU
starvation). A 12-game re-run logged 734 searches at **depth 3–7, median 4, mean
15 317 nodes**. The arena's `ab-plain --b-budget depth:4` spends ~3.3 M nodes
over ~110 plies — about 30 k nodes a move at the same depth. The deployed bot
and the offline oracle are the same searcher doing the same work; the hoster
calls `gamesearch.Choose` under the 1000 ms `ProductionBudget`, and
`crossplay.py` already pins `BOT_EXPLORE_EPSILON=0` so nothing randomises it
weaker. Row attribution in `games.db` was verified too, and is correct.

**Defect 1 — every game was played from one chair, and the chair is worth
something.** The server seats the challenger at P1 (`hub.go`
`handleAcceptChallenge`: `Player1: challenge.FromUser`) and only vsbot
challenged. Quantified with a 200-game colour-paired control:

```bash
./target/release/arena --a greedy --b ab-plain \
    --a-budget nodes:1 --b-budget depth:4 --games 200 --seed 777 --threads 1 \
    --per-game
```

```
greedy:n1 vs ab-plain:d4
  W-L-D 65-135-0 over 200 games (indicative)
  win rate 32.5% (draws not half-wins)  wilson95 [26.4%, 39.3%]
```

| Split | Result |
|---|---|
| greedy seated **P1** | 40 / 100 = **40.0%** |
| greedy seated **P2** | 25 / 100 = **25.0%** |
| P1 seat, either engine | 115 / 200 = **57.5%** of all games |

So moving first is worth about **+7.5 points of win rate** (~52 Elo) on this
board, and seating greedy exclusively at P1 lifts it from 25% to **40%**. Real,
and worth cancelling — but it does not turn 25% into 98%.

**Defect 2 — the 50 games were not 50 samples.** Nothing randomises a
cross-play opening and both bots play argmax, so the run *replays the same
game*. A 12-game re-run reproduced the anomaly exactly (12-0-0, 100%) and
contained only **5 distinct games**, with the opening identical across all
twelve. That is the whole gap: at a true one-chair rate of 40%, a genuine 12-0
has probability 0.4¹² ≈ 1.7 × 10⁻⁵. The games were not independent, so the
published Wilson interval `[89.5%, 99.6%]` — binomial over 50 assumed-
independent games — was measuring a sample size that did not exist.

**Why one opening returns ~100%.** Replay the pair with the opening noise off,
so the identical opening is played from both chairs:

```bash
./target/release/arena --a greedy --b ab-plain \
    --a-budget nodes:1 --b-budget depth:4 --games 2 --seed 777 --eps 0 \
    --threads 1 --per-game
```

```
game 0 seat_a=1 winner=1   greedy is P1 -> greedy wins
game 1 seat_a=2 winner=1   ab-plain is P1 -> ab-plain wins
```

**P1 won both.** In that position the first move decides the game whichever
engine holds it: the greedy floor beats the depth-4 oracle from the P1 chair,
and the oracle beats greedy from it. Cross-play locked onto exactly one opening
and always played it from the winning chair, so it returned that single game's
outcome fifty times. The arena's 25% is the same matchup averaged over *diverse*
openings with the colours paired — which is what makes it the trustworthy
number.

**What the harness does about it now.** `--direction alternate` splits a run
into two phases so half the games are played from each chair (needs an opponent
that can challenge back — see below); every report carries per-seat win rates
and a **distinct-game count**, fingerprinted over the move sequence with the
wall-clock `duration_cs` stripped; and a run whose games are mostly replays says
so loudly instead of quoting a confident interval. `crossplay --self-test`
covers that counting logic in CI.

The original conclusion stands, narrowed: the harness runs end to end and
records real, completed games. It was never measuring strength.

---

## Determinism

The gate, asserted by `crates/virus-arena/tests/determinism.rs` and re-run
end-to-end for this page:

* A `nodes:` gauntlet at a fixed seed reproduces **byte-identically** across two
  runs — not just W-L-D, but every game's winner, ply count, termination reason
  and per-side node count.
* The tally is invariant under `--threads` (1, 2, 3, 4, 8 all checked). Outcomes
  are folded in game-index order, never completion order.
* A configuration played against itself reads **exactly 50%**: both games of a
  colour pair are literally the same game, so the same seat wins both. If this
  ever stops holding, state is leaking between games and every number on this
  page is void.
* Lazy SMP is off in every arena arm. Helper threads write the shared
  transposition table, so a search with SMP on is not reproducible run to run.

Run end to end for this page on the row-1 configuration — 100 games, seed 11,
`nodes:60000` — once at `--threads 4` and once at `--threads 1`:

```
$ diff <(grep -vE '^arena:|elapsed' run1.txt) \
       <(grep -vE '^arena:|elapsed' run1b.txt)
$ echo $?
0
```

105 lines identical: the 100 `--per-game` records (winner, turns, plies,
termination, per-side node counts) plus the five summary lines. Only the elapsed
time differed — 495.8 s on 4 threads, 1568.7 s on 1.

`ms:` mode reproduces nothing. That is what a wall clock means, and a test
pretending otherwise would fail on a loaded box and teach everyone to ignore it.
Fixed-time rows above are single observations; re-running them will give
different numbers within the interval.

## What is not measured yet

| Gap | Why | Unblocked by |
|---|---|---|
| Anything at 400 games | Wall clock. Row 2 took 24 min for 100 games on 4 cores, so 400 is ~1.6 h; row 1 is ~33 min. Nothing blocks it but time. | A longer run |
| Cross-play with balanced colours **against the Go bot** | Structural, and now confirmed by reading the code rather than assumed: the bot-hoster's challenger targets `Manager.IsAcceptor(userID)`, which is false for every id outside its own pool, so it cannot challenge `vsbot` and cannot seat it at P2. `--direction theirs` is refused for that opponent rather than quietly returning another one-chair number | a bot-hoster change |
| A cross-play number that is a strength result | Even colour-balanced, cross-play has no opening randomisation, so its games are largely replays (§3a). The distinct-game count says how many samples a run really had; until a diversity knob exists, treat the count, not the game count, as `n` | an opening-randomisation lever on at least one side |
| Two different net artifacts in one gauntlet | The harness shares one loaded net across all games and threads; a net-vs-net run needs a second one threaded through the sides. Refused with an error rather than silently playing one artifact against itself | a follow-up bead |
| Rust:Java throughput ratio (superiority.md S0) | Needs criterion benches in `virus-mcts`, which are that bead's scope, not this one's | S0 |
