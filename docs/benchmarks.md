# Benchmarks

Every number here came from `virus-arena`. Read the caveat column before quoting
any of them.

The house rule (CLAUDE.md, ARCHITECTURE.md invariant 7) is that only gauntlets
count, and only at ≥400 games. Rows 1–3 are below that bar and are labelled
`INFORMATIONAL ONLY` or `indicative` by the harness itself; none of them may
gate a promotion. Row 4 (S1) *is* at 400 games, but its games are not 400
independent samples — read its distinct-game count before quoting it. They are
here because they are the project's first real datapoints and because
reproducing them is how you check the harness still works.

**Every cross-play number on this page (§3, §3a, §4) predates the opening
randomisation added in §3b and is superseded.** Their sample size is their
distinct-game count, not their game count. Re-run, don't quote.

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

### 3b. The diversity fix, and what it supersedes (bd `vsbot-t3q.2`, resolved)

**Every cross-play number recorded above — §3, §3a and §4 — was measured
without opening randomisation and is superseded.** Their tallies are real
games honestly counted, but their *sample sizes* are the distinct-game counts,
not the game counts, so none of them may be quoted as a rate or gate anything:
§3's `49-1` was 5 distinct games, §4's 400-game S1 was 65. Nothing below
re-states them; they stay on the page as the record of how the defect was
found. Re-run anything that needs to be a number.

The lever the earlier sections said did not exist now does, on the one side
this repository controls. `vsbot` takes `VSBOT_EXPLORE_EPS` /
`VSBOT_EXPLORE_TURNS` / `VSBOT_EXPLORE_SEED` (`crates/vsbot/src/explore.rs`):
inside an opening window it plays a uniformly random legal action instead of
the searched one, drawn from a per-game SplitMix64 stream derived
`mix64(seed ^ GOLDEN_GAMMA · (game + 1))` — the arena's derivation without its
colour-pair folding, which cross-play cannot use. `crossplay.py` drives it by
default (`--explore-eps 0.15 --explore-turns 8`) and hands every phase, every
vsbot instance and every pooled shard a **disjoint** derived seed, so nothing
in a run replays anything else in it. The window is counted in vsbot's *own*
turns because a client never sees the opponent's, and 8 of them is ~24 coin
flips — the same opening noise per game an `arena` run spends across both of
its sides.

Only vsbot explores. The Go bot-hoster's `BOT_EXPLORE_EPSILON` fires on every
turn of every game from an unseeded global RNG, which would weaken the opponent
throughout and unreproducibly; it stays pinned to `0` (`--opponent-explore-eps`
exists, and is never right for a gating run). The Java bot's
`CHALLENGER_EXPLORE` only reaches its `SEARCH=GOBOT` path, so it does nothing in
the `SEARCH=MCTS` configuration §4 measures. The asymmetry handicaps **vsbot** —
it plays a few random opening moves its opponent does not — which is the
conservative direction for a "vsbot is stronger" claim, and is why a re-run
number is a floor rather than a ceiling.

Verification, 50 games against the Go bot, same command shape as §3:

```bash
cargo build --release -p vsbot
python3 crates/virus-arena/crossplay/crossplay.py --games 50 --search GREEDY --json
```

```json
{ "vsbot_search": "GREEDY", "explore_eps": 0.15, "explore_turns": 8,
  "explore_seed": 20260813, "opponent_explore_eps": 0.0,
  "wins": 11, "losses": 39, "draws": 0, "games": 50,
  "as_p1": 50, "as_p2": 0, "win_rate": 22.0,
  "wilson95_low": 12.8, "wilson95_high": 35.2,
  "distinct_games": 50, "red_flag_terminations": 0 }
```

| | before (§3) | after |
|---|---|---|
| Distinct games | **5 / 50** | **50 / 50** |
| Low-diversity warning | fired | silent |
| W-L-D | 49-1-0 (98.0%) | 11-39-0 (22.0%) |
| Forfeits | 0 | 0 |

**50 of 50 games distinct**, and the harness's low-diversity warning stops
firing. The two vsbot instances logged 61 and 79 exploration moves over 24 and
28 games — ~2.7 a game, under the ~3.6 the window's ~24 flips at eps 0.15
predict, because games that end inside the window and plies whose search was
superseded do not consume it.

The second row of that table is the real headline. **§3's 98% collapses to 22%
once the opening is not a single replayed game** — right beside the arena's 25%
for `greedy` against the byte-exact GoBot oracle, and well below the 40% the
arena measures for `greedy` seated exclusively at P1, which is roughly what the
one-sided exploration handicap should cost. Two independent harnesses now agree
on the greedy floor, which is the strongest evidence yet that §3a's diagnosis
was right and that this harness measures what it claims to. (Run on a loaded box
— load average ~8 with a T2 curve run and a ponder gauntlet in flight — which
affects a `ms:`-budgeted arm's absolute numbers but not the diversity count.)

**What a seed pins, and what it cannot.** Two 10-game runs at the same
`--explore-seed 424242`, one vsbot and one Go bot each, reproduced **8 of their
10 games byte-identically and in the same order** (both 10/10 distinct). The two
that diverged account for the whole difference in tally, 1-9-0 against 0-10-0.
That is the honest ceiling: the Go bot
searches on a wall clock, so a cross-play *game* is no more reproducible than
the arena's `ms:` mode. What the seed does pin is the **exploration schedule** —
which of our plies are overridden and by which legal action — as a pure function
of `(seed, game index, position)`, asserted in `crates/vsbot/src/explore.rs`'s
tests and in `crossplay --self-test`. Deriving the coin from the position rather
than from a running stream is what makes that true in a client, where a
superseded search can still be in flight when its replacement starts.

What is still missing is unchanged and structural: colours cannot be balanced
against the Go bot, because its challenger only targets its own pool.

## 4. S1 — Rust MCTS vs the Java gen-5 champion, same net, 1 s/move

`superiority.md` S1: dethrone the Java champion at 1 s a move. Both sides run
**the same artifact** — `mcts_champion.json`, md5 `748c9289…`, verified
byte-identical in `artifacts/`, in the `nnue-trainer` checkout and baked into
the image — so this isolates *implementation throughput*, not net quality. The
Java bot has no JVM on this host and runs from
`ghcr.io/korjavin/nnue-trainer:latest` with `SEARCH=MCTS MCTS_VALUE=net
MCTS_MOVE_MILLIS=1000`; it logged `prior=mcts_champion.json+value cpuct=1.5`,
confirming it played the champion's value head and not a fallback.

Both bots connect to a **local** Go server booted by the harness — never
production. Colour-balanced by `--direction alternate`: half the games with
`vsbot` challenging (P1), half with the Java bot challenging (P2). Four
independent shards, pooled.

```bash
docker pull ghcr.io/korjavin/nnue-trainer:latest
cargo build --release -p vsbot
python3 crates/virus-arena/crossplay/crossplay_pool.py \
    --shards 4 --games 400 --opponent java --direction alternate \
    --search MCTS --move-millis 1000 --vsbot-instances 1 \
    --workdir /tmp/s1run
```

```json
{ "wins": 235, "losses": 165, "draws": 0, "games": 400,
  "as_p1": 200, "as_p2": 200,
  "win_rate_as_p1": 100.0, "win_rate_as_p2": 17.5,
  "pooled_score": 0.5875, "win_rate": 58.75,
  "wilson95_low": 53.86, "wilson95_high": 63.47,
  "distinct_games": 65, "red_flag_terminations": 0 }
```

| | |
|---|---|
| W-L-D | **235-165-0** over 400 games |
| Pooled `(W+0.5D)/N` | **0.5875** |
| Seats | 200 as P1, 200 as P2 — balanced |
| vsbot as P1 | **200 / 200 = 100.0%** |
| vsbot as P2 | 35 / 200 = 17.5% |
| Wilson 95% over 400 | [53.9%, 63.5%] |
| **Distinct games** | **65 of 400** |
| Wilson 95% over the *effective* 65 | **[46.3%, 69.6%]** |
| Forfeits (illegal move / timeout) | 0 |
| Median game length | 23 turns |

**Measured throughput: Rust runs ~23× the simulations of Java at the same
budget on the same net.** Java logged a median **163 sims** per 1 s move
(p10 149, p90 171, over 13 953 searches); `virus-mcts`'s microbench puts Rust
at **3797 net-value sims/s**. The comparison is generous to Rust — its figure
is a single-core microbench on a quiet box, Java's is in-game and carries JVM
and protocol overhead — but the order of magnitude is not in doubt. (Both
sides' raw sim counters spike into the millions in decided endgames, where a
simulation reaching a known-terminal node returns a cached value; that is why
the **median** is quoted, not the mean.)

**Verdict: S1's stated bar is met on the letter and not on the substance, so
this page does not crown a champion.**

* The pooled score **0.5875 clears the ≥ 0.55 acceptance over ≥ 400 games**,
  the run is seat-balanced, and there were zero forfeits.
* But only **65 of the 400 games were distinct.** Cross-play has no opening
  randomisation and both engines play argmax with no root noise, so the run
  replays a small set of games — the same defect that produced the 49-1 in §3a,
  caught this time by the harness's own diagnostic. Over the effective sample
  the interval is **[46.3%, 69.6%]**, which straddles both 0.55 *and* 0.50.
  Per CLAUDE.md (gauntlets only, ≥400 games) the game count is not the sample
  size, so **this may not gate a promotion.**

**What the seat split actually shows.** The P1 seat won **365 of 400 games
(91.3%)**. With the same net on both sides the engines are close enough that
moving first almost decides the game, and the entire margin is conversion of
the first move: **Rust converted 200/200 (100%) of its P1 games, Java converted
165/200 (82.5%)** of its. So the honest reading of the 23× sims advantage is
that it buys Rust near-perfect conversion of a won seat, and very little else —
it recovered only 17.5% of games from the losing seat. A throughput advantage
of that size producing a pooled 0.5875 is a *negative* result for the "more
sims wins" thesis, and is the loss analysis S1 asked for.

**What would make this gate-eligible**: an opening-randomisation lever on at
least one side, so 400 games are 400 samples. **That lever now exists** —
§3b — and this row predates it, so **the 0.5875 above is superseded and may not
gate.** Re-running S1 on `--explore-eps 0.15` is bd `vsbot-x70`'s job; the
number it returns will be a floor, because only vsbot carries the opening noise.
Until it is re-run, the distinct-game count is the number to read.

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
| Anything at 400 games **in the arena** | Wall clock. Row 2 took 24 min for 100 games on 4 cores, so 400 is ~1.6 h; row 1 is ~33 min. Nothing blocks it but time. (Row 4 is at 400 games, but through cross-play, where the game count is not the sample size.) | A longer run |
| Cross-play with balanced colours **against the Go bot** | Structural, and now confirmed by reading the code rather than assumed: the bot-hoster's challenger targets `Manager.IsAcceptor(userID)`, which is false for every id outside its own pool, so it cannot challenge `vsbot` and cannot seat it at P2. `--direction theirs` is refused for that opponent rather than quietly returning another one-chair number | a bot-hoster change |
| A cross-play number that is a strength result | The diversity half is **done** (§3b): `--explore-eps` makes 50 games 50 distinct games. What is left is the colour half — against the Go bot the chair cannot be alternated at all, and against the Java bot it can, so an `--direction alternate` S1 re-run on the new lever is the next real number. Every cross-play figure recorded before §3b is superseded | re-running §4 with `--explore-eps 0.15`; a bot-hoster change for the Go arm |
| Two different net artifacts in one gauntlet | The harness shares one loaded net across all games and threads; a net-vs-net run needs a second one threaded through the sides. Refused with an error rather than silently playing one artifact against itself | a follow-up bead |
| Rust:Java throughput ratio (superiority.md S0) | Needs criterion benches in `virus-mcts`, which are that bead's scope, not this one's | S0 |
