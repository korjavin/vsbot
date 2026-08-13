# virus-arena

The gauntlet harness. Every strength claim this project makes comes from here.

ARCHITECTURE.md invariant 7 is why: seven separate offline metrics were each
believed to predict playing strength and each one was wrong. CLAUDE.md's rule is
"never gate strength claims on offline metrics; gauntlets only (≥400 games)".
This crate is that gauntlet, and it is built to make the *undisciplined* use of
it awkward — a run under 100 games prints `INFORMATIONAL ONLY` and refuses to
state a direction, no matter how nice the number looks.

## Rust vs Rust: the `arena` binary

```bash
cargo build --release -p virus-arena

# Enhanced alpha-beta against the plain oracle at an equal node budget.
./target/release/arena \
    --a ab-enhanced --b ab-plain \
    --budget nodes:60000 --games 400 --seed 11 --threads 4

# MCTS at 1 s/move against enhanced alpha-beta at 1 s/move.
./target/release/arena \
    --a mcts:artifacts/mcts_champion.json --b ab-enhanced \
    --budget ms:1000 --games 400 --seed 20260813 --threads 4
```

`--help` lists every flag. Sides are `greedy`, `ab-plain`, `ab-enhanced`, and
`mcts[:artifact.json]`; budgets are `nodes:<n>`, `depth:<d>` and `ms:<n>`, and
each side can carry its own (`--a-budget` / `--b-budget`).

### What the harness does, and why each part is there

**Colours are paired.** Games `2k` and `2k+1` are the same opening seed with the
seats swapped, so first-mover advantage cancels *within* the pair rather than
being assumed to wash out. `--games` is rounded up to even for the same reason:
one unpaired game is a deterministic bias no amount of sampling removes.

The load-bearing consequence: **a configuration played against itself reads
exactly 50%**. Both games of a pair are literally the same game, so the same
seat wins both. `tests/determinism.rs` asserts this, and if it ever fails, state
is leaking between games and every number the harness has printed is suspect.

**Seeds are derived, not incremented.** The per-game seed is Java's

```
mix64(seed ^ (0x9E3779B97F4A7C15 * (game / 2 + 1)))
```

`game / 2` is the pair index, so a pair shares one opening. The golden-ratio
stride through a full-avalanche mixer is what makes base seeds 1 and 2 produce
unrelated openings — `seed + k` did not, and two "independent" runs launched at
nearby seeds turned out to be replaying each other's games (`nnue-trainer-riy`).
**When you pool two runs, space their `--seed` values far apart.**

**Openings are randomised, seeded.** Two deterministic engines replay one game
forever, so 400 repetitions of it are one sample and not 400. With probability
`--eps` (0.15) over the first `--explore-turns` (8) turns, a uniformly random
legal action is played instead of the searched one. The search still runs on
those plies and its answer is discarded — that is deliberate, because the
enhanced searcher accumulates a table, killers and history across its moves, and
skipping the search would leave a pair's two colours in different states.

**Draws are not half-wins in the headline.** A draw raises the denominator and
not the numerator, so a drawish engine's interval sags rather than sitting at a
comfortable 50%. The half-win pooled score `(W + 0.5·D)/N` that Gate A in
`docs/plans/superiority.md` compares against 0.55 is reported alongside it, as
its own labelled number.

**Wilson 95%, not the normal approximation.** Constant for constant from Go's
`arena.Wilson95`. At the extremes the normal interval leaves `[0, 100]` and
collapses to zero width at 0 wins, which reads as certainty.

**Capped games are draws.** Not territory decisions. Territory at the cap
answers "who was ahead in a game nobody finished", which is a different question
from who won. The territory verdict is still carried on each `GameOutcome` for
corpus recording, and the cap-hit count is reported separately so a run decided
by the cap cannot be mistaken for one decided by the engines.

**The cap and the opening window are counted in turns, from the actual seat
changes.** Java derives them as `turns × 3` plies, which is wrong given a rule
its own engine has: `PlaceNeutrals` spends a whole turn in a single action, so a
game with placements fits more than `--max-turns` turns inside `max_turns × 3`
plies. The drift is at most one turn per seat per game — but the flags say
turns, so they mean turns. `plies × 3` is kept only as a runaway guard.

### Determinism

In `nodes:` or `depth:` mode a run is a pure function of its configuration:
games are independent, seeded from `(seed, index)`, and no engine consults a
clock. `--threads` therefore changes only the wall time — outcomes are folded in
index order, never completion order. Lazy SMP is off in every arena arm and
`tests/determinism.rs` asserts it, because SMP helper threads write the shared
transposition table and a search with it on is not reproducible run to run.

`ms:` mode reproduces nothing, by construction. That is what a wall clock means.
It is the mode you need for cross-engine comparisons — an MCTS simulation and an
alpha-beta node are not the same unit of work — and it is the mode
`docs/plans/superiority.md` S0 called out as the gap that cost the Java harness
its first step. Each side enforces its deadline per action and the worst overrun
across the run is printed, so a side that quietly ignores the clock is visible
in the report instead of silently winning on time.

### Sample-size discipline

| games | label | what it may do |
|---|---|---|
| < 100 | `INFORMATIONAL ONLY` | be reported as a raw tally, and nothing else |
| 100–399 | `indicative` | state a direction with the interval attached |
| ≥ 400 | `gate-eligible` | gate a promotion (SE ≈ ±2.5 pts, so 0.55 ≈ 2σ) |

## Cross-play: `crossplay/crossplay.py`

Rust-vs-Rust settles internal A/Bs, but the bar this project has to clear is the
*deployed* Go bot, and that bot only exists behind a WebSocket server. So the
harness runs the real thing: it boots the Go server, boots the Go bot-hoster in
its default accept-only mode, points `vsbot` at it as a challenger, and reads
W-L-D off the server's own `games.db`. Nothing reimplements a rule or an
outcome — the server decides who won, exactly as in production.

```bash
cargo build --release -p vsbot
python3 crates/virus-arena/crossplay/crossplay.py --games 50 --search GREEDY
```

Or through cargo, env-gated the way `crates/vsbot/tests/live_games.rs` is:

```bash
VSBOT_CROSSPLAY=1 cargo test -p virus-arena --test crossplay -- --nocapture
```

| Variable | Default |
|---|---|
| `VSBOT_CROSSPLAY` | unset — the test skips |
| `VSBOT_CROSSPLAY_GAMES` | `50` |
| `VSBOT_CROSSPLAY_SEARCH` | `GREEDY` |
| `VSBOT_CROSSPLAY_TIMEOUT` | `1800` |
| `VSBOT_ITEST_BACKEND` | `$HOME/Project/virusgame/backend` |

### Opening diversity: why a game count is not a sample size

Both deployed engines play argmax with no root noise, so with nothing
randomising the opening a run **replays one game**. Measured (bd `vsbot-t3q.2`):
the 400-game S1 run contained 65 distinct games, and the 50-game run behind the
`49-1` contained 5. Every Wilson interval taken off the game count was therefore
far too narrow.

`vsbot` now carries the arena's lever: `--explore-eps` (default `0.15`) and
`--explore-turns` (default `8`) drive `VSBOT_EXPLORE_EPS` /
`VSBOT_EXPLORE_TURNS`, and inside that window it plays a uniformly random legal
action instead of the searched one. **50 of 50 games are now distinct** on the
Go-bot smoke, against 5 before.

Three details are load-bearing:

* **The window is counted in vsbot's own turns**, not the game's — a client
  never sees the opponent's turns. Eight of ours is ~24 coin flips, the same
  opening noise per game an `arena` run spends across *both* its sides, which is
  what it takes for one-sided exploration to reach the arena's diversity.
* **Seeds are derived, not reused.** Every phase, every `--vsbot-instances`
  process and every `crossplay_pool.py` shard gets a disjoint stream mixed off
  `--explore-seed` (default `20260813`); the per-game seed inside a process is
  `mix64(seed ^ GOLDEN_GAMMA * (game + 1))`. Two vsbots on one seed would derive
  the same schedule and replay each other, and `seed + k` is what caused
  `nnue-trainer-riy`. `--self-test` covers the derivation, so CI does too.
* **Only our side explores, on purpose.** The Go hoster's
  `BOT_EXPLORE_EPSILON` applies to every turn of every game from an unseeded
  global RNG (`--opponent-explore-eps` exposes it; it is never right for a
  gating run), and the Java bot's `CHALLENGER_EXPLORE` only reaches its
  `SEARCH=GOBOT` path, so it does nothing under `SEARCH=MCTS`. The asymmetry
  handicaps `vsbot`, which is the safe direction for a "vsbot is stronger"
  claim.

A seed pins the **exploration schedule** — which of our plies are overridden and
by which legal action — and not the games: a fixed-time engine against a live
opponent is a wall-clock measurement, exactly as `ms:` mode is in the arena.
`--explore-eps 0` reproduces the pre-fix harness, and the low-diversity warning
then says so.

Exploration is refused together with `VSBOT_PONDER`: a pondering session answers
actions without ever consulting the exploration wrapper, so the two together
would silently randomise nothing.

It is a script rather than an `arena` subcommand because it orchestrates three
external processes and reads an SQLite file — which this crate would otherwise
need a C dependency to open, for a job Python's standard library already does.

Four things it fixes relative to `nnue-trainer/eval_java_vs_go.py`, which it is
ported from:

* **The name filter is not seat-ordered.** The original counted only rows where
  its bot sat in seat 1 and silently discarded every game from the other chair.
  Both orders are counted here and the per-seat split is printed.
* **Draws are counted.** The original dropped `result = 0` from both numerator
  and denominator, so a drawish opponent looked like an even one.
* **Disconnects are discarded.** The poll that ends a run and the SIGTERM that
  follows it are not atomic, so an in-flight game lands in the table as a
  `disconnect` loss for whoever was killed first. `illegal_move` and `timeout`
  *are* counted — they are instant forfeits in production — but reported
  separately, because a run full of them is a bot problem, not an opponent.
* **The port is allocated, not assumed.** The Go server hard-codes `:8080`; the
  script builds it through `go build -overlay` with a patched `main.go` (the
  checkout on disk is never written to) so a run cannot fight whatever the
  developer already has listening. Same trick, same anchor line, as
  `crates/vsbot/tests/live_games.rs`.
* **Shutdown escalates.** The original sent `SIGTERM` to the process groups and
  never checked; a server holding a WebSocket read would survive it.

### Known limitations

**Colours cannot be balanced, so this number is not an `arena` number.** The
server seats the challenger at P1; only `vsbot` challenges (the Go pool is
accept-only so it does not spar with itself, and a Go challenger only ever
targets one of its own acceptors). Every cross-play game is therefore
vsbot-as-P1 against GoBot-as-P2, and P1 moves first on an empty board. The
`arena` gauntlet cancels that by pairing; this harness structurally cannot.
The counting handles both seat orders — free, and correct the day the server can
alternate — and until then the script prints a loud warning whenever the split
is lopsided. **Read a cross-play result as a plumbing check, not as strength.**

**`--vsbot-instances > 1` wastes roughly half its games.** `vsbot`'s challenger
picks any idle peer, including another `vsbot`, so a multi-instance run spends
about half its time on vsbot-vs-vsbot games. The name filter excludes them from
the tally — they are never miscounted, only wasted — and the trade for
concurrency is usually still worth it. (Each instance does get its own
exploration stream, so the wasted games are not also correlated ones.)

`SEARCH=ALPHABETA` and `SEARCH=MCTS` are rejected by the `vsbot` binary until
the engine wiring lands (`build_engine` in `crates/vsbot/src/main.rs`), so the
cross-play arm currently runs `SEARCH=GREEDY` — enough to prove the plumbing,
not a strength result. The script takes `--search` and needs no change once the
wiring merges.

Cross-play against the **Java** bot (`nnue-trainer` with `SEARCH=MCTS`) is not
implemented: there is no JVM on this host, so the arm could not have been run,
and shipping an untested boot sequence for it would be worse than not shipping
one. The script's process-spawning structure takes a third arm directly.
