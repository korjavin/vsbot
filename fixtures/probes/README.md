# `fixtures/probes` — the `PlaceNeutrals` regression probe set

`neutrals-v1.jsonl` is a fixed set of **neutral-decision positions**: positions
where the side to move may still spend its once-per-game `PlaceNeutrals`
(`movesLeft == 3`, `neutralUsed == false`). It exists to make bd `vsbot-07x` —
the gen-5 champion's taste for unmotivated neutral placements — measurable, and
therefore trainable.

**It is not a gate.** ARCHITECTURE.md invariant 7: only >=400-game gauntlets
say anything about strength. `probe run` prints that caveat on every
invocation and exits 0 whatever the numbers say. See `docs/probes.md` for the
gen-5 baseline and the analysis.

## Schema

One JSON object per line, no header line:

| field | meaning |
|---|---|
| `id` | stable identifier, unique in the set |
| `source` | `games-db` \| `ponder-repro` \| `live-owner-game` |
| `class` | `lost-advantage` \| `kept-advantage` \| `champion-chose-neutral` |
| `snapshot` | the position, in the same wire form the live client decodes |
| `provenance` | origin string, game id, seat names, turn, seed, the pair actually played |
| `labels` | the mining heuristic's measurements (see below) |

The position is carried as a `virus_core::Snapshot` rather than a bespoke
board encoding, so loading a probe re-runs the *same* validation the live
client runs on a server snapshot (ARCHITECTURE.md invariant 5). `probe run`
additionally refuses any record whose position cannot host a neutral decision,
so a stale or mis-mined fixture fails loudly instead of quietly measuring
nothing.

`labels` records `advantageBefore/After/Swing` (the mover's cell lead),
`immediateCost` (the advantage change over the mover's very next turn — the
action's mechanical price), `horizonTurns`, and `placerWon`. The heuristic,
including why the class label leans on the game's outcome rather than on the
material swing alone, is documented in `crates/virus-arena/src/probes.rs`.

## What is in v1

| source | n | what it is |
|---|---|---|
| `games-db` | 38 | real recorded games from the published snapshot, mined by the heuristic (30 suspects + 8 controls) |
| `ponder-repro` | 8 | positions the gen-5 champion itself answered with a neutral, on the `ponderrepro` trajectory generator |
| `live-owner-game` | 2 | live games where a non-bot-named seat faced a `SuperiorBot` seat, mined by the same heuristic |

48 positions in total.

### The `live-owner-game` half, added 2026-08-14

v1 originally shipped this source **empty**: bd `vsbot-07x` named the owner's
2026-08-13 game against a `SuperiorBot` seat as repro material, and the
published `games.db` at `https://vs.wandergeek.org/data/games.db`
(`Last-Modified: Sun, 09 Aug 2026 19:01:42 GMT`) stopped before it. The prod
database was then recovered by WAL replay to `work/gamesdb/games.db` — 2041
games, through 2026-08-14 08:53 — and the games are in that copy.

Two things a reader should know before quoting these two positions:

- **The bead's game identifier does not resolve.** No seat named `SuperiorBot
  Bot 1079` exists anywhere in the recovered corpus. The positions here are from
  `SuperiorBot Bot 1220` (2026-08-13) and `SuperiorBot Bot 7420` (2026-08-14).
- **Four neutral turns collapse to two positions.** Three of the four qualifying
  games reached the *identical* position at turn 6 — the opposing seats played
  an identical opening despite web-player-style names — and the bot answered
  each with the same pair, `(8,8)+(11,10)`. The miner's position hash dedups
  them. That repetition is one deterministic client, not four observations.

`docs/probes.md`, "The owner's live games", has the full identification and what
gen-5 says about the positions.

A seat is read as a bot iff its name contains "bot", which is the server's own
naming (`Bot 9261`, `NNUE Bot 1039`, `SuperiorBot Bot 1220`; a web player gets
`WiseBuffalo50`). It is a heuristic on a display string, it selects which games
are mined rather than what is measured, and each position records its seat names
so a reader can check.

## Regenerating

```bash
# 1. the corpus (records its own Last-Modified in the dump's meta line)
curl -sSD headers.txt -o games.db https://vs.wandergeek.org/data/games.db
python3 fixtures/probes/tools/dump_games.py \
    --db games.db --out games-dump.jsonl --only-neutral \
    --last-modified "$(grep -i '^last-modified:' headers.txt | cut -d' ' -f2-)"

# 2. the mined half (deterministic for a given dump)
cargo run --release -p virus-arena --bin probe -- mine-db \
    --games games-dump.jsonl --out db-probes.jsonl

# 3. the self-play half (deterministic for a given net, seed and sim count)
cargo run --release -p virus-arena --bin probe -- mine-play \
    --net artifacts/mcts_champion.json --out play-probes.jsonl \
    --games 8 --sims 800 --max 12

# 4. the live half — the same miner over a corpus narrowed to live games.
#    The recovered prod copy is NOT the published snapshot, so --source says so:
#    that string lands verbatim in every position's provenance.
python3 fixtures/probes/tools/dump_games.py \
    --db work/gamesdb/games.db --out live-dump.jsonl --only-neutral \
    --seat-name SuperiorBot --human-seat \
    --source "prod games.db recovered by WAL replay (work/gamesdb/games.db)" \
    --last-modified "Fri, 14 Aug 2026 08:53:08 GMT"
cargo run --release -p virus-arena --bin probe -- mine-db \
    --games live-dump.jsonl --out live-probes.jsonl \
    --source live-owner-game --max-suspect 30 --max-control 30

# 5. the set
cat db-probes.jsonl play-probes.jsonl live-probes.jsonl \
    > fixtures/probes/neutrals-v1.jsonl
```

All three mining steps are deterministic, so step 5 reproduces byte-for-byte
from the same corpora and net. **Do not regenerate `neutrals-v1.jsonl` in place
when comparing generations** — the whole point of a regression probe is that the
positions do not move. A new corpus or a new heuristic earns `neutrals-v2`.

Step 4 was added after v1 shipped, and it *appends*: the 46 records written by
steps 2–3 are byte-identical to what v1 shipped, so every per-position number
recorded against v1 still refers to the same position. Only the aggregate
denominator moved, from 46 to 48, and `docs/probes.md` reports the halves
separately for exactly that reason.

## Running it

```bash
cargo run --release -p virus-arena --bin probe -- run \
    --set fixtures/probes/neutrals-v1.jsonl --sims 192 --sims 1000 \
    --jsonl reports.jsonl
```

`192` is the deployed action's simulation floor and `1000` is a
comfortably-deeper reference; the bead's finding is that the extra search does
not rescue the decision, so both are reported side by side.
