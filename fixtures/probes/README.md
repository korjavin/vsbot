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
| `games-db` | 38 | real recorded games, mined by the heuristic (30 suspects + 8 controls) |
| `ponder-repro` | 10 | positions the gen-5 champion itself answered with a neutral, on the `ponderrepro` trajectory generator |
| `live-owner-game` | 0 | **absent — see below** |

### The owner's live game is not in here, and why

bd `vsbot-07x` names the owner's 2026-08-13 live game against `SuperiorBot Bot
1079` as repro material. It is **not** in the published corpus. The
`games.db` served at `https://vs.wandergeek.org/data/games.db` carries
`Last-Modified: Sun, 09 Aug 2026 19:01:42 GMT`, its newest `started_at` is
`2026-08-09 19:01:04`, and no row anywhere in it names `SuperiorBot` or `Bot
1079`. The published file is a periodic snapshot that had not been refreshed
past 2026-08-09 when this set was built, so that game could not be reached.

This is a provenance gap, not a silent omission: the `live-owner-game` source
exists in the schema and is empty. When a refreshed `games.db` (or the game's
PGN by any other route) becomes available, re-run the recipe below and the
positions land under that source without any code change.

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

# 4. the set
cat db-probes.jsonl play-probes.jsonl > fixtures/probes/neutrals-v1.jsonl
```

Both mining steps are deterministic, so step 4 reproduces byte-for-byte from
the same corpus and net. **Do not regenerate `neutrals-v1.jsonl` in place when
comparing generations** — the whole point of a regression probe is that the
positions do not move. A new corpus or a new heuristic earns `neutrals-v2`.

## Running it

```bash
cargo run --release -p virus-arena --bin probe -- run \
    --set fixtures/probes/neutrals-v1.jsonl --sims 192 --sims 1000 \
    --jsonl reports.jsonl
```

`192` is the deployed action's simulation floor and `1000` is a
comfortably-deeper reference; the bead's finding is that the extra search does
not rescue the decision, so both are reported side by side.
