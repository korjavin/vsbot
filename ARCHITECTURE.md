# vsbot — Rust virus-game bot

Goal: a bot that beats every existing bot (Go `~/Project/virusgame/backend`, Java `~/Project/nnue-trainer`) and the human owner, deployed via Docker/Portainer against `vs.wandergeek.org`.

## Why Rust, and what "best of both" means

Both existing engines are dominated by per-node allocation (full board clone + multi-BFS per node). Rust removes that tax; every saved microsecond converts to search depth / MCTS sims, which is the proven strength lever. We port the two things that measurably won:

1. **Enhanced alpha-beta stack** (Java `GoBotSearcher`, +64% vs baseline at equal time): staged movegen (TT-move-first), packed lockless XOR transposition table with generation aging + cross-move reuse, killers + history, turn-aware LMR (never reduce turn-ending moves), aspiration windows (δ=1500), lazy SMP, deadline salvage.
2. **MCTS/PUCT + conv policy+value net** (the current prod champion, `mcts_champion.json` gen-5): per-action nodes, absolute-frame backup (single flip at the leaf, no per-edge negation), PUCT cpuct=1.5, Dirichlet noise in self-play only.

Explicitly **not** ported: NNUE distillation (v1/v2/v3 all failed — labels are the binding constraint; see nnue-trainer `docs/nnue-v3-deep-labels.md`), the legacy negamax `SearchEngine`, frequency-ranked pattern features.

## Crate layout

```
virus-core    rules engine: state, movegen, apply, connectivity, elimination,
              Zobrist, snapshot decode/validate, strategic neutral-pair curation
virus-eval    integer-exact port of the hand-tuned eval (Voronoi space-race,
              Tarjan articulation cut-loss, predatory cuts, threat tempo)
virus-search  enhanced iterative-deepening alpha-beta (list above) + wedge opening book
virus-mcts    PUCT searcher + conv policy/value net inference (f32, single trunk pass)
virus-proto   WebSocket bot client (snapshot-authoritative, version-gated cancellation)
virus-arena   gauntlet harness (color-paired seeds, SplitMix64 derivation, Wilson95)
vsbot         binary crate wiring it all together (env-configured)
```

## Game rules (authoritative summary)

- Rectangular rows×cols (server default 12×12), 2–4 players. Bases fixed corners in seat order: P1 (0,0), P2 (rows-1,cols-1), P3 (0,cols-1), P4 (rows-1,0).
- Cells: Empty, Normal(owner), Base(owner), Fortified(owner), Neutral. Base/Fortified are invulnerable; Neutral is dead space.
- **3 actions per turn.** Move: place on an Empty cell or capture an enemy Normal (becomes YOUR Fortified). Target must be 8-adjacent to your base-connected component (BFS from base over 8-neighbours of your own cells).
- PlaceNeutrals: convert two of your OWN Normal cells to Neutral. Once per game per player, only at turn start (movesLeft==3), consumes the whole turn.
- After every action, any active player with no legal move is eliminated; **eliminated players' cells stay on the board and remain capturable**. Last active player wins. Turn-capped games are decided by territory (`outcomeWinner`).
- Server: 120 s per-move timer; **an illegal move is an instant forfeit**.

## Non-negotiable invariants (each one caused a real production bug)

1. **The mover does not alternate per ply.** ~47% of legal children flip the mover, 53% don't. Alpha-beta must use `maximizing = state.current == root_player`; MCTS must use absolute-frame backup with a single sign application at selection. Never write naive negamax negation.
2. **Never act on your own `neutrals_placed` ack.** Its snapshot still shows you as mover with movesLeft>0; acting caused 2 live forfeits (2026-08-08). `turn_change` is the authoritative turn driver. *A pondering search runs on positions off that driver entirely, so it is given no way to emit: it can only reply to a request the turn driver made with `current == me` (`virus-proto` `ponder`).*
3. **Deadline search must return the same move fixed-depth search returns at the depth it completed.** The Go wall-clock `choose()` bug made a parity-perfect engine lose 0-10 live. Partial-iteration salvage only when `bestScore > alphaOrig`. *The MCTS analogue is fallback-first: a legal answer is selected before the long search starts and played if the search overruns.*
4. **Capture fortifies**: taking an enemy Normal makes it your Fortified (not Normal). Getting this backwards silently corrupted a whole training run.
5. **Snapshot is the only board source.** Never reconstruct from move deltas. Re-validate every snapshot; new snapshot cancels the in-flight search (version counter). Known server quirk: `Active[]` may report eliminated-but-still-owning-cells seats as active — derive activity defensively.
6. **TT discipline**: hash must include movesLeft/neutralUsed/side; mate scores rebased on store/probe; TT-move plausibility guard must reject PlaceNeutrals TT moves (search enumerates only a curated subset of pairs).
7. **Offline metrics do not predict strength** (7 documented failures). Only ≥400-game gauntlets with color-paired seeds and Wilson intervals count.

## Parity strategy

Fixtures copied from `~/Project/nnue-trainer/src/test/resources/` (each records its source weights in meta):
- `gobot_search_parity.jsonl` (412 records) + `gobot_nodebudget_parity.jsonl` — search parity
- `gobot_staticeval_parity.jsonl` — eval parity (integer-exact)
- `mcts/mcts_policy_parity.json`, `mcts/mcts_value_parity.json` — net inference parity
Rust must match byte-exact (ints) / within fp tolerance (nets) before any enhancement work counts.

## Reference source (read, do not re-derive)

- Go: `~/Project/virusgame/backend/{game,search,cmd/bot-hoster,arena}/`
- Java: `~/Project/nnue-trainer/src/main/java/com/engine/nnue_trainer/{search/gobot,mcts,search/eval,protocol}/`
- Research docs: `~/Project/nnue-trainer/docs/` (start: `nnue-v3-deep-labels.md`, `plans/20260807-mcts-az-feasibility.md`, `plans/20260807-search-strength.md`); `~/Project/virusgame/docs/nnue-go-experiment.md`; `~/Project/virusgame/backend/search/evaluate.go:7-43` (failed eval ideas).
