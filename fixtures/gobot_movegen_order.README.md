# gobot_movegen_order.jsonl — GoBot move-enumeration + transition oracle

412 records, one per record of `gobot_search_parity.jsonl` (same index). Unlike
that fixture — which records what GoBot *chose* — this one records what GoBot
*enumerated*, in order, and what its rules engine produced when the recorded
action was applied.

It exists because search parity depends on identical **child ordering**, not
just on the final move: equal-scoring siblings are resolved "first wins", so a
reordered move list silently changes the played move. `virus-core`'s
`tests/movegen_parity.rs` checks every record byte-for-byte.

## Record schema

```json
{
  "index": 0,               // line index into gobot_search_parity.jsonl
  "legalCount": 3,          // len(State.LegalActions()) — exact, uncurated
  "strategic": false,       // true when the curated neutral-pair set replaced
                            // exact enumeration (moves + C(owned,2) > 32)
  "stateHash": 4483…,       // GoBot search.stateHash of the position
  "afterHash": 1461…,       // …and of state.Apply(<the recorded action>)
  "searchActions": [["M",0,1], ["M",1,0], ["N",3,4,5,6]]
}
```

`["M", row, col]` is a move; `["N", r1, c1, r2, c2]` is a neutral placement with
the pair in GoBot's emitted order. The list is exactly what
`game.Position.ForEachSearchAction` yielded.

`stateHash`/`afterHash` are FNV-1a over
`rows, cols, current, movesLeft, active|0x10, neutralUsed|0x20, cells`, i.e. the
key GoBot's transposition table uses. Matching `afterHash` on all 412 records
pins down the whole transition: capture-fortifies, neutral-consumes-turn,
elimination-leaves-cells and turn advance all feed it.

111 of the 412 records take the `strategic` branch, so the Tarjan-based
neutral-pair curation is covered as well as the exact path.

## Regenerating

The generator links the **real** Go engine (`game.FromSnapshot`,
`game.NewPosition`, `game.Position.ForEachSearchAction`, `game.State.Apply`), so
the golden file is GoBot's own output rather than a re-derivation.

```bash
cd fixtures/tools/ordergen
GOTOOLCHAIN=go1.24.0 go build -o ordergen .    # needs ~/Project/virusgame checked out
./ordergen -in ../../gobot_search_parity.jsonl -out ../../gobot_movegen_order.jsonl
```

`go.mod` has a `replace` pointing at `~/Project/virusgame/backend`; adjust it if
your checkout lives elsewhere.
