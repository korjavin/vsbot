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

<!-- RUN1 -->

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

<!-- RUN2 -->

## 3. Cross-play: vsbot vs the Go bot, through the real server

Not an engine comparison — a **plumbing check**. The Go server refereed, the Go
bot-hoster ran its default accept-only pool, and `vsbot` challenged. W-L-D was
read off the server's own `games.db`.

```bash
cargo build --release -p vsbot
python3 crates/virus-arena/crossplay/crossplay.py --games 50 --search GREEDY
```

<!-- RUN3 -->

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

<!-- DETERMINISM -->

`ms:` mode reproduces nothing. That is what a wall clock means, and a test
pretending otherwise would fail on a loaded box and teach everyone to ignore it.
Fixed-time rows above are single observations; re-running them will give
different numbers within the interval.

## What is not measured yet

| Gap | Why | Unblocked by |
|---|---|---|
| Anything at 400 games | Wall clock. A 400-game fixed-time run is ~4 h on 4 cores. | A longer run, or more cores |
| Cross-play with a real engine | `vsbot`'s `build_engine` rejects `SEARCH=ALPHABETA` and `SEARCH=MCTS` until the engine wiring lands, so the cross-play arm ran `SEARCH=GREEDY` | the vsbot engine-wiring bead |
| Cross-play vs the Java bot | No JVM on this host; shipping an untested boot sequence would be worse than shipping none | a host with a JDK |
| Two different net artifacts in one gauntlet | The harness shares one loaded net across all games and threads; a net-vs-net run needs a second one threaded through the sides. Refused with an error rather than silently playing one artifact against itself | a follow-up bead |
| Rust:Java throughput ratio (superiority.md S0) | Needs criterion benches in `virus-mcts`, which are that bead's scope, not this one's | S0 |
