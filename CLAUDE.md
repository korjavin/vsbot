# vsbot

Rust bot for the virus game. Read `ARCHITECTURE.md` first — especially the
"Non-negotiable invariants" section; every item there caused a real production
bug in the Go or Java predecessor.

## Build & test

- Toolchain: stable Rust via rustup (`~/.cargo/bin`).
- `cargo build --workspace` / `cargo test --workspace` must be green before any PR.
- `cargo clippy --workspace -- -D warnings` and `cargo fmt --check` are CI gates.
- Parity tests (fixtures under `fixtures/`) are hard gates — a parity break is
  never "close enough"; integer paths must match exactly.

## Conventions

- Workspace crates: `virus-core`, `virus-eval`, `virus-search`, `virus-mcts`,
  `virus-proto`, `virus-arena`, `virus-selfplay`, `vsbot` (bin). Keep dependency
  direction: core ← eval ← search/mcts; proto, arena and selfplay depend on the
  engine crates, never the reverse.
- No `unsafe` without a comment justifying it and a test covering it.
- All engine randomness is seeded and deterministic; production play paths take
  no RNG unless explicitly configured (exploration).
- Env-var config is read in one place (the `vsbot` bin), passed down as structs.
- Never gate strength claims on offline metrics; gauntlets only (≥400 games).

## Issue tracking

Uses bd (beads), prefix `vsbot-`. Close reasons must record what merged and what
was deferred.
