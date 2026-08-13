//! Enhanced iterative-deepening alpha-beta search for the virus game.
//!
//! # Two modes, one implementation
//!
//! The crate is deliberately built around a **plain** mode and an **enhanced**
//! mode sharing one search core:
//!
//! * **Plain** is a literal port of Go's `virusgame/backend/search/search.go`.
//!   It is the *oracle*: [`choose_depth`] reproduces
//!   `fixtures/gobot_search_parity.jsonl` (412 records, move and score exact)
//!   and [`choose_node_budget_plain`] reproduces
//!   `fixtures/gobot_nodebudget_parity.jsonl`. CLAUDE.md treats a parity break
//!   as a hard failure, never a rounding difference.
//! * **Enhanced** (the default, [`SearchOptions::default`]) is a port of Java's
//!   `GoBotSearcher` — measured at +64% strength over the baseline at equal
//!   time. Every feature is gated on [`SearchOptions::enhanced`], so the oracle
//!   stays reachable and any strength regression can be bisected against it.
//!
//! Keeping both alive is the point. The Go engine and the Java port each shipped
//! a search rewrite that silently changed the played move; without a byte-exact
//! oracle to diff against, neither was caught before it cost games.
//!
//! # The enhanced stack
//!
//! | Feature | Where |
//! |---|---|
//! | Staged movegen (TT move searched before siblings materialise) | [`Searcher`] |
//! | Packed lockless transposition table, 2^21 entries, cross-move | [`tt`] |
//! | Killers (2 per ply) + per-mover per-cell history | [`Searcher`] |
//! | Turn-aware late-move reduction | [`Searcher`] |
//! | Aspiration windows (delta = [`ASPIRATION_DELTA`]) | [`Searcher`] |
//! | Lazy SMP on the shared table (off by default) | [`SearchOptions::smp_threads`] |
//! | Soft deadline + guarded partial-iteration salvage | [`Searcher::search_with_deadline`] |
//! | The wedge opening book | [`book`] |
//!
//! Move-ordering tiers, highest first: TT move `+10_000_000`, killer
//! `+5_000_000`, immediate win `+1_000_000`, eliminations caused
//! `x100_000`, capture `+10_000`, history `<=9_000`, turn continuation `+100`.
//!
//! # The three invariants this crate exists to honour
//!
//! 1. **The mover does not alternate per ply.** A turn is three actions, so
//!    roughly 53% of legal children keep the mover. The search branches on
//!    `maximizing = state.current_player() == root_player` and never negates.
//!    Three- and four-player games run max^n over a `[Score; 4]` vector with no
//!    pruning beyond the exact mate cutoff.
//! 2. **A deadline search returns what the fixed-depth search returns at the
//!    depth it completed.** Go's wall-clock `choose()` bug made a parity-perfect
//!    engine lose 0-10 live. An aborted iteration is discarded; salvage happens
//!    only when `best_score > alpha_orig`.
//! 3. **Transposition discipline.** The key covers `moves_left`, `neutral_used`
//!    and the side to move; mate scores are rebased on store and probe; and a
//!    `PlaceNeutrals` TT move never takes the staged fast path, because the
//!    search enumerates only a curated subset of neutral pairs.
//!
//! # Example
//!
//! ```
//! use virus_core::State;
//!
//! let state = State::new(12, 12, 2).expect("12x12 two-player board");
//! // The oracle: deterministic, fixed depth, no opening book.
//! let plain = virus_search::choose_depth(&state, 2).expect("a legal action exists");
//! assert!(plain.action.is_some());
//!
//! // Production: the full strength stack under a node budget.
//! let enhanced = virus_search::choose_node_budget(&state, 20_000).expect("a legal action");
//! assert!(enhanced.action.is_some());
//! ```

#![deny(missing_docs)]

pub mod book;
mod searcher;
pub mod tt;

pub use searcher::{
    choose, choose_depth, choose_depth_with, choose_node_budget, choose_node_budget_plain,
    RootMove, SearchOptions, SearchResult, SearchStats, Searcher, ASPIRATION_DELTA, INF_SCORE,
    MAX_DEPTH, SMP_THREAD_CAP, SOFT_DEADLINE_PERCENT,
};

/// The search score type, re-exported so callers need not depend on
/// `virus-eval` directly. Always integral — a float score would make byte-exact
/// parity with the Go and Java oracles impossible.
pub use virus_eval::Score;

/// Terminal-position magnitude. A win at ply `p` scores `MATE_SCORE - p`, so a
/// faster mate always outranks a slower one.
pub use virus_eval::MATE_SCORE;
