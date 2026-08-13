//! Pins the workspace contract: after warm-up, evaluation allocates **nothing**.
//!
//! GoBot allocates ~12 slices per leaf (four connectivity masks, five Tarjan
//! buffers, two Voronoi buffers, one queue). At the node rates a search needs,
//! that allocation traffic dominates the profile — it is the main reason the Go
//! and Java engines top out around 55-65k evals/s. [`EvalWorkspace`] exists to
//! make the steady state allocation-free, and an "optimisation" that quietly
//! reintroduces a `Vec::new()` in the hot path would be invisible without this
//! test.
//!
//! This lives in its own integration-test binary because the counting allocator
//! is process-global: sharing a binary with other tests would have their
//! allocations, on other threads, counted here.

use std::alloc::{GlobalAlloc, Layout, System};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use serde::Deserialize;
use virus_core::fixture::parse_jsonl;
use virus_core::{Cell, State};
use virus_eval::{evaluate_all, EvalParams, EvalWorkspace};

/// Counts allocations while armed. Deallocation is not counted: a
/// grow-in-place-free steady state is exactly what we are asserting, and
/// dropping fixture data mid-measurement would otherwise show up as noise.
struct CountingAllocator;

static ARMED: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if ARMED.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: a pure pass-through to the system allocator; the counter has
        // no effect on the allocation itself.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` came from `System.alloc` above with this same layout.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if ARMED.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: `ptr`/`layout` came from this allocator; `new_size` is the
        // caller's, unmodified.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct StaticEvalRecord {
    board: Vec<Vec<Cell>>,
    player: u8,
    moves_left: u8,
    neutral_used: Vec<bool>,
}

fn states() -> Vec<State> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures")
        .join("gobot_staticeval_parity.jsonl");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("reading {}: {error}", path.display()));
    parse_jsonl::<StaticEvalRecord>(&text)
        .unwrap_or_else(|(line, error)| panic!("line {line}: {error}"))
        .into_iter()
        .map(|record| {
            let rows = record.board.len();
            let cols = record.board[0].len();
            let cells: Vec<Cell> = record.board.iter().flatten().copied().collect();
            State::from_grid(
                rows,
                cols,
                2,
                &cells,
                record.player,
                record.moves_left,
                &record.neutral_used,
            )
            .expect("fixture record is a legal position")
        })
        .collect()
}

#[test]
fn reused_workspace_reaches_a_zero_allocation_steady_state() {
    let states = states();
    let params = EvalParams::default();
    let mut workspace = EvalWorkspace::new();

    // Warm-up: the first pass grows every buffer to the board size and the
    // Tarjan stack to its high-water mark.
    let mut warm = 0i64;
    for state in &states {
        warm ^= evaluate_all(state, &params, &mut workspace)[0];
    }

    ARMED.store(true, Ordering::SeqCst);
    ALLOCATIONS.store(0, Ordering::SeqCst);
    let mut measured = 0i64;
    for _ in 0..3 {
        for state in &states {
            measured ^= evaluate_all(state, &params, &mut workspace)[0];
        }
    }
    let allocations = ALLOCATIONS.load(Ordering::SeqCst);
    ARMED.store(false, Ordering::SeqCst);

    assert_eq!(warm, measured, "evaluation is not deterministic");
    assert_eq!(
        allocations,
        0,
        "{} allocations across {} steady-state evaluations; the workspace is \
         leaking a per-node allocation",
        allocations,
        states.len() * 3
    );
}
