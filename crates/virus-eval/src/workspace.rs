//! The reusable, allocation-free evaluation workspace.
//!
//! Port of Go's `evalWorkspace`. GoBot's original evaluator allocated a fresh
//! `[]bool` per player per node for connectivity, plus five Tarjan buffers and
//! two Voronoi buffers — one allocation storm per leaf, and the single biggest
//! reason that engine is slow.
//!
//! Everything here is hoisted into buffers that are grown once and then reused
//! forever. After the first evaluation at a given board size, a steady-state
//! search performs **zero** allocations in the evaluator; `tests/no_alloc.rs`
//! pins that with a counting global allocator.
//!
//! **One workspace per searcher, never shared.** It is pure mutable scratch: two
//! threads evaluating through the same workspace would interleave their
//! connectivity masks and silently produce wrong scores. `&mut` access makes
//! that a compile error rather than a heisenbug.

use virus_core::MAX_PLAYERS;

/// One frame of the iterative Tarjan traversal.
///
/// The Go and Java originals recurse. A 50x50 board is a 2500-vertex chain in
/// the worst case, and this runs at every search leaf, so the traversal is
/// explicit here. `cursor` indexes the 3x3 neighbourhood scan (self included
/// and skipped), which is exactly the order Go's `neighbors` emits.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Frame {
    pub(crate) index: u32,
    pub(crate) cursor: u8,
    pub(crate) children: u32,
}

/// Buffers shared by every seat's analysis within one evaluation.
#[derive(Debug, Default)]
pub(crate) struct Scratch {
    /// FIFO for the connectivity flood-fill; re-borrowed by the Voronoi BFS,
    /// which only starts once connectivity has finished with it.
    pub(crate) queue: Vec<u32>,
    /// Deduplicates mobility/capture targets (Go's `analysisScratch.targets`).
    pub(crate) targets: Vec<bool>,
    /// Tarjan discovery times; `0` doubles as "unvisited", as in Go.
    pub(crate) discovery: Vec<u16>,
    /// Tarjan low-links.
    pub(crate) low: Vec<u16>,
    /// DFS parent, `-1` for a tree root.
    pub(crate) parent: Vec<i32>,
    /// DFS subtree sizes, the raw material of `cut_loss`.
    pub(crate) subtree: Vec<u16>,
    /// Explicit DFS stack.
    pub(crate) stack: Vec<Frame>,
    /// Voronoi BFS distance, `-1` for unreached.
    pub(crate) space_dist: Vec<i32>,
    /// Voronoi BFS owner: `-1` unset, `-2` contested, else a seat index.
    pub(crate) space_owner: Vec<i8>,
}

/// Per-searcher scratch space for [`crate::evaluate_all`].
///
/// Construct one per searcher (or per root worker) and pass it by `&mut` for
/// the lifetime of the search. See the module docs for why it must not be
/// shared across threads.
#[derive(Debug, Default)]
pub struct EvalWorkspace {
    /// Base-connected component per seat, indexed by `seat - 1`.
    pub(crate) connected: [Vec<bool>; MAX_PLAYERS],
    /// Articulation cells per seat, after the own-`Normal` filter.
    pub(crate) articulation: [Vec<bool>; MAX_PLAYERS],
    /// Cells lost when the matching articulation cell is captured.
    pub(crate) cut_loss: [Vec<u16>; MAX_PLAYERS],
    pub(crate) scratch: Scratch,
}

impl EvalWorkspace {
    /// An empty workspace. Buffers are allocated on the first evaluation and
    /// reused from then on.
    pub fn new() -> EvalWorkspace {
        EvalWorkspace::default()
    }

    /// Grows every buffer to `size` cells.
    ///
    /// `Vec::resize` only allocates when capacity is short, so repeated calls
    /// at the same board size are free — the whole point of the type. Values
    /// are not meaningful after this call; every consumer clears what it reads.
    pub(crate) fn ensure(&mut self, size: usize) {
        self.scratch.queue.resize(size, 0);
        self.scratch.targets.resize(size, false);
        self.scratch.discovery.resize(size, 0);
        self.scratch.low.resize(size, 0);
        self.scratch.parent.resize(size, -1);
        self.scratch.subtree.resize(size, 0);
        self.scratch.space_dist.resize(size, -1);
        self.scratch.space_owner.resize(size, -1);
        // The DFS never holds more frames than there are vertices; reserving
        // up front is what keeps the steady state allocation-free.
        if self.scratch.stack.capacity() < size {
            self.scratch
                .stack
                .reserve(size - self.scratch.stack.capacity());
        }
        for seat in 0..MAX_PLAYERS {
            self.connected[seat].resize(size, false);
            self.articulation[seat].resize(size, false);
            self.cut_loss[seat].resize(size, 0);
        }
    }
}
