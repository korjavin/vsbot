//! Reusable scratch buffers.
//!
//! The Go and Java engines allocate a fresh `[]bool`/`boolean[]` for every
//! connectivity flood-fill, every frontier scan and every Tarjan pass — one
//! allocation per node per player. That per-node allocation tax is the single
//! biggest reason those engines are slow (see ARCHITECTURE.md). Here every
//! buffer is hoisted into one reusable [`Scratch`] (the analogue of Go's
//! `evalWorkspace`), sized for the largest supported board and always used
//! through `..cells` prefix slices so a 12x12 board only ever touches 144 bytes.

use crate::MAX_CELLS;

/// Breadth-first-search buffers: connectivity flood-fill and frontier scan.
#[derive(Debug)]
pub(crate) struct BfsScratch {
    /// Visited marks for the flood-fill.
    pub(crate) seen: [bool; MAX_CELLS],
    /// FIFO of board indices. `u16` covers `MAX_CELLS` with room to spare.
    pub(crate) queue: [u16; MAX_CELLS],
    /// Legal-target marks, so the frontier comes out in board order.
    pub(crate) frontier: [bool; MAX_CELLS],
}

/// Tarjan articulation-point buffers.
#[derive(Debug)]
pub(crate) struct ArtScratch {
    pub(crate) discovery: [u32; MAX_CELLS],
    pub(crate) low: [u32; MAX_CELLS],
    pub(crate) parent: [i32; MAX_CELLS],
    pub(crate) cuts: [bool; MAX_CELLS],
}

/// All per-call working memory the rules engine needs.
///
/// The fields are separate so a caller can hold the mover's connectivity mask
/// while running a *second* flood-fill for an opponent — which is exactly what
/// the strategic neutral-pair curation does. Always box it: it is ~50 KiB.
#[derive(Debug)]
pub struct Scratch {
    pub(crate) bfs: BfsScratch,
    /// The mover's base-connected component, retained across a whole
    /// [`crate::Position`] construction.
    pub(crate) connected: [bool; MAX_CELLS],
    /// A second connectivity mask, so opponent flood-fills (threat detection in
    /// the neutral-pair curation) can run without clobbering `connected`.
    pub(crate) alt_connected: [bool; MAX_CELLS],
    /// Own `Normal` cells an active opponent could capture right now.
    pub(crate) threatened: [bool; MAX_CELLS],
    /// Articulation cells of the mover's component (full graph).
    pub(crate) cuts: [bool; MAX_CELLS],
    pub(crate) art: ArtScratch,
}

impl Scratch {
    /// Allocates a zeroed scratch space on the heap.
    pub fn new() -> Box<Scratch> {
        Box::new(Scratch {
            bfs: BfsScratch {
                seen: [false; MAX_CELLS],
                queue: [0; MAX_CELLS],
                frontier: [false; MAX_CELLS],
            },
            connected: [false; MAX_CELLS],
            alt_connected: [false; MAX_CELLS],
            threatened: [false; MAX_CELLS],
            cuts: [false; MAX_CELLS],
            art: ArtScratch {
                discovery: [0; MAX_CELLS],
                low: [0; MAX_CELLS],
                parent: [-1; MAX_CELLS],
                cuts: [false; MAX_CELLS],
            },
        })
    }
}

thread_local! {
    static THREAD_SCRATCH: std::cell::RefCell<Box<Scratch>> =
        std::cell::RefCell::new(Scratch::new());
}

/// Runs `body` with this thread's shared [`Scratch`].
///
/// Never call a `..._with`-free convenience API from inside `body`: the
/// `RefCell` borrow is held for the duration and re-entering would panic. All
/// engine-internal call sites thread `&mut Scratch` explicitly for this reason.
pub(crate) fn with_thread_scratch<R>(body: impl FnOnce(&mut Scratch) -> R) -> R {
    THREAD_SCRATCH.with(|cell| body(&mut cell.borrow_mut()))
}
