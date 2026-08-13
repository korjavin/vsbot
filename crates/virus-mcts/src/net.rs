//! Convolutional policy/value net inference.
//!
//! A port of `nnue-trainer/.../mcts/PolicyNetPrior.java`, which itself
//! hand-rolls the forward pass of the net exported by `python/mcts/
//! train_policy.py` / `train_selfplay.py`. No ONNX, no BLAS: at 12x12 with 32
//! channels the whole trunk is ~4.6M multiply-adds, which a plain f32 dot
//! product over contiguous slices vectorises into well under a millisecond.
//!
//! # Architecture (`conv-policy-value-v1`)
//!
//! ```text
//! 13 x 12 x 12 planes
//!   -> `layers` x [conv 3x3 same-padding, `channels` out, ReLU]
//!   -> move head   : 1x1 conv -> 144 per-cell logits
//!   -> pair head   : 1x1 conv -> 144 per-cell utilities `u`
//!   -> value head  : GAP over channels -> fc(32) ReLU -> fc(1) -> tanh
//! ```
//!
//! `logit(PlaceNeutrals{i,j}) = u[i] + u[j] + pair_bias` — the factored pair
//! head, which is what keeps a 20 880-wide action space down to 288 outputs.
//!
//! # The 13 planes
//!
//! Built from the node's **own** mover, exactly the training-time encoding
//! (`PatternContract.getSymbol` + `SelfPlayMcts.row`):
//!
//! | plane | contents |
//! |-------|----------|
//! | 0..=7 | one-hot of the cell symbol (see [`symbol`]) |
//! | 8..=10 | one-hot of `moves_left` (1, 2, 3), constant over the board |
//! | 11 | 1 iff the mover has spent its neutral placement |
//! | 12 | 1 iff the opponent has spent its neutral placement |
//!
//! # One trunk, both heads
//!
//! [`PolicyValueNet::forward`] returns policy *and* value from a single trunk
//! pass. The Java original runs the trunk twice per expanded node (once in
//! `priors`, once in `valueMover`); fusing them is a free ~2x on self-play
//! throughput and is the reason [`Heads`] carries the value alongside the
//! logits rather than exposing a separate `value()` entry point.
//!
//! # Numerics
//!
//! Java infers in `f64`; this port uses `f32` throughout. Both are fed weights
//! that were trained in `f32`, so `f32` inference is if anything closer to the
//! trainer. See `tests/net_parity.rs` for the measured agreement against the
//! python-computed fixtures.

use std::fmt;
use std::path::Path;

use serde::Deserialize;
use virus_core::{CellKind, Player, State};

/// Board edge the net was trained on.
pub const BOARD: usize = 12;

/// Number of board cells, `BOARD * BOARD`.
pub const CELLS: usize = BOARD * BOARD;

/// Number of input planes.
pub const PLANES: usize = 13;

/// Padded edge used by the same-padding convolution.
const PADDED: usize = BOARD + 2;

/// Number of distinct cell symbols the first eight planes one-hot encode.
const SYMBOLS: usize = 8;

/// Positions a batched trunk pass processes in one interleaved group.
///
/// The batched kernels lay the batch out as the **innermost** axis
/// (`[channel][cell][LANES]`), so one group's activations for a given cell are
/// `LANES` contiguous floats — exactly one 256-bit vector. Eight is the AVX2
/// register width; the single-position kernel's innermost run is the 12-wide
/// board row, which wastes a third of every vector and needs overlapping
/// unaligned loads for the kernel's three columns. See
/// [`PolicyValueNet::forward_batch`].
pub const BATCH_LANES: usize = 8;

/// Architectures this crate implements.
///
/// `conv-policy-v1` is the Phase 1 policy-only trunk; `conv-policy-value-v1`
/// adds the value head and is what the gen-5 champion declares.
const SUPPORTED_ARCH: [&str; 2] = ["conv-policy-v1", "conv-policy-value-v1"];

// ---------------------------------------------------------------- encoding

/// Cell symbol in the mover's frame, matching Java's `PatternContract`.
///
/// `0` empty, `1` neutral, `2`/`3` own/enemy base, `4`/`5` own/enemy normal,
/// `6`/`7` own/enemy fortified. The contract's eighth value (out of bounds) is
/// unreachable for an on-board cell and therefore has no plane here.
pub fn symbol(cell: virus_core::Cell, mover: Player) -> u8 {
    match cell.kind() {
        CellKind::Empty => 0,
        CellKind::Neutral => 1,
        CellKind::Base => {
            if cell.owner() == mover {
                2
            } else {
                3
            }
        }
        CellKind::Normal => {
            if cell.owner() == mover {
                4
            } else {
                5
            }
        }
        CellKind::Fortified => {
            if cell.owner() == mover {
                6
            } else {
                7
            }
        }
    }
}

/// The compact training-time encoding of one position: what the trainer wrote
/// into its JSONL rows, and what the parity fixtures replay.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Encoded {
    /// Per-cell symbol in the mover's frame, row-major.
    pub sym: [u8; CELLS],
    /// Actions remaining this turn, `1..=3`.
    pub moves_left: u8,
    /// Whether the mover has already spent its neutral placement.
    pub nu_own: bool,
    /// Whether the opponent has already spent its neutral placement.
    pub nu_opp: bool,
}

impl Encoded {
    /// Encodes a 12x12 two-player position from the current mover's point of
    /// view.
    ///
    /// "The opponent" is the other of seats 1 and 2 — the trainer's
    /// `3 - mover`. There is deliberately no three- or four-player fallback:
    /// the symbol alphabet lumps every non-mover into one "opponent" class and
    /// the neutral-used planes have room for exactly one other seat, so a
    /// wider game has no representation here at all. Encoding one anyway would
    /// hand the searcher confident-looking priors computed from a position the
    /// net has never seen a shape of.
    ///
    /// # Panics
    /// Panics unless the board is exactly `BOARD x BOARD` with two players.
    /// [`crate::MctsSearcher::new`] makes the same check once, up front, so in
    /// a search this can only fire on direct misuse.
    pub fn from_state(state: &State) -> Encoded {
        assert!(
            state.rows() == BOARD && state.cols() == BOARD && state.players() == 2,
            "policy net is {BOARD}x{BOARD} two-player only, got {}x{} with {} players",
            state.rows(),
            state.cols(),
            state.players()
        );
        let mover = state.current_player();
        let opponent = if mover == 1 { 2 } else { 1 };
        let mut sym = [0u8; CELLS];
        for (index, slot) in sym.iter_mut().enumerate() {
            *slot = symbol(state.cell_at(index), mover);
        }
        Encoded {
            sym,
            moves_left: state.moves_left(),
            nu_own: state.neutral_used(mover),
            nu_opp: state.neutral_used(opponent),
        }
    }
}

// ---------------------------------------------------------------- outputs

/// One position's raw head outputs.
#[derive(Clone, Debug)]
pub struct Heads {
    /// Per-cell move logits; index by row-major cell.
    pub move_logits: [f32; CELLS],
    /// Per-cell pair utilities `u`; a pair's logit is `u[i] + u[j] + pair_bias`.
    pub pair_u: [f32; CELLS],
    /// Mover-frame value in `[-1, 1]`, or `None` for a policy-only artifact.
    pub value: Option<f32>,
}

// ---------------------------------------------------------------- errors

/// Why an artifact could not be loaded.
///
/// Every variant is raised at load time. A net that is the wrong shape, or that
/// carries a non-finite weight, must fail here — never halfway through a
/// search.
#[derive(Debug)]
pub enum NetError {
    /// The file could not be read.
    Io(std::io::Error),
    /// The file is not the JSON the trainer exports.
    Json(serde_json::Error),
    /// The JSON parsed but describes a net this crate cannot run.
    Shape(String),
}

impl fmt::Display for NetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NetError::Io(error) => write!(f, "policy net: {error}"),
            NetError::Json(error) => write!(f, "policy net: malformed JSON: {error}"),
            NetError::Shape(message) => write!(f, "policy net: {message}"),
        }
    }
}

impl std::error::Error for NetError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            NetError::Io(error) => Some(error),
            NetError::Json(error) => Some(error),
            NetError::Shape(_) => None,
        }
    }
}

impl From<std::io::Error> for NetError {
    fn from(error: std::io::Error) -> NetError {
        NetError::Io(error)
    }
}

impl From<serde_json::Error> for NetError {
    fn from(error: serde_json::Error) -> NetError {
        NetError::Json(error)
    }
}

fn shape<T>(message: impl Into<String>) -> Result<T, NetError> {
    Err(NetError::Shape(message.into()))
}

// ---------------------------------------------------------------- raw JSON

#[derive(Deserialize)]
struct RawMeta {
    #[serde(default)]
    arch: String,
    #[serde(default = "minus_one")]
    board: i64,
    #[serde(default = "minus_one")]
    planes: i64,
    #[serde(default = "minus_one")]
    channels: i64,
    #[serde(default = "minus_one")]
    layers: i64,
}

fn minus_one() -> i64 {
    -1
}

#[derive(Deserialize)]
struct RawConv {
    /// `[out][in][3][3]`.
    w: Vec<Vec<Vec<Vec<f64>>>>,
    b: Vec<f64>,
}

#[derive(Deserialize)]
struct RawHead {
    /// `[channels][1][1]` — torch's `[1][channels][1][1]` with the output axis
    /// already stripped by the exporter.
    w: Vec<Vec<Vec<f64>>>,
    b: f64,
}

#[derive(Deserialize)]
struct RawValueHead {
    /// `[hidden][channels]`.
    fc1_w: Vec<Vec<f64>>,
    fc1_b: Vec<f64>,
    fc2_w: Vec<f64>,
    fc2_b: f64,
}

#[derive(Deserialize)]
struct RawNet {
    meta: RawMeta,
    conv: Vec<RawConv>,
    move_head: RawHead,
    pair_head: RawHead,
    pair_bias: f64,
    #[serde(default)]
    value_head: Option<RawValueHead>,
}

/// Narrows a JSON `f64` to the `f32` inference uses, rejecting non-finite
/// weights. The trainer exports `f32` values, so this never loses information
/// on a well-formed artifact.
fn finite(value: f64, what: &str) -> Result<f32, NetError> {
    let narrowed = value as f32;
    if narrowed.is_finite() {
        Ok(narrowed)
    } else {
        shape(format!("{what} is not finite ({value})"))
    }
}

fn vector(values: &[f64], expected: usize, what: &str) -> Result<Vec<f32>, NetError> {
    if values.len() != expected {
        return shape(format!(
            "{what} must have {expected} entries, got {}",
            values.len()
        ));
    }
    values
        .iter()
        .enumerate()
        .map(|(i, value)| finite(*value, &format!("{what}[{i}]")))
        .collect()
}

// ---------------------------------------------------------------- the net

/// The trained value head: global average pool over trunk channels, one hidden
/// ReLU layer, then a `tanh` scalar in the mover's frame.
#[derive(Clone, Debug)]
struct ValueHead {
    /// `[hidden * channels]`, row-major by hidden unit.
    fc1_w: Vec<f32>,
    fc1_b: Vec<f32>,
    fc2_w: Vec<f32>,
    fc2_b: f32,
}

/// A loaded policy(+value) net.
///
/// Immutable and `Sync`: share one across searcher threads and give each its
/// own [`NetScratch`].
#[derive(Clone, Debug)]
pub struct PolicyValueNet {
    arch: String,
    channels: usize,
    layers: usize,
    /// Per layer, `[out][in * 9]` flattened: `w[o * k + ic * 9 + kr * 3 + kc]`.
    /// That is exactly a weight row, so the convolution below is a run of plain
    /// contiguous dot products.
    conv_w: Vec<Vec<f32>>,
    conv_b: Vec<Vec<f32>>,
    move_w: Vec<f32>,
    move_b: f32,
    pair_w: Vec<f32>,
    pair_b: f32,
    pair_bias: f32,
    value: Option<ValueHead>,
    /// Whether this CPU has the AVX2 + FMA the fast convolution needs. Decided
    /// once at load, not per call.
    simd: bool,
}

/// Whether the running CPU supports the convolution's fast path.
fn simd_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

impl PolicyValueNet {
    /// Loads and fully validates the trainer's export.
    ///
    /// Every field is checked here — declared shape against actual shape, and
    /// every weight for finiteness — so a bad artifact fails once, at startup,
    /// instead of poisoning a search with `NaN` priors.
    pub fn load(path: impl AsRef<Path>) -> Result<PolicyValueNet, NetError> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)?;
        PolicyValueNet::from_json(&text).map_err(|error| match error {
            NetError::Shape(message) => {
                NetError::Shape(format!("{message} (in {})", path.display()))
            }
            other => other,
        })
    }

    /// [`PolicyValueNet::load`] from an in-memory artifact.
    pub fn from_json(text: &str) -> Result<PolicyValueNet, NetError> {
        let raw: RawNet = serde_json::from_str(text)?;
        PolicyValueNet::from_raw(raw)
    }

    fn from_raw(raw: RawNet) -> Result<PolicyValueNet, NetError> {
        // The architecture name, not just the tensor shapes. A future trunk
        // (residual blocks, a different head factorisation) could export the
        // same shapes with different semantics, and this loader would run it as
        // if it were the plain conv stack — the one wrong-net failure the shape
        // checks below cannot catch.
        if !SUPPORTED_ARCH.contains(&raw.meta.arch.as_str()) {
            return shape(format!(
                "meta arch {:?} unsupported, expected one of {SUPPORTED_ARCH:?}",
                raw.meta.arch
            ));
        }
        if raw.meta.board != BOARD as i64 || raw.meta.planes != PLANES as i64 {
            return shape(format!(
                "meta board/planes {}/{} unsupported, expected {BOARD}/{PLANES}",
                raw.meta.board, raw.meta.planes
            ));
        }
        if raw.meta.channels <= 0 || raw.meta.layers <= 0 {
            return shape(format!(
                "meta channels/layers must be positive, got {}/{}",
                raw.meta.channels, raw.meta.layers
            ));
        }
        let channels = raw.meta.channels as usize;
        let layers = raw.meta.layers as usize;
        if raw.conv.len() != layers {
            return shape(format!(
                "conv has {} layers, meta declares {layers}",
                raw.conv.len()
            ));
        }

        let mut conv_w = Vec::with_capacity(layers);
        let mut conv_b = Vec::with_capacity(layers);
        for (layer, entry) in raw.conv.iter().enumerate() {
            let in_channels = if layer == 0 { PLANES } else { channels };
            if entry.w.len() != channels {
                return shape(format!(
                    "conv[{layer}].w has {} rows, expected {channels}",
                    entry.w.len()
                ));
            }
            let k = in_channels * 9;
            let mut flat = vec![0.0f32; channels * k];
            for (o, row) in entry.w.iter().enumerate() {
                if row.len() != in_channels {
                    return shape(format!(
                        "conv[{layer}].w[{o}] has {} input channels, expected {in_channels}",
                        row.len()
                    ));
                }
                for (ic, kernel) in row.iter().enumerate() {
                    if kernel.len() != 3 {
                        return shape(format!(
                            "conv[{layer}].w[{o}][{ic}] has {} rows, expected 3",
                            kernel.len()
                        ));
                    }
                    for (kr, krow) in kernel.iter().enumerate() {
                        if krow.len() != 3 {
                            return shape(format!(
                                "conv[{layer}].w[{o}][{ic}][{kr}] has {} columns, expected 3",
                                krow.len()
                            ));
                        }
                        for (kc, value) in krow.iter().enumerate() {
                            flat[o * k + ic * 9 + kr * 3 + kc] =
                                finite(*value, &format!("conv[{layer}].w[{o}][{ic}][{kr}][{kc}]"))?;
                        }
                    }
                }
            }
            conv_w.push(flat);
            conv_b.push(vector(&entry.b, channels, &format!("conv[{layer}].b"))?);
        }

        let move_w = head_weights(&raw.move_head, channels, "move_head")?;
        let pair_w = head_weights(&raw.pair_head, channels, "pair_head")?;
        let move_b = finite(raw.move_head.b, "move_head.b")?;
        let pair_b = finite(raw.pair_head.b, "pair_head.b")?;
        let pair_bias = finite(raw.pair_bias, "pair_bias")?;

        let value = match &raw.value_head {
            None => None,
            Some(vh) => {
                if vh.fc1_w.is_empty() {
                    return shape("value_head.fc1_w is empty");
                }
                let hidden = vh.fc1_w.len();
                let mut fc1_w = Vec::with_capacity(hidden * channels);
                for (h, row) in vh.fc1_w.iter().enumerate() {
                    fc1_w.extend(vector(row, channels, &format!("value_head.fc1_w[{h}]"))?);
                }
                Some(ValueHead {
                    fc1_w,
                    fc1_b: vector(&vh.fc1_b, hidden, "value_head.fc1_b")?,
                    fc2_w: vector(&vh.fc2_w, hidden, "value_head.fc2_w")?,
                    fc2_b: finite(vh.fc2_b, "value_head.fc2_b")?,
                })
            }
        };

        Ok(PolicyValueNet {
            arch: raw.meta.arch,
            channels,
            layers,
            conv_w,
            conv_b,
            move_w,
            move_b,
            pair_w,
            pair_b,
            pair_bias,
            value,
            simd: simd_available(),
        })
    }

    /// Whether inference is using the AVX2 + FMA convolution.
    pub fn simd(&self) -> bool {
        self.simd
    }

    /// Forces the portable convolution, whatever the CPU supports.
    ///
    /// For the test that cross-checks the two code paths against each other,
    /// and for reproducing a number on a machine wider than the target.
    pub fn force_scalar(&mut self) {
        self.simd = false;
    }

    /// The artifact's declared architecture string, e.g. `conv-policy-value-v1`.
    pub fn arch(&self) -> &str {
        &self.arch
    }

    /// Trunk width.
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Number of trunk convolutions.
    pub fn layers(&self) -> usize {
        self.layers
    }

    /// The additive bias in `logit(pair) = u[i] + u[j] + pair_bias`.
    pub fn pair_bias(&self) -> f32 {
        self.pair_bias
    }

    /// Whether this artifact carries a trained value head.
    pub fn has_value_head(&self) -> bool {
        self.value.is_some()
    }

    /// Scratch space sized for this net. Hold one per thread and reuse it — the
    /// forward pass allocates nothing.
    pub fn scratch(&self) -> NetScratch {
        let width = self.channels.max(PLANES);
        NetScratch {
            cur: vec![0.0; width * CELLS],
            next: vec![0.0; width * CELLS],
            padded: vec![0.0; width * PADDED * PADDED],
        }
    }

    /// One trunk pass serving **both** heads and the value.
    ///
    /// This is the fused forward the Java original lacks: `priors` and
    /// `valueMover` there each run the whole trunk, so an expanded node paid
    /// for it twice.
    pub fn forward(&self, input: &Encoded, scratch: &mut NetScratch) -> Heads {
        let NetScratch { cur, next, padded } = scratch;

        // --- input planes ---
        cur[..PLANES * CELLS].fill(0.0);
        for (i, sym) in input.sym.iter().enumerate() {
            let plane = *sym as usize;
            debug_assert!(plane < SYMBOLS, "symbol {plane} has no plane");
            cur[plane.min(SYMBOLS - 1) * CELLS + i] = 1.0;
        }
        // The encoding contract is `moves_left in 1..=3`; clamping keeps a
        // malformed caller out of the symbol planes.
        let moves_left = (input.moves_left as usize).clamp(1, 3);
        let base = (SYMBOLS + moves_left - 1) * CELLS;
        cur[base..base + CELLS].fill(1.0);
        if input.nu_own {
            cur[11 * CELLS..12 * CELLS].fill(1.0);
        }
        if input.nu_opp {
            cur[12 * CELLS..13 * CELLS].fill(1.0);
        }

        // --- trunk ---
        let mut in_channels = PLANES;
        for layer in 0..self.layers {
            conv3x3_relu(
                self.simd,
                &cur[..in_channels * CELLS],
                ConvLayer {
                    in_channels,
                    out_channels: self.channels,
                    w: &self.conv_w[layer],
                    b: &self.conv_b[layer],
                },
                padded,
                next,
            );
            std::mem::swap(cur, next);
            in_channels = self.channels;
        }
        let trunk = &cur[..self.channels * CELLS];

        // --- heads (one trunk, three consumers) ---
        Heads {
            move_logits: self.head(trunk, &self.move_w, self.move_b),
            pair_u: self.head(trunk, &self.pair_w, self.pair_b),
            value: self.value.as_ref().map(|head| self.value_of(trunk, head)),
        }
    }

    /// Scratch space for [`PolicyValueNet::forward_batch`]. Hold one per
    /// thread and reuse it — a batched pass allocates nothing beyond the
    /// [`Heads`] it hands back.
    pub fn batch_scratch(&self) -> BatchScratch {
        let width = self.channels.max(PLANES);
        BatchScratch {
            cur: vec![0.0; width * CELLS * BATCH_LANES],
            next: vec![0.0; width * CELLS * BATCH_LANES],
            padded: vec![0.0; width * PADDED * PADDED * BATCH_LANES],
        }
    }

    /// One trunk pass over many positions at once, appending one [`Heads`] per
    /// input to `out` in input order.
    ///
    /// # Why this is faster than the same positions one at a time
    ///
    /// Nothing about the arithmetic changes — the same
    /// `channels * in_channels * 9 * 144` multiply-adds happen per position,
    /// in the same order, so a batched result is bit-identical to the serial
    /// one (`batched_forward_matches_the_serial_one` in `tests/net_parity.rs`
    /// pins that). What changes is the shape of the inner loop:
    ///
    /// * **Vector utilisation.** [`PolicyValueNet::forward`] reduces over a
    ///   12-wide board row, which is 1.5 AVX2 registers; the batched kernel
    ///   reduces over [`BATCH_LANES`] positions, which is exactly one.
    /// * **Loads.** The single-position kernel reads three *overlapping*
    ///   unaligned windows of the same row (`row[c]`, `row[c+1]`, `row[c+2]`)
    ///   for the kernel's three columns. With the batch innermost those become
    ///   three disjoint aligned vectors.
    /// * **Weights.** One `[out][in*9]` weight row is broadcast across
    ///   `BATCH_LANES` positions instead of one, so the ~125 KB of trunk
    ///   weights is streamed out of L2 once per group rather than once per
    ///   position.
    ///
    /// Inputs are processed in groups of [`BATCH_LANES`]; a partial trailing
    /// group leaves its unused lanes zeroed and discards their outputs, so a
    /// batch of 9 costs the same as a batch of 16. Callers that care should
    /// size their batches in multiples of [`BATCH_LANES`].
    pub fn forward_batch(
        &self,
        inputs: &[Encoded],
        scratch: &mut BatchScratch,
        out: &mut Vec<Heads>,
    ) {
        for group in inputs.chunks(BATCH_LANES) {
            self.forward_group(group, scratch, out);
        }
    }

    /// One trunk pass over a single lane group of at most [`BATCH_LANES`].
    fn forward_group(&self, group: &[Encoded], scratch: &mut BatchScratch, out: &mut Vec<Heads>) {
        debug_assert!(!group.is_empty() && group.len() <= BATCH_LANES);
        let BatchScratch { cur, next, padded } = scratch;
        let lanes = BATCH_LANES;

        // --- input planes, interleaved by lane; unused lanes stay zero ---
        cur[..PLANES * CELLS * lanes].fill(0.0);
        for (lane, input) in group.iter().enumerate() {
            for (i, sym) in input.sym.iter().enumerate() {
                let plane = *sym as usize;
                debug_assert!(plane < SYMBOLS, "symbol {plane} has no plane");
                cur[(plane.min(SYMBOLS - 1) * CELLS + i) * lanes + lane] = 1.0;
            }
            // Same clamp as the single-position path: the encoding contract is
            // `moves_left in 1..=3` and a malformed caller must not land in the
            // symbol planes.
            let moves_left = (input.moves_left as usize).clamp(1, 3);
            let mut set = |plane: usize| {
                for i in 0..CELLS {
                    cur[(plane * CELLS + i) * lanes + lane] = 1.0;
                }
            };
            set(SYMBOLS + moves_left - 1);
            if input.nu_own {
                set(11);
            }
            if input.nu_opp {
                set(12);
            }
        }

        // --- trunk ---
        let mut in_channels = PLANES;
        for layer in 0..self.layers {
            conv3x3_relu_batch(
                self.simd,
                &cur[..in_channels * CELLS * lanes],
                ConvLayer {
                    in_channels,
                    out_channels: self.channels,
                    w: &self.conv_w[layer],
                    b: &self.conv_b[layer],
                },
                padded,
                next,
            );
            std::mem::swap(cur, next);
            in_channels = self.channels;
        }
        let trunk = &cur[..self.channels * CELLS * lanes];

        // --- heads, one lane at a time ---
        for lane in 0..group.len() {
            out.push(Heads {
                move_logits: self.head_lane(trunk, &self.move_w, self.move_b, lane),
                pair_u: self.head_lane(trunk, &self.pair_w, self.pair_b, lane),
                value: self
                    .value
                    .as_ref()
                    .map(|head| self.value_of_lane(trunk, head, lane)),
            });
        }
    }

    /// 1x1 convolution from the trunk to a single 12x12 map.
    fn head(&self, trunk: &[f32], w: &[f32], bias: f32) -> [f32; CELLS] {
        let mut out = [bias; CELLS];
        for (ch, weight) in w.iter().enumerate() {
            let plane = &trunk[ch * CELLS..(ch + 1) * CELLS];
            for (slot, activation) in out.iter_mut().zip(plane) {
                *slot += weight * activation;
            }
        }
        out
    }

    /// [`PolicyValueNet::head`] over one lane of an interleaved trunk.
    ///
    /// Accumulates channel-major, exactly like the contiguous version, so the
    /// two agree bit for bit.
    fn head_lane(&self, trunk: &[f32], w: &[f32], bias: f32, lane: usize) -> [f32; CELLS] {
        let mut out = [bias; CELLS];
        for (ch, weight) in w.iter().enumerate() {
            let base = ch * CELLS * BATCH_LANES + lane;
            for (c, slot) in out.iter_mut().enumerate() {
                *slot += weight * trunk[base + c * BATCH_LANES];
            }
        }
        out
    }

    /// Global average pool -> fc1 ReLU -> fc2 -> tanh.
    fn value_of(&self, trunk: &[f32], head: &ValueHead) -> f32 {
        let mut gap = vec![0.0f32; self.channels];
        for (ch, slot) in gap.iter_mut().enumerate() {
            let plane = &trunk[ch * CELLS..(ch + 1) * CELLS];
            *slot = plane.iter().sum::<f32>() / CELLS as f32;
        }
        self.value_from_gap(&gap, head)
    }

    /// [`PolicyValueNet::value_of`] over one lane of an interleaved trunk.
    fn value_of_lane(&self, trunk: &[f32], head: &ValueHead, lane: usize) -> f32 {
        let mut gap = vec![0.0f32; self.channels];
        for (ch, slot) in gap.iter_mut().enumerate() {
            let base = ch * CELLS * BATCH_LANES + lane;
            // Summed in cell order, matching `Iterator::sum`'s sequential fold
            // over the contiguous plane, so the pooled value is identical.
            let mut sum = 0.0f32;
            for c in 0..CELLS {
                sum += trunk[base + c * BATCH_LANES];
            }
            *slot = sum / CELLS as f32;
        }
        self.value_from_gap(&gap, head)
    }

    /// The value head's dense tail, shared by both pooling paths.
    fn value_from_gap(&self, gap: &[f32], head: &ValueHead) -> f32 {
        let mut out = head.fc2_b;
        for (h, bias) in head.fc1_b.iter().enumerate() {
            let row = &head.fc1_w[h * self.channels..(h + 1) * self.channels];
            let acc = bias + dot(row, gap);
            if acc > 0.0 {
                out += head.fc2_w[h] * acc;
            }
        }
        out.tanh()
    }
}

fn head_weights(head: &RawHead, channels: usize, what: &str) -> Result<Vec<f32>, NetError> {
    if head.w.len() != channels {
        return shape(format!(
            "{what}.w has {} channels, expected {channels}",
            head.w.len()
        ));
    }
    let mut out = Vec::with_capacity(channels);
    for (ch, outer) in head.w.iter().enumerate() {
        // torch's trailing [1][1] spatial axes. Checked exactly rather than
        // just indexed into: a head with a wider kernel is a different
        // architecture, and silently using its top-left weight would run a
        // model this crate does not implement.
        if outer.len() != 1 || outer[0].len() != 1 {
            return shape(format!(
                "{what}.w[{ch}] must be a 1x1 kernel, got {}x{}",
                outer.len(),
                outer.first().map_or(0, Vec::len)
            ));
        }
        out.push(finite(outer[0][0], &format!("{what}.w[{ch}]"))?);
    }
    Ok(out)
}

// ---------------------------------------------------------------- kernels

/// Reusable buffers for [`PolicyValueNet::forward`].
#[derive(Clone, Debug)]
pub struct NetScratch {
    cur: Vec<f32>,
    next: Vec<f32>,
    padded: Vec<f32>,
}

/// Reusable buffers for [`PolicyValueNet::forward_batch`].
///
/// Laid out `[channel][cell][BATCH_LANES]` — the batch is the innermost,
/// contiguous axis, which is the whole point of the batched kernel.
#[derive(Clone, Debug)]
pub struct BatchScratch {
    cur: Vec<f32>,
    next: Vec<f32>,
    padded: Vec<f32>,
}

/// [`conv3x3_relu_scalar`], recompiled for AVX2 + FMA.
///
/// The body is identical — this is purely a code-generation switch. The
/// baseline `x86_64` target is SSE2-only, which caps the trunk at roughly
/// 4 multiply-adds per cycle; with FMA the same loop reaches 16. Measured on
/// this machine: 0.418 ms/forward baseline against 0.255 ms here, a 1.6x that
/// costs nothing but a feature check.
///
/// # Safety
/// `#[target_feature]` makes this callable only on a CPU that has AVX2 and FMA.
/// Every call site goes through [`PolicyValueNet::simd`], which is set at load
/// time from `is_x86_feature_detected!` for exactly these two features, so the
/// precondition is checked before the flag can ever be true. Covered by
/// `both_convolution_paths_hit_parity` in `tests/net_parity.rs`, which runs the
/// fixtures through this path and the portable one and checks both against the
/// oracle and against each other. They currently agree bit for bit — rustc does
/// not contract multiply-adds, so the whole gain is vector width — but the test
/// asserts a band rather than equality, because a toolchain that did start
/// contracting would still be correct.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn conv3x3_relu_avx2(
    input: &[f32],
    layer: ConvLayer<'_>,
    padded: &mut [f32],
    out: &mut [f32],
) {
    conv3x3_relu_scalar(input, layer, padded, out);
}

/// One convolution layer's immutable parameters.
#[derive(Clone, Copy, Debug)]
struct ConvLayer<'a> {
    in_channels: usize,
    out_channels: usize,
    /// `[out][in * 9]` flattened.
    w: &'a [f32],
    b: &'a [f32],
}

/// Dispatches the convolution to the widest instruction set this CPU has.
fn conv3x3_relu(
    simd: bool,
    input: &[f32],
    layer: ConvLayer<'_>,
    padded: &mut [f32],
    out: &mut [f32],
) {
    #[cfg(target_arch = "x86_64")]
    if simd {
        // SAFETY: `simd` is only ever set when `is_x86_feature_detected!`
        // reported both AVX2 and FMA for this CPU — see `PolicyValueNet::simd`
        // and the safety note on `conv3x3_relu_avx2`.
        unsafe {
            conv3x3_relu_avx2(input, layer, padded, out);
        }
        return;
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = simd;
    conv3x3_relu_scalar(input, layer, padded, out);
}

/// 3x3 same-padding convolution with ReLU.
///
/// Direct convolution, register-blocked one board row at a time. The obvious
/// alternative — im2col, which is what the Java original does — was measured at
/// 2.8x slower here: gathering each cell's `in_channels * 9` patch costs ~14k
/// three-element copies per layer, and that shuffling, not the arithmetic,
/// dominates at 12x12.
///
/// Instead the 12 accumulators for one output row live in registers across the
/// whole `in_channels * 9` reduction, and every operand is a contiguous
/// 12-float run of the zero-padded input against a broadcast weight — the shape
/// a compiler turns into plain FMAs with no data movement at all. One store per
/// output cell per layer, rather than one per multiply-add.
///
/// `#[inline(always)]` so the AVX2 wrapper above gets a genuinely recompiled
/// copy rather than a call into baseline-SSE2 code.
#[inline(always)]
fn conv3x3_relu_scalar(input: &[f32], layer: ConvLayer<'_>, padded: &mut [f32], out: &mut [f32]) {
    let ConvLayer {
        in_channels,
        out_channels,
        w,
        b,
    } = layer;
    let plane = PADDED * PADDED;
    let padded = &mut padded[..in_channels * plane];
    padded.fill(0.0);
    for ic in 0..in_channels {
        for r in 0..BOARD {
            let src = ic * CELLS + r * BOARD;
            let dst = ic * plane + (r + 1) * PADDED + 1;
            padded[dst..dst + BOARD].copy_from_slice(&input[src..src + BOARD]);
        }
    }

    let k = in_channels * 9;
    for o in 0..out_channels {
        let w_row = &w[o * k..(o + 1) * k];
        for r in 0..BOARD {
            let mut acc = [b[o]; BOARD];
            for ic in 0..in_channels {
                let kernel = &w_row[ic * 9..ic * 9 + 9];
                let channel = &padded[ic * plane..(ic + 1) * plane];
                // The three padded rows this kernel reads. `r` is a board row,
                // so padded row `r + kr` is board row `r + kr - 1`.
                for kr in 0..3 {
                    let base = (r + kr) * PADDED;
                    let row = &channel[base..base + BOARD + 2];
                    // The kernel's three columns share one pass over the 12
                    // accumulators and one set of row loads. Unrolling the
                    // kernel *rows* as well was measured slower: nine live
                    // vector operands exhaust the register file and the
                    // compiler starts spilling `acc`.
                    let (k0, k1, k2) = (kernel[kr * 3], kernel[kr * 3 + 1], kernel[kr * 3 + 2]);
                    for c in 0..BOARD {
                        acc[c] += k0 * row[c] + k1 * row[c + 1] + k2 * row[c + 2];
                    }
                }
            }
            let dst = &mut out[o * CELLS + r * BOARD..o * CELLS + r * BOARD + BOARD];
            for (slot, value) in dst.iter_mut().zip(&acc) {
                *slot = if *value > 0.0 { *value } else { 0.0 };
            }
        }
    }
}

/// [`conv3x3_relu_batch_scalar`], recompiled for AVX2 + FMA.
///
/// # Safety
/// Same contract as [`conv3x3_relu_avx2`]: `#[target_feature]` makes this
/// callable only on a CPU with AVX2 and FMA, and every call site goes through
/// the `simd` flag that [`simd_available`] set from `is_x86_feature_detected!`.
/// `batched_forward_matches_the_serial_one` in `tests/net_parity.rs` runs the
/// fixtures through this path.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn conv3x3_relu_batch_avx2(
    input: &[f32],
    layer: ConvLayer<'_>,
    padded: &mut [f32],
    out: &mut [f32],
) {
    conv3x3_relu_batch_scalar(input, layer, padded, out);
}

/// Dispatches the batched convolution to the widest instruction set available.
fn conv3x3_relu_batch(
    simd: bool,
    input: &[f32],
    layer: ConvLayer<'_>,
    padded: &mut [f32],
    out: &mut [f32],
) {
    #[cfg(target_arch = "x86_64")]
    if simd {
        // SAFETY: see `conv3x3_relu` — `simd` is only set when this CPU
        // reported both AVX2 and FMA at load time.
        unsafe {
            conv3x3_relu_batch_avx2(input, layer, padded, out);
        }
        return;
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = simd;
    conv3x3_relu_batch_scalar(input, layer, padded, out);
}

/// [`conv3x3_relu_scalar`] over [`BATCH_LANES`] interleaved positions.
///
/// Buffers are `[channel][cell][BATCH_LANES]`, so the innermost run is one
/// full-width vector of *positions* rather than a 12-wide board row. The
/// reduction order is unchanged — for every output cell the kernel's three
/// columns are still summed as one `k0*a + k1*b + k2*c` expression before
/// being added to the accumulator, and channels are still walked in order — so
/// this produces bit-identical results to the single-position kernel.
///
/// The accumulator is a `[[f32; BATCH_LANES]; C_TILE]` tile rather than a whole
/// board row: 12 lanes-wide accumulators plus the three loaded operand vectors
/// and three broadcast weights overflow the 16 AVX2 registers and start
/// spilling, which measured slower than re-walking the row in tiles.
#[inline(always)]
fn conv3x3_relu_batch_scalar(
    input: &[f32],
    layer: ConvLayer<'_>,
    padded: &mut [f32],
    out: &mut [f32],
) {
    /// Board columns whose accumulators are held live at once.
    const C_TILE: usize = 6;
    const L: usize = BATCH_LANES;

    let ConvLayer {
        in_channels,
        out_channels,
        w,
        b,
    } = layer;
    let plane = PADDED * PADDED * L;
    let padded = &mut padded[..in_channels * plane];
    padded.fill(0.0);
    for ic in 0..in_channels {
        for r in 0..BOARD {
            let src = (ic * CELLS + r * BOARD) * L;
            let dst = ic * plane + ((r + 1) * PADDED + 1) * L;
            padded[dst..dst + BOARD * L].copy_from_slice(&input[src..src + BOARD * L]);
        }
    }

    let k = in_channels * 9;
    for o in 0..out_channels {
        let w_row = &w[o * k..(o + 1) * k];
        for r in 0..BOARD {
            for c0 in (0..BOARD).step_by(C_TILE) {
                let mut acc = [[b[o]; L]; C_TILE];
                for ic in 0..in_channels {
                    let kernel = &w_row[ic * 9..ic * 9 + 9];
                    let channel = &padded[ic * plane..(ic + 1) * plane];
                    for kr in 0..3 {
                        // `r` is a board row, so padded row `r + kr` is board
                        // row `r + kr - 1` — the same offset the single
                        // position kernel uses.
                        let base = ((r + kr) * PADDED + c0) * L;
                        let row = &channel[base..base + (C_TILE + 2) * L];
                        let (k0, k1, k2) = (kernel[kr * 3], kernel[kr * 3 + 1], kernel[kr * 3 + 2]);
                        for (c, slot) in acc.iter_mut().enumerate() {
                            let window = &row[c * L..c * L + 3 * L];
                            for lane in 0..L {
                                slot[lane] += k0 * window[lane]
                                    + k1 * window[L + lane]
                                    + k2 * window[2 * L + lane];
                            }
                        }
                    }
                }
                let dst = (o * CELLS + r * BOARD + c0) * L;
                for (c, values) in acc.iter().enumerate() {
                    let slot = &mut out[dst + c * L..dst + c * L + L];
                    for lane in 0..L {
                        slot[lane] = if values[lane] > 0.0 {
                            values[lane]
                        } else {
                            0.0
                        };
                    }
                }
            }
        }
    }
}

/// Dot product of two equal-length `f32` slices.
///
/// Eight independent accumulators: `f32` addition is not associative, so the
/// compiler may not split the chain itself, and the serial dependency is what
/// bounds this loop. Written over `chunks_exact` so it lowers to one 256-bit
/// FMA per iteration where the target allows it.
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = [0.0f32; 8];
    let mut lhs = a.chunks_exact(8);
    let mut rhs = b.chunks_exact(8);
    for (x, y) in lhs.by_ref().zip(rhs.by_ref()) {
        for lane in 0..8 {
            acc[lane] += x[lane] * y[lane];
        }
    }
    let mut sum = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
    for (x, y) in lhs.remainder().iter().zip(rhs.remainder()) {
        sum += x * y;
    }
    sum
}
