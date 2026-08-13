//! Net-inference parity against the python-computed fixtures.
//!
//! `fixtures/mcts/mcts_policy_parity.json` was produced by `train_policy.py`
//! from `artifacts/mcts_policy.json` (the 32ch x 4-layer Phase 1 policy net);
//! `fixtures/mcts/mcts_value_parity.json` was produced by `train_selfplay.py`
//! from `fixtures/mcts/mcts_selfplay_tiny.json` (a deliberately tiny 8ch x
//! 2-layer net — the fixture pins the *forward pass*, not strength). Between
//! them they cover both trunk widths, both depths, the presence and the absence
//! of a value head.
//!
//! # Tolerance
//!
//! The fixtures carry no `meta` block and so state no tolerance of their own.
//! This test uses **1e-4 absolute** on head logits and value, applied as an
//! absolute rather than relative band: the compared quantities are pre-softmax
//! logits that legitimately pass through zero, where a relative band is
//! unbounded and meaningless. Head outputs here live in roughly `[-2, 2]`, so
//! 1e-4 absolute is at least as tight as 1e-4 relative everywhere it matters.
//!
//! Measured worst case: **6.7e-6** on the 32ch x 4-layer policy net and
//! **6.0e-8** on the 8ch x 2-layer value net — 15x and 1600x of headroom. This
//! port is in fact *closer* to the fixture than the reference Java
//! implementation, whose own parity test allows `1e-3`: the fixtures are
//! `float32` torch output and this port infers in `f32`, where Java widens
//! everything to `f64` and accumulates a different rounding trajectory.
//!
//! The gate that actually matters is downstream and is checked here too: the
//! softmax over legal actions, which is what the searcher consumes, agrees to
//! 8.6e-7 in probability.

use std::path::PathBuf;

use serde::Deserialize;
use virus_mcts::net::{Encoded, PolicyValueNet, CELLS};

/// Head-output agreement band; see the module docs.
const TOL: f32 = 1e-4;

#[derive(Debug, Deserialize)]
struct Sample {
    sym: Vec<u8>,
    ml: u8,
    nuo: u8,
    nux: u8,
    move_logits: Vec<f32>,
    pair_u: Vec<f32>,
    #[serde(default)]
    value: Option<f32>,
}

#[derive(Debug, Deserialize)]
struct Fixture {
    pair_bias: f32,
    samples: Vec<Sample>,
}

fn repo_path(relative: &str) -> PathBuf {
    // CARGO_MANIFEST_DIR is crates/virus-mcts.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(relative)
}

fn fixture(relative: &str) -> Fixture {
    let path = repo_path(relative);
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("reading {}: {error}", path.display()));
    serde_json::from_str(&text)
        .unwrap_or_else(|error| panic!("parsing {}: {error}", path.display()))
}

fn net(relative: &str) -> PolicyValueNet {
    PolicyValueNet::load(repo_path(relative)).expect("artifact loads")
}

impl Sample {
    fn encoded(&self) -> Encoded {
        assert_eq!(self.sym.len(), CELLS, "fixture sample is 12x12");
        let mut sym = [0u8; CELLS];
        sym.copy_from_slice(&self.sym);
        Encoded {
            sym,
            moves_left: self.ml,
            nu_own: self.nuo != 0,
            nu_opp: self.nux != 0,
        }
    }
}

/// Worst absolute deviation over every head output of every sample.
fn check(label: &str, weights: &str, fixture_path: &str, expect_value: bool) -> f32 {
    let simd = check_with(label, weights, fixture_path, expect_value, false);
    let scalar = check_with(
        &format!("{label}/scalar"),
        weights,
        fixture_path,
        expect_value,
        true,
    );
    simd.max(scalar)
}

fn check_with(
    label: &str,
    weights: &str,
    fixture_path: &str,
    expect_value: bool,
    force_scalar: bool,
) -> f32 {
    let mut net = net(weights);
    if force_scalar {
        net.force_scalar();
    }
    let fixture = fixture(fixture_path);
    assert!(
        (net.pair_bias() - fixture.pair_bias).abs() <= TOL,
        "{label}: pair_bias {} vs fixture {}",
        net.pair_bias(),
        fixture.pair_bias
    );
    assert!(!fixture.samples.is_empty(), "{label}: fixture has samples");
    assert_eq!(
        net.has_value_head(),
        expect_value,
        "{label}: value head presence"
    );

    let mut scratch = net.scratch();
    let mut worst = 0.0f32;
    for (s, sample) in fixture.samples.iter().enumerate() {
        let heads = net.forward(&sample.encoded(), &mut scratch);
        for i in 0..CELLS {
            let dm = (heads.move_logits[i] - sample.move_logits[i]).abs();
            let dp = (heads.pair_u[i] - sample.pair_u[i]).abs();
            assert!(
                dm <= TOL,
                "{label}: sample {s} move logit {i}: {} vs {} (delta {dm})",
                heads.move_logits[i],
                sample.move_logits[i]
            );
            assert!(
                dp <= TOL,
                "{label}: sample {s} pair u {i}: {} vs {} (delta {dp})",
                heads.pair_u[i],
                sample.pair_u[i]
            );
            worst = worst.max(dm).max(dp);
        }
        match (sample.value, heads.value) {
            (Some(expected), Some(actual)) => {
                assert!(
                    expected.abs() <= 1.0,
                    "{label}: sample {s} fixture value is a tanh output"
                );
                let dv = (actual - expected).abs();
                assert!(
                    dv <= TOL,
                    "{label}: sample {s} value: {actual} vs {expected} (delta {dv})"
                );
                worst = worst.max(dv);
            }
            (Some(_), None) => panic!("{label}: sample {s} has a value but the net has no head"),
            (None, _) => {}
        }
    }
    worst
}

#[test]
fn policy_heads_match_the_python_fixture() {
    let worst = check(
        "policy",
        "artifacts/mcts_policy.json",
        "fixtures/mcts/mcts_policy_parity.json",
        false,
    );
    println!("policy parity: worst absolute deviation {worst:e} (tolerance {TOL:e})");
}

#[test]
fn value_and_policy_heads_match_the_python_fixture() {
    let worst = check(
        "value",
        "fixtures/mcts/mcts_selfplay_tiny.json",
        "fixtures/mcts/mcts_value_parity.json",
        true,
    );
    println!("value parity: worst absolute deviation {worst:e} (tolerance {TOL:e})");
}

/// The searcher consumes a softmax, not raw logits, and the softmax is where
/// parity has to hold. Comparing the distributions directly bounds the error in
/// the quantity that actually reaches PUCT.
#[test]
fn softmax_over_the_board_agrees_far_inside_the_logit_band() {
    let net = net("artifacts/mcts_policy.json");
    let fixture = fixture("fixtures/mcts/mcts_policy_parity.json");
    let mut scratch = net.scratch();
    let mut worst = 0.0f32;
    for sample in &fixture.samples {
        let heads = net.forward(&sample.encoded(), &mut scratch);
        let ours = softmax(&heads.move_logits);
        let theirs = softmax(&sample.move_logits);
        for (a, b) in ours.iter().zip(&theirs) {
            worst = worst.max((a - b).abs());
        }
    }
    println!("policy softmax: worst absolute probability deviation {worst:e}");
    assert!(worst <= 1e-4, "softmax deviation {worst:e} exceeds 1e-4");
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.into_iter().map(|e| e / sum).collect()
}

/// The AVX2 convolution and the portable one must both land inside the parity
/// band, and must agree with each other far more tightly than the band allows.
///
/// This is the test the `unsafe` in `net.rs` is justified by: the fast path is
/// a `#[target_feature]` recompilation of the same source, so the only thing
/// that can differ is FP rounding. In practice nothing does — rustc does not
/// contract multiply-adds, so the two agree exactly and the speedup is purely
/// vector width — but the assertion is a band, not equality, so that a
/// toolchain which did start contracting would still pass.
#[test]
fn both_convolution_paths_hit_parity() {
    let net = net("artifacts/mcts_policy.json");
    println!("simd convolution available: {}", net.simd());
    let mut scalar = net.clone();
    scalar.force_scalar();
    assert!(!scalar.simd());

    let fixture = fixture("fixtures/mcts/mcts_policy_parity.json");
    let (mut a, mut b) = (net.scratch(), scalar.scratch());
    let mut worst = 0.0f32;
    for sample in &fixture.samples {
        let encoded = sample.encoded();
        let fast = net.forward(&encoded, &mut a);
        let slow = scalar.forward(&encoded, &mut b);
        for i in 0..CELLS {
            worst = worst
                .max((fast.move_logits[i] - slow.move_logits[i]).abs())
                .max((fast.pair_u[i] - slow.pair_u[i]).abs());
        }
    }
    println!("simd vs scalar: worst absolute deviation {worst:e}");
    assert!(
        worst <= 1e-5,
        "the two convolution paths diverged by {worst:e}"
    );
}

/// The champion is the artifact the bot actually plays: it must load, declare
/// the expected architecture, and carry a value head.
#[test]
fn the_gen5_champion_loads() {
    let net = net("artifacts/mcts_champion.json");
    assert_eq!(net.arch(), "conv-policy-value-v1");
    assert_eq!(net.channels(), 32);
    assert_eq!(net.layers(), 4);
    assert!(net.has_value_head(), "gen-5 champion carries a value head");
}

// ------------------------------------------------------- load-time validation

/// A minimal but well-formed artifact, as a template for the rejection tests.
fn tiny_artifact() -> serde_json::Value {
    let kernel = vec![vec![0.01f64; 3]; 3];
    serde_json::json!({
        "meta": {"arch": "conv-policy-value-v1", "board": 12, "planes": 13, "channels": 2, "layers": 1},
        "conv": [{"w": vec![vec![kernel.clone(); 13]; 2], "b": [0.0, 0.0]}],
        "move_head": {"w": [[[0.5]], [[0.25]]], "b": 0.1},
        "pair_head": {"w": [[[0.2]], [[0.3]]], "b": -0.1},
        "pair_bias": -0.05,
        "value_head": {"fc1_w": [[0.1, 0.2]], "fc1_b": [0.0], "fc2_w": [1.0], "fc2_b": 0.0}
    })
}

fn load(json: &serde_json::Value) -> Result<PolicyValueNet, virus_mcts::NetError> {
    PolicyValueNet::from_json(&json.to_string())
}

#[test]
fn a_well_formed_tiny_artifact_loads() {
    let net = load(&tiny_artifact()).expect("template artifact is valid");
    assert_eq!(net.channels(), 2);
    assert!(net.has_value_head());
}

#[test]
fn a_wrong_board_is_rejected_at_load() {
    let mut json = tiny_artifact();
    json["meta"]["board"] = serde_json::json!(19);
    assert!(load(&json).is_err(), "19x19 must not load");
}

#[test]
fn a_conv_layer_count_mismatch_is_rejected_at_load() {
    let mut json = tiny_artifact();
    json["meta"]["layers"] = serde_json::json!(2);
    assert!(load(&json).is_err(), "declared 2 layers, supplied 1");
}

#[test]
fn a_short_conv_row_is_rejected_at_load() {
    let mut json = tiny_artifact();
    json["conv"][0]["w"][0].as_array_mut().unwrap().truncate(12);
    assert!(load(&json).is_err(), "12 input channels instead of 13");
}

#[test]
fn a_short_bias_vector_is_rejected_at_load() {
    let mut json = tiny_artifact();
    json["conv"][0]["b"] = serde_json::json!([0.0]);
    assert!(load(&json).is_err(), "1 bias for 2 channels");
}

#[test]
fn a_non_finite_weight_is_rejected_at_load() {
    let mut json = tiny_artifact();
    // JSON has no NaN literal; an overflowing magnitude is how a corrupt export
    // actually shows up, and it narrows to `inf` in f32.
    json["conv"][0]["w"][0][0][0][0] = serde_json::json!(1e300);
    let error = load(&json).expect_err("infinite weight must not load");
    assert!(
        format!("{error}").contains("not finite"),
        "unhelpful error: {error}"
    );
}

#[test]
fn a_mis_sized_head_is_rejected_at_load() {
    let mut json = tiny_artifact();
    json["move_head"]["w"] = serde_json::json!([[[0.5]]]);
    assert!(load(&json).is_err(), "1 head channel for a 2-channel trunk");
}

/// A head kernel wider than 1x1 is a different architecture. Reading only its
/// top-left weight would run a model this crate does not implement, so the
/// loader must reject it rather than quietly truncate.
#[test]
fn a_head_kernel_wider_than_1x1_is_rejected_at_load() {
    let mut json = tiny_artifact();
    json["move_head"]["w"] = serde_json::json!([[[0.5, 0.4]], [[0.25, 0.1]]]);
    let error = load(&json).expect_err("a 1x2 head kernel must not load");
    assert!(
        format!("{error}").contains("1x1 kernel"),
        "unhelpful error: {error}"
    );

    let mut json = tiny_artifact();
    json["pair_head"]["w"] = serde_json::json!([[[0.2], [0.9]], [[0.3], [0.7]]]);
    assert!(load(&json).is_err(), "a 2x1 head kernel must not load");
}

#[test]
fn a_mis_sized_value_head_row_is_rejected_at_load() {
    let mut json = tiny_artifact();
    json["value_head"]["fc1_w"] = serde_json::json!([[0.1]]);
    assert!(load(&json).is_err(), "fc1 row is 1 wide for 2 channels");
}

#[test]
fn a_policy_only_artifact_loads_without_a_value_head() {
    let mut json = tiny_artifact();
    json.as_object_mut().unwrap().remove("value_head");
    let net = load(&json).expect("policy-only artifacts stay loadable");
    assert!(!net.has_value_head());
    let mut scratch = net.scratch();
    let heads = net.forward(
        &Encoded {
            sym: [0; CELLS],
            moves_left: 3,
            nu_own: false,
            nu_opp: false,
        },
        &mut scratch,
    );
    assert!(heads.value.is_none(), "no head, no value");
}
