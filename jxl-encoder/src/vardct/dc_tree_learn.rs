// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

// Module contains experimental/WIP code with some unused items and complex types.
// Allow various clippy warnings that don't affect correctness.
#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(clippy::type_complexity)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::let_and_return)]

//! DC coefficient tree learning for VarDCT encoding.
//!
//! Learns an optimal context tree for DC coding based on image content,
//! replacing the fixed GRADIENT_CONTEXT_LUT with a data-driven tree.
//! This can provide 0.3-1.0% compression improvement on DC stream.
//!
//! Port of libjxl's DC tree learning from `enc_modular.cc`.

use super::common::pack_signed;
use super::dc_coding::clamped_gradient;

/// Number of properties used in DC tree learning.
///
/// Must match libjxl's `kNumNonrefProperties = kNumStaticProperties(2) + 13 +
/// weighted::kNumProperties(1) = 16` (`context_predict.h:378-379`) and jxl-rs
/// decoder's property buffer layout:
/// - 0: channel (static, set by caller)
/// - 1: group_id/stream (static, typically 0 for DC)
/// - 2: y position
/// - 3: x position
/// - 4: |top|
/// - 5: |left|
/// - 6: top
/// - 7: left
/// - 8: local gradient (left - prev_left, maintained across row)
/// - 9: gradient (left + top - topleft) ← KEY SPLIT PROPERTY
/// - 10: left - topleft (FFV1)
/// - 11: topleft - top (FFV1)
/// - 12: top - topright (FFV1)
/// - 13: top - toptop (FFV1)
/// - 14: left - leftleft (FFV1)
/// - 15: wp_max_error (WeightedPredictor max abs error among teW/teN/teNW/teNE)
///   — `kWPProp = kNumNonrefProperties - 1 = 15` (`context_predict.h:381`).
const NUM_DC_PROPERTIES: usize = 16;

/// Number of candidate predictors per sample in Variable mode.
/// Matches libjxl `kNumModularPredictors = 14` (`modular/options.h:47` references
/// `Predictor::Variable + 1`; 14 simple predictors enumerated 0..=13).
const NUM_PREDICTORS_VARIABLE: usize = 14;

/// Properties to consider for splits.
/// Property 9 (gradient) is the most effective for DC coding; property 15
/// (wp_max_error) is what `kWPFixedDC` splits on and what libjxl's
/// `kNumNonrefProperties` Variable-mode learner ranks all the way through.
const SPLIT_PROPERTIES: &[usize] = &[
    9,  // gradient (left + top - topleft) - most important
    4,  // |top|
    5,  // |left|
    6,  // top
    7,  // left
    10, // left - topleft (FFV1)
];

/// Properties to consider for splits in Variable mode (stage 7c).
/// Adds wp_max_error (15) — required to access WP's adaptive prediction error
/// in the split decision, matching libjxl's `kWPProp` ranking in `FindBestSplit`.
const SPLIT_PROPERTIES_VARIABLE: &[usize] = &[
    9,  // gradient (left + top - topleft)
    15, // wp_max_error (kWPProp)
    4,  // |top|
    5,  // |left|
    6,  // top
    7,  // left
    10, // left - topleft (FFV1)
    11, // topleft - top (FFV1)
    12, // top - topright (FFV1)
    13, // top - toptop (FFV1)
    14, // left - leftleft (FFV1)
    8,  // local gradient
    2,  // y position
    3,  // x position
];

// W44-phase3-B3-RAYON-1 (2026-05-23) [RULED OUT]: chunking the
// `parallel_map(SPLIT_PROPERTIES_VARIABLE.len(), scan_property)` fan-out
// below — via `parallel_map_chunked(14, 7, ...)` — regressed 8-thread wall
// by 4-11 % across all 4 test cells (paired bench
// `benchmarks/w44_phase3_b3_rayon_1_dc_tree_batch_2026-05-23.{tsv,meta}`).
// Root cause: 14 fine-grained tasks already saturate the 8-worker pool in
// two waves; 2 chunked tasks of 7 leave 6 workers idle. Per-task overhead
// recovery (~3 pp) is overwhelmed by parallelism loss (8-12 pp). DO NOT
// respawn this chunk or its B3-RAYON-2 sibling (encode_ans histogram
// guard) without re-validating the trade-off premise on the target site.
// See `docs/LIBJXL_DIVERGENCES.md` Section F row "W44-phase3-B3-RAYON-1
// RULED OUT" and memo `~/.claude/projects/-home-lilith-work-zen-jxl-encoder/
// memory/w44_phase3_b3_rayon_1_dc_tree_batch_2026-05-23.md`.

/// Maximum tree depth to prevent overfitting.
const MAX_TREE_DEPTH: usize = 8;

/// Minimum samples per leaf to prevent overfitting.
const MIN_SAMPLES_PER_LEAF: usize = 64;

/// Predictor set evaluated by [`learn_dc_tree_variable_with_set`].
///
/// Mirrors libjxl's `Predictor::Variable` (all 14 simple predictors) vs
/// `Predictor::Best` (only Weighted + Gradient) meta-modes from
/// `modular/encoding/enc_ma.cc:542-552`.
///
/// libjxl picks one of these based on `speed_tier`
/// (`enc_modular.cc:1591-1597`):
///   - `speed_tier < SpeedTier::kKitten` (effort >= 9): `Predictor::Variable`
///   - `speed_tier < SpeedTier::kSquirrel` AND `speed_tier >= SpeedTier::kKitten`
///     (effort == 8): `Predictor::Best`
///   - otherwise (effort <= 7): `kWPFixedDC` (no `kLearn`)
///
/// W44-172 (2026-05-21): the W44-54 implementation always used
/// `Predictor::Variable` regardless of effort, evaluating 14 predictors per
/// split candidate × ~32 candidates × per node. This made
/// `build_tree_recursive_variable` consume ~48% of e8 CPU on 5+ MP
/// screenshots (terminal e8 d=0.5: 32× cjxl wall ratio). `PredictorSet::Best`
/// at e8 restores libjxl parity by limiting the predictor set to the same 2
/// predictors libjxl evaluates at kKitten.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PredictorSet {
    /// All 14 simple predictors (libjxl `Predictor::Variable`).
    /// Used at libjxl `speed_tier < kKitten` (our effort >= 9).
    Variable,
    /// Only Weighted (6) + Gradient (5) (libjxl `Predictor::Best`).
    /// Used at libjxl `speed_tier == kKitten` (our effort == 8).
    Best,
}

impl PredictorSet {
    /// Iterator over predictor indices to evaluate. For `Best` mode this is
    /// `&[Weighted, Gradient]` mirroring libjxl `enc_ma.cc:549`. For
    /// `Variable` it covers `0..=13` matching `enc_ma.cc:543`.
    #[inline]
    pub fn predictor_indices(self) -> &'static [u32] {
        match self {
            // Order matches libjxl `enc_ma.cc:543-547`: Variable swaps
            // Weighted → predictors[0], Gradient → predictors[1], then the
            // remaining 12 predictors in enum order.
            PredictorSet::Variable => &[
                6, // Weighted (swapped to slot 0 by libjxl)
                5, // Gradient (swapped to slot 1 by libjxl)
                0, // Zero
                1, // Left
                2, // Top
                3, // Average0
                4, // Select
                7, // TopRight
                8, // TopLeft
                9, // LeftLeft
                10, 11, 12, 13,
            ],
            // libjxl `enc_ma.cc:549`: `predictors = {Weighted, Gradient}`.
            PredictorSet::Best => &[6, 5],
        }
    }
}

/// HybridUint config for sample gathering: {4, 1, 2}.
const GATHER_SPLIT: u32 = 16; // 1 << 4
const GATHER_MSB_IN_TOKEN: u32 = 1;
const GATHER_LSB_IN_TOKEN: u32 = 2;

/// Encode a value using HybridUint config for gathering.
#[inline]
fn encode_hybrid_uint(value: u32) -> u32 {
    if value < GATHER_SPLIT {
        value
    } else {
        let n = 32 - value.leading_zeros(); // floor_log2(value) + 1
        let n_minus_split_exp = n - 4 - 1; // n - split_exponent - 1
        let token = GATHER_SPLIT + n_minus_split_exp * (GATHER_MSB_IN_TOKEN + GATHER_LSB_IN_TOKEN);
        token
    }
}

/// Collected samples for DC tree learning.
///
/// Holds residual tokens for every candidate predictor (Variable mode) and
/// property values for every sample. Backward-compatible: `add_sample` keeps
/// gradient-only behaviour for the stage 1-6 path (extra predictor slots stay
/// empty unless `add_sample_variable` is used).
pub struct DcTreeSamples {
    /// Number of samples collected.
    pub num_samples: usize,
    /// Gradient-predictor residual tokens (stage 1-6 path).
    residual_tokens: Vec<u32>,
    /// Per-predictor residual tokens (stage 7 Variable path). Index by
    /// `Predictor` id 0..=13 (`Predictor::all_simple()`).
    /// Empty until `add_sample_variable` is called.
    residual_tokens_per_predictor: Vec<Vec<u32>>,
    /// Property values: `props[property_idx][sample_idx]`. Size:
    /// `NUM_DC_PROPERTIES = 16` (stage 7b extended set).
    props: Vec<Vec<i32>>,
}

impl Default for DcTreeSamples {
    fn default() -> Self {
        Self::new()
    }
}

impl DcTreeSamples {
    /// Creates an empty DcTreeSamples structure.
    pub fn new() -> Self {
        Self {
            num_samples: 0,
            residual_tokens: Vec::new(),
            residual_tokens_per_predictor: Vec::new(),
            props: vec![Vec::new(); NUM_DC_PROPERTIES],
        }
    }

    /// Add a sample with its properties and a single gradient-predictor residual.
    /// Stage 1-6 compatibility path. Property 15 (`wp_max_error`) should be 0
    /// when WP state isn't tracked — splits on it become no-ops.
    #[inline]
    pub fn add_sample(&mut self, residual: i32, props: [i32; NUM_DC_PROPERTIES]) {
        let packed = pack_signed(residual);
        let token = encode_hybrid_uint(packed);
        self.residual_tokens.push(token);

        for (i, &p) in props.iter().enumerate() {
            self.props[i].push(p);
        }
        self.num_samples += 1;
    }

    /// Add a sample with multi-predictor residuals (stage 7 Variable mode).
    ///
    /// `residuals[i]` is the residual when predictor `i` (per `Predictor`
    /// enum: 0=Zero, 1=Left, 2=Top, 3=Average0, 4=Select, 5=Gradient,
    /// 6=Weighted, 7=TopRight, 8=TopLeft, 9=LeftLeft, 10=Average1, 11=Average2,
    /// 12=Average3, 13=Average4) is subtracted from the actual DC value.
    ///
    /// Mirrors libjxl `TreeSamples::AddSample` (`enc_ma.cc:711-730`) which
    /// stores one tokenized residual per predictor for later per-leaf
    /// best-predictor selection in `FindBestSplit`.
    #[inline]
    pub fn add_sample_variable(
        &mut self,
        residuals: [i32; NUM_PREDICTORS_VARIABLE],
        props: [i32; NUM_DC_PROPERTIES],
    ) {
        if self.residual_tokens_per_predictor.is_empty() {
            // Lazy-init per-predictor arrays on first variable-mode sample.
            // NOTE: `vec![Vec::with_capacity(..); N]` clones the template,
            // and a clone of an empty Vec does NOT retain capacity — every
            // slot would start at capacity 0. Build each slot explicitly.
            self.residual_tokens_per_predictor = (0..NUM_PREDICTORS_VARIABLE)
                .map(|_| Vec::with_capacity(self.num_samples + 1))
                .collect();
        }
        for (i, &r) in residuals.iter().enumerate() {
            let packed = pack_signed(r);
            let token = encode_hybrid_uint(packed);
            self.residual_tokens_per_predictor[i].push(token);
        }
        // Mirror gradient residual into legacy slot for stage 1-6 callers.
        let grad_token = encode_hybrid_uint(pack_signed(
            residuals[crate::modular::Predictor::Gradient as usize],
        ));
        self.residual_tokens.push(grad_token);

        for (i, &p) in props.iter().enumerate() {
            self.props[i].push(p);
        }
        self.num_samples += 1;
    }

    /// Returns true if multi-predictor samples were gathered (`add_sample_variable`).
    pub fn has_variable_residuals(&self) -> bool {
        !self.residual_tokens_per_predictor.is_empty()
    }
}

/// Compute properties for a DC value given its neighbors.
#[inline]
/// Compute DC properties matching jxl-rs decoder's property buffer layout.
///
/// # Arguments
/// * `channel_idx` - Channel index in encoding order (0=Y, 1=X, 2=B after reorder)
/// * `x` - X position in block coordinates
/// * `y` - Y position in block coordinates
/// * `top` - DC value of block above
/// * `left` - DC value of block to the left
/// * `topleft` - DC value of block diagonally above-left
/// * `topright` - DC value of block diagonally above-right
/// * `toptop` - DC value of block two rows above
/// * `leftleft` - DC value of block two columns left
/// * `prev_local_grad` - Previous local gradient (for property 8)
///
/// Returns (properties, new_local_grad) where new_local_grad should be passed
/// as prev_local_grad for the next pixel in the row.
pub fn compute_dc_properties(
    channel_idx: u32,
    x: usize,
    y: usize,
    top: i32,
    left: i32,
    topleft: i32,
    topright: i32,
    toptop: i32,
    leftleft: i32,
    prev_local_grad: i32,
) -> ([i32; NUM_DC_PROPERTIES], i32) {
    let mut props = [0i32; NUM_DC_PROPERTIES];

    // Static properties
    props[0] = channel_idx as i32;
    props[1] = 0; // group_id/stream, typically 0 for DC

    // Position
    props[2] = y as i32;
    props[3] = x as i32;

    // Absolute neighbors
    props[4] = top.wrapping_abs();
    props[5] = left.wrapping_abs();

    // Raw neighbors
    props[6] = top;
    props[7] = left;

    // Local gradient (left - prev_local_grad) - maintained across row
    let local_grad = left.wrapping_add(top).wrapping_sub(topleft);
    props[8] = left.wrapping_sub(prev_local_grad);

    // Gradient (left + top - topleft) - PRIMARY SPLIT PROPERTY
    props[9] = local_grad;

    // FFV1 context properties
    props[10] = left.wrapping_sub(topleft);
    props[11] = topleft.wrapping_sub(top);
    props[12] = top.wrapping_sub(topright);
    props[13] = top.wrapping_sub(toptop);
    props[14] = left.wrapping_sub(leftleft);

    (props, local_grad)
}

/// Gather DC samples with multi-predictor residuals + WP state (Variable mode).
///
/// Mirrors libjxl's per-pixel sample gathering when
/// `SetPredictor(Predictor::Variable)` is configured:
/// for each DC value, it computes:
/// - All 14 simple-predictor predictions (Zero through Average4) via
///   `Predictor::predict_from_neighbors` — matches `PredictOne` for
///   non-Weighted predictors (`context_predict.h:472-516`).
/// - The Weighted predictor's prediction + `wp_max_error` property by running
///   `WeightedPredictorState::predict_and_property` — matches
///   `weighted::State::Predict<true>` (`context_predict.h:133-193`).
///
/// Then stores `[i32; 14]` residuals (one per predictor) plus the 16-property
/// vector (property 15 = `wp_max_error`) via `add_sample_variable`.
///
/// Each channel gets a FRESH `WeightedPredictorState` — matches libjxl's
/// per-channel processing pattern.
pub fn gather_dc_samples_variable(samples: &mut DcTreeSamples, quant_dc: &[Vec<Vec<i16>>; 3]) {
    use crate::modular::predictor::{Neighbors, Predictor, WeightedPredictorState};

    if quant_dc[0].is_empty() || quant_dc[0][0].is_empty() {
        return;
    }

    let height = quant_dc[0].len();
    let width = quant_dc[0][0].len();

    // Gather in encoding channel order: Y (1), X (0), B (2)
    for (enc_idx, &c) in [1usize, 0, 2].iter().enumerate() {
        let channel = &quant_dc[c];
        let mut wp_state = WeightedPredictorState::with_defaults(width);

        for y in 0..height {
            let mut prev_local_grad = 0i32;

            for x in 0..width {
                let dc_val = channel[y][x] as i32;

                // Edge handling matching jxl-rs decoder + libjxl's PredictionData.
                let left = if x > 0 {
                    channel[y][x - 1] as i32
                } else if y > 0 {
                    channel[y - 1][x] as i32
                } else {
                    0
                };
                let top = if y > 0 {
                    channel[y - 1][x] as i32
                } else {
                    left
                };
                let topleft = if x > 0 && y > 0 {
                    channel[y - 1][x - 1] as i32
                } else {
                    left
                };
                let topright = if y > 0 && x + 1 < width {
                    channel[y - 1][x + 1] as i32
                } else {
                    top
                };
                let toptop = if y > 1 { channel[y - 2][x] as i32 } else { top };
                let leftleft = if x > 1 {
                    channel[y][x - 2] as i32
                } else {
                    left
                };
                let nee = if y > 0 && x + 2 < width {
                    channel[y - 1][x + 2] as i32
                } else {
                    topright
                };

                let neighbors = Neighbors {
                    n: top,
                    w: left,
                    nw: topleft,
                    ne: topright,
                    nn: toptop,
                    ww: leftleft,
                    nee,
                };

                // 14 simple predictions
                let mut residuals = [0i32; NUM_PREDICTORS_VARIABLE];
                for (i, pred) in Predictor::all_simple().iter().enumerate() {
                    if *pred == Predictor::Weighted {
                        continue; // filled below via WP state
                    }
                    let p = pred.predict_from_neighbors(&neighbors);
                    residuals[i] = dc_val - p;
                }

                // WP prediction + max_error property
                let (wp_pred, wp_max_error) =
                    wp_state.predict_and_property(x, y, width, &neighbors);
                residuals[Predictor::Weighted as usize] = dc_val - wp_pred as i32;

                // Update WP error state with actual value
                wp_state.update_errors(dc_val, x, y, width);

                // Compute extended property vector (16 entries, prop 15 = wp_max_error)
                let (mut props, new_local_grad) = compute_dc_properties(
                    enc_idx as u32,
                    x,
                    y,
                    top,
                    left,
                    topleft,
                    topright,
                    toptop,
                    leftleft,
                    prev_local_grad,
                );
                props[15] = wp_max_error;

                samples.add_sample_variable(residuals, props);

                prev_local_grad = new_local_grad;
            }
        }
    }
}

/// Gather DC samples from quantized DC values.
///
/// # Arguments
/// * `samples` - Sample collection to add to
/// * `quant_dc` - Quantized DC values [channel][y][x]
pub fn gather_dc_samples(samples: &mut DcTreeSamples, quant_dc: &[Vec<Vec<i16>>; 3]) {
    if quant_dc[0].is_empty() || quant_dc[0][0].is_empty() {
        return;
    }

    let height = quant_dc[0].len();
    let width = quant_dc[0][0].len();

    // Gather in encoding channel order: Y (1), X (0), B (2)
    for (enc_idx, &c) in [1usize, 0, 2].iter().enumerate() {
        let channel = &quant_dc[c];

        for y in 0..height {
            let mut prev_local_grad = 0i32;

            for x in 0..width {
                let dc_val = channel[y][x] as i32;

                // Get neighbors with edge handling matching jxl-rs decoder
                let left = if x > 0 {
                    channel[y][x - 1] as i32
                } else if y > 0 {
                    channel[y - 1][x] as i32
                } else {
                    0
                };

                let top = if y > 0 {
                    channel[y - 1][x] as i32
                } else {
                    left
                };

                let topleft = if x > 0 && y > 0 {
                    channel[y - 1][x - 1] as i32
                } else {
                    left
                };

                let topright = if y > 0 && x + 1 < width {
                    channel[y - 1][x + 1] as i32
                } else {
                    top
                };

                let toptop = if y > 1 { channel[y - 2][x] as i32 } else { top };

                let leftleft = if x > 1 {
                    channel[y][x - 2] as i32
                } else {
                    left
                };

                // Compute prediction and residual
                let prediction = clamped_gradient(top, left, topleft);
                let residual = dc_val - prediction;

                // Compute properties and add sample
                let (props, new_local_grad) = compute_dc_properties(
                    enc_idx as u32,
                    x,
                    y,
                    top,
                    left,
                    topleft,
                    topright,
                    toptop,
                    leftleft,
                    prev_local_grad,
                );
                samples.add_sample(residual, props);

                prev_local_grad = new_local_grad;
            }
        }
    }
}

/// A decision tree node for DC context assignment.
#[derive(Clone, Debug)]
pub struct DcTreeNode {
    /// Property to split on (-1 for leaf).
    pub property: i32,
    /// Split value (samples with property <= splitval go left).
    pub splitval: i32,
    /// Left child index (for internal nodes).
    pub lchild: usize,
    /// Right child index (for internal nodes).
    pub rchild: usize,
    /// Context ID (for leaf nodes).
    pub context_id: u32,
    /// Predictor for leaf nodes (0=Zero, 5=Gradient, etc.)
    pub predictor: u32,
}

impl Default for DcTreeNode {
    fn default() -> Self {
        Self {
            property: -1,
            splitval: 0,
            lchild: 0,
            rchild: 0,
            context_id: 0,
            predictor: 5, // Default: Gradient (matches DC prediction)
        }
    }
}

/// A learned DC context tree.
pub type DcTree = Vec<DcTreeNode>;

/// Estimate bits needed to encode tokens with a given distribution.
fn estimate_bits(counts: &[u32], total: u32) -> f64 {
    if total == 0 {
        return 0.0;
    }
    let total_f = total as f64;
    let mut bits = 0.0;

    for &count in counts {
        if count > 0 {
            let p = count as f64 / total_f;
            bits -= (count as f64) * jxl_simd::fast_log2f(p as f32) as f64;
        }
    }
    bits
}

/// Estimate entropy cost for a subset of samples.
fn estimate_subset_cost(samples: &DcTreeSamples, indices: &[usize], max_token: u32) -> f64 {
    if indices.is_empty() {
        return 0.0;
    }

    let histogram_size = (max_token + 1) as usize;
    let mut counts = vec![0u32; histogram_size];
    let mut total = 0u32;

    for &idx in indices {
        let tok = samples.residual_tokens[idx];
        if (tok as usize) < histogram_size {
            counts[tok as usize] += 1;
            total += 1;
        }
    }

    estimate_bits(&counts, total)
}

/// Find the best split for a set of samples.
///
/// Returns (property_idx, splitval, left_indices, right_indices, gain)
/// where gain is the entropy reduction from the split.
fn find_best_split(
    samples: &DcTreeSamples,
    indices: &[usize],
    max_token: u32,
) -> Option<(usize, i32, Vec<usize>, Vec<usize>, f64)> {
    if indices.len() < MIN_SAMPLES_PER_LEAF * 2 {
        return None;
    }

    let current_cost = estimate_subset_cost(samples, indices, max_token);
    let mut best_gain = 0.0f64;
    let mut best_split: Option<(usize, i32, Vec<usize>, Vec<usize>)> = None;

    for &prop_idx in SPLIT_PROPERTIES {
        // Collect unique split values for this property
        let props = &samples.props[prop_idx];
        let mut values: Vec<i32> = indices.iter().map(|&i| props[i]).collect();
        values.sort_unstable();
        values.dedup();

        // Try splits at quantile boundaries (for efficiency)
        let num_quantiles = 32.min(values.len() - 1);
        if num_quantiles == 0 {
            continue;
        }

        for q in 0..num_quantiles {
            let split_idx = (values.len() * (q + 1)) / (num_quantiles + 1);
            if split_idx == 0 || split_idx >= values.len() {
                continue;
            }
            let splitval = values[split_idx - 1];

            // Partition samples
            let (left, right): (Vec<usize>, Vec<usize>) =
                indices.iter().copied().partition(|&i| props[i] <= splitval);

            if left.len() < MIN_SAMPLES_PER_LEAF || right.len() < MIN_SAMPLES_PER_LEAF {
                continue;
            }

            // Compute cost reduction
            let left_cost = estimate_subset_cost(samples, &left, max_token);
            let right_cost = estimate_subset_cost(samples, &right, max_token);
            let new_cost = left_cost + right_cost;
            let gain = current_cost - new_cost;

            // Add overhead for the split itself (approximate)
            let overhead = 10.0; // bits for property + splitval encoding
            let net_gain = gain - overhead;

            if net_gain > best_gain {
                best_gain = net_gain;
                best_split = Some((prop_idx, splitval, left, right));
            }
        }
    }

    best_split.map(|(prop, sv, l, r)| (prop, sv, l, r, best_gain))
}

/// Recursively build the DC tree.
fn build_tree_recursive(
    samples: &DcTreeSamples,
    indices: &[usize],
    depth: usize,
    tree: &mut DcTree,
    next_context: &mut u32,
    max_token: u32,
) -> usize {
    let node_idx = tree.len();
    tree.push(DcTreeNode::default());

    // Check if we should make this a leaf
    if depth >= MAX_TREE_DEPTH || indices.len() < MIN_SAMPLES_PER_LEAF * 2 {
        tree[node_idx].property = -1;
        tree[node_idx].context_id = *next_context;
        *next_context += 1;
        return node_idx;
    }

    // Try to find a beneficial split
    if let Some((prop_idx, splitval, left_indices, right_indices, _gain)) =
        find_best_split(samples, indices, max_token)
    {
        // Build children first
        let lchild = build_tree_recursive(
            samples,
            &left_indices,
            depth + 1,
            tree,
            next_context,
            max_token,
        );
        let rchild = build_tree_recursive(
            samples,
            &right_indices,
            depth + 1,
            tree,
            next_context,
            max_token,
        );

        tree[node_idx].property = prop_idx as i32;
        tree[node_idx].splitval = splitval;
        tree[node_idx].lchild = lchild;
        tree[node_idx].rchild = rchild;
    } else {
        // No beneficial split found, make this a leaf
        tree[node_idx].property = -1;
        tree[node_idx].context_id = *next_context;
        *next_context += 1;
    }

    node_idx
}

/// Stage 7c: estimate per-predictor cost for a subset of samples.
///
/// Iterates the per-predictor residual tokens (populated by
/// `gather_dc_samples_variable`) and returns `[cost_pred_0, ..cost_pred_13]`.
/// Each entry is the entropy of the residual-token histogram for that
/// predictor on the subset, plus the extra-bits cost.
///
/// Mirrors libjxl `FindBestSplit` inner loop (`enc_ma.cc:206-227, 340-403`)
/// which computes both `EstimateBits` and tracks `tot_extra_bits[pred]`
/// separately. The HybridUint {4,1,2} extra-bits cost per token of value v
/// is `floor_log2(v) - 4` bits for v >= 16, 0 otherwise.
fn estimate_subset_cost_per_predictor(
    samples: &DcTreeSamples,
    indices: &[usize],
    max_token: u32,
    predictor_set: PredictorSet,
) -> [f64; NUM_PREDICTORS_VARIABLE] {
    // Predictors not in the active set return INFINITY so the `pick_best`
    // reduction skips them — matches libjxl's "only consider configured
    // predictors" semantic from `enc_ma.cc:542-552` even though that code
    // stores predictors in a vector instead of a fixed array.
    let mut out = [f64::INFINITY; NUM_PREDICTORS_VARIABLE];
    if indices.is_empty() || !samples.has_variable_residuals() {
        // Mirror prior behaviour for empty subsets: zero cost across the
        // board so the caller's gain accounting is a no-op.
        return [0.0f64; NUM_PREDICTORS_VARIABLE];
    }

    // Find true max token across the predictors we're actually going to
    // evaluate (mirrors libjxl `enc_ma.cc:207-213` but scoped to the active
    // set so `Best` mode doesn't oversize the histogram for predictors it
    // never touches). Falls back to `max_token` if the active set is empty
    // (defensive — predictor_indices() is never empty in practice).
    let pred_indices = predictor_set.predictor_indices();
    let mut true_max_token: u32 = 0;
    for &pred in pred_indices {
        let pred_tokens = &samples.residual_tokens_per_predictor[pred as usize];
        for &idx in indices {
            let t = pred_tokens[idx];
            if t > true_max_token {
                true_max_token = t;
            }
        }
    }
    if pred_indices.is_empty() {
        true_max_token = max_token;
    }
    let histogram_size = (true_max_token + 1) as usize;
    let mut counts = vec![0u32; histogram_size];

    for &pred_id in pred_indices {
        let pred = pred_id as usize;
        counts.iter_mut().for_each(|c| *c = 0);
        let mut total = 0u32;
        let mut extra_bits = 0.0f64;
        let tokens = &samples.residual_tokens_per_predictor[pred];
        for &idx in indices {
            let tok = tokens[idx];
            counts[tok as usize] += 1;
            total += 1;
            // HybridUint {4,1,2} extra bits: tokens >= 16 carry MSB+LSB bits.
            // libjxl `enc_ma.cc:218-225` tracks `extra_bits = rt.nbits * count`.
            // For our gather config (split_exponent=4, msb=1, lsb=2):
            //   token < 16: no extra bits
            //   token >= 16: bits in the encoded residual = (token-16)/3 * 1 + 2
            //                ≈ token-15 bits at worst (we approximate as the
            //                payload-bit count, sufficient for relative ranking).
            if tok >= GATHER_SPLIT {
                let n_minus_split_exp =
                    (tok - GATHER_SPLIT) / (GATHER_MSB_IN_TOKEN + GATHER_LSB_IN_TOKEN);
                // Extra bits = msb_count_used + lsb_count
                // For msb=1, lsb=2: extra_bits = 1 + 2 = 3 per token of this magnitude class
                extra_bits += (n_minus_split_exp as f64) + 2.0;
            }
        }
        out[pred] = estimate_bits(&counts, total) + extra_bits;
    }
    out
}

/// Stage 7c: find best split for Variable-mode samples.
///
/// At each candidate split: for each candidate predictor, compute the entropy
/// cost on the left + right side. The best split is the one minimizing
/// `min_pred(left_cost(pred)) + min_pred(right_cost(pred))`, with each side's
/// best predictor recorded.
///
/// Mirrors libjxl `FindBestSplit` (`enc_ma.cc:158-457`) — specifically the
/// per-predictor `costs_l[i]` / `costs_r[i]` accumulation and the
/// "pick best (cost, pred)" reduction. Penalties from libjxl (change-pred,
/// favor-Zero, disfavor-Weighted) match the source.
///
/// Returns (property_idx, splitval, left_indices, right_indices,
///          best_left_predictor, best_right_predictor, gain).
fn find_best_split_variable(
    samples: &DcTreeSamples,
    indices: &[usize],
    max_token: u32,
    current_predictor: u32,
    threshold: f64,
    predictor_set: PredictorSet,
) -> Option<(usize, i32, Vec<usize>, Vec<usize>, u32, u32, f64)> {
    if indices.len() < MIN_SAMPLES_PER_LEAF * 2 {
        return None;
    }
    if !samples.has_variable_residuals() {
        return None;
    }

    // Current (unsplit) cost with the current predictor — used as baseline.
    let base_per_pred =
        estimate_subset_cost_per_predictor(samples, indices, max_token, predictor_set);
    let base_cost = base_per_pred[current_predictor as usize];

    // Change-predictor penalty (libjxl enc_ma.cc:302): lower threshold = more noisy
    // estimates → discourage switching predictors. We use threshold=0 for DC
    // (libjxl ModularOptions default), but match the formula anyway.
    let change_pred_penalty = 800.0 / (100.0 + threshold);

    let pred_indices = predictor_set.predictor_indices();

    // W44-175: per-property parallel scan.
    //
    // Each property's (sort + dedup + per-quantile cost evaluation) is fully
    // independent — no shared mutable state, no order dependency on the
    // cost numbers themselves. The only ordering concern is tie-breaking:
    // the sequential code's `if net_gain > best_gain` keeps the FIRST
    // candidate when two split candidates produce equal `net_gain`, walking
    // properties in `SPLIT_PROPERTIES_VARIABLE` order and quantiles in
    // ascending `q` order within each property. We preserve that exact
    // tiebreaker by:
    //   (1) Computing each property's best candidate independently in
    //       parallel, returning `(prop_rank, q, net_gain, split-data)`
    //       where `prop_rank` is the index INTO `SPLIT_PROPERTIES_VARIABLE`
    //       (NOT the property ID itself).
    //   (2) Reducing serially across properties in `prop_rank` order with
    //       `if best.net_gain > current_best.net_gain` (strict >) — so on
    //       ties the first property wins, matching the sequential walk.
    //   (3) Within a single property, the inner `q` loop stays sequential
    //       (same per-property reduction order as before) so the inner
    //       tiebreaker also matches.
    //
    // Hash-lock invariant: this produces the SAME `(prop_idx, splitval,
    // left_indices, right_indices, lpred, rpred)` choice as the sequential
    // code on every input.
    //
    // Profiling baseline (terminal e8 d=0.5 single-thread, pre-W44-175):
    // `estimate_subset_cost_per_predictor` consumes ~23 % of CPU and
    // `partition` consumes ~8 %. Combined ~31 % of single-thread wall.
    // Parallelizing across the 14 properties gives up to 14× speedup on
    // the work inside this function — the actual wall-time win depends on
    // tree depth and the cost-vs-overhead trade of rayon dispatch per
    // node. See `benchmarks/w44_175_*_2026-05-21.{tsv,meta}`.
    type Candidate = (usize, i32, Vec<usize>, Vec<usize>, u32, u32, f64);

    let scan_property = |prop_rank: usize| -> Option<Candidate> {
        let prop_idx = SPLIT_PROPERTIES_VARIABLE[prop_rank];
        let props = &samples.props[prop_idx];
        let mut values: Vec<i32> = indices.iter().map(|&i| props[i]).collect();
        values.sort_unstable();
        values.dedup();

        if values.len() < 2 {
            return None;
        }
        // Quantile-based split candidates (matches stage 1-6 strategy)
        let num_quantiles = 32.min(values.len() - 1);
        if num_quantiles == 0 {
            return None;
        }

        let mut prop_best_gain = 0.0f64;
        let mut prop_best: Option<Candidate> = None;

        for q in 0..num_quantiles {
            let split_idx = (values.len() * (q + 1)) / (num_quantiles + 1);
            if split_idx == 0 || split_idx >= values.len() {
                continue;
            }
            let splitval = values[split_idx - 1];

            let (left, right): (Vec<usize>, Vec<usize>) =
                indices.iter().copied().partition(|&i| props[i] <= splitval);

            if left.len() < MIN_SAMPLES_PER_LEAF || right.len() < MIN_SAMPLES_PER_LEAF {
                continue;
            }

            let left_costs =
                estimate_subset_cost_per_predictor(samples, &left, max_token, predictor_set);
            let right_costs =
                estimate_subset_cost_per_predictor(samples, &right, max_token, predictor_set);

            // Per-side best-predictor selection with libjxl penalties
            // (enc_ma.cc:376-390): change-pred penalty if differs from
            // current_predictor & current isn't Weighted; favour Zero (-1e-8);
            // disfavour Weighted (+1e-8).
            //
            // W44-172: scoped to the active predictor set so `Best` mode
            // doesn't pretend to consider predictors whose `out[]` is INFINITY.
            // For `Variable` this still walks all 14 (pred_indices covers
            // 0..=13 in libjxl order); for `Best` it only walks {6, 5}.
            let pick_best = |costs: &[f64; NUM_PREDICTORS_VARIABLE]| -> (f64, u32) {
                let mut best_cost = f64::MAX;
                let mut best_pred: u32 = pred_indices[0];
                for &pred_id in pred_indices {
                    let c = costs[pred_id as usize];
                    if !c.is_finite() {
                        continue;
                    }
                    let mut penalty = 0.0;
                    let cur_pred_is_weighted =
                        current_predictor == crate::modular::Predictor::Weighted as u32;
                    if pred_id != current_predictor && !cur_pred_is_weighted {
                        penalty += change_pred_penalty;
                    }
                    if pred_id == crate::modular::Predictor::Weighted as u32 {
                        penalty += 1e-8;
                    }
                    if pred_id == crate::modular::Predictor::Zero as u32 {
                        penalty -= 1e-8;
                    }
                    if c + penalty < best_cost {
                        best_cost = c + penalty;
                        best_pred = pred_id;
                    }
                }
                (best_cost, best_pred)
            };

            let (lcost, lpred) = pick_best(&left_costs);
            let (rcost, rpred) = pick_best(&right_costs);
            let new_cost = lcost + rcost;
            let gain = base_cost - new_cost;

            // Split overhead estimate (libjxl `enc_ma.cc:441-455`):
            //   internal node tokens: property + splitval = ~10-20 bits
            //   2× leaf delta vs 1 leaf: ~30-50 bits of leaf tokens
            //   new ANS histogram per added context: ~30-50 bits
            // Plus the libjxl gate adds `threshold` (we use 0). Net 60-100 bits.
            // Without this, Variable mode over-splits on heavily-quantized DC
            // (terminal e6 d=6) and bloats LfGlobal (~1 KB vs cjxl ~15 B).
            let overhead = 60.0;
            let net_gain = gain - overhead;

            // Inner tiebreaker matches the sequential code: keep the first
            // candidate that strictly improves the running best.
            if net_gain > prop_best_gain {
                prop_best_gain = net_gain;
                prop_best = Some((prop_idx, splitval, left, right, lpred, rpred, net_gain));
            }
        }
        prop_best
    };

    let property_results: Vec<Option<Candidate>> =
        crate::parallel::parallel_map(SPLIT_PROPERTIES_VARIABLE.len(), scan_property);

    // Serial reduction in `prop_rank` order so on ties (equal `net_gain`)
    // the FIRST property in `SPLIT_PROPERTIES_VARIABLE` wins — matching
    // the sequential walk exactly.
    let mut best_gain = 0.0f64;
    let mut best_split: Option<Candidate> = None;
    for cand in property_results.into_iter().flatten() {
        if cand.6 > best_gain {
            best_gain = cand.6;
            best_split = Some(cand);
        }
    }

    best_split
}

/// W44-180: incremental-histogram port of [`find_best_split_variable`].
///
/// libjxl reference: `enc_ma.cc:280-439` (`FindBestSplit`, the per-property
/// `for prop in 0..num_properties` loop). The libjxl pattern:
/// 1. Pre-compute parent totals once: `counts[pred * max_symbols + tok]` and
///    `tot_extra_bits[pred]` (one O(N · P) pass).
/// 2. For each property:
///    a. Bucket samples by property value: `prop_value_used_count[i]`,
///    `count_increase[i * max_symbols + sym]`,
///    `extra_bits_increase[i]` per predictor (`enc_ma.cc:140-156`,
///    `CollectExtraBitsIncrease`).
///    b. Sweep i = first_used..last_used: transfer bucket `i` from
///    `counts_above` → `counts_below`, then compute lcost / rcost
///    from running histograms via `EstimateBits` (`enc_ma.cc:359-403`).
/// 3. The candidate split count is `last_used - first_used` (one per
///    distinct property value), NOT a fixed quantile grid.
///
/// **Byte-identical preservation strategy**: this Rust port keeps the SAME
/// 32-quantile candidate set as the pre-W44-180 code (the `values[split_idx-1]`
/// formula), but evaluates each candidate's cost via running incremental
/// histograms instead of re-scanning the subset. The candidates ARE a subset
/// of libjxl's "every distinct value" sweep — we honour the existing
/// quantile-coarsening for output stability, only replacing the inner cost
/// computation with the libjxl-style incremental pattern.
///
/// Complexity:
/// - Pre-W44-180: O(N · P · Q · 2) — Q=32 quantiles × 2 sides × re-scan
/// - W44-180: O(N · P + N log N + Q · S · P) — sort O(N log N) once,
///   bucket O(N · P) once, then Q EstimateBits calls per side
///   (S = histogram size, typically ≤ ~64 for DC tokens at e8+).
///
/// On terminal e8 d=0.5 (W44-175 baseline), `estimate_subset_cost_per_predictor`
/// plus `partition` were 31 % of single-thread CPU. The incremental pattern
/// collapses both: histogram building is O(N · P) once per property (vs Q
/// times in the old code), and partitioning happens only once at the end
/// for the winning split.
///
/// Returns the SAME tuple shape as [`find_best_split_variable`].
/// Hash-lock invariant: byte-identical output to the sequential path
/// (verified via `cargo test --lib` + the 36/36 hash-lock fixtures).
fn find_best_split_variable_incremental(
    samples: &DcTreeSamples,
    indices: &[usize],
    max_token: u32,
    current_predictor: u32,
    threshold: f64,
    predictor_set: PredictorSet,
) -> Option<(usize, i32, Vec<usize>, Vec<usize>, u32, u32, f64)> {
    if indices.len() < MIN_SAMPLES_PER_LEAF * 2 {
        return None;
    }
    if !samples.has_variable_residuals() {
        return None;
    }

    let pred_indices = predictor_set.predictor_indices();
    if pred_indices.is_empty() {
        return None;
    }

    // ─── Step 1: compute parent totals (mirrors libjxl enc_ma.cc:206-227) ───
    //
    // `true_max_token` is the maximum token value across the active predictor
    // set on this subset — sets the histogram width for `estimate_bits`. We
    // also accumulate `counts_total[pred * S + tok]` and
    // `extra_bits_total[pred]` so the per-property sweep starts from the
    // parent's totals (matches `counts_above = counts.data() + pred * S`
    // initialization at `enc_ma.cc:353`).
    let mut true_max_token: u32 = 0;
    for &pred in pred_indices {
        let pred_tokens = &samples.residual_tokens_per_predictor[pred as usize];
        for &idx in indices {
            let t = pred_tokens[idx];
            if t > true_max_token {
                true_max_token = t;
            }
        }
    }
    let s = (true_max_token + 1) as usize;
    let _ = max_token; // parent-supplied; superseded by true_max_token per libjxl.

    // Per-predictor totals laid out as [pred_slot * s + tok].
    // `pred_slot` is the position in `pred_indices` (NOT the raw Predictor enum
    // index), so the histogram is dense across the active set even for
    // `PredictorSet::Best` (which only walks {Weighted, Gradient}).
    let p = pred_indices.len();
    let mut counts_total = vec![0u32; p * s];
    let mut total_per_pred = vec![0u32; p];
    let mut extra_bits_total = vec![0.0f64; p];

    for (pred_slot, &pred_id) in pred_indices.iter().enumerate() {
        let pred = pred_id as usize;
        let tokens = &samples.residual_tokens_per_predictor[pred];
        let base = pred_slot * s;
        let mut total = 0u32;
        let mut eb = 0.0f64;
        for &idx in indices {
            let tok = tokens[idx];
            counts_total[base + tok as usize] += 1;
            total += 1;
            // HybridUint {4,1,2} extra bits — must match
            // `estimate_subset_cost_per_predictor` exactly so the per-side
            // costs sum back to the parent's base_per_pred values.
            if tok >= GATHER_SPLIT {
                let n_minus_split_exp =
                    (tok - GATHER_SPLIT) / (GATHER_MSB_IN_TOKEN + GATHER_LSB_IN_TOKEN);
                eb += (n_minus_split_exp as f64) + 2.0;
            }
        }
        total_per_pred[pred_slot] = total;
        extra_bits_total[pred_slot] = eb;
    }

    // Base cost = cost under current predictor (used to compute gain).
    // Equivalent to `estimate_subset_cost_per_predictor(..)[current_predictor]`
    // — we compute it from the totals we already built instead of doing a
    // second O(N) pass. If `current_predictor` is NOT in the active set, the
    // sequential path would have returned `INFINITY` for it and the gain
    // would compare against +inf (always positive). Mirror that here.
    let base_cost = match pred_indices
        .iter()
        .position(|&p_id| p_id == current_predictor)
    {
        Some(slot) => {
            let base_offset = slot * s;
            estimate_bits(
                &counts_total[base_offset..base_offset + s],
                total_per_pred[slot],
            ) + extra_bits_total[slot]
        }
        None => f64::INFINITY,
    };

    let change_pred_penalty = 800.0 / (100.0 + threshold);
    let pick_best = |costs: &[f64]| -> (f64, u32) {
        let mut best_cost = f64::MAX;
        let mut best_pred: u32 = pred_indices[0];
        let cur_pred_is_weighted = current_predictor == crate::modular::Predictor::Weighted as u32;
        for (slot, &pred_id) in pred_indices.iter().enumerate() {
            let c = costs[slot];
            if !c.is_finite() {
                continue;
            }
            let mut penalty = 0.0;
            if pred_id != current_predictor && !cur_pred_is_weighted {
                penalty += change_pred_penalty;
            }
            if pred_id == crate::modular::Predictor::Weighted as u32 {
                penalty += 1e-8;
            }
            if pred_id == crate::modular::Predictor::Zero as u32 {
                penalty -= 1e-8;
            }
            if c + penalty < best_cost {
                best_cost = c + penalty;
                best_pred = pred_id;
            }
        }
        (best_cost, best_pred)
    };

    type Candidate = (usize, i32, Vec<usize>, Vec<usize>, u32, u32, f64);

    // ─── Step 2: per-property incremental scan ───
    //
    // Each property gets its own bucket allocation (small for Variable mode:
    // 14 properties × P predictors × S tokens × |unique values| u32s — the
    // largest term is the bucket grid, but it's sized to the property's
    // actual distinct-value count plus indices, not to N).
    //
    // The parallelism shape matches the prior `scan_property`-by-prop_rank
    // pattern so the tiebreaker stays identical (first property in
    // `SPLIT_PROPERTIES_VARIABLE` order wins on equal net_gain).
    let scan_property = |prop_rank: usize| -> Option<Candidate> {
        let prop_idx = SPLIT_PROPERTIES_VARIABLE[prop_rank];
        let props = &samples.props[prop_idx];

        // Sort INDICES by property value (libjxl sorts samples themselves
        // via `SplitTreeSamples`; we keep `indices` immutable and build a
        // parallel sort permutation so the partition step at the end can
        // produce the exact left/right vectors the sequential path produced).
        //
        // Stable sort: preserves original `indices` order on ties — matches
        // the sequential partition's stability (Vec::partition preserves
        // element order within each side).
        let mut sorted: Vec<usize> = indices.to_vec();
        sorted.sort_by_key(|&i| props[i]);

        // Build the unique-values list (matches the pre-W44-180 sort+dedup).
        // We sweep `sorted` once to extract unique values in ascending order.
        let mut values: Vec<i32> = sorted.iter().map(|&i| props[i]).collect();
        values.dedup(); // already sorted, dedup() leaves unique ascending.

        if values.len() < 2 {
            return None;
        }
        let num_quantiles = 32.min(values.len() - 1);
        if num_quantiles == 0 {
            return None;
        }

        // Generate quantile splitvals in ascending order. This is the SAME
        // candidate set as the pre-W44-180 code: `values[split_idx-1]` where
        // `split_idx = values.len() * (q+1) / (num_quantiles+1)`.
        //
        // We pre-collect them so we can sweep `sorted` once and accumulate
        // into `counts_below` as the splitval boundary advances. Dedup on
        // splitval values keeps the work tight (different `q` indices can
        // collide on the same splitval when num_quantiles ~ values.len()).
        let mut splitvals: Vec<(usize, i32)> = Vec::with_capacity(num_quantiles);
        for q in 0..num_quantiles {
            let split_idx = (values.len() * (q + 1)) / (num_quantiles + 1);
            if split_idx == 0 || split_idx >= values.len() {
                continue;
            }
            splitvals.push((q, values[split_idx - 1]));
        }
        if splitvals.is_empty() {
            return None;
        }

        // Running histograms: counts_below grows, counts_above shrinks
        // (libjxl `enc_ma.cc:353-355`, `enc_ma.cc:365-369`).
        let mut counts_below = vec![0u32; p * s];
        let mut counts_above = counts_total.clone();
        let mut total_below = vec![0u32; p];
        let mut total_above = total_per_pred.clone();
        let mut eb_below = vec![0.0f64; p];
        let mut eb_above = extra_bits_total.clone();

        // Allocate per-predictor cost scratch (used by `pick_best`).
        let mut left_costs = vec![0.0f64; p];
        let mut right_costs = vec![0.0f64; p];

        let mut prop_best_gain = 0.0f64;
        let mut prop_best: Option<Candidate> = None;

        // Sweep sorted samples in ascending prop_value, advancing through
        // splitvals in lockstep. At each splitval boundary, compute the cost.
        //
        // `cursor`: next index in `sorted` to move from above → below.
        let mut cursor = 0usize;
        let n = sorted.len();

        for &(_q, splitval) in &splitvals {
            // Advance cursor: move all samples with props[sample] <= splitval
            // from `above` to `below`. Mirrors libjxl `enc_ma.cc:359-369`
            // (move bucket `i` from counts_above to counts_below); we
            // collapse it into a single sample-stream sweep because we
            // already have the sorted order.
            while cursor < n {
                let idx = sorted[cursor];
                if props[idx] > splitval {
                    break;
                }
                // Move this sample's contribution to all predictors from
                // counts_above → counts_below.
                for (pred_slot, &pred_id) in pred_indices.iter().enumerate() {
                    let pred = pred_id as usize;
                    let tok = samples.residual_tokens_per_predictor[pred][idx] as usize;
                    let base = pred_slot * s;
                    counts_below[base + tok] += 1;
                    counts_above[base + tok] -= 1;
                    total_below[pred_slot] += 1;
                    total_above[pred_slot] -= 1;
                    // Match `estimate_subset_cost_per_predictor` extra-bits
                    // formula exactly.
                    let tok_u32 = tok as u32;
                    if tok_u32 >= GATHER_SPLIT {
                        let n_minus_split_exp =
                            (tok_u32 - GATHER_SPLIT) / (GATHER_MSB_IN_TOKEN + GATHER_LSB_IN_TOKEN);
                        let eb_delta = (n_minus_split_exp as f64) + 2.0;
                        eb_below[pred_slot] += eb_delta;
                        eb_above[pred_slot] -= eb_delta;
                    }
                }
                cursor += 1;
            }

            // Min-leaf gate: same as pre-W44-180 (left = cursor, right = n-cursor).
            let left_count = cursor;
            let right_count = n - cursor;
            if left_count < MIN_SAMPLES_PER_LEAF || right_count < MIN_SAMPLES_PER_LEAF {
                continue;
            }

            // Compute per-predictor costs from running histograms.
            // Equivalent to `estimate_subset_cost_per_predictor` on the
            // implicit left/right subsets, but at O(P · S) per call instead
            // of O(N · P).
            for pred_slot in 0..p {
                let base = pred_slot * s;
                left_costs[pred_slot] =
                    estimate_bits(&counts_below[base..base + s], total_below[pred_slot])
                        + eb_below[pred_slot];
                right_costs[pred_slot] =
                    estimate_bits(&counts_above[base..base + s], total_above[pred_slot])
                        + eb_above[pred_slot];
            }

            let (lcost, lpred) = pick_best(&left_costs);
            let (rcost, rpred) = pick_best(&right_costs);
            let new_cost = lcost + rcost;
            let gain = base_cost - new_cost;

            // Same overhead constant as pre-W44-180 (see comment in
            // `find_best_split_variable` for derivation).
            let overhead = 60.0;
            let net_gain = gain - overhead;

            // Strict-> tiebreaker matches sequential code: first candidate
            // (smaller `q`) wins on equal `net_gain`.
            if net_gain > prop_best_gain {
                prop_best_gain = net_gain;
                // Reconstruct left/right vectors by partitioning indices the
                // same way the sequential code did. We do this only ONCE per
                // property (for the winning splitval) — that's the main
                // O(N) savings vs the per-quantile partition.
                let (left, right): (Vec<usize>, Vec<usize>) =
                    indices.iter().copied().partition(|&i| props[i] <= splitval);
                prop_best = Some((prop_idx, splitval, left, right, lpred, rpred, net_gain));
            }
        }
        prop_best
    };

    let property_results: Vec<Option<Candidate>> =
        crate::parallel::parallel_map(SPLIT_PROPERTIES_VARIABLE.len(), scan_property);

    let mut best_gain = 0.0f64;
    let mut best_split: Option<Candidate> = None;
    for cand in property_results.into_iter().flatten() {
        if cand.6 > best_gain {
            best_gain = cand.6;
            best_split = Some(cand);
        }
    }

    best_split
}

/// Stage 7c: recursively build a DC tree with per-leaf best-predictor selection.
///
/// Mirrors libjxl `FindBestSplit` BFS queue (`enc_ma.cc:177-525`) — at each
/// node, evaluates Variable-mode candidate splits, records best left+right
/// predictors, and recurses. Leaves store the best predictor for their subset.
fn build_tree_recursive_variable(
    samples: &DcTreeSamples,
    indices: &[usize],
    depth: usize,
    current_predictor: u32,
    tree: &mut DcTree,
    next_context: &mut u32,
    max_token: u32,
    predictor_set: PredictorSet,
) -> usize {
    let node_idx = tree.len();
    tree.push(DcTreeNode::default());

    // Leaf if depth cap or insufficient samples.
    if depth >= MAX_TREE_DEPTH || indices.len() < MIN_SAMPLES_PER_LEAF * 2 {
        tree[node_idx].property = -1;
        tree[node_idx].context_id = *next_context;
        tree[node_idx].predictor = current_predictor;
        *next_context += 1;
        return node_idx;
    }

    // libjxl uses threshold=0 for DC (ModularOptions::splitting_heuristics_node_threshold
    // default). enc_modular.cc:1166-1217 doesn't override.
    let threshold = 0.0f64;

    // W44-180: default to incremental-histogram split scan (libjxl
    // `enc_ma.cc:280-439` pattern, O(N · P + N log N + Q · S · P)).
    // Env hook `JXL_W44_180_FORCE_LEGACY_DC_TREE_SPLIT=1` falls back to the
    // pre-W44-180 per-quantile re-scan path for byte-equivalence diagnostics.
    // The two paths MUST produce byte-identical tree shape on every input —
    // the env hook exists only so a future agent can A/B-bisect if any cell
    // shows divergence, NOT because the legacy path is preferred.
    let use_legacy = {
        #[cfg(feature = "std")]
        {
            std::env::var_os("JXL_W44_180_FORCE_LEGACY_DC_TREE_SPLIT").is_some()
        }
        #[cfg(not(feature = "std"))]
        {
            false
        }
    };
    let split = if use_legacy {
        find_best_split_variable(
            samples,
            indices,
            max_token,
            current_predictor,
            threshold,
            predictor_set,
        )
    } else {
        find_best_split_variable_incremental(
            samples,
            indices,
            max_token,
            current_predictor,
            threshold,
            predictor_set,
        )
    };
    if let Some((prop_idx, splitval, left_indices, right_indices, lpred, rpred, _gain)) = split {
        let lchild = build_tree_recursive_variable(
            samples,
            &left_indices,
            depth + 1,
            lpred,
            tree,
            next_context,
            max_token,
            predictor_set,
        );
        let rchild = build_tree_recursive_variable(
            samples,
            &right_indices,
            depth + 1,
            rpred,
            tree,
            next_context,
            max_token,
            predictor_set,
        );

        tree[node_idx].property = prop_idx as i32;
        tree[node_idx].splitval = splitval;
        tree[node_idx].lchild = lchild;
        tree[node_idx].rchild = rchild;
    } else {
        // Leaf with current best predictor.
        tree[node_idx].property = -1;
        tree[node_idx].context_id = *next_context;
        tree[node_idx].predictor = current_predictor;
        *next_context += 1;
    }
    node_idx
}

/// Stage 7c: learn a DC tree with per-leaf Variable predictor.
///
/// Mirrors libjxl `ComputeBestTree` (`enc_ma.cc:503-525`) with
/// `SetPredictor(Predictor::Variable)`. Each leaf's `predictor` field holds
/// the best of 14 simple predictors for its subset, which the bitstream emits
/// per leaf (decoder uses one predictor per leaf — `Predictor::Variable` and
/// `Predictor::Best` are encoder-only meta-modes, `enc_ma.cc:1044`).
///
/// Starting predictor is Gradient (libjxl swaps Weighted→0, Gradient→1 then
/// initialises tree root predictor from `predictors[0]` at `enc_ma.cc:546-547,
/// 513`; for DC the per-leaf reseed via penalties dominates the initial choice).
pub fn learn_dc_tree_variable(samples: &DcTreeSamples, max_token: u32) -> (DcTree, u32) {
    learn_dc_tree_variable_with_set(samples, max_token, PredictorSet::Variable)
}

/// Like [`learn_dc_tree_variable`] but restricted to the libjxl `Predictor::Best`
/// predictor set (Weighted + Gradient only). Used at effort 8 to mirror
/// `enc_modular.cc:1593-1594` which selects `Predictor::Best` when
/// `cparams_.speed_tier == kKitten`.
///
/// W44-172: drop-in replacement for `learn_dc_tree_variable` at e8. Cuts the
/// per-split predictor evaluation count from 14 → 2, removing ~48 % of e8 CPU
/// on 5+ MP screenshots (terminal e8 d=0.5 was 32 × cjxl wall ratio; this fix
/// drops it back into ≤ 3 × cjxl territory matching e9's overhead).
pub fn learn_dc_tree_best(samples: &DcTreeSamples, max_token: u32) -> (DcTree, u32) {
    learn_dc_tree_variable_with_set(samples, max_token, PredictorSet::Best)
}

/// Shared learner driven by [`PredictorSet`]. Both `learn_dc_tree_variable`
/// (all 14) and `learn_dc_tree_best` (2) route through this — keeping a single
/// implementation of the BFS search + leaf-predictor selection.
pub fn learn_dc_tree_variable_with_set(
    samples: &DcTreeSamples,
    max_token: u32,
    predictor_set: PredictorSet,
) -> (DcTree, u32) {
    if samples.num_samples == 0 || !samples.has_variable_residuals() {
        // Empty / not variable-mode: single-leaf gradient-predictor tree
        // matches the stage 1-6 fallback.
        let tree = vec![DcTreeNode {
            property: -1,
            context_id: 0,
            predictor: crate::modular::Predictor::Gradient as u32,
            ..Default::default()
        }];
        return (tree, 1);
    }

    let mut tree = DcTree::new();
    let mut next_context = 0u32;
    let indices: Vec<usize> = (0..samples.num_samples).collect();

    // libjxl starts root at predictors[0] which (after swap) is Weighted.
    // But for DC the change-predictor penalty heavily favours sticking with
    // the initial choice; we use Gradient as the initial state because most
    // DC subsets favour it (gradient is the median across CID22 + gb82-sc).
    let initial_predictor = crate::modular::Predictor::Gradient as u32;

    build_tree_recursive_variable(
        samples,
        &indices,
        0,
        initial_predictor,
        &mut tree,
        &mut next_context,
        max_token,
        predictor_set,
    );

    (tree, next_context)
}

/// Learn an optimal DC context tree from samples.
///
/// # Arguments
/// * `samples` - Collected DC samples
/// * `max_token` - Maximum token value (for histogram sizing)
///
/// # Returns
/// A learned tree and the number of contexts it uses.
pub fn learn_dc_tree(samples: &DcTreeSamples, max_token: u32) -> (DcTree, u32) {
    if samples.num_samples == 0 {
        // Empty samples: return single-leaf tree
        let tree = vec![DcTreeNode {
            property: -1,
            context_id: 0,
            ..Default::default()
        }];
        return (tree, 1);
    }

    let mut tree = DcTree::new();
    let mut next_context = 0u32;
    let indices: Vec<usize> = (0..samples.num_samples).collect();

    build_tree_recursive(
        samples,
        &indices,
        0,
        &mut tree,
        &mut next_context,
        max_token,
    );

    (tree, next_context)
}

/// Traverse the learned tree to get a context for a DC value.
#[inline]
pub fn get_dc_context(tree: &DcTree, props: &[i32; NUM_DC_PROPERTIES]) -> u32 {
    let mut idx = 0;
    loop {
        let node = &tree[idx];
        if node.property < 0 {
            return node.context_id;
        }
        let pval = props[node.property as usize];
        if pval <= node.splitval {
            idx = node.lchild;
        } else {
            idx = node.rchild;
        }
    }
}

/// Convert a learned DC tree to context tree tokens for bitstream encoding.
///
/// The token format matches the modular tree format:
/// - Internal node: (property, splitval) pairs
/// - Leaf node: (predictor, multiplier, offset) but for DC we just use context
///
/// Format: sequence of (context, value) tokens that describe the tree structure.
///
/// IMPORTANT: Tokens must be in BFS (breadth-first/level-order) order, NOT DFS.
/// The decoder computes child indices assuming BFS order.
pub fn tree_to_tokens(tree: &DcTree) -> Vec<(u32, u32)> {
    use super::common::pack_signed;
    use alloc::collections::VecDeque;

    let mut tokens = Vec::new();
    let mut queue = VecDeque::new();
    queue.push_back(0usize);

    #[cfg(feature = "debug-tokens")]
    eprintln!("tree_to_tokens: tree has {} nodes", tree.len());
    #[cfg(feature = "debug-tokens")]
    let mut leaf_count = 0;

    while let Some(idx) = queue.pop_front() {
        let node = &tree[idx];

        if node.property < 0 {
            // Leaf node: emit predictor, multiplier, offset
            #[cfg(feature = "debug-tokens")]
            {
                eprintln!(
                    "  BFS node {}: LEAF (context_id={}, predictor={}, leaf_order={})",
                    idx, node.context_id, node.predictor, leaf_count
                );
                leaf_count += 1;
            }
            // Context 1: property = 0 signals leaf node (decoder subtracts 1, gets -1)
            tokens.push((1, 0));
            // Context 2: predictor (use node's predictor field)
            tokens.push((2, node.predictor));
            // Context 3: offset (0)
            tokens.push((3, 0));
            // Context 4: multiplier log (0 for multiplier=1 since (0+1)<<0 = 1)
            tokens.push((4, 0));
            // Context 5: multiplier bits (0)
            tokens.push((5, 0));
        } else {
            // Internal node: emit property and splitval
            #[cfg(feature = "debug-tokens")]
            eprintln!(
                "  BFS node {}: INTERNAL (prop={}, split={}, left={}, right={})",
                idx, node.property, node.splitval, node.lchild, node.rchild
            );
            // Context 1: property+1 (decoder subtracts 1 to get actual property index)
            let prop_token = (node.property + 1) as u32;
            tokens.push((1, prop_token));
            // Context 0: splitval (packed signed)
            tokens.push((0, pack_signed(node.splitval)));

            // Queue children for BFS traversal (left first, then right)
            queue.push_back(node.lchild);
            queue.push_back(node.rchild);
        }
    }

    #[cfg(feature = "debug-tokens")]
    eprintln!("  Total: {} tokens, {} leaves", tokens.len(), leaf_count);
    tokens
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_dc_properties() {
        // Test gradient property (property 9 = left + top - topleft)
        let (props, _) = compute_dc_properties(
            0,   // channel
            5,   // x
            3,   // y
            100, // top
            100, // left
            100, // topleft
            100, // topright
            100, // toptop
            100, // leftleft
            0,   // prev_local_grad
        );
        // Gradient: 100 + 100 - 100 = 100
        assert_eq!(props[9], 100);

        // Test position properties
        assert_eq!(props[2], 3); // y
        assert_eq!(props[3], 5); // x

        // Test absolute values
        assert_eq!(props[4], 100); // |top|
        assert_eq!(props[5], 100); // |left|

        // Test raw values
        assert_eq!(props[6], 100); // top
        assert_eq!(props[7], 100); // left

        // Test FFV1 properties
        let (props2, _) = compute_dc_properties(0, 0, 0, 200, 150, 100, 180, 200, 120, 0);
        // Gradient: 150 + 200 - 100 = 250
        assert_eq!(props2[9], 250);
        // FFV1 left - topleft: 150 - 100 = 50
        assert_eq!(props2[10], 50);
        // FFV1 topleft - top: 100 - 200 = -100
        assert_eq!(props2[11], -100);
    }

    #[test]
    fn test_gather_dc_samples_empty() {
        let quant_dc: [Vec<Vec<i16>>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        let mut samples = DcTreeSamples::new();
        gather_dc_samples(&mut samples, &quant_dc);
        assert_eq!(samples.num_samples, 0);
    }

    #[test]
    fn test_gather_dc_samples_variable_multi_predictor() {
        // 8x8 channel with stride values to exercise all predictors.
        // Channel: each row constant (so Top predictor wins easily on row 1+).
        let mk_channel = |base: i16| -> Vec<Vec<i16>> {
            let mut ch = Vec::with_capacity(8);
            for y in 0..8 {
                ch.push(vec![base + y as i16 * 4; 8]);
            }
            ch
        };
        let quant_dc: [Vec<Vec<i16>>; 3] = [mk_channel(50), mk_channel(100), mk_channel(30)];

        let mut samples = DcTreeSamples::new();
        gather_dc_samples_variable(&mut samples, &quant_dc);

        // 8*8 * 3 channels = 192 samples
        assert_eq!(samples.num_samples, 192);
        // Variable mode populates per-predictor residual arrays
        assert!(samples.has_variable_residuals());
        // All 14 predictor slots present
        assert_eq!(samples.residual_tokens_per_predictor.len(), 14);
        for slot in &samples.residual_tokens_per_predictor {
            assert_eq!(slot.len(), 192);
        }
        // Gradient residuals also mirrored into legacy slot
        assert_eq!(samples.residual_tokens.len(), 192);
        // Property 15 (wp_max_error) should have at least one non-zero entry
        // (the WP error term grows non-trivially through the iteration).
        let has_nonzero_wp = samples.props[15].iter().any(|&v| v != 0);
        assert!(has_nonzero_wp, "wp_max_error property should populate");
    }

    #[test]
    fn test_gather_dc_samples_simple() {
        // 4x4 constant DC values
        let channel = vec![vec![100i16; 4]; 4];
        let quant_dc: [Vec<Vec<i16>>; 3] = [channel.clone(), channel.clone(), channel];

        let mut samples = DcTreeSamples::new();
        gather_dc_samples(&mut samples, &quant_dc);

        // 4x4 x 3 channels = 48 samples
        assert_eq!(samples.num_samples, 48);
    }

    #[test]
    fn test_learn_dc_tree_empty() {
        let samples = DcTreeSamples::new();
        let (tree, num_contexts) = learn_dc_tree(&samples, 64);

        assert_eq!(tree.len(), 1);
        assert_eq!(tree[0].property, -1);
        assert_eq!(num_contexts, 1);
    }

    #[test]
    fn test_learn_dc_tree_constant() {
        // Constant DC values should produce single-leaf tree
        let channel = vec![vec![50i16; 8]; 8];
        let quant_dc: [Vec<Vec<i16>>; 3] = [channel.clone(), channel.clone(), channel];

        let mut samples = DcTreeSamples::new();
        gather_dc_samples(&mut samples, &quant_dc);

        let (tree, num_contexts) = learn_dc_tree(&samples, 64);

        // Should have at least 1 context
        assert!(num_contexts >= 1);
        // Root should exist
        assert!(!tree.is_empty());
    }

    #[test]
    fn test_get_dc_context() {
        // Create a simple 2-leaf tree that splits on gradient property (property 9)
        let tree = vec![
            DcTreeNode {
                property: 9, // Gradient (left + top - topleft)
                splitval: 150,
                lchild: 1,
                rchild: 2,
                context_id: 0,
                predictor: 0, // Not used for internal nodes
            },
            DcTreeNode {
                property: -1,
                context_id: 0,
                ..Default::default()
            },
            DcTreeNode {
                property: -1,
                context_id: 1,
                ..Default::default()
            },
        ];

        // Gradient <= 150 should go to context 0
        // top=100, left=100, topleft=100 => gradient = 100 + 100 - 100 = 100
        let (props_low, _) = compute_dc_properties(0, 0, 0, 100, 100, 100, 100, 100, 100, 0);
        assert_eq!(props_low[9], 100);
        assert_eq!(get_dc_context(&tree, &props_low), 0);

        // Gradient > 150 should go to context 1
        // top=200, left=100, topleft=50 => gradient = 100 + 200 - 50 = 250
        let (props_high, _) = compute_dc_properties(0, 0, 0, 200, 100, 50, 200, 200, 100, 0);
        assert_eq!(props_high[9], 250);
        assert_eq!(get_dc_context(&tree, &props_high), 1);
    }

    #[test]
    fn test_tree_to_tokens() {
        // Single leaf tree
        let tree = vec![DcTreeNode {
            property: -1,
            context_id: 0,
            ..Default::default()
        }];

        let tokens = tree_to_tokens(&tree);
        // Leaf emits 5 tokens: property marker, predictor, offset, multiplier, unused
        assert_eq!(tokens.len(), 5);
        assert_eq!(tokens[0], (1, 0)); // property = -1 (leaf marker)
    }

    /// W44-172: verify `PredictorSet::Best` returns exactly Weighted + Gradient
    /// in libjxl order (`enc_ma.cc:549`: `{Predictor::Weighted, Predictor::Gradient}`).
    #[test]
    fn test_predictor_set_best_indices() {
        let indices = PredictorSet::Best.predictor_indices();
        assert_eq!(indices, &[6, 5]);
        assert_eq!(crate::modular::Predictor::Weighted as u32, 6);
        assert_eq!(crate::modular::Predictor::Gradient as u32, 5);
    }

    /// W44-172: verify `PredictorSet::Variable` covers all 14 predictors with
    /// libjxl's swap (Weighted → slot 0, Gradient → slot 1, then the rest).
    /// Mirrors `enc_ma.cc:543-547`.
    #[test]
    fn test_predictor_set_variable_indices() {
        let indices = PredictorSet::Variable.predictor_indices();
        assert_eq!(indices.len(), 14);
        // libjxl swaps: slot 0 = Weighted, slot 1 = Gradient
        assert_eq!(indices[0], 6); // Weighted
        assert_eq!(indices[1], 5); // Gradient
        // Remaining slots are the leftover predictors (any permutation OK so
        // long as all 0..=13 appear exactly once).
        let mut sorted: alloc::vec::Vec<u32> = indices.iter().copied().collect();
        sorted.sort_unstable();
        let expected: alloc::vec::Vec<u32> = (0..14).collect();
        assert_eq!(sorted, expected);
    }

    /// W44-172: `learn_dc_tree_best` produces a valid tree on real-looking
    /// DC samples. Smoke test: must return at least one leaf, no panic.
    #[test]
    fn test_learn_dc_tree_best_smoke() {
        // 32×32 channel mixing two flat regions — enough samples to admit splits
        // and exercise the Best (2-predictor) path.
        let mut channel = alloc::vec![alloc::vec![0i16; 32]; 32];
        for (y, row) in channel.iter_mut().enumerate() {
            for (x, v) in row.iter_mut().enumerate() {
                *v = if x < 16 {
                    50 + (y as i16)
                } else {
                    200 - (y as i16)
                };
            }
        }
        let quant_dc: [alloc::vec::Vec<alloc::vec::Vec<i16>>; 3] =
            [channel.clone(), channel.clone(), channel];

        let mut samples = DcTreeSamples::new();
        gather_dc_samples_variable(&mut samples, &quant_dc);

        let (tree_best, n_ctx_best) = learn_dc_tree_best(&samples, 64);
        let (tree_var, n_ctx_var) = learn_dc_tree_variable(&samples, 64);

        // Both should produce at least the root.
        assert!(!tree_best.is_empty());
        assert!(!tree_var.is_empty());
        assert!(n_ctx_best >= 1);
        assert!(n_ctx_var >= 1);

        // Every leaf in the Best tree must use a predictor from the Best set
        // (Weighted=6 or Gradient=5). Variable tree leaves may use any predictor.
        for node in &tree_best {
            if node.property == -1 {
                let p = node.predictor;
                assert!(
                    p == 5 || p == 6,
                    "Best-mode leaf predictor {p} outside {{Gradient=5, Weighted=6}}"
                );
            }
        }
    }

    /// W44-172: shared `_with_set` core handles empty samples gracefully for
    /// both predictor sets (gradient-only single-leaf fallback).
    #[test]
    fn test_learn_dc_tree_variable_with_set_empty_samples() {
        let samples = DcTreeSamples::new();
        let (tree_best, n_best) = learn_dc_tree_variable_with_set(&samples, 64, PredictorSet::Best);
        let (tree_var, n_var) =
            learn_dc_tree_variable_with_set(&samples, 64, PredictorSet::Variable);
        for (tree, n) in [(tree_best, n_best), (tree_var, n_var)] {
            assert_eq!(tree.len(), 1);
            assert_eq!(tree[0].property, -1);
            assert_eq!(n, 1);
            // Empty-samples fallback uses Gradient as the default leaf predictor.
            assert_eq!(
                tree[0].predictor,
                crate::modular::Predictor::Gradient as u32
            );
        }
    }

    /// W44-175: `find_best_split_variable` default (parallel) path returns
    /// a self-consistent tree shape on a small non-trivial channel. The
    /// parallel-vs-sequential equivalence proof lives in the bench harness
    /// (`examples/w44_175_parallel_buttloop_ab.rs`) which sets the env
    /// hook `JXL_W44_175_FORCE_SEQUENTIAL_DC_TREE_SPLIT=1` from `examples/`
    /// scope (where `unsafe` for `env::set_var` is allowed; the lib crate
    /// forbids `unsafe`). The bench measures byte-identical output across
    /// 10 cells × 2 modes. Hash-lock regression tests (36/36 BYTE-IDENTICAL)
    /// cover the integration-level invariant.
    ///
    /// This unit test is a smoke check that the parallel path produces a
    /// reasonable tree shape (non-empty, predictor field set per leaf,
    /// at least one context) — it does NOT itself toggle the env hook.
    #[test]
    fn test_find_best_split_variable_parallel_smoke() {
        // 32×32 channel with two sharp regions + a gradient slope to give
        // the splitter actual work (multiple beneficial splits available).
        let mut channel = alloc::vec![alloc::vec![0i16; 32]; 32];
        for (y, row) in channel.iter_mut().enumerate() {
            for (x, v) in row.iter_mut().enumerate() {
                *v = if x < 16 {
                    (50 + (y as i16)) * (if x < 8 { 1 } else { 2 })
                } else {
                    (200 - (y as i16)) - (x as i16)
                };
            }
        }
        let quant_dc: [alloc::vec::Vec<alloc::vec::Vec<i16>>; 3] =
            [channel.clone(), channel.clone(), channel];

        let mut samples = DcTreeSamples::new();
        gather_dc_samples_variable(&mut samples, &quant_dc);

        let (tree_var, nctx_var) = learn_dc_tree_variable(&samples, 64);
        assert!(
            !tree_var.is_empty(),
            "Variable mode: tree must not be empty"
        );
        assert!(nctx_var >= 1, "Variable mode: context count must be >= 1");
        for node in &tree_var {
            if node.property == -1 {
                let p = node.predictor;
                assert!(
                    (0..=13).contains(&p),
                    "Variable-mode leaf predictor {p} outside the [0, 13] candidate range"
                );
            }
        }

        let (tree_best, nctx_best) = learn_dc_tree_best(&samples, 64);
        assert!(!tree_best.is_empty(), "Best mode: tree must not be empty");
        assert!(nctx_best >= 1, "Best mode: context count must be >= 1");
    }

    /// W44-180: `find_best_split_variable_incremental` must produce the SAME
    /// (prop_idx, splitval, left_indices, right_indices, lpred, rpred) tuple
    /// as the legacy `find_best_split_variable` on every input.
    ///
    /// The legacy path's tie-breaking and quantile-grid choices are part of
    /// the contract — the incremental port only changes the per-candidate
    /// cost-evaluation algorithm, not the candidate set or the reduction.
    ///
    /// This test exercises the equivalence on a small synthetic channel
    /// (sufficient samples to get past `MIN_SAMPLES_PER_LEAF * 2 = 128` so
    /// the split logic actually fires). The hash-lock fixtures and the
    /// `examples/w44_180_inc_hist_ab.rs` bench cover the integration-level
    /// invariant on real corpus images.
    #[test]
    fn test_find_best_split_variable_legacy_vs_incremental_byte_equivalent() {
        // 32×32 channel with structured content so the splitter has actual
        // beneficial splits to find (NUM_DC_PROPERTIES > 0 distinct property
        // values per axis).
        let mut channel = alloc::vec![alloc::vec![0i16; 32]; 32];
        for (y, row) in channel.iter_mut().enumerate() {
            for (x, v) in row.iter_mut().enumerate() {
                *v = if x < 16 {
                    (50 + (y as i16)) * (if x < 8 { 1 } else { 2 })
                } else {
                    (200 - (y as i16)) - (x as i16)
                };
            }
        }
        let quant_dc: [alloc::vec::Vec<alloc::vec::Vec<i16>>; 3] =
            [channel.clone(), channel.clone(), channel];

        let mut samples = DcTreeSamples::new();
        gather_dc_samples_variable(&mut samples, &quant_dc);

        // Get max_token consistently for both paths.
        let max_token = samples
            .residual_tokens_per_predictor
            .iter()
            .flat_map(|v| v.iter())
            .copied()
            .max()
            .unwrap_or(0);

        let indices: alloc::vec::Vec<usize> = (0..samples.num_samples).collect();

        // Skip if too few samples for either path to attempt a split — both
        // would return None, which is trivially equivalent.
        if indices.len() < MIN_SAMPLES_PER_LEAF * 2 {
            return;
        }

        // Test both predictor sets (Best at e8, Variable at e9+).
        for predictor_set in [PredictorSet::Best, PredictorSet::Variable] {
            // current_predictor follows the same seed as `build_tree_recursive_variable`:
            // first entry in `pred_indices` for the predictor set.
            let current_predictor = predictor_set.predictor_indices()[0];

            let legacy = find_best_split_variable(
                &samples,
                &indices,
                max_token,
                current_predictor,
                0.0,
                predictor_set,
            );
            let inc = find_best_split_variable_incremental(
                &samples,
                &indices,
                max_token,
                current_predictor,
                0.0,
                predictor_set,
            );

            match (&legacy, &inc) {
                (None, None) => continue,
                (Some(_), None) | (None, Some(_)) => {
                    panic!(
                        "W44-180 divergence (predictor_set={:?}): one path returned a split, \
                         the other returned None. legacy={:?} inc={:?}",
                        predictor_set,
                        legacy.as_ref().map(|c| (c.0, c.1, c.4, c.5)),
                        inc.as_ref().map(|c| (c.0, c.1, c.4, c.5)),
                    );
                }
                (Some(l), Some(r)) => {
                    assert_eq!(
                        l.0, r.0,
                        "W44-180 prop_idx divergence (predictor_set={:?}): legacy={} inc={}",
                        predictor_set, l.0, r.0
                    );
                    assert_eq!(
                        l.1, r.1,
                        "W44-180 splitval divergence (predictor_set={:?}, prop={}): \
                         legacy={} inc={}",
                        predictor_set, l.0, l.1, r.1
                    );
                    assert_eq!(
                        l.2, r.2,
                        "W44-180 left_indices divergence (predictor_set={:?})",
                        predictor_set
                    );
                    assert_eq!(
                        l.3, r.3,
                        "W44-180 right_indices divergence (predictor_set={:?})",
                        predictor_set
                    );
                    assert_eq!(
                        l.4, r.4,
                        "W44-180 lpred divergence (predictor_set={:?})",
                        predictor_set
                    );
                    assert_eq!(
                        l.5, r.5,
                        "W44-180 rpred divergence (predictor_set={:?})",
                        predictor_set
                    );
                    // gain is f64 — allow microscopic float jitter from the
                    // re-ordering of additions (the running update accumulates
                    // bucket additions in sorted-by-property order; the legacy
                    // re-scan accumulates in original-index order). Both
                    // produce the SAME unique-symbol-count histograms, so
                    // `estimate_bits` returns the same value, but the per-
                    // candidate extra_bits accumulation order differs.
                    // 1e-9 relative tolerance is well below the gain
                    // discrimination threshold (split decisions hinge on
                    // gain differences of bits, ~order 1-100).
                    let abs_diff = (l.6 - r.6).abs();
                    let rel_diff = abs_diff / l.6.abs().max(r.6.abs()).max(1e-9);
                    assert!(
                        abs_diff < 1e-6 || rel_diff < 1e-9,
                        "W44-180 net_gain divergence (predictor_set={:?}): \
                         legacy={:.9} inc={:.9} (abs_diff={:.3e}, rel_diff={:.3e})",
                        predictor_set,
                        l.6,
                        r.6,
                        abs_diff,
                        rel_diff
                    );
                }
            }
        }
    }
}

/// Number of AC metadata contexts (EPF=1, CfL=2, QF=4, ACS=4).
pub const NUM_AC_META_CONTEXTS: u32 = 11;

/// Create tree tokens for a merged MA tree with AC metadata routing and learned DC subtree.
///
/// Builds a tree where:
/// - Root splits on stream_id (property 1, splitval=2): LEFT → AC metadata, RIGHT → DC
/// - AC metadata subtree routes based on channel/y/left properties to 11 contexts
/// - DC subtree uses the learned tree for context assignment
/// - A padding chain pushes DC leaves deep enough in BFS that they appear after
///   all AC metadata leaves (dummy chain leaves get "wasted" context IDs)
///
/// Returns `(tokens, total_contexts, dc_ctx_remap, ac_meta_ctx_map)` where:
/// - `tokens`: BFS-ordered tree token stream for bitstream encoding
/// - `total_contexts`: total number of contexts (AC meta + dummy + DC)
/// - `dc_ctx_remap`: maps original DC context ID → BFS context ID
///   (needed because BFS leaf order may differ from DFS context assignment)
/// - `ac_meta_ctx_map`: maps original AC metadata context [0-10] → BFS context ID
pub fn tree_tokens_with_ac_metadata_prefix(
    dc_tree: &DcTree,
    learned_num_contexts: u32,
    num_dc_groups: usize,
) -> (
    Vec<(u32, u32)>,
    u32,
    Vec<u32>,
    [u32; NUM_AC_META_CONTEXTS as usize],
) {
    use super::common::pack_signed;
    use alloc::collections::VecDeque;

    // ─── Node types for building the merged tree ───

    enum LeafType {
        AcMeta(u32), // original AC metadata context 0-10
        Dummy,       // padding chain leaf (no tokens, wasted context)
        Dc(u32),     // original DC context from learned tree
    }

    struct FlatNode {
        property: i32,
        splitval: i32,
        predictor: u32,
        left: usize,
        right: usize,
        leaf_type: LeafType,
    }

    let mut flat: Vec<FlatNode> = Vec::new();

    let mk_internal =
        |flat: &mut Vec<FlatNode>, prop: i32, split: i32, l: usize, r: usize| -> usize {
            let idx = flat.len();
            flat.push(FlatNode {
                property: prop,
                splitval: split,
                predictor: 0,
                left: l,
                right: r,
                leaf_type: LeafType::Dummy,
            });
            idx
        };

    let mk_leaf = |flat: &mut Vec<FlatNode>, pred: u32, lt: LeafType| -> usize {
        let idx = flat.len();
        flat.push(FlatNode {
            property: -1,
            splitval: 0,
            predictor: pred,
            left: 0,
            right: 0,
            leaf_type: lt,
        });
        idx
    };

    // ─── Build AC metadata subtree (bottom-up for correct index references) ───
    //
    // Channel ordering (from jxl-oxide hf_metadata.rs):
    //   ch0 = x_from_y (YtoX CfL), ch1 = b_from_y (YtoB CfL),
    //   ch2 = block_info (ACS at y=0, QF at y=1), ch3 = sharpness (EPF)
    //
    // Context assignment (from dc_coding.rs):
    //   EPF=0(Zero), YtoB=1(Gradient), YtoX=2(Gradient),
    //   QF=3-6(Left), ACS=7-10(Zero)

    // QF leaves: predictor=1 (Left), contexts 3-6
    let qf3 = mk_leaf(&mut flat, 1, LeafType::AcMeta(3));
    let qf4 = mk_leaf(&mut flat, 1, LeafType::AcMeta(4));
    let qf5 = mk_leaf(&mut flat, 1, LeafType::AcMeta(5));
    let qf6 = mk_leaf(&mut flat, 1, LeafType::AcMeta(6));
    // ACS leaves: predictor=0 (Zero), contexts 7-10
    let acs7 = mk_leaf(&mut flat, 0, LeafType::AcMeta(7));
    let acs8 = mk_leaf(&mut flat, 0, LeafType::AcMeta(8));
    let acs9 = mk_leaf(&mut flat, 0, LeafType::AcMeta(9));
    let acs10 = mk_leaf(&mut flat, 0, LeafType::AcMeta(10));
    // QF splits on property 7 (left neighbor): >11, >5, >3, <=3
    let qf_l = mk_internal(&mut flat, 7, 11, qf3, qf4);
    let qf_r = mk_internal(&mut flat, 7, 3, qf5, qf6);
    let qf_root = mk_internal(&mut flat, 7, 5, qf_l, qf_r);
    // ACS splits on property 7 (left neighbor): same thresholds
    let acs_l = mk_internal(&mut flat, 7, 11, acs7, acs8);
    let acs_r = mk_internal(&mut flat, 7, 3, acs9, acs10);
    let acs_root = mk_internal(&mut flat, 7, 5, acs_l, acs_r);
    // Block info: property 2 (y), splitval=0 → LEFT=QF(y>0), RIGHT=ACS(y=0)
    let blockinfo = mk_internal(&mut flat, 2, 0, qf_root, acs_root);
    // Channel leaves
    let epf = mk_leaf(&mut flat, 0, LeafType::AcMeta(0)); // ch3, Zero pred
    let ytob = mk_leaf(&mut flat, 5, LeafType::AcMeta(1)); // ch1, Gradient pred
    let ytox = mk_leaf(&mut flat, 5, LeafType::AcMeta(2)); // ch0, Gradient pred
    // Channel routing: prop 0 (channel)
    let ch2 = mk_internal(&mut flat, 0, 2, epf, blockinfo); // ch>2→EPF, ch<=2→blockinfo
    let ch0 = mk_internal(&mut flat, 0, 0, ytob, ytox); // ch>0→YtoB, ch<=0→YtoX
    let ac_root = mk_internal(&mut flat, 0, 1, ch2, ch0); // ch>1→ch2, ch<=1→ch0

    // ─── Build DC subtree ───
    //
    // IMPORTANT: The JXL spec convention is LEFT = property > splitval,
    // RIGHT = property <= splitval. But our DC tree builder uses the opposite:
    // lchild = property <= splitval, rchild = property > splitval.
    // We SWAP the children here so the decoder evaluates correctly.

    let dc_start = flat.len();
    for node in dc_tree {
        if node.property < 0 {
            mk_leaf(&mut flat, node.predictor, LeafType::Dc(node.context_id));
        } else {
            mk_internal(
                &mut flat,
                node.property,
                node.splitval,
                dc_start + node.rchild, // JXL LEFT = property > splitval = our rchild
                dc_start + node.lchild, // JXL RIGHT = property <= splitval = our lchild
            );
        }
    }
    let dc_root_idx = dc_start;

    // ─── Build merged root ───
    //
    // No padding chain needed: we use a full context remap (dc_ctx_remap) that
    // correctly maps each DC tree context to its BFS position, regardless of
    // where DC leaves appear relative to AC metadata leaves in BFS order.
    //
    // Previous versions used a padding chain (property 1 splits) to push DC
    // leaves deeper in BFS, but decoders validate that splitval is within the
    // property's narrowing range, making repeated same-property splits fail.
    //
    // Property 1 (stream_id), splitval=num_dc_groups:
    //   LEFT (stream_id > num_dc_groups): AC metadata
    //   RIGHT (stream_id <= num_dc_groups): DC subtree
    //
    // DC groups have stream_ids 1..num_dc_groups (from ModularStreamId::VarDCTDC).
    // AC metadata groups have stream_ids 1+2*num_dc_groups.. (from ModularStreamId::ACMetadata).
    // So splitval=num_dc_groups correctly routes all DC groups to the DC subtree
    // and all AC metadata groups to the AC metadata subtree.
    let root = mk_internal(&mut flat, 1, num_dc_groups as i32, ac_root, dc_root_idx);

    // ─── BFS to generate token stream and track context ID mapping ───
    //
    // The decoder reads tokens in BFS order, assigning sequential context IDs
    // to leaves. Dummy leaves from the padding chain get context IDs between
    // AC metadata groups (they interleave at each BFS depth level).
    // We track the actual BFS context for each AC metadata and DC leaf.

    let mut tokens = Vec::new();
    let mut queue = VecDeque::new();
    let mut leaf_ctx = 0u32;
    let mut ac_meta_ctx_map = [0u32; NUM_AC_META_CONTEXTS as usize];
    let mut dc_ctx_map = Vec::new();

    // Emit root token
    let rn = &flat[root];
    tokens.push((1, (rn.property + 1) as u32));
    tokens.push((0, pack_signed(rn.splitval)));
    queue.push_back(root);

    while let Some(idx) = queue.pop_front() {
        for child_idx in [flat[idx].left, flat[idx].right] {
            let cn = &flat[child_idx];
            if cn.property < 0 {
                // Leaf: emit 5 tokens (property marker, predictor, offset, multiplier, unused)
                tokens.push((1, 0)); // property = -1 → encoded as 0
                tokens.push((2, cn.predictor));
                tokens.push((3, 0)); // offset
                tokens.push((4, 0)); // multiplier
                tokens.push((5, 0)); // unused
                match cn.leaf_type {
                    LeafType::AcMeta(orig) => {
                        ac_meta_ctx_map[orig as usize] = leaf_ctx;
                    }
                    LeafType::Dc(orig) => {
                        dc_ctx_map.push((orig, leaf_ctx));
                    }
                    LeafType::Dummy => {}
                }
                leaf_ctx += 1;
            } else {
                // Internal: emit 2 tokens (property, splitval)
                tokens.push((1, (cn.property + 1) as u32));
                tokens.push((0, pack_signed(cn.splitval)));
                queue.push_back(child_idx);
            }
        }
    }

    // Build DC context remap: dc_ctx_remap[orig_dc_ctx] = BFS context ID.
    // BFS and DFS can produce different leaf orderings for unbalanced trees,
    // plus the child swap changes BFS order, so we need a full remap.
    let mut dc_ctx_remap = vec![0u32; learned_num_contexts as usize];
    for &(orig, bfs) in &dc_ctx_map {
        dc_ctx_remap[orig as usize] = bfs;
    }
    let total_contexts = leaf_ctx;

    (tokens, total_contexts, dc_ctx_remap, ac_meta_ctx_map)
}

/// Build a context tree with AC metadata contexts only (no DC).
///
/// Used when `use_lf_frame` is true: DC is encoded in a separate frame,
/// so the main VarDCT frame's LfGlobal tree only needs AC metadata contexts.
///
/// Returns (tree_tokens, total_contexts, ac_meta_ctx_map).
pub fn ac_metadata_only_tree() -> (Vec<(u32, u32)>, u32, [u32; NUM_AC_META_CONTEXTS as usize]) {
    use super::common::pack_signed;
    use alloc::collections::VecDeque;

    enum LeafType {
        AcMeta(u32),
    }

    struct FlatNode {
        property: i32,
        splitval: i32,
        predictor: u32,
        left: usize,
        right: usize,
        leaf_type: Option<LeafType>,
    }

    let mut flat: Vec<FlatNode> = Vec::new();

    let mk_internal =
        |flat: &mut Vec<FlatNode>, prop: i32, split: i32, l: usize, r: usize| -> usize {
            let idx = flat.len();
            flat.push(FlatNode {
                property: prop,
                splitval: split,
                predictor: 0,
                left: l,
                right: r,
                leaf_type: None,
            });
            idx
        };

    let mk_leaf = |flat: &mut Vec<FlatNode>, pred: u32, lt: LeafType| -> usize {
        let idx = flat.len();
        flat.push(FlatNode {
            property: -1,
            splitval: 0,
            predictor: pred,
            left: 0,
            right: 0,
            leaf_type: Some(lt),
        });
        idx
    };

    // Build AC metadata subtree (same structure as in tree_tokens_with_ac_metadata_prefix)
    let qf3 = mk_leaf(&mut flat, 1, LeafType::AcMeta(3));
    let qf4 = mk_leaf(&mut flat, 1, LeafType::AcMeta(4));
    let qf5 = mk_leaf(&mut flat, 1, LeafType::AcMeta(5));
    let qf6 = mk_leaf(&mut flat, 1, LeafType::AcMeta(6));
    let acs7 = mk_leaf(&mut flat, 0, LeafType::AcMeta(7));
    let acs8 = mk_leaf(&mut flat, 0, LeafType::AcMeta(8));
    let acs9 = mk_leaf(&mut flat, 0, LeafType::AcMeta(9));
    let acs10 = mk_leaf(&mut flat, 0, LeafType::AcMeta(10));
    let qf_l = mk_internal(&mut flat, 7, 11, qf3, qf4);
    let qf_r = mk_internal(&mut flat, 7, 3, qf5, qf6);
    let qf_root = mk_internal(&mut flat, 7, 5, qf_l, qf_r);
    let acs_l = mk_internal(&mut flat, 7, 11, acs7, acs8);
    let acs_r = mk_internal(&mut flat, 7, 3, acs9, acs10);
    let acs_root = mk_internal(&mut flat, 7, 5, acs_l, acs_r);
    let blockinfo = mk_internal(&mut flat, 2, 0, qf_root, acs_root);
    let epf = mk_leaf(&mut flat, 0, LeafType::AcMeta(0));
    let ytob = mk_leaf(&mut flat, 5, LeafType::AcMeta(1));
    let ytox = mk_leaf(&mut flat, 5, LeafType::AcMeta(2));
    let ch2 = mk_internal(&mut flat, 0, 2, epf, blockinfo);
    let ch0 = mk_internal(&mut flat, 0, 0, ytob, ytox);
    let root = mk_internal(&mut flat, 0, 1, ch2, ch0);

    // BFS to generate token stream
    let mut tokens = Vec::new();
    let mut queue = VecDeque::new();
    let mut leaf_ctx = 0u32;
    let mut ac_meta_ctx_map = [0u32; NUM_AC_META_CONTEXTS as usize];

    let rn = &flat[root];
    tokens.push((1, (rn.property + 1) as u32));
    tokens.push((0, pack_signed(rn.splitval)));
    queue.push_back(root);

    while let Some(idx) = queue.pop_front() {
        for child_idx in [flat[idx].left, flat[idx].right] {
            let cn = &flat[child_idx];
            if cn.property < 0 {
                tokens.push((1, 0));
                tokens.push((2, cn.predictor));
                tokens.push((3, 0));
                tokens.push((4, 0));
                tokens.push((5, 0));
                if let Some(LeafType::AcMeta(orig)) = &cn.leaf_type {
                    ac_meta_ctx_map[*orig as usize] = leaf_ctx;
                }
                leaf_ctx += 1;
            } else {
                tokens.push((1, (cn.property + 1) as u32));
                tokens.push((0, pack_signed(cn.splitval)));
                queue.push_back(child_idx);
            }
        }
    }

    let total_contexts = leaf_ctx;
    (tokens, total_contexts, ac_meta_ctx_map)
}

/// Collect DC tokens using a learned Variable-mode tree.
///
/// Each leaf's `predictor` field selects one of 14 simple predictors for the
/// residual. WP state is **always** run in parallel (regardless of which
/// predictor each leaf chose) because:
///   1. It produces `wp_max_error` (property 15) which the tree traversal may
///      use for routing.
///   2. The decoder maintains identical WP state for any leaf that emitted
///      `Predictor::Weighted` (id 6), so the encoder must update state on
///      every pixel even when the current leaf chose a different predictor —
///      matches libjxl `EncodeModularChannelMAANS` (`enc_encoding.cc`)
///      which keeps `weighted::State::UpdateErrors` running unconditionally.
///
/// Mirrors libjxl `EncodeModularChannelMAANS` per-pixel loop. Each channel
/// gets a FRESH `WeightedPredictorState` (matches libjxl per-channel pass).
pub fn collect_dc_tokens_with_tree_variable(
    quant_dc: &[Vec<Vec<i16>>; 3],
    tree: &DcTree,
    start_bx: usize,
    start_by: usize,
    end_bx: usize,
    end_by: usize,
) -> Vec<crate::entropy_coding::token::Token> {
    use crate::entropy_coding::token::Token;
    use crate::modular::predictor::{Neighbors, Predictor, WeightedPredictorState};

    let region_width = end_bx - start_bx;
    let region_height = end_by - start_by;

    if region_width == 0 || region_height == 0 {
        return Vec::new();
    }

    let mut tokens = Vec::with_capacity(region_width * region_height * 3);

    // Encode in channel order: Y (1), X (0), B (2). Fresh WP state per channel.
    for (enc_idx, &c) in [1usize, 0, 2].iter().enumerate() {
        let channel = &quant_dc[c];
        let mut wp_state = WeightedPredictorState::with_defaults(region_width);

        for y in start_by..end_by {
            let mut prev_local_grad = 0i32;

            for x in start_bx..end_bx {
                let dc_val = channel[y][x] as i32;

                let left = if x > start_bx {
                    channel[y][x - 1] as i32
                } else if y > start_by {
                    channel[y - 1][x] as i32
                } else {
                    0
                };
                let top = if y > start_by {
                    channel[y - 1][x] as i32
                } else {
                    left
                };
                let topleft = if x > start_bx && y > start_by {
                    channel[y - 1][x - 1] as i32
                } else {
                    left
                };
                let topright = if y > start_by && x + 1 < end_bx {
                    channel[y - 1][x + 1] as i32
                } else {
                    top
                };
                let toptop = if y > start_by + 1 {
                    channel[y - 2][x] as i32
                } else {
                    top
                };
                let leftleft = if x > start_bx + 1 {
                    channel[y][x - 2] as i32
                } else {
                    left
                };
                let nee = if y > start_by && x + 2 < end_bx {
                    channel[y - 1][x + 2] as i32
                } else {
                    topright
                };

                let neighbors = Neighbors {
                    n: top,
                    w: left,
                    nw: topleft,
                    ne: topright,
                    nn: toptop,
                    ww: leftleft,
                    nee,
                };

                let local_x = x - start_bx;
                let local_y = y - start_by;

                // Always run WP state for property 15 (wp_max_error) and to
                // keep state consistent with the decoder for any
                // Weighted-predictor leaves.
                let (wp_pred, wp_max_error) =
                    wp_state.predict_and_property(local_x, local_y, region_width, &neighbors);

                // Compute extended property vector (16 entries)
                let (mut props, new_local_grad) = compute_dc_properties(
                    enc_idx as u32,
                    x - start_bx,
                    y - start_by,
                    top,
                    left,
                    topleft,
                    topright,
                    toptop,
                    leftleft,
                    prev_local_grad,
                );
                props[15] = wp_max_error;

                // Traverse the tree to find the leaf's context + predictor.
                let (ctx_id, leaf_predictor) = get_dc_context_and_predictor(tree, &props);

                // Compute prediction using leaf's chosen predictor.
                let prediction = if leaf_predictor == Predictor::Weighted as u32 {
                    wp_pred as i32
                } else if let Some(pred) = Predictor::from_id(leaf_predictor as u8) {
                    pred.predict_from_neighbors(&neighbors)
                } else {
                    // Defensive: unknown predictor id — use clamped gradient.
                    clamped_gradient(top, left, topleft)
                };

                let residual = dc_val - prediction;
                tokens.push(Token::new(ctx_id, pack_signed(residual)));

                // Always update WP error state (mandatory for state consistency).
                wp_state.update_errors(dc_val, local_x, local_y, region_width);

                prev_local_grad = new_local_grad;
            }
        }
    }

    tokens
}

/// Traverse the learned tree to get (context_id, predictor) for a DC value.
///
/// Like `get_dc_context` but also returns the leaf's predictor field.
#[inline]
pub fn get_dc_context_and_predictor(tree: &DcTree, props: &[i32; NUM_DC_PROPERTIES]) -> (u32, u32) {
    let mut idx = 0;
    loop {
        let node = &tree[idx];
        if node.property < 0 {
            return (node.context_id, node.predictor);
        }
        let pval = props[node.property as usize];
        if pval <= node.splitval {
            idx = node.lchild;
        } else {
            idx = node.rchild;
        }
    }
}

/// Collect DC tokens using a learned tree for context assignment.
///
/// This is the learned-tree version of `collect_dc_tokens_region()` from dc_coding.rs.
/// Instead of using GRADIENT_CONTEXT_LUT, it traverses the learned tree to get contexts.
pub fn collect_dc_tokens_with_tree(
    quant_dc: &[Vec<Vec<i16>>; 3],
    tree: &DcTree,
    start_bx: usize,
    start_by: usize,
    end_bx: usize,
    end_by: usize,
) -> Vec<crate::entropy_coding::token::Token> {
    use crate::entropy_coding::token::Token;

    let region_width = end_bx - start_bx;
    let region_height = end_by - start_by;

    if region_width == 0 || region_height == 0 {
        return Vec::new();
    }

    let mut tokens = Vec::with_capacity(region_width * region_height * 3);

    // Encode in channel order: Y (1), X (0), B (2)
    for (enc_idx, &c) in [1usize, 0, 2].iter().enumerate() {
        let channel = &quant_dc[c];

        for y in start_by..end_by {
            let mut prev_local_grad = 0i32;

            for x in start_bx..end_bx {
                let dc_val = channel[y][x] as i32;

                // Get neighbors with proper edge handling
                let left = if x > start_bx {
                    channel[y][x - 1] as i32
                } else if y > start_by {
                    channel[y - 1][x] as i32
                } else {
                    0
                };

                let top = if y > start_by {
                    channel[y - 1][x] as i32
                } else {
                    left
                };

                let topleft = if x > start_bx && y > start_by {
                    channel[y - 1][x - 1] as i32
                } else {
                    left
                };

                let topright = if y > start_by && x + 1 < end_bx {
                    channel[y - 1][x + 1] as i32
                } else {
                    top
                };

                let toptop = if y > start_by + 1 {
                    channel[y - 2][x] as i32
                } else {
                    top
                };

                let leftleft = if x > start_bx + 1 {
                    channel[y][x - 2] as i32
                } else {
                    left
                };

                // Compute prediction and residual
                let prediction = clamped_gradient(top, left, topleft);
                let residual = dc_val - prediction;

                // Compute properties and get context from tree
                let (props, new_local_grad) = compute_dc_properties(
                    enc_idx as u32,
                    x - start_bx,
                    y - start_by,
                    top,
                    left,
                    topleft,
                    topright,
                    toptop,
                    leftleft,
                    prev_local_grad,
                );
                let tree_ctx = get_dc_context(tree, &props);
                // DC tree assigns contexts starting from 0; the encoder adds
                // NUM_AC_METADATA_CONTEXTS (11) offset when building final tokens.
                let ctx_id = tree_ctx;

                tokens.push(Token::new(ctx_id, pack_signed(residual)));

                prev_local_grad = new_local_grad;
            }
        }
    }

    tokens
}

// ──────────────────────────────────────────────────────────────────
// kWPFixedDC tree — fixed balanced BSP on property 15 (wp_max_error)
// Matches libjxl's PredefinedTree(kWPFixedDC, ...) exactly.
// ──────────────────────────────────────────────────────────────────

/// kWPFixedDC cutoff values (from libjxl enc_encoding.cc).
/// These are the split thresholds for the wp_max_error property.
const WP_FIXED_DC_CUTOFFS: &[i32] = &[
    -500, -392, -255, -191, -127, -95, -63, -47, -31, -23, -15, -11, -7, -4, -3, -1, 0, 1, 3, 5, 7,
    11, 15, 23, 31, 47, 63, 95, 127, 191, 255, 392, 500,
];

/// Property index for wp_max_error in the JXL modular property list.
/// kNumStaticProperties(2) + 13 = 15. Used for tree serialization.
pub const WP_PROP_INDEX: i32 = 15;

/// Build the kWPFixedDC tree: a balanced BSP tree on wp_max_error (property 15)
/// with all leaves using Predictor::Weighted.
///
/// Matches libjxl's `MakeFixedTree(kWPProp, cutoffs, Predictor::Weighted, total_pixels, bitdepth)`.
///
/// # Arguments
/// * `total_pixels` - total DC pixels (width_blocks * height_blocks * 3 channels)
/// * `bitdepth` - bit depth of the DC values (typically 8)
pub fn build_wp_fixed_dc_tree(total_pixels: usize, bitdepth: u32) -> (DcTree, u32) {
    let log_px = if total_pixels > 0 {
        (usize::BITS - total_pixels.leading_zeros()) as usize // ceil_log2
    } else {
        0
    };
    let min_gap = if log_px < 14 { 8 * (14 - log_px) } else { 0 };
    let shift = if bitdepth > 11 {
        (bitdepth - 11).min(4)
    } else {
        0
    };
    let mul = 1i32 << shift;

    let cutoffs = WP_FIXED_DC_CUTOFFS;
    let mut tree = DcTree::new();
    let mut next_context = 0u32;

    build_wp_bsp_recursive(
        cutoffs,
        0,
        cutoffs.len(),
        min_gap,
        mul,
        &mut tree,
        &mut next_context,
    );

    (tree, next_context)
}

/// Recursively build a balanced BSP tree from sorted cutoffs.
///
/// Mirrors libjxl's MakeFixedTree BFS queue, but builds in DFS order
/// (our tree_tokens_with_ac_metadata_prefix handles the BFS conversion).
fn build_wp_bsp_recursive(
    cutoffs: &[i32],
    begin: usize,
    end: usize,
    min_gap: usize,
    mul: i32,
    tree: &mut DcTree,
    next_context: &mut u32,
) -> usize {
    let node_idx = tree.len();

    if begin + min_gap >= end {
        // Leaf node
        tree.push(DcTreeNode {
            property: -1,
            context_id: *next_context,
            predictor: 6, // Predictor::Weighted
            ..Default::default()
        });
        *next_context += 1;
        return node_idx;
    }

    let split = (begin + end) / 2;
    let cutoff = cutoffs[split] * mul;

    // Placeholder — filled after children are built
    tree.push(DcTreeNode::default());

    // rchild = values > cutoff → covers [split+1, end)
    let rchild = build_wp_bsp_recursive(cutoffs, split + 1, end, min_gap, mul, tree, next_context);
    // lchild = values <= cutoff → covers [begin, split)
    let lchild = build_wp_bsp_recursive(cutoffs, begin, split, min_gap, mul, tree, next_context);

    tree[node_idx] = DcTreeNode {
        property: WP_PROP_INDEX,
        splitval: cutoff,
        lchild,
        rchild,
        context_id: 0,
        predictor: 0,
    };

    node_idx
}

/// Traverse the kWPFixedDC tree using wp_max_error value.
///
/// Specialized traversal for the WP fixed tree — only uses the wp_max_error
/// property (property 15), which is the only property this tree splits on.
#[inline]
pub fn get_wp_dc_context(tree: &DcTree, wp_max_error: i32) -> u32 {
    let mut idx = 0;
    loop {
        let node = &tree[idx];
        if node.property < 0 {
            return node.context_id;
        }
        // All splits are on wp_max_error (property 15)
        if wp_max_error <= node.splitval {
            idx = node.lchild;
        } else {
            idx = node.rchild;
        }
    }
}

/// Compress statistics for learned DC tree.
pub struct DcTreeStats {
    /// Number of contexts used by the tree.
    pub num_contexts: u32,
    /// Number of samples collected.
    pub num_samples: usize,
    /// Estimated bits saved compared to fixed LUT (positive = better).
    pub bits_saved: f64,
}

/// Learn DC tree and collect tokens in one pass.
///
/// Returns (tree, tokens, stats) where:
/// - tree is the learned context tree
/// - tokens are DC tokens using the learned contexts
/// - stats contains compression statistics
pub fn learn_and_collect_dc_tokens(
    quant_dc: &[Vec<Vec<i16>>; 3],
    start_bx: usize,
    start_by: usize,
    end_bx: usize,
    end_by: usize,
) -> (
    DcTree,
    Vec<crate::entropy_coding::token::Token>,
    DcTreeStats,
) {
    // First pass: gather samples
    let mut samples = DcTreeSamples::new();

    if !quant_dc[0].is_empty() && !quant_dc[0][0].is_empty() {
        // Create a view of just this region for sample gathering
        let region_dc = extract_dc_region(quant_dc, start_bx, start_by, end_bx, end_by);
        gather_dc_samples(&mut samples, &region_dc);
    }

    // Learn tree
    let max_token = 64; // Reasonable max for DC residual tokens
    let (tree, num_contexts) = learn_dc_tree(&samples, max_token);

    // Collect tokens using learned tree
    let tokens = collect_dc_tokens_with_tree(quant_dc, &tree, start_bx, start_by, end_bx, end_by);

    let stats = DcTreeStats {
        num_contexts,
        num_samples: samples.num_samples,
        bits_saved: 0.0, // TODO: estimate actual savings
    };

    (tree, tokens, stats)
}

/// Extract a region of DC values for sample gathering.
#[allow(clippy::needless_range_loop)]
fn extract_dc_region(
    quant_dc: &[Vec<Vec<i16>>; 3],
    start_bx: usize,
    start_by: usize,
    end_bx: usize,
    end_by: usize,
) -> [Vec<Vec<i16>>; 3] {
    let width = end_bx - start_bx;
    let height = end_by - start_by;

    let mut result: [Vec<Vec<i16>>; 3] = [Vec::new(), Vec::new(), Vec::new()];

    for c in 0..3 {
        let mut channel = Vec::with_capacity(height);
        for y in start_by..end_by {
            let mut row = Vec::with_capacity(width);
            for x in start_bx..end_bx {
                row.push(quant_dc[c][y][x]);
            }
            channel.push(row);
        }
        result[c] = channel;
    }

    result
}

#[cfg(test)]
mod debug_tests {
    use super::*;
    use crate::bit_writer::BitWriter;
    use crate::vardct::context_tree::{write_context_tree, write_learned_context_tree};

    #[test]
    fn test_static_tokens_through_learned_path() {
        use crate::vardct::common::pack_signed;
        use crate::vardct::context_tree::CONTEXT_TREE_TOKENS;
        let num_dc_groups = 1;

        // Get the static tokens with num_dc_groups adjustment
        let mut static_token_pairs: Vec<(u32, u32)> = CONTEXT_TREE_TOKENS.to_vec();
        static_token_pairs[1].1 = pack_signed(1 + num_dc_groups as i32);

        // Write static tree via static path
        let mut static_writer = BitWriter::new();
        write_context_tree(num_dc_groups, &mut static_writer).unwrap();
        static_writer.zero_pad_to_byte();
        let static_bytes = static_writer.finish();

        // Write same tokens via learned path
        let mut learned_writer = BitWriter::new();
        write_learned_context_tree(&static_token_pairs, num_dc_groups, &mut learned_writer)
            .unwrap();
        learned_writer.zero_pad_to_byte();
        let learned_bytes = learned_writer.finish();

        eprintln!(
            "Static: {} bytes, Learned: {} bytes",
            static_bytes.len(),
            learned_bytes.len()
        );

        // They should be bit-identical since they use the same tokens
        assert_eq!(
            static_bytes, learned_bytes,
            "Static and learned paths produce different output for same tokens"
        );
    }
}

#[test]
fn test_wrapped_tree_tokens() {
    use super::*;

    // Single-leaf learned tree (1 DC context, depth 0)
    // Single-leaf DC tree: total = 11 AC meta + 1 DC = 12
    let tree = vec![DcTreeNode {
        property: -1,
        context_id: 0,
        ..Default::default()
    }];

    let (wrapped_tokens, total_contexts, dc_remap, ac_map) =
        tree_tokens_with_ac_metadata_prefix(&tree, 1, 1);
    eprintln!(
        "Merged tree: {} tokens, {} contexts, dc_remap={:?}, ac_map={:?}",
        wrapped_tokens.len(),
        total_contexts,
        dc_remap,
        ac_map,
    );

    assert_eq!(dc_remap.len(), 1);
    assert_eq!(total_contexts, 12); // 11 AC meta + 1 DC
    // All contexts (DC and AC meta) should be unique and within [0, total)
    let mut all_ctxs = std::collections::HashSet::new();
    for &bfs in &dc_remap {
        assert!(
            bfs < total_contexts,
            "DC ctx {} >= total {}",
            bfs,
            total_contexts
        );
        assert!(all_ctxs.insert(bfs), "Duplicate DC BFS context {}", bfs);
    }
    for &bfs in &ac_map {
        assert!(
            bfs < total_contexts,
            "AC meta ctx {} >= total {}",
            bfs,
            total_contexts
        );
        assert!(
            all_ctxs.insert(bfs),
            "Duplicate AC meta BFS context {}",
            bfs
        );
    }
}

#[test]
fn test_wrapped_tree_tokens_depth1_dc() {
    use super::*;

    // Depth-1 DC tree (2 leaves): total = 11 AC meta + 2 DC = 13
    let tree = vec![
        DcTreeNode {
            property: 9,
            splitval: 0,
            lchild: 1,
            rchild: 2,
            ..Default::default()
        },
        DcTreeNode {
            property: -1,
            context_id: 0,
            predictor: 5,
            ..Default::default()
        },
        DcTreeNode {
            property: -1,
            context_id: 1,
            predictor: 5,
            ..Default::default()
        },
    ];

    let (_, total_contexts, dc_remap, ac_map) = tree_tokens_with_ac_metadata_prefix(&tree, 2, 1);
    eprintln!(
        "Depth-1 DC: total={}, dc_remap={:?}, ac_map={:?}",
        total_contexts, dc_remap, ac_map
    );

    // 11 AC meta + 2 DC = 13 (no padding dummies)
    assert_eq!(total_contexts, 13);
    assert_eq!(dc_remap.len(), 2);
    // All contexts should be unique and within [0, total)
    let mut all_ctxs = std::collections::HashSet::new();
    for (i, &bfs) in dc_remap.iter().enumerate() {
        assert!(
            bfs < total_contexts,
            "DC remap[{}]={} >= total {}",
            i,
            bfs,
            total_contexts
        );
        assert!(
            all_ctxs.insert(bfs),
            "Duplicate DC ctx {} at remap[{}]",
            bfs,
            i
        );
    }
    for (i, &bfs) in ac_map.iter().enumerate() {
        assert!(
            bfs < total_contexts,
            "AC meta ctx {} >= total {} at map[{}]",
            bfs,
            total_contexts,
            i
        );
        assert!(
            all_ctxs.insert(bfs),
            "Duplicate AC meta ctx {} at map[{}]",
            bfs,
            i
        );
    }
}

#[test]
fn test_wrapped_tree_tokens_deep_dc() {
    use super::*;

    // DC tree with depth 5 (no padding needed):
    // Build a balanced binary tree with 32 leaves
    let mut tree = Vec::new();
    for i in 0..31 {
        tree.push(DcTreeNode {
            property: 9,
            splitval: (i as i32) * 10,
            lchild: i * 2 + 1,
            rchild: i * 2 + 2,
            ..Default::default()
        });
    }
    for i in 0..32 {
        tree.push(DcTreeNode {
            property: -1,
            context_id: i,
            predictor: 5,
            ..Default::default()
        });
    }

    let (_, total_contexts, dc_remap, ac_map) = tree_tokens_with_ac_metadata_prefix(&tree, 32, 1);
    eprintln!(
        "Deep DC: total={}, dc_remap={:?}, ac_map={:?}",
        total_contexts, dc_remap, ac_map
    );

    // No padding needed → no dummies → AC metadata contexts are exactly 0-10
    assert_eq!(total_contexts, 43); // 11 AC meta + 32 DC
    assert_eq!(dc_remap.len(), 32);
    // All DC contexts should be >= 11 and unique
    let mut dc_set = std::collections::HashSet::new();
    for (i, &bfs) in dc_remap.iter().enumerate() {
        assert!(bfs >= 11, "DC remap[{}]={} < 11", i, bfs);
        assert!(
            bfs < total_contexts,
            "DC remap[{}]={} >= total {}",
            i,
            bfs,
            total_contexts
        );
        assert!(
            dc_set.insert(bfs),
            "Duplicate DC BFS context {} at remap[{}]",
            bfs,
            i
        );
    }
    for i in 0..11u32 {
        assert_eq!(
            ac_map[i as usize], i,
            "AC meta {} not at expected BFS position",
            i
        );
    }
}
