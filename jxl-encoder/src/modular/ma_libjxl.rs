// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Strict `EncoderStrategy::Libjxl` port of libjxl's adaptive MA-tree
//! learning (`enc_ma.cc`, `enc_encoding.cc::GatherTreeData`,
//! `enc_modular.cc::ComputeTree`/`MergeTrees`).
//!
//! This module exists because the production learner in `tree_learn.rs`
//! is intentionally divergent (optimized split search, different sampling
//! and quantization). The strict path must reproduce libjxl's exact tree
//! *decisions*, which are sensitive to:
//!
//! - xorshift128+ sampling sequences (`GatherTreeData`, `CollectPixelSamples`)
//! - `EstimateBits` f32 accumulation order (AVX2 `SumOfLanes` pairing)
//! - `FastLog2f` rational-polynomial evaluation with fused multiply-add
//! - property quantization (`kPropertyRange = 511`, lazy abs-ordering)
//! - `FindBestSplit` candidate ordering and penalty terms
//! - `MergeTrees` stream-id splits + BFS `TokenizeTree` leaf numbering
//!
//! Everything here is a line-level port; do not "improve" anything.
//! In-memory nodes follow libjxl's convention (`lchild` = `>` side);
//! conversion to [`Tree`] happens once at the end.

use super::channel::{Channel, ModularImage};
use super::predictor::{Neighbors, Predictor, WeightedPredictorState, pack_signed};
use super::tree::{PropertyDecisionNode, Tree, assign_sequential_contexts};

/// libjxl `kNumStaticProperties` (channel index + group/stream id).
const NUM_STATIC_PROPS: usize = 2;
/// libjxl `kNumModularPredictors` (simple predictors only).
const NUM_MODULAR_PREDICTORS: usize = 14;
/// libjxl `TreeSamples::kPropertyRange` — quantized properties map the
/// [-511, 511] clamped value range onto `max_property_values` buckets.
const PROPERTY_RANGE: i32 = 511;
/// libjxl `ANS_TAB_SIZE` — `EstimateBits` probability floor.
const ANS_TAB_SIZE: f32 = 4096.0;
/// libjxl `kDedupEntryUnused`.
const DEDUP_UNUSED: u32 = u32::MAX;
/// libjxl `kWPProp` = `kNumNonrefProperties - weighted::kNumProperties`.
const WP_PROP: u32 = 15;
/// libjxl `kNumNonrefProperties` (no reference channels: max_properties=0).
const NUM_NONREF_PROPERTIES: usize = 16;
/// libjxl `kNumQuantTables`.
pub(crate) const NUM_QUANT_TABLES: usize = 17;

/// `ceil(x / 8)` — Highway AVX2 `Lanes(f32)`.
#[inline]
fn padded(x: usize) -> usize {
    (x + 7) & !7
}

/// libjxl `Predictor` meta-modes used at stream-option level.
///
/// `Variable`/`Best` are retained for completeness — libjxl's only
/// assignment site for them on the VarDCT DC stream is dead code
/// (`!nl_dc` is never true at `speed_tier < kSquirrel`; see
/// `vardct_stream_options`), but they mirror `SetPredictor`'s full
/// surface for any future non-VarDCT reuse.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StreamPredictor {
    /// `Predictor::Variable` — all 14 predictors, WP and Gradient first.
    #[allow(dead_code)]
    Variable,
    /// `Predictor::Best` — {Weighted, Gradient}.
    #[allow(dead_code)]
    Best,
    /// A single concrete predictor (Gradient for AC-metadata).
    Single(Predictor),
}

/// libjxl `ModularOptions::TreeMode` subset used by VarDCT streams.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WpTreeMode {
    /// `kDefault` — full property set.
    #[allow(dead_code)]
    Default,
    /// `kNoWP` — property 15 (`wp_max_error`) and the Weighted predictor
    /// are removed from consideration.
    NoWp,
    /// `kWPOnly` — predictors forced to `{Weighted}` and `props_to_use`
    /// forced to `{15}` (`enc_ma.cc:542-546,574-576`). libjxl sets this on
    /// the VarDCT DC stream unconditionally (`enc_modular.cc:1590-1591`).
    WpOnly,
}

/// The `ModularOptions` fields that influence MA-tree learning.
#[derive(Debug, Clone)]
pub(crate) struct LibjxlModularOptions {
    /// `options.predictor` (Best / Variable / concrete).
    pub predictor: StreamPredictor,
    /// `options.wp_tree_mode`.
    pub wp_tree_mode: WpTreeMode,
    /// `options.splitting_heuristics_properties`.
    pub properties: Vec<u32>,
    /// `options.nb_repeats` — sampling fraction for `GatherTreeData`.
    pub nb_repeats: f32,
    /// `options.max_property_values`.
    pub max_property_values: usize,
    /// `options.splitting_heuristics_node_threshold`.
    pub node_threshold: f32,
    /// `options.fast_decode_multiplier` (1.0 for VarDCT).
    pub fast_decode_multiplier: f32,
    /// `options.max_chan_size` (0xFFFFFF for VarDCT streams).
    pub max_chan_size: usize,
}

/// `cparams_.options` resolution shared by every VarDCT modular stream:
/// the non-squeeze `splitting_heuristics_properties` order (with the
/// "few groups → no group property" deletion) and the per-tier
/// `max_property_values`/`nb_mul` knobs (`enc_modular.cc:438-560`).
///
/// `tier` is libjxl's `SpeedTier` (`10 - effort`). `nb_mul` scales the
/// 0.5 `nb_repeats` base (`enc_modular.cc:555-559`).
fn tier_splitting_props(tier: i32, num_streams: usize) -> (Vec<u32>, usize, f32) {
    const K_TORTOISE: i32 = 1;
    const K_KITTEN: i32 = 2;
    const K_SQUIRREL: i32 = 3;
    // Non-squeeze prop order (`responsive == 0` — `ModularPartIsLossless()`
    // is true for VarDCT so `responsive` resolves to 0).
    let prop_order: Vec<u32> = vec![0, 1, 15, 9, 10, 11, 12, 13, 14, 2, 3, 4, 5, 6, 7, 8];
    // "if few groups, don't use group as a property" — `num_streams < 30
    // && tier > kTortoise && cparams_orig.ModularPartIsLossless()`.
    let mut prop_order = prop_order;
    if num_streams < 30 && tier > K_TORTOISE {
        prop_order.remove(1);
    }
    // `max_properties` is 0 by default → no reference-channel properties.
    let (num_properties, max_property_values, nb_mul) = match tier {
        // kGlacier / kTortoise
        t if t <= K_TORTOISE => (prop_order.len(), 256usize, 1.3f32),
        K_KITTEN => (10, 128usize, 1.1f32),
        K_SQUIRREL => (7, 96usize, 1.0f32),
        // kWombat
        4 => (5, 64usize, 0.7f32),
        // kHare
        5 => (4, 48usize, 0.5f32),
        // kCheetah and faster
        _ => (3, 32usize, 0.3f32),
    };
    prop_order.truncate(num_properties);
    (prop_order, max_property_values, nb_mul)
}

/// Build the resolved per-tier `ModularOptions` for a VarDCT frame,
/// mirroring `ModularFrameEncoder::Init` + `AddVarDCTDC`/`AddACMetadata`
/// overrides (`enc_modular.cc`).
///
/// `tier` is libjxl's `SpeedTier` (`10 - effort`): e8 → kKitten(2),
/// e9 → kTortoise(1), e10+ → kGlacier(0).
/// `num_streams` is `ModularStreamId::Num(frame_dim, passes)`.
///
/// Returns `(dc_options, ac_meta_options)` for the
/// `[VarDCTDC(0) .. ACMetadata(0))` and `[ACMetadata(0) .. num_streams)`
/// chunks respectively.
pub(crate) fn vardct_stream_options(
    tier: i32,
    num_streams: usize,
) -> (LibjxlModularOptions, LibjxlModularOptions) {
    // Start from the same cparams options as GlobalData, then apply only
    // AddVarDCTDC/AddACMetadata's stream-specific overrides.
    let mut ac_meta = global_stream_options(tier, num_streams, 0xFF_FFFF);
    ac_meta.wp_tree_mode = WpTreeMode::NoWp;

    // AddVarDCTDC (`enc_modular.cc:1587-1605`): the unconditional
    // assignment is `predictor = Weighted`, `wp_tree_mode = kWPOnly`.
    // The `speed_tier < kSquirrel && !nl_dc` override to
    // Best/Variable + kDefault is DEAD for VarDCT — the only call site
    // passes `nl_dc = speed_tier < kFalcon` (`enc_cache.cc:233`), which is
    // true at every effort where kLearn runs (e8+: tier <= kKitten <
    // kFalcon). The kGradientFixedDC arm only fires with
    // `decoding_speed_tier >= 1`.
    let dc = LibjxlModularOptions {
        predictor: StreamPredictor::Single(Predictor::Weighted),
        wp_tree_mode: WpTreeMode::WpOnly,
        ..ac_meta.clone()
    };
    (dc, ac_meta)
}

/// `stream_options_[0]` — the GlobalData stream takes `cparams_.options`
/// verbatim (`enc_modular.cc:675`), i.e. the state after the VarDCT
/// `modular_mode == false` overrides above it: predictor resolved to
/// `Predictor::Gradient` for lossy non-responsive VarDCT
/// (`enc_modular.cc:634-636`), `fast_decode_multiplier = 1.0`
/// (`enc_modular.cc:659`), and `max_chan_size = frame_dim_.group_dim`
/// (`enc_modular.cc:669`). `tree_kind` stays `kLearn` — the
/// `kWPFixedDC`/`kGradientFixedDC` overrides only fire at effort 3/2
/// (`enc_modular.cc:676-680`), outside the learned-tree range this
/// resolves for.
pub(crate) fn global_stream_options(
    tier: i32,
    num_streams: usize,
    group_dim: usize,
) -> LibjxlModularOptions {
    let (properties, max_property_values, nb_mul) = tier_splitting_props(tier, num_streams);
    LibjxlModularOptions {
        predictor: StreamPredictor::Single(Predictor::Gradient),
        wp_tree_mode: WpTreeMode::Default,
        properties,
        nb_repeats: (0.5f32 * nb_mul).min(1.0),
        max_property_values,
        node_threshold: (75 + 14 * tier) as f32,
        fast_decode_multiplier: 1.0,
        max_chan_size: group_dim,
    }
}

// ────────────────────────────────────────────────────────────────────────
// `Rng` — xorshift128+ port of `base/random.h`.
// ────────────────────────────────────────────────────────────────────────

struct Rng {
    s: [u64; 2],
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self {
            s: [
                0x94D0_49BB_1331_11EB,
                0xBF58_476D_1CE4_E5B9u64.wrapping_add(seed),
            ],
        }
    }

    #[inline]
    fn next(&mut self) -> u64 {
        let mut s1 = self.s[0];
        let s0 = self.s[1];
        let bits = s1.wrapping_add(s0);
        self.s[0] = s0;
        s1 ^= s1 << 23;
        s1 ^= s0 ^ (s1 >> 18) ^ (s0 >> 5);
        self.s[1] = s1;
        bits
    }

    /// `UniformF` — 23-bit uniform float in [begin, end).
    fn uniform_f(&mut self, begin: f32, end: f32) -> f32 {
        let u = ((self.next() >> (64 - 23)) | 0x3F80_0000) as u32;
        let f = f32::from_bits(u);
        (end - begin) * (f - 1.0) + begin
    }

    /// `MakeGeometric(p)` — note the C++ evaluates `1.0 / std::log(1 - p)`
    /// in double precision (the `1.0` literal is double), then narrows.
    fn make_geometric(p: f32) -> f32 {
        (1.0f64 / (1.0f32 - p).ln() as f64) as f32
    }

    /// `Geometric` — `static_cast<uint32_t>` truncates toward zero; the
    /// log-product is positive (two negatives), so this floors.
    fn geometric(&mut self, dist: f32) -> u32 {
        let f = self.uniform_f(0.0, 1.0);
        let log = (1.0f32 - f).ln() * dist;
        log as u32
    }
}

// ────────────────────────────────────────────────────────────────────────
// `FastLog2f` — scalar port of `fast_math-inl.h` (2,2 rational polynomial
// with fused multiply-add, IEEE division — matches HWY AVX2 semantics).
// ────────────────────────────────────────────────────────────────────────

#[inline]
fn fast_log2f(x: f32) -> f32 {
    const P: [f32; 3] = [
        -1.850_383_340_051_831E-06,
        1.428_716_047_008_375_5,
        7.424_587_332_782_056_6E-01,
    ];
    const Q: [f32; 3] = [
        9.903_281_427_759_071_9E-01,
        1.009_671_857_224_114_8,
        1.740_934_300_336_685_3E-01,
    ];
    let x_bits = x.to_bits() as i32;
    let exp_bits = x_bits.wrapping_sub(0x3f2a_aaab);
    let exp_shifted = exp_bits >> 23;
    let mantissa = f32::from_bits(x_bits.wrapping_sub(exp_shifted << 23) as u32);
    let exp_val = exp_shifted as f32;
    let xm1 = mantissa - 1.0;
    // EvalRationalPolynomial: Horner with fused multiply-add, then a real
    // IEEE division (`FastDivision` uses `Div` on x86).
    let yp = (P[2].mul_add(xm1, P[1])).mul_add(xm1, P[0]);
    let yq = (Q[2].mul_add(xm1, Q[1])).mul_add(xm1, Q[0]);
    yp / yq + exp_val
}

/// `enc_modular_simd.cc::EstimateCost` for strict Global-stream palettes.
/// Includes every channel, including palette metadata. Keep the integer and
/// fractional entropy sums separate, and round only once after all channels.
/// Histogram reduction uses the strict learner's canonical AVX2 lane order.
pub(crate) fn estimate_global_image_cost<'a>(channels: impl IntoIterator<Item = &'a Channel>) -> f32 {
    const CUTOFFS: [u32; 17] = [
        0, 1, 3, 5, 7, 11, 15, 23, 31, 47, 63, 95, 127, 191, 255, 392, 500,
    ];
    let mut integer_cost = 0u64;
    let mut fractional_cost = 0.0f32;
    let mut extra_bits = 0u64;
    let mut histograms = [[0u32; 128]; 17];
    for ch in channels {
        for y in 0..ch.height() {
            let row = ch.row(y);
            let prev = if y == 0 { row } else { ch.row(y - 1) };
            for x in 0..ch.width() {
                let left = if x != 0 {
                    row[x - 1]
                } else if y != 0 {
                    prev[x]
                } else {
                    0
                };
                let top = if y != 0 { prev[x] } else { left };
                let top_left = if x != 0 && y != 0 { prev[x - 1] } else { left };
                let spread = left
                    .max(top)
                    .max(top_left)
                    .abs_diff(left.min(top).min(top_left));
                let ctx = CUTOFFS.partition_point(|&v| v <= spread) - 1;
                let prediction = super::predictor::clamped_gradient(top, left, top_left);
                let packed = pack_signed(row[x].wrapping_sub(prediction));
                // Match the SIMD HybridUint(4,2,0) conversion, including the
                // right shift that avoids rounding large integers upward.
                let (token, nbits) = if packed < 16 {
                    (packed, 0)
                } else {
                    let large = packed > (1 << 22) - 1;
                    let fixed = if large { packed >> 10 } else { packed };
                    let bits = (fixed as f32).to_bits();
                    let nbits = (bits >> 23) - 129 + if large { 10 } else { 0 };
                    (8 + 4 * nbits + ((bits >> 21) & 3), nbits)
                };
                histograms[ctx][token as usize] += 1;
                extra_bits += u64::from(nbits);
            }
        }
        for histogram in &mut histograms {
            let total: u32 = histogram.iter().sum();
            if total != 0 {
                let total_f = total as f32;
                let inv_total = 1.0 / total_f;
                let mut lanes = [0.0f32; 8];
                for (i, &count) in histogram.iter().enumerate() {
                    if count != 0 && count != total {
                        lanes[i & 7] += 0.0 - count as f32 * fast_log2f(count as f32 * inv_total);
                    }
                }
                let cost = ((lanes[0] + lanes[4]) + (lanes[3] + lanes[7]))
                    + ((lanes[1] + lanes[5]) + (lanes[2] + lanes[6]));
                let whole = cost as u64;
                integer_cost += whole;
                fractional_cost += cost - whole as f32;
            }
            histogram.fill(0);
        }
    }
    (extra_bits + integer_cost + fractional_cost as u64) as f32
}

/// `EstimateBits` — exact AVX2-order port.
///
/// `counts` must be padded to a multiple of 8 (`Padded`). The integer
/// total is order-insensitive; the float accumulation uses Highway's
/// AVX2 `SumOfLanes` pairing `((v0+v4)+(v3+v7)) + ((v1+v5)+(v2+v6))`
/// (`ReduceAcrossBlocks` swaps 16-byte blocks, then the 4-lane
/// `ReduceWithinBlocks` computes `(w0+w3)+(w1+w2)`).
fn estimate_bits(counts: &[i32]) -> f32 {
    debug_assert_eq!(counts.len() % 8, 0);
    let mut total: i32 = 0;
    for &c in counts {
        total = total.wrapping_add(c);
    }
    let minprob = 1.0f32 / ANS_TAB_SIZE;
    let inv_total = 1.0f32 / total as f32;
    let mut lanes = [0.0f32; 8];
    for chunk in counts.as_chunks::<8>().0 {
        for (i, &c) in chunk.iter().enumerate() {
            let cf = c as f32;
            let mprobs = (cf * inv_total).max(minprob);
            // IfThenZeroElse(Eq(counts, total), log) — a single dominant
            // symbol contributes zero bits.
            let nbps = if c == total { 0.0 } else { fast_log2f(mprobs) };
            // Mul then Sub — separate ops, not fused.
            let prod = cf * nbps;
            lanes[i] -= prod;
        }
    }
    (lanes[0] + lanes[4] + (lanes[3] + lanes[7])) + ((lanes[1] + lanes[5]) + (lanes[2] + lanes[6]))
}

// ────────────────────────────────────────────────────────────────────────
// `TreeSamples` port.
// ────────────────────────────────────────────────────────────────────────

/// libjxl `ResidualToken`.
#[derive(Clone, Copy, Default)]
struct ResidualToken {
    tok: u8,
    nbits: u8,
}

/// `HybridUintConfig(4, 1, 2).Encode(PackSigned(v))`.
///
/// The residual `v` (`pixel_type_w`, i64) is narrowed to `int32_t` by the
/// `PackSigned` call site in libjxl; we do the same.
#[inline]
fn residual_token(v: i64) -> ResidualToken {
    let value = pack_signed(v as i32);
    // split_exponent=4 → split_token=16, msb_in_token=1, lsb_in_token=2.
    if value < 16 {
        return ResidualToken {
            tok: value as u8,
            nbits: 0,
        };
    }
    let n = 31 - value.leading_zeros();
    let m = value - (1 << n);
    let tok = 16 + ((n - 4) << 3) + ((m >> (n - 1)) << 2) + (m & 3);
    let nbits = n - 1 - 2;
    ResidualToken {
        tok: tok as u8,
        nbits: nbits as u8,
    }
}

/// `TreeSamples` port — deduplicated samples with quantized properties.
struct TreeSamples {
    /// Resolved predictor set (`predictors`), each with a residual list.
    predictors: Vec<Predictor>,
    /// `props_to_use`.
    props_to_use: Vec<u32>,
    num_static_props: usize,
    residuals: Vec<Vec<ResidualToken>>,
    sample_counts: Vec<u16>,
    /// Quantized static property values (only [0..num_static_props) used).
    static_props: [Vec<u32>; NUM_STATIC_PROPS],
    /// Quantized non-static property values.
    props: Vec<Vec<u8>>,
    /// `compact_properties` — per props_to_use entry, the dequantization
    /// table (threshold values).
    compact_properties: Vec<Vec<i32>>,
    /// `property_mapping` — per non-static props_to_use entry, map from
    /// clamped value + 511 to quantized bucket.
    property_mapping: Vec<Vec<u16>>,
    /// `static_property_mapping`.
    static_property_mapping: [Vec<u16>; NUM_STATIC_PROPS],
    /// `num_samples` — total merged occurrences (Σ sample_counts).
    num_samples: usize,
    dedup_table: Vec<u32>,
}

impl TreeSamples {
    fn new() -> Self {
        Self {
            predictors: Vec::new(),
            props_to_use: Vec::new(),
            num_static_props: 0,
            residuals: Vec::new(),
            sample_counts: Vec::new(),
            static_props: [Vec::new(), Vec::new()],
            props: Vec::new(),
            compact_properties: Vec::new(),
            property_mapping: Vec::new(),
            static_property_mapping: [Vec::new(), Vec::new()],
            num_samples: 0,
            dedup_table: Vec::new(),
        }
    }

    fn has_samples(&self) -> bool {
        !self.residuals.is_empty() && !self.residuals[0].is_empty()
    }
    fn num_distinct_samples(&self) -> usize {
        self.sample_counts.len()
    }
    fn num_samples(&self) -> usize {
        self.num_samples
    }
    fn num_predictors(&self) -> usize {
        self.predictors.len()
    }
    fn num_properties(&self) -> usize {
        self.props_to_use.len()
    }
    fn predictor_from_index(&self, i: usize) -> Predictor {
        self.predictors[i]
    }
    fn property_from_index(&self, i: usize) -> u32 {
        self.props_to_use[i]
    }
    fn predictor_index(&self, p: Predictor) -> usize {
        self.predictors.iter().position(|&x| x == p).unwrap()
    }
    fn token(&self, pred: usize, i: usize) -> usize {
        self.residuals[pred][i].tok as usize
    }
    fn rtokens(&self, pred: usize) -> &[ResidualToken] {
        &self.residuals[pred]
    }
    fn count(&self, i: usize) -> usize {
        self.sample_counts[i] as usize
    }
    /// `Property<S>` — `is_static` selects the table; `idx` is the index
    /// into the corresponding table (static: props_to_use index;
    /// non-static: props_to_use index - num_static_props).
    fn property(&self, is_static: bool, idx: usize, i: usize) -> usize {
        if is_static {
            self.static_props[idx][i] as usize
        } else {
            self.props[idx][i] as usize
        }
    }
    fn num_property_values(&self, prop_idx: usize) -> usize {
        self.compact_properties[prop_idx].len() + 1
    }
    fn unquantize_property(&self, prop_idx: usize, quant: u32) -> i32 {
        self.compact_properties[prop_idx][quant as usize]
    }
    fn quantize_property(&self, prop_idx: usize, v: i32) -> u32 {
        let v = v.clamp(-PROPERTY_RANGE, PROPERTY_RANGE) + PROPERTY_RANGE;
        self.property_mapping[prop_idx - self.num_static_props][v as usize] as u32
    }
    fn quantize_static_property(&self, prop_idx: usize, v: i32) -> u32 {
        let v = v.clamp(-PROPERTY_RANGE, PROPERTY_RANGE) + PROPERTY_RANGE;
        self.static_property_mapping[prop_idx][v as usize] as u32
    }

    /// `SetPredictor`.
    fn set_predictor(&mut self, predictor: StreamPredictor, wp_tree_mode: WpTreeMode) {
        if wp_tree_mode == WpTreeMode::WpOnly {
            self.predictors = vec![Predictor::Weighted];
            self.residuals = vec![Vec::new()];
            return;
        }
        match predictor {
            StreamPredictor::Variable => {
                for i in 0..NUM_MODULAR_PREDICTORS {
                    self.predictors.push(Predictor::from_id(i as u8).unwrap());
                }
                self.predictors.swap(0, Predictor::Weighted as usize);
                self.predictors.swap(1, Predictor::Gradient as usize);
            }
            StreamPredictor::Best => {
                self.predictors = vec![Predictor::Weighted, Predictor::Gradient];
            }
            StreamPredictor::Single(p) => {
                self.predictors = vec![p];
            }
        }
        if wp_tree_mode == WpTreeMode::NoWp {
            self.predictors.retain(|&p| p != Predictor::Weighted);
        }
        self.residuals = (0..self.predictors.len()).map(|_| Vec::new()).collect();
    }

    /// `SetProperties`.
    fn set_properties(&mut self, properties: &[u32], wp_tree_mode: WpTreeMode) {
        self.props_to_use = properties.to_vec();
        if wp_tree_mode == WpTreeMode::WpOnly {
            self.props_to_use = vec![WP_PROP];
        }
        if wp_tree_mode == WpTreeMode::NoWp {
            self.props_to_use.retain(|&p| p != WP_PROP);
        }
        debug_assert!(!self.props_to_use.is_empty());
        self.num_static_props = 0;
        for (i, &prop) in self.props_to_use.iter().enumerate() {
            if prop < NUM_STATIC_PROPS as u32 {
                debug_assert_eq!(i, prop as usize);
                self.num_static_props += 1;
            }
        }
        self.props = (0..self.props_to_use.len() - self.num_static_props)
            .map(|_| Vec::new())
            .collect();
    }

    /// `InitTable` + `PrepareForSamples`.
    fn init_table(&mut self, log_size: usize) {
        let size = 1usize << log_size;
        if self.dedup_table.len() == size {
            return;
        }
        self.dedup_table.clear();
        self.dedup_table.resize(size, DEDUP_UNUSED);
        for i in 0..self.num_distinct_samples() {
            if self.sample_counts[i] != u16::MAX {
                self.add_to_table(i);
            }
        }
    }

    fn prepare_for_samples(&mut self, extra_num_samples: usize) {
        for res in &mut self.residuals {
            res.reserve(extra_num_samples);
        }
        for p in &mut self.static_props {
            p.reserve(extra_num_samples);
        }
        for p in &mut self.props {
            p.reserve(extra_num_samples);
        }
        let total_num_samples = extra_num_samples + self.sample_counts.len();
        // CeilLog2Nonzero(total * 3 / 2)
        let n = total_num_samples.saturating_mul(3) / 2;
        let next_size = if n <= 1 {
            0
        } else {
            usize::BITS as usize - (n - 1).leading_zeros() as usize
        };
        self.init_table(next_size);
    }

    fn hash1(&self, a: usize) -> usize {
        const C: u64 = 0x1e35_a7bd;
        let mut h: u64 = C;
        for r in &self.residuals {
            h = h.wrapping_mul(C).wrapping_add(r[a].tok as u64);
            h = h.wrapping_mul(C).wrapping_add(r[a].nbits as u64);
        }
        for i in 0..self.num_static_props {
            h = h
                .wrapping_mul(C)
                .wrapping_add(self.static_props[i][a] as u64);
        }
        for p in &self.props {
            h = h.wrapping_mul(C).wrapping_add(p[a] as u64);
        }
        ((h >> 16) as usize) & (self.dedup_table.len() - 1)
    }

    fn hash2(&self, a: usize) -> usize {
        const C: u64 = 0x1e35_a7bd_1e35_a7bd;
        let mut h: u64 = C;
        for i in 0..self.num_static_props {
            h = h.wrapping_mul(C) ^ self.static_props[i][a] as u64;
        }
        for p in &self.props {
            h = h.wrapping_mul(C) ^ p[a] as u64;
        }
        for r in &self.residuals {
            h = h.wrapping_mul(C) ^ r[a].tok as u64;
            h = h.wrapping_mul(C) ^ r[a].nbits as u64;
        }
        ((h >> 16) as usize) & (self.dedup_table.len() - 1)
    }

    fn is_same_sample(&self, a: usize, b: usize) -> bool {
        for r in &self.residuals {
            if r[a].tok != r[b].tok || r[a].nbits != r[b].nbits {
                return false;
            }
        }
        for i in 0..self.num_static_props {
            if self.static_props[i][a] != self.static_props[i][b] {
                return false;
            }
        }
        for p in &self.props {
            if p[a] != p[b] {
                return false;
            }
        }
        true
    }

    fn add_to_table(&mut self, a: usize) {
        let pos1 = self.hash1(a);
        let pos2 = self.hash2(a);
        if self.dedup_table[pos1] == DEDUP_UNUSED {
            self.dedup_table[pos1] = a as u32;
        } else if self.dedup_table[pos2] == DEDUP_UNUSED {
            self.dedup_table[pos2] = a as u32;
        }
    }

    /// `AddToTableAndMerge` — true if merged into an existing entry.
    fn add_to_table_and_merge(&mut self, a: usize) -> bool {
        let pos1 = self.hash1(a);
        let pos2 = self.hash2(a);
        for pos in [pos1, pos2] {
            let entry = self.dedup_table[pos];
            if entry != DEDUP_UNUSED && self.is_same_sample(a, entry as usize) {
                self.sample_counts[entry as usize] += 1;
                if self.sample_counts[entry as usize] == u16::MAX {
                    self.dedup_table[pos] = DEDUP_UNUSED;
                }
                return true;
            }
        }
        self.add_to_table(a);
        false
    }

    /// `AddSample` — `properties` is the full 16-entry property array.
    fn add_sample(
        &mut self,
        pixel: i64,
        properties: &[i32],
        predictions: &[i64; NUM_MODULAR_PREDICTORS],
    ) {
        for (i, &pred) in self.predictors.iter().enumerate() {
            let v = pixel - predictions[pred as usize];
            self.residuals[i].push(residual_token(v));
        }
        for (i, &property) in properties[..self.num_static_props].iter().enumerate() {
            let q = self.quantize_static_property(i, property);
            self.static_props[i].push(q);
        }
        for (i, &prop) in self
            .props_to_use
            .iter()
            .enumerate()
            .skip(self.num_static_props)
        {
            let q = self.quantize_property(i, properties[prop as usize]);
            self.props[i - self.num_static_props].push(q as u8);
        }
        self.sample_counts.push(1);
        self.num_samples += 1;
        if self.add_to_table_and_merge(self.sample_counts.len() - 1) {
            for r in &mut self.residuals {
                r.pop();
            }
            for i in 0..self.num_static_props {
                self.static_props[i].pop();
            }
            for p in &mut self.props {
                p.pop();
            }
            self.sample_counts.pop();
        }
    }

    fn swap(&mut self, a: usize, b: usize) {
        if a == b {
            return;
        }
        for r in &mut self.residuals {
            r.swap(a, b);
        }
        for i in 0..self.num_static_props {
            self.static_props[i].swap(a, b);
        }
        for p in &mut self.props {
            p.swap(a, b);
        }
        self.sample_counts.swap(a, b);
    }

    fn all_samples_done(&mut self) {
        self.dedup_table.clear();
        self.dedup_table.shrink_to_fit();
    }

    /// `PreQuantizeProperties` — `multiplier_info` is always empty for
    /// VarDCT streams (`quants_` empty in `ComputeTree`), so the
    /// multiplier-threshold overrides are omitted.
    fn pre_quantize_properties(
        &mut self,
        group_pixel_count: &[u32],
        channel_pixel_count: &[u32],
        pixel_samples: &mut [i32],
        diff_samples: &mut [i32],
        max_property_values: usize,
    ) {
        let quantize_channel = || quantize_histogram(channel_pixel_count, max_property_values);
        let quantize_group_id = || quantize_histogram(group_pixel_count, max_property_values);
        let quantize_coordinate = || -> Vec<i32> {
            let mut quantized = Vec::with_capacity(max_property_values.saturating_sub(1));
            let mut i = 0usize;
            while i + 1 < max_property_values {
                quantized.push(((i + 1) * 256 / max_property_values) as i32 - 1);
                i += 1;
            }
            quantized
        };
        // These mirror the C++ lazy-evaluation order, including the
        // in-place abs() of the shared sample vectors.
        let mut pixel_thresholds: Vec<i32> = Vec::new();
        let mut abs_pixel_thresholds: Vec<i32> = Vec::new();
        let mut diff_thresholds: Vec<i32> = Vec::new();
        let mut abs_diff_thresholds: Vec<i32> = Vec::new();
        macro_rules! quantize_pixel {
            () => {{
                if pixel_thresholds.is_empty() {
                    pixel_thresholds = quantize_samples(pixel_samples, max_property_values);
                }
                &pixel_thresholds
            }};
        }
        macro_rules! quantize_abs_pixel {
            () => {{
                if abs_pixel_thresholds.is_empty() {
                    let _ = quantize_pixel!(); // non-abs thresholds first
                    for v in pixel_samples.iter_mut() {
                        *v = v.abs();
                    }
                    abs_pixel_thresholds = quantize_samples(pixel_samples, max_property_values);
                }
                &abs_pixel_thresholds
            }};
        }
        macro_rules! quantize_diff {
            () => {{
                if diff_thresholds.is_empty() {
                    diff_thresholds = quantize_samples(diff_samples, max_property_values);
                }
                &diff_thresholds
            }};
        }
        macro_rules! quantize_abs_diff {
            () => {{
                if abs_diff_thresholds.is_empty() {
                    let _ = quantize_diff!(); // non-abs thresholds first
                    for v in diff_samples.iter_mut() {
                        *v = v.abs();
                    }
                    abs_diff_thresholds = quantize_samples(diff_samples, max_property_values);
                }
                &abs_diff_thresholds
            }};
        }
        let quantize_wp = |max_property_values: usize| -> Vec<i32> {
            if max_property_values < 32 {
                vec![-127, -63, -31, -15, -7, -3, -1, 0, 1, 3, 7, 15, 31, 63, 127]
            } else if max_property_values < 64 {
                vec![
                    -255, -191, -127, -95, -63, -47, -31, -23, -15, -11, -7, -5, -3, -1, 0, 1, 3,
                    5, 7, 11, 15, 23, 31, 47, 63, 95, 127, 191, 255,
                ]
            } else {
                vec![
                    -255, -223, -191, -159, -127, -111, -95, -79, -63, -55, -47, -39, -31, -27,
                    -23, -19, -15, -13, -11, -9, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6,
                    7, 9, 11, 13, 15, 19, 23, 27, 31, 39, 47, 55, 63, 79, 95, 111, 127, 159, 191,
                    223, 255,
                ]
            }
        };

        self.compact_properties = (0..self.props_to_use.len()).map(|_| Vec::new()).collect();
        self.property_mapping = (0..self.props_to_use.len() - self.num_static_props)
            .map(|_| Vec::new())
            .collect();

        for i in 0..self.props_to_use.len() {
            let prop = self.props_to_use[i];
            self.compact_properties[i] = if prop == 0 {
                quantize_channel()
            } else if prop == 1 {
                quantize_group_id()
            } else if prop == 2 || prop == 3 {
                quantize_coordinate()
            } else if prop == 6
                || prop == 7
                || prop == 8
                || (prop as usize >= NUM_NONREF_PROPERTIES
                    && (prop as usize - NUM_NONREF_PROPERTIES) % 4 == 1)
            {
                quantize_pixel!().clone()
            } else if prop == 4
                || prop == 5
                || (prop as usize >= NUM_NONREF_PROPERTIES
                    && (prop as usize - NUM_NONREF_PROPERTIES).is_multiple_of(4))
            {
                quantize_abs_pixel!().clone()
            } else if prop as usize >= NUM_NONREF_PROPERTIES
                && (prop as usize - NUM_NONREF_PROPERTIES) % 4 == 2
            {
                quantize_abs_diff!().clone()
            } else if prop == WP_PROP {
                quantize_wp(max_property_values)
            } else {
                quantize_diff!().clone()
            };
            let num_pegs = (PROPERTY_RANGE * 2 + 1) as usize;
            let bias = PROPERTY_RANGE;
            let from = &self.compact_properties[i];
            let mut mapped = 0usize;
            let mut to = Vec::with_capacity(num_pegs);
            for peg in 0..num_pegs {
                while mapped < from.len() && (peg as i32 - bias) > from[mapped] {
                    mapped += 1;
                }
                to.push(mapped as u16);
            }
            if i < self.num_static_props {
                self.static_property_mapping[i] = to;
            } else {
                self.property_mapping[i - self.num_static_props] = to;
            }
        }
    }
}

/// `QuantizeHistogram`.
fn quantize_histogram(histogram: &[u32], num_chunks: usize) -> Vec<i32> {
    if histogram.is_empty() || num_chunks == 0 {
        return Vec::new();
    }
    let sum: u64 = histogram.iter().map(|&v| v as u64).sum();
    if sum == 0 {
        return Vec::new();
    }
    let mut thresholds = Vec::new();
    let mut cumsum: u64 = 0;
    let mut threshold: u64 = 1;
    for (i, &h) in histogram.iter().enumerate() {
        cumsum += h as u64;
        if cumsum * num_chunks as u64 >= threshold * sum {
            thresholds.push(i as i32);
            while cumsum * num_chunks as u64 >= threshold * sum {
                threshold += 1;
            }
        }
    }
    thresholds.pop();
    thresholds
}

/// `QuantizeSamples`.
fn quantize_samples(samples: &[i32], num_chunks: usize) -> Vec<i32> {
    if samples.is_empty() {
        return Vec::new();
    }
    const RANGE: i32 = 512;
    let min = samples.iter().copied().min().unwrap().clamp(-RANGE, RANGE);
    let mut counts = vec![0u32; (2 * RANGE + 1) as usize];
    for &s in samples {
        let off = s.clamp(-RANGE, RANGE) - min;
        counts[off as usize] += 1;
    }
    let mut thresholds = quantize_histogram(&counts, num_chunks);
    for v in &mut thresholds {
        *v += min;
    }
    thresholds
}

// ────────────────────────────────────────────────────────────────────────
// `CollectPixelSamples` + `GatherTreeData`.
// ────────────────────────────────────────────────────────────────────────

/// `CollectPixelSamples` — geometric subsampling for property
/// quantization. `group_id` is the stream id.
#[allow(clippy::too_many_arguments)]
fn collect_pixel_samples(
    image: &ModularImage,
    options: &LibjxlModularOptions,
    group_id: usize,
    group_pixel_count: &mut Vec<u32>,
    channel_pixel_count: &mut Vec<u32>,
    pixel_samples: &mut Vec<i32>,
    diff_samples: &mut Vec<i32>,
) {
    if options.nb_repeats == 0.0 {
        return;
    }
    if group_pixel_count.len() <= group_id {
        group_pixel_count.resize(group_id + 1, 0);
    }
    if channel_pixel_count.len() < image.channels.len() {
        channel_pixel_count.resize(image.channels.len(), 0);
    }
    let mut rng = Rng::new(group_id as u64);
    // Sample 10% of the final number of samples for property quantization.
    let fraction = (options.nb_repeats * 0.1).min(0.99);
    let dist = Rng::make_geometric(fraction);
    let mut total_pixels = 0usize;
    let mut channel_ids: Vec<usize> = Vec::new();
    for (i, ch) in image.channels.iter().enumerate() {
        // libjxl gates the early break on `i >= nb_meta_channels`
        // (`enc_ma.cc` CollectPixelSamples): meta channels are exempt.
        // Our `ModularImage` marks metas with `hshift == u32::MAX`
        // (libjxl's `-1` sentinel) instead of carrying a count.
        if ch.hshift != u32::MAX
            && (ch.width() > options.max_chan_size || ch.height() > options.max_chan_size)
        {
            break;
        }
        if ch.width() <= 1 || ch.height() == 0 {
            continue; // skip empty or width-1 channels
        }
        channel_ids.push(i);
        group_pixel_count[group_id] += (ch.width() * ch.height()) as u32;
        channel_pixel_count[i] += (ch.width() * ch.height()) as u32;
        total_pixels += ch.width() * ch.height();
    }
    if channel_ids.is_empty() {
        return;
    }
    let _ = total_pixels; // (reserves only)

    let mut i = 0usize;
    let mut y = 0usize;
    let mut x = 0usize;
    // `advance(amount)` — may run `i` past the end; the caller re-checks.
    macro_rules! advance {
        ($amount:expr) => {{
            x += $amount;
            while i < channel_ids.len() && x >= image.channels[channel_ids[i]].width() {
                x -= image.channels[channel_ids[i]].width();
                y += 1;
                if y == image.channels[channel_ids[i]].height() {
                    i += 1;
                    y = 0;
                }
            }
        }};
    }
    advance!(rng.geometric(dist) as usize);
    while i < channel_ids.len() {
        let row = image.channels[channel_ids[i]].row(y);
        pixel_samples.push(row[x]);
        let xp = if x == 0 { 1 } else { x - 1 };
        diff_samples.push(row[x].wrapping_sub(row[xp]));
        advance!(rng.geometric(dist) as usize + 1);
    }
}

/// `PredictOne` — full-width (`pixel_type_w`, i64) prediction.
fn predict_one_w(p: Predictor, n: &Neighbors, wp_pred: i64) -> i64 {
    let left = n.w as i64;
    let top = n.n as i64;
    let toptop = n.nn as i64;
    let topleft = n.nw as i64;
    let topright = n.ne as i64;
    let leftleft = n.ww as i64;
    let toprightright = n.nee as i64;
    match p {
        Predictor::Zero => 0,
        Predictor::Left => left,
        Predictor::Top => top,
        Predictor::Select => {
            // Select(left, top, topleft): pa = |top - topleft|,
            // pb = |left - topleft| — `pa < pb → left`.
            let pa = (top - topleft).abs();
            let pb = (left - topleft).abs();
            if pa < pb { left } else { top }
        }
        Predictor::Gradient => super::predictor::clamped_gradient(n.w, n.n, n.nw) as i64,
        Predictor::Weighted => wp_pred,
        Predictor::TopRight => topright,
        Predictor::TopLeft => topleft,
        Predictor::LeftLeft => leftleft,
        Predictor::Average0 => (left + top) / 2,
        Predictor::Average1 => (left + topleft) / 2,
        Predictor::Average2 => (topleft + top) / 2,
        Predictor::Average3 => (top + topright) / 2,
        Predictor::Average4 => {
            (6 * top - 2 * toptop + 7 * left + leftleft + toprightright + 3 * topright + 8) / 16
        }
    }
}

/// `GatherTreeData` — per-channel sample gathering with xorshift128+
/// subsampling (fixed seeds — NOT `Rng::new`). `group_id` is the stream id.
#[allow(clippy::too_many_arguments)]
fn gather_tree_data(
    channel: &Channel,
    chan: usize,
    group_id: u32,
    options: &LibjxlModularOptions,
    tree_samples: &mut TreeSamples,
    total_pixels: &mut usize,
) -> crate::error::Result<()> {
    let w = channel.width();
    let h = channel.height();
    let static_props = [chan as i32, group_id as i32];
    let mut properties = [0i32; NUM_NONREF_PROPERTIES];
    let mut pixel_fraction = options.nb_repeats.min(1.0f32) as f64;
    if pixel_fraction > 0.0 {
        pixel_fraction = pixel_fraction.max((1024.0 / (w * h) as f64).min(1.0));
    }
    let threshold = ((u64::MAX >> 32) as f64 * pixel_fraction) as u64;
    let mut s: [u64; 2] = [0x94D0_49BB_1331_11EB, 0xBF58_476D_1CE4_E5B9];
    let use_sample = |s: &mut [u64; 2]| -> bool {
        let mut s1 = s[0];
        let s0 = s[1];
        let bits = s1.wrapping_add(s0);
        s[0] = s0;
        s1 ^= s1 << 23;
        s1 ^= s0 ^ (s1 >> 18) ^ (s0 >> 5);
        s[1] = s1;
        (bits >> 32) <= threshold
    };

    let mut wp_state = WeightedPredictorState::with_defaults_and_budget(w, None)?;
    tree_samples.prepare_for_samples((pixel_fraction * (h * w) as f64 + 64.0) as usize);
    let multiple_predictors = tree_samples.num_predictors() != 1;

    for y in 0..h {
        // InitPropsRow
        properties[0] = static_props[0];
        properties[1] = static_props[1];
        properties[2] = y as i32;
        properties[9] = 0;
        let mut prev_gradient: i32 = 0;
        for x in 0..w {
            let n = Neighbors::gather(channel, x, y);
            let pixel = channel.get(x, y);
            // PredictLearn(All): kForceComputeProperties | kUseWP — the WP
            // predict writes wp_max_error into p[15]; errors update after.
            let (wp_pred, wp_max_error) = wp_state.predict_property_update(pixel, x, y, w, &n);
            super::tree_learn::compute_spec_properties_into(
                &mut properties,
                static_props[0] as u32,
                static_props[1] as u32,
                x,
                y,
                &n,
                prev_gradient,
                wp_max_error,
            );
            prev_gradient = properties[9];

            let mut predictions = [0i64; NUM_MODULAR_PREDICTORS];
            if multiple_predictors {
                for (i, slot) in predictions.iter_mut().enumerate() {
                    let pred = Predictor::from_id(i as u8).unwrap();
                    *slot = predict_one_w(pred, &n, wp_pred);
                }
            } else {
                let p = tree_samples.predictor_from_index(0);
                predictions[p as usize] = predict_one_w(p, &n, wp_pred);
            }

            *total_pixels += 1;
            if use_sample(&mut s) {
                tree_samples.add_sample(pixel as i64, &properties, &predictions);
            }
        }
    }
    Ok(())
}

// ────────────────────────────────────────────────────────────────────────
// The libjxl `Tree` (in-memory convention: lchild = `>` side).
// ────────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
pub(crate) struct JxlNode {
    /// -1 = leaf.
    pub(crate) property: i32,
    pub(crate) splitval: i32,
    /// `> splitval` side.
    pub(crate) lchild: u32,
    /// `<= splitval` side.
    pub(crate) rchild: u32,
    pub(crate) predictor: Predictor,
    pub(crate) predictor_offset: i64,
    pub(crate) multiplier: u32,
}

impl JxlNode {
    fn leaf(predictor: Predictor) -> Self {
        Self {
            property: -1,
            splitval: 0,
            lchild: 0,
            rchild: 0,
            predictor,
            predictor_offset: 0,
            multiplier: 1,
        }
    }
}

pub(crate) type JxlTree = Vec<JxlNode>;

/// `MakeSplitNode` — `lchild` gets the `>` (right-cost) leaf first.
#[allow(clippy::too_many_arguments)]
fn make_split_node(
    pos: usize,
    property: i32,
    splitval: i32,
    lpred: Predictor,
    loff: i64,
    rpred: Predictor,
    roff: i64,
    tree: &mut JxlTree,
) {
    tree[pos].lchild = tree.len() as u32;
    tree[pos].rchild = tree.len() as u32 + 1;
    tree[pos].splitval = splitval;
    tree[pos].property = property;
    let mut node = JxlNode::leaf(rpred);
    node.predictor_offset = roff;
    tree.push(node);
    let mut node = JxlNode::leaf(lpred);
    node.predictor_offset = loff;
    tree.push(node);
}

/// `SplitTreeSamples` — partition [begin, end) so [begin, pos) holds
/// `Property ≤ val` and [pos, end) holds `Property > val`.
fn split_tree_samples(
    tree_samples: &mut TreeSamples,
    is_static: bool,
    begin: usize,
    pos: usize,
    end: usize,
    prop: usize,
    val: u32,
) {
    let mut begin_pos = begin;
    let mut end_pos = pos;
    loop {
        while begin_pos < pos && tree_samples.property(is_static, prop, begin_pos) <= val as usize {
            begin_pos += 1;
        }
        while end_pos < end && tree_samples.property(is_static, prop, end_pos) > val as usize {
            end_pos += 1;
        }
        if begin_pos < pos && end_pos < end {
            tree_samples.swap(begin_pos, end_pos);
        }
        begin_pos += 1;
        end_pos += 1;
        if !(begin_pos < pos && end_pos < end) {
            break;
        }
    }
}

/// `StaticPropRange` — `[lo, hi)` bounds per static property.
type StaticPropRange = [[u32; 2]; NUM_STATIC_PROPS];

#[derive(Clone, Copy, Default)]
struct SplitInfo {
    prop: usize,
    val: u32,
    pos: usize,
    lcost: f32,
    rcost: f32,
    lpred: Predictor,
    rpred: Predictor,
}

impl SplitInfo {
    fn new() -> Self {
        Self {
            prop: 0,
            val: 0,
            pos: 0,
            lcost: f32::MAX,
            rcost: f32::MAX,
            lpred: Predictor::Zero,
            rpred: Predictor::Zero,
        }
    }
    fn cost(&self) -> f32 {
        self.lcost + self.rcost
    }
}

#[derive(Clone, Copy)]
struct CostInfo {
    cost: f32,
    extra_cost: f32,
    pred: Predictor,
}

impl CostInfo {
    fn new() -> Self {
        Self {
            cost: f32::MAX,
            extra_cost: 0.0,
            pred: Predictor::Zero,
        }
    }
    fn cost_total(&self) -> f32 {
        self.cost + self.extra_cost
    }
}

/// `FindBestSplit` — `mul_info` is always empty for VarDCT streams.
fn find_best_split(
    tree_samples: &mut TreeSamples,
    threshold: f32,
    initial_static_prop_range: StaticPropRange,
    fast_decode_multiplier: f32,
    tree: &mut JxlTree,
) {
    struct NodeInfo {
        pos: usize,
        begin: usize,
        end: usize,
        static_prop_range: StaticPropRange,
    }
    let mut nodes: Vec<NodeInfo> = Vec::new();
    nodes.push(NodeInfo {
        pos: 0,
        begin: 0,
        end: tree_samples.num_distinct_samples(),
        static_prop_range: initial_static_prop_range,
    });

    let num_predictors = tree_samples.num_predictors();
    let num_properties = tree_samples.num_properties();

    while let Some(NodeInfo {
        pos,
        begin,
        end,
        static_prop_range,
    }) = nodes.pop()
    {
        if begin == end {
            continue;
        }

        let mut best_split_static_constant = SplitInfo::new();
        let mut best_split_static = SplitInfo::new();
        let mut best_split_nonstatic = SplitInfo::new();
        let mut best_split_nowp = SplitInfo::new();

        // Maximum token in the range.
        let mut max_symbols = 0usize;
        for pred in 0..num_predictors {
            for i in begin..end {
                let tok = tree_samples.token(pred, i);
                max_symbols = max_symbols.max(tok + 1);
            }
        }
        max_symbols = padded(max_symbols);
        let mut counts = vec![0i32; max_symbols * num_predictors];
        let mut tot_extra_bits = vec![0usize; num_predictors];
        for pred in 0..num_predictors {
            let mut extra_bits = 0usize;
            let rtokens = tree_samples.rtokens(pred);
            for (offset, &rt) in rtokens[begin..end].iter().enumerate() {
                let i = begin + offset;
                let count = tree_samples.count(i);
                let eb = rt.nbits as usize * count;
                counts[pred * max_symbols + rt.tok as usize] += count as i32;
                extra_bits += eb;
            }
            tot_extra_bits[pred] = extra_bits;
        }

        let base_bits = {
            let pred = tree_samples.predictor_index(tree[pos].predictor);
            estimate_bits(&counts[pred * max_symbols..(pred + 1) * max_symbols])
                + tot_extra_bits[pred] as f32
        };

        let mut prop_value_used_count: Vec<i32> = Vec::new();
        let mut count_increase: Vec<i32> = Vec::new();
        let mut extra_bits_increase: Vec<usize> = Vec::new();
        let mut costs_l: Vec<CostInfo> = Vec::new();
        let mut costs_r: Vec<CostInfo> = Vec::new();
        let mut counts_above = vec![0i32; max_symbols];
        let mut counts_below = vec![0i32; max_symbols];

        // The lower the threshold, the higher the expected noisiness of
        // the estimate; discourage changing predictors.
        let change_pred_penalty = 800.0f32 / (100.0f32 + threshold);

        let mut prop = 0usize;
        while prop < num_properties && base_bits > threshold {
            costs_l.clear();
            costs_r.clear();
            let prop_size = tree_samples.num_property_values(prop);
            if extra_bits_increase.len() < prop_size {
                count_increase.resize(prop_size * max_symbols, 0);
                extra_bits_increase.resize(prop_size, 0);
            }
            prop_value_used_count.clear();
            prop_value_used_count.resize(prop_size, 0);

            let mut first_used = prop_size;
            let mut last_used = 0usize;

            if prop < tree_samples.num_static_props {
                for i in begin..end {
                    let p = tree_samples.property(true, prop, i);
                    prop_value_used_count[p] += 1;
                    last_used = last_used.max(p);
                    first_used = first_used.min(p);
                }
            } else {
                let prop_idx = prop - tree_samples.num_static_props;
                for i in begin..end {
                    let p = tree_samples.property(false, prop_idx, i);
                    prop_value_used_count[p] += 1;
                    last_used = last_used.max(p);
                    first_used = first_used.min(p);
                }
            }
            costs_l.resize(last_used - first_used, CostInfo::new());
            costs_r.resize(last_used - first_used, CostInfo::new());

            // For all predictors, compute the right and left costs of
            // each split.
            for pred in 0..num_predictors {
                let rtokens = tree_samples.rtokens(pred);
                // CollectExtraBitsIncrease
                if prop < tree_samples.num_static_props {
                    for (offset, &rt) in rtokens[begin..end].iter().enumerate() {
                        let i2 = begin + offset;
                        let cnt = tree_samples.count(i2);
                        let p = tree_samples.property(true, prop, i2);
                        let sym = rt.tok as usize;
                        let ebi = rt.nbits as usize * cnt;
                        count_increase[p * max_symbols + sym] += cnt as i32;
                        extra_bits_increase[p] += ebi;
                    }
                } else {
                    let prop_idx = prop - tree_samples.num_static_props;
                    for (offset, &rt) in rtokens[begin..end].iter().enumerate() {
                        let i2 = begin + offset;
                        let cnt = tree_samples.count(i2);
                        let p = tree_samples.property(false, prop_idx, i2);
                        let sym = rt.tok as usize;
                        let ebi = rt.nbits as usize * cnt;
                        count_increase[p * max_symbols + sym] += cnt as i32;
                        extra_bits_increase[p] += ebi;
                    }
                }
                counts_above.copy_from_slice(&counts[pred * max_symbols..(pred + 1) * max_symbols]);
                counts_below.fill(0);
                let mut extra_bits_below = 0usize;
                // Exclude last used: ensures neither side is empty.
                for i in first_used..last_used {
                    if prop_value_used_count[i] == 0 {
                        continue;
                    }
                    extra_bits_below += extra_bits_increase[i];
                    extra_bits_increase[i] = 0;
                    for sym in 0..max_symbols {
                        counts_above[sym] -= count_increase[i * max_symbols + sym];
                        counts_below[sym] += count_increase[i * max_symbols + sym];
                        count_increase[i * max_symbols + sym] = 0;
                    }
                    let rcost = estimate_bits(&counts_above) + tot_extra_bits[pred] as f32
                        - extra_bits_below as f32;
                    let lcost = estimate_bits(&counts_below) + extra_bits_below as f32;
                    let mut penalty = 0.0f32;
                    // Never discourage moving away from Weighted.
                    if tree_samples.predictor_from_index(pred) != tree[pos].predictor
                        && tree[pos].predictor != Predictor::Weighted
                    {
                        penalty = change_pred_penalty;
                    }
                    // Disfavour Weighted (slower) / favour Zero on ties.
                    if tree_samples.predictor_from_index(pred) == Predictor::Weighted {
                        penalty += 1e-8;
                    }
                    if tree_samples.predictor_from_index(pred) == Predictor::Zero {
                        penalty -= 1e-8;
                    }
                    if rcost + penalty < costs_r[i - first_used].cost_total() {
                        costs_r[i - first_used].cost = rcost;
                        costs_r[i - first_used].extra_cost = penalty;
                        costs_r[i - first_used].pred = tree_samples.predictor_from_index(pred);
                    }
                    if lcost + penalty < costs_l[i - first_used].cost_total() {
                        costs_l[i - first_used].cost = lcost;
                        costs_l[i - first_used].extra_cost = penalty;
                        costs_l[i - first_used].pred = tree_samples.predictor_from_index(pred);
                    }
                }
            }
            // Find the split minimizing the sum of side costs.
            let mut split = begin;
            for i in first_used..last_used {
                if prop_value_used_count[i] == 0 {
                    continue;
                }
                split += prop_value_used_count[i] as usize;
                let rcost = costs_r[i - first_used].cost;
                let lcost = costs_l[i - first_used].cost;

                let uses_wp = tree_samples.property_from_index(prop) == WP_PROP
                    || costs_l[i - first_used].pred == Predictor::Weighted
                    || costs_r[i - first_used].pred == Predictor::Weighted;
                let zero_entropy_side = rcost == 0.0 || lcost == 0.0;

                let is_static_prop =
                    tree_samples.property_from_index(prop) < NUM_STATIC_PROPS as u32;
                let best_ref = if is_static_prop {
                    if zero_entropy_side {
                        &mut best_split_static_constant
                    } else {
                        &mut best_split_static
                    }
                } else if uses_wp {
                    &mut best_split_nonstatic
                } else {
                    &mut best_split_nowp
                };
                if lcost + rcost < best_ref.cost() {
                    best_ref.prop = prop;
                    best_ref.val = i as u32;
                    best_ref.pos = split;
                    best_ref.lcost = lcost;
                    best_ref.lpred = costs_l[i - first_used].pred;
                    best_ref.rcost = rcost;
                    best_ref.rpred = costs_r[i - first_used].pred;
                }
            }
            // Clear for last_used.
            extra_bits_increase[last_used] = 0;
            for sym in 0..max_symbols {
                count_increase[last_used * max_symbols + sym] = 0;
            }
            prop += 1;
        }

        // Try to avoid introducing WP.
        let mut best = best_split_nonstatic;
        if best_split_nowp.cost() + threshold < base_bits
            && best_split_nowp.cost() <= fast_decode_multiplier * best.cost()
        {
            best = best_split_nowp;
        }
        // Split along static props if possible and not significantly more
        // expensive.
        if best_split_static.cost() + threshold < base_bits
            && best_split_static.cost() <= fast_decode_multiplier * best.cost()
        {
            best = best_split_static;
        }
        // Split along static props to create constant nodes if possible.
        if best_split_static_constant.cost() + threshold < base_bits {
            best = best_split_static_constant;
        }

        if best.cost() + threshold < base_bits {
            let p = tree_samples.property_from_index(best.prop);
            let dequant = tree_samples.unquantize_property(best.prop, best.val);
            make_split_node(pos, p as i32, dequant, best.lpred, 0, best.rpred, 0, tree);
            // "Sort" according to winning property.
            if best.prop < tree_samples.num_static_props {
                split_tree_samples(
                    tree_samples,
                    true,
                    begin,
                    best.pos,
                    end,
                    best.prop,
                    best.val,
                );
            } else {
                split_tree_samples(
                    tree_samples,
                    false,
                    begin,
                    best.pos,
                    end,
                    best.prop - tree_samples.num_static_props,
                    best.val,
                );
            }
            // Static-property range tracking (`mul_info` is always empty
            // for VarDCT so the ranges never get consulted, but the node
            // ordering contract must be preserved):
            //   rchild ([begin, pos)) gets the upper bound,
            //   lchild ([pos, end)) gets the lower bound.
            let mut new_sp_range = static_prop_range;
            if (p as usize) < NUM_STATIC_PROPS {
                new_sp_range[p as usize][1] = (dequant + 1) as u32;
            }
            nodes.push(NodeInfo {
                pos: tree[pos].rchild as usize,
                begin,
                end: best.pos,
                static_prop_range: new_sp_range,
            });
            let mut new_sp_range = static_prop_range;
            if (p as usize) < NUM_STATIC_PROPS {
                new_sp_range[p as usize][0] = (dequant + 1) as u32;
            }
            nodes.push(NodeInfo {
                pos: tree[pos].lchild as usize,
                begin: best.pos,
                end,
                static_prop_range: new_sp_range,
            });
        }
    }
}

// ────────────────────────────────────────────────────────────────────────
// `LearnTree` (per-chunk), `MergeTrees`, conversion, and orchestration.
// ────────────────────────────────────────────────────────────────────────

/// Inner `LearnTree` — `tree_samples` is already populated.
fn learn_tree_inner(
    tree_samples: &mut TreeSamples,
    total_pixels: usize,
    options: &LibjxlModularOptions,
    static_prop_range: StaticPropRange,
) -> JxlTree {
    let mut range = static_prop_range;
    for r in &mut range {
        if r[1] == 0 {
            r[1] = u32::MAX;
        }
    }
    if !tree_samples.has_samples() {
        return vec![JxlNode::leaf(tree_samples.predictor_from_index(0))];
    }
    let pixel_fraction = tree_samples.num_samples() as f32 / total_pixels as f32;
    let required_cost = pixel_fraction * 0.9 + 0.1;
    tree_samples.all_samples_done();
    let mut tree = vec![JxlNode::leaf(tree_samples.predictor_from_index(0))];
    find_best_split(
        tree_samples,
        options.node_threshold * required_cost,
        range,
        options.fast_decode_multiplier,
        &mut tree,
    );
    tree
}

/// Whether a `ModularImage` has pixels — `!image.empty()` = any channel
/// with nonzero w and h.
fn image_has_pixels(image: &ModularImage) -> bool {
    image
        .channels
        .iter()
        .any(|ch| ch.width() > 0 && ch.height() > 0)
}

/// `LearnTree(images, options, start, stop, {})` — learns the MA tree for
/// one `tree_splits_` chunk. `images` is the full stream image array;
/// `start..stop` are stream ids. All streams in the range must have
/// `tree_kind == kLearn`.
pub(crate) fn learn_tree(
    images: &[ModularImage],
    options: &[LibjxlModularOptions],
    start: u32,
    stop: u32,
) -> crate::error::Result<JxlTree> {
    let mut tree_samples = TreeSamples::new();
    tree_samples.set_predictor(
        options[start as usize].predictor,
        options[start as usize].wp_tree_mode,
    );
    tree_samples.set_properties(
        &options[start as usize].properties,
        options[start as usize].wp_tree_mode,
    );
    let mut max_c = 0usize;
    let mut pixel_samples: Vec<i32> = Vec::new();
    let mut diff_samples: Vec<i32> = Vec::new();
    let mut group_pixel_count: Vec<u32> = Vec::new();
    let mut channel_pixel_count: Vec<u32> = Vec::new();
    for i in start..stop {
        max_c = max_c.max(images[i as usize].channels.len());
        collect_pixel_samples(
            &images[i as usize],
            &options[i as usize],
            i as usize,
            &mut group_pixel_count,
            &mut channel_pixel_count,
            &mut pixel_samples,
            &mut diff_samples,
        );
    }
    let range: StaticPropRange = [[0, max_c as u32], [start, stop]];

    tree_samples.pre_quantize_properties(
        &group_pixel_count,
        &channel_pixel_count,
        &mut pixel_samples,
        &mut diff_samples,
        options[start as usize].max_property_values,
    );

    // `total_pixels` is the (clamped) pixel count of the FIRST stream's
    // channels only — GatherTreeData then accumulates every processed
    // pixel on top. Ported as-is.
    let mut total_pixels = 0usize;
    let first = &images[start as usize];
    for ch in &first.channels {
        if ch.width() > options[start as usize].max_chan_size
            || ch.height() > options[start as usize].max_chan_size
        {
            break;
        }
        total_pixels += ch.width() * ch.height();
    }
    total_pixels = total_pixels.max(1);

    for i in start..stop {
        let image = &images[i as usize];
        // `if (images[i].w == 0 || images[i].h == 0 || nb_channels < 1)`.
        if image.width() == 0 || image.height() == 0 || image.channels.is_empty() {
            continue;
        }
        for (c, ch) in image.channels.iter().enumerate() {
            if ch.width() > options[i as usize].max_chan_size
                || ch.height() > options[i as usize].max_chan_size
            {
                break;
            }
            if ch.width() == 0 || ch.height() == 0 {
                continue;
            }
            gather_tree_data(
                ch,
                c,
                i,
                &options[i as usize],
                &mut tree_samples,
                &mut total_pixels,
            )?;
        }
    }

    Ok(learn_tree_inner(
        &mut tree_samples,
        total_pixels,
        &options[start as usize],
        range,
    ))
}

/// `MergeTrees` — merges per-chunk trees under stream-id (property 1)
/// splits. `tree_splits` is the *useful* splits list (length = trees+1).
fn merge_trees(
    trees: &[JxlTree],
    tree_splits: &[usize],
    begin: usize,
    end: usize,
    out: &mut JxlTree,
) {
    debug_assert_eq!(trees.len() + 1, tree_splits.len());
    debug_assert!(end > begin);
    debug_assert!(end <= trees.len());
    if end == begin + 1 {
        // Insert the tree, offsetting child indices. Leaf ids become
        // wrong but TokenizeTree renumbers in BFS order.
        let sz = out.len();
        out.extend_from_slice(&trees[begin]);
        for node in &mut out[sz..] {
            node.lchild += sz as u32;
            node.rchild += sz as u32;
        }
        return;
    }
    let mid = (begin + end) / 2;
    let splitval = tree_splits[mid] - 1;
    let cur = out.len();
    out.push(JxlNode {
        property: 1, // stream_id
        splitval: splitval as i32,
        lchild: 0,
        rchild: 0,
        predictor: Predictor::Zero,
        predictor_offset: 0,
        multiplier: 1,
    });
    out[cur].lchild = out.len() as u32;
    merge_trees(trees, tree_splits, mid, end, out);
    out[cur].rchild = out.len() as u32;
    merge_trees(trees, tree_splits, begin, mid, out);
}

/// Convert the libjxl-convention tree (`lchild` = `>` side) into our
/// `Tree` (`lchild` = `≤` side, `rchild` = `>` side, emitted first in BFS)
/// and assign BFS-order context ids (matching `TokenizeTree` leaf
/// numbering).
pub(crate) fn to_property_tree(jxl: &JxlTree) -> Tree {
    let mut tree: Tree = jxl
        .iter()
        .map(|n| PropertyDecisionNode {
            property: n.property,
            splitval: n.splitval,
            predictor: n.predictor,
            predictor_offset: n.predictor_offset as i32,
            multiplier: n.multiplier as i32,
            lchild: n.rchild as usize, // ≤ side
            rchild: n.lchild as usize, // > side
            context_id: 0,
        })
        .collect();
    assign_sequential_contexts(&mut tree);
    tree
}

/// One VarDCT tree-learning chunk (a `tree_splits_` interval with pixels).
pub(crate) struct TreeChunk {
    /// Inclusive start / exclusive stop stream ids.
    pub start: u32,
    pub stop: u32,
}

/// `ModularFrameEncoder::ComputeTree` for the VarDCT streams: learns the
/// tree for each non-empty chunk and merges them under stream-id splits.
///
/// `images`/`options` are the full per-stream arrays; `chunks` are the
/// useful `[start, stop)` ranges in `tree_splits_` order.
///
/// Returns `None` when no stream has pixels (libjxl returns early and
/// keeps `tree_` empty).
pub(crate) fn compute_vardct_tree(
    images: &[ModularImage],
    options: &[LibjxlModularOptions],
    chunks: &[TreeChunk],
    num_streams: usize,
) -> crate::error::Result<Option<Tree>> {
    // `useful_splits`: keep only splits bounding a chunk with pixels.
    let mut useful_splits: Vec<usize> = Vec::with_capacity(chunks.len() + 1);
    for chunk in chunks {
        let mut has_pixels = false;
        for i in chunk.start..chunk.stop {
            if image_has_pixels(&images[i as usize]) {
                has_pixels = true;
            }
        }
        if has_pixels {
            useful_splits.push(chunk.start as usize);
        }
    }
    if useful_splits.is_empty() {
        return Ok(None);
    }
    useful_splits.push(num_streams);

    let mut trees: Vec<JxlTree> = Vec::with_capacity(useful_splits.len() - 1);
    for w in useful_splits.windows(2) {
        let (mut start, mut stop) = (w[0] as u32, w[1] as u32);
        while start < stop && !image_has_pixels(&images[start as usize]) {
            start += 1;
        }
        while start < stop && !image_has_pixels(&images[(stop - 1) as usize]) {
            stop -= 1;
        }
        trees.push(learn_tree(images, options, start, stop)?);
    }

    let mut merged = JxlTree::new();
    merge_trees(
        &trees,
        &useful_splits,
        0,
        useful_splits.len() - 1,
        &mut merged,
    );
    Ok(Some(to_property_tree(&merged)))
}

#[cfg(test)]
mod global_cost_tests {
    use super::*;

    #[test]
    fn global_palette_cost_matches_libjxl_v012() {
        for line in include_str!("../../../scripts/libjxl_estimate_cost_oracle/goldens.tsv").lines()
        {
            if line.starts_with('#') {
                continue;
            }
            let (name, expected) = line.split_once('\t').unwrap();
            let params: Vec<usize> = name.split(':').map(|v| v.parse().unwrap()).collect();
            let (w, h, mode, compact) = (params[0], params[1], params[2], params[3]);
            let mut state = 0x12345678u32;
            let mut data = Vec::new();
            for y in 0..h {
                for x in 0..w {
                    state = state.wrapping_mul(1103515245).wrapping_add(12345);
                    let r = state >> 16;
                    data.push(match mode {
                        0 => (r & 255) as i32,
                        1 => ((r & 1) * 255) as i32,
                        2 => (((x + y) % 16) * 17) as i32,
                        3 => {
                            if x == w - 1 && y == h - 1 {
                                255
                            } else {
                                ((x + y) % 15) as i32
                            }
                        }
                        4 => ((r & 255) * 257) as i32 - 32768,
                        5 => state as i32,
                        _ => unreachable!(),
                    });
                }
            }
            let mut channels = Vec::new();
            if compact != 0 {
                let mut palette = data.clone();
                palette.sort_unstable();
                palette.dedup();
                for value in &mut data {
                    *value = palette.binary_search(value).unwrap() as i32;
                }
                channels.push(Channel::from_vec(palette.clone(), palette.len(), 1).unwrap());
            }
            channels.push(Channel::from_vec(data, w, h).unwrap());
            channels.push(Channel::from_vec(vec![17, 0, 65535], 3, 1).unwrap());
            assert_eq!(
                estimate_global_image_cost(&channels),
                expected.parse::<f32>().unwrap(),
                "{name}"
            );
        }
    }
}
