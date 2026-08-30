//! Config enumeration + **computed unique-configs** for parameter sweeps
//! (`__expert`).
//!
//! A sweep varies a handful of knobs over candidate values and encodes the
//! cartesian product. Many knob combinations resolve to the **same**
//! effective [`EffortProfile`] — a knob that is a no-op at the chosen
//! effort, two override sets that collapse to the same schedule, etc. —
//! and those produce byte-identical encodes (modulo the image). Encoding
//! them more than once is wasted compute.
//!
//! This module computes the *unique* set: it resolves every candidate to an
//! [`EffortProfile`], fingerprints it, and deduplicates. The fingerprint is
//! the dedup primitive ([`EffortProfile::fingerprint`]); the
//! [`LossySweep`] / [`LosslessSweep`] grids are the convenience layer that
//! enumerates a product of [closure axes](LossySweep::axis) and returns one
//! [`UniqueLossyConfig`] / [`UniqueLosslessConfig`] per distinct resolved
//! profile, first-seen order preserved.
//!
//! The sweep surface is [`LossyInternalParams`] / [`LosslessInternalParams`]
//! (the full sweepable struct — every cost-model, AC-strategy, and
//! modular-tree knob is reachable through it). Resolution mirrors the encoder exactly: a base
//! schedule from `(effort, mode)` with the sparse params applied on top
//! (`apply_to`), so a unique profile here is a unique encode in production.
//!
//! Requires the `__expert` cargo feature. Not part of the stable API.

#[cfg(feature = "__expert")]
use alloc::boxed::Box;
use alloc::collections::BTreeSet;
use alloc::vec::Vec;
use core::hash::{Hash, Hasher};

#[cfg(feature = "__expert")]
use crate::api::EncoderMode;
use crate::effort::EffortProfile;
#[cfg(feature = "__expert")]
use crate::effort::{LosslessInternalParams, LossyInternalParams};

/// Tiny no_std FNV-1a hasher used for the profile fingerprint. We hash field
/// bytes ourselves (f32 via bit pattern, enums via discriminant) so the
/// fingerprint is stable and total-order-free — only equality matters for
/// dedup.
struct Fnv1a(u64);

impl Fnv1a {
    fn new() -> Self {
        Fnv1a(0xcbf2_9ce4_8422_2325)
    }
}

impl Hasher for Fnv1a {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 ^= b as u64;
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
}

impl EffortProfile {
    /// Stable content fingerprint over **every** field — `f32` by bit
    /// pattern, enums by discriminant — for deduping resolved configs in a
    /// sweep. Two profiles with equal fingerprints encode identically
    /// (modulo the image); collision probability is negligible for the
    /// candidate counts a sweep produces.
    ///
    /// Requires the `__expert` cargo feature. (The crate-internal
    /// [`Self::fingerprint_impl`] is always compiled — the e11+
    /// TectonicPlate schedule dedups with it.)
    #[cfg(feature = "__expert")]
    #[must_use]
    pub fn fingerprint(&self) -> u64 {
        self.fingerprint_impl()
    }

    /// Always-compiled body of [`Self::fingerprint`] (see there).
    #[must_use]
    pub(crate) fn fingerprint_impl(&self) -> u64 {
        let mut h = Fnv1a::new();

        // All bool fields (35).
        [
            self.use_ans,
            self.optimize_codes,
            self.custom_orders,
            self.gaborish,
            self.pixel_domain_loss,
            self.error_diffusion,
            self.patches,
            self.tree_learning,
            self.lz77,
            self.ac_strategy_enabled,
            self.try_dct16,
            self.try_dct32,
            self.try_dct64,
            self.try_dct4x8_afv,
            self.non_aligned_eval,
            self.chromacity_adjustment,
            self.enhanced_clustering_vardct,
            self.optimize_uint_configs_vardct,
            self.epf_dynamic_sharpness,
            self.cfl_two_pass,
            self.cfl_newton,
            self.cfl_newton_libjxl_parity,
            self.cfl_newton_libjxl_math_with_ls_warm_start,
            self.cfl_pass1_screenshot_x0_start,
            self.cfl_pass2_ls_at_low_effort,
            self.cfl_zero_for_search,
            self.use_adaptive_quant,
            self.adjust_quant_ac,
            self.use_libjxl_wp_dc_quant,
            self.patch_ref_tree_learning,
            self.use_streaming_dedup,
            self.gather_dedup,
            self.gather_dedup_phase3,
            self.tree_parallel_small_image_fallback,
            self.lloyd_max_buckets,
        ]
        .hash(&mut h);

        // u8 (8), u16 (1), u32 (3), usize (3).
        [
            self.effort,
            self.fine_grained_step,
            self.extra_dc_precision,
            self.nb_rcts_to_try,
            self.wp_num_param_sets,
            self.tree_num_properties,
            self.tree_learn_seeds,
            self.lossy_search_seeds,
        ]
        .hash(&mut h);
        self.tree_max_buckets.hash(&mut h);
        [
            self.butteraugli_iters,
            self.tree_parallel_max_depth,
            self.tree_max_samples_fixed,
        ]
        .hash(&mut h);
        [
            self.cfl_newton_max_iters,
            self.tree_parallel_floor,
            self.tree_parallel_root_threshold,
        ]
        .hash(&mut h);

        // Every f32 by bit pattern: 10 scalar + 4+4 thresholds + 5×3 cost
        // tuples (k8x8..k4x4) + 12 entropy_mul_table.
        let e = &self.entropy_mul_table;
        [
            self.cfl_newton_eps,
            self.initial_q_numerator,
            self.k_favor_2x2,
            self.k_avoid_transforms_base,
            self.k_info_loss_mul_base,
            self.k_zeros_mul_base,
            self.k_cost_delta_base,
            self.k_ac_quant,
            self.tree_threshold_base,
            self.tree_sample_fraction,
            self.fixed_thresholds_y[0],
            self.fixed_thresholds_y[1],
            self.fixed_thresholds_y[2],
            self.fixed_thresholds_y[3],
            self.adjust_thresholds[0],
            self.adjust_thresholds[1],
            self.adjust_thresholds[2],
            self.adjust_thresholds[3],
            self.k8x8.0,
            self.k8x8.1,
            self.k8x8.2,
            self.k16x8.0,
            self.k16x8.1,
            self.k16x8.2,
            self.k16x16.0,
            self.k16x16.1,
            self.k16x16.2,
            self.k4x8.0,
            self.k4x8.1,
            self.k4x8.2,
            self.k4x4.0,
            self.k4x4.1,
            self.k4x4.2,
            e.dct8,
            e.dct4x4,
            e.dct4x8,
            e.identity,
            e.dct2x2,
            e.afv,
            e.dct16x8,
            e.dct16x16,
            e.dct16x32,
            e.dct32x32,
            e.dct64x32,
            e.dct64x64,
        ]
        .map(f32::to_bits)
        .hash(&mut h);

        // Enums via discriminant / inner tag.
        core::mem::discriminant(&self.lz77_method).hash(&mut h);
        core::mem::discriminant(&self.ans_histogram_strategy_vardct).hash(&mut h);
        core::mem::discriminant(&self.forced_rct).hash(&mut h);
        if let Some(rct) = &self.forced_rct {
            // RctType is a `struct RctType(pub u8)` newtype — hash the tag.
            rct.0.hash(&mut h);
        }

        h.finish()
    }
}

#[cfg(feature = "__expert")]
/// One unique resolved lossy config from a sweep.
pub struct UniqueLossyConfig {
    /// The sparse override params that first produced this profile.
    pub params: LossyInternalParams,
    /// The resolved effective profile the encoder would consume.
    pub profile: EffortProfile,
    /// `profile.fingerprint()` — the dedup key.
    pub fingerprint: u64,
}

#[cfg(feature = "__expert")]
/// One unique resolved lossless config from a sweep.
pub struct UniqueLosslessConfig {
    /// The sparse override params that first produced this profile.
    pub params: LosslessInternalParams,
    /// The resolved effective profile the encoder would consume.
    pub profile: EffortProfile,
    /// `profile.fingerprint()` — the dedup key.
    pub fingerprint: u64,
}

/// Resolve each candidate against the `(effort, mode)` lossy schedule and
/// return the **unique** effective configs, deduplicated by fingerprint,
/// first-seen order preserved. The low-level primitive behind [`LossySweep`].
#[cfg(feature = "__expert")]
pub fn unique_lossy_configs(
    effort: u8,
    mode: EncoderMode,
    candidates: impl IntoIterator<Item = LossyInternalParams>,
) -> Vec<UniqueLossyConfig> {
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for params in candidates {
        let mut profile = EffortProfile::lossy(effort, mode);
        params.clone().apply_to(&mut profile);
        let fingerprint = profile.fingerprint();
        if seen.insert(fingerprint) {
            out.push(UniqueLossyConfig {
                params,
                profile,
                fingerprint,
            });
        }
    }
    out
}

/// Lossless counterpart of [`unique_lossy_configs`].
#[cfg(feature = "__expert")]
pub fn unique_lossless_configs(
    effort: u8,
    mode: EncoderMode,
    candidates: impl IntoIterator<Item = LosslessInternalParams>,
) -> Vec<UniqueLosslessConfig> {
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for params in candidates {
        let mut profile = EffortProfile::lossless(effort, mode);
        params.clone().apply_to(&mut profile);
        let fingerprint = profile.fingerprint();
        if seen.insert(fingerprint) {
            out.push(UniqueLosslessConfig {
                params,
                profile,
                fingerprint,
            });
        }
    }
    out
}

/// A mutator that sets one knob on a [`LossyInternalParams`] to one of the
/// candidate values being swept on an axis.
#[cfg(feature = "__expert")]
type LossyMutator = Box<dyn Fn(&mut LossyInternalParams)>;
#[cfg(feature = "__expert")]
/// Lossless counterpart of [`LossyMutator`].
type LosslessMutator = Box<dyn Fn(&mut LosslessInternalParams)>;

/// A lossy sweep grid: a base `(effort, mode)` plus a set of independent
/// axes. Each axis is a list of [mutators](LossyMutator), one per candidate
/// value on that axis. [`Self::unique`] enumerates the cartesian product,
/// resolves each combination, and returns the deduplicated unique configs.
///
/// ```ignore
/// use jxl_encoder::effort::{EncoderMode, Lz77Method};
/// use jxl_encoder::sweep::LossySweep;
/// let uniques = LossySweep::new(9, EncoderMode::Reference)
///     .axis(vec![
///         Box::new(|p| p.try_dct64 = Some(true)),
///         Box::new(|p| p.try_dct64 = Some(false)),
///     ])
///     .axis(vec![
///         Box::new(|p| p.k_ac_quant = Some(0.765)),
///         Box::new(|p| p.k_ac_quant = Some(0.65)),
///     ])
///     .unique();
/// // `uniques.len()` <= 4 — combinations that resolve identically collapse.
/// ```
#[cfg(feature = "__expert")]
pub struct LossySweep {
    effort: u8,
    mode: EncoderMode,
    axes: Vec<Vec<LossyMutator>>,
}

#[cfg(feature = "__expert")]
impl LossySweep {
    /// New grid over the `(effort, mode)` lossy schedule with no axes.
    #[must_use]
    pub fn new(effort: u8, mode: EncoderMode) -> Self {
        Self {
            effort,
            mode,
            axes: Vec::new(),
        }
    }

    /// Add an axis: one mutator per candidate value. An empty axis is
    /// ignored (it would zero the product).
    #[must_use]
    pub fn axis(mut self, values: Vec<LossyMutator>) -> Self {
        if !values.is_empty() {
            self.axes.push(values);
        }
        self
    }

    /// Total cartesian-product combinations (before dedup) = the product of
    /// the axis lengths (`1` when there are no axes).
    #[must_use]
    pub fn total_combinations(&self) -> usize {
        self.axes.iter().map(Vec::len).product::<usize>().max(1)
    }

    /// Enumerate the product, resolve each combination, and return the
    /// unique effective configs (deduplicated by fingerprint).
    #[must_use]
    pub fn unique(&self) -> Vec<UniqueLossyConfig> {
        let mut seen = BTreeSet::new();
        let mut out = Vec::new();
        for idx in 0..self.total_combinations() {
            let mut params = LossyInternalParams::default();
            // Mixed-radix decode of `idx` across the axes.
            let mut rem = idx;
            for axis in &self.axes {
                let pick = rem % axis.len();
                rem /= axis.len();
                axis[pick](&mut params);
            }
            let mut profile = EffortProfile::lossy(self.effort, self.mode);
            params.clone().apply_to(&mut profile);
            let fingerprint = profile.fingerprint();
            if seen.insert(fingerprint) {
                out.push(UniqueLossyConfig {
                    params,
                    profile,
                    fingerprint,
                });
            }
        }
        out
    }
}

#[cfg(feature = "__expert")]
/// Lossless counterpart of [`LossySweep`].
pub struct LosslessSweep {
    effort: u8,
    mode: EncoderMode,
    axes: Vec<Vec<LosslessMutator>>,
}

#[cfg(feature = "__expert")]
impl LosslessSweep {
    /// New grid over the `(effort, mode)` lossless schedule with no axes.
    #[must_use]
    pub fn new(effort: u8, mode: EncoderMode) -> Self {
        Self {
            effort,
            mode,
            axes: Vec::new(),
        }
    }

    /// Add an axis: one mutator per candidate value.
    #[must_use]
    pub fn axis(mut self, values: Vec<LosslessMutator>) -> Self {
        if !values.is_empty() {
            self.axes.push(values);
        }
        self
    }

    /// Total cartesian-product combinations (before dedup).
    #[must_use]
    pub fn total_combinations(&self) -> usize {
        self.axes.iter().map(Vec::len).product::<usize>().max(1)
    }

    /// Enumerate the product and return the unique effective configs.
    #[must_use]
    pub fn unique(&self) -> Vec<UniqueLosslessConfig> {
        let mut seen = BTreeSet::new();
        let mut out = Vec::new();
        for idx in 0..self.total_combinations() {
            let mut params = LosslessInternalParams::default();
            let mut rem = idx;
            for axis in &self.axes {
                let pick = rem % axis.len();
                rem /= axis.len();
                axis[pick](&mut params);
            }
            let mut profile = EffortProfile::lossless(self.effort, self.mode);
            params.clone().apply_to(&mut profile);
            let fingerprint = profile.fingerprint();
            if seen.insert(fingerprint) {
                out.push(UniqueLosslessConfig {
                    params,
                    profile,
                    fingerprint,
                });
            }
        }
        out
    }
}

#[cfg(all(test, feature = "__expert"))]
mod tests {
    use super::*;

    #[test]
    fn fingerprint_stable_and_distinguishing() {
        let a = EffortProfile::lossy(7, EncoderMode::Reference);
        let b = EffortProfile::lossy(7, EncoderMode::Reference);
        assert_eq!(a.fingerprint(), b.fingerprint(), "same profile -> same fp");
        let c = EffortProfile::lossy(5, EncoderMode::Reference);
        assert_ne!(
            a.fingerprint(),
            c.fingerprint(),
            "different effort -> different fp"
        );
        // f32 knob difference must change the fingerprint.
        let mut d = EffortProfile::lossy(7, EncoderMode::Reference);
        d.k_ac_quant += 0.01;
        assert_ne!(a.fingerprint(), d.fingerprint(), "f32 knob -> different fp");
    }

    #[test]
    fn sweep_dedups_noop_combinations() {
        // try_dct64 ∈ {true, false} crossed with k_ac_quant ∈ {0.765,
        // 0.765} (the second axis value is identical). The product is
        // 2×2 = 4, but the identical k_ac_quant axis MUST collapse, so
        // the unique set is strictly smaller than the product.
        let grid = LossySweep::new(9, EncoderMode::Reference)
            .axis(vec![
                Box::new(|p| p.try_dct64 = Some(true)),
                Box::new(|p| p.try_dct64 = Some(false)),
            ])
            .axis(vec![
                Box::new(|p| p.k_ac_quant = Some(0.765)),
                Box::new(|p| p.k_ac_quant = Some(0.765)),
            ]);
        let total = grid.total_combinations();
        let uniques = grid.unique();
        assert_eq!(total, 4);
        assert!(
            uniques.len() < total,
            "identical k_ac_quant axis must collapse ({} of {})",
            uniques.len(),
            total
        );
        assert_eq!(
            uniques.len(),
            2,
            "try_dct64 true vs false are the only distinct profiles"
        );
    }

    #[test]
    fn unique_lossy_configs_dedups() {
        let same = LossyInternalParams {
            try_dct64: Some(true),
            ..Default::default()
        };
        let also_same = LossyInternalParams {
            try_dct64: Some(true),
            ..Default::default()
        };
        let diff = LossyInternalParams {
            try_dct64: Some(false),
            ..Default::default()
        };
        let uniques = unique_lossy_configs(9, EncoderMode::Reference, [same, also_same, diff]);
        assert_eq!(uniques.len(), 2);
    }
}

// ─── TectonicPlate e11+ lossless trial schedule (issue #45) ─────────────────
//
// libjxl e11 (`SpeedTier::kTectonicPlate`, expert-gated, lossless-only)
// re-encodes the frame under ~22-26 whole-frame configurations — modular
// header / transform knobs at kGlacier search effort — and keeps the
// smallest (`enc_frame.cc:2576-2643` probe pair + branch,
// `TectonicPlateSettingsLessPalette` `:2363` / `...MorePalette` `:2471`).
// Our e11+ lossless schedule ports that trial set on top of the shifted
// extended-tier extras: trials run at e10 (the kGlacier analogue), the
// winning config is re-encoded at the ambient tier's full profile
// (e11: 2-seed tree learn; e12/e13: 16-seed), and the smallest stream
// overall wins. Consumed by `api::EncodeRequest::encode_lossless`.

/// One TectonicPlate trial configuration, transcribed 1:1 from the
/// libjxl `CompressParams` mutations (field names on the right of each
/// mapping are libjxl's):
///
/// - `palette_colors` ← `palette_colors` (0 disables)
/// - `channel_colors_group_percent` ← `channel_colors_percent`
/// - `channel_colors_global_percent` ← `channel_colors_pre_transform_percent`
/// - `group_size_shift` ← `modular_group_size_shift`
/// - `predictor` ← `options.predictor`, as our `-P` id
///   (`None` = tree-driven per-leaf ID3, ⊇ libjxl `Variable`/`Best`)
/// - `wp_no_wp` ← `options.wp_tree_mode == kNoWP` — transcribed but NOT
///   yet wired (we have no tree-level WP-exclusion knob; configs that
///   differ only here collapse in dedup — follow-up on #45)
/// - `patches` ← `patches != Override::kOff`
/// - `nb_repeats` ← `options.nb_repeats` (mapped to
///   `tree_sample_fraction = min(1.0, 1.3 × nb_repeats)` — our e9+
///   default 0.65 corresponds to libjxl's 0.5, so the ratio is
///   preserved; `0.0` maps to fraction `0.0`, where the gather floor +
///   the forced single-leaf `predictor = Zero` reproduce libjxl's
///   no-tree outcome)
/// - `nb_prev_channels` ← `options.max_properties` (always 4 in the
///   libjxl lists)
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct TectonicConfig {
    pub palette_colors: i64,
    pub channel_colors_group_percent: f32,
    pub channel_colors_global_percent: f32,
    pub group_size_shift: u8,
    /// Our `-P` id: `Some(0)` Zero, `Some(4)` Select, `Some(6)`
    /// Weighted; `None` = per-leaf ID3 (covers libjxl `Variable` and
    /// supersets `Best`, whose candidate set {Gradient, Weighted} is a
    /// subset of ID3's 14).
    pub predictor: Option<u8>,
    pub wp_no_wp: bool,
    pub patches: bool,
    pub nb_repeats: f32,
    pub nb_prev_channels: i32,
}

impl TectonicConfig {
    /// The `tree_sample_fraction` this config's `nb_repeats` maps to.
    pub(crate) fn tree_sample_fraction(&self) -> f32 {
        (1.3 * self.nb_repeats).min(1.0)
    }

    /// Dedup key: every wired knob, f32s by bit pattern. `wp_no_wp` is
    /// deliberately EXCLUDED while unwired so wp-only siblings collapse.
    pub(crate) fn dedup_key(&self) -> (i64, u32, u32, u8, Option<u8>, bool, u32, i32) {
        (
            self.palette_colors,
            self.channel_colors_group_percent.to_bits(),
            self.channel_colors_global_percent.to_bits(),
            self.group_size_shift,
            self.predictor,
            self.patches,
            self.tree_sample_fraction().to_bits(),
            self.nb_prev_channels,
        )
    }
}

/// libjxl predictor names used by the transcription below.
const P_VARIABLE: Option<u8> = None; // Variable → per-leaf ID3
const P_ZERO: Option<u8> = Some(0);
const P_SELECT: Option<u8> = Some(4);
const P_WEIGHTED: Option<u8> = Some(6);
/// libjxl `Best` = tree learn over {Gradient, Weighted} only; our ID3
/// evaluates all 14 candidates per leaf, a strict superset — map to ID3.
const P_BEST: Option<u8> = None;

/// The two probe encodes (`enc_frame.cc:2577-2597`): a no-palette and a
/// max-palette configuration whose sizes pick the branch below.
pub(crate) fn tectonic_probe_pair() -> [TectonicConfig; 2] {
    let mut a = TectonicConfig {
        palette_colors: 0,
        channel_colors_group_percent: 80.0,
        channel_colors_global_percent: 95.0,
        group_size_shift: 3,
        predictor: P_VARIABLE,
        wp_no_wp: false,
        patches: true,
        nb_repeats: 1.0,
        nb_prev_channels: 4,
    };
    let probe_a = a;
    a.predictor = P_ZERO;
    a.nb_repeats = 0.01;
    a.palette_colors = 70000;
    a.patches = false;
    a.wp_no_wp = true;
    [probe_a, a]
}

/// `TectonicPlateSettingsLessPalette` (`enc_frame.cc:2363-2470`),
/// transcribed mutation-for-mutation (24 configs).
pub(crate) fn tectonic_less_palette() -> Vec<TectonicConfig> {
    let mut v = Vec::with_capacity(24);
    let mut c = TectonicConfig {
        palette_colors: 1024,
        channel_colors_group_percent: 0.0,
        channel_colors_global_percent: 95.0,
        group_size_shift: 0,
        predictor: P_VARIABLE,
        wp_no_wp: false,
        patches: true,
        nb_repeats: 1.0,
        nb_prev_channels: 4,
    };
    v.push(c); // 1
    c.channel_colors_group_percent = 80.0;
    c.group_size_shift = 1;
    c.palette_colors = 0;
    c.channel_colors_global_percent = 0.0;
    v.push(c); // 2
    c.channel_colors_global_percent = 95.0;
    c.group_size_shift = 2;
    v.push(c); // 3
    c.group_size_shift = 3;
    c.patches = false;
    c.wp_no_wp = true;
    v.push(c); // 4
    c.palette_colors = 1024;
    c.wp_no_wp = false;
    v.push(c); // 5
    c.patches = true;
    c.wp_no_wp = true;
    v.push(c); // 6
    c.wp_no_wp = false;
    c.channel_colors_global_percent = 0.0;
    v.push(c); // 7
    c.channel_colors_global_percent = 95.0;
    c.nb_repeats = 0.9;
    c.group_size_shift = 2;
    v.push(c); // 8
    c.group_size_shift = 3;
    c.palette_colors = 0;
    c.wp_no_wp = true;
    v.push(c); // 9
    c.wp_no_wp = false;
    c.channel_colors_global_percent = 0.0;
    v.push(c); // 10
    c.palette_colors = 1024;
    c.nb_repeats = 0.95;
    c.group_size_shift = 1;
    c.channel_colors_group_percent = 0.0;
    v.push(c); // 11
    c.group_size_shift = 2;
    c.palette_colors = 0;
    v.push(c); // 12
    c.channel_colors_group_percent = 80.0;
    c.wp_no_wp = true;
    v.push(c); // 13
    c.palette_colors = 1024;
    c.channel_colors_global_percent = 95.0;
    c.wp_no_wp = false;
    c.group_size_shift = 3;
    v.push(c); // 14
    c.palette_colors = 0;
    c.patches = false;
    v.push(c); // 15
    c.patches = true;
    c.wp_no_wp = true;
    v.push(c); // 16
    c.palette_colors = 1024;
    c.patches = false;
    v.push(c); // 17
    c.nb_repeats = 0.5;
    c.patches = true;
    c.wp_no_wp = false;
    v.push(c); // 18
    c.predictor = P_ZERO;
    c.nb_repeats = 0.0;
    c.channel_colors_group_percent = 0.0;
    c.channel_colors_global_percent = 0.0;
    c.patches = false;
    v.push(c); // 19
    c.channel_colors_group_percent = 80.0;
    c.channel_colors_global_percent = 95.0;
    c.nb_repeats = 1.0;
    c.palette_colors = 0;
    v.push(c); // 20
    c.patches = true;
    c.predictor = P_BEST;
    v.push(c); // 21
    c.nb_repeats = 0.9;
    c.patches = false;
    v.push(c); // 22
    c.palette_colors = 1024;
    c.patches = true;
    c.predictor = P_WEIGHTED;
    c.nb_repeats = 1.0;
    v.push(c); // 23
    c.nb_repeats = 0.95;
    c.group_size_shift = 2;
    c.palette_colors = 0;
    c.channel_colors_global_percent = 0.0;
    v.push(c); // 24
    v
}

/// `TectonicPlateSettingsMorePalette` (`enc_frame.cc:2471-2560`),
/// transcribed mutation-for-mutation (20 configs).
pub(crate) fn tectonic_more_palette() -> Vec<TectonicConfig> {
    let mut v = Vec::with_capacity(20);
    let mut c = TectonicConfig {
        palette_colors: 70000,
        channel_colors_group_percent: 80.0,
        channel_colors_global_percent: 95.0,
        group_size_shift: 0,
        predictor: P_VARIABLE,
        wp_no_wp: false,
        patches: true,
        nb_repeats: 1.0,
        nb_prev_channels: 4,
    };
    v.push(c); // 1
    c.group_size_shift = 2;
    c.channel_colors_group_percent = 0.0;
    c.patches = false;
    c.wp_no_wp = true;
    v.push(c); // 2
    c.channel_colors_group_percent = 80.0;
    c.wp_no_wp = false;
    c.group_size_shift = 3;
    v.push(c); // 3
    c.nb_repeats = 0.9;
    v.push(c); // 4
    c.patches = true;
    c.nb_repeats = 0.95;
    c.group_size_shift = 0;
    v.push(c); // 5
    c.group_size_shift = 3;
    v.push(c); // 6
    c.patches = false;
    c.wp_no_wp = true;
    v.push(c); // 7
    c.nb_repeats = 0.5;
    v.push(c); // 8
    c.wp_no_wp = false;
    c.predictor = P_ZERO;
    c.nb_repeats = 0.0;
    v.push(c); // 9
    c.patches = true;
    c.channel_colors_global_percent = 0.0;
    v.push(c); // 10
    c.nb_repeats = 0.01;
    c.palette_colors = 0;
    c.patches = false;
    c.wp_no_wp = true;
    v.push(c); // 11
    c.channel_colors_global_percent = 95.0;
    c.wp_no_wp = false;
    c.palette_colors = 70000;
    v.push(c); // 12
    c.nb_repeats = 1.0;
    c.group_size_shift = 0;
    c.channel_colors_group_percent = 0.0;
    c.channel_colors_global_percent = 0.0;
    c.wp_no_wp = true;
    v.push(c); // 13
    c.channel_colors_global_percent = 95.0;
    c.group_size_shift = 1;
    v.push(c); // 14
    c.group_size_shift = 2;
    v.push(c); // 15
    c.channel_colors_group_percent = 80.0;
    c.wp_no_wp = false;
    c.group_size_shift = 3;
    v.push(c); // 16
    c.nb_repeats = 0.5;
    c.group_size_shift = 1;
    c.channel_colors_group_percent = 0.0;
    c.wp_no_wp = true;
    v.push(c); // 17
    c.wp_no_wp = false;
    c.group_size_shift = 2;
    v.push(c); // 18
    c.channel_colors_group_percent = 80.0;
    c.group_size_shift = 3;
    c.wp_no_wp = true;
    v.push(c); // 19
    c.wp_no_wp = false;
    c.predictor = P_SELECT;
    c.nb_repeats = 1.0;
    v.push(c); // 20
    v
}

/// Deduplicate a trial list on the wired-knob key, preserving first-seen
/// order. Configs that differ only in the not-yet-wired `wp_no_wp` axis
/// collapse here (the "wire the unique-config enumerator in" half of
/// issue #45 pick #3).
pub(crate) fn dedup_tectonic(configs: Vec<TectonicConfig>) -> Vec<TectonicConfig> {
    let mut seen = BTreeSet::new();
    let mut out = Vec::with_capacity(configs.len());
    for c in configs {
        if seen.insert(c.dedup_key()) {
            out.push(c);
        }
    }
    out
}

#[cfg(test)]
mod tectonic_tests {
    use super::*;

    #[test]
    fn list_lengths_match_libjxl() {
        assert_eq!(tectonic_probe_pair().len(), 2);
        assert_eq!(
            tectonic_less_palette().len(),
            24,
            "enc_frame.cc:2363 pushes 24"
        );
        assert_eq!(
            tectonic_more_palette().len(),
            20,
            "enc_frame.cc:2471 pushes 20"
        );
    }

    #[test]
    fn probe_pair_matches_libjxl_values() {
        let [a, b] = tectonic_probe_pair();
        assert_eq!(
            (a.palette_colors, a.group_size_shift, a.predictor, a.patches),
            (0, 3, P_VARIABLE, true)
        );
        assert_eq!(a.nb_repeats, 1.0);
        assert_eq!(
            (b.palette_colors, b.predictor, b.patches, b.wp_no_wp),
            (70000, P_ZERO, false, true)
        );
        assert_eq!(b.nb_repeats, 0.01);
        // Carried fields.
        assert_eq!(b.group_size_shift, 3);
        assert_eq!(b.channel_colors_group_percent, 80.0);
        assert_eq!(b.channel_colors_global_percent, 95.0);
    }

    #[test]
    fn spot_check_transcription() {
        let less = tectonic_less_palette();
        // #4: gss=3, patches off, NoWP, palette 0 (post-#2 palette drop).
        assert_eq!(less[3].group_size_shift, 3);
        assert!(!less[3].patches && less[3].wp_no_wp);
        assert_eq!(less[3].palette_colors, 0);
        // #19: the Zero/no-tree config.
        assert_eq!(less[18].predictor, P_ZERO);
        assert_eq!(less[18].nb_repeats, 0.0);
        assert_eq!(less[18].tree_sample_fraction(), 0.0);
        assert!(!less[18].patches);
        // #23: Weighted with palette back on.
        assert_eq!(less[22].predictor, P_WEIGHTED);
        assert_eq!(less[22].palette_colors, 1024);
        let more = tectonic_more_palette();
        // MorePalette #20: Select, nb 1.0, from the NoWP-cleared state.
        assert_eq!(more[19].predictor, P_SELECT);
        assert_eq!(more[19].nb_repeats, 1.0);
        assert!(!more[19].wp_no_wp);
        // Every MorePalette config keeps max_properties=4.
        assert!(more.iter().all(|c| c.nb_prev_channels == 4));
    }

    #[test]
    fn nb_repeats_fraction_mapping() {
        let f = |nb: f32| {
            TectonicConfig {
                nb_repeats: nb,
                ..tectonic_probe_pair()[0]
            }
            .tree_sample_fraction()
        };
        // Ratio-preserving 1.3× map: libjxl e9+ default 0.5 → our 0.65.
        assert_eq!(f(0.5), 0.65);
        assert_eq!(f(1.0), 1.0);
        assert_eq!(f(0.0), 0.0);
        assert!((f(0.01) - 0.013).abs() < 1e-6);
        // 0.95 / 0.9 both saturate at 1.0 (dedup collapses them).
        assert_eq!(f(0.95), 1.0);
        assert_eq!(f(0.9), 1.0);
    }

    #[test]
    fn dedup_collapses_wp_only_and_saturated_siblings() {
        let less = dedup_tectonic(tectonic_less_palette());
        let more = dedup_tectonic(tectonic_more_palette());
        assert!(
            less.len() < 24,
            "wp-only / fraction-saturated siblings must collapse (got {})",
            less.len()
        );
        assert!(more.len() < 20, "got {}", more.len());
        // But the surviving sets stay materially large — the trial
        // schedule is a real search, not a no-op.
        assert!(less.len() >= 12, "got {}", less.len());
        assert!(more.len() >= 10, "got {}", more.len());
        // First-seen order: config #1 survives as the head.
        assert_eq!(less[0], tectonic_less_palette()[0]);
    }
}
