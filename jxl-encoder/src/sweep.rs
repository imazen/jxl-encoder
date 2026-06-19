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
//! (the full sweepable struct — every effort-derived and cost-model knob is
//! reachable through it). Resolution mirrors the encoder exactly: a base
//! schedule from `(effort, mode)` with the sparse params applied on top
//! (`apply_to`), so a unique profile here is a unique encode in production.
//!
//! Requires the `__expert` cargo feature. Not part of the stable API.

use alloc::boxed::Box;
use alloc::collections::BTreeSet;
use alloc::vec::Vec;
use core::hash::{Hash, Hasher};

use crate::api::EncoderMode;
use crate::effort::{EffortProfile, LosslessInternalParams, LossyInternalParams};

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
    /// Requires the `__expert` cargo feature.
    #[must_use]
    pub fn fingerprint(&self) -> u64 {
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

/// One unique resolved lossy config from a sweep.
pub struct UniqueLossyConfig {
    /// The sparse override params that first produced this profile.
    pub params: LossyInternalParams,
    /// The resolved effective profile the encoder would consume.
    pub profile: EffortProfile,
    /// `profile.fingerprint()` — the dedup key.
    pub fingerprint: u64,
}

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
type LossyMutator = Box<dyn Fn(&mut LossyInternalParams)>;
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
pub struct LossySweep {
    effort: u8,
    mode: EncoderMode,
    axes: Vec<Vec<LossyMutator>>,
}

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

/// Lossless counterpart of [`LossySweep`].
pub struct LosslessSweep {
    effort: u8,
    mode: EncoderMode,
    axes: Vec<Vec<LosslessMutator>>,
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entropy_coding::Lz77Method;

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
            lz77_method: Some(Lz77Method::Greedy),
            ..Default::default()
        };
        let also_same = LossyInternalParams {
            lz77_method: Some(Lz77Method::Greedy),
            ..Default::default()
        };
        let diff = LossyInternalParams {
            lz77_method: Some(Lz77Method::Optimal),
            ..Default::default()
        };
        let uniques = unique_lossy_configs(9, EncoderMode::Reference, [same, also_same, diff]);
        assert_eq!(uniques.len(), 2);
    }
}
