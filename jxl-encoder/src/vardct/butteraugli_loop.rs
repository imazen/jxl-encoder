// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Butteraugli quantization loop for iterative quality refinement.
//!
//! Iteratively refines per-block quant_field by measuring perceptual distance
//! (butteraugli) between the original and reconstructed image.
//!
//! Matches libjxl's FindBestQuantization (enc_adaptive_quantization.cc:929-1115):
//! - Works in float quant field domain (values ~0.3-1.5), NOT integer (1-255)
//! - Recomputes global_scale each iteration via SetQuantField (median/MAD)
//! - Returns final DistanceParams for use in CfL pass 2 and encoding

use core::sync::atomic::{AtomicI32, Ordering};

use super::ac_strategy::AcStrategyMap;
use super::adaptive_quant::quantize_quant_field;
use super::chroma_from_luma::CflMap;
use super::common::*;
use super::encoder::VarDctEncoder;
use super::frame::DistanceParams;
use crate::debug_rect;
use crate::error::Result;

/// libjxl's hardcoded `kInitMul` (`enc_adaptive_quantization.cc:1042`)
/// that pulls the post-`kOriginalComparisonRound` quant field back toward
/// the initial AC heuristic field. Single-seed encodes use only this
/// value (bit-identical to libjxl).
pub(crate) const LIBJXL_INIT_MUL: f64 = 0.6;

// ===== Distance-aware buttloop tuning scaffolding (W38-2 #3.1).
//
// Infrastructure ported from GPU `d75bf7c` (memory
// `buttloop_rd_gap_2026-05-14.md`) for hot-swap A/B sweeps on the
// per-iter `(cur_pow, max_increase)` knobs. **Production defaults
// match libjxl at every regime** (`cur_pow=0.2`, `max_increase=100.0`
// ≈ "no cap"); the GPU's tuned LOW-regime values (`cur_pow=0.5`,
// `max_increase=1.3`) regressed RD-pareto on CPU at LOW (bfly +4-13 %
// on screenshots, +1-8 % on photos at d<2.0) when tested A/B — see
// `benchmarks/buttloop_distance_split_port_2026-05-18.tsv`. The GPU's
// tuning was calibrated against its own e7 baseline (≈9 % smaller
// bytes than cjxl e7); CPU's baseline differs and the same factor
// over-shrinks the quant field.
//
// The atomic overrides below let sweep harnesses search for a
// CPU-specific LOW value (or any other tuning) without rebuilds.
// Production code never sets them — see `resolved_cur_pow` /
// `resolved_max_increase` helpers.
//
// Hash-lock invariants at default e7 are preserved because the
// buttloop is gated at effort >= 8 (`speed_tier <= kKitten`).

/// Sweep override for `cur_pow` at low distances (`target_distance <
/// [`DEFAULT_DISTANCE_SPLIT`]`). Stored as `value × 1000` (so 500 = 0.5).
/// `i32::MIN` means "not overridden — use [`DEFAULT_CUR_POW_LOW`]".
pub static CUR_POW_X1000_LOW: AtomicI32 = AtomicI32::new(i32::MIN);

/// Sweep override for `cur_pow` at high distances (`target_distance >=
/// [`DEFAULT_DISTANCE_SPLIT`]`). `i32::MIN` means "use
/// [`DEFAULT_CUR_POW_HIGH`]".
pub static CUR_POW_X1000_HIGH: AtomicI32 = AtomicI32::new(i32::MIN);

/// Sweep override for `max_increase` (per-iter bad-block bump cap) at
/// low distances. Stored as `value × 1000`. `i32::MIN` means "use
/// [`DEFAULT_MAX_INCREASE_LOW`]".
pub static MAX_INCREASE_X1000_LOW: AtomicI32 = AtomicI32::new(i32::MIN);

/// Sweep override for `max_increase` at high distances. `i32::MIN` means
/// "use [`DEFAULT_MAX_INCREASE_HIGH`]".
pub static MAX_INCREASE_X1000_HIGH: AtomicI32 = AtomicI32::new(i32::MIN);

/// Sweep override for the threshold between LOW and HIGH regimes. The
/// per-iter loop picks LOW when `target_distance < threshold`, else HIGH.
/// Defaults to `2000` (= 2.0) — see [`DEFAULT_DISTANCE_SPLIT`].
///
/// Unlike the other overrides this slot is initialised to its default
/// value (NOT `i32::MIN`) so that `resolved_*` helpers always see a
/// valid split even when production runs without any harness present.
pub static DISTANCE_SPLIT_X1000: AtomicI32 = AtomicI32::new(2000);

/// Helper: read an `_X1000` override; return `default` when unset.
fn read_override_x1000(slot: &AtomicI32, default: f64) -> f64 {
    let v = slot.load(Ordering::Relaxed);
    if v == i32::MIN {
        default
    } else {
        v as f64 / 1000.0
    }
}

/// Production default for `cur_pow` in the LOW regime
/// (`target_distance < DEFAULT_DISTANCE_SPLIT`). Matches libjxl's
/// default — **the literal GPU port (`0.5`) was tested A/B on CPU
/// and over-reclaims, costing 1-13 % butteraugli at d<2.0** (see
/// `benchmarks/buttloop_distance_split_port_*.{tsv,meta}`). The
/// scaffolding stays so sweep harnesses can find a CPU-specific
/// LOW value via `CUR_POW_X1000_LOW`, but the default is the
/// libjxl-faithful value until that sweep lands.
///
/// GPU equivalent: `DEFAULT_CUR_POW_LOW = 0.5` in
/// `jxl-encoder-gpu/src/forks/butteraugli_loop.rs`. The GPU's
/// `0.5` tuning was calibrated to its own baseline (≈9 % smaller
/// bytes at e7 than cjxl); CPU's e7 baseline differs and the same
/// value is too aggressive here.
pub const DEFAULT_CUR_POW_LOW: f64 = 0.2;

/// Production default for `cur_pow` in the HIGH regime
/// (`target_distance >= DEFAULT_DISTANCE_SPLIT`). Matches libjxl's
/// default (`enc_adaptive_quantization.cc:1106`) — no change from
/// pre-port CPU behaviour.
pub const DEFAULT_CUR_POW_HIGH: f64 = 0.2;

/// Production default for `max_increase` (per-iter bad-block bump cap)
/// in the LOW regime. Matches libjxl's implicit "no cap" — set to
/// `100.0` (effectively infinite). See `DEFAULT_CUR_POW_LOW` for the
/// rationale on why the literal GPU port (`1.3`) is not the default.
pub const DEFAULT_MAX_INCREASE_LOW: f64 = 100.0;

/// Production default for `max_increase` in the HIGH regime. Matches
/// libjxl's implicit "no cap" — set to `100.0` (effectively infinite).
pub const DEFAULT_MAX_INCREASE_HIGH: f64 = 100.0;

/// Default split point between LOW and HIGH regimes.
/// `target_distance >= DEFAULT_DISTANCE_SPLIT` triggers the HIGH regime.
pub const DEFAULT_DISTANCE_SPLIT: f64 = 2.0;

/// Resolve `cur_pow` for the current iter + `target_distance`, honouring
/// any sweep overrides set in `CUR_POW_X1000_{LOW,HIGH}`.
///
/// Returns 0.0 for `iter >= 2` regardless of override (only iter < 2
/// has a good-block reclamation regime; later iters only bump bad
/// blocks — same as libjxl `enc_adaptive_quantization.cc:1106`).
pub(crate) fn resolved_cur_pow(iter: usize, target_distance: f64) -> f64 {
    if iter >= 2 {
        return 0.0;
    }
    let split = read_override_x1000(&DISTANCE_SPLIT_X1000, DEFAULT_DISTANCE_SPLIT);
    if target_distance < split {
        read_override_x1000(&CUR_POW_X1000_LOW, DEFAULT_CUR_POW_LOW)
    } else {
        read_override_x1000(&CUR_POW_X1000_HIGH, DEFAULT_CUR_POW_HIGH)
    }
}

/// Resolve `max_increase` (per-iter bad-block bump cap) for the current
/// `target_distance`, honouring sweep overrides.
pub(crate) fn resolved_max_increase(target_distance: f64) -> f64 {
    let split = read_override_x1000(&DISTANCE_SPLIT_X1000, DEFAULT_DISTANCE_SPLIT);
    if target_distance < split {
        read_override_x1000(&MAX_INCREASE_X1000_LOW, DEFAULT_MAX_INCREASE_LOW)
    } else {
        read_override_x1000(&MAX_INCREASE_X1000_HIGH, DEFAULT_MAX_INCREASE_HIGH)
    }
}

/// Seed values for the multi-seed butteraugli sweep (RFC#45 pick #1
/// chunk 3). Each seed runs the full quantization loop with a different
/// `kInitMul` (the constant that biases iter=1 toward the initial field
/// vs the per-iteration update). Different basins of the optimization
/// surface converge to different (qf, scale) pairs at the same butteraugli
/// target — we pick the seed with the largest mean(quant_field_float)
/// (proxy for smallest encoded bytes) that meets the butteraugli bound.
///
/// **Index 0 is ALWAYS the libjxl default** so the picker can never
/// regress below the single-seed baseline.
///
/// - `seeds = 1` ⇒ `[0.6]` — bit-identical to libjxl `FindBestQuantization`.
/// - `seeds = 2` ⇒ `[0.6, 0.4]` — adds a "trust the per-iter update more"
///   basin (smaller pullback toward initial → larger qf perturbation).
/// - `seeds = 3` ⇒ `[0.6, 0.4, 0.8]` — adds a "trust the initial more"
///   basin (more conservative; smaller qf perturbation, often hits
///   target with finer quant on noisy inputs).
/// - `seeds = 4` ⇒ `[0.6, 0.4, 0.8, 0.5]` — fills the gap near the
///   default with a fourth basin.
///
/// Capped at the length returned here (4); requesting more silently
/// saturates. The values are chosen empirically to span the
/// `kInitMul ∈ [0, 1]` interpolation interval without clustering near
/// the endpoints (where the loop degenerates to pure-update or
/// pure-pullback dynamics).
pub(crate) fn init_mul_seeds(seeds: u8) -> &'static [f64] {
    const ALL: [f64; 4] = [LIBJXL_INIT_MUL, 0.4, 0.8, 0.5];
    let n = (seeds.max(1) as usize).min(ALL.len());
    &ALL[..n]
}

/// Outcome of one butteraugli-loop seed used by the multi-seed picker
/// in [`VarDctEncoder::butteraugli_refine_quant_field`].
#[derive(Clone)]
struct SeedOutcome {
    /// Final `DistanceParams` after the loop's terminal SetQuantField.
    params: DistanceParams,
    /// `u8` quant_field after the loop's terminal SetQuantField (length
    /// `xsize_blocks * ysize_blocks`).
    quant_field: alloc::vec::Vec<u8>,
    /// Float quant_field at loop exit (length matches `quant_field`).
    quant_field_float: alloc::vec::Vec<f32>,
    /// Butteraugli score from the compare-only last iteration (`f64::INFINITY`
    /// if the reference compare failed at any point).
    final_score: f64,
    /// Mean of the final float quant_field — the smallest-bytes proxy
    /// (larger = coarser quantization = fewer non-zero coefficients).
    mean_qf: f64,
    /// `k_init_mul` value used for this seed (for debug logging).
    /// Only read when the `debug-rect` feature is enabled.
    #[cfg_attr(not(feature = "debug-rect"), allow(dead_code))]
    k_init_mul: f64,
}

impl VarDctEncoder {
    /// Butteraugli quantization loop: iteratively refines per-block quant_field
    /// by measuring perceptual distance (butteraugli) between the original image
    /// and the reconstruction from quantized coefficients.
    ///
    /// **Float-domain operation** (matching libjxl FindBestQuantization):
    /// The quant field is maintained in float domain (~0.3-1.5 range). Each
    /// iteration recomputes global_scale from the float field's median/MAD
    /// (matching libjxl's SetQuantField), then converts to u8 for quantization.
    ///
    /// Algorithm:
    /// For each iteration:
    ///   1. SetQuantField: recompute global_scale from float field, convert to u8
    ///   2. transform_and_quantize with current quant_field and new params
    ///   3. reconstruct XYB → apply gab → EPF → XYB-to-linear
    ///   4. butteraugli(original_linear, reconstructed_linear) → per-block distmap
    ///   5. Adjust float quant_field based on tile distances
    ///   6. Enforce deviation bounds from initial field
    ///
    /// AC strategy is FIXED throughout — only quant_field changes.
    /// Returns the final DistanceParams (with recomputed global_scale).
    #[cfg(feature = "butteraugli-loop")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn butteraugli_refine_quant_field(
        &self,
        linear_rgb: &[f32],
        width: usize,
        height: usize,
        xyb_x: &[f32],
        xyb_y: &[f32],
        xyb_b: &[f32],
        padded_width: usize,
        padded_height: usize,
        xsize_blocks: usize,
        ysize_blocks: usize,
        initial_params: &DistanceParams,
        quant_field: &mut [u8],
        quant_field_float: &mut [f32],
        initial_quant_field_float: &[f32],
        cfl_map: &CflMap,
        ac_strategy: &AcStrategyMap,
        patches_data: Option<&super::patches::PatchesData>,
        splines_data: Option<&super::splines::SplinesData>,
    ) -> Result<DistanceParams> {
        use crate::budget::MemoryBudget;

        // EX-J11 chunk 2: HDR-aware loss dispatch.
        //
        // Default path (`HdrLoss::Butteraugli`) is byte-identical to
        // every release prior to EX-J11 — the existing butteraugli
        // reference setup + per-iter compare below runs unchanged.
        //
        // `HdrLoss::Vdp2` (opt-in via [`crate::LossyConfig::with_hdr_loss`])
        // skips the butteraugli reference precompute and routes each
        // per-iter compare through [`super::hdr_vdp2_lite::compare_vdp2_planar`]
        // — a calibrated subset of HDR-VDP-2 that adapts to the encode's
        // `intensity_target`. See the module docs in
        // [`super::hdr_vdp2_lite`] for the deviations from the full paper
        // (cortex channels, chromatic sensitivity, masking model — all
        // chunk-3 follow-ons).
        if let Err(e) = super::hdr_metrics::validate_loss(self.hdr_loss) {
            return Err(crate::error::Error::NotImplemented(alloc::format!(
                "HDR loss dispatch: {e} (selected: {})",
                self.hdr_loss.as_str()
            )));
        }
        // EX-J11 chunk 4: belt-and-braces resolve of `HdrLoss::Auto`.
        // The public LossyConfig pipeline calls
        // `LossyConfig::resolve_hdr_loss(...)` before assigning
        // `enc.hdr_loss`, so by the time we reach this loop `Auto`
        // has normally been replaced by `Butteraugli` or `Vdp2`.
        // Direct construction of `VarDctEncoder` (e.g. from tests or
        // internal callers) may still leave `Auto` here — resolve with
        // `None` (no transfer-function hint available at this layer)
        // so we land on the SDR-safe `Butteraugli` path.
        let resolved_loss = self.hdr_loss.resolve(None);
        let use_vdp2 = matches!(resolved_loss, super::hdr_metrics::HdrLoss::Vdp2);

        let budget = self.budget.as_ref();
        let target_distance = self.distance;
        let num_blocks = xsize_blocks * ysize_blocks;
        let padded_pixels = padded_width * padded_height;

        // Precompute the perceptual reference from the original image ONCE.
        // Deinterleave to planar so both metric paths consume the same layout.
        //
        // For `HdrLoss::Butteraugli` we additionally build a `ButteraugliReference`
        // (the cached separated-frequencies + masking precompute). For
        // `HdrLoss::Vdp2` we skip that precompute — VDP2-lite has no separable
        // per-image cache; it walks both planes per-iter (the pyramid construction
        // dominates and is only sub-linear in the image size).
        //
        // The planar `ref_r/g/b` planes are kept alive for the full loop
        // duration so the VDP2 path can re-use them across iterations. Budget
        // is reserved permanently in the VDP2 branch (vs the butteraugli branch
        // where the planes are released after the reference precompute takes
        // ownership of an internal copy).
        let n = width * height;
        let mut ref_r = vec![0.0f32; n];
        let mut ref_g = vec![0.0f32; n];
        let mut ref_b = vec![0.0f32; n];
        for i in 0..n {
            ref_r[i] = linear_rgb[i * 3];
            ref_g[i] = linear_rgb[i * 3 + 1];
            ref_b[i] = linear_rgb[i * 3 + 2];
        }
        // intensity_target the VDP2-lite path uses to map linear-RGB [0,1]
        // onto absolute display luminance in cd/m². Pulled from the
        // VarDctEncoder field that the public LossyConfig::with_intensity_target
        // setter populates. SDR encodes default to 255.0, matching the
        // existing initialiser in vardct/encoder.rs:549.
        let vdp2_intensity_target = self.intensity_target;

        let reference: Option<butteraugli::ButteraugliReference> = if use_vdp2 {
            // VDP2 path: hold onto the planar refs permanently (one
            // n*4*3 reservation) and skip the butteraugli precompute.
            MemoryBudget::reserve_permanent_opt(budget, (n as u64).saturating_mul(4 * 3))?;
            None
        } else {
            // Butteraugli path: transient n*4*3 reservation released as
            // soon as the reference precompute owns its internal copy.
            let _g = MemoryBudget::reserve_opt(budget, (n as u64).saturating_mul(4 * 3))?;
            let butteraugli_params = butteraugli::ButteraugliParams::new()
                .with_intensity_target(80.0)
                .with_compute_diffmap(true);
            let r = match butteraugli::ButteraugliReference::new_linear_planar(
                &ref_r,
                &ref_g,
                &ref_b,
                width,
                height,
                width,
                butteraugli_params,
            ) {
                Ok(r) => r,
                Err(_) => return Ok(initial_params.clone()),
            };
            Some(r)
        };

        // Compute deviation bounds from the FLOAT initial field (libjxl lines 968-976).
        // These prevent the quant field from diverging too far from the initial field.
        let initial_qf_min = initial_quant_field_float
            .iter()
            .copied()
            .reduce(f32::min)
            .unwrap_or(0.01)
            .max(1e-6);
        let initial_qf_max = initial_quant_field_float
            .iter()
            .copied()
            .reduce(f32::max)
            .unwrap_or(1.0);
        let initial_qf_ratio = initial_qf_max / initial_qf_min;
        let qf_max_deviation_low = (250.0f32 / initial_qf_ratio).sqrt();
        let asymmetry = 2.0f32.min(qf_max_deviation_low);
        let qf_lower = initial_qf_min / (asymmetry * qf_max_deviation_low);
        let qf_higher = initial_qf_max * (qf_max_deviation_low / asymmetry);

        // Pre-allocate buffers reused across iterations.
        // These live for the duration of the loop — accounted permanently.
        // sharpness is u8, tile_dist is f32 of num_blocks, recon_* are f32 of padded_pixels.
        MemoryBudget::reserve_permanent_opt(
            budget,
            (num_blocks as u64)
                .saturating_add((num_blocks as u64).saturating_mul(4))
                .saturating_add((padded_pixels as u64).saturating_mul(4 * 3)),
        )?;
        let sharpness = vec![4u8; num_blocks];
        let mut tile_dist = vec![0.0f32; num_blocks];
        let mut recon_r = vec![0.0f32; padded_pixels];
        let mut recon_g = vec![0.0f32; padded_pixels];
        let mut recon_b = vec![0.0f32; padded_pixels];
        let mut transform_out =
            super::transform::TransformOutput::new(xsize_blocks, ysize_blocks, budget)?;

        // Saturate at consumption to bound worst-case CPU even when the
        // caller skipped LossyConfig::validate (which would have rejected
        // values > MAX_QUANT_LOOP_ITERS with IterCountOutOfRange). Each
        // iteration runs a full butteraugli pipeline; capping prevents
        // a malicious or buggy caller from DoS-ing the encoder.
        let iters = (self.butteraugli_iters.min(crate::api::MAX_QUANT_LOOP_ITERS)) as usize;
        // RFC#45 chunk 1 + chunk 2: e10/e11/e12 push butteraugli_iters to
        // 8/16/32 via the effort table (see effort.rs). The saturating
        // `.min()` above already bounds the loop; this debug-assert documents
        // the structural invariant so future effort levels can't sneak past
        // `MAX_QUANT_LOOP_ITERS` (= 32 after chunk 2) and underflow the
        // compare-only exit (`if iter == iters { break }` below).
        debug_assert!(
            iters <= crate::api::MAX_QUANT_LOOP_ITERS as usize,
            "butteraugli loop iters={} exceeds MAX_QUANT_LOOP_ITERS={} \
             (effort table must saturate at the cap)",
            iters,
            crate::api::MAX_QUANT_LOOP_ITERS,
        );

        // RFC#45 pick #1 chunk 3 — multi-seed butteraugli sweep.
        //
        // At e ≤ 9 the profile sets `lossy_search_seeds = 1` and the seed
        // table is `[LIBJXL_INIT_MUL]` (= 0.6) — bit-identical to the
        // single-seed libjxl `FindBestQuantization`. At e10/e11 we fan
        // out 2/4 different `kInitMul` values, run the full loop on a
        // clone of (`quant_field`, `quant_field_float`) per seed, then
        // pick the seed with the largest mean(`quant_field_float`)
        // (proxy for smallest encoded bytes — coarser quant produces
        // fewer non-zero AC coefficients and thus shorter Huffman/ANS
        // streams) whose final butteraugli score does not exceed
        // `K_BUTTERAUGLI_ACCEPT_FACTOR * target_distance`. If no seed
        // meets that bound (rare; usually means target_distance is so
        // small the loop didn't converge on any seed), the seed with
        // the smallest final score wins instead — the worst-case for
        // multi-seed is the same `final_score` as single-seed because
        // `init_mul_seeds[0]` is always `LIBJXL_INIT_MUL`.
        let seeds = init_mul_seeds(self.profile.lossy_search_seeds);
        const K_BUTTERAUGLI_ACCEPT_FACTOR: f64 = 1.05;

        // Snapshot the caller's starting buffers so each seed starts
        // from the SAME initial state (the caller hands us the post-AQ
        // float field; without snapshotting, seed N+1 would start from
        // seed N's post-loop field and the sweep would degenerate).
        let initial_qf_u8_snapshot = quant_field.to_vec();
        let initial_qf_float_snapshot = quant_field_float.to_vec();

        let mut outcomes: alloc::vec::Vec<SeedOutcome> =
            alloc::vec::Vec::with_capacity(seeds.len());

        for &k_init_mul in seeds {
            // Restore starting state for this seed (skipped on seed 0 because
            // quant_field/quant_field_float already hold it, but cheap enough
            // to always do for clarity).
            quant_field.copy_from_slice(&initial_qf_u8_snapshot);
            quant_field_float.copy_from_slice(&initial_qf_float_snapshot);

            let outcome = self.butteraugli_refine_quant_field_inner_seed(
                xyb_x,
                xyb_y,
                xyb_b,
                padded_width,
                padded_height,
                xsize_blocks,
                ysize_blocks,
                initial_params,
                quant_field,
                quant_field_float,
                initial_quant_field_float,
                cfl_map,
                ac_strategy,
                patches_data,
                splines_data,
                reference.as_ref(),
                &ref_r,
                &ref_g,
                &ref_b,
                width,
                height,
                use_vdp2,
                vdp2_intensity_target,
                qf_lower,
                qf_higher,
                &sharpness,
                &mut tile_dist,
                &mut recon_r,
                &mut recon_g,
                &mut recon_b,
                &mut transform_out,
                iters,
                k_init_mul,
            )?;
            outcomes.push(outcome);
        }

        // Pick the winner. Selection rule:
        //   1. Prefer seeds with final_score <= K_BUTTERAUGLI_ACCEPT_FACTOR * target.
        //   2. Among those, pick the largest mean_qf (proxy for smallest bytes).
        //   3. If none meet bound, pick the smallest final_score (degenerates
        //      to single-seed worst-case because seed 0 = LIBJXL_INIT_MUL).
        let accept_bound = K_BUTTERAUGLI_ACCEPT_FACTOR * target_distance as f64;
        let winner_idx = {
            let qualifying: alloc::vec::Vec<usize> = (0..outcomes.len())
                .filter(|&i| outcomes[i].final_score <= accept_bound)
                .collect();
            if !qualifying.is_empty() {
                qualifying
                    .into_iter()
                    .max_by(|&a, &b| outcomes[a].mean_qf.total_cmp(&outcomes[b].mean_qf))
                    .expect("non-empty by filter")
            } else {
                (0..outcomes.len())
                    .min_by(|&a, &b| outcomes[a].final_score.total_cmp(&outcomes[b].final_score))
                    .unwrap_or(0)
            }
        };

        // Emit a one-line debug summary of all seeds so post-hoc analysis
        // can spot when the picker is consistently choosing non-default
        // seeds (signal that the libjxl default `kInitMul=0.6` is
        // sub-optimal on this image / distance combination). The
        // `summary` is only formatted when the `debug-rect` feature
        // is enabled — without it the macro `if false {}` gate drops
        // the whole arm.
        #[cfg(feature = "debug-rect")]
        let summary: alloc::string::String = {
            use alloc::string::String;
            let mut s = String::new();
            for (i, o) in outcomes.iter().enumerate() {
                if i > 0 {
                    s.push(' ');
                }
                let marker = if i == winner_idx { "*" } else { "" };
                s.push_str(&alloc::format!(
                    "[{marker}{i} k={:.2} bfly={:.3} qf_mean={:.4}]",
                    o.k_init_mul,
                    o.final_score,
                    o.mean_qf
                ));
            }
            s
        };
        #[cfg(not(feature = "debug-rect"))]
        let summary = "";
        debug_rect!(
            "bfly/seeds",
            0,
            0,
            width,
            height,
            "n={} winner={} accept<={:.3} {}",
            outcomes.len(),
            winner_idx,
            accept_bound,
            summary,
        );

        // Promote winner into the caller's mutable buffers.
        let winner = outcomes.swap_remove(winner_idx);
        quant_field.copy_from_slice(&winner.quant_field);
        quant_field_float.copy_from_slice(&winner.quant_field_float);
        Ok(winner.params)
    }

    /// Inner per-seed body of the butteraugli quantization loop.
    /// Runs the full `iters + 1` iteration sequence on the supplied
    /// `quant_field_float` (and the matching u8 `quant_field`) and
    /// returns the resulting [`SeedOutcome`].
    ///
    /// `k_init_mul` selects the basin: it scales the iter-1 pullback
    /// toward `initial_quant_field_float` (libjxl uses 0.6;
    /// [`init_mul_seeds`] returns other values for the multi-seed
    /// sweep at e10/e11). Buffers (`tile_dist`, `recon_*`,
    /// `transform_out`) are re-used between seeds — the caller is
    /// responsible for resetting `quant_field`/`quant_field_float`
    /// between calls (each seed starts from the same initial state).
    #[cfg(feature = "butteraugli-loop")]
    #[allow(clippy::too_many_arguments)]
    fn butteraugli_refine_quant_field_inner_seed(
        &self,
        xyb_x: &[f32],
        xyb_y: &[f32],
        xyb_b: &[f32],
        padded_width: usize,
        padded_height: usize,
        xsize_blocks: usize,
        ysize_blocks: usize,
        initial_params: &DistanceParams,
        quant_field: &mut [u8],
        quant_field_float: &mut [f32],
        initial_quant_field_float: &[f32],
        cfl_map: &CflMap,
        ac_strategy: &AcStrategyMap,
        patches_data: Option<&super::patches::PatchesData>,
        splines_data: Option<&super::splines::SplinesData>,
        // `Some(reference)` for the butteraugli path (chunk-1 default).
        // `None` for the VDP2-lite path (chunk-2 opt-in via
        // `HdrLoss::Vdp2`) — the metric reads `ref_r/g/b` directly.
        reference: Option<&butteraugli::ButteraugliReference>,
        // Planar linear-RGB reference planes (always populated by the
        // top-level call). Sized `width × height` with stride = width.
        // Owned by the caller for the duration of the seed loop so we
        // can re-use them across iterations without re-deinterleaving.
        ref_r: &[f32],
        ref_g: &[f32],
        ref_b: &[f32],
        // Logical image extent. Distinct from `padded_width`/`padded_height`
        // (which describe the reconstruction buffer's row stride).
        width: usize,
        height: usize,
        // EX-J11 chunk 2: select the perceptual metric.
        use_vdp2: bool,
        // VDP2-lite display-luminance target in cd/m². Unused on the
        // butteraugli path (which hardcodes `intensity_target = 80`).
        vdp2_intensity_target: f32,
        qf_lower: f32,
        qf_higher: f32,
        sharpness: &[u8],
        tile_dist: &mut [f32],
        recon_r: &mut [f32],
        recon_g: &mut [f32],
        recon_b: &mut [f32],
        transform_out: &mut super::transform::TransformOutput,
        iters: usize,
        k_init_mul: f64,
    ) -> Result<SeedOutcome> {
        use super::epf;
        use super::reconstruct::{gab_smooth, reconstruct_xyb, xyb_to_linear_rgb_planar};

        let target_distance = self.distance;
        let num_blocks = xsize_blocks * ysize_blocks;
        let padded_pixels = padded_width * padded_height;
        debug_assert_eq!(ref_r.len(), width * height);
        debug_assert_eq!(ref_g.len(), width * height);
        debug_assert_eq!(ref_b.len(), width * height);
        debug_assert!(use_vdp2 == reference.is_none());
        debug_assert_eq!(padded_pixels, recon_r.len());
        let mut current_params;
        // Score from the final compare-only iteration (i == iters).
        // `INFINITY` until first compare succeeds — propagates to picker
        // selection: any seed that failed every compare is unselectable
        // unless every seed failed.
        let mut last_score: f64 = f64::INFINITY;

        // Loop runs iters+1 times (matching libjxl: last iteration is compare-only).
        // i=0..iters-1: SetQuantField + roundtrip + compare + adjust
        // i=iters: SetQuantField + roundtrip + compare + break
        for iter in 0..iters + 1 {
            // Step 1: SetQuantField — recompute global_scale from float field,
            // then convert float → u8.
            // (libjxl: quantizer.SetQuantField(initial_quant_dc, quant_field, &raw_quant_field))
            current_params =
                DistanceParams::compute_from_quant_field(target_distance, quant_field_float);
            // Preserve chromacity adjustments and EPF from initial params
            current_params.x_qm_scale = initial_params.x_qm_scale;
            current_params.b_qm_scale = initial_params.b_qm_scale;
            current_params.epf_iters = initial_params.epf_iters;

            // Convert float → u8 with current params' inv_scale
            // (libjxl: SetQuantFieldRect: ClampVal(row_qf[x] * inv_global_scale_ + 0.5f, 1, 255))
            let qf_vec = quantize_quant_field(quant_field_float, current_params.inv_scale);
            quant_field.copy_from_slice(&qf_vec);

            // Step 2: Transform and quantize with current params.
            // `transform_out` is `&mut TransformOutput` from our caller;
            // the helper wants `&mut TransformOutput` too, so reborrow.
            self.transform_and_quantize_into(
                xyb_x,
                xyb_y,
                xyb_b,
                padded_width,
                xsize_blocks,
                ysize_blocks,
                &current_params,
                quant_field,
                cfl_map,
                ac_strategy,
                &mut *transform_out,
            );

            // Step 3: Reconstruct XYB from quantized coefficients
            let mut planes = reconstruct_xyb(
                &transform_out.quant_dc,
                &transform_out.quant_ac,
                &current_params,
                quant_field,
                cfl_map,
                ac_strategy,
                xsize_blocks,
                ysize_blocks,
            );

            if self.enable_gaborish {
                gab_smooth(&mut planes, padded_width, padded_height);
            }

            if current_params.epf_iters > 0 {
                epf::apply_epf(
                    &mut planes,
                    quant_field,
                    sharpness,
                    current_params.scale,
                    current_params.epf_iters,
                    xsize_blocks,
                    ysize_blocks,
                    padded_width,
                    padded_height,
                    self.budget.as_ref(),
                )?;
            }

            if let Some(pd) = patches_data {
                super::patches::add_patches(&mut planes, padded_width, pd);
            }

            if let Some(sd) = splines_data {
                super::splines::add_splines(&mut planes, padded_width, width, height, sd);
            }

            // Step 4: Convert reconstructed XYB to planar linear RGB
            xyb_to_linear_rgb_planar(
                &planes[0],
                &planes[1],
                &planes[2],
                recon_r,
                recon_g,
                recon_b,
                padded_pixels,
            );

            // Debug hook (Layer-1 invariant for the quality-drift investigation):
            // capture the buttloop's INTERNAL reconstruction at the FINAL iter,
            // cropped to (width, height) — this is the linear-RGB image the loop
            // measures butteraugli against. The drift hypothesis is that this
            // diverges from what the user-facing decoder produces from the SHIPPED
            // bitstream (jxl-rs / jxl-oxide). Comparing the two pinpoints the bug.
            // See memory/quality_drift_investigation_2026-05-15.md.
            #[cfg(feature = "__internal_recon_hook")]
            if iter == iters && recon_hook::capture_enabled() {
                let mut cropped_r = alloc::vec![0.0f32; width * height];
                let mut cropped_g = alloc::vec![0.0f32; width * height];
                let mut cropped_b = alloc::vec![0.0f32; width * height];
                for y in 0..height {
                    let dst = y * width;
                    let src = y * padded_width;
                    cropped_r[dst..dst + width].copy_from_slice(&recon_r[src..src + width]);
                    cropped_g[dst..dst + width].copy_from_slice(&recon_g[src..src + width]);
                    cropped_b[dst..dst + width].copy_from_slice(&recon_b[src..src + width]);
                }
                // Snapshot per-block strategy + per-tile CfL for chunk-2's
                // diff-map correlation. These are cheap (a few bytes per block,
                // 2 i8 per tile) and only allocated when capture is enabled.
                let nblocks = xsize_blocks * ysize_blocks;
                let mut raw_strategy_v = alloc::vec![0u8; nblocks];
                let mut is_first_block = alloc::vec![false; nblocks];
                for by in 0..ysize_blocks {
                    for bx in 0..xsize_blocks {
                        let idx = by * xsize_blocks + bx;
                        raw_strategy_v[idx] = ac_strategy.raw_strategy(bx, by);
                        is_first_block[idx] = ac_strategy.is_first(bx, by);
                    }
                }
                recon_hook::store(recon_hook::InternalRecon {
                    width,
                    height,
                    r: cropped_r,
                    g: cropped_g,
                    b: cropped_b,
                    iter,
                    iters,
                    xsize_blocks,
                    ysize_blocks,
                    raw_strategy: raw_strategy_v,
                    is_first_block,
                    quant_field_u8: quant_field.to_vec(),
                    xsize_tiles: cfl_map.xsize_tiles,
                    ysize_tiles: cfl_map.ysize_tiles,
                    cfl_ytox: cfl_map.ytox.clone(),
                    cfl_ytob: cfl_map.ytob.clone(),
                });
            }

            // Step 5: Perceptual comparison.
            //
            // Dispatches on `use_vdp2`:
            //  - false (default): butteraugli `compare_linear_planar` against
            //    the precomputed reference (chunk-1 byte-identical path).
            //  - true (`HdrLoss::Vdp2` opt-in): VDP2-lite, walks ref + rec
            //    planar planes through the multi-scale CSF pyramid in
            //    [`super::hdr_vdp2_lite::compare_vdp2_planar`].
            //
            // Both metrics return `(score: f64, diffmap: Vec<f32>)` sized to
            // the logical `width × height` extent. On compare failure (rare —
            // typically NaN inputs the reconstruction shouldn't produce) we
            // bail out with the previous iter's score so the picker prefers
            // any seed that converged.
            let (iter_score, diffmap_vec): (f64, alloc::vec::Vec<f32>) = if use_vdp2 {
                match super::hdr_vdp2_lite::compare_vdp2_planar(
                    ref_r,
                    ref_g,
                    ref_b,
                    recon_r,
                    recon_g,
                    recon_b,
                    width,
                    height,
                    padded_width,
                    vdp2_intensity_target,
                ) {
                    Ok(r) => (r.score, r.diffmap),
                    Err(_) => {
                        let mean_qf = mean_qf_float(quant_field_float);
                        return Ok(SeedOutcome {
                            params: current_params,
                            quant_field: quant_field.to_vec(),
                            quant_field_float: quant_field_float.to_vec(),
                            final_score: last_score,
                            mean_qf,
                            k_init_mul,
                        });
                    }
                }
            } else {
                let bref = reference.expect(
                    "non-VDP2 path must carry a butteraugli reference (top-level invariant)",
                );
                let result =
                    match bref.compare_linear_planar(recon_r, recon_g, recon_b, padded_width) {
                        Ok(r) => r,
                        Err(_) => {
                            let mean_qf = mean_qf_float(quant_field_float);
                            return Ok(SeedOutcome {
                                params: current_params,
                                quant_field: quant_field.to_vec(),
                                quant_field_float: quant_field_float.to_vec(),
                                final_score: last_score,
                                mean_qf,
                                k_init_mul,
                            });
                        }
                    };
                let dm = match result.diffmap {
                    Some(dm) => dm,
                    None => {
                        let mean_qf = mean_qf_float(quant_field_float);
                        return Ok(SeedOutcome {
                            params: current_params,
                            quant_field: quant_field.to_vec(),
                            quant_field_float: quant_field_float.to_vec(),
                            final_score: last_score,
                            mean_qf,
                            k_init_mul,
                        });
                    }
                };
                // ImgVec is contiguous when produced by the butteraugli
                // crate (stride == width — confirmed in lib.rs:510). Take
                // ownership of the backing Vec to match the VDP2 branch's
                // owned-Vec return type without re-allocating.
                (result.score, dm.into_buf())
            };

            // Record metric score for the picker (rewritten every iter;
            // the value at loop exit is what the picker compares against the
            // target). Stored before the iter==iters early-break below so the
            // compare-only last iteration is included.
            last_score = iter_score;

            // Step 6: Compute per-block tile distance (16th-power norm, matching libjxl TileDistMap)
            const K_TILE_NORM: f32 = 1.2;
            let diffmap_buf: &[f32] = &diffmap_vec;
            tile_dist.fill(0.0);
            for by in 0..ysize_blocks {
                for bx in 0..xsize_blocks {
                    if !ac_strategy.is_first(bx, by) {
                        continue;
                    }
                    let covered_x = ac_strategy.covered_blocks_x(bx, by);
                    let covered_y = ac_strategy.covered_blocks_y(bx, by);
                    let px_start_x = bx * BLOCK_DIM;
                    let px_start_y = by * BLOCK_DIM;
                    let px_end_x = ((bx + covered_x) * BLOCK_DIM).min(width);
                    let px_end_y = ((by + covered_y) * BLOCK_DIM).min(height);
                    if px_start_x >= width || px_start_y >= height {
                        continue;
                    }
                    let mut dist_norm = 0.0f64;
                    let mut pixels = 0.0f64;
                    for py in px_start_y..px_end_y {
                        for px in px_start_x..px_end_x {
                            let v = diffmap_buf[py * width + px] as f64;
                            let v2 = v * v;
                            let v4 = v2 * v2;
                            let v8 = v4 * v4;
                            let v16 = v8 * v8;
                            dist_norm += v16;
                            pixels += 1.0;
                        }
                    }
                    if pixels == 0.0 {
                        pixels = 1.0;
                    }
                    let td = K_TILE_NORM * (dist_norm / pixels).sqrt().sqrt().sqrt().sqrt() as f32;
                    for sy in 0..covered_y {
                        for sx in 0..covered_x {
                            tile_dist[(by + sy) * xsize_blocks + (bx + sx)] = td;
                        }
                    }
                }
            }

            // Log per-iteration summary
            {
                let qf_min = quant_field_float
                    .iter()
                    .copied()
                    .reduce(f32::min)
                    .unwrap_or(0.0);
                let qf_max = quant_field_float
                    .iter()
                    .copied()
                    .reduce(f32::max)
                    .unwrap_or(0.0);
                let qf_sum: f64 = quant_field_float.iter().map(|&v| v as f64).sum();
                let qf_avg = qf_sum / quant_field_float.len() as f64;
                let td_max = tile_dist.iter().copied().reduce(f32::max).unwrap_or(0.0);
                let bad_blocks = tile_dist.iter().filter(|&&d| d > target_distance).count();
                debug_rect!(
                    "bfly/iter",
                    0,
                    0,
                    width,
                    height,
                    "iter={}/{} score={:.3} target={:.3} gs={} qf_avg={:.4} qf=[{:.4};{:.4}] td_max={:.2} bad={}",
                    iter,
                    iters,
                    iter_score,
                    target_distance,
                    current_params.global_scale,
                    qf_avg,
                    qf_min,
                    qf_max,
                    td_max,
                    bad_blocks
                );
            }

            // Last iteration is compare-only (libjxl: if (i == iters) break;)
            if iter == iters {
                break;
            }

            // Step 7: kOriginalComparisonRound = 1: constrain toward initial BEFORE adjustment.
            // Prevents oscillation by keeping qf from diverging too far from initial.
            // (libjxl enc_adaptive_quantization.cc:1039-1057)
            //
            // `k_init_mul` is the seed parameter — libjxl hardcodes 0.6 here;
            // RFC#45 chunk 3 sweeps multiple values at e10/e11 and picks the
            // best per-image outcome. See `init_mul_seeds()` and the
            // `lossy_search_seeds` field on [`EffortProfile`].
            const K_ORIGINAL_COMPARISON_ROUND: usize = 1;
            if iter == K_ORIGINAL_COMPARISON_ROUND {
                let k_one_minus_init_mul = 1.0 - k_init_mul;
                for bi in 0..num_blocks {
                    let init_qf = initial_quant_field_float[bi] as f64;
                    let cur_qf = quant_field_float[bi] as f64;
                    let clamp_val = k_one_minus_init_mul * cur_qf + k_init_mul * init_qf;
                    if cur_qf < clamp_val {
                        let mut v = clamp_val as f32;
                        if v > qf_higher {
                            v = qf_higher;
                        }
                        if v < qf_lower {
                            v = qf_lower;
                        }
                        quant_field_float[bi] = v;
                    }
                }
            }

            // Step 8: Adjust float quant_field based on tile distances.
            // (libjxl enc_adaptive_quantization.cc:1059-1110)
            //
            // Distance-aware tuning scaffolding (W38-2 #3.1; ported
            // from GPU `d75bf7c` as infrastructure-only).
            //
            // **Production defaults match libjxl at both regimes**
            // (`cur_pow=0.2`, `max_increase=100.0` ≈ "no cap"). The
            // literal GPU LOW-regime tuning (cur_pow=0.5,
            // max_increase=1.3) regressed bfly +4-13 % on CPU and was
            // not adopted as default — see
            // `benchmarks/buttloop_distance_split_port_2026-05-18.tsv`.
            //
            // The atomic overrides
            // (`CUR_POW_X1000_{LOW,HIGH}` / `MAX_INCREASE_X1000_{LOW,HIGH}`
            // / `DISTANCE_SPLIT_X1000`) let sweep harnesses search for
            // a CPU-specific LOW value that survives RD-pareto without
            // rebuilds; production code never sets them.
            //
            // `cur_pow` is 0.0 for `iter >= 2` regardless of regime
            // (only iter < 2 reduces quality of good blocks; later
            // iters only bump bad blocks — same as libjxl
            // `enc_adaptive_quantization.cc:1106`).
            let cur_pow: f64 = resolved_cur_pow(iter, target_distance as f64);
            let max_increase: f64 = resolved_max_increase(target_distance as f64);

            // InvGlobalScale and Scale from current iteration's params
            // (these change per iteration as global_scale is recomputed)
            let inv_global_scale = current_params.inv_scale; // = 65536 / global_scale
            let quantizer_scale = current_params.scale; // = global_scale / 65536

            if cur_pow == 0.0 {
                // Only adjust bad blocks (diff > 1.0)
                // (libjxl enc_adaptive_quantization.cc:1066-1086)
                for bi in 0..num_blocks {
                    // butteraugli's ButteraugliReference is finite by
                    // construction on any finite XYB input — non-finite
                    // here is always an upstream bug. The 270-encode
                    // trigger-fixture sweep + the math (XYB transform is
                    // total on ℝ via cbrt + bias) prove these never fire
                    // on legitimate input.
                    assert!(
                        tile_dist[bi].is_finite(),
                        "butteraugli loop: non-finite tile_dist[{bi}] = {} \
                         (upstream butteraugli should never produce non-finite)",
                        tile_dist[bi]
                    );
                    assert!(
                        quant_field_float[bi].is_finite(),
                        "butteraugli loop: non-finite quant_field_float[{bi}] = {} \
                         (clamps should keep this finite every iter)",
                        quant_field_float[bi]
                    );
                    let diff_raw = tile_dist[bi] / target_distance;
                    // W38-2 #3.1: cap the per-iter bump (no-op in HIGH
                    // regime where max_increase = 100.0).
                    let diff = diff_raw.min(max_increase as f32);
                    if diff > 1.0 {
                        let old = quant_field_float[bi];
                        quant_field_float[bi] = old * diff;
                        // Minimum step check: if rounding to integer quant produces
                        // the same value, bump by one quantizer step
                        // (libjxl: if (fi == pi) row_q[x] = old + quantizer.Scale())
                        let qf_old = (old * inv_global_scale + 0.5).floor() as i32;
                        let qf_new =
                            (quant_field_float[bi] * inv_global_scale + 0.5).floor() as i32;
                        if qf_old == qf_new {
                            quant_field_float[bi] = old + quantizer_scale;
                        }
                    }
                    quant_field_float[bi] = quant_field_float[bi].clamp(qf_lower, qf_higher);
                }
            } else {
                // Adjust both directions (libjxl enc_adaptive_quantization.cc:1087-1110)
                for bi in 0..num_blocks {
                    assert!(
                        tile_dist[bi].is_finite(),
                        "butteraugli loop: non-finite tile_dist[{bi}] = {}",
                        tile_dist[bi]
                    );
                    assert!(
                        quant_field_float[bi].is_finite(),
                        "butteraugli loop: non-finite quant_field_float[{bi}] = {}",
                        quant_field_float[bi]
                    );
                    let diff_raw = tile_dist[bi] / target_distance;
                    // W38-2 #3.1: cap the per-iter bump for bad blocks
                    // (no-op in HIGH regime where max_increase = 100.0,
                    // no-op for good blocks where diff <= 1.0 anyway).
                    let diff = diff_raw.min(max_increase as f32);
                    if diff <= 1.0 {
                        // Good quality: reduce precision to save bits.
                        // `diff` must be finite — NaN here indicates a real bug
                        // (target_distance == 0, or polluted reconstruction from
                        // a previous butteraugli iteration). Surface loudly rather
                        // than silently coercing to 0 via .max() (IEEE-754 ordered
                        // max returns the non-NaN operand, and 0.0.powf(x) = 0.0
                        // is finite, so the downstream assert can't catch it).
                        assert!(
                            diff.is_finite(),
                            "butteraugli loop: non-finite diff = {diff} \
                             (tile_dist={}, target_distance={target_distance})",
                            tile_dist[bi]
                        );
                        // Negative diff would produce NaN through powf for
                        // non-integer cur_pow — guard via max(0).
                        let safe_diff = diff.max(0.0) as f64;
                        let factor = safe_diff.powf(cur_pow) as f32;
                        assert!(
                            factor.is_finite(),
                            "butteraugli loop: non-finite powf factor diff={diff} pow={cur_pow}"
                        );
                        quant_field_float[bi] *= factor;
                    } else {
                        // Bad quality: increase precision
                        let old = quant_field_float[bi];
                        quant_field_float[bi] = old * diff;
                        // Minimum step check
                        let qf_old = (old * inv_global_scale + 0.5).floor() as i32;
                        let qf_new =
                            (quant_field_float[bi] * inv_global_scale + 0.5).floor() as i32;
                        if qf_old == qf_new {
                            quant_field_float[bi] = old + quantizer_scale;
                        }
                    }
                    quant_field_float[bi] = quant_field_float[bi].clamp(qf_lower, qf_higher);
                }
            }
        }

        // Final SetQuantField: compute definitive params from final float field
        // (libjxl enc_adaptive_quantization.cc:1112-1113)
        let mut final_params =
            DistanceParams::compute_from_quant_field(target_distance, quant_field_float);
        final_params.x_qm_scale = initial_params.x_qm_scale;
        final_params.b_qm_scale = initial_params.b_qm_scale;
        final_params.epf_iters = initial_params.epf_iters;

        // Convert final float → u8 with definitive params
        let qf_vec = quantize_quant_field(quant_field_float, final_params.inv_scale);
        quant_field.copy_from_slice(&qf_vec);

        let mean_qf = mean_qf_float(quant_field_float);
        Ok(SeedOutcome {
            params: final_params,
            quant_field: quant_field.to_vec(),
            quant_field_float: quant_field_float.to_vec(),
            final_score: last_score,
            mean_qf,
            k_init_mul,
        })
    }
}

/// Mean of the float quant_field — the picker's smallest-bytes proxy
/// in [`VarDctEncoder::butteraugli_refine_quant_field`]. Larger mean
/// means coarser per-block quantization, which empirically correlates
/// with smaller encoded bytes on photographic content (fewer non-zero
/// AC coefficients → shorter Huffman/ANS streams). Computed in `f64`
/// to avoid catastrophic cancellation on large block counts.
#[cfg(feature = "butteraugli-loop")]
fn mean_qf_float(quant_field_float: &[f32]) -> f64 {
    if quant_field_float.is_empty() {
        return 0.0;
    }
    let sum: f64 = quant_field_float.iter().map(|&v| v as f64).sum();
    sum / quant_field_float.len() as f64
}

/// Debug hook for capturing the buttloop's internal reconstruction at the
/// final iteration. Off by default; gated by `feature = "__internal_recon_hook"`.
///
/// The hook is single-threaded by design (a global `Mutex<Option<...>>`) — it's
/// only meant for the Layer-1 drift-investigation test, which runs one encode
/// at a time. Concurrent encodes with capture enabled will race and one will
/// overwrite the other's recon.
///
/// The recon stored here is exactly what the buttloop measures butteraugli
/// against on its last iteration: planar linear RGB, cropped to (width, height),
/// AFTER reconstruct_xyb → gab_smooth → EPF → add_patches → add_splines →
/// xyb_to_linear_rgb_planar. If this diverges from what the user-facing decoder
/// produces from the shipped bitstream, the buttloop is targeting an image the
/// decoder never delivers — that's the drift root cause.
#[cfg(feature = "__internal_recon_hook")]
pub mod recon_hook {
    use alloc::vec::Vec;
    use core::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Mutex;

    /// Captured internal reconstruction: the exact planar linear RGB image the
    /// buttloop compared against the original on its final iteration.
    ///
    /// `r`, `g`, `b` are each `width * height` f32 in linear RGB (NOT sRGB).
    /// Values are NOT clamped to [0, 1] — the encoder operates on linear-light
    /// floats and may produce values slightly outside that range near saturation.
    #[derive(Clone)]
    pub struct InternalRecon {
        pub width: usize,
        pub height: usize,
        pub r: Vec<f32>,
        pub g: Vec<f32>,
        pub b: Vec<f32>,
        pub iter: usize,
        pub iters: usize,
        // Per-block strategy info for chunk-2 diff-map correlation.
        // Length: xsize_blocks * ysize_blocks, row-major.
        pub xsize_blocks: usize,
        pub ysize_blocks: usize,
        pub raw_strategy: Vec<u8>,
        pub is_first_block: Vec<bool>,
        pub quant_field_u8: Vec<u8>,
        // Per-tile CfL state used by the buttloop's reconstruction.
        // Length: xsize_tiles * ysize_tiles, row-major.
        pub xsize_tiles: usize,
        pub ysize_tiles: usize,
        pub cfl_ytox: Vec<i8>,
        pub cfl_ytob: Vec<i8>,
    }

    static CAPTURE_ENABLED: AtomicBool = AtomicBool::new(false);
    static LAST_RECON: Mutex<Option<InternalRecon>> = Mutex::new(None);

    /// Enable or disable capture. Defaults to disabled — even with the feature
    /// compiled in, no recon is captured unless this is set to `true`.
    pub fn set_capture_enabled(enabled: bool) {
        CAPTURE_ENABLED.store(enabled, Ordering::SeqCst);
    }

    /// Returns the current capture-enabled state. Called by the buttloop on
    /// every final iteration; cheap relaxed load.
    pub fn capture_enabled() -> bool {
        CAPTURE_ENABLED.load(Ordering::Relaxed)
    }

    /// Store the recon from the buttloop's final iteration. Overwrites any
    /// prior recon — pair with `take_last` to drain between encodes.
    pub fn store(recon: InternalRecon) {
        let mut guard = LAST_RECON.lock().expect("recon_hook mutex poisoned");
        *guard = Some(recon);
    }

    /// Take (consume) the last captured recon, leaving `None` behind.
    /// Returns `None` if no encode has captured a recon since the last take
    /// (or since process start).
    pub fn take_last() -> Option<InternalRecon> {
        let mut guard = LAST_RECON.lock().expect("recon_hook mutex poisoned");
        guard.take()
    }
}

// ===== Distance-aware buttloop tuning unit tests (W38-2 #3.1) =====
//
// These tests share global atomic state with sweep harnesses. Run
// serially (`cargo test --lib -- --test-threads=1` if interleaved
// flakes appear). They mirror the GPU encoder's
// `forks/butteraugli_loop.rs::resolved_*` tests for parity.
#[cfg(test)]
mod tuning_tests {
    use super::*;

    fn reset_overrides() {
        CUR_POW_X1000_LOW.store(i32::MIN, Ordering::Relaxed);
        CUR_POW_X1000_HIGH.store(i32::MIN, Ordering::Relaxed);
        MAX_INCREASE_X1000_LOW.store(i32::MIN, Ordering::Relaxed);
        MAX_INCREASE_X1000_HIGH.store(i32::MIN, Ordering::Relaxed);
        DISTANCE_SPLIT_X1000.store(2000, Ordering::Relaxed);
    }

    #[test]
    fn resolved_cur_pow_uses_low_default_below_split() {
        reset_overrides();
        // d=1.0 < 2.0 → LOW regime.
        let v = resolved_cur_pow(0, 1.0);
        assert!(
            (v - DEFAULT_CUR_POW_LOW).abs() < 1e-9,
            "expected DEFAULT_CUR_POW_LOW={DEFAULT_CUR_POW_LOW}, got {v}"
        );
        // iter=1 also LOW regime.
        let v1 = resolved_cur_pow(1, 1.5);
        assert!((v1 - DEFAULT_CUR_POW_LOW).abs() < 1e-9);
    }

    #[test]
    fn resolved_cur_pow_uses_high_default_at_or_above_split() {
        reset_overrides();
        // d=2.0 >= 2.0 → HIGH regime.
        let v = resolved_cur_pow(0, 2.0);
        assert!(
            (v - DEFAULT_CUR_POW_HIGH).abs() < 1e-9,
            "expected DEFAULT_CUR_POW_HIGH={DEFAULT_CUR_POW_HIGH}, got {v}"
        );
        // d=3.0 — RD-pareto target; HIGH.
        let v3 = resolved_cur_pow(0, 3.0);
        assert!((v3 - DEFAULT_CUR_POW_HIGH).abs() < 1e-9);
    }

    #[test]
    fn resolved_cur_pow_zero_at_late_iterations() {
        reset_overrides();
        // iter >= 2 → 0.0 regardless of regime.
        assert_eq!(resolved_cur_pow(2, 1.0), 0.0);
        assert_eq!(resolved_cur_pow(3, 3.0), 0.0);
        assert_eq!(resolved_cur_pow(99, 5.0), 0.0);
    }

    #[test]
    fn resolved_max_increase_picks_per_regime_default() {
        reset_overrides();
        let v_low = resolved_max_increase(1.0);
        assert!((v_low - DEFAULT_MAX_INCREASE_LOW).abs() < 1e-9);
        let v_high = resolved_max_increase(3.0);
        assert!((v_high - DEFAULT_MAX_INCREASE_HIGH).abs() < 1e-9);
        // Edge: exactly at split → HIGH.
        let v_split = resolved_max_increase(2.0);
        assert!((v_split - DEFAULT_MAX_INCREASE_HIGH).abs() < 1e-9);
    }

    #[test]
    fn override_round_trip_x1000() {
        reset_overrides();
        // Confirm the X1000 encoding round-trips through resolve helpers.
        CUR_POW_X1000_HIGH.store(350, Ordering::Relaxed); // 0.350
        let v = resolved_cur_pow(0, 3.0);
        assert!((v - 0.35).abs() < 1e-9, "got {v}");
        MAX_INCREASE_X1000_LOW.store(1500, Ordering::Relaxed); // 1.500
        let m = resolved_max_increase(1.0);
        assert!((m - 1.5).abs() < 1e-9, "got {m}");
        reset_overrides();
    }

    /// Production defaults must match libjxl
    /// `enc_adaptive_quantization.cc:1106` at every regime — both LOW
    /// and HIGH ship libjxl-faithful values until A/B sweeps find
    /// CPU-specific tuning that survives RD-pareto.
    ///
    /// The atomic-override scaffolding is intentional (sweep harnesses
    /// can override LOW), but production CPU encodes are byte-identical
    /// to pre-port behaviour.
    #[test]
    fn production_defaults_are_libjxl_faithful() {
        // libjxl `kPow = {0.2, 0.2, 0, 0, ...}` (one entry per iter).
        assert_eq!(DEFAULT_CUR_POW_LOW, 0.2);
        assert_eq!(DEFAULT_CUR_POW_HIGH, 0.2);
        // libjxl applies no cap to `diff = tile_dist / target_distance`.
        // Encode as 100.0 ("effectively infinite" — block diffs of
        // 100× would already saturate at qf_higher).
        assert_eq!(DEFAULT_MAX_INCREASE_LOW, 100.0);
        assert_eq!(DEFAULT_MAX_INCREASE_HIGH, 100.0);
        assert_eq!(DEFAULT_DISTANCE_SPLIT, 2.0);
    }

    #[test]
    fn distance_split_override_shifts_regime() {
        reset_overrides();
        // Lower the split to 1.0 — then d=1.5 is HIGH.
        DISTANCE_SPLIT_X1000.store(1000, Ordering::Relaxed);
        let v = resolved_cur_pow(0, 1.5);
        assert!((v - DEFAULT_CUR_POW_HIGH).abs() < 1e-9, "got {v}");
        reset_overrides();
    }
}
