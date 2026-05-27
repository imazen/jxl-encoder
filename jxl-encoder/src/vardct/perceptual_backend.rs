// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Pluggable perceptual-metric backend for the quantization loop
//! (W44-phase3-B1; renamed from `ButteraugliBackend` for cvvdp-fork
//! Phase 2 on 2026-05-24 — see `docs/RFC_CVVDP_FORK.md` §2.1).
//!
//! The buttloop calls a perceptual metric once per iteration to measure the
//! perceptual distance between the original linear-RGB image and the current
//! iteration's reconstructed linear-RGB image. The result drives both the
//! global score (used to terminate / pick the best seed) and the per-pixel
//! diffmap (used to compute per-block tile-distance for the next iter's
//! qf adjustment).
//!
//! This module abstracts that step behind a [`PerceptualBackend`] trait so a
//! GPU backend can be plugged in opt-in. The default backend remains the
//! existing CPU `butteraugli` crate. The trait will host a `CvvdpBackend`
//! impl in cvvdp-fork Phase 3 (RFC §2.1) alongside the existing butteraugli
//! implementations.
//!
//! ## Backends
//!
//! - [`CpuButteraugliBackend`] — always available. Wraps
//!   `butteraugli::ButteraugliReference` + `compare_linear_planar`.
//!
//! - `GpuButteraugliBackend` (feature `gpu-butteraugli`) — wraps
//!   `butteraugli_gpu::Butteraugli<CudaRuntime>`. Accepts the same f32 planar
//!   linear-RGB inputs and converts on the host to sRGB-u8 packed format
//!   (the format the GPU pipeline expects). The 0.02% relative score drift
//!   measured in W44-RECON-DEEP/A7 vs the CPU backend comes from this
//!   linear-f32 → sRGB-u8 → GPU-linear-f32 round-trip; on a 1024×1024
//!   multires comparison the GPU is ~27× faster than rayon+avx512 CPU.
//!
//! ## When the GPU backend is active
//!
//! 1. Caller sets [`LossyConfig::with_gpu_butteraugli(true)`] AND
//! 2. The `gpu-butteraugli` cargo feature was enabled at build time AND
//! 3. The CUDA runtime initialised successfully at backend construction.
//!
//! If any of those fail, the buttloop falls back to the CPU backend silently
//! (defense-in-depth — the encoder is never broken by GPU misconfiguration).
//! The fallback is observable via `EncodeStats`-style logging only.

use alloc::format;

use crate::api::{DisplayConfig, PerceptualDevice, PerceptualMetric};
use crate::error::Result;

/// Result of one perceptual-metric comparison: aggregated max-norm score over
/// the linear-RGB plane diff. The diffmap itself is written into a caller-owned
/// `Vec<f32>` by [`PerceptualBackend::compare_with_reference`], so it can be
/// recycled across iterations (W44-phase3-B7a, 2026-05-23).
#[derive(Debug)]
pub(crate) struct BackendCompareResult {
    /// Max-norm score (the value the libjxl buttloop compares against
    /// `target_distance`). Same units as `butteraugli::ButteraugliResult::score`.
    pub(crate) score: f64,
}

/// Multi-metric Phase 0 (RFC #3 §4, 2026-05-25): bundled metric +
/// device selection passed to [`construct_backend`].
///
/// Built by
/// [`crate::api::LossyConfig::resolve_perceptual_metric_selection`].
/// The Libjxl strict-parity short-circuit (W44-126) has already fired
/// inside the resolver — `metric == Butteraugli` here unconditionally
/// means "construct a butteraugli backend", even if the caller
/// requested cvvdp with the Libjxl strategy.
///
/// The cargo-feature gate (silent fallback to butteraugli when the
/// requested metric's feature isn't compiled) has also fired in the
/// resolver, so `construct_backend` itself only needs to map the
/// triple `(metric, device, target_score)` to a concrete `Box<dyn
/// PerceptualBackend>` honouring per-backend `try_new` failures.
#[derive(Copy, Clone, Debug)]
pub(crate) struct MetricSelection {
    pub(crate) metric: PerceptualMetric,
    pub(crate) device: PerceptualDevice,
    /// Caller-supplied per-distance target override. `None` = use the
    /// metric's built-in calibration table; `Some(score)` = drive the
    /// buttloop against this score via the matching metric's inverse
    /// lookup (butteraugli → `vardct/butteraugli_targets.rs`, cvvdp →
    /// `vardct/cvvdp_targets.rs`, zensim → butteraugli's table after
    /// score normalization). Propagated by
    /// [`propagate_resolved_metric_to_encoder`] into
    /// [`crate::vardct::VarDctEncoder::perceptual_target_score`]; the
    /// buttloop body
    /// [`crate::vardct::perceptual_loop::run_buttloop`] reads it at
    /// the `effective_metric_target_distance` dispatch block.
    ///
    /// Phase 1 of RFC `docs/RFC_BUTTERAUGLI_TARGET_SYMMETRY.md`
    /// (2026-05-26) closed the wiring; pre-Phase-1 this field was a
    /// no-op for all three metrics.
    pub(crate) target_score: Option<f32>,
    /// Phase 1 display-config backfill (RFC
    /// `docs/RFC_DISPLAY_CONFIG_BACKFILL.md`, 2026-05-25): the resolved
    /// target display for cvvdp scoring. Already passed through the
    /// `EncoderStrategy::Libjxl` strict-parity short-circuit in
    /// [`crate::api::LossyConfig::resolve_target_display`].
    ///
    /// Has no effect on butteraugli / zensim dispatch (only consulted
    /// by the cvvdp backend ctors to construct a matching
    /// `CvvdpParams.display`); included on the bundled struct so the
    /// metric + device + display selection travels together.
    pub(crate) target_display: DisplayConfig,
}

/// Multi-metric Phase 0 (RFC #3, 2026-05-25): translate a resolved
/// [`MetricSelection`] into the four legacy bool fields the buttloop
/// body still reads on [`crate::vardct::VarDctEncoder`].
///
/// The buttloop body's internal field shape predates Phase 0; this
/// helper keeps the body unchanged and centralises the translation.
/// A future cleanup (out of scope for Phase 0) can pull the metric +
/// device into a typed enum on `VarDctEncoder` directly.
///
/// **Semantics** mirror the pre-Phase-0 resolvers exactly:
///
/// - `Butteraugli + Auto` → `gpu_butteraugli = cfg!(feature =
///   "gpu-butteraugli")` (matches W44-PHASE3-B5-flip default)
/// - `Butteraugli + Gpu`  → `gpu_butteraugli = true` (with silent CPU
///   fallback inside `construct_backend` when CUDA missing)
/// - `Butteraugli + Cpu`  → `gpu_butteraugli = false`
/// - `Cvvdp + Auto`       → `cvvdp_loop = true`, `cvvdp_use_cpu = false`
///   (prefer GPU; `construct_backend` falls back to CPU cvvdp if GPU
///   missing and `cvvdp-loop-cpu` is compiled, else butteraugli)
/// - `Cvvdp + Gpu`        → `cvvdp_loop = true`, `cvvdp_use_cpu = false`
///   (same as Auto for cvvdp; the GPU vs CPU CVVDP toggle is on
///   `cvvdp_use_cpu`, GPU is the implicit preference)
/// - `Cvvdp + Cpu`        → `cvvdp_loop = true`, `cvvdp_use_cpu = true`
/// - `Zensim + Auto`      → `zensim_loop = true`, `zensim_use_cpu = false`
///   (prefer GPU when `zensim-loop-gpu` is compiled; falls back to CPU
///   zensim if GPU unavailable and `zensim-loop` is compiled, else
///   butteraugli)
/// - `Zensim + Gpu`       → `zensim_loop = true`, `zensim_use_cpu = false`
/// - `Zensim + Cpu`       → `zensim_loop = true`, `zensim_use_cpu = true`
///
/// Strategy::Libjxl short-circuit has already fired in the resolver
/// (metric == Butteraugli for Libjxl regardless of caller field), so
/// no Libjxl branch is needed here.
#[cfg(feature = "butteraugli-loop")]
pub(crate) fn propagate_resolved_metric_to_encoder(
    selection: MetricSelection,
    enc: &mut crate::vardct::VarDctEncoder,
) {
    // Phase 1 display-config backfill (2026-05-25): the resolved
    // display travels with the metric selection. Field propagates
    // regardless of which metric is active; only the cvvdp backend
    // ctor and the per-display target lookup actually consume it.
    enc.target_display = selection.target_display;
    // Phase 1 of RFC `docs/RFC_BUTTERAUGLI_TARGET_SYMMETRY.md`
    // (2026-05-26): propagate the resolved per-distance target-score
    // override. `None` (the default) preserves the implicit-identity
    // arm in `perceptual_loop::run_buttloop`. The Libjxl strict-parity
    // short-circuit has already fired in
    // `LossyConfig::resolve_perceptual_target_score` (which forces
    // `None` for `EncoderStrategy::Libjxl` regardless of caller field).
    enc.perceptual_target_score = selection.target_score;
    match selection.metric {
        PerceptualMetric::Butteraugli => {
            enc.cvvdp_loop = false;
            enc.cvvdp_use_cpu = false;
            enc.zensim_loop = false;
            enc.zensim_use_cpu = false;
            enc.gpu_butteraugli = match selection.device {
                PerceptualDevice::Auto => cfg!(feature = "gpu-butteraugli"),
                PerceptualDevice::Cpu => false,
                PerceptualDevice::Gpu => true,
            };
        }
        PerceptualMetric::Cvvdp => {
            // cvvdp wins the construct_backend dispatch over
            // butteraugli regardless of `gpu_butteraugli`; we leave
            // `gpu_butteraugli` at the Auto-resolved value so that if
            // cvvdp falls back to butteraugli at runtime (no CUDA + no
            // cvvdp-loop-cpu), the butteraugli backend uses the
            // caller-requested device.
            enc.cvvdp_loop = true;
            enc.cvvdp_use_cpu = matches!(selection.device, PerceptualDevice::Cpu);
            enc.zensim_loop = false;
            enc.zensim_use_cpu = false;
            enc.gpu_butteraugli = match selection.device {
                PerceptualDevice::Auto => cfg!(feature = "gpu-butteraugli"),
                PerceptualDevice::Cpu => false,
                PerceptualDevice::Gpu => true,
            };
        }
        PerceptualMetric::Zensim => {
            // zensim-fork Phase 3 (2026-05-25): zensim wins the
            // construct_backend dispatch over both cvvdp and butteraugli
            // when its cargo feature is compiled in. `gpu_butteraugli` is
            // left at the Auto-resolved value as the final-fallback
            // butteraugli device choice (mirrors the cvvdp shape).
            enc.cvvdp_loop = false;
            enc.cvvdp_use_cpu = false;
            enc.zensim_loop = true;
            enc.zensim_use_cpu = matches!(selection.device, PerceptualDevice::Cpu);
            enc.gpu_butteraugli = match selection.device {
                PerceptualDevice::Auto => cfg!(feature = "gpu-butteraugli"),
                PerceptualDevice::Cpu => false,
                PerceptualDevice::Gpu => true,
            };
        }
    }
}

// ============================================================================
// cvvdp-fork Phase 8b/8c — diffmap distribution analysis + renormalization
// ============================================================================

/// cvvdp-fork Phase 8c (2026-05-25): per-pixel diffmap renormalization scale
/// applied INSIDE the cvvdp backends before returning to the buttloop.
///
/// The W44 cost-model / per-block reducer (`vardct/perceptual_loop.rs`,
/// 16th-power norm + `tile_dist[bi] / effective_metric_target_distance > 1`
/// bad-block predicate) was calibrated for butteraugli's per-pixel diffmap
/// value range. cvvdp's JOD-derived per-pixel signal lives in a different
/// numerical range; the Phase 8a Pareto diagnosis (40.3% Pareto-front
/// position vs butteraugli's 93.6%) showed the cvvdp loop over-allocates
/// qac to blocks the reducer flags as "bad" under cvvdp's distribution
/// shape. Scaling cvvdp's per-pixel diffmap by this factor brings the
/// reducer's bad-block predicate into the same statistical regime
/// butteraugli operates in, so the W44 calibration applies 1:1.
///
/// Value seeded from Phase 8b distribution capture
/// (`benchmarks/cvvdp_diffmap_distribution_2026-05-25.tsv` — see
/// `examples/cvvdp_phase8b_diffmap_distribution.rs`).
///
/// **The right scale aligns the BAD-BLOCK PREDICATE, not just the diffmap mean.**
/// The buttloop fires refinement when `tile_dist / effective_metric_target > 1`.
/// `effective_metric_target` differs between backends:
///
///   - butter: `target_b = distance` (e.g. 2.0 at d=2)
///   - cvvdp:  `target_c = CVVDP_DISTANCE_TARGETS` lookup (e.g. 0.0724 at d=2)
///
/// So the right scale satisfies
/// `(mean_c * scale) / target_c ≈ mean_b / target_b`,
/// i.e. `scale = (target_c / target_b) * (mean_b / mean_c)`.
///
/// Phase 8b 20-cell (5 fixtures × 4 distances) computation:
/// - **median scale = 0.0177**
/// - p25..p75: [0.0103, 0.0234]
/// - geometric mean: 0.01629
/// - range: 0.0036 (terminal d=0.5, outlier) — 0.0707 (imac_g3 d=1.0)
///
/// We pick **0.018** (rounded median) for the production constant. The
/// scale is fairly distance-independent within a 2-3× band, suggesting
/// a single global value is a reasonable Phase 8c shipping choice. Per-
/// distance refinement is Phase 8g (Intervention B) follow-on.
///
/// **Sentinel value `1.0`** disables renormalization (Phase 8b harness
/// behaviour when collecting raw cvvdp values for the ratio computation).
///
/// **Env override** `JXL_CVVDP_DIFFMAP_RENORM_SCALE=<float>` replaces
/// this constant for bench harnesses. Only consulted when the env var
/// is present AND parseable; production code uses the constant.
#[cfg(feature = "cvvdp-loop")]
pub(crate) const CVVDP_DIFFMAP_RENORM_SCALE: f32 = 0.018;

/// Read the active renorm scale, honouring the
/// `JXL_CVVDP_DIFFMAP_RENORM_SCALE` env override for Phase 8b harness use.
/// Production callers should treat the env hook as bench-only.
#[cfg(feature = "cvvdp-loop")]
#[inline]
pub(crate) fn resolved_cvvdp_diffmap_renorm_scale() -> f32 {
    if let Ok(s) = std::env::var("JXL_CVVDP_DIFFMAP_RENORM_SCALE")
        && let Ok(v) = s.parse::<f32>()
        && v.is_finite()
        && v > 0.0
    {
        return v;
    }
    CVVDP_DIFFMAP_RENORM_SCALE
}

// ============================================================================
// Phase 8d (2026-05-25): bytes-tighten exit pass constants.
// ============================================================================
//
// Variant 1 (batched single-probe) per
// `docs/RFC_CVVDP_PHASE8_PARETO_TARGETING.md` §3.3. After the cvvdp seed
// loop converges, run up to `MAX_OUTER_ITERS` iters where we globally
// bump `quant_field_float` by a multiplicative factor and re-score. If
// the new score still satisfies `iter_score <= target * (1 + TOLERANCE_FRAC)`
// we accept the loosened state (gives back bytes everywhere); else we
// revert to the last accepted state and either halve the step or break.
//
// The tightening pass is gated TWICE:
//  1. The `cvvdp-loop-tighten` cargo feature must be compiled in.
//  2. The runtime field `VarDctEncoder.cvvdp_bytes_tighten` must be true
//     AND `VarDctEncoder.cvvdp_loop` must be true.
//
// Both gates default OFF outside the feature so hash-locks stay
// byte-identical regardless of feature compilation. The default INSIDE
// the feature is "on when cvvdp_loop is on" — see
// `LossyConfig::resolve_cvvdp_bytes_tighten` for the full dispatch
// matrix.
//
// The pass NEVER fires on the butteraugli loop: the butteraugli
// per-block reducer is already calibrated to the W44 cost-model gates;
// loosening it post-convergence over-tightens the bytes/quality tradeoff
// (the buttloop's seed-picker mean_qf criterion already encodes the
// "biggest qf" preference among qualifying seeds, which is the natural
// bytes-tightening surface for butteraugli — see
// `vardct/perceptual_loop.rs` `accept_bound` block).

/// Maximum number of post-convergence tighten outer iters. Each iter
/// costs ~1 cvvdp score (transform + reconstruct + compare), so total
/// wall hit is bounded by `MAX_OUTER_ITERS × per_iter_wall`. At the
/// e=8 default (`butteraugli_iters = 3` → `iters + 1 = 4` seed iters),
/// this caps the additive wall at ~125% of the seed loop in the worst
/// case where every probe is accepted. Typical case is 1-2 iters before
/// the first reject closes the loop.
#[cfg(feature = "cvvdp-loop-tighten")]
pub(crate) const CVVDP_BYTES_TIGHTEN_MAX_OUTER_ITERS: u32 = 5;

/// Initial multiplicative bump applied to `quant_field_float` on each
/// tighten outer iter. `qf *= 1.0 + STEP` increases qac → coarser
/// quantization → fewer bytes. The step decays geometrically (halves
/// after each successful accept) so the search converges on the
/// largest-bump-that-still-passes within `MAX_OUTER_ITERS`.
///
/// 0.04 = 4% bytes-saving probe per iter (a few global qac steps).
#[cfg(feature = "cvvdp-loop-tighten")]
pub(crate) const CVVDP_BYTES_TIGHTEN_INITIAL_STEP: f32 = 0.04;

/// Tolerance fraction relative to the metric target. The probe is
/// accepted iff `iter_score <= target * (1.0 + TOLERANCE_FRAC)`. For
/// cvvdp at d=1.0 (target ~0.0314 in metric direction), this is a
/// ~0.5% slack — small enough to stay near the original convergence
/// point but large enough that the seed loop's residual under-shoot
/// (a typical seed converges slightly under target) provides room
/// for the probe to fit.
#[cfg(feature = "cvvdp-loop-tighten")]
pub(crate) const CVVDP_BYTES_TIGHTEN_TOLERANCE_FRAC: f32 = 0.005;

/// cvvdp-fork Phase 8d: env-overridable settings for bench harnesses.
/// `JXL_CVVDP_BYTES_TIGHTEN_MAX_ITERS=<u32>` overrides the max iter cap;
/// `JXL_CVVDP_BYTES_TIGHTEN_STEP=<float>` overrides the initial step;
/// `JXL_CVVDP_BYTES_TIGHTEN_TOL=<float>` overrides the tolerance fraction.
/// All three are checked once per call; production callers should treat
/// them as bench-only.
#[cfg(feature = "cvvdp-loop-tighten")]
#[inline]
pub(crate) fn resolved_cvvdp_bytes_tighten_settings() -> (u32, f32, f32) {
    let mut max_iters = CVVDP_BYTES_TIGHTEN_MAX_OUTER_ITERS;
    let mut step = CVVDP_BYTES_TIGHTEN_INITIAL_STEP;
    let mut tol = CVVDP_BYTES_TIGHTEN_TOLERANCE_FRAC;
    if let Ok(s) = std::env::var("JXL_CVVDP_BYTES_TIGHTEN_MAX_ITERS")
        && let Ok(v) = s.parse::<u32>()
    {
        max_iters = v;
    }
    if let Ok(s) = std::env::var("JXL_CVVDP_BYTES_TIGHTEN_STEP")
        && let Ok(v) = s.parse::<f32>()
        && v.is_finite()
        && v > 0.0
    {
        step = v;
    }
    if let Ok(s) = std::env::var("JXL_CVVDP_BYTES_TIGHTEN_TOL")
        && let Ok(v) = s.parse::<f32>()
        && v.is_finite()
        && v >= 0.0
    {
        tol = v;
    }
    (max_iters, step, tol)
}

/// cvvdp-fork Phase 8b (2026-05-25): when env var
/// `JXL_PHASE8B_DIFFMAP_DUMP` is set to a writable file path, every
/// `compare_with_reference` call appends one TSV row capturing the
/// diffmap distribution stats (mean / median / p25 / p75 / p95 / max).
/// Cheap (~one O(N) pass over the diffmap on top of the buttloop's
/// per-iter compare) and unconditionally disabled when the env var is
/// unset, so this has zero production cost.
///
/// Schema (tab-separated, header written once per file create):
/// `backend\tcompare_call\twidth\theight\tn_pixels\tmean\tmedian\tp25\tp75\tp95\tmax\tscore`
///
/// The CALLER passes its own `compare_call_idx` (the buttloop iter
/// number, or a synthetic counter for harness use); the dump function
/// records it verbatim. The score is recorded post-(10-JOD) mapping for
/// cvvdp backends and verbatim for butteraugli.
#[cfg(feature = "std")]
pub(crate) fn maybe_dump_diffmap_stats(
    backend_name: &str,
    compare_call_idx: u32,
    width: usize,
    height: usize,
    diffmap: &[f32],
    score: f64,
) {
    let path = match std::env::var("JXL_PHASE8B_DIFFMAP_DUMP") {
        Ok(p) if !p.is_empty() => p,
        _ => return,
    };
    // Compute stats. We allocate a sort buffer for percentiles —
    // O(N log N) overhead but bench-only so it's tolerable.
    if diffmap.is_empty() {
        return;
    }
    let n = diffmap.len();
    let mut sum = 0.0_f64;
    let mut max = f32::NEG_INFINITY;
    for &v in diffmap {
        sum += v as f64;
        if v > max {
            max = v;
        }
    }
    let mean = sum / n as f64;
    let mut sorted: alloc::vec::Vec<f32> = diffmap.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    let p25 = sorted[(n * 25 / 100).min(n - 1)];
    let median = sorted[(n / 2).min(n - 1)];
    let p75 = sorted[(n * 75 / 100).min(n - 1)];
    let p95 = sorted[(n * 95 / 100).min(n - 1)];

    use std::io::Write;
    // O_APPEND atomic append on POSIX so multi-process safety is
    // automatic. Header line is written by the harness on file creation
    // (it knows the backend list); we always append data rows.
    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path);
    if let Ok(mut f) = file {
        let _ = writeln!(
            f,
            "{backend}\t{idx}\t{w}\t{h}\t{n}\t{mean:.6e}\t{med:.6e}\t{p25:.6e}\t{p75:.6e}\t{p95:.6e}\t{max:.6e}\t{score:.6e}",
            backend = backend_name,
            idx = compare_call_idx,
            w = width,
            h = height,
            mean = mean,
            med = median,
            p25 = p25,
            p75 = p75,
            p95 = p95,
            max = max,
            score = score,
        );
    }
}

#[cfg(not(feature = "std"))]
#[inline]
pub(crate) fn maybe_dump_diffmap_stats(
    _backend_name: &str,
    _compare_call_idx: u32,
    _width: usize,
    _height: usize,
    _diffmap: &[f32],
    _score: f64,
) {
}

/// Pluggable backend for the buttloop's per-iter compare step.
///
/// Implementors capture the reference image once via [`Self::set_reference`]
/// and then service many [`Self::compare_with_reference`] calls — one per
/// buttloop iteration. Both reference and distorted are passed as
/// **planar linear-RGB f32** with stride = width (no padding); each plane
/// holds exactly `width * height` values in `[0, 1]` (pre-opsin).
///
/// `padded_width` is the row stride of the reconstruction buffer
/// (`recon_r/g/b` from the buttloop) — backends may need to handle non-tight
/// strides on the distorted side; the reference side is always tight
/// (`width == stride`).
pub(crate) trait PerceptualBackend: core::fmt::Debug {
    /// Backend identifier (for logging). e.g. `"cpu"`, `"gpu-cuda"`,
    /// `"gpu-fallback-cpu"`.
    fn name(&self) -> &'static str;

    /// Cache the reference image. After this returns `Ok(())`,
    /// [`Self::compare_with_reference`] can be called any number of times
    /// with distorted images of the same dimensions.
    ///
    /// `ref_r/g/b` are planar linear-RGB f32 with stride == width.
    fn set_reference(
        &mut self,
        ref_r: &[f32],
        ref_g: &[f32],
        ref_b: &[f32],
        width: usize,
        height: usize,
    ) -> Result<()>;

    /// Compare against the cached reference, writing the diffmap into the
    /// caller-owned `diffmap_out` buffer (B7a, 2026-05-23). The caller is
    /// expected to keep this Vec alive across iterations to reuse the
    /// allocation; the backend resizes/refills it on each call.
    ///
    /// - `dist_r/g/b` are planar linear-RGB f32 with `padded_width` stride;
    ///   the logical extent is `width × height` (read with the buttloop's
    ///   crop convention: `dist_r[y * padded_width + x]` for x in 0..width,
    ///   y in 0..height).
    /// - On success, `diffmap_out.len() == width * height` (row-major,
    ///   stride == width) and the returned [`BackendCompareResult`] carries
    ///   the max-norm score.
    ///
    /// Must return `Err(_)` only on dimension mismatch or transient GPU
    /// errors the caller should treat as "use the previous iter's score and
    /// stop refining." The buttloop bails to a `SeedOutcome` carrying the
    /// previous iter's score on error.
    fn compare_with_reference(
        &mut self,
        dist_r: &[f32],
        dist_g: &[f32],
        dist_b: &[f32],
        padded_width: usize,
        width: usize,
        height: usize,
        diffmap_out: &mut alloc::vec::Vec<f32>,
    ) -> Result<BackendCompareResult>;

    /// W44-PHASE3-B5b: returns `Some((divergence_pct, fell_back))` if the
    /// backend ran a divergence check during this cell, else `None`.
    /// `divergence_pct` is in `[0.0, 1.0]` (symmetric relative). The CPU
    /// backend always returns `None`. The GPU backend returns `Some(_)`
    /// after the first `compare_with_reference` call when the detector
    /// was enabled at construction.
    fn divergence_status(&self) -> Option<(f64, bool)> {
        None
    }
}

// ============================================================================
// CPU backend — always available
// ============================================================================

/// CPU butteraugli backend: wraps `butteraugli::ButteraugliReference` +
/// `compare_linear_planar`. Default backend. Bit-identical to pre-W44-phase3
/// behaviour: the trait dispatch is the only difference, and the CPU impl
/// makes the same two calls the buttloop used to make inline.
#[cfg(feature = "butteraugli-loop")]
pub(crate) struct CpuButteraugliBackend {
    /// Cached `ButteraugliReference`. `None` until `set_reference` runs.
    reference: Option<butteraugli::ButteraugliReference>,
    /// `ButteraugliParams` used when (re)building the reference. Captured
    /// once at construction. Mirrors the buttloop's pre-W44-phase3 usage —
    /// `intensity_target` is resolved at backend construction time via
    /// `libjxl_butteraugli_intensity_target` so callers don't need to know
    /// the dispatch matrix.
    params: butteraugli::ButteraugliParams,
    /// Phase 8b: per-instance compare-call counter (bumped on each
    /// successful `compare_with_reference`). Used only by the
    /// `JXL_PHASE8B_DIFFMAP_DUMP` env-gated TSV dump. Zero production
    /// cost when the env var is unset.
    compare_call_count: u32,
}

#[cfg(feature = "butteraugli-loop")]
impl core::fmt::Debug for CpuButteraugliBackend {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CpuButteraugliBackend")
            .field("has_reference", &self.reference.is_some())
            .finish()
    }
}

#[cfg(feature = "butteraugli-loop")]
impl CpuButteraugliBackend {
    /// Construct a CPU backend that will use `params` when building the
    /// reference. `params` MUST include `compute_diffmap = true`; the
    /// buttloop's per-tile distance computation REQUIRES the diffmap on
    /// every iter.
    pub(crate) fn new(params: butteraugli::ButteraugliParams) -> Self {
        Self {
            reference: None,
            params,
            compare_call_count: 0,
        }
    }
}

#[cfg(feature = "butteraugli-loop")]
impl PerceptualBackend for CpuButteraugliBackend {
    fn name(&self) -> &'static str {
        "cpu"
    }

    fn set_reference(
        &mut self,
        ref_r: &[f32],
        ref_g: &[f32],
        ref_b: &[f32],
        width: usize,
        height: usize,
    ) -> Result<()> {
        let r = butteraugli::ButteraugliReference::new_linear_planar(
            ref_r,
            ref_g,
            ref_b,
            width,
            height,
            width, // tight stride
            self.params.clone(),
        )
        .map_err(|e| crate::error::Error::InvalidInput(format!("butteraugli reference: {e}")))?;
        self.reference = Some(r);
        Ok(())
    }

    fn compare_with_reference(
        &mut self,
        dist_r: &[f32],
        dist_g: &[f32],
        dist_b: &[f32],
        padded_width: usize,
        width: usize,
        height: usize,
        diffmap_out: &mut alloc::vec::Vec<f32>,
    ) -> Result<BackendCompareResult> {
        let bref = self
            .reference
            .as_ref()
            .ok_or_else(|| crate::error::Error::InvalidInput("CPU backend: no reference".into()))?;

        // B7a (2026-05-23): bench-only env hook `JXL_W44_B7_DISABLE=1`
        // routes through the pre-B7 `compare_linear_planar` + `into_buf`
        // path that always allocates a fresh `Vec<f32>` per call. Used by
        // `examples/w44_phase3_b7_buffer_recycling_ab.rs` to produce a
        // paired A/B against the production buffer-recycling path. Default
        // (env unset) uses the new `_into` API.
        if std::env::var_os("JXL_W44_B7_DISABLE").is_some() {
            let r = bref
                .compare_linear_planar(dist_r, dist_g, dist_b, padded_width)
                .map_err(|e| {
                    crate::error::Error::InvalidInput(format!("butteraugli compare: {e}"))
                })?;
            let dm = r.diffmap.ok_or_else(|| {
                crate::error::Error::InvalidInput(
                    "CPU backend: butteraugli returned no diffmap despite compute_diffmap=true"
                        .into(),
                )
            })?;
            let buf = dm.into_buf();
            debug_assert_eq!(buf.len(), width * height);
            *diffmap_out = buf;
            let _ = (width, height);
            return Ok(BackendCompareResult { score: r.score });
        }

        // Production path: `compare_linear_planar_into` recycles the
        // diffmap backing allocation via the `ButteraugliReference`'s
        // persistent pool, and fills the caller-owned `diffmap_out` Vec —
        // eliminating the per-iter `width*height*4 B` allocation that the
        // prior `compare_linear_planar` → `into_buf` path produced.
        let (score, _pnorm_3) = bref
            .compare_linear_planar_into(dist_r, dist_g, dist_b, padded_width, diffmap_out)
            .map_err(|e| crate::error::Error::InvalidInput(format!("butteraugli compare: {e}")))?;
        debug_assert_eq!(diffmap_out.len(), width * height);
        // Phase 8b: optional diffmap distribution dump.
        maybe_dump_diffmap_stats(
            "B_CPU",
            self.compare_call_count,
            width,
            height,
            diffmap_out,
            score,
        );
        self.compare_call_count = self.compare_call_count.saturating_add(1);
        Ok(BackendCompareResult { score })
    }
}

// ============================================================================
// GPU backend — feature-gated, opt-in
// ============================================================================

/// W44-PHASE3-B5b (2026-05-24): in-loop GPU-vs-CPU score divergence check.
/// On the FIRST `compare_with_reference` call the GPU backend (when the
/// env var `JXL_W44_PHASE3_B5B_DETECTOR=1` is set) also runs a CPU
/// butteraugli compare, then compares `|score_gpu - score_cpu| /
/// max(|score_gpu|, |score_cpu|)`. If the relative divergence exceeds
/// this threshold, the backend transparently falls back to its internal
/// CPU shadow for all subsequent iters of the current cell.
///
/// The B5 wider-sweep (2026-05-23) measured 36/38 cells with relative
/// score drift well under 0.5%; the 2 outlier cells (cid22_3637739 e8/e9
/// d=2) had ~5.9% / ~2.6% butteraugli drift even though SSIM2 was
/// unchanged, which is the documented buttloop-convergence signature
/// (the ~1e-7 reduction-order divergence A7 measured perturbs the
/// gradient-descent trajectory on rare images and converges to a
/// slightly different local optimum). 0.5% is the W44-phase3-B1
/// documented threshold (25× the measured drift floor on agreeing
/// cells).
///
/// Default OFF — must be enabled per the W44-PHASE3-B5b task spec
/// (`JXL_W44_PHASE3_B5B_DETECTOR=1`). Once measurement on the 38-cell
/// sweep shows detector catches the 2 divergent cells without false
/// positives on the other 36, a follow-on chunk will flip the default.
#[cfg(feature = "gpu-butteraugli")]
pub(crate) const GPU_SCORE_DIVERGENCE_PCT: f64 = 0.005;

/// Env var name that gates the W44-PHASE3-B5b divergence detector.
#[cfg(feature = "gpu-butteraugli")]
pub(crate) const W44_PHASE3_B5B_ENV: &str = "JXL_W44_PHASE3_B5B_DETECTOR";

/// Process-global counters for the W44-PHASE3-B5b divergence detector.
/// Reset to zero by [`reset_b5b_counters`] (called by the bench harness
/// between cells); incremented by the GPU backend when the detector runs
/// and / or fires. Exposed via [`b5b_counters`] for the bench harness
/// only (the production encoder never reads these).
#[cfg(feature = "gpu-butteraugli")]
pub mod b5b_counters {
    use core::sync::atomic::{AtomicU64, Ordering};

    /// Number of cells where the iter-0 detector ran (i.e. GPU + shadow
    /// CPU both invoked, score divergence computed).
    pub static DETECTOR_RUN_COUNT: AtomicU64 = AtomicU64::new(0);
    /// Number of cells where the iter-0 detector tripped fallback
    /// (i.e. `|gpu-cpu|/max > 0.5%` → forced_to_cpu = true).
    pub static FALLBACK_TRIGGERED_COUNT: AtomicU64 = AtomicU64::new(0);
    /// Sum of absolute divergence percentages across all DETECTOR_RUN
    /// cells (for mean computation).
    pub static DIVERGENCE_PCT_SUM_MILLIONTHS: AtomicU64 = AtomicU64::new(0);
    /// Max absolute divergence percentage seen across all cells.
    pub static DIVERGENCE_PCT_MAX_MILLIONTHS: AtomicU64 = AtomicU64::new(0);

    /// Reset all four counters to zero. Bench harnesses call this
    /// once at the start of a run.
    pub fn reset() {
        DETECTOR_RUN_COUNT.store(0, Ordering::SeqCst);
        FALLBACK_TRIGGERED_COUNT.store(0, Ordering::SeqCst);
        DIVERGENCE_PCT_SUM_MILLIONTHS.store(0, Ordering::SeqCst);
        DIVERGENCE_PCT_MAX_MILLIONTHS.store(0, Ordering::SeqCst);
    }

    /// Snapshot the four counters.
    pub fn snapshot() -> Snapshot {
        Snapshot {
            run_count: DETECTOR_RUN_COUNT.load(Ordering::SeqCst),
            fallback_count: FALLBACK_TRIGGERED_COUNT.load(Ordering::SeqCst),
            divergence_pct_sum: (DIVERGENCE_PCT_SUM_MILLIONTHS.load(Ordering::SeqCst) as f64)
                / 1_000_000.0,
            divergence_pct_max: (DIVERGENCE_PCT_MAX_MILLIONTHS.load(Ordering::SeqCst) as f64)
                / 1_000_000.0,
        }
    }

    /// Record a single detector observation.
    pub(crate) fn record(divergence_pct: f64, fell_back: bool) {
        DETECTOR_RUN_COUNT.fetch_add(1, Ordering::SeqCst);
        if fell_back {
            FALLBACK_TRIGGERED_COUNT.fetch_add(1, Ordering::SeqCst);
        }
        let pct_int = (divergence_pct * 1_000_000.0) as u64;
        DIVERGENCE_PCT_SUM_MILLIONTHS.fetch_add(pct_int, Ordering::SeqCst);
        // Atomic max via CAS loop.
        loop {
            let cur = DIVERGENCE_PCT_MAX_MILLIONTHS.load(Ordering::SeqCst);
            if pct_int <= cur {
                break;
            }
            if DIVERGENCE_PCT_MAX_MILLIONTHS
                .compare_exchange_weak(cur, pct_int, Ordering::SeqCst, Ordering::SeqCst)
                .is_ok()
            {
                break;
            }
        }
    }

    /// Counter snapshot returned by [`snapshot`].
    #[derive(Debug, Clone, Copy)]
    pub struct Snapshot {
        pub run_count: u64,
        pub fallback_count: u64,
        pub divergence_pct_sum: f64,
        pub divergence_pct_max: f64,
    }
}

#[cfg(feature = "gpu-butteraugli")]
pub(crate) mod gpu {
    //! GPU butteraugli backend (CUDA via CubeCL).
    //!
    //! Constructed on demand by [`construct_backend`]. If CUDA init fails
    //! (e.g. no GPU, no driver), `try_new` returns `None` and the caller
    //! falls back to the CPU backend.
    //!
    //! ## W44-phase3-B4 — linear-planes bypass
    //!
    //! As of W44-phase3-B4 the backend uploads the encoder's host-side
    //! linear-f32 planes directly to the GPU and calls
    //! `set_reference_from_linear_planes` / `compute_with_reference_from_linear_planes`
    //! (butteraugli-gpu's `internals`-gated entry points). This skips:
    //! - the host-side linear→sRGB-u8 LUT pack (B1 path, ~5-15 ms / iter @ 1 MP)
    //! - the GPU-side sRGB-u8 upload + sRGB→linear kernel
    //!
    //! Both the encoder and butteraugli-gpu use the same IEC 61966-2-1
    //! sRGB transfer-function semantics; the bypass replaces one
    //! round-trip with a single host→GPU f32 plane upload. Verified
    //! bit-identical to the legacy path on 64×64 synthetic and within
    //! 1e-7 relative on 256×256 (butteraugli-gpu test
    //! `set_reference_from_linear_planes`).

    use super::*;

    use butteraugli_gpu::{Butteraugli, ButteraugliParams as GpuParams};
    use cubecl::Runtime;
    use cubecl::cuda::CudaRuntime;
    use cubecl::prelude::ComputeClient;
    use cubecl::server::Handle;

    // Internal helper: cubecl `prelude` re-exports `as_bytes` via the
    // `CubePrimitive`/`Pod` traits; the f32 `as_bytes` we need is the
    // associated fn on `f32` itself.
    use bytemuck;

    /// CUDA-backed butteraugli backend. Wraps `Butteraugli<CudaRuntime>`
    /// and uploads host-side linear-f32 planar input directly via
    /// butteraugli-gpu's `internals` API (W44-phase3-B4 — was sRGB-u8
    /// via a host-side LUT in W44-phase3-B1).
    pub(crate) struct GpuButteraugliBackend {
        inner: Butteraugli<CudaRuntime>,
        /// CubeCL compute client. Held for `create_from_slice` calls
        /// that upload the encoder's host linear planes to GPU buffers
        /// the inner pipeline adopts.
        client: ComputeClient<CudaRuntime>,
        /// Tight-stride f32 scratch for the distorted side's R/G/B
        /// planes. The buttloop hands us strided `recon_r/g/b` with
        /// `padded_width >= width`; we copy each row into this buffer
        /// before upload because the inner pipeline expects tight
        /// `width × height` planes (no padding). One plane's worth =
        /// `width * height` f32. We allocate three planes once and
        /// reuse the scratch every iter to avoid per-iter `Vec` churn.
        dist_plane_scratch: [alloc::vec::Vec<f32>; 3],
        params: GpuParams,
        width: u32,
        height: u32,
        /// W44-PHASE3-B5b: optional CPU shadow backend used by the
        /// in-loop divergence detector. `Some(_)` only when the
        /// detector is enabled (env var `JXL_W44_PHASE3_B5B_DETECTOR=1`)
        /// AND a CPU `ButteraugliParams` was threaded through
        /// `construct_backend`. On the FIRST `compare_with_reference`
        /// call after `set_reference`, the GPU backend also runs a CPU
        /// compute via this shadow and compares scores. If divergence
        /// exceeds [`GPU_SCORE_DIVERGENCE_PCT`], `forced_to_cpu` flips
        /// `true` and subsequent compares route through the shadow
        /// only (the GPU `inner` retains its uploaded reference but
        /// is not re-invoked).
        cpu_shadow: Option<alloc::boxed::Box<CpuButteraugliBackend>>,
        /// Iter counter (per cell). 0 on the FIRST compare after
        /// `set_reference`. Used to gate the once-per-cell detector
        /// check so the shadow CPU compute only runs once.
        compare_call_count: u32,
        /// W44-PHASE3-B5b: tripped by the iter-0 divergence detector
        /// when GPU and CPU scores diverge by more than
        /// [`GPU_SCORE_DIVERGENCE_PCT`]. Once `true`, all subsequent
        /// `compare_with_reference` calls route through `cpu_shadow`
        /// instead of `inner`. Persists for the lifetime of the
        /// backend (= one buttloop cell).
        forced_to_cpu: bool,
        /// W44-PHASE3-B5b: last measured GPU vs CPU score divergence
        /// from the iter-0 detector run. `None` until iter 0 runs;
        /// `Some(pct)` afterward where `pct` is the symmetric relative
        /// divergence (0.0..=1.0). Exposed via [`Self::last_divergence_pct`]
        /// so the bench harness can sample it via a getter on the
        /// trait object (downcast).
        last_divergence_pct: Option<f64>,
    }

    impl core::fmt::Debug for GpuButteraugliBackend {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.debug_struct("GpuButteraugliBackend")
                .field("width", &self.width)
                .field("height", &self.height)
                .field("forced_to_cpu", &self.forced_to_cpu)
                .field("last_divergence_pct", &self.last_divergence_pct)
                .finish()
        }
    }

    impl GpuButteraugliBackend {
        /// Construct a GPU backend for `width × height`. Returns `None` if
        /// the CUDA runtime fails to initialise (e.g. no GPU, no driver).
        ///
        /// The GPU pipeline is multi-resolution (mirrors CPU butteraugli's
        /// default). `intensity_target` is captured at construction; the
        /// reference must be re-cached if it changes mid-encode (it doesn't
        /// today — the buttloop fixes it once per encode).
        ///
        /// `cpu_shadow_params` is the CPU `ButteraugliParams` to use when
        /// constructing the W44-PHASE3-B5b shadow CPU backend for the
        /// in-loop divergence detector. Pass `None` to disable the
        /// detector (production default — the detector is opt-in via
        /// the [`W44_PHASE3_B5B_ENV`] env var, gated inside
        /// [`construct_backend`]).
        pub(crate) fn try_new(
            width: u32,
            height: u32,
            intensity_target: f32,
            cpu_shadow_params: Option<butteraugli::ButteraugliParams>,
        ) -> Option<Self> {
            // CubeCL client init. `client(&Default::default())` returns
            // a `ComputeClient<CudaRuntime>`; a panic inside CubeCL on a
            // CUDA-less host would surface as `try_init` failure. We
            // catch_unwind so a missing CUDA driver doesn't crash the
            // entire encode.
            let client = match std::panic::catch_unwind(|| CudaRuntime::client(&Default::default()))
            {
                Ok(c) => c,
                Err(_) => return None,
            };

            let inner = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                Butteraugli::<CudaRuntime>::new_multires(client.clone(), width, height)
            }));
            let inner = match inner {
                Ok(i) => i,
                Err(_) => return None,
            };

            let n = (width as usize).checked_mul(height as usize)?;
            let params = GpuParams::default().with_intensity_target(intensity_target);

            let cpu_shadow =
                cpu_shadow_params.map(|p| alloc::boxed::Box::new(CpuButteraugliBackend::new(p)));

            Some(Self {
                inner,
                client,
                dist_plane_scratch: [
                    alloc::vec![0.0f32; n],
                    alloc::vec![0.0f32; n],
                    alloc::vec![0.0f32; n],
                ],
                params,
                width,
                height,
                cpu_shadow,
                compare_call_count: 0,
                forced_to_cpu: false,
                last_divergence_pct: None,
            })
        }

        // W44-PHASE3-B5b note: the per-cell detector state
        // (`last_divergence_pct`, `forced_to_cpu`) is exposed via the
        // `PerceptualBackend::divergence_status` trait method on the
        // owning trait object (see `impl PerceptualBackend for
        // GpuButteraugliBackend` below). Bench harnesses use that
        // method to count fallback rate per cell.

        /// Upload one tight (width*height) f32 plane to a fresh GPU
        /// handle. Caller-supplied `plane.len()` MUST equal `n = width*height`.
        fn upload_plane(&self, plane: &[f32]) -> Handle {
            debug_assert_eq!(plane.len(), (self.width as usize) * (self.height as usize));
            // bytemuck::cast_slice<f32, u8> — the f32 plane → byte view
            // CubeCL needs for `create_from_slice`.
            self.client
                .create_from_slice(bytemuck::cast_slice::<f32, u8>(plane))
        }

        /// Copy one strided plane into the tight scratch slot. Returns
        /// a `&[f32]` view of the scratch which is then uploaded via
        /// [`Self::upload_plane`]. Tight rows are width-wide; strided
        /// input rows are padded_width-wide.
        fn copy_strided_row_into_scratch(
            scratch: &mut [f32],
            src: &[f32],
            padded_width: usize,
            width: usize,
            height: usize,
        ) {
            debug_assert_eq!(scratch.len(), width * height);
            // Fast path: stride == width means already tight; one
            // contiguous copy beats per-row loops in LLVM.
            if padded_width == width {
                let n = width * height;
                debug_assert!(src.len() >= n);
                scratch.copy_from_slice(&src[..n]);
                return;
            }
            for y in 0..height {
                let src_row = y * padded_width;
                let dst_row = y * width;
                scratch[dst_row..dst_row + width].copy_from_slice(&src[src_row..src_row + width]);
            }
        }
    }

    impl PerceptualBackend for GpuButteraugliBackend {
        fn name(&self) -> &'static str {
            if self.forced_to_cpu {
                "gpu-cuda-fallback-cpu"
            } else {
                "gpu-cuda"
            }
        }

        fn set_reference(
            &mut self,
            ref_r: &[f32],
            ref_g: &[f32],
            ref_b: &[f32],
            width: usize,
            height: usize,
        ) -> Result<()> {
            if width as u32 != self.width || height as u32 != self.height {
                return Err(crate::error::Error::InvalidInput(format!(
                    "GPU backend: dim mismatch in set_reference: expected {}×{}, got {}×{}",
                    self.width, self.height, width, height,
                )));
            }
            let n = width * height;
            if ref_r.len() < n || ref_g.len() < n || ref_b.len() < n {
                return Err(crate::error::Error::InvalidInput(format!(
                    "GPU backend: reference plane too short: expected {}, got R={} G={} B={}",
                    n,
                    ref_r.len(),
                    ref_g.len(),
                    ref_b.len(),
                )));
            }
            // Upload tight reference planes directly. Reference is tight
            // by the trait contract (`set_reference` doesn't take a
            // stride; `width == stride`).
            let r_h = self.upload_plane(&ref_r[..n]);
            let g_h = self.upload_plane(&ref_g[..n]);
            let b_h = self.upload_plane(&ref_b[..n]);
            let params = self.params;
            self.inner
                .set_reference_from_linear_planes_with_options(r_h, g_h, b_h, &params)
                .map_err(|e| {
                    crate::error::Error::InvalidInput(format!(
                        "GPU butteraugli set_reference_from_linear_planes: {e}"
                    ))
                })?;
            // W44-PHASE3-B5b: also cache the reference in the shadow
            // CPU backend (if the detector was enabled at construction).
            // This is cheap (just calls `ButteraugliReference::new_linear_planar`
            // which copies the planes into its own owned storage); the
            // savings of NOT running this once per encode would be
            // negligible vs the detector's safety guarantee.
            if let Some(shadow) = self.cpu_shadow.as_mut() {
                shadow.set_reference(ref_r, ref_g, ref_b, width, height)?;
            }
            // Reset per-cell detector state on each new reference.
            // (In practice `set_reference` is called once per buttloop
            // cell, but this protects against future code paths that
            // might re-cache mid-cell.)
            self.compare_call_count = 0;
            self.forced_to_cpu = false;
            self.last_divergence_pct = None;
            Ok(())
        }

        fn compare_with_reference(
            &mut self,
            dist_r: &[f32],
            dist_g: &[f32],
            dist_b: &[f32],
            padded_width: usize,
            width: usize,
            height: usize,
            diffmap_out: &mut alloc::vec::Vec<f32>,
        ) -> Result<BackendCompareResult> {
            if width as u32 != self.width || height as u32 != self.height {
                return Err(crate::error::Error::InvalidInput(format!(
                    "GPU backend: dim mismatch in compare: expected {}×{}, got {}×{}",
                    self.width, self.height, width, height,
                )));
            }

            // W44-PHASE3-B5b: if a prior iter's detector run flipped
            // `forced_to_cpu`, route this compare through the CPU
            // shadow (skipping the GPU entirely). The shadow already
            // has the reference cached (set by `set_reference`).
            if self.forced_to_cpu {
                let shadow = self
                    .cpu_shadow
                    .as_mut()
                    .expect("forced_to_cpu=true requires cpu_shadow=Some (set by set_reference)");
                self.compare_call_count = self.compare_call_count.saturating_add(1);
                return shadow.compare_with_reference(
                    dist_r,
                    dist_g,
                    dist_b,
                    padded_width,
                    width,
                    height,
                    diffmap_out,
                );
            }

            // Strided → tight copy into the per-instance scratch, then
            // upload each plane to its own GPU handle. The inner
            // pipeline adopts the handles; CubeCL refcounts so dropping
            // our local clones doesn't free the buffers prematurely.
            let [s_r, s_g, s_b] = &mut self.dist_plane_scratch;
            Self::copy_strided_row_into_scratch(s_r, dist_r, padded_width, width, height);
            Self::copy_strided_row_into_scratch(s_g, dist_g, padded_width, width, height);
            Self::copy_strided_row_into_scratch(s_b, dist_b, padded_width, width, height);
            // Borrow self.client/self.inner separately — upload_plane
            // borrows &self.client; the inner pipeline call borrows
            // &mut self.inner.
            let r_h = self
                .client
                .create_from_slice(bytemuck::cast_slice::<f32, u8>(s_r));
            let g_h = self
                .client
                .create_from_slice(bytemuck::cast_slice::<f32, u8>(s_g));
            let b_h = self
                .client
                .create_from_slice(bytemuck::cast_slice::<f32, u8>(s_b));
            let result = self
                .inner
                .compute_with_reference_from_linear_planes(r_h, g_h, b_h)
                .map_err(|e| {
                    crate::error::Error::InvalidInput(format!(
                        "GPU butteraugli compute_with_reference_from_linear_planes: {e}"
                    ))
                })?;
            // B7a (2026-05-23): write into caller-owned Vec to recycle the
            // diffmap allocation across iters.
            let needed = width * height;
            diffmap_out.clear();
            diffmap_out.resize(needed, 0.0);
            self.inner.copy_diffmap_to(diffmap_out).map_err(|e| {
                crate::error::Error::InvalidInput(format!("GPU butteraugli copy_diffmap: {e}"))
            })?;
            let gpu_score = result.score as f64;

            // W44-PHASE3-B5b: iter-0 divergence detector. On the FIRST
            // compare after `set_reference`, also run the shadow CPU
            // compute and compare scores. If `|gpu - cpu| /
            // max(|gpu|, |cpu|) > GPU_SCORE_DIVERGENCE_PCT`, flip
            // `forced_to_cpu` and return the CPU result for this iter
            // (so the buttloop's first refinement step already uses
            // the CPU-aligned diffmap, matching what a pure-CPU run
            // would have done).
            //
            // Iter 0 only — the buttloop converges from this point and
            // a per-iter shadow would 5×-10× wall-cost (defeats the
            // GPU speedup). On the W44-PHASE3-B5 2/38 divergent cells
            // the score gap is visible at iter 0 (the gap originates
            // in the GPU's reduction-tree-order, which is deterministic
            // per-input — divergence at iter 0 implies divergence at
            // iter N).
            //
            // Cost: ~1 full CPU butteraugli call per cell (only iter 0).
            // For the 36 non-divergent cells, this adds ~1 CPU compute
            // to a 2-4 GPU-compute buttloop = ~25-50% bytes-compute
            // overhead per CELL; encode-wall overhead is much lower
            // (buttloop is only 20-30% of encode wall; CPU-compute is
            // only 30-50% of buttloop wall). Net expected: +1.5-3%
            // encode wall.
            self.compare_call_count = self.compare_call_count.saturating_add(1);
            if self.compare_call_count == 1 {
                if let Some(shadow) = self.cpu_shadow.as_mut() {
                    // Use a separate scratch buffer for the CPU shadow's
                    // diffmap so we don't clobber `diffmap_out` if the
                    // GPU result wins (no divergence case).
                    let mut shadow_diffmap: alloc::vec::Vec<f32> =
                        alloc::vec::Vec::with_capacity(width * height);
                    let cpu_result = shadow.compare_with_reference(
                        dist_r,
                        dist_g,
                        dist_b,
                        padded_width,
                        width,
                        height,
                        &mut shadow_diffmap,
                    )?;
                    let cpu_score = cpu_result.score;
                    let denom = gpu_score.abs().max(cpu_score.abs()).max(f64::MIN_POSITIVE);
                    let divergence_pct = ((gpu_score - cpu_score).abs()) / denom;
                    self.last_divergence_pct = Some(divergence_pct);
                    let debug_log =
                        std::env::var("JXL_W44_PHASE3_B5B_DEBUG").ok().as_deref() == Some("1");
                    let trip = divergence_pct > super::GPU_SCORE_DIVERGENCE_PCT;
                    // Record into the process-global counters so bench
                    // harnesses can aggregate per-cell observations.
                    super::b5b_counters::record(divergence_pct, trip);
                    if trip {
                        self.forced_to_cpu = true;
                        if debug_log {
                            eprintln!(
                                "[W44-PHASE3-B5b] DIVERGENCE @ {}×{}: gpu={:.6} cpu={:.6} \
                                 delta_pct={:.4}% > threshold={:.4}% → FALLBACK TO CPU \
                                 for remainder of cell",
                                self.width,
                                self.height,
                                gpu_score,
                                cpu_score,
                                divergence_pct * 100.0,
                                super::GPU_SCORE_DIVERGENCE_PCT * 100.0,
                            );
                        }
                        // Return the CPU result for THIS iter — buttloop's
                        // first refinement step uses the CPU-aligned diffmap.
                        *diffmap_out = shadow_diffmap;
                        return Ok(cpu_result);
                    } else if debug_log {
                        eprintln!(
                            "[W44-PHASE3-B5b] OK @ {}×{}: gpu={:.6} cpu={:.6} \
                             delta_pct={:.4}% ≤ threshold={:.4}% → CONTINUE WITH GPU",
                            self.width,
                            self.height,
                            gpu_score,
                            cpu_score,
                            divergence_pct * 100.0,
                            super::GPU_SCORE_DIVERGENCE_PCT * 100.0,
                        );
                    }
                }
            }

            // Phase 8b: optional diffmap distribution dump.
            super::maybe_dump_diffmap_stats(
                "B_GPU",
                self.compare_call_count - 1,
                width,
                height,
                diffmap_out,
                gpu_score,
            );
            Ok(BackendCompareResult { score: gpu_score })
        }

        fn divergence_status(&self) -> Option<(f64, bool)> {
            self.last_divergence_pct
                .map(|pct| (pct, self.forced_to_cpu))
        }
    }

    // ====================================================================
    // W44-phase3-B4 — dead-code retained from B1 era
    // ====================================================================
    //
    // The 8193-entry linear→sRGB-u8 LUT below was the B1 workaround for
    // the API mismatch (butteraugli-gpu only accepted sRGB-u8 input on
    // the reference side). B4 added `set_reference_from_linear_planes`
    // upstream and the GPU backend now uploads linear-f32 planes
    // directly, so the LUT is no longer reachable.
    //
    // We keep the code under `#[allow(dead_code)]` rather than deleting
    // it because:
    // 1. It documents the exact sRGB conversion semantics we used to
    //    rely on (in case a future caller wants strict sRGB-u8 parity
    //    against a CPU butteraugli reference).
    // 2. If `internals` ever stops being on by default for our path
    //    (e.g. the upstream renames the feature), this is a one-line
    //    revert path.
    // 3. Compiled-out unit tests are still useful as a smoke for the
    //    LUT's correctness if it's ever resurrected.
    //
    // The unit tests are NOT compiled in this configuration to avoid
    // dead-test warnings; they ride along inside `#[cfg(test)]`.

    /// LUT-based linear-light f32 → 8-bit sRGB conversion. 8193-entry
    /// table indexed by `(x.clamp(0, 1) * 8192) as u32` with linear
    /// interpolation in u8 space. ~30-50× faster than the scalar `powf`
    /// path; the resulting sRGB-u8 values match the slow path within
    /// 1 ULP of u8 (verified by unit test) — well under the 0.5%
    /// butteraugli divergence threshold W44-RECON-DEEP/A7 measured.
    ///
    /// **Dead since W44-phase3-B4** — retained as documentation +
    /// quick-revert path. See module-level comment.
    #[allow(dead_code)]
    static LIN_TO_SRGB_LUT: once_cell::race::OnceBox<[u8; 8193]> = once_cell::race::OnceBox::new();

    #[allow(dead_code)]
    fn build_lut() -> alloc::boxed::Box<[u8; 8193]> {
        let mut t = alloc::boxed::Box::new([0u8; 8193]);
        for (i, slot) in t.iter_mut().enumerate() {
            let x = (i as f32) / 8192.0;
            let s = if x <= 0.0031308_f32 {
                12.92_f32 * x
            } else {
                1.055_f32 * x.powf(1.0 / 2.4) - 0.055_f32
            };
            let v = (s * 255.0_f32 + 0.5_f32).floor() as i32;
            *slot = v.clamp(0, 255) as u8;
        }
        t
    }

    /// sRGB encoding for one linear-light f32 value in `[0, 1]` (clamped).
    /// Returns the 8-bit sRGB code. Matches the IEC 61966-2-1 piecewise
    /// transfer function used by `srgb_u8_to_linear_planar_kernel` in
    /// `butteraugli-gpu` (which is the inverse).
    ///
    /// **Dead since W44-phase3-B4** — retained as documentation +
    /// quick-revert path. See module-level comment.
    #[allow(dead_code)]
    #[inline]
    fn linear_to_srgb_u8(linear: f32) -> u8 {
        let table = LIN_TO_SRGB_LUT.get_or_init(build_lut);
        let x = if linear.is_nan() {
            0.0
        } else if linear < 0.0 {
            0.0
        } else if linear > 1.0 {
            1.0
        } else {
            linear
        };
        // Map x in [0, 1] → index in [0, 8192]. Bias by 0.5 so the
        // nearest-cell lookup matches the slow path's round-to-nearest
        // behaviour on the segment endpoints.
        let idx = (x * 8192.0_f32 + 0.5_f32) as usize;
        let idx = idx.min(8192);
        table[idx]
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn linear_to_srgb_endpoints() {
            assert_eq!(linear_to_srgb_u8(0.0), 0);
            assert_eq!(linear_to_srgb_u8(1.0), 255);
            // Mid-gray sRGB 128 corresponds to ~0.2159 linear. Round-trip:
            // 0.2159 → encode → ~0.5 sRGB → round to 128.
            let mid = linear_to_srgb_u8(0.2159_f32);
            assert!((127..=129).contains(&mid), "linear=0.2159 → sRGB {}", mid);
        }

        #[test]
        fn linear_to_srgb_clamps() {
            assert_eq!(linear_to_srgb_u8(-1.0), 0);
            assert_eq!(linear_to_srgb_u8(2.0), 255);
            assert_eq!(linear_to_srgb_u8(f32::NAN), 0);
        }
    }
}

// ============================================================================
// Constructor: picks CPU or GPU based on caller policy + feature gate
// ============================================================================

/// Construct the active perceptual-metric backend for one buttloop run.
///
/// Routing priority order:
///
/// **(1)** `cvvdp_requested == true` — try CVVDP backends in this sub-order:
/// **1a** if `cvvdp_use_cpu_requested == true` AND feature `cvvdp-loop-cpu`
/// is ON → [`CpuCvvdpBackend`](crate::vardct::cvvdp_backend::cpu::CpuCvvdpBackend)
/// (cvvdp-fork Phase 5; pure Rust, no CUDA; falls back to GPU CVVDP if CPU
/// construction fails). **1b** else if feature `cvvdp-loop` is ON AND CUDA
/// init succeeds → [`GpuCvvdpBackend`](crate::vardct::cvvdp_backend::gpu::GpuCvvdpBackend)
/// (cvvdp-fork Phase 3, the default cvvdp branch). **1c** else if feature
/// `cvvdp-loop-cpu` is ON → CPU CVVDP (Phase 5 silent fallback when GPU
/// CVVDP is unavailable — hosts without CUDA still get CVVDP rather than
/// dropping to butteraugli). **1d** else fall through to step 2.
///
/// **(2)** `gpu_requested == true` AND feature `gpu-butteraugli` is ON AND
/// CUDA init succeeds → `GpuButteraugliBackend` (W44-phase3-B1).
///
/// **(3)** Otherwise → [`CpuButteraugliBackend`] (always available).
///
/// **Default policy when both CPU and GPU CVVDP are compiled in**:
/// GPU wins (Agent A's CPU port honest-stopped at 4.4× off the SIMD
/// floor — measured ~10× slower than `cvvdp-gpu` warm-ref). The CPU
/// backend exists for hosts without CUDA AND for callers who
/// explicitly opt in for deterministic-to-1e-4-JOD parity vs
/// pycvvdp v0.5.4 goldens (the CPU port carries no GPU
/// reduction-order variance).
///
/// **CVVDP and GPU butteraugli are mutually exclusive at the
/// construction site** — both wrap CudaRuntime; the caller's
/// [`LossyConfig::resolve_cvvdp_loop`](crate::api::LossyConfig::resolve_cvvdp_loop)
/// takes precedence when both fields are set, so the cvvdp branch fires
/// first. If cvvdp falls back (all variants unavailable), the GPU
/// butteraugli branch is consulted next (defense-in-depth).
///
/// **CPU fallback** is silent (single `eprintln!` when a higher-priority
/// branch is requested but unavailable) so users can see why their
/// requested backend didn't fire without breaking the encode.
#[cfg(feature = "butteraugli-loop")]
pub(crate) fn construct_backend(
    width: u32,
    height: u32,
    cpu_params: butteraugli::ButteraugliParams,
    #[allow(unused_variables)] intensity_target: f32,
    selection: MetricSelection,
) -> alloc::boxed::Box<dyn PerceptualBackend> {
    // Multi-metric Phase 0 (RFC #3 §4, 2026-05-25): the 7-arg
    // pre-Phase-0 signature collapsed into one bundled struct. The
    // dispatch body below preserves the priority order exactly:
    // cvvdp first when requested (mutually exclusive with butter-GPU
    // at the CudaRuntime layer), then butter-GPU, then butter-CPU.
    let gpu_requested = matches!(selection.device, PerceptualDevice::Gpu)
        || (matches!(selection.device, PerceptualDevice::Auto)
            && cfg!(feature = "gpu-butteraugli"));
    let cvvdp_requested = matches!(selection.metric, PerceptualMetric::Cvvdp);
    let cvvdp_use_cpu_requested = matches!(selection.device, PerceptualDevice::Cpu);
    // zensim-fork Phase 3 (2026-05-25): zensim wins the dispatch over
    // both cvvdp and butteraugli when its cargo feature is compiled
    // in. Mutually exclusive with cvvdp at the dispatch level
    // (`resolve_perceptual_metric` returns exactly one of Butteraugli /
    // Cvvdp / Zensim).
    let zensim_requested = matches!(selection.metric, PerceptualMetric::Zensim);
    let zensim_use_cpu_requested = matches!(selection.device, PerceptualDevice::Cpu);
    // Phase 1 of RFC `docs/RFC_BUTTERAUGLI_TARGET_SYMMETRY.md`
    // (2026-05-26): the `target_score` field IS consumed by the
    // buttloop body via `VarDctEncoder.perceptual_target_score`
    // (propagated through `propagate_resolved_metric_to_encoder`).
    // Backend ctors themselves do NOT read it — the per-distance
    // dispatch happens in `perceptual_loop::run_buttloop` at the
    // `effective_metric_target_distance` block. The bind below
    // silences any unused-field warning on no-feature builds where
    // the field is dead-stripped before propagation. The propagation
    // path is exercised by the `perceptual_target_score_drives_loop`
    // integration test.
    let _ = selection.target_score;
    // Debug hook: `JXL_W44_PHASE3_B1_DEBUG=1` logs which backend the
    // dispatch picks. Off by default to keep production logs clean.
    #[cfg(feature = "std")]
    let debug_log = std::env::var("JXL_W44_PHASE3_B1_DEBUG").ok().as_deref() == Some("1");
    #[cfg(not(feature = "std"))]
    let debug_log = false;

    // zensim-fork Phase 3 (2026-05-25): try the zensim backends first
    // when the caller has opted in via
    // `LossyConfig::with_perceptual_metric(PerceptualMetric::Zensim)`.
    // The zensim, cvvdp, and gpu-butteraugli paths are mutually
    // exclusive at the dispatch level
    // (`resolve_perceptual_metric` returns exactly one metric); zensim
    // wins when its feature is compiled in. Silent fallback to the
    // next dispatch tier on feature-off / CUDA-init-fail.
    //
    // Dispatch ordering inside the zensim branch (mirrors cvvdp Phase 5):
    //   (a) `zensim_use_cpu_requested == true` AND `zensim-loop`
    //       compiled: try CPU first; fall back to GPU if CPU
    //       construction fails (dims < 8×8).
    //   (b) else (default policy when both backends compiled): try
    //       GPU first; fall back to CPU if `zensim-loop` is compiled
    //       AND GPU construction failed; otherwise fall through to
    //       the cvvdp/butteraugli dispatch tier.
    //
    // Phase 1 (zensim-gpu) honest-stop carryover: the current GPU
    // diffmap delegates to the CPU pipeline (+1006% wall vs score-only
    // GPU). Until Phase 1b (pure-GPU kernels) lands, callers
    // prioritising wall time should explicitly select
    // `PerceptualDevice::Cpu`.
    #[cfg(any(feature = "zensim-loop", feature = "zensim-loop-gpu"))]
    {
        if zensim_requested {
            if debug_log {
                eprintln!(
                    "[zensim-fork P3] Zensim requested @ {}×{} \
                     (use_cpu_requested={zensim_use_cpu_requested}) — \
                     trying backends in priority order",
                    width, height,
                );
            }

            // (a) caller explicitly prefers CPU. Try CPU first.
            #[cfg(feature = "zensim-loop")]
            {
                if zensim_use_cpu_requested {
                    if let Some(c) =
                        crate::vardct::zensim_backend::cpu::CpuZensimBackend::try_new(width, height)
                    {
                        if debug_log {
                            eprintln!(
                                "[zensim-fork P3] CPU zensim backend ACTIVE @ {}×{} \
                                 (explicit opt-in)",
                                width, height
                            );
                        }
                        let _ = cpu_params;
                        return alloc::boxed::Box::new(c);
                    }
                    if debug_log {
                        eprintln!(
                            "[zensim-fork P3] CPU zensim construction failed @ {}×{} \
                             (dims likely below 8×8 minimum); trying GPU zensim next",
                            width, height,
                        );
                    }
                }
            }

            // (b default) try GPU zensim. This is the default path
            // when both backends are compiled and the caller hasn't
            // explicitly opted into CPU.
            #[cfg(feature = "zensim-loop-gpu")]
            {
                if let Some(g) =
                    crate::vardct::zensim_backend::gpu::GpuZensimBackend::try_new(width, height)
                {
                    if debug_log {
                        eprintln!(
                            "[zensim-fork P3] GPU zensim backend ACTIVE @ {}×{}",
                            width, height
                        );
                    }
                    let _ = cpu_params;
                    return alloc::boxed::Box::new(g);
                }
            }

            // (c silent fallback) GPU zensim failed (no CUDA, driver
            // issue, CubeCL panic) OR `zensim-loop-gpu` feature off.
            // If `zensim-loop` is compiled in, try CPU zensim as the
            // next-best perceptual metric — the caller asked for
            // zensim, so we honour that rather than dropping all the
            // way down to butteraugli.
            #[cfg(feature = "zensim-loop")]
            {
                if !zensim_use_cpu_requested {
                    if let Some(c) =
                        crate::vardct::zensim_backend::cpu::CpuZensimBackend::try_new(width, height)
                    {
                        eprintln!(
                            "[jxl-encoder zensim-fork P3] GPU zensim unavailable \
                             (CUDA missing/failed or `zensim-loop-gpu` off); \
                             falling back to CPU zensim @ {}×{}",
                            width, height,
                        );
                        let _ = cpu_params;
                        return alloc::boxed::Box::new(c);
                    }
                }
            }

            // All zensim variants failed; fall through to cvvdp /
            // butteraugli dispatch tiers below.
            eprintln!(
                "[jxl-encoder zensim-fork P3] Zensim backend requested but \
                 unavailable (CUDA init failed, no CPU fallback compiled, \
                 or dims invalid); falling back to next dispatch tier ({}×{})",
                width, height,
            );
        } else if debug_log {
            eprintln!(
                "[zensim-fork P3] Zensim not requested @ {}×{}, continuing dispatch",
                width, height
            );
        }
    }
    #[cfg(not(any(feature = "zensim-loop", feature = "zensim-loop-gpu")))]
    if debug_log && zensim_requested {
        eprintln!(
            "[zensim-fork P3] Zensim requested but neither `zensim-loop` \
             nor `zensim-loop-gpu` cargo features are compiled in; \
             continuing dispatch ({}×{})",
            width, height
        );
    }
    // Silence unused-variable warnings when zensim features are not compiled.
    let _ = zensim_requested;
    let _ = zensim_use_cpu_requested;

    // cvvdp-fork Phase 3 + Phase 5: try the CVVDP backends first when
    // the caller has opted in via `LossyConfig::with_cvvdp_loop`. The
    // cvvdp + gpu-butteraugli paths are mutually exclusive (both wrap
    // CudaRuntime); cvvdp wins when both are requested. Silent fallback
    // to the next dispatch tier on feature-off / CUDA-init-fail.
    //
    // Phase 5 (2026-05-24) introduces the CPU CVVDP backend. The
    // dispatch ordering inside the cvvdp branch:
    //   (a) `cvvdp_use_cpu_requested == true` AND `cvvdp-loop-cpu`
    //       compiled: try CPU first; fall back to GPU if CPU
    //       construction fails (e.g. dims < 8×8).
    //   (b) else (default policy when both backends compiled): try
    //       GPU first (10× faster per Agent A's honest-stop); fall
    //       back to CPU if `cvvdp-loop-cpu` is compiled AND GPU
    //       construction failed; otherwise fall through to the
    //       butteraugli dispatch tier.
    #[cfg(feature = "cvvdp-loop")]
    {
        if cvvdp_requested {
            if debug_log {
                eprintln!(
                    "[cvvdp-fork P3/P5] CVVDP requested @ {}×{} \
                     (use_cpu_requested={cvvdp_use_cpu_requested}) — \
                     trying backends in priority order",
                    width, height,
                );
            }

            // Phase 5 (a): caller explicitly prefers CPU. Try CPU first.
            #[cfg(feature = "cvvdp-loop-cpu")]
            {
                if cvvdp_use_cpu_requested {
                    if let Some(c) = crate::vardct::cvvdp_backend::cpu::CpuCvvdpBackend::try_new(
                        width,
                        height,
                        selection.target_display,
                    ) {
                        if debug_log {
                            eprintln!(
                                "[cvvdp-fork P5] CPU CVVDP backend ACTIVE @ {}×{} \
                                 (explicit opt-in)",
                                width, height
                            );
                        }
                        let _ = cpu_params;
                        return alloc::boxed::Box::new(c);
                    }
                    // CPU construction failed (dims < 8×8); fall through
                    // to GPU attempt + butteraugli fallback.
                    if debug_log {
                        eprintln!(
                            "[cvvdp-fork P5] CPU CVVDP construction failed @ {}×{} \
                             (dims likely below 8×8 minimum); trying GPU CVVDP next",
                            width, height,
                        );
                    }
                }
            }

            // Phase 3 (b default): try GPU CVVDP. This is the default
            // path when both backends are compiled and the caller hasn't
            // explicitly opted into CPU.
            if let Some(c) = crate::vardct::cvvdp_backend::gpu::GpuCvvdpBackend::try_new(
                width as u32,
                height as u32,
                selection.target_display,
            ) {
                if debug_log {
                    eprintln!(
                        "[cvvdp-fork P3] GPU CVVDP backend ACTIVE @ {}×{}",
                        width, height
                    );
                }
                let _ = cpu_params;
                return alloc::boxed::Box::new(c);
            }

            // Phase 5 (c silent fallback): GPU CVVDP failed (no CUDA,
            // driver issue, CubeCL panic). If `cvvdp-loop-cpu` is
            // compiled in, try CPU CVVDP as the next-best perceptual
            // metric — the caller asked for cvvdp, so we honour that
            // rather than dropping all the way down to butteraugli.
            #[cfg(feature = "cvvdp-loop-cpu")]
            {
                if let Some(c) = crate::vardct::cvvdp_backend::cpu::CpuCvvdpBackend::try_new(
                    width,
                    height,
                    selection.target_display,
                ) {
                    eprintln!(
                        "[jxl-encoder cvvdp-fork P5] GPU CVVDP unavailable \
                         (CUDA missing/failed); falling back to CPU CVVDP @ {}×{}",
                        width, height,
                    );
                    let _ = cpu_params;
                    return alloc::boxed::Box::new(c);
                }
            }

            // All CVVDP variants failed; fall through to butteraugli.
            eprintln!(
                "[jxl-encoder cvvdp-fork P3/P5] CVVDP backend requested but \
                 unavailable (CUDA init failed, no CPU fallback compiled, \
                 or dims invalid); falling back to next dispatch tier ({}×{})",
                width, height,
            );
        } else if debug_log {
            eprintln!(
                "[cvvdp-fork P3] CVVDP not requested @ {}×{}, continuing dispatch",
                width, height
            );
        }
    }
    // cvvdp-loop OFF but cvvdp-loop-cpu ON: still try CPU CVVDP when
    // requested. This is the "CPU-only" host configuration (no CUDA,
    // no GPU butteraugli, but the caller still wants cvvdp).
    #[cfg(all(not(feature = "cvvdp-loop"), feature = "cvvdp-loop-cpu"))]
    {
        if cvvdp_requested {
            if debug_log {
                eprintln!(
                    "[cvvdp-fork P5] CVVDP requested @ {}×{} \
                     (cvvdp-loop feature off, cvvdp-loop-cpu on) — \
                     trying CPU CVVDP",
                    width, height
                );
            }
            if let Some(c) = crate::vardct::cvvdp_backend::cpu::CpuCvvdpBackend::try_new(
                width,
                height,
                selection.target_display,
            ) {
                if debug_log {
                    eprintln!(
                        "[cvvdp-fork P5] CPU CVVDP backend ACTIVE @ {}×{} \
                         (cvvdp-loop-cpu only)",
                        width, height
                    );
                }
                let _ = cpu_params;
                let _ = cvvdp_use_cpu_requested;
                return alloc::boxed::Box::new(c);
            }
            eprintln!(
                "[jxl-encoder cvvdp-fork P5] CVVDP requested but CPU \
                 construction failed (dims < 8×8?); falling back to \
                 butteraugli ({}×{})",
                width, height,
            );
        }
    }
    #[cfg(all(not(feature = "cvvdp-loop"), not(feature = "cvvdp-loop-cpu")))]
    if debug_log && cvvdp_requested {
        eprintln!(
            "[cvvdp-fork P3/P5] CVVDP requested but neither `cvvdp-loop` \
             nor `cvvdp-loop-cpu` cargo features are compiled in; \
             continuing dispatch ({}×{})",
            width, height
        );
    }
    #[cfg(feature = "gpu-butteraugli")]
    {
        if gpu_requested {
            if debug_log {
                eprintln!(
                    "[W44-phase3-B1] GPU requested @ {}×{} — trying CUDA init",
                    width, height
                );
            }
            // W44-PHASE3-B5b: if the detector env var is set, also
            // clone the cpu_params so the GPU backend can construct a
            // CPU shadow for the iter-0 divergence check. Default
            // (env unset) → `None` → no shadow → behaves exactly like
            // pre-W44-PHASE3-B5b.
            let detector_enabled = std::env::var(W44_PHASE3_B5B_ENV).ok().as_deref() == Some("1");
            let shadow_params = if detector_enabled {
                Some(cpu_params.clone())
            } else {
                None
            };
            if debug_log && detector_enabled {
                eprintln!(
                    "[W44-PHASE3-B5b] detector ENABLED via {}=1 — shadow CPU will be built",
                    W44_PHASE3_B5B_ENV
                );
            }
            if let Some(g) =
                gpu::GpuButteraugliBackend::try_new(width, height, intensity_target, shadow_params)
            {
                if debug_log {
                    eprintln!("[W44-phase3-B1] GPU backend ACTIVE @ {}×{}", width, height);
                }
                return alloc::boxed::Box::new(g);
            }
            // Fallback. Don't spam — single one-shot warning so users
            // notice GPU didn't fire.
            eprintln!(
                "[jxl-encoder W44-phase3-B1] GPU butteraugli requested but \
                 CUDA init failed; falling back to CPU backend ({}×{})",
                width, height,
            );
        } else if debug_log {
            eprintln!(
                "[W44-phase3-B1] GPU not requested @ {}×{}, using CPU backend",
                width, height
            );
        }
    }
    #[cfg(not(feature = "gpu-butteraugli"))]
    if debug_log && gpu_requested {
        eprintln!(
            "[W44-phase3-B1] GPU requested but cargo feature `gpu-butteraugli` \
             is OFF; using CPU backend ({}×{})",
            width, height
        );
    }
    let _ = (width, height);
    alloc::boxed::Box::new(CpuButteraugliBackend::new(cpu_params))
}

#[cfg(all(test, feature = "butteraugli-loop"))]
mod tests {
    use super::*;

    /// Smoke: CPU backend builds + reference roundtrips on a flat field.
    /// Identical reference == identical distorted should yield score ≈ 0.
    #[test]
    fn cpu_backend_identical_zero_score() {
        let w = 64usize;
        let h = 64usize;
        let n = w * h;
        let r = alloc::vec![0.5f32; n];
        let g = alloc::vec![0.5f32; n];
        let b = alloc::vec![0.5f32; n];
        let params = butteraugli::ButteraugliParams::new().with_compute_diffmap(true);
        let mut backend = CpuButteraugliBackend::new(params);
        backend.set_reference(&r, &g, &b, w, h).unwrap();
        let mut diffmap = alloc::vec::Vec::new();
        let result = backend
            .compare_with_reference(&r, &g, &b, w, w, h, &mut diffmap)
            .unwrap();
        assert!(
            result.score < 1e-4,
            "identical images should score ~0, got {}",
            result.score
        );
        assert_eq!(diffmap.len(), n);
    }

    /// Smoke: CPU backend produces non-zero diffmap on perturbed input,
    /// and the diffmap length equals width*height.
    #[test]
    fn cpu_backend_diffmap_size() {
        let w = 64usize;
        let h = 64usize;
        let n = w * h;
        let r = alloc::vec![0.5f32; n];
        let g = alloc::vec![0.5f32; n];
        let b = alloc::vec![0.5f32; n];
        let mut r2 = r.clone();
        // Inject a perturbation in the middle so butteraugli reports
        // something non-trivial.
        for y in 24..40 {
            for x in 24..40 {
                r2[y * w + x] = 0.9;
            }
        }
        let params = butteraugli::ButteraugliParams::new().with_compute_diffmap(true);
        let mut backend = CpuButteraugliBackend::new(params);
        backend.set_reference(&r, &g, &b, w, h).unwrap();
        let mut diffmap = alloc::vec::Vec::new();
        let result = backend
            .compare_with_reference(&r2, &g, &b, w, w, h, &mut diffmap)
            .unwrap();
        assert_eq!(diffmap.len(), n);
        // Sanity — perturbation should produce a clearly non-zero score.
        assert!(
            result.score > 0.01,
            "perturbed image should score > 0.01, got {}",
            result.score
        );
    }

    /// B7a regression: calling compare_with_reference twice on the same
    /// backend must reuse the caller-owned diffmap Vec across calls
    /// (capacity should not grow between iters).
    #[test]
    fn cpu_backend_diffmap_recycles_across_calls() {
        let w = 32usize;
        let h = 32usize;
        let n = w * h;
        let r = alloc::vec![0.5f32; n];
        let g = alloc::vec![0.5f32; n];
        let b = alloc::vec![0.5f32; n];
        let params = butteraugli::ButteraugliParams::new().with_compute_diffmap(true);
        let mut backend = CpuButteraugliBackend::new(params);
        backend.set_reference(&r, &g, &b, w, h).unwrap();
        let mut diffmap = alloc::vec::Vec::new();
        let _ = backend
            .compare_with_reference(&r, &g, &b, w, w, h, &mut diffmap)
            .unwrap();
        let cap_after_first = diffmap.capacity();
        assert!(cap_after_first >= n);
        let _ = backend
            .compare_with_reference(&r, &g, &b, w, w, h, &mut diffmap)
            .unwrap();
        // Second call must not grow the buffer.
        assert_eq!(diffmap.capacity(), cap_after_first);
        assert_eq!(diffmap.len(), n);
    }

    #[test]
    fn cpu_backend_name() {
        let params = butteraugli::ButteraugliParams::new().with_compute_diffmap(true);
        let backend = CpuButteraugliBackend::new(params);
        assert_eq!(backend.name(), "cpu");
    }

    #[test]
    fn construct_backend_cpu_when_gpu_not_requested() {
        let params = butteraugli::ButteraugliParams::new().with_compute_diffmap(true);
        // Multi-metric Phase 0 (RFC #3 §4): construct_backend now takes
        // a single bundled `MetricSelection` struct instead of the
        // four trailing bools. `Butteraugli + Cpu` mirrors the
        // production default (`LossyConfig::default()` produces
        // `Butteraugli + Auto`, which resolves to CPU when
        // `gpu-butteraugli` is not compiled in — same baseline as
        // pre-Phase-0).
        let backend = construct_backend(
            64,
            64,
            params,
            80.0,
            MetricSelection {
                metric: PerceptualMetric::Butteraugli,
                device: PerceptualDevice::Cpu,
                target_score: None,
                target_display: DisplayConfig::WebSdr80,
            },
        );
        assert_eq!(backend.name(), "cpu");
    }

    /// Multi-metric Phase 0: when `metric == Butteraugli` AND
    /// `device == Cpu`, the dispatch returns the CPU butteraugli
    /// backend regardless of the `cvvdp-loop` cargo feature.
    /// Verifies the cvvdp branch doesn't accidentally fire when the
    /// caller's metric is Butteraugli.
    #[cfg(feature = "cvvdp-loop")]
    #[test]
    fn construct_backend_cpu_when_cvvdp_not_requested() {
        let params = butteraugli::ButteraugliParams::new().with_compute_diffmap(true);
        let backend = construct_backend(
            64,
            64,
            params,
            80.0,
            MetricSelection {
                metric: PerceptualMetric::Butteraugli,
                device: PerceptualDevice::Cpu,
                target_score: None,
                target_display: DisplayConfig::WebSdr80,
            },
        );
        assert_eq!(backend.name(), "cpu");
    }

    /// Multi-metric Phase 0 (RFC #3 §4): when `metric == Cvvdp` AND
    /// `device == Cpu` AND `cvvdp-loop-cpu` is compiled in, the
    /// dispatch returns the CPU CVVDP backend. We can only assert the
    /// backend's `name()` ends up as `"cvvdp-cpu"`; on hosts without
    /// CUDA the GPU fallback path would still be unreachable (we
    /// asked for CPU explicitly).
    #[cfg(feature = "cvvdp-loop-cpu")]
    #[test]
    fn construct_backend_cvvdp_cpu_when_use_cpu_requested() {
        let params = butteraugli::ButteraugliParams::new().with_compute_diffmap(true);
        let backend = construct_backend(
            64,
            64,
            params,
            80.0,
            MetricSelection {
                metric: PerceptualMetric::Cvvdp,
                device: PerceptualDevice::Cpu,
                target_score: None,
                target_display: DisplayConfig::WebSdr80,
            },
        );
        assert_eq!(
            backend.name(),
            "cvvdp-cpu",
            "explicit CPU CVVDP opt-in must return the CPU CVVDP backend \
             on a 64×64 (≥ 8×8) buffer"
        );
    }

    /// Multi-metric Phase 0: when `metric == Cvvdp` AND `device == Auto`
    /// (default-policy GPU first), the dispatch returns either:
    /// - `"cvvdp-gpu-cuda"` on hosts with CUDA, OR
    /// - `"cvvdp-cpu"` on hosts without CUDA but with `cvvdp-loop-cpu`
    ///   compiled in (Phase 5 silent fallback per the dispatch matrix).
    ///
    /// Either is acceptable; the test fails only if the backend name
    /// is `"cpu"` (= butteraugli fell through, which means the cvvdp
    /// fallback chain didn't fire).
    #[cfg(feature = "cvvdp-loop-cpu")]
    #[test]
    fn construct_backend_cvvdp_falls_back_to_cpu_when_no_cuda() {
        let params = butteraugli::ButteraugliParams::new().with_compute_diffmap(true);
        let backend = construct_backend(
            64,
            64,
            params,
            80.0,
            MetricSelection {
                metric: PerceptualMetric::Cvvdp,
                device: PerceptualDevice::Auto,
                target_score: None,
                target_display: DisplayConfig::WebSdr80,
            },
        );
        let name = backend.name();
        assert!(
            name == "cvvdp-gpu-cuda" || name == "cvvdp-cpu",
            "Cvvdp + Auto with cvvdp-loop-cpu compiled must \
             land on a CVVDP backend (GPU when CUDA OK, CPU otherwise); \
             got {name}"
        );
    }

    /// W44-PHASE3-B5b: CPU backend never reports divergence (the
    /// detector is GPU-only — CPU is the reference).
    #[test]
    fn cpu_backend_divergence_status_always_none() {
        let w = 32usize;
        let h = 32usize;
        let n = w * h;
        let r = alloc::vec![0.5f32; n];
        let g = alloc::vec![0.5f32; n];
        let b = alloc::vec![0.5f32; n];
        let params = butteraugli::ButteraugliParams::new().with_compute_diffmap(true);
        let mut backend = CpuButteraugliBackend::new(params);
        backend.set_reference(&r, &g, &b, w, h).unwrap();
        let mut diffmap = alloc::vec::Vec::new();
        let _ = backend
            .compare_with_reference(&r, &g, &b, w, w, h, &mut diffmap)
            .unwrap();
        // Trait default impl returns None.
        assert!(backend.divergence_status().is_none());
    }

    /// W44-PHASE3-B5b: counters reset works and exposes zero state.
    #[cfg(feature = "gpu-butteraugli")]
    #[test]
    fn b5b_counters_reset_zero_state() {
        super::b5b_counters::reset();
        let snap = super::b5b_counters::snapshot();
        assert_eq!(snap.run_count, 0);
        assert_eq!(snap.fallback_count, 0);
        assert_eq!(snap.divergence_pct_sum, 0.0);
        assert_eq!(snap.divergence_pct_max, 0.0);
    }

    /// W44-PHASE3-B5b: counters record + snapshot round-trip.
    #[cfg(feature = "gpu-butteraugli")]
    #[test]
    fn b5b_counters_record_round_trip() {
        super::b5b_counters::reset();
        // 3 observations: 0.001 (no trip), 0.003 (no trip), 0.010 (trip)
        super::b5b_counters::record(0.001, false);
        super::b5b_counters::record(0.003, false);
        super::b5b_counters::record(0.010, true);
        let snap = super::b5b_counters::snapshot();
        assert_eq!(snap.run_count, 3);
        assert_eq!(snap.fallback_count, 1);
        // Sum: 0.001 + 0.003 + 0.010 = 0.014
        assert!((snap.divergence_pct_sum - 0.014).abs() < 1e-6);
        // Max: 0.010
        assert!((snap.divergence_pct_max - 0.010).abs() < 1e-6);
        super::b5b_counters::reset();
    }

    /// W44-PHASE3-B5b: the threshold constant is the documented 0.5%.
    #[cfg(feature = "gpu-butteraugli")]
    #[test]
    fn b5b_threshold_constant_is_0_5_pct() {
        assert!((super::GPU_SCORE_DIVERGENCE_PCT - 0.005).abs() < f64::EPSILON);
        assert_eq!(super::W44_PHASE3_B5B_ENV, "JXL_W44_PHASE3_B5B_DETECTOR");
    }
}
