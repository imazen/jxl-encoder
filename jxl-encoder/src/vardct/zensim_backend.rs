// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! zensim-based [`PerceptualBackend`] implementations (zensim-fork
//! Phase 3, 2026-05-25 — see `docs/RFC_ZENSIM_FORK_PLAN.md` §5,
//! `docs/RFC_ZENSIM_BUTTLOOP_AUDIT.md`, and `docs/RFC_MULTI_METRIC_PERCEPTUAL_BACKEND.md`).
//!
//! Phase 3 ships the backend impl + the opt-in dispatch only. The
//! buttloop body still consumes butteraugli-direction targets; Phase 4
//! (separate chunk) plumbs the zensim signal through `run_buttloop` via
//! `vardct/zensim_targets.rs`. Hash-locks therefore stay byte-identical
//! at default features AND when a zensim feature is enabled but no
//! caller has opted in via
//! [`LossyConfig::with_perceptual_metric`](crate::api::LossyConfig::with_perceptual_metric)`(PerceptualMetric::Zensim)`.
//!
//! ## Backends
//!
//! - [`cpu::CpuZensimBackend`] (feature `zensim-loop`) — wraps
//!   `zensim::Zensim` + `precompute_reference_linear_planar` +
//!   `compute_with_ref_and_diffmap_linear_planar`. CPU implementation,
//!   no CUDA. Always available when the `zensim-loop` feature is
//!   compiled.
//! - [`gpu::GpuZensimBackend`] (feature `zensim-loop-gpu`) — wraps
//!   `zensim_gpu::ZensimOpaque` via the Phase 1 (`1175b49`)
//!   `*_from_linear_planes_*` API surface. The current zensim-gpu
//!   diffmap implementation delegates to the canonical CPU pipeline
//!   (Phase 1 honest-stop, see `crates/zensim-gpu/docs/DIFFMAP_DIVERGENCES.md`);
//!   the GPU backend is wired through the same trait shape for
//!   forward-compatibility with the Phase 1b pure-GPU kernel chain.
//!
//! ## Score-direction normalization
//!
//! zensim's native score lives in `[0, 100]` with `100 = identical`
//! (higher = better). Butteraugli direction is `smaller = better, 0 =
//! identical`. Conversion at the trait boundary:
//!
//! ```text
//! butter_score = (100.0 - zensim_score).clamp(0.0, 100.0)
//! ```
//!
//! The clamp protects against rare numerical overshoots above 100 on
//! identical-plus-noise inputs.
//!
//! ## When the zensim backends are active
//!
//! 1. Caller sets
//!    [`LossyConfig::with_perceptual_metric`](crate::api::LossyConfig::with_perceptual_metric)`(PerceptualMetric::Zensim)`
//!    AND
//! 2. A zensim cargo feature (`zensim-loop` for CPU,
//!    `zensim-loop-gpu` for GPU) was enabled at build time AND
//! 3. (GPU) The CUDA runtime initialised successfully at backend
//!    construction AND
//! 4. The active strategy is NOT
//!    [`EncoderStrategy::Libjxl`](crate::api::EncoderStrategy::Libjxl)
//!    (the strict cjxl-parity invariant forces butteraugli regardless
//!    of the field — see
//!    [`LossyConfig::resolve_perceptual_metric`](crate::api::LossyConfig::resolve_perceptual_metric)).
//!
//! If any of (1-4) fail, the buttloop falls back to the next dispatch
//! tier silently (CPU zensim → CPU butteraugli when `zensim-loop-gpu`
//! is selected but CUDA missing; CPU butteraugli when no zensim
//! feature is on; CPU butteraugli when Libjxl strategy is active).

#![cfg(any(feature = "zensim-loop", feature = "zensim-loop-gpu"))]

use alloc::format;

use crate::error::Result;

use super::perceptual_backend::{BackendCompareResult, PerceptualBackend};

// ============================================================================
// Score-direction normalization helper (shared CPU + GPU)
// ============================================================================

/// Convert zensim's native higher-is-better `[0, 100]` score to a
/// butteraugli-direction normalised score (smaller=better, `0 =
/// identical`) per RFC #1 §1.1 + `RFC_ZENSIM_BUTTLOOP_AUDIT.md` §1.1.
///
/// The clamp guards against rare numerical overshoots above 100 on
/// identical-plus-noise inputs (e.g. when the f32 reduction-order
/// noise on a flat-field comparison drifts the score by a few ULP
/// above the algebraic ceiling).
#[inline]
fn zensim_score_to_butter_direction(zensim_score: f64) -> f64 {
    (100.0_f64 - zensim_score).clamp(0.0, 100.0)
}

// ============================================================================
// CPU zensim backend — `zensim-loop` feature
// ============================================================================

#[cfg(feature = "zensim-loop")]
pub(crate) mod cpu {
    //! CPU zensim backend (pure Rust, no CUDA).
    //!
    //! Constructed on demand by
    //! [`super::super::perceptual_backend::construct_backend`] when the
    //! caller has opted in via
    //! [`crate::api::LossyConfig::with_perceptual_metric`]`(PerceptualMetric::Zensim)`
    //! AND (a) explicitly preferred CPU via
    //! [`crate::api::LossyConfig::with_perceptual_device`]`(PerceptualDevice::Cpu)`,
    //! OR (b) the GPU zensim backend is unavailable (CUDA missing /
    //! `zensim-loop-gpu` feature off / CUDA init fails — defense-in-depth
    //! fallback).
    //!
    //! ## API surface this wraps (zensim 0.2.7+)
    //!
    //! - [`zensim::Zensim::new`] — construct a scorer for a chosen
    //!   [`zensim::ZensimProfile`]. zensim 0.3+ ships only
    //!   [`zensim::ZensimProfile::A`] (the canonical codec-target profile);
    //!   the legacy linear `PreviewV0_1` / `PreviewV0_2` variants were
    //!   removed. This backend pins `A`.
    //! - [`zensim::Zensim::precompute_reference_linear_planar`] —
    //!   build the per-cell `PrecomputedReference` ONCE from the
    //!   linear-RGB planes the buttloop hands us (no sRGB byte
    //!   round-trip on the host).
    //! - [`zensim::Zensim::compute_with_ref_and_diffmap_linear_planar`]
    //!   — run a compare against the precomputed reference, returning
    //!   a [`zensim::DiffmapResult`] carrying the scalar score + a
    //!   `Vec<f32>` per-pixel diffmap. We copy the diffmap into the
    //!   caller-owned `Vec<f32>` via `clear() + extend_from_slice`
    //!   (B7a recycling). The per-iter ~`W*H*4` extra bytes is dwarfed
    //!   by zensim's internal multi-scale buffers.
    //!
    //! ## Phase 1 honest-stop carryover
    //!
    //! zensim 0.2.7 has no `_into` variant of
    //! `compute_with_ref_and_diffmap_linear_planar` that fills a
    //! caller-owned `Vec<f32>`. Per `RFC_ZENSIM_BUTTLOOP_AUDIT.md`
    //! §3.2.1 we ship Option B (wrap inside the backend, copy the
    //! diffmap out). Adds one O(W*H) memcpy per iter — ~1-2 ms at
    //! 1024², negligible vs zensim compute. Option A (zensim-side
    //! `_into` variant) is a queued zensim 0.3+ follow-on; not
    //! required for Phase 3 opt-in shipment.

    use super::*;

    use alloc::vec::Vec;
    use zensim::{DiffmapOptions, PrecomputedReference, Zensim, ZensimProfile};

    /// Pure-Rust CPU zensim backend. Wraps [`Zensim`] + a per-cell
    /// [`PrecomputedReference`].
    ///
    /// One instance is scoped to a single buttloop cell. The reference
    /// is precomputed on [`PerceptualBackend::set_reference`]; each
    /// [`PerceptualBackend::compare_with_reference`] call runs the
    /// distorted side through the warm-ref pipeline.
    pub(crate) struct CpuZensimBackend {
        /// The scorer (cheap to construct; carries profile + parallel +
        /// max_pixels config). Kept in `Option` so we can rebuild
        /// across `set_reference` calls without dragging the
        /// `PrecomputedReference` lifetime through self-referential
        /// territory.
        scorer: Zensim,
        /// Tight-stride f32 scratch for the reference planes. Filled
        /// by `set_reference`; passed as `&[f32]` to
        /// `precompute_reference_linear_planar`.
        ref_planes: [Vec<f32>; 3],
        /// Tight-stride f32 scratch for the distorted side. The
        /// buttloop hands us strided `recon_r/g/b` with
        /// `padded_width >= width`; we copy each row into this buffer
        /// before passing the slice to
        /// `compute_with_ref_and_diffmap_linear_planar` because the
        /// linear-planar API expects `stride >= width`. We normalise
        /// to `stride == width` to keep the code path uniform and the
        /// buffer allocation amortised across iters.
        dist_plane_scratch: [Vec<f32>; 3],
        /// Per-cell precomputed reference. Built on `set_reference`;
        /// reused on every compare call.
        precomputed: Option<PrecomputedReference>,
        width: u32,
        height: u32,
        /// Phase 8b: per-instance compare-call counter (bench-only
        /// dump; mirrors the cvvdp backend shape so callers wiring
        /// `JXL_PHASE8B_DIFFMAP_DUMP` see Z_CPU tagged rows alongside
        /// C_CPU / C_GPU).
        compare_call_count: u32,
    }

    impl core::fmt::Debug for CpuZensimBackend {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.debug_struct("CpuZensimBackend")
                .field("width", &self.width)
                .field("height", &self.height)
                .field("reference_warmed", &self.precomputed.is_some())
                .finish()
        }
    }

    impl CpuZensimBackend {
        /// Construct a CPU zensim backend for `width × height`. Returns
        /// `None` if the dimensions reject (only fails when `min(w, h)
        /// < 8` — zensim's pyramid minimum). CPU construction never
        /// panics (no driver init, no GPU runtime — pure Rust + alloc).
        /// We still wrap the size-times-channels checked-multiply in a
        /// guard so a u32 overflow returns `None` cleanly.
        pub(crate) fn try_new(width: u32, height: u32) -> Option<Self> {
            if width < 8 || height < 8 {
                return None;
            }
            let n = (width as usize).checked_mul(height as usize)?;
            // Profile A (the v47-strict-QAT bake, external name `zensim-a`) — the
            // standardized profile as of the 2026-07-01 "ban PreviewV0_2, A only"
            // directive. The local zensim 0.3.x (`../zensim/zensim`) provides it, and
            // it is the profile the canonical training data + zenmetrics production
            // scoring now use (`latest_preview()` also resolves to A). Pinned
            // explicitly rather than via `latest_preview()` so this encoder's
            // zensim-loop RD target stays reproducible across zensim revisions.
            // zensim deprecated `A` in favour of `B` (2026-07); staying on `A`
            // is deliberate until the Phase 8-zensim recalibration re-seeds the
            // target table against a new profile.
            //
            // RD-experiment override (2026-07-18, diffmap-RD worktree): the
            // profile is selectable via `JXL_ZENSIM_RD_PROFILE=a|b|latest` so
            // the target-zensim RD eval can compare loop drivers WITHOUT
            // changing the shipped default (unset → A, hash-locks intact).
            let scorer = match std::env::var("JXL_ZENSIM_RD_PROFILE").as_deref() {
                Ok("b") => Zensim::new(ZensimProfile::B),
                Ok("latest") => Zensim::new(ZensimProfile::latest_preview()),
                _ => {
                    #[allow(deprecated)]
                    Zensim::new(ZensimProfile::A)
                }
            };
            Some(Self {
                scorer,
                ref_planes: [
                    alloc::vec![0.0f32; n],
                    alloc::vec![0.0f32; n],
                    alloc::vec![0.0f32; n],
                ],
                dist_plane_scratch: [
                    alloc::vec![0.0f32; n],
                    alloc::vec![0.0f32; n],
                    alloc::vec![0.0f32; n],
                ],
                precomputed: None,
                width,
                height,
                compare_call_count: 0,
            })
        }

        /// Copy one strided plane into a tight-stride destination.
        /// Mirrors `CpuCvvdpBackend::copy_strided_row_into_scratch` —
        /// fast-path when `padded_width == width`, per-row copy
        /// otherwise.
        fn copy_strided_row_into_scratch(
            scratch: &mut [f32],
            src: &[f32],
            padded_width: usize,
            width: usize,
            height: usize,
        ) {
            debug_assert_eq!(scratch.len(), width * height);
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

    impl PerceptualBackend for CpuZensimBackend {
        fn name(&self) -> &'static str {
            "zensim-cpu"
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
                    "CPU zensim backend: dim mismatch in set_reference: \
                     expected {}×{}, got {}×{}",
                    self.width, self.height, width, height,
                )));
            }
            let n = width * height;
            if ref_r.len() < n || ref_g.len() < n || ref_b.len() < n {
                return Err(crate::error::Error::InvalidInput(format!(
                    "CPU zensim backend: reference plane too short: \
                     expected {}, got R={} G={} B={}",
                    n,
                    ref_r.len(),
                    ref_g.len(),
                    ref_b.len(),
                )));
            }
            // Reference is tight (trait contract; `set_reference`
            // doesn't take a stride). Copy into our cached buffers
            // (zensim's API takes `&[f32]` per plane; we hold them
            // owned so the `PrecomputedReference` it produces stays
            // valid across compare iters).
            self.ref_planes[0].clear();
            self.ref_planes[1].clear();
            self.ref_planes[2].clear();
            self.ref_planes[0].extend_from_slice(&ref_r[..n]);
            self.ref_planes[1].extend_from_slice(&ref_g[..n]);
            self.ref_planes[2].extend_from_slice(&ref_b[..n]);

            let pre = self
                .scorer
                .precompute_reference_linear_planar(
                    [
                        &self.ref_planes[0],
                        &self.ref_planes[1],
                        &self.ref_planes[2],
                    ],
                    width,
                    height,
                    width, // tight stride
                )
                .map_err(|e| {
                    crate::error::Error::InvalidInput(format!(
                        "CPU zensim precompute_reference_linear_planar: {e}"
                    ))
                })?;
            self.precomputed = Some(pre);
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
            diffmap_out: &mut Vec<f32>,
        ) -> Result<BackendCompareResult> {
            let pre = self.precomputed.as_ref().ok_or_else(|| {
                crate::error::Error::InvalidInput(
                    "CPU zensim backend: compare_with_reference called \
                     before set_reference"
                        .into(),
                )
            })?;
            if width as u32 != self.width || height as u32 != self.height {
                return Err(crate::error::Error::InvalidInput(format!(
                    "CPU zensim backend: dim mismatch in compare: \
                     expected {}×{}, got {}×{}",
                    self.width, self.height, width, height,
                )));
            }

            // Strided → tight copy into the per-instance scratch.
            let [s_r, s_g, s_b] = &mut self.dist_plane_scratch;
            Self::copy_strided_row_into_scratch(s_r, dist_r, padded_width, width, height);
            Self::copy_strided_row_into_scratch(s_g, dist_g, padded_width, width, height);
            Self::copy_strided_row_into_scratch(s_b, dist_b, padded_width, width, height);

            // Call zensim's diffmap-bearing linear-planar entry point
            // with tight stride (we already normalised the dist
            // scratch above; the precomputed reference was also built
            // tight).
            let result = self
                .scorer
                .compute_with_ref_and_diffmap_linear_planar(
                    pre,
                    [
                        &self.dist_plane_scratch[0],
                        &self.dist_plane_scratch[1],
                        &self.dist_plane_scratch[2],
                    ],
                    width,
                    height,
                    width, // tight stride
                    // RD-experiment knob (2026-07-18): `JXL_ZENSIM_DIFFMAP_SIGNALS=all`
                    // turns on the edge/mse/hf per-pixel signals (the coherence
                    // matrix showed the ssim-only default is part of the
                    // steering bottleneck). Unset → default (shipped behavior).
                    if std::env::var("JXL_ZENSIM_DIFFMAP_SIGNALS").as_deref() == Ok("all") {
                        DiffmapOptions {
                            include_edge_mse: true,
                            include_hf: true,
                            ..DiffmapOptions::default()
                        }
                    } else {
                        DiffmapOptions::default()
                    },
                )
                .map_err(|e| {
                    crate::error::Error::InvalidInput(format!(
                        "CPU zensim compute_with_ref_and_diffmap_linear_planar: {e}"
                    ))
                })?;

            // Copy zensim's owned diffmap into the caller-owned Vec.
            // Phase 1 honest-stop carryover: zensim 0.2.7 has no
            // `_into` variant; this `clear + extend_from_slice` is
            // the Option B wrap from RFC #2 §3.2.1. Amortised-O(1)
            // once the capacity high-watermark is reached because the
            // buttloop reuses `diffmap_out` across iters.
            let zensim_diffmap = result.diffmap();
            debug_assert_eq!(zensim_diffmap.len(), width * height);
            diffmap_out.clear();
            diffmap_out.extend_from_slice(zensim_diffmap);

            // Direction-normalize zensim's native higher-is-better score
            // to the butteraugli-direction lower-is-better surface the
            // buttloop's `target_distance` comparison expects.
            let zensim_score = result.score();
            let score = zensim_score_to_butter_direction(zensim_score);

            // Phase 8b: pre-renorm dump (mirrors C_CPU shape so harness
            // TSVs from the cvvdp arc can scrape Z_CPU rows the same way).
            super::super::perceptual_backend::maybe_dump_diffmap_stats(
                "Z_CPU_PRE",
                self.compare_call_count,
                width,
                height,
                diffmap_out,
                score,
            );

            // Phase 8c-equivalent: renormalize the zensim diffmap before
            // returning. Phase 3 ships the placeholder `1.0` constant
            // (no renorm) per the RFC #1 §4 + RFC §5 task spec ("DO NOT
            // set the CVVDP_DIFFMAP_RENORM_SCALE-equivalent away from
            // 1.0 yet — Phase 4 fits this"). When the constant departs
            // from 1.0, the `(renorm - 1.0).abs() > EPSILON` guard
            // skips the whole multiplication pass (zero overhead at
            // unit scale).
            let renorm = super::resolved_zensim_diffmap_renorm_scale();
            if (renorm - 1.0).abs() > f32::EPSILON {
                for v in diffmap_out.iter_mut() {
                    *v *= renorm;
                }
            }

            // Phase 8b: post-renorm dump.
            super::super::perceptual_backend::maybe_dump_diffmap_stats(
                "Z_CPU_POST",
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
}

// ============================================================================
// GPU zensim backend — `zensim-loop-gpu` feature
// ============================================================================

#[cfg(feature = "zensim-loop-gpu")]
pub(crate) mod gpu {
    //! GPU zensim backend (CUDA via CubeCL).
    //!
    //! Constructed on demand by
    //! [`super::super::perceptual_backend::construct_backend`]. If CUDA
    //! init fails (e.g. no GPU, no driver), [`GpuZensimBackend::try_new`]
    //! returns `None` and the caller falls back to the next dispatch
    //! tier (CPU zensim when `zensim-loop` is compiled; CPU
    //! butteraugli otherwise).
    //!
    //! ## API surface this wraps (zenmetrics master `1175b49`)
    //!
    //! - [`zensim_gpu::ZensimOpaque::new`] — single-resolution Backend
    //!   selector (`zensim_gpu::Backend::Cuda`).
    //! - [`zensim_gpu::ZensimOpaque::warm_reference_from_linear_planes`]
    //!   — caches a linear-RGB reference for many subsequent compares.
    //! - [`zensim_gpu::ZensimOpaque::score_from_linear_planes_with_warm_ref_diffmap`]
    //!   — runs a compare against the warm reference, filling a
    //!   caller-owned `Vec<f32>` diffmap.
    //!
    //! All three methods take linear-f32 planes (no sRGB byte-pack on
    //! the host).
    //!
    //! ## Phase 1 honest-stop carryover (zensim-gpu side)
    //!
    //! The Phase 1 zensim-gpu diffmap kernel chain delegates to the
    //! canonical zensim CPU pipeline (see
    //! `crates/zensim-gpu/docs/DIFFMAP_DIVERGENCES.md`). Per-iter wall
    //! overhead at 1024² is ~+1006% vs the score-only GPU path,
    //! making GPU zensim with diffmap currently SLOWER than CPU
    //! zensim. Phase 1b (pure-GPU kernels) is the path to closing
    //! this; the current overhead is acceptable for Phase 3 opt-in
    //! mode. Callers prioritising wall time should prefer
    //! [`crate::api::PerceptualDevice::Cpu`] until Phase 1b lands.
    //!
    //! ## Phase 1 zensim-gpu working-tree caveat (2026-05-25)
    //!
    //! At Phase 3 implementation time, the path-pinned
    //! `~/work/zen/zenmetrics/crates/zensim-gpu` working tree was at
    //! `f4cf509b` (parent of `1175b49`); the Phase 1 source must be
    //! present on disk for `zensim-loop-gpu` to build. The build
    //! verifies this via the `score_from_linear_planes_with_warm_ref_diffmap`
    //! call below. If the build fails at this site, the operator
    //! needs to update the zenmetrics checkout to `master` (or a
    //! descendant) before the Phase 3 GPU path compiles.

    use super::*;

    use alloc::vec::Vec;
    use zensim_gpu::{Backend as ZensimBackend, ZensimOpaque, ZensimParams};

    /// CUDA-backed zensim backend. Wraps [`ZensimOpaque`] and uploads
    /// host-side linear-f32 planar input directly via the
    /// `*_from_linear_planes_*` API surface.
    ///
    /// One instance is scoped to a single buttloop cell: the reference
    /// is set once via [`PerceptualBackend::set_reference`] then
    /// [`PerceptualBackend::compare_with_reference`] is called once
    /// per buttloop iteration with the current iter's reconstructed
    /// planes.
    pub(crate) struct GpuZensimBackend {
        inner: ZensimOpaque,
        /// Tight-stride f32 scratch for the distorted side's R/G/B
        /// planes. Mirrors `GpuCvvdpBackend::dist_plane_scratch` —
        /// the linear-planes API expects tight `width × height`
        /// planes (no padding); we copy each row before the call to
        /// `score_from_linear_planes_with_warm_ref_diffmap`.
        dist_plane_scratch: [Vec<f32>; 3],
        width: u32,
        height: u32,
        reference_warmed: bool,
        /// Phase 8b: per-instance compare-call counter (bench-only).
        compare_call_count: u32,
    }

    impl core::fmt::Debug for GpuZensimBackend {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.debug_struct("GpuZensimBackend")
                .field("width", &self.width)
                .field("height", &self.height)
                .field("reference_warmed", &self.reference_warmed)
                .finish()
        }
    }

    impl GpuZensimBackend {
        /// Construct a GPU zensim backend for `width × height`.
        /// Returns `None` if the CUDA runtime fails to initialise
        /// (e.g. no GPU, no driver, panic inside CubeCL). Mirrors the
        /// `GpuButteraugliBackend::try_new` / `GpuCvvdpBackend::try_new`
        /// defense-in-depth pattern: `catch_unwind` around the
        /// constructor so a missing CUDA driver never aborts the
        /// encode.
        ///
        /// zensim doesn't expose an `intensity_target` knob — its
        /// model is SDR-only (sRGB / BT.709 / linear with sRGB-aware
        /// gamut mapping; see `RFC_ZENSIM_BUTTLOOP_AUDIT.md` §3.4).
        /// HDR encodes should not opt into zensim; the resolver
        /// currently doesn't short-circuit HDR → butteraugli but the
        /// caller's `LossyConfig::resolve_hdr_loss` upstream typically
        /// catches PQ / HLG before this backend gets constructed.
        pub(crate) fn try_new(width: u32, height: u32) -> Option<Self> {
            let inner = std::panic::catch_unwind(|| {
                ZensimOpaque::new(ZensimBackend::Cuda, width, height, ZensimParams::default())
            });
            let inner = match inner {
                Ok(Ok(c)) => c,
                Ok(Err(_)) => return None,
                Err(_) => return None,
            };

            let n = (width as usize).checked_mul(height as usize)?;
            Some(Self {
                inner,
                dist_plane_scratch: [
                    alloc::vec![0.0f32; n],
                    alloc::vec![0.0f32; n],
                    alloc::vec![0.0f32; n],
                ],
                width,
                height,
                reference_warmed: false,
                compare_call_count: 0,
            })
        }

        /// Copy one strided plane into the tight scratch slot. Mirrors
        /// `GpuCvvdpBackend::copy_strided_row_into_scratch`.
        fn copy_strided_row_into_scratch(
            scratch: &mut [f32],
            src: &[f32],
            padded_width: usize,
            width: usize,
            height: usize,
        ) {
            debug_assert_eq!(scratch.len(), width * height);
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

    impl PerceptualBackend for GpuZensimBackend {
        fn name(&self) -> &'static str {
            "zensim-gpu-cuda"
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
                    "GPU zensim backend: dim mismatch in set_reference: \
                     expected {}×{}, got {}×{}",
                    self.width, self.height, width, height,
                )));
            }
            let n = width * height;
            if ref_r.len() < n || ref_g.len() < n || ref_b.len() < n {
                return Err(crate::error::Error::InvalidInput(format!(
                    "GPU zensim backend: reference plane too short: \
                     expected {}, got R={} G={} B={}",
                    n,
                    ref_r.len(),
                    ref_g.len(),
                    ref_b.len(),
                )));
            }
            // Reference is tight by the trait contract.
            self.inner
                .warm_reference_from_linear_planes(&ref_r[..n], &ref_g[..n], &ref_b[..n])
                .map_err(|e| {
                    crate::error::Error::InvalidInput(format!(
                        "GPU zensim warm_reference_from_linear_planes: {e}"
                    ))
                })?;
            self.reference_warmed = true;
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
            diffmap_out: &mut Vec<f32>,
        ) -> Result<BackendCompareResult> {
            if !self.reference_warmed {
                return Err(crate::error::Error::InvalidInput(
                    "GPU zensim backend: compare_with_reference called \
                     before set_reference"
                        .into(),
                ));
            }
            if width as u32 != self.width || height as u32 != self.height {
                return Err(crate::error::Error::InvalidInput(format!(
                    "GPU zensim backend: dim mismatch in compare: \
                     expected {}×{}, got {}×{}",
                    self.width, self.height, width, height,
                )));
            }

            // Strided → tight copy.
            let [s_r, s_g, s_b] = &mut self.dist_plane_scratch;
            Self::copy_strided_row_into_scratch(s_r, dist_r, padded_width, width, height);
            Self::copy_strided_row_into_scratch(s_g, dist_g, padded_width, width, height);
            Self::copy_strided_row_into_scratch(s_b, dist_b, padded_width, width, height);

            // The Phase 1 zensim-gpu API takes a `&mut Vec<f32>` that
            // it `clear()`s + extends with the diffmap (semantics
            // matching butteraugli-gpu and cvvdp-gpu B7a recycling).
            // We pre-clear here for explicit ownership semantics; the
            // upstream API would `clear()` anyway.
            diffmap_out.clear();
            // Pre-reserve capacity so the upstream extend doesn't
            // reallocate when the high-watermark is first reached.
            let needed = width * height;
            diffmap_out.reserve(needed);

            // Phase 1 returns the butter-direction normalized f32
            // score directly (see Phase 1 helper
            // `normalize_zensim_score` at zenmetrics
            // `crates/zensim-gpu/src/pipeline.rs:1651`). We promote
            // to f64 for the `BackendCompareResult` contract.
            let score_normalized = self
                .inner
                .score_from_linear_planes_with_warm_ref_diffmap(
                    &self.dist_plane_scratch[0],
                    &self.dist_plane_scratch[1],
                    &self.dist_plane_scratch[2],
                    diffmap_out,
                )
                .map_err(|e| {
                    crate::error::Error::InvalidInput(format!(
                        "GPU zensim score_from_linear_planes_with_warm_ref_diffmap: {e}"
                    ))
                })?;
            debug_assert_eq!(diffmap_out.len(), needed);
            // The Phase 1 API already applies the `(100 - score)
            // .clamp(0, 100)` direction normalization internally, so
            // we promote to f64 verbatim. Defense-in-depth: clamp
            // again here in case the upstream contract drifts in a
            // future zensim-gpu version.
            let score = (score_normalized as f64).clamp(0.0, 100.0);

            // Phase 8b: pre-renorm dump.
            super::super::perceptual_backend::maybe_dump_diffmap_stats(
                "Z_GPU_PRE",
                self.compare_call_count,
                width,
                height,
                diffmap_out,
                score,
            );

            // Phase 8c-equivalent renorm (placeholder 1.0; Phase 4 fits).
            let renorm = super::resolved_zensim_diffmap_renorm_scale();
            if (renorm - 1.0).abs() > f32::EPSILON {
                for v in diffmap_out.iter_mut() {
                    *v *= renorm;
                }
            }

            // Phase 8b: post-renorm dump.
            super::super::perceptual_backend::maybe_dump_diffmap_stats(
                "Z_GPU_POST",
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
}

// ============================================================================
// Diffmap renormalization scale (placeholder for Phase 4 fit)
// ============================================================================

/// zensim-fork Phase 3 (RFC #3 §7 + RFC_ZENSIM_FORK_PLAN.md §10.2,
/// 2026-05-25): per-pixel diffmap renormalization scale applied INSIDE
/// the zensim backends before returning to the buttloop.
///
/// **Phase 3 placeholder**: `1.0` (no renorm). Per the task spec — "DO
/// NOT set the CVVDP_DIFFMAP_RENORM_SCALE-equivalent away from 1.0
/// yet — Phase 4 fits this". The `(renorm - 1.0).abs() > EPSILON`
/// guards inside the backends skip the multiplication pass entirely at
/// unit scale, so Phase 3 ships with zero overhead.
///
/// Phase 4 will fit this constant via a Phase 8c-style harness (capture
/// the zensim diffmap distribution against butteraugli on a held-out
/// corpus, compute `scale = (target_z / target_b) * (mean_b / mean_z)`
/// per RFC #1 §4.2). See `RFC_ZENSIM_FORK_PLAN.md` §10.2 (Phase 8b/8c
/// equivalent for zensim).
///
/// **Env override** `JXL_ZENSIM_DIFFMAP_RENORM_SCALE=<float>` replaces
/// this constant for bench harnesses. Only consulted when the env var
/// is present AND parseable; production code uses the constant.
pub(crate) const ZENSIM_DIFFMAP_RENORM_SCALE: f32 = 1.0;

/// Read the active renorm scale, honouring the
/// `JXL_ZENSIM_DIFFMAP_RENORM_SCALE` env override for Phase 4+ harness
/// use. Production callers should treat the env hook as bench-only.
#[inline]
pub(crate) fn resolved_zensim_diffmap_renorm_scale() -> f32 {
    if let Ok(s) = std::env::var("JXL_ZENSIM_DIFFMAP_RENORM_SCALE")
        && let Ok(v) = s.parse::<f32>()
        && v.is_finite()
        && v > 0.0
    {
        return v;
    }
    ZENSIM_DIFFMAP_RENORM_SCALE
}

// ============================================================================
// Unit tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// zensim score → butteraugli-direction mapping endpoints.
    ///
    /// Guards the mapping applied inside both
    /// [`cpu::CpuZensimBackend::compare_with_reference`] and
    /// [`gpu::GpuZensimBackend::compare_with_reference`]:
    /// - zensim_score = 100 (identical) → butter score 0 (matches
    ///   butteraugli's "perfect" reading).
    /// - zensim_score = 0 (maximally different) → butter score 100
    ///   (worst).
    /// - zensim overshoots above 100 (rare numerical edge case) →
    ///   clamped to 0 (not negative).
    /// - zensim undershoots below 0 (degenerate inputs) → clamped to 100.
    #[test]
    fn zensim_to_butteraugli_direction_mapping() {
        let map = zensim_score_to_butter_direction;
        assert!(
            (map(100.0) - 0.0).abs() < 1e-9,
            "identical zensim score → butter 0"
        );
        assert!(
            (map(0.0) - 100.0).abs() < 1e-9,
            "worst zensim score → butter 100"
        );
        assert!(
            (map(50.0) - 50.0).abs() < 1e-9,
            "mid zensim score → mid butter"
        );
        assert!(
            (map(101.0) - 0.0).abs() < 1e-9,
            "zensim overshoot clamps to 0 (not negative)"
        );
        assert!(
            (map(-1.0) - 100.0).abs() < 1e-9,
            "zensim undershoot clamps to 100 (not above 100)"
        );
    }

    /// `ZENSIM_DIFFMAP_RENORM_SCALE` is the Phase 3 placeholder.
    /// Guards against accidental promotion to a measured value before
    /// Phase 4 fits the constant via the diffmap distribution harness.
    /// If you're seeing this test fire, you almost certainly want to
    /// run the Phase 4 calibration sweep first.
    #[test]
    fn zensim_diffmap_renorm_scale_is_phase3_placeholder() {
        assert_eq!(
            ZENSIM_DIFFMAP_RENORM_SCALE, 1.0,
            "Phase 3 ships `1.0` (no renorm) per RFC #1 §4 + \
             RFC_ZENSIM_FORK_PLAN.md §5 task spec. Bumping this \
             constant requires the Phase 4 diffmap-distribution \
             harness output as evidence."
        );
    }

    // ========================================================================
    // CPU zensim backend (gated on `zensim-loop` feature)
    // ========================================================================

    /// CPU `try_new` succeeds on a 32×32 buffer (no CUDA required —
    /// pure Rust + alloc).
    #[cfg(feature = "zensim-loop")]
    #[test]
    fn cpu_zensim_try_new_constructs_on_32x32() {
        let backend = cpu::CpuZensimBackend::try_new(32, 32);
        assert!(
            backend.is_some(),
            "CPU zensim backend must construct on 32×32 (≥ 8×8 minimum)"
        );
    }

    /// CPU `try_new` returns `None` on dimensions below zensim's
    /// minimum (`min(w, h) < 8`).
    #[cfg(feature = "zensim-loop")]
    #[test]
    fn cpu_zensim_try_new_rejects_tiny() {
        let backend = cpu::CpuZensimBackend::try_new(4, 4);
        assert!(
            backend.is_none(),
            "CPU zensim backend must reject 4×4 (zensim minimum is 8×8)"
        );
    }

    /// CPU backend reports the stable name `"zensim-cpu"` for log
    /// scraping.
    #[cfg(feature = "zensim-loop")]
    #[test]
    fn cpu_zensim_name_is_stable() {
        let backend = cpu::CpuZensimBackend::try_new(32, 32)
            .expect("32×32 construct succeeds (no CUDA needed)");
        assert_eq!(backend.name(), "zensim-cpu");
    }

    /// CPU `compare_with_reference` errors cleanly when called before
    /// `set_reference` — does NOT panic.
    #[cfg(feature = "zensim-loop")]
    #[test]
    fn cpu_zensim_compare_before_set_reference_errors() {
        let mut backend = cpu::CpuZensimBackend::try_new(32, 32).expect("32×32 construct succeeds");
        let n = 32 * 32;
        let r = alloc::vec![0.5f32; n];
        let g = alloc::vec![0.5f32; n];
        let b = alloc::vec![0.5f32; n];
        let mut diffmap = alloc::vec::Vec::new();
        let err = backend.compare_with_reference(&r, &g, &b, 32, 32, 32, &mut diffmap);
        assert!(
            err.is_err(),
            "compare before set_reference MUST return Err, not panic"
        );
    }

    /// CPU backend round-trip on a flat field: set a uniform
    /// reference, compare against the SAME uniform distorted →
    /// expect a zensim score near 100 (= identical), which maps to
    /// butteraugli-direction score near 0. The diffmap length must
    /// match width × height.
    #[cfg(feature = "zensim-loop")]
    #[test]
    fn cpu_zensim_identical_scores_near_zero() {
        let w = 32usize;
        let h = 32usize;
        let n = w * h;
        let r = alloc::vec![0.5f32; n];
        let g = alloc::vec![0.5f32; n];
        let b = alloc::vec![0.5f32; n];
        let mut backend =
            cpu::CpuZensimBackend::try_new(w as u32, h as u32).expect("32×32 construct succeeds");
        backend
            .set_reference(&r, &g, &b, w, h)
            .expect("set_reference on flat field must succeed");
        let mut diffmap = alloc::vec::Vec::new();
        let result = backend
            .compare_with_reference(&r, &g, &b, w, w, h, &mut diffmap)
            .expect("compare_with_reference on flat field must succeed");
        // Identical inputs → zensim score ≈ 100 → butter score ≈ 0.
        // Use a generous 5.0 threshold so any future zensim profile
        // tweak that perturbs identical-input scores by a few percent
        // still passes (f32 reduction-order noise on flat fields can
        // show a few-unit deviation from algebraic identity).
        assert!(
            result.score < 5.0,
            "identical inputs must score near 0 (butter-direction); got {}",
            result.score
        );
        assert_eq!(diffmap.len(), n, "diffmap length must equal width × height");
    }

    /// CPU backend produces a non-trivial diffmap + score on
    /// perturbed input.
    #[cfg(feature = "zensim-loop")]
    #[test]
    fn cpu_zensim_perturbed_diffmap_nonzero() {
        let w = 32usize;
        let h = 32usize;
        let n = w * h;
        let r = alloc::vec![0.5f32; n];
        let g = alloc::vec![0.5f32; n];
        let b = alloc::vec![0.5f32; n];
        let mut r2 = r.clone();
        for y in 12..20 {
            for x in 12..20 {
                r2[y * w + x] = 0.9;
            }
        }
        let mut backend =
            cpu::CpuZensimBackend::try_new(w as u32, h as u32).expect("32×32 construct succeeds");
        backend
            .set_reference(&r, &g, &b, w, h)
            .expect("set_reference must succeed");
        let mut diffmap = alloc::vec::Vec::new();
        let result = backend
            .compare_with_reference(&r2, &g, &b, w, w, h, &mut diffmap)
            .expect("compare must succeed");
        assert_eq!(diffmap.len(), n);
        assert!(
            result.score > 0.001,
            "perturbed image must score > 0.001 (butter-direction); got {}",
            result.score
        );
    }

    /// CPU backend handles strided distorted input (padded_width >
    /// width). Verifies the strided-row copy path doesn't accidentally
    /// pick up padding bytes. Reference is always tight (trait
    /// contract).
    #[cfg(feature = "zensim-loop")]
    #[test]
    fn cpu_zensim_strided_distorted_works() {
        let w = 16usize;
        let h = 16usize;
        let padded_w = 24usize;
        let n_tight = w * h;
        let n_strided = padded_w * h;
        let r = alloc::vec![0.5f32; n_tight];
        let g = alloc::vec![0.5f32; n_tight];
        let b = alloc::vec![0.5f32; n_tight];
        let mut r_s = alloc::vec![f32::NAN; n_strided];
        let mut g_s = alloc::vec![f32::NAN; n_strided];
        let mut b_s = alloc::vec![f32::NAN; n_strided];
        for y in 0..h {
            for x in 0..w {
                r_s[y * padded_w + x] = 0.5;
                g_s[y * padded_w + x] = 0.5;
                b_s[y * padded_w + x] = 0.5;
            }
        }
        let mut backend =
            cpu::CpuZensimBackend::try_new(w as u32, h as u32).expect("16×16 construct succeeds");
        backend.set_reference(&r, &g, &b, w, h).unwrap();
        let mut diffmap = alloc::vec::Vec::new();
        let result = backend
            .compare_with_reference(&r_s, &g_s, &b_s, padded_w, w, h, &mut diffmap)
            .expect("strided distorted compare must succeed");
        assert!(
            result.score.is_finite(),
            "strided compare must produce finite score; got {}",
            result.score
        );
        assert!(
            result.score < 5.0,
            "strided identical inputs must score near 0; got {}",
            result.score
        );
        for (i, v) in diffmap.iter().enumerate() {
            assert!(
                v.is_finite(),
                "strided compare diffmap[{i}] must be finite; got {v}"
            );
        }
    }

    /// Dimension mismatch in `set_reference` returns Err, not panic.
    #[cfg(feature = "zensim-loop")]
    #[test]
    fn cpu_zensim_set_reference_dim_mismatch_errors() {
        let mut backend = cpu::CpuZensimBackend::try_new(32, 32).expect("32×32 construct succeeds");
        let r = alloc::vec![0.5f32; 16 * 16];
        let g = alloc::vec![0.5f32; 16 * 16];
        let b = alloc::vec![0.5f32; 16 * 16];
        let err = backend.set_reference(&r, &g, &b, 16, 16);
        assert!(
            err.is_err(),
            "set_reference with mismatched dims MUST return Err"
        );
    }

    // ========================================================================
    // GPU zensim backend (gated on `zensim-loop-gpu` feature)
    // ========================================================================

    /// GPU constructor's `try_new` returns `None` cleanly if CUDA
    /// isn't present — never panics. Verified by calling `try_new`
    /// on a trivial size and accepting either `Some(_)` (CUDA OK) or
    /// `None` (CUDA missing); the test passes in both environments.
    #[cfg(feature = "zensim-loop-gpu")]
    #[test]
    fn gpu_zensim_try_new_does_not_panic() {
        let result = std::panic::catch_unwind(|| gpu::GpuZensimBackend::try_new(32, 32));
        assert!(
            result.is_ok(),
            "GpuZensimBackend::try_new MUST NOT panic; got: {:?}",
            result.err()
        );
    }
}
