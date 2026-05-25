// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! cvvdp-based [`PerceptualBackend`] implementations (cvvdp-fork Phase 3,
//! 2026-05-24 — see `docs/RFC_CVVDP_FORK.md` §2.1 and
//! `docs/RFC_CVVDP_PHASE3_BRIEF.md`).
//!
//! Phase 3 ships the backend impl + the opt-in API surface only. The
//! buttloop body still consumes butteraugli; Phase 4 (separate chunk)
//! plumbs the cvvdp signal through `run_buttloop`. Hash-locks therefore
//! stay byte-identical at default features AND when the `cvvdp-loop`
//! feature is enabled but no caller has opted in via
//! [`LossyConfig::with_cvvdp_loop(Some(true))`](crate::api::LossyConfig::with_cvvdp_loop).
//!
//! ## Backends
//!
//! - [`GpuCvvdpBackend`] (feature `cvvdp-loop`) — wraps
//!   `cvvdp_gpu::CvvdpOpaque`, uploads linear-f32 planes directly via the
//!   `*_from_linear_planes_*` API surface Agent B shipped (zenmetrics
//!   commit `8b658b4`). Produces a per-pixel diffmap + scalar JOD;
//!   the scalar is mapped to butteraugli-direction semantics
//!   (smaller = better) via `score = (10.0 - jod).clamp(0.0, 10.0)` so
//!   the buttloop's `target_distance` comparison surface stays uniform
//!   across backends.
//!
//! ## When the GPU CVVDP backend is active
//!
//! 1. Caller sets [`LossyConfig::with_cvvdp_loop(Some(true))`](crate::api::LossyConfig::with_cvvdp_loop) AND
//! 2. The `cvvdp-loop` cargo feature was enabled at build time AND
//! 3. The CUDA runtime initialised successfully at backend construction AND
//! 4. The active strategy is NOT [`EncoderStrategy::Libjxl`](crate::api::EncoderStrategy::Libjxl)
//!    (the strict cjxl-parity invariant forces butteraugli regardless of
//!    the field — see [`LossyConfig::resolve_cvvdp_loop`](crate::api::LossyConfig::resolve_cvvdp_loop)).
//!
//! If any of those fail, the buttloop falls back to the CPU butteraugli
//! backend silently (defense-in-depth — the encoder is never broken by
//! GPU misconfiguration). Phase 4 / 5 will surface the active backend
//! via `EncodeStats`-style logging.

use alloc::format;

use crate::error::Result;

use super::perceptual_backend::{BackendCompareResult, PerceptualBackend};

// ============================================================================
// GPU CVVDP backend — feature-gated, opt-in
// ============================================================================

#[cfg(feature = "cvvdp-loop")]
pub(crate) mod gpu {
    //! GPU CVVDP backend (CUDA via CubeCL).
    //!
    //! Constructed on demand by
    //! [`super::super::perceptual_backend::construct_backend`]. If CUDA
    //! init fails (e.g. no GPU, no driver), [`GpuCvvdpBackend::try_new`]
    //! returns `None` and the caller falls back to the CPU butteraugli
    //! backend — same shape as `GpuButteraugliBackend`.
    //!
    //! ## API surface this wraps (zenmetrics master `8b658b4`)
    //!
    //! - [`cvvdp_gpu::CvvdpOpaque::new`] — single-resolution Backend
    //!   selector (`cvvdp_gpu::Backend::Cuda`). Multi-resolution is
    //!   inherent to CVVDP (it builds a pyramid internally).
    //! - [`cvvdp_gpu::CvvdpOpaque::warm_reference_from_linear_planes`]
    //!   — caches a linear-RGB reference for many subsequent compares.
    //! - [`cvvdp_gpu::CvvdpOpaque::compute_with_warm_ref_from_linear_planes`]
    //!   — runs a compare against the warm reference, optionally
    //!   filling a caller-owned `Vec<f32>` diffmap.
    //!
    //! All three methods take linear-f32 planes (no sRGB byte-pack on
    //! the host) — the upstream API mirrors butteraugli-gpu's
    //! W44-phase3-B4 linear-planes bypass that landed in B4.

    use super::*;

    use cvvdp_gpu::{Backend as CvvdpBackend, CvvdpOpaque, CvvdpParams};

    /// CUDA-backed CVVDP backend. Wraps [`CvvdpOpaque`] and uploads
    /// host-side linear-f32 planar input directly via the
    /// `*_from_linear_planes_*` API surface.
    ///
    /// One instance is scoped to a single buttloop cell: the reference
    /// is set once via [`PerceptualBackend::set_reference`] then
    /// [`PerceptualBackend::compare_with_reference`] is called once per
    /// buttloop iteration with the current iter's reconstructed planes.
    pub(crate) struct GpuCvvdpBackend {
        inner: CvvdpOpaque,
        /// Tight-stride f32 scratch for the distorted side's R/G/B
        /// planes. The buttloop hands us strided `recon_r/g/b` with
        /// `padded_width >= width`; we copy each row into this buffer
        /// before passing the slice to `compute_with_warm_ref_from_linear_planes`,
        /// because CVVDP's linear-planes entry points expect tight
        /// `width × height` planes (no padding). Allocated once and
        /// reused every iter to avoid per-iter `Vec` churn.
        dist_plane_scratch: [alloc::vec::Vec<f32>; 3],
        width: u32,
        height: u32,
        reference_warmed: bool,
        /// Phase 8b: per-instance compare-call counter for the
        /// `JXL_PHASE8B_DIFFMAP_DUMP` env-gated TSV dump (bench-only).
        /// Zero production cost when the env var is unset.
        compare_call_count: u32,
    }

    impl core::fmt::Debug for GpuCvvdpBackend {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.debug_struct("GpuCvvdpBackend")
                .field("width", &self.width)
                .field("height", &self.height)
                .field("reference_warmed", &self.reference_warmed)
                .finish()
        }
    }

    impl GpuCvvdpBackend {
        /// Construct a GPU CVVDP backend for `width × height`. Returns
        /// `None` if the CUDA runtime fails to initialise (e.g. no GPU,
        /// no driver, panic inside CubeCL). Mirrors the
        /// `GpuButteraugliBackend::try_new` defense-in-depth pattern:
        /// `catch_unwind` around the constructor so a missing CUDA
        /// driver never aborts the encode.
        ///
        /// CVVDP doesn't expose an `intensity_target` knob the same way
        /// butteraugli does — the metric internally targets ~80 cd/m²
        /// for SDR (controlled via the underlying `CvvdpParams::display`
        /// / `geometry` fields, both defaulted upstream). Callers that
        /// need HDR-aware CVVDP scoring should re-bake the upstream
        /// params, which is out of scope for this chunk (Phase 4
        /// follow-on).
        pub(crate) fn try_new(width: u32, height: u32) -> Option<Self> {
            // CubeCL panic-on-CUDA-missing protection. Mirror
            // `GpuButteraugliBackend::try_new` — wrap the entire
            // constructor in `catch_unwind` so the encode survives
            // even if CubeCL aborts mid-init.
            let inner = std::panic::catch_unwind(|| {
                CvvdpOpaque::new(CvvdpBackend::Cuda, width, height, CvvdpParams::default())
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
        /// `GpuButteraugliBackend::copy_strided_row_into_scratch` — same
        /// fast-path when `padded_width == width`, same per-row copy
        /// otherwise. The buttloop's reconstructed planes carry the
        /// `padded_width` stride (one block-row of padding); CVVDP
        /// needs tight stride.
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

    impl PerceptualBackend for GpuCvvdpBackend {
        fn name(&self) -> &'static str {
            "cvvdp-gpu-cuda"
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
                    "GPU CVVDP backend: dim mismatch in set_reference: \
                     expected {}×{}, got {}×{}",
                    self.width, self.height, width, height,
                )));
            }
            let n = width * height;
            if ref_r.len() < n || ref_g.len() < n || ref_b.len() < n {
                return Err(crate::error::Error::InvalidInput(format!(
                    "GPU CVVDP backend: reference plane too short: \
                     expected {}, got R={} G={} B={}",
                    n,
                    ref_r.len(),
                    ref_g.len(),
                    ref_b.len(),
                )));
            }
            // Reference is tight by the trait contract — `set_reference`
            // doesn't take a stride; `width == stride`.
            self.inner
                .warm_reference_from_linear_planes(&ref_r[..n], &ref_g[..n], &ref_b[..n])
                .map_err(|e| {
                    crate::error::Error::InvalidInput(format!(
                        "GPU CVVDP warm_reference_from_linear_planes: {e}"
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
            diffmap_out: &mut alloc::vec::Vec<f32>,
        ) -> Result<BackendCompareResult> {
            if !self.reference_warmed {
                return Err(crate::error::Error::InvalidInput(
                    "GPU CVVDP backend: compare_with_reference called \
                     before set_reference"
                        .into(),
                ));
            }
            if width as u32 != self.width || height as u32 != self.height {
                return Err(crate::error::Error::InvalidInput(format!(
                    "GPU CVVDP backend: dim mismatch in compare: \
                     expected {}×{}, got {}×{}",
                    self.width, self.height, width, height,
                )));
            }

            // Strided → tight copy into the per-instance scratch.
            let [s_r, s_g, s_b] = &mut self.dist_plane_scratch;
            Self::copy_strided_row_into_scratch(s_r, dist_r, padded_width, width, height);
            Self::copy_strided_row_into_scratch(s_g, dist_g, padded_width, width, height);
            Self::copy_strided_row_into_scratch(s_b, dist_b, padded_width, width, height);

            // Resize the caller-owned diffmap Vec — CVVDP's linear-planes
            // diffmap path requires the buffer to hold exactly W*H
            // entries on entry (verified by zenmetrics
            // `tests/diffmap_dispatch.rs`). The buttloop reuses the
            // allocation across iters; `resize(_, 0.0)` is amortized
            // O(1) once the high watermark is reached.
            let needed = width * height;
            diffmap_out.clear();
            diffmap_out.resize(needed, 0.0);

            let score_obj = self
                .inner
                .compute_with_warm_ref_from_linear_planes(
                    &self.dist_plane_scratch[0],
                    &self.dist_plane_scratch[1],
                    &self.dist_plane_scratch[2],
                    Some(diffmap_out),
                )
                .map_err(|e| {
                    crate::error::Error::InvalidInput(format!(
                        "GPU CVVDP compute_with_warm_ref_from_linear_planes: {e}"
                    ))
                })?;
            debug_assert_eq!(diffmap_out.len(), needed);

            // JOD → butteraugli-direction (smaller = better) mapping.
            // CVVDP's JOD lives in [0, 10] with 10 = identical
            // (no perceptible diff). butteraugli's score is non-negative
            // with 0 = identical. The buttloop's `target_distance`
            // comparison surface assumes the latter (smaller = better),
            // so we map `score = (10.0 - jod).clamp(0.0, 10.0)`. The
            // clamp protects against rare JOD overshoots above 10 on
            // identical-plus-noise inputs.
            //
            // Note: this mapping preserves direction but NOT magnitude
            // — a CVVDP-aware buttloop (Phase 4) MUST recalibrate its
            // `target_distance` thresholds. The Phase 4 brief queues
            // a per-distance JOD-target table at
            // `vardct/cvvdp_targets.rs` to handle this; Phase 3 ships
            // the mapping unchanged so smoke tests can run.
            let jod = score_obj.value as f64;
            let score = (10.0_f64 - jod).clamp(0.0, 10.0);

            // Phase 8b: optional pre-renorm diffmap distribution dump.
            // Captured BEFORE renormalization so the dump captures the
            // raw cvvdp signal — Phase 8b's job is to compute the scale
            // ratio against butteraugli's diffmap.
            super::super::perceptual_backend::maybe_dump_diffmap_stats(
                "C_GPU_PRE",
                self.compare_call_count,
                width,
                height,
                diffmap_out,
                score,
            );

            // Phase 8c (2026-05-25): renormalize the cvvdp diffmap before
            // returning to the buttloop. The W44 per-block reducer
            // (`vardct/perceptual_loop.rs` 16th-power norm + `tile_dist /
            // effective_metric_target_distance > 1` bad-block predicate)
            // was calibrated for butteraugli's diffmap distribution; cvvdp's
            // mean is structurally larger per Phase 8b measurement, so
            // without this scale the bad-block set is over-populated and
            // the loop over-allocates qac → 2-4× bytes vs butteraugli
            // for ~equal cvvdp JOD (Phase 8a Pareto diagnosis: 40.3%
            // vs 93.6% Pareto-front position).
            let renorm = super::super::perceptual_backend::resolved_cvvdp_diffmap_renorm_scale();
            if (renorm - 1.0).abs() > f32::EPSILON {
                for v in diffmap_out.iter_mut() {
                    *v *= renorm;
                }
            }

            // Phase 8b: optional post-renorm dump for harnesses that want
            // to confirm the rescaling lands in the right band.
            super::super::perceptual_backend::maybe_dump_diffmap_stats(
                "C_GPU_POST",
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
// CPU CVVDP backend — cvvdp-fork Phase 5 (2026-05-24)
// ============================================================================

#[cfg(feature = "cvvdp-loop-cpu")]
pub(crate) mod cpu {
    //! CPU CVVDP backend (pure Rust, no CUDA, rayon-backed per-band
    //! fan-out via the `cvvdp-cpu` crate's `parallel` feature).
    //!
    //! Constructed on demand by
    //! [`super::super::perceptual_backend::construct_backend`] when the
    //! caller has opted in via [`crate::api::LossyConfig::with_cvvdp_loop(Some(true))`]
    //! AND (a) explicitly preferred CPU via
    //! [`crate::api::LossyConfig::with_cvvdp_use_cpu(Some(true))`], OR
    //! (b) the GPU CVVDP backend is unavailable (CUDA missing / `cvvdp-loop`
    //! feature off / CUDA init fails — defense-in-depth fallback).
    //!
    //! ## API surface this wraps (zenmetrics master `da81694`+`a177c89`+`c3c56ee`)
    //!
    //! Agent A's `cvvdp-cpu` v0.0.1 exposes:
    //! - [`cvvdp_cpu::Cvvdp::new`] / [`cvvdp_cpu::Cvvdp::with_geometry`]
    //!   — construct a scorer for a fixed `(width, height)` with
    //!   per-call scratch.
    //! - sRGB-byte one-shots: `score` / `score_with_diffmap`.
    //! - sRGB-byte warm-ref: `warm_reference` / `score_with_warm_ref` /
    //!   `score_with_warm_ref_diffmap`.
    //! - **Linear-f32 planar one-shots**: `score_from_linear_planes` /
    //!   `score_from_linear_planes_with_diffmap` — the entry points
    //!   the JPEG XL buttloop needs.
    //!
    //! ## Phase 5b follow-on (NOT this chunk)
    //!
    //! `cvvdp-cpu` v0.0.1 lacks the linear-planes warm-reference
    //! companion (`warm_reference_from_linear_planes` +
    //! `score_from_linear_planes_with_warm_ref_diffmap`) that Agent B
    //! added to `cvvdp-gpu` master `8b658b4`. Without those, every
    //! buttloop iteration must re-bake the reference DKL + weber
    //! pyramid (~half the per-iter compute). We work around it by
    //! caching the reference planes locally in `CpuCvvdpBackend` and
    //! passing them as the `ref_*` args to
    //! `score_from_linear_planes_with_diffmap` on every compare. The
    //! extra cost is ~5-10 % wall on a 4-iter buttloop at 1024² (each
    //! iter re-builds the ref pyramid that warm_reference would have
    //! cached). Agent A queued the linear-planes warm-ref as a v0.1.0
    //! follow-on; until that lands, this is the structural cost of
    //! the CPU backend integration.

    use super::*;

    use cvvdp_cpu::{Cvvdp, CvvdpParams};

    /// Pure-Rust CPU CVVDP backend. Wraps [`Cvvdp`] and feeds it
    /// linear-f32 planar input directly via the `score_from_linear_planes_*`
    /// API surface (no sRGB byte round-trip on the host).
    ///
    /// One instance is scoped to a single buttloop cell. The reference
    /// planes are cached locally on [`PerceptualBackend::set_reference`]
    /// in the `ref_planes` slot (the underlying `cvvdp-cpu` v0.0.1
    /// lacks a linear-planes warm-reference API; see module-level
    /// Phase 5b follow-on note). On each
    /// [`PerceptualBackend::compare_with_reference`] call we pass the
    /// cached reference + the current distorted planes through the
    /// one-shot `score_from_linear_planes_with_diffmap` entry point.
    pub(crate) struct CpuCvvdpBackend {
        inner: Cvvdp,
        /// Cached reference planes in tight-stride (`width == stride`)
        /// f32 layout. Filled by `set_reference`; reused on every
        /// `compare_with_reference` call until the backend is dropped.
        /// `[r, g, b]`.
        ref_planes: [alloc::vec::Vec<f32>; 3],
        /// Tight-stride f32 scratch for the distorted side. Mirrors
        /// `GpuCvvdpBackend::dist_plane_scratch`: the buttloop hands us
        /// strided `recon_r/g/b` with `padded_width >= width`; we copy
        /// each row into this buffer before passing the slice to
        /// `score_from_linear_planes_with_diffmap`, because the CPU
        /// scorer (like the GPU one) expects tight `width × height`
        /// planes (or a single `padded_width` shared across all 6
        /// planes — we normalise to tight stride for simplicity since
        /// the buffer is allocated once and reused).
        dist_plane_scratch: [alloc::vec::Vec<f32>; 3],
        width: u32,
        height: u32,
        reference_warmed: bool,
        /// Phase 8b: per-instance compare-call counter (bench-only dump).
        compare_call_count: u32,
    }

    impl core::fmt::Debug for CpuCvvdpBackend {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.debug_struct("CpuCvvdpBackend")
                .field("width", &self.width)
                .field("height", &self.height)
                .field("reference_warmed", &self.reference_warmed)
                .finish()
        }
    }

    impl CpuCvvdpBackend {
        /// Construct a CPU CVVDP backend for `width × height`. Returns
        /// `None` if `cvvdp-cpu` rejects the dimensions (only fails
        /// when `min(w, h) < 8` — the smallest cvvdp pyramid baseband
        /// is 4×4 and the algorithm needs one step of reduce, so
        /// effective minimum is 8×8).
        ///
        /// CPU construction never panics (no driver init, no GPU
        /// runtime — pure Rust + alloc). We still wrap in
        /// `catch_unwind` for defense-in-depth so a misbehaving
        /// transitive dep can't crash the encode.
        pub(crate) fn try_new(width: u32, height: u32) -> Option<Self> {
            let inner =
                std::panic::catch_unwind(|| Cvvdp::new(width, height, CvvdpParams::default()));
            let inner = match inner {
                Ok(Ok(c)) => c,
                Ok(Err(_)) => return None,
                Err(_) => return None,
            };

            let n = (width as usize).checked_mul(height as usize)?;
            Some(Self {
                inner,
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
                width,
                height,
                reference_warmed: false,
                compare_call_count: 0,
            })
        }

        /// Copy one strided plane into a tight-stride destination.
        /// Mirrors `GpuCvvdpBackend::copy_strided_row_into_scratch` —
        /// same fast-path when `padded_width == width`, same per-row
        /// copy otherwise.
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

    impl PerceptualBackend for CpuCvvdpBackend {
        fn name(&self) -> &'static str {
            "cvvdp-cpu"
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
                    "CPU CVVDP backend: dim mismatch in set_reference: \
                     expected {}×{}, got {}×{}",
                    self.width, self.height, width, height,
                )));
            }
            let n = width * height;
            if ref_r.len() < n || ref_g.len() < n || ref_b.len() < n {
                return Err(crate::error::Error::InvalidInput(format!(
                    "CPU CVVDP backend: reference plane too short: \
                     expected {}, got R={} G={} B={}",
                    n,
                    ref_r.len(),
                    ref_g.len(),
                    ref_b.len(),
                )));
            }
            // Reference is tight by the trait contract — `set_reference`
            // doesn't take a stride; `width == stride`. Cache the
            // planes locally for the per-iter compare path (cvvdp-cpu
            // v0.0.1 lacks a linear-planes warm-reference API; see
            // module-level Phase 5b follow-on note).
            self.ref_planes[0].clear();
            self.ref_planes[1].clear();
            self.ref_planes[2].clear();
            self.ref_planes[0].extend_from_slice(&ref_r[..n]);
            self.ref_planes[1].extend_from_slice(&ref_g[..n]);
            self.ref_planes[2].extend_from_slice(&ref_b[..n]);
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
            diffmap_out: &mut alloc::vec::Vec<f32>,
        ) -> Result<BackendCompareResult> {
            if !self.reference_warmed {
                return Err(crate::error::Error::InvalidInput(
                    "CPU CVVDP backend: compare_with_reference called \
                     before set_reference"
                        .into(),
                ));
            }
            if width as u32 != self.width || height as u32 != self.height {
                return Err(crate::error::Error::InvalidInput(format!(
                    "CPU CVVDP backend: dim mismatch in compare: \
                     expected {}×{}, got {}×{}",
                    self.width, self.height, width, height,
                )));
            }

            // Strided → tight copy into the per-instance scratch.
            let [s_r, s_g, s_b] = &mut self.dist_plane_scratch;
            Self::copy_strided_row_into_scratch(s_r, dist_r, padded_width, width, height);
            Self::copy_strided_row_into_scratch(s_g, dist_g, padded_width, width, height);
            Self::copy_strided_row_into_scratch(s_b, dist_b, padded_width, width, height);

            // Call cvvdp-cpu's one-shot linear-planes entry point with
            // the cached reference + current distorted planes. The
            // caller-owned `diffmap_out` is OVERWRITTEN by the cvvdp-cpu
            // API (it does `*diffmap_out = diff.expect(...)`), which
            // preserves the buttloop's amortised-O(1) Vec recycle
            // pattern as long as the high-watermark cap stays the same
            // (a `Vec::with_capacity(width * height)` allocation gets
            // replaced by a new Vec of the same length — Rust's
            // allocator typically returns the same block; the user-
            // observed behaviour is identical to the GPU backend's
            // `resize` path).
            //
            // Tight `padded_width = width` is passed because we've
            // already normalised the dist scratch above and the ref
            // planes are tight by construction.
            let tight_width = width;
            let jod = self
                .inner
                .score_from_linear_planes_with_diffmap(
                    &self.ref_planes[0],
                    &self.ref_planes[1],
                    &self.ref_planes[2],
                    &self.dist_plane_scratch[0],
                    &self.dist_plane_scratch[1],
                    &self.dist_plane_scratch[2],
                    tight_width,
                    diffmap_out,
                )
                .map_err(|e| {
                    crate::error::Error::InvalidInput(format!(
                        "CPU CVVDP score_from_linear_planes_with_diffmap: {e}"
                    ))
                })?;
            debug_assert_eq!(diffmap_out.len(), width * height);

            // JOD → butteraugli-direction mapping. Identical to the
            // GPU backend's mapping (`score = (10.0 - jod).clamp(0.0,
            // 10.0)`) — the buttloop's `target_distance` comparison
            // surface treats CPU and GPU cvvdp as the same metric
            // (Phase 4 calibration via `CVVDP_DISTANCE_TARGETS`).
            // **DO NOT** introduce per-backend score offsets here; if
            // the CPU vs GPU score drift ever exceeds the Phase 5
            // smoke test's ±0.05 JOD tolerance, root-cause the
            // underlying numeric divergence (likely the pycvvdp v0.5.4
            // reference disagreement Agent A documented at 4.4× off
            // SIMD floor) rather than papering over with a mapping
            // tweak.
            let jod = jod as f64;
            let score = (10.0_f64 - jod).clamp(0.0, 10.0);

            // Phase 8b: pre-renorm dump (see GPU backend for full doc).
            super::super::perceptual_backend::maybe_dump_diffmap_stats(
                "C_CPU_PRE",
                self.compare_call_count,
                width,
                height,
                diffmap_out,
                score,
            );

            // Phase 8c: same renormalization as GPU backend — keeps
            // both cvvdp paths semantically equivalent at the
            // perceptual-loop boundary.
            let renorm = super::super::perceptual_backend::resolved_cvvdp_diffmap_renorm_scale();
            if (renorm - 1.0).abs() > f32::EPSILON {
                for v in diffmap_out.iter_mut() {
                    *v *= renorm;
                }
            }

            // Phase 8b: post-renorm dump.
            super::super::perceptual_backend::maybe_dump_diffmap_stats(
                "C_CPU_POST",
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
// Unit tests
// ============================================================================

#[cfg(all(test, feature = "cvvdp-loop"))]
mod tests {
    use super::*;

    /// JOD → butteraugli-direction mapping endpoints.
    ///
    /// This guards the mapping we apply in
    /// [`gpu::GpuCvvdpBackend::compare_with_reference`] AND in
    /// [`cpu::CpuCvvdpBackend::compare_with_reference`] (Phase 5):
    /// - JOD = 10 (identical inputs) → score 0 (matches butteraugli's
    ///   "perfect" reading).
    /// - JOD = 0 (maximally different) → score 10 (worst).
    /// - JOD overshoots above 10 (rare numerical edge case) → clamped
    ///   to score 0 (not negative).
    /// - JOD undershoots below 0 (degenerate inputs) → clamped to
    ///   score 10.
    ///
    /// Re-implemented locally because the mapping lives inside impl
    /// methods and is otherwise un-callable from tests. Phase 5's CPU
    /// backend MUST use the EXACT same mapping as the GPU backend —
    /// the buttloop's `target_distance` comparison surface (and the
    /// Phase 4 `CVVDP_DISTANCE_TARGETS` calibration table) treat them
    /// as the same metric.
    #[test]
    fn jod_to_butteraugli_direction_mapping() {
        fn map(jod: f64) -> f64 {
            (10.0_f64 - jod).clamp(0.0, 10.0)
        }
        assert!((map(10.0) - 0.0).abs() < 1e-9, "identical JOD → score 0");
        assert!((map(0.0) - 10.0).abs() < 1e-9, "worst JOD → score 10");
        assert!((map(5.0) - 5.0).abs() < 1e-9, "mid JOD → mid score");
        assert!(
            (map(11.0) - 0.0).abs() < 1e-9,
            "JOD overshoot clamps to 0 (not negative)"
        );
        assert!(
            (map(-1.0) - 10.0).abs() < 1e-9,
            "JOD undershoot clamps to 10 (not above 10)"
        );
    }

    /// GPU constructor's `try_new` returns `None` cleanly if CUDA isn't
    /// present — never panics. Verified by calling `try_new` on a
    /// trivial size and accepting either `Some(_)` (CUDA OK) or `None`
    /// (CUDA missing); the test passes in both environments.
    #[test]
    fn gpu_cvvdp_try_new_does_not_panic() {
        // Wrap in a closure + assert_ok to make the no-panic
        // expectation explicit. `try_new` itself uses `catch_unwind`
        // internally; this test guards against accidental removal of
        // that safety net.
        let result = std::panic::catch_unwind(|| gpu::GpuCvvdpBackend::try_new(32, 32));
        assert!(
            result.is_ok(),
            "GpuCvvdpBackend::try_new MUST NOT panic; got: {:?}",
            result.err()
        );
        // The Option value (Some/None) depends on whether CUDA is
        // available; both are acceptable.
    }

    // ========================================================================
    // CPU CVVDP backend (Phase 5) — gated on `cvvdp-loop-cpu` feature.
    // ========================================================================

    /// CPU `try_new` succeeds on a 32×32 buffer (no CUDA required —
    /// pure Rust + alloc).
    #[cfg(feature = "cvvdp-loop-cpu")]
    #[test]
    fn cpu_cvvdp_try_new_constructs_on_32x32() {
        let backend = cpu::CpuCvvdpBackend::try_new(32, 32);
        assert!(
            backend.is_some(),
            "CPU CVVDP backend must construct on 32×32 (≥ 8×8 minimum)"
        );
    }

    /// CPU `try_new` returns `None` on dimensions below cvvdp's
    /// minimum (`min(w, h) < 8`). Mirrors the underlying
    /// `cvvdp_cpu::Error::InvalidImageSize` case.
    #[cfg(feature = "cvvdp-loop-cpu")]
    #[test]
    fn cpu_cvvdp_try_new_rejects_tiny() {
        let backend = cpu::CpuCvvdpBackend::try_new(4, 4);
        assert!(
            backend.is_none(),
            "CPU CVVDP backend must reject 4×4 (cvvdp minimum is 8×8)"
        );
    }

    /// CPU backend reports the stable name `"cvvdp-cpu"` for log
    /// scraping. Matches the Phase 5 brief's documented backend
    /// identifier.
    #[cfg(feature = "cvvdp-loop-cpu")]
    #[test]
    fn cpu_cvvdp_name_is_stable() {
        let backend = cpu::CpuCvvdpBackend::try_new(32, 32)
            .expect("32×32 construct succeeds (no CUDA needed)");
        assert_eq!(backend.name(), "cvvdp-cpu");
    }

    /// CPU `compare_with_reference` errors cleanly when called before
    /// `set_reference` — does NOT panic.
    #[cfg(feature = "cvvdp-loop-cpu")]
    #[test]
    fn cpu_cvvdp_compare_before_set_reference_errors() {
        let mut backend = cpu::CpuCvvdpBackend::try_new(32, 32).expect("32×32 construct succeeds");
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

    /// CPU backend round-trip on a flat field: set a uniform reference,
    /// compare against the SAME uniform distorted → expect a JOD score
    /// near 10 (= identical), which maps to butteraugli-direction
    /// score near 0. The diffmap length must match width × height.
    #[cfg(feature = "cvvdp-loop-cpu")]
    #[test]
    fn cpu_cvvdp_identical_scores_near_zero() {
        let w = 32usize;
        let h = 32usize;
        let n = w * h;
        let r = alloc::vec![0.5f32; n];
        let g = alloc::vec![0.5f32; n];
        let b = alloc::vec![0.5f32; n];
        let mut backend =
            cpu::CpuCvvdpBackend::try_new(w as u32, h as u32).expect("32×32 construct succeeds");
        backend
            .set_reference(&r, &g, &b, w, h)
            .expect("set_reference on flat field must succeed");
        let mut diffmap = alloc::vec::Vec::new();
        let result = backend
            .compare_with_reference(&r, &g, &b, w, w, h, &mut diffmap)
            .expect("compare_with_reference on flat field must succeed");
        // Identical inputs → JOD ≈ 10 → score ≈ 0. Use a generous
        // 0.1 threshold so any future internal-constant tweak that
        // perturbs identical-input scores by ≤ 1% of the [0, 10] band
        // still passes (the CPU port targets ≤ 1e-3 JOD parity vs
        // pycvvdp v0.5.4 on real cells, but f32 reduction-order
        // noise on flat fields can show ~1e-2 JOD).
        assert!(
            result.score < 0.1,
            "identical inputs must score near 0 (butteraugli-direction); got {}",
            result.score
        );
        assert_eq!(diffmap.len(), n, "diffmap length must equal width × height");
    }

    /// CPU backend produces a non-trivial diffmap + score on perturbed
    /// input. Mirrors the existing `cpu_backend_diffmap_size` test for
    /// the butteraugli CPU backend.
    #[cfg(feature = "cvvdp-loop-cpu")]
    #[test]
    fn cpu_cvvdp_perturbed_diffmap_nonzero() {
        let w = 32usize;
        let h = 32usize;
        let n = w * h;
        let r = alloc::vec![0.5f32; n];
        let g = alloc::vec![0.5f32; n];
        let b = alloc::vec![0.5f32; n];
        let mut r2 = r.clone();
        // Inject a perturbation in the middle so cvvdp reports
        // something non-trivial.
        for y in 12..20 {
            for x in 12..20 {
                r2[y * w + x] = 0.9;
            }
        }
        let mut backend =
            cpu::CpuCvvdpBackend::try_new(w as u32, h as u32).expect("32×32 construct succeeds");
        backend
            .set_reference(&r, &g, &b, w, h)
            .expect("set_reference must succeed");
        let mut diffmap = alloc::vec::Vec::new();
        let result = backend
            .compare_with_reference(&r2, &g, &b, w, w, h, &mut diffmap)
            .expect("compare must succeed");
        assert_eq!(diffmap.len(), n);
        // Perturbed inputs should NOT score near 0; the exact value
        // depends on the cvvdp display model + viewing geometry, but
        // we expect something measurably > 0.01.
        assert!(
            result.score > 0.001,
            "perturbed image must score > 0.001 (butteraugli-direction); got {}",
            result.score
        );
    }

    /// CPU backend handles strided distorted input (padded_width >
    /// width). Verifies the strip-tile copy path doesn't accidentally
    /// pick up padding bytes. Reference is always tight (trait
    /// contract).
    #[cfg(feature = "cvvdp-loop-cpu")]
    #[test]
    fn cpu_cvvdp_strided_distorted_works() {
        let w = 16usize;
        let h = 16usize;
        let padded_w = 24usize; // 8 cols of padding per row.
        let n_tight = w * h;
        let n_strided = padded_w * h;
        let r = alloc::vec![0.5f32; n_tight];
        let g = alloc::vec![0.5f32; n_tight];
        let b = alloc::vec![0.5f32; n_tight];
        // Strided distorted buffer with sentinel `f32::NAN` in the
        // padding columns — if our copy accidentally reads padding,
        // cvvdp would propagate NaN and the diffmap would be NaN.
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
            cpu::CpuCvvdpBackend::try_new(w as u32, h as u32).expect("16×16 construct succeeds");
        backend.set_reference(&r, &g, &b, w, h).unwrap();
        let mut diffmap = alloc::vec::Vec::new();
        let result = backend
            .compare_with_reference(&r_s, &g_s, &b_s, padded_w, w, h, &mut diffmap)
            .expect("strided distorted compare must succeed");
        // Identical inputs (after strip copy) → score near 0; if our
        // strip copy bug had pulled NaN from padding, score would be
        // NaN or huge.
        assert!(
            result.score.is_finite(),
            "strided compare must produce finite score; got {}",
            result.score
        );
        assert!(
            result.score < 0.1,
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
    #[cfg(feature = "cvvdp-loop-cpu")]
    #[test]
    fn cpu_cvvdp_set_reference_dim_mismatch_errors() {
        let mut backend = cpu::CpuCvvdpBackend::try_new(32, 32).expect("32×32 construct succeeds");
        let r = alloc::vec![0.5f32; 16 * 16];
        let g = alloc::vec![0.5f32; 16 * 16];
        let b = alloc::vec![0.5f32; 16 * 16];
        let err = backend.set_reference(&r, &g, &b, 16, 16);
        assert!(
            err.is_err(),
            "set_reference with mismatched dims MUST return Err"
        );
    }
}
