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

            Ok(BackendCompareResult { score })
        }
    }
}

// ============================================================================
// CPU CVVDP backend stub (Phase 5)
// ============================================================================

/// Stub for the forthcoming CPU CVVDP backend. Agent A shipped the
/// `cvvdp-cpu` crate at zenmetrics master `da81694`+`a177c89`+`c3c56ee`;
/// integration is queued for Phase 5 once the crate exposes a parity-
/// tested `score_with_diffmap` mirroring `cvvdp-gpu`'s Phase 1 API.
///
/// Reserved name. Not constructed by `construct_backend` in this chunk.
#[cfg(feature = "cvvdp-loop")]
#[allow(dead_code)]
pub(crate) struct CpuCvvdpBackend {
    width: u32,
    height: u32,
}

#[cfg(feature = "cvvdp-loop")]
impl core::fmt::Debug for CpuCvvdpBackend {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CpuCvvdpBackend")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("note", &"Phase 5 — not yet implemented")
            .finish()
    }
}

#[cfg(feature = "cvvdp-loop")]
impl PerceptualBackend for CpuCvvdpBackend {
    fn name(&self) -> &'static str {
        "cvvdp-cpu-stub"
    }

    fn set_reference(
        &mut self,
        _ref_r: &[f32],
        _ref_g: &[f32],
        _ref_b: &[f32],
        _width: usize,
        _height: usize,
    ) -> Result<()> {
        Err(crate::error::Error::InvalidInput(
            "CpuCvvdpBackend: Phase 5 stub — wire `cvvdp-cpu` to land this backend".into(),
        ))
    }

    fn compare_with_reference(
        &mut self,
        _dist_r: &[f32],
        _dist_g: &[f32],
        _dist_b: &[f32],
        _padded_width: usize,
        _width: usize,
        _height: usize,
        _diffmap_out: &mut alloc::vec::Vec<f32>,
    ) -> Result<BackendCompareResult> {
        Err(crate::error::Error::InvalidInput(
            "CpuCvvdpBackend: Phase 5 stub — wire `cvvdp-cpu` to land this backend".into(),
        ))
    }
}

// ============================================================================
// Unit tests
// ============================================================================

#[cfg(all(test, feature = "cvvdp-loop"))]
mod tests {
    use super::*;

    /// Smoke: CPU CVVDP stub returns a clean error rather than
    /// `unimplemented!()` panic. Phase 5 will replace the stub.
    #[test]
    fn cpu_cvvdp_stub_returns_error_not_panic() {
        let mut backend = CpuCvvdpBackend {
            width: 64,
            height: 64,
        };
        let n = 64 * 64;
        let r = alloc::vec![0.5f32; n];
        let g = alloc::vec![0.5f32; n];
        let b = alloc::vec![0.5f32; n];
        let err = backend.set_reference(&r, &g, &b, 64, 64);
        assert!(err.is_err(), "Phase 5 CPU stub must error on set_reference");
        let mut diffmap = alloc::vec::Vec::new();
        let err = backend.compare_with_reference(&r, &g, &b, 64, 64, 64, &mut diffmap);
        assert!(
            err.is_err(),
            "Phase 5 CPU stub must error on compare_with_reference"
        );
    }

    /// `CpuCvvdpBackend::name()` is stable for log scraping.
    #[test]
    fn cpu_cvvdp_stub_name() {
        let backend = CpuCvvdpBackend {
            width: 64,
            height: 64,
        };
        assert_eq!(backend.name(), "cvvdp-cpu-stub");
    }

    /// JOD → butteraugli-direction mapping endpoints.
    ///
    /// This guards the mapping we apply in
    /// [`gpu::GpuCvvdpBackend::compare_with_reference`]:
    /// - JOD = 10 (identical inputs) → score 0 (matches butteraugli's
    ///   "perfect" reading).
    /// - JOD = 0 (maximally different) → score 10 (worst).
    /// - JOD overshoots above 10 (rare numerical edge case) → clamped
    ///   to score 0 (not negative).
    /// - JOD undershoots below 0 (degenerate inputs) → clamped to
    ///   score 10.
    ///
    /// Re-implemented locally because the mapping lives inside an
    /// impl method and is otherwise un-callable from tests.
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

    /// CVVDP backend module compiles + the module-public types are
    /// reachable when the feature is enabled.
    #[test]
    fn cvvdp_module_smoke() {
        // Just instantiating to verify visibility — the constructor
        // doesn't touch CUDA. (`try_new` is the one that does.)
        let backend = CpuCvvdpBackend {
            width: 32,
            height: 32,
        };
        let _: &dyn PerceptualBackend = &backend;
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
}
