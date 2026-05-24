// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Pluggable butteraugli backend for the quantization loop (W44-phase3-B1).
//!
//! The buttloop calls butteraugli once per iteration to measure the perceptual
//! distance between the original linear-RGB image and the current iteration's
//! reconstructed linear-RGB image. The result drives both the global score
//! (used to terminate / pick the best seed) and the per-pixel diffmap (used
//! to compute per-block tile-distance for the next iter's qf adjustment).
//!
//! This module abstracts that step behind a [`ButteraugliBackend`] trait so a
//! GPU backend can be plugged in opt-in. The default backend remains the
//! existing CPU `butteraugli` crate.
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

use crate::error::Result;

/// Result of one butteraugli comparison: aggregated max-norm score over the
/// linear-RGB plane diff. The diffmap itself is written into a caller-owned
/// `Vec<f32>` by [`ButteraugliBackend::compare_with_reference`], so it can be
/// recycled across iterations (W44-phase3-B7a, 2026-05-23).
#[derive(Debug)]
pub(crate) struct BackendCompareResult {
    /// Max-norm score (the value the libjxl buttloop compares against
    /// `target_distance`). Same units as `butteraugli::ButteraugliResult::score`.
    pub(crate) score: f64,
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
pub(crate) trait ButteraugliBackend: core::fmt::Debug {
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
        }
    }
}

#[cfg(feature = "butteraugli-loop")]
impl ButteraugliBackend for CpuButteraugliBackend {
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
        let _ = (width, height);
        Ok(BackendCompareResult { score })
    }
}

// ============================================================================
// GPU backend — feature-gated, opt-in
// ============================================================================

/// Documented threshold for an in-loop GPU-vs-CPU score divergence check.
/// **Not currently wired** — recorded as a follow-on note for future chunks
/// that may want to add a per-iter sanity check. W44-RECON-DEEP/A7 measured
/// a 0.02-0.03% drift floor on real corpus images; this 0.5% threshold is
/// 25× higher (kept in source as the canonical "this much divergence is
/// suspicious" magic number).
///
/// W44-phase3-B1 does NOT implement per-iter cross-checking because the
/// buttloop's compare step is on the critical path and a CPU-shadow
/// invocation would defeat the GPU speedup. A future follow-on chunk may
/// add `--gpu-butteraugli-validate` that runs both backends on iter 0 and
/// falls back to CPU if scores diverge by more than this threshold.
#[cfg(feature = "gpu-butteraugli")]
#[allow(dead_code)]
pub(crate) const GPU_SCORE_DIVERGENCE_PCT: f64 = 0.5;

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
    use cubecl::cuda::CudaRuntime;
    use cubecl::prelude::ComputeClient;
    use cubecl::server::Handle;
    use cubecl::Runtime;

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
    }

    impl core::fmt::Debug for GpuButteraugliBackend {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.debug_struct("GpuButteraugliBackend")
                .field("width", &self.width)
                .field("height", &self.height)
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
        pub(crate) fn try_new(
            width: u32,
            height: u32,
            intensity_target: f32,
        ) -> Option<Self> {
            // CubeCL client init. `client(&Default::default())` returns
            // a `ComputeClient<CudaRuntime>`; a panic inside CubeCL on a
            // CUDA-less host would surface as `try_init` failure. We
            // catch_unwind so a missing CUDA driver doesn't crash the
            // entire encode.
            let client = match std::panic::catch_unwind(|| {
                CudaRuntime::client(&Default::default())
            }) {
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
            })
        }

        /// Upload one tight (width*height) f32 plane to a fresh GPU
        /// handle. Caller-supplied `plane.len()` MUST equal `n = width*height`.
        fn upload_plane(&self, plane: &[f32]) -> Handle {
            debug_assert_eq!(plane.len(), (self.width as usize) * (self.height as usize));
            // bytemuck::cast_slice<f32, u8> — the f32 plane → byte view
            // CubeCL needs for `create_from_slice`.
            self.client.create_from_slice(bytemuck::cast_slice::<f32, u8>(plane))
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
                scratch[dst_row..dst_row + width]
                    .copy_from_slice(&src[src_row..src_row + width]);
            }
        }
    }

    impl ButteraugliBackend for GpuButteraugliBackend {
        fn name(&self) -> &'static str {
            "gpu-cuda"
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
            let r_h = self.client.create_from_slice(bytemuck::cast_slice::<f32, u8>(s_r));
            let g_h = self.client.create_from_slice(bytemuck::cast_slice::<f32, u8>(s_g));
            let b_h = self.client.create_from_slice(bytemuck::cast_slice::<f32, u8>(s_b));
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
            self.inner
                .copy_diffmap_to(diffmap_out)
                .map_err(|e| {
                    crate::error::Error::InvalidInput(format!("GPU butteraugli copy_diffmap: {e}"))
                })?;
            Ok(BackendCompareResult {
                score: result.score as f64,
            })
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
    static LIN_TO_SRGB_LUT: once_cell::race::OnceBox<[u8; 8193]> =
        once_cell::race::OnceBox::new();

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

/// Construct the active butteraugli backend for one buttloop run.
///
/// Routing:
/// - `gpu_requested == false` OR feature `gpu-butteraugli` is OFF → CPU backend.
/// - `gpu_requested == true` AND feature is ON AND CUDA init succeeds → GPU backend.
/// - `gpu_requested == true` AND feature is ON AND CUDA init fails → CPU backend
///   (silent fallback; emits one `eprintln!` so users can see why GPU didn't fire).
#[cfg(feature = "butteraugli-loop")]
pub(crate) fn construct_backend(
    width: u32,
    height: u32,
    cpu_params: butteraugli::ButteraugliParams,
    #[allow(unused_variables)] intensity_target: f32,
    #[allow(unused_variables)] gpu_requested: bool,
) -> alloc::boxed::Box<dyn ButteraugliBackend> {
    // Debug hook: `JXL_W44_PHASE3_B1_DEBUG=1` logs which backend the
    // dispatch picks. Off by default to keep production logs clean.
    #[cfg(feature = "std")]
    let debug_log = std::env::var("JXL_W44_PHASE3_B1_DEBUG").ok().as_deref() == Some("1");
    #[cfg(not(feature = "std"))]
    let debug_log = false;
    #[cfg(feature = "gpu-butteraugli")]
    {
        if gpu_requested {
            if debug_log {
                eprintln!(
                    "[W44-phase3-B1] GPU requested @ {}×{} — trying CUDA init",
                    width, height
                );
            }
            if let Some(g) = gpu::GpuButteraugliBackend::try_new(width, height, intensity_target) {
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
        let backend = construct_backend(64, 64, params, 80.0, false);
        assert_eq!(backend.name(), "cpu");
    }
}
