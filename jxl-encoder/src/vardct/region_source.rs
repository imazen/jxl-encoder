// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Region-source abstraction for the XYB planes consumed by
//! [`super::transform::VarDctEncoder::transform_and_quantize`] and the
//! downstream per-DC-group tokenization path landed by chunk 8a
//! (`ceb05e13`).
//!
//! ## Streaming refactor chunk 8b (#11)
//!
//! Chunk 8a moved token collection into a per-DC-group parallel fan-out
//! so each task could (eventually) drop its slice of the XYB / quant
//! buffers immediately after returning. The blocker for that drop is
//! that `transform_and_quantize` upstream of `encode_two_pass` still
//! takes three whole-image `&[f32]` slices and the encoder's caller
//! has no way to expose "give me only this DC group's region" — the
//! XYB planes are constructed once at the top of `encode_inner` and
//! consumed end-to-end as `Vec<f32>` borrows.
//!
//! This module introduces the *seam*: `XybRegionSource` is the
//! pull-style trait that lets the encoder ask "give me the XYB data
//! covering region R" and "you can free region R now."
//!
//! Chunk 8b ships only the trait + a `WholeImageXybSource` blanket
//! implementation that returns the existing whole-image borrows
//! unchanged — every existing call site gets byte-identical output.
//! Chunk 8c will wire a streaming source that materialises one DC
//! group at a time + drops it after `transform_and_quantize` returns.
//!
//! ### Why this trait (not a `&[f32]` x3)
//!
//! - `transform_and_quantize` reads XYB blocks in a parallel fan-out
//!   over AC groups. Each AC group is 32×32 blocks (256×256 px); a
//!   DC group is 8×8 AC groups (2048×2048 px). Random-access reads
//!   across the whole image are required because the per-tile AC
//!   strategy search references neighbour blocks (1-block border) and
//!   the existing rayon `parallel_map` over `num_groups` makes no
//!   assumption about pixel locality.
//! - Today, the encoder holds three `Vec<f32>` planes for the entire
//!   padded image (~24 B/pixel = ~190 MiB at 4096×4096). A streaming
//!   source could replace those with a sliding window over DC groups
//!   (~3 MiB per group + the active borrow); the trait abstraction
//!   keeps the existing `&[f32]`-based reader paths unchanged.
//!
//! ### What chunk 8b does NOT do
//!
//! - **No emission-order change.** Per-DC-group sections are still
//!   accumulated into the whole-image vectors that `encode_two_pass`
//!   downstream code expects. Bitstream output is byte-identical.
//! - **No `WritableSeek` sink wiring.** The chunk-6 buffered-output
//!   sink remains owned by `encode_two_pass`; chunk 8c plumbs the
//!   per-DC-group sections into it.
//! - **No quant_dc/quant_ac/nzeros region slicing.** The transform
//!   output is still produced at whole-image scope. Chunk 8c will
//!   refactor `TransformOutput` to own per-DC-group storage and drop
//!   it once tokenization consumes it.
//! - **No actual per-region xyb buffer dropping in the default
//!   path.** The blanket `WholeImageXybSource` is reference-only —
//!   the caller still owns the `Vec<f32>` planes. The trait makes
//!   the call site *willing* to release per-region storage; the
//!   producer side (chunk 8c streaming source) actually does the
//!   release.
//!
//! ### Downstream whole-image consumers still in place
//!
//! These functions read whole-image XYB after `transform_and_quantize`
//! and must be addressed before we can fully eliminate the whole-image
//! plane buffers. Listed in dispatch order:
//!
//! 1. `compute_epf_sharpness` ([`super::epf`]) — reads
//!    `[xyb_x, xyb_y, xyb_b]` for per-block sharpness derivation.
//!    Gated on `params.epf_iters > 0 && distance >= 0.5 &&
//!    profile.epf_dynamic_sharpness`.
//!    *Chunk-8c plan*: run sharpness derivation per DC group with a
//!    2-block top/bottom border (same border the gaborish step
//!    already requires).
//! 2. `compute_mask1x1_with_budget` ([`super::adaptive_quant`]) —
//!    fallback path inside the sharpness branch when `mask1x1` was
//!    not precomputed. *Chunk-8c plan*: lift the mask1x1 dependency
//!    out of the sharpness branch so the fallback isn't needed.
//! 3. `butteraugli_loop` (feature-gated) — multi-iteration internal
//!    reconstruction that needs whole-image XYB. *Chunk-8c plan*:
//!    keep `FullBuffered` for `butteraugli_iters > 0` (mirrors libjxl
//!    `CanDoStreamingEncoding` which gates streaming on
//!    `!use_butteraugli_loop`).
//! 4. Splines auto-detection and `simplify_invisible` (when enabled)
//!    — both run on whole-image XYB pre-transform; not affected by
//!    8b/8c (they run earlier than transform_and_quantize).
//!
//! See also: [`super::precomputed::compute_dc_group`] (chunk 3
//! per-region precompute) for the symmetric pull-style approach on
//! the quant_field / mask / CfL / AC-strategy side.

#![allow(dead_code)]

use alloc::vec::Vec;

/// Source that supplies XYB plane data to [`super::transform::VarDctEncoder::
/// transform_and_quantize_with_source`] (chunk 8b) and the downstream
/// per-DC-group token collection path (chunk 8a).
///
/// All implementors must:
/// - Return slices whose stride is `padded_width` (i.e. the canonical
///   row-major layout the encoder uses end-to-end).
/// - Return slices that include any edge-padding rows / columns (the
///   `convert_to_xyb_padded` pad-to-block-boundary contract).
/// - Make `xyb_full` cheap (typically a borrow); it's called on the hot
///   path from `transform_and_quantize` and any per-region wrapper.
///
/// The trait is `Sync` because [`super::transform::VarDctEncoder::
/// transform_and_quantize_with_source`] hands the source into
/// rayon-parallel AC-group tasks. The whole-image implementation is
/// trivially `Sync`; future streaming sources should hold their
/// internal buffer behind a `Mutex` only if they actually mutate it
/// during a `xyb_full` call (the chunk-8c per-DC-group source plans
/// to materialise one region at a time *between* `transform_and_quantize`
/// calls, not inside one).
pub(crate) trait XybRegionSource: Sync {
    /// Logical image width before block-padding.
    fn width(&self) -> usize;
    /// Logical image height before block-padding.
    fn height(&self) -> usize;
    /// Block-padded width (`xsize_blocks * 8`); equal to the row
    /// stride of every plane.
    fn padded_width(&self) -> usize;
    /// Block-padded height (`ysize_blocks * 8`).
    fn padded_height(&self) -> usize;

    /// Returns a borrow of the three XYB planes that covers the full
    /// `padded_width × padded_height` image.
    ///
    /// For the whole-image implementation this is a constant-cost
    /// borrow. For the chunk-8c streaming source, this materialises
    /// one DC group's region on demand and panics if asked for a
    /// region outside the active window (the encoder must call
    /// `release_dc_region` then re-pull).
    fn xyb_full(&self) -> (&[f32], &[f32], &[f32]);

    /// Hint that the caller is done with the DC group at
    /// `(dc_x, dc_y)` (DC-group coordinates) and the source may
    /// release any storage backing that region. Default: no-op.
    ///
    /// The whole-image source ignores the hint (the planes live for
    /// the lifetime of the encode). The chunk-8c streaming source
    /// drops the region's storage so the peak working set tracks the
    /// active DC group rather than the whole image.
    fn release_dc_region(&self, _dc_x: u32, _dc_y: u32) {}
}

/// Whole-image XYB source: wraps owned `Vec<f32>` planes and lets
/// the caller mutate them in place (gaborish_inverse, sanitize, etc.)
/// then hand the same buffers to [`super::transform::VarDctEncoder::
/// transform_and_quantize_with_source`].
///
/// This is the only implementation chunk 8b ships. It preserves the
/// existing byte-identical output contract — every hash-locked test
/// (`hash_lock_features.rs`, 36 tests) and every buffering-dispatch
/// test (`buffering_dispatch.rs`, 15 tests) passes unchanged.
///
/// The fields are `pub(crate)` so the encoder can construct, mutate,
/// and *take* the planes out at end of encode (the existing
/// `drop(xyb_x); drop(xyb_y); drop(xyb_b);` at
/// `encoder.rs:2132-2134` becomes a `drop(source)` after this
/// landing).
pub(crate) struct WholeImageXybSource {
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) padded_width: usize,
    pub(crate) padded_height: usize,
    pub(crate) xyb_x: Vec<f32>,
    pub(crate) xyb_y: Vec<f32>,
    pub(crate) xyb_b: Vec<f32>,
}

impl WholeImageXybSource {
    /// Construct from three already-padded XYB planes. The caller
    /// asserts that `xyb_x.len() == xyb_y.len() == xyb_b.len() ==
    /// padded_width * padded_height`.
    pub(crate) fn new(
        width: usize,
        height: usize,
        padded_width: usize,
        padded_height: usize,
        xyb_x: Vec<f32>,
        xyb_y: Vec<f32>,
        xyb_b: Vec<f32>,
    ) -> Self {
        let n = padded_width.saturating_mul(padded_height);
        debug_assert_eq!(xyb_x.len(), n, "WholeImageXybSource: xyb_x length mismatch");
        debug_assert_eq!(xyb_y.len(), n, "WholeImageXybSource: xyb_y length mismatch");
        debug_assert_eq!(xyb_b.len(), n, "WholeImageXybSource: xyb_b length mismatch");
        Self {
            width,
            height,
            padded_width,
            padded_height,
            xyb_x,
            xyb_y,
            xyb_b,
        }
    }

    /// Consume the source and return the owned planes. Mirrors the
    /// existing `(Vec<f32>, Vec<f32>, Vec<f32>)` shape so existing
    /// post-transform code that mutates / drops the planes continues
    /// to compile unchanged when the chunk-8b walker isn't engaged.
    #[allow(dead_code)]
    pub(crate) fn into_planes(self) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        (self.xyb_x, self.xyb_y, self.xyb_b)
    }
}

impl XybRegionSource for WholeImageXybSource {
    #[inline]
    fn width(&self) -> usize {
        self.width
    }
    #[inline]
    fn height(&self) -> usize {
        self.height
    }
    #[inline]
    fn padded_width(&self) -> usize {
        self.padded_width
    }
    #[inline]
    fn padded_height(&self) -> usize {
        self.padded_height
    }
    #[inline]
    fn xyb_full(&self) -> (&[f32], &[f32], &[f32]) {
        (&self.xyb_x, &self.xyb_y, &self.xyb_b)
    }
}

/// View-only XYB source backed by three external `&[f32]` slices.
/// Used when the encoder already owns the planes by some other path
/// (e.g. the `EncoderPrecomputed`-driven `encode_from_precomputed`
/// fast path where the planes live on the precomputed struct and the
/// encoder borrows them).
///
/// The lifetime is `'a`, so this is `Sync` whenever the underlying
/// slices outlive the `transform_and_quantize` call.
pub(crate) struct BorrowedXybSource<'a> {
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) padded_width: usize,
    pub(crate) padded_height: usize,
    pub(crate) xyb_x: &'a [f32],
    pub(crate) xyb_y: &'a [f32],
    pub(crate) xyb_b: &'a [f32],
}

impl<'a> BorrowedXybSource<'a> {
    pub(crate) fn new(
        width: usize,
        height: usize,
        padded_width: usize,
        padded_height: usize,
        xyb_x: &'a [f32],
        xyb_y: &'a [f32],
        xyb_b: &'a [f32],
    ) -> Self {
        let n = padded_width.saturating_mul(padded_height);
        debug_assert_eq!(xyb_x.len(), n, "BorrowedXybSource: xyb_x length mismatch");
        debug_assert_eq!(xyb_y.len(), n, "BorrowedXybSource: xyb_y length mismatch");
        debug_assert_eq!(xyb_b.len(), n, "BorrowedXybSource: xyb_b length mismatch");
        Self {
            width,
            height,
            padded_width,
            padded_height,
            xyb_x,
            xyb_y,
            xyb_b,
        }
    }
}

impl<'a> XybRegionSource for BorrowedXybSource<'a> {
    #[inline]
    fn width(&self) -> usize {
        self.width
    }
    #[inline]
    fn height(&self) -> usize {
        self.height
    }
    #[inline]
    fn padded_width(&self) -> usize {
        self.padded_width
    }
    #[inline]
    fn padded_height(&self) -> usize {
        self.padded_height
    }
    #[inline]
    fn xyb_full(&self) -> (&[f32], &[f32], &[f32]) {
        (self.xyb_x, self.xyb_y, self.xyb_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn make_planes(n: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let x: Vec<f32> = (0..n).map(|i| i as f32 * 0.001).collect();
        let y: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
        let b: Vec<f32> = (0..n).map(|i| -(i as f32) * 0.25).collect();
        (x, y, b)
    }

    #[test]
    fn whole_image_source_round_trip() {
        let (x, y, b) = make_planes(16 * 16);
        let src = WholeImageXybSource::new(16, 16, 16, 16, x.clone(), y.clone(), b.clone());
        assert_eq!(src.width(), 16);
        assert_eq!(src.height(), 16);
        assert_eq!(src.padded_width(), 16);
        assert_eq!(src.padded_height(), 16);
        let (sx, sy, sb) = src.xyb_full();
        assert_eq!(sx, x.as_slice());
        assert_eq!(sy, y.as_slice());
        assert_eq!(sb, b.as_slice());
        // Default release hint is a no-op; verify it doesn't crash.
        src.release_dc_region(0, 0);
        src.release_dc_region(7, 3);
        let (rx, ry, rb) = src.into_planes();
        assert_eq!(rx, x);
        assert_eq!(ry, y);
        assert_eq!(rb, b);
    }

    #[test]
    fn borrowed_source_round_trip() {
        let (x, y, b) = make_planes(8 * 8);
        let src = BorrowedXybSource::new(8, 8, 8, 8, &x, &y, &b);
        assert_eq!(src.padded_width(), 8);
        let (sx, sy, sb) = src.xyb_full();
        assert_eq!(sx.as_ptr(), x.as_ptr());
        assert_eq!(sy.as_ptr(), y.as_ptr());
        assert_eq!(sb.as_ptr(), b.as_ptr());
        // Default release hint is a no-op.
        src.release_dc_region(0, 0);
    }

    #[test]
    fn whole_image_source_is_sync() {
        fn assert_sync<T: Sync>(_: &T) {}
        let (x, y, b) = make_planes(64);
        let src = WholeImageXybSource::new(8, 8, 8, 8, x, y, b);
        assert_sync(&src);
    }

    #[test]
    fn borrowed_source_is_sync() {
        fn assert_sync<T: Sync>(_: &T) {}
        let v = vec![0.0f32; 64];
        let src = BorrowedXybSource::new(8, 8, 8, 8, &v, &v, &v);
        assert_sync(&src);
    }
}
