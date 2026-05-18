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
//!
//! ## Streaming refactor chunk 8c step D (#11)
//!
//! Adds [`StreamingXybSource`]: a concrete `XybRegionSource` impl
//! with per-DC-group release tracking + an explicit
//! [`StreamingXybSource::into_planes`] consumer that drops the
//! contiguous plane storage. The implementation is `#![forbid
//! (unsafe_code)]`-compatible (no slice-lifetime extension via raw
//! pointers); the borrow checker's `'self` lifetime on
//! [`XybRegionSource::xyb_full`] is the safety guarantee.
//!
//! Step D ships the type + 6 unit tests covering construction,
//! release tracking, idempotency, out-of-bounds release, `Sync`,
//! and the walker-contract pattern (borrow → release → into_planes).
//! Step E does NOT wire `StreamingXybSource` into
//! `super::encoder::VarDctEncoder::encode_inner` for the production
//! path because the encoder's post-transform pipeline has two
//! whole-image consumers (`compute_epf_sharpness` + the `mask1x1`
//! fallback) that read XYB after the walker would call
//! `into_planes`. Wiring would require chunk-9's per-region EPF
//! sharpness port (step A). The full chunk-9 wireup plan is
//! documented on the [`StreamingXybSource`] type itself.
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

/// Streaming XYB source: tracks per-DC-group release state and
/// exposes [`Self::into_planes`] as the single point at which the
/// contiguous plane storage returns to the allocator.
///
/// **Chunk 8c step D (#11) — structural foundation only.** This
/// type is the concrete consumer the chunk-8b walker seam
/// (`encoder.rs:2129-2137`) was waiting for; it is *not yet wired*
/// into [`super::encoder::VarDctEncoder::encode_inner`] because
/// the encoder's post-transform whole-image consumers
/// (`compute_epf_sharpness` + the `mask1x1` fallback) would read
/// XYB *after* the walker calls `into_planes`. Wiring the
/// streaming source today would either (a) require step A's
/// per-region EPF port (not landed — see "Wireup blocker" below)
/// or (b) silently disable EPF dynamic sharpness on the streaming
/// path (a quality regression). The walker conditional in
/// [`super::encoder::VarDctEncoder::encode_inner`] therefore
/// keeps engaging `WholeImageXybSource` until chunk 9 ports
/// `compute_epf_sharpness_for_region`.
///
/// What this impl lands: a concrete trait impl with per-DC-group
/// release tracking and an explicit `into_planes` consumer so
/// chunk-9 wireup has a stable surface to plug into. Unit tests
/// in this file's `tests` mod lock down the trait contract.
///
/// ### Memory model
///
/// Owns three XYB planes as contiguous `Vec<f32>`s (same layout
/// as [`WholeImageXybSource`]). On construction:
/// - The three plane Vecs.
/// - `released: Vec<bool>` of length `num_dc_groups_x *
///   num_dc_groups_y` tracking per-region release state.
/// - `released_count: u32` is an O(1) "all released" probe.
///
/// **Why drop happens at [`Self::into_planes`], not inside
/// [`release_dc_region`].** The trait method [`xyb_full`] returns
/// borrowed slices with `'self` lifetime — that lifetime is the
/// borrow checker's evidence that the planes are still alive when
/// the caller dereferences the slice. `release_dc_region` is
/// `&self` (required by the trait's `Sync` constraint for parallel
/// fan-out compatibility); it cannot drop the planes in place
/// without `unsafe` (this crate is `#![forbid(unsafe_code)]`).
/// `into_planes` takes `self` by value, which forces the borrow
/// checker to confirm every prior `xyb_full` borrow has been
/// dropped before the call — no unsafe required, zero risk of
/// dangling slices.
///
/// The walker contract is:
///
///   1. Construct the source with the XYB planes + DC-group grid.
///   2. Call [`xyb_full`] once and hand the borrow to the
///      parallel AC-group fan-out inside `transform_and_quantize`.
///   3. Wait for the fan-out to join — all borrows dropped.
///   4. Walk the DC-group grid and call [`release_dc_region`] for
///      each (bookkeeping only).
///   5. Confirm [`Self::all_released`] returns `true`, then call
///      [`Self::into_planes`]. This is the single point at which
///      the three plane Vecs return to the allocator.
///
/// ### Wireup blocker (chunk-9 prerequisite)
///
/// The encoder's post-transform pipeline has two whole-image
/// consumers that read XYB after the walker would call
/// `into_planes`:
///
///   1. [`super::epf::compute_epf_sharpness`] — IDCT-based
///      reconstruction, 3-step EPF filter (cross-block stencil up
///      to 3 px), per-block L2 errors with histogram-based
///      pass-2 context refinement that touches the *entire*
///      sharpness grid. Gated on
///      `params.epf_iters > 0 && distance >= 0.5 &&
///      profile.epf_dynamic_sharpness`.
///   2. The `mask1x1` fallback — calls
///      [`super::adaptive_quant::compute_mask1x1_with_budget`] on
///      `xyb_y`. Already hoisted into a single
///      [`super::adaptive_quant::resolve_mask1x1_for_sharpness`]
///      call by chunk-8c step B (`f434350b`), but still reads
///      whole-image `xyb_y`.
///
/// Lifting `compute_epf_sharpness` per-DC-group is the chunk-8c
/// step A that didn't ship: byte-identity bar is high because
/// pass-2 multipliers are quantised via `size_t / size_t`
/// integer division — any rounding drift in a per-DC-group
/// accumulation changes the shipped sharpness map. Chunk-9 task:
/// port `compute_epf_sharpness_for_region(dc_x, dc_y, src,
/// 2-block-border)` with byte-identity validation on the
/// rd-regression corpus + every hash_lock_features test.
///
/// ### Why mid-encode RSS doesn't drop at 4K
///
/// Even with chunk-9 wireup, the 4096² peak RSS (~2895 MB
/// measured at chunk-8c, unchanged from chunk-8b baseline) is
/// dominated by *two-pass tokenization*: BitWriter capacity
/// (~268 MB at 4K via `width * height * 4` heuristic), token
/// vectors per AC-group, ANS reverse-stream scratch, per-DC-group
/// section writers. The XYB planes (~200 MB) are already dropped
/// at the end of the EPF branch (`encoder.rs:2186-2192`) before
/// two-pass starts; freeing them per-region during transform
/// won't bring peak RSS down because (a) `quant_ac` is allocated
/// next (~192 MB) and lives through two-pass, and (b) glibc /
/// jemalloc rarely `madvise(DONTNEED)` on mid-arena Vec drops, so
/// OS-reported RSS stays high until the next allocator coalescing
/// cycle.
///
/// The real RSS wins land when a future chunk replaces the
/// whole-image two-pass tokenizer with per-DC-group tokenization
/// that streams sections to a [`std::io::Write`] sink. This
/// `StreamingXybSource` is the upstream contract that
/// per-DC-group tokenization needs to call `release_dc_region`
/// on — without it, the tokenizer would have no defined upstream
/// "release this region" semantics.
pub(crate) struct StreamingXybSource {
    width: usize,
    height: usize,
    padded_width: usize,
    padded_height: usize,
    num_dc_groups_x: u32,
    num_dc_groups_y: u32,
    planes_x: Vec<f32>,
    planes_y: Vec<f32>,
    planes_b: Vec<f32>,
    /// Per-region release bookkeeping. `std::sync::Mutex` is used
    /// because `release_dc_region` is `&self` (trait constraint
    /// for `Sync` parallel fan-out compatibility).
    bookkeeping: std::sync::Mutex<StreamingBookkeeping>,
}

struct StreamingBookkeeping {
    released: Vec<bool>,
    released_count: u32,
}

impl StreamingXybSource {
    /// Construct a streaming source from three already-padded XYB
    /// planes plus the DC-group grid dimensions. Asserts the
    /// plane lengths match `padded_width * padded_height`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        width: usize,
        height: usize,
        padded_width: usize,
        padded_height: usize,
        num_dc_groups_x: u32,
        num_dc_groups_y: u32,
        xyb_x: Vec<f32>,
        xyb_y: Vec<f32>,
        xyb_b: Vec<f32>,
    ) -> Self {
        let n = padded_width.saturating_mul(padded_height);
        debug_assert_eq!(xyb_x.len(), n, "StreamingXybSource: xyb_x length mismatch");
        debug_assert_eq!(xyb_y.len(), n, "StreamingXybSource: xyb_y length mismatch");
        debug_assert_eq!(xyb_b.len(), n, "StreamingXybSource: xyb_b length mismatch");
        let num_dc_groups = (num_dc_groups_x as usize) * (num_dc_groups_y as usize);
        Self {
            width,
            height,
            padded_width,
            padded_height,
            num_dc_groups_x,
            num_dc_groups_y,
            planes_x: xyb_x,
            planes_y: xyb_y,
            planes_b: xyb_b,
            bookkeeping: std::sync::Mutex::new(StreamingBookkeeping {
                released: alloc::vec![false; num_dc_groups],
                released_count: 0,
            }),
        }
    }

    /// Number of DC regions whose `release_dc_region` hint has
    /// fired. Useful for tests and for the walker to sanity-check
    /// that it covered the full grid before calling
    /// [`Self::into_planes`].
    #[allow(dead_code)]
    pub(crate) fn released_count(&self) -> u32 {
        self.bookkeeping.lock().unwrap().released_count
    }

    /// `true` when every DC region has been released. The walker
    /// uses this to gate the [`Self::into_planes`] call.
    #[allow(dead_code)]
    pub(crate) fn all_released(&self) -> bool {
        let bk = self.bookkeeping.lock().unwrap();
        let total = self.num_dc_groups_x.saturating_mul(self.num_dc_groups_y);
        bk.released_count >= total
    }

    /// Consume the source, returning the three plane `Vec`s in
    /// constructor order `(xyb_x, xyb_y, xyb_b)`.
    ///
    /// This is the actual point at which the contiguous plane
    /// storage returns to the allocator. The caller (the walker)
    /// must have dropped every prior [`Self::xyb_full`] borrow
    /// before calling — Rust's borrow checker enforces this at
    /// compile time because `into_planes` takes `self` by value.
    /// No `unsafe` required.
    #[allow(dead_code)]
    pub(crate) fn into_planes(self) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        (self.planes_x, self.planes_y, self.planes_b)
    }
}

impl XybRegionSource for StreamingXybSource {
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
        (&self.planes_x, &self.planes_y, &self.planes_b)
    }

    /// Mark `(dc_x, dc_y)` as released. Bookkeeping only — the
    /// actual plane Vec storage is freed by [`Self::into_planes`]
    /// (which takes `self` by value, so the borrow checker
    /// forbids any concurrent [`xyb_full`] borrow).
    ///
    /// Out-of-bounds `(dc_x, dc_y)` are silently ignored — same
    /// contract as the trait's default no-op.
    ///
    /// Idempotent: calling release twice for the same `(dc_x,
    /// dc_y)` advances `released_count` only once.
    fn release_dc_region(&self, dc_x: u32, dc_y: u32) {
        if dc_x >= self.num_dc_groups_x || dc_y >= self.num_dc_groups_y {
            return;
        }
        let idx = (dc_y as usize) * (self.num_dc_groups_x as usize) + (dc_x as usize);
        let mut bk = self.bookkeeping.lock().unwrap();
        if idx < bk.released.len() && !bk.released[idx] {
            bk.released[idx] = true;
            bk.released_count = bk.released_count.saturating_add(1);
        }
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

    #[test]
    fn streaming_source_round_trips_xyb_full_before_release() {
        let (x, y, b) = make_planes(16 * 16);
        let src = StreamingXybSource::new(16, 16, 16, 16, 1, 1, x.clone(), y.clone(), b.clone());
        assert_eq!(src.width(), 16);
        assert_eq!(src.padded_width(), 16);
        let (sx, sy, sb) = src.xyb_full();
        assert_eq!(sx, x.as_slice());
        assert_eq!(sy, y.as_slice());
        assert_eq!(sb, b.as_slice());
        assert!(!src.all_released());
        assert_eq!(src.released_count(), 0);
    }

    #[test]
    fn streaming_source_tracks_release_state() {
        // 4×3 = 12 DC regions; release them all, expect
        // `all_released` to flip and `into_planes` to return the
        // original storage in order.
        let (x, y, b) = make_planes(32 * 24);
        let src = StreamingXybSource::new(32, 24, 32, 24, 4, 3, x.clone(), y.clone(), b.clone());
        assert!(!src.all_released(), "fresh source must not be all-released");
        // Release 11 of 12 — counter advances, all_released still false.
        for i in 0..11u32 {
            src.release_dc_region(i % 4, i / 4);
        }
        assert!(
            !src.all_released(),
            "all_released must remain false until every region is released"
        );
        assert_eq!(src.released_count(), 11);
        // Final release flips all_released.
        src.release_dc_region(3, 2);
        assert!(
            src.all_released(),
            "all_released must fire on final release"
        );
        assert_eq!(src.released_count(), 12);
        // into_planes returns the original storage in order.
        // (The walker calls this after all_released to drop the
        // planes.)
        let (rx, ry, rb) = src.into_planes();
        assert_eq!(rx, x);
        assert_eq!(ry, y);
        assert_eq!(rb, b);
    }

    #[test]
    fn streaming_source_release_is_idempotent() {
        let (x, y, b) = make_planes(16 * 8);
        let src = StreamingXybSource::new(16, 8, 16, 8, 2, 1, x, y, b);
        // Same region released twice — counter advances once.
        src.release_dc_region(0, 0);
        src.release_dc_region(0, 0);
        src.release_dc_region(0, 0);
        assert_eq!(src.released_count(), 1);
        assert!(!src.all_released());
        // Final region.
        src.release_dc_region(1, 0);
        assert!(src.all_released());
    }

    #[test]
    fn streaming_source_release_out_of_bounds_is_noop() {
        let (x, y, b) = make_planes(16 * 8);
        let src = StreamingXybSource::new(16, 8, 16, 8, 2, 1, x, y, b);
        src.release_dc_region(2, 0); // x past grid
        src.release_dc_region(0, 1); // y past grid
        src.release_dc_region(42, 42); // far past grid
        assert_eq!(src.released_count(), 0);
        assert!(!src.all_released());
    }

    #[test]
    fn streaming_source_is_sync() {
        fn assert_sync<T: Sync>(_: &T) {}
        let (x, y, b) = make_planes(64);
        let src = StreamingXybSource::new(8, 8, 8, 8, 1, 1, x, y, b);
        assert_sync(&src);
    }

    #[test]
    fn streaming_source_xyb_full_remains_valid_across_releases() {
        // The walker contract: xyb_full() borrows are held during
        // transform_and_quantize; release_dc_region calls run
        // after the borrows drop. This test verifies the
        // bookkeeping doesn't invalidate prior borrows (borrow
        // checker would reject the pattern at compile time if it
        // did, since release takes &self).
        let (x, y, b) = make_planes(16 * 8);
        let src = StreamingXybSource::new(16, 8, 16, 8, 2, 1, x.clone(), y.clone(), b.clone());
        // Borrow snapshot before any releases.
        let (sx, sy, sb) = src.xyb_full();
        assert_eq!(sx, x.as_slice());
        // Release some regions while borrows are still live (they
        // are NOT invalidated because into_planes hasn't run).
        src.release_dc_region(0, 0);
        assert_eq!(sy, y.as_slice());
        src.release_dc_region(1, 0);
        assert_eq!(sb, b.as_slice());
        // Source still has its planes; into_planes would take
        // them. all_released gates that step.
        assert!(src.all_released());
    }
}
