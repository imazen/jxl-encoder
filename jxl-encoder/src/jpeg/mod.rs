// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! JPEG parsing and lossless reencoding into JPEG XL.
//!
//! This module provides a complete JPEG parser that extracts quantized DCT
//! coefficients, quantization/Huffman tables, and all metadata needed for
//! bit-exact JPEG reconstruction from a JPEG XL container.

mod data;
mod encode;
mod jbrd;
mod lossy;
mod parse;

pub use data::JpegData;
pub use encode::{
    encode_jpeg_to_jxl, encode_jpeg_to_jxl_container, encode_jpeg_to_jxl_container_with_effort,
    encode_jpeg_to_jxl_with_effort,
};
// Cancellable (Stop-polling) variants — internal; the public transcode entry
// points in `api.rs` use these for `*_with_stop`. See issue #77 item 2.
pub(crate) use encode::{
    encode_jpeg_to_jxl_container_with_effort_stop, encode_jpeg_to_jxl_with_effort_stop,
};
pub use lossy::{
    coarsen_coefficients, coarsen_coefficients_auto, coarsen_coefficients_dz,
    coarsen_coefficients_planar, coarsen_policy,
};
pub(crate) use parse::read_jpeg_with_stop;
pub use parse::{JpegError, read_jpeg};

use crate::api::Limits;
use crate::budget::MemoryBudget;
use alloc::sync::Arc;
use enough::Stop;

/// Shared decode + resource-bound setup for the PreserveJxl recompress entry
/// points (#77 follow-up).
///
/// Builds a per-encode [`MemoryBudget`] from `limits` (the lossless default
/// cap when unset, since PreserveJxl emits a lossless-floored transcode),
/// applies the pre-flight SOF pixel cap, polls `stop` during the decode, and
/// reserves the coefficient buffers — then returns the parsed [`JpegData`]
/// alongside the budget for the downstream encode(s) to share. `None` limits /
/// stop give the secure default cap / an unstoppable run.
fn read_jpeg_for_recompress(
    jpeg_bytes: &[u8],
    limits: Option<&Limits>,
    stop: Option<&dyn Stop>,
) -> Result<(JpegData, Arc<MemoryBudget>), crate::error::Error> {
    let max_pixels = limits.and_then(|l| l.max_pixels());
    let cap = limits
        .and_then(|l| l.max_memory_bytes())
        .unwrap_or(Limits::DEFAULT_MAX_MEMORY_BYTES_LOSSLESS);
    let fallible = limits.is_some_and(|l| l.fallible_alloc());
    let budget = MemoryBudget::with_alloc_policy(cap, fallible);
    let jpeg =
        read_jpeg_with_stop(jpeg_bytes, max_pixels, stop, Some(&budget)).map_err(|e| match e {
            JpegError::Cancelled => crate::error::Error::Cancelled,
            // A budget rejection is a resource limit, not malformed input — map
            // it to the limit variant rather than funnelling it into
            // `InvalidInput` (which wrongly implies a bad JPEG). The structured
            // fields were stringified crossing the public `JpegError` boundary,
            // but we own the budget here, so reconstruct from its live state:
            // `cap`/`used` are exact at the point of failure; the failing
            // `requested` amount is not recoverable across the boundary (0).
            JpegError::ResourceLimit(_) => crate::error::Error::AllocationLimit {
                requested: 0,
                used: budget.used(),
                cap: budget.cap(),
            },
            other => crate::error::Error::InvalidInput(alloc::format!("JPEG parse: {other:?}")),
        })?;
    Ok((jpeg, budget))
}

/// PreserveJxl: coefficient-domain lossy JPEG → bare JXL codestream.
///
/// Parses `jpeg_bytes`, coarsens its quantized DCT coefficients in the DCT
/// domain by `scale` (> 1.0; near-uniform scale of the source's own quant
/// tables) with AC deadzone widening `dz` in `[0.0, 0.5]` (see
/// [`coarsen_coefficients_dz`]), then losslessly transcodes the coarsened
/// coefficients to a YCbCr JXL codestream (no JBRD). The output decodes to
/// the coarsened image; `scale <= 1.0` is identical to a lossless transcode.
///
/// The result is **guaranteed ≤ the lossless transcode size**: if coarsening
/// does not shrink the codestream (e.g. very gentle settings on an already-
/// sparse source), the lossless transcode is returned instead — never a
/// larger, quality-degraded file.
///
/// `limits` bounds this untrusted-bytes path (pre-flight SOF pixel cap +
/// per-encode [`MemoryBudget`]); `None` applies the secure defaults
/// ([`Limits::DEFAULT_MAX_JPEG_TRANSCODE_PIXELS`] = 120 MP and the lossless
/// memory default). `stop` ([`enough::Stop`]) is polled at coarse boundaries
/// during both the decode and the two encodes, returning
/// [`crate::error::Error::Cancelled`] on cancellation; `None` is unstoppable.
///
/// Requires the `jpeg-reencoding` feature.
pub fn encode_jpeg_recompress_codestream(
    jpeg_bytes: &[u8],
    scale: f32,
    dz: f32,
    effort: u8,
    limits: Option<&Limits>,
    stop: Option<&dyn Stop>,
) -> Result<alloc::vec::Vec<u8>, crate::error::Error> {
    let (jpeg, budget) = read_jpeg_for_recompress(jpeg_bytes, limits, stop)?;
    // Lossless transcode is the "do no harm" floor.
    let lossless = encode_jpeg_to_jxl_with_effort_stop(&jpeg, effort, stop, Some(&budget))?;
    // NaN-safe: NaN selects the lossless floor, same as scale <= 1.0.
    if scale.is_nan() || scale <= 1.0 {
        return Ok(lossless);
    }
    let mut coarsened = jpeg;
    coarsen_coefficients_dz(&mut coarsened, scale, dz);
    let lossy = encode_jpeg_to_jxl_with_effort_stop(&coarsened, effort, stop, Some(&budget))?;
    // No-size-regression guard (RECOMPRESSION_COMPENDIUM §10.6): never ship a
    // larger, quality-degraded file than the lossless transcode.
    if lossy.len() < lossless.len() {
        Ok(lossy)
    } else {
        Ok(lossless)
    }
}

/// PreserveJxl with the bundled single-knob [`coarsen_policy`] (deadzone + mild
/// chroma lead, the proven RD-frontier defaults). The caller's quality loop
/// only moves one `scale` dial; `scale <= 1.0` is the lossless transcode. Same
/// no-size-regression guard as [`encode_jpeg_recompress_codestream`].
///
/// This is the recommended PreserveJxl entry point — it bakes in the frontier
/// findings so callers do not hand-tune deadzone/chroma.
///
/// See [`encode_jpeg_recompress_codestream`] for the `limits` / `stop`
/// resource-bound + cancellation contract.
///
/// Requires the `jpeg-reencoding` feature.
pub fn encode_jpeg_recompress_auto_codestream(
    jpeg_bytes: &[u8],
    scale: f32,
    effort: u8,
    limits: Option<&Limits>,
    stop: Option<&dyn Stop>,
) -> Result<alloc::vec::Vec<u8>, crate::error::Error> {
    let (jpeg, budget) = read_jpeg_for_recompress(jpeg_bytes, limits, stop)?;
    let lossless = encode_jpeg_to_jxl_with_effort_stop(&jpeg, effort, stop, Some(&budget))?;
    // NaN-safe: NaN selects the lossless floor, same as scale <= 1.0.
    if scale.is_nan() || scale <= 1.0 {
        return Ok(lossless);
    }
    let mut coarsened = jpeg;
    coarsen_coefficients_auto(&mut coarsened, scale);
    let lossy = encode_jpeg_to_jxl_with_effort_stop(&coarsened, effort, stop, Some(&budget))?;
    if lossy.len() < lossless.len() {
        Ok(lossy)
    } else {
        Ok(lossless)
    }
}

/// PreserveJxl with **separate luma/chroma** coarsening (see
/// [`coarsen_coefficients_planar`]). Same no-size-regression guard as
/// [`encode_jpeg_recompress_codestream`].
///
/// See [`encode_jpeg_recompress_codestream`] for the `limits` / `stop`
/// resource-bound + cancellation contract.
///
/// Requires the `jpeg-reencoding` feature.
// Separate luma/chroma scale+deadzone (4) + effort + bytes + limits + stop = 8;
// these are the PreserveJxl ablation knobs and this entry is slated to move to
// zenjxl, so the wide signature is acceptable.
#[allow(clippy::too_many_arguments)]
pub fn encode_jpeg_recompress_planar_codestream(
    jpeg_bytes: &[u8],
    luma_scale: f32,
    luma_dz: f32,
    chroma_scale: f32,
    chroma_dz: f32,
    effort: u8,
    limits: Option<&Limits>,
    stop: Option<&dyn Stop>,
) -> Result<alloc::vec::Vec<u8>, crate::error::Error> {
    let (jpeg, budget) = read_jpeg_for_recompress(jpeg_bytes, limits, stop)?;
    let lossless = encode_jpeg_to_jxl_with_effort_stop(&jpeg, effort, stop, Some(&budget))?;
    if !(luma_scale > 1.0 || chroma_scale > 1.0) {
        return Ok(lossless);
    }
    let mut coarsened = jpeg;
    coarsen_coefficients_planar(&mut coarsened, luma_scale, luma_dz, chroma_scale, chroma_dz);
    let lossy = encode_jpeg_to_jxl_with_effort_stop(&coarsened, effort, stop, Some(&budget))?;
    if lossy.len() < lossless.len() {
        Ok(lossy)
    } else {
        Ok(lossless)
    }
}

// Re-export for tests that need direct JBRD access.
#[doc(hidden)]
pub use jbrd::encode_jbrd;

// Re-exports for the chunk-4 chroma-subsampling Sub420 path
// (`vardct::chroma_subsampling::encode_rgb8_sub420_via_jpeg_path`),
// which synthesises a [`JpegData`] from raw RGB pixels and hands it
// to [`encode_jpeg_to_jxl`]. The synthesised payload needs the
// component / quant-table / component-type types — these were
// already `pub` on their parent module; the re-export just makes
// them reachable through `crate::jpeg::*` without exposing the
// `data` submodule itself.
#[doc(hidden)]
pub use data::{JpegComponent, JpegComponentType, JpegQuantTable};

/// Fast check: do the supplied bytes look like a JPEG file?
///
/// Returns `true` if `bytes` starts with the JPEG SOI (Start Of Image)
/// marker `0xFF 0xD8` followed by another `0xFF` marker byte (any
/// well-formed JPEG follows SOI immediately with another marker, no
/// padding). Returns `false` for shorter inputs.
///
/// This is a lightweight signature sniff for routing decisions
/// (e.g., "did the caller hand us JPEG bytes, route to transcode?").
/// It does not validate the JPEG structure — use
/// [`read_jpeg`] for full parsing.
#[inline]
pub fn is_jpeg_signature(bytes: &[u8]) -> bool {
    bytes.len() >= 3 && bytes[0] == 0xFF && bytes[1] == 0xD8 && bytes[2] == 0xFF
}
