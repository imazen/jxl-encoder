// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Extra-channel plumbing for the VarDCT encoder.
//!
//! VarDCT itself only encodes the three color channels (XYB). Any
//! additional channels — alpha, depth, spot color, selection mask,
//! thermal, CFA — are encoded as a **single modular sub-bitstream**
//! that travels alongside the VarDCT data.
//!
//! All extras share one [`crate::bit_writer::BitWriter`] frame within
//! the sub-bitstream: a single GroupHeader, one local tree (gradient
//! prediction, one context), and one entropy code carrying each
//! channel's residuals in sequence. The decoder pulls
//! `channel_width * channel_height` tokens per channel out of the
//! shared stream, in the same order [`VardctExtra`]s were passed in.
//!
//! For now the writer supports any combination of 8-bit / 16-bit
//! buffers at `dim_shift == 0` (full-resolution channels). Non-zero
//! `dim_shift` values in lossy are guarded upstream with explicit
//! `Unsupported` errors so the wire format stays correct as that path
//! is filled in.

use crate::headers::extra_channels::ExtraChannelInfo;

/// Internal view of one extra channel passed to the VarDCT encoder.
///
/// Borrowed-only: encode does not take ownership of the channel data.
/// `data` may be u8 or u16; bit depth is governed by `info.bit_depth`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct VardctExtra<'a> {
    pub info: &'a ExtraChannelInfo,
    pub data: VardctExtraBuf<'a>,
}

/// Per-channel pixel buffer, exactly mirroring the public
/// [`crate::api::ExtraChannelBuf`] enum but living inside the
/// `vardct` module so internal call sites avoid the `api::` prefix.
#[derive(Debug, Clone, Copy)]
pub(crate) enum VardctExtraBuf<'a> {
    /// `channel_width * channel_height` u8 samples, row-major.
    U8(&'a [u8]),
    /// `channel_width * channel_height` u16 samples (native byte
    /// order), row-major.
    U16(&'a [u16]),
}

impl<'a> VardctExtraBuf<'a> {
    /// Read a single sample as `i32`. u8 channels return values in
    /// `[0, 255]`; u16 channels return `[0, 65535]`.
    #[inline]
    pub fn sample(&self, idx: usize) -> i32 {
        match self {
            VardctExtraBuf::U8(s) => s[idx] as i32,
            VardctExtraBuf::U16(s) => s[idx] as i32,
        }
    }
}

impl<'a> VardctExtra<'a> {
    /// Build a VardctExtra view from the public [`crate::api::ExtraChannel`].
    pub fn from_api(ec: &'a crate::api::ExtraChannel<'a>) -> Self {
        let data = match ec.data() {
            crate::api::ExtraChannelBuf::U8(s) => VardctExtraBuf::U8(s),
            crate::api::ExtraChannelBuf::U16(s) => VardctExtraBuf::U16(s),
        };
        Self {
            info: ec.info(),
            data,
        }
    }

    /// Channel width given the (full-resolution) image width. Uses
    /// the standard `image_width >> dim_shift` reduction.
    #[inline]
    pub fn channel_width(&self, image_width: usize) -> usize {
        image_width >> self.info.dim_shift
    }

    /// Scan the channel covering the rectangular region
    /// `[x0, x0+region_width) × [y0, y0+region_height)` (sampled at
    /// this channel's resolution via `dim_shift`) for the single-value
    /// constant-channel ChannelCompact opportunity.
    ///
    /// Returns `Some(value)` when every sampled pixel is equal to the
    /// same `value`. Used by
    /// [`super::bitstream::write_modular_extras_subbitstream`] to emit
    /// a libjxl-parity `kPalette` (num_c=1, nb_colors=1) transform so
    /// the original constant value survives lossy alpha quantization
    /// (`alpha_distance > 0` → `q > 1` would otherwise snap
    /// `255 → 252` at `q == 7`, W13-4 audit gap).
    ///
    /// Returns `None` on empty regions or when at least two distinct
    /// values are seen. Stops scanning at the first mismatch.
    pub(crate) fn detect_constant_value(
        &self,
        image_width: usize,
        x0: usize,
        y0: usize,
        region_width: usize,
        region_height: usize,
    ) -> Option<i32> {
        let ch_w = self.channel_width(image_width);
        let ch_x0 = x0 >> self.info.dim_shift;
        let ch_y0 = y0 >> self.info.dim_shift;
        let ch_rw = region_width >> self.info.dim_shift;
        let ch_rh = region_height >> self.info.dim_shift;
        if ch_rw == 0 || ch_rh == 0 {
            return None;
        }
        let first = self.data.sample(ch_y0 * ch_w + ch_x0);
        for y in 0..ch_rh {
            for x in 0..ch_rw {
                let v = self.data.sample((ch_y0 + y) * ch_w + (ch_x0 + x));
                if v != first {
                    return None;
                }
            }
        }
        Some(first)
    }
}
