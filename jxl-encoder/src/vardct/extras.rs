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

use crate::error::Result;
use crate::headers::extra_channels::ExtraChannelInfo;
use crate::modular::channel::Channel;
use crate::modular::squeeze::SqueezeParams;

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

/// One squeeze-output sub-channel of an alpha extra, after default
/// `apply_squeeze` + per-channel `QuantizeChannel` (libjxl
/// `enc_modular.cc:1024` parity). Carries the (hshift, vshift) pair the
/// decoder needs to reconstruct the original channel via inverse
/// squeeze, plus the integer quantizer `q` so the gradient-prediction
/// residuals can be divided exactly.
///
/// Used by [`super::bitstream::VarDctEncoder::write_modular_extras_alpha_squeezed`].
#[derive(Debug, Clone)]
pub(crate) struct AlphaSqueezeSubChannel {
    /// Squeeze-domain channel data (already quantized via `snap-to-q`).
    pub channel: Channel,
    /// Integer pixel quantizer for this sub-channel (libjxl
    /// `enc_modular.cc:1010-1022`, `responsive=1` luma branch). `1` =
    /// lossless leaf.
    pub q: u32,
}

/// Squeeze pipeline output for a single alpha extra.
///
/// Built by [`build_alpha_squeeze_pipeline`] when
/// [`super::encoder::VarDctEncoder::alpha_squeeze_engaged`] is `true`.
/// Holds (a) the squeeze params the decoder needs to undo the wavelet
/// (signalled in the modular GroupHeader as `nb_transforms` ≥ 1) and
/// (b) the per-sub-channel quantized data + per-sub-channel `q` so the
/// bitstream writer can dispatch via the channel-split tree.
#[derive(Debug, Clone)]
pub(crate) struct AlphaSqueezePipeline {
    /// Sub-channels in apply-order. The decoder's inverse squeeze
    /// reverses these to recover the original alpha plane.
    pub sub_channels: alloc::vec::Vec<AlphaSqueezeSubChannel>,
    /// Squeeze descriptor list (each entry → one `SqueezeParam` in the
    /// modular subbitstream GroupHeader `nb_transforms` block).
    pub squeeze_params: alloc::vec::Vec<SqueezeParams>,
}

/// Build the squeeze pipeline for a single alpha extra. Mirrors libjxl
/// `enc_modular.cc:937-1027` responsive=1 path narrowed to the
/// extras-only ModularImage (one alpha channel, no color).
///
/// Steps:
/// 1. Materialize the alpha extra as a single i32 [`Channel`] at full
///    resolution (`dim_shift == 0` and `dim_shift > 0` both supported —
///    `dim_shift > 0` callers feed in the already-subsampled buffer).
/// 2. Build a 1-channel [`crate::modular::channel::ModularImage`] and
///    invoke [`crate::modular::squeeze::default_squeeze_params`] +
///    [`crate::modular::squeeze::apply_squeeze`]. Default squeeze
///    halves alternating axes until both ≤ 8 px; produces N
///    sub-channels (1 lowest-frequency average + N-1 HF residuals).
/// 3. For each sub-channel compute its integer quantizer via the
///    shift-aware quantizer formula
///    ([`super::encoder::VarDctEncoder::compute_extra_pixel_quantizer_shifted`])
///    using `shift = (hshift + vshift) - 1` clamped to `[0, 15]`
///    (libjxl `enc_modular.cc:1006-1008`).
/// 4. In-place `QuantizeChannel` (libjxl `enc_modular.cc:141`):
///    `snap-to-multiple-of-q` round-toward-zero, lossless leaves
///    (`q == 1`) untouched.
///
/// Returns the [`AlphaSqueezePipeline`] for the bitstream writer to
/// consume. On caller-shape inputs that can't be squeezed (e.g. a
/// channel ≤ 8×8) returns an empty `squeeze_params` and a single
/// pass-through sub-channel at `shift = 0` — the writer can still emit
/// a `nb_transforms = 0` GroupHeader and use the shift=0 quantizer.
pub(crate) fn build_alpha_squeeze_pipeline(
    alpha: &VardctExtra<'_>,
    image_width: usize,
    image_height: usize,
    shift0_quantizer: u32,
    shifted_quantizer: impl Fn(u32) -> u32,
) -> Result<AlphaSqueezePipeline> {
    use crate::modular::channel::ModularImage;
    use crate::modular::squeeze::{apply_squeeze, default_squeeze_params};

    // Materialize alpha as a single i32 Channel at this channel's
    // resolution (`image_width >> dim_shift` × `image_height >> dim_shift`).
    let ch_w = alpha.channel_width(image_width);
    let ch_h = image_height >> alpha.info.dim_shift;
    debug_assert!(ch_w > 0 && ch_h > 0, "alpha squeeze: empty channel");

    let mut data: alloc::vec::Vec<i32> = alloc::vec::Vec::with_capacity(ch_w * ch_h);
    for y in 0..ch_h {
        for x in 0..ch_w {
            data.push(alpha.data.sample(y * ch_w + x));
        }
    }
    let alpha_channel = Channel::from_vec(data, ch_w, ch_h)?;

    // Single-channel modular image to drive default_squeeze_params /
    // apply_squeeze. is_grayscale=true keeps the param generator from
    // treating multiple channels as chroma siblings.
    let mut mi = ModularImage {
        channels: alloc::vec![alpha_channel],
        bit_depth: alpha.info.bit_depth.bits_per_sample,
        is_grayscale: true,
        has_alpha: false,
    };

    let params = default_squeeze_params(&mi);
    if !params.is_empty() {
        apply_squeeze(&mut mi, &params)?;
    }

    // Per-sub-channel quantize using the shift-aware quantizer.
    let mut sub_channels: alloc::vec::Vec<AlphaSqueezeSubChannel> =
        alloc::vec::Vec::with_capacity(mi.channels.len());
    for ch in mi.channels.into_iter() {
        let shift = ch.hshift + ch.vshift;
        // Libjxl `enc_modular.cc:1006-1008`: `if (shift > 0) shift--;`
        // so the lowest-frequency post-squeeze band lands at the
        // qtable[0] row. `shift == 0` (pre-squeeze pass-through) uses
        // the caller's shift0 quantizer (lets the
        // no-default-squeeze-applied fallback path keep its existing
        // calibration).
        let mut ch_mut = ch;
        let q = if shift == 0 {
            shift0_quantizer
        } else {
            shifted_quantizer(shift - 1)
        };
        quantize_channel_inplace(&mut ch_mut, q);
        sub_channels.push(AlphaSqueezeSubChannel { channel: ch_mut, q });
    }

    Ok(AlphaSqueezePipeline {
        sub_channels,
        squeeze_params: params,
    })
}

/// In-place `snap-to-multiple-of-q` quantizer. Mirrors libjxl
/// `enc_modular.cc:141` `QuantizeChannel`. `q == 1` is a no-op
/// (lossless leaf). Negative values are reflected (`-((-x+q/2)/q)*q`)
/// so the snap stays symmetric around zero.
fn quantize_channel_inplace(ch: &mut Channel, q: u32) {
    if q <= 1 {
        return;
    }
    let qi = q as i32;
    let half = qi / 2;
    let w = ch.width();
    let h = ch.height();
    for y in 0..h {
        for x in 0..w {
            let v = ch.get(x, y);
            let snapped = if v >= 0 {
                ((v + half) / qi) * qi
            } else {
                -(((-v) + half) / qi) * qi
            };
            ch.set(x, y, snapped);
        }
    }
}

#[cfg(test)]
mod squeeze_pipeline_tests {
    use super::*;
    use crate::headers::extra_channels::ExtraChannelType;

    fn alpha_info(bits: u32) -> ExtraChannelInfo {
        let mut info = ExtraChannelInfo::default();
        info.ec_type = ExtraChannelType::Alpha;
        info.bit_depth.bits_per_sample = bits;
        info
    }

    #[test]
    fn alpha_squeeze_pipeline_32x32_produces_subchannels_with_mixed_q() {
        // 32×32 alpha → default squeeze halves until both dims ≤ 8.
        // Expected: 4 alternating squeezes (32→16→8 on each axis),
        // producing 5 sub-channels (1 lowest-frequency average +
        // 4 HF residual bands).
        //
        // Sub-channel ordering after `apply_squeeze` with `in_place=true`:
        // `channels[0]` is the *lowest-frequency average* (it
        // accumulates `hshift + vshift = num_squeezes` after every
        // step — the squeezed averages keep flowing through index 0).
        // Higher indices are residual bands at shallower shift depths.
        // libjxl `enc_modular.cc:1006-1008` clamps `shift > 16` to 16
        // before decrementing, so very deep bands collapse to the
        // qtable's deepest (smallest) value.
        let info = alpha_info(8);
        let pixels: alloc::vec::Vec<u8> = (0..32u32 * 32)
            .map(|i| ((i * 13 + 5) % 256) as u8)
            .collect();
        let extra = VardctExtra {
            info: &info,
            data: VardctExtraBuf::U8(&pixels),
        };
        let pipe = build_alpha_squeeze_pipeline(
            &extra,
            32,
            32,
            /* shift0 */ 100,
            // Fake shifted_quantizer: q = max(1, 100 / 2^shift) so
            // deep bands drop toward q = 1. Mirrors the qtable shape
            // (halves per shift) without pulling in the real
            // constants.
            |shift| (100u32 >> shift).max(1),
        )
        .expect("squeeze pipeline build");
        assert!(
            !pipe.squeeze_params.is_empty(),
            "32×32 alpha should produce ≥1 squeeze param"
        );
        assert!(
            pipe.sub_channels.len() >= 2,
            "≥1 squeeze produces ≥2 sub-channels, got {}",
            pipe.sub_channels.len()
        );
        // Per-band q values should span a range — lowest-frequency
        // average (channel[0], after cumulative squeezes) takes a
        // smaller q than the last HF residual at the shallowest shift
        // (channel[N-1]). Both ends should be present and distinct
        // unless the chunk has exactly 2 sub-channels with identical
        // shifts.
        let qs: alloc::vec::Vec<u32> = pipe.sub_channels.iter().map(|sc| sc.q).collect();
        let min_q = *qs.iter().min().unwrap();
        let max_q = *qs.iter().max().unwrap();
        assert!(
            min_q < max_q,
            "sub-channel q values should span a range; got qs={qs:?}"
        );
        // The shallowest-shift residual sees the `shift0_quantizer`
        // value (100 in this test). At least one sub-channel must
        // pick that up.
        assert!(
            qs.contains(&100),
            "at least one sub-channel must use shift0_quantizer=100; got qs={qs:?}"
        );
    }

    #[test]
    fn alpha_squeeze_pipeline_8x8_skips_squeeze() {
        // Both dimensions ≤ MAX_FIRST_PREVIEW_SIZE (8) → no squeeze.
        let info = alpha_info(8);
        let pixels = alloc::vec![128u8; 8 * 8];
        let extra = VardctExtra {
            info: &info,
            data: VardctExtraBuf::U8(&pixels),
        };
        let pipe = build_alpha_squeeze_pipeline(&extra, 8, 8, 7, |_| 1).expect("build");
        assert!(
            pipe.squeeze_params.is_empty(),
            "≤8×8 input must skip squeeze"
        );
        assert_eq!(pipe.sub_channels.len(), 1, "no squeeze ⇒ 1 sub-channel");
        assert_eq!(pipe.sub_channels[0].q, 7);
    }

    #[test]
    fn quantize_channel_inplace_snaps_to_multiple_of_q() {
        let mut ch = Channel::from_vec(alloc::vec![0, 3, 5, 7, 8, 11], 6, 1).unwrap();
        quantize_channel_inplace(&mut ch, 4);
        // q=4, half=2. Snap: 0→0, 3→4, 5→4, 7→8, 8→8, 11→12
        assert_eq!(ch.get(0, 0), 0);
        assert_eq!(ch.get(1, 0), 4);
        assert_eq!(ch.get(2, 0), 4);
        assert_eq!(ch.get(3, 0), 8);
        assert_eq!(ch.get(4, 0), 8);
        assert_eq!(ch.get(5, 0), 12);
    }

    #[test]
    fn quantize_channel_inplace_q_eq_1_is_lossless() {
        let mut ch = Channel::from_vec(alloc::vec![1, 2, 3, 250], 4, 1).unwrap();
        quantize_channel_inplace(&mut ch, 1);
        assert_eq!(ch.get(0, 0), 1);
        assert_eq!(ch.get(1, 0), 2);
        assert_eq!(ch.get(2, 0), 3);
        assert_eq!(ch.get(3, 0), 250);
    }

    #[test]
    fn quantize_channel_inplace_handles_negative() {
        let mut ch = Channel::from_vec(alloc::vec![-5, -3, -1, 0, 1, 3, 5], 7, 1).unwrap();
        quantize_channel_inplace(&mut ch, 4);
        // q=4: -5→-4, -3→-4, -1→0, 0→0, 1→0, 3→4, 5→4
        assert_eq!(ch.get(0, 0), -4);
        assert_eq!(ch.get(1, 0), -4);
        assert_eq!(ch.get(2, 0), 0);
        assert_eq!(ch.get(3, 0), 0);
        assert_eq!(ch.get(4, 0), 0);
        assert_eq!(ch.get(5, 0), 4);
        assert_eq!(ch.get(6, 0), 4);
    }
}
