// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Pixel-ingestion and request-plumbing helpers: transfer-function decode
//! (sRGB/PQ/HLG/BT.709/gamma, u8/u16/f16/f32 to linear f32), CMYK and
//! grayscale expansion, alpha extraction, strided unpacking, container
//! wrapping, level computation. All `pub(crate)`, called from the config and
//! request impls in `super` (`crate::api`).

use super::*;

pub(crate) const SRGB_U8_TO_LINEAR: [f32; 256] = {
    let mut table = [0.0f32; 256];
    let mut i = 0u16;
    while i < 256 {
        let c = i as f64 / 255.0;
        // Use f64 for accuracy during const eval, then truncate to f32.
        // powf is not const, so we use exp(2.4 * ln(x)) via a manual series.
        // For const context, we precompute using the piecewise sRGB TF.
        table[i as usize] = if c <= 0.04045 {
            (c / 12.92) as f32
        } else {
            // ((c + 0.055) / 1.055)^2.4
            // = exp(2.4 * ln((c + 0.055) / 1.055))
            // Approximate via repeated squaring: x^2.4 = x^2 * x^0.4
            // x^0.4 = (x^0.5)^0.8 = ((x^0.5)^0.5)^... too complex for const.
            // Instead, use the identity: x^2.4 = (x^12)^(1/5)
            // and compute fifth root via Newton's method in f64.
            let base = (c + 0.055) / 1.055;
            // x^12 = ((x^2)^2)^3
            let x2 = base * base;
            let x4 = x2 * x2;
            let x8 = x4 * x4;
            let x12 = x8 * x4;
            // Fifth root of x^12 = x^(12/5) = x^2.4
            // Newton: y_{n+1} = y_n - (y_n^5 - x12) / (5 * y_n^4)
            //       = (4*y_n + x12/y_n^4) / 5
            let mut y = base * base; // initial guess ~x^2
            // 8 iterations of Newton's method for fifth root (converges in ~6 for f64)
            let mut iter = 0;
            while iter < 8 {
                let y2 = y * y;
                let y4 = y2 * y2;
                y = (4.0 * y + x12 / y4) / 5.0;
                iter += 1;
            }
            y as f32
        };
        i += 1;
    }
    table
};

/// sRGB u8 → linear f32 via LUT.
#[inline]
pub(crate) fn srgb_to_linear(c: u8) -> f32 {
    SRGB_U8_TO_LINEAR[c as usize]
}

pub(crate) fn srgb_u8_to_linear_f32(data: &[u8], channels: usize) -> Vec<f32> {
    let num_pixels = data.len() / channels;
    let mut out = vec![0.0f32; num_pixels * 3];
    let lut = &SRGB_U8_TO_LINEAR;
    // Row-strip parallel fill (disjoint chunks, same values — exact).
    // The sequential loop was ~50 ms at 4K and sat OUTSIDE encode_inner,
    // the single biggest piece of the lossy e3 CLI-vs-core wall gap.
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        const STRIP_PX: usize = 1 << 14;
        if num_pixels >= STRIP_PX * 4 {
            out.par_chunks_mut(3 * STRIP_PX)
                .zip(data.par_chunks(channels * STRIP_PX))
                .for_each(|(o, d)| {
                    for (px, rgb) in d.chunks_exact(channels).zip(o.chunks_exact_mut(3)) {
                        rgb[0] = lut[px[0] as usize];
                        rgb[1] = lut[px[1] as usize];
                        rgb[2] = lut[px[2] as usize];
                    }
                });
            return out;
        }
    }
    // zip chunks to eliminate output bounds checks; u8 index into [f32; 256] is always in bounds
    for (px, rgb) in data.chunks_exact(channels).zip(out.chunks_exact_mut(3)) {
        rgb[0] = lut[px[0] as usize];
        rgb[1] = lut[px[1] as usize];
        rgb[2] = lut[px[2] as usize];
    }
    out
}

/// PQ u8 → linear f32 RGB. Uses a 256-entry LUT (avoids per-pixel
/// powf — matches the gamma_u8_to_linear_f32 optimization). 8-bit
/// PQ is unusual in practice (PQ's headroom rewards wider precision)
/// but accepting it lets callers tag low-bit-depth content correctly.
/// BT.2100 HLG OOTF gamma for a display peak (libjxl
/// `ApplyHlgOotf`, `jxl_cms.cc:868`): `1.2 * 1.111^log2(nits/1000)`.
/// Returns `None` inside the 295..=305-nit band where gamma ~= 1 and
/// libjxl skips the pass entirely.
pub(crate) fn hlg_ootf_gamma(intensity_target: f32) -> Option<f32> {
    if (295.0..=305.0).contains(&intensity_target) {
        return None;
    }
    Some(1.2 * 1.111_f32.powf((intensity_target * 1e-3).log2()))
}

/// Primaries luminances for the HLG OOTF's luma weighting (libjxl
/// `GetPrimariesLuminances` outputs for the enum primaries; the Y row
/// of the RGB->XYZ matrix). `Custom` falls back to sRGB.
pub(crate) fn hlg_ootf_luminances(
    primaries: crate::headers::color_encoding::Primaries,
) -> [f32; 3] {
    use crate::headers::color_encoding::Primaries;
    match primaries {
        Primaries::Bt2100 => [0.262_700_2, 0.677_998_07, 0.059_301_7],
        Primaries::P3 => [0.228_974_64, 0.691_738_55, 0.079_286_91],
        Primaries::Srgb | Primaries::Custom => [0.212_639_06, 0.715_168_65, 0.072_192_32],
    }
}

/// Forward HLG OOTF (scene light -> display light), libjxl
/// `ApplyHlgOotf(forward=true)` parity: per-pixel luma-weighted ratio
/// `Y_s^(gamma-1)`, with the hue-preserving gamut normalize when
/// gamma < 1 pushes highlights above 1.0. The encoder MUST apply this
/// before XYB for HLG input (issue #73 follow-up: without it, every
/// decoder's linear->HLG conversion applies the inverse OOTF and the
/// round-trip lands at scene^(1/gamma) — a constant ~22 dB wedge,
/// distance-flat).
pub(crate) fn apply_hlg_forward_ootf(rgb: &mut [f32], luminances: [f32; 3], gamma: f32) {
    let [lr, lg, lb] = luminances;
    for px in rgb.chunks_exact_mut(3) {
        let luminance = px[0] * lr + px[1] * lg + px[2] * lb;
        let ratio = luminance.powf(gamma - 1.0);
        if ratio.is_finite() {
            px[0] *= ratio;
            px[1] *= ratio;
            px[2] *= ratio;
            if gamma < 1.0 {
                let maximum = px[0].max(px[1]).max(px[2]);
                if maximum > 1.0 {
                    let normalizer = 1.0 / maximum;
                    px[0] *= normalizer;
                    px[1] *= normalizer;
                    px[2] *= normalizer;
                }
            }
        }
    }
}

pub(crate) fn pq_u8_to_linear_f32(data: &[u8], channels: usize) -> Vec<f32> {
    let lut: [f32; 256] = core::array::from_fn(|i| pq_to_linear_f(i as f32 / 255.0));
    data.chunks(channels)
        .flat_map(|px| {
            [
                lut[px[0] as usize],
                lut[px[1] as usize],
                lut[px[2] as usize],
            ]
        })
        .collect()
}

/// HLG u8 → linear f32 RGB. 256-entry LUT.
pub(crate) fn hlg_u8_to_linear_f32(data: &[u8], channels: usize) -> Vec<f32> {
    let lut: [f32; 256] = core::array::from_fn(|i| hlg_to_linear_f(i as f32 / 255.0));
    data.chunks(channels)
        .flat_map(|px| {
            [
                lut[px[0] as usize],
                lut[px[1] as usize],
                lut[px[2] as usize],
            ]
        })
        .collect()
}

/// BT.709 u8 → linear f32 RGB. 256-entry LUT.
pub(crate) fn bt709_u8_to_linear_f32(data: &[u8], channels: usize) -> Vec<f32> {
    let lut: [f32; 256] = core::array::from_fn(|i| bt709_to_linear_f(i as f32 / 255.0));
    data.chunks(channels)
        .flat_map(|px| {
            [
                lut[px[0] as usize],
                lut[px[1] as usize],
                lut[px[2] as usize],
            ]
        })
        .collect()
}

/// PQ u16 → linear f32. `u16_max` mirrors the convention in
/// `srgb_u16_to_linear_f32` — the divisor for input normalization.
/// Output is the normalized SMPTE ST 2084 EOTF: linear [0..1] where
/// 1.0 = 10,000 nits (the PQ peak). The lossy path pairs this with
/// `intensity_target = 10_000` (libjxl `SetIntensityTarget` parity —
/// issue #73); with any other intensity_target the decoder
/// misinterprets the scale. Closes PQ portion of #17.
pub(crate) fn pq_u16_to_linear_f32(data: &[u8], channels: usize, u16_max: f32) -> Vec<f32> {
    let pixels: &[u16] = &cast_pixel_lanes(data);
    pixels
        .chunks(channels)
        .flat_map(|px| {
            [
                pq_to_linear_f(px[0] as f32 / u16_max),
                pq_to_linear_f(px[1] as f32 / u16_max),
                pq_to_linear_f(px[2] as f32 / u16_max),
            ]
        })
        .collect()
}

/// BT.709 u16 → linear f32. Same shape as `pq_u16_to_linear_f32`.
/// Closes BT.709 portion of #17.
pub(crate) fn bt709_u16_to_linear_f32(data: &[u8], channels: usize, u16_max: f32) -> Vec<f32> {
    let pixels: &[u16] = &cast_pixel_lanes(data);
    pixels
        .chunks(channels)
        .flat_map(|px| {
            [
                bt709_to_linear_f(px[0] as f32 / u16_max),
                bt709_to_linear_f(px[1] as f32 / u16_max),
                bt709_to_linear_f(px[2] as f32 / u16_max),
            ]
        })
        .collect()
}

/// HLG u16 → linear scene-light f32. Same shape as
/// `pq_u16_to_linear_f32`. Closes HLG portion of #17.
pub(crate) fn hlg_u16_to_linear_f32(data: &[u8], channels: usize, u16_max: f32) -> Vec<f32> {
    let pixels: &[u16] = &cast_pixel_lanes(data);
    pixels
        .chunks(channels)
        .flat_map(|px| {
            [
                hlg_to_linear_f(px[0] as f32 / u16_max),
                hlg_to_linear_f(px[1] as f32 / u16_max),
                hlg_to_linear_f(px[2] as f32 / u16_max),
            ]
        })
        .collect()
}

/// sRGB u16 → linear f32 (IEC 61966-2-1).
///
/// `u16_max` is the divisor for input normalization — `65535.0` for
/// full 16-bit input (the default), or `(1 << bits) - 1` for narrower
/// precision (e.g., 1023 for 10-bit, 4095 for 12-bit, 16383 for 14-bit).
/// See `EncodeRequest::with_bits_per_sample`.
pub(crate) fn srgb_u16_to_linear_f32(data: &[u8], channels: usize, u16_max: f32) -> Vec<f32> {
    let pixels: &[u16] = &cast_pixel_lanes(data);
    pixels
        .chunks(channels)
        .flat_map(|px| {
            [
                srgb_to_linear_f(px[0] as f32 / u16_max),
                srgb_to_linear_f(px[1] as f32 / u16_max),
                srgb_to_linear_f(px[2] as f32 / u16_max),
            ]
        })
        .collect()
}

/// sRGB transfer function: normalized float [0,1] → linear float.
#[inline]
pub(crate) fn srgb_to_linear_f(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        jxl_simd::fast_powf((c + 0.055) / 1.055, 2.4)
    }
}

/// PQ (SMPTE ST 2084) EOTF: PQ-encoded normalized [0,1] → linear [0,1]
/// where 1.0 = peak luminance (= the encoder's `intensity_target`,
/// typically 10 000 nits for full-spec PQ). Closes PQ portion of #17.
///
/// Constants per SMPTE ST 2084-2014 (m1 / m2 / c1 / c2 / c3). Negative
/// inputs are clamped to 0; outputs are non-negative by construction.
#[inline]
pub(crate) fn pq_to_linear_f(c: f32) -> f32 {
    const M1: f32 = 2610.0 / 16384.0; // 0.1593017578125
    const M2: f32 = (2523.0 / 4096.0) * 128.0; // 78.84375
    const C1: f32 = 3424.0 / 4096.0; // 0.8359375
    const C2: f32 = (2413.0 / 4096.0) * 32.0; // 18.8515625
    const C3: f32 = (2392.0 / 4096.0) * 32.0; // 18.6875
    let e = c.max(0.0);
    let n = jxl_simd::fast_powf(e, 1.0 / M2);
    // numerator clamped at 0; denominator can't reach 0 in [0,1] domain
    // (c2 = 18.85, c3 = 18.69, c3*N <= 18.69 at N=1, c2 - c3*N >= 0.16)
    let num = (n - C1).max(0.0);
    let den = C2 - C3 * n;
    jxl_simd::fast_powf(num / den, 1.0 / M1)
}

/// BT.709 inverse OETF (Rec. ITU-R BT.709-6, the broadcast camera
/// transfer): encoded normalized [0,1] → linear scene-light [0,1].
///
/// Piecewise: linear toe below 0.081 (= 4.5 × 0.018) plus a power
/// curve above with effective inverse gamma ≈ 2.222. Note this is
/// the SCENE-light EOTF (the inverse of the broadcast OETF), NOT the
/// display EOTF (which would be a pure gamma 2.4 per BT.1886).
/// Matches libjxl's interpretation of `TransferFunction::Bt709` for
/// encoder input. Closes BT.709 portion of #17.
#[inline]
pub(crate) fn bt709_to_linear_f(c: f32) -> f32 {
    // Threshold = beta * alpha = 0.018 * 4.5 = 0.081 (encoded value
    // below which the toe is linear). Some references quote 0.0812
    // due to the alpha = 1.099 derivation; we use the spec's exact
    // 0.081 cutoff per Rec. BT.709-6 §1.2.
    const TOE_CUTOFF: f32 = 0.081;
    let e = c.max(0.0);
    if e <= TOE_CUTOFF {
        e / 4.5
    } else {
        jxl_simd::fast_powf((e + 0.099) / 1.099, 1.0 / 0.45)
    }
}

/// HLG (Hybrid Log-Gamma, BT.2100 / ARIB STD-B67) inverse OETF:
/// HLG-encoded normalized [0,1] → linear scene-light [0,1].
///
/// HLG is piecewise: a square-root-like toe in the lower half plus a
/// logarithmic shoulder in the upper half. Scene-light output is in
/// [0, 1] where 1.0 = peak signal; downstream display mapping (the
/// HLG OOTF) is the decoder's responsibility, NOT the encoder's.
///
/// Closes HLG portion of #17.
#[inline]
pub(crate) fn hlg_to_linear_f(c: f32) -> f32 {
    const A: f32 = 0.17883277;
    const B: f32 = 1.0 - 4.0 * A; // 0.28466892
    // c_const = 0.5 - a * ln(4 * a). Hard-coded literal because the
    // spec gives this value to high precision and we want bit-exact
    // agreement with reference decoders.
    const C_CONST: f32 = 0.55991073;
    let e = c.max(0.0);
    if e <= 0.5 {
        // Lower half: square-root-like toe. L = E²/3.
        (e * e) / 3.0
    } else {
        // Upper half: logarithmic shoulder. L = (exp((E - c)/a) + b)/12.
        // The /12 normalization keeps L in [0, 1] for E in [0, 1]
        // (HLG peak signal corresponds to 12 × the SDR diffuse white).
        ((((e - C_CONST) / A).exp()) + B) / 12.0
    }
}

/// Gamma u8 → linear f32 RGB. `linear = (encoded/255)^(1/gamma)`
pub(crate) fn gamma_u8_to_linear_f32(data: &[u8], channels: usize, gamma: f32) -> Vec<f32> {
    // Build 256-entry LUT for u8 values (avoids per-pixel powf)
    let inv_gamma = 1.0 / gamma;
    let lut: [f32; 256] =
        core::array::from_fn(|i| jxl_simd::fast_powf(i as f32 / 255.0, inv_gamma));
    data.chunks(channels)
        .flat_map(|px| {
            [
                lut[px[0] as usize],
                lut[px[1] as usize],
                lut[px[2] as usize],
            ]
        })
        .collect()
}

/// Gamma u16 → linear f32 RGB. `linear = (encoded/u16_max)^(1/gamma)`
pub(crate) fn gamma_u16_to_linear_f32(
    data: &[u8],
    channels: usize,
    gamma: f32,
    u16_max: f32,
) -> Vec<f32> {
    let inv_gamma = 1.0 / gamma;
    let pixels: &[u16] = &cast_pixel_lanes(data);
    pixels
        .chunks(channels)
        .flat_map(|px| {
            [
                jxl_simd::fast_powf(px[0] as f32 / u16_max, inv_gamma),
                jxl_simd::fast_powf(px[1] as f32 / u16_max, inv_gamma),
                jxl_simd::fast_powf(px[2] as f32 / u16_max, inv_gamma),
            ]
        })
        .collect()
}

/// CMY u8 + K u8 → linear-light f32 RGB via the naive uncalibrated
/// subtractive model: `R = (1 - C/255) * (1 - K/255)` etc., where
/// each ink absorbs its complementary primary in linear light.
///
/// This is the chunk-3 follow-on to the chunk-2 placeholder that
/// treated CMY as if it were sRGB-encoded RGB bytes (which had no
/// physical basis at all — a fully-saturated cyan ink would encode
/// as bright red in XYB and decode to an entirely wrong colour
/// family). The 1-CMY model is still an approximation: it ignores
/// per-ink chromaticity, dot-gain, illuminant, and printer profile,
/// so output won't match a colorimetric CMYK→sRGB conversion done
/// through an ICC profile. But it puts the colours in the right
/// half of the gamut — a pure cyan input now encodes as cyan-ish
/// (no red component), which the XYB perceptual model can quantise
/// sensibly. A future chunk can wire the caller's CMYK ICC profile
/// (option A) or a hardcoded SWOP/FOGRA matrix (option B) for
/// colorimetric accuracy.
///
/// `K` is also kept as a modular extra channel further down the
/// pipeline so the K plane round-trips losslessly — the CMY→RGB
/// transform here is purely for perceptual quantisation of the
/// colour content.
pub(crate) fn cmyk_u8_to_linear_f32_rgb(cmy: &[u8], k: &[u8]) -> Vec<f32> {
    debug_assert_eq!(cmy.len(), k.len() * 3);
    let inv = 1.0f32 / 255.0;
    let mut out = Vec::with_capacity(k.len() * 3);
    for (px, &kv) in cmy.chunks_exact(3).zip(k.iter()) {
        let one_minus_k = 1.0 - (kv as f32) * inv;
        out.push((1.0 - (px[0] as f32) * inv) * one_minus_k);
        out.push((1.0 - (px[1] as f32) * inv) * one_minus_k);
        out.push((1.0 - (px[2] as f32) * inv) * one_minus_k);
    }
    out
}

/// CMY u16 + K u16 → linear-light f32 RGB. Same 1-CMY × (1-K) model
/// as the 8-bit variant; `u16_max` is the bit-depth normaliser (e.g.
/// `65535.0` for full-precision 16-bit input).
pub(crate) fn cmyk_u16_to_linear_f32_rgb(cmy: &[u8], k: &[u16], u16_max: f32) -> Vec<f32> {
    let cmy_u16: &[u16] = &cast_pixel_lanes(cmy);
    debug_assert_eq!(cmy_u16.len(), k.len() * 3);
    let inv = 1.0f32 / u16_max;
    let mut out = Vec::with_capacity(k.len() * 3);
    for (px, &kv) in cmy_u16.chunks_exact(3).zip(k.iter()) {
        let one_minus_k = 1.0 - (kv as f32) * inv;
        out.push((1.0 - (px[0] as f32) * inv) * one_minus_k);
        out.push((1.0 - (px[1] as f32) * inv) * one_minus_k);
        out.push((1.0 - (px[2] as f32) * inv) * one_minus_k);
    }
    out
}

/// Gamma u8 grayscale → linear f32 RGB (gray→R=G=B). `linear = (encoded/255)^(1/gamma)`
pub(crate) fn gamma_gray_u8_to_linear_f32_rgb(data: &[u8], stride: usize, gamma: f32) -> Vec<f32> {
    let inv_gamma = 1.0 / gamma;
    let lut: [f32; 256] =
        core::array::from_fn(|i| jxl_simd::fast_powf(i as f32 / 255.0, inv_gamma));
    data.chunks(stride)
        .flat_map(|px| {
            let v = lut[px[0] as usize];
            [v, v, v]
        })
        .collect()
}

/// Gamma u16 grayscale → linear f32 RGB (gray→R=G=B). `linear = (encoded/u16_max)^(1/gamma)`
pub(crate) fn gamma_gray_u16_to_linear_f32_rgb(
    data: &[u8],
    stride: usize,
    gamma: f32,
    u16_max: f32,
) -> Vec<f32> {
    let inv_gamma = 1.0 / gamma;
    let pixels: &[u16] = &cast_pixel_lanes(data);
    pixels
        .chunks(stride)
        .flat_map(|px| {
            let v = jxl_simd::fast_powf(px[0] as f32 / u16_max, inv_gamma);
            [v, v, v]
        })
        .collect()
}

/// Extract alpha channel from interleaved 16-bit pixel data as u8 (quantized).
///
/// `u16_max` is the source-precision max value (65535 for 16-bit,
/// `(1 << bits) - 1` for narrower precision). Used to scale alpha
/// from `0..=u16_max` to `0..=255` correctly.
pub(crate) fn extract_alpha_u16(
    data: &[u8],
    stride: usize,
    alpha_offset: usize,
    u16_max: f32,
) -> Vec<u8> {
    let pixels: &[u16] = &cast_pixel_lanes(data);
    pixels
        .chunks(stride)
        .map(|px| ((px[alpha_offset] as f32 / u16_max).clamp(0.0, 1.0) * 255.0 + 0.5) as u8)
        .collect()
}

/// W44-91: compute the cheap zenanalyze-equivalent proxies for the
/// high-distance smooth-photo gate widening when the input layout is
/// 8-bit sRGB-like. Returns `None` for all other layouts (16-bit,
/// linear-f32, grayscale, HDR, CMYK) where the M3 colourfulness scale
/// and per-block range threshold are not well-defined.
///
/// See [`crate::vardct::encoder::ZenanalyzeProxies::compute_srgb_u8`]
/// for the per-byte definitions (matches zenanalyze `src/tier1.rs`
/// colourfulness and `flat_color_blocks` accumulators exactly).
pub(crate) fn compute_w44_91_zenanalyze_proxies(
    pixels: &[u8],
    width: usize,
    height: usize,
    layout: PixelLayout,
) -> Option<crate::vardct::encoder::ZenanalyzeProxies> {
    use crate::vardct::encoder::ZenanalyzeProxies;
    // Per-layout (R offset, G offset, B offset, bytes per pixel). Only the
    // 8-bit sRGB layouts have a meaningful M3 colourfulness scale —
    // everything else stays `None`.
    let (r_off, g_off, b_off, bpp) = match layout {
        PixelLayout::Rgb8 => (0, 1, 2, 3),
        PixelLayout::Rgba8 => (0, 1, 2, 4),
        PixelLayout::Bgr8 => (2, 1, 0, 3),
        PixelLayout::Bgra8 => (2, 1, 0, 4),
        _ => return None,
    };
    let expected_len = width.checked_mul(height)?.checked_mul(bpp)?;
    if pixels.len() < expected_len || width == 0 || height == 0 {
        return None;
    }
    Some(ZenanalyzeProxies::compute_srgb_u8(
        pixels, width, height, bpp, r_off, g_off, b_off,
    ))
}

/// Swap B and R channels in-place equivalent: BGR(A) → RGB(A).
pub(crate) fn bgr_to_rgb(data: &[u8], stride: usize) -> Vec<u8> {
    let mut out = data.to_vec();
    for chunk in out.chunks_mut(stride) {
        chunk.swap(0, 2);
    }
    out
}

/// Extract a single channel from interleaved pixel data.
pub(crate) fn extract_alpha(data: &[u8], stride: usize, alpha_offset: usize) -> Vec<u8> {
    data.chunks(stride).map(|px| px[alpha_offset]).collect()
}

/// Extract alpha from interleaved f32 pixel data, converting to u8 (0..255).
pub(crate) fn extract_alpha_f32(data: &[f32], stride: usize, alpha_offset: usize) -> Vec<u8> {
    data.chunks(stride)
        .map(|px| (px[alpha_offset].clamp(0.0, 1.0) * 255.0 + 0.5) as u8)
        .collect()
}

/// Expand 8-bit sRGB grayscale to linear f32 RGB (gray→R=G=B).
pub(crate) fn gray_u8_to_linear_f32_rgb(data: &[u8], stride: usize) -> Vec<f32> {
    data.chunks(stride)
        .flat_map(|px| {
            let v = srgb_to_linear(px[0]);
            [v, v, v]
        })
        .collect()
}

/// Expand 16-bit sRGB grayscale to linear f32 RGB (gray→R=G=B).
pub(crate) fn gray_u16_to_linear_f32_rgb(data: &[u8], stride: usize, u16_max: f32) -> Vec<f32> {
    let pixels: &[u16] = &cast_pixel_lanes(data);
    pixels
        .chunks(stride)
        .flat_map(|px| {
            let v = srgb_to_linear_f(px[0] as f32 / u16_max);
            [v, v, v]
        })
        .collect()
}

/// Expand u8 grayscale to linear f32 RGB via the PQ EOTF. Uses a
/// 256-entry LUT to avoid per-pixel powf, mirroring the PQ u8 RGB
/// helper. Closes Gray PQ portion of #17.
pub(crate) fn pq_gray_u8_to_linear_f32_rgb(data: &[u8], stride: usize) -> Vec<f32> {
    let lut: [f32; 256] = core::array::from_fn(|i| pq_to_linear_f(i as f32 / 255.0));
    data.chunks(stride)
        .flat_map(|px| {
            let v = lut[px[0] as usize];
            [v, v, v]
        })
        .collect()
}

/// Expand u16 grayscale to linear f32 RGB via the PQ EOTF.
pub(crate) fn pq_gray_u16_to_linear_f32_rgb(data: &[u8], stride: usize, u16_max: f32) -> Vec<f32> {
    let pixels: &[u16] = &cast_pixel_lanes(data);
    pixels
        .chunks(stride)
        .flat_map(|px| {
            let v = pq_to_linear_f(px[0] as f32 / u16_max);
            [v, v, v]
        })
        .collect()
}

/// Expand u8 grayscale to linear f32 RGB via the HLG inverse OETF.
pub(crate) fn hlg_gray_u8_to_linear_f32_rgb(data: &[u8], stride: usize) -> Vec<f32> {
    let lut: [f32; 256] = core::array::from_fn(|i| hlg_to_linear_f(i as f32 / 255.0));
    data.chunks(stride)
        .flat_map(|px| {
            let v = lut[px[0] as usize];
            [v, v, v]
        })
        .collect()
}

/// Expand u16 grayscale to linear f32 RGB via the HLG inverse OETF.
pub(crate) fn hlg_gray_u16_to_linear_f32_rgb(data: &[u8], stride: usize, u16_max: f32) -> Vec<f32> {
    let pixels: &[u16] = &cast_pixel_lanes(data);
    pixels
        .chunks(stride)
        .flat_map(|px| {
            let v = hlg_to_linear_f(px[0] as f32 / u16_max);
            [v, v, v]
        })
        .collect()
}

/// Expand u8 grayscale to linear f32 RGB via the BT.709 inverse OETF.
pub(crate) fn bt709_gray_u8_to_linear_f32_rgb(data: &[u8], stride: usize) -> Vec<f32> {
    let lut: [f32; 256] = core::array::from_fn(|i| bt709_to_linear_f(i as f32 / 255.0));
    data.chunks(stride)
        .flat_map(|px| {
            let v = lut[px[0] as usize];
            [v, v, v]
        })
        .collect()
}

/// Expand u16 grayscale to linear f32 RGB via the BT.709 inverse OETF.
pub(crate) fn bt709_gray_u16_to_linear_f32_rgb(
    data: &[u8],
    stride: usize,
    u16_max: f32,
) -> Vec<f32> {
    let pixels: &[u16] = &cast_pixel_lanes(data);
    pixels
        .chunks(stride)
        .flat_map(|px| {
            let v = bt709_to_linear_f(px[0] as f32 / u16_max);
            [v, v, v]
        })
        .collect()
}

/// PQ-encoded f32 → linear f32 RGB. Input is interleaved
/// `stride`-channels-per-pixel where each channel is a PQ-encoded
/// `[0, 1]` value. Output is linear `[0, 1]` (where 1.0 = peak
/// luminance per the encoder's `intensity_target`).
///
/// A3 chunk 1b (issue #46). No LUT — input is already float, so the
/// per-pixel `powf` cost is unavoidable. Use the u8/u16 helpers for
/// quantized input.
pub(crate) fn pq_f32_to_linear_f32_rgb(data: &[f32], stride: usize) -> Vec<f32> {
    data.chunks(stride)
        .flat_map(|px| {
            [
                pq_to_linear_f(px[0]),
                pq_to_linear_f(px[1]),
                pq_to_linear_f(px[2]),
            ]
        })
        .collect()
}

/// HLG-encoded f32 → linear (scene-light) f32 RGB. See
/// [`pq_f32_to_linear_f32_rgb`] for shape. A3 chunk 1b (issue #46).
pub(crate) fn hlg_f32_to_linear_f32_rgb(data: &[f32], stride: usize) -> Vec<f32> {
    data.chunks(stride)
        .flat_map(|px| {
            [
                hlg_to_linear_f(px[0]),
                hlg_to_linear_f(px[1]),
                hlg_to_linear_f(px[2]),
            ]
        })
        .collect()
}

/// BT.709-encoded f32 → linear f32 RGB. See
/// [`pq_f32_to_linear_f32_rgb`] for shape. A3 chunk 1b (issue #46).
pub(crate) fn bt709_f32_to_linear_f32_rgb(data: &[f32], stride: usize) -> Vec<f32> {
    data.chunks(stride)
        .flat_map(|px| {
            [
                bt709_to_linear_f(px[0]),
                bt709_to_linear_f(px[1]),
                bt709_to_linear_f(px[2]),
            ]
        })
        .collect()
}

/// Expand linear f32 grayscale to linear f32 RGB (gray→R=G=B).
pub(crate) fn gray_f32_to_linear_f32_rgb(data: &[f32], stride: usize) -> Vec<f32> {
    data.chunks(stride)
        .flat_map(|px| {
            let v = px[0];
            [v, v, v]
        })
        .collect()
}

// ─── f16 (linear) input helpers ───────────────────────────────────
// Closes the FLOAT16 portion of #18. Storage is native-endian u16
// per channel; conversion via `crate::f16::f16_bits_to_f32`.

/// Convert interleaved linear f16 RGB(A) bytes (`stride` channels per
/// pixel) to interleaved linear f32 RGB (stride 3, alpha dropped).
/// `bytes` must contain exactly `n_pixels * stride * 2` u16-bytes.
pub(crate) fn f16_to_linear_f32_rgb(bytes: &[u8], stride: usize) -> Vec<f32> {
    use crate::f16::f16_bits_to_f32;
    let pixels: &[u16] = &cast_pixel_lanes(bytes);
    pixels
        .chunks(stride)
        .flat_map(|px| {
            [
                f16_bits_to_f32(px[0]),
                f16_bits_to_f32(px[1]),
                f16_bits_to_f32(px[2]),
            ]
        })
        .collect()
}

/// Expand interleaved linear f16 grayscale (`stride=1` for gray-only,
/// `stride=2` for gray+alpha) to interleaved linear f32 RGB.
pub(crate) fn f16_gray_to_linear_f32_rgb(bytes: &[u8], stride: usize) -> Vec<f32> {
    use crate::f16::f16_bits_to_f32;
    let pixels: &[u16] = &cast_pixel_lanes(bytes);
    pixels
        .chunks(stride)
        .flat_map(|px| {
            let v = f16_bits_to_f32(px[0]);
            [v, v, v]
        })
        .collect()
}

/// Repack a row-strided pixel buffer into a tightly-packed `Vec<u8>`.
/// Closes row-stride portion of #18.
///
/// Caller must ensure `stride >= width * bytes_per_pixel`. The result
/// has `height * width * bytes_per_pixel` bytes; padding bytes from
/// each source row are discarded.
///
/// Returns `Err(EncodeError::InvalidInput)` if the source buffer is
/// too small to hold `height * stride` bytes (would index out of
/// bounds during the row copy).
pub(crate) fn unpack_strided_pixels(
    src: &[u8],
    width: usize,
    height: usize,
    bytes_per_pixel: usize,
    stride: usize,
) -> Result<Vec<u8>> {
    let row_bytes = width * bytes_per_pixel;
    if stride < row_bytes {
        return Err(at!(EncodeError::InvalidInput {
            message: format!(
                "row_stride {stride} is less than width*bytes_per_pixel = {width}*{bytes_per_pixel} = {row_bytes}",
            ),
        }));
    }
    let needed = height.checked_mul(stride).ok_or_else(|| {
        at!(EncodeError::InvalidInput {
            message: "height * row_stride overflows usize".into(),
        })
    })?;
    if src.len() < needed {
        return Err(at!(EncodeError::InvalidInput {
            message: format!(
                "pixel buffer too small for strided input: need {needed} bytes (height {height} × stride {stride}), got {}",
                src.len(),
            ),
        }));
    }
    let mut packed = Vec::with_capacity(height * row_bytes);
    for y in 0..height {
        let row_start = y * stride;
        packed.extend_from_slice(&src[row_start..row_start + row_bytes]);
    }
    Ok(packed)
}

/// Dispatch container-wrap by Brotli setting (closes #15 wire-up).
///
/// When `brotli_quality` is `Some(q)` AND the `brotli-metadata`
/// feature is enabled, routes through `wrap_in_container_with_brob`
/// (each metadata blob falls back to plain box if brob would be
/// bigger). Otherwise (or when feature is off), uses the plain
/// `wrap_in_container`. Centralizing the dispatch keeps the 3 call
/// sites (encode_inner, LossyEncoder::finish_inner,
/// LosslessEncoder::finish_inner) aligned.
///
/// `level` is the codestream level (5 or 10). When `level != 5` a
/// `jxll` (level) box is emitted directly after `ftyp`. For level 5
/// the byte layout is byte-identical to the historical wrap. See
/// [`crate::container::compute_codestream_level`].
pub(crate) fn wrap_metadata_container(
    codestream: &[u8],
    exif: Option<&[u8]>,
    xmp: Option<&[u8]>,
    jumbf: Option<&[u8]>,
    brotli_quality: Option<u32>,
    level: u8,
) -> Vec<u8> {
    #[cfg(feature = "brotli-metadata")]
    {
        if let Some(q) = brotli_quality {
            return crate::container::wrap_in_container_with_brob_and_level_and_jumbf(
                codestream, exif, xmp, jumbf, q, level,
            );
        }
    }
    let _ = brotli_quality;
    crate::container::wrap_in_container_with_level_and_jumbf(codestream, exif, xmp, level, jumbf)
}

/// Pick the codestream level required for an image with the given
/// dimensions, ICC size, and extra channels. Wraps
/// [`crate::container::compute_codestream_level`] and translates the
/// `None` (unencodable) case into [`EncodeError::InvalidInput`].
///
/// `num_extra_channels` MUST already include the alpha channel when
/// the pixel layout carries alpha — the level-5 cap is `<= 4` extras
/// *including* alpha, matching libjxl `VerifyLevelSettings` which
/// reads `m.num_extra_channels` (alpha is one of them).
pub(crate) fn compute_required_level(
    width: u32,
    height: u32,
    num_extra_channels: u32,
    has_black_channel: bool,
    icc_size: u64,
) -> Result<u8> {
    crate::container::compute_codestream_level(
        width,
        height,
        num_extra_channels,
        has_black_channel,
        icc_size,
    )
    .ok_or_else(|| {
        at!(EncodeError::InvalidInput {
            message: format!(
                "image {width}x{height} ({} px), {num_extra_channels} extra channels, \
                 {icc_size}-byte ICC exceeds JPEG XL level 10 limits",
                u64::from(width).saturating_mul(u64::from(height)),
            ),
        })
    })
}

/// Divide premultiplied (associated) linear RGB values by alpha so the
/// encoded codestream stores straight (unassociated) color. Mirrors
/// libjxl `UnpremultiplyAlpha` in `lib/jxl/alpha.cc:106`. Pairs with
/// `alpha_associated=true` in the codestream header — the decoder is
/// responsible for re-premultiplying the output.
///
/// `alpha_u8` is the per-pixel alpha after our standard u8 quantization
/// (matching the codestream's 8-bit BitDepth default). Using the same
/// quantized value the decoder will see ensures the round-trip
/// premultiplied → encode → decode → re-premultiplied closes.
///
/// `kSmallAlpha = 1.0 / (1<<26)` floor on the divisor — matches
/// libjxl `lib/jxl/alpha.h:21`. Lifts division-by-zero on alpha=0
/// pixels (where the original color is undefined anyway).
pub(crate) fn unpremultiply_alpha_inplace(linear_rgb_interleaved: &mut [f32], alpha_u8: &[u8]) {
    const K_SMALL_ALPHA: f32 = 1.0_f32 / ((1u32 << 26) as f32);
    debug_assert_eq!(linear_rgb_interleaved.len(), alpha_u8.len() * 3);
    for (rgb, &a) in linear_rgb_interleaved
        .chunks_exact_mut(3)
        .zip(alpha_u8.iter())
    {
        let a_f = (a as f32) / 255.0;
        let inv = 1.0 / a_f.max(K_SMALL_ALPHA);
        rgb[0] *= inv;
        rgb[1] *= inv;
        rgb[2] *= inv;
    }
}

/// Extract alpha from interleaved f16 pixel data, converting to u8
/// (0..255). Mirrors `extract_alpha_f32` but reads u16 bytes via f16
/// conversion before clamping.
pub(crate) fn extract_alpha_f16(bytes: &[u8], stride: usize, alpha_offset: usize) -> Vec<u8> {
    use crate::f16::f16_bits_to_f32;
    let pixels: &[u16] = &cast_pixel_lanes(bytes);
    pixels
        .chunks(stride)
        .map(|px| (f16_bits_to_f32(px[alpha_offset]).clamp(0.0, 1.0) * 255.0 + 0.5) as u8)
        .collect()
}
