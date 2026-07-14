//! Smart-dispatch content detection: the W44-35 DCT64 smooth-photo
//! gate and W44-164 auto content-class discriminator, plus the
//! box-filter channel downsample companion to
//! [`ExtraChannel::with_dim_shift`].
//!
//! All free functions (no types); re-exported at the `api` level.

use super::*;

/// Box-filter downsample a u8 channel by `factor` (mirrors libjxl
/// `DoDownsampleImage` in `lib/jxl/image_ops.cc`). Companion to
/// [`ExtraChannel::with_dim_shift`] for the
/// `--ec_resampling`-style workflow where the caller owns
/// full-resolution channel data (e.g., alpha plane sliced out of
/// RGBA8 input) but wants to store it at lower resolution.
///
/// Output dimensions are `(width.div_ceil(factor),
/// height.div_ceil(factor))`. Edge cells average only the
/// in-bounds samples (partial 2×2 / 4×4 / 8×8 patches), matching
/// the libjxl behaviour.
///
/// `factor` should be one of `{1, 2, 4, 8}` to match libjxl's
/// `--ec_resampling N` accepted values; passing `factor == 0`
/// returns an empty `Vec` and `factor == 1` returns a clone of the
/// input.
///
/// # Example
///
/// ```ignore
/// use jxl_encoder::api::{downsample_channel_u8, ExtraChannel};
/// let alpha_half = downsample_channel_u8(&alpha_fullres, w, h, 2);
/// let extras = [ExtraChannel::from_alpha_buf(&alpha_half, false)
///     .with_dim_shift(1)];
/// ```
#[must_use]
pub fn downsample_channel_u8(src: &[u8], width: usize, height: usize, factor: u32) -> Vec<u8> {
    if factor == 0 {
        return Vec::new();
    }
    if factor == 1 {
        return src.to_vec();
    }
    debug_assert_eq!(
        src.len(),
        width * height,
        "downsample_channel_u8: src len {} != width*height {}",
        src.len(),
        width * height,
    );
    let factor = factor as usize;
    let out_w = width.div_ceil(factor);
    let out_h = height.div_ceil(factor);
    let mut out = Vec::with_capacity(out_w * out_h);
    for oy in 0..out_h {
        let y0 = oy * factor;
        for ox in 0..out_w {
            let x0 = ox * factor;
            let mut sum: u32 = 0;
            let mut cnt: u32 = 0;
            for iy in 0..factor {
                let y = y0 + iy;
                if y >= height {
                    break;
                }
                let row_off = y * width;
                for ix in 0..factor {
                    let x = x0 + ix;
                    if x >= width {
                        break;
                    }
                    sum += src[row_off + x] as u32;
                    cnt += 1;
                }
            }
            // cnt is guaranteed > 0 because oy < out_h, ox < out_w
            // imply at least the (y0, x0) sample is in-bounds.
            let avg = (sum + cnt / 2) / cnt; // round-to-nearest
            out.push(avg.min(255) as u8);
        }
    }
    out
}

// ── W44-35: smooth-photo DCT64 admission detector ──────────────────────────

/// Mean-abs adjacent-luma-diff threshold (4×-downsampled luma plane,
/// integer scale 0..255) below which the input is "smooth enough" for
/// DCT64-class transforms to win on the gated-cell (W44-35).
/// Calibration: smooth photos 1418519 / 7256805 / 1025469 all measure
/// `<= 0.10`; textured photos and most screenshots measure `>= 0.20`.
const W44_35_PROXY_EDGE_DENSITY_MAX: f32 = 0.15;

/// Fraction-of-8×8-blocks-with-luma-variance-below-5 threshold above
/// which the input looks like solid-color screen content (W44-35).
/// Calibration: smooth photos measure `~0.50`; solid-color UI screenshots
/// (codec_wiki, terminal) measure `>= 0.67`.
const W44_35_PROXY_FLAT_SOLID_MAX: f32 = 0.60;

/// `mean(|2c - l - r| over rows + |2c - t - b| over cols) / mean(|r-l| + |b-t|)`
/// threshold above which the input has too much high-frequency energy to
/// benefit from DCT64-class merges (W44-35). Calibration: smooth photos
/// 1418519 / 7256805 measure `<= 0.95`; textured photo 1025469 measures
/// `1.38`; screenshots measure `>= 1.30`.
const W44_35_PROXY_HF_RATIO_MAX: f32 = 1.0;

/// Auto detector for the W44-35 smooth-photo DCT64 admission gate.
///
/// Returns `true` when the input classifies as a smooth photo that
/// benefits from DCT64-class transforms even on the small + low-distance
/// gated cell. Cheap cost (~0.2 ms on 512×512 RGB on a modern desktop):
/// reads the input once, computes three integer aggregates on a
/// 4×-downsampled luma plane.
///
/// Discriminator (all 3 must hold):
/// - Edge density proxy < [`W44_35_PROXY_EDGE_DENSITY_MAX`] = 0.15
/// - Solid-color block ratio < [`W44_35_PROXY_FLAT_SOLID_MAX`] = 0.60
/// - HF energy ratio < [`W44_35_PROXY_HF_RATIO_MAX`] = 1.0
///
/// **Provenance**: `benchmarks/w44_35_dct64_smart_dispatch_calibrate.tsv`
/// and `examples/dct64_smart_dispatch_proxy_calibrate.rs`. Validated on
/// 10 stratified images across 8 cells each (e ∈ {6, 7}, d ∈ {1.0, 1.2,
/// 1.6, 2.0}): admits 1418519 (-5 to -7 % bytes), admits 7256805
/// (-2 to -4 %), correctly skips 1025469 (mixed +/- < 1.3 %), correctly
/// skips all 4 screenshots / pixel-art.
///
/// **Callers**: each lossy entry point in [`crate::api`] computes this
/// once on the input RGB before resolving the per-image profile via
/// [`LossyConfig::effective_profile_for_image_with_smoothness`]. The
/// `StrategyOverrides::smooth_photo_dct64_hint = Some(_)` override
/// (set via [`LossyConfig::with_strategy_overrides`]) always wins.
///
/// **Layout dispatch**: this function takes raw `pixels` plus a
/// [`PixelLayout`] descriptor and only fires on the u8 sRGB-encoded
/// layouts (`Rgb8`, `Rgba8`, `Bgr8`, `Bgra8`). For other layouts
/// (16-bit, float, gray) it returns `false` — the gate falls through
/// to the default `adapt_to_image_lossy` behaviour. Callers wanting
/// the admission on non-u8 layouts should set
/// [`LossyConfig::with_strategy_overrides`] with
/// `smooth_photo_dct64_hint: Some(true)`.
#[must_use]
pub(crate) fn detect_smooth_photo_for_dct64_from_layout(
    pixels: &[u8],
    w: u32,
    h: u32,
    layout: PixelLayout,
) -> bool {
    // Only sRGB u8 layouts are supported by the cheap proxy. Other
    // layouts return `false` (conservative — the gate stays as today).
    let (bpp, r_offset) = match layout {
        PixelLayout::Rgb8 => (3usize, 0usize),
        PixelLayout::Rgba8 => (4usize, 0usize),
        PixelLayout::Bgr8 => (3usize, 2usize), // R at +2, B at +0
        PixelLayout::Bgra8 => (4usize, 2usize),
        _ => return false,
    };
    detect_smooth_photo_for_dct64_inner(pixels, w, h, bpp, r_offset)
}

/// Internal kernel for [`detect_smooth_photo_for_dct64_from_layout`].
/// Operates on interleaved u8 sRGB at `bpp` bytes per pixel with the
/// R channel at byte offset `r_offset` (0 for RGB-order, 2 for BGR-order).
/// Luma is approximated as `(R + 2G + B) >> 2`.
fn detect_smooth_photo_for_dct64_inner(
    rgb: &[u8],
    w: u32,
    h: u32,
    bpp: usize,
    r_offset: usize,
) -> bool {
    // Cheap proxy: only meaningful when the gate condition would fire
    // (pixels < 500_000). Skip the work otherwise to keep the entry-point
    // dispatch overhead-free on production-sized inputs.
    let pixels = (w as u64) * (h as u64);
    if pixels >= crate::effort::LOSSY_SMALL_IMAGE_PIXEL_THRESHOLD {
        return false;
    }
    let w = w as usize;
    let h = h as usize;
    if w * h * bpp != rgb.len() {
        return false;
    }
    let stride = w * bpp;
    // Channel-byte offsets for R/G/B (G is always between R and B
    // in RGB-order; for BGR-order R is at +2, G is at +1, B is at +0).
    let g_off = 1usize;
    let b_off = if r_offset == 0 { 2 } else { 0 };
    let ds = 4usize;
    let dw = w / ds;
    let dh = h / ds;
    if dw < 3 || dh < 3 {
        return false;
    }
    // Build integer luma plane on the 4×-downsampled grid.
    let mut luma = vec![0i32; dw * dh];
    for dy in 0..dh {
        let y = dy * ds;
        let row_off = y * stride;
        for dx in 0..dw {
            let x = dx * ds;
            let i = row_off + x * bpp;
            let r = rgb[i + r_offset] as i32;
            let g = rgb[i + g_off] as i32;
            let b = rgb[i + b_off] as i32;
            // Approximate luma: (R + 2G + B) / 4, range 0..255.
            luma[dy * dw + dx] = (r + 2 * g + b) >> 2;
        }
    }
    // Proxy 1: mean abs adjacent luma diff (edge density).
    let mut sum_d1: u64 = 0;
    let mut count_d1: u64 = 0;
    for y in 0..dh {
        for x in 1..dw {
            let a = luma[y * dw + x - 1];
            let b = luma[y * dw + x];
            sum_d1 += (a - b).unsigned_abs() as u64;
            count_d1 += 1;
        }
    }
    for y in 1..dh {
        for x in 0..dw {
            let a = luma[(y - 1) * dw + x];
            let b = luma[y * dw + x];
            sum_d1 += (a - b).unsigned_abs() as u64;
            count_d1 += 1;
        }
    }
    let mean_d1 = (sum_d1 as f32) / (count_d1.max(1) as f32);
    let proxy_edge = mean_d1 / 64.0;
    if proxy_edge >= W44_35_PROXY_EDGE_DENSITY_MAX {
        return false;
    }
    // Proxy 2: HF energy ratio (mean abs 2nd-derivative / mean abs 1st-derivative).
    let mut sum_d2: u64 = 0;
    let mut count_d2: u64 = 0;
    let mut sum_d1_center: u64 = 0;
    for y in 0..dh {
        for x in 1..dw - 1 {
            let l = luma[y * dw + x - 1];
            let c = luma[y * dw + x];
            let r = luma[y * dw + x + 1];
            sum_d2 += (2 * c - l - r).unsigned_abs() as u64;
            sum_d1_center += (r - l).unsigned_abs() as u64;
            count_d2 += 1;
        }
    }
    for y in 1..dh - 1 {
        for x in 0..dw {
            let t = luma[(y - 1) * dw + x];
            let c = luma[y * dw + x];
            let b = luma[(y + 1) * dw + x];
            sum_d2 += (2 * c - t - b).unsigned_abs() as u64;
            sum_d1_center += (b - t).unsigned_abs() as u64;
            count_d2 += 1;
        }
    }
    let mean_d2 = (sum_d2 as f32) / (count_d2.max(1) as f32);
    let mean_d1c = (sum_d1_center as f32) / (count_d2.max(1) as f32) + 0.001;
    let proxy_hf = mean_d2 / mean_d1c;
    if proxy_hf >= W44_35_PROXY_HF_RATIO_MAX {
        return false;
    }
    // Proxy 3: fraction of 8×8 luma blocks with intra-block variance < 5
    // (= near-solid-color regions, typical of screen content). Run on the
    // full-resolution input.
    let block = 8usize;
    let bw = w / block;
    let bh = h / block;
    if bw == 0 || bh == 0 {
        // Image smaller than a single 8×8 block — can't be DCT64 candidate.
        return false;
    }
    let mut flat: u64 = 0;
    let mut total: u64 = 0;
    for by in 0..bh {
        for bx in 0..bw {
            let mut sum: u32 = 0;
            let mut sum_sq: u32 = 0;
            for py in 0..block {
                let y = by * block + py;
                let row_off = y * stride;
                for px in 0..block {
                    let x = bx * block + px;
                    let i = row_off + x * bpp;
                    let r = rgb[i + r_offset] as u32;
                    let g = rgb[i + g_off] as u32;
                    let b = rgb[i + b_off] as u32;
                    let luma_v = (r + 2 * g + b) >> 2;
                    sum += luma_v;
                    sum_sq += luma_v * luma_v;
                }
            }
            let n = (block * block) as u32;
            let mean = sum as f32 / n as f32;
            let var = (sum_sq as f32 / n as f32) - mean * mean;
            if var < 5.0 {
                flat += 1;
            }
            total += 1;
        }
    }
    let ratio = (flat as f32) / (total.max(1) as f32);
    if ratio >= W44_35_PROXY_FLAT_SOLID_MAX {
        return false;
    }
    true
}

// ─────────────────────────────────────────────────────────────────────────
// W44-164: Smart-Zenjxl auto-classifier for `ImageContentClass`
// ─────────────────────────────────────────────────────────────────────────
//
// The encoder ships an opt-in `LossyConfig::with_content_class(Some(...))`
// setter that feeds [`crate::effort::EffortProfile::adapt_to_image_content`]
// (today: enables `patches` at e ∈ {5, 6} on `Screenshot` content). The
// auto-classifier below derives the class from the same cheap zenanalyze
// proxies the W44-91 / W44-96 / W44-124 gates already consume, so the
// dispatch fires automatically on `EncoderStrategy::Zenjxl` /
// `Aggressive` without the caller having to plumb a class through.
//
// Discriminator (sRGB u8 corpus measurements; see
// `benchmarks/entropy_mul_smart_dispatch_2026-05-18.meta` for the 10
// GB82-SC screenshots × per-image proxy table and
// `benchmarks/w44_91_zenanalyze_dispatch_2026-05-19.meta` for the 41
// CID22 validation photos + 6 W44-78 regression-band photos):
//
//   class      fcbr range (gb82-sc)    fcbr range (CID22 photos)
//   --------   ----------------------  -------------------------------
//   Screenshot 0.360 (windows95) –     never seen ≥ 0.10
//              0.907 (gmessages)
//   Photo      never seen ≥ 0.30       0.0034 (1189261) – 0.098 (297394)
//
// `fcbr >= 0.30` cleanly captures every screenshot in the corpus
// (smallest 0.36) while staying well above every photo (largest 0.098,
// 3× margin). We use the slightly higher 0.35 threshold to leave
// windows95-class outliers (very-low palette pixel-art) on the safe
// side — the W23-2 honest-stop showed the W22-1 *entropy_mul lift* hurts
// windows95, but `adapt_to_image_content` only flips `patches` (which
// helps repetitive UI content). Photos with anomalous fcbr in the
// 0.10–0.30 deadband stay `Unknown` (no dispatch).

/// W44-164 fcbr threshold for the `Screenshot` class. Calibrated from
/// 10 GB82-SC screenshots (range 0.360–0.907) vs 41 CID22 photos +
/// 6 W44-78 regression-band photos (range 0.0034–0.098). The 0.35
/// threshold sits inside the 0.098 → 0.360 gap, ~3.5× above the
/// photo ceiling and just below the screenshot floor — see the table
/// above for the source data.
const W44_164_FCBR_SCREENSHOT_MIN: f32 = 0.35;

/// W44-164 minimum pixel count for the auto-classifier to compute the
/// proxies. Matches [`crate::effort::CONTENT_CLASS_MIN_PIXELS`]
/// (= 65,536); below this both (a) the proxies are unreliable on
/// thumbnail content AND (b) `adapt_to_image_content` short-circuits
/// anyway. Skipping the proxy compute saves the O(W·H) scan on the
/// throwaway-small inputs.
const W44_164_MIN_PIXELS: u64 = crate::effort::CONTENT_CLASS_MIN_PIXELS;

/// W44-164: derive an [`crate::effort::ImageContentClass`] from the
/// caller-supplied 8-bit sRGB pixel buffer when the auto-classifier
/// dispatch is active. Returns `None` for layouts where the proxy
/// would be unreliable (16-bit, linear-f32, grayscale, HDR), for
/// images below the pixel threshold, and for content sitting in the
/// fcbr deadband between the photo ceiling and screenshot floor
/// (`fcbr` ∈ [0.10, [`W44_164_FCBR_SCREENSHOT_MIN`])) — in those
/// cases the dispatch falls through to the existing default profile
/// (no class adapter fires).
///
/// Discriminator (see module-level prose above for the source data):
///
/// - `fcbr >= W44_164_FCBR_SCREENSHOT_MIN (= 0.35)` → `Screenshot`
/// - `fcbr < 0.10 AND m3_colourfulness >= 5.0` → `Photo`
/// - else → `None`
///
/// The `m3 >= 5.0` photo floor excludes pure-grayscale gradients
/// (which have m3 ≈ 0); they sit in the `Unknown` bucket where the
/// adapter is a no-op today anyway.
///
/// **Layout dispatch**: only the 8-bit sRGB layouts (`Rgb8`, `Rgba8`,
/// `Bgr8`, `Bgra8`) compute the proxies — for everything else this
/// returns `None` (same surface as `compute_w44_91_zenanalyze_proxies`).
#[must_use]
pub(crate) fn auto_classify_content_class_from_layout(
    pixels: &[u8],
    w: u32,
    h: u32,
    layout: PixelLayout,
) -> Option<crate::effort::ImageContentClass> {
    let pixel_count = (w as u64).checked_mul(h as u64)?;
    if pixel_count < W44_164_MIN_PIXELS {
        return None;
    }
    let proxies = compute_w44_91_zenanalyze_proxies(pixels, w as usize, h as usize, layout)?;
    Some(classify_from_proxies(&proxies))
}

/// Pure function: classify content given precomputed proxies. Split
/// out for unit testing so the discriminator predicate can be
/// exercised without instantiating an input buffer.
#[must_use]
pub(crate) fn classify_from_proxies(
    p: &crate::vardct::encoder::ZenanalyzeProxies,
) -> crate::effort::ImageContentClass {
    use crate::effort::ImageContentClass;
    if p.flat_color_block_ratio >= W44_164_FCBR_SCREENSHOT_MIN {
        ImageContentClass::Screenshot
    } else if p.flat_color_block_ratio < 0.10 && p.m3_colourfulness >= 5.0 {
        ImageContentClass::Photo
    } else {
        // Deadband fcbr ∈ [0.10, 0.35) or near-grayscale photo:
        // leave `Unknown` so the adapter is a no-op (default behaviour
        // — preserves byte parity on edge cases).
        ImageContentClass::Unknown
    }
}
