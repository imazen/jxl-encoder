// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Squeeze (Haar wavelet) transform for modular encoding.
//!
//! Decomposes image channels into low-frequency (average) and high-frequency
//! (residual/detail) components by halving resolution in one direction per step.
//! Enables progressive decoding and improves compression on smooth content.

use super::channel::{Channel, ModularImage};
use crate::error::Result;

/// Parameters for a single squeeze step.
#[derive(Debug, Clone, Copy)]
pub struct SqueezeParams {
    /// True = halve width (horizontal), false = halve height (vertical).
    pub horizontal: bool,
    /// True = insert residuals in-place after data channels, false = append to end.
    pub in_place: bool,
    /// First channel index to squeeze.
    pub begin_c: u32,
    /// Number of consecutive channels to squeeze.
    pub num_c: u32,
}

/// Maximum size before squeeze is applied (matches libjxl).
const MAX_FIRST_PREVIEW_SIZE: usize = 8;

/// Compute smooth tendency prediction to reduce ringing artifacts.
///
/// B = previous value, a = current average, n = next average.
/// Returns predicted difference for smooth (monotonic) regions.
#[inline]
fn smooth_tendency(b: i32, a: i32, n: i32) -> i32 {
    let mut diff = 0i32;
    if b >= a && a >= n {
        diff = (4 * b - 3 * n - a + 6) / 12;
        if diff - (diff & 1) > 2 * (b - a) {
            diff = 2 * (b - a) + 1;
        }
        if diff + (diff & 1) > 2 * (a - n) {
            diff = 2 * (a - n);
        }
    } else if b <= a && a <= n {
        diff = (4 * b - 3 * n - a - 6) / 12;
        if diff + (diff & 1) < 2 * (b - a) {
            diff = 2 * (b - a) - 1;
        }
        if diff - (diff & 1) < 2 * (a - n) {
            diff = 2 * (a - n);
        }
    }
    diff
}

/// Rounded average matching libjxl's AVERAGE macro.
#[inline]
fn average(x: i32, y: i32) -> i32 {
    (x + y + (if x > y { 1 } else { 0 })) >> 1
}

/// Forward horizontal squeeze of a single channel.
///
/// Input channel (w, h) → average channel ((w+1)/2, h) + residual channel (w-(w+1)/2, h).
/// Both output channels inherit vshift and get hshift+1 (matching libjxl MetaSqueeze).
fn fwd_h_squeeze(channel: &Channel) -> Result<(Channel, Channel)> {
    let w = channel.width();
    let h = channel.height();
    let avg_w = w.div_ceil(2);
    let res_w = w - avg_w;

    let mut avg = Channel::new(avg_w, h)?;
    let mut res = Channel::new(res_w, h)?;

    // Both avg and res get incremented hshift, same vshift (matching libjxl MetaSqueeze)
    avg.hshift = channel.hshift + 1;
    avg.vshift = channel.vshift;
    res.hshift = channel.hshift + 1;
    res.vshift = channel.vshift;

    for y in 0..h {
        for x in 0..res_w {
            let a = channel.get(x * 2, y);
            let b = channel.get(x * 2 + 1, y);
            let av = average(a, b);
            avg.set(x, y, av);

            let diff = a - b;

            let next_avg = if x + 1 < res_w {
                let c = channel.get(x * 2 + 2, y);
                let d = channel.get(x * 2 + 3, y);
                average(c, d)
            } else if w & 1 != 0 {
                channel.get(x * 2 + 2, y)
            } else {
                av
            };

            let left = if x > 0 { channel.get(x * 2 - 1, y) } else { av };
            let tendency = smooth_tendency(left, av, next_avg);
            res.set(x, y, diff - tendency);
        }

        // Odd width: last pixel goes directly to average
        if w & 1 != 0 {
            let x = avg_w - 1;
            avg.set(x, y, channel.get(x * 2, y));
        }
    }

    Ok((avg, res))
}

/// Forward vertical squeeze of a single channel.
///
/// Input channel (w, h) → average channel (w, (h+1)/2) + residual channel (w, h-(h+1)/2).
/// Both output channels inherit hshift and get vshift+1 (matching libjxl MetaSqueeze).
fn fwd_v_squeeze(channel: &Channel) -> Result<(Channel, Channel)> {
    let w = channel.width();
    let h = channel.height();
    let avg_h = h.div_ceil(2);
    let res_h = h - avg_h;

    let mut avg = Channel::new(w, avg_h)?;
    let mut res = Channel::new(w, res_h)?;

    // Both avg and res get incremented vshift, same hshift (matching libjxl MetaSqueeze)
    avg.hshift = channel.hshift;
    avg.vshift = channel.vshift + 1;
    res.hshift = channel.hshift;
    res.vshift = channel.vshift + 1;

    for y in 0..res_h {
        for x in 0..w {
            let a = channel.get(x, y * 2);
            let b = channel.get(x, y * 2 + 1);
            let av = average(a, b);
            avg.set(x, y, av);

            let diff = a - b;

            let next_avg = if y + 1 < res_h {
                let c = channel.get(x, y * 2 + 2);
                let d = channel.get(x, y * 2 + 3);
                average(c, d)
            } else if h & 1 != 0 {
                channel.get(x, y * 2 + 2)
            } else {
                av
            };

            let top = if y > 0 { channel.get(x, y * 2 - 1) } else { av };
            let tendency = smooth_tendency(top, av, next_avg);
            res.set(x, y, diff - tendency);
        }
    }

    // Odd height: last row goes directly to average
    if h & 1 != 0 {
        let y = avg_h - 1;
        for x in 0..w {
            avg.set(x, y, channel.get(x, y * 2));
        }
    }

    Ok((avg, res))
}

/// Generate default squeeze parameters for an image.
///
/// Follows libjxl's default strategy:
/// 1. Optional 4:2:0 chroma squeeze for channels 1-2
/// 2. Alternate horizontal/vertical squeezes until both dimensions ≤ 8
pub fn default_squeeze_params(image: &ModularImage) -> Vec<SqueezeParams> {
    let nb_channels = image.channels.len();
    if nb_channels == 0 {
        return Vec::new();
    }

    let mut params = Vec::new();
    let mut w = image.channels[0].width();
    let mut h = image.channels[0].height();

    // Skip squeeze entirely if both dimensions are already small
    if w <= MAX_FIRST_PREVIEW_SIZE && h <= MAX_FIRST_PREVIEW_SIZE {
        return params;
    }

    // 4:2:0 chroma squeeze if channels 1 and 2 have same dimensions as channel 0
    // Skip squeeze directions that would produce 0-sized residual channels.
    if nb_channels > 2
        && image.channels[1].width() == w
        && image.channels[1].height() == h
        && image.channels[2].width() == w
        && image.channels[2].height() == h
    {
        if w > 1 {
            params.push(SqueezeParams {
                horizontal: true,
                in_place: false,
                begin_c: 1,
                num_c: 2,
            });
        }
        if h > 1 {
            params.push(SqueezeParams {
                horizontal: false,
                in_place: false,
                begin_c: 1,
                num_c: 2,
            });
        }
    }

    let wide = w > h;

    let sp = SqueezeParams {
        horizontal: false, // will be set per iteration
        in_place: true,
        begin_c: 0,
        num_c: nb_channels as u32,
    };

    // Tall image: vertical first
    if !wide && h > MAX_FIRST_PREVIEW_SIZE {
        let mut p = sp;
        p.horizontal = false;
        params.push(p);
        h = h.div_ceil(2);
    }

    // Alternate horizontal/vertical until both ≤ 8
    while w > MAX_FIRST_PREVIEW_SIZE || h > MAX_FIRST_PREVIEW_SIZE {
        if w > MAX_FIRST_PREVIEW_SIZE {
            let mut p = sp;
            p.horizontal = true;
            params.push(p);
            w = w.div_ceil(2);
        }
        if h > MAX_FIRST_PREVIEW_SIZE {
            let mut p = sp;
            p.horizontal = false;
            params.push(p);
            h = h.div_ceil(2);
        }
    }

    params
}

/// Apply forward squeeze transform to a modular image.
///
/// Modifies the image in-place, replacing channels with average+residual pairs.
/// Returns the squeeze parameters that were applied (for bitstream serialization).
pub fn apply_squeeze(image: &mut ModularImage, params: &[SqueezeParams]) -> Result<()> {
    for param in params {
        let begin_c = param.begin_c as usize;
        let end_c = begin_c + param.num_c as usize - 1;
        let offset = if param.in_place {
            end_c + 1
        } else {
            image.channels.len()
        };

        // Process channels in order, inserting residuals.
        // Matches C++ FwdSqueeze: c iterates beginc..=endc, insert at offset+c-beginc.
        // For in_place, offset = end_c+1 so residuals go right after data channels.
        // For not in_place, offset = original channels.len() so residuals append.
        for c in begin_c..=end_c {
            let rc = offset + c - begin_c;

            let (avg, res) = if param.horizontal {
                fwd_h_squeeze(&image.channels[c])?
            } else {
                fwd_v_squeeze(&image.channels[c])?
            };

            image.channels[c] = avg;
            image.channels.insert(rc, res);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_smooth_tendency_monotonic_decreasing() {
        // B=10, a=5, n=2 → monotonically decreasing
        let t = smooth_tendency(10, 5, 2);
        // Positive tendency (predicts positive diff in smooth region)
        assert!(t >= 0);
    }

    #[test]
    fn test_smooth_tendency_monotonic_increasing() {
        let t = smooth_tendency(2, 5, 10);
        assert!(t <= 0);
    }

    #[test]
    fn test_smooth_tendency_non_monotonic() {
        // Non-monotonic: no tendency
        assert_eq!(smooth_tendency(5, 2, 10), 0);
        assert_eq!(smooth_tendency(10, 5, 8), 0);
    }

    #[test]
    fn test_average() {
        assert_eq!(average(4, 6), 5);
        assert_eq!(average(5, 6), 5); // 5 > 6 is false, (5+6+0)/2 = 5
        assert_eq!(average(6, 5), 6); // 6 > 5 is true, (6+5+1)/2 = 6
        assert_eq!(average(0, 0), 0);
        assert_eq!(average(1, 0), 1); // (1+0+1)/2 = 1
    }

    #[test]
    fn test_h_squeeze_even_width() {
        // 4x2 channel → avg 2x2, res 2x2
        let ch = Channel::from_vec(vec![10, 20, 30, 40, 50, 60, 70, 80], 4, 2).unwrap();
        let (avg, res) = fwd_h_squeeze(&ch).unwrap();
        assert_eq!(avg.width(), 2);
        assert_eq!(avg.height(), 2);
        assert_eq!(res.width(), 2);
        assert_eq!(res.height(), 2);
    }

    #[test]
    fn test_h_squeeze_odd_width() {
        // 5x1 → avg 3x1, res 2x1
        let ch = Channel::from_vec(vec![10, 20, 30, 40, 50], 5, 1).unwrap();
        let (avg, res) = fwd_h_squeeze(&ch).unwrap();
        assert_eq!(avg.width(), 3);
        assert_eq!(res.width(), 2);
        // Last pixel (50) goes to avg[2]
        assert_eq!(avg.get(2, 0), 50);
    }

    #[test]
    fn test_v_squeeze_even_height() {
        // 2x4 → avg 2x2, res 2x2
        let ch = Channel::from_vec(vec![10, 20, 30, 40, 50, 60, 70, 80], 2, 4).unwrap();
        let (avg, res) = fwd_v_squeeze(&ch).unwrap();
        assert_eq!(avg.width(), 2);
        assert_eq!(avg.height(), 2);
        assert_eq!(res.width(), 2);
        assert_eq!(res.height(), 2);
    }

    #[test]
    fn test_default_params_small_image() {
        // 4x4 image: both dimensions ≤ 8, no squeeze needed
        let image = ModularImage::from_gray8(&[0u8; 16], 4, 4).unwrap();
        let params = default_squeeze_params(&image);
        assert!(params.is_empty());
    }

    #[test]
    fn test_default_params_16x16() {
        // 16x16: needs squeeze to get below 8
        let image = ModularImage::from_gray8(&[0u8; 256], 16, 16).unwrap();
        let params = default_squeeze_params(&image);
        assert!(!params.is_empty());
        // Should produce at least 2 squeeze steps (H + V or V + H)
        assert!(params.len() >= 2);
    }

    #[test]
    fn test_apply_squeeze_gray_16x16() {
        let mut data = vec![0u8; 16 * 16];
        for y in 0..16 {
            for x in 0..16 {
                data[y * 16 + x] = (x * 16 + y * 4) as u8;
            }
        }
        let mut image = ModularImage::from_gray8(&data, 16, 16).unwrap();
        let params = default_squeeze_params(&image);
        assert!(!params.is_empty());

        let orig_channels = image.channels.len();
        apply_squeeze(&mut image, &params).unwrap();

        // Should have more channels now (averages + residuals)
        assert!(image.channels.len() > orig_channels);
    }
}
