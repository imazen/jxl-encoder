// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Lossy modular quantization for Squeeze-transformed channels.
//!
//! Port of libjxl `enc_modular.cc:80-99,141-153,973-1027` — the quantization
//! tables, per-channel quantizer computation, channel pre-quantization, and
//! multiplier info construction used by the `responsive=1` encoding path.
//!
//! The key idea: after Squeeze decomposes channels into avg+residual at multiple
//! scales, each channel gets a quantizer `q` derived from distance and its
//! position in the Squeeze pyramid. Pixels are rounded to the nearest multiple
//! of `q`, and the tree leaf's `multiplier` field is set to `q` so the decoder
//! can reconstruct: `pixel = unpack_signed(token) * multiplier + prediction`.

use super::channel::Channel;

/// XYB squeeze quantization tables from libjxl (enc_modular.cc:92-98).
///
/// Index: `[component][shift]` where `shift = hshift + vshift - 1` (clamped to 0..15).
/// Component 0 = Y, 1 = X, 2 = B-Y.
const SQUEEZE_XYB_QTABLE: [[f32; 16]; 3] = [
    // Y
    [
        163.84, 81.92, 40.96, 20.48, 10.24, 5.12, 2.56, 1.28, 0.64, 0.32, 0.16, 0.08, 0.04, 0.02,
        0.01, 0.005,
    ],
    // X
    [
        1024.0, 512.0, 256.0, 128.0, 64.0, 32.0, 16.0, 8.0, 4.0, 2.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5,
    ],
    // B-Y
    [
        2048.0, 1024.0, 512.0, 256.0, 128.0, 64.0, 32.0, 16.0, 8.0, 4.0, 2.0, 1.0, 0.5, 0.5, 0.5,
        0.5,
    ],
];

/// Quality factor for XYB squeeze quantization (enc_modular.cc:89).
const SQUEEZE_QUALITY_FACTOR_XYB: f32 = 4.0;

/// Extra quality factor for Y component (enc_modular.cc:90).
const SQUEEZE_QUALITY_FACTOR_Y: f32 = 1.5;

/// Compute the quantizer for an XYB channel after Squeeze transform.
///
/// Matches libjxl enc_modular.cc:1011-1014:
/// ```text
/// q = quantizers[component] * squeeze_quality_factor_xyb *
///     squeeze_xyb_qtable[component][shift];
/// if (component == 0) q *= squeeze_quality_factor_y;
/// ```
///
/// Where `quantizers[c] = 0.25 * distance^1.2` for all XYB components.
///
/// # Arguments
/// * `component` - 0=Y, 1=X, 2=B-Y
/// * `hshift` - Horizontal squeeze shift of the channel
/// * `vshift` - Vertical squeeze shift of the channel
/// * `distance` - Butteraugli distance
pub fn compute_channel_quantizer_xyb(
    component: usize,
    hshift: u32,
    vshift: u32,
    distance: f32,
) -> i32 {
    debug_assert!(component < 3);

    // Base quantizer: 0.25 * distance^1.2 (enc_modular.cc:988-989)
    let base_quantizer = 0.25 * jxl_simd::fast_powf(distance, 1.2);

    // Shift index: hshift + vshift, with the -1 adjustment from libjxl
    // (enc_modular.cc:1006: `shift = ch.hshift + ch.vshift; if shift > 0: shift--`)
    let shift = (hshift + vshift) as usize;
    let shift = if shift > 0 { shift - 1 } else { 0 };
    let shift = shift.min(15);

    let mut q = base_quantizer * SQUEEZE_QUALITY_FACTOR_XYB * SQUEEZE_XYB_QTABLE[component][shift];
    if component == 0 {
        q *= SQUEEZE_QUALITY_FACTOR_Y;
    }

    // Clamp to at least 1 (enc_modular.cc:1024)
    (q as i32).max(1)
}

/// Pre-quantize a channel: round every pixel to the nearest multiple of `q`.
///
/// Matches libjxl QuantizeChannel (enc_modular.cc:141-153):
/// ```text
/// if (row[x] < 0) row[x] = -((-row[x] + q/2) / q) * q;
/// else             row[x] =  (( row[x] + q/2) / q) * q;
/// ```
pub fn quantize_channel(channel: &mut Channel, q: i32) {
    if q == 1 {
        return;
    }
    let half = q / 2;
    for y in 0..channel.height() {
        let row = channel.row_mut(y);
        for val in row.iter_mut() {
            if *val < 0 {
                *val = -((-*val + half) / q) * q;
            } else {
                *val = ((*val + half) / q) * q;
            }
        }
    }
}

/// Result of checking how a multiplier info box intersects a node's range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntersectionType {
    /// No overlap between ranges.
    None,
    /// Partial overlap — need to force a split on the boundary.
    Partial,
    /// Needle is fully inside haystack — can set multiplier directly.
    Inside,
}

/// A range on the two static properties (channel, group_id) plus a multiplier.
///
/// Matches libjxl `ModularMultiplierInfo` (modular/options.h:54-57).
/// `range[0]` = channel range `[begin, end)`, `range[1]` = stream/group range `[begin, end)`.
#[derive(Debug, Clone)]
pub struct ModularMultiplierInfo {
    pub range: [[u32; 2]; 2],
    pub multiplier: u32,
}

/// Check how a node's current static property range (needle) intersects
/// a multiplier info box (haystack).
///
/// Matches libjxl `BoxIntersects` (enc_ma.cc:93-117).
///
/// Returns `(intersection_type, partial_axis, partial_val)` where:
/// - `partial_axis` is 0 (channel) or 1 (group_id) where the partial overlap occurs
/// - `partial_val` is the split value to use for the forced split
pub fn box_intersects(
    needle: &[[u32; 2]; 2],
    haystack: &[[u32; 2]; 2],
) -> (IntersectionType, usize, u32) {
    let mut partial = false;
    let mut partial_axis = 0usize;
    let mut partial_val = 0u32;

    for i in 0..2 {
        // No overlap
        if haystack[i][0] >= needle[i][1] {
            return (IntersectionType::None, 0, 0);
        }
        if haystack[i][1] <= needle[i][0] {
            return (IntersectionType::None, 0, 0);
        }
        // Needle fully contains haystack on this axis
        if haystack[i][0] <= needle[i][0] && haystack[i][1] >= needle[i][1] {
            continue;
        }
        // Partial overlap
        partial = true;
        partial_axis = i;
        if haystack[i][0] > needle[i][0] && haystack[i][0] < needle[i][1] {
            partial_val = haystack[i][0] - 1;
        } else {
            debug_assert!(haystack[i][1] > needle[i][0] && haystack[i][1] < needle[i][1]);
            partial_val = haystack[i][1] - 1;
        }
    }

    if partial {
        (IntersectionType::Partial, partial_axis, partial_val)
    } else {
        (IntersectionType::Inside, 0, 0)
    }
}

/// Build multiplier info from per-channel quantizers.
///
/// For a single-group LfFrame, all channels share stream_id=0, so the range
/// on axis 1 (group_id) is always `[0, 1)`.
///
/// Channels with the same quantizer that are adjacent get merged into a single
/// range, matching libjxl enc_modular.cc:1107-1161.
///
/// # Arguments
/// * `quants` - Per-channel quantizer values (length = num data channels)
/// * `nb_meta_channels` - Number of meta channels to skip (typically 0 for LfFrame)
pub fn build_multiplier_info(
    quants: &[i32],
    nb_meta_channels: usize,
) -> Vec<ModularMultiplierInfo> {
    let mut info = Vec::new();

    for (i, &q) in quants.iter().enumerate() {
        let ch_idx = (i + nb_meta_channels) as u32;
        let stream_id = 0u32; // Single-group: always stream 0

        if let Some(last) = info.last_mut() {
            let last: &mut ModularMultiplierInfo = last;
            if last.range[1][0] == stream_id && last.multiplier == q as u32 {
                // Same stream and quantizer — extend channel range
                last.range[0][1] = ch_idx + 1;
                continue;
            }
        }

        info.push(ModularMultiplierInfo {
            range: [[ch_idx, ch_idx + 1], [stream_id, stream_id + 1]],
            multiplier: q as u32,
        });
    }

    info
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_channel_basic() {
        let mut ch = Channel::from_vec(vec![0, 5, 10, 15, -5, -10, -15, 7, 3], 3, 3).unwrap();
        quantize_channel(&mut ch, 10);
        // 0 → 0, 5 → 10, 10 → 10, 15 → 20, -5 → -10, -10 → -10, -15 → -20, 7 → 10, 3 → 0
        assert_eq!(ch.get(0, 0), 0);
        assert_eq!(ch.get(1, 0), 10);
        assert_eq!(ch.get(2, 0), 10);
        assert_eq!(ch.get(0, 1), 20);
        assert_eq!(ch.get(1, 1), -10);
        assert_eq!(ch.get(2, 1), -10);
        assert_eq!(ch.get(0, 2), -20);
        assert_eq!(ch.get(1, 2), 10);
        assert_eq!(ch.get(2, 2), 0);
    }

    #[test]
    fn test_quantize_channel_q1_noop() {
        let mut ch = Channel::from_vec(vec![1, 2, 3, 4], 2, 2).unwrap();
        quantize_channel(&mut ch, 1);
        assert_eq!(ch.get(0, 0), 1);
        assert_eq!(ch.get(1, 1), 4);
    }

    #[test]
    fn test_quantize_channel_divisibility() {
        // After quantization, every pixel must be divisible by q
        let mut ch = Channel::from_vec(vec![13, -27, 100, -99, 0, 51, -1, 200, 3], 3, 3).unwrap();
        let q = 7;
        quantize_channel(&mut ch, q);
        for y in 0..3 {
            for x in 0..3 {
                assert_eq!(
                    ch.get(x, y) % q,
                    0,
                    "pixel ({},{}) = {} not divisible by {}",
                    x,
                    y,
                    ch.get(x, y),
                    q
                );
            }
        }
    }

    #[test]
    fn test_compute_quantizer_xyb() {
        // At distance 1.0, shift 0 (hshift=0, vshift=0 → shift=0 → clamped to 0):
        // q = 0.25 * 1.0^1.2 * 4.0 * qtable[c][0]
        // Y: 0.25 * 4.0 * 163.84 * 1.5 = 245.76 → 245
        let q_y = compute_channel_quantizer_xyb(0, 0, 0, 1.0);
        assert_eq!(q_y, 245);
        // X: 0.25 * 4.0 * 1024.0 = 1024 (allow ±1 from fast_powf approximation)
        let q_x = compute_channel_quantizer_xyb(1, 0, 0, 1.0);
        assert!(
            (q_x - 1024).unsigned_abs() <= 1,
            "q_x={q_x}, expected ~1024"
        );
    }

    #[test]
    fn test_compute_quantizer_with_shift() {
        // Component Y, hshift=1, vshift=1 → total shift=2, adjusted=1
        // q = 0.25 * 1.0^1.2 * 4.0 * 81.92 * 1.5 = 122.88 → 122
        let q = compute_channel_quantizer_xyb(0, 1, 1, 1.0);
        assert_eq!(q, 122);
    }

    #[test]
    fn test_box_intersects_none() {
        let needle = [[0, 3], [0, 1]];
        let haystack = [[5, 8], [0, 1]];
        let (t, _, _) = box_intersects(&needle, &haystack);
        assert_eq!(t, IntersectionType::None);
    }

    #[test]
    fn test_box_intersects_inside() {
        let needle = [[0, 3], [0, 1]];
        let haystack = [[0, 3], [0, 1]];
        let (t, _, _) = box_intersects(&needle, &haystack);
        assert_eq!(t, IntersectionType::Inside);
    }

    #[test]
    fn test_box_intersects_partial() {
        let needle = [[0, 6], [0, 1]];
        let haystack = [[0, 3], [0, 1]];
        let (t, axis, val) = box_intersects(&needle, &haystack);
        assert_eq!(t, IntersectionType::Partial);
        assert_eq!(axis, 0); // channel axis
        assert_eq!(val, 2); // split at channel 2 (haystack end - 1)
    }

    #[test]
    fn test_build_multiplier_info_basic() {
        // 3 channels with quantizers [10, 20, 20]
        let quants = vec![10, 20, 20];
        let info = build_multiplier_info(&quants, 0);
        assert_eq!(info.len(), 2);
        assert_eq!(info[0].range[0], [0, 1]);
        assert_eq!(info[0].multiplier, 10);
        assert_eq!(info[1].range[0], [1, 3]);
        assert_eq!(info[1].multiplier, 20);
    }

    #[test]
    fn test_build_multiplier_info_all_same() {
        let quants = vec![5, 5, 5];
        let info = build_multiplier_info(&quants, 0);
        assert_eq!(info.len(), 1);
        assert_eq!(info[0].range[0], [0, 3]);
        assert_eq!(info[0].multiplier, 5);
    }

    #[test]
    fn test_build_multiplier_info_all_different() {
        let quants = vec![1, 2, 3];
        let info = build_multiplier_info(&quants, 0);
        assert_eq!(info.len(), 3);
    }
}
