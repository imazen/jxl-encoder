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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

/// Largest sample magnitude for which the forward squeeze's i32 arithmetic
/// cannot overflow: `4*b - 3*n - a + 6` stays within 2^30 and the residual
/// within 2^29, and later squeeze steps only reuse averages, which stay
/// inside the input range.
const SQUEEZE_MAX_ABS_SAMPLE: i32 = 1 << 27;

/// Whether every sample is small enough for squeeze to be lossless. Wider
/// data (f32 bit patterns, integers above 28 bits) overflowed and decoded to
/// different samples; callers skip squeeze for it.
pub(crate) fn squeeze_fits(image: &ModularImage) -> bool {
    image.channels.iter().all(|ch| {
        ch.data()
            .iter()
            .all(|&v| (-SQUEEZE_MAX_ABS_SAMPLE..=SQUEEZE_MAX_ABS_SAMPLE).contains(&v))
    })
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
fn fwd_h_squeeze(channel: &Channel, stop: Option<&dyn enough::Stop>) -> Result<(Channel, Channel)> {
    let w = channel.width();
    let h = channel.height();
    let avg_w = w.div_ceil(2);
    let res_w = w - avg_w;

    let mut avg = Channel::new(avg_w, h)?;
    let mut res = Channel::new(res_w, h)?;

    // Both avg and res get incremented hshift, same vshift (matching libjxl MetaSqueeze)
    avg.hshift = channel.hshift + 1;
    avg.vshift = channel.vshift;
    avg.component = channel.component;
    res.hshift = channel.hshift + 1;
    res.vshift = channel.vshift;
    res.component = channel.component;

    for y in 0..h {
        if y & 0x3F == 0 {
            crate::error::check_stop(stop)?;
        }
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
fn fwd_v_squeeze(channel: &Channel, stop: Option<&dyn enough::Stop>) -> Result<(Channel, Channel)> {
    let w = channel.width();
    let h = channel.height();
    let avg_h = h.div_ceil(2);
    let res_h = h - avg_h;

    let mut avg = Channel::new(w, avg_h)?;
    let mut res = Channel::new(w, res_h)?;

    // Both avg and res get incremented vshift, same hshift (matching libjxl MetaSqueeze)
    avg.hshift = channel.hshift;
    avg.vshift = channel.vshift + 1;
    avg.component = channel.component;
    res.hshift = channel.hshift;
    res.vshift = channel.vshift + 1;
    res.component = channel.component;

    for y in 0..res_h {
        if y & 0x3F == 0 {
            crate::error::check_stop(stop)?;
        }
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
    if image.channels.len() <= MAX_EXPLICIT_SQUEEZE_CHANNELS as usize {
        return build_default_squeeze_params(image, false);
    }
    // Larger channel sets use the implicit wire form. Apply exactly the
    // decoder's default, and retain the existing no-squeeze fallback when
    // that schedule would require a zero-sized residual channel.
    if image.channels[0].width() <= MAX_FIRST_PREVIEW_SIZE
        && image.channels[0].height() <= MAX_FIRST_PREVIEW_SIZE
    {
        return Vec::new();
    }
    let params = build_default_squeeze_params(image, true);
    if default_dimensions_fit(image, &params) {
        params
    } else {
        Vec::new()
    }
}

pub(crate) const MAX_EXPLICIT_SQUEEZE_CHANNELS: u32 = 19;

fn default_dimensions_fit(image: &ModularImage, params: &[SqueezeParams]) -> bool {
    // Default schedules only revisit the original channels; residual bands
    // are inserted after that prefix or appended, never selected again.
    let mut dimensions: Vec<_> = image
        .channels
        .iter()
        .map(|c| (c.width(), c.height()))
        .collect();
    for param in params {
        let begin = param.begin_c as usize;
        let end = begin + param.num_c as usize;
        let Some(selected) = dimensions.get_mut(begin..end) else {
            return false;
        };
        for (width, height) in selected {
            let dimension = if param.horizontal { width } else { height };
            if *dimension < 2 {
                return false;
            }
            *dimension = dimension.div_ceil(2);
        }
    }
    true
}

fn build_default_squeeze_params(image: &ModularImage, implicit: bool) -> Vec<SqueezeParams> {
    let nb_channels = image.channels.len();
    if nb_channels == 0 {
        return Vec::new();
    }

    let mut params = Vec::new();
    let mut w = image.channels[0].width();
    let mut h = image.channels[0].height();

    // Skip squeeze entirely if both dimensions are already small
    if !implicit && w <= MAX_FIRST_PREVIEW_SIZE && h <= MAX_FIRST_PREVIEW_SIZE {
        return params;
    }

    // 4:2:0 chroma squeeze if channels 1 and 2 have same dimensions as channel 0
    // Skip squeeze directions that would produce 0-sized residual channels.
    if nb_channels > 2
        && image.channels[1].width() == w
        && image.channels[1].height() == h
        && (implicit || (image.channels[2].width() == w && image.channels[2].height() == h))
    {
        if implicit || w > 1 {
            params.push(SqueezeParams {
                horizontal: true,
                in_place: false,
                begin_c: 1,
                num_c: 2,
            });
        }
        if implicit || h > 1 {
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

/// Derive an encodable multi-group schedule without changing valid full plans.
///
/// Remaining channels must have nonzero tiles in their assigned group grid.
/// On an invalid full plan, retain v0.12's conservative explicit prefix, or
/// decline squeeze when that prefix cannot use the explicit field grammar.
pub(crate) fn default_squeeze_params_for_groups(
    image: &ModularImage,
    group_dim: usize,
    fallible: bool,
) -> Result<Vec<SqueezeParams>> {
    let mut params = default_squeeze_params(image);
    if params.is_empty() {
        return Ok(params);
    }
    if !group_dim.is_power_of_two() {
        return Err(crate::error::Error::InvalidInput(
            "invalid squeeze group dimension".into(),
        ));
    }
    let mut native_limit = params.len();
    let (mut horizontal, mut vertical) = (0_u32, 0_u32);
    for (index, step) in params.iter().enumerate() {
        if step.horizontal {
            horizontal += 1;
        } else {
            vertical += 1;
        }
        let boost = if horizontal.min(vertical) >= 3 { 3 } else { 0 };
        if horizontal.max(vertical) > group_dim.ilog2() + boost {
            native_limit = index.saturating_sub(1);
            break;
        }
    }
    // Every operation selects the original prefix or its two chroma channels.
    // Public channel/dimension limits bound this metadata-only allocation.
    let capacity = image.channels.len() + params.iter().map(|p| p.num_c as usize).sum::<usize>();
    let mut channels = crate::budget::vec_with_capacity_fallible(fallible, capacity)?;
    channels.extend(image.channels.iter().map(|channel| SqueezeGeometry {
        width: channel.width(),
        height: channel.height(),
        hshift: channel.hshift,
        vshift: channel.vshift,
    }));
    let mut keep = 0;
    let mut full_valid = false;
    for (index, step) in params.iter().enumerate() {
        if !squeeze_geometry_step(&mut channels, *step) {
            break;
        }
        let valid = squeeze_group_geometry_fits(&channels, group_dim);
        if valid && index < native_limit {
            keep = index + 1;
        }
        if index + 1 == params.len() {
            full_valid = valid;
        }
    }
    if full_valid {
        return Ok(params);
    }
    if params
        .iter()
        .any(|p| p.num_c > MAX_EXPLICIT_SQUEEZE_CHANNELS)
    {
        // v0.12 derives the full implicit default. A truncated implicit plan
        // would disagree with it; an explicit prefix cannot encode this count.
        params.clear();
    } else {
        params.truncate(keep);
    }
    Ok(params)
}

#[derive(Clone, Copy)]
struct SqueezeGeometry {
    width: usize,
    height: usize,
    hshift: u32,
    vshift: u32,
}

fn squeeze_geometry_step(channels: &mut Vec<SqueezeGeometry>, step: SqueezeParams) -> bool {
    let begin = step.begin_c as usize;
    let end = begin + step.num_c as usize;
    let offset = if step.in_place { end } else { channels.len() };
    for index in begin..end {
        let mut average = channels[index];
        let (size, shift) = if step.horizontal {
            (&mut average.width, &mut average.hshift)
        } else {
            (&mut average.height, &mut average.vshift)
        };
        if *size < 2 || *shift >= 31 {
            return false;
        }
        let residual_size = *size / 2;
        *size = size.div_ceil(2);
        *shift += 1;
        let mut residual = average;
        if step.horizontal {
            residual.width = residual_size;
        } else {
            residual.height = residual_size;
        }
        channels[index] = average;
        channels.insert(offset + index - begin, residual);
    }
    true
}

fn squeeze_group_geometry_fits(channels: &[SqueezeGeometry], group_dim: usize) -> bool {
    let cutoff = channels
        .iter()
        .position(|c| c.width > group_dim || c.height > group_dim)
        .unwrap_or(channels.len());
    channels[cutoff..].iter().all(|c| {
        if c.width == 0 || c.height == 0 {
            return true;
        }
        let extent = if c.hshift.min(c.vshift) >= 3 {
            8 * group_dim
        } else {
            group_dim
        };
        extent.checked_shr(c.hshift.max(c.vshift)).unwrap_or(0) != 0
    })
}

/// libjxl's `DefaultSqueezeParameters` (squeeze.cc:387-445) as the lossy
/// modular path needs it. Unlike [`default_squeeze_params`] (kept for the
/// lossless squeeze option, whose output is locked), this applies the 4:2:0
/// chroma squeeze even when the image is already 8x8 or smaller, as libjxl
/// does. libjxl also squeezes a chroma dimension of 1, leaving the samples
/// unchanged and a zero-sized residual; our channels cannot be zero-sized,
/// so those steps are left out and returned as extra (horizontal, vertical)
/// shift for the chroma channels' quantizers. The decoder never sees
/// quantizers (the tree multipliers carry them), so the reconstruction
/// matches libjxl's.
pub(crate) fn lossy_default_squeeze_params(image: &ModularImage) -> (Vec<SqueezeParams>, u32, u32) {
    let nb_channels = image.channels.len();
    if nb_channels == 0 {
        return (Vec::new(), 0, 0);
    }
    let mut params = Vec::new();
    let (mut extra_h, mut extra_v) = (0, 0);
    let mut w = image.channels[0].width();
    let mut h = image.channels[0].height();
    if nb_channels > 2
        && image.channels[1].width() == w
        && image.channels[1].height() == h
        && image.channels[2].width() == w
        && image.channels[2].height() == h
    {
        for horizontal in [true, false] {
            let dim = if horizontal { w } else { h };
            if dim > 1 {
                params.push(SqueezeParams {
                    horizontal,
                    in_place: false,
                    begin_c: 1,
                    num_c: 2,
                });
            } else if horizontal {
                extra_h = 1;
            } else {
                extra_v = 1;
            }
        }
    }
    let sp = SqueezeParams {
        horizontal: false,
        in_place: true,
        begin_c: 0,
        // LF frames append lossless zero extra-channel placeholders. Above
        // the explicit field limit, squeeze only the XYB prefix: channel
        // transforms are independent, so its coefficients stay identical.
        // Keep this explicit rather than using a default schedule: omitted
        // degenerate chroma steps carry quantizer shifts in this producer.
        num_c: if nb_channels > MAX_EXPLICIT_SQUEEZE_CHANNELS as usize {
            3
        } else {
            nb_channels as u32
        },
    };
    if w <= h && h > MAX_FIRST_PREVIEW_SIZE {
        params.push(sp);
        h = h.div_ceil(2);
    }
    while w > MAX_FIRST_PREVIEW_SIZE || h > MAX_FIRST_PREVIEW_SIZE {
        if w > MAX_FIRST_PREVIEW_SIZE {
            params.push(SqueezeParams {
                horizontal: true,
                ..sp
            });
            w = w.div_ceil(2);
        }
        if h > MAX_FIRST_PREVIEW_SIZE {
            params.push(sp);
            h = h.div_ceil(2);
        }
    }
    (params, extra_h, extra_v)
}

/// Apply forward squeeze transform to a modular image.
///
/// Modifies the image in-place, replacing channels with average+residual pairs.
/// Returns the squeeze parameters that were applied (for bitstream serialization).
pub fn apply_squeeze(image: &mut ModularImage, params: &[SqueezeParams]) -> Result<()> {
    apply_squeeze_stop(image, params, None)
}

/// `apply_squeeze` with cooperative cancellation.
pub(crate) fn apply_squeeze_stop(
    image: &mut ModularImage,
    params: &[SqueezeParams],
    stop: Option<&dyn enough::Stop>,
) -> Result<()> {
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
            crate::error::check_stop(stop)?;
            let rc = offset + c - begin_c;

            let (avg, res) = if param.horizontal {
                fwd_h_squeeze(&image.channels[c], stop)?
            } else {
                fwd_v_squeeze(&image.channels[c], stop)?
            };

            image.channels[c] = avg;
            image.channels.insert(rc, res);
        }
    }
    Ok(())
}

// ── Inverse Squeeze (unsqueeze) ──────────────────────────────────────────────
//
// Used by the LfFrame encode-back to reconstruct original-resolution channel
// values from lossy-quantized Squeeze-domain channels. Scalar-only since it
// runs once on small DC images (typically 64x64).

/// Smooth tendency for inverse squeeze, computed in i64 to avoid overflow.
/// Matches jxl-rs smooth_tendency_scalar (which matches the JXL spec).
#[allow(dead_code)]
fn smooth_tendency_i64(b: i64, a: i64, n: i64) -> i64 {
    let mut diff = 0i64;
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

/// Core unsqueeze: given avg, res, next_avg, prev → (a, b).
/// Matches jxl-rs `unsqueeze_scalar`.
#[allow(dead_code)]
fn unsqueeze_pair(avg: i32, res: i32, next_avg: i32, prev: i32) -> (i32, i32) {
    let tendency = smooth_tendency_i64(prev as i64, avg as i64, next_avg as i64);
    let diff = (res as i64) + tendency;
    let a = (avg as i64) + (diff / 2);
    let b = a - diff;
    (a as i32, b as i32)
}

/// Inverse horizontal squeeze: reconstruct original channel from avg + res.
///
/// avg has width `(orig_w+1)/2`, res has width `orig_w - (orig_w+1)/2`.
/// Output has width `orig_w`.
#[allow(dead_code)]
fn inv_h_squeeze(avg: &Channel, res: &Channel, orig_w: usize) -> Result<Channel> {
    let h = avg.height();
    let res_w = res.width();
    let mut out = Channel::new(orig_w, h)?;
    let has_tail = orig_w & 1 != 0;

    for y in 0..h {
        // prev_b: for x=0, use avg[0] as prev (no prior reconstructed value)
        let mut prev_b = avg.get(0, y);

        for x in 0..res_w {
            let next_avg = if x + 1 < res_w {
                avg.get(x + 1, y)
            } else if has_tail {
                // Tail avg pixel
                avg.get(res_w, y)
            } else {
                avg.get(x, y) // self-reference at boundary
            };

            let (a, b) = unsqueeze_pair(avg.get(x, y), res.get(x, y), next_avg, prev_b);
            out.set(x * 2, y, a);
            out.set(x * 2 + 1, y, b);
            prev_b = b;
        }

        // Odd width: tail pixel from avg
        if has_tail {
            out.set(orig_w - 1, y, avg.get(avg.width() - 1, y));
        }
    }

    out.hshift = avg.hshift.saturating_sub(1);
    out.vshift = avg.vshift;
    out.component = avg.component;
    Ok(out)
}

/// Inverse vertical squeeze: reconstruct original channel from avg + res.
///
/// avg has height `(orig_h+1)/2`, res has height `orig_h - (orig_h+1)/2`.
/// Output has height `orig_h`.
#[allow(dead_code)]
fn inv_v_squeeze(avg: &Channel, res: &Channel, orig_h: usize) -> Result<Channel> {
    let w = avg.width();
    let res_h = res.height();
    let mut out = Channel::new(w, orig_h)?;
    let has_tail = orig_h & 1 != 0;

    for x in 0..w {
        // prev_b: for y=0, use avg[0] as prev
        let mut prev_b = avg.get(x, 0);

        for y in 0..res_h {
            let next_avg = if y + 1 < res_h {
                avg.get(x, y + 1)
            } else if has_tail {
                avg.get(x, res_h)
            } else {
                avg.get(x, y) // self-reference at boundary
            };

            let (a, b) = unsqueeze_pair(avg.get(x, y), res.get(x, y), next_avg, prev_b);
            out.set(x, y * 2, a);
            out.set(x, y * 2 + 1, b);
            prev_b = b;
        }

        // Odd height: tail row from avg
        if has_tail {
            out.set(x, orig_h - 1, avg.get(x, avg.height() - 1));
        }
    }

    out.hshift = avg.hshift;
    out.vshift = avg.vshift.saturating_sub(1);
    out.component = avg.component;
    Ok(out)
}

/// Apply inverse squeeze to reconstruct original-resolution channels from
/// Squeeze-domain channels.
///
/// Takes the channel list produced by `apply_squeeze` and the same params,
/// and undoes the squeeze steps in reverse order.
#[allow(dead_code)]
pub fn inverse_squeeze(image: &mut ModularImage, params: &[SqueezeParams]) -> Result<()> {
    // Process in reverse order (undo last squeeze first)
    for param in params.iter().rev() {
        let begin_c = param.begin_c as usize;
        let end_c = begin_c + param.num_c as usize - 1;
        let offset = if param.in_place {
            end_c + 1
        } else {
            // Not in-place: residuals are at the end
            image.channels.len() - param.num_c as usize
        };

        // Reconstruct channels in reverse order (matching forward insertion order)
        for c in (begin_c..=end_c).rev() {
            let rc = offset + c - begin_c;
            let res = image.channels.remove(rc);
            let avg = &image.channels[c];

            // Determine original dimension
            let reconstructed = if param.horizontal {
                let orig_w = avg.width() + res.width();
                inv_h_squeeze(avg, &res, orig_w)?
            } else {
                let orig_h = avg.height() + res.height();
                inv_v_squeeze(avg, &res, orig_h)?
            };

            image.channels[c] = reconstructed;
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
        let (avg, res) = fwd_h_squeeze(&ch, None).unwrap();
        assert_eq!(avg.width(), 2);
        assert_eq!(avg.height(), 2);
        assert_eq!(res.width(), 2);
        assert_eq!(res.height(), 2);
    }

    #[test]
    fn test_h_squeeze_odd_width() {
        // 5x1 → avg 3x1, res 2x1
        let ch = Channel::from_vec(vec![10, 20, 30, 40, 50], 5, 1).unwrap();
        let (avg, res) = fwd_h_squeeze(&ch, None).unwrap();
        assert_eq!(avg.width(), 3);
        assert_eq!(res.width(), 2);
        // Last pixel (50) goes to avg[2]
        assert_eq!(avg.get(2, 0), 50);
    }

    #[test]
    fn test_v_squeeze_even_height() {
        // 2x4 → avg 2x2, res 2x2
        let ch = Channel::from_vec(vec![10, 20, 30, 40, 50, 60, 70, 80], 2, 4).unwrap();
        let (avg, res) = fwd_v_squeeze(&ch, None).unwrap();
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

    #[test]
    fn test_squeeze_roundtrip_gray_16x16() {
        // Verify forward+inverse squeeze is lossless
        let mut data = vec![0i32; 16 * 16];
        for y in 0..16 {
            for x in 0..16 {
                data[y * 16 + x] = (x * 16 + y * 4) as i32;
            }
        }
        let ch = Channel::from_vec(data.clone(), 16, 16).unwrap();
        let mut image = ModularImage {
            channels: vec![ch],
            bit_depth: 8,
            is_grayscale: true,
            has_alpha: false,
        };

        let params = default_squeeze_params(&image);
        assert!(!params.is_empty());

        apply_squeeze(&mut image, &params).unwrap();
        assert!(image.channels.len() > 1);

        inverse_squeeze(&mut image, &params).unwrap();
        assert_eq!(image.channels.len(), 1);
        assert_eq!(image.channels[0].width(), 16);
        assert_eq!(image.channels[0].height(), 16);

        // Check pixel-exact roundtrip
        for y in 0..16 {
            for x in 0..16 {
                assert_eq!(
                    image.channels[0].get(x, y),
                    data[y * 16 + x],
                    "mismatch at ({}, {}): got {}, expected {}",
                    x,
                    y,
                    image.channels[0].get(x, y),
                    data[y * 16 + x]
                );
            }
        }
    }

    #[test]
    fn test_squeeze_roundtrip_rgb_32x32() {
        // Test with 3 channels (RGB-like) and larger size
        let mut image = ModularImage {
            channels: Vec::new(),
            bit_depth: 16,
            is_grayscale: false,
            has_alpha: false,
        };
        let mut originals = Vec::new();
        for c in 0..3 {
            let mut data = vec![0i32; 32 * 32];
            for y in 0..32 {
                for x in 0..32 {
                    data[y * 32 + x] = ((x + c * 7) * 100 + (y + c * 3) * 50) as i32;
                }
            }
            originals.push(data.clone());
            let ch = Channel::from_vec(data, 32, 32).unwrap();
            image.channels.push(ch);
        }

        let params = default_squeeze_params(&image);
        apply_squeeze(&mut image, &params).unwrap();
        inverse_squeeze(&mut image, &params).unwrap();

        assert_eq!(image.channels.len(), 3);
        for (c, (channel, original)) in image.channels.iter().zip(originals.iter()).enumerate() {
            assert_eq!(channel.width(), 32);
            assert_eq!(channel.height(), 32);
            for y in 0..32 {
                for x in 0..32 {
                    assert_eq!(
                        channel.get(x, y),
                        original[y * 32 + x],
                        "ch{c} mismatch at ({x}, {y})"
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod default_squeeze_wire_tests {
    use super::*;
    use alloc::collections::BTreeMap;

    fn image(width: usize, height: usize, channels: usize) -> ModularImage {
        ModularImage {
            channels: (0..channels)
                .map(|_| Channel::new(width, height).unwrap())
                .collect(),
            bit_depth: 8,
            is_grayscale: false,
            has_alpha: false,
        }
    }

    #[test]
    fn implicit_defaults_match_unmodified_v012() {
        let mut cases: BTreeMap<(usize, usize, usize), Vec<SqueezeParams>> = BTreeMap::new();
        for line in include_str!("squeeze_defaults_v012.tsv").lines() {
            let fields: Vec<usize> = line.split('\t').map(|v| v.parse().unwrap()).collect();
            assert_eq!(fields.len(), 8);
            let params = cases.entry((fields[0], fields[1], fields[2])).or_default();
            assert_eq!(params.len(), fields[3], "reference index order");
            assert!(fields[4] <= 1 && fields[5] <= 1);
            params.push(SqueezeParams {
                horizontal: fields[4] != 0,
                in_place: fields[5] != 0,
                begin_c: fields[6] as u32,
                num_c: fields[7] as u32,
            });
        }
        for ((width, height, channels), expected) in cases {
            let source = image(width, height, channels);
            assert_eq!(
                build_default_squeeze_params(&source, true),
                expected,
                "unmodified reference {width}x{height}/{channels}"
            );
            if channels < 3 {
                let (lossy, h, v) = lossy_default_squeeze_params(&source);
                assert_eq!(lossy, expected);
                assert_eq!((h, v), (0, 0));
            }
        }
    }

    #[test]
    fn implicit_default_admission_retains_small_and_zero_residual_fallbacks() {
        assert!(default_squeeze_params(&image(16, 16, 0)).is_empty());
        assert!(!default_squeeze_params(&image(16, 16, 20)).is_empty());
        assert!(default_squeeze_params(&image(8, 8, 20)).is_empty());
        assert!(default_squeeze_params(&image(1, 17, 20)).is_empty());
        assert!(default_squeeze_params(&image(17, 1, 20)).is_empty());
        // A full-size alpha followed by a legal eighth-size depth makes the
        // implicit chroma pre-squeeze exhaust that depth before the main step.
        let mut source = image(16, 16, 20);
        source.channels[2] = Channel::new(2, 2).unwrap();
        source.channels[2].hshift = 3;
        source.channels[2].vshift = 3;
        assert!(default_squeeze_params(&source).is_empty());
        let invalid = [SqueezeParams {
            horizontal: true,
            in_place: true,
            begin_c: 20,
            num_c: 1,
        }];
        assert!(!default_dimensions_fit(&source, &invalid));
    }

    #[test]
    fn lf_explicit_schedules_keep_color_prefix_and_degenerate_quantizer_shifts() {
        assert_eq!(
            lossy_default_squeeze_params(&image(16, 16, 0)),
            (Vec::new(), 0, 0)
        );
        for (width, height) in [(1, 1), (1, 17), (17, 1), (16, 16), (257, 257)] {
            let (baseline, h, v) = lossy_default_squeeze_params(&image(width, height, 19));
            for channels in [20, 21] {
                let (actual, actual_h, actual_v) =
                    lossy_default_squeeze_params(&image(width, height, channels));
                let expected: Vec<_> = baseline
                    .iter()
                    .map(|step| SqueezeParams {
                        num_c: if step.num_c == 19 { 3 } else { step.num_c },
                        ..*step
                    })
                    .collect();
                assert_eq!(actual, expected);
                assert_eq!((actual_h, actual_v), (h, v));
                assert!(
                    actual
                        .iter()
                        .all(|step| step.num_c <= MAX_EXPLICIT_SQUEEZE_CHANNELS)
                );
            }
        }
    }
}

#[cfg(test)]
mod group_schedule_tests {
    use super::*;

    fn image(width: usize, height: usize, count: usize) -> ModularImage {
        ModularImage {
            channels: (0..count)
                .map(|_| Channel::new(width, height).unwrap())
                .collect(),
            bit_depth: 8,
            is_grayscale: false,
            has_alpha: false,
        }
    }
    fn geometry(image: &ModularImage) -> Vec<SqueezeGeometry> {
        image
            .channels
            .iter()
            .map(|c| SqueezeGeometry {
                width: c.width(),
                height: c.height(),
                hshift: c.hshift,
                vshift: c.vshift,
            })
            .collect()
    }

    #[test]
    fn valid_full_group_schedules_remain_unchanged() {
        for count in [1, 2, 3, 19, 20, 21] {
            let image = image(259, 17, count);
            assert_eq!(
                default_squeeze_params_for_groups(&image, 128, true).unwrap(),
                default_squeeze_params(&image)
            );
        }
        assert!(
            default_squeeze_params_for_groups(&image(8, 8, 3), 256, true)
                .unwrap()
                .is_empty()
        );
        assert!(default_squeeze_params_for_groups(&image(16, 16, 3), 0, true).is_err());
    }

    #[test]
    fn tall_explicit_prefix_avoids_invalid_remaining_tiles() {
        for (width, height) in [(8, 65665), (65665, 8)] {
            let original = image(width, height, 3);
            let full = default_squeeze_params(&original);
            let mut unbounded = original.clone();
            apply_squeeze(&mut unbounded, &full).unwrap();
            assert_eq!(unbounded.channels.len(), 49);
            assert!(!squeeze_group_geometry_fits(&geometry(&unbounded), 256));
            let params = default_squeeze_params_for_groups(&original, 256, true).unwrap();
            assert!(!params.is_empty());
            assert!(params.len() < full.len());
            let mut bounded = original;
            apply_squeeze(&mut bounded, &params).unwrap();
            assert!(squeeze_group_geometry_fits(&geometry(&bounded), 256));
        }
    }

    #[test]
    fn over_range_count_declines_a_truncated_implicit_schedule() {
        assert!(
            default_squeeze_params_for_groups(&image(8, 16449, 20), 128, true)
                .unwrap()
                .is_empty()
        );
        assert!(
            default_squeeze_params_for_groups(&image(16449, 8, 20), 128, true)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn ordered_global_prefix_does_not_absorb_later_small_bands() {
        let global = SqueezeGeometry {
            width: 4,
            height: 64,
            hshift: 1,
            vshift: 10,
        };
        let first_remaining = SqueezeGeometry {
            width: 8,
            height: 257,
            hshift: 0,
            vshift: 8,
        };
        let later_small = SqueezeGeometry {
            width: 4,
            height: 128,
            hshift: 1,
            vshift: 9,
        };
        assert!(!squeeze_group_geometry_fits(
            &[global, first_remaining, later_small],
            256
        ));
        assert!(!squeeze_group_geometry_fits(
            &[global, first_remaining, later_small, first_remaining],
            256
        ));
        assert!(squeeze_group_geometry_fits(
            &[global, later_small, first_remaining],
            256
        ));
        let lf = SqueezeGeometry {
            width: 300,
            height: 1,
            hshift: 3,
            vshift: 11,
        };
        assert!(squeeze_group_geometry_fits(&[lf], 256));
        assert!(!squeeze_group_geometry_fits(
            &[SqueezeGeometry { vshift: 12, ..lf }],
            256
        ));
    }

    #[test]
    fn unsupported_zero_residual_and_shift_overflow_are_not_applied() {
        let step = SqueezeParams {
            horizontal: true,
            in_place: true,
            begin_c: 0,
            num_c: 1,
        };
        assert!(!squeeze_geometry_step(
            &mut vec![SqueezeGeometry {
                width: 1,
                height: 17,
                hshift: 0,
                vshift: 0
            }],
            step
        ));
        assert!(!squeeze_geometry_step(
            &mut vec![SqueezeGeometry {
                width: 16,
                height: 17,
                hshift: 31,
                vshift: 0
            }],
            step
        ));
        assert!(squeeze_group_geometry_fits(
            &[SqueezeGeometry {
                width: 0,
                height: 300,
                hshift: 31,
                vshift: 31
            }],
            256
        ));
        assert!(squeeze_group_geometry_fits(
            &[SqueezeGeometry {
                width: 300,
                height: 0,
                hshift: 31,
                vshift: 31
            }],
            256,
        ));
        assert!(!squeeze_group_geometry_fits(
            &[SqueezeGeometry {
                width: 300,
                height: 1,
                hshift: u32::MAX,
                vshift: 0
            }],
            256,
        ));
        assert!(squeeze_group_geometry_fits(&[], 256));
    }
    #[test]
    fn inconsistent_internal_channel_geometry_declines_squeeze() {
        // These internal shapes cannot be produced by valid uniform public
        // dimension shifts; the metadata guard must still stop cleanly.
        let mut mixed = image(259, 17, 3);
        mixed.channels[1] = Channel::new(1, 1).unwrap();
        assert!(
            default_squeeze_params_for_groups(&mixed, 128, true)
                .unwrap()
                .is_empty()
        );
        let mut shifted = image(259, 17, 1);
        shifted.channels[0].hshift = 31;
        assert!(
            default_squeeze_params_for_groups(&shifted, 128, true)
                .unwrap()
                .is_empty()
        );
    }
}

#[cfg(test)]
mod gray_global_native_tests {
    use super::*;
    #[test]
    fn global_high_shifts_match_native_bands_without_invalid_remaining_tiles() {
        let rows: Vec<Vec<usize>> = include_str!("squeeze_gray_global_v012.tsv")
            .lines()
            .filter(|line| !line.starts_with('#'))
            .map(|line| line.split('\t').map(|v| v.parse().unwrap()).collect())
            .collect();
        for (width, height) in [(2, 16449), (16449, 2)] {
            for count in [1, 2] {
                let mut image = ModularImage {
                    channels: (0..count)
                        .map(|_| Channel::new(width, height).unwrap())
                        .collect(),
                    bit_depth: 8,
                    is_grayscale: true,
                    has_alpha: count == 2,
                };
                let params = default_squeeze_params(&image);
                assert_eq!(
                    default_squeeze_params_for_groups(&image, 128, true).unwrap(),
                    params
                );
                apply_squeeze(&mut image, &params).unwrap();
                let expected: Vec<_> = rows
                    .iter()
                    .filter(|r| r[..3] == [width, height, count])
                    .collect();
                assert_eq!(image.channels.len(), expected.len());
                let mut global = true;
                for (index, (channel, row)) in image.channels.iter().zip(expected).enumerate() {
                    global &= channel.width() <= 128 && channel.height() <= 128;
                    assert_eq!(
                        [
                            index,
                            channel.width(),
                            channel.height(),
                            channel.hshift as usize,
                            channel.vshift as usize,
                            usize::from(global)
                        ],
                        row[3..]
                    );
                    if !global {
                        let extent = if channel.hshift.min(channel.vshift) >= 3 {
                            1024_usize
                        } else {
                            128
                        };
                        assert_ne!(extent >> channel.hshift.max(channel.vshift), 0);
                    }
                }
            }
        }
    }
}
