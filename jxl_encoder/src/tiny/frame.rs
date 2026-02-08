// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Frame header writing for the tiny encoder.

use super::common::clamp;
use crate::bit_writer::BitWriter;
#[cfg(feature = "debug-tokens")]
use crate::debug_log;
use crate::error::Result;

/// Distance-dependent encoding parameters.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct DistanceParams {
    /// Target distance (quality).
    pub distance: f32,
    /// Global quantization scale.
    pub global_scale: i32,
    /// DC quantization parameter.
    pub quant_dc: i32,
    /// Scale factor (global_scale / 65536).
    pub scale: f32,
    /// Inverse scale factor.
    pub inv_scale: f32,
    /// DC scale factor.
    pub scale_dc: f32,
    /// X channel quant matrix scale (2-5).
    pub x_qm_scale: u32,
    /// B channel quant matrix scale (2-5).
    pub b_qm_scale: u32,
    /// Number of EPF iterations (0-3).
    pub epf_iters: u32,
}

/// Pixel-level statistics for chroma quantization adjustment.
///
/// Ported from libjxl enc_frame.cc:572-645.
/// Computes max horizontal/vertical gradients of X and B-Y channels
/// to determine how much chroma quantization can be coarsened.
pub(crate) struct PixelStatsForChromacityAdjustment {
    /// Max gradient of X (opsin) channel.
    dx: f32,
    /// Max gradient of B-Y channel.
    db: f32,
    /// Exposed blue metric (B pixels much brighter than Y).
    exposed_blue: f32,
}

impl PixelStatsForChromacityAdjustment {
    /// Compute max horizontal/vertical gradient of a single plane.
    pub(crate) fn calc_plane(plane: &[f32], width: usize, height: usize) -> f32 {
        let mut xmax: f32 = 0.0;
        let mut ymax: f32 = 0.0;
        for ty in 1..height {
            for tx in 1..width {
                let cur = plane[ty * width + tx];
                let prev_row = plane[(ty - 1) * width + tx];
                let prev = plane[ty * width + (tx - 1)];
                xmax = xmax.max((cur - prev).abs());
                ymax = ymax.max((cur - prev_row).abs());
            }
        }
        xmax.max(ymax)
    }

    /// Compute B-Y gradient and exposed blue metric.
    pub(crate) fn calc_exposed_blue(
        plane_y: &[f32],
        plane_b: &[f32],
        width: usize,
        height: usize,
    ) -> (f32, f32) {
        let mut eb: f32 = 0.0;
        let mut xmax: f32 = 0.0;
        let mut ymax: f32 = 0.0;
        for ty in 1..height {
            for tx in 1..width {
                let cur_y = plane_y[ty * width + tx];
                let cur_b = plane_b[ty * width + tx];
                let exposed_b = cur_b - cur_y * 1.2;
                let diff_b = cur_b - cur_y;
                let prev_row_b = plane_b[(ty - 1) * width + tx];
                let prev_b = plane_b[ty * width + (tx - 1)];
                let diff_prev_row = prev_row_b - plane_y[(ty - 1) * width + tx];
                let diff_prev = prev_b - plane_y[ty * width + (tx - 1)];
                xmax = xmax.max((diff_b - diff_prev).abs());
                ymax = ymax.max((diff_b - diff_prev_row).abs());
                if exposed_b >= 0.0 {
                    let eb_val = exposed_b * ((cur_b - prev_b).abs() + (cur_b - prev_row_b).abs());
                    eb = eb.max(eb_val);
                }
            }
        }
        (xmax.max(ymax), eb)
    }

    /// Compute all pixel stats from XYB image.
    pub(crate) fn calc(
        xyb_x: &[f32],
        xyb_y: &[f32],
        xyb_b: &[f32],
        width: usize,
        height: usize,
    ) -> Self {
        let dx = Self::calc_plane(xyb_x, width, height);
        let (db, exposed_blue) = Self::calc_exposed_blue(xyb_y, xyb_b, width, height);
        Self {
            dx,
            db,
            exposed_blue,
        }
    }

    /// How much X channel quantization can be coarsened (0-3).
    pub(crate) fn how_much_is_x_channel_pixelized(&self) -> u32 {
        if self.dx >= 0.026 {
            return 3;
        }
        if self.dx >= 0.022 {
            return 2;
        }
        if self.dx >= 0.015 {
            return 1;
        }
        0
    }

    /// How much B channel quantization can be coarsened (0-3).
    pub(crate) fn how_much_is_b_channel_pixelized(&self) -> u32 {
        let add = if self.exposed_blue >= 0.13 { 1 } else { 0 };
        if self.db > 0.38 {
            return 2 + add;
        }
        if self.db > 0.33 {
            return 1 + add;
        }
        if self.db > 0.28 {
            return add;
        }
        0
    }
}

/// Compute DC quantization scale from distance.
fn quant_dc(distance: f32) -> f32 {
    // Full libjxl constants (from enc_adaptive_quantization.cc)
    const DC_QUANT_POW: f32 = 0.83;
    const DC_QUANT: f32 = 1.095_924;
    const DC_MUL: f32 = 0.3;

    let effective_dist = DC_MUL * (distance / DC_MUL).powf(DC_QUANT_POW);
    let effective_dist = clamp(effective_dist, 0.5 * distance, distance);
    (DC_QUANT / effective_dist).min(50.0)
}

impl DistanceParams {
    /// Compute distance-dependent parameters using fixed global_scale formula.
    /// This is the fallback when no quant field is available.
    pub fn compute(distance: f32) -> Self {
        // Use median=AC_QUANT/distance, MAD=0 for fixed formula (matches libjxl-tiny)
        Self::compute_internal(distance, None)
    }

    /// Compute distance-dependent parameters using content-adaptive global_scale.
    ///
    /// This matches full libjxl's SetQuantField behavior: global_scale is derived
    /// from the median and MAD (median absolute deviation) of the quant field.
    /// For high-variance images, MAD is large, so (median - MAD) is smaller,
    /// giving a smaller global_scale (finer quantization, better quality).
    #[allow(dead_code)]
    pub fn compute_from_quant_field(distance: f32, quant_field: &[f32]) -> Self {
        if quant_field.is_empty() {
            return Self::compute(distance);
        }

        // Compute median using nth_element equivalent (partial sort)
        let mut data: Vec<f32> = quant_field.to_vec();
        let mid = data.len() / 2;
        data.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap());
        let quant_median = data[mid];

        // Compute median absolute deviation from median
        let mut deviations: Vec<f32> = data.iter().map(|&x| (x - quant_median).abs()).collect();
        deviations.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap());
        let quant_median_absd = deviations[mid];

        #[cfg(feature = "debug-tokens")]
        eprintln!(
            "[adaptive] d={:.2} median={:.4} mad={:.4} (median-mad)={:.4}",
            distance,
            quant_median,
            quant_median_absd,
            quant_median - quant_median_absd
        );
        Self::compute_internal(distance, Some((quant_median, quant_median_absd)))
    }

    /// Internal implementation shared by both compute methods.
    fn compute_internal(distance: f32, quant_stats: Option<(f32, f32)>) -> Self {
        const GLOBAL_SCALE_DENOM: i32 = 1 << 16;
        const GLOBAL_SCALE_NUMERATOR: i32 = 4096;
        const AC_QUANT: f32 = 0.765;
        const QUANT_FIELD_TARGET: f32 = 5.0;

        let qdc = quant_dc(distance);

        // Compute global_scale from quant field content when available.
        // libjxl's ComputeGlobalScaleAndQuant uses (median - MAD) of the quant
        // field to adapt quantization precision to image content. For high-variance
        // images, MAD is large so global_scale is smaller (coarser discretization
        // but better range), which preserves the adaptive quant field's variation.
        let scale = if let Some((quant_median, quant_median_absd)) = quant_stats {
            // Content-adaptive: matches libjxl quantizer.cc:ComputeGlobalScaleAndQuant
            (GLOBAL_SCALE_DENOM as f32) * (quant_median - quant_median_absd) / QUANT_FIELD_TARGET
        } else {
            // Fixed formula fallback (libjxl-tiny style)
            (GLOBAL_SCALE_DENOM as f32) * AC_QUANT / (distance * QUANT_FIELD_TARGET)
        };
        let scale = clamp(scale, 1.0, (1 << 15) as f32);

        let scaled_quant_dc = (qdc * (GLOBAL_SCALE_NUMERATOR as f32) * 1.6) as i32;
        let global_scale = clamp(scale as i32, 1, scaled_quant_dc);

        let scale = (global_scale as f32) / (GLOBAL_SCALE_DENOM as f32);
        let inv_scale = 1.0 / scale;

        #[cfg(feature = "debug-tokens")]
        {
            let mode = if quant_stats.is_some() {
                "adaptive"
            } else {
                "fixed"
            };
            eprintln!(
                "[global_scale] d={:.2} mode={} global_scale={} inv_scale={:.4}",
                distance, mode, global_scale, inv_scale
            );
            if let Some((median, mad)) = quant_stats {
                eprintln!(
                    "[global_scale] median={:.4} mad={:.4} (median-mad)={:.4}",
                    median,
                    mad,
                    median - mad
                );
            }
        }

        let quant_dc = clamp((qdc / scale + 0.5) as i32, 1, 1 << 16);
        let scale_dc = (quant_dc as f32) * scale;

        // X quant matrix scale - full libjxl formula (enc_frame.cc:655-661)
        // Starts at 3, steps at [2.5, 5.5, 9.5] (vs libjxl-tiny: starts at 2, steps [1.25, 9.0])
        let mut x_qm_scale = 3u32;
        let x_qm_scale_steps = [2.5f32, 5.5f32, 9.5f32];
        for step in &x_qm_scale_steps {
            if distance > *step {
                x_qm_scale += 1;
            }
        }

        // B quant matrix scale defaults to 2 (will be adjusted by pixel stats if available)
        let b_qm_scale = 2u32;

        // EPF iterations
        const EPF_THRESHOLDS: [f32; 3] = [0.7, 1.5, 4.0];
        let mut epf_iters = 0u32;
        for threshold in &EPF_THRESHOLDS {
            if distance >= *threshold {
                epf_iters += 1;
            }
        }

        Self {
            distance,
            global_scale,
            quant_dc,
            scale,
            inv_scale,
            scale_dc,
            x_qm_scale,
            b_qm_scale,
            epf_iters,
        }
    }

    /// Compute raw quantization field value for a uniform (constant) image.
    ///
    /// For adaptive quantization with a uniform image, the quant field is
    /// approximately 0.73-0.78 (not 1.0) due to the masking computations.
    /// This value was determined empirically by comparing with libjxl-tiny output.
    ///
    /// raw_quant = clamp(round(quant_field * inv_scale + 0.5), 1, 255)
    ///
    /// For distance=1.0 with quant_field≈0.73:
    ///   raw_quant = round(0.73 * 8.93 + 0.5) ≈ 7
    #[allow(dead_code)]
    pub fn raw_quant_uniform(&self) -> u8 {
        // Use 0.73 as the approximate quant_field for uniform images.
        // This value was determined empirically by comparing with libjxl-tiny output.
        //
        // Note: For proper adaptive quantization, this should be computed per-block
        // based on image masking. The uniform value of ~7 works well for smooth images.
        // High-frequency images (checkerboard, noise) have different masking and
        // libjxl-tiny computes different raw_qf values per-block.
        const UNIFORM_QUANT_FIELD: f32 = 0.73;
        clamp(
            (UNIFORM_QUANT_FIELD * self.inv_scale + 0.5).round() as i32,
            1,
            255,
        ) as u8
    }

    /// Apply pixel-level chromacity adjustments from pre-computed pixel stats.
    ///
    /// Matches libjxl's `ComputeChromacityAdjustments` (enc_frame.cc:647-674):
    /// - x_qm_scale = max(distance_based, 2 + HowMuchIsXChannelPixelized())
    /// - b_qm_scale = 2 + HowMuchIsBChannelPixelized()
    ///
    /// IMPORTANT: The pixel stats must be computed from the XYB image BEFORE
    /// gaborish inverse, matching libjxl's pipeline order. Gaborish sharpening
    /// inflates gradients and would produce overly aggressive chromacity adjustment.
    pub fn apply_chromacity_adjustment(&mut self, x_pixelized: u32, b_pixelized: u32) {
        // For X, take the most severe adjustment (max of distance-based and pixel-based)
        self.x_qm_scale = self.x_qm_scale.max(2 + x_pixelized);

        // B only adjusted by pixel-based approach
        self.b_qm_scale = 2 + b_pixelized;

        #[cfg(feature = "debug-tokens")]
        eprintln!(
            "[chromacity] x_pixelized={} b_pixelized={} -> x_qm_scale={} b_qm_scale={}",
            x_pixelized, b_pixelized, self.x_qm_scale, self.b_qm_scale,
        );
    }
}

/// Write the frame header.
///
/// When `enable_noise` is true, sets the ENABLE_NOISE flag (bit 0) in addition
/// to SKIP_ADAPTIVE_LF_SMOOTHING (bit 7). Flags value: 128 without noise, 129 with.
///
/// When `enable_gaborish` is true, signals gab=1 in the loop filter so the
/// decoder applies its 3x3 Gabor-like blur. The encoder must have applied
/// the inverse sharpening pre-filter to compensate.
pub fn write_frame_header(
    x_qm_scale: u32,
    b_qm_scale: u32,
    epf_iters: u32,
    enable_noise: bool,
    enable_gaborish: bool,
    num_extra_channels: usize,
    writer: &mut BitWriter,
) -> Result<()> {
    // Flags: SKIP_ADAPTIVE_LF_SMOOTHING (0x80) | optional ENABLE_NOISE (0x01)
    let flags: u64 = 128 | if enable_noise { 1 } else { 0 };
    // U64 encoding: flags is in range [17, 272], so selector=2, data=flags-17
    let flags_data = flags - 17;

    writer.write(1, 0)?; // not all default
    writer.write(2, 0)?; // regular frame
    writer.write(1, 0)?; // vardct (not modular)
    writer.write(2, 2)?; // flags U64 selector (17 .. 272)
    writer.write(8, flags_data)?; // flags value
    writer.write(2, 0)?; // no upsampling

    // ec_upsampling: one U2S(1,2,4,8) per extra channel
    for _ in 0..num_extra_channels {
        writer.write(2, 0)?; // selector 0 = no upsampling (1)
    }

    writer.write(3, x_qm_scale as u64)?;
    writer.write(3, b_qm_scale as u64)?;
    writer.write(2, 0)?; // one pass
    writer.write(1, 0)?; // no custom frame size or origin
    writer.write(2, 0)?; // replace blend mode

    // ec_blending_info: one BlendingInfo per extra channel
    for _ in 0..num_extra_channels {
        writer.write(2, 0)?; // mode = Replace (selector 0)
    }

    writer.write(1, 1)?; // last frame
    writer.write(2, 0)?; // no name

    // Loop filter: all_default=1 means gab=true, epf_iters=2
    if enable_gaborish && epf_iters == 2 {
        writer.write(1, 1)?; // all_default (gab=true, epf_iters=2)
    } else {
        writer.write(1, 0)?; // not all default
        writer.write(1, enable_gaborish as u64)?; // gab
        if enable_gaborish {
            writer.write(1, 0)?; // gab_custom=false (use default decoder weights)
        }
        writer.write(2, epf_iters as u64)?;
        if epf_iters > 0 {
            writer.write(1, 0)?; // default epf sharpness
            writer.write(1, 0)?; // default epf weights
            writer.write(1, 0)?; // default epf sigma
        }
        writer.write(2, 0)?; // no loop filter extensions
    }
    writer.write(2, 0)?; // no frame header extensions
    Ok(())
}

/// Write quantization scales.
pub fn write_quant_scales(global_scale: i32, quant_dc: i32, writer: &mut BitWriter) -> Result<()> {
    if global_scale < 2049 {
        writer.write(2, 0)?;
        writer.write(11, (global_scale - 1) as u64)?;
    } else if global_scale < 4097 {
        writer.write(2, 1)?;
        writer.write(11, (global_scale - 2049) as u64)?;
    } else if global_scale < 8193 {
        writer.write(2, 2)?;
        writer.write(12, (global_scale - 4097) as u64)?;
    } else {
        writer.write(2, 3)?;
        writer.write(16, (global_scale - 8193) as u64)?;
    }

    if quant_dc == 16 {
        writer.write(2, 0)?;
    } else if quant_dc < 33 {
        writer.write(2, 1)?;
        writer.write(5, (quant_dc - 1) as u64)?;
    } else if quant_dc < 257 {
        writer.write(2, 2)?;
        writer.write(8, (quant_dc - 1) as u64)?;
    } else {
        writer.write(2, 3)?;
        writer.write(16, (quant_dc - 1) as u64)?;
    }
    Ok(())
}

/// Write the TOC (table of contents).
pub fn write_toc(section_sizes: &[usize], writer: &mut BitWriter) -> Result<()> {
    writer.write(1, 0)?; // no permutation
    writer.zero_pad_to_byte(); // before TOC entries

    const BITS: [usize; 4] = [10, 14, 22, 30];

    #[allow(clippy::unused_enumerate_index)]
    for (_idx, &section_size) in section_sizes.iter().enumerate() {
        let mut offset = 0;
        let mut success = false;
        for (i, &bits) in BITS.iter().enumerate() {
            if section_size < offset + (1 << bits) {
                #[cfg(feature = "debug-tokens")]
                debug_log!(
                    "TOC[{}]: size={}, selector={}, bits={}, value={}",
                    _idx,
                    section_size,
                    i,
                    bits,
                    section_size - offset
                );
                writer.write(2, i as u64)?;
                writer.write(bits, (section_size - offset) as u64)?;
                success = true;
                break;
            }
            offset += 1 << bits;
        }
        assert!(success, "Section size {} too large", section_size);
    }
    writer.zero_pad_to_byte();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distance_params() {
        let params = DistanceParams::compute(1.0);
        assert!(params.global_scale > 0);
        assert!(params.quant_dc > 0);
        assert!(params.scale > 0.0);
        // x_qm_scale: starts at 3 (full libjxl), distance 1.0 < 2.5 so no increment
        assert_eq!(params.x_qm_scale, 3);
        // b_qm_scale defaults to 2 (adjusted by pixel stats when available)
        assert_eq!(params.b_qm_scale, 2);
        // EPF iterations for distance 1.0: >= 0.7 (1 iter), but < 1.5 (not 2 iters)
        assert_eq!(params.epf_iters, 1);

        let params_low = DistanceParams::compute(0.5);
        assert!(params_low.global_scale >= params.global_scale);
        // Lower distance = fewer EPF iterations (0.5 < 0.7)
        assert_eq!(params_low.epf_iters, 0);

        // Higher distance increases x_qm_scale
        let params_high = DistanceParams::compute(3.0);
        // 3.0 > 2.5 -> x_qm_scale = 4, 3.0 < 5.5 -> still 4
        assert_eq!(params_high.x_qm_scale, 4);
        // 2.0 >= 0.7 and >= 1.5 -> epf_iters = 2
        assert_eq!(params_high.epf_iters, 2);

        // Very high distance
        let params_vhigh = DistanceParams::compute(10.0);
        // 10.0 > 2.5 > 5.5 > 9.5 -> x_qm_scale = 6
        assert_eq!(params_vhigh.x_qm_scale, 6);
    }

    #[test]
    fn test_quant_dc() {
        // Higher distance = lower quality = smaller quant_dc
        let qdc_low = quant_dc(0.5);
        let qdc_high = quant_dc(2.0);
        assert!(qdc_low > qdc_high);
    }
}
