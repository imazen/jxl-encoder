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
    /// Number of EPF iterations (0-3).
    pub epf_iters: u32,
}

/// Compute DC quantization scale from distance.
fn quant_dc(distance: f32) -> f32 {
    const DC_QUANT_POW: f32 = 0.57;
    const DC_QUANT: f32 = 1.12;
    const DC_MUL: f32 = 2.9;

    let effective_dist = DC_MUL * (distance / DC_MUL).powf(DC_QUANT_POW);
    let effective_dist = clamp(effective_dist, 0.5 * distance, distance);
    (DC_QUANT / effective_dist).min(50.0)
}

impl DistanceParams {
    /// Compute distance-dependent parameters.
    pub fn compute(distance: f32) -> Self {
        const GLOBAL_SCALE_DENOM: i32 = 1 << 16;
        const GLOBAL_SCALE_NUMERATOR: i32 = 4096;
        const AC_QUANT: f32 = 0.8;
        const QUANT_FIELD_TARGET: f32 = 5.0;

        let qdc = quant_dc(distance);
        let scale = (GLOBAL_SCALE_DENOM as f32) * AC_QUANT / (distance * QUANT_FIELD_TARGET);
        let scale = clamp(scale, 1.0, (1 << 15) as f32);

        let scaled_quant_dc = (qdc * (GLOBAL_SCALE_NUMERATOR as f32) * 1.6) as i32;
        let global_scale = clamp(scale as i32, 1, scaled_quant_dc);

        let scale = (global_scale as f32) / (GLOBAL_SCALE_DENOM as f32);
        let inv_scale = 1.0 / scale;

        let quant_dc = clamp((qdc / scale + 0.5) as i32, 1, 1 << 16);
        let scale_dc = (quant_dc as f32) * scale;

        // X quant matrix scale
        let mut x_qm_scale = 2u32;
        let x_qm_scale_steps = [1.25f32, 9.0f32];
        for step in &x_qm_scale_steps {
            if distance > *step {
                x_qm_scale += 1;
            }
        }
        if distance < 0.299 {
            x_qm_scale += 1;
        }

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
    epf_iters: u32,
    enable_noise: bool,
    enable_gaborish: bool,
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
    writer.write(3, x_qm_scale as u64)?;
    writer.write(3, 2)?; // b_qm_scale
    writer.write(2, 0)?; // one pass
    writer.write(1, 0)?; // no custom frame size or origin
    writer.write(2, 0)?; // replace blend mode
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
        // x_qm_scale: starts at 2, distance 1.0 < 1.25 so no increment
        assert_eq!(params.x_qm_scale, 2);
        // EPF iterations for distance 1.0: >= 0.7 (1 iter), but < 1.5 (not 2 iters)
        assert_eq!(params.epf_iters, 1);

        let params_low = DistanceParams::compute(0.5);
        assert!(params_low.global_scale >= params.global_scale);
        // Lower distance = fewer EPF iterations (0.5 < 0.7)
        assert_eq!(params_low.epf_iters, 0);

        // Higher distance increases x_qm_scale
        let params_high = DistanceParams::compute(2.0);
        // 2.0 > 1.25 -> x_qm_scale = 3, 2.0 < 9.0 -> still 3
        assert_eq!(params_high.x_qm_scale, 3);
        // 2.0 >= 0.7 and >= 1.5 -> epf_iters = 2
        assert_eq!(params_high.epf_iters, 2);
    }

    #[test]
    fn test_quant_dc() {
        // Higher distance = lower quality = smaller quant_dc
        let qdc_low = quant_dc(0.5);
        let qdc_high = quant_dc(2.0);
        assert!(qdc_low > qdc_high);
    }
}
