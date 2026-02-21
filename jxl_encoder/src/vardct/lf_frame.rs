// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! LfFrame (separate DC frame) encoder.
//!
//! Encodes DC coefficients as a separate JXL frame (frame_type=1, dc_level=1)
//! before the main VarDCT frame. The DC frame uses modular encoding with
//! distance-scaled quantization factors, matching libjxl's enc_cache.cc behavior.
//!
//! The float XYB DC values are scaled by custom `enc_factors`, converted to
//! integers in [Y, X, B-Y] channel order, and encoded losslessly. The distance
//! is baked into the scaling factors, so the modular encoding itself is lossless.

use crate::bit_writer::BitWriter;
use crate::error::Result;
use crate::f16::f16_roundtrip;
use crate::headers::frame_header::FrameHeader;
use crate::modular::channel::{Channel, ModularImage};

/// Minimum butteraugli distance (libjxl kMinButteraugliDistance).
/// libjxl enc_params.h:201: "Below d0.05 is not useful and risks going outside
/// Level 5 limits (in particular modular_16bit_buffers becomes an issue for DC)"
const K_MIN_BUTTERAUGLI_DISTANCE: f32 = 0.05;

/// Block dimension (must match common::BLOCK_DIM).
const BLOCK_DIM: usize = 8;

/// Compute float DC values from padded XYB image.
///
/// For each 8x8 block, computes the DC coefficient as `sum(block_pixels) / 64.0`,
/// matching the forward DCT8x8 DC coefficient (the [0][0] output of forward DCT).
///
/// Our forward DCT8x8 applies 1/8 normalization per dimension (row and column),
/// so the 2D DC = sum(all 64 pixels) * (1/8) * (1/8) = sum / 64.
///
/// libjxl's DC frame (enc_cache.cc:106) uses the raw forward DCT DC output
/// before quantization, which equals sum/64 in our DCT convention.
///
/// # Arguments
/// * `xyb` - [X, Y, B] channel data, padded to block boundaries (row stride = padded_width)
/// * `padded_width` - Row stride in pixels (= xsize_blocks * 8)
/// * `xsize_blocks` - Number of 8x8 blocks horizontally
/// * `ysize_blocks` - Number of 8x8 blocks vertically
///
/// # Returns
/// `[X, Y, B]` arrays, each of length `ysize_blocks * xsize_blocks`, flat row-major.
pub(crate) fn compute_float_dc(
    xyb: [&[f32]; 3],
    padded_width: usize,
    xsize_blocks: usize,
    ysize_blocks: usize,
) -> [Vec<f32>; 3] {
    let n = xsize_blocks * ysize_blocks;
    let mut dc: [Vec<f32>; 3] = core::array::from_fn(|_| vec![0.0f32; n]);

    for c in 0..3 {
        let ch = xyb[c];
        for by in 0..ysize_blocks {
            for bx in 0..xsize_blocks {
                let mut sum = 0.0f32;
                let base_y = by * BLOCK_DIM;
                let base_x = bx * BLOCK_DIM;
                for dy in 0..BLOCK_DIM {
                    let row_offset = (base_y + dy) * padded_width + base_x;
                    for dx in 0..BLOCK_DIM {
                        sum += ch[row_offset + dx];
                    }
                }
                dc[c][by * xsize_blocks + bx] = sum / 64.0;
            }
        }
    }

    dc
}

/// Custom DC quantization factors computed from distance.
///
/// These are the values written to the LfFrame's LfGlobal section as
/// `dc_quant[c] * 128.0` in F16 format. The decoder reads them back
/// and uses `1.0 / dc_quant[c]` (= inv_dc_quant) to convert integers
/// back to float XYB values.
#[derive(Debug, Clone, Copy)]
pub(crate) struct DcQuantFactors {
    /// dc_quant values [X, Y, B] after F16 roundtrip.
    /// These are 1/enc_factor for each channel.
    pub dc_quant: [f32; 3],
    /// Inverse dc_quant = enc_factor for each channel, after F16 roundtrip.
    /// Used to convert float DC to integers: `int = round(float_dc * inv_dc_quant)`.
    pub inv_dc_quant: [f32; 3],
}

impl DcQuantFactors {
    /// Compute DC quantization factors from the main frame's butteraugli distance.
    ///
    /// Matches libjxl enc_modular.cc:749-768:
    /// ```text
    /// enc_factors[0] = 65536 / (1 + 23*dc_distance)  // X
    /// enc_factors[1] = 4096 / (1 + 14*dc_distance)   // Y
    /// enc_factors[2] = 4096 / (1 + 14*dc_distance)   // B
    /// dc_quant[c] = 1/enc_factors[c]
    /// ```
    /// Then F16-roundtripped through `dc_quant[c] * 128.0` for exact decoder parity.
    pub fn compute(main_distance: f32) -> Self {
        let dc_distance = (main_distance * 0.02).max(K_MIN_BUTTERAUGLI_DISTANCE * 0.02);

        let mut enc_factors = [65536.0f32, 4096.0, 4096.0]; // [X, Y, B]
        enc_factors[0] /= 1.0 + 23.0 * dc_distance;
        enc_factors[1] /= 1.0 + 14.0 * dc_distance;
        enc_factors[2] /= 1.0 + 14.0 * dc_distance;

        // F16 roundtrip to get exact decoder-matching factors.
        // The bitstream stores dc_quant[c] * 128.0 as F16.
        // The decoder reads F16, divides by 128 → gets dc_quant.
        // inv_dc_quant = 1.0 / dc_quant = the effective enc_factor.
        let dc_quant: [f32; 3] = [
            f16_roundtrip(128.0 / enc_factors[0]) / 128.0,
            f16_roundtrip(128.0 / enc_factors[1]) / 128.0,
            f16_roundtrip(128.0 / enc_factors[2]) / 128.0,
        ];
        let inv_dc_quant: [f32; 3] = dc_quant.map(|q| 1.0 / q);

        Self {
            dc_quant,
            inv_dc_quant,
        }
    }
}

/// Round half-away-from-zero (matching libjxl's `float_to_int` for Y and X channels).
///
/// libjxl enc_modular.cc:796-806:
/// `out = (int)(val * factor + (val < 0 ? -0.5f : 0.5f))`
fn round_hafz(val: f32) -> i32 {
    (val + if val < 0.0 { -0.5 } else { 0.5 }) as i32
}

/// Encode the LfFrame (separate DC frame) to the bitstream.
///
/// The LfFrame contains pre-quantization float XYB DC coefficients at 1/8
/// resolution, converted to integers via distance-scaled quantization factors.
/// It is written as a complete JXL frame (frame_type=1) before the main VarDCT frame.
///
/// Returns the decoded-back DC values in `[X, Y, B]` channel order — the exact
/// float values the decoder will reconstruct from the LfFrame integers. This
/// matches libjxl's decode-back step (enc_cache.cc:195-222) where the encoded
/// LfFrame is immediately decoded to get exact decoder DC for the main frame.
///
/// # Arguments
/// * `float_dc` - Pre-quantization float XYB DC values: [XYB channel][yb * xsize_blocks + xb]
/// * `main_distance` - Main frame's butteraugli distance
/// * `xsize_blocks` - Number of 8x8 blocks horizontally
/// * `ysize_blocks` - Number of 8x8 blocks vertically
/// * `use_ans` - Whether to use ANS entropy coding
/// * `effort` - Effort level (1-10)
pub(crate) fn encode_lf_frame(
    float_dc: &[Vec<f32>; 3],
    main_distance: f32,
    xsize_blocks: usize,
    ysize_blocks: usize,
    use_ans: bool,
    effort: u8,
    writer: &mut BitWriter,
) -> Result<[Vec<f32>; 3]> {
    let factors = DcQuantFactors::compute(main_distance);

    #[cfg(feature = "trace-bitstream")]
    {
        eprintln!("LFRAME: dc_quant = {:?}", factors.dc_quant);
        eprintln!("LFRAME: inv_dc_quant = {:?}", factors.inv_dc_quant);
        eprintln!("LFRAME: dim = {}x{}", xsize_blocks, ysize_blocks);
    }

    let n = xsize_blocks * ysize_blocks;

    // Convert float DC to [Y, X, B-Y] integers.
    //
    // Channel order in modular: [0=Y, 1=X, 2=B-Y]
    // XYB input: [0=X, 1=Y, 2=B]
    //
    // Rounding matches libjxl enc_modular.cc:796-814:
    // - Y and X: round-half-away-from-zero
    // - B: always +0.5 (then subtract Y_int)
    let mut ch_y_data = Vec::with_capacity(n);
    let mut ch_x_data = Vec::with_capacity(n);
    let mut ch_by_data = Vec::with_capacity(n);

    for ((&dc_x, &dc_y), &dc_b) in float_dc[0]
        .iter()
        .zip(float_dc[1].iter())
        .zip(float_dc[2].iter())
    {
        let y_int = round_hafz(dc_y * factors.inv_dc_quant[1]); // Y
        let x_int = round_hafz(dc_x * factors.inv_dc_quant[0]); // X
        // B channel: always +0.5 (not sign-dependent), then subtract Y_int
        let b_int = (dc_b * factors.inv_dc_quant[2] + 0.5) as i32 - y_int;
        ch_y_data.push(y_int);
        ch_x_data.push(x_int);
        ch_by_data.push(b_int);
    }

    // Decode-back: compute the exact float DC values the decoder will reconstruct.
    //
    // libjxl (enc_cache.cc:195-222) decodes the encoded LfFrame to get exact decoder
    // DC values. Since modular encoding is lossless for integers, the decode-back is
    // equivalent to: decoded_float = integer * dc_quant.
    //
    // Channel conversion: modular [Y, X, B-Y] integers → [X, Y, B] XYB floats
    //   decoded_Y = y_int * dc_quant[1]
    //   decoded_X = x_int * dc_quant[0]
    //   decoded_B = (by_int + y_int) * dc_quant[2]
    let decoded_dc = {
        let mut dc_x = Vec::with_capacity(n);
        let mut dc_y = Vec::with_capacity(n);
        let mut dc_b = Vec::with_capacity(n);
        for i in 0..n {
            dc_y.push(ch_y_data[i] as f32 * factors.dc_quant[1]);
            dc_x.push(ch_x_data[i] as f32 * factors.dc_quant[0]);
            dc_b.push((ch_by_data[i] + ch_y_data[i]) as f32 * factors.dc_quant[2]);
        }
        [dc_x, dc_y, dc_b]
    };

    #[cfg(feature = "trace-bitstream")]
    {
        let y_min = ch_y_data.iter().copied().min().unwrap_or(0);
        let y_max = ch_y_data.iter().copied().max().unwrap_or(0);
        let x_min = ch_x_data.iter().copied().min().unwrap_or(0);
        let x_max = ch_x_data.iter().copied().max().unwrap_or(0);
        let by_min = ch_by_data.iter().copied().min().unwrap_or(0);
        let by_max = ch_by_data.iter().copied().max().unwrap_or(0);
        eprintln!("LFRAME: Y int range [{y_min}, {y_max}]");
        eprintln!("LFRAME: X int range [{x_min}, {x_max}]");
        eprintln!("LFRAME: B-Y int range [{by_min}, {by_max}]");
        // Check reconstruction: float_dc ≈ int * dc_quant
        if !ch_y_data.is_empty() {
            let y0_recon = ch_y_data[0] as f32 * factors.dc_quant[1];
            let x0_recon = ch_x_data[0] as f32 * factors.dc_quant[0];
            eprintln!(
                "LFRAME: first pixel: float_dc=[{:.6}, {:.6}, {:.6}]",
                float_dc[0][0], float_dc[1][0], float_dc[2][0]
            );
            eprintln!(
                "LFRAME: reconstructed: x={:.6}, y={:.6}",
                x0_recon, y0_recon
            );
        }
    }

    // Build LfFrame header
    let fh = FrameHeader::lf_frame(xsize_blocks as u32, ysize_blocks as u32, 1);
    fh.write(writer)?;

    // Build modular image with 3 channels [Y, X, B-Y]
    let mod_channels = vec![
        Channel::from_vec(ch_y_data, xsize_blocks, ysize_blocks)?,
        Channel::from_vec(ch_x_data, xsize_blocks, ysize_blocks)?,
        Channel::from_vec(ch_by_data, xsize_blocks, ysize_blocks)?,
    ];
    let image = ModularImage {
        channels: mod_channels,
        bit_depth: 16, // Fixed-point representation
        is_grayscale: false,
        has_alpha: false,
    };

    // Determine encoding parameters.
    // libjxl (enc_cache.cc:134-136) uses one speed_tier SLOWER (= more effort) for DC:
    //   speed_tier' = max(kTortoise, speed_tier - 1)
    // Lower speed_tier = more effort in libjxl. Our effort scale is reversed (higher = more).
    // So DC gets effort + 1, capped at 10.
    let lf_effort = (effort + 1).min(10);
    let mut profile =
        crate::effort::EffortProfile::lossless(lf_effort, crate::api::EncoderMode::Reference);
    // libjxl (enc_cache.cc:121) disables patches for DC frames.
    // Patch detection is wasteful on tiny DC images (32x32 to 128x128).
    profile.patches = false;
    // Disable LZ77 for DC frames — the DC image is small (typically 32-128px)
    // and smooth, making backward references ineffective. LZ77 header overhead
    // outweighs any savings on such small data.
    profile.lz77 = false;

    let num_groups_x = xsize_blocks.div_ceil(crate::GROUP_DIM);
    let num_groups_y = ysize_blocks.div_ceil(crate::GROUP_DIM);
    let num_groups = num_groups_x * num_groups_y;

    if num_groups == 1 {
        // Single group: use write_modular_stream_with_tree_dc_quant for
        // combined dc_quant + tree learning + modular encoding.
        let mut section_writer = BitWriter::new();

        crate::modular::encode::write_modular_stream_with_tree_dc_quant(
            &image,
            &mut section_writer,
            &profile,
            false, // no RCT (XYB integer channels)
            profile.lz77,
            profile.lz77_method,
            Some(factors.dc_quant),
            false, // no Squeeze for pre-quantized DC data
        )?;

        let section_data = section_writer.finish();

        // Write TOC (single entry, unpermuted)
        writer.write(1, 0)?; // permuted = false
        writer.zero_pad_to_byte();
        write_toc_entry(writer, section_data.len() as u32)?;
        writer.zero_pad_to_byte();

        // Write section data
        writer.append_bytes(&section_data)?;
    } else {
        // Multi-group: DC image larger than 256×256 blocks (original > 2048×2048).
        // Section layout: LfGlobal | LfGroup×N (empty) | HfGlobal (empty) | PassGroup×N
        encode_lf_frame_multi_group(
            &image,
            &factors,
            &profile,
            xsize_blocks,
            ysize_blocks,
            use_ans,
            writer,
        )?;
    }

    Ok(decoded_dc)
}

/// Multi-group LfFrame encoding.
///
/// For DC images larger than 256×256 blocks (original image > 2048×2048),
/// the DC frame itself needs multi-group encoding.
///
/// Section layout matches the standard JXL multi-group modular frame:
/// - Section 0: LfGlobal (dc_quant + tree + histogram)
/// - Sections 1..num_lf_groups: LfGroup (empty for modular)
/// - Section num_lf_groups+1: HfGlobal (empty for modular)
/// - Sections num_lf_groups+2..: PassGroup (modular data per group)
fn encode_lf_frame_multi_group(
    image: &ModularImage,
    factors: &DcQuantFactors,
    profile: &crate::effort::EffortProfile,
    xsize_blocks: usize,
    ysize_blocks: usize,
    _use_ans: bool,
    writer: &mut BitWriter,
) -> Result<()> {
    use crate::modular::encode::write_group_modular_section;
    use crate::modular::section::write_global_modular_section_with_tree_dc_quant;

    let num_groups_x = xsize_blocks.div_ceil(crate::GROUP_DIM);
    let num_groups_y = ysize_blocks.div_ceil(crate::GROUP_DIM);
    let num_groups = num_groups_x * num_groups_y;

    // LF groups are 8× larger than regular groups
    let lf_group_dim = crate::GROUP_DIM * 8;
    let num_lf_groups_x = xsize_blocks.div_ceil(lf_group_dim);
    let num_lf_groups_y = ysize_blocks.div_ceil(lf_group_dim);
    let num_lf_groups = num_lf_groups_x * num_lf_groups_y;

    let num_passes = 1;

    // Step 1: Extract group images
    let mut group_images: Vec<ModularImage> = Vec::with_capacity(num_groups);
    for group_idx in 0..num_groups {
        let gx = group_idx % num_groups_x;
        let gy = group_idx / num_groups_x;
        let x_start = gx * crate::GROUP_DIM;
        let y_start = gy * crate::GROUP_DIM;
        let x_end = (x_start + crate::GROUP_DIM).min(xsize_blocks);
        let y_end = (y_start + crate::GROUP_DIM).min(ysize_blocks);
        let group_image = image.extract_region(x_start, y_start, x_end, y_end)?;
        group_images.push(group_image);
    }

    // Step 2: Write LfGlobal section (custom dc_quant + tree + histogram)
    let mut lf_global_writer = BitWriter::new();
    let global_state = write_global_modular_section_with_tree_dc_quant(
        &group_images,
        &mut lf_global_writer,
        profile,
        None, // no RCT (XYB integer channels)
        false,
        profile.lz77_method,
        Some(factors.dc_quant),
    )?;
    let lf_global_data = lf_global_writer.finish();

    // Step 3: LfGroup sections (empty for modular encoding)
    let lf_group_data: Vec<Vec<u8>> = (0..num_lf_groups).map(|_| Vec::new()).collect();

    // Step 4: HfGlobal section (empty for modular encoding)
    let hf_global_data: Vec<u8> = Vec::new();

    // Step 5: Write PassGroup data
    let mut pass_group_data: Vec<Vec<u8>> = Vec::with_capacity(num_groups * num_passes);
    for group_image in &group_images {
        for _pass in 0..num_passes {
            let mut group_writer = BitWriter::new();
            write_group_modular_section(group_image, &global_state, &mut group_writer)?;
            pass_group_data.push(group_writer.finish());
        }
    }

    // Step 6: Collect section sizes in JXL order:
    // LfGlobal | LfGroup[0..N] | HfGlobal | PassGroup[0..M]
    let mut section_sizes = Vec::with_capacity(2 + num_lf_groups + num_groups * num_passes);
    section_sizes.push(lf_global_data.len());
    for data in &lf_group_data {
        section_sizes.push(data.len());
    }
    section_sizes.push(hf_global_data.len());
    for data in &pass_group_data {
        section_sizes.push(data.len());
    }

    // Step 7: Write TOC
    writer.write(1, 0)?; // permuted = false
    writer.zero_pad_to_byte();
    for &size in &section_sizes {
        write_toc_entry(writer, size as u32)?;
    }
    writer.zero_pad_to_byte();

    // Step 8: Write section data
    for byte in &lf_global_data {
        writer.write_u8(*byte)?;
    }
    for data in &lf_group_data {
        for byte in data {
            writer.write_u8(*byte)?;
        }
    }
    for byte in &hf_global_data {
        writer.write_u8(*byte)?;
    }
    for data in &pass_group_data {
        for byte in data {
            writer.write_u8(*byte)?;
        }
    }

    Ok(())
}

/// Write a single TOC entry using u2S encoding.
fn write_toc_entry(writer: &mut BitWriter, size: u32) -> Result<()> {
    // u2S(Bits(10), Bits(14)+1024, Bits(22)+17408, Bits(30)+4211712)
    if size < 1024 {
        writer.write(2, 0)?;
        writer.write(10, size as u64)?;
    } else if size < 17408 {
        writer.write(2, 1)?;
        writer.write(14, (size - 1024) as u64)?;
    } else if size < 4211712 {
        writer.write(2, 2)?;
        writer.write(22, (size - 17408) as u64)?;
    } else {
        writer.write(2, 3)?;
        writer.write(30, (size - 4211712) as u64)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dc_quant_factors_d1() {
        let f = DcQuantFactors::compute(1.0);
        // At d=1.0: dc_distance = 0.02
        // enc_factors[0] = 65536 / (1 + 23*0.02) = 65536 / 1.46 ≈ 44887
        // enc_factors[1] = 4096 / (1 + 14*0.02) = 4096 / 1.28 = 3200
        assert!(f.inv_dc_quant[0] > 40000.0 && f.inv_dc_quant[0] < 50000.0);
        assert!(f.inv_dc_quant[1] > 3000.0 && f.inv_dc_quant[1] < 3500.0);
        assert_eq!(f.inv_dc_quant[1], f.inv_dc_quant[2]); // Y and B use same k=14
    }

    #[test]
    fn test_dc_quant_factors_d0_5() {
        let f = DcQuantFactors::compute(0.5);
        // Lower distance → larger enc_factors (less quantization)
        let f1 = DcQuantFactors::compute(1.0);
        assert!(f.inv_dc_quant[0] > f1.inv_dc_quant[0]);
        assert!(f.inv_dc_quant[1] > f1.inv_dc_quant[1]);
    }

    #[test]
    fn test_dc_quant_f16_roundtrip() {
        let f = DcQuantFactors::compute(1.0);
        // dc_quant values should survive F16 roundtrip
        for c in 0..3 {
            let rt = f16_roundtrip(f.dc_quant[c] * 128.0) / 128.0;
            assert_eq!(rt, f.dc_quant[c], "channel {c}");
        }
    }

    #[test]
    fn test_round_hafz() {
        assert_eq!(round_hafz(0.5), 1);
        assert_eq!(round_hafz(-0.5), -1);
        assert_eq!(round_hafz(0.4), 0);
        assert_eq!(round_hafz(-0.4), 0);
        assert_eq!(round_hafz(1.5), 2);
        assert_eq!(round_hafz(-1.5), -2);
    }
}
