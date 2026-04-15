// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Frame header writing for the tiny encoder.

#[cfg(feature = "jpeg-reencoding")]
use super::ac_strategy::AcStrategyMap;
use super::common::clamp;
#[cfg(feature = "jpeg-reencoding")]
use super::common::{DC_GROUP_DIM_IN_BLOCKS, ceil_log2_nonzero};
use crate::bit_writer::BitWriter;
#[cfg(feature = "debug-tokens")]
use crate::debug_log;
#[cfg(feature = "jpeg-reencoding")]
use crate::entropy_coding::token::Token;
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
    /// Compute all pixel stats from XYB image.
    ///
    /// Serial path: single-pass with early exit once all thresholds saturate.
    /// Parallel path: row strips processed independently, results max-reduced.
    /// The parallel path skips early-exit — saturated-early images become
    /// slightly slower, but these are rare and the phase is <1% of encode
    /// time anyway. Max-reduction is associative/commutative for finite f32,
    /// so the output is bit-exact regardless of strip count.
    pub(crate) fn calc(
        xyb_x: &[f32],
        xyb_y: &[f32],
        xyb_b: &[f32],
        width: usize,
        height: usize,
    ) -> Self {
        // Thresholds from how_much_is_x_channel_pixelized / how_much_is_b_channel_pixelized
        const DX_MAX_THRESH: f32 = 0.026;
        const DB_MAX_THRESH: f32 = 0.38;
        const EB_THRESH: f32 = 0.13;

        // Rows 1..height only (row 0 is skipped because the inner loop needs a prev row).
        if height < 2 {
            return Self {
                dx: 0.0,
                db: 0.0,
                exposed_blue: 0.0,
            };
        }

        // Each strip processes a range of ROW INDICES ty_start..ty_end (never 0).
        // Requires xyb_*[(ty_start - 1) * width ..] readable, so chunk the input
        // index space to `row_span = ty_end - ty_start` rows with 1 row of back-overlap.
        let calc_strip = |ty_start: usize, ty_end: usize| -> (f32, f32, f32) {
            let mut dx: f32 = 0.0;
            let mut db: f32 = 0.0;
            let mut exposed_blue: f32 = 0.0;
            for ty in ty_start..ty_end {
                let x_row = &xyb_x[ty * width..(ty + 1) * width];
                let y_row = &xyb_y[ty * width..(ty + 1) * width];
                let b_row = &xyb_b[ty * width..(ty + 1) * width];
                let x_prev_row = &xyb_x[(ty - 1) * width..ty * width];
                let y_prev_row = &xyb_y[(ty - 1) * width..ty * width];
                let b_prev_row = &xyb_b[(ty - 1) * width..ty * width];
                for tx in 1..width {
                    let cur_x = x_row[tx];
                    dx = dx
                        .max((cur_x - x_row[tx - 1]).abs())
                        .max((cur_x - x_prev_row[tx]).abs());
                    let cur_y = y_row[tx];
                    let cur_b = b_row[tx];
                    let diff_b = cur_b - cur_y;
                    let diff_prev = b_row[tx - 1] - y_row[tx - 1];
                    let diff_prev_row = b_prev_row[tx] - y_prev_row[tx];
                    db = db
                        .max((diff_b - diff_prev).abs())
                        .max((diff_b - diff_prev_row).abs());
                    let exposed_b = cur_b - cur_y * 1.2;
                    if exposed_b >= 0.0 {
                        let eb_val = exposed_b
                            * ((cur_b - b_row[tx - 1]).abs() + (cur_b - b_prev_row[tx]).abs());
                        exposed_blue = exposed_blue.max(eb_val);
                    }
                }
            }
            (dx, db, exposed_blue)
        };

        #[cfg(feature = "parallel")]
        {
            // Only parallelize for images tall enough to amortize task overhead.
            // Below 256 rows, the serial early-exit path is almost certainly faster.
            const PAR_MIN_ROWS: usize = 256;
            if height >= PAR_MIN_ROWS {
                const STRIP_ROWS: usize = 64;
                let total = height - 1; // rows 1..height
                let n_strips = total.div_ceil(STRIP_ROWS);
                let results: Vec<(f32, f32, f32)> = crate::parallel::parallel_map(n_strips, |s| {
                    let start = 1 + s * STRIP_ROWS;
                    let end = (start + STRIP_ROWS).min(height);
                    calc_strip(start, end)
                });
                let (dx, db, exposed_blue) =
                    results.into_iter().fold((0.0f32, 0.0f32, 0.0f32), |a, b| {
                        (a.0.max(b.0), a.1.max(b.1), a.2.max(b.2))
                    });
                return Self {
                    dx,
                    db,
                    exposed_blue,
                };
            }
        }

        // Serial path with early exit (short images or parallel disabled).
        let mut dx: f32 = 0.0;
        let mut db: f32 = 0.0;
        let mut exposed_blue: f32 = 0.0;
        for ty in 1..height {
            let (sdx, sdb, seb) = calc_strip(ty, ty + 1);
            dx = dx.max(sdx);
            db = db.max(sdb);
            exposed_blue = exposed_blue.max(seb);
            if dx >= DX_MAX_THRESH && db > DB_MAX_THRESH && exposed_blue >= EB_THRESH {
                break;
            }
        }
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

    let effective_dist = DC_MUL * jxl_simd::fast_powf(distance / DC_MUL, DC_QUANT_POW);
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

    /// Compute distance-dependent parameters from the effort profile.
    ///
    /// Uses `profile.initial_q_numerator` to determine the global_scale:
    /// - Effort < 5 (speed_tier > kHare): q = 0.79 / distance
    /// - Effort >= 5 (speed_tier <= kHare): q = 0.39 / distance
    ///
    /// The adaptive median/MAD formula is ONLY used inside the butteraugli
    /// quantization loop (effort >= 8), where SetQuantField recomputes
    /// global_scale after each iteration.
    pub fn compute_for_profile(distance: f32, profile: &crate::effort::EffortProfile) -> Self {
        let q = profile.initial_q_numerator / distance;
        Self::compute_from_q(distance, q)
    }

    /// Compute distance-dependent parameters using content-adaptive global_scale.
    ///
    /// This matches full libjxl's SetQuantField behavior: global_scale is derived
    /// from the median and MAD (median absolute deviation) of the quant field.
    /// For high-variance images, MAD is large, so (median - MAD) is smaller,
    /// giving a smaller global_scale (finer quantization, better quality).
    ///
    /// NOTE: In libjxl, this is ONLY called inside the butteraugli quantization
    /// loop (effort >= 8). At effort 5-7, global_scale uses the fixed formula
    /// from `compute_for_profile()`. Use this method only for butteraugli loop
    /// refinement.
    #[allow(dead_code)]
    pub fn compute_from_quant_field(distance: f32, quant_field: &[f32]) -> Self {
        if quant_field.is_empty() {
            return Self::compute(distance);
        }

        // Compute median using nth_element equivalent (partial sort)
        let mut data: Vec<f32> = quant_field.to_vec();
        let mid = data.len() / 2;
        data.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
        let quant_median = data[mid];

        // Compute median absolute deviation from median
        let mut deviations: Vec<f32> = data.iter().map(|&x| (x - quant_median).abs()).collect();
        deviations.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
        let quant_median_absd = deviations[mid];

        #[cfg(feature = "debug-tokens")]
        eprintln!(
            "[adaptive] d={:.2} median={:.4} mad={:.4} (median-mad)={:.4}",
            distance,
            quant_median,
            quant_median_absd,
            quant_median - quant_median_absd
        );
        Self::compute_from_q(distance, quant_median - quant_median_absd)
    }

    /// Compute distance-dependent parameters from a given `q` value.
    ///
    /// The `q` parameter determines global_scale: `global_scale = 65536 * q / 5.0`.
    /// This is the core formula from libjxl quantizer.cc:ComputeGlobalScaleAndQuant
    /// with quant_median_absd=0 (i.e. `q = quant_median - 0 = quant_median`).
    fn compute_from_q(distance: f32, q: f32) -> Self {
        Self::compute_internal(distance, Some(q))
    }

    /// Internal implementation shared by all compute methods.
    fn compute_internal(distance: f32, q_for_global_scale: Option<f32>) -> Self {
        const GLOBAL_SCALE_DENOM: i32 = 1 << 16;
        const GLOBAL_SCALE_NUMERATOR: i32 = 4096;
        const AC_QUANT: f32 = 0.765;
        const QUANT_FIELD_TARGET: f32 = 5.0;

        let qdc = quant_dc(distance);

        // Compute global_scale from the q parameter.
        // libjxl's ComputeGlobalScaleAndQuant: scale = kGlobalScaleDenom * q / kQuantFieldTarget
        // where q comes from:
        // - Fixed formula: 0.39/d (effort >= 5) or 0.79/d (effort < 5)
        // - Adaptive: (median - MAD) of quant field (butteraugli loop only)
        // - Fallback: kAcQuant / distance (libjxl-tiny compat)
        let scale = if let Some(q) = q_for_global_scale {
            (GLOBAL_SCALE_DENOM as f32) * q / QUANT_FIELD_TARGET
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
            let mode = if q_for_global_scale.is_some() {
                "q-based"
            } else {
                "fallback"
            };
            eprintln!(
                "[global_scale] d={:.2} mode={} global_scale={} inv_scale={:.4}",
                distance, mode, global_scale, inv_scale
            );
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

/// Write a DC group section from pre-collected tokens.
///
/// Writes the DC group header, DC tokens, AC metadata sub-header, then AC
/// metadata tokens. Used by both normal VarDCT and JPEG reencoding paths.
///
/// The `write_tokens` closure handles the actual entropy-coded token writing,
/// allowing callers to use either `BuiltEntropyCode` or `OwnedAnsEntropyCode`.
#[cfg(feature = "jpeg-reencoding")]
#[allow(clippy::too_many_arguments)]
pub fn write_dc_group_from_tokens(
    dc_group_idx: usize,
    xsize_blocks: usize,
    ysize_blocks: usize,
    xsize_dc_groups: usize,
    dc_tokens: &[Token],
    ac_metadata_tokens: &[Token],
    ac_strategy: &AcStrategyMap,
    write_tokens: &dyn Fn(&[Token], &mut BitWriter) -> Result<()>,
    writer: &mut BitWriter,
) -> Result<()> {
    let dc_gx = dc_group_idx % xsize_dc_groups;
    let dc_gy = dc_group_idx / xsize_dc_groups;
    let start_bx = dc_gx * DC_GROUP_DIM_IN_BLOCKS;
    let start_by = dc_gy * DC_GROUP_DIM_IN_BLOCKS;
    let end_bx = (start_bx + DC_GROUP_DIM_IN_BLOCKS).min(xsize_blocks);
    let end_by = (start_by + DC_GROUP_DIM_IN_BLOCKS).min(ysize_blocks);
    let region_xsize = end_bx - start_bx;
    let region_ysize = end_by - start_by;

    // DC group header
    writer.write(2, 0)?; // extra_dc_precision = 0
    writer.write(4, 3)?; // use global tree, default wp, no transforms

    // Write DC tokens
    write_tokens(dc_tokens, writer)?;

    // AC metadata sub-header — count first blocks (distinct transforms)
    let num_blocks = region_xsize * region_ysize;
    let mut num_ac_blocks = 0;
    for ry in start_by..end_by {
        for rx in start_bx..end_bx {
            if ac_strategy.is_first(rx, ry) {
                num_ac_blocks += 1;
            }
        }
    }
    let nb_bits = ceil_log2_nonzero(num_blocks);
    if nb_bits != 0 {
        writer.write(nb_bits as usize, (num_ac_blocks - 1) as u64)?;
    }
    writer.write(4, 3)?; // use global tree, default wp, no transforms

    // Write AC metadata tokens
    write_tokens(ac_metadata_tokens, writer)?;

    Ok(())
}

/// Assemble VarDCT frame sections into the output bitstream.
///
/// Handles both single-group (bit-level combination) and multi-group (byte-aligned
/// sections with TOC) assembly. This is shared by both the normal VarDCT encoder
/// and the JPEG reencoding path.
///
/// Section order: DC global, DC groups, AC global, AC groups (per JXL spec).
#[cfg(feature = "jpeg-reencoding")]
pub fn assemble_frame_sections(
    dc_global: BitWriter,
    dc_groups: Vec<BitWriter>,
    ac_global: BitWriter,
    ac_groups: Vec<BitWriter>,
    writer: &mut BitWriter,
) -> Result<()> {
    let num_dc_groups = dc_groups.len();
    let num_ac_groups = ac_groups.len();
    let num_sections = 2 + num_dc_groups + num_ac_groups;

    if num_sections == 4 {
        // Single-group: combine all sections at the bit level (no byte alignment between them)
        let mut combined = dc_global;
        combined.append_unaligned(&dc_groups[0])?;
        combined.append_unaligned(&ac_global)?;
        combined.append_unaligned(&ac_groups[0])?;
        combined.zero_pad_to_byte();
        let combined_bytes = combined.finish();

        write_toc(&[combined_bytes.len()], writer)?;
        writer.append_bytes(&combined_bytes)?;
    } else {
        // Multi-group: each section is independently byte-aligned
        let mut sections: Vec<Vec<u8>> = Vec::with_capacity(num_sections);

        // DC Global
        let mut dc_global = dc_global;
        dc_global.zero_pad_to_byte();
        sections.push(dc_global.finish());

        // DC Groups
        for mut dc_group in dc_groups {
            dc_group.zero_pad_to_byte();
            sections.push(dc_group.finish());
        }

        // AC Global
        let mut ac_global = ac_global;
        ac_global.zero_pad_to_byte();
        sections.push(ac_global.finish());

        // AC Groups
        for mut ac_group in ac_groups {
            ac_group.zero_pad_to_byte();
            sections.push(ac_group.finish());
        }

        let section_sizes: Vec<usize> = sections.iter().map(|s| s.len()).collect();
        write_toc(&section_sizes, writer)?;
        for section in sections {
            writer.append_bytes(&section)?;
        }
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
