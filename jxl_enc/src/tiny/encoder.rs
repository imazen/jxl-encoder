// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Main tiny encoder implementation.

use super::ac_group::{
    AC_STRATEGY_DCT8, num_nonzero_8x8_except_dc, predict_from_top_and_left,
    tokenize_ac_coefficients,
};
use super::common::*;
use super::dc_coding::write_dc_tokens;
use super::dct::dct_8x8;
use super::frame::{DistanceParams, write_frame_header, write_quant_scales, write_toc};
use super::quant::{DC_QUANT, QUANT_WEIGHTS};
use super::static_codes::{get_ac_entropy_code, get_dc_entropy_code};
use crate::bit_writer::BitWriter;
use crate::color::xyb::linear_rgb_to_xyb;
use crate::error::Result;

/// Tiny JPEG XL encoder.
///
/// This is a simplified VarDCT encoder based on libjxl-tiny that uses:
/// - Only DCT8, DCT8x16, DCT16x8 transforms
/// - Only Huffman entropy coding
/// - Default zig-zag coefficient order
/// - Fixed context tree for DC
pub struct TinyEncoder {
    /// Target distance (quality). 1.0 = visually lossless.
    pub distance: f32,
}

impl Default for TinyEncoder {
    fn default() -> Self {
        Self { distance: 1.0 }
    }
}

impl TinyEncoder {
    /// Create a new tiny encoder with the given distance.
    pub fn new(distance: f32) -> Self {
        Self { distance }
    }

    /// Encode an image in linear sRGB format.
    ///
    /// Input should be 3 channels (RGB) of f32 values in [0, 1] range.
    /// Values outside [0, 1] are allowed for out-of-gamut colors.
    pub fn encode(&self, width: usize, height: usize, linear_rgb: &[f32]) -> Result<Vec<u8>> {
        assert_eq!(linear_rgb.len(), width * height * 3);

        // Compute distance parameters
        let params = DistanceParams::compute(self.distance);

        // Calculate dimensions
        let xsize_blocks = div_ceil(width, BLOCK_DIM);
        let ysize_blocks = div_ceil(height, BLOCK_DIM);
        let xsize_groups = div_ceil(width, GROUP_DIM);
        let ysize_groups = div_ceil(height, GROUP_DIM);
        let xsize_dc_groups = div_ceil(width, DC_GROUP_DIM);
        let ysize_dc_groups = div_ceil(height, DC_GROUP_DIM);
        let num_groups = xsize_groups * ysize_groups;
        let num_dc_groups = xsize_dc_groups * ysize_dc_groups;

        // Number of sections: DC global + DC groups + AC global + AC groups
        let num_sections = 2 + num_dc_groups + num_groups;

        // Convert to XYB
        let (xyb_x, xyb_y, xyb_b) = self.convert_to_xyb(width, height, linear_rgb);

        // Pad to block boundary
        let padded_width = xsize_blocks * BLOCK_DIM;
        let padded_height = ysize_blocks * BLOCK_DIM;

        // Perform DCT and quantization
        let (quant_dc, quant_ac, nzeros) = self.transform_and_quantize(
            &xyb_x,
            &xyb_y,
            &xyb_b,
            width,
            height,
            padded_width,
            padded_height,
            xsize_blocks,
            ysize_blocks,
            &params,
        );

        // Get static entropy codes
        let dc_code = get_dc_entropy_code();
        let ac_code = get_ac_entropy_code();

        // Create main writer
        let mut writer = BitWriter::with_capacity(width * height * 4);

        // Write JXL signature
        writer.write(8, 0xFF)?;
        writer.write(8, 0x0A)?;
        #[cfg(test)]
        eprintln!("After signature: bit {}", writer.bits_written());

        // Write size header (simple format for small images)
        self.write_file_header(width, height, &mut writer)?;
        #[cfg(test)]
        eprintln!("After file header: bit {} (byte {})", writer.bits_written(), writer.bits_written() / 8);

        // Write frame header
        write_frame_header(params.x_qm_scale, params.epf_iters, &mut writer)?;
        #[cfg(test)]
        eprintln!("After frame header: bit {} (byte {})", writer.bits_written(), writer.bits_written() / 8);

        // Create section writers
        let mut sections: Vec<Vec<u8>> = Vec::with_capacity(num_sections);

        // DC Global section
        let mut dc_global = BitWriter::new();
        self.write_dc_global(&params, num_dc_groups, &dc_code, &mut dc_global)?;
        dc_global.zero_pad_to_byte();
        sections.push(dc_global.finish());

        // DC group sections
        for dc_group_idx in 0..num_dc_groups {
            let mut dc_group = BitWriter::new();
            self.write_dc_group(
                dc_group_idx,
                &quant_dc,
                xsize_blocks,
                ysize_blocks,
                &dc_code,
                &mut dc_group,
            )?;
            dc_group.zero_pad_to_byte();
            sections.push(dc_group.finish());
        }

        // AC Global section
        let mut ac_global = BitWriter::new();
        self.write_ac_global(num_groups, &ac_code, &mut ac_global)?;
        ac_global.zero_pad_to_byte();
        sections.push(ac_global.finish());

        // AC group sections
        for group_idx in 0..num_groups {
            let mut ac_group_writer = BitWriter::new();
            self.write_ac_group(
                group_idx,
                &quant_ac,
                &nzeros,
                xsize_blocks,
                ysize_blocks,
                &ac_code,
                &mut ac_group_writer,
            )?;
            ac_group_writer.zero_pad_to_byte();
            sections.push(ac_group_writer.finish());
        }

        // Write TOC
        let section_sizes: Vec<usize> = sections.iter().map(|s| s.len()).collect();

        // Debug: print section sizes
        #[cfg(test)]
        eprintln!(
            "Section sizes: DC_global={}, DC_group={}, AC_global={}, AC_group={}",
            section_sizes.get(0).unwrap_or(&0),
            section_sizes.get(1).unwrap_or(&0),
            section_sizes.get(2).unwrap_or(&0),
            section_sizes.get(3).unwrap_or(&0),
        );

        // For single-group images, combine all sections into one
        if sections.len() == 4 {
            let mut combined = sections.remove(0);
            for section in sections {
                combined.extend(section);
            }
            #[cfg(test)]
            {
                eprintln!("Combined section size: {}", combined.len());
                eprintln!("Before TOC: bit {} (byte {})", writer.bits_written(), writer.bits_written() / 8);
            }
            write_toc(&[combined.len()], &mut writer)?;
            #[cfg(test)]
            eprintln!("After TOC: bit {} (byte {})", writer.bits_written(), writer.bits_written() / 8);
            writer.append_bytes(&combined)?;
        } else {
            write_toc(&section_sizes, &mut writer)?;
            for section in sections {
                writer.append_bytes(&section)?;
            }
        }

        Ok(writer.finish_with_padding())
    }

    /// Convert linear RGB to XYB color space.
    fn convert_to_xyb(
        &self,
        width: usize,
        height: usize,
        linear_rgb: &[f32],
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let n = width * height;
        let mut xyb_x = vec![0.0f32; n];
        let mut xyb_y = vec![0.0f32; n];
        let mut xyb_b = vec![0.0f32; n];

        for i in 0..n {
            let r = linear_rgb[i * 3];
            let g = linear_rgb[i * 3 + 1];
            let b = linear_rgb[i * 3 + 2];
            let (x, y, b_out) = linear_rgb_to_xyb(r, g, b);
            xyb_x[i] = x;
            xyb_y[i] = y;
            xyb_b[i] = b_out;
        }

        (xyb_x, xyb_y, xyb_b)
    }

    /// Perform DCT and quantization on all blocks.
    ///
    /// Returns (quantized_dc, quantized_ac, nzeros)
    fn transform_and_quantize(
        &self,
        xyb_x: &[f32],
        xyb_y: &[f32],
        xyb_b: &[f32],
        width: usize,
        height: usize,
        padded_width: usize,
        padded_height: usize,
        xsize_blocks: usize,
        ysize_blocks: usize,
        params: &DistanceParams,
    ) -> (
        [Vec<Vec<i16>>; 3],
        [Vec<Vec<[i32; DCT_BLOCK_SIZE]>>; 3],
        [Vec<Vec<u8>>; 3],
    ) {
        // Initialize output arrays
        let mut quant_dc: [Vec<Vec<i16>>; 3] = [
            vec![vec![0i16; xsize_blocks]; ysize_blocks],
            vec![vec![0i16; xsize_blocks]; ysize_blocks],
            vec![vec![0i16; xsize_blocks]; ysize_blocks],
        ];

        let mut quant_ac: [Vec<Vec<[i32; DCT_BLOCK_SIZE]>>; 3] = [
            vec![vec![[0i32; DCT_BLOCK_SIZE]; xsize_blocks]; ysize_blocks],
            vec![vec![[0i32; DCT_BLOCK_SIZE]; xsize_blocks]; ysize_blocks],
            vec![vec![[0i32; DCT_BLOCK_SIZE]; xsize_blocks]; ysize_blocks],
        ];

        let mut nzeros: [Vec<Vec<u8>>; 3] = [
            vec![vec![0u8; xsize_blocks]; ysize_blocks],
            vec![vec![0u8; xsize_blocks]; ysize_blocks],
            vec![vec![0u8; xsize_blocks]; ysize_blocks],
        ];

        // Channel data
        let channels = [xyb_x, xyb_y, xyb_b];

        // Process each block
        for by in 0..ysize_blocks {
            for bx in 0..xsize_blocks {
                for c in 0..3 {
                    // Extract 8x8 block with edge padding (flat array)
                    let mut block = [0.0f32; DCT_BLOCK_SIZE];
                    for dy in 0..BLOCK_DIM {
                        for dx in 0..BLOCK_DIM {
                            let py = (by * BLOCK_DIM + dy).min(height - 1);
                            let px = (bx * BLOCK_DIM + dx).min(width - 1);
                            block[dy * BLOCK_DIM + dx] = channels[c][py * width + px];
                        }
                    }

                    // Perform DCT
                    let mut dct_block = [0.0f32; DCT_BLOCK_SIZE];
                    dct_8x8(&block, &mut dct_block);

                    // Quantize DC
                    let dc = dct_block[0];
                    let dc_scale = DC_QUANT[c] * params.scale;
                    let qdc = (dc * dc_scale).round() as i16;
                    quant_dc[c][by][bx] = qdc;

                    // Quantize AC coefficients
                    let ac_scale = params.scale;
                    let weights = &QUANT_WEIGHTS[..DCT_BLOCK_SIZE]; // DCT8 uses first 64 weights

                    let mut qblock = [0i32; DCT_BLOCK_SIZE];
                    for idx in 0..DCT_BLOCK_SIZE {
                        if idx == 0 {
                            // DC is handled separately
                            qblock[0] = 0;
                        } else {
                            let coef = dct_block[idx];
                            let weight = weights[idx];
                            let qval = (coef * ac_scale / weight).round() as i32;
                            qblock[idx] = qval;
                        }
                    }
                    quant_ac[c][by][bx] = qblock;

                    // Count non-zeros
                    let _nz = num_nonzero_8x8_except_dc(&qblock, &mut nzeros[c][by][bx]);
                }
            }
        }

        (quant_dc, quant_ac, nzeros)
    }

    /// Write the file header (SizeHeader + ImageMetadata).
    ///
    /// Follows libjxl-tiny's enc_file.cc exactly:
    /// 1. SizeHeader with small=0 (U32 encoding)
    /// 2. ImageMetadata with float samples (32-bit, 8 exp bits)
    /// 3. ColorEncoding with sRGB primaries, Linear transfer
    /// 4. all_default_transform_data = 1
    /// 5. Zero padding to byte
    fn write_file_header(&self, width: usize, height: usize, writer: &mut BitWriter) -> Result<()> {
        // 1. SizeHeader - use U32 format (small=0), same as libjxl-tiny
        self.write_size(width, height, writer)?;

        // 2. ImageMetadata
        writer.write(1, 0)?; // not all default
        writer.write(1, 0)?; // no extra fields

        // Bit depth - 32-bit float with 8 exponent bits (libjxl-tiny format)
        writer.write(1, 1)?; // float = 1
        writer.write(2, 0)?; // bits_per_sample selector 0 = 32 bits
        writer.write(4, 7)?; // exp_bits = 8 (encoded as 7+1)

        writer.write(1, 0)?; // modular 16 bit buffer NOT sufficient (for float)

        // Extra channels - none
        writer.write(2, 0)?; // selector 0 = 0 extra channels

        // xyb_encoded = 1 (required for VarDCT)
        writer.write(1, 1)?;

        // Color encoding - sRGB primaries/white, Linear transfer
        writer.write(1, 0)?; // not all default
        writer.write(1, 0)?; // no ICC profile
        writer.write(2, 0)?; // color_space = RGB (0)
        writer.write(2, 1)?; // white_point = D65 (1)
        writer.write(2, 1)?; // primaries = sRGB (1)
        writer.write(1, 0)?; // no gamma (use transfer function)
        // TransferFunction: U32(0, 1, 2+Read(4), 18+Read(6))
        // For Linear (value 8): selector=2, extra=6 (8 = 2 + 6)
        writer.write(2, 2)?; // selector 2
        writer.write(4, 6)?; // value 6 -> transfer_function = 2+6 = 8 = Linear
        writer.write(2, 1)?; // rendering_intent = relative (1)

        // Extensions
        writer.write(2, 0)?; // no extensions

        // 3. all_default_transform_data = 1 (required before frame)
        writer.write(1, 1)?;

        // 4. Zero pad to byte before frame
        writer.zero_pad_to_byte();

        Ok(())
    }

    /// Write image size header.
    ///
    /// Uses U32 format (small=0) to match libjxl-tiny exactly.
    /// Format: small=0, height U32, ratio=0, width U32
    fn write_size(&self, width: usize, height: usize, writer: &mut BitWriter) -> Result<()> {
        // Helper to write a dimension using U32 encoding
        // Matches libjxl-tiny's WriteSize() exactly
        fn write_dim(size: usize, writer: &mut BitWriter) -> Result<()> {
            let size_m1 = (size.saturating_sub(1)) as u32;
            // U32 selectors: 9 bits, 13 bits, 18 bits, 30 bits
            // Select first one where value fits
            let k_bits: [u32; 4] = [9, 13, 18, 30];
            for (i, &bits) in k_bits.iter().enumerate() {
                if size_m1 < (1u32 << bits) {
                    writer.write(2, i as u64)?;
                    writer.write(bits as usize, size_m1 as u64)?;
                    return Ok(());
                }
            }
            // Shouldn't reach here for valid sizes
            Ok(())
        }

        // small = 0 (use U32 encoding)
        writer.write(1, 0)?;
        write_dim(height, writer)?;
        writer.write(3, 0)?; // ratio = 0 (explicit width)
        write_dim(width, writer)?;

        Ok(())
    }

    /// Write DC global section.
    ///
    /// This follows the libjxl-tiny WriteDCGlobal pattern:
    /// 1. Default dequant DC
    /// 2. Quant scales
    /// 3. Non-default BlockCtxMap + compact block context map
    /// 4. Default DC cmap
    /// 5. Context tree for modular stream
    /// 6. No LZ77
    /// 7. DC entropy code
    fn write_dc_global(
        &self,
        params: &DistanceParams,
        num_dc_groups: usize,
        dc_code: &super::entropy_code::EntropyCode,
        writer: &mut BitWriter,
    ) -> Result<()> {
        #[cfg(test)]
        let start_bits = writer.bits_written();

        writer.write(1, 1)?; // default dequant dc

        #[cfg(test)]
        let after_dequant_dc = writer.bits_written();

        write_quant_scales(params.global_scale, params.quant_dc, writer)?;

        #[cfg(test)]
        let after_quant = writer.bits_written();

        // BlockCtxMap - non-default, write compact map
        writer.write(1, 0)?; // non-default BlockCtxMap
        writer.write(16, 0)?; // no dc ctx, no qft

        // Write compact block context map
        super::context_tree::write_block_context_map(writer)?;

        #[cfg(test)]
        let after_block_ctx = writer.bits_written();

        writer.write(1, 1)?; // default DC cmap

        // Write context tree for modular stream DC header
        super::context_tree::write_context_tree(num_dc_groups, writer)?;

        #[cfg(test)]
        let after_ctx_tree = writer.bits_written();

        writer.write(1, 0)?; // no lz77

        #[cfg(test)]
        let after_lz77 = writer.bits_written();

        // Write DC entropy code
        self.write_entropy_code_header(dc_code, writer)?;

        #[cfg(test)]
        {
            let after_dc_code = writer.bits_written();
            let total_bits = after_dc_code - start_bits;
            let bytes_before_pad = (total_bits + 7) / 8;
            eprintln!("DC_global detailed breakdown:");
            eprintln!("  dequant_dc: {} bits (1)", after_dequant_dc - start_bits);
            eprintln!("  quant_scales: {} bits", after_quant - after_dequant_dc);
            eprintln!("  block_ctx_map: {} bits (1+16+map)", after_block_ctx - after_quant);
            eprintln!("  dc_cmap: 1 bit (default=1)");
            eprintln!("  context_tree: {} bits", after_ctx_tree - after_block_ctx - 1);
            eprintln!("  lz77: 1 bit (no=0)");
            eprintln!("  dc_entropy_code: {} bits", after_dc_code - after_lz77);
            eprintln!("  total bits: {}, bytes before pad: {}", total_bits, bytes_before_pad);
        }

        Ok(())
    }

    /// Write DC group section.
    fn write_dc_group(
        &self,
        _dc_group_idx: usize,
        quant_dc: &[Vec<Vec<i16>>; 3],
        _xsize_blocks: usize,
        _ysize_blocks: usize,
        dc_code: &super::entropy_code::EntropyCode,
        writer: &mut BitWriter,
    ) -> Result<()> {
        // DC group header
        writer.write(2, 0)?; // extra_dc_precision = 0
        writer.write(4, 3)?; // use global tree, default wp, no transforms

        // Write DC tokens using gradient predictor
        write_dc_tokens(quant_dc, dc_code, writer)?;

        // TODO: Write AC metadata (YtoX, YtoB, AC strategy, quant field, EPF)
        // For now, we'll write minimal metadata

        Ok(())
    }

    /// Write AC global section.
    fn write_ac_global(
        &self,
        num_groups: usize,
        ac_code: &super::entropy_code::EntropyCode,
        writer: &mut BitWriter,
    ) -> Result<()> {
        #[cfg(test)]
        let start_bits = writer.bits_written();

        writer.write(1, 1)?; // all default quant matrices

        let num_histo_bits = ceil_log2_nonzero(num_groups);
        if num_histo_bits != 0 {
            writer.write(num_histo_bits as usize, 0)?;
        }

        writer.write(2, 3)?;
        writer.write(13, 0)?; // all default coeff order

        writer.write(1, 0)?; // no lz77

        #[cfg(test)]
        let before_ac_code = writer.bits_written();

        // Write entropy code
        self.write_entropy_code_header(ac_code, writer)?;

        #[cfg(test)]
        {
            let after_ac_code = writer.bits_written();
            eprintln!("AC_global breakdown:");
            eprintln!("  header: {} bits", before_ac_code - start_bits);
            eprintln!("  ac_entropy_code: {} bits ({} contexts, {} prefix codes)",
                      after_ac_code - before_ac_code,
                      ac_code.num_contexts,
                      ac_code.num_prefix_codes);
        }

        Ok(())
    }

    /// Write AC group section.
    fn write_ac_group(
        &self,
        _group_idx: usize,
        quant_ac: &[Vec<Vec<[i32; DCT_BLOCK_SIZE]>>; 3],
        nzeros: &[Vec<Vec<u8>>; 3],
        xsize_blocks: usize,
        ysize_blocks: usize,
        ac_code: &super::entropy_code::EntropyCode,
        writer: &mut BitWriter,
    ) -> Result<()> {
        // Process blocks in channel order: Y (1), X (0), B (2)
        for &c in &[1usize, 0, 2] {
            for by in 0..ysize_blocks {
                for bx in 0..xsize_blocks {
                    let block = &quant_ac[c][by][bx];
                    let nz = nzeros[c][by][bx];

                    // Predict nzeros from neighbors
                    let row_top = if by > 0 {
                        Some(nzeros[c][by - 1].as_slice())
                    } else {
                        None
                    };
                    let predicted_nz = predict_from_top_and_left(row_top, &nzeros[c][by], bx, 32);

                    // Tokenize AC coefficients
                    tokenize_ac_coefficients(
                        block,
                        c,
                        AC_STRATEGY_DCT8,
                        nz,
                        predicted_nz,
                        ac_code,
                        writer,
                    )?;
                }
            }
        }

        Ok(())
    }

    /// Write entropy code (context map + prefix codes).
    fn write_entropy_code_header(
        &self,
        code: &super::entropy_code::EntropyCode,
        writer: &mut BitWriter,
    ) -> Result<()> {
        super::entropy_code::write_entropy_code(code, writer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_creation() {
        let encoder = TinyEncoder::new(1.0);
        assert_eq!(encoder.distance, 1.0);

        let encoder_default = TinyEncoder::default();
        assert_eq!(encoder_default.distance, 1.0);
    }

    #[test]
    fn test_encode_small_image() {
        let encoder = TinyEncoder::new(1.0);

        // Create a simple 8x8 red image
        let width = 8;
        let height = 8;
        let mut linear_rgb = vec![0.0f32; width * height * 3];
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) * 3;
                linear_rgb[idx] = 1.0; // R
                linear_rgb[idx + 1] = 0.0; // G
                linear_rgb[idx + 2] = 0.0; // B
            }
        }

        // This should at least not panic - full encoding not yet implemented
        let result = encoder.encode(width, height, &linear_rgb);
        // For now, just check it produces some output
        assert!(result.is_ok());
        let bytes = result.unwrap();
        assert!(bytes.len() > 2);
        assert_eq!(bytes[0], 0xFF);
        assert_eq!(bytes[1], 0x0A);
    }

    #[test]
    fn test_convert_to_xyb() {
        let encoder = TinyEncoder::new(1.0);

        // Gray pixel
        let linear_rgb = vec![0.5, 0.5, 0.5];
        let (x, y, b) = encoder.convert_to_xyb(1, 1, &linear_rgb);

        // Gray should have X ≈ 0 (equal L and M)
        assert!(x[0].abs() < 0.01, "X should be near zero for gray");
        assert!(y[0] > 0.0, "Y should be positive");
        assert!(b[0] > 0.0, "B should be positive");
    }
}
