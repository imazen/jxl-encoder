// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Frame encoder - assembles complete JXL frames.

use crate::GROUP_DIM;
use crate::bit_writer::BitWriter;
use crate::entropy_coding::ans::AnsDistribution;
use crate::error::Result;
use crate::headers::ColorEncoding;
use crate::heuristics::HeuristicLevel;
use crate::modular::channel::ModularImage;
use crate::modular::improved::{
    build_histogram_from_residuals, collect_all_residuals, write_global_modular_section,
    write_group_modular_section, write_improved_modular_stream,
};
use crate::vardct::tokenize::Token;
use crate::vardct::transform::{transform_xyb_image, transform_xyb_image_with_strategy};
use crate::vardct::{VarDctEncoder, VarDctOptions};

/// Options for frame encoding.
#[derive(Debug, Clone)]
pub struct FrameEncoderOptions {
    /// Use modular mode (lossless).
    pub use_modular: bool,
    /// Effort level (1-10, higher = better compression, slower).
    pub effort: u8,
}

impl Default for FrameEncoderOptions {
    fn default() -> Self {
        Self {
            use_modular: true, // Default to lossless
            effort: 7,
        }
    }
}

/// Encodes a single frame.
pub struct FrameEncoder {
    /// Encoding options.
    #[allow(dead_code)]
    options: FrameEncoderOptions,
    /// Image width.
    width: usize,
    /// Image height.
    height: usize,
}

impl FrameEncoder {
    /// Creates a new frame encoder.
    pub fn new(width: usize, height: usize, options: FrameEncoderOptions) -> Self {
        Self {
            options,
            width,
            height,
        }
    }

    /// Encodes a modular image into a frame.
    pub fn encode_modular(
        &self,
        image: &ModularImage,
        _color_encoding: &ColorEncoding,
        writer: &mut BitWriter,
    ) -> Result<()> {
        // Write frame header
        self.write_frame_header(writer)?;

        let num_groups = self.num_groups();

        if num_groups == 1 {
            // Single group: all sections combined into one TOC entry
            let mut section_writer = BitWriter::new();
            write_improved_modular_stream(image, &mut section_writer)?;
            let section_data = section_writer.finish();
            let section_size = section_data.len();

            eprintln!("DEBUG: section_size = {} bytes", section_size);

            // Write TOC
            self.write_toc(writer, section_size)?;

            // Append section data (already byte-aligned)
            for byte in section_data {
                writer.write_u8(byte)?;
            }
        } else {
            // Multi-group: separate TOC entries for global and each group
            self.encode_modular_multi_group(image, writer)?;
        }

        Ok(())
    }

    /// Encodes a modular image using multi-group format (>256x256 images).
    ///
    /// For multi-group frames, the JXL spec requires this TOC structure:
    /// - Section 0: LfGlobal (dc_quant + tree + histograms)
    /// - Section 1: HfGlobal (empty for modular encoding)
    /// - Section 2..2+num_lf_groups: LfGroup (empty for modular encoding)
    /// - Section 2+num_lf_groups..: PassGroup (GroupHeader + pixel data per 256x256 region)
    fn encode_modular_multi_group(
        &self,
        image: &ModularImage,
        writer: &mut BitWriter,
    ) -> Result<()> {
        let num_groups = self.num_groups();
        let num_lf_groups = self.num_lf_groups();
        let num_passes = 1;

        eprintln!(
            "MULTI_GROUP: Encoding {}x{} image with {} groups, {} lf_groups",
            self.width, self.height, num_groups, num_lf_groups
        );

        // Step 1: Collect residuals from the entire image to build global histogram
        let (all_residuals, max_residual) = collect_all_residuals(image);
        let histogram = build_histogram_from_residuals(&all_residuals, max_residual);

        eprintln!(
            "MULTI_GROUP: {} total residuals, max={}, {} unique symbols",
            all_residuals.len(),
            max_residual,
            histogram.iter().filter(|&&c| c > 0).count()
        );

        // Step 2: Write LfGlobal section (tree + histogram)
        let mut lf_global_writer = BitWriter::new();
        let global_state =
            write_global_modular_section(&histogram, max_residual, &mut lf_global_writer)?;
        let lf_global_data = lf_global_writer.finish();

        eprintln!(
            "MULTI_GROUP: LfGlobal section = {} bytes",
            lf_global_data.len()
        );

        // Step 3: HfGlobal is empty for modular encoding (0 bytes)
        let hf_global_data: Vec<u8> = Vec::new();
        eprintln!("MULTI_GROUP: HfGlobal section = 0 bytes (empty for modular)");

        // Step 4: LfGroup sections are empty for modular encoding
        let lf_group_data: Vec<Vec<u8>> = (0..num_lf_groups).map(|_| Vec::new()).collect();
        eprintln!(
            "MULTI_GROUP: {} LfGroup sections = 0 bytes each (empty for modular)",
            num_lf_groups
        );

        // Step 5: Write each PassGroup's data (GroupHeader + pixel data)
        let mut pass_group_data: Vec<Vec<u8>> = Vec::with_capacity(num_groups * num_passes);
        for group_idx in 0..num_groups {
            for _pass in 0..num_passes {
                let (x_start, y_start, x_end, y_end) = self.group_bounds(group_idx);
                let group_image = image.extract_region(x_start, y_start, x_end, y_end)?;

                eprintln!(
                    "MULTI_GROUP: Group {} bounds ({}, {}) - ({}, {}), size {}x{}",
                    group_idx,
                    x_start,
                    y_start,
                    x_end,
                    y_end,
                    group_image.width(),
                    group_image.height()
                );

                let mut group_writer = BitWriter::new();
                write_group_modular_section(&group_image, &global_state, &mut group_writer)?;
                pass_group_data.push(group_writer.finish());

                eprintln!(
                    "MULTI_GROUP: PassGroup {} section = {} bytes",
                    group_idx,
                    pass_group_data.last().unwrap().len()
                );
            }
        }

        // Step 6: Collect all section sizes in correct order and write TOC
        // Order: LfGlobal, HfGlobal, LfGroup[0..num_lf_groups], PassGroup[0..num_groups*num_passes]
        let mut section_sizes = Vec::with_capacity(2 + num_lf_groups + num_groups * num_passes);
        section_sizes.push(lf_global_data.len());
        section_sizes.push(hf_global_data.len());
        for data in &lf_group_data {
            section_sizes.push(data.len());
        }
        for data in &pass_group_data {
            section_sizes.push(data.len());
        }

        eprintln!(
            "MULTI_GROUP: {} total sections, sizes = {:?}",
            section_sizes.len(),
            section_sizes
        );

        self.write_toc_multi(writer, &section_sizes)?;

        // Step 7: Append all section data in same order
        for byte in lf_global_data {
            writer.write_u8(byte)?;
        }
        for byte in hf_global_data {
            writer.write_u8(byte)?;
        }
        for data in lf_group_data {
            for byte in data {
                writer.write_u8(byte)?;
            }
        }
        for data in pass_group_data {
            for byte in data {
                writer.write_u8(byte)?;
            }
        }

        Ok(())
    }

    /// Encodes an RGB image using VarDCT (lossy).
    pub fn encode_vardct(
        &self,
        xyb_data: &[f32],
        distance: f32,
        _color_encoding: &ColorEncoding,
        writer: &mut BitWriter,
    ) -> Result<()> {
        let options = VarDctOptions {
            distance,
            use_default_quant_matrices: true,
            use_default_block_ctx: true,
            ..Default::default()
        };

        let mut vardct_encoder = VarDctEncoder::new(self.width, self.height, options.clone());
        let num_groups = vardct_encoder.num_groups();

        // Compute heuristics from image content
        let use_strategy = options.ac_strategy_heuristics != HeuristicLevel::Dct8Only;
        if use_strategy {
            vardct_encoder.compute_ac_strategies(xyb_data);
        }
        if options.cfl_enabled {
            vardct_encoder.compute_color_correlation(xyb_data);
        }
        if options.adaptive_quant {
            vardct_encoder.compute_quant_field(xyb_data);
        }

        // Transform XYB image data into quantized DCT coefficients
        let quantizer = vardct_encoder.quantizer();

        // Use strategy-aware transform for DCT16/32 support
        let (dc_coeffs, ac_coeffs, tokens, distributions) = if use_strategy {
            let transformed = transform_xyb_image_with_strategy(
                xyb_data,
                self.width,
                self.height,
                quantizer,
                vardct_encoder.ac_strategy_map(),
            );
            eprintln!(
                "TRANSFORM_STRAT: dc_coeffs={}, ac_coeffs={}",
                transformed.dc_coeffs.len(),
                transformed.ac_coeffs.len()
            );
            let (tokens, distributions) = vardct_encoder.tokenize_ac_with_strategy(&transformed)?;
            eprintln!(
                "TOKENIZE_STRAT: {} tokens, {} distributions",
                tokens.len(),
                distributions.len()
            );
            (
                transformed.dc_coeffs,
                transformed.ac_coeffs,
                tokens,
                distributions,
            )
        } else {
            let transformed = transform_xyb_image(xyb_data, self.width, self.height, quantizer);
            eprintln!(
                "TRANSFORM: dc_coeffs={}, ac_coeffs={}",
                transformed.dc_coeffs.len(),
                transformed.ac_coeffs.len()
            );
            let (tokens, distributions) =
                vardct_encoder.tokenize_ac_coefficients(&transformed.ac_coeffs)?;
            eprintln!(
                "TOKENIZE: {} tokens, {} distributions",
                tokens.len(),
                distributions.len()
            );
            (
                transformed.dc_coeffs,
                transformed.ac_coeffs,
                tokens,
                distributions,
            )
        };

        // Write VarDCT frame header
        vardct_encoder.write_frame_header(writer)?;

        // NOTE: No padding between frame header and TOC - the TOC starts
        // immediately after the frame header (the TOC itself handles alignment)

        if num_groups == 1 {
            // Single group: all sections combined into one TOC entry
            self.encode_vardct_single_group(
                &vardct_encoder,
                &dc_coeffs,
                &tokens,
                &distributions,
                writer,
            )?;
        } else {
            // Multi-group: separate TOC entries for each section
            // Pass ac_coeffs so we can tokenize per group
            self.encode_vardct_multi_group(
                &vardct_encoder,
                &dc_coeffs,
                &ac_coeffs,
                &distributions,
                writer,
            )?;
        }

        Ok(())
    }

    /// Encode VarDCT for single-group images (≤256x256).
    fn encode_vardct_single_group(
        &self,
        vardct_encoder: &VarDctEncoder,
        dc_coeffs: &[i32],
        tokens: &[Token],
        distributions: &[AnsDistribution],
        writer: &mut BitWriter,
    ) -> Result<()> {
        // For single-group VarDCT, we can use a single TOC entry containing all sections.
        // The sections are concatenated in order: LfGlobal, HfGlobal, LfGroup, PassGroup
        let mut section_writer = BitWriter::new();

        // LF Global section
        vardct_encoder.write_lf_global(&mut section_writer)?;
        section_writer.zero_pad_to_byte();
        let lf_global_size = section_writer.bytes_written();
        eprintln!("SECTION: LF Global = {} bytes", lf_global_size);

        // HF Global section (with histograms)
        vardct_encoder.write_hf_global(distributions, &mut section_writer)?;
        section_writer.zero_pad_to_byte();
        let hf_global_size = section_writer.bytes_written() - lf_global_size;
        eprintln!("SECTION: HF Global = {} bytes", hf_global_size);

        // LF Group (DC coefficients)
        vardct_encoder.write_lf_group(dc_coeffs, &mut section_writer)?;
        section_writer.zero_pad_to_byte();
        let lf_group_size = section_writer.bytes_written() - lf_global_size - hf_global_size;
        eprintln!("SECTION: LF Group = {} bytes", lf_group_size);

        // Pass Group (AC coefficients)
        vardct_encoder.write_pass_group(tokens, distributions, &mut section_writer)?;
        section_writer.zero_pad_to_byte();
        let pass_group_size =
            section_writer.bytes_written() - lf_global_size - hf_global_size - lf_group_size;
        eprintln!("SECTION: Pass Group = {} bytes", pass_group_size);

        let section_data = section_writer.finish();
        let section_size = section_data.len();
        eprintln!("SECTION: Total = {} bytes", section_size);

        // Write single TOC entry
        self.write_toc(writer, section_size)?;

        // Append section data
        for byte in section_data {
            writer.write_u8(byte)?;
        }

        Ok(())
    }

    /// Encode VarDCT for multi-group images (>256x256).
    fn encode_vardct_multi_group(
        &self,
        vardct_encoder: &VarDctEncoder,
        dc_coeffs: &[i32],
        ac_coeffs: &[i32],
        distributions: &[AnsDistribution],
        writer: &mut BitWriter,
    ) -> Result<()> {
        let num_groups = vardct_encoder.num_groups();
        let num_lf_groups = self.num_lf_groups();
        let num_passes = 1; // Single pass for now

        // For multi-group, we need separate sections:
        // [0] LfGlobal
        // [1] HfGlobal
        // [2..2+num_lf_groups] LfGroup for each LF group
        // [2+num_lf_groups..] PassGroup for each (group, pass)

        let mut section_data: Vec<Vec<u8>> = Vec::new();

        // Section 0: LF Global
        let mut lf_global_writer = BitWriter::new();
        vardct_encoder.write_lf_global(&mut lf_global_writer)?;
        lf_global_writer.zero_pad_to_byte();
        section_data.push(lf_global_writer.finish());

        // Section 1: HF Global
        let mut hf_global_writer = BitWriter::new();
        vardct_encoder.write_hf_global(distributions, &mut hf_global_writer)?;
        hf_global_writer.zero_pad_to_byte();
        section_data.push(hf_global_writer.finish());

        // Sections 2..2+num_lf_groups: LF Group for each LF group
        // For images ≤2048x2048, there's only 1 LF group
        for _lf_group_idx in 0..num_lf_groups {
            let mut lf_group_writer = BitWriter::new();
            vardct_encoder.write_lf_group(dc_coeffs, &mut lf_group_writer)?;
            lf_group_writer.zero_pad_to_byte();
            section_data.push(lf_group_writer.finish());
        }

        // Sections for PassGroup: tokenize and write AC coefficients per group
        for group_idx in 0..num_groups {
            for _pass in 0..num_passes {
                let mut pass_group_writer = BitWriter::new();

                // Tokenize AC coefficients for this specific group
                let group_tokens =
                    vardct_encoder.tokenize_ac_coefficients_for_group(ac_coeffs, group_idx);

                // Write tokens for this group
                vardct_encoder.write_pass_group(
                    &group_tokens,
                    distributions,
                    &mut pass_group_writer,
                )?;

                pass_group_writer.zero_pad_to_byte();
                section_data.push(pass_group_writer.finish());
            }
        }

        // Collect section sizes
        let section_sizes: Vec<usize> = section_data.iter().map(|s| s.len()).collect();

        // Write TOC with all section sizes
        self.write_toc_multi(writer, &section_sizes)?;

        // Append all section data
        for section in section_data {
            for byte in section {
                writer.write_u8(byte)?;
            }
        }

        Ok(())
    }

    /// Writes the frame header for a simple lossless modular frame.
    fn write_frame_header(&self, writer: &mut BitWriter) -> Result<()> {
        eprintln!(
            "FRMH [bit {}]: Starting frame header",
            writer.bits_written()
        );

        // all_default = false (because we use Modular encoding, not VarDCT default)
        writer.write(1, 0)?;
        eprintln!("FRMH [bit {}]: all_default = 0", writer.bits_written());

        // frame_type = RegularFrame (0)
        writer.write(2, 0)?;
        eprintln!("FRMH [bit {}]: frame_type = 0", writer.bits_written());

        // encoding = Modular (1)
        writer.write(1, 1)?;
        eprintln!(
            "FRMH [bit {}]: encoding = 1 (Modular)",
            writer.bits_written()
        );

        // flags = 0 (U64 encoding: selector 0 with 2 bits means value is 0)
        writer.write(2, 0)?;
        eprintln!("FRMH [bit {}]: flags = 0", writer.bits_written());

        // do_ycbcr = false (only for non-xyb_encoded, which is our case for lossless)
        // For lossless modular, xyb_encoded should be false in the image metadata
        writer.write(1, 0)?;
        eprintln!("FRMH [bit {}]: do_ycbcr = 0", writer.bits_written());

        // upsampling = 1 (selector 0 in u2S(1,2,4,8))
        writer.write(2, 0)?;
        eprintln!("FRMH [bit {}]: upsampling = 0", writer.bits_written());

        // ec_upsampling - for each extra channel (none for RGB)
        // (already handled by not writing anything)

        // group_size_shift: 0 = 128, 1 = 256, 2 = 512, 3 = 1024
        // We use GROUP_DIM = 256, so shift must be 1
        writer.write(2, 1)?; // selector 1 = 256 pixels
        eprintln!(
            "FRMH [bit {}]: group_size_shift = 1 (256)",
            writer.bits_written()
        );

        // passes (only if frame_type != ReferenceOnly)
        // num_passes = 1 (selector 0 in u2S(1,2,3,Bits(3)+4))
        writer.write(2, 0)?;
        eprintln!("FRMH [bit {}]: passes = 0", writer.bits_written());

        // lf_level - not written (only for LFFrame)

        // have_crop = false (only if frame_type != LFFrame)
        writer.write(1, 0)?;
        eprintln!("FRMH [bit {}]: have_crop = 0", writer.bits_written());

        // blending_info (only for RegularFrame or SkipProgressive)
        // mode = Replace (selector 0 in u2S(0,1,2,Bits(2)+3))
        writer.write(2, 0)?;
        eprintln!("FRMH [bit {}]: blending = 0", writer.bits_written());

        // ec_blending_info - for each extra channel (none for RGB)

        // duration - not written (no animation)
        // timecode - not written (no timecode)

        // is_last = true (only for RegularFrame or SkipProgressive)
        writer.write(1, 1)?;
        eprintln!("FRMH [bit {}]: is_last = 1", writer.bits_written());

        // save_as_reference - not written (is_last = true)

        // save_before_ct - not written (conditions not met)

        // name = empty string
        // name_length = 0 using u2S(0, Bits(4), Bits(5)+16, Bits(10)+48)
        writer.write(2, 0)?; // selector 0 = value 0
        eprintln!("FRMH [bit {}]: name = 0", writer.bits_written());

        // restoration_filter - MUST disable filters for lossless modular encoding!
        // Default has gab=true (Gaborish) and epf_iters=2 (Edge-Preserving Filter)
        // which would blur the image. For lossless, we disable both.
        writer.write(1, 0)?; // all_default = false
        writer.write(1, 0)?; // gab = false (disable Gaborish)
        writer.write(2, 0)?; // epf_iters = 0 (disable EPF)
        eprintln!(
            "FRMH [bit {}]: restoration = disabled (gab=false, epf=0)",
            writer.bits_written()
        );

        // extensions = 0 (no extensions)
        // u64 encoding: selector 0 (2 bits) means value 0
        writer.write(2, 0)?;
        eprintln!(
            "FRMH [bit {}]: extensions = 0, frame header done",
            writer.bits_written()
        );

        // NOTE: #[aligned] in jxl-rs means byte alignment at START of reading,
        // not at the end. The caller (encoder.rs) handles the alignment before
        // calling encode_modular. We do NOT byte-align here - the TOC follows
        // immediately and handles its own alignment after the permuted bit.

        Ok(())
    }

    /// Writes the table of contents with a single section.
    fn write_toc(&self, writer: &mut BitWriter, section_size: usize) -> Result<()> {
        self.write_toc_multi(writer, &[section_size])
    }

    /// Writes the table of contents with multiple sections.
    fn write_toc_multi(&self, writer: &mut BitWriter, section_sizes: &[usize]) -> Result<()> {
        eprintln!("TOC [bit {}]: Writing permuted = 0", writer.bits_written());
        // permuted = false
        writer.write(1, 0)?;

        eprintln!(
            "TOC [bit {}]: After permuted, byte aligning",
            writer.bits_written()
        );
        // Byte align before TOC entries (permutation reads, then aligns)
        writer.zero_pad_to_byte();

        // Write TOC entries using u2S(Bits(10), Bits(14)+1024, Bits(22)+17408, Bits(30)+4211712)
        for (i, &size) in section_sizes.iter().enumerate() {
            eprintln!(
                "TOC [bit {}]: Writing entry {} size={}",
                writer.bits_written(),
                i,
                size
            );
            self.write_toc_entry(writer, size as u32)?;
        }
        eprintln!("TOC [bit {}]: After TOC entries", writer.bits_written());

        // Byte align after TOC entries
        writer.zero_pad_to_byte();

        Ok(())
    }

    /// Writes a single TOC entry.
    fn write_toc_entry(&self, writer: &mut BitWriter, size: u32) -> Result<()> {
        // u2S(Bits(10), Bits(14)+1024, Bits(22)+17408, Bits(30)+4211712)
        if size < 1024 {
            writer.write(2, 0)?; // selector 0
            writer.write(10, size as u64)?;
        } else if size < 17408 {
            writer.write(2, 1)?; // selector 1
            writer.write(14, (size - 1024) as u64)?;
        } else if size < 4211712 {
            writer.write(2, 2)?; // selector 2
            writer.write(22, (size - 17408) as u64)?;
        } else {
            writer.write(2, 3)?; // selector 3
            writer.write(30, (size - 4211712) as u64)?;
        }
        Ok(())
    }

    /// Returns the number of groups in this frame.
    pub fn num_groups(&self) -> usize {
        let num_groups_x = self.width.div_ceil(GROUP_DIM);
        let num_groups_y = self.height.div_ceil(GROUP_DIM);
        num_groups_x * num_groups_y
    }

    /// Returns the number of groups in X direction.
    pub fn num_groups_x(&self) -> usize {
        self.width.div_ceil(GROUP_DIM)
    }

    /// Returns the number of groups in Y direction.
    pub fn num_groups_y(&self) -> usize {
        self.height.div_ceil(GROUP_DIM)
    }

    /// Returns the number of LF groups (DC groups).
    /// LF groups are 8x the size of regular groups (2048x2048 pixels).
    pub fn num_lf_groups(&self) -> usize {
        let lf_group_dim = GROUP_DIM * 8; // 2048
        let lf_groups_x = self.width.div_ceil(lf_group_dim);
        let lf_groups_y = self.height.div_ceil(lf_group_dim);
        lf_groups_x * lf_groups_y
    }

    /// Returns the number of TOC entries for this frame.
    /// Single group: 1 entry
    /// Multi-group: 2 + num_lf_groups + num_groups * num_passes
    pub fn num_toc_entries(&self, num_passes: usize) -> usize {
        let num_groups = self.num_groups();
        if num_groups == 1 && num_passes == 1 {
            1
        } else {
            2 + self.num_lf_groups() + num_groups * num_passes
        }
    }

    /// Get the pixel bounds for a group.
    /// Returns (x_start, y_start, x_end, y_end).
    pub fn group_bounds(&self, group_idx: usize) -> (usize, usize, usize, usize) {
        let num_groups_x = self.num_groups_x();
        let gx = group_idx % num_groups_x;
        let gy = group_idx / num_groups_x;

        let x_start = gx * GROUP_DIM;
        let y_start = gy * GROUP_DIM;
        let x_end = (x_start + GROUP_DIM).min(self.width);
        let y_end = (y_start + GROUP_DIM).min(self.height);

        (x_start, y_start, x_end, y_end)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_encoder_creation() {
        let encoder = FrameEncoder::new(256, 256, FrameEncoderOptions::default());
        assert_eq!(encoder.num_groups(), 1);
    }

    #[test]
    fn test_frame_encoder_multi_group() {
        let encoder = FrameEncoder::new(512, 512, FrameEncoderOptions::default());
        assert_eq!(encoder.num_groups(), 4); // 2x2 groups
        assert_eq!(encoder.num_groups_x(), 2);
        assert_eq!(encoder.num_groups_y(), 2);
        assert_eq!(encoder.num_lf_groups(), 1); // 512 < 2048
    }

    #[test]
    fn test_group_bounds() {
        let encoder = FrameEncoder::new(512, 512, FrameEncoderOptions::default());

        // Group 0: top-left
        let (x0, y0, x1, y1) = encoder.group_bounds(0);
        assert_eq!((x0, y0, x1, y1), (0, 0, 256, 256));

        // Group 1: top-right
        let (x0, y0, x1, y1) = encoder.group_bounds(1);
        assert_eq!((x0, y0, x1, y1), (256, 0, 512, 256));

        // Group 2: bottom-left
        let (x0, y0, x1, y1) = encoder.group_bounds(2);
        assert_eq!((x0, y0, x1, y1), (0, 256, 256, 512));

        // Group 3: bottom-right
        let (x0, y0, x1, y1) = encoder.group_bounds(3);
        assert_eq!((x0, y0, x1, y1), (256, 256, 512, 512));
    }

    #[test]
    fn test_group_bounds_partial() {
        // 300x200 image: 2x1 groups, second group is partial
        let encoder = FrameEncoder::new(300, 200, FrameEncoderOptions::default());
        assert_eq!(encoder.num_groups(), 2); // 2x1

        let (x0, y0, x1, y1) = encoder.group_bounds(0);
        assert_eq!((x0, y0, x1, y1), (0, 0, 256, 200));

        let (x0, y0, x1, y1) = encoder.group_bounds(1);
        assert_eq!((x0, y0, x1, y1), (256, 0, 300, 200)); // Clamped to image bounds
    }

    #[test]
    fn test_num_toc_entries() {
        // Single group, single pass
        let encoder = FrameEncoder::new(256, 256, FrameEncoderOptions::default());
        assert_eq!(encoder.num_toc_entries(1), 1);

        // 4 groups, single pass: 2 + 1 + 4 = 7
        let encoder = FrameEncoder::new(512, 512, FrameEncoderOptions::default());
        assert_eq!(encoder.num_toc_entries(1), 7);

        // 4 groups, 2 passes: 2 + 1 + 8 = 11
        assert_eq!(encoder.num_toc_entries(2), 11);
    }

    #[test]
    fn test_encode_multi_group_image() {
        // 300x300 RGB image - requires 2x2 = 4 groups
        let mut data = Vec::with_capacity(300 * 300 * 3);
        for y in 0..300 {
            for x in 0..300 {
                // Smooth gradient for good compression
                data.push(((x + y) % 256) as u8); // R
                data.push(((x * 2) % 256) as u8); // G
                data.push(((y * 2) % 256) as u8); // B
            }
        }

        let image = ModularImage::from_rgb8(&data, 300, 300).unwrap();

        let encoder = FrameEncoder::new(300, 300, FrameEncoderOptions::default());
        assert_eq!(encoder.num_groups(), 4); // 2x2 groups

        let mut writer = BitWriter::new();
        let color_encoding = ColorEncoding::srgb();

        encoder
            .encode_modular(&image, &color_encoding, &mut writer)
            .unwrap();

        let bytes = writer.finish_with_padding();
        eprintln!("Multi-group modular: {} bytes", bytes.len());
        assert!(!bytes.is_empty());
        // Should have reasonable size (not huge, not tiny)
        assert!(bytes.len() > 100); // Has content
        assert!(bytes.len() < 300 * 300 * 3); // Better than raw
    }

    #[test]
    fn test_encode_small_image() {
        // 4x4 RGB image with only 4 unique values (max for simple Huffman)
        // Pattern: checkerboard of two colors
        let mut data = Vec::with_capacity(4 * 4 * 3);
        for y in 0..4 {
            for x in 0..4 {
                let v = if (x + y) % 2 == 0 { 0u8 } else { 128u8 };
                data.push(v); // R
                data.push(v); // G
                data.push(v); // B
            }
        }

        let image = ModularImage::from_rgb8(&data, 4, 4).unwrap();

        let encoder = FrameEncoder::new(4, 4, FrameEncoderOptions::default());
        let mut writer = BitWriter::new();
        let color_encoding = ColorEncoding::srgb();

        encoder
            .encode_modular(&image, &color_encoding, &mut writer)
            .unwrap();

        let bytes = writer.finish_with_padding();
        assert!(!bytes.is_empty());
    }
}
