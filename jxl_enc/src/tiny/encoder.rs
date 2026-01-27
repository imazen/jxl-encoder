// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Main tiny encoder implementation.

use super::common::*;
use super::frame::{DistanceParams, write_frame_header, write_quant_scales, write_toc};
use crate::bit_writer::BitWriter;
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
        let xsize_groups = div_ceil(width, GROUP_DIM);
        let ysize_groups = div_ceil(height, GROUP_DIM);
        let xsize_dc_groups = div_ceil(width, DC_GROUP_DIM);
        let ysize_dc_groups = div_ceil(height, DC_GROUP_DIM);
        let num_groups = xsize_groups * ysize_groups;
        let num_dc_groups = xsize_dc_groups * ysize_dc_groups;

        // Number of sections: DC global + DC groups + AC global + AC groups
        let num_sections = 2 + num_dc_groups + num_groups;

        // Create main writer
        let mut writer = BitWriter::with_capacity(width * height * 4);

        // Write JXL signature
        writer.write(8, 0xFF)?;
        writer.write(8, 0x0A)?;

        // Write size header (simple format for small images)
        self.write_file_header(width, height, &mut writer)?;

        // Write frame header
        write_frame_header(params.x_qm_scale, params.epf_iters, &mut writer)?;

        // TODO: Encode DC and AC groups
        // For now, write minimal placeholder sections

        // Create section writers
        let mut sections: Vec<Vec<u8>> = Vec::with_capacity(num_sections);

        // DC Global section
        let mut dc_global = BitWriter::new();
        self.write_dc_global(&params, num_dc_groups, &mut dc_global)?;
        dc_global.zero_pad_to_byte();
        sections.push(dc_global.finish());

        // DC group sections
        for _ in 0..num_dc_groups {
            let mut dc_group = BitWriter::new();
            // TODO: Write actual DC group data
            dc_group.write(6, 12)?; // placeholder: extra_dc_precision=0, use global tree
            dc_group.zero_pad_to_byte();
            sections.push(dc_group.finish());
        }

        // AC Global section
        let mut ac_global = BitWriter::new();
        self.write_ac_global(num_groups, &mut ac_global)?;
        ac_global.zero_pad_to_byte();
        sections.push(ac_global.finish());

        // AC group sections
        for _ in 0..num_groups {
            let mut ac_group = BitWriter::new();
            // TODO: Write actual AC group data
            ac_group.zero_pad_to_byte();
            sections.push(ac_group.finish());
        }

        // Write TOC
        let section_sizes: Vec<usize> = sections.iter().map(|s| s.len()).collect();

        // For single-group images, combine all sections into one
        if sections.len() == 4 {
            let mut combined = sections.remove(0);
            for section in sections {
                combined.extend(section);
            }
            write_toc(&[combined.len()], &mut writer)?;
            writer.append_bytes(&combined)?;
        } else {
            write_toc(&section_sizes, &mut writer)?;
            for section in sections {
                writer.append_bytes(&section)?;
            }
        }

        Ok(writer.finish_with_padding())
    }

    /// Write the file header (size box).
    fn write_file_header(&self, width: usize, height: usize, writer: &mut BitWriter) -> Result<()> {
        // Simple approach: write all_default=0, then size fields
        writer.write(1, 0)?; // not all default

        // xyb_encoded (1 bit) - 1 for VarDCT
        writer.write(1, 1)?;

        // Size encoding
        self.write_size(width, height, writer)?;

        // Use defaults for everything else
        writer.write(1, 1)?; // default bit depth (8-bit)
        writer.write(1, 1)?; // default modular 16 bit buffers
        writer.write(1, 0)?; // no extra channels
        writer.write(1, 1)?; // all_default color encoding
        writer.write(2, 0)?; // no extensions

        Ok(())
    }

    /// Write image size.
    fn write_size(&self, width: usize, height: usize, writer: &mut BitWriter) -> Result<()> {
        // div8 = 0 means dimensions directly
        writer.write(1, 0)?;

        let h = height as u64;
        let w = width as u64;

        // Write height
        if height <= 256 {
            writer.write(2, 0)?; // selector 0
            writer.write(9, h - 1)?;
        } else if height <= 8449 {
            writer.write(2, 1)?; // selector 1
            writer.write(13, h - 1 - 256)?;
        } else {
            writer.write(2, 2)?;
            writer.write(18, h - 1 - 8449)?;
        }

        // ratio = 0 (no aspect ratio)
        writer.write(3, 0)?;

        // Write width
        if width <= 256 {
            writer.write(2, 0)?;
            writer.write(9, w - 1)?;
        } else if width <= 8449 {
            writer.write(2, 1)?;
            writer.write(13, w - 1 - 256)?;
        } else {
            writer.write(2, 2)?;
            writer.write(18, w - 1 - 8449)?;
        }

        Ok(())
    }

    /// Write DC global section.
    fn write_dc_global(
        &self,
        params: &DistanceParams,
        _num_dc_groups: usize,
        writer: &mut BitWriter,
    ) -> Result<()> {
        writer.write(1, 1)?; // default dequant dc
        write_quant_scales(params.global_scale, params.quant_dc, writer)?;

        // BlockCtxMap
        writer.write(1, 0)?; // non-default BlockCtxMap
        writer.write(16, 0)?; // no dc ctx, no qft
        // Context map for block context (simplified)
        writer.write(3, 1)?; // simple context map

        writer.write(1, 1)?; // default DC camp

        // Context tree (simplified - write empty tree for now)
        writer.write(1, 0)?; // empty tree

        writer.write(1, 0)?; // no lz77

        // Entropy code will be written when we have actual data
        Ok(())
    }

    /// Write AC global section.
    fn write_ac_global(&self, num_groups: usize, writer: &mut BitWriter) -> Result<()> {
        writer.write(1, 1)?; // all default quant matrices

        let num_histo_bits = ceil_log2_nonzero(num_groups);
        if num_histo_bits != 0 {
            writer.write(num_histo_bits as usize, 0)?;
        }

        writer.write(2, 3)?;
        writer.write(13, 0)?; // all default coeff order
        writer.write(1, 0)?; // no lz77

        // Entropy code will be written when we have actual data
        Ok(())
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
}
