// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Frame encoder - assembles complete JXL frames.

use crate::GROUP_DIM;
use crate::bit_writer::BitWriter;
use crate::error::Result;
use crate::headers::ColorEncoding;
use crate::modular::channel::ModularImage;
use crate::modular::minimal::write_minimal_modular_stream;

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

        // Encode the image data to a temporary buffer to know its size
        let mut section_writer = BitWriter::new();
        write_minimal_modular_stream(image, &mut section_writer)?;
        let section_data = section_writer.finish();
        let section_size = section_data.len();

        eprintln!("DEBUG: section_size = {} bytes", section_size);

        // Write TOC
        self.write_toc(writer, section_size)?;

        // Append section data (already byte-aligned)
        for byte in section_data {
            writer.write_u8(byte)?;
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

        // group_size_shift = 1 (default, 256x256 base -> 512x512 with shift=1)
        // Actually, looking at jxl-rs, default is 1, but let's use 0 for 256x256
        writer.write(2, 0)?; // 0 = 128, 1 = 256, 2 = 512, 3 = 1024
        eprintln!("FRMH [bit {}]: group_size_shift = 0", writer.bits_written());

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

    /// Writes the table of contents.
    fn write_toc(&self, writer: &mut BitWriter, section_size: usize) -> Result<()> {
        let num_groups = self.num_groups();
        let num_toc_entries = if num_groups == 1 { 1 } else { 2 + num_groups };

        eprintln!("TOC [bit {}]: Writing permuted = 0", writer.bits_written());
        // permuted = false
        writer.write(1, 0)?;

        eprintln!(
            "TOC [bit {}]: After permuted, byte aligning",
            writer.bits_written()
        );
        // Byte align before TOC entries (permutation reads, then aligns)
        writer.zero_pad_to_byte();

        eprintln!(
            "TOC [bit {}]: Writing TOC entry for size={}",
            writer.bits_written(),
            section_size
        );
        // Write TOC entries using u2S(Bits(10), Bits(14)+1024, Bits(22)+17408, Bits(30)+4211712)
        if num_toc_entries == 1 {
            // Single section
            self.write_toc_entry(writer, section_size as u32)?;
            eprintln!("TOC [bit {}]: After TOC entry", writer.bits_written());
        } else {
            // Multiple sections - placeholder for now
            for _ in 0..num_toc_entries {
                self.write_toc_entry(writer, 0)?;
            }
        }

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
