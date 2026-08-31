// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! JXL file header (SizeHeader + ImageMetadata).

use crate::JXL_SIGNATURE;
use crate::bit_writer::BitWriter;
use crate::error::Result;

use super::color_encoding::ColorEncoding;
use super::extra_channels::ExtraChannelInfo;

/// Orientation of the image.
///
/// The full 8-value JXL spec table (ISO/IEC 18181-1 ImageMetadata
/// orientation). The encoder currently only ever writes
/// [`Orientation::Identity`]; the other variants document the wire
/// values and are kept for that purpose (#76).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
#[allow(dead_code)]
pub enum Orientation {
    #[default]
    Identity = 1,
    FlipHorizontal = 2,
    Rotate180 = 3,
    FlipVertical = 4,
    Transpose = 5,
    Rotate90CW = 6,
    AntiTranspose = 7,
    Rotate90CCW = 8,
}

/// Bit depth specification.
#[derive(Debug, Clone, Copy)]
pub struct BitDepth {
    /// True if floating point, false if integer.
    pub float_sample: bool,
    /// Bits per sample (for integer) or exponent bits (for float).
    pub bits_per_sample: u32,
    /// Exponent bits for floating point samples.
    pub exponent_bits: u32,
}

impl Default for BitDepth {
    fn default() -> Self {
        Self {
            float_sample: false,
            bits_per_sample: 8,
            exponent_bits: 0,
        }
    }
}

impl BitDepth {
    /// Creates an 8-bit integer depth.
    pub fn uint8() -> Self {
        Self::default()
    }

    /// Creates a 16-bit integer depth.
    pub fn uint16() -> Self {
        Self {
            float_sample: false,
            bits_per_sample: 16,
            exponent_bits: 0,
        }
    }

    /// Creates a 32-bit float depth.
    pub fn float32() -> Self {
        Self {
            float_sample: true,
            bits_per_sample: 32,
            exponent_bits: 8,
        }
    }

    /// Creates a 16-bit half-float depth.
    pub fn float16() -> Self {
        Self {
            float_sample: true,
            bits_per_sample: 16,
            exponent_bits: 5,
        }
    }
}

/// Animation parameters.
#[derive(Debug, Clone, Default)]
pub struct AnimationHeader {
    /// Ticks per second numerator.
    pub tps_numerator: u32,
    /// Ticks per second denominator.
    pub tps_denominator: u32,
    /// Number of loops (0 = infinite).
    pub num_loops: u32,
    /// Whether frames have varying durations.
    pub have_timecodes: bool,
}

impl AnimationHeader {
    /// Writes the AnimationHeader to the bitstream.
    ///
    /// Matches libjxl's `AnimationHeader::VisitFields`:
    /// - tps_numerator: u2S(100, 1000, Bits(10)+1, Bits(30)+1)
    /// - tps_denominator: u2S(1, 1001, Bits(8)+1, Bits(10)+1)
    /// - num_loops: u2S(0, Bits(3), Bits(16), Bits(32))
    /// - have_timecodes: Bool(false)
    pub fn write(&self, writer: &mut BitWriter) -> Result<()> {
        // tps_numerator: u2S(100, 1000, BitsOffset(10,1), BitsOffset(30,1))
        match self.tps_numerator {
            100 => writer.write(2, 0)?,
            1000 => writer.write(2, 1)?,
            v if (1..=1024).contains(&v) => {
                writer.write(2, 2)?;
                writer.write(10, (v - 1) as u64)?;
            }
            v => {
                debug_assert!(v >= 1, "tps_numerator must be >= 1");
                writer.write(2, 3)?;
                writer.write(30, (v - 1) as u64)?;
            }
        }

        // tps_denominator: u2S(1, 1001, BitsOffset(8,1), BitsOffset(10,1))
        match self.tps_denominator {
            1 => writer.write(2, 0)?,
            1001 => writer.write(2, 1)?,
            v @ 2..=256 => {
                writer.write(2, 2)?;
                writer.write(8, (v - 1) as u64)?;
            }
            v => {
                debug_assert!((1..=1025).contains(&v), "tps_denominator {v} out of range");
                writer.write(2, 3)?;
                writer.write(10, (v - 1) as u64)?;
            }
        }

        // num_loops: u2S(0, Bits(3), Bits(16), Bits(32))
        match self.num_loops {
            0 => writer.write(2, 0)?,
            v @ 1..=7 => {
                writer.write(2, 1)?;
                writer.write(3, v as u64)?;
            }
            v @ 8..=65535 => {
                writer.write(2, 2)?;
                writer.write(16, v as u64)?;
            }
            v => {
                writer.write(2, 3)?;
                writer.write(32, v as u64)?;
            }
        }

        // have_timecodes: Bool(default=false)
        writer.write_bit(self.have_timecodes)?;

        Ok(())
    }
}

/// Image metadata that appears once per file.
#[derive(Debug, Clone)]
pub struct ImageMetadata {
    /// Bit depth configuration.
    pub bit_depth: BitDepth,
    /// Color encoding (color space, transfer function, etc.).
    pub color_encoding: ColorEncoding,
    /// Extra channels (alpha, depth, etc.).
    pub extra_channels: Vec<ExtraChannelInfo>,
    /// Image orientation.
    pub orientation: Orientation,
    /// Animation parameters (None if not animated).
    pub animation: Option<AnimationHeader>,
    /// Intensity target for HDR in nits.
    pub intensity_target: f32,
    /// Minimum nits for tone mapping.
    pub min_nits: f32,
    /// `ToneMapping.relative_to_max_display` (default `false`). When
    /// `true`, [`Self::linear_below`] is interpreted as a ratio in
    /// `[0, 1]` of the maximum display brightness rather than an
    /// absolute nit value. Mirrors libjxl `ToneMapping`
    /// (`image_metadata.h:169`) / jxl-rs `ToneMapping`
    /// (`headers/image_metadata.rs:147`). Closes issue #46 chunk 1a.
    pub relative_to_max_display: bool,
    /// `ToneMapping.linear_below` (default `0.0`). The tone-mapping
    /// curve leaves pixels strictly below this value unchanged
    /// (linear). Interpretation depends on
    /// [`Self::relative_to_max_display`] — ratio when `true`, absolute
    /// nits when `false`. Mirrors libjxl `ToneMapping`
    /// (`image_metadata.h:174`) / jxl-rs `ToneMapping`
    /// (`headers/image_metadata.rs:149`). Closes issue #46 chunk 1a.
    pub linear_below: f32,
    /// Whether intrinsic size differs from coded size.
    pub have_intrinsic_size: bool,
    /// Intrinsic width (if have_intrinsic_size).
    pub intrinsic_width: u32,
    /// Intrinsic height (if have_intrinsic_size).
    pub intrinsic_height: u32,
    /// Whether image uses XYB color encoding (true for lossy, false for lossless).
    pub xyb_encoded: bool,
    /// Force `modular_16bit_buffer_sufficient = false` in the header even when
    /// `bits_per_sample <= 12`. Set by the VarDCT encoder when the quantized DC
    /// exceeds the i16 range so a spec decoder reconstructs the LF/DC modular
    /// image into i32 (not i16) buffers — otherwise oversized DC wraps and
    /// desynchronises the DC ANS stream. Default `false`. (#94)
    pub force_modular_32bit: bool,
}

impl Default for ImageMetadata {
    fn default() -> Self {
        Self {
            bit_depth: BitDepth::default(),
            color_encoding: ColorEncoding::default(),
            extra_channels: Vec::new(),
            orientation: Orientation::default(),
            animation: None,
            intensity_target: 255.0,
            min_nits: 0.0,
            relative_to_max_display: false,
            linear_below: 0.0,
            have_intrinsic_size: false,
            intrinsic_width: 0,
            intrinsic_height: 0,
            xyb_encoded: false, // Default to lossless (non-XYB)
            force_modular_32bit: false,
        }
    }
}

/// Complete JXL file header.
#[derive(Debug, Clone)]
pub struct FileHeader {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Image metadata.
    pub metadata: ImageMetadata,
    /// Decoder upsampling LUT selection for the active frame-header
    /// `upsampling` factor (libjxl
    /// `JxlEncoderSetUpsamplingMode(_, factor, mode)`):
    ///
    /// - `None` (default) / `Some(-1)` — fancy default upsampling.
    ///   `CustomTransformData` stays at `all_default = true`.
    /// - `Some(0)` — nearest-neighbour LUT for the matching factor.
    /// - `Some(1)` — "pixel dots" (nearest with cut corners). Only
    ///   meaningful for `upsampling_factor` 4 / 8; for factor 2 it
    ///   behaves as nearest because the libjxl table is empty there.
    pub upsampling_mode: Option<i32>,
    /// Upsampling factor (1, 2, 4, or 8) the encoder will emit in the
    /// frame header (libjxl `frame_header.upsampling`). Tracked here
    /// so [`Self::write`] can select the right LUT slot when
    /// [`Self::upsampling_mode`] is set.
    pub upsampling_factor: u32,
    /// Permit the `ImageMetadata.all_default = 1` one-bit fast path when
    /// [`Self::is_metadata_default`] holds. `false` on every path except
    /// [`crate::api::EncoderStrategy::Libjxl`] — see that method's docs
    /// and the `ImageMetadata.all_default` row in
    /// `docs/LIBJXL_DIVERGENCES.md` Section D for why the default is the
    /// larger encoding.
    pub header_all_default_fast_paths: bool,
}

impl FileHeader {
    /// Creates a new file header for an RGB image.
    pub fn new_rgb(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            metadata: ImageMetadata::default(),
            upsampling_mode: None,
            upsampling_factor: 1,
            // Opt-in only; see `is_metadata_default`.
            header_all_default_fast_paths: false,
        }
    }

    /// Creates a new file header for an RGBA image.
    pub fn new_rgba(width: u32, height: u32) -> Self {
        let mut header = Self::new_rgb(width, height);
        header
            .metadata
            .extra_channels
            .push(ExtraChannelInfo::alpha());
        header
    }

    /// Creates a new file header for a grayscale image.
    pub fn new_gray(width: u32, height: u32) -> Self {
        let mut header = Self::new_rgb(width, height);
        header.metadata.color_encoding = ColorEncoding::gray();
        header
    }

    /// Writes the JXL signature.
    pub fn write_signature(writer: &mut BitWriter) -> Result<()> {
        writer.write_u8(JXL_SIGNATURE[0])?;
        writer.write_u8(JXL_SIGNATURE[1])?;
        Ok(())
    }

    /// Writes the size header.
    ///
    /// JXL Size format:
    /// - small: Bool (1 bit) - true if both dimensions are multiples of 8 and <= 256
    /// - If small:
    ///   - ysize_div8: Bits(5) + 1 (height/8, range 1-32)
    ///   - ratio: Bits(3)
    ///   - If ratio == 0: xsize_div8: Bits(5) + 1 (width/8, range 1-32)
    /// - If !small:
    ///   - ysize: 1 + u2S(Bits(9), Bits(13), Bits(18), Bits(30))
    ///   - ratio: Bits(3)
    ///   - If ratio == 0: xsize: 1 + u2S(Bits(9), Bits(13), Bits(18), Bits(30))
    fn write_size_header(&self, writer: &mut BitWriter) -> Result<()> {
        // small = true if both dimensions are multiples of 8 and fit in 5 bits (8-256)
        let h_div8 = self.height.is_multiple_of(8) && self.height / 8 >= 1 && self.height / 8 <= 32;
        let w_div8 = self.width.is_multiple_of(8) && self.width / 8 >= 1 && self.width / 8 <= 32;
        let small = h_div8 && w_div8;

        crate::trace::debug_eprintln!(
            "SIZE_HDR: {}x{}, small={}, h_div8={}, w_div8={}",
            self.width,
            self.height,
            small,
            h_div8,
            w_div8
        );
        writer.write_bit(small)?;

        if small {
            // ysize_div8_minus_1: Bits(5), decoder adds 1 then multiplies by 8
            crate::trace::debug_eprintln!("SIZE_HDR: ysize_div8_minus_1 = {}", self.height / 8 - 1);
            writer.write(5, (self.height / 8 - 1) as u64)?;

            let ratio = self.compute_ratio();
            crate::trace::debug_eprintln!("SIZE_HDR: ratio = {}", ratio);
            writer.write(3, ratio as u64)?;

            if ratio == 0 {
                // xsize_div8_minus_1: Bits(5), decoder adds 1 then multiplies by 8
                crate::trace::debug_eprintln!(
                    "SIZE_HDR: xsize_div8_minus_1 = {}",
                    self.width / 8 - 1
                );
                writer.write(5, (self.width / 8 - 1) as u64)?;
            }
        } else {
            // ysize: 1 + u2S(Bits(9), Bits(13), Bits(18), Bits(30))
            // Write height - 1 using u2S encoding
            self.write_size_u2s(writer, self.height - 1)?;

            let ratio = self.compute_ratio();
            writer.write(3, ratio as u64)?;

            if ratio == 0 {
                // xsize: 1 + u2S(Bits(9), Bits(13), Bits(18), Bits(30))
                self.write_size_u2s(writer, self.width - 1)?;
            }
        }

        Ok(())
    }

    /// Writes a size value using u2S(Bits(9), Bits(13), Bits(18), Bits(30)) encoding.
    /// The decoder adds 1 to the result, so we write value directly (not value-1).
    fn write_size_u2s(&self, writer: &mut BitWriter, value: u32) -> Result<()> {
        if value < (1 << 9) {
            writer.write(2, 0)?; // selector 0
            writer.write(9, value as u64)?;
        } else if value < (1 << 13) {
            writer.write(2, 1)?; // selector 1
            writer.write(13, value as u64)?;
        } else if value < (1 << 18) {
            writer.write(2, 2)?; // selector 2
            writer.write(18, value as u64)?;
        } else {
            writer.write(2, 3)?; // selector 3
            writer.write(30, value as u64)?;
        }
        Ok(())
    }

    /// Computes the aspect ratio selector (0 = explicit width).
    fn compute_ratio(&self) -> u8 {
        // Ratio selectors: 1=1:1, 2=12:10, 3=4:3, 4=3:2, 5=16:9, 6=5:4, 7=2:1
        if self.width == self.height {
            1 // 1:1
        } else if self.width * 10 == self.height * 12 {
            2 // 12:10
        } else if self.width * 3 == self.height * 4 {
            3 // 4:3
        } else if self.width * 2 == self.height * 3 {
            4 // 3:2
        } else if self.width * 9 == self.height * 16 {
            5 // 16:9
        } else if self.width * 4 == self.height * 5 {
            6 // 5:4
        } else if self.width == self.height * 2 {
            7 // 2:1
        } else {
            0 // Explicit
        }
    }

    /// Writes the complete file header (signature + size + metadata + transform_data).
    pub fn write(&self, writer: &mut BitWriter) -> Result<()> {
        crate::trace::debug_eprintln!("FHDR [bit {}]: Starting file header", writer.bits_written());
        Self::write_signature(writer)?;
        crate::trace::debug_eprintln!("FHDR [bit {}]: After signature", writer.bits_written());
        self.write_size_header(writer)?;
        crate::trace::debug_eprintln!("FHDR [bit {}]: After size header", writer.bits_written());
        self.write_image_metadata(writer)?;
        crate::trace::debug_eprintln!("FHDR [bit {}]: After metadata", writer.bits_written());
        // CustomTransformData - written after ImageMetadata
        // For simple images, all_default = true (just 1 bit)
        self.write_transform_data(writer)?;
        crate::trace::debug_eprintln!("FHDR [bit {}]: After transform_data", writer.bits_written());
        Ok(())
    }

    /// Writes the CustomTransformData bundle.
    ///
    /// Wire format (jxl-rs `CustomTransformData`,
    /// libjxl `image_metadata.cc:VisitFields`):
    /// 1. `all_default` (1 bit).
    /// 2. If `!all_default && xyb_encoded`: `OpsinInverseMatrix` block.
    ///    We always emit `OpsinInverseMatrix.all_default = true` (1 bit)
    ///    because our encoder does not yet expose a non-default opsin
    ///    matrix.
    /// 3. `custom_weights_mask` (3 bits).
    /// 4. If `(mask & 1)`: 15 F16 weights for upsampling factor 2.
    /// 5. If `(mask & 2)`: 55 F16 weights for upsampling factor 4.
    /// 6. If `(mask & 4)`: 210 F16 weights for upsampling factor 8.
    ///
    /// We take the all_default fast path (1 bit) when
    /// [`Self::upsampling_mode`] is `None` / `Some(-1)`, OR when
    /// [`Self::upsampling_factor`] <= 1. In that case the decoder picks
    /// the default fancy-upsampling kernels.
    fn write_transform_data(&self, writer: &mut BitWriter) -> Result<()> {
        let factor = self.upsampling_factor;
        let mode = self.upsampling_mode.unwrap_or(-1);
        let emit_custom = factor > 1 && mode >= 0 && matches!(factor, 2 | 4 | 8);
        if !emit_custom {
            crate::trace::debug_eprintln!(
                "XFRM [bit {}]: transform_data.all_default = true",
                writer.bits_written()
            );
            writer.write_bit(true)?;
            return Ok(());
        }

        // !all_default
        crate::trace::debug_eprintln!(
            "XFRM [bit {}]: transform_data.all_default = false (custom upsampling LUT)",
            writer.bits_written()
        );
        writer.write_bit(false)?;

        // OpsinInverseMatrix (conditional on xyb_encoded) — always
        // default for now.
        if self.metadata.xyb_encoded {
            crate::trace::debug_eprintln!(
                "XFRM [bit {}]: opsin_inverse_matrix.all_default = true",
                writer.bits_written()
            );
            writer.write_bit(true)?;
        }

        // custom_weights_mask (3 bits). Only the bit matching the
        // active factor flips on; the other factors stay default.
        let mask_bit: u32 = match factor {
            2 => 1,
            4 => 2,
            8 => 4,
            _ => 0,
        };
        crate::trace::debug_eprintln!(
            "XFRM [bit {}]: custom_weights_mask = {}",
            writer.bits_written(),
            mask_bit
        );
        writer.write(3, mask_bit as u64)?;

        // Compute the requested weights as f32, then emit as F16.
        let weights = upsampling_lut_weights(factor, mode);
        for w in weights {
            crate::f16::write_f16(w, writer)?;
        }
        Ok(())
    }

    /// Writes the image metadata.
    fn write_image_metadata(&self, writer: &mut BitWriter) -> Result<()> {
        let meta = &self.metadata;

        // all_default flag
        let all_default = self.is_metadata_default();
        crate::trace::debug_eprintln!(
            "META [bit {}]: all_default = {}",
            writer.bits_written(),
            all_default
        );
        writer.write_bit(all_default)?;

        if all_default {
            return Ok(());
        }

        // extra_fields flag — any non-default field in the ImageMetadata
        // "extra_fields" cluster must trigger this (file_header.rs:485
        // and jxl-rs `headers/image_metadata.rs:184-200`). The
        // ToneMapping sub-bundle has 4 fields; any non-default value on
        // any of them needs `extra_fields = true` so the decoder reads
        // the bundle.
        let extra_fields = meta.animation.is_some()
            || meta.orientation != Orientation::Identity
            || meta.have_intrinsic_size
            || meta.intensity_target != 255.0
            || meta.min_nits != 0.0
            || meta.relative_to_max_display
            || meta.linear_below != 0.0;
        crate::trace::debug_eprintln!(
            "META [bit {}]: extra_fields = {}",
            writer.bits_written(),
            extra_fields
        );
        writer.write_bit(extra_fields)?;

        if extra_fields {
            // orientation - 1 (3 bits)
            writer.write(3, (meta.orientation as u8 - 1) as u64)?;

            // have_intrinsic_size
            writer.write_bit(meta.have_intrinsic_size)?;
            if meta.have_intrinsic_size {
                // Intrinsic size uses same u2S encoding as Size
                self.write_size_u2s(writer, meta.intrinsic_width - 1)?;
                self.write_size_u2s(writer, meta.intrinsic_height - 1)?;
            }

            // have_preview (not implemented)
            writer.write_bit(false)?;

            // have_animation
            writer.write_bit(meta.animation.is_some())?;
            if let Some(ref anim) = meta.animation {
                anim.write(writer)?;
            }
        }

        // bit_depth
        crate::trace::debug_eprintln!("META [bit {}]: Writing bit_depth", writer.bits_written());
        meta.bit_depth.write(writer)?;
        crate::trace::debug_eprintln!("META [bit {}]: After bit_depth", writer.bits_written());

        // modular_16_bit_buffer_sufficient
        // Default is true for bit depths <= 12, BUT the VarDCT encoder forces it
        // false when the quantized DC exceeds i16 (#94): a decoder honouring the
        // "true" promise reconstructs the LF/DC modular image into i16 buffers,
        // where oversized DC wraps and desynchronises the DC ANS stream.
        let mod16_sufficient = meta.bit_depth.bits_per_sample <= 12 && !meta.force_modular_32bit;
        crate::trace::debug_eprintln!(
            "META [bit {}]: modular_16_bit_buffer_sufficient = {}",
            writer.bits_written(),
            mod16_sufficient
        );
        writer.write_bit(mod16_sufficient)?;

        // num_extra_channels — per jxl-rs ImageMetadata
        // `extra_channel_info: Vec<...>` with
        // `#[size_coder(implicit(u2S(0, 1, Bits(4) + 2, Bits(12) + 1)))]`.
        // Selectors:
        //   0 → Val(0)        (2 bits)
        //   1 → Val(1)        (2 bits)
        //   2 → Bits(4) + 2   (2 + 4 bits, range 2..=17)
        //   3 → Bits(12) + 1  (2 + 12 bits, range 1..)
        // The previous call `write_u32_coder(n, 0, 1, 2, 1, 12)`
        // emitted selector 2 = Val(2) — wrong for n >= 2 (2 bits
        // instead of 6) which shifted every subsequent header field
        // and broke 2+ extra-channel decodes (refs #9). Fix mirrors
        // libjxl + jxl-rs spec exactly.
        let num_extra = meta.extra_channels.len() as u32;
        crate::trace::debug_eprintln!(
            "META [bit {}]: num_extra_channels = {}",
            writer.bits_written(),
            num_extra
        );
        if num_extra == 0 {
            writer.write(2, 0)?;
        } else if num_extra == 1 {
            writer.write(2, 1)?;
        } else if num_extra <= 17 {
            // selector 2: Bits(4) + 2 — values 2..=17
            writer.write(2, 2)?;
            writer.write(4, (num_extra - 2) as u64)?;
        } else {
            // selector 3: Bits(12) + 1 — values 1..=4096; we never
            // emit this for <= 17 since selector 2 covers it.
            debug_assert!((1..=4096).contains(&num_extra));
            writer.write(2, 3)?;
            writer.write(12, (num_extra - 1) as u64)?;
        }

        for ec in &meta.extra_channels {
            ec.write(writer)?;
        }

        // xyb_encoded (true for lossy, false for lossless)
        crate::trace::debug_eprintln!(
            "META [bit {}]: xyb_encoded = {}",
            writer.bits_written(),
            meta.xyb_encoded
        );
        writer.write_bit(meta.xyb_encoded)?;

        // color_encoding
        crate::trace::debug_eprintln!(
            "META [bit {}]: Writing color_encoding",
            writer.bits_written()
        );
        meta.color_encoding
            .write_with_spec_default_fast_path(writer, self.header_all_default_fast_paths)?;
        crate::trace::debug_eprintln!("META [bit {}]: After color_encoding", writer.bits_written());

        // tone_mapping - only if extra_fields
        //
        // ToneMapping is an `all_default`-gated bundle (jxl-rs
        // `headers/image_metadata.rs:139-150`): the bundle short-circuits
        // when every field is at its spec default
        // (intensity_target=255.0, min_nits=0.0,
        // relative_to_max_display=false, linear_below=0.0). When any
        // field differs, the encoder writes `all_default=0` followed by
        // the four field values in spec order. Closes issue #46 chunk
        // 1a: `relative_to_max_display` and `linear_below` are no
        // longer hardcoded to the default.
        if extra_fields {
            let tone_all_default = meta.intensity_target == 255.0
                && meta.min_nits == 0.0
                && !meta.relative_to_max_display
                && meta.linear_below == 0.0;
            writer.write_bit(tone_all_default)?;
            if !tone_all_default {
                crate::f16::write_f16(meta.intensity_target, writer)?;
                crate::f16::write_f16(meta.min_nits, writer)?;
                writer.write_bit(meta.relative_to_max_display)?;
                crate::f16::write_f16(meta.linear_below, writer)?;
            }
        }

        // extensions (u64 selector, 0 = no extensions)
        // u64 encoding: 2-bit selector, 0 means value 0
        writer.write(2, 0)?;

        Ok(())
    }

    /// Whether every serialized `ImageMetadata` field is at its JXL
    /// spec default, so the encoder may write `all_default = 1` (one
    /// bit) instead of spelling the whole block out (27 bits for the
    /// common 8-bit sRGB no-alpha lossy case).
    ///
    /// Spec defaults, read from libjxl `d089091a`
    /// `image_metadata.cc::ImageMetadata::VisitFields` +
    /// `color_encoding_internal.cc::ColorEncoding::VisitFields` (their
    /// comment there: *"we set the defaults to the most common values
    /// so ImageMetadata.all_default is true in the common case"*):
    ///
    /// - `extra_fields = false` — orientation identity, no intrinsic
    ///   size, no preview, no animation, and `ToneMapping` all-default
    ///   (`intensity_target = 255`, `min_nits = 0`,
    ///   `relative_to_max_display = false`, `linear_below = 0`)
    /// - `bit_depth` = 8-bit integer
    /// - `modular_16_bit_buffer_sufficient = true`
    /// - `num_extra_channels = 0`
    /// - `xyb_encoded = true` *(spec default — NOT false; an earlier
    ///   comment here had this backwards, and our own
    ///   `ImageMetadata::default()` differs from the SPEC default on
    ///   exactly this field, which is why this predicate tests the spec
    ///   values field by field rather than comparing against
    ///   `Default::default()`)*
    /// - `color_encoding` = sRGB defaults with `want_icc = false`
    /// - no extensions
    ///
    /// Implication: lossless modular encodes need `xyb_encoded = false`
    /// and so can never take this path; lossy VarDCT sRGB / 8-bit /
    /// no-alpha / no-metadata-deviation encodes always can.
    ///
    /// **Gated on [`Self::header_all_default_fast_paths`], which is set
    /// only under [`crate::api::EncoderStrategy::Libjxl`].** The gate is
    /// not about correctness — the two forms decode identically and the
    /// short one is strictly 27 bits smaller. It exists because flipping
    /// it moves every zen-mode hash lock, a re-bake that needs owner
    /// sign-off. See the `ImageMetadata.all_default` row in
    /// `docs/LIBJXL_DIVERGENCES.md` Section D.
    fn is_metadata_default(&self) -> bool {
        if !self.header_all_default_fast_paths {
            return false;
        }
        let m = &self.metadata;
        let tone_mapping_default = m.intensity_target == 255.0
            && m.min_nits == 0.0
            && !m.relative_to_max_display
            && m.linear_below == 0.0;
        let extra_fields = m.animation.is_some()
            || m.orientation != Orientation::Identity
            || m.have_intrinsic_size
            || !tone_mapping_default;
        if extra_fields {
            return false;
        }
        if m.bit_depth.float_sample
            || m.bit_depth.bits_per_sample != 8
            || m.bit_depth.exponent_bits != 0
        {
            return false;
        }
        // `modular_16_bit_buffer_sufficient` is written as
        // `bits_per_sample <= 12 && !force_modular_32bit`; the spec
        // default is `true`, so #94's 32-bit escape hatch forces the
        // long form.
        if m.force_modular_32bit {
            return false;
        }
        if !m.extra_channels.is_empty() {
            return false;
        }
        if !m.xyb_encoded {
            return false;
        }
        m.color_encoding.is_spec_default()
    }
}

/// Build the upsampling LUT weights for the given factor + mode.
///
/// Mirrors libjxl `JxlEncoderSetUpsamplingMode` (encode.cc:1393).
///
/// - `factor` must be `2`, `4`, or `8` (caller has already gated).
/// - `mode == 0`: nearest-neighbour. Single 1.0 impulse at the LUT
///   slot the libjxl table picks for the factor; everything else 0.
/// - `mode == 1`: "pixel dots" — starts from nearest then zeros /
///   halves a small set of slots. For factor 2 the pixel-dots branch
///   is empty in libjxl (no edits), so the LUT degenerates to nearest;
///   we preserve that behaviour for byte parity.
/// - Any other mode: returns the nearest LUT (caller is responsible
///   for rejecting out-of-range modes upstream).
///
/// LUT lengths from libjxl:
///   factor 2 → 15 weights
///   factor 4 → 55 weights
///   factor 8 → 210 weights
fn upsampling_lut_weights(factor: u32, mode: i32) -> Vec<f32> {
    let count = match factor {
        2 => 15usize,
        4 => 55usize,
        8 => 210usize,
        _ => return Vec::new(),
    };
    let mut w = vec![0f32; count];
    // Nearest-neighbour base (mode 0): single 1.0 impulse(s)
    // matching libjxl encode.cc:1417-1429.
    match factor {
        2 => {
            w[9] = 1.0;
        }
        4 => {
            for &i in &[19usize, 24, 49] {
                w[i] = 1.0;
            }
        }
        8 => {
            for &i in &[39usize, 44, 49, 54, 119, 124, 129, 174, 179, 204] {
                w[i] = 1.0;
            }
        }
        _ => {}
    }
    // Pixel-dots mode adjustments (libjxl encode.cc:1430-1439).
    // Builds on top of nearest. For factor 2 the table is empty,
    // i.e. pixel-dots == nearest.
    if mode == 1 {
        match factor {
            4 => {
                w[19] = 0.0;
                w[24] = 0.5;
            }
            8 => {
                for &i in &[39usize, 44, 49, 119] {
                    w[i] = 0.0;
                }
                for &i in &[54usize, 124] {
                    w[i] = 0.5;
                }
            }
            _ => {}
        }
    }
    w
}

impl BitDepth {
    /// Writes the bit depth to the bitstream.
    pub fn write(&self, writer: &mut BitWriter) -> Result<()> {
        writer.write_bit(self.float_sample)?;
        if self.float_sample {
            // bits_per_sample for float: u2S(32, 16, 24, 1 + Bits(6))
            writer.write_u32_coder(self.bits_per_sample, 32, 16, 24, 1, 6)?;
            // exponent_bits: 1 + Bits(4)
            writer.write(4, (self.exponent_bits - 1) as u64)?;
        } else {
            // bits_per_sample for int: u2S(8, 10, 12, 1 + Bits(6))
            writer.write_u32_coder(self.bits_per_sample, 8, 10, 12, 1, 6)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signature() {
        let mut writer = BitWriter::new();
        FileHeader::write_signature(&mut writer).unwrap();
        let bytes = writer.finish();
        assert_eq!(bytes, vec![0xFF, 0x0A]);
    }

    #[test]
    fn test_simple_header() {
        let header = FileHeader::new_rgb(256, 256);
        let mut writer = BitWriter::new();
        header.write(&mut writer).unwrap();

        let bytes = writer.finish_with_padding();
        // Should start with JXL signature
        assert_eq!(&bytes[0..2], &[0xFF, 0x0A]);
    }
}
