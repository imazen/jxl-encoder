// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Frame encoder - assembles complete JXL frames.

use super::channel::ModularImage;
use super::encode::{
    build_histogram_from_residuals, collect_all_residuals, write_global_modular_section,
    write_group_modular_section, write_improved_modular_stream, write_modular_stream_with_tree,
};
use super::section::write_global_modular_section_with_tree;
use crate::GROUP_DIM;
use crate::bit_writer::BitWriter;
use crate::error::Result;
use crate::headers::ColorEncoding;
use crate::headers::frame_header::{BlendMode, FrameCrop, FrameHeader};

/// Options for frame encoding.
#[derive(Debug, Clone)]
pub struct FrameEncoderOptions {
    /// Use modular mode (lossless).
    pub use_modular: bool,
    /// Effort level (1-10, higher = better compression, slower).
    pub effort: u8,
    /// Use ANS entropy coding instead of Huffman for modular.
    pub use_ans: bool,
    /// Use content-adaptive MA tree learning for modular encoding.
    pub use_tree_learning: bool,
    /// Use squeeze (Haar wavelet) transform for modular encoding.
    pub use_squeeze: bool,
    /// Whether this frame is part of an animation (enables duration field in header).
    pub have_animation: bool,
    /// Duration of this frame in ticks (only used when have_animation is true).
    pub duration: u32,
    /// Whether this is the last frame in the image/animation.
    pub is_last: bool,
    /// Optional crop rectangle for this frame (None = full frame).
    pub crop: Option<FrameCrop>,
}

impl Default for FrameEncoderOptions {
    fn default() -> Self {
        Self {
            use_modular: true, // Default to lossless
            effort: 7,
            use_ans: false,
            use_tree_learning: false,
            use_squeeze: false,
            have_animation: false,
            duration: 0,
            is_last: true,
            crop: None,
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
    #[allow(dead_code)]
    /// Number of extra channels (e.g., 1 for alpha).
    num_extra_channels: usize,
}

impl FrameEncoder {
    /// Creates a new frame encoder.
    pub fn new(width: usize, height: usize, options: FrameEncoderOptions) -> Self {
        Self {
            options,
            width,
            height,
            num_extra_channels: 0,
        }
    }

    /// Creates a new frame encoder with extra channel support.
    pub fn new_with_extra_channels(
        width: usize,
        height: usize,
        options: FrameEncoderOptions,
        num_extra_channels: usize,
    ) -> Self {
        Self {
            options,
            width,
            height,
            num_extra_channels,
        }
    }

    /// Encodes a modular image into a frame.
    pub fn encode_modular(
        &self,
        image: &ModularImage,
        _color_encoding: &ColorEncoding,
        writer: &mut BitWriter,
    ) -> Result<()> {
        // Compute num_extra_channels from image
        let num_extra_channels = if image.has_alpha { 1 } else { 0 };

        // Write frame header using unified FrameHeader
        {
            let mut fh = FrameHeader::lossless();
            fh.ec_upsampling = vec![1; num_extra_channels];
            fh.ec_blend_modes = vec![BlendMode::Replace; num_extra_channels];
            fh.have_animation = self.options.have_animation;
            fh.duration = self.options.duration;
            fh.is_last = self.options.is_last;
            if let Some(ref crop) = self.options.crop {
                fh.x0 = crop.x0;
                fh.y0 = crop.y0;
                fh.width = crop.width;
                fh.height = crop.height;
                fh.blend_mode = BlendMode::Replace;
                fh.blend_source = 1;
            }
            // For animation, save non-last frames to reference slot 1
            // so crop frames can composite onto the previous canvas.
            if self.options.have_animation && !self.options.is_last {
                fh.save_as_reference = 1;
            }
            fh.write(writer)?;
        }

        let num_groups = self.num_groups();

        if num_groups == 1 {
            // Single group: all sections combined into one TOC entry
            let mut section_writer = BitWriter::new();
            let has_squeeze = self.options.use_squeeze
                && !super::squeeze::default_squeeze_params(image).is_empty();

            if has_squeeze && self.options.use_tree_learning && self.options.use_ans {
                // Combined squeeze + tree learning: best compression
                super::encode::write_modular_stream_with_squeeze_and_tree(
                    image,
                    &mut section_writer,
                    self.options.effort,
                )?;
            } else if has_squeeze {
                // Squeeze without tree learning (lower effort levels)
                super::encode::write_modular_stream_with_squeeze(
                    image,
                    &mut section_writer,
                    self.options.use_ans,
                )?;
            } else if self.options.use_tree_learning
                && self.options.use_ans
                && super::palette::should_use_palette(image).is_none()
            {
                // Tree learning without squeeze: skip for images that benefit from palette
                // (palette + tree learning has a meta-channel encoding mismatch)
                write_modular_stream_with_tree(
                    image,
                    &mut section_writer,
                    self.options.effort,       // effort-dependent tree params
                    image.channels.len() >= 3, // RCT for RGB
                )?;
            } else if image.channels.len() >= 3 {
                super::encode::write_modular_stream_with_rct(
                    image,
                    &mut section_writer,
                    self.options.use_ans,
                )?;
            } else {
                write_improved_modular_stream(image, &mut section_writer, self.options.use_ans)?;
            }

            let section_data = section_writer.finish();
            let section_size = section_data.len();

            crate::trace::debug_eprintln!("FRAME_ENCODER: section_size = {} bytes", section_size);

            // Write TOC
            self.write_toc(writer, section_size)?;

            // Append section data (already byte-aligned)
            for byte in section_data {
                writer.write_u8(byte)?;
            }
        } else if self.options.use_squeeze
            && !super::squeeze::default_squeeze_params(image).is_empty()
        {
            if self.options.use_tree_learning && self.options.use_ans {
                // Multi-group with squeeze + tree learning: best compression
                self.encode_modular_multi_group_squeeze_with_tree(image, writer)?;
            } else {
                // Multi-group with squeeze: gradient predictor, single context
                self.encode_modular_multi_group_squeeze(image, writer)?;
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

        crate::trace::debug_eprintln!(
            "MULTI_GROUP: Encoding {}x{} image with {} groups, {} lf_groups",
            self.width,
            self.height,
            num_groups,
            num_lf_groups
        );

        // Step 0: Apply RCT (YCoCg) to full image for RGB before extracting groups
        let has_rct = image.channels.len() >= 3;
        let rct_type = if has_rct {
            Some(super::rct::RctType::YCOCG)
        } else {
            None
        };
        let transformed;
        let source_image = if has_rct {
            transformed = {
                let mut img = image.clone();
                super::rct::forward_rct(&mut img.channels, 0, super::rct::RctType::YCOCG)?;
                img
            };
            &transformed
        } else {
            image
        };

        // Step 1: Extract each group image (from RCT-transformed image if applicable)
        let mut group_images: Vec<ModularImage> = Vec::with_capacity(num_groups);
        for group_idx in 0..num_groups {
            let (x_start, y_start, x_end, y_end) = self.group_bounds(group_idx);
            let group_image = source_image.extract_region(x_start, y_start, x_end, y_end)?;
            group_images.push(group_image);
        }

        // Step 2: Write LfGlobal section (tree + histogram)
        let mut lf_global_writer = BitWriter::new();
        let global_state = if self.options.use_tree_learning && self.options.use_ans {
            // Tree learning path: gather samples, learn tree, build multi-context ANS
            write_global_modular_section_with_tree(
                &group_images,
                &mut lf_global_writer,
                self.options.effort, // effort-dependent tree params
                rct_type,
            )?
        } else {
            // Standard path: collect residuals with gradient predictor
            let mut all_residuals = Vec::new();
            let mut max_residual: u32 = 0;
            for group_image in &group_images {
                let (group_residuals, group_max) = collect_all_residuals(group_image);
                all_residuals.extend(group_residuals);
                max_residual = max_residual.max(group_max);
            }
            let (histogram, max_token) =
                build_histogram_from_residuals(&all_residuals, max_residual);

            crate::trace::debug_eprintln!(
                "MULTI_GROUP: {} total residuals, max_raw={}, max_token={}, {} unique tokens",
                all_residuals.len(),
                max_residual,
                max_token,
                histogram.iter().filter(|&&c| c > 0).count()
            );

            write_global_modular_section(
                &all_residuals,
                &histogram,
                max_token,
                &mut lf_global_writer,
                self.options.use_ans,
                rct_type,
            )?
        };
        let lf_global_data = lf_global_writer.finish();

        crate::trace::debug_eprintln!(
            "MULTI_GROUP: LfGlobal section = {} bytes",
            lf_global_data.len()
        );

        // Step 3: HfGlobal is empty for modular encoding (0 bytes)
        let hf_global_data: Vec<u8> = Vec::new();
        crate::trace::debug_eprintln!(
            "MULTI_GROUP: HfGlobal section = 0 bytes (empty for modular)"
        );

        // Step 4: LfGroup sections are empty for modular encoding
        let lf_group_data: Vec<Vec<u8>> = (0..num_lf_groups).map(|_| Vec::new()).collect();
        crate::trace::debug_eprintln!(
            "MULTI_GROUP: {} LfGroup sections = 0 bytes each (empty for modular)",
            num_lf_groups
        );

        // Step 5: Write each PassGroup's data (GroupHeader + pixel data)
        // Use the pre-extracted group_images to ensure residual consistency
        let mut pass_group_data: Vec<Vec<u8>> = Vec::with_capacity(num_groups * num_passes);
        for (group_idx, group_image) in group_images.iter().enumerate() {
            for _pass in 0..num_passes {
                let (_x_start, _y_start, _x_end, _y_end) = self.group_bounds(group_idx);

                crate::trace::debug_eprintln!(
                    "MULTI_GROUP: Group {} bounds ({}, {}) - ({}, {}), size {}x{}",
                    group_idx,
                    _x_start,
                    _y_start,
                    _x_end,
                    _y_end,
                    group_image.width(),
                    group_image.height()
                );

                let mut group_writer = BitWriter::new();
                write_group_modular_section(group_image, &global_state, &mut group_writer)?;
                pass_group_data.push(group_writer.finish());

                crate::trace::debug_eprintln!(
                    "MULTI_GROUP: PassGroup {} section = {} bytes",
                    group_idx,
                    pass_group_data.last().unwrap().len()
                );
            }
        }

        // Step 6: Collect all section sizes in correct order and write TOC
        // JXL spec order: LfGlobal, LfGroup[0..num_lf_groups], HfGlobal, PassGroup[0..num_groups*num_passes]
        // Note: LfGroup comes BEFORE HfGlobal!
        let mut section_sizes = Vec::with_capacity(2 + num_lf_groups + num_groups * num_passes);
        section_sizes.push(lf_global_data.len());
        for data in &lf_group_data {
            section_sizes.push(data.len());
        }
        section_sizes.push(hf_global_data.len());
        for data in &pass_group_data {
            section_sizes.push(data.len());
        }

        crate::trace::debug_eprintln!(
            "MULTI_GROUP: {} total sections, sizes = {:?}",
            section_sizes.len(),
            section_sizes
        );

        self.write_toc_multi(writer, &section_sizes)?;

        // Step 7: Append all section data in same order
        for byte in lf_global_data {
            writer.write_u8(byte)?;
        }
        for data in lf_group_data {
            for byte in data {
                writer.write_u8(byte)?;
            }
        }
        for byte in hf_global_data {
            writer.write_u8(byte)?;
        }
        for data in pass_group_data {
            for byte in data {
                writer.write_u8(byte)?;
            }
        }

        Ok(())
    }

    /// Encodes a modular image using multi-group format with squeeze (Haar wavelet) transform.
    ///
    /// After squeeze, channels are partitioned by resolution:
    /// - **LfGlobal**: channels small enough to fit in GROUP_DIM (tree + histogram + data)
    /// - **LfGroup**: channels with min(hshift, vshift) >= 3 (DC-group-sized regions)
    /// - **PassGroup**: channels with min(hshift, vshift) < 3 (group-sized regions)
    fn encode_modular_multi_group_squeeze(
        &self,
        image: &ModularImage,
        writer: &mut BitWriter,
    ) -> Result<()> {
        use super::encode::{
            write_gradient_tree_tokens, write_rct_transform, write_squeeze_transform,
            write_tree_histogram_for_gradient,
        };
        use super::predictor::pack_signed;
        use super::rct::{RctType, forward_rct};
        use super::squeeze::{apply_squeeze, default_squeeze_params};
        use crate::entropy_coding::encode::{build_entropy_code_ans, write_tokens_ans};
        use crate::entropy_coding::hybrid_uint::HybridUintConfig;
        use crate::entropy_coding::token::Token as AnsToken;

        const MODULAR_HYBRID_UINT: HybridUintConfig = HybridUintConfig {
            split_exponent: 4,
            split: 16,
            msb_in_token: 2,
            lsb_in_token: 0,
        };

        let num_groups = self.num_groups();
        let num_lf_groups = self.num_lf_groups();
        let lf_group_dim = GROUP_DIM * 8; // 2048

        // Step 1: Apply RCT (YCoCg) before squeeze for RGB images, then squeeze
        let squeeze_params = default_squeeze_params(image);
        let mut squeezed = image.clone();
        let has_rct = squeezed.channels.len() >= 3;
        if has_rct {
            forward_rct(&mut squeezed.channels, 0, RctType::YCOCG)?;
        }
        apply_squeeze(&mut squeezed, &squeeze_params)?;

        #[cfg(test)]
        {
            eprintln!(
                "SQUEEZE_MULTI: {} steps, {} → {} channels, image {}x{}",
                squeeze_params.len(),
                image.channels.len(),
                squeezed.channels.len(),
                self.width,
                self.height,
            );
            for (i, ch) in squeezed.channels.iter().enumerate() {
                eprintln!(
                    "  ch[{}]: {}x{} hshift={} vshift={} min_shift={}",
                    i,
                    ch.width(),
                    ch.height(),
                    ch.hshift,
                    ch.vshift,
                    ch.hshift.min(ch.vshift),
                );
            }
        }

        // Step 2: Partition channels by size/shift
        // Global channels: both dimensions <= GROUP_DIM
        let global_cutoff = squeezed
            .channels
            .iter()
            .position(|c| c.width() > GROUP_DIM || c.height() > GROUP_DIM)
            .unwrap_or(squeezed.channels.len());

        crate::trace::debug_eprintln!(
            "SQUEEZE_MULTI: {} global channels (<={}x{}), {} group channels",
            global_cutoff,
            GROUP_DIM,
            GROUP_DIM,
            squeezed.channels.len() - global_cutoff,
        );

        // Classify non-global channels by shift bracket
        // LfGroup: min(hshift, vshift) >= 3
        // PassGroup: min(hshift, vshift) < 3
        let mut lf_channel_indices: Vec<usize> = Vec::new();
        let mut pass_channel_indices: Vec<usize> = Vec::new();
        for i in global_cutoff..squeezed.channels.len() {
            let ch = &squeezed.channels[i];
            let min_shift = ch.hshift.min(ch.vshift);
            if min_shift >= 3 {
                lf_channel_indices.push(i);
            } else {
                pass_channel_indices.push(i);
            }
        }

        #[cfg(test)]
        eprintln!(
            "SQUEEZE_MULTI: {} global, {} LfGroup (shift>=3), {} PassGroup (shift<3) channels",
            global_cutoff,
            lf_channel_indices.len(),
            pass_channel_indices.len(),
        );

        // Step 3: Collect residuals from ALL channels for histogram building
        let predict_gradient = |left: i32, top: i32, topleft: i32| -> i32 {
            let grad = left + top - topleft;
            grad.clamp(left.min(top), left.max(top))
        };

        let collect_channel_residuals = |channel: &super::channel::Channel| -> Vec<u32> {
            let w = channel.width();
            let h = channel.height();
            let mut residuals = Vec::with_capacity(w * h);
            for y in 0..h {
                for x in 0..w {
                    let pixel = channel.get(x, y);
                    let left = if x > 0 { channel.get(x - 1, y) } else { 0 };
                    let top = if y > 0 { channel.get(x, y - 1) } else { left };
                    let topleft = if x > 0 && y > 0 {
                        channel.get(x - 1, y - 1)
                    } else {
                        left
                    };
                    let prediction = predict_gradient(left, top, topleft);
                    residuals.push(pack_signed(pixel - prediction));
                }
            }
            residuals
        };

        // 3a: Global channel residuals (full channels)
        let mut all_residuals: Vec<u32> = Vec::new();
        for i in 0..global_cutoff {
            all_residuals.extend(collect_channel_residuals(&squeezed.channels[i]));
        }

        // 3b: LfGroup channel residuals (cropped to each DC group rect)
        // Use extract_grid_cell matching decoder's get_grid_rect: computes regions
        // in channel space via grid_dim = (group_dim >> hshift, group_dim >> vshift).
        let num_lf_groups_x = self.width.div_ceil(lf_group_dim);
        let mut lf_group_channel_data: Vec<Vec<Vec<u32>>> = vec![Vec::new(); num_lf_groups]; // [lf_group_idx][channel_within_group] = residuals
        for &ch_idx in &lf_channel_indices {
            let ch = &squeezed.channels[ch_idx];
            for (lg, lg_channels) in lf_group_channel_data
                .iter_mut()
                .enumerate()
                .take(num_lf_groups)
            {
                let lg_x = lg % num_lf_groups_x;
                let lg_y = lg / num_lf_groups_x;
                if let Some(cropped) = ch.extract_grid_cell(lg_x, lg_y, lf_group_dim) {
                    let residuals = collect_channel_residuals(&cropped);
                    all_residuals.extend(&residuals);
                    lg_channels.push(residuals);
                }
            }
        }

        // 3c: PassGroup channel residuals (cropped to each group rect)
        // Use extract_grid_cell matching decoder's get_grid_rect logic.
        let num_groups_x = self.num_groups_x();
        let mut pass_group_channel_data: Vec<Vec<Vec<u32>>> = vec![Vec::new(); num_groups]; // [group_idx][channel_within_group] = residuals
        for &ch_idx in &pass_channel_indices {
            let ch = &squeezed.channels[ch_idx];
            for (g, g_channels) in pass_group_channel_data
                .iter_mut()
                .enumerate()
                .take(num_groups)
            {
                let gx = g % num_groups_x;
                let gy = g / num_groups_x;
                if let Some(cropped) = ch.extract_grid_cell(gx, gy, GROUP_DIM) {
                    let residuals = collect_channel_residuals(&cropped);
                    all_residuals.extend(&residuals);
                    g_channels.push(residuals);
                }
            }
        }

        // Step 4: Build histogram and entropy codes
        let mut max_token: u32 = 0;
        for &r in &all_residuals {
            let (token, _, _) = MODULAR_HYBRID_UINT.encode(r);
            max_token = max_token.max(token);
        }

        // Step 5: Write LfGlobal section
        let mut lf_global_writer = BitWriter::new();

        // dc_quant.all_default = true
        lf_global_writer.write(1, 1)?;
        // has_tree = true
        lf_global_writer.write(1, 1)?;

        // Tree histogram + tokens (gradient predictor)
        let (tree_depths, tree_codes) = write_tree_histogram_for_gradient(&mut lf_global_writer)?;
        write_gradient_tree_tokens(&mut lf_global_writer, &tree_depths, &tree_codes)?;

        // Data histogram (Huffman or ANS) — covers ALL channels across ALL sections
        let use_ans = self.options.use_ans;

        // Build the entropy coding state
        enum EntropyState {
            Huffman {
                depths: Vec<u8>,
                codes: Vec<u16>,
            },
            Ans {
                code: crate::entropy_coding::encode::OwnedAnsEntropyCode,
            },
        }

        let entropy_state = if use_ans {
            let tokens: Vec<AnsToken> =
                all_residuals.iter().map(|&r| AnsToken::new(0, r)).collect();
            let code = build_entropy_code_ans(&tokens, 1);
            super::section::write_ans_modular_header(&mut lf_global_writer, &code)?;
            EntropyState::Ans { code }
        } else {
            let histogram_size = (max_token + 1) as usize;
            let mut histogram = vec![0u32; histogram_size];
            for &r in &all_residuals {
                let (token, _, _) = MODULAR_HYBRID_UINT.encode(r);
                histogram[token as usize] += 1;
            }
            let (depths, codes) = super::encode::write_hybrid_data_histogram(
                &mut lf_global_writer,
                &histogram,
                max_token,
            )?;
            EntropyState::Huffman { depths, codes }
        };

        // GroupHeader for global modular stream — includes RCT (if RGB) + squeeze transform
        lf_global_writer.write(1, 1)?; // use_global_tree = true
        lf_global_writer.write(1, 1)?; // wp_params.default_wp = true
        if has_rct {
            // nb_transforms = 2: U32 BitsOffset(4,2), offset=0
            lf_global_writer.write(2, 2)?;
            lf_global_writer.write(4, 0)?;
            write_rct_transform(&mut lf_global_writer, 0, RctType::YCOCG)?;
            write_squeeze_transform(&mut lf_global_writer, &squeeze_params)?;
        } else {
            lf_global_writer.write(2, 1)?; // nb_transforms = 1
            write_squeeze_transform(&mut lf_global_writer, &squeeze_params)?;
        }

        // Encode global channel data (small channels that fit within GROUP_DIM)
        let encode_residuals =
            |residuals: &[u32], writer: &mut BitWriter, state: &EntropyState| -> Result<()> {
                match state {
                    EntropyState::Huffman { depths, codes } => {
                        for &r in residuals {
                            let (token, extra_bits, num_extra) = MODULAR_HYBRID_UINT.encode(r);
                            let depth = depths.get(token as usize).copied().unwrap_or(0);
                            let code = codes.get(token as usize).copied().unwrap_or(0);
                            if depth > 0 {
                                writer.write(depth as usize, code as u64)?;
                            }
                            if num_extra > 0 {
                                writer.write(num_extra as usize, extra_bits as u64)?;
                            }
                        }
                    }
                    EntropyState::Ans { code } => {
                        let tokens: Vec<AnsToken> =
                            residuals.iter().map(|&r| AnsToken::new(0, r)).collect();
                        write_tokens_ans(&tokens, code, None, writer)?;
                    }
                }
                Ok(())
            };

        // Write global channel residuals
        let mut global_residuals: Vec<u32> = Vec::new();
        for i in 0..global_cutoff {
            global_residuals.extend(collect_channel_residuals(&squeezed.channels[i]));
        }
        encode_residuals(&global_residuals, &mut lf_global_writer, &entropy_state)?;

        lf_global_writer.zero_pad_to_byte();
        let lf_global_data = lf_global_writer.finish();

        crate::trace::debug_eprintln!(
            "SQUEEZE_MULTI: LfGlobal = {} bytes ({} global channels)",
            lf_global_data.len(),
            global_cutoff,
        );

        // Step 6: Write LfGroup sections
        let mut lf_group_data: Vec<Vec<u8>> = Vec::with_capacity(num_lf_groups);
        for (_lg, lg_channels) in lf_group_channel_data.iter().enumerate().take(num_lf_groups) {
            let mut lg_writer = BitWriter::new();

            if lg_channels.is_empty() {
                // Empty LfGroup (no channels assigned)
                lf_group_data.push(lg_writer.finish());
                continue;
            }

            // GroupHeader
            lg_writer.write(1, 1)?; // use_global_tree = true
            lg_writer.write(1, 1)?; // wp_params.default_wp = true
            lg_writer.write(2, 0)?; // nb_transforms = 0

            // Concatenate all channel residuals for this section, then encode once.
            // ANS requires a single encoder per section (one ANS state per section).
            let mut section_residuals: Vec<u32> = Vec::new();
            for channel_residuals in lg_channels {
                section_residuals.extend(channel_residuals);
            }
            encode_residuals(&section_residuals, &mut lg_writer, &entropy_state)?;

            lg_writer.zero_pad_to_byte();
            let data = lg_writer.finish();
            crate::trace::debug_eprintln!(
                "SQUEEZE_MULTI: LfGroup[{}] = {} bytes ({} channels)",
                _lg,
                data.len(),
                lg_channels.len(),
            );
            lf_group_data.push(data);
        }

        // Step 7: HfGlobal is empty for modular
        let hf_global_data: Vec<u8> = Vec::new();

        // Step 8: Write PassGroup sections
        let mut pass_group_data: Vec<Vec<u8>> = Vec::with_capacity(num_groups);
        for (_g, g_channels) in pass_group_channel_data.iter().enumerate().take(num_groups) {
            let mut pg_writer = BitWriter::new();

            if g_channels.is_empty() {
                // Empty PassGroup (no channels assigned)
                pass_group_data.push(pg_writer.finish());
                continue;
            }

            // GroupHeader
            pg_writer.write(1, 1)?; // use_global_tree = true
            pg_writer.write(1, 1)?; // wp_params.default_wp = true
            pg_writer.write(2, 0)?; // nb_transforms = 0

            // Concatenate all channel residuals for this section, then encode once.
            // ANS requires a single encoder per section (one ANS state per section).
            let mut section_residuals: Vec<u32> = Vec::new();
            for channel_residuals in g_channels {
                section_residuals.extend(channel_residuals);
            }
            encode_residuals(&section_residuals, &mut pg_writer, &entropy_state)?;

            pg_writer.zero_pad_to_byte();
            let data = pg_writer.finish();
            crate::trace::debug_eprintln!(
                "SQUEEZE_MULTI: PassGroup[{}] = {} bytes ({} channels)",
                _g,
                data.len(),
                g_channels.len(),
            );
            pass_group_data.push(data);
        }

        // Step 9: Assemble TOC and sections
        // Section order: LfGlobal, LfGroup[0..n], HfGlobal, PassGroup[0..m]
        let mut section_sizes = Vec::with_capacity(2 + num_lf_groups + num_groups);
        section_sizes.push(lf_global_data.len());
        for data in &lf_group_data {
            section_sizes.push(data.len());
        }
        section_sizes.push(hf_global_data.len());
        for data in &pass_group_data {
            section_sizes.push(data.len());
        }

        #[cfg(test)]
        eprintln!(
            "SQUEEZE_MULTI: {} sections, sizes = {:?}",
            section_sizes.len(),
            section_sizes,
        );

        self.write_toc_multi(writer, &section_sizes)?;

        // Write all section data
        for byte in lf_global_data {
            writer.write_u8(byte)?;
        }
        for data in lf_group_data {
            for byte in data {
                writer.write_u8(byte)?;
            }
        }
        for byte in hf_global_data {
            writer.write_u8(byte)?;
        }
        for data in pass_group_data {
            for byte in data {
                writer.write_u8(byte)?;
            }
        }

        Ok(())
    }

    /// Encodes a multi-group modular image with squeeze + tree learning.
    ///
    /// This combines the Haar wavelet (squeeze) transform with learned MA tree
    /// for multi-context ANS encoding across all sections. The tree is learned
    /// from the full squeezed image and shared across all sections.
    ///
    /// Pipeline: RCT -> squeeze -> partition channels -> gather samples ->
    /// learn tree -> collect residuals per section -> multi-context ANS
    fn encode_modular_multi_group_squeeze_with_tree(
        &self,
        image: &ModularImage,
        writer: &mut BitWriter,
    ) -> Result<()> {
        use super::encode::{write_rct_transform, write_squeeze_transform, write_tree};
        use super::rct::{RctType, forward_rct};
        use super::squeeze::{apply_squeeze, default_squeeze_params};
        use super::tree::count_contexts;
        use super::tree_learn::{
            TreeLearningParams, TreeSamples, collect_residuals_with_tree, compute_best_tree,
            compute_gather_stride, gather_samples_strided,
        };
        use crate::entropy_coding::encode::{
            build_entropy_code_ans, write_entropy_code_ans, write_tokens_ans,
        };
        use crate::entropy_coding::token::Token as AnsToken;

        let num_groups = self.num_groups();
        let num_lf_groups = self.num_lf_groups();
        let lf_group_dim = GROUP_DIM * 8; // 2048

        // Step 1: Apply RCT (YCoCg) before squeeze for RGB images, then squeeze
        let squeeze_params = default_squeeze_params(image);
        let mut squeezed = image.clone();
        let has_rct = squeezed.channels.len() >= 3;
        if has_rct {
            forward_rct(&mut squeezed.channels, 0, RctType::YCOCG)?;
        }
        apply_squeeze(&mut squeezed, &squeeze_params)?;

        crate::trace::debug_eprintln!(
            "SQUEEZE_TREE_MULTI: {} steps, {} → {} channels, image {}x{}",
            squeeze_params.len(),
            image.channels.len(),
            squeezed.channels.len(),
            self.width,
            self.height,
        );

        // Step 2: Partition channels by size/shift
        let global_cutoff = squeezed
            .channels
            .iter()
            .position(|c| c.width() > GROUP_DIM || c.height() > GROUP_DIM)
            .unwrap_or(squeezed.channels.len());

        let mut lf_channel_indices: Vec<usize> = Vec::new();
        let mut pass_channel_indices: Vec<usize> = Vec::new();
        for i in global_cutoff..squeezed.channels.len() {
            let ch = &squeezed.channels[i];
            let min_shift = ch.hshift.min(ch.vshift);
            if min_shift >= 3 {
                lf_channel_indices.push(i);
            } else {
                pass_channel_indices.push(i);
            }
        }

        crate::trace::debug_eprintln!(
            "SQUEEZE_TREE_MULTI: {} global, {} LfGroup, {} PassGroup channels",
            global_cutoff,
            lf_channel_indices.len(),
            pass_channel_indices.len(),
        );

        // Step 3: Build sub-images for each section and gather samples
        // Compute stride from total pixel count for subsampling
        let total_pixels: usize = squeezed
            .channels
            .iter()
            .map(|ch| ch.width() * ch.height())
            .sum();
        let stride = compute_gather_stride(total_pixels, self.options.effort);
        let mut samples = TreeSamples::new();

        // 3a: Global channels (full, no cropping needed)
        let global_sub = ModularImage {
            channels: squeezed.channels[..global_cutoff].to_vec(),
            bit_depth: squeezed.bit_depth,
            is_grayscale: squeezed.is_grayscale,
            has_alpha: false,
        };
        // group_id=0 for global section, channel_offset=0
        gather_samples_strided(&mut samples, &global_sub, 0, 0, stride);

        // 3b: LfGroup channels — crop to each LfGroup rect
        let num_lf_groups_x = self.width.div_ceil(lf_group_dim);
        // Store cropped sub-images: [lf_group_idx] = Vec<Channel>
        let mut lf_group_sub_images: Vec<Vec<super::channel::Channel>> =
            vec![Vec::new(); num_lf_groups];
        for &ch_idx in &lf_channel_indices {
            let ch = &squeezed.channels[ch_idx];
            for (lg, lg_channels) in lf_group_sub_images.iter_mut().enumerate() {
                let lg_x = lg % num_lf_groups_x;
                let lg_y = lg / num_lf_groups_x;
                if let Some(cropped) = ch.extract_grid_cell(lg_x, lg_y, lf_group_dim) {
                    lg_channels.push(cropped);
                }
            }
        }
        // Gather samples from LfGroup sub-images
        // The first LfGroup channel in the squeezed image is at lf_channel_indices[0],
        // but we don't need channel_offset here because LfGroup channels form a separate
        // sub-image with their own local channel indices for the decoder.
        // Actually — the decoder uses a SINGLE tree across all sections. Property[0] = channel
        // index within the sub-image (modular stream). For squeeze multi-group, the decoder
        // reconstructs each section as a separate modular sub-image. The global section has
        // channels 0..gc, each LfGroup section has its own channels (starting from 0),
        // and each PassGroup section has its own channels (starting from 0).
        //
        // BUT — the tree was trained on the full image where these had specific channel indices.
        // The decoder assigns local indices per-section. So we need channel_offset to map:
        //   - Global: channels 0..gc → offset 0 (correct by default)
        //   - LfGroup: decoder's ch[0..n] → should map to squeezed ch[lf_channel_indices[0]..end]
        //   - PassGroup: decoder's ch[0..n] → should map to squeezed ch[pass_channel_indices[0]..end]
        //
        // Wait — I need to verify what the decoder actually does. Let me think about this more carefully.
        //
        // In JXL multi-group modular, the decoder processes:
        //   1. LfGlobal: reads tree, histograms, then decodes channels 0..gc using the tree
        //   2. LfGroup[i]: each section is a new modular stream with its own GroupHeader.
        //      The decoder maps these channels to the overall image by shift classification.
        //      Within each section, channels start from index 0.
        //   3. PassGroup[i]: same — channels start from index 0 within each section.
        //
        // The tree's property[0] (channel index) sees 0..n within each section.
        // So for tree learning to work correctly across sections, we should:
        //   - Either use a channel_offset to remap local indices to global indices (what PLAN.md suggests)
        //   - Or accept that the tree splits on local channel indices (may be less optimal)
        //
        // libjxl's approach: the tree IS global, but each section's channels are numbered locally.
        // The tree learns to split on "channel 0 vs channel 1" etc. which means different things
        // in different sections. But typically LfGroup has 0 or 3 channels (one per color component)
        // and PassGroup has 3 channels too, so the splits transfer well.
        //
        // For simplicity and correctness: use local channel indices (no offset) with per-section
        // group_id to disambiguate. This matches how the decoder will traverse the tree.
        for (lg, lg_channels) in lf_group_sub_images.iter().enumerate() {
            if lg_channels.is_empty() {
                continue;
            }
            let sub_image = ModularImage {
                channels: lg_channels.clone(),
                bit_depth: squeezed.bit_depth,
                is_grayscale: squeezed.is_grayscale,
                has_alpha: false,
            };
            // group_id for LfGroup sections — offset by 1 to distinguish from global (0)
            gather_samples_strided(&mut samples, &sub_image, (lg + 1) as u32, 0, stride);
        }

        // 3c: PassGroup channels — crop to each group rect
        let num_groups_x = self.num_groups_x();
        let mut pass_group_sub_images: Vec<Vec<super::channel::Channel>> =
            vec![Vec::new(); num_groups];
        for &ch_idx in &pass_channel_indices {
            let ch = &squeezed.channels[ch_idx];
            for (g, g_channels) in pass_group_sub_images.iter_mut().enumerate() {
                let gx = g % num_groups_x;
                let gy = g / num_groups_x;
                if let Some(cropped) = ch.extract_grid_cell(gx, gy, GROUP_DIM) {
                    g_channels.push(cropped);
                }
            }
        }
        // Gather samples from PassGroup sub-images
        for (g, g_channels) in pass_group_sub_images.iter().enumerate() {
            if g_channels.is_empty() {
                continue;
            }
            let sub_image = ModularImage {
                channels: g_channels.clone(),
                bit_depth: squeezed.bit_depth,
                is_grayscale: squeezed.is_grayscale,
                has_alpha: false,
            };
            // group_id offset: after global (0) and LfGroups (1..num_lf_groups)
            gather_samples_strided(
                &mut samples,
                &sub_image,
                (num_lf_groups + 1 + g) as u32,
                0,
                stride,
            );
        }

        // Step 4: Learn tree
        let pixel_fraction = if total_pixels > 0 {
            samples.num_samples as f64 / total_pixels as f64
        } else {
            1.0
        };
        let tree_params =
            TreeLearningParams::for_effort(self.options.effort).with_pixel_fraction(pixel_fraction);
        let tree = compute_best_tree(&mut samples, &tree_params);
        let num_contexts = count_contexts(&tree) as usize;

        crate::trace::debug_eprintln!(
            "SQUEEZE_TREE_MULTI: {} tree nodes, {} contexts from {} samples (pf={:.3})",
            tree.len(),
            num_contexts,
            samples.num_samples,
            pixel_fraction,
        );

        // Step 5: Collect residuals per section with the learned tree
        // Global section tokens
        let global_tokens = collect_residuals_with_tree(&global_sub, &tree, 0);

        // LfGroup section tokens
        let mut lf_group_tokens: Vec<Vec<AnsToken>> = Vec::with_capacity(num_lf_groups);
        for (lg, lg_channels) in lf_group_sub_images.iter().enumerate() {
            if lg_channels.is_empty() {
                lf_group_tokens.push(Vec::new());
                continue;
            }
            let sub_image = ModularImage {
                channels: lg_channels.clone(),
                bit_depth: squeezed.bit_depth,
                is_grayscale: squeezed.is_grayscale,
                has_alpha: false,
            };
            let tokens = collect_residuals_with_tree(&sub_image, &tree, (lg + 1) as u32);
            lf_group_tokens.push(tokens);
        }

        // PassGroup section tokens
        let mut pass_group_tokens: Vec<Vec<AnsToken>> = Vec::with_capacity(num_groups);
        for (g, g_channels) in pass_group_sub_images.iter().enumerate() {
            if g_channels.is_empty() {
                pass_group_tokens.push(Vec::new());
                continue;
            }
            let sub_image = ModularImage {
                channels: g_channels.clone(),
                bit_depth: squeezed.bit_depth,
                is_grayscale: squeezed.is_grayscale,
                has_alpha: false,
            };
            let tokens =
                collect_residuals_with_tree(&sub_image, &tree, (num_lf_groups + 1 + g) as u32);
            pass_group_tokens.push(tokens);
        }

        // Step 6: Build ANS codes from ALL tokens
        let mut all_tokens: Vec<AnsToken> = Vec::new();
        all_tokens.extend(&global_tokens);
        for lg_tokens in &lf_group_tokens {
            all_tokens.extend(lg_tokens);
        }
        for pg_tokens in &pass_group_tokens {
            all_tokens.extend(pg_tokens);
        }
        let code = build_entropy_code_ans(&all_tokens, num_contexts);

        // Step 7: Write LfGlobal section
        let mut lf_global_writer = BitWriter::new();

        // dc_quant.all_default = true
        lf_global_writer.write(1, 1)?;
        // has_tree = true
        lf_global_writer.write(1, 1)?;

        // Write the learned tree
        write_tree(&mut lf_global_writer, &tree)?;

        // Write ANS histogram (multi-context if num_contexts > 1)
        if num_contexts > 1 {
            lf_global_writer.write(1, 0)?; // lz77.enabled = 0
            write_entropy_code_ans(&code, &mut lf_global_writer)?;
        } else {
            super::section::write_ans_modular_header(&mut lf_global_writer, &code)?;
        }

        // GroupHeader for global modular stream — includes RCT (if RGB) + squeeze transform
        lf_global_writer.write(1, 1)?; // use_global_tree = true
        lf_global_writer.write(1, 1)?; // wp_params.default_wp = true
        if has_rct {
            // nb_transforms = 2: U32 BitsOffset(4,2), offset=0
            lf_global_writer.write(2, 2)?;
            lf_global_writer.write(4, 0)?;
            write_rct_transform(&mut lf_global_writer, 0, RctType::YCOCG)?;
            write_squeeze_transform(&mut lf_global_writer, &squeeze_params)?;
        } else {
            lf_global_writer.write(2, 1)?; // nb_transforms = 1
            write_squeeze_transform(&mut lf_global_writer, &squeeze_params)?;
        }

        // Write global channel tokens
        write_tokens_ans(&global_tokens, &code, None, &mut lf_global_writer)?;

        lf_global_writer.zero_pad_to_byte();
        let lf_global_data = lf_global_writer.finish();

        crate::trace::debug_eprintln!(
            "SQUEEZE_TREE_MULTI: LfGlobal = {} bytes ({} global channels, {} contexts)",
            lf_global_data.len(),
            global_cutoff,
            num_contexts,
        );

        // Step 8: Write LfGroup sections
        let mut lf_group_data: Vec<Vec<u8>> = Vec::with_capacity(num_lf_groups);
        for lg_tokens in &lf_group_tokens {
            let mut lg_writer = BitWriter::new();

            if lg_tokens.is_empty() {
                lf_group_data.push(lg_writer.finish());
                continue;
            }

            // GroupHeader
            lg_writer.write(1, 1)?; // use_global_tree = true
            lg_writer.write(1, 1)?; // wp_params.default_wp = true
            lg_writer.write(2, 0)?; // nb_transforms = 0

            write_tokens_ans(lg_tokens, &code, None, &mut lg_writer)?;

            lg_writer.zero_pad_to_byte();
            let data = lg_writer.finish();
            crate::trace::debug_eprintln!(
                "SQUEEZE_TREE_MULTI: LfGroup = {} bytes ({} tokens)",
                data.len(),
                lg_tokens.len(),
            );
            lf_group_data.push(data);
        }

        // Step 9: HfGlobal is empty for modular
        let hf_global_data: Vec<u8> = Vec::new();

        // Step 10: Write PassGroup sections
        let mut pass_group_data: Vec<Vec<u8>> = Vec::with_capacity(num_groups);
        for pg_tokens in &pass_group_tokens {
            let mut pg_writer = BitWriter::new();

            if pg_tokens.is_empty() {
                pass_group_data.push(pg_writer.finish());
                continue;
            }

            // GroupHeader
            pg_writer.write(1, 1)?; // use_global_tree = true
            pg_writer.write(1, 1)?; // wp_params.default_wp = true
            pg_writer.write(2, 0)?; // nb_transforms = 0

            write_tokens_ans(pg_tokens, &code, None, &mut pg_writer)?;

            pg_writer.zero_pad_to_byte();
            let data = pg_writer.finish();
            crate::trace::debug_eprintln!(
                "SQUEEZE_TREE_MULTI: PassGroup = {} bytes ({} tokens)",
                data.len(),
                pg_tokens.len(),
            );
            pass_group_data.push(data);
        }

        // Step 11: Assemble TOC and sections
        let mut section_sizes = Vec::with_capacity(2 + num_lf_groups + num_groups);
        section_sizes.push(lf_global_data.len());
        for data in &lf_group_data {
            section_sizes.push(data.len());
        }
        section_sizes.push(hf_global_data.len());
        for data in &pass_group_data {
            section_sizes.push(data.len());
        }

        self.write_toc_multi(writer, &section_sizes)?;

        // Write all section data
        for byte in lf_global_data {
            writer.write_u8(byte)?;
        }
        for data in lf_group_data {
            for byte in data {
                writer.write_u8(byte)?;
            }
        }
        for byte in hf_global_data {
            writer.write_u8(byte)?;
        }
        for data in pass_group_data {
            for byte in data {
                writer.write_u8(byte)?;
            }
        }

        Ok(())
    }

    /// Writes the table of contents with a single section.
    fn write_toc(&self, writer: &mut BitWriter, section_size: usize) -> Result<()> {
        self.write_toc_multi(writer, &[section_size])
    }

    /// Writes the table of contents with multiple sections.
    fn write_toc_multi(&self, writer: &mut BitWriter, section_sizes: &[usize]) -> Result<()> {
        crate::trace::debug_eprintln!("TOC [bit {}]: Writing permuted = 0", writer.bits_written());
        // permuted = false
        writer.write(1, 0)?;

        crate::trace::debug_eprintln!(
            "TOC [bit {}]: After permuted, byte aligning",
            writer.bits_written()
        );
        // Byte align before TOC entries (permutation reads, then aligns)
        writer.zero_pad_to_byte();

        // Write TOC entries using u2S(Bits(10), Bits(14)+1024, Bits(22)+17408, Bits(30)+4211712)
        #[allow(clippy::unused_enumerate_index)]
        for (_i, &size) in section_sizes.iter().enumerate() {
            crate::trace::debug_eprintln!(
                "TOC [bit {}]: Writing entry {} size={}",
                writer.bits_written(),
                _i,
                size
            );
            self.write_toc_entry(writer, size as u32)?;
        }
        crate::trace::debug_eprintln!("TOC [bit {}]: After TOC entries", writer.bits_written());

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
        crate::trace::debug_eprintln!("Multi-group modular: {} bytes", bytes.len());
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
