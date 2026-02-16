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

            if self.options.use_squeeze
                && !super::squeeze::default_squeeze_params(image).is_empty()
            {
                super::encode::write_modular_stream_with_squeeze(
                    image,
                    &mut section_writer,
                    self.options.use_ans,
                )?;
            } else if self.options.use_tree_learning && self.options.use_ans {
                write_modular_stream_with_tree(
                    image,
                    &mut section_writer,
                    256,                       // max_nodes
                    1.0,                       // split_threshold
                    image.channels.len() >= 3, // RCT for RGB
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
            // Multi-group with squeeze: apply global squeeze, partition channels
            self.encode_modular_multi_group_squeeze(image, writer)?;
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

        // Step 1: Extract each group image
        let mut group_images: Vec<ModularImage> = Vec::with_capacity(num_groups);
        for group_idx in 0..num_groups {
            let (x_start, y_start, x_end, y_end) = self.group_bounds(group_idx);
            let group_image = image.extract_region(x_start, y_start, x_end, y_end)?;
            group_images.push(group_image);
        }

        // Step 2: Write LfGlobal section (tree + histogram)
        let mut lf_global_writer = BitWriter::new();
        let global_state = if self.options.use_tree_learning && self.options.use_ans {
            // Tree learning path: gather samples, learn tree, build multi-context ANS
            write_global_modular_section_with_tree(
                &group_images,
                &mut lf_global_writer,
                256, // max_nodes
                1.0, // split_threshold
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
            write_gradient_tree_tokens, write_squeeze_transform, write_tree_histogram_for_gradient,
        };
        use super::predictor::pack_signed;
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

        // Step 1: Apply squeeze transform
        let squeeze_params = default_squeeze_params(image);
        let mut squeezed = image.clone();
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

        // GroupHeader for global modular stream — includes squeeze transform
        lf_global_writer.write(1, 1)?; // use_global_tree = true
        lf_global_writer.write(1, 1)?; // wp_params.default_wp = true
        lf_global_writer.write(2, 1)?; // nb_transforms = 1
        write_squeeze_transform(&mut lf_global_writer, &squeeze_params)?;

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
