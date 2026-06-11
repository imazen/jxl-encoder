// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Frame encoder - assembles complete JXL frames.

use super::channel::{Channel, ModularImage};
use super::encode::{
    build_histogram_from_residuals, select_best_rct_at, write_group_modular_section_idx,
};
use super::palette::{CHANNEL_COLORS_PERCENT, analyze_channel_compact};
use crate::bit_writer::BitWriter;
use crate::entropy_coding::lz77::Lz77Method;
use crate::error::Result;
use crate::headers::ColorEncoding;
use crate::headers::frame_header::{BlendMode, FrameCrop, FrameHeader};

/// Options for frame encoding.
#[derive(Debug, Clone)]
pub struct FrameEncoderOptions {
    /// Use modular mode (lossless).
    pub use_modular: bool,
    /// Effort level (1-12, higher = better compression, slower; e10/e11/e12
    /// extends libjxl kTortoise=9 with extended search budgets — e12 doubles
    /// butteraugli iters 16 → 32 on the lossy path).
    pub effort: u8,
    /// Use ANS entropy coding instead of Huffman for modular.
    pub use_ans: bool,
    /// Use content-adaptive MA tree learning for modular encoding.
    pub use_tree_learning: bool,
    /// Use squeeze (Haar wavelet) transform for modular encoding.
    pub use_squeeze: bool,
    /// Enable LZ77 compression on modular token streams.
    pub enable_lz77: bool,
    /// LZ77 method to use when enable_lz77 is true.
    pub lz77_method: Lz77Method,
    /// Use lossy delta palette for near-lossless modular encoding.
    pub lossy_palette: bool,
    /// Encoder mode: Reference (match libjxl) or Experimental (own improvements).
    pub encoder_mode: crate::api::EncoderMode,
    /// Effort profile with all effort-derived parameters.
    pub profile: crate::effort::EffortProfile,
    /// Whether this frame is part of an animation (enables duration field in header).
    pub have_animation: bool,
    /// Whether the file-level animation header signals `have_timecodes`.
    /// When true, an extra 32-bit timecode field is emitted for animated frames.
    pub have_timecodes: bool,
    /// Duration of this frame in ticks (only used when have_animation is true).
    pub duration: u32,
    /// Whether this is the last frame in the image/animation.
    pub is_last: bool,
    /// Optional crop rectangle for this frame (None = full frame).
    pub crop: Option<FrameCrop>,
    /// Skip RCT even for 3-channel images (e.g., XYB channels already decorrelated).
    pub skip_rct: bool,
    /// Optional per-frame blend mode override. `None` keeps the encoder default
    /// (Replace + blend_source=1 when a crop is set, otherwise Replace + 0).
    pub blend_mode: Option<BlendMode>,
    /// Optional `blend_source` override (0–3). `None` keeps the encoder default.
    pub blend_source: Option<u32>,
    /// Optional `save_as_reference` override (0–3). `None` keeps the encoder default
    /// (1 for non-last animated frames).
    pub save_as_reference: Option<u32>,
    /// Optional per-extra-channel blend mode override. `Some(mode)`
    /// replaces every extra channel's [`BlendMode::Replace`] default
    /// with `mode` in the frame header's `ec_blend_modes` vector.
    /// Used by the chunk-2 `with_auto_delta_frames` path on RGBA inputs
    /// so an `Add`-over-zero main also leaves alpha unchanged. `None`
    /// keeps the existing `Replace` default.
    pub ec_blend_mode_override: Option<BlendMode>,
    /// Encode as `FrameType::ReferenceOnly`. The frame is written into
    /// its `save_as_reference` slot but is NOT a displayable keyframe;
    /// later regular frames can composite against it via
    /// `BlendingInfo::source`. Rejected by the public API on the last
    /// frame of an animation.
    pub reference_only: bool,
    /// Optional frame name. `None` writes no name.
    pub name: Option<String>,
    /// Optional SMPTE timecode (requires `have_timecodes=true`). `None` writes `0`.
    pub timecode: Option<u32>,
    /// Optional modular-encoder tuning knobs (libjxl `cjxl -P`,
    /// `--modular_palette_colors`, `--modular_channel_colors_*`,
    /// `-E N` parity). All fields default to `None`; in that state the
    /// encoder uses the built-in libjxl-faithful constants
    /// ([`super::palette::MAX_PALETTE_COLORS`],
    /// [`super::palette::CHANNEL_COLORS_PERCENT`],
    /// [`super::palette::CHANNEL_COLORS_GROUP_PERCENT`]). Wired through
    /// from [`crate::api::LosslessConfig`] by the public encode entry
    /// points in `api.rs`.
    pub modular_knobs: super::palette::ModularKnobs,
    /// Modular group-size shift (libjxl `cjxl -g 0..3`,
    /// `cparams.modular_group_size_shift`). `None` keeps the current
    /// 256-pixel default (shift = 1). `Some(n)` for `n in 0..=3`
    /// maps to group dimensions `128 << n` = {128, 256, 512, 1024}.
    /// Affects both the frame-header signal and the modular encoder's
    /// per-group partitioning. VarDCT is unaffected (libjxl + this
    /// encoder both fix VarDCT groups at 256). See
    /// [`crate::api::LosslessConfig::with_modular_group_size`].
    pub modular_group_size_shift: Option<u8>,
    /// Optional custom DC quantization factors for the LfGlobal
    /// `LfQuantFactors` block (libjxl `DequantMatricesEncodeDC` in
    /// `enc_quant_weights.cc:144-180`). When `Some([x, y, b])`, writes
    /// `all_default = 0` followed by 3 F16-encoded `dc_quant * 128.0`
    /// values into the modular frame's LfGlobal. When `None` (the
    /// default), writes `all_default = 1` (spec defaults `{1/4096,
    /// 1/512, 1/256}` per `quant_weights.h:289-293`).
    ///
    /// Used by the patches reference frame at lossy distance (libjxl
    /// `enc_modular.cc:755-774` scales the base `{65536, 4096, 4096}`
    /// by `1 / (1 + 23·d)` (X) and `1 / (1 + 14·d)` (Y, B-Y)). At
    /// `distance == 0.0` the legacy spec-default DC quant is preserved.
    pub dc_quant_custom: Option<[f32; 3]>,
}

impl Default for FrameEncoderOptions {
    fn default() -> Self {
        Self {
            use_modular: true, // Default to lossless
            effort: 7,
            use_ans: false,
            use_tree_learning: false,
            use_squeeze: false,
            enable_lz77: false,
            lz77_method: Lz77Method::Rle,
            lossy_palette: false,
            encoder_mode: crate::api::EncoderMode::Reference,
            profile: crate::effort::EffortProfile::lossless(7, crate::api::EncoderMode::Reference),
            have_animation: false,
            have_timecodes: false,
            duration: 0,
            is_last: true,
            crop: None,
            skip_rct: false,
            blend_mode: None,
            blend_source: None,
            save_as_reference: None,
            ec_blend_mode_override: None,
            reference_only: false,
            name: None,
            timecode: None,
            modular_knobs: super::palette::ModularKnobs::default(),
            modular_group_size_shift: None,
            dc_quant_custom: None,
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
    /// Optional per-encode allocation budget. When present, dimension-
    /// driven heap allocations inside the modular path charge against
    /// the cap and fail with [`crate::error::Error::AllocationLimit`]
    /// rather than panicking on OOM.
    pub(crate) budget: Option<alloc::sync::Arc<crate::budget::MemoryBudget>>,
}

impl FrameEncoder {
    /// Creates a new frame encoder.
    pub fn new(width: usize, height: usize, options: FrameEncoderOptions) -> Self {
        Self {
            options,
            width,
            height,
            num_extra_channels: 0,
            budget: None,
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
            budget: None,
        }
    }

    /// Attach a per-encode allocation budget. Returns `self` for builder
    /// chaining. Caller-supplied budget is threaded through to the
    /// modular path's hot allocation sites.
    pub(crate) fn with_budget(
        mut self,
        budget: alloc::sync::Arc<crate::budget::MemoryBudget>,
    ) -> Self {
        self.budget = Some(budget);
        self
    }

    /// Apply the animation / per-frame override fields from
    /// [`FrameEncoderOptions`] to the in-progress `FrameHeader`.
    ///
    /// Centralises the (have_animation, have_timecodes, duration,
    /// is_last, crop, blend_mode, blend_source, save_as_reference,
    /// name, timecode) logic so the three modular header-building call
    /// sites (encode_modular, encode_modular_with_patches,
    /// multi-group inner) stay in sync.
    fn apply_animation_to_header(&self, fh: &mut FrameHeader) {
        fh.have_animation = self.options.have_animation;
        fh.have_timecodes = self.options.have_timecodes;
        fh.duration = self.options.duration;
        fh.is_last = self.options.is_last;
        if let Some(tc) = self.options.timecode {
            fh.timecode = tc;
        }
        if let Some(ref name) = self.options.name {
            fh.name = name.clone();
        }

        // Crop sets blend_mode/blend_source defaults; per-frame
        // overrides win below.
        if let Some(ref crop) = self.options.crop {
            fh.x0 = crop.x0;
            fh.y0 = crop.y0;
            fh.width = crop.width;
            fh.height = crop.height;
            fh.blend_mode = BlendMode::Replace;
            fh.blend_source = 1;
            // Mirror the main `blend_source` onto every extra channel.
            // Without this, ec defaults to `source=0` (the empty
            // initial canvas) — `Replace`-over-source-0 on a crop
            // resets the canvas alpha to the encoded pixels and zeros
            // everywhere else, which decodes as alpha=0 outside the
            // crop region (the long-standing RGBA animation-crop
            // bug surfaced by the chunk-2 RGBA tests).
            //
            // `ec_blend_modes` is set by the calling sites
            // (`encode_modular`, `encode_modular_with_patches`,
            // multi-group) before `apply_animation_to_header` runs, so
            // size it from there.
            if !fh.ec_blend_modes.is_empty() {
                fh.ec_blend_sources = vec![fh.blend_source; fh.ec_blend_modes.len()];
            }
        }
        if let Some(mode) = self.options.blend_mode {
            fh.blend_mode = mode;
        }
        if let Some(source) = self.options.blend_source {
            fh.blend_source = source;
        }

        // Default: non-last animated frames save to reference slot 1
        // so successor crop frames can composite over the canvas.
        // An explicit per-frame `save_as_reference` always wins.
        if self.options.have_animation && !self.options.is_last {
            fh.save_as_reference = 1;
        }
        if let Some(slot) = self.options.save_as_reference {
            fh.save_as_reference = slot;
        }

        // Chunk-2 `with_auto_delta_frames` RGBA support: when the
        // caller has set `Add` on the main frame (e.g. identity-frame
        // short-circuit on RGBA), the alpha extra channel must also
        // get `Add` AND must composite against the *same* reference
        // slot as the main frame so that `Add`-over-zero-alpha is a
        // canvas-preserving no-op. Default behaviour (`Replace` with
        // source 0) would clobber alpha to the encoded zero — both
        // because Replace overwrites instead of adding and because
        // source 0 is the empty initial canvas, not the previous-frame
        // slot the main RGB is compositing against. The caller is
        // responsible for matching the override to the main blend
        // mode and zeroing the extra-channel pixel buffer.
        if let Some(ec_mode) = self.options.ec_blend_mode_override {
            for slot in fh.ec_blend_modes.iter_mut() {
                *slot = ec_mode;
            }
            // Mirror the main `blend_source` onto every ec so the
            // decoder composites the extras against the same reference
            // canvas the main frame is using. `apply_animation_to_header`
            // has already finalised `fh.blend_source` by the time we
            // reach this block.
            let n = fh.ec_blend_modes.len();
            fh.ec_blend_sources = vec![fh.blend_source; n];
        }

        // Reference-only frames are written to a save slot but NOT
        // presented as displayable keyframes. The decoder skips them
        // during playback. The spec forbids them as the last frame —
        // the public animation entry points reject that combination
        // (api.rs::validate_animation_input). Force `is_last=false`
        // and clear `have_animation` so the spec's "no duration / no
        // is_last for ReferenceOnly" bitstream layout is respected by
        // FrameHeader::write (frame_header.rs:386-419, normal_frame
        // gate).
        if self.options.reference_only {
            fh.frame_type = crate::headers::frame_header::FrameType::ReferenceOnly;
            fh.is_last = false;
            // Default to slot 1 if the caller didn't pick one (matches
            // the "non-last animated frame" default above, but applies
            // even when the slot wasn't set explicitly via
            // `with_save_as_reference`).
            if self.options.save_as_reference.is_none() && fh.save_as_reference == 0 {
                fh.save_as_reference = 1;
            }
            // ReferenceOnly frames always write `save_before_ct`
            // (frame_header.rs:428). Default to `true` so the decoder
            // stores the pre-color-transform data — what libjxl uses
            // for `kReferenceOnly + xyb_encoded=true` animation
            // backgrounds (matches the patches path at
            // patches.rs:1941).
            fh.save_before_ct = true;
        }
    }

    /// Encodes a modular image into a frame with optional patches.
    ///
    /// When patches are provided, sets PATCHES_FLAG in the frame header and
    /// writes the patches section at the start of LfGlobal data.
    pub(crate) fn encode_modular_with_patches(
        &self,
        image: &ModularImage,
        color_encoding: &ColorEncoding,
        writer: &mut BitWriter,
        patches: Option<&crate::vardct::patches::PatchesData>,
    ) -> Result<()> {
        if patches.is_none() {
            return self.encode_modular(image, color_encoding, writer);
        }
        let patches = patches.unwrap();

        // Compute num_extra_channels from image. Channels beyond the
        // base color set (1 for grayscale, 3 for RGB) are extra
        // channels — alpha is the historical case but [`ModularImage`]
        // can hold N additional channels (refs #9 depth/spot color).
        let num_color = if image.is_grayscale { 1 } else { 3 };
        let num_extra_channels = image.channels.len().saturating_sub(num_color);

        // Write frame header with PATCHES_FLAG
        {
            use crate::headers::frame_header::PATCHES_FLAG;
            let mut fh = FrameHeader::lossless();
            fh.flags |= PATCHES_FLAG;
            fh.ec_upsampling = vec![1; num_extra_channels];
            fh.ec_blend_modes = vec![BlendMode::Replace; num_extra_channels];
            fh.group_size_shift = self.modular_group_size_shift() as u32;
            self.apply_animation_to_header(&mut fh);
            fh.write(writer)?;
        }

        let num_groups = self.num_groups();

        if num_groups == 1 {
            // Single group: combine patches section + modular data into one TOC entry
            let mut section_writer = BitWriter::new();

            // Write patches section first (within the single TOC section)
            crate::vardct::patches::encode_patches_section(
                patches,
                self.options.use_ans,
                &mut section_writer,
            )?;

            // Then write modular data (same logic as encode_modular)
            let has_squeeze = self.options.use_squeeze
                && !super::squeeze::default_squeeze_params(image).is_empty();

            if self.options.lossy_palette && image.channels.len() >= 3 {
                // Honour `--modular_palette_colors` override; fall back
                // to the bit-depth-derived default (libjxl parity).
                let max_colors = self
                    .options
                    .modular_knobs
                    .palette_colors_or_default()
                    .unwrap_or(1usize << image.bit_depth.min(12));
                super::encode::write_modular_stream_with_lossy_palette_budget_knobs(
                    image,
                    &mut section_writer,
                    self.options.use_ans,
                    0,
                    image.channels.len().min(3),
                    max_colors,
                    self.budget.as_ref(),
                    &self.options.modular_knobs,
                )?;
            } else if has_squeeze && self.options.use_tree_learning && self.options.use_ans {
                super::encode::write_modular_stream_with_squeeze_and_tree(
                    image,
                    &mut section_writer,
                    &self.options.profile,
                    self.options.enable_lz77,
                    self.options.lz77_method,
                )?;
            } else if has_squeeze {
                super::encode::write_modular_stream_with_squeeze(
                    image,
                    &mut section_writer,
                    self.options.use_ans,
                )?;
            } else if self.options.use_tree_learning && self.options.use_ans {
                // Tree learning: handles palette internally when beneficial
                super::encode::write_modular_stream_with_tree_knobs(
                    image,
                    &mut section_writer,
                    &self.options.profile,
                    image.channels.len() >= 3,
                    self.options.enable_lz77,
                    self.options.lz77_method,
                    &self.options.modular_knobs,
                )?;
            } else if image.channels.len() >= 3 {
                super::encode::write_modular_stream_with_rct_knobs(
                    image,
                    &mut section_writer,
                    self.options.use_ans,
                    &self.options.modular_knobs,
                )?;
            } else {
                let predictor_id =
                    super::encode::resolve_fixed_predictor(&self.options.modular_knobs);
                super::encode::write_improved_modular_stream_with_predictor(
                    image,
                    &mut section_writer,
                    self.options.use_ans,
                    predictor_id,
                )?;
            }

            let section_data = section_writer.finish();
            self.write_toc(writer, section_data.len())?;
            for byte in section_data {
                writer.write_u8(byte)?;
            }
        } else {
            // Multi-group with patches: patches section goes into LfGlobal.
            // Squeeze + patches is not yet supported; use non-squeeze multi-group path.
            self.encode_modular_multi_group_inner(image, writer, Some(patches))?;
        }

        Ok(())
    }

    /// Encodes a modular image into a frame.
    pub fn encode_modular(
        &self,
        image: &ModularImage,
        _color_encoding: &ColorEncoding,
        writer: &mut BitWriter,
    ) -> Result<()> {
        // Compute num_extra_channels from image. Channels beyond the
        // base color set (1 for grayscale, 3 for RGB) are extra
        // channels — alpha is the historical case but [`ModularImage`]
        // can hold N additional channels (refs #9 depth/spot color).
        let num_color = if image.is_grayscale { 1 } else { 3 };
        let num_extra_channels = image.channels.len().saturating_sub(num_color);

        // Write frame header using unified FrameHeader
        {
            let mut fh = FrameHeader::lossless();
            fh.ec_upsampling = vec![1; num_extra_channels];
            fh.ec_blend_modes = vec![BlendMode::Replace; num_extra_channels];
            fh.group_size_shift = self.modular_group_size_shift() as u32;
            self.apply_animation_to_header(&mut fh);
            fh.write(writer)?;
        }

        let num_groups = self.num_groups();

        if num_groups == 1 {
            // Single group: all sections combined into one TOC entry
            let mut section_writer = BitWriter::new();
            let has_squeeze = self.options.use_squeeze
                && !super::squeeze::default_squeeze_params(image).is_empty();

            if self.options.lossy_palette && image.channels.len() >= 3 {
                // Lossy delta palette: near-lossless with error diffusion.
                // Honour `--modular_palette_colors` override; fall back to
                // the bit-depth-derived default (libjxl parity).
                let max_colors = self
                    .options
                    .modular_knobs
                    .palette_colors_or_default()
                    .unwrap_or(1usize << image.bit_depth.min(12));
                super::encode::write_modular_stream_with_lossy_palette_budget_knobs(
                    image,
                    &mut section_writer,
                    self.options.use_ans,
                    0,
                    image.channels.len().min(3),
                    max_colors,
                    self.budget.as_ref(),
                    &self.options.modular_knobs,
                )?;
            } else if has_squeeze && self.options.use_tree_learning && self.options.use_ans {
                // Combined squeeze + tree learning: best compression
                super::encode::write_modular_stream_with_squeeze_and_tree(
                    image,
                    &mut section_writer,
                    &self.options.profile,
                    self.options.enable_lz77,
                    self.options.lz77_method,
                )?;
            } else if has_squeeze {
                // Squeeze without tree learning (lower effort levels)
                super::encode::write_modular_stream_with_squeeze(
                    image,
                    &mut section_writer,
                    self.options.use_ans,
                )?;
            } else if self.options.use_tree_learning && self.options.use_ans {
                // Tree learning: handles palette internally when beneficial
                super::encode::write_modular_stream_with_tree_knobs(
                    image,
                    &mut section_writer,
                    &self.options.profile,     // effort-dependent tree params
                    image.channels.len() >= 3, // RCT for RGB
                    self.options.enable_lz77,
                    self.options.lz77_method,
                    &self.options.modular_knobs,
                )?;
            } else if image.channels.len() >= 3 {
                super::encode::write_modular_stream_with_rct_knobs(
                    image,
                    &mut section_writer,
                    self.options.use_ans,
                    &self.options.modular_knobs,
                )?;
            } else {
                let predictor_id =
                    super::encode::resolve_fixed_predictor(&self.options.modular_knobs);
                super::encode::write_improved_modular_stream_with_predictor(
                    image,
                    &mut section_writer,
                    self.options.use_ans,
                    predictor_id,
                )?;
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
        } else if self.options.lossy_palette && image.channels.len() >= 3 {
            // Multi-group lossy palette: palette meta in LfGlobal, index across groups
            self.encode_modular_multi_group_lossy_palette(image, writer)?;
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
        self.encode_modular_multi_group_inner(image, writer, None)
    }

    /// Inner multi-group encoder that accepts optional patches.
    /// When patches are provided, writes patches section at the start of LfGlobal.
    fn encode_modular_multi_group_inner(
        &self,
        image: &ModularImage,
        writer: &mut BitWriter,
        patches: Option<&crate::vardct::patches::PatchesData>,
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

        // Step 0: ChannelCompact + RCT + split into meta-image / per-group index images.
        //
        // ChannelCompact (per-channel palette) dramatically reduces bit depth for
        // screenshots with sparse per-channel values (e.g., R uses 30/256 values).
        // Applied BEFORE RCT because RCT spreads values, negating compaction benefit.
        //
        // The key insight: palette meta-channels (small, e.g. 30×1) stay in the global
        // section, while index channels (image-sized) get extract_region per-group.
        // This avoids the root cause of previous failures: extract_region corrupting
        // tiny meta-channels by forcing them to group dimensions.

        // Step 0a: ChannelCompact on raw image (before RCT)
        // Only try ChannelCompact when tree learning + ANS are enabled (the global
        // meta-channel path requires the AnsWithTree codepath in section.rs).
        let has_rct = !self.options.skip_rct && image.channels.len() >= 3;
        let num_color_channels = if has_rct {
            3
        } else {
            image.channels.len().min(3)
        };
        let try_compact = self.options.use_tree_learning && self.options.use_ans;

        let compact_analyses: Vec<(usize, super::palette::PaletteAnalysis)> = if try_compact {
            // For multi-group, compact overhead is higher (meta-channels in global section,
            // tree quality dilution across many groups). Require density <= 50%
            // (i.e. range >= 2x unique), which means >= 1 bit/pixel entropy savings.
            // Below this threshold, savings are eaten by per-group overhead.
            //
            // Honour `--modular_channel_colors_group_percent` override
            // when set, otherwise keep the historical
            // `CHANNEL_COLORS_PERCENT` default (95.0). libjxl's
            // `enc_params.h:channel_colors_percent` defaults to 80.0 for
            // the per-group pass; we ship the global 95.0 here for
            // bitstream stability with the existing hash-locks and let
            // callers dial down via the CLI override.
            let group_percent = self
                .options
                .modular_knobs
                .channel_colors_group_percent
                .unwrap_or(CHANNEL_COLORS_PERCENT);
            (0..num_color_channels)
                .filter_map(|ch_idx| {
                    let analysis = analyze_channel_compact(&image.channels[ch_idx], group_percent)?;
                    // Reject if unique values use >50% of the range (< 1 bit/pixel savings)
                    let ch = &image.channels[ch_idx];
                    let mut min_v = i32::MAX;
                    let mut max_v = i32::MIN;
                    for y in 0..ch.height() {
                        for x in 0..ch.width() {
                            let v = ch.get(x, y);
                            min_v = min_v.min(v);
                            max_v = max_v.max(v);
                        }
                    }
                    let range = (max_v as i64 - min_v as i64 + 1).max(1) as f64;
                    let density = analysis.num_colors as f64 / range;
                    crate::trace::debug_eprintln!(
                        "COMPACT_FILTER: ch={} unique={} range={:.0} density={:.3}",
                        ch_idx,
                        analysis.num_colors,
                        range,
                        density
                    );
                    if density > 0.5 {
                        return None;
                    }
                    Some((ch_idx, analysis))
                })
                .collect()
        } else {
            Vec::new()
        };

        let (meta_image, source_image_owned, compact_info, rct_type);
        if !compact_analyses.is_empty() {
            // Build palette meta-channels + index channels.
            // Layout: [pal_N-1, ..., pal_0, idx_0, ch_1, idx_2, ...extra]
            // (palettes reversed for decoder MetaPalette insertion order)
            let mut palettes: Vec<Channel> = Vec::new();
            let mut non_meta: Vec<Channel> = Vec::new();
            let mut info: Vec<(usize, usize)> = Vec::new();
            let mut nb_meta = 0usize;

            for (orig_idx, ch) in image.channels.iter().enumerate() {
                if let Some((_, analysis)) =
                    compact_analyses.iter().find(|(idx, _)| *idx == orig_idx)
                {
                    // Create palette meta-channel (nb_colors wide, 1 high)
                    let mut pal_ch = Channel::new(analysis.num_colors, 1)?;
                    for (i, color) in analysis.palette.iter().enumerate() {
                        pal_ch.set(i, 0, color[0]);
                    }
                    palettes.push(pal_ch);

                    // Create index channel (same dimensions as original)
                    let mut idx_ch = Channel::new(ch.width(), ch.height())?;
                    for y in 0..ch.height() {
                        for x in 0..ch.width() {
                            let val = ch.get(x, y);
                            let index = analysis.color_to_index[&vec![val]];
                            idx_ch.set(x, y, index);
                        }
                    }
                    non_meta.push(idx_ch);

                    // begin_c for the transform descriptor
                    let begin_c = orig_idx + nb_meta;
                    info.push((begin_c, analysis.num_colors));
                    nb_meta += 1;
                } else {
                    non_meta.push(ch.clone());
                }
            }

            // Decoder's MetaPalette inserts each palette at position 0,
            // so earlier palettes end up deeper. Reverse to match.
            palettes.reverse();

            crate::trace::debug_eprintln!(
                "MULTI_GROUP_COMPACT: {} channels compacted, {} meta + {} non-meta, info={:?}",
                compact_analyses.len(),
                nb_meta,
                non_meta.len(),
                info,
            );

            // Split: meta-channels stay whole in global section
            let mut meta_img = image.clone();
            meta_img.channels = palettes;

            // Non-meta channels: index + non-compacted + extra (full-size, to be split)
            let mut work = image.clone();
            work.channels = non_meta;

            // Step 0b: RCT on index channels (starting at position 0 in non-meta)
            if has_rct && work.channels.len() >= 3 {
                let (selected_rct, transformed) = select_best_rct_at(
                    &work,
                    0,
                    self.options.profile.nb_rcts_to_try,
                    self.options.profile.forced_rct,
                );
                rct_type = Some(selected_rct);
                work = transformed;
            } else {
                rct_type = None;
            }

            meta_image = Some(meta_img);
            source_image_owned = work;
            compact_info = info;
        } else {
            // No ChannelCompact — standard RCT-only path
            crate::profile_time!("modular/rct_select", {
                if has_rct {
                    let (selected_rct, rct_image) = super::encode::select_best_rct(
                        image,
                        self.options.profile.nb_rcts_to_try,
                        self.options.profile.forced_rct,
                    );
                    rct_type = Some(selected_rct);
                    source_image_owned = rct_image;
                } else {
                    rct_type = None;
                    source_image_owned = image.clone();
                }
            });
            meta_image = None;
            compact_info = Vec::new();
        };

        let global_transforms = super::section::GlobalTransforms {
            compact_info,
            rct_type,
        };

        // Step 1: Extract each group image from the index/non-meta channels only.
        // Meta-channels (palettes) are NOT split — they go whole in the global section.
        let mut group_images: Vec<ModularImage> = Vec::with_capacity(num_groups);
        let group_transforms: Vec<super::section::GroupTransforms> =
            vec![super::section::GroupTransforms::none(); num_groups];
        for group_idx in 0..num_groups {
            let (x_start, y_start, x_end, y_end) = self.group_bounds(group_idx);
            let group_image = source_image_owned.extract_region(x_start, y_start, x_end, y_end)?;
            group_images.push(group_image);
        }

        // Step 2: Write LfGlobal section (patches + tree + histogram)
        let mut lf_global_writer = BitWriter::new();

        // If patches are provided, write patches section first in LfGlobal
        if let Some(pd) = patches {
            crate::vardct::patches::encode_patches_section(
                pd,
                self.options.use_ans,
                &mut lf_global_writer,
            )?;
        }

        let global_state = if self.options.dc_quant_custom.is_some() {
            // When the caller supplied custom DC quant (patches ref frame at
            // lossy distance), always use the tree-learning `_dc_quant_knobs`
            // variant — it is the only multi-group path that writes a
            // non-default `LfQuantFactors`. The non-tree path
            // (`write_global_modular_section_with_predictor`) hardcodes
            // `all_default = true`. Sending custom dc_quant through it would
            // silently emit spec defaults and the decoder would dequantize
            // with the wrong scale, corrupting the patches.
            super::section::write_global_modular_section_with_tree_dc_quant_knobs(
                &group_images,
                &mut lf_global_writer,
                &self.options.profile,
                global_transforms,
                self.options.enable_lz77,
                self.options.lz77_method,
                self.options.dc_quant_custom,
                meta_image.as_ref(),
                &self.options.modular_knobs,
                super::section::modular_hf_stream_id_base(self.num_lf_groups() as u32),
            )?
        } else if self.options.use_tree_learning && self.options.use_ans {
            // Tree learning path: gather samples, learn tree, build multi-context ANS.
            // Honours [`super::palette::ModularKnobs::modular_predictor`]
            // (libjxl `cjxl -P N` / `--modular_predictor`) by routing
            // through the `_knobs` variant, which builds a single-leaf
            // tree pinned to the requested predictor when set.
            super::section::write_global_modular_section_with_tree_knobs(
                &group_images,
                &mut lf_global_writer,
                &self.options.profile, // effort-dependent tree params
                global_transforms,
                self.options.enable_lz77,
                self.options.lz77_method,
                meta_image.as_ref(),
                &self.options.modular_knobs,
                super::section::modular_hf_stream_id_base(self.num_lf_groups() as u32),
            )?
        } else {
            // Standard path: collect residuals using the requested predictor
            // (libjxl `cjxl -P` / `--modular_predictor`). No per-channel WP
            // state in this path — fold id 6 → 5 for self-consistency.
            let predictor_id =
                super::encode::resolve_fixed_predictor_for_simple_path(&self.options.modular_knobs);
            let mut all_residuals = Vec::new();
            let mut max_residual: u32 = 0;
            for group_image in &group_images {
                let (group_residuals, group_max) =
                    super::section::collect_all_residuals_with_predictor(group_image, predictor_id);
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

            super::section::write_global_modular_section_with_predictor(
                &all_residuals,
                &histogram,
                max_token,
                &mut lf_global_writer,
                self.options.use_ans,
                global_transforms,
                predictor_id,
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
        //
        // Property-1 (`group_id`) values MUST be the spec stream ids the
        // decoder evaluates when walking the tree (#68 second cause):
        // pass-group g at pass 0 is `1 + 3*num_lf_groups +
        // NUM_QUANT_TABLE_STREAMS + g` (the global/meta channels are
        // stream 0). The old `1 + group_idx` collision-avoidance ids
        // matched section.rs's tree learning but not the decoders, so
        // any e9+ tree splitting on property 1 desynced. num_passes is
        // 1 today; the pass term is `num_groups * pass` when that
        // changes. Must stay in lockstep with the gather/apply ids in
        // section.rs (both now derive from `modular_hf_stream_id_base`).
        let per_group_id_offset: u32 =
            super::section::modular_hf_stream_id_base(num_lf_groups as u32);
        // PassGroup sections — parallelizable (each group writes to its own BitWriter)
        let pass_group_data: Vec<Vec<u8>> =
            crate::parallel::parallel_map_result(num_groups * num_passes, |flat_idx| {
                let group_idx = flat_idx / num_passes;
                let group_image = &group_images[group_idx];

                let mut group_writer = BitWriter::new();
                write_group_modular_section_idx(
                    group_image,
                    &global_state,
                    group_idx as u32 + per_group_id_offset,
                    &group_transforms[group_idx],
                    &mut group_writer,
                )?;

                crate::trace::debug_eprintln!(
                    "MULTI_GROUP: PassGroup {} section = {} bytes",
                    group_idx,
                    group_writer.bits_written() / 8,
                );
                Ok(group_writer.finish())
            })?;

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

    /// Encodes a modular image using multi-group format with lossy (delta) palette.
    ///
    /// After applying the lossy palette transform, the channel layout is:
    /// - Channel 0: palette meta-channel (width=total_size, height=num_c) — SMALL
    /// - Channel 1: index channel (width=image_width, height=image_height) — LARGE
    /// - Channel 2+: optional alpha/extra channels
    ///
    /// The palette meta-channel goes into LfGlobal (alongside the tree + histogram).
    /// The index and extra channels are split across PassGroups by 256x256 regions.
    /// The palette transform descriptor is written in the LfGlobal GroupHeader.
    fn encode_modular_multi_group_lossy_palette(
        &self,
        image: &ModularImage,
        writer: &mut BitWriter,
    ) -> Result<()> {
        use super::encode::{
            predict_pixel_with_id, resolve_fixed_predictor_for_simple_path,
            write_gradient_tree_tokens, write_hybrid_data_histogram, write_predictor_tree_tokens,
            write_tree_histogram_for_gradient, write_tree_histogram_for_predictor,
        };
        use super::encode_transforms::write_palette_transform;
        use super::predictor::pack_signed;
        use crate::entropy_coding::encode::{build_entropy_code_ans, write_tokens_ans};
        use crate::entropy_coding::hybrid_uint::HybridUintConfig;
        use crate::entropy_coding::token::Token as AnsToken;

        const MODULAR_HYBRID_UINT: HybridUintConfig = HybridUintConfig {
            split_exponent: 4,
            split: 16,
            msb_in_token: 2,
            lsb_in_token: 0,
        };

        // Multi-group lossy palette has no per-channel WP state for the
        // index/palette residual pass — fold predictor id 6 → 5.
        let predictor_id = resolve_fixed_predictor_for_simple_path(&self.options.modular_knobs);

        let num_groups = self.num_groups();
        let num_lf_groups = self.num_lf_groups();

        // Step 1: Apply lossy palette to full image
        let mut transformed = image.clone();
        let max_colors = 1usize << image.bit_depth.min(12);
        let num_c = image.channels.len().min(3);
        let result = super::palette::apply_lossy_palette_with_budget(
            &mut transformed,
            0,
            num_c,
            max_colors,
            self.budget.as_ref(),
        );
        let result = match result {
            Some(r) => r,
            None => {
                // Lossy palette not beneficial, fall back to standard multi-group
                return self.encode_modular_multi_group_inner(image, writer, None);
            }
        };

        crate::trace::debug_eprintln!(
            "LOSSY_PALETTE_MULTI: {} colors + {} deltas, predictor={}, {} → {} channels, {}x{}",
            result.nb_colors,
            result.nb_deltas,
            result.predictor,
            image.channels.len(),
            transformed.channels.len(),
            self.width,
            self.height,
        );

        // After palette: transformed.channels = [palette_meta, index, ...extra]
        // Separate palette_meta (small, global) from spatial channels (split across groups)
        let palette_meta = transformed.channels[0].clone();

        // Build a ModularImage of only the spatial channels (index + alpha)
        let spatial_image = ModularImage {
            channels: transformed.channels[1..].to_vec(),
            bit_depth: transformed.bit_depth,
            is_grayscale: transformed.is_grayscale,
            has_alpha: transformed.has_alpha,
        };

        // Step 2: Extract group images from spatial channels only
        let mut group_images: Vec<ModularImage> = Vec::with_capacity(num_groups);
        for group_idx in 0..num_groups {
            let (x_start, y_start, x_end, y_end) = self.group_bounds(group_idx);
            let group_image = spatial_image.extract_region(x_start, y_start, x_end, y_end)?;
            group_images.push(group_image);
        }

        // Step 3: Collect ALL residuals (palette_meta + all groups) for histogram
        // Honours `--modular-predictor` for the no-tree-learning fallback path.
        let collect_channel_residuals = |channel: &super::channel::Channel| -> Vec<u32> {
            let w = channel.width();
            let h = channel.height();
            let mut residuals = Vec::with_capacity(w * h);
            for y in 0..h {
                for x in 0..w {
                    let pixel = channel.get(x, y);
                    let prediction = predict_pixel_with_id(channel, x, y, predictor_id);
                    residuals.push(pack_signed(pixel - prediction));
                }
            }
            residuals
        };

        // Palette meta-channel residuals (goes to LfGlobal)
        let palette_residuals = collect_channel_residuals(&palette_meta);

        // All residuals: palette_meta + all group spatial channels
        let mut all_residuals = palette_residuals.clone();
        for group_image in &group_images {
            for channel in &group_image.channels {
                all_residuals.extend(collect_channel_residuals(channel));
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

        // Tree histogram + tokens for the requested predictor (5=Gradient default).
        let (tree_depths, tree_codes) = if predictor_id == 5 {
            write_tree_histogram_for_gradient(&mut lf_global_writer)?
        } else {
            write_tree_histogram_for_predictor(&mut lf_global_writer, predictor_id)?
        };
        if predictor_id == 5 {
            write_gradient_tree_tokens(&mut lf_global_writer, &tree_depths, &tree_codes)?;
        } else {
            write_predictor_tree_tokens(
                &mut lf_global_writer,
                &tree_depths,
                &tree_codes,
                predictor_id,
            )?;
        }

        // Build entropy coding state
        let use_ans = self.options.use_ans;

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
            let (depths, codes) =
                write_hybrid_data_histogram(&mut lf_global_writer, &histogram, max_token)?;
            EntropyState::Huffman { depths, codes }
        };

        // GroupHeader with palette transform
        lf_global_writer.write(1, 1)?; // use_global_tree = true
        lf_global_writer.write(1, 1)?; // wp_params.default_wp = true
        lf_global_writer.write(2, 1)?; // nb_transforms = 1
        write_palette_transform(
            &mut lf_global_writer,
            0,
            num_c,
            result.nb_colors,
            result.nb_deltas,
            result.predictor,
        )?;

        // Encode palette_meta residuals in LfGlobal
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

        encode_residuals(&palette_residuals, &mut lf_global_writer, &entropy_state)?;

        lf_global_writer.zero_pad_to_byte();
        let lf_global_data = lf_global_writer.finish();

        crate::trace::debug_eprintln!(
            "LOSSY_PALETTE_MULTI: LfGlobal = {} bytes (palette_meta {}x{})",
            lf_global_data.len(),
            palette_meta.width(),
            palette_meta.height(),
        );

        // Step 6: HfGlobal is empty for modular
        let hf_global_data: Vec<u8> = Vec::new();

        // Step 7: LfGroup sections are empty for modular
        let lf_group_data: Vec<Vec<u8>> = (0..num_lf_groups).map(|_| Vec::new()).collect();

        // Step 8: Write each PassGroup's data — parallelizable
        let pass_group_data: Vec<Vec<u8>> =
            crate::parallel::parallel_map_result(num_groups, |g| {
                let group_image = &group_images[g];
                let mut group_writer = BitWriter::new();

                // GroupHeader
                group_writer.write(1, 1)?; // use_global_tree = true
                group_writer.write(1, 1)?; // wp_params.default_wp = true
                group_writer.write(2, 0)?; // nb_transforms = 0

                // Collect and encode spatial channel residuals for this group
                let mut section_residuals: Vec<u32> = Vec::new();
                for channel in &group_image.channels {
                    section_residuals.extend(collect_channel_residuals(channel));
                }
                encode_residuals(&section_residuals, &mut group_writer, &entropy_state)?;

                group_writer.zero_pad_to_byte();
                let data = group_writer.finish();
                crate::trace::debug_eprintln!(
                    "LOSSY_PALETTE_MULTI: PassGroup[{}] = {} bytes",
                    g,
                    data.len(),
                );
                Ok(data)
            })?;

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

        self.write_toc_multi(writer, &section_sizes)?;

        // Write all section data in same order
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
            predict_pixel_with_id, resolve_fixed_predictor_for_simple_path,
            write_gradient_tree_tokens, write_predictor_tree_tokens, write_rct_transform,
            write_squeeze_transform, write_tree_histogram_for_gradient,
            write_tree_histogram_for_predictor,
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

        // Squeeze multi-group has no per-channel WP state for residual
        // pass — fold predictor id 6 → 5.
        let predictor_id = resolve_fixed_predictor_for_simple_path(&self.options.modular_knobs);

        let num_groups = self.num_groups();
        let num_lf_groups = self.num_lf_groups();
        let group_dim = self.group_dim();
        let lf_group_dim = group_dim * 8;

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
            .position(|c| c.width() > group_dim || c.height() > group_dim)
            .unwrap_or(squeezed.channels.len());

        crate::trace::debug_eprintln!(
            "SQUEEZE_MULTI: {} global channels (<={}x{}), {} group channels",
            global_cutoff,
            group_dim,
            group_dim,
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
        // Honours `--modular-predictor` for the squeeze multi-group fallback.
        let collect_channel_residuals = |channel: &super::channel::Channel| -> Vec<u32> {
            let w = channel.width();
            let h = channel.height();
            let mut residuals = Vec::with_capacity(w * h);
            for y in 0..h {
                for x in 0..w {
                    let pixel = channel.get(x, y);
                    let prediction = predict_pixel_with_id(channel, x, y, predictor_id);
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
                if let Some(cropped) = ch.extract_grid_cell(gx, gy, group_dim) {
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

        // Tree histogram + tokens for the requested predictor (5=Gradient default).
        let (tree_depths, tree_codes) = if predictor_id == 5 {
            write_tree_histogram_for_gradient(&mut lf_global_writer)?
        } else {
            write_tree_histogram_for_predictor(&mut lf_global_writer, predictor_id)?
        };
        if predictor_id == 5 {
            write_gradient_tree_tokens(&mut lf_global_writer, &tree_depths, &tree_codes)?;
        } else {
            write_predictor_tree_tokens(
                &mut lf_global_writer,
                &tree_depths,
                &tree_codes,
                predictor_id,
            )?;
        }

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
        // Step 8: Write PassGroup sections — parallelizable
        let pass_group_data: Vec<Vec<u8>> =
            crate::parallel::parallel_map_result(num_groups, |g| {
                let g_channels = &pass_group_channel_data[g];
                let mut pg_writer = BitWriter::new();

                if g_channels.is_empty() {
                    // Empty PassGroup (no channels assigned)
                    return Ok(pg_writer.finish());
                }

                // GroupHeader
                pg_writer.write(1, 1)?; // use_global_tree = true
                pg_writer.write(1, 1)?; // wp_params.default_wp = true
                pg_writer.write(2, 0)?; // nb_transforms = 0

                // Concatenate all channel residuals for this section, then encode once.
                let mut section_residuals: Vec<u32> = Vec::new();
                for channel_residuals in g_channels {
                    section_residuals.extend(channel_residuals);
                }
                encode_residuals(&section_residuals, &mut pg_writer, &entropy_state)?;

                pg_writer.zero_pad_to_byte();
                let data = pg_writer.finish();
                crate::trace::debug_eprintln!(
                    "SQUEEZE_MULTI: PassGroup[{}] = {} bytes ({} channels)",
                    g,
                    data.len(),
                    g_channels.len(),
                );
                Ok(data)
            })?;

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
            GatherDedupTable, TreeLearningParams, TreeSamples,
            collect_residuals_with_tree_with_budget, compute_best_tree_with_budget,
            compute_gather_stride_from_profile, gather_samples_strided_with_budget_inner,
        };
        use crate::entropy_coding::encode::build_entropy_code_ans_with_options;
        use crate::entropy_coding::encode::{write_entropy_code_ans, write_tokens_ans};
        use crate::entropy_coding::token::Token as AnsToken;

        let num_groups = self.num_groups();
        let num_lf_groups = self.num_lf_groups();
        let group_dim = self.group_dim();
        let lf_group_dim = group_dim * 8;

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
            .position(|c| c.width() > group_dim || c.height() > group_dim)
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
        let stride = compute_gather_stride_from_profile(total_pixels, &self.options.profile);

        // Find best WP parameters (effort-dependent search)
        let wp_params = if self.options.profile.wp_num_param_sets > 0 {
            super::predictor::find_best_wp_params(
                &squeezed.channels,
                self.options.profile.wp_num_param_sets,
            )
        } else {
            super::predictor::WeightedPredictorParams::default()
        };

        let mut samples = TreeSamples::new();

        // Phase 2 of issue #41: when the profile asks for gather-time
        // dedup, all three gather phases (global / LfGroup / PassGroup)
        // share one cuckoo table so cross-section duplicates merge into
        // the same unique row. Size from the global pixel count to keep
        // load < 2/3 throughout — channels in LfGroup/PassGroup are
        // already covered by the same estimate (squeezed total).
        //
        // The hash uses `params.properties` (post-y/x skip) so the
        // gather-time merge mirrors what the post-sort dedup would
        // collapse anyway. This squeeze path uses
        // `TreeLearningParams::from_profile` (NOT `_squeeze`) — squeeze
        // training happens inside the same routine but `from_profile`
        // matches what we feed to `compute_best_tree_with_budget`
        // downstream.
        let mut shared_dedup_table = if self.options.profile.gather_dedup {
            let est_samples = total_pixels.div_ceil(stride.max(1)).max(1);
            let props = TreeLearningParams::from_profile(&self.options.profile)
                .properties
                .clone();
            Some(GatherDedupTable::new_with_properties(est_samples, &props))
        } else {
            None
        };

        // Compute stream_id values matching the decoder's ModularStreamId formula.
        // The decoder assigns stream_id = property[1] during tree traversal:
        //   GlobalData:    0
        //   ModularLF(g):  1 + num_lf_groups + g
        //   ModularHF(p,g): 1 + 3*num_lf_groups + NUM_QUANT_TABLES + num_groups*p + g
        // These must match between encoder (tree training + residual collection) and decoder.
        const NUM_QUANT_TABLES: usize = 17;
        let stream_id_lf_base = 1 + num_lf_groups;
        let stream_id_hf_base = 1 + 3 * num_lf_groups + NUM_QUANT_TABLES;

        // 3a: Global channels (full, no cropping needed)
        let global_sub = ModularImage {
            channels: squeezed.channels[..global_cutoff].to_vec(),
            bit_depth: squeezed.bit_depth,
            is_grayscale: squeezed.is_grayscale,
            has_alpha: false,
        };
        // group_id=0 for global section, channel_offset=0
        gather_samples_strided_with_budget_inner(
            &mut samples,
            &global_sub,
            0,
            0,
            stride,
            &wp_params,
            self.budget.as_ref(),
            shared_dedup_table.as_mut(),
        )?;

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
            // stream_id for LfGroup: ModularLF(lg) = 1 + num_lf_groups + lg
            gather_samples_strided_with_budget_inner(
                &mut samples,
                &sub_image,
                (stream_id_lf_base + lg) as u32,
                0,
                stride,
                &wp_params,
                self.budget.as_ref(),
                shared_dedup_table.as_mut(),
            )?;
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
                if let Some(cropped) = ch.extract_grid_cell(gx, gy, group_dim) {
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
            // stream_id for PassGroup: ModularHF(pass=0, group=g) = 1 + 3*num_lf_groups + 17 + g
            gather_samples_strided_with_budget_inner(
                &mut samples,
                &sub_image,
                (stream_id_hf_base + g) as u32,
                0,
                stride,
                &wp_params,
                self.budget.as_ref(),
                shared_dedup_table.as_mut(),
            )?;
        }

        // Step 4: Learn tree. See section.rs for the gather-time dedup
        // pixel_fraction rationale (`total_gathered_weight()` recovers
        // the pre-dedup gathered count when sample_counts is populated).
        let pixel_fraction = if total_pixels > 0 {
            samples.total_gathered_weight() as f64 / total_pixels as f64
        } else {
            1.0
        };
        let tree_params = TreeLearningParams::from_profile(&self.options.profile)
            .with_pixel_fraction(pixel_fraction)
            .with_total_pixels(total_pixels);
        let tree = compute_best_tree_with_budget(&mut samples, &tree_params, self.budget.as_ref())?;
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
        let mut global_tokens = collect_residuals_with_tree_with_budget(
            &global_sub,
            &tree,
            0,
            &wp_params,
            self.budget.as_ref(),
        )?;

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
            let tokens = collect_residuals_with_tree_with_budget(
                &sub_image,
                &tree,
                (stream_id_lf_base + lg) as u32,
                &wp_params,
                self.budget.as_ref(),
            )?;
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
            let tokens = collect_residuals_with_tree_with_budget(
                &sub_image,
                &tree,
                (stream_id_hf_base + g) as u32,
                &wp_params,
                self.budget.as_ref(),
            )?;
            pass_group_tokens.push(tokens);
        }

        // Step 5b: Optionally apply LZ77 to each section's tokens independently
        // IMPORTANT: dist_multiplier must be computed PER-SECTION from that section's
        // channel widths, because the decoder creates a fresh LZ77 state per section
        // with dist_multiplier = max(section_channel_widths).
        let use_lz77 = self.options.enable_lz77;
        let lz77_method = self.options.lz77_method;
        let lz77_params = if use_lz77 {
            use crate::entropy_coding::lz77::apply_lz77;

            let try_lz77 = |tokens: &[AnsToken], dist_multiplier: i32| -> Vec<AnsToken> {
                if tokens.is_empty() {
                    return tokens.to_vec();
                }
                match apply_lz77(tokens, num_contexts, false, lz77_method, dist_multiplier) {
                    Some((lz77_tokens, _)) => lz77_tokens,
                    None => tokens.to_vec(),
                }
            };

            // Global section: dist_multiplier from global channels
            let global_dm = squeezed.channels[..global_cutoff]
                .iter()
                .map(|c| c.width())
                .max()
                .unwrap_or(0) as i32;
            global_tokens = try_lz77(&global_tokens, global_dm);

            // LfGroup sections: dist_multiplier from each LfGroup's channels
            for (lg, lg_tokens) in lf_group_tokens.iter_mut().enumerate() {
                let dm = lf_group_sub_images[lg]
                    .iter()
                    .map(|c| c.width())
                    .max()
                    .unwrap_or(0) as i32;
                *lg_tokens = try_lz77(lg_tokens, dm);
            }

            // PassGroup sections: dist_multiplier from each PassGroup's channels
            for (g, pg_tokens) in pass_group_tokens.iter_mut().enumerate() {
                let dm = pass_group_sub_images[g]
                    .iter()
                    .map(|c| c.width())
                    .max()
                    .unwrap_or(0) as i32;
                *pg_tokens = try_lz77(pg_tokens, dm);
            }

            // Check if any section has LZ77 references
            let has_lz77 = global_tokens.iter().any(|t| t.is_lz77_length())
                || lf_group_tokens
                    .iter()
                    .any(|ts| ts.iter().any(|t| t.is_lz77_length()))
                || pass_group_tokens
                    .iter()
                    .any(|ts| ts.iter().any(|t| t.is_lz77_length()));

            if has_lz77 {
                let mut params = crate::entropy_coding::lz77::Lz77Params::new(num_contexts, false);
                params.enabled = true;
                Some(params)
            } else {
                None
            }
        } else {
            None
        };
        let ans_num_contexts = if lz77_params.is_some() {
            num_contexts + 1
        } else {
            num_contexts
        };

        // Step 6: Build ANS codes from ALL tokens
        let mut all_tokens: Vec<AnsToken> = Vec::new();
        all_tokens.extend(&global_tokens);
        for lg_tokens in &lf_group_tokens {
            all_tokens.extend(lg_tokens);
        }
        for pg_tokens in &pass_group_tokens {
            all_tokens.extend(pg_tokens);
        }
        let code = build_entropy_code_ans_with_options(
            &all_tokens,
            ans_num_contexts,
            true, // enhanced clustering (pair-merge refinement)
            true, // optimize uint configs
            lz77_params.as_ref(),
            Some(total_pixels),
        );

        // Step 7: Write LfGlobal section
        let mut lf_global_writer = BitWriter::new();

        // dc_quant.all_default = true
        lf_global_writer.write(1, 1)?;
        // has_tree = true
        lf_global_writer.write(1, 1)?;

        // Write the learned tree
        write_tree(&mut lf_global_writer, &tree)?;

        // Write LZ77 header + ANS histogram
        if ans_num_contexts > 1 {
            crate::entropy_coding::lz77::write_lz77_header(
                lz77_params.as_ref(),
                &mut lf_global_writer,
            )?;
            write_entropy_code_ans(&code, &mut lf_global_writer)?;
        } else {
            super::section::write_ans_modular_header(&mut lf_global_writer, &code)?;
        }

        // GroupHeader for global modular stream — includes RCT (if RGB) + squeeze transform
        lf_global_writer.write(1, 1)?; // use_global_tree = true
        super::encode::write_wp_header(&mut lf_global_writer, &wp_params)?;
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
        write_tokens_ans(
            &global_tokens,
            &code,
            lz77_params.as_ref(),
            &mut lf_global_writer,
        )?;

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
            super::encode::write_wp_header(&mut lg_writer, &wp_params)?;
            lg_writer.write(2, 0)?; // nb_transforms = 0

            write_tokens_ans(lg_tokens, &code, lz77_params.as_ref(), &mut lg_writer)?;

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

        // Step 10: Write PassGroup sections — parallelizable
        let pass_group_data: Vec<Vec<u8>> =
            crate::parallel::parallel_map_result(num_groups, |g| {
                let pg_tokens = &pass_group_tokens[g];
                let mut pg_writer = BitWriter::new();

                if pg_tokens.is_empty() {
                    return Ok(pg_writer.finish());
                }

                // GroupHeader
                pg_writer.write(1, 1)?; // use_global_tree = true
                super::encode::write_wp_header(&mut pg_writer, &wp_params)?;
                pg_writer.write(2, 0)?; // nb_transforms = 0

                write_tokens_ans(pg_tokens, &code, lz77_params.as_ref(), &mut pg_writer)?;

                pg_writer.zero_pad_to_byte();
                let data = pg_writer.finish();
                crate::trace::debug_eprintln!(
                    "SQUEEZE_TREE_MULTI: PassGroup = {} bytes ({} tokens)",
                    data.len(),
                    pg_tokens.len(),
                );
                Ok(data)
            })?;

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

    /// Encode modular image body (TOC + sections) without writing a frame header.
    ///
    /// Caller is responsible for writing the frame header before calling this.
    /// This enables encoding reference frames with custom frame headers (e.g.,
    /// `FrameType::ReferenceOnly`, `save_before_ct=true`) while getting full
    /// FrameEncoder features (RCT, multi-group, histogram optimization, ANS).
    pub(crate) fn encode_modular_body(
        &self,
        image: &ModularImage,
        writer: &mut BitWriter,
    ) -> Result<()> {
        let num_groups = self.num_groups();

        if num_groups == 1 {
            // Single group: all sections combined into one TOC entry
            let mut section_writer = BitWriter::new();

            let use_rct = image.channels.len() >= 3 && !self.options.skip_rct;

            // When the caller supplied custom DC quant (patches ref frame at
            // lossy distance), always route through the tree-learning
            // `_dc_quant_knobs` variant — it is the only single-group path
            // that wires the `LfQuantFactors` header through to
            // `write_lf_quant`. The non-tree paths hardcode
            // `writer.write(1, 1) // all_default = true`. Sending Some(dq)
            // through them would silently emit the spec defaults and the
            // decoder would dequantize with the wrong scale.
            //
            // At `dc_quant_custom == None` we preserve the existing routing
            // (no perturbation to lossless hash-locks).
            if self.options.dc_quant_custom.is_some() {
                super::encode::write_modular_stream_with_tree_dc_quant_knobs(
                    image,
                    &mut section_writer,
                    &self.options.profile,
                    use_rct,
                    self.options.enable_lz77,
                    self.options.lz77_method,
                    self.options.dc_quant_custom,
                    None, // not lossy modular (no Squeeze + multiplier-info path)
                    true, // palette detection ok
                    &self.options.modular_knobs,
                )?;
            } else if self.options.use_tree_learning && self.options.use_ans {
                super::encode::write_modular_stream_with_tree_knobs(
                    image,
                    &mut section_writer,
                    &self.options.profile,
                    use_rct,
                    self.options.enable_lz77,
                    self.options.lz77_method,
                    &self.options.modular_knobs,
                )?;
            } else if use_rct {
                super::encode::write_modular_stream_with_rct_knobs(
                    image,
                    &mut section_writer,
                    self.options.use_ans,
                    &self.options.modular_knobs,
                )?;
            } else {
                let predictor_id =
                    super::encode::resolve_fixed_predictor(&self.options.modular_knobs);
                super::encode::write_improved_modular_stream_with_predictor(
                    image,
                    &mut section_writer,
                    self.options.use_ans,
                    predictor_id,
                )?;
            }

            let section_data = section_writer.finish();
            self.write_toc(writer, section_data.len())?;
            for byte in section_data {
                writer.write_u8(byte)?;
            }
        } else {
            // Multi-group: use the standard multi-group encoder (no patches in body)
            self.encode_modular_multi_group_inner(image, writer, None)?;
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

    /// Returns the modular group dimension in pixels.
    ///
    /// Maps the optional `modular_group_size_shift` knob (libjxl
    /// `cjxl -g 0..3`) to a pixel dimension. When `None`, returns the
    /// historical 256-pixel default (shift = 1) so existing callers
    /// keep byte-identical output.
    ///
    /// Shift → dim mapping (matches libjxl
    /// `frame_dim.Set(..., group_size_shift, ...)`):
    /// `0 → 128`, `1 → 256`, `2 → 512`, `3 → 1024`.
    pub fn group_dim(&self) -> usize {
        let shift = self.modular_group_size_shift() as usize;
        128usize << shift
    }

    /// Returns the effective `group_size_shift` (0..=3) for this frame.
    /// Defaults to `1` (256 pixels) when the knob is unset.
    pub fn modular_group_size_shift(&self) -> u8 {
        self.options
            .modular_group_size_shift
            .map(|s| s.min(3))
            .unwrap_or(1)
    }

    /// Returns the number of groups in this frame.
    pub fn num_groups(&self) -> usize {
        self.num_groups_x() * self.num_groups_y()
    }

    /// Returns the number of groups in X direction.
    pub fn num_groups_x(&self) -> usize {
        self.width.div_ceil(self.group_dim())
    }

    /// Returns the number of groups in Y direction.
    pub fn num_groups_y(&self) -> usize {
        self.height.div_ceil(self.group_dim())
    }

    /// Returns the number of LF groups (DC groups).
    /// LF groups are 8x the size of regular modular groups.
    pub fn num_lf_groups(&self) -> usize {
        let lf_group_dim = self.group_dim() * 8;
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
        let group_dim = self.group_dim();
        let gx = group_idx % num_groups_x;
        let gy = group_idx / num_groups_x;

        let x_start = gx * group_dim;
        let y_start = gy * group_dim;
        let x_end = (x_start + group_dim).min(self.width);
        let y_end = (y_start + group_dim).min(self.height);

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
