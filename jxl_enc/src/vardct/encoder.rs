//! VarDCT frame encoder.
//!
//! Produces VarDCT (lossy) encoded frames from RGB images.

use crate::BLOCK_DIM;
use crate::bit_writer::BitWriter;
use crate::entropy_coding::ans::{AnsDistribution, AnsEncoder};
use crate::error::Result;
use crate::heuristics::{
    AcStrategyMap, ColorCorrelationMap, HeuristicLevel, QuantField, select_ac_strategies,
};
use crate::modular::channel::{Channel, ModularImage};
use crate::modular::improved::write_vardct_modular_substream;
#[allow(unused_imports)]
use crate::{trace_note, trace_section, trace_write};

use super::AcStrategy;
use super::context::BlockContextMap;
use super::enc_coeff::pack_signed;
use super::histogram::HistogramBuilder;
use super::quant_weights::DequantMatrices;
use super::quantizer::QuantizerParams;
use super::tokenize::{
    NATURAL_ORDER_8X8, Token, generate_natural_order, log2_covered_blocks_for_strategy,
};
use super::transform::TransformedDataWithStrategy;

/// VarDCT frame encoding options.
#[derive(Clone, Debug)]
pub struct VarDctOptions {
    /// Butteraugli distance target (0.0 = lossless, 1.0 = high quality).
    pub distance: f32,
    /// Use default quant matrices.
    pub use_default_quant_matrices: bool,
    /// Use default block context map.
    pub use_default_block_ctx: bool,
    /// AC strategy selection heuristics level.
    pub ac_strategy_heuristics: HeuristicLevel,
    /// Enable Chroma-from-Luma correlation.
    pub cfl_enabled: bool,
    /// Enable adaptive quantization field.
    pub adaptive_quant: bool,
    /// Adaptive quantization strength (0.0 = uniform, 1.0 = full).
    pub adaptive_quant_strength: f32,
}

impl Default for VarDctOptions {
    fn default() -> Self {
        Self {
            distance: 1.0,
            use_default_quant_matrices: true,
            use_default_block_ctx: true,
            ac_strategy_heuristics: HeuristicLevel::VarianceBased, // Adaptive DCT sizes
            cfl_enabled: true,    // Chroma-from-Luma for better compression
            adaptive_quant: true, // Per-block quality for perceptual quality
            adaptive_quant_strength: 0.5,
        }
    }
}

/// VarDCT frame encoder.
pub struct VarDctEncoder {
    options: VarDctOptions,
    width: usize,
    height: usize,
    quantizer: QuantizerParams,
    dequant_matrices: DequantMatrices,
    block_ctx_map: BlockContextMap,
    ac_strategy_map: AcStrategyMap,
    color_correlation: ColorCorrelationMap,
    quant_field: QuantField,
}

impl VarDctEncoder {
    /// Create a new VarDCT encoder.
    pub fn new(width: usize, height: usize, options: VarDctOptions) -> Self {
        let quantizer = QuantizerParams::from_distance(options.distance);
        let blocks_x = width.div_ceil(BLOCK_DIM);
        let blocks_y = height.div_ceil(BLOCK_DIM);

        // Create default DCT8-only map; can be updated with compute_ac_strategies
        let ac_strategy_map = AcStrategyMap::new_dct8(blocks_x, blocks_y);

        // Create default color correlation (no CfL)
        let color_correlation = ColorCorrelationMap::new_default(width, height);

        // Create uniform quant field; can be updated with compute_quant_field
        let base_quant = quantizer.quant_dc.min(255) as u8;
        let quant_field = QuantField::uniform(blocks_x, blocks_y, base_quant);

        Self {
            options,
            width,
            height,
            quantizer,
            dequant_matrices: DequantMatrices::default(),
            block_ctx_map: BlockContextMap::default(),
            ac_strategy_map,
            color_correlation,
            quant_field,
        }
    }

    /// Compute AC strategy map based on image content.
    ///
    /// Call this after creating the encoder and before encoding.
    /// The Y plane should be the luminance channel in linear space (or XYB Y).
    pub fn compute_ac_strategies(&mut self, y_plane: &[f32]) {
        self.ac_strategy_map = select_ac_strategies(
            y_plane,
            self.width,
            self.height,
            self.options.ac_strategy_heuristics,
        );
    }

    /// Compute adaptive quant field from image data.
    ///
    /// Call this after creating the encoder and before encoding.
    /// The Y plane should be the luminance channel in linear space (or XYB Y).
    pub fn compute_quant_field(&mut self, y_plane: &[f32]) {
        if self.options.adaptive_quant {
            let base_quant = self.quantizer.quant_dc.min(255) as u8;
            self.quant_field = QuantField::compute_adaptive(
                y_plane,
                self.width,
                self.height,
                base_quant,
                self.options.adaptive_quant_strength,
            );
        }
    }

    /// Get the quant field.
    pub fn quant_field(&self) -> &QuantField {
        &self.quant_field
    }

    /// Compute CfL correlation from XYB image data.
    ///
    /// Call this after creating the encoder and before encoding.
    /// The XYB data should be interleaved (X, Y, B per pixel).
    pub fn compute_color_correlation(&mut self, xyb_data: &[f32]) {
        if self.options.cfl_enabled {
            self.color_correlation =
                ColorCorrelationMap::compute_from_xyb(xyb_data, self.width, self.height);
        }
    }

    /// Get the color correlation map.
    pub fn color_correlation(&self) -> &ColorCorrelationMap {
        &self.color_correlation
    }

    /// Get the AC strategy map.
    pub fn ac_strategy_map(&self) -> &AcStrategyMap {
        &self.ac_strategy_map
    }

    /// Number of 8x8 blocks in X direction.
    pub fn num_blocks_x(&self) -> usize {
        self.width.div_ceil(BLOCK_DIM)
    }

    /// Number of 8x8 blocks in Y direction.
    pub fn num_blocks_y(&self) -> usize {
        self.height.div_ceil(BLOCK_DIM)
    }

    /// Get the quantizer parameters.
    pub fn quantizer(&self) -> &QuantizerParams {
        &self.quantizer
    }

    /// Group dimension in pixels (256x256).
    const GROUP_DIM: usize = 256;

    /// Number of groups in X direction.
    pub fn num_groups_x(&self) -> usize {
        self.width.div_ceil(Self::GROUP_DIM)
    }

    /// Number of groups in Y direction.
    pub fn num_groups_y(&self) -> usize {
        self.height.div_ceil(Self::GROUP_DIM)
    }

    /// Total number of groups.
    pub fn num_groups(&self) -> usize {
        self.num_groups_x() * self.num_groups_y()
    }

    /// Get block range for a group.
    /// Returns (start_bx, start_by, end_bx, end_by) in block coordinates.
    pub fn group_block_range(&self, group_idx: usize) -> (usize, usize, usize, usize) {
        let num_groups_x = self.num_groups_x();
        let gx = group_idx % num_groups_x;
        let gy = group_idx / num_groups_x;

        // Group bounds in pixels
        let px_start_x = gx * Self::GROUP_DIM;
        let px_start_y = gy * Self::GROUP_DIM;
        let px_end_x = (px_start_x + Self::GROUP_DIM).min(self.width);
        let px_end_y = (px_start_y + Self::GROUP_DIM).min(self.height);

        // Convert to block coordinates
        let start_bx = px_start_x / BLOCK_DIM;
        let start_by = px_start_y / BLOCK_DIM;
        let end_bx = px_end_x.div_ceil(BLOCK_DIM);
        let end_by = px_end_y.div_ceil(BLOCK_DIM);

        (start_bx, start_by, end_bx, end_by)
    }

    /// Write the VarDCT frame header.
    ///
    /// This differs from modular by setting encoding=0 and includes
    /// VarDCT-specific fields like x_qm_scale and b_qm_scale.
    /// Note: group_size_shift is ONLY for Modular frames, not VarDCT!
    pub fn write_frame_header(&self, writer: &mut BitWriter) -> Result<()> {
        trace_section!(begin "FRAME_HEADER", writer);

        // Write explicit frame header (matching libjxl reference output)
        trace_write!(writer, 1, 0, "all_default", "false - need VarDCT fields")?;
        trace_write!(writer, 2, 0, "frame_type", "RegularFrame")?;
        trace_write!(writer, 1, 0, "encoding", "VarDCT")?;
        trace_write!(writer, 2, 0, "flags", "U64 selector=0 → value=0")?;

        // NOTE: do_ycbcr is ONLY present when xyb_encoded=false in the file header.
        // Since VarDCT uses XYB color space (xyb_encoded=true), we do NOT write do_ycbcr.
        // The decoder skips this field when xyb_encoded=true.
        //
        // If we were using a non-XYB VarDCT (xyb_encoded=false), we would write:
        //   trace_write!(writer, 1, 0, "do_ycbcr", "false - use YCbCr")?;
        //   If do_ycbcr=1: jpeg_upsampling (3 channels × 2 bits = 6 bits)

        trace_write!(writer, 2, 0, "upsampling", "u2S selector=0 → 1x")?;
        trace_write!(writer, 3, 3, "x_qm_scale", "VarDCT XYB quant matrix scale")?;
        trace_write!(writer, 3, 2, "b_qm_scale", "VarDCT XYB quant matrix scale")?;
        trace_write!(writer, 2, 0, "passes.num_passes", "u2S selector=0 → 1 pass")?;
        trace_write!(writer, 1, 0, "have_crop", "false")?;
        trace_write!(writer, 2, 0, "blending_info.mode", "Replace")?;
        trace_write!(writer, 1, 1, "is_last", "true")?;
        trace_write!(writer, 2, 0, "name_length", "u2S selector=0 → 0")?;

        // restoration_filter section
        trace_section!(begin "RESTORATION_FILTER", writer);
        trace_write!(writer, 1, 0, "all_default", "false - need epf_iters=1")?;
        trace_write!(writer, 1, 1, "gab", "true")?;
        trace_write!(writer, 1, 0, "gab_custom", "false")?;
        trace_write!(
            writer,
            2,
            1,
            "epf_iters",
            "1 (reference uses 1, not default 2)"
        )?;
        trace_write!(writer, 1, 0, "epf_sharp_custom", "false")?;
        trace_write!(writer, 1, 0, "epf_weight_custom", "false")?;
        trace_write!(writer, 1, 0, "epf_sigma_custom", "false")?;
        trace_section!(end "RESTORATION_FILTER", writer);

        trace_write!(writer, 2, 0, "extensions", "U64 selector=0 → none")?;

        trace_section!(end "FRAME_HEADER", writer);
        Ok(())
    }

    /// Write the LF Global section.
    ///
    /// Contains: LfQuantFactors, QuantizerParams, BlockCtxMap, ColorCorrelation, Tree, ModularGlobal.
    pub fn write_lf_global(&self, writer: &mut BitWriter) -> Result<()> {
        trace_section!(begin "LF_GLOBAL", writer);

        // Write LF quant factors (use defaults)
        trace_write!(writer, 1, 1, "lf_quant_factors.all_default", "use defaults")?;

        // Write quantizer params (global_scale, quant_dc)
        self.quantizer.write_traced(writer);
        trace_note!(
            writer,
            "quantizer_params: gs={}, qdc={}",
            self.quantizer.global_scale,
            self.quantizer.quant_dc
        );

        // Write block context map (default = 1 bit)
        self.block_ctx_map.write_traced(writer)?;

        // Write color correlation (LF)
        self.write_color_correlation(writer)?;

        // Write global tree presence (0 = no global tree for VarDCT)
        trace_write!(
            writer,
            1,
            0,
            "has_global_tree",
            "false - VarDCT uses per-group trees"
        )?;

        // ModularGlobal is empty for VarDCT without extra channels
        // (no channels to encode, so nothing to write)

        trace_section!(end "LF_GLOBAL", writer);
        Ok(())
    }

    /// Write color correlation parameters.
    fn write_color_correlation(&self, writer: &mut BitWriter) -> Result<()> {
        trace_section!(begin "COLOR_CORRELATION", writer);
        let cmap = &self.color_correlation;

        // If using default correlation (no CfL), just write all_default=true
        if !self.options.cfl_enabled || cmap.is_default() {
            trace_write!(writer, 1, 1, "all_default", "true - no CfL")?;
            trace_section!(end "COLOR_CORRELATION", writer);
            return Ok(());
        }

        trace_write!(writer, 1, 0, "all_default", "false - custom CfL")?;

        // Write color_factor using U32Enc kColorFactorDist
        // U32Enc: Val(84), Val(256), BitsOffset(8,2), BitsOffset(16,258)
        let color_factor = cmap.color_factor;
        if color_factor == 84 {
            trace_write!(writer, 2, 0, "color_factor", "selector=0 → 84")?;
        } else if color_factor == 256 {
            trace_write!(writer, 2, 1, "color_factor", "selector=1 → 256")?;
        } else if (2..258).contains(&color_factor) {
            trace_write!(writer, 2, 2, "color_factor.selector", "Bits(8)+2")?;
            trace_write!(writer, 8, (color_factor - 2) as u64, "color_factor.value")?;
        } else {
            trace_write!(writer, 2, 3, "color_factor.selector", "Bits(16)+258")?;
            trace_write!(
                writer,
                16,
                (color_factor.saturating_sub(258)) as u64,
                "color_factor.value"
            )?;
        }

        trace_write!(writer, 16, 0, "base_correlation_x", "F16: 0.0")?;
        trace_write!(writer, 16, 0x3C00, "base_correlation_b", "F16: 1.0")?;

        // Write x_factor_lf and b_factor_lf (unsigned 8 bits each, default=128)
        // NOTE: These are the LF channel factors, NOT ytox_dc/ytob_dc (which are DC offsets).
        // The decoder expects u(8) here, not signed varints!
        trace_write!(writer, 8, 128, "x_factor_lf", "default=128")?;
        trace_write!(writer, 8, 128, "b_factor_lf", "default=128")?;

        trace_section!(end "COLOR_CORRELATION", writer);
        Ok(())
    }

    /// Tokenize AC coefficients and build histograms.
    ///
    /// Returns (tokens, distributions) for use in HF global and pass group.
    pub fn tokenize_ac_coefficients(
        &self,
        ac_coeffs: &[i32],
    ) -> Result<(Vec<Token>, Vec<AnsDistribution>)> {
        let blocks_x = self.num_blocks_x();
        let blocks_y = self.num_blocks_y();
        let num_blocks = blocks_x * blocks_y;
        let num_contexts = self.block_ctx_map.num_ac_contexts();

        // Collect tokens from all blocks
        let mut tokens = Vec::with_capacity(num_blocks * 64);

        for block_idx in 0..num_blocks {
            // Get quant field value for this block (uniform)
            let qf = self.quantizer.quant_dc;

            // Process each channel (X=0, Y=1, B=2)
            for c in 0..3 {
                // Get block context
                let lf_idx = 0; // No LF thresholds in default mode
                let order_id = 0; // DCT8
                let block_context = self.block_ctx_map.block_context(lf_idx, qf, order_id, c);

                // Get AC coefficients for this block/channel (63 coeffs per channel)
                let ac_start = block_idx * 3 * 63 + c * 63;
                let block_ac = &ac_coeffs[ac_start..ac_start + 63];

                // Count non-zeros
                let nzeros: usize = block_ac.iter().filter(|&&x| x != 0).count();

                // Emit non-zero count token
                let nz_ctx = self.block_ctx_map.nonzero_context(nzeros, block_context) as u32;
                tokens.push(Token::new(nz_ctx, nzeros as u32));

                if nzeros > 0 {
                    // Get zero-density context offset
                    let histo_offset = self
                        .block_ctx_map
                        .zero_density_context_offset(block_context);

                    // Process coefficients in natural order
                    let mut nzeros_left = nzeros;
                    let mut prev = if nzeros > 4 { 0 } else { 1 };

                    for k in 0..63 {
                        if nzeros_left == 0 {
                            break;
                        }

                        let coeff = block_ac[NATURAL_ORDER_8X8[k + 1] - 1]; // -1 because AC starts at 0
                        let ctx = histo_offset
                            + super::context::zero_density_context(
                                nzeros_left,
                                k + 1, // k is 1-based in context computation
                                0,     // log_num_blocks = 0 for DCT8
                                prev,
                            );

                        let u_coeff = pack_signed(coeff);
                        tokens.push(Token::new(ctx as u32, u_coeff));

                        if coeff != 0 {
                            prev = 1;
                            nzeros_left -= 1;
                        } else {
                            prev = 0;
                        }
                    }
                }
            }
        }

        // Build histograms from tokens
        let mut builder = HistogramBuilder::new(num_contexts);
        builder.add_tokens(&tokens);
        let distributions = builder.build_distributions()?;

        Ok((tokens, distributions))
    }

    /// Tokenize AC coefficients with strategy-aware variable block sizes.
    ///
    /// Handles DCT8, DCT16, and DCT32 blocks based on the strategy map.
    pub fn tokenize_ac_with_strategy(
        &self,
        transformed: &TransformedDataWithStrategy,
    ) -> Result<(Vec<Token>, Vec<AnsDistribution>)> {
        let blocks_x = self.num_blocks_x();
        let blocks_y = self.num_blocks_y();
        let num_blocks = blocks_x * blocks_y;
        let num_contexts = self.block_ctx_map.num_ac_contexts();

        let mut tokens = Vec::with_capacity(num_blocks * 64);

        // Track which blocks have been processed (for DCT16/32)
        let mut processed = vec![false; num_blocks];

        // Generate natural orders for different block sizes
        let order_8 = NATURAL_ORDER_8X8.to_vec();
        let order_16 = generate_natural_order(2, 2);
        let order_32 = generate_natural_order(4, 4);

        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let block_idx = by * blocks_x + bx;

                if processed[block_idx] {
                    continue;
                }

                let strategy = transformed.strategies.get(bx, by);
                let (cx, cy) = match strategy {
                    AcStrategy::Dct32x32 if bx + 3 < blocks_x && by + 3 < blocks_y => (4, 4),
                    AcStrategy::Dct16x16 if bx + 1 < blocks_x && by + 1 < blocks_y => (2, 2),
                    _ => (1, 1),
                };

                // Mark blocks as processed
                for dy in 0..cy {
                    for dx in 0..cx {
                        processed[(by + dy) * blocks_x + (bx + dx)] = true;
                    }
                }

                let order_id = strategy.order_id();
                let log2_blocks = log2_covered_blocks_for_strategy(cx, cy);
                let order: &[usize] = match (cx, cy) {
                    (4, 4) => &order_32,
                    (2, 2) => &order_16,
                    _ => &order_8,
                };

                let covered_blocks = cx * cy;
                let block_size = covered_blocks * 64;
                let ac_size = block_size - covered_blocks; // Subtract DCs

                // Get quant field value
                let qf = self.quantizer.quant_dc;

                // Process each channel
                for c in 0..3 {
                    let block_context = self.block_ctx_map.block_context(0, qf, order_id, c);

                    // Get AC coefficients for this block/channel
                    let ac_start = transformed.ac_offsets[block_idx * 3 + c];
                    let ac_end = if block_idx * 3 + c + 1 < transformed.ac_offsets.len() {
                        transformed.ac_offsets[block_idx * 3 + c + 1]
                    } else {
                        transformed.ac_coeffs.len()
                    };
                    let block_ac = &transformed.ac_coeffs[ac_start..ac_end];

                    // For larger blocks, need to handle coefficient ordering properly
                    // For now, use the AC directly (skipping LLF which is in dc_coeffs)
                    let effective_ac = if block_ac.len() == ac_size {
                        block_ac
                    } else {
                        // Fallback for DCT8
                        block_ac
                    };

                    // Count non-zeros
                    let nzeros: usize = effective_ac.iter().filter(|&&x| x != 0).count();

                    // Emit non-zero count token
                    let nz_ctx = self.block_ctx_map.nonzero_context(nzeros, block_context) as u32;
                    tokens.push(Token::new(nz_ctx, nzeros as u32));

                    if nzeros > 0 {
                        let histo_offset = self
                            .block_ctx_map
                            .zero_density_context_offset(block_context);

                        let mut nzeros_left = nzeros;
                        let mut prev = if nzeros > ac_size / 16 { 0 } else { 1 };

                        // Process coefficients in scan order
                        for k in 0..effective_ac.len() {
                            if nzeros_left == 0 {
                                break;
                            }

                            // Use the appropriate order
                            let coeff_idx = if k + covered_blocks < order.len() {
                                order[k + covered_blocks]
                            } else {
                                k
                            };

                            let coeff = if coeff_idx < block_size {
                                // Map to position within our AC array
                                if coeff_idx < effective_ac.len() {
                                    effective_ac[coeff_idx]
                                } else {
                                    0
                                }
                            } else {
                                0
                            };

                            let ctx = histo_offset
                                + super::context::zero_density_context(
                                    nzeros_left,
                                    k + 1,
                                    log2_blocks,
                                    prev,
                                );

                            let u_coeff = pack_signed(coeff);
                            tokens.push(Token::new(ctx as u32, u_coeff));

                            if coeff != 0 {
                                prev = 1;
                                nzeros_left -= 1;
                            } else {
                                prev = 0;
                            }
                        }
                    }
                }
            }
        }

        // Build histograms from tokens
        let mut builder = HistogramBuilder::new(num_contexts);
        builder.add_tokens(&tokens);
        let distributions = builder.build_distributions()?;

        Ok((tokens, distributions))
    }

    /// Tokenize AC coefficients for a specific group.
    ///
    /// Returns tokens only for blocks within the specified group.
    /// The histograms must have been built from all groups beforehand.
    pub fn tokenize_ac_coefficients_for_group(
        &self,
        ac_coeffs: &[i32],
        group_idx: usize,
    ) -> Vec<Token> {
        let blocks_x = self.num_blocks_x();
        let (start_bx, start_by, end_bx, end_by) = self.group_block_range(group_idx);

        let group_blocks_x = end_bx - start_bx;
        let group_blocks_y = end_by - start_by;
        let num_group_blocks = group_blocks_x * group_blocks_y;

        let mut tokens = Vec::with_capacity(num_group_blocks * 64);

        // Process blocks within this group's bounds
        for by in start_by..end_by {
            for bx in start_bx..end_bx {
                let block_idx = by * blocks_x + bx;

                // Get quant field value for this block (uniform)
                let qf = self.quantizer.quant_dc;

                // Process each channel (X=0, Y=1, B=2)
                for c in 0..3 {
                    // Get block context
                    let lf_idx = 0; // No LF thresholds in default mode
                    let order_id = 0; // DCT8
                    let block_context = self.block_ctx_map.block_context(lf_idx, qf, order_id, c);

                    // Get AC coefficients for this block/channel (63 coeffs per channel)
                    let ac_start = block_idx * 3 * 63 + c * 63;
                    let block_ac = &ac_coeffs[ac_start..ac_start + 63];

                    // Count non-zeros
                    let nzeros: usize = block_ac.iter().filter(|&&x| x != 0).count();

                    // Emit non-zero count token
                    let nz_ctx = self.block_ctx_map.nonzero_context(nzeros, block_context) as u32;
                    tokens.push(Token::new(nz_ctx, nzeros as u32));

                    if nzeros > 0 {
                        // Get zero-density context offset
                        let histo_offset = self
                            .block_ctx_map
                            .zero_density_context_offset(block_context);

                        // Process coefficients in natural order
                        let mut nzeros_left = nzeros;
                        let mut prev = if nzeros > 4 { 0 } else { 1 };

                        for k in 0..63 {
                            if nzeros_left == 0 {
                                break;
                            }

                            let coeff = block_ac[NATURAL_ORDER_8X8[k + 1] - 1];
                            let ctx = histo_offset
                                + super::context::zero_density_context(
                                    nzeros_left,
                                    k + 1,
                                    0, // log_num_blocks = 0 for DCT8
                                    prev,
                                );

                            let u_coeff = pack_signed(coeff);
                            tokens.push(Token::new(ctx as u32, u_coeff));

                            if coeff != 0 {
                                prev = 1;
                                nzeros_left -= 1;
                            } else {
                                prev = 0;
                            }
                        }
                    }
                }
            }
        }

        tokens
    }

    /// Write the HF Global section.
    ///
    /// Contains: DequantMatrices, num_histograms, coeff_orders, histograms.
    pub fn write_hf_global(
        &self,
        tokens: &[Token],
        distributions: &[AnsDistribution],
        writer: &mut BitWriter,
    ) -> Result<()> {
        // Write dequant matrices (default = 1 bit)
        self.dequant_matrices.write(writer);

        // Write num_hf_presets for multi-group frames
        // Format: (num_hf_presets - 1) in ceil_log2(num_groups) bits
        // For single group, ceil_log2(1) = 0 bits (nothing written)
        let num_groups = self.num_groups();
        let num_hf_presets_bits = num_groups.next_power_of_two().trailing_zeros() as usize;
        if num_hf_presets_bits > 0 {
            // Write num_hf_presets - 1 = 0 (we use 1 preset)
            writer.write(num_hf_presets_bits, 0)?;
            eprintln!(
                "HF_GLOBAL [bit {}]: num_hf_presets = 1 (wrote 0 in {} bits)",
                writer.bits_written(),
                num_hf_presets_bits
            );
        }

        // Coefficient order encoding
        // used_orders = 0 means all default orders (no custom permutations)
        writer.write(2, 2)?; // Selector 2 = value 0 (used_orders = 0)

        // Write histograms for each pass (we have 1 pass)
        self.write_histograms(tokens, distributions, writer)?;

        Ok(())
    }

    /// Write histogram set for AC coefficients.
    ///
    /// Uses prefix codes for simplicity. For large alphabets, uses HybridUint
    /// encoding to keep token values bounded.
    fn write_histograms(
        &self,
        tokens: &[Token],
        distributions: &[AnsDistribution],
        writer: &mut BitWriter,
    ) -> Result<()> {
        let start_bit = writer.bits_written();
        eprintln!("HF_HIST [bit {}]: Starting histogram", start_bit);

        // LZ77 enabled = false
        writer.write(1, 0)?;
        eprintln!("HF_HIST [bit {}]: lz77.enabled = 0", writer.bits_written());

        // Context map
        // For simplicity, map all contexts to cluster 0
        let num_contexts = distributions.len();
        eprintln!("HF_HIST: num_contexts = {}", num_contexts);

        if num_contexts == 1 {
            // When num_contexts = 1, read_clusters returns immediately without reading bits
            // (implicit single histogram, no context map written)
        } else {
            // Use simple context map encoding: is_simple=1, bits_per_entry=0
            // This maps all contexts to cluster 0
            writer.write(1, 1)?; // is_simple = true
            writer.write(2, 0)?; // bits_per_entry = 0 (all contexts map to 0)
            eprintln!(
                "HF_HIST [bit {}]: context_map (is_simple=1, bits=0)",
                writer.bits_written()
            );
        }

        // Use prefix codes (Huffman) for simplicity
        writer.write(1, 1)?; // use_prefix_code = true
        eprintln!(
            "HF_HIST [bit {}]: use_prefix_code = 1",
            writer.bits_written()
        );

        // Compute the maximum raw symbol value
        let max_symbol = tokens.iter().map(|t| t.value as usize).max().unwrap_or(0);
        eprintln!("HF_HIST: max_symbol = {}", max_symbol);

        // Use HybridUint encoding to bound token values
        // With split_exponent=4, msb_in_token=2, lsb_in_token=0:
        // - Values 0-15: token = value (direct)
        // - Values 16+: token is bounded, extra bits written separately
        let hybrid_config = crate::entropy_coding::hybrid_uint::HybridUintConfig::new(4, 2, 0);

        // Compute max token value after HybridUint encoding
        let max_token = if max_symbol < hybrid_config.split as usize {
            max_symbol
        } else {
            let (token, _, _) = hybrid_config.encode(max_symbol as u32);
            token as usize
        };
        let alphabet_size = max_token + 1;
        eprintln!(
            "HF_HIST: max_token = {}, alphabet_size = {}",
            max_token, alphabet_size
        );

        // IntegerConfig: split_exponent determines direct vs hybrid encoding
        // log_alphabet_size = 15 for prefix codes (decoder uses 15 for alphabet_size read)
        //
        // Format: split_exponent_bits = add_log2_ceil(15) = 4 bits
        // When split_exponent != 15, also read msb_in_token and lsb_in_token
        writer.write(4, 4)?; // split_exponent = 4

        // msb_in_token (uses add_log2_ceil(4) = 3 bits)
        writer.write(3, 2)?; // msb_in_token = 2

        // lsb_in_token (uses add_log2_ceil(4 - 2) = 2 bits)
        writer.write(2, 0)?; // lsb_in_token = 0
        eprintln!(
            "HF_HIST [bit {}]: IntegerConfig (split=4, msb=2, lsb=0)",
            writer.bits_written()
        );

        // Write alphabet size
        write_alphabet_size(writer, alphabet_size)?;
        eprintln!(
            "HF_HIST [bit {}]: After alphabet_size = {}",
            writer.bits_written(),
            alphabet_size
        );

        // Write prefix codes for the bounded token alphabet
        write_prefix_code(writer, alphabet_size)?;
        eprintln!(
            "HF_HIST [bit {}]: After prefix_code, total = {} bits",
            writer.bits_written(),
            writer.bits_written() - start_bit
        );

        Ok(())
    }

    /// Write the LF Group section.
    ///
    /// VarDCT LF Group contains:
    /// 1. extra_precision (2 bits)
    /// 2. VarDCTLF modular stream (DC coefficients)
    /// 3. ModularLF stream (for extra channels, empty for RGB)
    /// 4. HF metadata (count + 4 modular channels: ytox, ytob, transform, epf)
    pub fn write_lf_group(&self, dc_coeffs: &[i32], writer: &mut BitWriter) -> Result<()> {
        let blocks_x = self.num_blocks_x();
        let blocks_y = self.num_blocks_y();
        let _num_blocks = blocks_x * blocks_y;

        eprintln!(
            "LF_GROUP [bit {}]: Starting, {}x{} blocks",
            writer.bits_written(),
            blocks_x,
            blocks_y
        );

        // 1. extra_precision (2 bits) - 0 for standard precision
        writer.write(2, 0)?;
        eprintln!(
            "LF_GROUP [bit {}]: extra_precision = 0",
            writer.bits_written()
        );

        // 2. VarDCTLF modular stream (DC coefficients)
        self.write_dc_coeffs(dc_coeffs, writer)?;
        eprintln!(
            "LF_GROUP [bit {}]: After DC coefficients",
            writer.bits_written()
        );

        // 3. ModularLF stream - for extra channels
        // We don't have extra channels, so this is empty (nothing to write)

        // 4. HF metadata
        self.write_hf_metadata(writer)?;
        eprintln!(
            "LF_GROUP [bit {}]: After HF metadata, LF Group done",
            writer.bits_written()
        );

        Ok(())
    }

    /// Write HF metadata for the LF Group.
    ///
    /// HF metadata contains:
    /// - count (ceil_log2(upper_bound) bits) - number of transform blocks
    /// - 4 modular channels: ytox_map, ytob_map, transform_image, epf_map
    fn write_hf_metadata(&self, writer: &mut BitWriter) -> Result<()> {
        let blocks_x = self.num_blocks_x();
        let blocks_y = self.num_blocks_y();
        let num_blocks = blocks_x * blocks_y;

        // Color tile size (8x8 blocks per tile)
        let tiles_x = blocks_x.div_ceil(8);
        let tiles_y = blocks_y.div_ceil(8);
        let num_tiles = tiles_x * tiles_y;

        // Count = number of transform blocks (for DCT8, every block is distinct)
        // The count is encoded using ceil_log2(upper_bound) bits
        let upper_bound = num_blocks;
        let count_bits = if upper_bound <= 1 {
            0
        } else {
            (usize::BITS - (upper_bound - 1).leading_zeros()) as usize
        };
        let count = num_blocks; // Every block is a distinct transform
        if count_bits > 0 {
            writer.write(count_bits, (count - 1) as u64)?;
        }
        eprintln!(
            "HF_META [bit {}]: count = {} ({} bits)",
            writer.bits_written(),
            count,
            count_bits
        );

        // Build 4 modular channels for HF metadata:
        // - ytox_map: color tiles size, i8 values
        // - ytob_map: color tiles size, i8 values
        // - transform_image: (count, 2) - pairs of (transform_type, raw_quant-1)
        // - epf_map: block size, epf level values

        // ytox_map: default = 0 (no YtoX correlation)
        let ytox_data: Vec<i32> = vec![0i32; num_tiles];
        let ytox_channel = Channel::from_vec(ytox_data, tiles_x, tiles_y)?;

        // ytob_map: default = 0 (no YtoB correlation beyond default)
        let ytob_data: Vec<i32> = vec![0i32; num_tiles];
        let ytob_channel = Channel::from_vec(ytob_data, tiles_x, tiles_y)?;

        // transform_image: (count, 2) array
        // Row 0: transform_type (0 = DCT8)
        // Row 1: raw_quant - 1
        let mut transform_data = vec![0i32; count * 2];
        for i in 0..count {
            // Transform type (DCT8 = 0)
            transform_data[i] = 0;
            // Raw quant - 1 (stored in second row)
            let bx = i % blocks_x;
            let by = i / blocks_x;
            let quant = self.quant_field.get(bx, by) as i32;
            transform_data[count + i] = (quant - 1).max(0);
        }
        let transform_channel = Channel::from_vec(transform_data, count, 2)?;

        // epf_map: block size, EPF level per block (4 = default enabled)
        let epf_data: Vec<i32> = vec![4i32; num_blocks];
        let epf_channel = Channel::from_vec(epf_data, blocks_x, blocks_y)?;

        // Create modular image with all 4 channels
        let hf_meta_image = ModularImage {
            channels: vec![ytox_channel, ytob_channel, transform_channel, epf_channel],
            bit_depth: 8, // Small values
            is_grayscale: false,
            has_alpha: false,
        };

        // Write GroupHeader for this subbitstream
        self.write_group_header(writer)?;

        // Write HF metadata using modular encoder (Tree + Histogram + Data only)
        write_vardct_modular_substream(&hf_meta_image, writer)?;

        Ok(())
    }

    /// Write a minimal GroupHeader for modular subbitstreams.
    ///
    /// GroupHeader contains:
    /// - use_global_tree (1 bit)
    /// - wp_header (WeightedHeader with all_default bit)
    /// - transforms (count encoded with u2S)
    fn write_group_header(&self, writer: &mut BitWriter) -> Result<()> {
        trace_section!(begin "GROUP_HEADER", writer);

        // use_global_tree = false (we write our own tree)
        trace_write!(writer, 1, 0, "use_global_tree", "false - write local tree")?;

        // wp_header.all_default = true (use default weighted params)
        trace_write!(writer, 1, 1, "wp_header.all_default", "true - use defaults")?;

        // transforms count = 0 (no transforms)
        // u2S(0, 1, Bits(4)+2, Bits(8)+18): selector 0 = value 0
        trace_write!(writer, 2, 0, "transforms", "u2S selector=0 → count=0")?;

        trace_section!(end "GROUP_HEADER", writer);
        Ok(())
    }

    /// Write DC coefficients using modular encoding.
    fn write_dc_coeffs(&self, dc_coeffs: &[i32], writer: &mut BitWriter) -> Result<()> {
        let blocks_x = self.num_blocks_x();
        let blocks_y = self.num_blocks_y();
        let num_blocks = blocks_x * blocks_y;

        // Deinterleave DC coefficients into 3 channels (X, Y, B)
        // Note: jxl-rs expects order Y, X, B (channel 1, 0, 2 due to XYB ordering)
        let mut y_dc = vec![0i32; num_blocks];
        let mut x_dc = vec![0i32; num_blocks];
        let mut b_dc = vec![0i32; num_blocks];

        for i in 0..num_blocks {
            x_dc[i] = dc_coeffs[i * 3];
            y_dc[i] = dc_coeffs[i * 3 + 1];
            b_dc[i] = dc_coeffs[i * 3 + 2];
        }

        eprintln!(
            "DC_COEFFS: blocks={}x{}, y_dc={:?}, x_dc={:?}, b_dc={:?}",
            blocks_x,
            blocks_y,
            &y_dc[..y_dc.len().min(10)],
            &x_dc[..x_dc.len().min(10)],
            &b_dc[..b_dc.len().min(10)]
        );

        // Create ModularImage from DC channels (Y, X, B order for VarDCTLF)
        let dc_image = ModularImage {
            channels: vec![
                Channel::from_vec(y_dc, blocks_x, blocks_y)?,
                Channel::from_vec(x_dc, blocks_x, blocks_y)?,
                Channel::from_vec(b_dc, blocks_x, blocks_y)?,
            ],
            bit_depth: 16, // DC coefficients can be larger
            is_grayscale: false,
            has_alpha: false,
        };

        // Write GroupHeader for this subbitstream
        self.write_group_header(writer)?;

        // Write DC coefficients using the modular encoder (Tree + Histogram + Data only)
        write_vardct_modular_substream(&dc_image, writer)?;

        Ok(())
    }

    /// Write the Pass Group section.
    ///
    /// Contains: AC coefficients for this group/pass.
    /// Uses HybridUint encoding with prefix codes to match write_histograms.
    pub fn write_pass_group(
        &self,
        tokens: &[Token],
        _distributions: &[AnsDistribution],
        writer: &mut BitWriter,
    ) -> Result<()> {
        // For multi-group: histograms_id would be written here
        // For single group, it's implicit (id = 0)

        if tokens.is_empty() {
            return Ok(());
        }

        // Use the same HybridUint config as write_histograms
        let hybrid_config = crate::entropy_coding::hybrid_uint::HybridUintConfig::new(4, 2, 0);

        // Compute max token value after HybridUint encoding (for alphabet_size)
        let max_symbol = tokens.iter().map(|t| t.value as usize).max().unwrap_or(0);
        let max_token = if max_symbol < hybrid_config.split as usize {
            max_symbol
        } else {
            let (token, _, _) = hybrid_config.encode(max_symbol as u32);
            token as usize
        };
        let alphabet_size = max_token + 1;

        if alphabet_size <= 1 {
            // Single symbol - nothing to write (decoder knows the only possibility)
            return Ok(());
        }

        // Build canonical Huffman codes matching write_complex_prefix_code
        // For near-flat distribution with n symbols:
        // - d = ceil(log2(n))
        // - (2^d - n) symbols get depth d-1
        // - (2n - 2^d) symbols get depth d
        let (codes, code_lengths) = build_canonical_huffman_codes(alphabet_size);

        eprintln!(
            "PASS_GROUP: alphabet_size={}, first 10 codes: {:?}, first 10 lengths: {:?}",
            alphabet_size,
            &codes[..alphabet_size.min(10)],
            &code_lengths[..alphabet_size.min(10)]
        );

        // Encode each value using HybridUint, then write with correct Huffman code
        let mut token_count = 0;
        for token in tokens {
            let (encoded_token, extra_bits, num_extra_bits) = hybrid_config.encode(token.value);
            let sym = encoded_token as usize;
            let code = codes[sym];
            let len = code_lengths[sym] as usize;
            if token_count < 5 {
                eprintln!(
                    "PASS_GROUP: token {}: value={}, encoded_token={}, code={:#b} ({} bits), extra_bits={:#x} ({} bits)",
                    token_count, token.value, encoded_token, code, len, extra_bits, num_extra_bits
                );
            }
            writer.write(len, code as u64)?;
            if num_extra_bits > 0 {
                writer.write(num_extra_bits as usize, extra_bits as u64)?;
            }
            token_count += 1;
        }

        Ok(())
    }

    /// Write the Pass Group section using ANS encoding.
    ///
    /// This is the proper ANS-encoded version for better compression.
    #[allow(dead_code)]
    pub fn write_pass_group_ans(
        &self,
        tokens: &[Token],
        distributions: &[AnsDistribution],
        context_map: &[usize],
        writer: &mut BitWriter,
    ) -> Result<()> {
        if tokens.is_empty() {
            return Ok(());
        }

        let mut encoder = AnsEncoder::new();

        // Process tokens in reverse order (ANS requirement)
        for token in tokens.iter().rev() {
            let ctx = token.context as usize;
            let dist_idx = context_map.get(ctx).copied().unwrap_or(0);

            if let Some(dist) = distributions.get(dist_idx)
                && let Some(info) = dist.get(token.value as usize)
            {
                encoder.put_symbol(info);
            }
        }

        // Finalize and write the encoded data
        encoder.finalize(writer)?;

        Ok(())
    }
}

/// Write a variable-length uint8.
#[allow(dead_code)] // May be used in future for custom quant matrices
fn write_var_len_uint8(writer: &mut BitWriter, n: u8) -> Result<()> {
    if n == 0 {
        writer.write(1, 0)?;
    } else {
        writer.write(1, 1)?;
        let nbits = 8 - n.leading_zeros();
        writer.write(3, (nbits - 1) as u64)?;
        writer.write((nbits - 1) as usize, (n as u64) - (1u64 << (nbits - 1)))?;
    }
    Ok(())
}

/// Reverse the bits of a value within a given bit length.
fn bit_reverse(value: u32, len: u8) -> u32 {
    if len == 0 {
        return 0;
    }
    let mut result = 0u32;
    let mut v = value;
    for _ in 0..len {
        result = (result << 1) | (v & 1);
        v >>= 1;
    }
    result
}

/// Build canonical Huffman codes for near-flat distribution.
///
/// Returns (codes, code_lengths) where:
/// - codes[i] is the bit-reversed canonical code for symbol i (for LSB-first bitstream)
/// - code_lengths[i] is the bit length for symbol i
///
/// For n symbols where d = ceil(log2(n)):
/// - First (2^d - n) symbols get depth d-1
/// - Remaining symbols get depth d
fn build_canonical_huffman_codes(alphabet_size: usize) -> (Vec<u32>, Vec<u8>) {
    if alphabet_size <= 1 {
        return (vec![0], vec![0]);
    }

    let d = (usize::BITS - (alphabet_size - 1).leading_zeros()) as usize;
    let pow2_d = 1usize << d;

    // Number of symbols at each depth
    let symbols_short = pow2_d.saturating_sub(alphabet_size); // depth d-1
    let symbols_long = alphabet_size.saturating_sub(symbols_short); // depth d

    let (depth_short, depth_long) = if d <= 1 || symbols_short == 0 {
        (d, d)
    } else {
        (d - 1, d)
    };

    // Build code lengths array
    let mut code_lengths = vec![0u8; alphabet_size];
    for i in 0..symbols_short {
        code_lengths[i] = depth_short as u8;
    }
    for i in symbols_short..alphabet_size {
        code_lengths[i] = depth_long as u8;
    }

    // Build canonical codes
    // Sort symbols by (code_length, symbol) - already sorted since shorter symbols come first
    let mut codes = vec![0u32; alphabet_size];
    let mut code = 0u32;
    let mut prev_len = 0u8;

    for sym in 0..alphabet_size {
        let len = code_lengths[sym];
        if len == 0 {
            continue;
        }
        if len > prev_len {
            code <<= len - prev_len;
        }
        // Store bit-reversed code for LSB-first bitstream
        codes[sym] = bit_reverse(code, len);
        code += 1;
        prev_len = len;
    }

    (codes, code_lengths)
}

/// Write alphabet size for Huffman/prefix codes.
///
/// The format for prefix codes is:
/// - If count == 1: write 0 (1 bit)
/// - If count > 1: write 1 (1 bit), then encode (count - 1) as:
///   - n = ceil(log2(count - 1)) bits needed
///   - write n (4 bits)
///   - write (count - 1 - (1 << n)) using n bits
fn write_alphabet_size(writer: &mut BitWriter, size: usize) -> Result<()> {
    if size == 1 {
        writer.write(1, 0)?; // count = 1
    } else {
        writer.write(1, 1)?; // count > 1
        let count_minus_1 = size - 1;
        // Find n such that 1 + (1 << n) + extra = count, where extra < (1 << n)
        // count - 1 = (1 << n) + extra
        // n = floor(log2(count - 1))
        let n = (usize::BITS - 1 - count_minus_1.leading_zeros()) as usize;
        let extra = count_minus_1 - (1 << n);
        writer.write(4, n as u64)?;
        writer.write(n, extra as u64)?;
    }
    Ok(())
}

/// Write a prefix code (Huffman) for a given alphabet size.
///
/// The format is based on Brotli prefix codes:
/// - hskip (2 bits): 1 = simple code, 0/2/3 = complex code with skip count
///
/// For simple codes (alphabet_size <= 4):
/// - nsym - 1 (2 bits)
/// - symbols (alphabet_bits each)
/// - tree_selector (1 bit) for 4 symbols
///
/// For complex codes (alphabet_size > 4):
/// - Uses Brotli-style code length encoding
fn write_prefix_code(writer: &mut BitWriter, alphabet_size: usize) -> Result<()> {
    if alphabet_size == 0 {
        return Ok(());
    }

    // Calculate bits needed to represent symbols
    let max_bits = if alphabet_size <= 1 {
        0
    } else {
        (usize::BITS - (alphabet_size - 1).leading_zeros()) as usize
    };

    if alphabet_size == 1 {
        // Single symbol: simple code with nsym=1
        // hskip=1, nsym-1=0, symbol=0
        writer.write(2, 1)?; // hskip = 1 (simple)
        writer.write(2, 0)?; // nsym - 1 = 0
        writer.write(max_bits.max(1), 0)?; // symbol 0 (at least 1 bit)
    } else if alphabet_size == 2 {
        // Two symbols: simple code with nsym=2
        // hskip=1, nsym-1=1, sym0, sym1
        writer.write(2, 1)?; // hskip = 1 (simple)
        writer.write(2, 1)?; // nsym - 1 = 1
        writer.write(max_bits, 0)?; // symbol 0
        writer.write(max_bits, 1)?; // symbol 1
    } else if alphabet_size == 3 {
        // Three symbols: simple code with nsym=3
        // Depths: 1, 2, 2 → codes: 0, 10, 11
        writer.write(2, 1)?; // hskip = 1 (simple)
        writer.write(2, 2)?; // nsym - 1 = 2
        writer.write(max_bits, 0)?; // symbol 0 (depth 1)
        writer.write(max_bits, 1)?; // symbol 1 (depth 2)
        writer.write(max_bits, 2)?; // symbol 2 (depth 2)
    } else if alphabet_size == 4 {
        // Four symbols: simple code with nsym=4
        // Depths: 2, 2, 2, 2 (flat) → tree_selector=0
        writer.write(2, 1)?; // hskip = 1 (simple)
        writer.write(2, 3)?; // nsym - 1 = 3
        writer.write(max_bits, 0)?; // symbol 0
        writer.write(max_bits, 1)?; // symbol 1
        writer.write(max_bits, 2)?; // symbol 2
        writer.write(max_bits, 3)?; // symbol 3
        writer.write(1, 0)?; // tree_selector = 0 (flat: depths 2,2,2,2)
    } else {
        // Complex code: use Brotli-style encoding
        // For simplicity, encode a flat distribution where all symbols have
        // the same code length.
        write_complex_prefix_code(writer, alphabet_size)?;
    }

    Ok(())
}

/// Write a flat ANS histogram for large alphabets.
///
/// For flat ANS distribution:
/// - use_prefix_code = 0 (use ANS)
/// - IntegerConfig with appropriate log_alphabet_size
/// - is_flat = 1 (flat distribution)
/// - log_alphabet_size bits for the alphabet size
fn write_flat_ans_histogram(writer: &mut BitWriter, alphabet_size: usize) -> Result<()> {
    // use_prefix_code = 0 (use ANS)
    writer.write(1, 0)?;

    // Calculate log_alphabet_size (ceil(log2(alphabet_size)))
    let log_alpha = if alphabet_size <= 1 {
        0
    } else {
        (usize::BITS - (alphabet_size - 1).leading_zeros()) as usize
    };

    // IntegerConfig for ANS
    // split_exponent_bits = add_log2_ceil(log_alphabet_size) + 3
    // When log_alphabet_size is known, split_exponent is read with the calculated bits
    // For log_alphabet_size, use add_log2_ceil which returns ceil(log2(max + 1)) for range [0, max]
    // log_alphabet_size fits in [0, 15] so we need 4 bits for it
    let split_bits = if log_alpha == 0 { 0 } else { log_alpha };
    writer.write(4, split_bits as u64)?; // split_exponent = log_alpha (raw symbols)

    // When split_exponent == log_alphabet_size (which it does for raw symbols),
    // msb_in_token and lsb_in_token are implicitly 0 (not read).

    // Write alphabet_size using the standard encoding
    write_alphabet_size(writer, alphabet_size)?;

    // Now write the ANS distribution
    // is_flat = 1 (flat distribution - all symbols equiprobable)
    writer.write(1, 1)?;

    // For flat distribution, write log_alphabet_size bits for (alphabet_size - 1)
    // This tells the decoder how many symbols are in the flat distribution
    if log_alpha > 0 {
        writer.write(log_alpha, (alphabet_size - 1) as u64)?;
    }

    Ok(())
}

/// Write a complex prefix code using Brotli-style encoding.
///
/// For alphabet_size > 4, we need to encode using the Brotli format:
/// 1. Write hskip (2 bits): 0, 2, or 3 (number of code length symbols to skip)
/// 2. Write code length code lengths for each symbol in storage order
/// 3. Write actual code lengths using the code length Huffman tree
fn write_complex_prefix_code(writer: &mut BitWriter, alphabet_size: usize) -> Result<()> {
    eprintln!(
        "COMPLEX_PREFIX [bit {}]: alphabet_size={}",
        writer.bits_written(),
        alphabet_size
    );

    // Compute code lengths for a near-flat distribution
    // For n symbols with ceil(log2(n)) = d:
    // - Some symbols get depth d-1 (shorter codes)
    // - Remaining symbols get depth d (longer codes)
    let d = (usize::BITS - (alphabet_size - 1).leading_zeros()) as usize;
    let pow2_d = 1usize << d;

    // Number of symbols at depth d-1 vs depth d
    // x + y = n, 2x + y = 2^d → x = 2^d - n
    let symbols_short = pow2_d.saturating_sub(alphabet_size); // depth d-1
    let symbols_long = alphabet_size.saturating_sub(symbols_short); // depth d

    // If d == 1, all symbols have depth 1 (special case)
    let (depth_short, depth_long) = if d <= 1 {
        (1, 1)
    } else {
        (d - 1, d)
    };

    eprintln!(
        "COMPLEX_PREFIX: d={}, symbols_short={} (depth {}), symbols_long={} (depth {})",
        d, symbols_short, depth_short, symbols_long, depth_long
    );

    // Build code length array
    let mut code_lengths = vec![0u8; alphabet_size];
    for i in 0..symbols_short {
        code_lengths[i] = depth_short as u8;
    }
    for i in symbols_short..alphabet_size {
        code_lengths[i] = depth_long as u8;
    }

    // Now encode using Brotli format
    // Storage order for code length symbols
    const STORAGE_ORDER: [u8; 18] = [1, 2, 3, 4, 0, 5, 17, 6, 16, 7, 8, 9, 10, 11, 12, 13, 14, 15];

    // Code length code lengths encoding:
    // Decoder uses U32(0, 4, 3, 8) then additional bits for base=8.
    // In LSB-first bitstream order:
    // 0 → 00 (2 bits) - selector 0 → 0
    // 1 → 0111 (4 bits) - selector 3 (11) → 8, then 1, then 0 → len 1
    // 2 → 011 (3 bits) - selector 3 (11) → 8, then 0 → len 2
    // 3 → 10 (2 bits) - selector 2 → 3
    // 4 → 01 (2 bits) - selector 1 → 4
    // 5 → 1111 (4 bits) - selector 3 (11) → 8, then 1, then 1 → len 5
    const CL_CODE_BITS: [u8; 6] = [0b00, 0b0111, 0b011, 0b10, 0b01, 0b1111];
    const CL_CODE_LENS: [u8; 6] = [2, 4, 3, 2, 2, 4];

    // Determine which code length symbols are used
    let cl_used_short = depth_short; // code length symbol for short codes
    let cl_used_long = depth_long; // code length symbol for long codes

    // Build code length tree for the used symbols
    // We only use 1-2 distinct code lengths
    let single_depth = symbols_short == 0 || symbols_long == 0 || depth_short == depth_long;

    if single_depth {
        // All symbols have the same code length
        let cl_sym = if symbols_short > 0 { cl_used_short } else { cl_used_long };

        // Find position in storage order
        let pos = STORAGE_ORDER.iter().position(|&x| x == cl_sym as u8).unwrap_or(0);

        // hskip: number of zeros at start of code length code lengths
        // We can skip up to 3 of the first symbols in storage order if they're 0
        let skip = pos.min(3);

        // Write hskip (2 bits)
        // 0 = no skip, 2 = skip first 2, 3 = skip first 3
        let hskip = if skip >= 2 { skip } else { 0 };
        writer.write(2, hskip as u64)?;

        // Write code length code lengths
        // For single symbol in code length tree, we write code length 0 for it
        // (special case: nonzero_count == 1)
        for i in hskip..=pos {
            let cl_sym_at_i = STORAGE_ORDER[i];
            let cl_cl = if cl_sym_at_i as usize == cl_sym { 1u8 } else { 0u8 };
            writer.write(CL_CODE_LENS[cl_cl as usize] as usize, CL_CODE_BITS[cl_cl as usize] as u64)?;
        }

        // Now emit code lengths for each symbol
        // With single code length symbol having code 0 (since it's the only one),
        // we don't emit any bits per symbol
        // But we need to emit `alphabet_size` instances...
        // Actually, the decoder reads using the code length Huffman tree.
        // With only one symbol (cl_sym), each read returns that symbol with 0 bits.

        // Nothing more to write - decoder will read 0 bits per code length
    } else {
        // Two distinct code lengths
        // Build a code length tree with symbols depth_short and depth_long

        // Find positions in storage order
        let pos_short = STORAGE_ORDER.iter().position(|&x| x == cl_used_short as u8).unwrap_or(0);
        let pos_long = STORAGE_ORDER.iter().position(|&x| x == cl_used_long as u8).unwrap_or(0);
        let min_pos = pos_short.min(pos_long);
        let max_pos = pos_short.max(pos_long);

        // Skip if possible (both positions > skip)
        let skip = if min_pos >= 3 { 3 } else if min_pos >= 2 { 2 } else { 0 };

        eprintln!(
            "COMPLEX_PREFIX: pos_short={}, pos_long={}, min={}, max={}, skip={}",
            pos_short, pos_long, min_pos, max_pos, skip
        );

        // hskip
        writer.write(2, skip as u64)?;
        eprintln!(
            "COMPLEX_PREFIX [bit {}]: wrote hskip={}",
            writer.bits_written(),
            skip
        );

        // Write code length code lengths up to and including max_pos
        // Both symbols get code length 1 (since we have exactly 2 symbols,
        // they each get 1 bit: 0 for one, 1 for the other)
        let mut space = 32i32;
        let mut num_codes = 0;
        for i in skip..=max_pos {
            let cl_sym_at_i = STORAGE_ORDER[i] as usize;
            let cl_cl = if cl_sym_at_i == cl_used_short || cl_sym_at_i == cl_used_long {
                1u8 // code length 1 for this symbol
            } else {
                0u8 // not used
            };
            writer.write(CL_CODE_LENS[cl_cl as usize] as usize, CL_CODE_BITS[cl_cl as usize] as u64)?;
            eprintln!(
                "COMPLEX_PREFIX [bit {}]: storage[{}]={} -> cl_cl={}, bits={:#b} ({} bits)",
                writer.bits_written(),
                i,
                cl_sym_at_i,
                cl_cl,
                CL_CODE_BITS[cl_cl as usize],
                CL_CODE_LENS[cl_cl as usize]
            );
            if cl_cl != 0 {
                space -= 32 >> cl_cl;
                num_codes += 1;
            }
        }
        eprintln!(
            "COMPLEX_PREFIX: After cl-cl: space={}, num_codes={}",
            space, num_codes
        );

        // Assign codes to the two code length symbols
        // The one that appears first in storage order gets code 0
        // The one that appears second gets code 1
        let (first_cl, _second_cl) = if pos_short < pos_long {
            (cl_used_short, cl_used_long)
        } else {
            (cl_used_long, cl_used_short)
        };

        // Now emit code lengths for each alphabet symbol
        // symbols_short symbols use first_cl (if pos_short < pos_long) or second_cl
        // symbols_long symbols use the other
        let start_bit = writer.bits_written();
        for i in 0..alphabet_size {
            let sym_cl = code_lengths[i] as usize;
            let bit = if sym_cl == first_cl { 0u64 } else { 1u64 };
            writer.write(1, bit)?;
        }
        eprintln!(
            "COMPLEX_PREFIX [bit {}]: wrote {} code lengths ({} bits)",
            writer.bits_written(),
            alphabet_size,
            writer.bits_written() - start_bit
        );
    }

    Ok(())
}

/// Write a signed 32-bit integer using JXL S32 encoding.
///
/// S32 uses a variable-length encoding:
/// - Small values (±63) use fewer bits
/// - Larger values use more bits
fn write_signed_varint_traced(writer: &mut BitWriter, value: i32, field: &str) -> Result<()> {
    // Convert signed to unsigned using zigzag encoding
    let unsigned = if value >= 0 {
        (value as u32) << 1
    } else {
        ((-value as u32) << 1) - 1
    };

    // Use variable-length encoding similar to U32
    if unsigned == 0 {
        trace_write!(
            writer,
            2,
            0,
            field,
            &format!("selector=0 → {} (zigzag=0)", value)
        )?;
    } else if unsigned <= 16 {
        trace_write!(writer, 2, 1, &format!("{}.selector", field), "1")?;
        trace_write!(
            writer,
            4,
            (unsigned - 1) as u64,
            &format!("{}.value", field),
            &format!("{} (zigzag={})", value, unsigned)
        )?;
    } else if unsigned <= 272 {
        trace_write!(writer, 2, 2, &format!("{}.selector", field), "2")?;
        trace_write!(
            writer,
            8,
            (unsigned - 17) as u64,
            &format!("{}.value", field),
            &format!("{} (zigzag={})", value, unsigned)
        )?;
    } else {
        trace_write!(writer, 2, 3, &format!("{}.selector", field), "3")?;
        trace_write!(
            writer,
            12,
            (unsigned.saturating_sub(273)) as u64,
            &format!("{}.value", field),
            &format!("{} (zigzag={})", value, unsigned)
        )?;
    }

    Ok(())
}

fn write_signed_varint(writer: &mut BitWriter, value: i32) -> Result<()> {
    // Convert signed to unsigned using zigzag encoding
    let unsigned = if value >= 0 {
        (value as u32) << 1
    } else {
        ((-value as u32) << 1) - 1
    };

    // Use variable-length encoding similar to U32
    // U32Enc: Val(0), BitsOffset(4, 1), BitsOffset(8, 17), BitsOffset(12, 273)
    if unsigned == 0 {
        writer.write(2, 0)?; // selector 0 = 0
    } else if unsigned <= 16 {
        writer.write(2, 1)?; // selector 1
        writer.write(4, (unsigned - 1) as u64)?;
    } else if unsigned <= 272 {
        writer.write(2, 2)?; // selector 2
        writer.write(8, (unsigned - 17) as u64)?;
    } else {
        writer.write(2, 3)?; // selector 3
        writer.write(12, (unsigned.saturating_sub(273)) as u64)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vardct::quantizer::GLOBAL_SCALE_DENOM;

    #[test]
    fn test_vardct_encoder_creation() {
        let enc = VarDctEncoder::new(64, 64, VarDctOptions::default());
        assert_eq!(enc.num_blocks_x(), 8);
        assert_eq!(enc.num_blocks_y(), 8);
    }

    #[test]
    fn test_write_frame_header() {
        let enc = VarDctEncoder::new(64, 64, VarDctOptions::default());
        let mut writer = BitWriter::new();
        enc.write_frame_header(&mut writer).unwrap();
        assert!(writer.bits_written() > 0);
    }

    #[test]
    fn test_write_lf_global() {
        let enc = VarDctEncoder::new(64, 64, VarDctOptions::default());
        let mut writer = BitWriter::new();
        enc.write_lf_global(&mut writer).unwrap();
        assert!(writer.bits_written() > 0);
    }

    #[test]
    fn test_quantizer_from_distance() {
        // Distance 1.0 should give reasonable quantizer
        let params = QuantizerParams::from_distance(1.0);
        assert!(params.global_scale > 0);
        assert!(params.global_scale <= GLOBAL_SCALE_DENOM);
        assert!(params.quant_dc > 0);
    }
}
