//! VarDCT frame encoder.
//!
//! Produces VarDCT (lossy) encoded frames from RGB images.

use crate::BLOCK_DIM;
use crate::bit_writer::BitWriter;
use crate::entropy_coding::ans::{AnsDistribution, AnsEncoder};
use crate::entropy_coding::encode_context_map;
use crate::error::Result;
use crate::heuristics::{
    AcStrategyMap, ColorCorrelationMap, HeuristicLevel, QuantField, select_ac_strategies,
};
use crate::modular::channel::{Channel, ModularImage};
use crate::modular::improved::write_vardct_modular_substream;
#[allow(unused_imports)]
use crate::{trace_note, trace_section, trace_write};

use super::context::BlockContextMap;
use super::enc_coeff::pack_signed;
use super::histogram::{ClusteredHistogramSet, HistogramBuilder};
use super::prefix_codes::{build_canonical_huffman_codes, write_alphabet_size, write_prefix_code};
use super::quant_weights::DequantMatrices;
use super::quantizer::{Quantizer, QuantizerParams};
use super::tokenize::{Token, ZIGZAG_ORDER_8X8, generate_natural_order};
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
            ac_strategy_heuristics: HeuristicLevel::VarianceBased, // DCT8/16/32 based on image content
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
        // The quant field stores raw AC quant values, computed from quant_field_target = 5.0
        let quant = Quantizer::new(quantizer.clone());
        let base_quant = quant.quant_from_field(5.0).min(255) as u8;
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
    /// Takes interleaved XYB data and extracts the Y plane for variance analysis.
    pub fn compute_ac_strategies(&mut self, xyb_interleaved: &[f32]) {
        // Extract Y plane from interleaved XYB data (Y is at index 1 of each pixel)
        let num_pixels = self.width * self.height;
        let y_plane: Vec<f32> = (0..num_pixels)
            .map(|i| xyb_interleaved[i * 3 + 1])
            .collect();

        self.ac_strategy_map = select_ac_strategies(
            &y_plane,
            self.width,
            self.height,
            self.options.ac_strategy_heuristics,
        );
    }

    /// Compute adaptive quant field from image data.
    ///
    /// Call this after creating the encoder and before encoding.
    /// Takes interleaved XYB data and extracts the Y plane for analysis.
    pub fn compute_quant_field(&mut self, xyb_interleaved: &[f32]) {
        if self.options.adaptive_quant {
            // Extract Y plane from interleaved XYB data
            let num_pixels = self.width * self.height;
            let y_plane: Vec<f32> = (0..num_pixels)
                .map(|i| xyb_interleaved[i * 3 + 1])
                .collect();

            // The quant field stores raw AC quant values, computed from quant_field_target = 5.0
            let quant = Quantizer::new(self.quantizer.clone());
            let base_quant = quant.quant_from_field(5.0).min(255) as u8;
            self.quant_field = QuantField::compute_adaptive(
                &y_plane,
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

        // Track non-zeros per column per channel (for prediction)
        // This matches the decoder's non_zeros_grid_row
        let mut nz_grid: [Vec<u32>; 3] = [vec![0; blocks_x], vec![0; blocks_x], vec![0; blocks_x]];

        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let block_idx = by * blocks_x + bx;

                // Get quant field value for this block (uniform)
                let qf = self.quantizer.quant_dc;

                // Process channels matching decoder order.
                // jxl-oxide uses loop index (0, 1, 2) for context computation (ch_idx),
                // but remaps to [1, 0, 2] for actual data access (Y, X, B).
                const CHANNEL_REMAP: [usize; 3] = [1, 0, 2];
                for ctx_idx in 0..3 {
                    let c = CHANNEL_REMAP[ctx_idx]; // Data channel

                    // Get block context - uses ctx_idx (0, 1, 2), NOT remapped channel
                    let lf_idx = 0; // No LF thresholds in default mode
                    let order_id = 0; // DCT8
                    let block_context = self.block_ctx_map.block_context(lf_idx, qf, order_id, ctx_idx);

                    // Get AC coefficients for this block/channel (63 coeffs per channel)
                    let ac_start = block_idx * 3 * 63 + c * 63;
                    let block_ac = &ac_coeffs[ac_start..ac_start + 63];

                    // Count actual non-zeros
                    let nzeros: usize = block_ac.iter().filter(|&&x| x != 0).count();

                    // Compute predicted non-zeros from neighbors (matching decoder)
                    // Decoder uses channel index (c) for grid, not iteration index
                    let predicted = if by == 0 {
                        if bx == 0 {
                            32usize // First block uses fixed prediction
                        } else {
                            nz_grid[c][bx - 1] as usize
                        }
                    } else if bx == 0 {
                        nz_grid[c][bx] as usize
                    } else {
                        ((nz_grid[c][bx] + nz_grid[c][bx - 1] + 1) >> 1) as usize
                    };

                    // Use prediction for context (matching decoder)
                    let nz_ctx =
                        self.block_ctx_map.nonzero_context(predicted, block_context) as u32;

                    // Emit actual non-zeros value with predicted context
                    tokens.push(Token::new(nz_ctx, nzeros as u32));

                    // Store actual non-zeros for future predictions
                    // Use channel index (c) for grid
                    nz_grid[c][bx] = nzeros as u32;

                    if nzeros > 0 {
                        // Get zero-density context offset
                        let histo_offset = self
                            .block_ctx_map
                            .zero_density_context_offset(block_context);

                        // Process coefficients in natural order
                        let mut nzeros_left = nzeros;
                        // Decoder: is_prev_coeff_nonzero = (non_zeros <= num_blocks * 4) as u32
                        // For DCT8x8 (num_blocks=1), this is: (nzeros <= 4) ? 1 : 0
                        let mut prev = if nzeros <= 4 { 1 } else { 0 };

                        for k in 0..63 {
                            if nzeros_left == 0 {
                                break;
                            }

                            // Pre-transpose coefficient index for DCT8 because jxl-oxide
                            // transposes coordinates when h >= w (which is true for 8x8).
                            // Our DCT output is dct[v*8+u]. To compensate for decoder transpose,
                            // we access dct[u*8+v] instead (swap row/col indices).
                            let orig_idx = ZIGZAG_ORDER_8X8[k + 1];
                            let transposed_idx = (orig_idx % 8) * 8 + (orig_idx / 8);
                            let coeff = block_ac[transposed_idx - 1]; // -1 because AC starts at 0
                            let ctx = histo_offset
                                + super::context::zero_density_context(
                                    nzeros_left,
                                    k, // k is 0-based, matching decoder's idx
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

        // Track non-zeros per column per channel (for prediction, matching decoder)
        let mut nz_grid: [Vec<u32>; 3] = [vec![0; blocks_x], vec![0; blocks_x], vec![0; blocks_x]];

        // Generate natural orders for different block sizes
        // jxl-oxide's natural_order for DCT8 matches ZIGZAG_ORDER_8X8.
        let order_8 = ZIGZAG_ORDER_8X8.to_vec();
        let order_16 = generate_natural_order(2, 2);
        let order_32 = generate_natural_order(4, 4);

        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let block_idx = by * blocks_x + bx;

                if processed[block_idx] {
                    continue;
                }

                let strategy = transformed.strategies.get(bx, by);
                // Get the actual coverage from the strategy
                let (cx, cy) = strategy.covered_blocks();

                // Clamp to image bounds for safety (shouldn't happen with proper strategy selection)
                let actual_cx = cx.min(blocks_x - bx);
                let actual_cy = cy.min(blocks_y - by);

                // Mark all covered blocks as processed
                for dy in 0..actual_cy {
                    for dx in 0..actual_cx {
                        processed[(by + dy) * blocks_x + (bx + dx)] = true;
                    }
                }

                let order_id = strategy.order_id();
                let log2_blocks = strategy.log2_covered_blocks() as usize;
                let order: &[usize] = match (cx, cy) {
                    (4, 4) => &order_32,
                    (2, 2) => &order_16,
                    _ => &order_8,
                };

                let covered_blocks = cx * cy;

                // Get quant field value
                let qf = self.quantizer.quant_dc;

                // Process channels matching decoder order.
                // jxl-oxide uses loop index (0, 1, 2) for context computation (ch_idx),
                // but remaps to [1, 0, 2] for actual data access (Y, X, B).
                // - ctx_idx=0 -> data channel 1 (Y)
                // - ctx_idx=1 -> data channel 0 (X)
                // - ctx_idx=2 -> data channel 2 (B)
                const CHANNEL_REMAP: [usize; 3] = [1, 0, 2]; // Y, X, B
                for ctx_idx in 0..3 {
                    let c = CHANNEL_REMAP[ctx_idx]; // Data channel
                    // Context uses ctx_idx (0, 1, 2), NOT the remapped channel
                    let block_context = self.block_ctx_map.block_context(0, qf, order_id, ctx_idx);

                    // Get AC coefficients for this block/channel
                    let ac_start = transformed.ac_offsets[block_idx * 3 + c];
                    let ac_end = if block_idx * 3 + c + 1 < transformed.ac_offsets.len() {
                        transformed.ac_offsets[block_idx * 3 + c + 1]
                    } else {
                        transformed.ac_coeffs.len()
                    };
                    let block_ac = &transformed.ac_coeffs[ac_start..ac_end];

                    // Use the actual coefficients provided by the transform
                    let effective_ac = block_ac;

                    // Count actual non-zeros
                    let nzeros: usize = effective_ac.iter().filter(|&&x| x != 0).count();

                    // Compute predicted non-zeros from neighbors (matching decoder)
                    // Decoder uses channel index (c) for grid
                    let predicted = if by == 0 {
                        if bx == 0 {
                            32usize // First block uses fixed prediction
                        } else {
                            nz_grid[c][bx - 1] as usize
                        }
                    } else if bx == 0 {
                        nz_grid[c][bx] as usize
                    } else {
                        ((nz_grid[c][bx] + nz_grid[c][bx - 1] + 1) >> 1) as usize
                    };

                    // Use prediction for context (matching decoder)
                    let nz_ctx =
                        self.block_ctx_map.nonzero_context(predicted, block_context) as u32;

                    // Emit actual non-zeros value with predicted context
                    tokens.push(Token::new(nz_ctx, nzeros as u32));

                    // Store actual non-zeros for future predictions
                    // For larger blocks (DCT16/32), store to all covered columns
                    // Use channel index (c) for grid
                    let nz_val = nzeros as u32;
                    for dx in 0..cx {
                        if bx + dx < blocks_x {
                            nz_grid[c][bx + dx] = nz_val;
                        }
                    }

                    if nzeros > 0 {
                        let histo_offset = self
                            .block_ctx_map
                            .zero_density_context_offset(block_context);

                        // Debug: print first few coefficients for first block/channel
                        let is_first = bx == 0 && by == 0 && ctx_idx == 0;
                        if is_first {
                            eprintln!("DEBUG tokenize_ac_with_strategy: first block channel X");
                            eprintln!("  nzeros={}, covered_blocks={}, log2_blocks={}", nzeros, covered_blocks, log2_blocks);
                            eprintln!("  effective_ac.len()={}", effective_ac.len());
                            eprintln!("  First 10 effective_ac: {:?}", &effective_ac[..10.min(effective_ac.len())]);
                        }

                        let mut nzeros_left = nzeros;
                        // Decoder: is_prev_coeff_nonzero = (non_zeros <= num_blocks * 4) as u32
                        let num_8x8_blocks = covered_blocks;
                        let mut prev = if nzeros <= num_8x8_blocks * 4 { 1 } else { 0 };

                        // Process coefficients in scan order
                        for k in 0..effective_ac.len() {
                            if nzeros_left == 0 {
                                break;
                            }

                            // Use the appropriate order to get the position in the block
                            // The decoder uses permutation[k] to determine where to store the coefficient,
                            // so we need to send the coefficient that should go at that position.
                            let coeff_idx = if k + covered_blocks < order.len() {
                                order[k + covered_blocks]
                            } else {
                                k + covered_blocks
                            };

                            // Pre-transpose coefficient index for square blocks.
                            // jxl-oxide (and libjxl decoder) transposes coordinates when h >= w,
                            // which is true for all square transforms (DCT8, DCT16, DCT32).
                            // We need to pre-transpose so that after the decoder's transpose,
                            // coefficients end up in the correct positions.
                            let block_dim = cx * 8; // 8 for DCT8, 16 for DCT16, 32 for DCT32
                            let transposed_coeff_idx = if cx == cy {
                                // Square block - apply transpose: (x,y) -> (y,x)
                                // Position i = y * width + x becomes x * width + y
                                (coeff_idx % block_dim) * block_dim + (coeff_idx / block_dim)
                            } else {
                                // Non-square block - no transpose
                                coeff_idx
                            };

                            // Map from full-block position to AC array index
                            // effective_ac contains coefficients starting from position covered_blocks (after LLF)
                            // For DCT8: position 1 -> ac_index 0, position 63 -> ac_index 62
                            // For DCT16/32: we skip covered_blocks LLF positions
                            let ac_index = if transposed_coeff_idx >= covered_blocks {
                                transposed_coeff_idx - covered_blocks
                            } else {
                                // LLF position - shouldn't happen in AC processing
                                k
                            };

                            let coeff = if ac_index < effective_ac.len() {
                                effective_ac[ac_index]
                            } else {
                                0
                            };

                            // jxl-oxide uses 0-based idx for the loop, starting from
                            // the first AC coefficient. Our k=0 matches idx=0.
                            let ctx = histo_offset
                                + super::context::zero_density_context(
                                    nzeros_left,
                                    k, // 0-based index matching jxl-oxide
                                    log2_blocks,
                                    prev,
                                );

                            let u_coeff = pack_signed(coeff);
                            tokens.push(Token::new(ctx as u32, u_coeff));

                            // Debug for first block/channel
                            if is_first && k < 10 {
                                eprintln!("  k={}: coeff_idx={} -> transposed={} -> ac_index={} -> coeff={} -> u_coeff={}, ctx={}",
                                    k, coeff_idx, transposed_coeff_idx, ac_index, coeff, u_coeff, ctx);
                            }

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

    /// Build a clustered histogram set from tokens.
    ///
    /// This uses the histogram clustering infrastructure to group similar
    /// contexts together for better compression.
    pub fn build_clustered_histogram_set(
        &self,
        tokens: &[Token],
        clustering_type: crate::entropy_coding::ClusteringType,
    ) -> Result<ClusteredHistogramSet> {
        ClusteredHistogramSet::from_tokens(tokens, &self.block_ctx_map, clustering_type)
    }

    /// Tokenize AC coefficients for a specific group.
    ///
    /// Returns tokens only for blocks within the specified group.
    /// The histograms must have been built from all groups beforehand.
    ///
    /// NOTE: For proper multi-group prediction, this needs the prediction grid
    /// from previous rows. For now, each group starts with a fresh grid.
    /// TODO: Add prediction grid state parameter for cross-group prediction.
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

        // Track non-zeros per column per channel (for prediction)
        // CRITICAL: Uses iteration order (0,1,2), not channel index
        let mut nz_grid: [Vec<u32>; 3] = [
            vec![0; group_blocks_x],
            vec![0; group_blocks_x],
            vec![0; group_blocks_x],
        ];

        // Process blocks within this group's bounds
        for by in start_by..end_by {
            let local_by = by - start_by;
            for bx in start_bx..end_bx {
                let local_bx = bx - start_bx;
                let block_idx = by * blocks_x + bx;

                // Get quant field value for this block (uniform)
                let qf = self.quantizer.quant_dc;

                // Process channels matching decoder order.
                // jxl-oxide uses loop index (0, 1, 2) for context computation (ch_idx),
                // but remaps to [1, 0, 2] for actual data access (Y, X, B).
                const CHANNEL_REMAP: [usize; 3] = [1, 0, 2];
                for ctx_idx in 0..3 {
                    let c = CHANNEL_REMAP[ctx_idx]; // Data channel

                    // Get block context - uses ctx_idx (0, 1, 2), NOT remapped channel
                    let lf_idx = 0; // No LF thresholds in default mode
                    let order_id = 0; // DCT8
                    let block_context = self.block_ctx_map.block_context(lf_idx, qf, order_id, ctx_idx);

                    // Get AC coefficients for this block/channel (63 coeffs per channel)
                    let ac_start = block_idx * 3 * 63 + c * 63;
                    let block_ac = &ac_coeffs[ac_start..ac_start + 63];

                    // Count actual non-zeros
                    let nzeros: usize = block_ac.iter().filter(|&&x| x != 0).count();

                    // Compute predicted non-zeros from neighbors (matching decoder)
                    // Use local coordinates within group for grid
                    // Decoder uses channel index (c) for grid
                    let predicted = if local_by == 0 {
                        if local_bx == 0 {
                            32usize // First block uses fixed prediction
                        } else {
                            nz_grid[c][local_bx - 1] as usize
                        }
                    } else if local_bx == 0 {
                        nz_grid[c][local_bx] as usize
                    } else {
                        ((nz_grid[c][local_bx] + nz_grid[c][local_bx - 1] + 1) >> 1) as usize
                    };

                    // Use prediction for context (matching decoder)
                    let nz_ctx =
                        self.block_ctx_map.nonzero_context(predicted, block_context) as u32;

                    // Emit actual non-zeros value with predicted context
                    tokens.push(Token::new(nz_ctx, nzeros as u32));

                    // Store actual non-zeros for future predictions
                    nz_grid[c][local_bx] = nzeros as u32;

                    if nzeros > 0 {
                        // Get zero-density context offset
                        let histo_offset = self
                            .block_ctx_map
                            .zero_density_context_offset(block_context);

                        // Process coefficients in natural order
                        let mut nzeros_left = nzeros;
                        let mut prev = if nzeros <= 4 { 1 } else { 0 };

                        for k in 0..63 {
                            if nzeros_left == 0 {
                                break;
                            }

                            // Pre-transpose coefficient index for DCT8 because jxl-oxide
                            // transposes coordinates when h >= w (which is true for 8x8).
                            let orig_idx = ZIGZAG_ORDER_8X8[k + 1];
                            let transposed_idx = (orig_idx % 8) * 8 + (orig_idx / 8);
                            let coeff = block_ac[transposed_idx - 1]; // -1 because AC starts at 0
                            let ctx = histo_offset
                                + super::context::zero_density_context(
                                    nzeros_left,
                                    k, // k is 0-based, matching decoder's idx
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

    /// Tokenize AC coefficients for a specific group, using strategy-aware offsets.
    ///
    /// This version correctly handles variable-size DCT blocks (DCT8, DCT16, DCT32)
    /// by using the ac_offsets from TransformedDataWithStrategy.
    ///
    /// Returns tokens only for blocks within the specified group.
    pub fn tokenize_ac_with_strategy_for_group(
        &self,
        transformed: &TransformedDataWithStrategy,
        group_idx: usize,
    ) -> Vec<Token> {
        let blocks_x = self.num_blocks_x();
        let blocks_y = self.num_blocks_y();
        let (start_bx, start_by, end_bx, end_by) = self.group_block_range(group_idx);

        let group_blocks_x = end_bx - start_bx;
        let num_group_blocks = group_blocks_x * (end_by - start_by);

        let mut tokens = Vec::with_capacity(num_group_blocks * 64);

        // Track which blocks have been processed (for DCT16/32)
        let mut processed = vec![vec![false; blocks_x]; blocks_y];

        // Track non-zeros per column per channel (for prediction, matching decoder)
        let mut nz_grid: [Vec<u32>; 3] = [
            vec![0; group_blocks_x],
            vec![0; group_blocks_x],
            vec![0; group_blocks_x],
        ];

        // Generate natural orders for different block sizes
        let order_8 = ZIGZAG_ORDER_8X8.to_vec();
        let order_16 = generate_natural_order(2, 2);
        let order_32 = generate_natural_order(4, 4);

        // Debug: count transforms processed for first group
        let debug_group = group_idx == 0;
        let mut debug_block_count = 0;

        for by in start_by..end_by {
            let local_by = by - start_by;
            for bx in start_bx..end_bx {
                let local_bx = bx - start_bx;
                let block_idx = by * blocks_x + bx;

                if processed[by][bx] {
                    continue;
                }

                let strategy = transformed.strategies.get(bx, by);
                let (cx, cy) = strategy.covered_blocks();

                if debug_group && debug_block_count < 3 {
                    eprintln!(
                        "DEBUG group0 block({},{}): strategy={:?}, coverage=({},{}), block_idx={}",
                        bx, by, strategy, cx, cy, block_idx
                    );
                    // Check ac_offsets for this block
                    for c in 0..3 {
                        let ac_start = transformed.ac_offsets[block_idx * 3 + c];
                        let ac_end = if block_idx * 3 + c + 1 < transformed.ac_offsets.len() {
                            transformed.ac_offsets[block_idx * 3 + c + 1]
                        } else {
                            transformed.ac_coeffs.len()
                        };
                        eprintln!(
                            "  ch{}: ac_offsets[{}..{}] len={}",
                            c, ac_start, ac_end, ac_end - ac_start
                        );
                    }
                    debug_block_count += 1;
                }

                // Clamp to image bounds
                let actual_cx = cx.min(blocks_x - bx);
                let actual_cy = cy.min(blocks_y - by);

                // Mark all covered blocks as processed
                for dy in 0..actual_cy {
                    for dx in 0..actual_cx {
                        if by + dy < blocks_y && bx + dx < blocks_x {
                            processed[by + dy][bx + dx] = true;
                        }
                    }
                }

                let order_id = strategy.order_id();
                let log2_blocks = strategy.log2_covered_blocks() as usize;
                let order: &[usize] = match (cx, cy) {
                    (4, 4) => &order_32,
                    (2, 2) => &order_16,
                    _ => &order_8,
                };

                let covered_blocks = cx * cy;
                let qf = self.quantizer.quant_dc;

                // Process channels matching decoder order
                const CHANNEL_REMAP: [usize; 3] = [1, 0, 2];
                for ctx_idx in 0..3 {
                    let c = CHANNEL_REMAP[ctx_idx];
                    let block_context = self.block_ctx_map.block_context(0, qf, order_id, ctx_idx);

                    // Get AC coefficients using offsets
                    let ac_start = transformed.ac_offsets[block_idx * 3 + c];
                    let ac_end = if block_idx * 3 + c + 1 < transformed.ac_offsets.len() {
                        transformed.ac_offsets[block_idx * 3 + c + 1]
                    } else {
                        transformed.ac_coeffs.len()
                    };
                    let block_ac = &transformed.ac_coeffs[ac_start..ac_end];

                    let nzeros: usize = block_ac.iter().filter(|&&x| x != 0).count();

                    // Compute predicted non-zeros from neighbors (local coords for group)
                    let predicted = if local_by == 0 {
                        if local_bx == 0 {
                            32usize
                        } else {
                            nz_grid[c][local_bx - 1] as usize
                        }
                    } else if local_bx == 0 {
                        nz_grid[c][local_bx] as usize
                    } else {
                        ((nz_grid[c][local_bx] + nz_grid[c][local_bx - 1] + 1) >> 1) as usize
                    };

                    let nz_ctx =
                        self.block_ctx_map.nonzero_context(predicted, block_context) as u32;
                    tokens.push(Token::new(nz_ctx, nzeros as u32));

                    // Store actual non-zeros for future predictions
                    let nz_val = nzeros as u32;
                    for dx in 0..cx {
                        if local_bx + dx < group_blocks_x {
                            nz_grid[c][local_bx + dx] = nz_val;
                        }
                    }

                    if nzeros > 0 {
                        let histo_offset = self
                            .block_ctx_map
                            .zero_density_context_offset(block_context);

                        let mut nzeros_left = nzeros;
                        let num_8x8_blocks = covered_blocks;
                        let mut prev = if nzeros <= num_8x8_blocks * 4 { 1 } else { 0 };

                        // Debug: trace first few coefficients for first DCT16 block
                        let debug_this_block = debug_group && debug_block_count <= 3 && cx > 1 && ctx_idx == 0;

                        for k in 0..block_ac.len() {
                            if nzeros_left == 0 {
                                break;
                            }

                            let coeff_idx = if k + covered_blocks < order.len() {
                                order[k + covered_blocks]
                            } else {
                                k + covered_blocks
                            };

                            // Pre-transpose for square blocks
                            let block_dim = cx * 8;
                            let transposed_coeff_idx = if cx == cy {
                                (coeff_idx % block_dim) * block_dim + (coeff_idx / block_dim)
                            } else {
                                coeff_idx
                            };

                            // Map from full-block position to AC array index
                            // (same formula as tokenize_ac_with_strategy)
                            let ac_index = if transposed_coeff_idx >= covered_blocks {
                                transposed_coeff_idx - covered_blocks
                            } else {
                                // LLF position - fallback to k
                                k
                            };

                            let coeff = if ac_index < block_ac.len() {
                                block_ac[ac_index]
                            } else {
                                0
                            };

                            if debug_this_block && k < 5 {
                                eprintln!(
                                    "  DCT16 k={}: coeff_idx={} transposed={} ac_index={} coeff={}",
                                    k, coeff_idx, transposed_coeff_idx, ac_index, coeff
                                );
                            }

                            let ctx = histo_offset
                                + super::context::zero_density_context(
                                    nzeros_left,
                                    k,
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
            crate::trace::debug_eprintln!(
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

    /// Write the HF Global section using clustered histograms.
    ///
    /// This version uses histogram clustering for better compression.
    pub fn write_hf_global_clustered(
        &self,
        tokens: &[Token],
        histogram_set: &ClusteredHistogramSet,
        writer: &mut BitWriter,
    ) -> Result<()> {
        // Write dequant matrices (default = 1 bit)
        self.dequant_matrices.write(writer);

        // Write num_hf_presets for multi-group frames
        let num_groups = self.num_groups();
        let num_hf_presets_bits = num_groups.next_power_of_two().trailing_zeros() as usize;
        if num_hf_presets_bits > 0 {
            // Write num_hf_presets - 1 = 0 (we use 1 preset)
            writer.write(num_hf_presets_bits, 0)?;
            crate::trace::debug_eprintln!(
                "HF_GLOBAL_CLUSTERED [bit {}]: num_hf_presets = 1 (wrote 0 in {} bits)",
                writer.bits_written(),
                num_hf_presets_bits
            );
        }

        // Coefficient order encoding
        // used_orders = 0 means all default orders (no custom permutations)
        writer.write(2, 2)?; // Selector 2 = value 0 (used_orders = 0)

        // Write clustered histograms for each pass (we have 1 pass)
        self.write_histograms_clustered(histogram_set, tokens, writer)?;

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
        crate::trace::debug_eprintln!("HF_HIST [bit {}]: Starting histogram", start_bit);

        // LZ77 enabled = false
        writer.write(1, 0)?;
        crate::trace::debug_eprintln!("HF_HIST [bit {}]: lz77.enabled = 0", writer.bits_written());

        // Context map
        // For simplicity, map all contexts to cluster 0
        let num_contexts = distributions.len();
        crate::trace::debug_eprintln!("HF_HIST: num_contexts = {}", num_contexts);

        if num_contexts == 1 {
            // When num_contexts = 1, read_clusters returns immediately without reading bits
            // (implicit single histogram, no context map written)
        } else {
            // Use simple context map encoding: is_simple=1, bits_per_entry=0
            // This maps all contexts to cluster 0
            writer.write(1, 1)?; // is_simple = true
            writer.write(2, 0)?; // bits_per_entry = 0 (all contexts map to 0)
            crate::trace::debug_eprintln!(
                "HF_HIST [bit {}]: context_map (is_simple=1, bits=0)",
                writer.bits_written()
            );
        }

        // Use prefix codes (Huffman) for simplicity
        writer.write(1, 1)?; // use_prefix_code = true
        crate::trace::debug_eprintln!(
            "HF_HIST [bit {}]: use_prefix_code = 1",
            writer.bits_written()
        );

        // Compute the maximum raw symbol value
        let max_symbol = tokens.iter().map(|t| t.value as usize).max().unwrap_or(0);
        crate::trace::debug_eprintln!("HF_HIST: max_symbol = {}", max_symbol);

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
        crate::trace::debug_eprintln!(
            "HF_HIST: max_token = {}, alphabet_size = {}",
            max_token,
            alphabet_size
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
        crate::trace::debug_eprintln!(
            "HF_HIST [bit {}]: IntegerConfig (split=4, msb=2, lsb=0)",
            writer.bits_written()
        );

        // Write alphabet size
        write_alphabet_size(writer, alphabet_size)?;
        crate::trace::debug_eprintln!(
            "HF_HIST [bit {}]: After alphabet_size = {}",
            writer.bits_written(),
            alphabet_size
        );

        // Write prefix codes for the bounded token alphabet
        write_prefix_code(writer, alphabet_size)?;
        crate::trace::debug_eprintln!(
            "HF_HIST [bit {}]: After prefix_code, total = {} bits",
            writer.bits_written(),
            writer.bits_written() - start_bit
        );

        Ok(())
    }

    /// Write histogram set for AC coefficients using clustered histograms.
    ///
    /// This version uses the histogram clustering infrastructure for better
    /// compression. Similar contexts are merged into clusters, reducing
    /// the number of histograms that need to be encoded.
    fn write_histograms_clustered(
        &self,
        histogram_set: &ClusteredHistogramSet,
        _tokens: &[Token],
        writer: &mut BitWriter,
    ) -> Result<()> {
        let start_bit = writer.bits_written();
        crate::trace::debug_eprintln!(
            "HF_HIST_CLUSTERED [bit {}]: Starting, {} clusters",
            start_bit,
            histogram_set.num_clusters()
        );

        // LZ77 enabled = false
        writer.write(1, 0)?;
        crate::trace::debug_eprintln!(
            "HF_HIST_CLUSTERED [bit {}]: lz77.enabled = 0",
            writer.bits_written()
        );

        // Context map encoding
        let num_clusters = histogram_set.num_clusters();
        let num_contexts = histogram_set.num_contexts;
        crate::trace::debug_eprintln!(
            "HF_HIST_CLUSTERED: num_clusters = {}, num_contexts = {}",
            num_clusters,
            num_contexts
        );

        if num_contexts == 1 {
            // When num_contexts = 1, no context map is written
        } else if num_clusters == 1 {
            // Single cluster: simple context map encoding
            writer.write(1, 1)?; // is_simple = true
            writer.write(2, 0)?; // bits_per_entry = 0 (all contexts map to 0)
            crate::trace::debug_eprintln!(
                "HF_HIST_CLUSTERED [bit {}]: context_map (is_simple=1, bits=0)",
                writer.bits_written()
            );
        } else {
            // Use proper context map encoding for multiple clusters
            encode_context_map(&histogram_set.context_map, num_clusters, writer)?;
            crate::trace::debug_eprintln!(
                "HF_HIST_CLUSTERED [bit {}]: context_map encoded ({} clusters)",
                writer.bits_written(),
                num_clusters
            );
        }

        // Use prefix codes (Huffman) for simplicity
        writer.write(1, 1)?; // use_prefix_code = true
        crate::trace::debug_eprintln!(
            "HF_HIST_CLUSTERED [bit {}]: use_prefix_code = 1",
            writer.bits_written()
        );

        // Use the global alphabet size from histogram_set (computed at build time from all tokens)
        // This MUST match what write_pass_group_clustered uses to ensure consistent encoding
        let alphabet_size = histogram_set.global_alphabet_size;
        crate::trace::debug_eprintln!(
            "HF_HIST_CLUSTERED: using global_alphabet_size = {}",
            alphabet_size
        );

        // Write IntegerConfig for EACH histogram (num_clusters histograms)
        // Format: For log_alpha_size=15 (HUFFMAN_MAX_BITS):
        //   - split_exponent: ceil_log2(16) = 4 bits
        //   - msb_in_token: ceil_log2(split_exponent + 1) = ceil_log2(5) = 3 bits
        //   - lsb_in_token: ceil_log2(split_exponent - msb_in_token + 1) = ceil_log2(3) = 2 bits
        for cluster_idx in 0..num_clusters {
            writer.write(4, 4)?; // split_exponent = 4
            writer.write(3, 2)?; // msb_in_token = 2
            writer.write(2, 0)?; // lsb_in_token = 0
            crate::trace::debug_eprintln!(
                "HF_HIST_CLUSTERED [bit {}]: IntegerConfig {} (split=4, msb=2, lsb=0)",
                writer.bits_written(),
                cluster_idx
            );
        }

        // Write alphabet sizes for EACH histogram (all alphabet sizes first)
        // HuffmanCodes::decode reads all alphabet sizes, then all prefix codes
        for cluster_idx in 0..num_clusters {
            write_alphabet_size(writer, alphabet_size)?;
            crate::trace::debug_eprintln!(
                "HF_HIST_CLUSTERED [bit {}]: alphabet_size[{}] = {}",
                writer.bits_written(),
                cluster_idx,
                alphabet_size
            );
        }

        // Write prefix codes for EACH cluster (after all alphabet sizes)
        for cluster_idx in 0..num_clusters {
            write_prefix_code(writer, alphabet_size)?;
            crate::trace::debug_eprintln!(
                "HF_HIST_CLUSTERED [bit {}]: After prefix_code for cluster {}",
                writer.bits_written(),
                cluster_idx
            );
        }

        crate::trace::debug_eprintln!(
            "HF_HIST_CLUSTERED [bit {}]: Complete, total = {} bits",
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

        crate::trace::debug_eprintln!(
            "LF_GROUP [bit {}]: Starting, {}x{} blocks",
            writer.bits_written(),
            blocks_x,
            blocks_y
        );

        // 1. extra_precision (2 bits) - 0 for standard precision
        writer.write(2, 0)?;
        crate::trace::debug_eprintln!(
            "LF_GROUP [bit {}]: extra_precision = 0",
            writer.bits_written()
        );

        // 2. VarDCTLF modular stream (DC coefficients)
        self.write_dc_coeffs(dc_coeffs, writer)?;
        crate::trace::debug_eprintln!(
            "LF_GROUP [bit {}]: After DC coefficients",
            writer.bits_written()
        );

        // 3. ModularLF stream - for extra channels
        // We don't have extra channels, so this is empty (nothing to write)

        // 4. HF metadata
        self.write_hf_metadata(writer)?;
        crate::trace::debug_eprintln!(
            "LF_GROUP [bit {}]: After HF metadata, LF Group done",
            writer.bits_written()
        );

        Ok(())
    }

    /// Write HF metadata for the LF Group.
    ///
    /// HF metadata contains:
    /// - count (ceil_log2(upper_bound) bits) - number of distinct transform blocks
    /// - 4 modular channels: ytox_map, ytob_map, transform_image, epf_map
    ///
    /// For DCT16/32, multiple 8x8 blocks are covered by a single transform.
    /// Only the top-left block of each transform is written to transform_image.
    fn write_hf_metadata(&self, writer: &mut BitWriter) -> Result<()> {
        let blocks_x = self.num_blocks_x();
        let blocks_y = self.num_blocks_y();
        let num_blocks = blocks_x * blocks_y;

        // Color tile size (8x8 blocks per tile)
        let tiles_x = blocks_x.div_ceil(8);
        let tiles_y = blocks_y.div_ceil(8);
        let num_tiles = tiles_x * tiles_y;

        // Build transform entries by walking grid and tracking processed blocks.
        // For DCT8, every block is distinct. For DCT16/32, multiple blocks share one transform.
        let mut processed = vec![false; num_blocks];
        let mut transform_entries: Vec<(i32, i32)> = Vec::new();

        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let block_idx = by * blocks_x + bx;
                if processed[block_idx] {
                    continue;
                }

                let strategy = self.ac_strategy_map.get(bx, by);
                let (cx, cy) = strategy.covered_blocks();

                // For boundary safety, clamp coverage to image bounds
                let actual_cx = cx.min(blocks_x - bx);
                let actual_cy = cy.min(blocks_y - by);

                // Mark all covered blocks as processed
                for dy in 0..actual_cy {
                    for dx in 0..actual_cx {
                        processed[(by + dy) * blocks_x + (bx + dx)] = true;
                    }
                }

                // Store transform entry (only top-left block)
                let quant = self.quant_field.get(bx, by) as i32;
                transform_entries.push((strategy as i32, (quant - 1).max(0)));
            }
        }

        let count = transform_entries.len();

        // Count is encoded using ceil_log2(upper_bound) bits
        // Upper bound is still num_blocks (worst case: all DCT8)
        let upper_bound = num_blocks;
        let count_bits = if upper_bound <= 1 {
            0
        } else {
            (usize::BITS - (upper_bound - 1).leading_zeros()) as usize
        };
        if count_bits > 0 {
            writer.write(count_bits, (count - 1) as u64)?;
        }
        crate::trace::debug_eprintln!(
            "HF_META [bit {}]: count = {} distinct transforms ({} bits, num_blocks={})",
            writer.bits_written(),
            count,
            count_bits,
            num_blocks
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
        // Row 0: transform_type (enum value from AcStrategy)
        // Row 1: raw_quant - 1
        let mut transform_data = vec![0i32; count * 2];
        for (i, (strategy, quant)) in transform_entries.iter().enumerate() {
            transform_data[i] = *strategy;
            transform_data[count + i] = *quant;
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

        crate::trace::debug_eprintln!(
            "DC_COEFFS: blocks={}x{}, y_dc={:?}, x_dc={:?}, b_dc={:?}",
            blocks_x,
            blocks_y,
            &y_dc[..y_dc.len().min(10)],
            &x_dc[..x_dc.len().min(10)],
            &b_dc[..b_dc.len().min(10)]
        );

        // Create ModularImage from DC channels
        // Order: Y (c=0), X (c=1), B (c=2) - standard XYB channel order
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

        crate::trace::debug_eprintln!(
            "PASS_GROUP: alphabet_size={}, first 10 codes: {:?}, first 10 lengths: {:?}",
            alphabet_size,
            &codes[..alphabet_size.min(10)],
            &code_lengths[..alphabet_size.min(10)]
        );

        // Encode each value using HybridUint, then write with correct Huffman code
        for (token_count, token) in tokens.iter().enumerate() {
            let (encoded_token, extra_bits, num_extra_bits) = hybrid_config.encode(token.value);
            let sym = encoded_token as usize;
            let code = codes[sym];
            let len = code_lengths[sym] as usize;
            if token_count < 5 {
                crate::trace::debug_eprintln!(
                    "PASS_GROUP: token {}: value={}, encoded_token={}, code={:#b} ({} bits), extra_bits={:#x} ({} bits)",
                    token_count,
                    token.value,
                    encoded_token,
                    code,
                    len,
                    extra_bits,
                    num_extra_bits
                );
            }
            writer.write(len, code as u64)?;
            if num_extra_bits > 0 {
                writer.write(num_extra_bits as usize, extra_bits as u64)?;
            }
        }

        Ok(())
    }

    /// Write the Pass Group section using clustered histograms.
    ///
    /// This version uses the context map to select the appropriate
    /// Huffman code for each token based on its context's cluster.
    ///
    /// Note: histograms_id is only written when num_hf_presets > 1.
    /// We always use 1 preset, so histograms_id is never written.
    pub fn write_pass_group_clustered(
        &self,
        tokens: &[Token],
        histogram_set: &ClusteredHistogramSet,
        writer: &mut BitWriter,
    ) -> Result<()> {
        // Note: histograms_id would be written here ONLY if num_hf_presets > 1.
        // We always use 1 preset, so skip it.

        if tokens.is_empty() {
            return Ok(());
        }

        // Use the same HybridUint config as write_histograms_clustered
        let hybrid_config = crate::entropy_coding::hybrid_uint::HybridUintConfig::new(4, 2, 0);

        // CRITICAL: Use the global alphabet size from histogram_set, NOT computed from local tokens!
        // This must match what was written in write_histograms_clustered.
        // Computing it locally from group tokens causes mismatches when different groups
        // have different max symbol values, leading to "non_zeros too large" decode errors.
        let alphabet_size = histogram_set.global_alphabet_size;

        if alphabet_size <= 1 {
            // Single symbol - nothing to write (decoder knows the only possibility)
            return Ok(());
        }

        let num_clusters = histogram_set.num_clusters();

        // Build canonical Huffman codes for EACH cluster
        // For now, we use the same flat distribution for all clusters
        // (matching what we wrote in write_histograms_clustered)
        let codes_per_cluster: Vec<(Vec<u32>, Vec<u8>)> = (0..num_clusters)
            .map(|_| build_canonical_huffman_codes(alphabet_size))
            .collect();

        // Encode each token using the appropriate cluster's Huffman codes
        for token in tokens {
            let context = token.context as usize;
            let cluster_idx = if context < histogram_set.context_map.len() {
                histogram_set.context_map[context] as usize
            } else {
                0
            };

            let (ref codes, ref code_lengths) = codes_per_cluster[cluster_idx];

            let (encoded_token, extra_bits, num_extra_bits) = hybrid_config.encode(token.value);
            let sym = encoded_token as usize;
            let code = codes[sym];
            let len = code_lengths[sym] as usize;

            writer.write(len, code as u64)?;
            if num_extra_bits > 0 {
                writer.write(num_extra_bits as usize, extra_bits as u64)?;
            }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vardct::AcStrategy;
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

    #[test]
    fn test_write_hf_metadata_dct8() {
        // Test that write_hf_metadata correctly counts all blocks for DCT8-only
        let enc = VarDctEncoder::new(16, 16, VarDctOptions::default());
        assert_eq!(enc.num_blocks_x(), 2);
        assert_eq!(enc.num_blocks_y(), 2);

        let mut writer = BitWriter::new();
        // Need to write group header and HF metadata
        enc.write_hf_metadata(&mut writer).unwrap();
        assert!(writer.bits_written() > 0);
    }

    #[test]
    fn test_write_hf_metadata_dct16() {
        // Test with a 16x16 image (2x2 blocks) using DCT16 for all
        use crate::heuristics::AcStrategyMap;

        let options = VarDctOptions {
            ac_strategy_heuristics: HeuristicLevel::VarianceBased,
            ..Default::default()
        };
        let mut enc = VarDctEncoder::new(16, 16, options);

        // Manually set all blocks to DCT16 (covers 2x2 blocks)
        let mut strategy_map = AcStrategyMap::new_dct8(2, 2);
        for by in 0..2 {
            for bx in 0..2 {
                strategy_map.set(bx, by, AcStrategy::Dct16x16);
            }
        }
        enc.ac_strategy_map = strategy_map;

        let mut writer = BitWriter::new();
        enc.write_hf_metadata(&mut writer).unwrap();
        // With DCT16 covering all 4 blocks as one transform, count should be 1
        assert!(writer.bits_written() > 0);
    }

    #[test]
    fn test_write_hf_metadata_dct32() {
        // Test with a 32x32 image (4x4 blocks) using DCT32 for all
        use crate::heuristics::AcStrategyMap;

        let options = VarDctOptions {
            ac_strategy_heuristics: HeuristicLevel::VarianceBased,
            ..Default::default()
        };
        let mut enc = VarDctEncoder::new(32, 32, options);

        // Manually set all blocks to DCT32 (covers 4x4 blocks)
        let mut strategy_map = AcStrategyMap::new_dct8(4, 4);
        for by in 0..4 {
            for bx in 0..4 {
                strategy_map.set(bx, by, AcStrategy::Dct32x32);
            }
        }
        enc.ac_strategy_map = strategy_map;

        let mut writer = BitWriter::new();
        enc.write_hf_metadata(&mut writer).unwrap();
        // With DCT32 covering all 16 blocks as one transform, count should be 1
        assert!(writer.bits_written() > 0);
    }

    #[test]
    fn test_write_hf_metadata_mixed_strategies() {
        // Test with mixed DCT8 and DCT16
        use crate::heuristics::AcStrategyMap;

        let options = VarDctOptions {
            ac_strategy_heuristics: HeuristicLevel::VarianceBased,
            ..Default::default()
        };
        let mut enc = VarDctEncoder::new(32, 16, options);
        // 4x2 blocks = 8 blocks total

        // Set left half to DCT16 (covers 2x2), right half to DCT8
        let mut strategy_map = AcStrategyMap::new_dct8(4, 2);
        // (0,0), (1,0), (0,1), (1,1) -> DCT16 (1 transform)
        strategy_map.set(0, 0, AcStrategy::Dct16x16);
        strategy_map.set(1, 0, AcStrategy::Dct16x16);
        strategy_map.set(0, 1, AcStrategy::Dct16x16);
        strategy_map.set(1, 1, AcStrategy::Dct16x16);
        // (2,0), (3,0), (2,1), (3,1) -> DCT8 (4 transforms)
        // Already set to DCT8 by default
        enc.ac_strategy_map = strategy_map;

        let mut writer = BitWriter::new();
        enc.write_hf_metadata(&mut writer).unwrap();
        // Should have 1 DCT16 transform + 4 DCT8 transforms = 5 total
        assert!(writer.bits_written() > 0);
    }
}
