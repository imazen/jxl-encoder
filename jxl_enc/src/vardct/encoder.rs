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
        eprintln!(
            "VARDCT_FRMH [bit {}]: Starting VarDCT frame header",
            writer.bits_written()
        );

        // Write explicit frame header (matching libjxl reference output)
        // all_default = false (need to specify VarDCT-specific fields)
        writer.write(1, 0)?;
        eprintln!(
            "VARDCT_FRMH [bit {}]: all_default = 0",
            writer.bits_written()
        );

        // frame_type = RegularFrame (0)
        writer.write(2, 0)?;

        // encoding = VarDCT (0)
        writer.write(1, 0)?;

        // flags = 0 (U64 encoding: selector 0 means value 0)
        writer.write(2, 0)?;

        // upsampling = 1 (selector 0 in u2S(1,2,4,8))
        writer.write(2, 0)?;

        // x_qm_scale = 3 (only for VarDCT with xyb_encoded)
        writer.write(3, 3)?;

        // b_qm_scale = 2 (only for VarDCT with xyb_encoded)
        writer.write(3, 2)?;

        // passes.num_passes = 1 (selector 0 in u2S(1,2,3,Bits(3)+4))
        writer.write(2, 0)?;

        // have_crop = false (only if frame_type != LFFrame)
        writer.write(1, 0)?;

        // blending_info.mode = Replace (selector 0 in u2S(0,1,2,Bits(2)+3))
        writer.write(2, 0)?;

        // is_last = true (only for RegularFrame or SkipProgressive)
        writer.write(1, 1)?;

        // name_length = 0 using u2S(0, Bits(4), Bits(5)+16, Bits(10)+48)
        writer.write(2, 0)?; // selector 0 = value 0

        // restoration_filter - explicit settings matching reference
        // all_default = false (we need to specify epf_iters=1)
        writer.write(1, 0)?;

        // gab = true
        writer.write(1, 1)?;
        // gab_custom = false
        writer.write(1, 0)?;

        // epf_iters = 1 (reference uses 1, not default 2)
        writer.write(2, 1)?;
        // epf_sharp_custom = false
        writer.write(1, 0)?;
        // epf_weight_custom = false
        writer.write(1, 0)?;
        // epf_sigma_custom = false
        writer.write(1, 0)?;

        // extensions = 0 (no extensions)
        // U64 encoding: selector 0 (2 bits) means value 0
        writer.write(2, 0)?;
        eprintln!(
            "VARDCT_FRMH [bit {}]: frame header done",
            writer.bits_written()
        );

        Ok(())
    }

    /// Write the LF Global section.
    ///
    /// Contains: LfQuantFactors, QuantizerParams, BlockCtxMap, ColorCorrelation, Tree, ModularGlobal.
    pub fn write_lf_global(&self, writer: &mut BitWriter) -> Result<()> {
        eprintln!(
            "LF_GLOBAL [bit {}]: Starting LF Global section",
            writer.bits_written()
        );

        // Write LF quant factors (use defaults)
        // Format: bit 1 = use default LF_QUANT values
        writer.write(1, 1)?; // all_default = true
        eprintln!(
            "LF_GLOBAL [bit {}]: lf_quant_factors.all_default = 1",
            writer.bits_written()
        );

        // Write quantizer params (global_scale, quant_dc)
        self.quantizer.write(writer);
        eprintln!(
            "LF_GLOBAL [bit {}]: quantizer_params (gs={}, qdc={})",
            writer.bits_written(),
            self.quantizer.global_scale,
            self.quantizer.quant_dc
        );

        // Write block context map (default = 1 bit)
        self.block_ctx_map.write(writer)?;
        eprintln!(
            "LF_GLOBAL [bit {}]: block_context_map (default={})",
            writer.bits_written(),
            self.block_ctx_map.use_default
        );

        // Write color correlation (LF)
        self.write_color_correlation(writer)?;
        eprintln!(
            "LF_GLOBAL [bit {}]: color_correlation",
            writer.bits_written()
        );

        // Write global tree presence (0 = no global tree for VarDCT)
        writer.write(1, 0)?;
        eprintln!(
            "LF_GLOBAL [bit {}]: has_global_tree = 0, LF Global done",
            writer.bits_written()
        );

        // ModularGlobal is empty for VarDCT without extra channels
        // (no channels to encode, so nothing to write)

        Ok(())
    }

    /// Write color correlation parameters.
    fn write_color_correlation(&self, writer: &mut BitWriter) -> Result<()> {
        let cmap = &self.color_correlation;

        // If using default correlation (no CfL), just write all_default=true
        if !self.options.cfl_enabled || cmap.is_default() {
            writer.write(1, 1)?; // all_default = true
            return Ok(());
        }

        // all_default = false
        writer.write(1, 0)?;

        // Write color_factor using U32Enc kColorFactorDist
        // U32Enc: Val(84), Val(256), BitsOffset(8,2), BitsOffset(16,258)
        let color_factor = cmap.color_factor;
        if color_factor == 84 {
            writer.write(2, 0)?; // selector 0 = 84
        } else if color_factor == 256 {
            writer.write(2, 1)?; // selector 1 = 256
        } else if (2..258).contains(&color_factor) {
            writer.write(2, 2)?; // selector 2 = Bits(8) + 2
            writer.write(8, (color_factor - 2) as u64)?;
        } else {
            writer.write(2, 3)?; // selector 3 = Bits(16) + 258
            writer.write(16, (color_factor.saturating_sub(258)) as u64)?;
        }

        // Write base_correlation_x (F16)
        // For simplicity, use 0.0 (encoded as 0x0000 in half-precision)
        writer.write(16, 0)?; // base_correlation_x = 0.0

        // Write base_correlation_b (F16)
        // Default is 1.0 = 0x3C00 in half-precision
        writer.write(16, 0x3C00)?; // base_correlation_b = 1.0

        // Write ytox_dc (S32 signed integer)
        write_signed_varint(writer, cmap.ytox_dc)?;

        // Write ytob_dc (S32 signed integer)
        write_signed_varint(writer, cmap.ytob_dc)?;

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
        distributions: &[AnsDistribution],
        writer: &mut BitWriter,
    ) -> Result<()> {
        // Write dequant matrices (default = 1 bit)
        self.dequant_matrices.write(writer);

        // num_histograms = 1 (only for multi-group, skip for single)
        // For single group, this is not written

        // Coefficient order encoding
        // used_orders = 0 means all default orders (no custom permutations)
        writer.write(2, 2)?; // Selector 2 = value 0 (used_orders = 0)

        // Write histograms for each pass (we have 1 pass)
        self.write_histograms(distributions, writer)?;

        Ok(())
    }

    /// Write histogram set for AC coefficients.
    fn write_histograms(
        &self,
        distributions: &[AnsDistribution],
        writer: &mut BitWriter,
    ) -> Result<()> {
        // LZ77 enabled = false
        writer.write(1, 0)?;

        // Context map
        // For simplicity, map all contexts to cluster 0
        let num_contexts = distributions.len();

        if num_contexts == 1 {
            // When num_contexts = 1, read_clusters returns immediately without reading bits
            // (implicit single histogram, no context map written)
        } else {
            // Use simple context map encoding: is_simple=1, bits_per_entry=0
            // This maps all contexts to cluster 0
            writer.write(1, 1)?; // is_simple = true
            writer.write(2, 0)?; // bits_per_entry = 0 (all contexts map to 0)
        }

        // Use prefix code (Huffman) instead of ANS for simplicity
        writer.write(1, 1)?; // use_prefix_code = true

        // IntegerConfig: When use_prefix_code=1, decoder uses log_alphabet_size=15
        // for parsing IntegerConfig. Use split_exponent=15 for raw symbol encoding.
        // split_exponent_bits = add_log2_ceil(15) = 4 bits
        // When split_exponent == log_alphabet_size, msb/lsb are implicit 0
        writer.write(4, 15)?; // split_exponent = 15 (raw symbols)

        // Alphabet size
        let alphabet_size = distributions
            .first()
            .map(|d| d.alphabet_size())
            .unwrap_or(1);
        write_alphabet_size(writer, alphabet_size)?;

        // Write Huffman codes for the single histogram
        // For now, use a simple flat code
        if alphabet_size <= 1 {
            // Single symbol - trivial encoding
            writer.write(1, 1)?; // is_simple = true
            writer.write(2, 0)?; // nsym_minus_1 = 0
        // No symbol bits needed for single symbol
        } else if alphabet_size <= 4 {
            // Simple code for 2-4 symbols
            writer.write(1, 1)?; // is_simple = true
            writer.write(2, (alphabet_size - 1) as u64)?; // nsym_minus_1

            if alphabet_size == 2 {
                writer.write(1, 0)?; // symbol 0
                writer.write(1, 1)?; // symbol 1
            } else {
                // Write symbols
                let nbits = 8 - (alphabet_size as u32 - 1).leading_zeros();
                for i in 0..alphabet_size {
                    writer.write(nbits as usize, i as u64)?;
                }
            }
        } else {
            // Full Huffman tree - use flat distribution for now
            writer.write(1, 0)?; // is_simple = false
            writer.write(1, 1)?; // is_flat = true
            // log_counts encoding
            let log_alpha = 8 - (alphabet_size as u32 - 1).leading_zeros();
            writer.write(log_alpha as usize, (alphabet_size - 1) as u64)?;
        }

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
        // use_global_tree = false (we write our own tree)
        writer.write(1, 0)?;

        // wp_header.all_default = true (use default weighted params)
        writer.write(1, 1)?;

        // transforms count = 0 (no transforms)
        // u2S(0, 1, Bits(4)+2, Bits(8)+18): selector 0 = value 0
        writer.write(2, 0)?;

        eprintln!(
            "GROUP_HDR [bit {}]: use_global_tree=0, wp_header.all_default=1, transforms=0",
            writer.bits_written()
        );

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
    pub fn write_pass_group(
        &self,
        tokens: &[Token],
        distributions: &[AnsDistribution],
        writer: &mut BitWriter,
    ) -> Result<()> {
        // For multi-group: histograms_id would be written here
        // For single group, it's implicit (id = 0)

        // Encode tokens using ANS
        // First, we need to emit them in the correct order
        // For Huffman/prefix codes with a single cluster, we can emit directly

        if tokens.is_empty() {
            eprintln!("PASS_GROUP: tokens is empty, returning");
            return Ok(());
        }
        eprintln!("PASS_GROUP: {} tokens to write", tokens.len());

        // Get the single distribution (we use cluster 0 for all contexts)
        let dist = distributions.first().ok_or_else(|| {
            crate::error::Error::InvalidHistogram("no distributions for pass group".to_string())
        })?;

        // For Huffman encoding, we need to emit symbols directly
        // Use a simple encoding: each token value is written with fixed bits
        let alphabet_size = dist.alphabet_size();
        eprintln!("PASS_GROUP: alphabet_size = {}", alphabet_size);
        if alphabet_size <= 1 {
            // Single symbol - no bits needed per token
            eprintln!("PASS_GROUP: single symbol, returning");
            return Ok(());
        }

        let nbits = 8 - (alphabet_size as u32 - 1).leading_zeros();

        // Emit each token
        for token in tokens {
            let val = (token.value as usize).min(alphabet_size - 1);
            writer.write(nbits as usize, val as u64)?;
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

/// Write alphabet size for Huffman codes.
fn write_alphabet_size(writer: &mut BitWriter, size: usize) -> Result<()> {
    // Encode alphabet size using varint16
    let n = size as u16;
    if n <= 1 {
        writer.write(2, 0)?; // selector 0 = 1
    } else if n <= 2 {
        writer.write(2, 1)?; // selector 1 = value + 1
        writer.write(4, (n - 1) as u64)?;
    } else if n <= 18 {
        writer.write(2, 2)?; // selector 2 = value + 1
        writer.write(8, (n - 1) as u64)?;
    } else {
        writer.write(2, 3)?; // selector 3 = value + 1
        writer.write(16, (n - 1) as u64)?;
    }
    Ok(())
}

/// Write a signed 32-bit integer using JXL S32 encoding.
///
/// S32 uses a variable-length encoding:
/// - Small values (±63) use fewer bits
/// - Larger values use more bits
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
