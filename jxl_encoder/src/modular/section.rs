// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Modular section encoding for multi-group images.
//!
//! Handles GlobalModularState and section writing for large images that
//! are split into multiple groups.

use super::channel::ModularImage;
use super::encode::{
    write_gradient_tree_tokens, write_hybrid_data_histogram, write_rct_transform,
    write_tree_histogram_for_gradient,
};
use super::predictor::pack_signed;
use super::rct::RctType;
use crate::bit_writer::BitWriter;
use crate::entropy_coding::encode::{
    OwnedAnsEntropyCode, build_entropy_code_ans, write_tokens_ans,
};
use crate::entropy_coding::hybrid_uint::HybridUintConfig;
use crate::entropy_coding::token::Token as AnsToken;
use crate::error::Result;

/// Default HybridUint config for modular data: split_exponent=4, msb_in_token=2, lsb_in_token=0.
const MODULAR_HYBRID_UINT: HybridUintConfig = HybridUintConfig {
    split_exponent: 4,
    split: 16, // 1 << 4
    msb_in_token: 2,
    lsb_in_token: 0,
};

/// Gradient prediction (ClampedGradient).
#[inline]
fn predict_gradient(left: i32, top: i32, topleft: i32) -> i32 {
    let grad = left + top - topleft;
    // Clamp to [min(left, top), max(left, top)]
    let min = left.min(top);
    let max = left.max(top);
    grad.clamp(min, max)
}

pub fn collect_all_residuals(image: &ModularImage) -> (Vec<u32>, u32) {
    let mut residuals = Vec::new();
    let mut max_residual: u32 = 0;

    for channel in &image.channels {
        let width = channel.width();
        let height = channel.height();

        for y in 0..height {
            for x in 0..width {
                let pixel = channel.get(x, y);

                // Get neighbors (matching JXL decoder)
                let left = if x > 0 { channel.get(x - 1, y) } else { 0 };
                let top = if y > 0 { channel.get(x, y - 1) } else { left };
                let topleft = if x > 0 && y > 0 {
                    channel.get(x - 1, y - 1)
                } else {
                    left
                };

                // Predict using ClampedGradient (predictor 5)
                let prediction = predict_gradient(left, top, topleft);
                let residual = pixel - prediction;
                let packed = pack_signed(residual);

                residuals.push(packed);
                max_residual = max_residual.max(packed);
            }
        }
    }

    (residuals, max_residual)
}

/// Builds a histogram from residuals, encoding through HybridUint {4,2,0}.
/// Returns (histogram_on_tokens, max_token).
pub fn build_histogram_from_residuals(residuals: &[u32], _max_residual: u32) -> (Vec<u32>, u32) {
    let mut max_token: u32 = 0;
    // First pass: find max token
    for &r in residuals {
        let (token, _, _) = MODULAR_HYBRID_UINT.encode(r);
        max_token = max_token.max(token);
    }
    // Second pass: build histogram on tokens
    let histogram_size = (max_token + 1) as usize;
    let mut histogram = vec![0u32; histogram_size];
    for &r in residuals {
        let (token, _, _) = MODULAR_HYBRID_UINT.encode(r);
        histogram[token as usize] += 1;
    }
    (histogram, max_token)
}

/// Result of writing the global modular section.
/// Contains the entropy codes needed to encode pixel data in group sections.
pub enum GlobalModularState {
    /// Huffman entropy coding state.
    Huffman {
        /// Huffman bit depths for each HybridUint token.
        depths: Vec<u8>,
        /// Huffman codes for each HybridUint token.
        codes: Vec<u16>,
        /// Maximum HybridUint token value.
        max_token: u32,
    },
    /// ANS entropy coding state (single-context gradient tree).
    Ans {
        /// The ANS entropy code (distributions, context map, etc.)
        code: OwnedAnsEntropyCode,
    },
    /// ANS entropy coding with learned MA tree (multi-context).
    AnsWithTree {
        /// The ANS entropy code (multiple distributions, context map).
        code: OwnedAnsEntropyCode,
        /// The learned MA tree for per-pixel predictor/context selection.
        tree: super::tree::Tree,
    },
}

/// CeilLog2Nonzero matching the JXL spec.
fn ceil_log2_nonzero(x: u32) -> u32 {
    debug_assert!(x > 0);
    let floor = 31 - x.leading_zeros();
    if x.is_power_of_two() {
        floor
    } else {
        floor + 1
    }
}

/// Write ANS data histogram header for a single-context modular stream.
///
/// For modular with a single-leaf MA tree (num_dist=1), the context map is NOT written.
/// Layout: lz77.enabled=0 + use_prefix_code=0 + log_alpha_size + HybridUint config + ANS distribution
pub(super) fn write_ans_modular_header(
    writer: &mut BitWriter,
    code: &OwnedAnsEntropyCode,
) -> Result<()> {
    assert_eq!(
        code.histograms.len(),
        1,
        "modular ANS header only supports single-distribution (single-leaf tree)"
    );

    // lz77.enabled = 0
    writer.write(1, 0)?;

    // NO context map for num_dist=1

    // use_prefix_code = 0 (ANS, not Huffman)
    writer.write(1, 0)?;

    // log_alpha_size - 5 (2 bits)
    let las = code.log_alpha_size;
    writer.write(2, (las - 5) as u64)?;

    // HybridUint config: {4, 2, 0}
    let se_bits = ceil_log2_nonzero(las as u32 + 1);
    writer.write(se_bits as usize, 4)?; // split_exponent = 4
    writer.write(3, 2)?; // msb_in_token = 2
    writer.write(2, 0)?; // lsb_in_token = 0

    // Write the single ANS distribution
    code.histograms[0].write(writer)?;

    Ok(())
}

/// Writes the global modular section (tree + histogram) for multi-group encoding.
///
/// This writes:
/// - dc_quant.all_default = 1
/// - has_tree = 1
/// - Tree histogram and tokens (Gradient predictor)
/// - Data histogram with HybridUint {4,2,0} (Huffman or ANS)
///
/// `all_residuals` are the raw packed residuals from all groups (needed for ANS histogram building).
/// `histogram` and `max_token` are built from HybridUint-encoded tokens (not raw residuals).
/// Returns the entropy coding state needed to encode pixel data in group sections.
pub fn write_global_modular_section(
    all_residuals: &[u32],
    histogram: &[u32],
    max_token: u32,
    writer: &mut BitWriter,
    use_ans: bool,
    rct_type: Option<RctType>,
) -> Result<GlobalModularState> {
    crate::trace::debug_eprintln!(
        "GLOBAL_MODULAR [bit {}]: Starting global section (ans={})",
        writer.bits_written(),
        use_ans
    );

    // dc_quant.all_default = true
    writer.write(1, 1)?;
    // has_tree = true
    writer.write(1, 1)?;

    // Tree histogram (supports symbols 0-5 for Gradient predictor)
    let (tree_depths, tree_codes) = write_tree_histogram_for_gradient(writer)?;
    write_gradient_tree_tokens(writer, &tree_depths, &tree_codes)?;

    if use_ans {
        // Build ANS code from all residuals across all groups
        let tokens: Vec<AnsToken> = all_residuals.iter().map(|&r| AnsToken::new(0, r)).collect();
        let code = build_entropy_code_ans(&tokens, 1); // 1 context for single-leaf tree

        // Write ANS data header (distribution + config)
        write_ans_modular_header(writer, &code)?;

        // Write GlobalModular's ModularHeader
        writer.write(1, 1)?; // use_global_tree = true
        writer.write(1, 1)?; // wp_params.default_wp = true
        write_global_transforms(writer, rct_type)?;

        // Byte-align at end of global section
        writer.zero_pad_to_byte();
        crate::trace::debug_eprintln!(
            "GLOBAL_MODULAR [bit {}]: Global section done (ANS)",
            writer.bits_written()
        );

        Ok(GlobalModularState::Ans { code })
    } else {
        // Data histogram with HybridUint {4,2,0} + Huffman
        let (depths, codes) = write_hybrid_data_histogram(writer, histogram, max_token)?;

        // Write GlobalModular's ModularHeader
        writer.write(1, 1)?; // use_global_tree = true
        writer.write(1, 1)?; // wp_params.default_wp = true
        write_global_transforms(writer, rct_type)?;

        // Byte-align at end of global section
        writer.zero_pad_to_byte();
        crate::trace::debug_eprintln!(
            "GLOBAL_MODULAR [bit {}]: Global section done (Huffman)",
            writer.bits_written()
        );

        Ok(GlobalModularState::Huffman {
            depths,
            codes,
            max_token,
        })
    }
}

/// Writes the global modular section with a learned MA tree for multi-group encoding.
///
/// This writes:
/// - dc_quant.all_default = 1
/// - has_tree = 1
/// - Learned tree (write_tree)
/// - lz77.enabled = 0
/// - Multi-context ANS data histogram (write_entropy_code_ans)
/// - GroupHeader (use_global_tree=1, wp_header.all_default=1, num_transforms=0)
pub fn write_global_modular_section_with_tree(
    images: &[ModularImage],
    writer: &mut BitWriter,
    effort: u8,
    rct_type: Option<RctType>,
) -> Result<GlobalModularState> {
    use super::encode::write_tree;
    use super::tree::count_contexts;
    use super::tree_learn::{
        TreeLearningParams, TreeSamples, collect_residuals_with_tree, compute_best_tree,
        compute_gather_stride, gather_samples_strided,
    };
    use crate::entropy_coding::encode::{build_entropy_code_ans, write_entropy_code_ans};

    // Step 1: Gather samples from all groups (with subsampling for large images)
    let total_pixels: usize = images
        .iter()
        .flat_map(|img| img.channels.iter())
        .map(|ch| ch.width() * ch.height())
        .sum();
    let stride = compute_gather_stride(total_pixels);
    let mut samples = TreeSamples::new();
    for (group_idx, group_image) in images.iter().enumerate() {
        gather_samples_strided(&mut samples, group_image, group_idx as u32, 0, stride);
    }

    // Step 2: Learn tree with effort-dependent parameters
    let params = TreeLearningParams::for_effort(effort);
    let tree = compute_best_tree(&mut samples, &params);
    let num_contexts = count_contexts(&tree) as usize;

    crate::trace::debug_eprintln!(
        "GLOBAL_MODULAR_TREE: {} nodes, {} leaves/contexts from {} samples",
        tree.len(),
        num_contexts,
        samples.num_samples
    );

    // Step 3: Collect residuals from all groups with tree
    let mut all_tokens = Vec::new();
    for (group_idx, group_image) in images.iter().enumerate() {
        let group_tokens = collect_residuals_with_tree(group_image, &tree, group_idx as u32);
        all_tokens.extend(group_tokens);
    }

    // Step 4: Build multi-context ANS code
    let code = build_entropy_code_ans(&all_tokens, num_contexts);

    // Step 5: Write bitstream
    // dc_quant.all_default = true
    writer.write(1, 1)?;
    // has_tree = true
    writer.write(1, 1)?;

    // Write the learned tree
    write_tree(writer, &tree)?;

    // Write ANS data histogram.
    // JXL spec: context map is only written when num_contexts > 1.
    if num_contexts > 1 {
        writer.write(1, 0)?; // lz77.enabled = 0
        write_entropy_code_ans(&code, writer)?;
    } else {
        // write_ans_modular_header writes lz77.enabled=0 + omits context map
        write_ans_modular_header(writer, &code)?;
    }

    // GroupHeader (global modular group)
    writer.write(1, 1)?; // use_global_tree = true
    writer.write(1, 1)?; // wp_params.default_wp = true
    write_global_transforms(writer, rct_type)?;

    writer.zero_pad_to_byte();

    Ok(GlobalModularState::AnsWithTree { code, tree })
}

/// Write num_transforms + optional RCT transform for the global GroupHeader.
fn write_global_transforms(writer: &mut BitWriter, rct_type: Option<RctType>) -> Result<()> {
    if let Some(rct) = rct_type {
        writer.write(2, 1)?; // nb_transforms = 1
        write_rct_transform(writer, 0, rct)?;
    } else {
        writer.write(2, 0)?; // nb_transforms = 0
    }
    Ok(())
}

/// Collect packed residuals from a group image using gradient prediction.
fn collect_group_residuals(group_image: &ModularImage) -> Vec<u32> {
    let mut residuals = Vec::new();
    for channel in &group_image.channels {
        let width = channel.width();
        let height = channel.height();
        for y in 0..height {
            for x in 0..width {
                let pixel = channel.get(x, y);
                let left = if x > 0 { channel.get(x - 1, y) } else { 0 };
                let top = if y > 0 { channel.get(x, y - 1) } else { left };
                let topleft = if x > 0 && y > 0 {
                    channel.get(x - 1, y - 1)
                } else {
                    left
                };
                let prediction = predict_gradient(left, top, topleft);
                let residual = pixel - prediction;
                residuals.push(pack_signed(residual));
            }
        }
    }
    residuals
}

/// Writes a group's data section for multi-group modular encoding.
///
/// This writes:
/// - GroupHeader (use_global_tree=1, wp_header.all_default=1, num_transforms=0)
/// - Encoded pixel residuals using HybridUint {4,2,0} + global entropy codes
///
/// The `group_image` should be the extracted region for this group.
pub fn write_group_modular_section(
    group_image: &ModularImage,
    state: &GlobalModularState,
    writer: &mut BitWriter,
) -> Result<()> {
    crate::trace::debug_eprintln!(
        "GROUP_MODULAR [bit {}]: Starting group section ({}x{})",
        writer.bits_written(),
        group_image.width(),
        group_image.height()
    );

    // GroupHeader
    writer.write(1, 1)?; // use_global_tree = true
    writer.write(1, 1)?; // wp_header.all_default = true
    writer.write(2, 0)?; // num_transforms = 0

    match state {
        GlobalModularState::Huffman {
            depths,
            codes,
            max_token: _,
        } => {
            // Encode residuals with HybridUint {4,2,0} + Huffman
            for channel in &group_image.channels {
                let width = channel.width();
                let height = channel.height();
                for y in 0..height {
                    for x in 0..width {
                        let pixel = channel.get(x, y);
                        let left = if x > 0 { channel.get(x - 1, y) } else { 0 };
                        let top = if y > 0 { channel.get(x, y - 1) } else { left };
                        let topleft = if x > 0 && y > 0 {
                            channel.get(x - 1, y - 1)
                        } else {
                            left
                        };
                        let prediction = predict_gradient(left, top, topleft);
                        let residual = pixel - prediction;
                        let packed = pack_signed(residual);

                        let (token, extra_bits, num_extra) = MODULAR_HYBRID_UINT.encode(packed);
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
            }
        }
        GlobalModularState::Ans { code } => {
            // Collect residuals for this group and encode with ANS
            let residuals = collect_group_residuals(group_image);
            let tokens: Vec<AnsToken> = residuals.iter().map(|&r| AnsToken::new(0, r)).collect();
            write_tokens_ans(&tokens, code, None, writer)?;
        }
        GlobalModularState::AnsWithTree { code, tree } => {
            // Collect residuals using the learned tree (multi-context)
            let tokens = super::tree_learn::collect_residuals_with_tree(group_image, tree, 0);
            write_tokens_ans(&tokens, code, None, writer)?;
        }
    }

    // Byte-align at end of group section
    writer.zero_pad_to_byte();
    crate::trace::debug_eprintln!(
        "GROUP_MODULAR [bit {}]: Group section done",
        writer.bits_written()
    );

    Ok(())
}
