// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Modular encoder orchestration.
//!
//! High-level entry points for modular (lossless) encoding: gradient prediction,
//! LZ77, RCT, palette, squeeze, weighted predictor, and tree-learned streams.
//!
//! Low-level primitives are in `encode_primitives`, tree writing in `encode_tree`,
//! and transform descriptors in `encode_transforms`. All public items are re-exported
//! here so consumers can continue using `crate::modular::encode::*`.

use crate::bit_writer::BitWriter;
#[allow(unused_imports)]
use crate::debug_rect;
use crate::entropy_coding::encode::write_tokens_ans;
use crate::error::Result;
use crate::modular::channel::ModularImage;
use crate::modular::rct::{RctType, forward_rct};

// Re-export everything from sub-modules so existing import paths work unchanged.
pub(crate) use super::encode_primitives::*;
pub(crate) use super::encode_transforms::*;
pub(crate) use super::encode_tree::*;

/// Collect residuals using gradient prediction and identify LZ77 runs.
fn collect_residuals_with_prediction(image: &ModularImage) -> Vec<Token> {
    let mut tokens = Vec::new();
    let mut current_run = 0usize;
    let mut num_decoded = 0usize; // Track how many values we've output (for LZ77 validity)
    let mut last_value = 0u32; // Track last value output (for LZ77 copy)
    let mut debug_count = 0;

    for channel in &image.channels {
        // Flush any accumulated run at channel boundary
        // LZ77 should not span channel boundaries because each channel is decoded separately
        if current_run > K_LZ77_MIN_LENGTH {
            tokens.push(Token::Lz77Run(current_run));
            num_decoded += current_run;
        } else {
            for _ in 0..current_run {
                tokens.push(Token::Raw(last_value));
                num_decoded += 1;
            }
        }
        current_run = 0;
        // Reset last_value to an impossible value to prevent LZ77 from first pixel of new channel
        last_value = u32::MAX;

        let width = channel.width();
        let height = channel.height();

        for y in 0..height {
            for x in 0..width {
                let pixel = channel.get(x, y);

                // Get neighbors (matching JXL decoder and write_simple_modular_stream)
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

                if debug_count < 20 {
                    let _channel_idx = image
                        .channels
                        .iter()
                        .position(|c| std::ptr::eq(c, channel))
                        .unwrap();
                    crate::trace::debug_eprintln!(
                        "RESIDUAL[{}]: ch={} y={} x={} pixel={}, pred={}, residual={}, packed={}",
                        debug_count,
                        _channel_idx,
                        y,
                        x,
                        pixel,
                        prediction,
                        residual,
                        packed
                    );
                    debug_count += 1;
                }

                // LZ77 with distance=1 copies the last value.
                // We can only use LZ77 if:
                // 1. We have at least one value in the window (num_decoded > 0)
                // 2. The value to repeat (packed) matches the last value
                let can_use_lz77 = num_decoded > 0 && packed == last_value;

                if can_use_lz77 {
                    current_run += 1;
                } else {
                    // Flush any accumulated run
                    if current_run > K_LZ77_MIN_LENGTH {
                        tokens.push(Token::Lz77Run(current_run));
                        num_decoded += current_run;
                        // Note: after LZ77 copy, last_value stays the same
                    } else {
                        // Output individual copies of last_value
                        for _ in 0..current_run {
                            tokens.push(Token::Raw(last_value));
                            num_decoded += 1;
                        }
                    }
                    current_run = 0;
                    tokens.push(Token::Raw(packed));
                    num_decoded += 1;
                    last_value = packed;
                }
            }
        }
    }

    // Flush final run
    if current_run > K_LZ77_MIN_LENGTH {
        tokens.push(Token::Lz77Run(current_run));
    } else {
        for _ in 0..current_run {
            tokens.push(Token::Raw(last_value));
        }
    }

    tokens
}

/// Writes an improved modular stream with gradient prediction and LZ77.
///
/// For VarDCT subbitstreams, set `skip_group_header = true` since the GroupHeader
/// is written separately before calling this function.
pub fn write_improved_modular_stream(
    image: &ModularImage,
    writer: &mut BitWriter,
    use_ans: bool,
) -> Result<()> {
    write_improved_modular_stream_inner(image, writer, false, use_ans)
}

fn write_improved_modular_stream_inner(
    image: &ModularImage,
    writer: &mut BitWriter,
    _skip_group_header: bool,
    use_ans: bool,
) -> Result<()> {
    // Collect residuals with gradient prediction
    let tokens = collect_residuals_with_prediction(image);

    // Build sparse histogram [0..18] + [224..256]
    let sparse_counts = build_sparse_histogram(&tokens);

    let _num_raw_used = sparse_counts[..K_NUM_RAW_SYMBOLS]
        .iter()
        .filter(|&&c| c > 0)
        .count();
    let _num_lz77_used = sparse_counts[K_LZ77_MIN_SYMBOL..]
        .iter()
        .filter(|&&c| c > 0)
        .count();
    let num_lz77_runs = tokens
        .iter()
        .filter(|t| matches!(t, Token::Lz77Run(_)))
        .count();

    crate::trace::debug_eprintln!(
        "IMPROVED: {} tokens, {} raw symbols used, {} lz77 tokens used, {} lz77 runs",
        tokens.len(),
        _num_raw_used,
        _num_lz77_used,
        num_lz77_runs
    );

    // If no LZ77 runs, fall back to simple encoding
    if num_lz77_runs == 0 {
        return write_simple_modular_stream(image, writer, use_ans);
    }

    // === Global section (LfGlobal) ===
    writer.write(1, 1)?; // dc_quant.all_default = true
    writer.write(1, 1)?; // has_tree = true

    // Tree histogram for single-leaf tree with Gradient predictor
    let (tree_depths, tree_codes) = write_tree_histogram_for_gradient(writer)?;
    write_gradient_tree_tokens(writer, &tree_depths, &tree_codes)?;

    // Data histogram with LZ77 (sparse alphabet)
    let (depths, codes) = write_sparse_lz77_histogram(writer, &sparse_counts)?;

    // GroupHeader
    writer.write(1, 1)?; // use_global_tree = true
    writer.write(1, 1)?; // wp_header.all_default = true
    writer.write(2, 0)?; // num_transforms = 0

    // Encode tokens using sparse alphabet
    for token in &tokens {
        match token {
            Token::Raw(value) => {
                // Encode value using hybrid uint (split_exponent=0, msb_in_token=0, lsb_in_token=0)
                let (tok, nbits, extra) = encode_hybrid_uint_000(*value);
                let symbol = tok as usize;
                let depth = depths[symbol];
                let code = codes[symbol];
                if depth > 0 {
                    writer.write(depth as usize, code as u64)?;
                }
                // Write extra bits
                if nbits > 0 {
                    writer.write(nbits as usize, extra as u64)?;
                }
            }
            Token::Lz77Run(count) => {
                // LZ77: encode as single symbol >= 224
                // Length encoding: symbol = min_symbol + HybridUint(length - min_length)
                let adjusted = count - K_LZ77_MIN_LENGTH;
                let (tok, nbits, extra) = encode_hybrid_uint_lz77_length(adjusted as u32);

                // Symbol in sparse alphabet
                let symbol = K_LZ77_MIN_SYMBOL + tok as usize;
                let depth = depths[symbol];
                let code = codes[symbol];
                if depth > 0 {
                    writer.write(depth as usize, code as u64)?;
                }

                // Write extra bits for length
                if nbits > 0 {
                    writer.write(nbits as usize, extra as u64)?;
                }

                // Write distance symbol for distance=1 (RLE)
                // With dist_multiplier = image_width, distance formula is:
                //   distance = dist_multiplier * dist + offset
                // SPECIAL_DISTANCES[0] = (0, 1): distance = width * 1 + 0 = width
                // SPECIAL_DISTANCES[1] = (1, 0): distance = width * 0 + 1 = 1 ✓
                // So for distance=1, we need distance symbol 1, not 0!
                let dist_symbol = 1u32;
                let (dist_tok, dist_nbits, dist_extra) = encode_hybrid_uint_000(dist_symbol);
                let dist_depth = depths[dist_tok as usize];
                let dist_code = codes[dist_tok as usize];
                if dist_depth > 0 {
                    writer.write(dist_depth as usize, dist_code as u64)?;
                }
                if dist_nbits > 0 {
                    writer.write(dist_nbits as usize, dist_extra as u64)?;
                }
            }
        }
    }

    crate::trace::debug_eprintln!(
        "LZ77 [bit {}]: Encoded {} tokens",
        writer.bits_written(),
        tokens.len()
    );

    writer.zero_pad_to_byte();
    Ok(())
}

/// Clamped gradient predictor (predictor 5 in JXL spec).
/// Returns `left + top - topleft` clamped to [min(left,top), max(left,top)]
/// when topleft is outside that range.
#[inline]
fn predict_gradient(left: i32, top: i32, topleft: i32) -> i32 {
    let min = left.min(top);
    let max = left.max(top);
    let grad = left + top - topleft;
    let grad_clamp_max = if topleft < min { max } else { grad };
    if topleft > max { min } else { grad_clamp_max }
}

// Set to true to use Zero predictor for debugging
// Try: false = gradient tree path, true = zero tree path (works)
const USE_ZERO_PREDICTOR: bool = false;

/// Simpler stream without LZ77 but with gradient prediction.
pub fn write_simple_modular_stream(
    image: &ModularImage,
    writer: &mut BitWriter,
    use_ans: bool,
) -> Result<()> {
    // Collect residuals with gradient prediction
    let mut residuals = Vec::new();

    for channel in &image.channels {
        let width = channel.width();
        let height = channel.height();

        for y in 0..height {
            for x in 0..width {
                let pixel = channel.get(x, y);

                let prediction = if USE_ZERO_PREDICTOR {
                    0
                } else {
                    let left = if x > 0 { channel.get(x - 1, y) } else { 0 };
                    let top = if y > 0 { channel.get(x, y - 1) } else { left };
                    let topleft = if x > 0 && y > 0 {
                        channel.get(x - 1, y - 1)
                    } else {
                        left
                    };
                    predict_gradient(left, top, topleft)
                };

                let residual = pixel - prediction;
                let packed = pack_signed(residual);
                residuals.push(packed);
            }
        }
    }

    // === Global section ===
    writer.write(1, 1)?; // dc_quant.all_default = true
    writer.write(1, 1)?; // has_tree = true

    if USE_ZERO_PREDICTOR {
        write_tree_histogram_for_zero(writer)?;
    } else {
        let (tree_depths, tree_codes) = write_tree_histogram_for_gradient(writer)?;
        write_gradient_tree_tokens(writer, &tree_depths, &tree_codes)?;
    }

    if use_ans {
        // ANS entropy coding path
        let (tokens, code) = build_ans_modular_code(&residuals);
        write_ans_modular_header(writer, &code)?;

        // GroupHeader
        writer.write(1, 1)?; // use_global_tree = true
        writer.write(1, 1)?; // wp_header.all_default = true
        writer.write(2, 0)?; // num_transforms = 0

        write_ans_modular_tokens(writer, &tokens, &code)?;
    } else {
        // Huffman path with HybridUint {4,2,0}
        let (encoded, max_token) = encode_residuals_hybrid(&residuals);
        let histogram = build_token_histogram(&encoded, max_token);
        let (depths, codes) = write_hybrid_data_histogram(writer, &histogram, max_token)?;

        // GroupHeader
        writer.write(1, 1)?; // use_global_tree = true
        writer.write(1, 1)?; // wp_header.all_default = true
        writer.write(2, 0)?; // num_transforms = 0

        write_hybrid_residuals(writer, &encoded, &depths, &codes)?;
    }

    writer.zero_pad_to_byte();
    Ok(())
}

/// Write modular stream with palette transform for few-color images.
///
/// When an image has few unique colors (≤256 for 8-bit), palette encoding
/// replaces multi-channel data with a palette meta-channel + index channel.
/// This provides 19-57% compression improvement on graphics/screenshots.
pub fn write_modular_stream_with_palette(
    image: &ModularImage,
    writer: &mut BitWriter,
    use_ans: bool,
    begin_c: usize,
    num_c: usize,
) -> Result<()> {
    use super::palette::{analyze_palette, apply_palette};

    let max_colors = 1usize << image.bit_depth.min(12);
    let analysis = analyze_palette(image, begin_c, num_c, max_colors);

    if !analysis.use_palette {
        // Fallback: use RCT if RGB, otherwise simple
        if image.channels.len() >= 3 {
            return write_modular_stream_with_rct(image, writer, use_ans);
        } else {
            return write_simple_modular_stream(image, writer, use_ans);
        }
    }

    // Apply palette transform
    let mut transformed = image.clone();
    let nb_colors = apply_palette(&mut transformed, begin_c, num_c, &analysis)?;

    crate::trace::debug_eprintln!(
        "PALETTE: {} unique colors, {} channels → palette({}) + index",
        analysis.num_colors,
        num_c,
        nb_colors
    );

    // Collect residuals with gradient prediction on transformed channels
    let mut residuals = Vec::new();

    for channel in &transformed.channels {
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

                residuals.push(packed);
            }
        }
    }

    // === Global section ===
    writer.write(1, 1)?; // dc_quant.all_default = true
    writer.write(1, 1)?; // has_tree = true

    let (tree_depths, tree_codes) = write_tree_histogram_for_gradient(writer)?;
    write_gradient_tree_tokens(writer, &tree_depths, &tree_codes)?;

    if use_ans {
        let (tokens, code) = build_ans_modular_code(&residuals);

        write_ans_modular_header(writer, &code)?;

        // GroupHeader with 1 transform (Palette)
        writer.write(1, 1)?; // use_global_tree = true
        writer.write(1, 1)?; // wp_header.all_default = true
        writer.write(2, 1)?; // num_transforms = 1
        write_palette_transform(writer, begin_c, num_c, nb_colors, 0, 0)?;

        write_ans_modular_tokens(writer, &tokens, &code)?;
    } else {
        let (encoded, max_token) = encode_residuals_hybrid(&residuals);
        let histogram = build_token_histogram(&encoded, max_token);
        let (depths, codes) = write_hybrid_data_histogram(writer, &histogram, max_token)?;

        // GroupHeader with 1 transform (Palette)
        writer.write(1, 1)?; // use_global_tree = true
        writer.write(1, 1)?; // wp_header.all_default = true
        writer.write(2, 1)?; // num_transforms = 1
        write_palette_transform(writer, begin_c, num_c, nb_colors, 0, 0)?;

        write_hybrid_residuals(writer, &encoded, &depths, &codes)?;
    }

    writer.zero_pad_to_byte();
    Ok(())
}

/// Write modular stream with lossy delta palette transform.
///
/// Uses a two-pass algorithm matching libjxl's FwdPalette:
/// 1. Discovers frequent color deltas (residuals from prediction)
/// 2. Applies palette with error diffusion using discovered deltas
///
/// The lossy palette quantizes colors to a small palette + delta entries,
/// producing smaller files at the cost of some color accuracy.
pub fn write_modular_stream_with_lossy_palette(
    image: &ModularImage,
    writer: &mut BitWriter,
    use_ans: bool,
    begin_c: usize,
    num_c: usize,
    max_palette_colors: usize,
) -> Result<()> {
    use super::palette::apply_lossy_palette;

    let mut transformed = image.clone();
    let result = apply_lossy_palette(&mut transformed, begin_c, num_c, max_palette_colors);

    let result = match result {
        Some(r) => r,
        None => {
            // Lossy palette not beneficial, fall back to lossless RCT
            if image.channels.len() >= 3 {
                return write_modular_stream_with_rct(image, writer, use_ans);
            } else {
                return write_simple_modular_stream(image, writer, use_ans);
            }
        }
    };

    let nb_colors = result.nb_colors;
    let nb_deltas = result.nb_deltas;
    let predictor = result.predictor;

    crate::trace::debug_eprintln!(
        "LOSSY PALETTE: {} colors + {} deltas, predictor={}, {} channels → palette + index",
        nb_colors,
        nb_deltas,
        predictor,
        num_c,
    );

    // Collect residuals with gradient prediction on transformed channels
    let mut residuals = Vec::new();
    for channel in &transformed.channels {
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
                residuals.push(packed);
            }
        }
    }

    // === Global section ===
    writer.write(1, 1)?; // dc_quant.all_default = true
    writer.write(1, 1)?; // has_tree = true

    let (tree_depths, tree_codes) = write_tree_histogram_for_gradient(writer)?;
    write_gradient_tree_tokens(writer, &tree_depths, &tree_codes)?;

    if use_ans {
        let (tokens, code) = build_ans_modular_code(&residuals);
        write_ans_modular_header(writer, &code)?;

        // GroupHeader with 1 transform (Palette)
        writer.write(1, 1)?; // use_global_tree = true
        writer.write(1, 1)?; // wp_header.all_default = true
        writer.write(2, 1)?; // num_transforms = 1
        write_palette_transform(writer, begin_c, num_c, nb_colors, nb_deltas, predictor)?;

        write_ans_modular_tokens(writer, &tokens, &code)?;
    } else {
        let (encoded, max_token) = encode_residuals_hybrid(&residuals);
        let histogram = build_token_histogram(&encoded, max_token);
        let (depths, codes) = write_hybrid_data_histogram(writer, &histogram, max_token)?;

        // GroupHeader with 1 transform (Palette)
        writer.write(1, 1)?; // use_global_tree = true
        writer.write(1, 1)?; // wp_header.all_default = true
        writer.write(2, 1)?; // num_transforms = 1
        write_palette_transform(writer, begin_c, num_c, nb_colors, nb_deltas, predictor)?;

        write_hybrid_residuals(writer, &encoded, &depths, &codes)?;
    }

    writer.zero_pad_to_byte();
    Ok(())
}

/// Write modular stream with Squeeze (Haar wavelet) transform.
///
/// Decomposes channels into low-frequency (average) + high-frequency (residual)
/// pairs by halving resolution. Enables progressive decoding and improves
/// compression on smooth content.
pub fn write_modular_stream_with_squeeze(
    image: &ModularImage,
    writer: &mut BitWriter,
    use_ans: bool,
) -> Result<()> {
    use super::squeeze::{apply_squeeze, default_squeeze_params};

    let params = default_squeeze_params(image);
    if params.is_empty() {
        // Image too small for squeeze, fall back
        if image.channels.len() >= 3 {
            return write_modular_stream_with_rct(image, writer, use_ans);
        } else {
            return write_simple_modular_stream(image, writer, use_ans);
        }
    }

    // Apply RCT (YCoCg) before squeeze for RGB images to decorrelate channels
    let mut transformed = image.clone();
    let has_rct = transformed.channels.len() >= 3;
    if has_rct {
        let rct_type = RctType::YCOCG;
        forward_rct(&mut transformed.channels, 0, rct_type)?;
    }

    // Apply forward squeeze
    apply_squeeze(&mut transformed, &params)?;

    crate::trace::debug_eprintln!(
        "SQUEEZE: {} steps, {} → {} channels, rct={}",
        params.len(),
        image.channels.len(),
        transformed.channels.len(),
        has_rct,
    );

    // Collect residuals with gradient prediction on all transformed channels
    let mut residuals = Vec::new();
    let mut max_residual: u32 = 0;

    for channel in &transformed.channels {
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

                residuals.push(packed);
                max_residual = max_residual.max(packed);
            }
        }
    }

    // === Global section ===
    writer.write(1, 1)?; // dc_quant.all_default = true
    writer.write(1, 1)?; // has_tree = true

    let (tree_depths, tree_codes) = write_tree_histogram_for_gradient(writer)?;
    write_gradient_tree_tokens(writer, &tree_depths, &tree_codes)?;

    if use_ans {
        let (tokens, code) = build_ans_modular_code(&residuals);
        write_ans_modular_header(writer, &code)?;

        // GroupHeader with transforms: RCT (if RGB) + Squeeze
        writer.write(1, 1)?; // use_global_tree = true
        writer.write(1, 1)?; // wp_header.all_default = true
        if has_rct {
            // num_transforms = 2: U32 BitsOffset(4,2), offset=0
            writer.write(2, 2)?;
            writer.write(4, 0)?;
            write_rct_transform(writer, 0, RctType::YCOCG)?;
            write_squeeze_transform(writer, &params)?;
        } else {
            writer.write(2, 1)?; // num_transforms = 1
            write_squeeze_transform(writer, &params)?;
        }

        write_ans_modular_tokens(writer, &tokens, &code)?;
    } else {
        let (encoded, max_token) = encode_residuals_hybrid(&residuals);
        let histogram = build_token_histogram(&encoded, max_token);
        let (depths, codes) = write_hybrid_data_histogram(writer, &histogram, max_token)?;

        // GroupHeader with transforms: RCT (if RGB) + Squeeze
        writer.write(1, 1)?; // use_global_tree = true
        writer.write(1, 1)?; // wp_header.all_default = true
        if has_rct {
            // num_transforms = 2: U32 BitsOffset(4,2), offset=0
            writer.write(2, 2)?;
            writer.write(4, 0)?;
            write_rct_transform(writer, 0, RctType::YCOCG)?;
            write_squeeze_transform(writer, &params)?;
        } else {
            writer.write(2, 1)?; // num_transforms = 1
            write_squeeze_transform(writer, &params)?;
        }

        write_hybrid_residuals(writer, &encoded, &depths, &codes)?;
    }

    writer.zero_pad_to_byte();
    Ok(())
}

/// Write modular stream with RCT (YCoCg) transform for RGB images.
///
/// This function:
/// 1. Applies YCoCg RCT to decorrelate RGB channels
/// 2. Signals the transform in the bitstream
/// 3. Encodes the transformed data
///
/// YCoCg improves compression by 15-20% for typical RGB images.
pub fn write_modular_stream_with_rct(
    image: &ModularImage,
    writer: &mut BitWriter,
    use_ans: bool,
) -> Result<()> {
    // Check if palette is more beneficial than RCT
    if let Some((begin_c, num_c)) = super::palette::should_use_palette(image) {
        return write_modular_stream_with_palette(image, writer, use_ans, begin_c, num_c);
    }

    // Only apply RCT to RGB images (3+ channels)
    if image.channels.len() < 3 {
        return write_simple_modular_stream(image, writer, use_ans);
    }

    // Clone the image and apply forward RCT (YCoCg)
    let mut transformed = image.clone();
    let rct_type = RctType::YCOCG;
    forward_rct(&mut transformed.channels, 0, rct_type)?;

    crate::trace::debug_eprintln!(
        "RCT: Applied YCoCg transform to {} channels",
        transformed.channels.len()
    );

    // Collect residuals with gradient prediction on transformed channels
    let mut residuals = Vec::new();
    let mut max_residual: u32 = 0;

    for channel in &transformed.channels {
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

                residuals.push(packed);
                max_residual = max_residual.max(packed);
            }
        }
    }

    // === Global section ===
    writer.write(1, 1)?; // dc_quant.all_default = true
    writer.write(1, 1)?; // has_tree = true

    let (tree_depths, tree_codes) = write_tree_histogram_for_gradient(writer)?;
    write_gradient_tree_tokens(writer, &tree_depths, &tree_codes)?;

    if use_ans {
        let (tokens, code) = build_ans_modular_code(&residuals);
        write_ans_modular_header(writer, &code)?;

        // GroupHeader with 1 transform
        writer.write(1, 1)?; // use_global_tree = true
        writer.write(1, 1)?; // wp_header.all_default = true
        writer.write(2, 1)?; // num_transforms = 1
        write_rct_transform(writer, 0, rct_type)?;

        write_ans_modular_tokens(writer, &tokens, &code)?;
    } else {
        let (encoded, max_token) = encode_residuals_hybrid(&residuals);
        let histogram = build_token_histogram(&encoded, max_token);
        let (depths, codes) = write_hybrid_data_histogram(writer, &histogram, max_token)?;

        // GroupHeader with 1 transform
        writer.write(1, 1)?; // use_global_tree = true
        writer.write(1, 1)?; // wp_header.all_default = true
        writer.write(2, 1)?; // num_transforms = 1
        write_rct_transform(writer, 0, rct_type)?;

        write_hybrid_residuals(writer, &encoded, &depths, &codes)?;
    }

    writer.zero_pad_to_byte();
    Ok(())
}

/// Write modular stream using the Weighted predictor for better compression.
///
/// The Weighted predictor adapts to local image statistics and typically
/// achieves better compression than the Gradient predictor for natural images.
pub fn write_modular_stream_with_weighted(
    image: &ModularImage,
    writer: &mut BitWriter,
    use_ans: bool,
) -> Result<()> {
    use super::predictor::{Neighbors, WeightedPredictorParams, WeightedPredictorState};

    let params = WeightedPredictorParams::default();

    // Collect residuals with weighted prediction
    let mut residuals = Vec::new();

    for channel in &image.channels {
        let width = channel.width();
        let height = channel.height();
        let mut wp_state = WeightedPredictorState::new(&params, width);

        for y in 0..height {
            for x in 0..width {
                let pixel = channel.get(x, y);
                let neighbors = Neighbors::gather(channel, x, y);
                let prediction = wp_state.predict(x, y, width, &neighbors);

                let residual = pixel - prediction;
                let packed = pack_signed(residual);
                residuals.push(packed);

                wp_state.update_errors(pixel, x, y, width);
            }
        }
    }

    // === Global section ===
    writer.write(1, 1)?; // dc_quant.all_default = true
    writer.write(1, 1)?; // has_tree = true

    write_tree_histogram_for_weighted(writer)?;
    write_weighted_tree_tokens(writer)?;

    if use_ans {
        let (tokens, code) = build_ans_modular_code(&residuals);
        write_ans_modular_header(writer, &code)?;

        writer.write(1, 1)?; // use_global_tree = true
        write_wp_header(writer, &params)?;
        writer.write(2, 0)?; // num_transforms = 0

        write_ans_modular_tokens(writer, &tokens, &code)?;
    } else {
        let (encoded, max_token) = encode_residuals_hybrid(&residuals);
        let histogram = build_token_histogram(&encoded, max_token);
        let (depths, codes) = write_hybrid_data_histogram(writer, &histogram, max_token)?;

        writer.write(1, 1)?; // use_global_tree = true
        write_wp_header(writer, &params)?;
        writer.write(2, 0)?; // num_transforms = 0

        write_hybrid_residuals(writer, &encoded, &depths, &codes)?;
    }

    writer.zero_pad_to_byte();
    Ok(())
}

/// Write modular stream with RCT and Weighted predictor for best compression.
///
/// Combines YCoCg color transform with adaptive weighted prediction.
pub fn write_modular_stream_with_rct_weighted(
    image: &ModularImage,
    writer: &mut BitWriter,
    use_ans: bool,
) -> Result<()> {
    use super::predictor::{Neighbors, WeightedPredictorParams, WeightedPredictorState};

    // Check if palette is more beneficial
    if let Some((begin_c, num_c)) = super::palette::should_use_palette(image) {
        return write_modular_stream_with_palette(image, writer, use_ans, begin_c, num_c);
    }

    if image.channels.len() < 3 {
        return write_modular_stream_with_weighted(image, writer, use_ans);
    }

    let mut transformed = image.clone();
    let rct_type = RctType::YCOCG;
    forward_rct(&mut transformed.channels, 0, rct_type)?;

    let params = WeightedPredictorParams::default();

    // Collect residuals with weighted prediction on transformed channels
    let mut residuals = Vec::new();

    for channel in &transformed.channels {
        let width = channel.width();
        let height = channel.height();
        let mut wp_state = WeightedPredictorState::new(&params, width);

        for y in 0..height {
            for x in 0..width {
                let pixel = channel.get(x, y);
                let neighbors = Neighbors::gather(channel, x, y);
                let prediction = wp_state.predict(x, y, width, &neighbors);

                let residual = pixel - prediction;
                residuals.push(pack_signed(residual));

                wp_state.update_errors(pixel, x, y, width);
            }
        }
    }

    // === Global section ===
    writer.write(1, 1)?; // dc_quant.all_default = true
    writer.write(1, 1)?; // has_tree = true

    write_tree_histogram_for_weighted(writer)?;
    write_weighted_tree_tokens(writer)?;

    if use_ans {
        let (tokens, code) = build_ans_modular_code(&residuals);
        write_ans_modular_header(writer, &code)?;

        writer.write(1, 1)?; // use_global_tree = true
        write_wp_header(writer, &params)?;
        writer.write(2, 1)?; // num_transforms = 1
        write_rct_transform(writer, 0, rct_type)?;

        write_ans_modular_tokens(writer, &tokens, &code)?;
    } else {
        let (encoded, max_token) = encode_residuals_hybrid(&residuals);
        let histogram = build_token_histogram(&encoded, max_token);
        let (depths, codes) = write_hybrid_data_histogram(writer, &histogram, max_token)?;

        writer.write(1, 1)?; // use_global_tree = true
        write_wp_header(writer, &params)?;
        writer.write(2, 1)?; // num_transforms = 1
        write_rct_transform(writer, 0, rct_type)?;

        write_hybrid_residuals(writer, &encoded, &depths, &codes)?;
    }

    writer.zero_pad_to_byte();
    Ok(())
}

/// Fast cost estimate for an image using gradient prediction.
///
/// Matches libjxl's `EstimateCost()` in enc_modular_simd.cc.
/// Uses clamped gradient prediction with residuals bucketed by local complexity
/// (max neighbor difference). Returns total estimated bits.
fn estimate_cost(image: &ModularImage) -> f64 {
    use super::predictor::pack_signed;
    use crate::entropy_coding::hybrid_uint::HybridUintConfig;

    let config = HybridUintConfig::new(4, 2, 0);
    let cutoffs: &[u32] = &[
        0, 1, 3, 5, 7, 11, 15, 23, 31, 47, 63, 95, 127, 191, 255, 392, 500,
    ];
    let nc = cutoffs.len() + 1; // 18 context buckets

    let mut total_bits: f64 = 0.0;
    let mut extra_bits: u64 = 0;
    let mut histograms: Vec<Vec<u32>> = vec![vec![]; nc];

    for ch in &image.channels {
        let w = ch.width();
        let h = ch.height();
        if w == 0 || h == 0 {
            continue;
        }

        for y in 0..h {
            for x in 0..w {
                let val = ch.data()[y * w + x];
                let left = if x > 0 {
                    ch.data()[y * w + x - 1]
                } else if y > 0 {
                    ch.data()[(y - 1) * w + x]
                } else {
                    0
                };
                let top = if y > 0 {
                    ch.data()[(y - 1) * w + x]
                } else {
                    left
                };
                let topleft = if x > 0 && y > 0 {
                    ch.data()[(y - 1) * w + x - 1]
                } else {
                    left
                };

                let max_diff = left.max(top).max(topleft) - left.min(top).min(topleft);
                let max_diff = max_diff as u32;

                // Find context bucket (count how many cutoffs are > max_diff)
                let mut ctx = 0usize;
                for &c in cutoffs {
                    if max_diff < c {
                        ctx += 1;
                    }
                }

                // Gradient prediction residual
                let grad = left + top - topleft;
                let pred = grad.max(left.min(top)).min(left.max(top)); // clamped gradient
                let res = val - pred;
                let packed = pack_signed(res);

                let (token, _bits, nbits) = config.encode(packed);
                if histograms[ctx].len() <= token as usize {
                    histograms[ctx].resize(token as usize + 1, 0);
                }
                histograms[ctx][token as usize] += 1;
                extra_bits += nbits as u64;
            }
        }

        // Sum Shannon entropy per context bucket, then reset
        for hist in &mut histograms {
            let total: u32 = hist.iter().sum();
            if total > 0 {
                let total_f = total as f64;
                for &count in hist.iter() {
                    if count > 0 {
                        let p = count as f64 / total_f;
                        total_bits -= count as f64 * p.log2();
                    }
                }
            }
            hist.clear();
        }
    }

    total_bits + extra_bits as f64
}

/// RCT variants to try at each effort level, matching libjxl's enc_modular.cc.
/// The order is: identity, YCoCg, YCbCr-like, GBR+SubGR, RBG+YCoCg, BGR+YCoCg, GBR+YCoCg.
#[allow(clippy::identity_op, clippy::erasing_op)]
const RCT_CANDIDATES: &[u8] = &[
    0 * 7 + 0, // identity (no transform)
    0 * 7 + 6, // YCoCg
    0 * 7 + 5, // type 5
    1 * 7 + 3, // GBR + SubGR
    3 * 7 + 5, // RBG + type 5
    5 * 7 + 5, // BGR + type 5
    1 * 7 + 5, // GBR + type 5
];

/// Select the best RCT variant by trying candidates and picking the lowest cost.
///
/// Returns the RctType and the transformed image. At effort 7, tries 7 RCT variants
/// matching libjxl's kSquirrel behavior.
pub(crate) fn select_best_rct(image: &ModularImage, nb_rcts_to_try: u8) -> (RctType, ModularImage) {
    use super::rct::{RctType, forward_rct};

    let nb_rcts_to_try = nb_rcts_to_try as usize;

    if nb_rcts_to_try == 0 || image.channels.len() < 3 {
        // Default to YCoCg
        let mut transformed = image.clone();
        forward_rct(&mut transformed.channels, 0, RctType::YCOCG).ok();
        return (RctType::YCOCG, transformed);
    }

    let mut best_cost = f64::MAX;
    let mut best_rct = RctType::YCOCG;
    let mut best_image = None;

    for (i, &rct_val) in RCT_CANDIDATES.iter().enumerate() {
        if i >= nb_rcts_to_try {
            break;
        }
        let rct_type = RctType(rct_val);

        if rct_type.is_noop() {
            // Identity: estimate cost of the original image
            let cost = estimate_cost(image);
            crate::trace::debug_eprintln!("  RCT {:2}: cost={:.0}", rct_val, cost);
            if cost < best_cost {
                best_cost = cost;
                best_rct = rct_type;
                best_image = Some(image.clone());
            }
        } else {
            let mut transformed = image.clone();
            if forward_rct(&mut transformed.channels, 0, rct_type).is_ok() {
                let cost = estimate_cost(&transformed);
                crate::trace::debug_eprintln!("  RCT {:2}: cost={:.0}", rct_val, cost);
                if cost < best_cost {
                    best_cost = cost;
                    best_rct = rct_type;
                    best_image = Some(transformed);
                }
            }
        }
    }

    let work_image = best_image.unwrap_or_else(|| {
        let mut t = image.clone();
        forward_rct(&mut t.channels, 0, RctType::YCOCG).ok();
        t
    });

    crate::trace::debug_eprintln!(
        "RCT_SELECT: best={} (cost={:.0}), tried {} variants",
        best_rct.0,
        best_cost,
        nb_rcts_to_try.min(RCT_CANDIDATES.len()),
    );

    (best_rct, work_image)
}

/// Write a modular stream using a learned MA tree with multi-context ANS.
///
/// This is the single-group version. For multi-group, see section.rs.
///
/// Layout:
/// - dc_quant.all_default = 1
/// - has_tree = 1
/// - Tree (write_tree)
/// - lz77.enabled = 0 for data
/// - Multi-context ANS histogram (write_entropy_code_ans)
/// - GroupHeader (use_global_tree=1, wp_header.all_default=1, num_transforms=0 or 1)
/// - ANS-encoded residuals (write_tokens_ans)
/// - byte padding
pub fn write_modular_stream_with_tree(
    image: &ModularImage,
    writer: &mut BitWriter,
    profile: &crate::effort::EffortProfile,
    rct: bool,
    use_lz77: bool,
    lz77_method: crate::entropy_coding::lz77::Lz77Method,
) -> Result<()> {
    use super::tree::count_contexts;
    use super::tree_learn::{
        TreeLearningParams, TreeSamples, collect_residuals_with_tree, compute_best_tree,
        compute_gather_stride_from_profile, gather_samples_strided,
    };
    use crate::entropy_coding::encode::build_entropy_code_ans_with_options;
    use crate::entropy_coding::encode::write_entropy_code_ans;
    use crate::entropy_coding::lz77::{apply_lz77, write_lz77_header};

    // Select best RCT variant by trying multiple candidates (effort-dependent).
    // Tree learning uses RCT only (not palette) because palette+tree has a
    // meta-channel encoding mismatch that causes decoder failures.
    let (work_image, rct_type) = if rct && image.channels.len() >= 3 {
        let (selected_rct, transformed) = select_best_rct(image, profile.nb_rcts_to_try);
        (transformed, Some(selected_rct))
    } else {
        (image.clone(), None)
    };

    // Step 0: Find best WP parameters (effort-dependent search)
    let wp_params = if profile.wp_num_param_sets > 0 {
        super::predictor::find_best_wp_params(&work_image.channels, profile.wp_num_param_sets)
    } else {
        super::predictor::WeightedPredictorParams::default()
    };

    // Step 1: Gather samples (with subsampling for large images)
    let total_pixels: usize = work_image
        .channels
        .iter()
        .map(|ch| ch.width() * ch.height())
        .sum();
    let stride = compute_gather_stride_from_profile(total_pixels, profile);
    let mut samples = TreeSamples::new();
    gather_samples_strided(&mut samples, &work_image, 0, 0, stride, &wp_params);

    // Step 2: Learn tree with effort-dependent parameters
    let pixel_fraction = if total_pixels > 0 {
        samples.num_samples as f64 / total_pixels as f64
    } else {
        1.0
    };
    let params = TreeLearningParams::from_profile(profile)
        .with_pixel_fraction(pixel_fraction)
        .with_total_pixels(total_pixels);
    let tree = compute_best_tree(&mut samples, &params);
    let num_contexts = count_contexts(&tree) as usize;

    crate::trace::debug_eprintln!(
        "TREE_LEARN: effort={}, {} props, {} max_buckets, threshold={:.0}*{:.3}={:.1}, \
         {} nodes, {} leaves/contexts, {} samples",
        profile.effort,
        params.properties.len(),
        params.max_property_values,
        params.split_threshold,
        params.pixel_fraction * 0.9 + 0.1,
        params.split_threshold * (params.pixel_fraction * 0.9 + 0.1),
        tree.len(),
        num_contexts,
        samples.num_samples
    );

    // Step 3: Collect residuals with learned tree
    let tokens = collect_residuals_with_tree(&work_image, &tree, 0, &wp_params);

    // Step 3b: Optionally apply LZ77 to the token stream
    let dist_multiplier = work_image
        .channels
        .iter()
        .map(|c| c.width())
        .max()
        .unwrap_or(0) as i32;
    let (tokens, lz77_params) = if use_lz77 {
        // LZ77 application
        match apply_lz77(&tokens, num_contexts, false, lz77_method, dist_multiplier) {
            Some((lz77_tokens, params)) => (lz77_tokens, Some(params)),
            None => (tokens, None),
        }
    } else {
        (tokens, None)
    };
    let ans_num_contexts = if lz77_params.is_some() {
        num_contexts + 1
    } else {
        num_contexts
    };

    // Step 4: Build multi-context ANS code with enhanced clustering
    let code = build_entropy_code_ans_with_options(
        &tokens,
        ans_num_contexts,
        true, // enhanced clustering (pair-merge refinement)
        lz77_params.as_ref(),
        Some(total_pixels),
    );

    // Step 5: Write bitstream
    // dc_quant.all_default = true
    writer.write(1, 1)?;
    // has_tree = true
    writer.write(1, 1)?;

    // Write the learned tree
    write_tree(writer, &tree)?;

    // Write LZ77 header + ANS data histogram.
    if ans_num_contexts > 1 {
        write_lz77_header(lz77_params.as_ref(), writer)?;
        write_entropy_code_ans(&code, writer)?;
    } else {
        use super::section::write_ans_modular_header;
        write_ans_modular_header(writer, &code)?;
    }

    // GroupHeader
    writer.write(1, 1)?; // use_global_tree = true
    write_wp_header(writer, &wp_params)?;

    if let Some(rct_type) = rct_type {
        writer.write(2, 1)?; // num_transforms = 1
        write_rct_transform(writer, 0, rct_type)?;
    } else {
        writer.write(2, 0)?; // num_transforms = 0
    }

    // Debug: verify ANS encoding correctness (skip when LZ77 active — verify doesn't handle LZ77 tokens)
    #[cfg(debug_assertions)]
    if lz77_params.is_none() {
        let roundtrip_result = crate::entropy_coding::encode::verify_ans_roundtrip(&tokens, &code);
        if roundtrip_result.is_err() {
            debug_rect!(
                "ans/verify",
                0,
                0,
                image.width(),
                image.height(),
                "ROUNDTRIP FAILED for tree learning data (ctx={} histo={} tokens={}): {:?}",
                num_contexts,
                code.histograms.len(),
                tokens.len(),
                roundtrip_result
            );
        }
    }

    // Write ANS tokens
    write_tokens_ans(&tokens, &code, lz77_params.as_ref(), writer)?;

    writer.zero_pad_to_byte();
    Ok(())
}

/// Write a single-group modular stream using squeeze + tree learning.
///
/// Combines the Haar wavelet (squeeze) transform with learned MA tree
/// for multi-context ANS encoding. This gives the benefits of both:
/// - Squeeze decorrelates spatial frequencies (better for smooth gradients)
/// - Tree learning adapts prediction and contexts per-channel/per-region
///
/// Pipeline: RCT → squeeze → gather samples → learn tree → collect residuals → ANS
pub fn write_modular_stream_with_squeeze_and_tree(
    image: &ModularImage,
    writer: &mut BitWriter,
    profile: &crate::effort::EffortProfile,
    use_lz77: bool,
    lz77_method: crate::entropy_coding::lz77::Lz77Method,
) -> Result<()> {
    use super::rct::{RctType, forward_rct};
    use super::squeeze::{apply_squeeze, default_squeeze_params};
    use super::tree::count_contexts;
    use super::tree_learn::{
        TreeLearningParams, TreeSamples, collect_residuals_with_tree, compute_best_tree,
        compute_gather_stride_from_profile, gather_samples_strided,
    };
    use crate::entropy_coding::encode::build_entropy_code_ans_with_options;
    use crate::entropy_coding::encode::write_entropy_code_ans;
    use crate::entropy_coding::lz77::{apply_lz77, write_lz77_header};

    let params = default_squeeze_params(image);
    if params.is_empty() {
        // Image too small for squeeze, fall back to tree learning without squeeze
        return write_modular_stream_with_tree(
            image,
            writer,
            profile,
            image.channels.len() >= 3,
            use_lz77,
            lz77_method,
        );
    }

    // Step 1: Apply RCT (YCoCg) before squeeze for RGB images
    let mut transformed = image.clone();
    let has_rct = transformed.channels.len() >= 3;
    if has_rct {
        let rct_type = RctType::YCOCG;
        forward_rct(&mut transformed.channels, 0, rct_type)?;
    }

    // Step 2: Apply forward squeeze
    apply_squeeze(&mut transformed, &params)?;

    crate::trace::debug_eprintln!(
        "SQUEEZE+TREE: {} squeeze steps, {} → {} channels, rct={}",
        params.len(),
        image.channels.len(),
        transformed.channels.len(),
        has_rct,
    );

    // Step 2b: Find best WP parameters (effort-dependent search)
    let wp_params = if profile.wp_num_param_sets > 0 {
        super::predictor::find_best_wp_params(&transformed.channels, profile.wp_num_param_sets)
    } else {
        super::predictor::WeightedPredictorParams::default()
    };

    // Step 3: Gather samples from squeezed image (with subsampling for large images)
    let total_pixels: usize = transformed
        .channels
        .iter()
        .map(|ch| ch.width() * ch.height())
        .sum();
    let stride = compute_gather_stride_from_profile(total_pixels, profile);
    let mut samples = TreeSamples::new();
    gather_samples_strided(&mut samples, &transformed, 0, 0, stride, &wp_params);

    // Step 4: Learn tree with effort-dependent parameters
    let pixel_fraction = if total_pixels > 0 {
        samples.num_samples as f64 / total_pixels as f64
    } else {
        1.0
    };
    let tree_params = TreeLearningParams::from_profile(profile)
        .with_pixel_fraction(pixel_fraction)
        .with_total_pixels(total_pixels);
    let tree = compute_best_tree(&mut samples, &tree_params);
    let num_contexts = count_contexts(&tree) as usize;

    crate::trace::debug_eprintln!(
        "SQUEEZE+TREE: effort={}, {} nodes, {} contexts, {} samples (pf={:.3})",
        profile.effort,
        tree.len(),
        num_contexts,
        samples.num_samples,
        pixel_fraction,
    );

    // Step 5: Collect residuals with learned tree
    let tokens = collect_residuals_with_tree(&transformed, &tree, 0, &wp_params);

    // Step 5b: Optionally apply LZ77 to the token stream
    let dist_multiplier = transformed
        .channels
        .iter()
        .map(|c| c.width())
        .max()
        .unwrap_or(0) as i32;
    let (tokens, lz77_params) = if use_lz77 {
        match apply_lz77(&tokens, num_contexts, false, lz77_method, dist_multiplier) {
            Some((lz77_tokens, params)) => (lz77_tokens, Some(params)),
            None => (tokens, None),
        }
    } else {
        (tokens, None)
    };
    let ans_num_contexts = if lz77_params.is_some() {
        num_contexts + 1
    } else {
        num_contexts
    };

    // Step 6: Build multi-context ANS code with enhanced clustering
    let code = build_entropy_code_ans_with_options(
        &tokens,
        ans_num_contexts,
        true, // enhanced clustering (pair-merge refinement)
        lz77_params.as_ref(),
        Some(total_pixels),
    );

    // Step 7: Write bitstream
    // dc_quant.all_default = true
    writer.write(1, 1)?;
    // has_tree = true
    writer.write(1, 1)?;

    // Write the learned tree
    write_tree(writer, &tree)?;

    // Write LZ77 header + ANS data histogram
    if ans_num_contexts > 1 {
        write_lz77_header(lz77_params.as_ref(), writer)?;
        write_entropy_code_ans(&code, writer)?;
    } else {
        use super::section::write_ans_modular_header;
        write_ans_modular_header(writer, &code)?;
    }

    // GroupHeader with transforms: RCT (if RGB) + Squeeze
    writer.write(1, 1)?; // use_global_tree = true
    write_wp_header(writer, &wp_params)?;

    if has_rct {
        // num_transforms = 2: U32 BitsOffset(4,2), offset=0
        writer.write(2, 2)?;
        writer.write(4, 0)?;
        write_rct_transform(writer, 0, RctType::YCOCG)?;
        write_squeeze_transform(writer, &params)?;
    } else {
        writer.write(2, 1)?; // num_transforms = 1
        write_squeeze_transform(writer, &params)?;
    }

    // Debug: verify ANS encoding correctness (skip when LZ77 active — verify doesn't handle LZ77 tokens)
    #[cfg(debug_assertions)]
    if lz77_params.is_none() {
        let roundtrip_result = crate::entropy_coding::encode::verify_ans_roundtrip(&tokens, &code);
        if roundtrip_result.is_err() {
            debug_rect!(
                "ans/verify",
                0,
                0,
                image.width(),
                image.height(),
                "ROUNDTRIP FAILED for squeeze+tree data (ctx={} histo={} tokens={}): {:?}",
                num_contexts,
                code.histograms.len(),
                tokens.len(),
                roundtrip_result
            );
        }
    }

    // Write ANS tokens
    write_tokens_ans(&tokens, &code, lz77_params.as_ref(), writer)?;

    writer.zero_pad_to_byte();
    Ok(())
}

// ===== Multi-group support =====
// These functions are now in the section module for better organization

pub use super::section::{
    build_histogram_from_residuals, collect_all_residuals, write_global_modular_section,
    write_group_modular_section,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_signed() {
        assert_eq!(pack_signed(0), 0);
        assert_eq!(pack_signed(-1), 1);
        assert_eq!(pack_signed(1), 2);
        assert_eq!(pack_signed(-2), 3);
        assert_eq!(pack_signed(2), 4);
    }

    #[test]
    fn test_predict_gradient() {
        // Smooth gradient: should predict correctly
        assert_eq!(predict_gradient(10, 10, 10), 10);

        // Gradient clamping
        // predict_gradient(left, top, topleft) = clamp(left + top - topleft, min, max)
        // where min = min(left, top), max = max(left, top)
        assert_eq!(predict_gradient(20, 10, 10), 20); // grad = 20, clamped to [10,20]
        assert_eq!(predict_gradient(10, 20, 30), 10); // grad = 0, topleft > max, return min
    }

    #[test]
    fn test_encode_hybrid_uint() {
        assert_eq!(encode_hybrid_uint_000(0), (0, 0, 0));
        assert_eq!(encode_hybrid_uint_000(1), (1, 0, 0));
        assert_eq!(encode_hybrid_uint_000(2), (2, 1, 0));
        assert_eq!(encode_hybrid_uint_000(3), (2, 1, 1));
    }

    #[test]
    fn test_gradient_stream() {
        // 4x4 image
        let data: Vec<u8> = vec![
            100, 101, 102, 103, 101, 102, 103, 104, 102, 103, 104, 105, 103, 104, 105, 106,
        ];
        let image = ModularImage::from_gray8(&data, 4, 4).unwrap();

        let mut writer = BitWriter::new();
        write_simple_modular_stream(&image, &mut writer, false).unwrap();

        let _bytes = writer.finish_with_padding();
        crate::trace::debug_eprintln!("Gradient stream: {} bytes", _bytes.len());
    }

    #[test]
    fn test_rct_stream() {
        // 4x4 RGB image with smooth gradients (good for RCT)
        let mut data = Vec::new();
        for y in 0..4 {
            for x in 0..4 {
                let base = (y * 4 + x) * 10;
                data.push(base as u8); // R
                data.push((base + 5) as u8); // G
                data.push((base + 10) as u8); // B
            }
        }
        let image = ModularImage::from_rgb8(&data, 4, 4).unwrap();

        let mut writer = BitWriter::new();
        write_modular_stream_with_rct(&image, &mut writer, false).unwrap();

        let bytes = writer.finish_with_padding();
        crate::trace::debug_eprintln!("RCT stream: {} bytes", bytes.len());
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_write_rct_transform() {
        use crate::bit_writer::BitWriter;

        let mut writer = BitWriter::new();
        write_rct_transform(&mut writer, 0, RctType::YCOCG).unwrap();

        // YCoCg with begin_c=0 should be:
        // - TransformId=RCT: 2 bits (00)
        // - begin_c=0: 5 bits (00 + 000)
        // - rct_type=6: 2 bits (00)
        // Total: 9 bits
        assert_eq!(writer.bits_written(), 9);
    }

    #[test]
    fn test_rct_type_u32_encoding() {
        use crate::modular::rct::RctType;

        // Verify that write_rct_transform produces correct bit lengths
        // for all rct_type values. The U32 distribution is:
        // U32(Val(6), Bits(2), BitsOffset(4, 2), BitsOffset(6, 10))
        // TransformId (2 bits) + begin_c (5 bits for 0) + rct_type
        let base_bits = 2 + 5; // TransformId + begin_c=0

        for rct_val in 0..42u8 {
            let rct_type = RctType(rct_val);
            let mut writer = BitWriter::new();
            write_rct_transform(&mut writer, 0, rct_type).unwrap();

            let expected_rct_bits = if rct_val == 6 {
                2 // selector only (Val(6))
            } else if rct_val < 2 {
                2 + 2 // selector + Bits(2)
            } else if rct_val < 10 {
                2 + 4 // selector + BitsOffset(4, 2)
            } else {
                2 + 6 // selector + BitsOffset(6, 10)
            };
            let total_expected = base_bits + expected_rct_bits;

            assert_eq!(
                writer.bits_written(),
                total_expected,
                "Wrong bit count for rct_type={}: expected {} bits, got {}",
                rct_val,
                total_expected,
                writer.bits_written()
            );
        }
    }

    #[test]
    fn test_weighted_stream() {
        // 4x4 grayscale image with gradient
        let data: Vec<u8> = vec![
            100, 101, 102, 103, 101, 102, 103, 104, 102, 103, 104, 105, 103, 104, 105, 106,
        ];
        let image = ModularImage::from_gray8(&data, 4, 4).unwrap();

        let mut writer = BitWriter::new();
        write_modular_stream_with_weighted(&image, &mut writer, false).unwrap();

        let bytes = writer.finish_with_padding();
        crate::trace::debug_eprintln!("Weighted stream: {} bytes", bytes.len());
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_rct_weighted_stream() {
        // 4x4 RGB image with smooth gradients
        let mut data = Vec::new();
        for y in 0..4 {
            for x in 0..4 {
                let base = (y * 4 + x) * 10;
                data.push(base as u8);
                data.push((base + 5) as u8);
                data.push((base + 10) as u8);
            }
        }
        let image = ModularImage::from_rgb8(&data, 4, 4).unwrap();

        let mut writer = BitWriter::new();
        write_modular_stream_with_rct_weighted(&image, &mut writer, false).unwrap();

        let bytes = writer.finish_with_padding();
        crate::trace::debug_eprintln!("RCT+Weighted stream: {} bytes", bytes.len());
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_lz77_bit_trace() {
        // Create a 16x16 image that will trigger LZ77
        // Wider rows = runs of 15 zeros per row (>7, so LZ77 triggers)
        let mut data = Vec::new();
        for _ in 0..256 {
            // 16x16 image
            data.push(100u8);
            data.push(100u8);
            data.push(100u8);
        }
        let image = ModularImage::from_rgb8(&data, 16, 16).unwrap();

        crate::trace::debug_eprintln!("\n=== LZ77 BIT TRACE TEST ===");

        let mut writer = BitWriter::new();
        write_improved_modular_stream(&image, &mut writer, false).unwrap();

        let _bytes = writer.finish_with_padding();
        crate::trace::debug_eprintln!("LZ77 stream: {} bytes", _bytes.len());
        crate::trace::debug_eprintln!("Raw bytes: {:02x?}", &_bytes[.._bytes.len().min(50)]);

        // Now let's trace through what the decoder expects:
        crate::trace::debug_eprintln!("\n=== EXPECTED DECODER INTERPRETATION ===");
        crate::trace::debug_eprintln!("Bit 0: dc_quant.all_default = 1");
        crate::trace::debug_eprintln!("Bit 1: has_tree = 1");
        crate::trace::debug_eprintln!("--- TREE HISTOGRAM (6 contexts) ---");
        crate::trace::debug_eprintln!("Bit 2: lz77.enabled = 0");
        crate::trace::debug_eprintln!("Bits 3-5: context_map (is_simple=1, bits_per_entry=0)");
        // ... etc
    }

    #[test]
    fn test_ans_roundtrip_gray() {
        use crate::headers::{ColorEncoding, FileHeader};
        use crate::modular::frame::{FrameEncoder, FrameEncoderOptions};

        let data: Vec<u8> = vec![
            100, 101, 102, 103, 101, 102, 103, 104, 102, 103, 104, 105, 103, 104, 105, 106,
        ];
        let image = ModularImage::from_gray8(&data, 4, 4).unwrap();

        // Build full JXL bitstream with ANS modular
        let mut writer = BitWriter::new();
        let file_header = FileHeader::new_gray(4, 4);
        file_header.write(&mut writer).unwrap();
        writer.zero_pad_to_byte();

        let frame_options = FrameEncoderOptions {
            use_modular: true,
            effort: 7,
            use_ans: true,
            use_tree_learning: false,
            use_squeeze: false,
            ..Default::default()
        };
        let frame_encoder = FrameEncoder::new(4, 4, frame_options);
        let color_encoding = ColorEncoding::srgb();
        frame_encoder
            .encode_modular(&image, &color_encoding, &mut writer)
            .unwrap();

        let bytes = writer.finish_with_padding();
        eprintln!("ANS modular gray 4x4: {} bytes", bytes.len());

        // Decode with jxl-oxide
        let jxl_image = jxl_oxide::JxlImage::builder()
            .read(std::io::Cursor::new(&bytes))
            .unwrap_or_else(|e| panic!("jxl-oxide parse failed: {}", e));

        assert_eq!(jxl_image.width(), 4);
        assert_eq!(jxl_image.height(), 4);

        let render = jxl_image
            .render_frame(0)
            .unwrap_or_else(|e| panic!("jxl-oxide render failed: {}", e));

        let fb = render.image_all_channels();
        let decoded_f32 = fb.buf();
        let decoded: Vec<u8> = decoded_f32
            .iter()
            .map(|&v| (v * 255.0).round().clamp(0.0, 255.0) as u8)
            .collect();

        assert_eq!(
            decoded.len(),
            data.len(),
            "decoded size mismatch: {} vs {}",
            decoded.len(),
            data.len()
        );

        for (i, (&orig, &dec)) in data.iter().zip(decoded.iter()).enumerate() {
            assert_eq!(
                orig, dec,
                "pixel {} differs: orig={} decoded={}",
                i, orig, dec
            );
        }
    }

    #[test]
    fn test_ans_roundtrip_gray_varied() {
        use crate::headers::{ColorEncoding, FileHeader};
        use crate::modular::frame::{FrameEncoder, FrameEncoderOptions};

        let data = vec![0u8, 64, 128, 192, 255, 100, 50, 200];
        let image = ModularImage::from_gray8(&data, 4, 2).unwrap();

        // First write with Huffman to get reference bytes
        {
            let mut writer = BitWriter::new();
            let file_header = FileHeader::new_gray(4, 2);
            file_header.write(&mut writer).unwrap();
            writer.zero_pad_to_byte();

            let frame_options = FrameEncoderOptions {
                use_modular: true,
                effort: 7,
                use_ans: false,
                use_tree_learning: false,
                use_squeeze: false,
                ..Default::default()
            };
            let frame_encoder = FrameEncoder::new(4, 2, frame_options);
            let color_encoding = ColorEncoding::srgb();
            frame_encoder
                .encode_modular(&image, &color_encoding, &mut writer)
                .unwrap();
            let huf_bytes = writer.finish_with_padding();
            eprintln!("Huffman modular gray varied 4x2: {} bytes", huf_bytes.len());
            eprintln!("Huffman bytes: {:02x?}", &huf_bytes);
        }

        // Now write with ANS
        let mut writer = BitWriter::new();
        let file_header = FileHeader::new_gray(4, 2);
        file_header.write(&mut writer).unwrap();
        writer.zero_pad_to_byte();

        let frame_options = FrameEncoderOptions {
            use_modular: true,
            effort: 7,
            use_ans: true,
            use_tree_learning: false,
            use_squeeze: false,
            ..Default::default()
        };
        let frame_encoder = FrameEncoder::new(4, 2, frame_options);
        let color_encoding = ColorEncoding::srgb();
        frame_encoder
            .encode_modular(&image, &color_encoding, &mut writer)
            .unwrap();

        let bytes = writer.finish_with_padding();
        eprintln!("ANS modular gray varied 4x2: {} bytes", bytes.len());
        eprintln!("ANS bytes: {:02x?}", &bytes);

        // Save for external debugging
        std::fs::write(std::env::temp_dir().join("ans_modular_varied.jxl"), &bytes).ok();

        let jxl_image = jxl_oxide::JxlImage::builder()
            .read(std::io::Cursor::new(&bytes))
            .unwrap_or_else(|e| panic!("jxl-oxide parse failed: {}", e));

        let render = jxl_image
            .render_frame(0)
            .unwrap_or_else(|e| panic!("jxl-oxide render failed: {}", e));

        let fb = render.image_all_channels();
        let decoded_f32 = fb.buf();
        let decoded: Vec<u8> = decoded_f32
            .iter()
            .map(|&v| (v * 255.0).round().clamp(0.0, 255.0) as u8)
            .collect();

        for (i, (&orig, &dec)) in data.iter().zip(decoded.iter()).enumerate() {
            assert_eq!(
                orig, dec,
                "pixel {} differs: orig={} decoded={}",
                i, orig, dec
            );
        }
    }

    #[test]
    fn test_ans_roundtrip_rgb_gradient() {
        use crate::headers::{ColorEncoding, FileHeader};
        use crate::modular::frame::{FrameEncoder, FrameEncoderOptions};

        let mut data = vec![0u8; 8 * 8 * 3];
        for y in 0..8 {
            for x in 0..8 {
                let idx = (y * 8 + x) * 3;
                data[idx] = (x * 32) as u8;
                data[idx + 1] = (y * 32) as u8;
                data[idx + 2] = ((x + y) * 16) as u8;
            }
        }
        let image = ModularImage::from_rgb8(&data, 8, 8).unwrap();

        let mut writer = BitWriter::new();
        let file_header = FileHeader::new_rgb(8, 8);
        file_header.write(&mut writer).unwrap();
        writer.zero_pad_to_byte();

        let frame_options = FrameEncoderOptions {
            use_modular: true,
            effort: 7,
            use_ans: true,
            use_tree_learning: false,
            use_squeeze: false,
            ..Default::default()
        };
        let frame_encoder = FrameEncoder::new(8, 8, frame_options);
        let color_encoding = ColorEncoding::srgb();
        frame_encoder
            .encode_modular(&image, &color_encoding, &mut writer)
            .unwrap();

        let bytes = writer.finish_with_padding();
        eprintln!("ANS modular RGB gradient 8x8: {} bytes", bytes.len());

        let jxl_image = jxl_oxide::JxlImage::builder()
            .read(std::io::Cursor::new(&bytes))
            .unwrap_or_else(|e| panic!("jxl-oxide parse failed: {}", e));

        let render = jxl_image
            .render_frame(0)
            .unwrap_or_else(|e| panic!("jxl-oxide render failed: {}", e));

        let fb = render.image_all_channels();
        let decoded_f32 = fb.buf();
        let decoded: Vec<u8> = decoded_f32
            .iter()
            .map(|&v| (v * 255.0).round().clamp(0.0, 255.0) as u8)
            .collect();

        assert_eq!(decoded.len(), data.len());
        let mut max_diff = 0i32;
        for (i, (&orig, &dec)) in data.iter().zip(decoded.iter()).enumerate() {
            let diff = (orig as i32 - dec as i32).abs();
            if diff > max_diff {
                max_diff = diff;
                eprintln!(
                    "pixel {} ch {}: orig={} decoded={} diff={}",
                    i / 3,
                    i % 3,
                    orig,
                    dec,
                    diff
                );
            }
        }
        assert_eq!(max_diff, 0, "lossless roundtrip should have zero diff");
    }

    #[test]
    fn test_ans_simple_stream() {
        let data: Vec<u8> = vec![
            100, 101, 102, 103, 101, 102, 103, 104, 102, 103, 104, 105, 103, 104, 105, 106,
        ];
        let image = ModularImage::from_gray8(&data, 4, 4).unwrap();

        let mut writer = BitWriter::new();
        write_simple_modular_stream(&image, &mut writer, true).unwrap();

        let bytes = writer.finish_with_padding();
        assert!(
            !bytes.is_empty(),
            "ANS stream should produce non-empty output"
        );
    }

    #[test]
    fn test_ans_rct_stream() {
        let mut data = Vec::new();
        for y in 0..4 {
            for x in 0..4 {
                let base = (y * 4 + x) * 10;
                data.push(base as u8);
                data.push((base + 5) as u8);
                data.push((base + 10) as u8);
            }
        }
        let image = ModularImage::from_rgb8(&data, 4, 4).unwrap();

        let mut writer = BitWriter::new();
        write_modular_stream_with_rct(&image, &mut writer, true).unwrap();

        let bytes = writer.finish_with_padding();
        assert!(
            !bytes.is_empty(),
            "ANS RCT stream should produce non-empty output"
        );
    }

    #[test]
    fn test_ans_weighted_stream() {
        let data: Vec<u8> = vec![
            100, 101, 102, 103, 101, 102, 103, 104, 102, 103, 104, 105, 103, 104, 105, 106,
        ];
        let image = ModularImage::from_gray8(&data, 4, 4).unwrap();

        let mut writer = BitWriter::new();
        write_modular_stream_with_weighted(&image, &mut writer, true).unwrap();

        let bytes = writer.finish_with_padding();
        assert!(
            !bytes.is_empty(),
            "ANS weighted stream should produce non-empty output"
        );
    }

    #[test]
    fn test_ans_rct_weighted_stream() {
        let mut data = Vec::new();
        for y in 0..4 {
            for x in 0..4 {
                let base = (y * 4 + x) * 10;
                data.push(base as u8);
                data.push((base + 5) as u8);
                data.push((base + 10) as u8);
            }
        }
        let image = ModularImage::from_rgb8(&data, 4, 4).unwrap();

        let mut writer = BitWriter::new();
        write_modular_stream_with_rct_weighted(&image, &mut writer, true).unwrap();

        let bytes = writer.finish_with_padding();
        assert!(
            !bytes.is_empty(),
            "ANS RCT+weighted stream should produce non-empty output"
        );
    }

    /// Compare ANS vs Huffman file sizes for a lossless encode.
    #[test]
    fn test_ans_vs_huffman_size() {
        use crate::{LosslessConfig, PixelLayout};

        // Create a non-trivial 32x32 RGB image
        let mut data = vec![0u8; 32 * 32 * 3];
        for y in 0..32 {
            for x in 0..32 {
                let idx = (y * 32 + x) * 3;
                data[idx] = ((x * 8 + y * 2) % 256) as u8;
                data[idx + 1] = ((y * 8 + x * 3) % 256) as u8;
                data[idx + 2] = (((x + y) * 5) % 256) as u8;
            }
        }

        // Encode with Huffman
        let huf_encoded = LosslessConfig::new()
            .with_ans(false)
            .encode(&data, 32, 32, PixelLayout::Rgb8)
            .unwrap();

        // Encode with ANS
        let ans_encoded = LosslessConfig::new()
            .with_ans(true)
            .encode(&data, 32, 32, PixelLayout::Rgb8)
            .unwrap();

        eprintln!(
            "32x32 RGB: Huffman={} bytes, ANS={} bytes, savings={:.1}%",
            huf_encoded.len(),
            ans_encoded.len(),
            (1.0 - ans_encoded.len() as f64 / huf_encoded.len() as f64) * 100.0
        );

        // ANS should not be significantly larger than Huffman
        // (for small images the overhead can make ANS larger, but it should be close)
        assert!(
            ans_encoded.len() <= huf_encoded.len() + huf_encoded.len() / 5,
            "ANS should not be >20% larger than Huffman"
        );
    }
}
