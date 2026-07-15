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
use crate::modular::channel::{Channel, ModularImage};
use crate::modular::rct::{RctType, forward_rct};

// Re-export everything from sub-modules so existing import paths work unchanged.
pub(crate) use super::encode_primitives::*;
pub(crate) use super::encode_transforms::*;
pub(crate) use super::encode_tree::*;

/// Write U32-encoded num_transforms value.
///
/// Encoding: U32(Val(0), Val(1), BitsOffset(4,2), BitsOffset(8,18))
pub(crate) fn write_num_transforms(writer: &mut BitWriter, num_transforms: u32) -> Result<()> {
    match num_transforms {
        0 => writer.write(2, 0)?,
        1 => writer.write(2, 1)?,
        2..=17 => {
            writer.write(2, 2)?;
            writer.write(4, (num_transforms - 2) as u64)?;
        }
        _ => {
            writer.write(2, 3)?;
            writer.write(8, (num_transforms - 18) as u64)?;
        }
    }
    Ok(())
}

/// Collect residuals using the given fixed `predictor_id` and identify
/// LZ77 runs. The default `predictor_id == 5` (Gradient) keeps the
/// legacy bytestream that hash-locks are pinned to.
fn collect_residuals_with_prediction_id(image: &ModularImage, predictor_id: u8) -> Vec<Token> {
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

                // Predict using the requested predictor (default: ClampedGradient = 5)
                let prediction = predict_pixel_with_id(channel, x, y, predictor_id);
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
#[allow(dead_code)] // public wrapper retained for API stability;
// internal callers route through `write_improved_modular_stream_with_predictor`
pub fn write_improved_modular_stream(
    image: &ModularImage,
    writer: &mut BitWriter,
    use_ans: bool,
) -> Result<()> {
    write_improved_modular_stream_inner(image, writer, false, use_ans, 5)
}

/// Knob-aware variant of [`write_improved_modular_stream`] that honours
/// [`super::palette::ModularKnobs::modular_predictor`]. Routes
/// `predictor_id == 6` (Weighted) to
/// [`write_modular_stream_with_weighted`] (the simple LZ77 path has no
/// per-channel WP state).
pub(crate) fn write_improved_modular_stream_with_predictor(
    image: &ModularImage,
    writer: &mut BitWriter,
    use_ans: bool,
    predictor_id: u8,
) -> Result<()> {
    if predictor_id == 6 {
        return write_modular_stream_with_weighted(image, writer, use_ans);
    }
    write_improved_modular_stream_inner(image, writer, false, use_ans, predictor_id)
}

fn write_improved_modular_stream_inner(
    image: &ModularImage,
    writer: &mut BitWriter,
    _skip_group_header: bool,
    use_ans: bool,
    predictor_id: u8,
) -> Result<()> {
    // Collect residuals using the requested predictor (5 = Gradient default).
    let tokens = collect_residuals_with_prediction_id(image, predictor_id);

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

    // If no LZ77 runs, fall back to simple encoding (carry predictor through).
    if num_lz77_runs == 0 {
        return write_simple_modular_stream_with_predictor(image, writer, use_ans, predictor_id);
    }

    // === Global section (LfGlobal) ===
    writer.write(1, 1)?; // dc_quant.all_default = true
    writer.write(1, 1)?; // has_tree = true

    // Tree histogram for single-leaf tree with the requested predictor.
    let (tree_depths, tree_codes) = if predictor_id == 5 {
        write_tree_histogram_for_gradient(writer)?
    } else {
        write_tree_histogram_for_predictor(writer, predictor_id)?
    };
    if predictor_id == 5 {
        write_gradient_tree_tokens(writer, &tree_depths, &tree_codes)?;
    } else {
        write_predictor_tree_tokens(writer, &tree_depths, &tree_codes, predictor_id)?;
    }

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

/// Predict a single pixel using a fixed `predictor_id` in `0..=13`.
///
/// Use [`predict_pixel_with_id`] for the no-tree-learning modular paths
/// that honour [`super::palette::ModularKnobs::modular_predictor`].
///
/// `predictor_id == 6` (Weighted) is **not** real weighted prediction —
/// it falls back to clamped gradient, matching
/// [`super::predictor::Predictor::Weighted::predict_from_neighbors`]'s
/// placeholder. Callers that need true Weighted prediction must use
/// [`write_modular_stream_with_weighted`] (which carries
/// [`super::predictor::WeightedPredictorState`] through every pixel).
/// Unknown ids fall back to clamped gradient too.
#[inline]
pub(crate) fn predict_pixel_with_id(
    channel: &Channel,
    x: usize,
    y: usize,
    predictor_id: u8,
) -> i32 {
    use super::predictor::{Neighbors, Predictor};
    // Fast-path the default (gradient) case to keep the
    // `--modular-predictor` flag off the hot loop when unset.
    if predictor_id == 5 || Predictor::from_id(predictor_id).is_none() {
        let left = if x > 0 { channel.get(x - 1, y) } else { 0 };
        let top = if y > 0 { channel.get(x, y - 1) } else { left };
        let topleft = if x > 0 && y > 0 {
            channel.get(x - 1, y - 1)
        } else {
            left
        };
        return predict_gradient(left, top, topleft);
    }
    let neighbors = Neighbors::gather(channel, x, y);
    // Safe: from_id checked above.
    Predictor::from_id(predictor_id)
        .unwrap()
        .predict_from_neighbors(&neighbors)
}

// Set to true to use Zero predictor for debugging
// Try: false = gradient tree path, true = zero tree path (works)
const USE_ZERO_PREDICTOR: bool = false;

/// Resolve the fixed predictor that the no-tree-learning modular paths
/// should use, given a [`super::palette::ModularKnobs`] snapshot.
///
/// Returns `5` (Gradient — the legacy default) when:
/// - `modular_predictor` is `None`,
/// - the requested id is out of range,
/// - the requested id is `14` (libjxl `Best`) or `15` (libjxl `Variable`)
///   — both encoder-only meta-modes that imply tree learning, which the
///   non-tree paths don't perform; the fixed Gradient fallback keeps
///   the byte output stable across these surface-only inputs.
///
/// Otherwise returns the raw `0..=13` id.
#[inline]
pub(crate) fn resolve_fixed_predictor(knobs: &super::palette::ModularKnobs) -> u8 {
    match knobs.modular_predictor {
        None => 5,
        Some(id) if id <= 13 => id,
        Some(_) => 5,
    }
}

/// Same as [`resolve_fixed_predictor`] but additionally folds Weighted
/// (id `6`) down to Gradient (id `5`) for paths that don't carry per-
/// channel [`super::predictor::WeightedPredictorState`].
///
/// Used by the palette and lossy-palette stream writers, the squeeze
/// multi-group path, and the multi-group non-tree-learn path — these
/// all compute residuals via [`predict_pixel_with_id`] which has only a
/// scalar (gradient-equivalent) Weighted placeholder. Emitting tree
/// token `6` while using gradient-equivalent residuals would diverge
/// from the decoder; falling back to `5` keeps the bitstream
/// self-consistent.
///
/// The dedicated weighted writers ([`write_modular_stream_with_weighted`],
/// [`write_modular_stream_with_rct_weighted`]) carry the WP state
/// themselves and so honour the raw id 6 — the simpler writers route to
/// those when [`super::palette::ModularKnobs::modular_predictor`] is `Some(6)`.
#[inline]
pub(crate) fn resolve_fixed_predictor_for_simple_path(knobs: &super::palette::ModularKnobs) -> u8 {
    match resolve_fixed_predictor(knobs) {
        6 => 5,
        id => id,
    }
}

/// Resolve the optional tree-learn force-predictor override.
///
/// Returns `Some(predictor)` when the caller has explicitly asked for a
/// fixed predictor (libjxl `cjxl -P N` / `--modular_predictor`) AND that
/// id maps to a real [`super::predictor::Predictor`] variant in
/// `0..=13`. The tree-learning path can then build a single-leaf tree
/// with this predictor instead of running ID3 — matching libjxl's
/// behaviour when an explicit `-P N` is supplied alongside its tree
/// learner: the leaf predictor is forced to `N` even though the
/// surrounding plumbing (gather, learn, residual collection) is the same
/// shape as `Predictor::Variable`.
///
/// Returns `None` to mean "no override — fall through to the per-leaf
/// ID3 selection". Triggered when:
/// - [`super::palette::ModularKnobs::modular_predictor`] is `None`,
/// - the requested id is `14` (libjxl `Best`) or `15` (libjxl `Variable`)
///   — both encoder-only meta-modes that explicitly want tree-driven
///   per-leaf selection,
/// - the id is out of range (`>= 16`),
/// - the id is `5` (Gradient — the legacy default that hash-locks pin),
///   so the tree-learn default path stays byte-identical when callers
///   pass through `--modular-predictor 5` explicitly.
///
/// Note id 6 (Weighted) IS returned as `Some(Weighted)` — the tree-learn
/// residual collector ([`super::tree_learn::collect_residuals_with_tree`])
/// already carries `WeightedPredictorState` per channel and handles
/// `Predictor::Weighted` leaves natively, so there is no need to fold
/// `6 → 5` the way [`resolve_fixed_predictor_for_simple_path`] does for
/// the no-tree-learn paths.
#[inline]
pub(crate) fn resolve_tree_learn_force_predictor(
    knobs: &super::palette::ModularKnobs,
) -> Option<super::predictor::Predictor> {
    use super::predictor::Predictor;
    match knobs.modular_predictor {
        None => None,
        Some(5) => None, // Default — fall through to ID3 to keep hash-locks green.
        Some(id) => Predictor::from_id(id), // 0..=4, 6..=13 → Some(_); 14/15/>=16 → None
    }
}

/// Resolve the RIGED (Sharma 2018, Resolution-Independent Gradient-aware
/// Edge Detection) override for the tree-learning path.
///
/// Returns `Some(tree)` (a hand-crafted multi-leaf MA tree implementing
/// the RIGED switching rule) when the caller asked for predictor id
/// `14`, an encoder-only meta-mode in libjxl. The tree-learn path can
/// then bypass ID3 and use this fixed-shape tree directly — same
/// integration as [`resolve_tree_learn_force_predictor`] but routed
/// through a multi-leaf tree instead of a single-leaf one.
///
/// Returns `None` for all other ids — including libjxl's other encoder-
/// only meta-mode `15` (`Variable`), which keeps falling through to ID3
/// to match its libjxl semantics ("tree learner picks per-leaf").
///
/// `bit_depth` is the modular channel bit depth ([`super::channel::ModularImage::bit_depth`]),
/// used to scale the RIGED threshold (`T = 44` for ≤ 8-bit, `T = 768`
/// for 16-bit).
///
/// See [`super::tree::riged_tree`] for the tree shape and references.
#[inline]
pub(crate) fn resolve_tree_learn_riged_tree(
    knobs: &super::palette::ModularKnobs,
    bit_depth: u32,
) -> Option<super::tree::Tree> {
    match knobs.modular_predictor {
        Some(14) => Some(super::tree::riged_tree(bit_depth)),
        _ => None,
    }
}

/// Simpler stream without LZ77 but with gradient prediction.
pub fn write_simple_modular_stream(
    image: &ModularImage,
    writer: &mut BitWriter,
    use_ans: bool,
) -> Result<()> {
    write_simple_modular_stream_with_predictor(image, writer, use_ans, 5)
}

/// Knob-aware variant of [`write_simple_modular_stream`] that honours
/// [`super::palette::ModularKnobs::modular_predictor`] (libjxl `cjxl -P`
/// / `--modular_predictor`). Routes `predictor_id == 6` (Weighted) to
/// [`write_modular_stream_with_weighted`] so the on-pixel prediction
/// uses the full [`super::predictor::WeightedPredictorState`] (the
/// simple-stream path otherwise carries no per-channel WP state and
/// would diverge from the decoder).
pub(crate) fn write_simple_modular_stream_with_predictor(
    image: &ModularImage,
    writer: &mut BitWriter,
    use_ans: bool,
    predictor_id: u8,
) -> Result<()> {
    if predictor_id == 6 {
        return write_modular_stream_with_weighted(image, writer, use_ans);
    }

    // Collect residuals using the requested predictor (5 = Gradient default).
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
                    predict_pixel_with_id(channel, x, y, predictor_id)
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
        write_zero_tree_complete(writer)?;
    } else if predictor_id == 5 {
        // Default path — keep the byte-identical helpers so hash-locks
        // stay green when `modular_predictor` is unset.
        let (tree_depths, tree_codes) = write_tree_histogram_for_gradient(writer)?;
        write_gradient_tree_tokens(writer, &tree_depths, &tree_codes)?;
    } else {
        let (tree_depths, tree_codes) = write_tree_histogram_for_predictor(writer, predictor_id)?;
        write_predictor_tree_tokens(writer, &tree_depths, &tree_codes, predictor_id)?;
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
    write_modular_stream_with_palette_knobs(
        image,
        writer,
        use_ans,
        begin_c,
        num_c,
        &super::palette::ModularKnobs::default(),
    )
}

/// Knob-aware variant of [`write_modular_stream_with_palette`].
///
/// Honours [`super::palette::ModularKnobs::palette_colors`] (libjxl
/// `--modular_palette_colors`). When `knobs.palette_colors_or_default()`
/// returns `None` (caller set `palette_colors = Some(0)`), palette
/// detection is disabled and the function falls through to the
/// RCT / simple modular paths exactly as the default `analyze_palette`
/// rejection path would.
pub fn write_modular_stream_with_palette_knobs(
    image: &ModularImage,
    writer: &mut BitWriter,
    use_ans: bool,
    begin_c: usize,
    num_c: usize,
    knobs: &super::palette::ModularKnobs,
) -> Result<()> {
    use super::palette::{analyze_palette, apply_palette};

    // Resolve forced predictor. Fall-through paths route via the full
    // resolver (so id 6 keeps WP semantics), but the in-palette
    // residual emission folds 6 → 5 (this path has no per-channel WP
    // state; see resolve_fixed_predictor_for_simple_path).
    let fallback_predictor_id = resolve_fixed_predictor(knobs);

    let max_colors = match knobs.palette_colors_or_default() {
        Some(n) => n,
        None => {
            // Caller explicitly disabled palette via knob; fall through to
            // the same path `analyze_palette` would take on rejection.
            if image.channels.len() >= 3 {
                return write_modular_stream_with_rct_only_with_predictor(
                    image,
                    writer,
                    use_ans,
                    fallback_predictor_id,
                );
            } else {
                return write_simple_modular_stream_with_predictor(
                    image,
                    writer,
                    use_ans,
                    fallback_predictor_id,
                );
            }
        }
    };
    let analysis = analyze_palette(image, begin_c, num_c, max_colors);

    if !analysis.use_palette {
        // Fallback: use RCT if RGB, otherwise simple.
        // IMPORTANT: call write_modular_stream_with_rct_only (not _with_rct) to avoid
        // re-entering should_use_palette → write_modular_stream_with_palette → infinite recursion.
        if image.channels.len() >= 3 {
            return write_modular_stream_with_rct_only_with_predictor(
                image,
                writer,
                use_ans,
                fallback_predictor_id,
            );
        } else {
            return write_simple_modular_stream_with_predictor(
                image,
                writer,
                use_ans,
                fallback_predictor_id,
            );
        }
    }

    // We're committed to the in-palette residual path now — fold 6 → 5
    // because this path lacks per-channel WP state.
    let predictor_id = resolve_fixed_predictor_for_simple_path(knobs);

    // Apply palette transform
    let mut transformed = image.clone();
    let nb_colors = apply_palette(&mut transformed, begin_c, num_c, &analysis)?;

    crate::trace::debug_eprintln!(
        "PALETTE: {} unique colors, {} channels → palette({}) + index",
        analysis.num_colors,
        num_c,
        nb_colors
    );

    // Collect residuals on the transformed channels with the requested predictor.
    let mut residuals = Vec::new();

    for channel in &transformed.channels {
        let width = channel.width();
        let height = channel.height();

        for y in 0..height {
            for x in 0..width {
                let pixel = channel.get(x, y);

                let prediction = predict_pixel_with_id(channel, x, y, predictor_id);

                let residual = pixel - prediction;
                let packed = pack_signed(residual);

                residuals.push(packed);
            }
        }
    }

    // === Global section ===
    writer.write(1, 1)?; // dc_quant.all_default = true
    writer.write(1, 1)?; // has_tree = true

    let (tree_depths, tree_codes) = if predictor_id == 5 {
        write_tree_histogram_for_gradient(writer)?
    } else {
        write_tree_histogram_for_predictor(writer, predictor_id)?
    };
    if predictor_id == 5 {
        write_gradient_tree_tokens(writer, &tree_depths, &tree_codes)?;
    } else {
        write_predictor_tree_tokens(writer, &tree_depths, &tree_codes, predictor_id)?;
    }

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
#[allow(dead_code)] // public wrapper; internal callers thread budget via the `_budget` variant
pub fn write_modular_stream_with_lossy_palette(
    image: &ModularImage,
    writer: &mut BitWriter,
    use_ans: bool,
    begin_c: usize,
    num_c: usize,
    max_palette_colors: usize,
) -> Result<()> {
    write_modular_stream_with_lossy_palette_budget(
        image,
        writer,
        use_ans,
        begin_c,
        num_c,
        max_palette_colors,
        None,
    )
}

/// `write_modular_stream_with_lossy_palette` with explicit allocation
/// budget. The budget is forwarded to `apply_lossy_palette` so the
/// dim-driven scratch buffers (deltas, quant rows, error diffusion,
/// index channel) are charged against the per-encode cap.
pub(crate) fn write_modular_stream_with_lossy_palette_budget(
    image: &ModularImage,
    writer: &mut BitWriter,
    use_ans: bool,
    begin_c: usize,
    num_c: usize,
    max_palette_colors: usize,
    budget: Option<&alloc::sync::Arc<crate::budget::MemoryBudget>>,
) -> Result<()> {
    write_modular_stream_with_lossy_palette_budget_knobs(
        image,
        writer,
        use_ans,
        begin_c,
        num_c,
        max_palette_colors,
        budget,
        &super::palette::ModularKnobs::default(),
    )
}

/// Knob-aware variant of
/// [`write_modular_stream_with_lossy_palette_budget`]. Honours
/// [`super::palette::ModularKnobs::modular_predictor`] for the residual
/// pass and tree-token emission. The palette transform itself still
/// uses the libjxl two-pass discovery; only the post-palette residual
/// encoding is affected.
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_modular_stream_with_lossy_palette_budget_knobs(
    image: &ModularImage,
    writer: &mut BitWriter,
    use_ans: bool,
    begin_c: usize,
    num_c: usize,
    max_palette_colors: usize,
    budget: Option<&alloc::sync::Arc<crate::budget::MemoryBudget>>,
    knobs: &super::palette::ModularKnobs,
) -> Result<()> {
    use super::palette::apply_lossy_palette_with_budget;

    let fallback_predictor_id = resolve_fixed_predictor(knobs);

    let mut transformed = image.clone();
    let result = apply_lossy_palette_with_budget(
        &mut transformed,
        begin_c,
        num_c,
        max_palette_colors,
        budget,
    );

    let result = match result {
        Some(r) => r,
        None => {
            // Lossy palette not beneficial, fall back to lossless RCT
            // (honouring the predictor override on the way down).
            if image.channels.len() >= 3 {
                return write_modular_stream_with_rct_knobs(image, writer, use_ans, knobs);
            } else {
                return write_simple_modular_stream_with_predictor(
                    image,
                    writer,
                    use_ans,
                    fallback_predictor_id,
                );
            }
        }
    };

    // In-palette residual path lacks per-channel WP state — fold 6 → 5.
    let predictor_id = resolve_fixed_predictor_for_simple_path(knobs);

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

    // Collect residuals using the requested predictor on transformed channels.
    let mut residuals = Vec::new();
    for channel in &transformed.channels {
        let width = channel.width();
        let height = channel.height();
        for y in 0..height {
            for x in 0..width {
                let pixel = channel.get(x, y);
                let prediction = predict_pixel_with_id(channel, x, y, predictor_id);
                let residual = pixel - prediction;
                let packed = pack_signed(residual);
                residuals.push(packed);
            }
        }
    }

    // === Global section ===
    writer.write(1, 1)?; // dc_quant.all_default = true
    writer.write(1, 1)?; // has_tree = true

    let (tree_depths, tree_codes) = if predictor_id == 5 {
        write_tree_histogram_for_gradient(writer)?
    } else {
        write_tree_histogram_for_predictor(writer, predictor_id)?
    };
    if predictor_id == 5 {
        write_gradient_tree_tokens(writer, &tree_depths, &tree_codes)?;
    } else {
        write_predictor_tree_tokens(writer, &tree_depths, &tree_codes, predictor_id)?;
    }

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

    // Collect residuals with Zero prediction on all transformed channels.
    // libjxl forces Predictor::Zero for squeeze residuals (enc_modular.cc:629-633).
    // Squeeze already decorrelates via Haar wavelet; adding prediction doesn't help.
    let mut residuals = Vec::new();
    let mut max_residual: u32 = 0;

    for channel in &transformed.channels {
        let width = channel.width();
        let height = channel.height();

        for y in 0..height {
            for x in 0..width {
                let pixel = channel.get(x, y);
                let packed = pack_signed(pixel);

                residuals.push(packed);
                max_residual = max_residual.max(packed);
            }
        }
    }

    // === Global section ===
    writer.write(1, 1)?; // dc_quant.all_default = true
    writer.write(1, 1)?; // has_tree = true

    // Use Zero predictor tree for squeeze residuals (matching libjxl enc_modular.cc:629-633)
    write_zero_tree_complete(writer)?;

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
    write_modular_stream_with_rct_knobs(
        image,
        writer,
        use_ans,
        &super::palette::ModularKnobs::default(),
    )
}

/// Knob-aware variant of [`write_modular_stream_with_rct`].
///
/// Threads [`super::palette::ModularKnobs`] into the palette short-circuit
/// (and any nested palette write) so CLI overrides
/// (`--modular_palette_colors=0` disables palette detection,
/// `--modular_palette_colors=N` caps the search) take effect on the RCT
/// entry point too.
pub fn write_modular_stream_with_rct_knobs(
    image: &ModularImage,
    writer: &mut BitWriter,
    use_ans: bool,
    knobs: &super::palette::ModularKnobs,
) -> Result<()> {
    // Check if palette is more beneficial than RCT (respecting the
    // `palette_colors` override — `Some(0)` skips the check entirely).
    if knobs.palette_colors_or_default().is_some()
        && let Some((begin_c, num_c)) = super::palette::should_use_palette(image)
    {
        return write_modular_stream_with_palette_knobs(
            image, writer, use_ans, begin_c, num_c, knobs,
        );
    }

    let predictor_id = resolve_fixed_predictor(knobs);
    write_modular_stream_with_rct_only_with_predictor(image, writer, use_ans, predictor_id)
}

/// Write modular stream with RCT (YCoCg), without checking palette.
/// Used as a fallback from `write_modular_stream_with_palette` to avoid
/// infinite recursion through `should_use_palette`. Honours
/// [`super::palette::ModularKnobs::modular_predictor`] when called via
/// [`write_modular_stream_with_rct_knobs`]. Routes `predictor_id == 6`
/// (Weighted) to [`write_modular_stream_with_rct_weighted`].
fn write_modular_stream_with_rct_only_with_predictor(
    image: &ModularImage,
    writer: &mut BitWriter,
    use_ans: bool,
    predictor_id: u8,
) -> Result<()> {
    if predictor_id == 6 {
        return write_modular_stream_with_rct_weighted(image, writer, use_ans);
    }

    // Only apply RCT to RGB images (3+ channels)
    if image.channels.len() < 3 {
        return write_simple_modular_stream_with_predictor(image, writer, use_ans, predictor_id);
    }

    // Clone the image and apply forward RCT (YCoCg)
    let mut transformed = image.clone();
    let rct_type = RctType::YCOCG;
    forward_rct(&mut transformed.channels, 0, rct_type)?;

    crate::trace::debug_eprintln!(
        "RCT: Applied YCoCg transform to {} channels",
        transformed.channels.len()
    );

    // Collect residuals using the requested predictor on transformed channels.
    let mut residuals = Vec::new();
    let mut max_residual: u32 = 0;

    for channel in &transformed.channels {
        let width = channel.width();
        let height = channel.height();

        for y in 0..height {
            for x in 0..width {
                let pixel = channel.get(x, y);

                let prediction = predict_pixel_with_id(channel, x, y, predictor_id);

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

    let (tree_depths, tree_codes) = if predictor_id == 5 {
        write_tree_histogram_for_gradient(writer)?
    } else {
        write_tree_histogram_for_predictor(writer, predictor_id)?
    };
    if predictor_id == 5 {
        write_gradient_tree_tokens(writer, &tree_depths, &tree_codes)?;
    } else {
        write_predictor_tree_tokens(writer, &tree_depths, &tree_codes, predictor_id)?;
    }

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
    write_modular_stream_with_weighted_with_budget(image, writer, use_ans, None)
}

/// `write_modular_stream_with_weighted` with explicit allocation budget.
///
/// Per-channel `WeightedPredictorState` scratch is reserved against the
/// cap. `budget = None` is zero-overhead.
pub(crate) fn write_modular_stream_with_weighted_with_budget(
    image: &ModularImage,
    writer: &mut BitWriter,
    use_ans: bool,
    budget: Option<&alloc::sync::Arc<crate::budget::MemoryBudget>>,
) -> Result<()> {
    use super::predictor::{Neighbors, WeightedPredictorParams, WeightedPredictorState};

    let params = WeightedPredictorParams::default();

    // Collect residuals with weighted prediction
    let mut residuals = Vec::new();

    for channel in &image.channels {
        let width = channel.width();
        let height = channel.height();
        let mut wp_state = WeightedPredictorState::new_with_budget(&params, width, budget)?;

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
    write_modular_stream_with_rct_weighted_with_budget(image, writer, use_ans, None)
}

/// `write_modular_stream_with_rct_weighted` with explicit allocation budget.
///
/// Per-channel `WeightedPredictorState` scratch is reserved against the
/// cap. `budget = None` is zero-overhead.
pub(crate) fn write_modular_stream_with_rct_weighted_with_budget(
    image: &ModularImage,
    writer: &mut BitWriter,
    use_ans: bool,
    budget: Option<&alloc::sync::Arc<crate::budget::MemoryBudget>>,
) -> Result<()> {
    use super::predictor::{Neighbors, WeightedPredictorParams, WeightedPredictorState};

    // Check if palette is more beneficial
    if let Some((begin_c, num_c)) = super::palette::should_use_palette(image) {
        return write_modular_stream_with_palette(image, writer, use_ans, begin_c, num_c);
    }

    if image.channels.len() < 3 {
        return write_modular_stream_with_weighted_with_budget(image, writer, use_ans, budget);
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
        let mut wp_state = WeightedPredictorState::new_with_budget(&params, width, budget)?;

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

    // Context LUT (perf: this fn runs once per RCT trial — 7x per image
    // at e7 — and the cutoff scan was a 17-compare loop per pixel,
    // 11.4 % of CPU on terminal post-radix). ctx = number of cutoffs
    // strictly greater than max_diff; constant for max_diff >= 500
    // (= 0), so a 501-entry table indexed by min(max_diff, 500)
    // reproduces the scan exactly — byte-identical.
    let mut ctx_lut = [0u8; 501];
    for (d, slot) in ctx_lut.iter_mut().enumerate() {
        *slot = cutoffs.iter().filter(|&&c| (d as u32) < c).count() as u8;
    }

    let mut total_bits: f64 = 0.0;
    let mut extra_bits: u64 = 0;
    let mut histograms: Vec<Vec<u32>> = vec![vec![]; nc];

    // Per-pixel core, shared by the edge and interior paths. Identical
    // math to the original scalar loop; only neighbor SELECTION moved to
    // the callers (hoisted rows, no per-pixel bounds branches inside).
    let tally = |val: i32,
                 left: i32,
                 top: i32,
                 topleft: i32,
                 histograms: &mut Vec<Vec<u32>>,
                 extra_bits: &mut u64| {
        let max_diff = (left.max(top).max(topleft) - left.min(top).min(topleft)) as u32;
        let ctx = ctx_lut[(max_diff as usize).min(500)] as usize;

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
        *extra_bits += nbits as u64;
    };

    for ch in &image.channels {
        let w = ch.width();
        let h = ch.height();
        if w == 0 || h == 0 {
            continue;
        }

        let data = ch.data();

        // Row y == 0: left = previous pixel (0 at x == 0), top = left,
        // topleft = left — matches the original per-pixel fallbacks.
        {
            let row = &data[..w];
            tally(row[0], 0, 0, 0, &mut histograms, &mut extra_bits);
            for x in 1..w {
                let left = row[x - 1];
                tally(row[x], left, left, left, &mut histograms, &mut extra_bits);
            }
        }

        for y in 1..h {
            let row = &data[y * w..y * w + w];
            let prev = &data[(y - 1) * w..y * w];
            // x == 0: left = top = prev[0], topleft = left (original
            // fallbacks: left -> N, top -> N, topleft -> left).
            tally(
                row[0],
                prev[0],
                prev[0],
                prev[0],
                &mut histograms,
                &mut extra_bits,
            );
            for x in 1..w {
                tally(
                    row[x],
                    row[x - 1],
                    prev[x],
                    prev[x - 1],
                    &mut histograms,
                    &mut extra_bits,
                );
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
                        total_bits -= count as f64 * jxl_simd::fast_log2f(p as f32) as f64;
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
///
/// When `forced_rct` is `Some`, skips the search entirely and applies the
/// caller-supplied RCT directly. Mirrors libjxl's `cparams.colorspace`.
pub(crate) fn select_best_rct(
    image: &ModularImage,
    nb_rcts_to_try: u8,
    forced_rct: Option<RctType>,
) -> (RctType, ModularImage) {
    use super::rct::{RctType, forward_rct};

    if let Some(rct_type) = forced_rct {
        let mut transformed = image.clone();
        forward_rct(&mut transformed.channels, 0, rct_type).ok();
        return (rct_type, transformed);
    }

    let nb_rcts_to_try = nb_rcts_to_try as usize;

    if nb_rcts_to_try == 0 || image.channels.len() < 3 {
        // Default to RCT-10 (GBR + Subtract-Green): -1.19% bytes vs YCoCg on a
        // diverse 490-image corpus (see RctType::GBR_SUBGR doc + commit 287d915).
        let mut transformed = image.clone();
        forward_rct(&mut transformed.channels, 0, RctType::GBR_SUBGR).ok();
        return (RctType::GBR_SUBGR, transformed);
    }

    let num_to_try = nb_rcts_to_try.min(RCT_CANDIDATES.len());

    // Each RCT trial is independent: clone the image, forward_rct (mutates
    // the clone only), estimate_cost (pure read). Fan out across trials,
    // then reduce in trial order to preserve the deterministic tie-break
    // ("first wins" on equal cost, matching the previous serial logic).
    let trial_results: Vec<Option<(f64, ModularImage)>> =
        crate::parallel::parallel_map(num_to_try, |i| {
            let rct_val = RCT_CANDIDATES[i];
            let rct_type = RctType(rct_val);

            if rct_type.is_noop() {
                let cost = estimate_cost(image);
                Some((cost, image.clone()))
            } else {
                let mut transformed = image.clone();
                if forward_rct(&mut transformed.channels, 0, rct_type).is_ok() {
                    let cost = estimate_cost(&transformed);
                    Some((cost, transformed))
                } else {
                    None
                }
            }
        });

    let mut best_cost = f64::MAX;
    // Fallback (all-trials-failed) RCT matches the nb_rcts_to_try=0 fallback.
    let mut best_rct = RctType::GBR_SUBGR;
    let mut best_image = None;
    for (i, result) in trial_results.into_iter().enumerate() {
        let Some((cost, transformed)) = result else {
            continue;
        };
        let rct_val = RCT_CANDIDATES[i];
        let rct_type = RctType(rct_val);
        crate::trace::debug_eprintln!("  RCT {:2}: cost={:.0}", rct_val, cost);
        // Strict `<` preserves "first wins" tie-break since we iterate
        // candidates in their original order.
        if cost < best_cost {
            best_cost = cost;
            best_rct = rct_type;
            best_image = Some(transformed);
        }
    }

    let work_image = best_image.unwrap_or_else(|| {
        let mut t = image.clone();
        forward_rct(&mut t.channels, 0, RctType::GBR_SUBGR).ok();
        t
    });

    crate::trace::debug_eprintln!(
        "RCT_SELECT: best={} (cost={:.0}), tried {} variants",
        best_rct.0,
        best_cost,
        num_to_try,
    );

    (best_rct, work_image)
}

/// Like [`select_best_rct`] but applies RCT starting at `begin_c` instead of 0.
///
/// Used after ChannelCompact inserts meta channels at the front, so the color
/// channels to decorrelate are at `begin_c..begin_c+3`.
pub(crate) fn select_best_rct_at(
    image: &ModularImage,
    begin_c: usize,
    nb_rcts_to_try: u8,
    forced_rct: Option<RctType>,
) -> (RctType, ModularImage) {
    use super::rct::{RctType, forward_rct};

    if let Some(rct_type) = forced_rct {
        let mut transformed = image.clone();
        forward_rct(&mut transformed.channels, begin_c, rct_type).ok();
        return (rct_type, transformed);
    }

    let nb_rcts_to_try = nb_rcts_to_try as usize;

    if nb_rcts_to_try == 0 || image.channels.len() < begin_c + 3 {
        // Default to RCT-10 (GBR + Subtract-Green): -1.19% bytes vs YCoCg on a
        // diverse 490-image corpus (see RctType::GBR_SUBGR doc + commit 287d915).
        let mut transformed = image.clone();
        forward_rct(&mut transformed.channels, begin_c, RctType::GBR_SUBGR).ok();
        return (RctType::GBR_SUBGR, transformed);
    }

    let num_to_try = nb_rcts_to_try.min(RCT_CANDIDATES.len());

    // Each RCT trial is independent (clone + forward_rct + estimate_cost).
    // Fan out across trials and reduce in trial order to keep the
    // deterministic "first wins" tie-break on equal cost.
    let trial_results: Vec<Option<(f64, ModularImage)>> =
        crate::parallel::parallel_map(num_to_try, |i| {
            let rct_val = RCT_CANDIDATES[i];
            let rct_type = RctType(rct_val);

            if rct_type.is_noop() {
                let cost = estimate_cost(image);
                Some((cost, image.clone()))
            } else {
                let mut transformed = image.clone();
                if forward_rct(&mut transformed.channels, begin_c, rct_type).is_ok() {
                    let cost = estimate_cost(&transformed);
                    Some((cost, transformed))
                } else {
                    None
                }
            }
        });

    let mut best_cost = f64::MAX;
    // Fallback (all-trials-failed) RCT matches the nb_rcts_to_try=0 fallback.
    let mut best_rct = RctType::GBR_SUBGR;
    let mut best_image = None;
    for (i, result) in trial_results.into_iter().enumerate() {
        let Some((cost, transformed)) = result else {
            continue;
        };
        let rct_val = RCT_CANDIDATES[i];
        let rct_type = RctType(rct_val);
        crate::trace::debug_eprintln!(
            "  RCT {:2} (begin_c={}): cost={:.0}",
            rct_val,
            begin_c,
            cost
        );
        if cost < best_cost {
            best_cost = cost;
            best_rct = rct_type;
            best_image = Some(transformed);
        }
    }

    let work_image = best_image.unwrap_or_else(|| {
        let mut t = image.clone();
        forward_rct(&mut t.channels, begin_c, RctType::GBR_SUBGR).ok();
        t
    });

    crate::trace::debug_eprintln!(
        "RCT_SELECT: best={} (cost={:.0}), tried {} variants, begin_c={}",
        best_rct.0,
        best_cost,
        num_to_try,
        begin_c,
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
/// - GroupHeader (use_global_tree=1, wp_header, num_transforms=0..2)
/// - ANS-encoded residuals (write_tokens_ans)
/// - byte padding
#[allow(clippy::too_many_arguments)]
pub fn write_modular_stream_with_tree(
    image: &ModularImage,
    writer: &mut BitWriter,
    profile: &crate::effort::EffortProfile,
    rct: bool,
    use_lz77: bool,
    lz77_method: crate::entropy_coding::lz77::Lz77Method,
    budget: Option<&alloc::sync::Arc<crate::budget::MemoryBudget>>,
) -> Result<()> {
    write_modular_stream_with_tree_dc_quant(
        image,
        writer,
        profile,
        rct,
        use_lz77,
        lz77_method,
        None,
        None, // no lossy modular options
        true, // enable palette detection
        budget,
    )
}

/// Knob-aware variant of [`write_modular_stream_with_tree`].
///
/// Threads [`super::palette::ModularKnobs`] (libjxl `cjxl -P`,
/// `--modular_palette_colors`, `--modular_channel_colors_global_percent`,
/// `-E N`) into the tree-learning + palette + channel-compact pipeline.
/// `None` knobs (the default) gives byte-identical output to
/// [`write_modular_stream_with_tree`].
#[allow(clippy::too_many_arguments)]
pub fn write_modular_stream_with_tree_knobs(
    image: &ModularImage,
    writer: &mut BitWriter,
    profile: &crate::effort::EffortProfile,
    rct: bool,
    use_lz77: bool,
    lz77_method: crate::entropy_coding::lz77::Lz77Method,
    knobs: &super::palette::ModularKnobs,
    budget: Option<&alloc::sync::Arc<crate::budget::MemoryBudget>>,
) -> Result<()> {
    write_modular_stream_with_tree_dc_quant_knobs(
        image,
        writer,
        profile,
        rct,
        use_lz77,
        lz77_method,
        None,
        None, // no lossy modular options
        true, // enable palette detection
        knobs,
        budget,
    )
}

/// Options for lossy modular encoding (Squeeze + quantization + tree leaf multipliers).
///
/// When enabled, the encoder:
/// 1. Always applies Squeeze transform
/// 2. Computes per-channel quantizers from distance and XYB qtables
/// 3. Pre-quantizes channel pixels to nearest multiples of q
/// 4. Forces tree splits at channel boundaries so each leaf gets its multiplier
/// 5. Divides residuals by multiplier (decoder reconstructs via multiplication)
/// 6. Forces Zero predictor (guarantees residual divisibility invariant)
#[derive(Debug, Clone, Copy)]
pub struct LossyModularOptions {
    /// Butteraugli distance for quantizer computation.
    pub distance: f32,
}

/// Like [`write_modular_stream_with_tree`] but with a custom dc_quant for LfFrame support
/// and optional lossy modular quantization.
///
/// When `dc_quant_custom` is `Some([x, y, b])`, writes custom DC quantization factors
/// instead of `all_default=true`. Used by the LfFrame encoder to embed distance-scaled
/// DC quant values in the modular frame's LfGlobal section.
///
/// When `lossy_options` is `Some(...)`, enables the full lossy modular pipeline:
/// Squeeze + pre-quantization + forced tree splits with multipliers + Zero predictor.
/// This replaces the `use_squeeze: bool` parameter for the responsive path.
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_modular_stream_with_tree_dc_quant(
    image: &ModularImage,
    writer: &mut BitWriter,
    profile: &crate::effort::EffortProfile,
    rct: bool,
    use_lz77: bool,
    lz77_method: crate::entropy_coding::lz77::Lz77Method,
    dc_quant_custom: Option<[f32; 3]>,
    lossy_options: Option<LossyModularOptions>,
    palette: bool,
    budget: Option<&alloc::sync::Arc<crate::budget::MemoryBudget>>,
) -> Result<()> {
    write_modular_stream_with_tree_dc_quant_knobs(
        image,
        writer,
        profile,
        rct,
        use_lz77,
        lz77_method,
        dc_quant_custom,
        lossy_options,
        palette,
        &super::palette::ModularKnobs::default(),
        budget,
    )
}

/// Minimum gather stride below which stride-aliasing cannot occur. A
/// randomized re-gather with a `[1, 2·stride−1]` window is too narrow to
/// de-alias at stride `< 8`, and small strides already sample densely (so the
/// fixed grid can't systematically miss a periodic structure).
const TREE_SELF_REPAIR_MIN_STRIDE: usize = 8;
/// Minimum learned-tree node count before a self-repair is worth attempting.
/// This is only a *cost gate* (skip the second tree-learn on trivially small
/// trees), NOT an aliasing discriminator — measurement (2026-07-15) showed
/// node count does NOT separate aliased docs from detailed photos (5336 641
/// nodes vs a photo's 4147; both over-split). The aliasing verdict is made by
/// real coded cost in [`tree_self_repair_keep_by_cost`], not by size.
const TREE_SELF_REPAIR_MIN_NODES: usize = 256;
/// Switch to the de-aliased tree only when it is estimated to code the *actual
/// residuals* at least this much cheaper than the fixed-stride tree. A 2 %
/// confidence moat (mirrors `EffortProfile::cfl_keep_best`'s
/// `CFL_KEEP_BEST_MARGIN`) keeps non-aliased content — where the two trees
/// cost within noise — on the fixed-stride tree ⇒ byte-identical, and fires
/// only on the clear aliasing win (5336: `cost_r/cost_a` ≈ 0.73).
const TREE_SELF_REPAIR_COST_MARGIN: f64 = 0.98;
/// Aliasing pre-filter threshold: the ratio of a tree's REAL clustered ANS cost
/// to its IDEAL per-context entropy. A stride-aliased tree spreads samples
/// across many thin contexts that collapse badly under the ≤96-histogram
/// clustering, so its clustered cost runs far above its ideal entropy
/// (measured 2026-07-15: 5336 ratio 2.00, 5334 1.30). Non-aliased content
/// clusters almost for free (photos 1.00, flat docs 1.02–1.17). Only when the
/// ratio exceeds this do we pay for the de-aliased re-gather; below it the
/// fixed-stride tree is kept without the second tree-learn (bounds the
/// wall-time cost to genuinely-suspect content). Set low (5 %) so a
/// borderline-aliased doc is never skipped — a false positive only wastes the
/// second pass (real cost still keeps the fixed tree), a false negative would
/// miss a fix.
const TREE_SELF_REPAIR_ALIASING_RATIO: f64 = 1.05;

/// `JXL_TREE_SELF_REPAIR=1` opts into the content-agnostic aliased-tree
/// self-repair (default OFF during validation ⇒ byte-identical). The repair
/// learns the tree twice — once from the fixed-stride sample, once from a
/// de-aliased randomized re-gather — and keeps whichever codes the real
/// residuals cheaper (see [`tree_self_repair_should_try`] /
/// [`tree_self_repair_keep_by_cost`]).
#[cfg(feature = "std")]
fn tree_self_repair_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("JXL_TREE_SELF_REPAIR").is_some())
}
#[cfg(not(feature = "std"))]
fn tree_self_repair_enabled() -> bool {
    false
}

/// Should we attempt an aliased-tree self-repair? True only when the feature
/// is enabled AND the gather stride is large enough for aliasing to occur AND
/// the fixed-stride tree is non-trivial (the node floor is a cost gate, not an
/// aliasing test — see [`TREE_SELF_REPAIR_MIN_NODES`]). Shared by every
/// tree-learning site so all paths use identical thresholds.
pub(crate) fn tree_self_repair_should_try(stride: usize, tree_a_nodes: usize) -> bool {
    tree_self_repair_enabled()
        && stride >= TREE_SELF_REPAIR_MIN_STRIDE
        && tree_a_nodes >= TREE_SELF_REPAIR_MIN_NODES
}

/// Keep the de-aliased tree only when it codes the actual residuals materially
/// cheaper than the fixed-stride tree (by the [`TREE_SELF_REPAIR_COST_MARGIN`]
/// moat). `cost_*` are `estimate_token_cost` bit estimates over the collected
/// residuals — the same estimator the multi-seed picker trusts. Because the
/// cheaper tree is kept, the repair can only shrink the output (never a byte
/// regression); on non-aliased content the two costs are within noise so the
/// fixed-stride tree is kept ⇒ byte-identical.
pub(crate) fn tree_self_repair_keep_by_cost(cost_r: f64, cost_a: f64) -> bool {
    cost_r < cost_a * TREE_SELF_REPAIR_COST_MARGIN
}

/// Cheap aliasing pre-filter: does the fixed-stride tree's real clustered ANS
/// cost run far enough above its ideal per-context entropy to suspect
/// stride-aliasing (and thus be worth a de-aliased re-learn)? See
/// [`TREE_SELF_REPAIR_ALIASING_RATIO`]. Keeps the wall-time cost of the
/// self-repair off non-aliased content (photos, flat docs), whose clustered
/// and ideal costs are within a few percent.
pub(crate) fn tree_self_repair_ratio_flags_aliasing(clustered_cost: f64, ideal_cost: f64) -> bool {
    clustered_cost > ideal_cost.max(1.0) * TREE_SELF_REPAIR_ALIASING_RATIO
}

/// Knob-aware variant of [`write_modular_stream_with_tree_dc_quant`].
///
/// Honours [`super::palette::ModularKnobs::palette_colors`] (caps the
/// multi-channel palette colour search and disables it entirely when
/// `Some(0)`), `channel_colors_global_percent` (overrides the global /
/// single-group channel-compact threshold, libjxl
/// `enc_params.h:channel_colors_pre_transform_percent`), and
/// `nb_prev_channels` (caps `max_ref_channels` for tree-learner reference
/// properties, libjxl `options.max_properties`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_modular_stream_with_tree_dc_quant_knobs(
    image: &ModularImage,
    writer: &mut BitWriter,
    profile: &crate::effort::EffortProfile,
    rct: bool,
    use_lz77: bool,
    lz77_method: crate::entropy_coding::lz77::Lz77Method,
    dc_quant_custom: Option<[f32; 3]>,
    lossy_options: Option<LossyModularOptions>,
    palette: bool,
    knobs: &super::palette::ModularKnobs,
    budget: Option<&alloc::sync::Arc<crate::budget::MemoryBudget>>,
) -> Result<()> {
    use super::tree::count_contexts;
    use super::tree_learn::{
        TreeLearningParams, TreeSamples, collect_residuals_with_tree, compute_best_tree,
        compute_best_tree_with_multipliers, compute_gather_stride_from_profile,
        gather_samples_strided, gather_samples_strided_with_dedup_backend, max_ref_channels,
    };
    use crate::entropy_coding::encode::build_entropy_code_ans_with_options;
    use crate::entropy_coding::encode::write_entropy_code_ans;
    use crate::entropy_coding::lz77::{apply_lz77, write_lz77_header};

    let is_lossy = lossy_options.is_some();

    // Check if multi-channel palette is beneficial (only for lossless, non-lossy images).
    // When palette is active, it replaces RCT for the color channels — palette
    // already decorrelates them by converting to a single index channel.
    //
    // `knobs.palette_colors_or_default()` returns `None` when the caller
    // disabled palette via `--modular_palette_colors=0` — in that case
    // we skip detection entirely. Otherwise it returns the cap (default
    // `MAX_PALETTE_COLORS`, or the caller-supplied override).
    let palette_info = if let Some(max_colors) = knobs.palette_colors_or_default() {
        if palette && !is_lossy && image.channels.len() >= 2 {
            if let Some((begin_c, num_c)) = super::palette::should_use_palette(image) {
                let analysis = super::palette::analyze_palette(image, begin_c, num_c, max_colors);
                if analysis.use_palette {
                    Some((begin_c, num_c, analysis))
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    };

    // ChannelCompact: per-channel value compaction for sparse channels.
    // Applied when multi-channel palette didn't fire (too many unique RGB colors,
    // but individual channels may be sparse — common in screenshots).
    // Matches libjxl enc_modular.cc:395-438 with channel_colors_pre_transform_percent=95.
    //
    // Honours `--modular_channel_colors_global_percent` override; falls
    // back to `CHANNEL_COLORS_PERCENT` (95.0).
    let channel_colors_percent = knobs.channel_colors_global_percent_or_default();
    let compact_analyses: Vec<(usize, super::palette::PaletteAnalysis)> = if palette_info.is_none()
        && !is_lossy
        && palette
        && image.channels.len() >= 2
    {
        // Color channels = base color set (1 gray or 3 RGB).
        // Everything beyond is extra (alpha, depth, spot, …) and
        // ChannelCompact must skip them. The previous formula
        // `len - 1 if has_alpha` was wrong for >1 extras (refs #9):
        // it would treat the spot/depth/etc as a color channel and
        // try to compact it alongside RGB, blowing the layout.
        let num_color_channels = if image.is_grayscale { 1 } else { 3 };
        (0..num_color_channels.min(image.channels.len()))
            .filter_map(|i| {
                super::palette::analyze_channel_compact(&image.channels[i], channel_colors_percent)
                    .map(|a| (i, a))
            })
            .collect()
    } else {
        Vec::new()
    };

    // Apply transforms: multi-channel palette, ChannelCompact + RCT, or RCT only.
    // compact_info tracks ChannelCompact transforms for the bitstream:
    // Vec of (begin_c_in_bitstream, nb_colors).
    let (work_image, rct_type, palette_result, compact_info) = if let Some((
        begin_c,
        num_c,
        ref analysis,
    )) = palette_info
    {
        // Multi-channel palette path: replaces RCT entirely
        let mut palettized = image.clone();
        let nb_colors = super::palette::apply_palette(&mut palettized, begin_c, num_c, analysis)?;
        crate::trace::debug_eprintln!(
            "PALETTE+TREE: {} unique colors, {} channels palettized, begin_c={}",
            nb_colors,
            num_c,
            begin_c,
        );
        (
            palettized,
            None,
            Some((begin_c, num_c, nb_colors)),
            Vec::new(),
        )
    } else if !compact_analyses.is_empty() {
        // ChannelCompact path: compact individual channels, then apply RCT.
        // Channel ordering must match decoder's MetaPalette expectations:
        // palettes are inserted at position 0 (front), so after N compacts
        // the layout is [pal_N-1, ..., pal_0, idx_0, ch_1, idx_2, ...].
        let num_compacted = compact_analyses.len();
        let mut palettes: Vec<Channel> = Vec::new();
        let mut non_meta: Vec<Channel> = Vec::new();
        let mut info: Vec<(usize, usize)> = Vec::new();
        let mut nb_meta = 0usize;

        for (orig_idx, ch) in image.channels.iter().enumerate() {
            if let Some((_, analysis)) = compact_analyses.iter().find(|(idx, _)| *idx == orig_idx) {
                // Create palette meta-channel (nb_colors wide, 1 high for num_c=1)
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

                // Compute begin_c for the bitstream transform descriptor.
                // Each prior compact added one meta channel, shifting begin_c.
                let begin_c = orig_idx + nb_meta;
                info.push((begin_c, analysis.num_colors));
                nb_meta += 1;
            } else {
                // Non-compacted channel passes through unchanged
                non_meta.push(ch.clone());
            }
        }

        // Decoder's MetaPalette inserts each palette at position 0,
        // so earlier palettes end up deeper. Reverse to match.
        palettes.reverse();

        let mut work = image.clone();
        work.channels = palettes;
        work.channels.extend(non_meta);

        crate::trace::debug_eprintln!(
            "CHANNEL_COMPACT+TREE: {} channels compacted, {} meta + {} non-meta channels, info={:?}",
            num_compacted,
            nb_meta,
            work.channels.len() - nb_meta,
            info,
        );

        // Apply RCT to the non-meta channels (starting at nb_meta)
        let rct_begin_c = num_compacted;
        if rct && work.channels.len() >= rct_begin_c + 3 {
            let (selected_rct, transformed) = select_best_rct_at(
                &work,
                rct_begin_c,
                profile.nb_rcts_to_try,
                profile.forced_rct,
            );
            (transformed, Some(selected_rct), None, info)
        } else {
            (work, None, None, info)
        }
    } else if !is_lossy && rct && image.channels.len() >= 3 {
        // RCT only path (no palette, no ChannelCompact)
        let (selected_rct, transformed) =
            select_best_rct(image, profile.nb_rcts_to_try, profile.forced_rct);
        (transformed, Some(selected_rct), None, Vec::new())
    } else {
        (image.clone(), None, None, Vec::new())
    };

    // Apply Squeeze (Haar wavelet) for spatial decorrelation.
    // For lossy modular, Squeeze is always applied (matching libjxl responsive=1).
    // For lossless, it's never applied here (separate squeeze+tree path exists).
    let squeeze_params = if is_lossy {
        use super::squeeze::default_squeeze_params;
        let params = default_squeeze_params(&work_image);
        if !params.is_empty() {
            Some(params)
        } else {
            None
        }
    } else {
        None
    };
    let mut work_image = work_image;
    if let Some(ref params) = squeeze_params {
        super::squeeze::apply_squeeze(&mut work_image, params)?;
    }

    // Lossy modular: compute per-channel quantizers, pre-quantize, build multiplier info.
    let multiplier_info = if let Some(lossy) = lossy_options {
        use super::quantize::{
            build_multiplier_info, compute_channel_quantizer_xyb, quantize_channel,
        };

        let mut quants = Vec::new();
        for ch in work_image.channels.iter_mut() {
            let component = ch.component;
            if !(0..3).contains(&component) {
                // Non-XYB channel (shouldn't happen for LfFrame, but be safe)
                quants.push(1);
                continue;
            }
            let q = compute_channel_quantizer_xyb(
                component as usize,
                ch.hshift,
                ch.vshift,
                lossy.distance,
            );
            quantize_channel(ch, q);
            quants.push(q);
        }

        let info = build_multiplier_info(&quants, 0);

        crate::trace::debug_eprintln!(
            "LOSSY_MODULAR: distance={:.2}, {} channels, quants={:?}, {} mul_info entries",
            lossy.distance,
            work_image.channels.len(),
            quants,
            info.len(),
        );

        Some(info)
    } else {
        None
    };

    // Step 0: WP parameters.
    // For lossy modular with Zero predictor, WP is unused but we still need
    // valid params for the gather phase (which computes WP for all predictors).
    let wp_params = if !is_lossy && profile.wp_num_param_sets > 0 {
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
    let num_refs = if is_lossy {
        0
    } else {
        // Cap `max_ref_channels` to honour `--modular_nb_prev_channels`
        // (libjxl `options.max_properties`). `None` (default) keeps the
        // natural image-derived maximum.
        knobs.nb_prev_channels_cap(max_ref_channels(&work_image))
    };
    let mut samples = TreeSamples::new_with_ref_channels(num_refs);
    // Phase 2 of issue #41: thread the same property list the tree
    // learner will use into the gather-time dedup hash so the merge
    // stays at-or-below the post-sort dedup in aggressiveness.
    let dedup_properties: Vec<usize> = if profile.gather_dedup {
        TreeLearningParams::from_profile(profile)
            .with_ref_properties(num_refs, profile.effort)
            .properties
            .clone()
    } else {
        Vec::new()
    };
    if profile.gather_dedup {
        let _ = gather_samples_strided_with_dedup_backend(
            &mut samples,
            &work_image,
            0,
            0,
            stride,
            &wp_params,
            None,
            true,
            profile.gather_dedup_phase3,
            &dedup_properties,
        );
    } else {
        gather_samples_strided(&mut samples, &work_image, 0, 0, stride, &wp_params);
    }

    // Step 2: Learn tree with effort-dependent parameters
    //
    // Pixel fraction must reflect the *gathered* sample count
    // (`total_gathered_weight()` recovers it whether or not gather-time
    // dedup ran — see section.rs).
    let pixel_fraction = if total_pixels > 0 {
        samples.total_gathered_weight() as f64 / total_pixels as f64
    } else {
        1.0
    };
    let params = TreeLearningParams::from_profile(profile)
        .with_ref_properties(num_refs, profile.effort)
        .with_pixel_fraction(pixel_fraction)
        .with_total_pixels(total_pixels);

    // Tree-learn force-predictor override (libjxl `cjxl -P N` /
    // `--modular_predictor`). When `modular_predictor` is `Some(0..=13)`
    // and we are NOT in the lossy modular path (which has its own
    // multiplier-info forced-split discipline + Zero predictor invariant),
    // bypass ID3 and build a single-leaf tree pinned to the requested
    // predictor. This matches libjxl's behaviour when an explicit `-P N`
    // accompanies its tree learner: the leaf predictor is forced even
    // though the surrounding plumbing (gather, residual collection)
    // proceeds as usual.
    //
    // The `5` default (Gradient) falls through to ID3 by design — see
    // [`resolve_tree_learn_force_predictor`] — so hash-locks stay green
    // when callers pass `--modular-predictor 5` explicitly.
    let force_predictor = if is_lossy {
        None
    } else {
        resolve_tree_learn_force_predictor(knobs)
    };
    // RIGED meta-mode override (id 14). Mutually exclusive with
    // `force_predictor` since `Predictor::from_id(14) → None`. Skipped on
    // the lossy modular path for the same reason as `force_predictor`:
    // lossy needs the forced-split multiplier tree, not an override.
    let riged_override = if is_lossy {
        None
    } else {
        resolve_tree_learn_riged_tree(knobs, image.bit_depth)
    };

    let tree = if let Some(ref mul_info) = multiplier_info {
        // Lossy: use forced-split tree learning with multiplier info
        let num_channels = work_image.channels.len() as u32;
        let initial_range = [[0, num_channels], [0, 1]];
        compute_best_tree_with_multipliers(&mut samples, &params, mul_info, initial_range)
    } else if let Some(forced) = force_predictor {
        // Force-predictor override: single-leaf tree, no ID3.
        super::tree::simple_tree(forced)
    } else if let Some(riged) = riged_override {
        // RIGED override (id 14): hand-crafted 3-leaf gradient-aware
        // tree, no ID3.
        riged
    } else {
        compute_best_tree(&mut samples, &params)
    };
    let num_contexts = count_contexts(&tree) as usize;

    crate::trace::debug_eprintln!(
        "TREE_LEARN: effort={}, {} props, {} max_buckets, threshold={:.0}*{:.3}={:.1}, \
         {} nodes, {} leaves/contexts, {} samples, lossy={}",
        profile.effort,
        params.properties.len(),
        params.max_property_values,
        params.split_threshold,
        params.pixel_fraction * 0.9 + 0.1,
        params.split_threshold * (params.pixel_fraction * 0.9 + 0.1),
        tree.len(),
        num_contexts,
        samples.num_samples,
        is_lossy,
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
        match apply_lz77(
            &tokens,
            num_contexts,
            false,
            lz77_method,
            dist_multiplier,
            budget,
        )? {
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
        true, // optimize uint configs
        lz77_params.as_ref(),
        Some(total_pixels),
    );

    // Step 5: Write bitstream
    crate::f16::write_lf_quant(writer, dc_quant_custom)?;
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

    {
        let has_palette = palette_result.is_some();
        let has_rct = rct_type.is_some();
        let has_squeeze = squeeze_params.is_some();
        let num_transforms =
            compact_info.len() as u32 + has_palette as u32 + has_rct as u32 + has_squeeze as u32;
        write_num_transforms(writer, num_transforms)?;

        // ChannelCompact transforms first (per-channel palette, num_c=1)
        for &(begin_c, nb_colors) in &compact_info {
            write_palette_transform(writer, begin_c, 1, nb_colors, 0, 0)?;
        }
        // Multi-channel palette (if any)
        if let Some((begin_c, num_c, nb_colors)) = palette_result {
            write_palette_transform(writer, begin_c, num_c, nb_colors, 0, 0)?;
        }
        // RCT (begin_c adjusted for ChannelCompact meta channels)
        if let Some(rct_type) = rct_type {
            let rct_begin_c = compact_info.len();
            write_rct_transform(writer, rct_begin_c, rct_type)?;
        }
        if let Some(ref params) = squeeze_params {
            write_squeeze_transform(writer, params)?;
        }
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

/// Write a pre-squeezed, pre-quantized modular stream with tree leaf multipliers.
///
/// This function is used by the LfFrame encoder when lossy modular quantization
/// is active. The caller has already:
/// 1. Applied Squeeze transform to the image
/// 2. Pre-quantized each channel (pixels are multiples of their channel's q)
/// 3. Built the multiplier_info for forced tree splits
///
/// This function handles:
/// - Writing dc_quant (custom values for LfFrame)
/// - Writing the Squeeze transform descriptor
/// - Tree learning with forced splits at channel boundaries
/// - Residual collection with division by multiplier
/// - ANS entropy coding and bitstream assembly
///
/// Zero predictor is forced for all leaves with multiplier > 1, guaranteeing
/// the residual divisibility invariant (residual % multiplier == 0).
#[allow(clippy::too_many_arguments, dead_code)]
pub(crate) fn write_modular_stream_with_tree_dc_quant_presqueezed(
    image: &ModularImage,
    writer: &mut BitWriter,
    profile: &crate::effort::EffortProfile,
    use_lz77: bool,
    lz77_method: crate::entropy_coding::lz77::Lz77Method,
    dc_quant_custom: Option<[f32; 3]>,
    squeeze_params: &[super::squeeze::SqueezeParams],
    multiplier_info: &[super::quantize::ModularMultiplierInfo],
    _quants: &[i32],
    budget: Option<&alloc::sync::Arc<crate::budget::MemoryBudget>>,
) -> Result<()> {
    use super::tree::count_contexts;
    use super::tree_learn::{
        TreeLearningParams, TreeSamples, collect_residuals_with_tree, compute_best_tree,
        compute_best_tree_with_multipliers, compute_gather_stride_from_profile,
        gather_samples_strided, gather_samples_strided_with_dedup_backend,
    };
    use crate::entropy_coding::encode::build_entropy_code_ans_with_options;
    use crate::entropy_coding::encode::write_entropy_code_ans;
    use crate::entropy_coding::lz77::{apply_lz77, write_lz77_header};

    // WP parameters: default (Zero predictor is forced for lossy leaves,
    // but WP params are still needed for the gather phase).
    let wp_params = super::predictor::WeightedPredictorParams::default();

    // Step 1: Gather samples (with subsampling for large images)
    let total_pixels: usize = image
        .channels
        .iter()
        .map(|ch| ch.width() * ch.height())
        .sum();
    let stride = compute_gather_stride_from_profile(total_pixels, profile);
    let mut samples = TreeSamples::new();
    let dedup_properties: Vec<usize> = if profile.gather_dedup {
        TreeLearningParams::from_profile(profile).properties.clone()
    } else {
        Vec::new()
    };
    if profile.gather_dedup {
        let _ = gather_samples_strided_with_dedup_backend(
            &mut samples,
            image,
            0,
            0,
            stride,
            &wp_params,
            None,
            true,
            profile.gather_dedup_phase3,
            &dedup_properties,
        );
    } else {
        gather_samples_strided(&mut samples, image, 0, 0, stride, &wp_params);
    }

    // Step 2: Learn tree with forced splits for multiplier info.
    // See section.rs for the gather-time dedup pixel_fraction rationale.
    let pixel_fraction = if total_pixels > 0 {
        samples.total_gathered_weight() as f64 / total_pixels as f64
    } else {
        1.0
    };
    let params = TreeLearningParams::from_profile(profile)
        .with_pixel_fraction(pixel_fraction)
        .with_total_pixels(total_pixels);

    let tree = if !multiplier_info.is_empty() {
        let num_channels = image.channels.len() as u32;
        let initial_range = [[0, num_channels], [0, 1]];
        compute_best_tree_with_multipliers(&mut samples, &params, multiplier_info, initial_range)
    } else {
        compute_best_tree(&mut samples, &params)
    };
    let num_contexts = count_contexts(&tree) as usize;

    crate::trace::debug_eprintln!(
        "PRESQUEEZED_TREE: {} nodes, {} contexts, {} samples, {} mul_info entries",
        tree.len(),
        num_contexts,
        samples.num_samples,
        multiplier_info.len(),
    );

    // Step 3: Collect residuals with learned tree
    let tokens = collect_residuals_with_tree(image, &tree, 0, &wp_params);

    // Step 3b: Optionally apply LZ77 to the token stream
    let dist_multiplier = image.channels.iter().map(|c| c.width()).max().unwrap_or(0) as i32;
    let (tokens, lz77_params) = if use_lz77 {
        match apply_lz77(
            &tokens,
            num_contexts,
            false,
            lz77_method,
            dist_multiplier,
            budget,
        )? {
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

    // Step 4: Build multi-context ANS code
    let code = build_entropy_code_ans_with_options(
        &tokens,
        ans_num_contexts,
        true,
        true, // optimize uint configs
        lz77_params.as_ref(),
        Some(total_pixels),
    );

    // Step 5: Write bitstream
    // dc_quant header
    crate::f16::write_lf_quant(writer, dc_quant_custom)?;
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

    // GroupHeader
    writer.write(1, 1)?; // use_global_tree = true
    write_wp_header(writer, &wp_params)?;

    // Squeeze transform descriptor (1 transform)
    let has_squeeze = !squeeze_params.is_empty();
    if has_squeeze {
        writer.write(2, 1)?; // num_transforms = 1
        write_squeeze_transform(writer, squeeze_params)?;
    } else {
        writer.write(2, 0)?; // num_transforms = 0
    }

    // Debug: verify ANS encoding correctness
    #[cfg(debug_assertions)]
    if lz77_params.is_none() {
        let roundtrip_result = crate::entropy_coding::encode::verify_ans_roundtrip(&tokens, &code);
        if roundtrip_result.is_err() {
            crate::debug_rect!(
                "ans/verify",
                0,
                0,
                image.width(),
                image.height(),
                "ROUNDTRIP FAILED for presqueezed data (ctx={} histo={} tokens={}): {:?}",
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
    budget: Option<&alloc::sync::Arc<crate::budget::MemoryBudget>>,
) -> Result<()> {
    use super::rct::{RctType, forward_rct};
    use super::squeeze::{apply_squeeze, default_squeeze_params};
    use super::tree::count_contexts;
    use super::tree_learn::{
        TreeLearningParams, TreeSamples, collect_residuals_with_tree, compute_best_tree,
        compute_gather_stride_from_profile, gather_samples_strided,
        gather_samples_strided_with_dedup_backend,
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
            budget,
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
    // For squeeze, WP is only used as a property (property 15) for tree splitting,
    // not as a predictor (libjxl forces Predictor::Zero for squeeze residuals).
    let wp_params = if profile.wp_num_param_sets > 0 {
        super::predictor::find_best_wp_params(&transformed.channels, profile.wp_num_param_sets)
    } else {
        super::predictor::WeightedPredictorParams::default()
    };

    // Step 3: Gather samples from squeezed image (with subsampling for large images)
    // Use squeeze-specific samples: only Zero predictor candidate (matching libjxl
    // enc_modular.cc:629-633: "zero predictor for Squeeze residues").
    let total_pixels: usize = transformed
        .channels
        .iter()
        .map(|ch| ch.width() * ch.height())
        .sum();
    let stride = compute_gather_stride_from_profile(total_pixels, profile);
    let mut samples = TreeSamples::new_for_squeeze();
    let dedup_properties: Vec<usize> = if profile.gather_dedup {
        TreeLearningParams::from_profile_squeeze(profile)
            .properties
            .clone()
    } else {
        Vec::new()
    };
    if profile.gather_dedup {
        let _ = gather_samples_strided_with_dedup_backend(
            &mut samples,
            &transformed,
            0,
            0,
            stride,
            &wp_params,
            None,
            true,
            profile.gather_dedup_phase3,
            &dedup_properties,
        );
    } else {
        gather_samples_strided(&mut samples, &transformed, 0, 0, stride, &wp_params);
    }

    // Step 4: Learn tree with effort-dependent parameters
    // Use squeeze-specific property order (libjxl enc_modular.cc:538-541).
    // See section.rs for the gather-time dedup pixel_fraction rationale.
    let pixel_fraction = if total_pixels > 0 {
        samples.total_gathered_weight() as f64 / total_pixels as f64
    } else {
        1.0
    };
    let tree_params = TreeLearningParams::from_profile_squeeze(profile)
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
        match apply_lz77(
            &tokens,
            num_contexts,
            false,
            lz77_method,
            dist_multiplier,
            budget,
        )? {
            Some((lz77_tokens, params)) => {
                crate::trace::debug_eprintln!(
                    "SQUEEZE LZ77: {} → {} tokens ({:.1}x), method={:?}, dm={}",
                    tokens.len(),
                    lz77_tokens.len(),
                    tokens.len() as f64 / lz77_tokens.len() as f64,
                    lz77_method,
                    dist_multiplier,
                );
                (lz77_tokens, Some(params))
            }
            None => {
                crate::trace::debug_eprintln!(
                    "SQUEEZE LZ77: not cost-effective, method={:?}, dm={}, {} tokens",
                    lz77_method,
                    dist_multiplier,
                    tokens.len(),
                );
                (tokens, None)
            }
        }
    } else {
        crate::trace::debug_eprintln!("SQUEEZE LZ77: disabled");
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
        true, // optimize uint configs
        lz77_params.as_ref(),
        Some(total_pixels),
    );

    // Step 7: Write bitstream
    let _bit0 = writer.bits_written();
    // dc_quant.all_default = true
    writer.write(1, 1)?;
    // has_tree = true
    writer.write(1, 1)?;

    // Write the learned tree
    write_tree(writer, &tree)?;
    let _bit_after_tree = writer.bits_written();

    // Write LZ77 header + ANS data histogram
    if ans_num_contexts > 1 {
        write_lz77_header(lz77_params.as_ref(), writer)?;
        write_entropy_code_ans(&code, writer)?;
    } else {
        use super::section::write_ans_modular_header;
        write_ans_modular_header(writer, &code)?;
    }
    let _bit_after_histo = writer.bits_written();

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
    let _bit_after_header = writer.bits_written();
    crate::trace::debug_eprintln!(
        "SQUEEZE OVERHEAD: tree={} bits ({:.0}B), histograms={} bits ({:.0}B), header={} bits ({:.0}B), total_overhead={:.0}B",
        _bit_after_tree - _bit0,
        (_bit_after_tree - _bit0) as f64 / 8.0,
        _bit_after_histo - _bit_after_tree,
        (_bit_after_histo - _bit_after_tree) as f64 / 8.0,
        _bit_after_header - _bit_after_histo,
        (_bit_after_header - _bit_after_histo) as f64 / 8.0,
        (_bit_after_header - _bit0) as f64 / 8.0,
    );

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
    let _bit_before_data = writer.bits_written();
    write_tokens_ans(&tokens, &code, lz77_params.as_ref(), writer)?;
    let _bit_after_data = writer.bits_written();
    crate::trace::debug_eprintln!(
        "SQUEEZE DATA: {} bits ({:.0}B), {} tokens, {} histograms",
        _bit_after_data - _bit_before_data,
        (_bit_after_data - _bit_before_data) as f64 / 8.0,
        tokens.len(),
        code.histograms.len(),
    );
    crate::trace::debug_eprintln!(
        "SQUEEZE TOTAL: {:.0}B (overhead {:.0}B + data {:.0}B)",
        (_bit_after_data - _bit0) as f64 / 8.0,
        (_bit_after_header - _bit0) as f64 / 8.0,
        (_bit_after_data - _bit_before_data) as f64 / 8.0,
    );

    writer.zero_pad_to_byte();
    Ok(())
}

// ===== Multi-group support =====
// These functions are now in the section module for better organization

pub use super::section::{
    build_histogram_from_residuals, collect_all_residuals, write_global_modular_section,
    write_group_modular_section, write_group_modular_section_idx,
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
            .encode_modular(&image, &color_encoding, &mut writer, None)
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
                .encode_modular(&image, &color_encoding, &mut writer, None)
                .unwrap();
            let huf_bytes = writer.finish_with_padding();
            eprintln!("Huffman modular gray varied 4x2: {} bytes", huf_bytes.len());
            eprintln!("Huffman bytes: {:02x?}", huf_bytes);
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
            .encode_modular(&image, &color_encoding, &mut writer, None)
            .unwrap();

        let bytes = writer.finish_with_padding();
        eprintln!("ANS modular gray varied 4x2: {} bytes", bytes.len());
        eprintln!("ANS bytes: {:02x?}", bytes);

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
            .encode_modular(&image, &color_encoding, &mut writer, None)
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
