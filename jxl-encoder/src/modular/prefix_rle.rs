// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

use alloc::sync::Arc;

use crate::{
    bit_writer::BitWriter,
    budget::{MemoryBudget, vec_with_capacity_fallible},
    entropy_coding::{
        encode::{build_entropy_code_unclustered, write_dyn_prefix_codes},
        hybrid_uint::HybridUintConfig,
        lz77::{Lz77Params, write_lz77_header},
        token::Token,
    },
    error::{Result, check_stop},
};

use super::{
    channel::ModularImage,
    encode::{write_gradient_tree_tokens, write_tree_histogram_for_gradient},
    section::collect_all_residuals_with_predictor,
};

/// Compare independently coded group streams against the shared prefix code.
pub(super) fn try_local_prefix(
    image: &ModularImage,
    incumbent: &mut BitWriter,
    budget: Option<&Arc<MemoryBudget>>,
    stop: Option<&dyn enough::Stop>,
) -> Result<()> {
    let num_samples = image
        .channels
        .iter()
        .map(|c| c.width() * c.height())
        .sum::<usize>();
    if num_samples == 0 {
        return Ok(());
    }
    let _reservation = MemoryBudget::reserve_opt(
        budget,
        num_samples as u64 * (size_of::<u32>() + 2 * size_of::<Token>()) as u64,
    )?;
    let fallible = budget.is_some_and(|budget| budget.is_fallible());
    let (residuals, _) = collect_all_residuals_with_predictor(image, 5);
    let mut plain = vec_with_capacity_fallible(fallible, num_samples)?;
    plain.extend(residuals.iter().map(|&value| Token::new(0, value)));
    keep_smaller(incumbent, serialize(&plain, None)?);

    let mut rle = vec_with_capacity_fallible(fallible, num_samples)?;
    let mut params = Lz77Params::new(1, true);
    params.enabled = true;
    let mut start = 0;
    for channel in &image.channels {
        let end = start + channel.width() * channel.height();
        let mut values = &residuals[start..end];
        while let Some((&value, _)) = values.split_first() {
            check_stop(stop)?;
            let count = values.iter().take_while(|&&other| other == value).count();
            let copies = count - 1;
            rle.push(Token::new(0, value));
            if copies >= params.min_length as usize {
                rle.push(Token::lz77_length(0, copies as u32 - params.min_length));
                // The decoder uses distance symbol 1 when the stream has a width.
                rle.push(Token::new(params.distance_context, 1));
            } else {
                rle.extend(core::iter::repeat_n(Token::new(0, value), copies));
            }
            values = &values[count..];
        }
        start = end;
    }
    if rle.len() < plain.len() {
        keep_smaller(incumbent, serialize(&rle, Some(&params))?);
    }
    Ok(())
}

fn keep_smaller(incumbent: &mut BitWriter, candidate: BitWriter) {
    // The section and its TOC entry both have nondecreasing byte cost.
    if candidate.bits_written().div_ceil(8) < incumbent.bits_written().div_ceil(8) {
        *incumbent = candidate;
    }
}

fn serialize(tokens: &[Token], lz77: Option<&Lz77Params>) -> Result<BitWriter> {
    let configs = [
        HybridUintConfig::new(4, 2, 0),
        HybridUintConfig::new(0, 0, 0),
    ];
    let code =
        build_entropy_code_unclustered(tokens, &configs[..1 + usize::from(lz77.is_some())], lz77);
    let mut writer = BitWriter::new();
    writer.write(1, 0)?; // local tree
    writer.write(1, 1)?; // default weighted predictor parameters
    super::encode::write_num_transforms(&mut writer, 0)?;
    let (depths, codes) = write_tree_histogram_for_gradient(&mut writer)?;
    write_gradient_tree_tokens(&mut writer, &depths, &codes)?;
    write_lz77_header(lz77, &mut writer)?;
    if lz77.is_some() {
        code.write_header(&mut writer)?;
    } else {
        // A stream with only one context has no context map on the wire.
        write_dyn_prefix_codes(&code.prefix_codes, &code.uint_configs, &mut writer)?;
    }
    code.write_tokens_owned(tokens, lz77, &mut writer)?;
    writer.zero_pad_to_byte();
    Ok(writer)
}
