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
        lz77::write_lz77_header,
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
        num_samples as u64 * (size_of::<u32>() + size_of::<Token>()) as u64,
    )?;
    let fallible = budget.is_some_and(|budget| budget.is_fallible());
    let (residuals, _) = collect_all_residuals_with_predictor(image, 5);
    let mut plain = vec_with_capacity_fallible(fallible, num_samples)?;
    plain.extend(residuals.iter().map(|&value| Token::new(0, value)));
    check_stop(stop)?;
    keep_smaller(incumbent, serialize(&plain)?);

    Ok(())
}

fn keep_smaller(incumbent: &mut BitWriter, candidate: BitWriter) {
    // The section and its TOC entry both have nondecreasing byte cost.
    if candidate.bits_written().div_ceil(8) < incumbent.bits_written().div_ceil(8) {
        *incumbent = candidate;
    }
}

fn serialize(tokens: &[Token]) -> Result<BitWriter> {
    let config = HybridUintConfig::new(4, 2, 0);
    let code = build_entropy_code_unclustered(tokens, &[config], None);
    let mut writer = BitWriter::new();
    writer.write(1, 0)?; // local tree
    writer.write(1, 1)?; // default weighted predictor parameters
    super::encode::write_num_transforms(&mut writer, 0)?;
    let (depths, codes) = write_tree_histogram_for_gradient(&mut writer)?;
    write_gradient_tree_tokens(&mut writer, &depths, &codes)?;
    write_lz77_header(None, &mut writer)?;
    write_dyn_prefix_codes(&code.prefix_codes, &code.uint_configs, &mut writer)?;
    code.write_tokens_owned(tokens, None, &mut writer)?;
    writer.zero_pad_to_byte();
    Ok(writer)
}
