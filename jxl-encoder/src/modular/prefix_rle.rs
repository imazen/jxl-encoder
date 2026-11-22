// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

use alloc::{sync::Arc, vec::Vec};

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
    encode_tree::write_tree,
    section::{
        GlobalTransforms, GroupTransforms, collect_all_residuals_with_predictor,
        write_global_transforms_full, write_group_header,
    },
    tree::{Tree, assign_sequential_contexts, per_channel_tree},
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
    let mut rle = vec_with_capacity_fallible(fallible, num_samples)?;
    let channels = image.channels.len();
    for num_contexts in [1, channels] {
        let tree = if num_contexts == 1 {
            None
        } else {
            if !(2..=4).contains(&channels) {
                break;
            }
            let mut tree = per_channel_tree(channels);
            assign_sequential_contexts(&mut tree);
            Some(tree)
        };
        let mut params = Lz77Params::new(num_contexts, true);
        params.enabled = true;
        plain.clear();
        rle.clear();
        let mut start = 0;
        for (index, channel) in image.channels.iter().enumerate() {
            let context = tree
                .as_ref()
                .map_or(0, |tree| tree[channels - 1 + index].context_id);
            let end = start + channel.width() * channel.height();
            let values = &residuals[start..end];
            plain.extend(values.iter().map(|&value| Token::new(context, value)));
            append_runs(values, context, &params, &mut rle, stop)?;
            start = end;
        }
        keep_smaller(
            incumbent,
            serialize(&plain, tree.as_ref(), num_contexts, None)?,
        );
        if rle.len() < plain.len() {
            keep_smaller(
                incumbent,
                serialize(&rle, tree.as_ref(), num_contexts, Some(&params))?,
            );
        }
        if channels == 1 {
            break;
        }
    }
    Ok(())
}

/// Collect each group's tokens independently, then build one code for the frame.
pub(super) fn write_shared_prefix(
    groups: &[ModularImage],
    transforms: &GlobalTransforms,
    global: &mut BitWriter,
    use_rle: bool,
    budget: Option<&Arc<MemoryBudget>>,
    stop: Option<&dyn enough::Stop>,
) -> Result<Vec<Vec<u8>>> {
    let channels = groups[0].channels.len();
    let mut tree = per_channel_tree(channels);
    assign_sequential_contexts(&mut tree);

    let mut params = Lz77Params::new(channels, true);
    params.enabled = true;
    let lz77 = use_rle.then_some(&params);
    let num_samples = groups
        .iter()
        .flat_map(|group| &group.channels)
        .map(|channel| channel.width() as u64 * channel.height() as u64)
        .sum::<u64>();
    let _reservation = MemoryBudget::reserve_opt(
        budget,
        num_samples * (size_of::<u32>() + size_of::<Token>()) as u64,
    )?;
    let fallible = budget.is_some_and(|budget| budget.is_fallible());
    let mut group_tokens = vec_with_capacity_fallible(fallible, groups.len())?;
    for group in groups {
        check_stop(stop)?;
        let (residuals, _) = collect_all_residuals_with_predictor(group, 5);
        let mut tokens = vec_with_capacity_fallible(fallible, residuals.len())?;
        let mut start = 0;
        for (index, channel) in group.channels.iter().enumerate() {
            let context = tree[channels - 1 + index].context_id;
            let end = start + channel.width() * channel.height();
            let values = &residuals[start..end];
            if use_rle {
                append_runs(values, context, &params, &mut tokens, stop)?;
            } else {
                tokens.extend(values.iter().map(|&value| Token::new(context, value)));
            }
            start = end;
        }
        group_tokens.push(tokens);
    }
    let mut configs = [HybridUintConfig::new(4, 2, 0); 5];
    configs[channels] = HybridUintConfig::new(0, 0, 0);
    let code = build_entropy_code_unclustered(
        group_tokens.iter().flatten(),
        &configs[..channels + usize::from(use_rle)],
        lz77,
    );
    global.write(1, 1)?; // default LfQuantFactors
    global.write(1, 1)?; // global MA tree
    write_tree(global, &tree)?;
    write_lz77_header(lz77, global)?;
    code.write_header(global)?;
    global.write(1, 1)?; // use the global tree
    global.write(1, 1)?; // default weighted predictor parameters
    write_global_transforms_full(global, transforms)?;
    global.zero_pad_to_byte();

    let transforms = GroupTransforms::none();
    crate::parallel::parallel_map_result(groups.len(), |group| {
        let mut writer = BitWriter::new();
        write_group_header(&mut writer, None, &transforms)?;
        code.write_tokens_owned(&group_tokens[group], lz77, &mut writer)?;
        writer.zero_pad_to_byte();
        Ok(writer.finish())
    })
}

fn append_runs(
    mut values: &[u32],
    context: u32,
    params: &Lz77Params,
    rle: &mut Vec<Token>,
    stop: Option<&dyn enough::Stop>,
) -> Result<()> {
    while let Some((&value, _)) = values.split_first() {
        check_stop(stop)?;
        let count = values.iter().take_while(|&&other| other == value).count();
        let copies = count - 1;
        rle.push(Token::new(context, value));
        if copies >= params.min_length as usize {
            rle.push(Token::lz77_length(
                context,
                copies as u32 - params.min_length,
            ));
            // The decoder uses distance symbol 1 when the stream has a width.
            rle.push(Token::new(params.distance_context, 1));
        } else {
            rle.extend(core::iter::repeat_n(Token::new(context, value), copies));
        }
        values = &values[count..];
    }
    Ok(())
}

fn keep_smaller(incumbent: &mut BitWriter, candidate: BitWriter) {
    // The section and its TOC entry both have nondecreasing byte cost.
    if candidate.bits_written().div_ceil(8) < incumbent.bits_written().div_ceil(8) {
        *incumbent = candidate;
    }
}

fn serialize(
    tokens: &[Token],
    tree: Option<&Tree>,
    num_contexts: usize,
    lz77: Option<&Lz77Params>,
) -> Result<BitWriter> {
    let mut configs = [HybridUintConfig::new(4, 2, 0); 5];
    configs[num_contexts] = HybridUintConfig::new(0, 0, 0);
    let code = build_entropy_code_unclustered(
        tokens,
        &configs[..num_contexts + usize::from(lz77.is_some())],
        lz77,
    );
    let mut writer = BitWriter::new();
    writer.write(1, 0)?; // local tree
    writer.write(1, 1)?; // default weighted predictor parameters
    super::encode::write_num_transforms(&mut writer, 0)?;
    if let Some(tree) = tree {
        write_tree(&mut writer, tree)?;
    } else {
        let (depths, codes) = write_tree_histogram_for_gradient(&mut writer)?;
        write_gradient_tree_tokens(&mut writer, &depths, &codes)?;
    }
    write_lz77_header(lz77, &mut writer)?;
    if num_contexts > 1 || lz77.is_some() {
        code.write_header(&mut writer)?;
    } else {
        // A stream with only one context has no context map on the wire.
        write_dyn_prefix_codes(&code.prefix_codes, &code.uint_configs, &mut writer)?;
    }
    code.write_tokens_owned(tokens, lz77, &mut writer)?;
    writer.zero_pad_to_byte();
    Ok(writer)
}
