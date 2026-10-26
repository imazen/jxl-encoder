// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Lossy modular (Squeeze + quantization) for a frame larger than one group.
//!
//! The single-group lossy stream lives in
//! [`super::encode::write_modular_stream_with_tree_dc_quant`]. This writer
//! covers the multi-group case the LF frame needs above 2048 px: libjxl
//! encodes such a DC frame exactly like a small one (Squeeze, per-channel
//! quantizers, Zero-predictor tree with multipliers; enc_modular.cc), with the
//! squeezed channels spread over the frame's sections by the spec rule:
//! leading channels that fit one group go to LfGlobal, channels with both
//! shifts >= 3 to the LfGroups, the rest to the PassGroups, each cropped to
//! its section at its own resolution.

use super::channel::{Channel, ModularImage};
use super::quantize::ModularMultiplierInfo;
use crate::bit_writer::BitWriter;
use crate::entropy_coding::token::Token as AnsToken;
use crate::error::Result;

/// Spec stream id offset of the PassGroups: GlobalData, then ModularLF and
/// VarDCT DC/AC-metadata streams per LF group, then the quant tables.
const NUM_QUANT_TABLES: usize = 17;

/// One section's sub-image: its channels and, for each, the index of the
/// squeezed channel it was cropped from (whose quantizer applies).
struct Section {
    stream_id: u32,
    channels: Vec<Channel>,
    source: Vec<usize>,
}

/// Writes the TOC and every section of a multi-group lossy modular frame of
/// `image` at `distance`, with custom DC quantization factors `dc_quant`
/// (LF frame). The frame header is the caller's. Returns the samples the
/// decoder reconstructs.
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_lossy_multi_group_sections(
    image: &ModularImage,
    dc_quant: [f32; 3],
    distance: f32,
    profile: &crate::effort::EffortProfile,
    group_dim: usize,
    writer: &mut BitWriter,
    budget: Option<&alloc::sync::Arc<crate::budget::MemoryBudget>>,
) -> Result<ModularImage> {
    use super::encode::{
        lossy_squeeze_quantize, write_squeeze_transform, write_tree, write_wp_header,
    };
    use super::tree::count_contexts;
    use super::tree_learn::{
        TreeLearningParams, TreeSamples, collect_residuals_with_tree_with_budget,
        compute_best_tree_with_multipliers_stop, compute_gather_stride_from_profile,
        gather_samples_strided_with_budget_inner,
    };
    use crate::entropy_coding::encode::{
        build_entropy_code_ans_with_options_stop, write_entropy_code_ans, write_tokens_ans_stop,
    };

    let (width, height) = (image.width(), image.height());
    let mut squeezed = image.clone();
    let (params, quants) = lossy_squeeze_quantize(&mut squeezed, distance)?;

    let lf_group_dim = group_dim * 8;
    let num_groups_x = width.div_ceil(group_dim);
    let num_groups = num_groups_x * height.div_ceil(group_dim);
    let num_lf_groups_x = width.div_ceil(lf_group_dim);
    let num_lf_groups = num_lf_groups_x * height.div_ceil(lf_group_dim);

    // Partition the squeezed channels into sections (spec G.1.3 / libjxl
    // ModularFrameEncoder): the leading run that fits one group is global.
    let global_cutoff = squeezed
        .channels
        .iter()
        .position(|c| c.width() > group_dim || c.height() > group_dim)
        .unwrap_or(squeezed.channels.len());
    let mut sections = vec![Section {
        stream_id: 0,
        channels: squeezed.channels[..global_cutoff].to_vec(),
        source: (0..global_cutoff).collect(),
    }];
    let crop = |indices: &[usize], count: usize, cells_x: usize, dim: usize, base: usize| {
        (0..count)
            .map(|s| {
                let mut section = Section {
                    stream_id: (base + s) as u32,
                    channels: Vec::new(),
                    source: Vec::new(),
                };
                for &i in indices {
                    if let Some(c) =
                        squeezed.channels[i].extract_grid_cell(s % cells_x, s / cells_x, dim)
                    {
                        section.channels.push(c);
                        section.source.push(i);
                    }
                }
                section
            })
            .collect::<Vec<_>>()
    };
    let (lf_indices, pass_indices): (Vec<usize>, Vec<usize>) = (global_cutoff
        ..squeezed.channels.len())
        .partition(|&i| squeezed.channels[i].hshift.min(squeezed.channels[i].vshift) >= 3);
    sections.extend(crop(
        &lf_indices,
        num_lf_groups,
        num_lf_groups_x,
        lf_group_dim,
        1 + num_lf_groups,
    ));
    sections.extend(crop(
        &pass_indices,
        num_groups,
        num_groups_x,
        group_dim,
        1 + 3 * num_lf_groups + NUM_QUANT_TABLES,
    ));

    // One multiplier box per (local channel, stream), as libjxl builds them:
    // the tree must split every box apart so each leaf knows its quantizer.
    let mut multipliers: Vec<ModularMultiplierInfo> = Vec::new();
    for s in &sections {
        for (k, &src) in s.source.iter().enumerate() {
            let q = quants[src] as u32;
            let k = k as u32;
            if let Some(last) = multipliers.last_mut()
                && last.range[1] == [s.stream_id, s.stream_id + 1]
                && last.range[0][1] == k
                && last.multiplier == q
            {
                last.range[0][1] = k + 1;
                continue;
            }
            multipliers.push(ModularMultiplierInfo {
                range: [[k, k + 1], [s.stream_id, s.stream_id + 1]],
                multiplier: q,
            });
        }
    }
    let max_channels = sections.iter().map(|s| s.channels.len()).max().unwrap_or(0) as u32;
    let max_stream = sections.last().map_or(0, |s| s.stream_id) + 1;

    let sub_image = |s: &Section| ModularImage {
        channels: s.channels.clone(),
        bit_depth: squeezed.bit_depth,
        is_grayscale: squeezed.is_grayscale,
        has_alpha: false,
    };
    // The Zero predictor the multiplier tree uses ignores the weighted
    // predictor, but gathering still computes it.
    let wp_params = super::predictor::WeightedPredictorParams::default();
    let total_pixels: usize = squeezed
        .channels
        .iter()
        .map(|c| c.width() * c.height())
        .sum();
    let stride = compute_gather_stride_from_profile(total_pixels, profile);
    let mut samples = TreeSamples::new();
    for s in sections.iter().filter(|s| !s.channels.is_empty()) {
        gather_samples_strided_with_budget_inner(
            &mut samples,
            &sub_image(s),
            s.stream_id,
            0,
            stride,
            &wp_params,
            budget,
            None,
            None,
        )?;
    }
    let pixel_fraction = if total_pixels > 0 {
        samples.total_gathered_weight() as f64 / total_pixels as f64
    } else {
        1.0
    };
    let tree_params = TreeLearningParams::from_profile(profile)
        .with_pixel_fraction(pixel_fraction)
        .with_total_pixels(total_pixels);
    let tree = compute_best_tree_with_multipliers_stop(
        &mut samples,
        &tree_params,
        &multipliers,
        [[0, max_channels], [0, max_stream]],
        None,
    )?;
    let num_contexts = count_contexts(&tree) as usize;

    let tokens: Vec<Vec<AnsToken>> = sections
        .iter()
        .map(|s| {
            if s.channels.is_empty() {
                Ok(Vec::new())
            } else {
                collect_residuals_with_tree_with_budget(
                    &sub_image(s),
                    &tree,
                    s.stream_id,
                    &wp_params,
                    budget,
                    None,
                )
            }
        })
        .collect::<Result<_>>()?;
    // LZ77 per section, as libjxl's modular coder applies it: each section
    // is a fresh LZ77 state whose distance multiplier is its widest channel.
    // `apply_lz77` keeps the plain tokens unless the matches pay.
    let mut tokens = tokens;
    let mut lz77_params = None;
    if profile.lz77 {
        for (s, t) in sections.iter().zip(tokens.iter_mut()) {
            if t.is_empty() {
                continue;
            }
            let dm = s.channels.iter().map(|c| c.width()).max().unwrap_or(0) as i32;
            if let Some((lz, _)) = crate::entropy_coding::lz77::apply_lz77_stop(
                t,
                num_contexts,
                false,
                profile.lz77_method,
                dm,
                budget,
                None,
            )? {
                *t = lz;
            }
        }
        if tokens.iter().flatten().any(|t| t.is_lz77_length()) {
            let mut p = crate::entropy_coding::lz77::Lz77Params::new(num_contexts, false);
            p.enabled = true;
            lz77_params = Some(p);
        }
    }
    let ans_num_contexts = num_contexts + usize::from(lz77_params.is_some());
    let all_tokens: Vec<AnsToken> = tokens.iter().flatten().copied().collect();
    let code = build_entropy_code_ans_with_options_stop(
        &all_tokens,
        ans_num_contexts,
        true,
        true,
        lz77_params.as_ref(),
        Some(total_pixels),
        None,
    );

    let group_header = |w: &mut BitWriter, transforms: bool| -> Result<()> {
        w.write(1, 1)?; // use_global_tree
        write_wp_header(w, &wp_params)?;
        match (&params, transforms) {
            (Some(p), true) => {
                w.write(2, 1)?; // nb_transforms = 1
                write_squeeze_transform(w, p)?;
            }
            _ => w.write(2, 0)?, // nb_transforms = 0
        }
        Ok(())
    };

    // LfGlobal: DC quant, tree, histograms, then the global stream.
    let mut global = BitWriter::new();
    crate::f16::write_lf_quant(&mut global, Some(dc_quant))?;
    global.write(1, 1)?; // has_tree
    write_tree(&mut global, &tree)?;
    if ans_num_contexts > 1 {
        crate::entropy_coding::lz77::write_lz77_header(lz77_params.as_ref(), &mut global)?;
        write_entropy_code_ans(&code, &mut global)?;
    } else {
        super::section::write_ans_modular_header(&mut global, &code)?;
    }
    group_header(&mut global, true)?;
    write_tokens_ans_stop(&tokens[0], &code, lz77_params.as_ref(), &mut global, None)?;
    global.zero_pad_to_byte();

    let section_bytes = |i: usize| -> Result<Vec<u8>> {
        let mut w = BitWriter::new();
        if !sections[i].channels.is_empty() {
            group_header(&mut w, false)?;
            write_tokens_ans_stop(&tokens[i], &code, lz77_params.as_ref(), &mut w, None)?;
            w.zero_pad_to_byte();
        }
        Ok(w.finish())
    };
    // Section order: LfGlobal | LfGroup x N | HfGlobal (empty) | PassGroup x M.
    let mut data = vec![global.finish()];
    for i in 1..=num_lf_groups {
        data.push(section_bytes(i)?);
    }
    data.push(Vec::new());
    for i in 1 + num_lf_groups..sections.len() {
        data.push(section_bytes(i)?);
    }

    writer.write(1, 0)?; // TOC not permuted
    writer.zero_pad_to_byte();
    for d in &data {
        crate::vardct::lf_frame::write_toc_entry(writer, d.len() as u32)?;
    }
    writer.zero_pad_to_byte();
    for d in &data {
        writer.append_bytes(d)?;
    }

    super::encode::lossy_modular_reconstruction(image, distance)
}
