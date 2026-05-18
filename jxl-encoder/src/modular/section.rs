// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Modular section encoding for multi-group images.
//!
//! Handles GlobalModularState and section writing for large images that
//! are split into multiple groups.

use super::channel::ModularImage;
use super::encode::{
    predict_pixel_with_id, write_gradient_tree_tokens, write_hybrid_data_histogram,
    write_palette_transform, write_predictor_tree_tokens, write_rct_transform,
    write_tree_histogram_for_gradient, write_tree_histogram_for_predictor,
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

pub fn collect_all_residuals(image: &ModularImage) -> (Vec<u32>, u32) {
    collect_all_residuals_with_predictor(image, 5)
}

/// Knob-aware variant of [`collect_all_residuals`] that honours the
/// libjxl `--modular_predictor` override. `predictor_id == 5` (Gradient)
/// matches the legacy hash-locked output. See
/// [`super::encode::resolve_fixed_predictor`].
pub(crate) fn collect_all_residuals_with_predictor(
    image: &ModularImage,
    predictor_id: u8,
) -> (Vec<u32>, u32) {
    let mut residuals = Vec::new();
    let mut max_residual: u32 = 0;

    for channel in &image.channels {
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
        /// Fixed predictor id (libjxl `cjxl -P`); `5` (Gradient) is the
        /// legacy default that hash-locks pin against.
        predictor_id: u8,
    },
    /// ANS entropy coding state (single-context gradient tree).
    Ans {
        /// The ANS entropy code (distributions, context map, etc.)
        code: OwnedAnsEntropyCode,
        /// Fixed predictor id used for per-group residual collection.
        /// `5` (Gradient) is the legacy default.
        predictor_id: u8,
    },
    /// ANS entropy coding with learned MA tree (multi-context).
    AnsWithTree {
        /// The ANS entropy code (multiple distributions, context map).
        code: OwnedAnsEntropyCode,
        /// The learned MA tree for per-pixel predictor/context selection.
        tree: super::tree::Tree,
        /// WP parameters used during tree learning and residual collection.
        wp_params: super::predictor::WeightedPredictorParams,
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

    // HybridUint config (per-histogram optimized, or default {4,2,0})
    let config = code
        .uint_configs
        .first()
        .copied()
        .unwrap_or(crate::entropy_coding::hybrid_uint::HybridUintConfig::default_config());
    let se_bits = ceil_log2_nonzero(las as u32 + 1);
    writer.write(se_bits as usize, config.split_exponent as u64)?;
    if (config.split_exponent as usize) != las {
        let msb_bits = ceil_log2_nonzero(config.split_exponent + 1);
        writer.write(msb_bits as usize, config.msb_in_token as u64)?;
        let lsb_bits = ceil_log2_nonzero(config.split_exponent - config.msb_in_token + 1);
        writer.write(lsb_bits as usize, config.lsb_in_token as u64)?;
    }

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
    transforms: GlobalTransforms,
) -> Result<GlobalModularState> {
    write_global_modular_section_with_predictor(
        all_residuals,
        histogram,
        max_token,
        writer,
        use_ans,
        transforms,
        5,
    )
}

/// Knob-aware variant of [`write_global_modular_section`] that honours
/// `predictor_id` (libjxl `cjxl -P`). Default `5` (Gradient) keeps the
/// hash-locked output bit-identical.
pub fn write_global_modular_section_with_predictor(
    all_residuals: &[u32],
    histogram: &[u32],
    max_token: u32,
    writer: &mut BitWriter,
    use_ans: bool,
    transforms: GlobalTransforms,
    predictor_id: u8,
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

    // Tree histogram + tokens for the requested predictor.
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
        // Build ANS code from all residuals across all groups
        let tokens: Vec<AnsToken> = all_residuals.iter().map(|&r| AnsToken::new(0, r)).collect();
        let code = build_entropy_code_ans(&tokens, 1); // 1 context for single-leaf tree

        // Write ANS data header (distribution + config)
        write_ans_modular_header(writer, &code)?;

        // Write GlobalModular's ModularHeader
        writer.write(1, 1)?; // use_global_tree = true
        writer.write(1, 1)?; // wp_params.default_wp = true
        write_global_transforms_full(writer, &transforms)?;

        // Empty ANS stream terminator for the global modular sub-bitstream.
        // libjxl writes the same 32-bit initial state via `WriteTokens` even when
        // the LfGlobal carries no tokens — pre-fix jxl-oxide unconditionally calls
        // `Decoder::begin()` (which reads 32 bits) before checking buffer dims. djxl
        // and jxl-rs short-circuit before this read when no channels are decodable in
        // this section, so the extra 4 bytes are simply padding to them. See
        // `imazen/jxl-oxide@fd4e2c3` for the matching decoder fix.
        write_tokens_ans(&[], &code, None, writer)?;

        // Byte-align at end of global section
        writer.zero_pad_to_byte();
        crate::trace::debug_eprintln!(
            "GLOBAL_MODULAR [bit {}]: Global section done (ANS)",
            writer.bits_written()
        );

        Ok(GlobalModularState::Ans { code, predictor_id })
    } else {
        // Data histogram with HybridUint {4,2,0} + Huffman
        let (depths, codes) = write_hybrid_data_histogram(writer, histogram, max_token)?;

        // Write GlobalModular's ModularHeader
        writer.write(1, 1)?; // use_global_tree = true
        writer.write(1, 1)?; // wp_params.default_wp = true
        write_global_transforms_full(writer, &transforms)?;

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
            predictor_id,
        })
    }
}

/// Writes the global modular section with a learned MA tree for multi-group encoding.
///
/// This writes:
/// - dc_quant (all_default=1, or custom if dc_quant_custom is Some)
/// - has_tree = 1
/// - Learned tree (write_tree)
/// - lz77.enabled = 0
/// - Multi-context ANS data histogram (write_entropy_code_ans)
/// - GroupHeader (use_global_tree=1, wp_header.all_default=1, num_transforms=0)
pub fn write_global_modular_section_with_tree(
    images: &[ModularImage],
    writer: &mut BitWriter,
    profile: &crate::effort::EffortProfile,
    transforms: GlobalTransforms,
    use_lz77: bool,
    lz77_method: crate::entropy_coding::lz77::Lz77Method,
    meta_image: Option<&ModularImage>,
) -> Result<GlobalModularState> {
    write_global_modular_section_with_tree_dc_quant(
        images,
        writer,
        profile,
        transforms,
        use_lz77,
        lz77_method,
        None,
        meta_image,
    )
}

/// Like [`write_global_modular_section_with_tree`] but with custom dc_quant for LfFrame.
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_global_modular_section_with_tree_dc_quant(
    images: &[ModularImage],
    writer: &mut BitWriter,
    profile: &crate::effort::EffortProfile,
    transforms: GlobalTransforms,
    use_lz77: bool,
    lz77_method: crate::entropy_coding::lz77::Lz77Method,
    dc_quant_custom: Option<[f32; 3]>,
    meta_image: Option<&ModularImage>,
) -> Result<GlobalModularState> {
    use super::encode::write_tree;
    use super::encode::write_wp_header;
    use super::predictor::WeightedPredictorParams;
    use super::tree::Tree;
    use super::tree::count_contexts;
    use super::tree_learn::{
        TreeLearningParams, TreeSamples, collect_residuals_with_tree, compute_best_tree,
        compute_gather_stride_from_profile, derive_seeded_max_property_values,
        derive_seeded_params, derive_seeded_properties_truncation, derive_seeded_sample_fraction,
        derive_seeded_stride, estimate_token_cost, gather_samples_strided,
        gather_samples_strided_with_dedup_backend, gather_samples_strided_with_offset,
        max_ref_channels, stride_for_seeded_sample_fraction,
    };
    use crate::entropy_coding::encode::build_entropy_code_ans_with_options;
    use crate::entropy_coding::encode::write_entropy_code_ans;
    use crate::entropy_coding::lz77::write_lz77_header;

    // Step 0: Find best WP parameters (effort-dependent search)
    let all_channels: Vec<&super::channel::Channel> = meta_image
        .into_iter()
        .chain(images.iter())
        .flat_map(|img| img.channels.iter())
        .collect();
    let wp_params = crate::profile_time!("modular/wp_params_search", {
        if profile.wp_num_param_sets > 0 {
            // Collect channel references for cost estimation
            let channels_for_wp: Vec<super::channel::Channel> =
                all_channels.iter().map(|c| (*c).clone()).collect();
            super::predictor::find_best_wp_params(&channels_for_wp, profile.wp_num_param_sets)
        } else {
            WeightedPredictorParams::default()
        }
    });

    // Step 1: Gather samples from all groups (with subsampling for large images)
    let total_pixels: usize = meta_image
        .into_iter()
        .chain(images.iter())
        .flat_map(|img| img.channels.iter())
        .map(|ch| ch.width() * ch.height())
        .sum();
    let stride = compute_gather_stride_from_profile(total_pixels, profile);
    // Compute max ref channels across all images for cross-channel prediction
    let num_refs = {
        let mut mr = 0;
        if let Some(meta) = meta_image {
            mr = mr.max(max_ref_channels(meta));
        }
        for img in images.iter() {
            mr = mr.max(max_ref_channels(img));
        }
        mr
    };
    let per_group_id_offset = if meta_image.is_some() { 1u32 } else { 0u32 };
    // Phase 2 of issue #41: when the profile asks for gather-time dedup,
    // every per-group task gets its own `GatherDedupTable`. Concatenation
    // via `append_from` joins the per-task sample_counts; the post-gather
    // sort dedup then collapses cross-task duplicates (and any bucket-
    // equivalence collisions the raw-value hash missed).
    //
    // The dedup hash mirrors `params.properties` (post-y/x skip) so the
    // merge is provably at-or-below the post-sort merge in
    // aggressiveness — every gather-time match would also have collapsed
    // under the bucket-key sort, just possibly with other rows.
    let enable_gather_dedup = profile.gather_dedup;
    // Phase 3 of issue #41: switch the gather-time dedup table to
    // [`InlineDedupTable`]. Only meaningful when `enable_gather_dedup` is
    // also `true` — the backend wrapper falls back to Phase 2 at
    // construction time when the inline-key packing budget can't hold
    // the configured property × predictor count.
    let enable_phase3 = profile.gather_dedup_phase3;
    let dedup_properties: Vec<usize> = if enable_gather_dedup {
        // Borrow the same property list `compute_best_tree` will build
        // from this profile so the gather hash uses the same slot set.
        TreeLearningParams::from_profile(profile)
            .with_ref_properties(num_refs, profile.effort)
            .properties
            .clone()
    } else {
        Vec::new()
    };

    // Closure: gather samples for a given seed, with optional per-seed
    // stride override (RFC#45 chunk 3 + chunk 4). Seed 0 always uses the
    // canonical `stride` + `start_offset = 0` and the canonical
    // `CANDIDATE_PREDICTORS` order so seed-0 is byte-identical to the
    // pre-RFC#45-chunk-2 gather. Higher seeds shift the offset, may use
    // a perturbed `seed_stride` (different sample density), and pick a
    // per-seed predictor permutation (chunk-4 evaluation-order variance).
    let gather_for_seed = |seed: u64, seed_stride: usize| -> TreeSamples {
        let start_offset = if seed_stride > 1 {
            (seed as usize) % seed_stride
        } else {
            0
        };
        // Chunk 4: predictor-order shuffle — seed 0 yields the canonical
        // `CANDIDATE_PREDICTORS`, higher seeds rotate through 3 alternate
        // permutations. All TreeSamples merged via `append_from` MUST use
        // the SAME predictor order (the assert at append_from line 506-510
        // checks lengths; the SoA columns are predictor-indexed so they
        // must agree).
        let mut samples = TreeSamples::new_with_predictor_order_for_seed(num_refs, seed);
        // Meta channels first.
        if let Some(meta) = meta_image {
            if enable_gather_dedup && seed == 0 {
                // Gather-time dedup only honors the canonical seed-0 path.
                // Higher seeds skip dedup — the post-sort arbiter still
                // collapses bucket-equivalent rows downstream.
                let _ = gather_samples_strided_with_dedup_backend(
                    &mut samples,
                    meta,
                    0,
                    0,
                    seed_stride,
                    &wp_params,
                    None,
                    true,
                    enable_phase3,
                    &dedup_properties,
                );
            } else if start_offset == 0 {
                gather_samples_strided(&mut samples, meta, 0, 0, seed_stride, &wp_params);
            } else {
                gather_samples_strided_with_offset(
                    &mut samples,
                    meta,
                    0,
                    0,
                    seed_stride,
                    start_offset,
                    &wp_params,
                );
            }
        }
        // Per-group gather — embarrassingly parallel across groups.
        let per_group_samples: Vec<TreeSamples> =
            crate::parallel::parallel_map(images.len(), |group_idx| {
                // Same per-seed predictor order as the meta init above.
                let mut local = TreeSamples::new_with_predictor_order_for_seed(num_refs, seed);
                if enable_gather_dedup && seed == 0 {
                    let _ = gather_samples_strided_with_dedup_backend(
                        &mut local,
                        &images[group_idx],
                        group_idx as u32 + per_group_id_offset,
                        0,
                        seed_stride,
                        &wp_params,
                        None,
                        true,
                        enable_phase3,
                        &dedup_properties,
                    );
                } else if start_offset == 0 {
                    gather_samples_strided(
                        &mut local,
                        &images[group_idx],
                        group_idx as u32 + per_group_id_offset,
                        0,
                        seed_stride,
                        &wp_params,
                    );
                } else {
                    gather_samples_strided_with_offset(
                        &mut local,
                        &images[group_idx],
                        group_idx as u32 + per_group_id_offset,
                        0,
                        seed_stride,
                        start_offset,
                        &wp_params,
                    );
                }
                local
            });
        // Reserve total capacity up-front to avoid Vec growth during merge,
        // then concatenate in deterministic group-index order.
        let total_extra: usize = per_group_samples.iter().map(|s| s.num_samples).sum();
        samples.reserve(total_extra);
        for local in per_group_samples {
            samples.append_from(local);
        }
        samples
    };

    // Multi-seed dispatch — RFC#45 chunk 2 (start-offset variance), chunk 3
    // (broader variance: stride / split_threshold / property-order), chunk
    // 4 (sample-fraction jitter + predictor-evaluation-order shuffle),
    // chunk 5 (seed-slot split + budget expansion at e11), and chunk 6
    // (split-bucket-count + properties-slice truncation, e11 budget
    // doubled again 8 → 16).
    //
    // At e ≤ 9 `tree_learn_seeds = 1` (libjxl-equivalent, byte-identical
    // hash-locks). At e10 we fan out 2 seeded runs; at e11 chunk 6 fans
    // out 16 (was 8) and pick the tree whose tokens have the lowest
    // entropy cost.
    //
    // Chunk 6 seed-slot layout (relevant when `seeds >= 4`):
    //   - seeds 0..=3:   chunk-3 perturbations active
    //                    (split_threshold jitter via [`derive_seeded_params`],
    //                    property-order rotation, per-seed stride via
    //                    [`derive_seeded_stride`]). Chunks 4/5/6 helpers
    //                    are no-ops here.
    //   - seeds 4..=7:   chunk-4 perturbations active on top of chunk-3
    //                    ([`derive_seeded_sample_fraction`] takes precedence
    //                    over [`derive_seeded_stride`] when it returns
    //                    Some(_); [`derive_seeded_predictor_order`] cycles
    //                    through 4 permutations of [`CANDIDATE_PREDICTORS`]).
    //                    Chunk-6 helpers are no-ops here.
    //   - seeds 8..=11:  chunk-6 split-bucket-count override active
    //                    ([`derive_seeded_max_property_values`] cycles
    //                    through Some(64) / Some(128) / Some(192) / None
    //                    — coarser-than-canonical bucket grids for the
    //                    `find_best_split` value quantization). Chunk-4
    //                    helpers held to canonical no-op.
    //   - seeds 12..=15: chunk-6 properties-slice truncation active
    //                    ([`derive_seeded_properties_truncation`] cycles
    //                    through Some(8) / Some(10) / Some(12) / None —
    //                    a structural-regularization fallback that forces
    //                    the greedy builder to choose among fewer
    //                    high-information properties first). Chunk-4 +
    //                    chunk-6-bucket helpers held to canonical no-op.
    //   - seed 0 stays the canonical libjxl run for all dimensions.
    //
    // The split-per-dimension pattern (chunk-3 → chunk-4 → chunk-5 →
    // chunk-6) preserves the wins of each prior chunk in dedicated seed
    // slots, exactly as chunk 5 reserved seeds 0..=3 for chunk-3-only
    // perturbations after chunk 4's recombined-budget regression.
    let seeds = profile.tree_learn_seeds.max(1);
    let mut best: Option<(Vec<crate::entropy_coding::token::Token>, usize, Tree, f64)> = None;

    // Baseline params shared across seeds. derive_seeded_params clones and
    // mutates per-seed; the with_pixel_fraction call is per-seed because
    // pixel_fraction depends on the actual gathered weight, which varies
    // with stride.
    let base_params = TreeLearningParams::from_profile(profile)
        .with_ref_properties(num_refs, profile.effort)
        .with_total_pixels(total_pixels);

    for seed in 0..(seeds as u64) {
        // Chunk 4: per-seed sample-fraction override takes precedence
        // over chunk-3's stride perturbation. Seed 0 returns None → fall
        // through to the chunk-3 path. Higher seeds with Some(frac) map
        // an absolute target fraction onto a stride; the override is a
        // no-op when total_pixels already fits under the 65 K floor.
        let seed_stride = match derive_seeded_sample_fraction(seed) {
            Some(frac) => stride_for_seeded_sample_fraction(total_pixels, frac),
            None => derive_seeded_stride(stride, seed),
        };
        // Gather + tree-learn for this seed.
        let mut samples = crate::profile_time!("modular/gather_samples", {
            gather_for_seed(seed, seed_stride)
        });

        let pixel_fraction = if total_pixels > 0 {
            samples.total_gathered_weight() as f64 / total_pixels as f64
        } else {
            1.0
        };
        // Per-seed parameter variance — seed 0 is a no-op clone.
        let mut params =
            derive_seeded_params(&base_params, seed).with_pixel_fraction(pixel_fraction);
        // Chunk-6 dim A — split-bucket-count override (seeds 8..=11).
        // Seeds 0..=7 return None → canonical bucket count preserved.
        if let Some(buckets) = derive_seeded_max_property_values(seed) {
            params.max_property_values = buckets;
        }
        // Chunk-6 dim B — properties-slice truncation (seeds 12..=15).
        // Seeds 0..=11 return None → canonical slice length preserved.
        // Clamp to current slice length so a truncation cap longer than
        // the property Vec is a no-op rather than an invalid index.
        if let Some(prop_cap) = derive_seeded_properties_truncation(seed) {
            let cap = prop_cap.min(params.properties.len());
            params.properties.truncate(cap);
        }
        let tree = crate::profile_time!("modular/compute_best_tree", {
            compute_best_tree(&mut samples, &params)
        });

        crate::trace::debug_eprintln!(
            "GLOBAL_MODULAR_TREE seed={}/{}: {} nodes from {} samples \
             (pixel_fraction={:.3}, stride={}, split_threshold={:.2}, \
              max_property_values={}, properties.len={})",
            seed,
            seeds,
            tree.len(),
            samples.num_samples,
            pixel_fraction,
            seed_stride,
            params.split_threshold,
            params.max_property_values,
            params.properties.len(),
        );

        // Collect residuals for this candidate tree.
        let (all_tokens, nb_meta_tokens) =
            crate::profile_time!("modular/collect_residuals_global", {
                let per_group_tokens: Vec<Vec<crate::entropy_coding::token::Token>> =
                    crate::parallel::parallel_map(images.len(), |group_idx| {
                        collect_residuals_with_tree(
                            &images[group_idx],
                            &tree,
                            group_idx as u32 + per_group_id_offset,
                            &wp_params,
                        )
                    });
                let meta_tokens_opt =
                    meta_image.map(|meta| collect_residuals_with_tree(meta, &tree, 0, &wp_params));
                let nb_meta_tokens = meta_tokens_opt.as_ref().map(|t| t.len()).unwrap_or(0);
                let total_len: usize =
                    nb_meta_tokens + per_group_tokens.iter().map(|t| t.len()).sum::<usize>();
                let mut all_tokens =
                    Vec::<crate::entropy_coding::token::Token>::with_capacity(total_len);
                if let Some(meta_tokens) = meta_tokens_opt {
                    all_tokens.extend(meta_tokens);
                }
                for tokens in per_group_tokens {
                    all_tokens.extend(tokens);
                }
                (all_tokens, nb_meta_tokens)
            });

        // Score: skip cost estimate entirely when seeds == 1 (the legacy
        // single-pass path doesn't need to know the cost). For seeds > 1
        // we compute the entropy cost (with per-context header term,
        // see `estimate_token_cost`) and keep the cheapest candidate.
        if seeds == 1 {
            best = Some((all_tokens, nb_meta_tokens, tree, 0.0));
            break;
        }
        let cost = estimate_token_cost(&all_tokens);
        crate::trace::debug_eprintln!(
            "MULTI_SEED_TREE_PICK seed={}/{} cost={:.0} bits ({} tokens, {} nodes)",
            seed,
            seeds,
            cost,
            all_tokens.len(),
            tree.len(),
        );
        match best {
            None => best = Some((all_tokens, nb_meta_tokens, tree, cost)),
            Some((_, _, _, prev_cost)) if cost < prev_cost => {
                best = Some((all_tokens, nb_meta_tokens, tree, cost));
            }
            _ => {}
        }
    }

    let (all_tokens, nb_meta_tokens, tree, _final_cost) =
        best.expect("seeds >= 1 guarantees at least one candidate");
    let num_contexts = count_contexts(&tree) as usize;

    // Note: LZ77 is NOT applied in this path. The per-group sections
    // (write_group_modular_section) re-collect tokens independently without LZ77.
    // Applying LZ77 to the combined stream would cause a histogram mismatch because
    // the ANS code would include LZ77 symbols that per-group sections don't emit.
    // The squeeze multi-group path (frame.rs) handles LZ77 correctly per-section.
    let _ = (use_lz77, lz77_method); // suppress unused warnings
    let lz77_params: Option<crate::entropy_coding::lz77::Lz77Params> = None;
    let ans_num_contexts = if lz77_params.is_some() {
        num_contexts + 1
    } else {
        num_contexts
    };

    // Step 4: Build multi-context ANS code with enhanced clustering
    let code = crate::profile_time!("modular/build_ans_code", {
        build_entropy_code_ans_with_options(
            &all_tokens,
            ans_num_contexts,
            true, // enhanced clustering (pair-merge refinement)
            true, // optimize uint configs
            lz77_params.as_ref(),
            Some(total_pixels),
        )
    });

    // Per-seed diagnostics are emitted inside the loop via the
    // `MULTI_SEED_TREE_PICK` trace; here we just summarise the picked tree.
    eprintln!(
        "DIAG tree: {} nodes, {} contexts, {} total_tokens (seeds={})",
        tree.len(),
        num_contexts,
        all_tokens.len(),
        seeds,
    );
    eprintln!(
        "DIAG code: {} histograms (from {} contexts), rct={:?}, compact={}",
        code.histograms.len(),
        ans_num_contexts,
        transforms.rct_type,
        transforms.compact_info.len(),
    );

    // Step 5: Write bitstream
    let bits_before = writer.bits_written();
    crate::f16::write_lf_quant(writer, dc_quant_custom)?;
    // has_tree = true
    writer.write(1, 1)?;

    // Write the learned tree
    let bits_before_tree = writer.bits_written();
    write_tree(writer, &tree)?;
    let tree_bits = writer.bits_written() - bits_before_tree;

    // Write LZ77 header + ANS data histogram.
    let bits_before_histo = writer.bits_written();
    if ans_num_contexts > 1 {
        write_lz77_header(lz77_params.as_ref(), writer)?;
        write_entropy_code_ans(&code, writer)?;
    } else {
        write_ans_modular_header(writer, &code)?;
    }
    let histo_bits = writer.bits_written() - bits_before_histo;

    // GroupHeader (global modular group)
    writer.write(1, 1)?; // use_global_tree = true
    write_wp_header(writer, &wp_params)?;
    write_global_transforms_full(writer, &transforms)?;

    // Write meta-channel tokens (palette data) in the global section, after GroupHeader.
    // These are part of the global modular image — they stay whole (not split across groups).
    //
    // Even when nb_meta_tokens == 0, we still emit a 32-bit ANS initial state so the
    // section forms a valid (empty) ANS stream. Pre-fix jxl-oxide always calls
    // `decoder.begin()` here regardless of buffers — without these bits we'd EOF
    // mid-LfGlobal. libjxl is bug-compatible by writing the same 32 bits via its
    // `WriteTokens`/`ANSCoder::Flush` codepath. djxl and jxl-rs short-circuit before
    // reading the state when there are no decodable channels in this section (the
    // `num_chans == 0` / `is_empty` early-returns), so the extra 4 bytes are simply
    // padding to them. See `imazen/jxl-oxide@fd4e2c3` for the matching decoder fix.
    let meta_token_slice = &all_tokens[..nb_meta_tokens];
    write_tokens_ans(meta_token_slice, &code, None, writer)?;

    let total_lf_global_bits = writer.bits_written() - bits_before;
    eprintln!(
        "DIAG LfGlobal: tree={} bits ({} B), histo={} bits ({} B), \
         meta_tokens={}, total={} bits ({} B)",
        tree_bits,
        tree_bits / 8,
        histo_bits,
        histo_bits / 8,
        nb_meta_tokens,
        total_lf_global_bits,
        total_lf_global_bits / 8,
    );

    writer.zero_pad_to_byte();

    Ok(GlobalModularState::AnsWithTree {
        code,
        tree,
        wp_params,
    })
}

/// Info about global transforms to write in the LfGlobal GroupHeader.
pub struct GlobalTransforms {
    /// Per-channel ChannelCompact transforms: (begin_c, nb_colors).
    pub compact_info: Vec<(usize, usize)>,
    /// Optional RCT type (begin_c is adjusted for ChannelCompact meta channels).
    pub rct_type: Option<RctType>,
}

impl GlobalTransforms {
    pub fn rct_only(rct_type: Option<RctType>) -> Self {
        Self {
            compact_info: Vec::new(),
            rct_type,
        }
    }
}

/// Write num_transforms + transform descriptors for the global GroupHeader.
///
/// When `compact_info` is present, writes ChannelCompact (kPalette with num_c=1)
/// transforms first, then RCT with begin_c shifted by the number of compact meta channels.
fn write_global_transforms_full(
    writer: &mut BitWriter,
    transforms: &GlobalTransforms,
) -> Result<()> {
    let num_transforms =
        transforms.compact_info.len() as u32 + transforms.rct_type.is_some() as u32;
    super::encode::write_num_transforms(writer, num_transforms)?;

    // ChannelCompact transforms first (per-channel palette, num_c=1)
    for &(begin_c, nb_colors) in &transforms.compact_info {
        write_palette_transform(writer, begin_c, 1, nb_colors, 0, 0)?;
    }
    // RCT (begin_c adjusted for ChannelCompact meta channels)
    if let Some(rct) = transforms.rct_type {
        let rct_begin_c = transforms.compact_info.len();
        write_rct_transform(writer, rct_begin_c, rct)?;
    }
    Ok(())
}

/// Collect packed residuals from a group image using gradient prediction.
#[allow(dead_code)] // legacy wrapper retained for hash-lock parity
fn collect_group_residuals(group_image: &ModularImage) -> Vec<u32> {
    collect_group_residuals_with_predictor(group_image, 5)
}

/// Knob-aware variant of [`collect_group_residuals`] that honours the
/// libjxl `--modular_predictor` override.
fn collect_group_residuals_with_predictor(
    group_image: &ModularImage,
    predictor_id: u8,
) -> Vec<u32> {
    let mut residuals = Vec::new();
    for channel in &group_image.channels {
        let width = channel.width();
        let height = channel.height();
        for y in 0..height {
            for x in 0..width {
                let pixel = channel.get(x, y);
                let prediction = predict_pixel_with_id(channel, x, y, predictor_id);
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
    write_group_modular_section_idx(group_image, state, 0, &GroupTransforms::none(), writer)
}

/// Like [`write_group_modular_section`] but with an explicit group index
/// for tree property 1 (group_id). Required when the learned tree splits on group_id.
///
/// `rct_type`: Optional per-group RCT transform to write in this group's GroupHeader.
/// When `Some`, the group data is assumed to be already RCT-transformed and the
/// decoder will apply inverse RCT when decoding this group.
/// Per-group transform info for ChannelCompact + RCT.
#[derive(Clone)]
pub struct GroupTransforms {
    /// Per-channel ChannelCompact transforms: (begin_c, nb_colors).
    pub compact_info: Vec<(usize, usize)>,
    /// Optional RCT type (begin_c is adjusted for ChannelCompact meta channels).
    pub rct_type: Option<RctType>,
}

impl GroupTransforms {
    pub fn none() -> Self {
        Self {
            compact_info: Vec::new(),
            rct_type: None,
        }
    }
}

pub fn write_group_modular_section_idx(
    group_image: &ModularImage,
    state: &GlobalModularState,
    group_idx: u32,
    transforms: &GroupTransforms,
    writer: &mut BitWriter,
) -> Result<()> {
    crate::trace::debug_eprintln!(
        "GROUP_MODULAR [bit {}]: Starting group section ({}x{}, compact={}, rct={:?})",
        writer.bits_written(),
        group_image.width(),
        group_image.height(),
        transforms.compact_info.len(),
        transforms.rct_type,
    );

    // GroupHeader
    writer.write(1, 1)?; // use_global_tree = true
    // Write WP params matching the global section's params
    match state {
        GlobalModularState::AnsWithTree { wp_params, .. } => {
            super::encode::write_wp_header(writer, wp_params)?;
        }
        _ => {
            writer.write(1, 1)?; // wp_params.default_wp = true
        }
    }
    // Per-group transforms: ChannelCompact(s) + optional RCT
    let num_transforms =
        transforms.compact_info.len() as u32 + transforms.rct_type.is_some() as u32;
    super::encode::write_num_transforms(writer, num_transforms)?;
    for &(begin_c, nb_colors) in &transforms.compact_info {
        write_palette_transform(writer, begin_c, 1, nb_colors, 0, 0)?;
    }
    if let Some(rct) = transforms.rct_type {
        let rct_begin_c = transforms.compact_info.len();
        write_rct_transform(writer, rct_begin_c, rct)?;
    }

    match state {
        GlobalModularState::Huffman {
            depths,
            codes,
            max_token: _,
            predictor_id,
        } => {
            // Encode residuals with HybridUint {4,2,0} + Huffman, honouring
            // the global section's forced predictor (libjxl `cjxl -P`).
            let predictor_id = *predictor_id;
            for channel in &group_image.channels {
                let width = channel.width();
                let height = channel.height();
                for y in 0..height {
                    for x in 0..width {
                        let pixel = channel.get(x, y);
                        let prediction = predict_pixel_with_id(channel, x, y, predictor_id);
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
        GlobalModularState::Ans { code, predictor_id } => {
            // Collect residuals for this group and encode with ANS
            let residuals = collect_group_residuals_with_predictor(group_image, *predictor_id);
            let tokens: Vec<AnsToken> = residuals.iter().map(|&r| AnsToken::new(0, r)).collect();
            write_tokens_ans(&tokens, code, None, writer)?;
        }
        GlobalModularState::AnsWithTree {
            code,
            tree,
            wp_params,
        } => {
            // Collect residuals using the learned tree (multi-context).
            // Per-group images use 0-based channel indices (matching the decoder,
            // which builds per-group images with only non-meta channels).
            let tokens = crate::profile_time!("modular/collect_residuals_per_group", {
                super::tree_learn::collect_residuals_with_tree(
                    group_image,
                    tree,
                    group_idx,
                    wp_params,
                )
            });
            crate::profile_time!("modular/write_tokens_per_group", {
                write_tokens_ans(&tokens, code, None, writer)?;
            });
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
