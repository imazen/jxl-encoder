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
/// How many groups' `TreeSamples` may be live at once during the tree-learning
/// gather, before they are merged into the accumulator and dropped.
///
/// The gather is embarrassingly parallel across groups, but holding every
/// group's samples until a single trailing merge doubles the peak: the
/// per-group vec and the merged accumulator are both resident. Merging in
/// waves bounds that transient without changing the merged result (waves are
/// consumed in ascending group order; `append_from` concatenates).
///
/// The wave is the live transient: at the 4K/e9 per-group size of ~11.6 MiB,
/// 64 groups is ~740 MB in flight, 16 is ~185 MB, 8 is ~93 MB. It only needs to
/// be wide enough to keep the worker pool fed — the useful thread count is the
/// floor, and anything past a small multiple of it buys throughput nothing
/// while costing peak linearly.
///
/// MEASURED 2026-08-13: 16 was chosen by reasoning ("wide enough to keep the
/// pool fed") and was wrong. A backtrace captured AT the instant `peak_live` is
/// set — not from an RSS-polled snapshot, which samples a different moment and
/// had been misattributing this — puts the lossless peak inside this very
/// `parallel_map`. Re-swept with that knowledge (3840x2160 lossless,
/// byte-identical at every setting, alloc count and wall time flat):
///
///   wave  e7 peak_live  e9 peak_live
///     16      1966 MB      1966 MB
///      8      1873 MB         -
///      4      1851 MB      1898 MB
///      2      1851 MB         -
///      1      1851 MB         -
///
/// 4 captures the whole available reduction and flattens below it, so it is the
/// knee rather than a guess. Wall is unchanged (14.7 s e7 / 84.0 s e9), so the
/// extra barriers cost nothing measurable — the "keep every worker fed" concern
/// that motivated 16 was unfounded at this width.
///
/// Overridable at runtime via `JXL_TREE_GATHER_WAVE` (sweep knob; unset in
/// production). The value is observationally invisible — waves are consumed in
/// ascending group order, so the merged result is identical at every setting.
const GATHER_MERGE_WAVE_GROUPS: usize = 4;

/// `JXL_TREE_GATHER_WAVE=<n>` overrides [`GATHER_MERGE_WAVE_GROUPS`].
fn gather_wave_groups() -> usize {
    use std::sync::OnceLock;
    static WAVE: OnceLock<usize> = OnceLock::new();
    *WAVE.get_or_init(|| {
        std::env::var("JXL_TREE_GATHER_WAVE")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|n| *n > 0)
            .unwrap_or(GATHER_MERGE_WAVE_GROUPS)
    })
}

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
        /// Per-section LZ77 (issue #69 item 1). `Some` means the global
        /// entropy code was built over LZ77-transformed per-section token
        /// streams, and every per-group section MUST re-apply the same
        /// `(method, dist_multiplier)` transform at write time — the
        /// decoder creates a fresh LZ77 state per section with
        /// `dist_multiplier = max(section channel widths)`. `None` means
        /// LZ77 is off for this frame (the pre-#69 behaviour).
        lz77: Option<SectionLz77>,
    },
}

/// Per-section LZ77 configuration carried by
/// [`GlobalModularState::AnsWithTree`]: the schedule's method plus the
/// header params the global entropy code was built with.
pub type SectionLz77 = (
    crate::entropy_coding::lz77::Lz77Method,
    crate::entropy_coding::lz77::Lz77Params,
);

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

/// Number of quant-table modular streams in the spec stream numbering
/// (libjxl `DequantMatrices::kNum`; zenjxl-decoder
/// `quantizer::NUM_QUANT_TABLES`).
pub(crate) const NUM_QUANT_TABLE_STREAMS: u32 = 17;

/// First ModularHF stream id (pass 0, group 0) in the spec's modular
/// stream numbering — the value decoders feed into tree property 1
/// (`group_id`) when decoding a pass-group section: stream 0 is
/// GlobalData, then `num_lf_groups` each of VarDCT-LF / ModularLF /
/// LFMeta, then the 17 quant-table streams. (#68 second cause: the
/// encoder used ad-hoc `meta_offset + group_idx` ids for gather/apply,
/// which desynced every decoder whenever an e9+ tree split on property
/// 1 — only reachable on multi-group images, since a single group makes
/// the property constant and unsplittable.)
pub(crate) fn modular_hf_stream_id_base(num_lf_groups: u32) -> u32 {
    1 + 3 * num_lf_groups + NUM_QUANT_TABLE_STREAMS
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
#[allow(dead_code)]
// public wrapper retained for API stability;
// internal callers route through `write_global_modular_section_with_tree_knobs`
#[allow(clippy::too_many_arguments)] // mirrors the knobs variant's signature
pub fn write_global_modular_section_with_tree(
    images: &[ModularImage],
    writer: &mut BitWriter,
    profile: &crate::effort::EffortProfile,
    transforms: GlobalTransforms,
    use_lz77: bool,
    lz77_method: crate::entropy_coding::lz77::Lz77Method,
    meta_image: Option<&ModularImage>,
    hf_stream_id_base: u32,
    budget: Option<&alloc::sync::Arc<crate::budget::MemoryBudget>>,
) -> Result<GlobalModularState> {
    write_global_modular_section_with_tree_dc_quant_knobs(
        images,
        writer,
        profile,
        transforms,
        use_lz77,
        lz77_method,
        None,
        meta_image,
        &super::palette::ModularKnobs::default(),
        hf_stream_id_base,
        budget,
    )
}

/// Knob-aware variant of [`write_global_modular_section_with_tree`].
///
/// When [`super::palette::ModularKnobs::modular_predictor`] resolves to a
/// concrete `0..=13` predictor (excluding `5` Gradient — the default
/// keeps the legacy ID3 path for hash-lock parity, and `14`/`15` are
/// libjxl's `Best`/`Variable` meta-modes that explicitly request
/// per-leaf selection), the tree learner is skipped entirely and a
/// single-leaf tree with the requested predictor is emitted. Per-group
/// residual collection in [`write_group_modular_section`] picks up the
/// override via the [`GlobalModularState::AnsWithTree`] tree handle,
/// keeping the bitstream self-consistent end-to-end.
///
/// Mirrors libjxl `cjxl -P N` / `--modular_predictor`: forcing the leaf
/// predictor in the tree-learn path matches the libjxl behaviour where
/// `options.predictor` overrides what would otherwise be the tree
/// learner's per-leaf choice.
#[allow(clippy::too_many_arguments)]
pub fn write_global_modular_section_with_tree_knobs(
    images: &[ModularImage],
    writer: &mut BitWriter,
    profile: &crate::effort::EffortProfile,
    transforms: GlobalTransforms,
    use_lz77: bool,
    lz77_method: crate::entropy_coding::lz77::Lz77Method,
    meta_image: Option<&ModularImage>,
    knobs: &super::palette::ModularKnobs,
    hf_stream_id_base: u32,
    budget: Option<&alloc::sync::Arc<crate::budget::MemoryBudget>>,
) -> Result<GlobalModularState> {
    write_global_modular_section_with_tree_dc_quant_knobs(
        images,
        writer,
        profile,
        transforms,
        use_lz77,
        lz77_method,
        None,
        meta_image,
        knobs,
        hf_stream_id_base,
        budget,
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
    hf_stream_id_base: u32,
    budget: Option<&alloc::sync::Arc<crate::budget::MemoryBudget>>,
) -> Result<GlobalModularState> {
    write_global_modular_section_with_tree_dc_quant_knobs(
        images,
        writer,
        profile,
        transforms,
        use_lz77,
        lz77_method,
        dc_quant_custom,
        meta_image,
        &super::palette::ModularKnobs::default(),
        hf_stream_id_base,
        budget,
    )
}

/// Knob-aware + LfFrame-aware variant of
/// [`write_global_modular_section_with_tree`]. See
/// [`write_global_modular_section_with_tree_knobs`] for the override
/// semantics.
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_global_modular_section_with_tree_dc_quant_knobs(
    images: &[ModularImage],
    writer: &mut BitWriter,
    profile: &crate::effort::EffortProfile,
    transforms: GlobalTransforms,
    use_lz77: bool,
    lz77_method: crate::entropy_coding::lz77::Lz77Method,
    dc_quant_custom: Option<[f32; 3]>,
    meta_image: Option<&ModularImage>,
    knobs: &super::palette::ModularKnobs,
    hf_stream_id_base: u32,
    budget: Option<&alloc::sync::Arc<crate::budget::MemoryBudget>>,
) -> Result<GlobalModularState> {
    use super::encode::write_tree;
    use super::encode::write_wp_header;
    use super::predictor::WeightedPredictorParams;
    use super::tree::Tree;
    use super::tree::count_contexts;
    use super::tree_learn::{
        MULTI_SEED_EARLY_OUT_PROBE_SEEDS, TreeLearningParams, TreeSamples,
        collect_residuals_with_tree, compute_best_tree, compute_gather_stride_from_profile,
        derive_seeded_max_property_values, derive_seeded_params,
        derive_seeded_properties_truncation, derive_seeded_sample_fraction, derive_seeded_stride,
        estimate_token_cost, gather_samples_strided, gather_samples_strided_with_dedup_backend,
        gather_samples_strided_with_offset, max_ref_channels, multi_seed_early_out_after_probe,
        stride_for_seeded_sample_fraction,
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
    // Property-1 (`group_id`) values MUST be the spec stream ids the
    // decoder evaluates (#68 second cause): pass-group g (single pass)
    // is `hf_stream_id_base + g`. The meta/global channels use stream 0
    // (GlobalData). The old ad-hoc `meta_offset + group_idx` numbering
    // desynced any e9+ tree that split on property 1.
    let per_group_id_offset = hf_stream_id_base;
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
    // Property-column storage mask for this path: the canonical
    // `TreeLearningParams::properties` list is the ONLY set of raw property
    // columns pre-quantize/dedup/tree-build read here (seed perturbations
    // permute or truncate it, never extend it), so the gather skips storing
    // the rest — at 4K e7 that is 15-17 of 24 columns, ~400 MiB of the
    // gather-phase peak. Columns outside the mask stay EMPTY, which every
    // consumer already skips. See `TreeSamples::active_props`.
    let active_prop_list: Vec<usize> = TreeLearningParams::from_profile(profile)
        .with_ref_properties(num_refs, profile.effort)
        .properties
        .clone();

    let gather_for_seed = |seed: u64, seed_stride: usize, randomize: bool| -> TreeSamples {
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
        let active_mask: alloc::boxed::Box<[bool]> = {
            let mut m = vec![false; samples.total_num_properties()].into_boxed_slice();
            for &p in &active_prop_list {
                if p < m.len() {
                    m[p] = true;
                }
            }
            m
        };
        samples.set_active_props(active_mask.clone());
        // Self-repair re-gather (task #14): draw de-aliased randomized
        // samples instead of the fixed stride. `false` on the normal path
        // ⇒ byte-identical.
        samples.randomize_gather = randomize;
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
        // Per-group gather — embarrassingly parallel across groups, but run in
        // WAVES so only `wave` groups' samples are live at once.
        //
        // Gathering all groups into one `Vec<TreeSamples>` and merging
        // afterwards costs a second full copy of every sample: the per-group
        // vec and the merged accumulator are both alive during the merge. On a
        // 4K lossless e9 encode that was the encoder's peak — the RSS timeline
        // spiked to ~5.0 GB in the first 6 s (gather+merge) before settling to
        // ~2.9 GB for the 70 s of actual tree learning. Merging each wave as it
        // completes bounds the transient to `wave` groups instead of all of
        // them, and the steady-state accumulator is unchanged.
        //
        // Byte-identical: waves are consumed in ascending group order and
        // `append_from` is a concatenation, so the merged column order is
        // exactly what the all-at-once merge produced. Each group's gather is
        // independent (fresh `local`, `group_idx` passed explicitly, no
        // cross-group state), so wave boundaries cannot change any sample.
        //
        // The wave is sized to keep every worker busy — an over-small wave
        // would serialize the gather at a barrier per wave.
        //
        // The accumulator is sized ONCE here, to an exact upper bound, so no
        // column ever reallocates as waves are merged in. Growing it per wave
        // would trade the old single duplicate for repeated 48 MiB-column
        // reallocations — each holding old+new at once and leaving a hole —
        // which costs more, and costs it on every allocator.
        let gather_upper_bound: usize = images
            .iter()
            .flat_map(|img| img.channels.iter())
            .map(|ch| (ch.width() * ch.height()).div_ceil(seed_stride.max(1)))
            .sum::<usize>()
            + samples.num_samples;
        samples.reserve_exact_total(gather_upper_bound);

        let wave = gather_wave_groups().max(1);
        let mut wave_start = 0usize;
        while wave_start < images.len() {
            let wave_len = wave.min(images.len() - wave_start);
            let wave_samples: Vec<TreeSamples> = crate::parallel::parallel_map(wave_len, |i| {
                let group_idx = wave_start + i;
                // Same per-seed predictor order as the meta init above.
                let mut local = TreeSamples::new_with_predictor_order_for_seed(num_refs, seed);
                local.set_active_props(active_mask.clone());
                local.randomize_gather = randomize;
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
            // No per-wave reserve: `reserve_exact_total` above already sized
            // every column past the final count, so these appends never grow.
            for local in wave_samples {
                samples.append_from(local);
            }
            wave_start += wave_len;
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
    // Tree-learn force-predictor override (libjxl `cjxl -P N` /
    // `--modular_predictor`). When the caller has asked for a fixed
    // `0..=13` predictor (excluding `5` Gradient — keeps hash-locks green
    // — and excluding `14`/`15` meta-modes), bypass ID3 entirely: build a
    // single-leaf tree with the requested predictor, collect residuals
    // against it once (no multi-seed search — all seeds would produce
    // identical output), and skip straight to ANS code building.
    //
    // Mirrors libjxl's behaviour where `options.predictor` overrides
    // what would otherwise be the tree learner's per-leaf choice. The
    // returned [`GlobalModularState::AnsWithTree`] carries the
    // single-leaf tree so per-group sections pick up the override via
    // the same per-pixel residual path used for ID3-learned trees.
    let force_predictor = super::encode::resolve_tree_learn_force_predictor(knobs);

    // RIGED meta-mode override (libjxl `cjxl -P 14` slot in our wiring —
    // Sharma 2018 Resolution-Independent Gradient-aware Edge Detection).
    // Mutually exclusive with `force_predictor` because `from_id(14) →
    // None` keeps `force_predictor = None` for id 14. When set, the
    // tree-learn path bypasses ID3 and uses a hand-crafted multi-leaf
    // tree implementing the RIGED switch rule. Like `force_predictor`,
    // the per-pixel residual collector + ANS code build path is shared
    // with the ID3-learned tree, so this is a tree-shape override only.
    //
    // bit_depth is taken from the first image (all images in a multi-
    // group encode share the same bit depth — see `ModularImage`).
    let riged_bit_depth = images.first().map(|img| img.bit_depth).unwrap_or(8);
    let riged_override = super::encode::resolve_tree_learn_riged_tree(knobs, riged_bit_depth);

    let seeds = if force_predictor.is_some() || riged_override.is_some() {
        1
    } else {
        profile.tree_learn_seeds.max(1)
    };
    /// Winning multi-seed candidate: (all_tokens, nb_meta_tokens,
    /// per-group token ranges within all_tokens, learned tree, cost).
    type SeedCandidate = (
        Vec<crate::entropy_coding::token::Token>,
        usize,
        Vec<core::ops::Range<usize>>,
        Tree,
        f64,
    );
    let mut best: Option<SeedCandidate> = None;

    // Perf (task #14 default-on enabler): when the self-repair runs it already
    // builds the winning tree's real clustered ANS code (in `real_ans_cost`),
    // which — LZ77 being off on the e5/e6 tree-lift path — is BYTE-IDENTICAL to
    // the Step-4 build below. Cache it here and reuse it downstream so the
    // self-repair costs NO extra entropy build on the common (non-aliased,
    // KEEP/SKIP) path. Only ever set at seeds==1 (the single tree-lift seed);
    // stays `None` when the self-repair does not run ⇒ default rebuild.
    let mut cached_winner_code: Option<OwnedAnsEntropyCode> = None;

    // Per-seed cost log for the chunk-7 early-out decision. Indexed by
    // completed seed (0..seeds). Populated inside the loop after the
    // entropy estimate is computed for seed >= 0. Capacity matches the
    // budget so the Vec never reallocates during the hot loop.
    let mut seed_costs: Vec<f64> = Vec::with_capacity(seeds.max(1) as usize);

    // Baseline params shared across seeds. derive_seeded_params clones and
    // mutates per-seed; the with_pixel_fraction call is per-seed because
    // pixel_fraction depends on the actual gathered weight, which varies
    // with stride.
    let base_params = TreeLearningParams::from_profile(profile)
        .with_ref_properties(num_refs, profile.effort)
        .with_total_pixels(total_pixels);

    // Collect residuals for a candidate tree across meta + every group,
    // returning the concatenated token stream, the meta prefix length, and
    // per-group ranges into it. Shared by the normal per-seed scoring and the
    // task-#14 self-repair (which collects a second, de-aliased candidate).
    let collect_for_tree = |tree: &Tree| -> (
        Vec<crate::entropy_coding::token::Token>,
        usize,
        Vec<core::ops::Range<usize>>,
    ) {
        let per_group_tokens: Vec<Vec<crate::entropy_coding::token::Token>> =
            crate::parallel::parallel_map(images.len(), |group_idx| {
                collect_residuals_with_tree(
                    &images[group_idx],
                    tree,
                    group_idx as u32 + per_group_id_offset,
                    &wp_params,
                )
            });
        let meta_tokens_opt =
            meta_image.map(|meta| collect_residuals_with_tree(meta, tree, 0, &wp_params));
        let nb_meta_tokens = meta_tokens_opt.as_ref().map(|t| t.len()).unwrap_or(0);
        let total_len: usize =
            nb_meta_tokens + per_group_tokens.iter().map(|t| t.len()).sum::<usize>();
        let mut all_tokens = Vec::<crate::entropy_coding::token::Token>::with_capacity(total_len);
        if let Some(meta_tokens) = meta_tokens_opt {
            all_tokens.extend(meta_tokens);
        }
        let mut group_ranges = Vec::with_capacity(images.len());
        for tokens in per_group_tokens {
            let start = all_tokens.len();
            all_tokens.extend(tokens);
            group_ranges.push(start..all_tokens.len());
        }
        (all_tokens, nb_meta_tokens, group_ranges)
    };

    // Real ANS-coded cost (histogram tables + coded tokens) of a candidate's
    // residual stream. Unlike `estimate_token_cost`'s per-context *ideal*
    // entropy, this runs the actual clustering + ANS build, so it captures the
    // ≤96-histogram clustering penalty that makes a stride-aliased tree code
    // far worse than its node count or ideal entropy imply (5336 e5: the ideal
    // estimate sees 1.7 %, the real ANS build sees ~27 %). Used only by the
    // task-#14 self-repair to pick between the fixed and de-aliased trees.
    let real_ans_cost = |tokens: &[crate::entropy_coding::token::Token],
                         tree: &Tree|
     -> Result<(usize, OwnedAnsEntropyCode)> {
        let num_contexts = count_contexts(tree) as usize;
        // Match the real write path's build (section.rs Step 4): enhanced
        // pair-merge clustering + uint-config optimization. The clustering
        // (≤ histogram cap) is precisely where a stride-aliased tree's many
        // thin contexts collapse into shared histograms and code badly —
        // the simple `build_entropy_code_ans` (near-ideal, uncapped) misses
        // it. LZ77 is off on the e5/e6 tree-lift path this repair fires on, so
        // this build is BYTE-IDENTICAL to the Step-4 build for the same tree —
        // the winning candidate's code is cached and reused there (perf).
        let code = build_entropy_code_ans_with_options(
            tokens,
            num_contexts.max(1),
            true, // enhanced clustering (pair-merge refinement)
            true, // optimize uint configs
            None, // LZ77 off on the tree-lift path
            Some(total_pixels),
        );
        if tokens.is_empty() {
            return Ok((0, code));
        }
        let mut w = BitWriter::new();
        write_entropy_code_ans(&code, &mut w)?;
        write_tokens_ans(tokens, &code, None, &mut w)?;
        Ok((w.bits_written(), code))
    };

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
        // Build seeded params from a gathered sample set. Shared by the base
        // per-seed tree-learn and the task-#14 self-repair re-gather so both
        // apply the same per-seed variance (pixel_fraction / bucket / prop caps).
        let build_params = |samples: &TreeSamples| -> TreeLearningParams {
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
            params
        };
        // Gather + tree-learn for this seed — skipped entirely when the
        // force-predictor override is active (the single-leaf tree below
        // does not consume samples, and gather is the dominant cost at
        // e7+). The RIGED override (predictor id 14) takes the same
        // gather-skip shortcut and substitutes a 3-leaf gradient-aware
        // tree instead of a single-leaf one.
        let tree = if let Some(forced) = force_predictor {
            super::tree::simple_tree(forced)
        } else if let Some(ref riged) = riged_override {
            riged.clone()
        } else {
            let mut samples = crate::profile_time!("modular/gather_samples", {
                gather_for_seed(seed, seed_stride, false)
            });
            let params = build_params(&samples);
            crate::profile_time!("modular/compute_best_tree", {
                compute_best_tree(&mut samples, &params)
            })
        };

        if force_predictor.is_none() && riged_override.is_none() {
            crate::trace::debug_eprintln!(
                "GLOBAL_MODULAR_TREE seed={}/{}: {} nodes (ID3 learned, seed_stride={})",
                seed,
                seeds,
                tree.len(),
                seed_stride,
            );
        } else if riged_override.is_some() {
            crate::trace::debug_eprintln!(
                "GLOBAL_MODULAR_TREE seed={}/{}: {} nodes (RIGED predictor-14 override)",
                seed,
                seeds,
                tree.len(),
            );
        } else {
            crate::trace::debug_eprintln!(
                "GLOBAL_MODULAR_TREE seed={}/{}: {} nodes (force-predictor override)",
                seed,
                seeds,
                tree.len(),
            );
        }

        // Collect residuals for this candidate tree. `group_ranges[g]` is
        // group g's slice of `all_tokens` — kept so the per-section LZ77
        // transform below can re-slice the winning seed's streams without
        // holding a second per-group copy.
        let (all_tokens, nb_meta_tokens, group_ranges) =
            crate::profile_time!("modular/collect_residuals_global", {
                collect_for_tree(&tree)
            });

        // Content-agnostic self-repair (task #14, #24): our fixed-stride sample
        // gather can ALIAS against periodic content (document text-line
        // spacing), yielding a non-representative sample whose learned tree
        // codes the real residuals badly (5336 e5: +36.9% vs cjxl). When that
        // is possible (large stride, non-trivial tree) learn a SECOND tree from
        // a DE-ALIASED randomized re-gather and keep whichever codes the actual
        // residuals cheaper (`estimate_token_cost` — the same estimator the
        // multi-seed picker trusts). Node count does NOT separate aliased docs
        // from detailed photos (both over-split); only real coded cost does.
        // Picking the cheaper tree can never regress bytes; on non-aliased
        // content the two costs sit within the moat so the fixed-stride tree is
        // kept ⇒ byte-identical. Costs a second tree-learn + collect when it
        // fires (gated to the base seed path, stride >= 8, non-trivial tree).
        let (all_tokens, nb_meta_tokens, group_ranges, tree) = if force_predictor.is_none()
            && riged_override.is_none()
            && super::encode::tree_self_repair_should_try(
                profile.tree_self_repair,
                seed_stride,
                tree.len(),
            ) {
            // Cheap aliasing pre-filter (bounds the wall-time cost): a
            // stride-aliased tree loses far more to the ≤96-histogram
            // clustering than a well-sampled one, so its REAL clustered ANS
            // cost runs well above its IDEAL per-context entropy. Compute both
            // for the fixed-stride tree and only pay for the de-aliased
            // re-gather when the ratio flags aliasing. Non-aliased content
            // (photos, flat docs; ratio ≈ 1.0) skips the second pass entirely.
            let ideal_a = estimate_token_cost(&all_tokens);
            let (cost_a_bits, code_a) = real_ans_cost(&all_tokens, &tree)?;
            let cost_a = cost_a_bits as f64;
            if !super::encode::tree_self_repair_ratio_flags_aliasing(cost_a, ideal_a) {
                crate::trace::debug_eprintln!(
                    "SELF_REPAIR seed={} stride={} tree_a={}n clustered={:.0} ideal={:.0} ratio={:.3} -> SKIP (clusters clean, not aliased)",
                    seed,
                    seed_stride,
                    tree.len(),
                    cost_a,
                    ideal_a,
                    cost_a / ideal_a.max(1.0),
                );
                // KEEP fixed-stride tree: cache its already-built code for the
                // Step-4 reuse (this is why the SKIP path costs ~0 extra).
                cached_winner_code = Some(code_a);
                (all_tokens, nb_meta_tokens, group_ranges, tree)
            } else {
                let mut samples_r = gather_for_seed(seed, seed_stride, true);
                let params_r = build_params(&samples_r);
                let tree_r = compute_best_tree(&mut samples_r, &params_r);
                let (tokens_r, meta_r, ranges_r) =
                    crate::profile_time!("modular/collect_residuals_global", {
                        collect_for_tree(&tree_r)
                    });
                let (cost_r_bits, code_r) = real_ans_cost(&tokens_r, &tree_r)?;
                let cost_r = cost_r_bits as f64;
                let switch = super::encode::tree_self_repair_keep_by_cost(cost_r, cost_a);
                crate::trace::debug_eprintln!(
                    "SELF_REPAIR seed={} stride={} tree_a={}n clustered={:.0} ideal={:.0} ratio={:.3} tree_r={}n clustered_r={:.0} -> {}",
                    seed,
                    seed_stride,
                    tree.len(),
                    cost_a,
                    ideal_a,
                    cost_a / ideal_a.max(1.0),
                    tree_r.len(),
                    cost_r,
                    if switch {
                        "SWITCH to de-aliased tree"
                    } else {
                        "KEEP fixed-stride tree"
                    },
                );
                // Cache the WINNER's already-built code for the Step-4 reuse.
                if switch {
                    cached_winner_code = Some(code_r);
                    (tokens_r, meta_r, ranges_r, tree_r)
                } else {
                    cached_winner_code = Some(code_a);
                    (all_tokens, nb_meta_tokens, group_ranges, tree)
                }
            }
        } else {
            (all_tokens, nb_meta_tokens, group_ranges, tree)
        };

        // Score: skip cost estimate entirely when seeds == 1 (the legacy
        // single-pass path doesn't need to know the cost). For seeds > 1
        // we compute the entropy cost (with per-context header term,
        // see `estimate_token_cost`) and keep the cheapest candidate.
        if seeds == 1 {
            best = Some((all_tokens, nb_meta_tokens, group_ranges, tree, 0.0));
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
        seed_costs.push(cost);
        match best {
            None => best = Some((all_tokens, nb_meta_tokens, group_ranges, tree, cost)),
            Some((_, _, _, _, prev_cost)) if cost < prev_cost => {
                best = Some((all_tokens, nb_meta_tokens, group_ranges, tree, cost));
            }
            _ => {}
        }

        // RFC#45 chunk 7 — Pareto-aware wall-clock early-out.
        //
        // After completing the probe seeds (chunk-3 perturbation slot),
        // check whether the spread of token costs is below the threshold.
        // Low chunk-3 spread → skip the remaining 12 seeds. This trades a
        // small bytes regression (~+0.09% on the 5-image chunk-6 bench)
        // for a large wall-clock speedup (~3.36× at e11). See the helper's
        // doc comment for the full trade-off table.
        //
        // Fires at most once, when the probe window first closes — `break`
        // exits the loop immediately so the helper isn't reconsulted on
        // subsequent iterations.
        let completed = seed_costs.len();
        if completed == MULTI_SEED_EARLY_OUT_PROBE_SEEDS
            && completed < seeds as usize
            && multi_seed_early_out_after_probe(
                &seed_costs,
                MULTI_SEED_EARLY_OUT_PROBE_SEEDS,
                seeds as usize,
            )
        {
            crate::trace::debug_eprintln!(
                "MULTI_SEED_EARLY_OUT after {}/{} seeds (cost spread converged below {:.3}%)",
                completed,
                seeds,
                super::tree_learn::MULTI_SEED_EARLY_OUT_SPREAD_THRESHOLD * 100.0,
            );
            break;
        }
    }

    let (all_tokens, nb_meta_tokens, group_ranges, tree, _final_cost) =
        best.expect("seeds >= 1 guarantees at least one candidate");
    let num_contexts = count_contexts(&tree) as usize;

    // Per-section LZ77 (issue #69 item 1) — mirrors the squeeze multi-group
    // path (frame.rs Step 5b). Every section's token stream is transformed
    // INDEPENDENTLY: the decoder creates a fresh LZ77 state per section with
    // dist_multiplier = max(that section's channel widths). The global ANS
    // code is then built over the TRANSFORMED streams, so its LZ77 length
    // context matches exactly what the sections emit — the histogram
    // mismatch that historically kept LZ77 off this path only existed when
    // the combined stream was transformed as one unit.
    //
    // Two deliberate orderings:
    // - Seed selection above scores UNTRANSFORMED streams. Transforming
    //   every candidate would multiply the e9 Optimal parse cost by the
    //   seed count; "pick tree, then LZ77" is the cheap order.
    // - Per-group sections re-collect tokens at write time
    //   (write_group_modular_section_idx) and re-apply this same transform.
    //   apply_lz77 is deterministic on identical inputs (same tree, same
    //   group_id, same num_contexts, same dist_multiplier), so the
    //   histogram-time slices and the write-time streams stay in lockstep.
    let lz77_applied = if use_lz77 {
        use crate::entropy_coding::lz77::{Lz77Params, apply_lz77};
        let try_lz77 = |tokens: &[AnsToken], dist_multiplier: i32| -> Result<Vec<AnsToken>> {
            if tokens.is_empty() {
                return Ok(Vec::new());
            }
            Ok(
                match apply_lz77(
                    tokens,
                    num_contexts,
                    false,
                    lz77_method,
                    dist_multiplier,
                    budget,
                )? {
                    Some((lz77_tokens, _)) => lz77_tokens,
                    None => tokens.to_vec(),
                },
            )
        };
        let meta_dm = meta_image
            .map(|m| m.channels.iter().map(|c| c.width()).max().unwrap_or(0))
            .unwrap_or(0) as i32;
        let mut transformed = Vec::with_capacity(all_tokens.len());
        transformed.extend(try_lz77(&all_tokens[..nb_meta_tokens], meta_dm)?);
        let transformed_nb_meta = transformed.len();
        for (g, range) in group_ranges.iter().enumerate() {
            let dm = images[g]
                .channels
                .iter()
                .map(|c| c.width())
                .max()
                .unwrap_or(0) as i32;
            transformed.extend(try_lz77(&all_tokens[range.clone()], dm)?);
        }
        // Header params come from the same (num_contexts, force_huffman)
        // construction apply_lz77 uses internally, so min_symbol/min_length
        // agree across all sections (same contract as the squeeze path).
        if transformed.iter().any(|t| t.is_lz77_length()) {
            let mut params = Lz77Params::new(num_contexts, false);
            params.enabled = true;
            Some((transformed, transformed_nb_meta, params))
        } else {
            // No section materialized a reference: without refs the
            // transform is the identity, so drop it and write the
            // pre-#69 lz77.enabled=0 layout.
            None
        }
    } else {
        None
    };
    let (all_tokens, nb_meta_tokens, lz77_params) = match lz77_applied {
        Some((tokens, nb_meta, params)) => (tokens, nb_meta, Some(params)),
        None => (all_tokens, nb_meta_tokens, None),
    };
    let ans_num_contexts = if lz77_params.is_some() {
        num_contexts + 1
    } else {
        num_contexts
    };

    // Step 4: Build multi-context ANS code with enhanced clustering.
    //
    // Reuse the self-repair's already-built code when present (task #14 perf):
    // it fires only on the LZ77-off e5/e6 tree-lift path where this build's
    // params are IDENTICAL — same winning tokens (no LZ77 transform, so
    // `all_tokens` is unchanged), `ans_num_contexts == num_contexts ==
    // count_contexts(tree)`, same clustering/uint flags, `lz77_params = None`.
    // So the cached code is byte-for-byte what this build would produce; the
    // `!use_lz77` guard is a belt-and-braces assertion of that invariant.
    let code = crate::profile_time!("modular/build_ans_code", {
        match cached_winner_code.take() {
            Some(cached) if !use_lz77 => cached,
            _ => build_entropy_code_ans_with_options(
                &all_tokens,
                ans_num_contexts,
                true, // enhanced clustering (pair-merge refinement)
                true, // optimize uint configs
                lz77_params.as_ref(),
                Some(total_pixels),
            ),
        }
    });

    // Per-seed diagnostics are emitted inside the loop via the
    // `MULTI_SEED_TREE_PICK` trace; here we just summarise the picked tree.
    crate::trace::debug_eprintln!(
        "DIAG tree: {} nodes, {} contexts, {} total_tokens (seeds={})",
        tree.len(),
        num_contexts,
        all_tokens.len(),
        seeds,
    );
    crate::trace::debug_eprintln!(
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
    let _tree_bits = writer.bits_written() - bits_before_tree;

    // Write LZ77 header + ANS data histogram.
    let bits_before_histo = writer.bits_written();
    if ans_num_contexts > 1 {
        write_lz77_header(lz77_params.as_ref(), writer)?;
        write_entropy_code_ans(&code, writer)?;
    } else {
        write_ans_modular_header(writer, &code)?;
    }
    let _histo_bits = writer.bits_written() - bits_before_histo;

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
    write_tokens_ans(meta_token_slice, &code, lz77_params.as_ref(), writer)?;

    let _total_lf_global_bits = writer.bits_written() - bits_before;
    crate::trace::debug_eprintln!(
        "DIAG LfGlobal: tree={} bits ({} B), histo={} bits ({} B), \
         meta_tokens={}, total={} bits ({} B)",
        _tree_bits,
        _tree_bits / 8,
        _histo_bits,
        _histo_bits / 8,
        nb_meta_tokens,
        _total_lf_global_bits,
        _total_lf_global_bits / 8,
    );

    writer.zero_pad_to_byte();

    Ok(GlobalModularState::AnsWithTree {
        code,
        tree,
        wp_params,
        lz77: lz77_params.map(|p| (lz77_method, p)),
    })
}

/// Info about global transforms to write in the LfGlobal GroupHeader.
pub struct GlobalTransforms {
    /// Full-image palette transform (issue #69 item 2):
    /// `(begin_c, num_c, nb_colors)`. Mutually exclusive with
    /// `compact_info`/`rct_type` (indices are nominal — RCT is skipped,
    /// and a full palette subsumes per-channel compaction).
    pub full_palette: Option<(usize, usize, usize)>,
    /// Per-channel ChannelCompact transforms: (begin_c, nb_colors).
    pub compact_info: Vec<(usize, usize)>,
    /// Optional RCT type (begin_c is adjusted for ChannelCompact meta channels).
    pub rct_type: Option<RctType>,
}

impl GlobalTransforms {
    pub fn rct_only(rct_type: Option<RctType>) -> Self {
        Self {
            full_palette: None,
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
    let num_transforms = transforms.full_palette.is_some() as u32
        + transforms.compact_info.len() as u32
        + transforms.rct_type.is_some() as u32;
    super::encode::write_num_transforms(writer, num_transforms)?;

    // Full-image palette (issue #69 item 2) — written exactly like the
    // single-group palette path (nb_deltas=0, d_pred=0, lossless).
    if let Some((begin_c, num_c, nb_colors)) = transforms.full_palette {
        write_palette_transform(writer, begin_c, num_c, nb_colors, 0, 0)?;
    }

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
// `MemoryBudget` is a `pub(crate)` type; this `pub` fn lives in the
// `pub(crate) mod section` (re-exported only inside `pub(crate) mod encode`),
// so it is not externally reachable despite the `pub` keyword. The budget
// param is an internal allocation-policy detail.
/// LfGlobal for the sectioned-tree mode: byte-shape-identical to the proven
/// global-tree LfGlobal (has_tree = 1, tree, single-context histograms,
/// GroupHeader, empty meta-token ANS stream) but with a TRIVIAL single-leaf
/// tree that stream 0 never uses for pixels — every image-sized channel is
/// group-streamed, and the pass groups carry their own local trees. Global
/// transforms (RCT) are signaled here exactly like the global-tree path, so
/// decoders apply them at full-image reconstruction as today.
pub(crate) fn write_local_trees_lf_global(
    writer: &mut BitWriter,
    transforms: GlobalTransforms,
) -> Result<()> {
    use super::encode::{write_tree, write_wp_header};
    use super::predictor::WeightedPredictorParams;

    let tree = super::tree::simple_tree(super::predictor::Predictor::Gradient);
    // The histogram must be buildable (a zero-count context cannot
    // normalize), so derive the 1-context code from one dummy token; the
    // stream itself still carries ZERO tokens (just the ANS final state).
    let seed_tokens =
        alloc::vec![crate::entropy_coding::token::Token::new(0, 0)];
    let code = build_entropy_code_ans(&seed_tokens, 1);
    let tokens: alloc::vec::Vec<crate::entropy_coding::token::Token> = alloc::vec::Vec::new();

    crate::f16::write_lf_quant(writer, None)?;
    writer.write(1, 1)?; // has_tree
    write_tree(writer, &tree)?;
    write_ans_modular_header(writer, &code)?;
    // GroupHeader for the (channel-empty at this stream) global modular image.
    writer.write(1, 1)?; // use_global_tree = true (the trivial tree above)
    write_wp_header(writer, &WeightedPredictorParams::default())?;
    write_global_transforms_full(writer, &transforms)?;
    // Empty meta-token ANS stream: the 32-bit final state that keeps every
    // decoder's begin()/final-state check happy (same rationale as the
    // global-tree writer's zero-meta case).
    write_tokens_ans(&tokens, &code, None, writer)?;
    writer.zero_pad_to_byte();
    Ok(())
}

/// Writes a PassGroup modular section that carries its OWN MA tree and
/// histograms (`use_global_tree = false`) — the sectioned-tree lossless
/// memory mode (imazen/jxl-encoder#96).
///
/// The stream is fully self-contained: GroupHeader (local tree, wp params,
/// optional per-group RCT descriptor), the tree learned from THIS group's
/// samples only, its entropy code, then the group's tokens. Peak memory for
/// the whole encode becomes the image copies plus ONE group's tree-learn
/// working set instead of the whole-image sample accumulator — measured
/// tradeoff in `benchmarks/jxl_sectioned_tree_tradeoff_2026-08-13.md`.
///
/// The caller writes an LfGlobal with `has_tree = 0` and NO global stream
/// content (grammar: decoders early-out on the empty global channel set
/// before reading a GroupHeader, so global transforms cannot be signaled
/// there — which is why the RCT descriptor rides in each group's header;
/// RCT is pointwise per pixel, so per-group application is equivalent).
#[allow(private_interfaces)]
#[allow(clippy::too_many_arguments)]
pub fn write_group_modular_section_local_tree(
    group_image: &ModularImage,
    stream_id: u32,
    profile: &crate::effort::EffortProfile,
    use_lz77: bool,
    lz77_method: crate::entropy_coding::lz77::Lz77Method,
    rct_type: Option<RctType>,
    writer: &mut BitWriter,
    budget: Option<&alloc::sync::Arc<crate::budget::MemoryBudget>>,
) -> Result<()> {
    use super::encode::{write_num_transforms, write_tree, write_wp_header};
    use super::predictor::WeightedPredictorParams;
    use super::tree::count_contexts;
    use super::tree_learn::{
        TreeLearningParams, TreeSamples, collect_residuals_with_tree_with_budget,
        compute_best_tree, compute_gather_stride_from_profile, gather_samples_strided,
        max_ref_channels,
    };
    use crate::entropy_coding::encode::build_entropy_code_ans_with_options;
    use crate::entropy_coding::encode::write_entropy_code_ans;
    use crate::entropy_coding::lz77::{apply_lz77, write_lz77_header};

    // v1 keeps the default WP parameter set (no per-group search): the
    // params used for learning are the params written in this group's
    // header, so encode and decode agree by construction.
    let wp_params = WeightedPredictorParams::default();
    let total_pixels: usize = group_image
        .channels
        .iter()
        .map(|c| c.width() * c.height())
        .sum();
    let stride = compute_gather_stride_from_profile(total_pixels, profile);
    let num_refs = max_ref_channels(group_image);

    // Learn this group's tree from this group's samples only. Property 1
    // (group id) is the spec stream id, matching what the residual
    // collector below feeds the tree at encode time.
    let mut samples = TreeSamples::new_with_ref_channels(num_refs);
    gather_samples_strided(&mut samples, group_image, stream_id, 0, stride, &wp_params);
    let pixel_fraction = if total_pixels > 0 {
        samples.total_gathered_weight() as f64 / total_pixels as f64
    } else {
        1.0
    };
    let params = TreeLearningParams::from_profile(profile)
        .with_ref_properties(num_refs, profile.effort)
        .with_total_pixels(total_pixels)
        .with_pixel_fraction(pixel_fraction);
    let tree = compute_best_tree(&mut samples, &params);
    drop(samples);

    let tokens =
        collect_residuals_with_tree_with_budget(group_image, &tree, stream_id, &wp_params, budget)?;
    let num_contexts = count_contexts(&tree) as usize;

    // Same LZ77 construction as the single-group tree writer.
    let dist_multiplier = group_image
        .channels
        .iter()
        .map(|c| c.width())
        .max()
        .unwrap_or(0) as i32;
    let (tokens, lz77_params) = if use_lz77 {
        match apply_lz77(&tokens, num_contexts, false, lz77_method, dist_multiplier, budget)? {
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
    let code = build_entropy_code_ans_with_options(
        &tokens,
        ans_num_contexts,
        true,
        true,
        lz77_params.as_ref(),
        Some(total_pixels),
    );

    // GroupHeader: local tree, the wp params used above, per-group RCT.
    writer.write(1, 0)?; // use_global_tree = false
    write_wp_header(writer, &wp_params)?;
    let num_transforms = u32::from(rct_type.is_some());
    write_num_transforms(writer, num_transforms)?;
    if let Some(rct) = rct_type {
        write_rct_transform(writer, 0, rct)?;
    }

    // Local tree + its entropy code, exactly the single-group serialization.
    write_tree(writer, &tree)?;
    if ans_num_contexts > 1 {
        write_lz77_header(lz77_params.as_ref(), writer)?;
        write_entropy_code_ans(&code, writer)?;
    } else {
        write_ans_modular_header(writer, &code)?;
    }
    write_tokens_ans(&tokens, &code, lz77_params.as_ref(), writer)?;
    // Sections are byte-delimited by the TOC; pad to the byte boundary like
    // every other section writer does before the caller's `finish()`.
    writer.zero_pad_to_byte();
    Ok(())
}

#[allow(private_interfaces)]
pub fn write_group_modular_section(
    group_image: &ModularImage,
    state: &GlobalModularState,
    writer: &mut BitWriter,
    budget: Option<&alloc::sync::Arc<crate::budget::MemoryBudget>>,
) -> Result<()> {
    write_group_modular_section_idx(
        group_image,
        state,
        0,
        &GroupTransforms::none(),
        writer,
        budget,
    )
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
    budget: Option<&alloc::sync::Arc<crate::budget::MemoryBudget>>,
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
            lz77,
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
            // Per-section LZ77 (issue #69 item 1): re-apply the exact
            // transform the global histogram was built over — same method,
            // same num_contexts (from the same tree), and this section's
            // dist_multiplier (max channel width of THIS group, matching
            // the decoder's fresh per-section LZ77 state). apply_lz77 is
            // deterministic, so this stream is identical to the
            // histogram-time slice in the global section.
            let (tokens, lz77_params) = match lz77 {
                Some((method, params)) => {
                    let dist_multiplier = group_image
                        .channels
                        .iter()
                        .map(|c| c.width())
                        .max()
                        .unwrap_or(0) as i32;
                    let num_contexts = super::tree::count_contexts(tree) as usize;
                    let transformed = match crate::entropy_coding::lz77::apply_lz77(
                        &tokens,
                        num_contexts,
                        false,
                        *method,
                        dist_multiplier,
                        budget,
                    )? {
                        Some((lz77_tokens, _)) => lz77_tokens,
                        None => tokens,
                    };
                    (transformed, Some(params))
                }
                None => (tokens, None),
            };
            crate::profile_time!("modular/write_tokens_per_group", {
                write_tokens_ans(&tokens, code, lz77_params, writer)?;
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
