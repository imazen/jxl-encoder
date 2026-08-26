// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Zensim-based quantization loop for iterative quality refinement.
//!
//! Alternative to the butteraugli loop: uses zensim's psychovisual metric for
//! both global quality tracking and per-pixel spatial error map (diffmap).
//!
//! Uses planar linear RGB input directly — no RGBA interleaving overhead.
//! The diffmap includes SSIM + edge artifact + MSE features for comprehensive
//! spatial error detection (blocking, ringing, detail loss).
//!
//! Per-tile distance calibration uses `approx_butteraugli()` — a trained mapping
//! from zensim scores to butteraugli-compatible distance units.

use super::ac_strategy::AcStrategyMap;
use super::adaptive_quant::quantize_quant_field;
use super::chroma_from_luma::CflMap;
use super::common::*;
use super::encoder::VarDctEncoder;
use super::frame::DistanceParams;
use crate::debug_rect;
use crate::error::Result;

/// diffmap-RD worktree (2026-07-18): bake bytes for `JXL_ZENSIM_RD_PROFILE=bake:<path>`
/// — mounts an arbitrary ZNPR bake as the loop scorer via `ZensimProfile::Custom`
/// (zensim `custom-profiles` feature; enabled in this worktree's Cargo.toml).
static RD_BAKE_BYTES: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
fn rd_bake_bytes() -> &'static [u8] {
    RD_BAKE_BYTES.get().expect("rd bake bytes set").as_slice()
}
/// Resolved (profile, n_inputs) for the RD experiment — computed once per process.
static RD_PROFILE: std::sync::OnceLock<(zensim::ZensimProfile, usize)> = std::sync::OnceLock::new();

/// Feature-width probe: the bakes we mount are exact-width, so the first width
/// the forward accepts is the model's caller-facing input width (dependency-free
/// — avoids a zenpredict dep just to read the header).
///
/// 23shot-sota944 (2026-08-05): probes SMALLEST-FIRST. The forward accepts any
/// width `>=` the bake's caller width (prefix branch) plus `caller == W + 4`
/// (size axes), so the first accepted width is the tightest one. Largest-first
/// with the folded widths in the list would resolve EVERY bake to 944 and break
/// the ≤372 arms' gradient probes. Folded widths (720/924/944) route the loop
/// through the folded-class score path below. Note a PRUNED bake
/// (`FeatureTransform::Drop`) probes at its CALLER width, not its internal
/// layer-0 width — exactly what the extraction must produce.
fn rd_infer_n_inputs(profile: zensim::ZensimProfile) -> usize {
    let feats = vec![0.0f64; 944];
    for n in [156usize, 228, 300, 372, 720, 924, 944] {
        if zensim::score_features_with_profile(profile, &feats[..n], 64, 64).is_ok() {
            return n;
        }
    }
    0
}

/// Map-side profile for the folded-class score route: NO mlp, walk config
/// (weights / num_scales / blur) identical to the bake mounts' builder
/// defaults — so the Trained diffmap driving redistribution is the same map
/// every `*_base` arm uses, and the mounted 944-class bake is the ONLY arm
/// difference (it judges/controls; it cannot score through the 372-class
/// compare, whose forward would fail and silently emit the seed).
static RD_MAP_PROFILE: std::sync::OnceLock<zensim::ZensimProfile> = std::sync::OnceLock::new();
fn rd_map_profile() -> zensim::ZensimProfile {
    *RD_MAP_PROFILE.get_or_init(|| {
        let params = zensim::profile::ProfileParams::builder()
            .skip_score_mapping(true)
            .extrapolate_score(true)
            .build();
        let params: &'static zensim::profile::ProfileParams = Box::leak(Box::new(params));
        zensim::ZensimProfile::Custom {
            params,
            name: "rd-map-folded",
        }
    })
}

/// Appendix N (2026-08-05): the map-side profile for folded-class
/// ATTR-family arms — `rd_map_profile` plus `extended_features`, because
/// the fused folded-944 compare's v1 walk derives attribution coefficients
/// from every basic stat. Walk config (weights / num_scales / blur) is
/// still the builder defaults, so the redistribution-relevant geometry is
/// unchanged; only stat coverage grows. The baseline folded arm keeps the
/// plain `rd_map_profile` (byte-identity with appendix M).
static RD_ATTR_MAP_PROFILE: std::sync::OnceLock<zensim::ZensimProfile> = std::sync::OnceLock::new();
fn rd_attr_map_profile() -> zensim::ZensimProfile {
    *RD_ATTR_MAP_PROFILE.get_or_init(|| {
        let params = zensim::profile::ProfileParams::builder()
            .skip_score_mapping(true)
            .extrapolate_score(true)
            .extended_features(true)
            .build();
        let params: &'static zensim::profile::ProfileParams = Box::leak(Box::new(params));
        zensim::ZensimProfile::Custom {
            params,
            name: "rd-attr-map-folded944",
        }
    })
}

/// `JXL_ZENSIM_RD_PROFILE=a|b|latest|bake:<path>` → (profile, n_inputs).
/// n_inputs = 0 means "model-map unsupported for this profile" (A's feature
/// regime is not probed; the shipped default needs no gradient).
fn rd_profile_from_env() -> (zensim::ZensimProfile, usize) {
    let v = std::env::var("JXL_ZENSIM_RD_PROFILE").unwrap_or_default();
    if let Some(path) = v.strip_prefix("bake:") {
        let bytes = std::fs::read(path)
            .unwrap_or_else(|e| panic!("JXL_ZENSIM_RD_PROFILE bake {path}: {e}"));
        RD_BAKE_BYTES.set(bytes).expect("rd bake set once");
        let params = zensim::profile::ProfileParams::builder()
            .mlp(rd_bake_bytes)
            .skip_score_mapping(true)
            .extrapolate_score(true)
            .extended_features(true)
            .compute_iw_features(true)
            .build();
        let params: &'static zensim::profile::ProfileParams = Box::leak(Box::new(params));
        let profile = zensim::ZensimProfile::Custom {
            params,
            name: "rd-bake",
        };
        let n_in = rd_infer_n_inputs(profile);
        // 23shot-sota944 loud guard: a mounted bake whose forward accepts no
        // probed width can never score — the loop's compare errors would then
        // silently emit seed-quality bitstreams (`Err(_) => Ok(current_params)`).
        // Refuse at mount instead.
        assert!(
            n_in != 0,
            "JXL_ZENSIM_RD_PROFILE bake:{path}: the bake's forward accepts no \
             probed feature width (156/228/300/372/720/924/944) — refusing to \
             mount a bake that cannot score (silent mount = seed-quality output)"
        );
        return (profile, n_in);
    }
    match v.as_str() {
        "b" => (zensim::ZensimProfile::B, 372),
        "latest" => (zensim::ZensimProfile::latest_preview(), 372),
        _ =>
        {
            #[allow(deprecated)]
            (zensim::ZensimProfile::A, 0)
        }
    }
}

/// Tunable parameters for the zensim loop, readable from environment variables.
/// Allows parameter sweeps without recompilation.
#[derive(Debug, Clone)]
struct ZensimParams {
    // Diffmap options
    masking_strength: Option<f32>, // ZENSIM_MASKING (f32 or "none")
    sqrt: bool,                    // ZENSIM_SQRT (0/1)
    include_hf: bool,              // ZENSIM_HF (0/1)
    include_edge_mse: bool,        // ZENSIM_EDGE_MSE (0/1)
    // Tile aggregation
    norm_power: f32,     // ZENSIM_NORM (2.0=L2, 4.0=L4, etc.)
    spatial_weight: f32, // ZENSIM_SPATIAL_W (0.0-1.0)
    ratio_max: f32,      // ZENSIM_RATIO_MAX (1.0-10.0)
    // Redistribution
    alpha_base: f32, // ZENSIM_ALPHA (0.01-1.0)
    factor_max: f32, // ZENSIM_FACTOR_MAX (1.01-2.0)
}

impl ZensimParams {
    fn from_env() -> Self {
        Self {
            // Defaults tuned by parameter sweep (2026-03-08, 20 images validated).
            // masking=8,sqrt=0: preserves diffmap dynamic range for redistribution.
            // L2/sw=0.6/rmax=3.0: best quality gain across diverse images.
            //   → +1.262 SSIM2 at +1.10% size (e7-zen4 vs e7, 20-img avg)
            //   → +0.000 SSIM2 at -4.61% size (e8-zen4 vs e7, 20-img avg)
            // L6/sw=1.0/rmax=2.0 tested: lower size cost but -11% less quality
            // gain and e8-zen4 regression (-0.114 SSIM2). Overfits on 4-img sample.
            masking_strength: Self::env_masking("ZENSIM_MASKING", Some(8.0)),
            sqrt: Self::env_bool("ZENSIM_SQRT", false),
            include_hf: Self::env_bool("ZENSIM_HF", true),
            include_edge_mse: Self::env_bool("ZENSIM_EDGE_MSE", true),
            norm_power: Self::env_f32("ZENSIM_NORM", 2.0),
            spatial_weight: Self::env_f32("ZENSIM_SPATIAL_W", 0.6),
            ratio_max: Self::env_f32("ZENSIM_RATIO_MAX", 3.0),
            alpha_base: Self::env_f32("ZENSIM_ALPHA", 0.25),
            factor_max: Self::env_f32("ZENSIM_FACTOR_MAX", 1.15),
        }
    }

    fn env_f32(name: &str, default: f32) -> f32 {
        std::env::var(name)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(default)
    }

    fn env_bool(name: &str, default: bool) -> bool {
        match std::env::var(name).ok().as_deref() {
            Some("0" | "false" | "no") => false,
            Some("1" | "true" | "yes") => true,
            _ => default,
        }
    }

    fn env_masking(name: &str, default: Option<f32>) -> Option<f32> {
        match std::env::var(name).ok().as_deref() {
            Some("none" | "0" | "off") => None,
            Some(s) => s.parse().ok().map(Some).unwrap_or(default),
            None => default,
        }
    }
}

/// Compute per-block tile distance from a zensim diffmap.
///
/// For each transform in the strategy map, computes the L4 norm of diffmap
/// pixels in the covered region (emphasizing worst-case pixels within each
/// tile), then normalizes so the global average tile distance matches
/// `anchor_dist`.
///
/// The anchor_dist is typically `target_distance` (budget-neutral redistribution)
/// with an optional small correction from the measured quality. This avoids
/// the bias in `approx_butteraugli()` which caused +55% file size inflation
/// when used directly as the anchor.
///
/// L4 norm is a compromise: mean would ignore spatial peaks (missing bad pixels),
/// L16 (as butteraugli uses) would be too aggressive for SSIM-based diffmap
/// values which already have 11×11 spatial smoothing.
#[allow(clippy::too_many_arguments)]
/// C10 fail-loud (plan §5, 2026-08-26): the 372-class compare paths used to swallow
/// every error as `Err(_) => Ok(seed)`, silently emitting a SEED-QUALITY bitstream —
/// indistinguishable from a converged encode at the call site. A compare failure is a
/// wiring bug (mount probes and width guards run earlier), so it panics with the arm
/// name, like the folded-class routes have since 23shot-sota944.
fn loud_compare<T, E: std::fmt::Debug>(r: Result<T, E>, what: &str) -> T {
    match r {
        Ok(v) => v,
        Err(e) => panic!(
            "zensim loop {what} compare failed (loud by design; the old `Err(_) =>              Ok(seed)` silently emitted a seed-quality bitstream): {e:?}"
        ),
    }
}

fn compute_tile_dist(
    diffmap: &[f32],
    width: usize,
    height: usize,
    ac_strategy: &AcStrategyMap,
    xsize_blocks: usize,
    ysize_blocks: usize,
    anchor_dist: f32,
    tile_dist: &mut [f32],
    params: &ZensimParams,
) {
    tile_dist.fill(0.0);

    let norm_power = params.norm_power as f64;
    let inv_power = 1.0 / norm_power;

    let mut sum_raw = 0.0f64;
    let mut n_tiles = 0u32;
    for by in 0..ysize_blocks {
        for bx in 0..xsize_blocks {
            if !ac_strategy.is_first(bx, by) {
                continue;
            }
            let covered_x = ac_strategy.covered_blocks_x(bx, by);
            let covered_y = ac_strategy.covered_blocks_y(bx, by);
            let px_start_x = bx * BLOCK_DIM;
            let px_start_y = by * BLOCK_DIM;
            let px_end_x = ((bx + covered_x) * BLOCK_DIM).min(width);
            let px_end_y = ((by + covered_y) * BLOCK_DIM).min(height);
            if px_start_x >= width || px_start_y >= height {
                continue;
            }

            let mut norm_sum = 0.0f64;
            let mut pixels = 0.0f64;
            for py in px_start_y..px_end_y {
                for px in px_start_x..px_end_x {
                    let v = diffmap[py * width + px] as f64;
                    if !v.is_finite() {
                        continue;
                    }
                    norm_sum += v.abs().powf(norm_power);
                    pixels += 1.0;
                }
            }
            if pixels == 0.0 {
                pixels = 1.0;
            }
            // Lp norm: (sum(|v|^p) / n)^(1/p)
            let tile_norm = (norm_sum / pixels).powf(inv_power) as f32;

            for sy in 0..covered_y {
                for sx in 0..covered_x {
                    tile_dist[(by + sy) * xsize_blocks + (bx + sx)] = tile_norm;
                }
            }

            sum_raw += tile_norm as f64;
            n_tiles += 1;
        }
    }

    if n_tiles == 0 || sum_raw < 1e-12 {
        tile_dist.fill(anchor_dist);
        return;
    }
    let avg_raw = (sum_raw / n_tiles as f64) as f32;
    let spatial_weight = params.spatial_weight;
    let ratio_max = params.ratio_max;
    for td in tile_dist.iter_mut() {
        let ratio = if avg_raw > 1e-10 {
            (*td / avg_raw).min(ratio_max)
        } else {
            1.0
        };
        let blended = 1.0 - spatial_weight + spatial_weight * ratio;
        *td = anchor_dist * blended;
    }
}

/// C3b (task #67): per-tile steering from the C3a attribution map — the
/// per-tile raw value is the CLAMPED mean attribution density over the
/// tile's pixel rect (`max(0, query_rect)/pixels`; clamping at the TILE
/// level is the signed-map analogue of the fold arm's per-pixel clamp —
/// negative-sum tiles read "refining here loses score" and coarsen under
/// redistribution, the directionally-correct action). The normalization /
/// `spatial_weight` / `ratio_max` blend tail is IDENTICAL to
/// [`compute_tile_dist`] so the A/B isolates the MAP change alone.
#[allow(clippy::too_many_arguments)]
fn compute_tile_dist_attr(
    attr: &zensim::AttributionResult,
    ac_strategy: &AcStrategyMap,
    xsize_blocks: usize,
    ysize_blocks: usize,
    anchor_dist: f32,
    tile_dist: &mut [f32],
    params: &ZensimParams,
) {
    tile_dist.fill(0.0);
    let (width, height) = (attr.width(), attr.height());
    let mut sum_raw = 0.0f64;
    let mut n_tiles = 0u32;
    for by in 0..ysize_blocks {
        for bx in 0..xsize_blocks {
            if !ac_strategy.is_first(bx, by) {
                continue;
            }
            let covered_x = ac_strategy.covered_blocks_x(bx, by);
            let covered_y = ac_strategy.covered_blocks_y(bx, by);
            let px_start_x = bx * BLOCK_DIM;
            let px_start_y = by * BLOCK_DIM;
            let px_end_x = ((bx + covered_x) * BLOCK_DIM).min(width);
            let px_end_y = ((by + covered_y) * BLOCK_DIM).min(height);
            if px_start_x >= width || px_start_y >= height {
                continue;
            }
            let pixels = ((px_end_x - px_start_x) * (px_end_y - px_start_y)).max(1) as f64;
            // O(1) SAT rectangle query — the C3a steering contract: the
            // tile's summed density ≈ the first-order score gain from
            // refining exactly this tile.
            let gain = attr.query_rect(px_start_x, px_start_y, px_end_x, px_end_y);
            let tile_norm = (gain.max(0.0) / pixels) as f32;

            for sy in 0..covered_y {
                for sx in 0..covered_x {
                    tile_dist[(by + sy) * xsize_blocks + (bx + sx)] = tile_norm;
                }
            }
            sum_raw += tile_norm as f64;
            n_tiles += 1;
        }
    }

    if n_tiles == 0 || sum_raw < 1e-30 {
        tile_dist.fill(anchor_dist);
        return;
    }
    let avg_raw = (sum_raw / n_tiles as f64) as f32;
    let spatial_weight = params.spatial_weight;
    let ratio_max = params.ratio_max;
    for td in tile_dist.iter_mut() {
        let ratio = if avg_raw > 1e-30 {
            (*td / avg_raw).min(ratio_max)
        } else {
            1.0
        };
        let blended = 1.0 - spatial_weight + spatial_weight * ratio;
        *td = anchor_dist * blended;
    }
}

/// #69 H arms: per-tile SIGNED attribution values — `tile_signed[b]` = the
/// tile's MEAN density (score-units/pixel, signed; H1/H2's field) and
/// `tile_q[b]` = the tile's TOTAL `query_rect` (score units; H3's field).
/// No clamp, no normalization — the arm-specific redistribution applies its
/// own registered rule.
fn compute_tile_signed_attr(
    attr: &zensim::AttributionResult,
    ac_strategy: &AcStrategyMap,
    xsize_blocks: usize,
    ysize_blocks: usize,
    tile_signed: &mut [f32],
    tile_q: &mut [f32],
) {
    tile_signed.fill(0.0);
    tile_q.fill(0.0);
    let (width, height) = (attr.width(), attr.height());
    for by in 0..ysize_blocks {
        for bx in 0..xsize_blocks {
            if !ac_strategy.is_first(bx, by) {
                continue;
            }
            let covered_x = ac_strategy.covered_blocks_x(bx, by);
            let covered_y = ac_strategy.covered_blocks_y(bx, by);
            let px_start_x = bx * BLOCK_DIM;
            let px_start_y = by * BLOCK_DIM;
            let px_end_x = ((bx + covered_x) * BLOCK_DIM).min(width);
            let px_end_y = ((by + covered_y) * BLOCK_DIM).min(height);
            if px_start_x >= width || px_start_y >= height {
                continue;
            }
            let pixels = ((px_end_x - px_start_x) * (px_end_y - px_start_y)).max(1) as f64;
            let gain = attr.query_rect(px_start_x, px_start_y, px_end_x, px_end_y);
            let s = (gain / pixels) as f32;
            let q = gain as f32;
            for sy in 0..covered_y {
                for sx in 0..covered_x {
                    tile_signed[(by + sy) * xsize_blocks + (bx + sx)] = s;
                    tile_q[(by + sy) * xsize_blocks + (bx + sx)] = q;
                }
            }
        }
    }
}

/// Refine AC strategy by splitting large transforms with high perceptual error.
///
/// Scans the strategy map for multi-block transforms (DCT16+) where the
/// tile distance exceeds `split_threshold * target_distance`. Those blocks
/// are split one level down (e.g., DCT32x32 → four DCT16x16).
///
/// Returns the number of splits performed.
///
/// Currently disabled: splitting large transforms with high error causes size
/// inflation (+5-12%) without proportional quality gain. Kept for future
/// re-enablement once RD impact is characterized.
#[allow(dead_code)]
fn refine_strategy_from_diffmap(
    ac_strategy: &mut AcStrategyMap,
    tile_dist: &[f32],
    xsize_blocks: usize,
    ysize_blocks: usize,
    target_distance: f32,
    split_threshold: f32,
) -> usize {
    let threshold = split_threshold * target_distance;
    let mut splits = 0;

    for by in 0..ysize_blocks {
        for bx in 0..xsize_blocks {
            if !ac_strategy.is_first(bx, by) {
                continue;
            }
            let cx = ac_strategy.covered_blocks_x(bx, by);
            let cy = ac_strategy.covered_blocks_y(bx, by);
            // Only consider multi-block transforms
            if cx <= 1 && cy <= 1 {
                continue;
            }
            let td = tile_dist[by * xsize_blocks + bx];
            if td > threshold && ac_strategy.split_one_level(bx, by) {
                splits += 1;
            }
        }
    }

    splits
}

impl VarDctEncoder {
    /// Zensim quantization loop: iteratively refines per-block quant_field
    /// using zensim's psychovisual metric for both global quality and spatial
    /// error distribution.
    ///
    /// Same structure as butteraugli_refine_quant_field. The spatial error map
    /// comes from zensim's diffmap (per-pixel SSIM + edge + MSE error in XYB
    /// space, weighted across channels). The global quality is tracked via
    /// `approx_butteraugli()` which maps zensim scores to butteraugli-compatible
    /// distance units.
    #[cfg(feature = "zensim-loop")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn zensim_refine_quant_field(
        &self,
        linear_rgb: &[f32],
        width: usize,
        height: usize,
        xyb_x: &[f32],
        xyb_y: &[f32],
        xyb_b: &[f32],
        padded_width: usize,
        padded_height: usize,
        xsize_blocks: usize,
        ysize_blocks: usize,
        initial_params: &DistanceParams,
        quant_field: &mut [u8],
        quant_field_float: &mut [f32],
        initial_quant_field_float: &[f32],
        cfl_map: &CflMap,
        ac_strategy: &mut AcStrategyMap,
        patches_data: Option<&super::patches::PatchesData>,
        splines_data: Option<&super::splines::SplinesData>,
    ) -> Result<DistanceParams> {
        use super::epf;
        use super::reconstruct::{gab_smooth, reconstruct_xyb, xyb_to_linear_rgb_planar};
        use crate::budget::MemoryBudget;

        let budget = self.budget.as_ref();
        let target_distance = self.distance;
        let num_blocks = xsize_blocks * ysize_blocks;
        let padded_pixels = padded_width * padded_height;
        let n = width * height;

        // Read tunable parameters from environment variables.
        // Defaults match the benchmark-validated configuration.
        let params = ZensimParams::from_env();

        // Efficiency study (2026-07-31): env-gated global quant-field
        // pre-scale — the actuator for the harness-level bytes-target outer
        // loop (E7). The loop itself never entropy-codes, so per-iterate
        // bytes do not exist here; the outer harness measures real bytes per
        // full encode and drives this scale with the same damped-controller
        // formula the in-loop score controller uses. Unset / 1.0 = no-op
        // (default behavior unchanged; byte-identity gated in the study's R0).
        if let Some(g) = std::env::var("JXL_ZENSIM_QF_GLOBAL_SCALE")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            && g.is_finite()
            && g > 0.0
            && g != 1.0
        {
            for v in quant_field_float.iter_mut() {
                *v *= g;
            }
        }

        // Pinned to `ZensimProfile::A` (the shipped codec-target metric, v47-strict
        // as of 2026-05-27) rather than the deprecated `latest()` — the
        // ZENSIM_DISTANCE_TARGETS calibration table is seeded against a SPECIFIC
        // profile, so the scoring profile must be pinned to keep the loop and its
        // calibration coherent across zensim releases.
        // zensim deprecated `A` in favour of `B` (2026-07); the pin stays until
        // the calibration table is re-seeded against a new profile.
        // RD-experiment override (2026-07-18): `JXL_ZENSIM_RD_PROFILE=
        // a|b|latest|bake:<path>` selects the loop's scoring profile (bake:
        // mounts an arbitrary ZNPR via ZensimProfile::Custom); unset → A
        // (shipped behavior). `JXL_ZENSIM_MODEL_MAP=signed|abs` additionally
        // steers iterations 2+ with the MODEL'S OWN gradient s_k
        // (DiffmapWeighting::ModelSensitivity) — s_k measured numerically at
        // the first iteration's features, fold per the 2026-07-18 coherence
        // matrix (signed for MLP gradients, abs for additive solves).
        let (rd_profile, rd_n_in) = *RD_PROFILE.get_or_init(rd_profile_from_env);
        // 23shot-sota944 (2026-08-05): folded-class (720/924/944) bakes cannot
        // score through the 372-class compare below — its forward wants more
        // features than that walk extracts, and the compare's `Err(_) =>
        // Ok(current_params)` would silently emit the seed. Route: the
        // redistribution map comes from the no-mlp map profile (identical walk
        // config — same Trained diffmap as every `*_base` arm), and the score
        // comes from the canonical folded extraction + the full-bundle forward
        // (`score_features_with_profile`, which sizes by the bake's CALLER
        // input width — a pruned bake carries `FeatureTransform::Drop` and
        // must be handed its caller-width vector, never its internal width).
        let folded_class = rd_n_in >= 720;
        let model_map = std::env::var("JXL_ZENSIM_MODEL_MAP").ok();
        let model_map_requested = matches!(model_map.as_deref(), Some(s) if !s.is_empty());
        // Appendix N (2026-08-05): folded-class ATTR-FAMILY arms score+steer
        // through the FUSED folded-944 compare
        // (`compute_folded944_score_and_attribution`), whose v1 walk must
        // compute every basic feature — so their map-side profile is the
        // extended twin of `rd_map_profile` (same walk config + weights).
        // The baseline folded arm keeps the plain map profile: its route is
        // byte-identical to appendix M (the substrate probe's carried-rows
        // guarantee).
        let z = zensim::Zensim::new(if folded_class && model_map_requested {
            rd_attr_map_profile()
        } else if folded_class {
            rd_map_profile()
        } else {
            rd_profile
        })
        .with_parallel(false);
        // C3b (task #67): `attr` steers iterations 2+ with the C3a FUSED
        // compare (`compute_with_ref_score_and_attribution` — ONE call =
        // score + attribution map, replacing the separate compare +
        // ModelSensitivity fold); `attr-stale` additionally steers with the
        // PREVIOUS iteration's map (pricing the stale-scalar single-pass
        // lever: does steering quality survive a one-iteration-stale map?).
        // Both derive s_k exactly like `signed` at iteration 1. 372-class
        // profiles only (the fused entry's contract).
        // #69 loop-steering arms (PLAN_LOOP_STEERING_69.md, frozen gates):
        //   h1-signed — signed redistribution: the attribution map's signed
        //     per-tile mass drives the qf factor directly (negative tiles
        //     push coarser proportionally; no clamp-at-0, no anchor blend).
        //   h2-ctrl  — controller separation: the map steers only the
        //     ZERO-SUM residual (tile value minus mean); the damped
        //     controller owns the level from scalar error alone.
        //   h3-mag   — magnitude steering: per-tile step ∝ query_rect
        //     score-units (× ZENSIM_H3_GAIN, default 10.0), capped by
        //     factor_max; NOT ratio-normalized.
        // All three share the C3b attr arm's fused compare, controller,
        // qf bounds, and sum-renormalization — the A/B isolates the
        // redistribution rule.
        // 23shot-sota944 loud guards (2026-08-05) — the loop69-registered
        // hazard was that an unknown JXL_ZENSIM_MODEL_MAP value fell through
        // to baseline SILENTLY (caught in-run there only by a control-arm
        // mismatch). The refusals that replace the fall-through:
        //   1. unknown arm name        -> panic (typo'd arms never masquerade
        //      as baseline);
        //   2. arm set but rd_n_in == 0 (profile A / unprobed) -> panic (the
        //      arm would silently run as baseline);
        //   3. a FOLD-family arm (signed/abs) on a folded-class bake -> panic:
        //      the per-scale ModelSensitivity fold has no folded-class
        //      integrands. ATTR-family arms on a folded-class bake are LIVE
        //      since appendix N — they route through the fused folded-944
        //      compare instead of panicking.
        // Empty string counts as unset (baseline), matching `remove_var`
        // drivers that clear the arm between runs.
        // #70 item 2: `JXL_ZENSIM_SINGLEPASS=1` routes the fused compare
        // through zensim's stale-scalar single-pass entry
        // (`compute_with_ref_score_and_attribution_stale`) — score +
        // steering map from ONE pipeline pass. The map combines the CURRENT
        // iterate's planes with the PREVIOUS compare's pooled scalars (the
        // proven-free one-iterate-lag semantics; the session's first fused
        // call primes via the fresh path). Default OFF = the fresh fused
        // call, byte-identical to shipped behavior (R0-gated). The session
        // lives per encode (one reference), created lazily at the first
        // steered iteration.
        // Appendix P lever 2: on a FOLDED-class bake the same env means the
        // stale-MAP single-pass — the first steered iteration pays the
        // fused folded-944 compare and caches its map; every later steered
        // iteration is a score-only canonical extraction + forward steering
        // with the cached map (marginal map cost ≈ 0; gate G-P5).
        let singlepass = matches!(
            std::env::var("JXL_ZENSIM_SINGLEPASS").ok().as_deref(),
            Some("1" | "true" | "yes")
        );
        if model_map_requested {
            let known = matches!(
                model_map.as_deref(),
                Some("signed")
                    | Some("abs")
                    | Some("attr")
                    | Some("attr-stale")
                    | Some("h1-signed")
                    | Some("h2-ctrl")
                    | Some("h3-mag")
                    | Some("h3-mag-stale")
            );
            assert!(
                known,
                "JXL_ZENSIM_MODEL_MAP='{}' is not a known arm \
                 (signed|abs|attr|attr-stale|h1-signed|h2-ctrl|h3-mag|h3-mag-stale) \
                 — refusing the silent fall-through to baseline",
                model_map.as_deref().unwrap_or_default()
            );
            assert!(
                rd_n_in > 0,
                "JXL_ZENSIM_MODEL_MAP='{}' is set but the loop profile has no \
                 probed feature width (JXL_ZENSIM_RD_PROFILE unset / profile A) \
                 — the arm would silently run as baseline; refusing",
                model_map.as_deref().unwrap_or_default()
            );
            let fold_family = matches!(model_map.as_deref(), Some("signed") | Some("abs"));
            assert!(
                !(folded_class && fold_family),
                "JXL_ZENSIM_MODEL_MAP='{}' requested on a folded-class bake \
                 (probed width {rd_n_in}): the signed/abs ModelSensitivity \
                 fold is 372-class only — use an attr-family arm \
                 (attr|attr-stale|h1-signed|h2-ctrl|h3-mag|h3-mag-stale), \
                 which routes through the fused folded-944 compare — \
                 refusing the silent downgrade to baseline",
                model_map.as_deref().unwrap_or_default()
            );
            // Folded-class + SINGLEPASS is LIVE since appendix P lever 2:
            // the stale-MAP single-pass (first steered iteration fused +
            // map cached; later iterations score-only extraction steering
            // with the cached map). The 372-class SINGLEPASS keeps its
            // stale-SCALAR meaning (`compute_with_ref_score_and_attribution_stale`)
            // unchanged. SINGLEPASS on a folded-class bake requires an
            // attr-family arm (checked by the fold-family refusal above).
        }
        let model_map_active = model_map_requested;
        let attr_mode = matches!(
            model_map.as_deref(),
            Some("attr")
                | Some("attr-stale")
                | Some("h1-signed")
                | Some("h2-ctrl")
                | Some("h3-mag")
                | Some("h3-mag-stale")
        );
        // `*-stale` steers with the PREVIOUS iteration's map (#69 G4 pricing
        // of the single-pass endpoint for a G1+G2-passing arm).
        let attr_stale = matches!(
            model_map.as_deref(),
            Some("attr-stale") | Some("h3-mag-stale")
        );
        let h_arm: Option<u8> = match model_map.as_deref() {
            Some("h1-signed") => Some(1),
            Some("h2-ctrl") => Some(2),
            Some("h3-mag") | Some("h3-mag-stale") => Some(3),
            _ => None,
        };
        let h3_gain: f32 = std::env::var("ZENSIM_H3_GAIN")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(10.0);
        // A-Y1 (campaign appendix Y): adaptive H3 gain shapes. Unset/"fixed"
        // = shipped constant gain (byte-identical); "decay" = gain·0.5^iter;
        // "err" = gain·min(1, |err_i|/|err_0|) — error-proportional, needs a
        // target (falls back to fixed without one).
        let h3_gain_mode = std::env::var("ZENSIM_H3_GAIN_MODE").ok();
        let mut h3_err0: Option<f64> = None;
        // A-Y3 (campaign appendix Y): EMA blend of the H-arm per-tile query
        // field across iterations — tile_q_i = α·fresh + (1−α)·prev, the
        // middle point between the fresh and stale endpoints. Unset = OFF
        // (byte-identical). Under stale-map single-pass the cached map makes
        // this a near-no-op by construction.
        let map_ema_alpha: Option<f32> = std::env::var("JXL_ZENSIM_MAP_EMA")
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|a: &f32| a.is_finite() && *a > 0.0 && *a < 1.0);
        let mut prev_tile_q: Vec<f32> = Vec::new();
        let mut model_s: Option<&'static [f64]> = None;
        // Level-2 binned attribution (zensim d0f624eb): JXL var-DCT tiles
        // are 8-px aligned and the loop reads the map ONLY through
        // `query_rect` on tile rects (8-multiples or image-edge clamped),
        // so `bin = 8` answers every steering query EXACTLY while the map
        // fold + SAT shrink 64× and no full-resolution canvas/trim/SAT is
        // built per compare. `ZENSIM_ATTR_BIN=1` restores the per-pixel
        // path (byte-identical pre-L2 behavior) for A/B.
        let attr_bin: usize = std::env::var("ZENSIM_ATTR_BIN")
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|b: &usize| *b >= 1)
            .unwrap_or(8);
        // attr mode: interleaved LinearF32Rgba view of the recon (reused),
        // and the previous iteration's map for the stale arm.
        let mut attr_rgba: Vec<f32> = Vec::new();
        let mut prev_attr: Option<zensim::AttributionResult> = None;
        let mut attr_session: Option<zensim::AttributionSession> = None;
        // Appendix N: the fused folded-944 session (extraction scratch +
        // walk retention, ~42 MB at 576² reused across compares), created
        // lazily at the first folded-class steered iteration; one per
        // encode.
        let mut fused944_session: Option<zensim::Fused944Session> = None;
        // C3b target-seeking controller (shared by ALL arms so the A/B
        // isolates the steering map): `JXL_ZENSIM_TARGET_SCORE=<native
        // 0-100>` adds a damped global quant-field step toward the target
        // each iteration (redistribution stays sum-preserving; the global
        // step owns the score dial), with early stop at
        // `JXL_ZENSIM_TARGET_TOL` (default 0.25). `JXL_ZENSIM_RD_STATS`
        // appends one TSV line per encode (iters, final score, ms/iter) —
        // experiment instrumentation, env-gated, defaults unchanged.
        let target_native: Option<f64> = std::env::var("JXL_ZENSIM_TARGET_SCORE")
            .ok()
            .and_then(|s| s.parse().ok());
        let target_tol: f64 = std::env::var("JXL_ZENSIM_TARGET_TOL")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.25);
        // #70 best-so-far emission (efficiency-study finding #7: the loop
        // emits the LAST iterate, and overshoot past the sweet spot does not
        // self-correct — h3 judged err 0.59@k6 → 1.02@k12). With
        // `JXL_ZENSIM_EMIT_BEST=1` AND a target set, snapshot the float quant
        // field measured by the compare whose INTERNAL score is closest to
        // the target (`<=`: the LATEST iterate wins exact ties, so a tied
        // last iterate emits exactly the default bitstream) and restore it
        // for the final SetQuantField — the emitted bitstream then
        // corresponds to the BEST measured iterate, not the last. The
        // internal fused score is the trusted proxy (E6: judged−internal med
        // ±0.13 on photos); no decoded scoring is added in-loop. Default
        // OFF: env unset, or no `JXL_ZENSIM_TARGET_SCORE` ("best" is
        // undefined without a target) = shipped emit-last behavior
        // (byte-identity gated in the study's R0).
        let emit_best = target_native.is_some()
            && matches!(
                std::env::var("JXL_ZENSIM_EMIT_BEST").ok().as_deref(),
                Some("1" | "true" | "yes")
            );
        let mut best_err = f64::INFINITY;
        let mut best_score = f64::NAN;
        let mut best_iter = usize::MAX;
        let mut best_qf: Vec<f32> = Vec::new();
        // Metric-matrix study (2026-07-31): env-gated override of the damped
        // Controller per-step clamp. DEFAULT 2.00 since the beats-butter
        // study (2026-08-07): at exp 1.0 the clamp dose-response peaks at
        // 2.0 (k3 census 20→24/27, nonphoto 3/9→7/9, photo unchanged,
        // med |err| 0.564→0.535; 2.5 regresses to 23) — the old 1.35 could
        // not descend far enough on the nonphoto overshoot class in 2-3
        // steps. Controller-only confirmation ran in the same study.
        // Values <= 1.0 are ignored (a step clamp must exceed 1).
        let ctrl_clamp: f64 = std::env::var("JXL_ZENSIM_CTRL_CLAMP")
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|c: &f64| c.is_finite() && *c > 1.0)
            .unwrap_or(2.00);
        // Controller exponent. DEFAULT 1.0 (pure proportional) since the
        // beats-butter study adopted the appendix-Y Part-2 result: monotone
        // dose-response 0.45<<0.6<0.8<1.0>=1.2 at BOTH budgets, 26W/1L vs
        // the old 0.6, med |err| 1.659→0.564 on the frontier arm;
        // controller-only is 19W/4L. Values outside (0, 2] are ignored.
        let ctrl_exp: f64 = std::env::var("JXL_ZENSIM_CTRL_EXP")
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|e: &f64| e.is_finite() && *e > 0.0 && *e <= 2.0)
            .unwrap_or(1.0);
        // Diffmap secant (plan §5.1, 2026-08-25): the global step uses a
        // measured elasticity ε̂ = Δln L / Δln S from the last two iterates
        // instead of the fixed-exponent power law. In THIS codebase higher
        // quant_field = more bits = LESS loss, so ε̂ is NEGATIVE; the secant is
        // used only when ε̂ < 0 and the last two scales differ, else the power
        // law is the fallback (and the mandatory first-iterate step). The
        // existing clamp still bounds the step. Default OFF (opt-in A/B arm).
        // Budget-optimal default (user directive 2026-08-25 "adjust defaults to
        // be optimal based on budget"): ON. Measured on the shipped C recipe
        // (benchmarks/zensim_secant_2026-08-25.md) — +3 k2 census, −55%/−71%
        // median error, no regression, at BOTH budgets. `JXL_ZENSIM_SECANT=0`
        // opts back to the pure power law.
        let secant_enabled: bool = std::env::var("JXL_ZENSIM_SECANT")
            .ok()
            .map(|s| s == "1" || s.eq_ignore_ascii_case("true"))
            .unwrap_or(true);
        let stats_path = std::env::var("JXL_ZENSIM_RD_STATS").ok();
        // Efficiency study (2026-07-31): per-COMPARE trace — one TSV line per
        // iteration: `trace_id  iter  score  qf_mean  qf_min  qf_max  iter_ms`.
        // Env-gated, append-only, default off. NO bytes column: the loop
        // never entropy-codes (no true bytes, no in-loop estimate) — a
        // registered limitation; per-iterate bytes come from budget-capped
        // full encodes (deterministic trajectory).
        let trace_path = std::env::var("JXL_ZENSIM_TRACE").ok();
        let trace_id = std::env::var("JXL_ZENSIM_TRACE_ID").unwrap_or_default();
        let t_loop = std::time::Instant::now();
        let mut iter_ms: Vec<f64> = Vec::new();
        let mut compares_used = 0usize;
        let mut final_score = f64::NAN;
        // Secant controller state (JXL_ZENSIM_SECANT): cumulative ln of the
        // controller's applied global scale (the g-product; the redistribution
        // is sum-preserving so it does not move global scale) + the previous
        // iterate's (ln S, ln L). NaN until the first controller step runs.
        let mut cum_log_s = 0.0f64;
        let mut prev_log_s = f64::NAN;
        let mut prev_log_l = f64::NAN;
        // Folded-class score route: the canonical folded extraction takes the
        // SOURCE as an ImageSource per compare (streaming, no prepared-ref
        // form since the C5 switchover), so build the tight LinearF32Rgba
        // interleave of the clean source once per encode.
        let mut folded_src_rgba: Vec<f32> = Vec::new();
        if folded_class {
            folded_src_rgba.resize(n * 4, 1.0);
            for i in 0..n {
                folded_src_rgba[i * 4] = linear_rgb[i * 3];
                folded_src_rgba[i * 4 + 1] = linear_rgb[i * 3 + 1];
                folded_src_rgba[i * 4 + 2] = linear_rgb[i * 3 + 2];
            }
        }
        // Appendix N: the folded-class gradient probe's feature base — the
        // FIRST compare's folded extraction (captured in the score branch,
        // consumed once by the derivation block below).
        let mut folded_feats_for_grad: Option<Vec<f64>> = None;

        // The deinterleaved planes are transient: only used to build the zensim
        // precomputed reference, then dropped.
        let precomputed = {
            let _g = MemoryBudget::reserve_opt(budget, (n as u64).saturating_mul(4 * 3))?;
            let (src_r, src_g, src_b) = deinterleave_rgb(linear_rgb, n);
            loud_compare(
                z.precompute_reference_linear_planar([&src_r, &src_g, &src_b], width, height, width),
                "precompute_reference_linear_planar (372-class reference precompute)",
            )
            // src_r/g/b drop here; _g releases the reservation.
        };

        let mut diffmap_opts = zensim::DiffmapOptions {
            weighting: zensim::DiffmapWeighting::Trained,
            include_edge_mse: params.include_edge_mse,
            include_hf: params.include_hf,
            masking_strength: params.masking_strength,
            sqrt: params.sqrt,
            guided_coarse_redistribution: false,
        };

        // Deviation bounds (same as butteraugli loop).
        let initial_qf_min = initial_quant_field_float
            .iter()
            .copied()
            .reduce(f32::min)
            .unwrap_or(0.01)
            .max(1e-6);
        let initial_qf_max = initial_quant_field_float
            .iter()
            .copied()
            .reduce(f32::max)
            .unwrap_or(1.0);
        let initial_qf_ratio = initial_qf_max / initial_qf_min;
        let qf_max_deviation_low = (250.0f32 / initial_qf_ratio).sqrt();
        let asymmetry = 2.0f32.min(qf_max_deviation_low);
        let qf_lower = initial_qf_min / (asymmetry * qf_max_deviation_low);
        let qf_higher = initial_qf_max * (qf_max_deviation_low / asymmetry);

        // Pre-allocate buffers reused across iterations.
        MemoryBudget::reserve_permanent_opt(
            budget,
            (num_blocks as u64)
                .saturating_add((num_blocks as u64).saturating_mul(4))
                .saturating_add((padded_pixels as u64).saturating_mul(4 * 3)),
        )?;
        let sharpness = vec![4u8; num_blocks];
        let mut tile_dist = vec![0.0f32; num_blocks];
        // #69 H arms: per-tile SIGNED mean attribution density + per-tile
        // total query (score units); `h_steered` marks iterations where an
        // H redistribution replaces the shared ratio redistribution.
        let mut tile_signed = vec![0.0f32; num_blocks];
        let mut tile_q = vec![0.0f32; num_blocks];
        let mut h_steered = false;
        let mut recon_r = vec![0.0f32; padded_pixels];
        let mut recon_g = vec![0.0f32; padded_pixels];
        let mut recon_b = vec![0.0f32; padded_pixels];
        let mut transform_out =
            super::transform::TransformOutput::new(xsize_blocks, ysize_blocks, budget)?;

        // Saturate at consumption — see butteraugli_loop.rs for rationale.
        let iters = (self.zensim_iters.min(crate::api::MAX_QUANT_LOOP_ITERS)) as usize;
        let mut current_params;

        for iter in 0..iters + 1 {
            let t_iter = std::time::Instant::now();
            // Step 1: SetQuantField — recompute global_scale from float field
            current_params =
                DistanceParams::compute_from_quant_field(target_distance, quant_field_float);
            current_params.x_qm_scale = initial_params.x_qm_scale;
            current_params.b_qm_scale = initial_params.b_qm_scale;
            current_params.epf_iters = initial_params.epf_iters;

            let qf_vec = quantize_quant_field(quant_field_float, current_params.inv_scale);
            quant_field.copy_from_slice(&qf_vec);

            // Step 2: Transform and quantize
            self.transform_and_quantize_into(
                xyb_x,
                xyb_y,
                xyb_b,
                padded_width,
                xsize_blocks,
                ysize_blocks,
                &current_params,
                quant_field,
                cfl_map,
                ac_strategy,
                &mut transform_out,
            );

            // Step 3: Reconstruct XYB → linear RGB
            let mut planes = reconstruct_xyb(
                &transform_out.quant_dc,
                &transform_out.quant_ac,
                &current_params,
                quant_field,
                cfl_map,
                ac_strategy,
                xsize_blocks,
                ysize_blocks,
            );

            if self.enable_gaborish {
                gab_smooth(&mut planes, padded_width, padded_height);
            }

            if current_params.epf_iters > 0 {
                epf::apply_epf(
                    &mut planes,
                    quant_field,
                    &sharpness,
                    current_params.scale,
                    current_params.epf_iters,
                    xsize_blocks,
                    ysize_blocks,
                    padded_width,
                    padded_height,
                    budget,
                )?;
            }

            if let Some(pd) = patches_data {
                super::patches::add_patches(&mut planes, padded_width, pd);
            }

            if let Some(sd) = splines_data {
                super::splines::add_splines(&mut planes, padded_width, width, height, sd);
            }

            // Step 4: Convert reconstructed XYB to planar linear RGB
            xyb_to_linear_rgb_planar(
                &planes[0],
                &planes[1],
                &planes[2],
                &mut recon_r,
                &mut recon_g,
                &mut recon_b,
                padded_pixels,
            );

            // Step 5+6, C3b attr arm (iterations 2+, s_k known): ONE fused
            // call = scalar score + attribution map; per-tile steering via
            // O(1) SAT rectangle queries (clamp-at-0 at the TILE level — the
            // signed-map analogue of the fold arm's per-pixel clamp — then
            // the IDENTICAL normalization/blend tail as `compute_tile_dist`).
            let dm_result;
            let zensim_score;
            let measured_dist;
            if let (true, Some(s)) = (attr_mode, model_s) {
                let n_px = width * height;
                attr_rgba.resize(n_px * 4, 1.0);
                for y in 0..height {
                    let row = y * padded_width;
                    let out = y * width * 4;
                    for x in 0..width {
                        attr_rgba[out + x * 4] = recon_r[row + x];
                        attr_rgba[out + x * 4 + 1] = recon_g[row + x];
                        attr_rgba[out + x * 4 + 2] = recon_b[row + x];
                        attr_rgba[out + x * 4 + 3] = 1.0;
                    }
                }
                let src = zensim::StridedBytes::with_alpha_mode(
                    bytemuck::cast_slice(&attr_rgba),
                    width,
                    height,
                    width * 16,
                    zensim::PixelFormat::LinearF32Rgba,
                    zensim::AlphaMode::Opaque,
                );
                // Appendix P lever 2 (stale-map single-pass): with
                // `JXL_ZENSIM_SINGLEPASS=1` on a folded-class bake, only
                // the FIRST steered iteration pays the fused compare (its
                // map is cached in `prev_attr`); every later steered
                // iteration is a SCORE-ONLY canonical extraction + forward
                // (the identical score route the baseline folded arm uses)
                // steering with the cached map — marginal map cost ≈ 0.
                // Level control stays with the damped controller + shared
                // sum-renormalization; staleness affects allocation SHAPE
                // only (gate G-P5: the 27-cell decoded-judged A/B).
                let stale_map_iter = folded_class && singlepass && prev_attr.is_some();
                let (res, attr, folded_score) = if stale_map_iter {
                    let src_clean = zensim::StridedBytes::with_alpha_mode(
                        bytemuck::cast_slice(&folded_src_rgba),
                        width,
                        height,
                        width * 16,
                        zensim::PixelFormat::LinearF32Rgba,
                        zensim::AlphaMode::Opaque,
                    );
                    let v2 = match rd_n_in {
                        944 => z.compute_folded720_append2_features(&src_clean, &src),
                        924 => z.compute_folded720_append_features(&src_clean, &src),
                        _ => z.compute_folded720_features(&src_clean, &src),
                    }
                    .expect("folded-class extraction failed (loud by design)");
                    let feats = v2.features();
                    let sc = zensim::score_features_with_profile(
                        rd_profile,
                        &feats[..rd_n_in.min(feats.len())],
                        width as u32,
                        height as u32,
                    )
                    .expect("folded-class forward failed after a passing mount probe (wiring bug)");
                    (None, None, Some(sc))
                } else if folded_class {
                    // Appendix N: the FUSED folded-944 compare — canonical
                    // 944 extraction (features bitwise the standalone
                    // extraction, G-N1) + the full-coverage attribution map
                    // from ONE extraction pass; the score is the bake's own
                    // forward over those features, exactly the baseline
                    // folded score path. Failures PANIC (study path, loud
                    // by design): the mount probe already proved the width,
                    // and the swallow would silently emit the seed.
                    let src_clean = zensim::StridedBytes::with_alpha_mode(
                        bytemuck::cast_slice(&folded_src_rgba),
                        width,
                        height,
                        width * 16,
                        zensim::PixelFormat::LinearF32Rgba,
                        zensim::AlphaMode::Opaque,
                    );
                    let sess = fused944_session.get_or_insert_with(zensim::Fused944Session::new);
                    let (res, v2, attr) = z
                        .compute_folded944_score_and_attribution_binned(
                            &src_clean,
                            &precomputed,
                            &src,
                            s,
                            sess,
                            attr_bin,
                        )
                        .expect("fused folded-944 compare failed (loud by design)");
                    let feats = v2.features();
                    let sc = zensim::score_features_with_profile(
                        rd_profile,
                        &feats[..rd_n_in.min(feats.len())],
                        width as u32,
                        height as u32,
                    )
                    .expect("folded-class forward failed after a passing mount probe (wiring bug)");
                    (Some(res), Some(attr), Some(sc))
                } else if singlepass {
                    let sess = attr_session.get_or_insert_with(zensim::AttributionSession::new);
                    {
                        let (r, a) = loud_compare(
                            z.compute_with_ref_score_and_attribution_stale_binned(
                                &precomputed,
                                &src,
                                s,
                                sess,
                                attr_bin,
                            ),
                            "compute_with_ref_score_and_attribution_stale_binned (372-class stale single-pass arm)",
                        );
                        (Some(r), Some(a), None)
                    }
                } else {
                    {
                        let (r, a) = loud_compare(
                            z.compute_with_ref_score_and_attribution_binned(
                                &precomputed,
                                &src,
                                s,
                                attr_bin,
                            ),
                            "compute_with_ref_score_and_attribution_binned (372-class fused arm)",
                        );
                        (Some(r), Some(a), None)
                    }
                };
                zensim_score = match folded_score {
                    Some(sc) => sc,
                    None => res
                        .as_ref()
                        .expect("non-folded arms carry a result")
                        .score(),
                };
                // Log-only value; the score-only stale-map path runs no v1
                // walk, so there is nothing to approximate — NaN, never 0.
                measured_dist = res
                    .as_ref()
                    .map(|r| r.approx_butteraugli() as f32)
                    .unwrap_or(f32::NAN);
                // Stale arm: steer with the PREVIOUS iteration's map when
                // one exists (first steered iteration uses the fresh map).
                // Stale-map single-pass iterations HAVE no fresh map — the
                // cached one is the steering signal by construction.
                let steer_map = match (&attr, attr_stale) {
                    (Some(a), true) => prev_attr.as_ref().unwrap_or(a),
                    (Some(a), false) => a,
                    (None, _) => prev_attr
                        .as_ref()
                        .expect("stale-map single-pass without a cached map (wiring bug)"),
                };
                if h_arm.is_some() {
                    compute_tile_signed_attr(
                        steer_map,
                        ac_strategy,
                        xsize_blocks,
                        ysize_blocks,
                        &mut tile_signed,
                        &mut tile_q,
                    );
                    // A-Y3: EMA-blend the per-tile query field with the
                    // previous iteration's (the fresh↔stale middle point).
                    if let Some(a) = map_ema_alpha {
                        if prev_tile_q.len() == tile_q.len() {
                            for (q, p) in tile_q.iter_mut().zip(prev_tile_q.iter()) {
                                *q = a * *q + (1.0 - a) * *p;
                            }
                        }
                        prev_tile_q.clear();
                        prev_tile_q.extend_from_slice(&tile_q);
                    }
                    // tile_dist only feeds the shared log line on this path.
                    tile_dist.fill(target_distance);
                    h_steered = true;
                } else {
                    compute_tile_dist_attr(
                        steer_map,
                        ac_strategy,
                        xsize_blocks,
                        ysize_blocks,
                        target_distance,
                        &mut tile_dist,
                        &params,
                    );
                }
                // Smoke-test probe: one line per attr-steered iteration with
                // the tile-dist stats (env-gated; test asserts sane values).
                if let Ok(pp) = std::env::var("JXL_ZENSIM_ATTR_PROBE") {
                    use std::io::Write;
                    let probe_src: &[f32] = if h_arm.is_some() {
                        &tile_signed
                    } else {
                        &tile_dist
                    };
                    let (mut mn, mut mx, mut sum) = (f32::INFINITY, f32::NEG_INFINITY, 0.0f64);
                    for &v in probe_src.iter() {
                        mn = mn.min(v);
                        mx = mx.max(v);
                        sum += v as f64;
                    }
                    if let Ok(mut f) = std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(pp)
                    {
                        let _ = writeln!(
                            f,
                            "attr_iter\t{mn}\t{mx}\t{}",
                            sum / probe_src.len().max(1) as f64
                        );
                    }
                }
                // Cache the map: lagged-fresh for the `*-stale` arms; the
                // FIRST fused map for the folded-class single-pass (later
                // iterations produce none and keep reusing it).
                if let Some(attr) = attr
                    && (attr_stale || (folded_class && singlepass))
                {
                    prev_attr = Some(attr);
                }
                drop(res);
                dm_result = None;
            } else {
                // Step 5: Zensim comparison → global score + per-pixel diffmap.
                // Pass planar linear RGB directly — no RGBA interleaving, no byte cast.
                let dm = loud_compare(
                    z.compute_with_ref_and_diffmap_linear_planar(
                        &precomputed,
                        [&recon_r, &recon_g, &recon_b],
                        width,
                        height,
                        padded_width, // stride = padded_width (encoder pads rows for SIMD)
                        diffmap_opts,
                    ),
                    "compute_with_ref_and_diffmap_linear_planar (372-class two-pass diffmap arm)",
                );
                zensim_score = if folded_class {
                    // Folded-class score: canonical 720/924/944 extraction over
                    // (clean source, current recon) + full-bundle forward in
                    // the bake's own units. Failures PANIC (study path, loud
                    // by design): the mount probe already proved the width, so
                    // an error here is a wiring bug, and the alternative — the
                    // compare-error swallow above — silently emits the seed.
                    let n_px = width * height;
                    attr_rgba.resize(n_px * 4, 1.0);
                    for y in 0..height {
                        let row = y * padded_width;
                        let out = y * width * 4;
                        for x in 0..width {
                            attr_rgba[out + x * 4] = recon_r[row + x];
                            attr_rgba[out + x * 4 + 1] = recon_g[row + x];
                            attr_rgba[out + x * 4 + 2] = recon_b[row + x];
                            attr_rgba[out + x * 4 + 3] = 1.0;
                        }
                    }
                    let src = zensim::StridedBytes::with_alpha_mode(
                        bytemuck::cast_slice(&folded_src_rgba),
                        width,
                        height,
                        width * 16,
                        zensim::PixelFormat::LinearF32Rgba,
                        zensim::AlphaMode::Opaque,
                    );
                    let dst = zensim::StridedBytes::with_alpha_mode(
                        bytemuck::cast_slice(&attr_rgba),
                        width,
                        height,
                        width * 16,
                        zensim::PixelFormat::LinearF32Rgba,
                        zensim::AlphaMode::Opaque,
                    );
                    let v2 = match rd_n_in {
                        944 => z.compute_folded720_append2_features(&src, &dst),
                        924 => z.compute_folded720_append_features(&src, &dst),
                        _ => z.compute_folded720_features(&src, &dst),
                    }
                    .expect("folded-class extraction failed (loud by design)");
                    // Appendix N: a folded-class attr-family arm derives its
                    // gradient from the FOLDED features (the bake's true
                    // inputs) — capture the first compare's vector for the
                    // probe below (the 372-class walk's features are the
                    // wrong regime for a folded bake's forward).
                    if model_map_active && model_s.is_none() {
                        folded_feats_for_grad = Some(v2.features().to_vec());
                    }
                    zensim::score_features_with_profile(
                        rd_profile,
                        v2.features(),
                        width as u32,
                        height as u32,
                    )
                    .expect("folded-class forward failed after a passing mount probe (wiring bug)")
                } else {
                    dm.score()
                };
                let diffmap = dm.diffmap();

                // RD-experiment: a SIGNED model-sensitivity map carries negative
                // values where refining LOSES score. The L4 tile fold below is
                // sign-blind, so clamp at 0 — those tiles read as "no error" and
                // the redistribution coarsens them (the directionally-correct
                // action). No-op for the non-negative Trained map.
                let clamped_map;
                let diffmap: &[f32] = if model_s.is_some() {
                    clamped_map = diffmap.iter().map(|&v| v.max(0.0)).collect::<Vec<f32>>();
                    &clamped_map
                } else {
                    diffmap
                };

                // Step 6: Compute per-block error from zensim diffmap.
                // Use L4 norm per tile to capture spatial error distribution.
                measured_dist = dm.result().approx_butteraugli() as f32;
                compute_tile_dist(
                    diffmap,
                    width,
                    height,
                    ac_strategy,
                    xsize_blocks,
                    ysize_blocks,
                    target_distance,
                    &mut tile_dist,
                    &params,
                );
                dm_result = Some(dm);
            }

            // RD-experiment: derive the model-sensitivity weighting from the
            // FIRST compare's features (numerical central differences through
            // the production features→score runtime; `abs` fold = pass −|s|
            // through the signed ModelSensitivity fold). Iterations 2+ steer
            // with the model's own gradient.
            if model_map_active && model_s.is_none() {
                // Gradient base: folded-class arms use the FOLDED features
                // captured by the first compare's score extraction (appendix
                // N — the bake's true input regime); 372-class arms read the
                // compare walk's features as before.
                let base_opt: Option<Vec<f64>> = if folded_class {
                    folded_feats_for_grad
                        .take()
                        .map(|f| f[..rd_n_in.min(f.len())].to_vec())
                } else {
                    dm_result.as_ref().and_then(|dm| {
                        let feats = dm.result().features();
                        (feats.len() >= rd_n_in).then(|| feats[..rd_n_in].to_vec())
                    })
                };
                if let Some(base) = base_opt {
                    // L-Y1 (campaign appendix Y): the batched FD-gradient
                    // entry parses the bake once + reuses one predictor for
                    // all 2·N forwards — bitwise-equal to the sequential
                    // recipe (zensim gate `fd_gradient_bitwise_matches_
                    // sequential`). `JXL_ZENSIM_FDPROBE=seq` keeps the
                    // legacy per-probe path for A/B verification.
                    let seq_probe = std::env::var("JXL_ZENSIM_FDPROBE").is_ok_and(|v| v == "seq");
                    let mut s = if seq_probe {
                        let sf = |f: &[f64]| {
                            zensim::score_features_with_profile(
                                rd_profile,
                                f,
                                width as u32,
                                height as u32,
                            )
                            .unwrap_or(f64::NAN)
                        };
                        let mut s = vec![0.0f64; rd_n_in];
                        let mut probe = base.clone();
                        for (k, sk) in s.iter_mut().enumerate() {
                            let eps = (base[k].abs() * 1e-3).max(1e-5);
                            probe[k] = base[k] + eps;
                            let up = sf(&probe);
                            probe[k] = base[k] - eps;
                            let dn = sf(&probe);
                            probe[k] = base[k];
                            *sk = if up.is_finite() && dn.is_finite() {
                                (up - dn) / (2.0 * eps)
                            } else {
                                0.0
                            };
                        }
                        s
                    } else {
                        // Err (model failed to parse) matches the sequential
                        // recipe's all-NaN → all-zero gradient disposition.
                        zensim::score_features_fd_gradient_with_profile(
                            rd_profile,
                            &base,
                            width as u32,
                            height as u32,
                        )
                        .unwrap_or_else(|_| vec![0.0f64; rd_n_in])
                    };
                    if model_map.as_deref() == Some("abs") {
                        for v in &mut s {
                            *v = -v.abs();
                        }
                    }
                    let s: &'static [f64] = Box::leak(s.into_boxed_slice());
                    model_s = Some(s);
                    if !attr_mode {
                        diffmap_opts.weighting = zensim::DiffmapWeighting::ModelSensitivity(s);
                        diffmap_opts.sqrt = false; // sqrt(negative) is NaN on a signed map
                    }
                }
            }

            // Strategy refinement disabled: splitting large transforms with high
            // error causes size inflation (+5-12%) without proportional quality gain.
            // Pure quant redistribution is more size-neutral.
            // TODO: re-enable with better thresholds once RD impact is characterized.

            // Log per-iteration summary
            {
                let qf_min = quant_field_float
                    .iter()
                    .copied()
                    .reduce(f32::min)
                    .unwrap_or(0.0);
                let qf_max = quant_field_float
                    .iter()
                    .copied()
                    .reduce(f32::max)
                    .unwrap_or(0.0);
                let qf_sum: f64 = quant_field_float.iter().map(|&v| v as f64).sum();
                let qf_avg = qf_sum / quant_field_float.len() as f64;
                let td_max = tile_dist.iter().copied().reduce(f32::max).unwrap_or(0.0);
                debug_rect!(
                    "zensim/iter",
                    0,
                    0,
                    width,
                    height,
                    // bfly≈NaN on the stale score-only arm (no v1 walk to approximate) — expected, log-only.
                    "iter={}/{} zensim={:.2} bfly≈{:.3} target={:.3} gs={} qf_avg={:.4} qf=[{:.4};{:.4}] td_max={:.2}",
                    iter,
                    iters,
                    zensim_score,
                    measured_dist,
                    target_distance,
                    current_params.global_scale,
                    qf_avg,
                    qf_min,
                    qf_max,
                    td_max
                );
            }

            compares_used = iter + 1;
            final_score = zensim_score;
            // #70 emit-best: `quant_field_float` still holds the field this
            // compare MEASURED (redistribution/controller run later), so a
            // snapshot here reproduces this iterate exactly at emission.
            if let (true, Some(tgt)) = (emit_best, target_native) {
                let err = (zensim_score - tgt).abs();
                if err.is_finite() && err <= best_err {
                    best_err = err;
                    best_score = zensim_score;
                    best_iter = iter;
                    best_qf.clear();
                    best_qf.extend_from_slice(quant_field_float);
                }
            }
            iter_ms.push(t_iter.elapsed().as_secs_f64() * 1e3);
            // Efficiency-study trace (env-gated; see decl above). The qf
            // stats describe the field MEASURED by this compare (the field
            // that produced iterate `iter`), not the post-redistribution one.
            if let Some(tp) = &trace_path {
                use std::io::Write;
                let (mut qmn, mut qmx, mut qsum) = (f32::INFINITY, f32::NEG_INFINITY, 0.0f64);
                for &v in quant_field_float.iter() {
                    qmn = qmn.min(v);
                    qmx = qmx.max(v);
                    qsum += v as f64;
                }
                let qmean = qsum / quant_field_float.len().max(1) as f64;
                let ms = iter_ms.last().copied().unwrap_or(f64::NAN);
                if let Ok(mut f) = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(tp)
                {
                    let _ = writeln!(
                        f,
                        "{trace_id}\t{iter}\t{zensim_score:.4}\t{qmean:.6}\t{qmn:.6}\t{qmx:.6}\t{ms:.1}"
                    );
                }
            }
            // Target convergence: stop once the measured score is within
            // tolerance (needs >= 1 redistribution behind it so the map
            // actually steered something).
            if let Some(tgt) = target_native
                && iter >= 1
                && (zensim_score - tgt).abs() <= target_tol
            {
                break;
            }
            // Last iteration is compare-only
            if iter == iters {
                break;
            }

            // Step 7 (#69 H arms): the arm-specific redistribution rule.
            // Shared with every other arm: qf deviation bounds, the exact
            // sum-renormalization, and the damped controller below.
            let h_this_iter = h_steered;
            h_steered = false;
            if h_this_iter {
                let n_f = num_blocks as f64;
                let mean_s: f32 = (tile_signed.iter().map(|&v| v as f64).sum::<f64>() / n_f) as f32;
                let qf_sum_before: f64 = quant_field_float.iter().map(|&v| v as f64).sum();
                match h_arm.unwrap_or(0) {
                    1 | 2 => {
                        // H1: signed value as-is; H2: zero-sum residual.
                        let center = if h_arm == Some(2) { mean_s } else { 0.0 };
                        let mean_abs: f32 = (tile_signed
                            .iter()
                            .map(|&v| (v - center).abs() as f64)
                            .sum::<f64>()
                            / n_f) as f32;
                        if mean_abs > 1e-20 {
                            // Adaptive alpha mirrors the shared rule:
                            // alpha_base / (1 + CV of the |field|).
                            let var: f64 = tile_signed
                                .iter()
                                .map(|&v| {
                                    let d = ((v - center).abs() - mean_abs) as f64;
                                    d * d
                                })
                                .sum::<f64>()
                                / n_f;
                            let cv = (var.sqrt() / mean_abs as f64) as f32;
                            let k_alpha = params.alpha_base / (1.0 + cv);
                            for bi in 0..num_blocks {
                                let r = ((tile_signed[bi] - center) / mean_abs)
                                    .clamp(-params.ratio_max, params.ratio_max);
                                let factor = (1.0 + k_alpha * r)
                                    .clamp(1.0 / params.factor_max, params.factor_max);
                                quant_field_float[bi] =
                                    (quant_field_float[bi] * factor).clamp(qf_lower, qf_higher);
                            }
                        }
                    }
                    _ => {
                        // H3: step ∝ per-tile query_rect score-units, capped.
                        // A-Y1: gain optionally adapts per iteration (the
                        // default "fixed" reproduces the shipped constant).
                        let h3_gain_eff: f32 = match h3_gain_mode.as_deref() {
                            Some("decay") => h3_gain * 0.5f32.powi(iter as i32),
                            Some("err") => {
                                if let Some(tgt) = target_native {
                                    let err_i = (zensim_score - tgt).abs();
                                    let err0 = *h3_err0.get_or_insert(err_i.max(1e-9));
                                    h3_gain * ((err_i / err0).min(1.0) as f32)
                                } else {
                                    h3_gain
                                }
                            }
                            _ => h3_gain,
                        };
                        for bi in 0..num_blocks {
                            let factor = (1.0 + h3_gain_eff * tile_q[bi])
                                .clamp(1.0 / params.factor_max, params.factor_max);
                            quant_field_float[bi] =
                                (quant_field_float[bi] * factor).clamp(qf_lower, qf_higher);
                        }
                    }
                }
                // Shared sum-renormalization (size control), as in every arm.
                let qf_sum_after: f64 = quant_field_float.iter().map(|&v| v as f64).sum();
                if qf_sum_after > 1e-10 {
                    let scale = (qf_sum_before / qf_sum_after) as f32;
                    for v in quant_field_float.iter_mut() {
                        *v *= scale;
                    }
                }
            }

            // Step 7: Symmetric sum-preserving redistribution of quant_field.
            //
            // Tiles with above-average error get lower quant_field (more bits),
            // tiles with below-average error get higher quant_field (fewer bits).
            // Adaptive K_ALPHA scales by 1/(1+CV). Factor clamped per iteration.
            let td_sum: f64 = tile_dist.iter().map(|&v| v as f64).sum();
            let td_avg = (td_sum / num_blocks as f64) as f32;

            if !h_this_iter && td_avg > 1e-10 {
                let td_var: f64 = tile_dist
                    .iter()
                    .map(|&v| {
                        let d = v as f64 - td_avg as f64;
                        d * d
                    })
                    .sum::<f64>()
                    / num_blocks as f64;
                let td_cv = (td_var.sqrt() / td_avg as f64) as f32;
                let k_alpha = params.alpha_base / (1.0 + td_cv);
                let factor_max = params.factor_max;

                let qf_sum_before: f64 = quant_field_float.iter().map(|&v| v as f64).sum();

                for bi in 0..num_blocks {
                    let ratio = tile_dist[bi] / td_avg;
                    let factor =
                        (1.0 + k_alpha * (ratio - 1.0)).clamp(1.0 / factor_max, factor_max);
                    quant_field_float[bi] *= factor;

                    if quant_field_float[bi] > qf_higher {
                        quant_field_float[bi] = qf_higher;
                    }
                    if quant_field_float[bi] < qf_lower {
                        quant_field_float[bi] = qf_lower;
                    }
                }

                // Renormalize: preserve original sum to control size growth.
                let qf_sum_after: f64 = quant_field_float.iter().map(|&v| v as f64).sum();
                if qf_sum_after > 1e-10 {
                    let scale = (qf_sum_before / qf_sum_after) as f32;
                    for v in quant_field_float.iter_mut() {
                        *v *= scale;
                    }
                }
            }

            // C3b: damped global step toward the native-score target.
            // Loss (100 − score) above target ⇒ too lossy ⇒ scale the whole
            // field up (more bits). Defaults JXL_ZENSIM_CTRL_EXP=1.0 +
            // JXL_ZENSIM_CTRL_CLAMP=2.00 (adopted 2026-08-07, campaign
            // appendix AB.3; the earlier 0.6 / 1.35 was superseded there).
            if let Some(tgt) = target_native {
                let achieved_loss = (100.0 - zensim_score).max(0.05);
                let target_loss = (100.0 - tgt).max(0.05);
                let cur_log_l = achieved_loss.ln();
                // Power-law step: the default, and the secant's fallback.
                let g_pow = (achieved_loss / target_loss).powf(ctrl_exp);
                let g_raw = if secant_enabled
                    && prev_log_l.is_finite()
                    && (cum_log_s - prev_log_s).abs() > 1e-6
                    // Loss must have MOVED enough for a reliable elasticity —
                    // near-equal iterates give ε̂ ≈ 0 and a divide-by-noise step
                    // (the 2026-08-25 smoke's t70 iter-2 overshoot). Fall back
                    // to the power law when it barely moved.
                    && (cur_log_l - prev_log_l).abs() > 1e-3
                {
                    // ε̂ = Δln L / Δln S; valid only when loss falls as scale
                    // rises (ε̂ < 0). Step: ln S_target = ln S + (ln L_t − ln L)/ε̂.
                    let eps_hat = (cur_log_l - prev_log_l) / (cum_log_s - prev_log_s);
                    if eps_hat < -1e-6 {
                        ((target_loss.ln() - cur_log_l) / eps_hat).exp()
                    } else {
                        g_pow
                    }
                } else {
                    g_pow
                };
                let g = g_raw.clamp(1.0 / ctrl_clamp, ctrl_clamp);
                // Record the iterate (S, L) at which this score was measured,
                // then advance the cumulative scale by the applied step.
                prev_log_s = cum_log_s;
                prev_log_l = cur_log_l;
                cum_log_s += g.ln();
                let gf = g as f32;
                for v in quant_field_float.iter_mut() {
                    *v = (*v * gf).clamp(qf_lower, qf_higher);
                }
            }
        }

        // #70 emit-best: restore the best iterate's measured field for the
        // final SetQuantField below. Skipped when the last executed compare
        // WAS the best (`<=` tracking makes the latest tie win), so the
        // emitted bitstream is bit-for-bit the default one in that case.
        // `final_score` (the RD_STATS line) reports the EMITTED iterate's
        // internal score — the honest value for downstream consumers.
        if emit_best && best_iter != usize::MAX && best_iter + 1 != compares_used {
            quant_field_float.copy_from_slice(&best_qf);
            final_score = best_score;
            debug_rect!(
                "zensim/emit-best",
                0,
                0,
                width,
                height,
                "restoring iterate {} (score {:.3}, |err| {:.3}) over last iterate {}",
                best_iter,
                best_score,
                best_err,
                compares_used - 1
            );
        }

        if let Some(sp) = &stats_path {
            use std::io::Write;
            let per_iter = iter_ms
                .iter()
                .map(|v| format!("{v:.1}"))
                .collect::<Vec<_>>()
                .join(",");
            let line = format!(
                "{compares_used}\t{final_score:.4}\t{:.1}\t{per_iter}\n",
                t_loop.elapsed().as_secs_f64() * 1e3
            );
            if let Ok(mut f) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(sp)
            {
                let _ = f.write_all(line.as_bytes());
            }
        }

        // Final SetQuantField
        let mut final_params =
            DistanceParams::compute_from_quant_field(target_distance, quant_field_float);
        final_params.x_qm_scale = initial_params.x_qm_scale;
        final_params.b_qm_scale = initial_params.b_qm_scale;
        final_params.epf_iters = initial_params.epf_iters;

        let qf_vec = quantize_quant_field(quant_field_float, final_params.inv_scale);
        quant_field.copy_from_slice(&qf_vec);

        Ok(final_params)
    }
}

/// Deinterleave `[R, G, B, R, G, B, ...]` into three separate channel buffers.
fn deinterleave_rgb(rgb: &[f32], n: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut r = vec![0.0f32; n];
    let mut g = vec![0.0f32; n];
    let mut b = vec![0.0f32; n];
    for i in 0..n {
        r[i] = rgb[i * 3];
        g[i] = rgb[i * 3 + 1];
        b[i] = rgb[i * 3 + 2];
    }
    (r, g, b)
}

#[cfg(test)]
mod c10_loud_tests {
    #[test]
    #[should_panic(expected = "compare failed")]
    fn forced_compare_failure_panics_with_arm_name() {
        super::loud_compare::<(), &str>(Err("forced"), "unit-test-arm (forced failure)");
    }
}
