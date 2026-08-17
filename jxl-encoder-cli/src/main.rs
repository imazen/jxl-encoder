// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing
#![forbid(unsafe_code)]

//! Command-line JPEG XL encoder.

use clap::{Parser, ValueEnum};
use jxl_encoder::{
    AnimationFrame, AnimationParams, Buffering, ContainerMode, EpfDispatch, LosslessConfig,
    LossyConfig, Lz77Method, PixelLayout, PixelLossDispatch, PremultipliedAlphaMode,
    ProgressiveMode,
};
// W44-131 Chunk E — `--strategy` CLI flag. The four named variants
// (`libjxl`, `lean-faster`, `zenjxl`, `aggressive`) map to
// [`jxl_encoder::api::EncoderStrategy`] variants of the same name.
// `Custom` is API-only per design doc §7 Q7 (`docs/COMPATIBILITY_MODES.md`).

/// CLI-facing strategy enum. Mirrors
/// [`jxl_encoder::api::EncoderStrategy`] minus the API-only `Custom`
/// payload. See `docs/COMPATIBILITY_MODES.md` §4.1 for behaviour.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, ValueEnum)]
enum StrategyArg {
    /// Strict libjxl-parity bundle. Disables every W44-* improvement,
    /// flips Section A effort-gate divergences, and re-enables the
    /// Section D KNOWN-BUG that we currently work around. Verified
    /// against `cjxl` output as a regression gate. Bytes may be
    /// LARGER than `zenjxl` — that IS the point.
    Libjxl,
    /// Lean Faster. Drops the heavy per-image content gates and the
    /// EPF/buttloop corrections to keep encode time leaner. Keeps the
    /// at-parity algorithm fixes plus the cheap photo-class entropy-mul
    /// lowering. Effort-gate divergences stay at ours (not libjxl).
    LeanFaster,
    /// Zenjxl — production default. What we ship today. Every
    /// Section B content-aware gate fires on its auto discriminator.
    /// Byte-identical to invoking `cjxl-rs` without `--strategy`.
    #[default]
    Zenjxl,
    /// Aggressive — forward-compatible slot for opt-ins with
    /// too-narrow Zenjxl discriminators. Currently equivalent to
    /// `zenjxl` (W44-124 obsoleted the prior global DCT32 keep lift).
    Aggressive,
}

/// Apply the CLI `--strategy` pick to a [`LossyConfig`] together with
/// the per-dispatch overrides parsed from `--epf-dispatch` and
/// `--pixel-loss-dispatch`. For `Zenjxl` (default) we keep the
/// existing W44-130 Chunk D wiring — wrap the dispatches into a
/// `Custom` payload whose other fields are `Default::default()` (=
/// Zenjxl-equivalent), so byte output is identical to the no-`--strategy`
/// CLI invocation. For `Libjxl`/`LeanFaster`/`Aggressive` we use the
/// bare variant directly; dispatch overrides are dropped (a warning
/// is emitted upstream).
fn apply_strategy_to_lossy(
    cfg: LossyConfig,
    strategy: StrategyArg,
    epf_dispatch: EpfDispatch,
    pixel_loss_dispatch: PixelLossDispatch,
) -> LossyConfig {
    use jxl_encoder::api::{EncoderImprovementsCustom, EncoderStrategy};
    match strategy {
        StrategyArg::Zenjxl => {
            // Byte-identical to pre-Chunk-E CLI: wrap dispatches into
            // a `Custom` payload defaulted to Zenjxl-equivalent fields.
            let custom = EncoderImprovementsCustom {
                epf_dispatch,
                pixel_loss_dispatch,
                ..EncoderImprovementsCustom::default()
            };
            cfg.with_strategy(EncoderStrategy::Custom(Box::new(custom)))
        }
        StrategyArg::Libjxl => cfg.with_strategy(EncoderStrategy::Libjxl),
        StrategyArg::LeanFaster => cfg.with_strategy(EncoderStrategy::LeanFaster),
        StrategyArg::Aggressive => cfg.with_strategy(EncoderStrategy::Aggressive),
    }
}
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(name = "cjxl-rs")]
#[command(author, version, about = "JPEG XL encoder in Rust", long_about = None)]
struct Args {
    /// Input image file (PNG or PNM)
    #[arg(required = true)]
    input: PathBuf,

    /// Output JXL file
    #[arg(required = true)]
    output: PathBuf,

    /// Quality setting (0-100, 100 = lossless)
    #[arg(short, long, default_value = "90")]
    quality: u32,

    /// Effort level (1-12, higher = slower but better compression).
    /// e10/e11/e12 are our extensions past libjxl's kTortoise=9: longer
    /// butteraugli convergence (8 / 16 / 32 iters) plus multi-seed tree
    /// learning + multi-seed butteraugli sweep at e10/e11. Bitstreams stay
    /// 100% spec-valid. See RFC issue #45.
    #[arg(short, long, default_value = "7")]
    effort: u8,

    /// Force lossless encoding
    #[arg(long)]
    lossless: bool,

    /// Encoder strategy bundle (lossy only). One of `libjxl`,
    /// `lean-faster`, `zenjxl`, `aggressive`. Default `zenjxl`
    /// (production-shipping behaviour). Sets the high-level
    /// per-divergence policy stack documented in
    /// `docs/COMPATIBILITY_MODES.md` §4.1. `libjxl` is strict
    /// libjxl-parity (every W44-* improvement off, Section A
    /// effort-gate flips on, Section D KNOWN-BUG re-enabled —
    /// bytes may be larger than `zenjxl`, that IS the point).
    /// `Custom` is API-only — drive it from Rust via
    /// `LossyConfig::with_strategy(EncoderStrategy::Custom(...))`
    /// (W44-131 Chunk E). Mutually exclusive with `--lossless`.
    /// Until Chunk G ships, `--strategy libjxl` flips only the
    /// Section B/D divergences; the Section A effort-gate
    /// consultation lands in Chunk G.
    #[arg(
        long,
        value_enum,
        default_value_t = StrategyArg::default(),
        conflicts_with = "lossless",
    )]
    strategy: StrategyArg,

    /// Distance (alternative to quality, 0 = lossless, 1 = visually lossless)
    #[arg(short, long)]
    distance: Option<f32>,

    /// Disable dynamic Huffman code optimization (use static codes)
    #[arg(long)]
    no_optimize_codes: bool,

    /// Use Huffman instead of ANS entropy coding (ANS is default, 4-10% smaller)
    #[arg(long)]
    no_ans: bool,

    /// Disable custom coefficient ordering
    #[arg(long)]
    no_custom_orders: bool,

    /// Enable noise synthesis (estimates and encodes noise parameters)
    #[arg(long)]
    noise: bool,

    /// Enable Wiener denoising pre-filter (implies --noise)
    /// Removes estimated noise before encoding; decoder re-adds it.
    /// Provides 1-8% file savings with near-zero perceptual quality impact.
    #[arg(long)]
    denoise: bool,

    /// Synthesise camera-ISO photon noise instead of estimating from
    /// content. Mirrors libjxl `--photon_noise=ISO`. Bypasses
    /// `--noise`. Typical: 100 (bright outdoors), 800 (indoor),
    /// 6400+ (low-light grainy).
    #[arg(long, value_name = "ISO")]
    photon_noise_iso: Option<f32>,

    /// Caller-supplied source-image distance for re-encode pipelines.
    /// Mirrors libjxl `--original_butteraugli_distance`. When the
    /// source is already lossy (re-encoding a JPEG / JXL), pass its
    /// approximate distance; the encoder's distance-based heuristics
    /// (x_qm_scale) ramp against this rather than the target.
    #[arg(long, value_name = "DIST")]
    original_distance: Option<f32>,

    /// Multiplier on the AC quantiser's `global_scale` after the
    /// standard distance-driven compute. Mirrors libjxl
    /// `--quant_ac_rescale`. r < 1.0 → finer AC quant (larger files,
    /// higher quality); r > 1.0 → coarser. Reasonable range
    /// 0.5..=2.0; 1.0 is no-op.
    #[arg(long, value_name = "R")]
    quant_ac_rescale: Option<f32>,

    /// Force a specific Reversible Color Transform colorspace for
    /// lossless encoding (skips the per-effort RCT search). Mirrors
    /// libjxl `--colorspace`. Common values: 0 (none), 3
    /// (subtract-green), 6 (YCoCg, default fallback). Full RCT table
    /// 0..=41 (permutation × transform).
    #[arg(long, value_name = "RCT")]
    force_rct: Option<u8>,

    /// Disable all encoder-side perceptual heuristics (gaborish,
    /// patches, dot detection, noise, pixel-domain loss) in one
    /// switch. Mirrors libjxl `--disable_perceptual_optimizations`.
    /// Useful for spec-strict mode, decoder testing, picker training
    /// without perceptual confounds.
    #[arg(long)]
    no_perceptual_optimizations: bool,

    /// Override the tree-learning pixel-sampling fraction for
    /// lossless effort 7+. Lower values trade compression for speed
    /// (refs #23 — the e6→e7 cliff). Reasonable range 0.10..=0.50;
    /// libjxl effort defaults are 0.50 at e7, 0.55 at e8, 0.65 at e9+.
    /// Use 0.10..=0.20 for an "e7-lite" tier between e6 (no tree)
    /// and e7-default. Clamped to [0.0, 1.0].
    #[arg(long, value_name = "F")]
    tree_learning_sample_fraction: Option<f32>,

    /// Per-image smart-fanout for parallel tree learning (lossless only).
    /// Re-tunes `tree_parallel_max_depth` / `tree_parallel_floor` based
    /// on input pixel count instead of effort alone. Bitstream-equivalent;
    /// only changes rayon fanout shape. Wins 5-18% wall-clock on
    /// small/medium photos at e7/e8/e9; large+e9 is unchanged (per
    /// `smart_fanout_sweep_2026-05-17`).
    #[arg(long)]
    smart_fanout: bool,

    /// Enable the opt-in small-image parallel-tree-learning fallback
    /// (lossless only). When enabled, the fallback bypasses the
    /// thread-local SplitWorkspace cache for inputs smaller than 1 MP
    /// at effort ≤ 7, while keeping the parallel root-split and
    /// borrowed-view fan-out on. Bitstream-equivalent — flipping
    /// this only changes the workspace allocation strategy.
    /// Default: OFF — the audit-claimed cb5e202 cache regression no
    /// longer reproduces on top of chunk-3c (paired bench data in
    /// `benchmarks/small_image_fallback_paired_2026-05-17.tsv`).
    /// Kept as opt-in for future investigation. See audit item #10:
    /// `rejected_optimizations_conditional_value_2026-05-17.md`.
    #[arg(long)]
    small_image_fallback: bool,

    /// Disable gaborish inverse pre-filter (on by default).
    /// Without gaborish, the decoder skips its 3x3 blur post-filter.
    #[arg(long)]
    no_gaborish: bool,

    /// EX-J13 — per-tile contrast-adaptive gaborish kernel strength on
    /// the Y (luma) channel (off by default). Encoder-only; decoder
    /// always applies the fixed 3x3 inverse blur. No-op when
    /// `--no-gaborish` is set or when the distance gate disables
    /// gaborish.
    #[arg(long)]
    adaptive_gaborish: bool,

    /// Edge-preserving filter strength override (mirrors libjxl `cjxl --epf`).
    /// `-1` (default) = encoder chooses by distance; `0` = off;
    /// `1`/`2`/`3` = forced iteration count (heavier smoothing).
    /// Values outside `-1..=3` are clamped to that range.
    #[arg(long, value_name = "LEVEL", allow_hyphen_values = true, default_value_t = -1)]
    epf: i8,

    /// Adaptive-dispatch policy for the per-block EPF sharpness
    /// search (W36-2). `auto` (default) skips the per-block search
    /// on smooth regions (uniform default sharpness emitted
    /// instead). `always-select` runs the full search whenever EPF
    /// and dynamic sharpness are active — byte-identical to
    /// historical builds. `always-default` skips the
    /// search unconditionally. The search is `compute_epf_sharpness`
    /// in `vardct/epf.rs`; per the W36-1 baseline it is 45.5% of e6
    /// wall-clock and 33.8% of e7.
    #[arg(long, value_name = "POLICY", default_value = "auto")]
    epf_dispatch: String,

    /// Adaptive-dispatch policy for the pixel-domain loss term in the
    /// AC-strategy search cost (W38-2). `always-on` (default) keeps
    /// the loss term whenever `pixel_domain_loss` is enabled —
    /// byte-identical to historical builds. `auto` drops the loss
    /// term on smooth content (per-image `median(mask1x1) > 80`)
    /// where it rarely changes which strategy wins. `always-off`
    /// disables the loss term unconditionally (equivalent to
    /// `--no-pixel-domain-loss`). Per the W38-1 baseline the loss
    /// path adds ~11 ms/MP on photos and ~70 ms/MP on screenshots at
    /// effort 5.
    #[arg(long, value_name = "POLICY", default_value = "always-on")]
    pixel_loss_dispatch: String,

    /// Force DCT8 only (disable AC strategy selection)
    #[arg(long)]
    dct8_only: bool,

    /// Force a specific AC strategy (0=DCT8, 1=DCT16x8, 2=DCT8x16, 3=DCT16x16,
    /// 4=DCT32x32, 5=DCT4x8, 6=DCT8x4, 7=DCT4x4)
    #[arg(long)]
    force_strategy: Option<u8>,

    /// Maximum AC strategy transform size (8, 16, 32, or 64).
    /// 8 = only 8x8-class transforms, 16 = up to 16x16, 32 = up to 32x32,
    /// 64 = no restriction (default).
    #[arg(long, value_name = "SIZE")]
    max_strategy_size: Option<u8>,

    /// Enable error diffusion in AC quantization (propagates 1/4 quantization error
    /// to the next coefficient in zigzag order). Off by default — libjxl's QuantizeBlockAC
    /// accepts an error_diffusion parameter but never references it in the function body,
    /// so the feature is effectively a no-op in the reference encoder. Our implementation
    /// does implement it, but it hurts quality on images with bright features in dark
    /// regions, especially when combined with gaborish (up to +33% butteraugli regression).
    #[arg(long)]
    error_diffusion: bool,

    /// Disable pixel-domain loss in AC strategy selection.
    /// Pixel-domain loss (full libjxl cost model) is on by default.
    #[arg(long)]
    no_pixel_domain_loss: bool,

    /// Disable patches (dictionary-based repeated pattern detection).
    /// Patches are on by default. Huge wins on screenshots, zero cost on photos.
    #[arg(long)]
    no_patches: bool,

    /// Force-enable libjxl-style dot detection (refs #19). On by default,
    /// mirroring libjxl's `cjxl --dots` "encoder chooses" semantics. The
    /// detector is internally gated to effort >= 7, distance >= 3.0, and
    /// no text-like patches in the same image, so it's a no-op outside
    /// the niche star-field / specular-highlight content range. Passing
    /// this flag is equivalent to the default; provided for symmetry with
    /// `--no-dot-detection`.
    #[arg(long)]
    dot_detection: bool,

    /// Force-disable dot detection. Mirrors libjxl `cjxl --dots=0`.
    /// Use when you want bit-exact reproducibility on content that
    /// could otherwise trip the detector (astronomy, specular highlights),
    /// or when running picker / cost-model sweeps without perceptual
    /// confounds.
    #[arg(long, conflicts_with = "dot_detection")]
    no_dot_detection: bool,

    /// Enable LZ77 backward references (on by default at effort 9+).
    #[arg(long)]
    lz77: bool,

    /// Disable LZ77 backward references.
    #[arg(long, conflicts_with = "lz77")]
    no_lz77: bool,

    /// LZ77 method to use when LZ77 is active.
    /// - rle: Only matches consecutive identical tokens (fast, limited on photos)
    /// - greedy: Hash chain backward references (slower but better compression)
    /// - optimal: Viterbi DP minimum-cost parse (slowest, best compression)
    ///
    /// Default: auto-selected by effort level (e0-7=rle, e8=greedy, e9+=optimal)
    #[arg(long, value_name = "METHOD")]
    lz77_method: Option<String>,

    /// Enable content-adaptive MA tree learning (on by default at effort 8+).
    /// ANS-only (implies ANS). For lossless encoding.
    #[arg(long)]
    tree_learning: bool,

    /// Disable content-adaptive MA tree learning.
    #[arg(long, conflicts_with = "tree_learning")]
    no_tree_learning: bool,

    /// Enable squeeze (Haar wavelet) transform (on by default at effort 7+).
    /// For lossless encoding.
    #[arg(long)]
    squeeze: bool,

    /// Disable squeeze (Haar wavelet) transform.
    #[arg(long, conflicts_with = "squeeze")]
    no_squeeze: bool,

    /// Enable lossy delta palette for near-lossless modular encoding.
    /// Quantizes colors to a small palette + delta entries with error diffusion.
    /// NOT pixel-exact — trades color accuracy for smaller files.
    #[arg(long)]
    lossy_palette: bool,

    /// Enable 3-pass progressive encoding (DC/VLF → LF → Full AC).
    /// Enables staged previews at reduced quality before full decode.
    #[arg(long)]
    progressive: bool,

    /// Enable 2-pass quantized progressive encoding.
    /// All AC at reduced precision first, then full precision refinement.
    #[arg(long, conflicts_with = "progressive")]
    qprogressive: bool,

    /// Enable separate DC frame (LfFrame).
    /// Encodes DC coefficients in a separate modular frame before the VarDCT frame.
    /// Matches libjxl's progressive_dc >= 1.
    #[arg(long)]
    lf_frame: bool,

    /// Use experimental encoder mode (encoder-specific improvements).
    /// Default is reference mode (matches libjxl algorithm choices).
    #[arg(long)]
    experimental: bool,

    /// Enable iterative rate control for improved distance targeting.
    /// Encodes multiple times, adjusting quantization to match target distance.
    /// Requires the rate-control feature. Off by default.
    #[arg(short = 'r', long)]
    rate_control: bool,

    /// Maximum iterations for rate control (default: 3).
    /// Only used when --rate-control is enabled.
    #[arg(long, value_name = "N", default_value = "3")]
    rc_iterations: usize,

    /// Number of butteraugli quantization loop iterations.
    /// Default depends on effort: e1-7=0, e8=2, e9+=4 (matching libjxl).
    /// Requires the butteraugli-loop feature. Use --no-butteraugli to disable.
    #[arg(long, value_name = "N")]
    butteraugli_iters: Option<u32>,

    /// Disable butteraugli quantization loop (equivalent to --butteraugli-iters 0).
    #[arg(long)]
    no_butteraugli: bool,

    /// Number of SSIM2 quantization loop iterations.
    /// Alternative to butteraugli loop: uses per-block RMSE + full-image SSIM2.
    /// Requires the ssim2-loop feature.
    #[arg(long, value_name = "N")]
    ssim2_iters: Option<u32>,

    /// Number of zensim quantization loop iterations.
    /// Alternative to butteraugli loop: uses zensim psychovisual metric with
    /// per-pixel diffmap in XYB space. Requires the zensim-loop feature.
    #[arg(long, value_name = "N")]
    zensim_iters: Option<u32>,

    /// EXIF metadata file to embed in the output JXL container
    #[arg(long, value_name = "FILE")]
    exif: Option<PathBuf>,

    /// XMP metadata file to embed in the output JXL container
    #[arg(long, value_name = "FILE")]
    xmp: Option<PathBuf>,

    /// JUMBF (ISO 19566-5, C2PA / Content Authenticity Initiative)
    /// payload file to embed in the output JXL container as a `jumb`
    /// box. The file must contain a valid JUMBF superbox (typically
    /// produced by the `c2pa` tooling); we pass the bytes through
    /// verbatim without validation.
    #[arg(long, value_name = "FILE")]
    jumbf: Option<PathBuf>,

    /// ICC profile file to embed in the JXL codestream
    #[arg(long, value_name = "FILE")]
    icc: Option<PathBuf>,

    /// Override frame rate for APNG animation (frames per second).
    /// Default: derive from APNG per-frame delays.
    #[arg(long, value_name = "FPS")]
    fps: Option<u32>,

    /// Number of animation loops (0 = infinite).
    /// Default: use APNG loop count.
    #[arg(long, value_name = "N")]
    loops: Option<u32>,

    /// Number of threads for parallel encoding (0 = auto, 1 = sequential).
    /// Requires the parallel feature.
    #[arg(long, value_name = "N", default_value = "0")]
    threads: usize,

    /// Downsample extra channels (alpha, depth, …) by this factor
    /// before encoding. Accepts `{1, 2, 4, 8}` (mirrors libjxl
    /// `cjxl --ec_resampling`). `1` is the default (no
    /// downsampling). Box-filter; output dimensions follow
    /// `width.div_ceil(N) × height.div_ceil(N)`. Currently applies
    /// only to RGBA input on the lossless path — alpha is sliced
    /// out, downsampled with [`jxl_encoder::downsample_channel_u8`],
    /// and attached as an extra channel with
    /// `dim_shift = log2(N)`. Lossy + extras > alpha follow when
    /// the lossy `dim_shift > 0` guard lifts.
    ///
    /// Both `--ec_resampling` (libjxl style) and `--ec-resampling`
    /// (clap default) are accepted.
    #[arg(
        long = "ec_resampling",
        alias = "ec-resampling",
        value_name = "N",
        default_value = "1"
    )]
    ec_resampling: u32,

    /// Be quiet (minimal output)
    #[arg(long)]
    quiet: bool,

    /// Force JPEG → JXL lossless transcoding for the input file (mirrors
    /// cjxl `--lossless_jpeg=1`). Requires the `jpeg-reencoding` cargo
    /// feature. When the feature is enabled, `.jpg` / `.jpeg` inputs are
    /// also auto-detected by extension and routed through the transcode
    /// path unless `--no-lossless-jpeg` is passed. The resulting JXL
    /// container holds a JBRD reconstruction box, so
    /// `djxl out.jxl out.jpg --reconstruct_jpeg` reproduces the exact
    /// original JPEG bytes.
    #[arg(long)]
    lossless_jpeg: bool,

    /// Disable the automatic JPEG → JXL transcode path even when the
    /// input has a `.jpg` / `.jpeg` extension. Useful for re-encoding
    /// JPEG content as a fresh lossy VarDCT JXL (via pixel decode), or
    /// for testing the non-transcode code path on JPEG inputs.
    #[arg(long)]
    no_lossless_jpeg: bool,

    /// PreserveJxl: lossy JPEG → JXL by coefficient-domain coarsening.
    /// Takes a `scale` > 1.0 (coarser = smaller; 1.0 = lossless transcode).
    /// Re-quantizes the JPEG's own DCT coefficients to a coarser, same-family
    /// scale of its quant tables (bundled deadzone + mild chroma lead — the
    /// proven RD-frontier policy), then transcodes — no pixel round-trip, no
    /// JBRD (lossy). Guaranteed not larger than the lossless transcode. Best
    /// for gentle / near-lossless reduction, where it beats a full pixel
    /// re-encode (see docs/JPEG_LOSSY_RECOMPRESSION.md). Requires the
    /// `jpeg-reencoding` feature.
    #[arg(long, value_name = "SCALE")]
    jpeg_coarsen: Option<f32>,

    // ── A1 CLI passthrough — libjxl `cjxl` parity flags ──────────────
    //
    // These flags forward through to `LossyConfig` / `LosslessConfig`
    // builders added in the matching commit. Several are stored on the
    // config and acted on opportunistically; full encoder-side wiring
    // is queued as follow-on work per CLI passthrough audit.
    /// Peak display luminance in nits (cd/m²) for HDR content. Mirrors
    /// libjxl `cjxl --intensity_target`. Written to the JXL codestream
    /// `ToneMapping.intensity_target` field. Default keeps the file
    /// header's existing value (255.0 = SDR for typical sRGB inputs).
    /// Typical: 4000 / 10000 for PQ HDR.
    #[arg(long, value_name = "NITS")]
    intensity_target: Option<f32>,

    /// Brotli effort (0-11) for `brob` (Brotli-compressed) container
    /// metadata boxes. Mirrors libjxl `cjxl --brotli_effort`. Higher =
    /// smaller container, slower encode. Default `None` = plain
    /// `Exif`/`xml ` boxes. Requires the `brotli-metadata` cargo
    /// feature; ignored otherwise. libjxl default quality is 4.
    #[arg(long, value_name = "Q")]
    brotli_effort: Option<u32>,

    /// Separate butteraugli distance for the alpha extra channel.
    /// Mirrors libjxl `cjxl -a` / `--alpha_distance`. `0` = lossless
    /// alpha (libjxl default behaviour). Stored on the config; alpha
    /// is still encoded losslessly until the lossy-alpha pipeline
    /// lands.
    #[arg(short = 'a', long, value_name = "D")]
    alpha_distance: Option<f32>,

    /// Modular group encoding order. Mirrors libjxl `cjxl
    /// --group_order`. `0` = scanline order (default), `1` =
    /// center-first (equivalent to the existing center-first AC group
    /// reorder), `2` = reserved.
    #[arg(long, value_name = "N")]
    group_order: Option<u8>,

    /// Centre X coordinate for the center-first AC group reorder.
    /// Mirrors libjxl `cjxl --center_x`. `-1` = image centre (default).
    /// Requires `--group-order 1`; ignored otherwise. Stored on the
    /// config — non-default centre honouring is queued follow-on work.
    #[arg(long, value_name = "X", allow_hyphen_values = true)]
    center_x: Option<i64>,

    /// Centre Y coordinate for the center-first AC group reorder.
    /// Mirrors libjxl `cjxl --center_y`. `-1` = image centre (default).
    /// Requires `--group-order 1`; ignored otherwise.
    #[arg(long, value_name = "Y", allow_hyphen_values = true)]
    center_y: Option<i64>,

    /// Decoder upsampling mode. Mirrors libjxl `cjxl --upsampling_mode`.
    /// `-1` = non-separable (default), `0` = nearest neighbour
    /// (pixel-art), `1` = reserved. Stored on the config; emission of
    /// a custom upsampling LUT is queued follow-on work.
    #[arg(long, value_name = "N", allow_hyphen_values = true)]
    upsampling_mode: Option<i32>,

    /// Force a fixed modular predictor (lossless path). Mirrors libjxl
    /// `cjxl -P` / `--modular_predictor`. `0..=13` = the corresponding
    /// `jxl_encoder::modular::Predictor` variant (Zero..Average4); `15`
    /// = libjxl `Variable` meta-mode (falls through to ID3 tree
    /// learning).
    ///
    /// `14` is reserved by libjxl for the `Best` meta-mode. In this
    /// encoder we repurpose it as **RIGED** — Sharma et al. 2018
    /// Resolution-Independent Gradient-aware Edge Detection. Triggers a
    /// hand-crafted 3-leaf MA tree that switches between
    /// `Top`/`Left`/`Average((W+N)/2)` per pixel based on the dominant
    /// local gradient direction. Encoder-only meta-mode; the wire
    /// bitstream is decoder-legal (uses only wire predictors 1/2/3 and
    /// wire properties 10/13). Mostly a win on textured content.
    #[arg(short = 'P', long, value_name = "N")]
    modular_predictor: Option<u8>,

    /// Override the palette-transform colour cap (lossless path).
    /// Mirrors libjxl `cjxl --modular_palette_colors`. `0` disables
    /// palette detection. Default keeps the built-in `MAX_PALETTE_COLORS`
    /// constant (1024).
    #[arg(long, value_name = "N", allow_hyphen_values = true)]
    modular_palette_colors: Option<i64>,

    /// Override the global channel-colours percentage cap (lossless
    /// path). Mirrors libjxl `cjxl
    /// --modular_channel_colors_global_percent`. `0..=100`. Default
    /// keeps the built-in `CHANNEL_COLORS_PERCENT` constant (95.0).
    #[arg(long, value_name = "P")]
    modular_channel_colors_global_percent: Option<f32>,

    /// Override the per-group channel-colours percentage cap (lossless
    /// path). Mirrors libjxl `cjxl
    /// --modular_channel_colors_group_percent`. `0..=100`. Default
    /// keeps the libjxl default (80.0).
    #[arg(long, value_name = "P")]
    modular_channel_colors_group_percent: Option<f32>,

    /// Override the previous-channel context-properties limit for tree
    /// learning (lossless path). Mirrors libjxl `cjxl -E` /
    /// `--modular_nb_prev_channels`. `-1` = libjxl default sentinel.
    /// Stored on the config; our tree learner does not yet consume
    /// previous-channel properties.
    #[arg(short = 'E', long, value_name = "N", allow_hyphen_values = true)]
    modular_nb_prev_channels: Option<i32>,

    /// Bias the encoder toward simpler bitstreams that decode faster, at
    /// the cost of compression. Mirrors libjxl `cjxl --faster_decoding
    /// 0..4`. `0` (default) keeps the existing behaviour. Higher tiers
    /// progressively drop encoder features (Weighted predictor → MA
    /// tree learner → EPF → DCT32+ + gaborish). See
    /// `LossyConfig::with_faster_decoding` / `LosslessConfig::with_faster_decoding`
    /// for per-tier specifics. Applied on both lossy and lossless
    /// paths.
    #[arg(long, value_name = "TIER", default_value = "0")]
    faster_decoding: u8,

    /// Modular group-size override (lossless / modular path only).
    /// Mirrors libjxl `cjxl -g 0..3` / `cparams.modular_group_size_shift`.
    /// The value is the `group_size_shift` written into the frame
    /// header; group dimension = `128 << shift` pixels:
    /// `0`=128, `1`=256 (default), `2`=512, `3`=1024.
    /// `None` (omitted flag) keeps the existing 256-pixel default so
    /// bitstreams are unchanged. Ignored on VarDCT (lossy) encodes —
    /// libjxl and this encoder both fix VarDCT groups at 256 pixels.
    #[arg(short = 'g', long, value_name = "SHIFT", value_parser = clap::value_parser!(u8).range(0..=3))]
    modular_group_size: Option<u8>,

    /// Container-wrap policy. Mirrors libjxl `cjxl --container 0|1`.
    /// `-1` = auto (default: wrap only when metadata or codestream
    /// level requires it). `0` = never wrap (bare codestream — drops
    /// EXIF / XMP / JUMBF silently, errors on level-10 codestreams).
    /// `1` = always wrap. Applied on both lossy and lossless paths.
    #[arg(
        long,
        value_name = "MODE",
        default_value = "-1",
        allow_hyphen_values = true
    )]
    container: i8,

    /// Explicit progressive-DC level. Mirrors libjxl `cjxl
    /// --progressive_dc 0..2`. `0` = no progressive DC (default), `1` =
    /// one LfFrame (equivalent to `--lf-frame`), `2` = two nested
    /// LfFrames (libjxl path; currently emits a single LfFrame).
    /// Lossy only; implies `--lf-frame` when non-zero.
    #[arg(long, value_name = "N", default_value = "0")]
    progressive_dc: u8,

    /// Premultiplied (associated) alpha mode. Mirrors libjxl
    /// `cjxl --premultiply -1|0|1`. `0` (default) = straight alpha,
    /// `1` = premultiplied alpha, `-1` = auto-detect from input via a
    /// one-pass scan. Applied to lossy encodes on layouts with alpha.
    #[arg(
        long,
        value_name = "MODE",
        default_value = "0",
        allow_hyphen_values = true
    )]
    premultiply: i8,

    /// Input/output buffering policy (streaming refactor scaffolding,
    /// jxl-encoder#11). Mirrors libjxl `cjxl --buffering -1..3`.
    ///
    /// `-1` = auto (default): encoder picks per-image — ≤ 2048² folds
    /// to `0` (full-buffered); larger images fold to `2`
    /// (stream-input + buffered-output), matching libjxl post-`032d39a`.
    /// `0` = buffer everything (today's one-shot path).
    /// `1` = buffer ≤ 2048², stream-input + buffered-output otherwise.
    /// `2` = always stream-input + buffered-output.
    /// `3` = stream input AND stream output (requires seek-back on sink;
    /// bitstream is not progressively decodable).
    ///
    /// **Chunk 1 scaffolding** — accepted on the CLI and surfaced on
    /// the config types but no dispatch is wired; output bytes are
    /// identical regardless of value. Chunks 2-7 land the per-DC-group
    /// split and the active streaming paths. Both `--buffering` and
    /// `--buffering=N` forms are accepted; both lossy and lossless
    /// configs receive the value.
    #[arg(
        long,
        value_name = "MODE",
        default_value = "-1",
        allow_hyphen_values = true
    )]
    buffering: i8,

    /// Route the encode through the streaming
    /// [`LossyEncoder`](jxl_encoder::LossyEncoder) /
    /// [`LosslessEncoder`](jxl_encoder::LosslessEncoder) API by chunking
    /// the decoded pixel buffer into row-groups (`STREAM_CHUNK_ROWS`)
    /// fed via `push_rows()` + finalized with `finish()` /
    /// [`finish_to`](jxl_encoder::LossyEncoder::finish_to).
    ///
    /// Mirrors `cjxl --streaming_input` at the CLI surface. The input
    /// file itself is still PNG/PNM-decoded into memory first because
    /// our streaming PNM reader is not yet wired (that's the multi-day
    /// follow-on tracked alongside `--streaming-input` in the A1 audit);
    /// this flag exercises the encoder-side row-streaming pipeline so
    /// the API is regularly covered from end-to-end. Bitstreams emitted
    /// via `--streaming-input` are bit-identical to the bulk path on the
    /// supported subset (basic RGB / RGBA / Gray / Gray+Alpha lossless
    /// or lossy without rate-control, JPEG transcode, animation, or
    /// ec_resampling). When combined with an unsupported path the flag
    /// is logged and ignored.
    ///
    /// Both `--streaming_input` (libjxl style) and `--streaming-input`
    /// (clap default) are accepted.
    #[arg(long = "streaming_input", alias = "streaming-input")]
    streaming_input: bool,

    /// Write the encoded codestream directly into the output file via
    /// [`LossyEncoder::finish_to`](jxl_encoder::LossyEncoder::finish_to) /
    /// [`LosslessEncoder::finish_to`](jxl_encoder::LosslessEncoder::finish_to)
    /// instead of buffering the full encoded `Vec<u8>` in memory first.
    ///
    /// Mirrors `cjxl --streaming_output`. Only effective alongside
    /// `--streaming-input`; the bulk one-shot path still goes through a
    /// `Vec<u8>` because it has no incremental writer surface. The
    /// streaming-encoder `finish_to` sink is `std::io::Write`-backed, so
    /// we wrap the destination in a `BufWriter<File>` and let the
    /// encoder push bytes as they are emitted.
    ///
    /// Both `--streaming_output` (libjxl style) and `--streaming-output`
    /// (clap default) are accepted.
    #[arg(long = "streaming_output", alias = "streaming-output")]
    streaming_output: bool,
}

/// Number of pixel rows pushed per `push_rows` call when
/// `--streaming-input` is on. Picked to (a) keep per-chunk pixel byte
/// volume below ~16 MiB at typical pixel widths and (b) match the
/// 256-row JXL group height so deinterleave / channel writes line up
/// with downstream group boundaries.
const STREAM_CHUNK_ROWS: u32 = 64;

fn main() {
    let args = Args::parse();

    if !args.quiet {
        println!("JPEG XL Encoder (Rust)");
        println!("=====================");
    }

    // Determine distance
    let distance = if args.lossless || args.distance == Some(0.0) {
        0.0
    } else if let Some(d) = args.distance {
        d
    } else {
        quality_to_distance(args.quality)
    };

    if !args.quiet {
        println!("Input:    {}", args.input.display());
        println!("Output:   {}", args.output.display());
        println!(
            "Distance: {} {}",
            distance,
            if distance == 0.0 { "(lossless)" } else { "" }
        );
        println!("Effort:   {}", args.effort);
        println!();
    }

    let lz77_method = args
        .lz77_method
        .as_deref()
        .map(|m| match m.to_lowercase().as_str() {
            "rle" => Lz77Method::Rle,
            "greedy" => Lz77Method::Greedy,
            "optimal" => Lz77Method::Optimal,
            other => {
                eprintln!(
                    "Error: unknown --lz77-method '{other}' (expected rle | greedy | optimal)"
                );
                std::process::exit(1);
            }
        });

    let epf_dispatch = match args.epf_dispatch.to_lowercase().as_str() {
        "always-select" | "always_select" | "select" => EpfDispatch::AlwaysSelect,
        "always-default" | "always_default" | "default" | "skip" => EpfDispatch::AlwaysDefault,
        "auto" => EpfDispatch::Auto,
        other => {
            eprintln!(
                "Error: unknown --epf-dispatch '{other}' (expected always-select | \
                 always-default | auto)"
            );
            std::process::exit(1);
        }
    };

    let pixel_loss_dispatch = match args.pixel_loss_dispatch.to_lowercase().as_str() {
        "always-on" | "always_on" | "on" => PixelLossDispatch::AlwaysOn,
        "always-off" | "always_off" | "off" => PixelLossDispatch::AlwaysOff,
        "auto" => PixelLossDispatch::Auto,
        other => {
            eprintln!(
                "Error: unknown --pixel-loss-dispatch '{other}' (expected always-on | \
                 always-off | auto)"
            );
            std::process::exit(1);
        }
    };

    // W44-131 Chunk E: when the caller picks a named non-Zenjxl
    // strategy AND also explicitly overrides one of the perf
    // dispatches (epf / pixel-loss), warn that the dispatch override
    // is dropped — non-Custom strategies don't carry per-dispatch
    // overrides. Callers wanting both must drive the encoder from
    // Rust via `EncoderStrategy::Custom(...)`.
    let dispatches_overridden = epf_dispatch != EpfDispatch::default()
        || pixel_loss_dispatch != PixelLossDispatch::default();
    if args.strategy != StrategyArg::Zenjxl && dispatches_overridden && !args.quiet {
        eprintln!(
            "Warning: --strategy {:?} discards --epf-dispatch / --pixel-loss-dispatch overrides.\n\
             Named non-Zenjxl strategies carry their own perf-dispatch defaults; to keep both, \
             drive the encoder via `LossyConfig::with_strategy(EncoderStrategy::Custom(...))` \
             from Rust.",
            args.strategy
        );
    }
    let strategy_arg = args.strategy;

    // Determine input format from extension
    let ext = args
        .input
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    let is_pnm = matches!(ext.as_str(), "pnm" | "ppm" | "pgm" | "pfm" | "pam");
    let is_jpeg_ext = matches!(ext.as_str(), "jpg" | "jpeg" | "jpe" | "jfif");

    let start = Instant::now();

    // ── JPEG → JXL lossless transcoding ────────────────────────────────
    //
    // Routes through `LosslessConfig::encode_jpeg_transcode` whenever
    // either:
    //   (a) the user explicitly passed `--lossless-jpeg`, OR
    //   (b) the input has a `.jpg` / `.jpeg` / `.jpe` / `.jfif`
    //       extension AND `--no-lossless-jpeg` was NOT passed.
    //
    // Output is a JXL container (signature box + codestream + JBRD
    // reconstruction box). The JBRD box lets `djxl --reconstruct_jpeg`
    // reproduce the original JPEG byte-for-byte.
    //
    // Only available when the `jpeg-reencoding` cargo feature is on.
    // When the feature is off, a `.jpg` input falls through to the
    // (failing) PNG path so users get a clear "not a PNG" error rather
    // than silently producing garbage.
    #[cfg(feature = "jpeg-reencoding")]
    {
        // PreserveJxl lossy path: explicit `--jpeg-coarsen <scale>` on a JPEG
        // input. Routes through the coefficient-domain coarsener (no JBRD).
        if let Some(scale) = args.jpeg_coarsen {
            let jpeg_bytes = match std::fs::read(&args.input) {
                Ok(b) => b,
                Err(e) => {
                    eprintln!("Error reading JPEG input: {}", e);
                    std::process::exit(1);
                }
            };
            if !jxl_encoder::jpeg::is_jpeg_signature(&jpeg_bytes) {
                eprintln!(
                    "Error: --jpeg-coarsen requires a JPEG input; {} has no SOI marker.",
                    args.input.display()
                );
                std::process::exit(1);
            }
            if !args.quiet {
                println!(
                    "PreserveJxl lossy JPEG → JXL (coarsen scale {:.3}, input {} bytes)",
                    scale,
                    jpeg_bytes.len()
                );
            }
            let encoded = match jxl_encoder::jpeg::encode_jpeg_recompress_auto_codestream(
                &jpeg_bytes,
                scale,
                args.effort,
                None,
                None,
            ) {
                Ok(b) => b,
                Err(e) => {
                    eprintln!("PreserveJxl recompress failed: {}", e);
                    std::process::exit(1);
                }
            };
            let encode_time = start.elapsed();
            if let Err(e) = write_output(&args.output, &encoded) {
                eprintln!("Error writing output: {}", e);
                std::process::exit(1);
            }
            if !args.quiet {
                let input_size = jpeg_bytes.len() as u64;
                let output_size = encoded.len() as u64;
                println!();
                println!("Input size:  {} bytes (JPEG)", input_size);
                println!(
                    "Output size: {} bytes (JXL codestream, lossy, no JBRD)",
                    output_size
                );
                println!(
                    "Ratio:       {:.2}x ({:.1}% of original JPEG)",
                    output_size as f64 / input_size as f64,
                    output_size as f64 / input_size as f64 * 100.0
                );
                println!("Time:        {:.2?}", encode_time);
            } else {
                println!("{}", args.output.display());
            }
            return;
        }

        let want_jpeg_transcode = args.lossless_jpeg || (is_jpeg_ext && !args.no_lossless_jpeg);
        if want_jpeg_transcode {
            let jpeg_bytes = match std::fs::read(&args.input) {
                Ok(b) => b,
                Err(e) => {
                    eprintln!("Error reading JPEG input: {}", e);
                    std::process::exit(1);
                }
            };
            // Sniff signature so we don't try to transcode a mis-extensioned PNG.
            if !jxl_encoder::jpeg::is_jpeg_signature(&jpeg_bytes) {
                eprintln!(
                    "Error: {} does not look like a JPEG file (no SOI marker). \
                     Skip JPEG transcoding with --no-lossless-jpeg.",
                    args.input.display()
                );
                std::process::exit(1);
            }
            if !args.quiet {
                println!(
                    "JPEG → JXL lossless transcoding (input {} bytes)",
                    jpeg_bytes.len()
                );
            }
            let cfg = LosslessConfig::new()
                .with_effort(args.effort)
                .with_threads(args.threads);
            let encoded = match cfg.encode_jpeg_transcode(&jpeg_bytes) {
                Ok(b) => b,
                Err(e) => {
                    eprintln!("JPEG transcode failed: {}", e);
                    std::process::exit(1);
                }
            };
            let encode_time = start.elapsed();
            if let Err(e) = write_output(&args.output, &encoded) {
                eprintln!("Error writing output: {}", e);
                std::process::exit(1);
            }
            let input_size = jpeg_bytes.len() as u64;
            let output_size = encoded.len() as u64;
            if !args.quiet {
                println!();
                println!("Input size:  {} bytes (JPEG)", input_size);
                println!(
                    "Output size: {} bytes (JXL container, JBRD box included)",
                    output_size
                );
                println!(
                    "Ratio:       {:.2}x ({:.1}% of original JPEG)",
                    output_size as f64 / input_size as f64,
                    output_size as f64 / input_size as f64 * 100.0
                );
                println!("Time:        {:.2?}", encode_time);
                println!();
                println!(
                    "Reconstruct original JPEG: djxl {} <out.jpg> --reconstruct_jpeg",
                    args.output.display()
                );
            } else {
                println!("{}", args.output.display());
            }
            return;
        }
    }
    #[cfg(not(feature = "jpeg-reencoding"))]
    {
        // Feature off — silence the unused-args warning for the JPEG knobs.
        let _ = (args.lossless_jpeg, args.no_lossless_jpeg, is_jpeg_ext);
        if args.lossless_jpeg {
            eprintln!("Error: --lossless-jpeg requires the `jpeg-reencoding` cargo feature.");
            std::process::exit(1);
        }
        if args.jpeg_coarsen.is_some() {
            eprintln!("Error: --jpeg-coarsen requires the `jpeg-reencoding` cargo feature.");
            std::process::exit(1);
        }
    }

    // Check for APNG (animated PNG) — handle before single-frame path
    if !is_pnm {
        match read_apng(&args.input) {
            Ok(Some(apng)) => {
                if !args.quiet {
                    println!(
                        "APNG:     {}x{} {:?}, {} frames, {} loops",
                        apng.width,
                        apng.height,
                        apng.color_type,
                        apng.frames.len(),
                        apng.num_loops
                    );
                }

                let layout = if apng.has_alpha {
                    PixelLayout::Rgba8
                } else {
                    PixelLayout::Rgb8
                };

                // Build animation params
                let (tps_numerator, tps_denominator) = if let Some(fps) = args.fps {
                    (fps, 1)
                } else {
                    (1000, 1) // millisecond precision
                };

                let num_loops = args.loops.unwrap_or(apng.num_loops);

                let animation = AnimationParams {
                    tps_numerator,
                    tps_denominator,
                    num_loops,
                    // CLI never receives premultiplied input — APNG is
                    // straight alpha; matches still-image CLI default.
                    premultiplied_alpha: false,
                };

                // Build frames with durations
                let anim_frames: Vec<AnimationFrame<'_>> = apng
                    .frames
                    .iter()
                    .map(|f| {
                        AnimationFrame::new(
                            &f.pixels,
                            if args.fps.is_some() {
                                1 // 1 tick per frame when fps is explicit
                            } else {
                                f.delay_ms // millisecond ticks
                            },
                        )
                    })
                    .collect();

                let lossy_supported = matches!(
                    layout,
                    PixelLayout::Rgb8
                        | PixelLayout::Rgba8
                        | PixelLayout::Gray8
                        | PixelLayout::GrayAlpha8
                );

                let encoded = if distance > 0.0 && lossy_supported {
                    let mut cfg = LossyConfig::new(distance)
                        .with_effort(args.effort)
                        .with_threads(args.threads);
                    if let Some(method) = lz77_method {
                        cfg = cfg.with_lz77_method(method);
                    }
                    if args.no_ans {
                        cfg = cfg.with_ans(false);
                    }
                    if args.no_gaborish {
                        cfg = cfg.with_gaborish(false);
                    }
                    if args.adaptive_gaborish {
                        cfg = cfg.with_adaptive_gaborish(true);
                    }
                    if args.epf != -1 {
                        cfg = cfg.with_epf_level(args.epf);
                    }
                    // W44-130 Chunk D: dispatch policies absorbed into
                    // `EncoderImprovementsCustom`; setters deleted.
                    // W44-131 Chunk E: `--strategy` flag drives the
                    // top-level preset; `Zenjxl` (default) keeps the
                    // Chunk-D wrap so dispatch overrides survive.
                    cfg = apply_strategy_to_lossy(
                        cfg,
                        strategy_arg,
                        epf_dispatch,
                        pixel_loss_dispatch,
                    );
                    if args.noise || args.denoise {
                        cfg = cfg.with_noise(true);
                    }
                    if args.denoise {
                        cfg = cfg.with_denoise(true);
                    }
                    // libjxl-parity Option<f32> knobs — None is a no-op
                    cfg = cfg.with_photon_noise_iso(args.photon_noise_iso);
                    cfg = cfg.with_original_distance(args.original_distance);
                    cfg = cfg.with_quant_ac_rescale(args.quant_ac_rescale);
                    if args.no_perceptual_optimizations {
                        cfg = cfg.with_perceptual_optimizations(false);
                    }
                    if args.error_diffusion {
                        cfg = cfg.with_error_diffusion(true);
                    }
                    if args.no_pixel_domain_loss {
                        cfg = cfg.with_pixel_domain_loss(false);
                    }
                    if args.no_patches {
                        cfg = cfg.with_patches(false);
                    }
                    if args.no_dot_detection {
                        cfg = cfg.with_dot_detection(false);
                    } else if args.dot_detection {
                        cfg = cfg.with_dot_detection(true);
                    }
                    if args.lz77 {
                        cfg = cfg.with_lz77(true);
                    }
                    if args.no_lz77 {
                        cfg = cfg.with_lz77(false);
                    }

                    if args.progressive {
                        cfg = cfg.with_progressive(ProgressiveMode::DcVlfLfAc);
                    }
                    if args.qprogressive {
                        cfg = cfg.with_progressive(ProgressiveMode::QuantizedAcFullAc);
                    }
                    if args.lf_frame {
                        cfg = cfg.with_lf_frame(true);
                    }
                    if args.experimental {
                        cfg = cfg.with_mode(jxl_encoder::EncoderMode::Experimental);
                    }

                    if args.dct8_only {
                        cfg = cfg.with_force_strategy(Some(0));
                    }
                    if let Some(s) = args.force_strategy {
                        cfg = cfg.with_force_strategy(Some(s));
                    }
                    if let Some(s) = args.max_strategy_size {
                        cfg = cfg.with_max_strategy_size(Some(s));
                    }

                    // ── A1 passthrough — libjxl cjxl parity knobs ─────
                    cfg = cfg.with_alpha_distance(args.alpha_distance);
                    cfg = cfg.with_group_order(args.group_order);
                    cfg = cfg.with_center_x(args.center_x);
                    cfg = cfg.with_center_y(args.center_y);
                    cfg = cfg.with_upsampling_mode(args.upsampling_mode);
                    cfg = cfg.with_faster_decoding(args.faster_decoding);
                    cfg = cfg.with_container_mode(container_mode_from_cli(args.container));
                    cfg = cfg.with_progressive_dc(args.progressive_dc);
                    cfg = cfg.with_buffering(Buffering::from_i8(args.buffering));

                    #[cfg(feature = "butteraugli-loop")]
                    {
                        if args.no_butteraugli {
                            cfg = cfg.with_butteraugli_iters(0);
                        } else if let Some(n) = args.butteraugli_iters {
                            cfg = cfg.with_butteraugli_iters(n);
                        }
                        if !args.quiet && cfg.butteraugli_iters() > 0 {
                            println!("Butteraugli loop: {} iterations", cfg.butteraugli_iters());
                        }
                    }
                    #[cfg(feature = "ssim2-loop")]
                    if let Some(n) = args.ssim2_iters {
                        cfg = cfg.with_ssim2_iters(n);
                        if !args.quiet && n > 0 {
                            println!("SSIM2 loop: {} iterations", n);
                        }
                    }
                    #[cfg(feature = "zensim-loop")]
                    if let Some(n) = args.zensim_iters {
                        cfg = cfg.with_zensim_iters(n);
                        if !args.quiet && n > 0 {
                            println!("Zensim loop: {} iterations", n);
                        }
                    }

                    cfg.encode_animation(apng.width, apng.height, layout, &animation, &anim_frames)
                } else {
                    {
                        let mut lcfg = LosslessConfig::new()
                            .with_effort(args.effort)
                            .with_threads(args.threads);
                        if args.no_ans {
                            lcfg = lcfg.with_ans(false);
                        }
                        if args.tree_learning {
                            lcfg = lcfg.with_tree_learning(true).with_ans(true);
                        }
                        if args.no_tree_learning {
                            lcfg = lcfg.with_tree_learning(false);
                        }
                        if args.squeeze {
                            lcfg = lcfg.with_squeeze(true);
                        }
                        if args.no_squeeze {
                            lcfg = lcfg.with_squeeze(false);
                        }
                        if args.no_patches {
                            lcfg = lcfg.with_patches(false);
                        }
                        if args.lz77 {
                            lcfg = lcfg.with_lz77(true);
                        }
                        if args.no_lz77 {
                            lcfg = lcfg.with_lz77(false);
                        }
                        if args.lossy_palette {
                            lcfg = lcfg.with_lossy_palette(true);
                        }
                        if let Some(rct) = args.force_rct {
                            lcfg = lcfg.with_force_rct(Some(jxl_encoder::RctType(rct)));
                        }
                        if let Some(f) = args.tree_learning_sample_fraction {
                            lcfg = lcfg.with_tree_learning_sample_fraction(f);
                        }
                        if args.smart_fanout {
                            lcfg = lcfg.with_smart_fanout(true);
                        }
                        if args.small_image_fallback {
                            lcfg = lcfg.with_small_image_fallback_override(Some(true));
                        }
                        if args.experimental {
                            lcfg = lcfg.with_mode(jxl_encoder::EncoderMode::Experimental);
                        }
                        // ── A1 passthrough — libjxl cjxl modular knobs ─
                        lcfg = lcfg.with_modular_predictor(args.modular_predictor);
                        lcfg = lcfg.with_modular_palette_colors(args.modular_palette_colors);
                        lcfg = lcfg.with_modular_channel_colors_global_percent(
                            args.modular_channel_colors_global_percent,
                        );
                        lcfg = lcfg.with_modular_channel_colors_group_percent(
                            args.modular_channel_colors_group_percent,
                        );
                        lcfg = lcfg.with_modular_nb_prev_channels(args.modular_nb_prev_channels);
                        lcfg = lcfg.with_faster_decoding(args.faster_decoding);
                        lcfg = lcfg.with_modular_group_size(args.modular_group_size);
                        lcfg = lcfg.with_container_mode(container_mode_from_cli(args.container));
                        lcfg = lcfg.with_buffering(Buffering::from_i8(args.buffering));
                        lcfg
                    }
                    .encode_animation(
                        apng.width,
                        apng.height,
                        layout,
                        &animation,
                        &anim_frames,
                    )
                };

                let encoded = match encoded {
                    Ok(data) => data,
                    Err(e) => {
                        eprintln!("Error encoding animation: {}", e);
                        std::process::exit(1);
                    }
                };

                let encode_time = start.elapsed();

                match write_output(&args.output, &encoded) {
                    Ok(()) => {}
                    Err(e) => {
                        eprintln!("Error writing output: {}", e);
                        std::process::exit(1);
                    }
                }

                let input_size = std::fs::metadata(&args.input).map(|m| m.len()).unwrap_or(0);
                let output_size = encoded.len() as u64;

                if !args.quiet {
                    println!();
                    println!("Input size:  {} bytes", input_size);
                    println!("Output size: {} bytes", output_size);
                    println!(
                        "Ratio:       {:.2}x",
                        if input_size > 0 {
                            output_size as f64 / input_size as f64
                        } else {
                            0.0
                        }
                    );
                    println!("Time:        {:.2?}", encode_time);
                } else {
                    println!("{}", args.output.display());
                }

                return;
            }
            Ok(None) => {} // Not animated, fall through to single-frame path
            Err(e) => {
                eprintln!("Error reading input: {}", e);
                std::process::exit(1);
            }
        }
    } // end if !is_pnm

    // Read image (PNG or PNM single frame)
    let _t_load = Instant::now();
    let (width, height, color_type, bit_depth, data, source_gamma, cicp) = if is_pnm {
        let (w, h, ct, bd, d) = match read_pnm(&args.input) {
            Ok(result) => result,
            Err(e) => {
                eprintln!("Error reading PNM input: {}", e);
                std::process::exit(1);
            }
        };
        (w, h, ct, bd, d, None, None) // PNM has no gamma/cICP metadata
    } else {
        match read_png(&args.input) {
            Ok(result) => result,
            Err(e) => {
                eprintln!("Error reading input: {}", e);
                std::process::exit(1);
            }
        }
    };

    let is_16bit = bit_depth == png::BitDepth::Sixteen;

    if !args.quiet {
        println!(
            "Image:    {}x{} {:?} {}bpc",
            width,
            height,
            color_type,
            if is_16bit { 16 } else { 8 }
        );
        if let Some(gamma) = source_gamma {
            println!("Gamma:    {:.6} (display gamma {:.1})", gamma, 1.0 / gamma);
        }
        if let Some(c) = cicp {
            println!(
                "cICP:     cp={} tc={} mc={} full_range={}",
                c.color_primaries,
                c.transfer_function,
                c.matrix_coefficients,
                c.is_video_full_range_image
            );
        }
    }

    // Determine pixel layout
    let layout = match (color_type, is_16bit) {
        (png::ColorType::Rgb, false) => PixelLayout::Rgb8,
        (png::ColorType::Rgba, false) => PixelLayout::Rgba8,
        (png::ColorType::Grayscale, false) => PixelLayout::Gray8,
        (png::ColorType::Rgb, true) => PixelLayout::Rgb16,
        (png::ColorType::Rgba, true) => PixelLayout::Rgba16,
        (png::ColorType::Grayscale, true) => PixelLayout::Gray16,
        _ => {
            eprintln!(
                "Error: Unsupported color type: {:?} {:?}",
                color_type, bit_depth
            );
            std::process::exit(1);
        }
    };

    // Read optional EXIF/XMP metadata files
    let exif_data = args.exif.as_ref().map(|p| {
        std::fs::read(p).unwrap_or_else(|e| {
            eprintln!("Error reading EXIF file {}: {}", p.display(), e);
            std::process::exit(1);
        })
    });
    let xmp_data = args.xmp.as_ref().map(|p| {
        std::fs::read(p).unwrap_or_else(|e| {
            eprintln!("Error reading XMP file {}: {}", p.display(), e);
            std::process::exit(1);
        })
    });
    let jumbf_data = args.jumbf.as_ref().map(|p| {
        std::fs::read(p).unwrap_or_else(|e| {
            eprintln!("Error reading JUMBF file {}: {}", p.display(), e);
            std::process::exit(1);
        })
    });
    let icc_data = args.icc.as_ref().map(|p| {
        std::fs::read(p).unwrap_or_else(|e| {
            eprintln!("Error reading ICC file {}: {}", p.display(), e);
            std::process::exit(1);
        })
    });

    let metadata = if exif_data.is_some()
        || xmp_data.is_some()
        || jumbf_data.is_some()
        || icc_data.is_some()
    {
        let mut meta = jxl_encoder::ImageMetadata::new();
        if let Some(ref exif) = exif_data {
            meta = meta.with_exif(exif);
        }
        if let Some(ref xmp) = xmp_data {
            meta = meta.with_xmp(xmp);
        }
        if let Some(ref jumbf) = jumbf_data {
            meta = meta.with_jumbf(jumbf);
        }
        if let Some(ref icc) = icc_data {
            meta = meta.with_icc_profile(icc);
        }
        Some(meta)
    } else {
        None
    };

    // Lossy VarDCT supported for RGB/RGBA/Gray layouts (8-bit, 16-bit, f32)
    let lossy_supported = matches!(
        layout,
        PixelLayout::Rgb8
            | PixelLayout::Rgba8
            | PixelLayout::Bgr8
            | PixelLayout::Bgra8
            | PixelLayout::Rgb16
            | PixelLayout::Rgba16
            | PixelLayout::RgbLinearF32
            | PixelLayout::RgbaLinearF32
            | PixelLayout::Gray8
            | PixelLayout::GrayAlpha8
            | PixelLayout::Gray16
            | PixelLayout::GrayAlpha16
            | PixelLayout::GrayLinearF32
            | PixelLayout::GrayAlphaLinearF32
    );

    // Encode using new API
    if std::env::var_os("CJXLRS_TIMING").is_some() {
        eprintln!("[cli-timing] load+parse: {:?}", _t_load.elapsed());
    }
    let _t_enc = Instant::now();
    let encoded = if distance > 0.0 && lossy_supported {
        // Lossy VarDCT path — effort sets defaults, flags override
        let mut cfg = LossyConfig::new(distance)
            .with_effort(args.effort)
            .with_threads(args.threads);
        if let Some(method) = lz77_method {
            cfg = cfg.with_lz77_method(method);
        }
        if args.no_ans {
            cfg = cfg.with_ans(false);
        }
        if args.no_gaborish {
            cfg = cfg.with_gaborish(false);
        }
        if args.adaptive_gaborish {
            cfg = cfg.with_adaptive_gaborish(true);
        }
        if args.epf != -1 {
            cfg = cfg.with_epf_level(args.epf);
        }
        // W44-130 Chunk D: dispatch policies absorbed into
        // `EncoderImprovementsCustom`; setters deleted.
        // W44-131 Chunk E: `--strategy` flag drives the top-level
        // preset; `Zenjxl` (default) keeps the Chunk-D wrap so
        // dispatch overrides survive.
        cfg = apply_strategy_to_lossy(cfg, strategy_arg, epf_dispatch, pixel_loss_dispatch);
        if args.noise || args.denoise {
            cfg = cfg.with_noise(true);
        }
        if args.denoise {
            cfg = cfg.with_denoise(true);
        }
        // libjxl-parity Option<f32> knobs — None is a no-op
        cfg = cfg.with_photon_noise_iso(args.photon_noise_iso);
        cfg = cfg.with_original_distance(args.original_distance);
        cfg = cfg.with_quant_ac_rescale(args.quant_ac_rescale);
        if args.no_perceptual_optimizations {
            cfg = cfg.with_perceptual_optimizations(false);
        }
        if args.error_diffusion {
            cfg = cfg.with_error_diffusion(true);
        }
        if args.no_pixel_domain_loss {
            cfg = cfg.with_pixel_domain_loss(false);
        }
        if args.no_patches {
            cfg = cfg.with_patches(false);
        }
        if args.no_dot_detection {
            cfg = cfg.with_dot_detection(false);
        } else if args.dot_detection {
            cfg = cfg.with_dot_detection(true);
        }
        if args.lz77 {
            cfg = cfg.with_lz77(true);
        }
        if args.no_lz77 {
            cfg = cfg.with_lz77(false);
        }

        if args.progressive {
            cfg = cfg.with_progressive(ProgressiveMode::DcVlfLfAc);
        }
        if args.qprogressive {
            cfg = cfg.with_progressive(ProgressiveMode::QuantizedAcFullAc);
        }
        if args.lf_frame {
            cfg = cfg.with_lf_frame(true);
        }
        if args.experimental {
            cfg = cfg.with_mode(jxl_encoder::EncoderMode::Experimental);
        }

        if args.dct8_only {
            cfg = cfg.with_force_strategy(Some(0));
        }
        if let Some(s) = args.force_strategy {
            cfg = cfg.with_force_strategy(Some(s));
        }
        if let Some(s) = args.max_strategy_size {
            cfg = cfg.with_max_strategy_size(Some(s));
        }

        // ── A1 passthrough — libjxl cjxl parity knobs ─────────────
        cfg = cfg.with_alpha_distance(args.alpha_distance);
        cfg = cfg.with_group_order(args.group_order);
        cfg = cfg.with_center_x(args.center_x);
        cfg = cfg.with_center_y(args.center_y);
        cfg = cfg.with_upsampling_mode(args.upsampling_mode);
        cfg = cfg.with_faster_decoding(args.faster_decoding);
        cfg = cfg.with_container_mode(container_mode_from_cli(args.container));
        cfg = cfg.with_progressive_dc(args.progressive_dc);
        cfg = cfg.with_buffering(Buffering::from_i8(args.buffering));
        if (args.center_x.is_some() || args.center_y.is_some())
            && !matches!(args.group_order, Some(1))
        {
            eprintln!(
                "Warning: --center-x / --center-y require --group-order 1 \
                 (center-first); the values will be stored but the AC group \
                 reorder will not engage. Mirrors libjxl cjxl behaviour."
            );
        }

        #[cfg(feature = "butteraugli-loop")]
        {
            if args.no_butteraugli {
                cfg = cfg.with_butteraugli_iters(0);
            } else if let Some(n) = args.butteraugli_iters {
                cfg = cfg.with_butteraugli_iters(n);
            }
            // else: use effort-derived default from with_effort()
            if !args.quiet && cfg.butteraugli_iters() > 0 {
                println!("Butteraugli loop: {} iterations", cfg.butteraugli_iters());
            }
        }
        #[cfg(not(feature = "butteraugli-loop"))]
        if args.butteraugli_iters.is_some() && !args.no_butteraugli {
            eprintln!("Warning: --butteraugli-iters requires the butteraugli-loop feature");
            eprintln!("Rebuild with: cargo build --features butteraugli-loop");
        }
        #[cfg(feature = "ssim2-loop")]
        if let Some(n) = args.ssim2_iters {
            cfg = cfg.with_ssim2_iters(n);
            if !args.quiet && n > 0 {
                println!("SSIM2 loop: {} iterations", n);
            }
        }
        #[cfg(not(feature = "ssim2-loop"))]
        if args.ssim2_iters.is_some() {
            eprintln!("Warning: --ssim2-iters requires the ssim2-loop feature");
            eprintln!("Rebuild with: cargo build --features ssim2-loop");
        }
        #[cfg(feature = "zensim-loop")]
        if let Some(n) = args.zensim_iters {
            cfg = cfg.with_zensim_iters(n);
            if !args.quiet && n > 0 {
                println!("Zensim loop: {} iterations", n);
            }
        }
        #[cfg(not(feature = "zensim-loop"))]
        if args.zensim_iters.is_some() {
            eprintln!("Warning: --zensim-iters requires the zensim-loop feature");
            eprintln!("Rebuild with: cargo build --features zensim-loop");
        }

        // Rate control path (uses internal VarDctEncoder directly)
        #[cfg(feature = "rate-control")]
        if args.rate_control {
            // Rate control needs the internal VarDctEncoder for multi-pass
            use jxl_encoder::vardct::VarDctEncoder;
            let mut tiny = VarDctEncoder::new(distance);
            tiny.effort = args.effort;
            // Rate control doesn't go through LossyConfig, so apply effort defaults manually
            tiny.use_ans = if args.no_ans { false } else { args.effort >= 4 };
            tiny.optimize_codes = args.effort >= 2;
            tiny.custom_orders = args.effort >= 3;
            tiny.ac_strategy_enabled = args.effort >= 3;
            tiny.enable_noise = args.noise || args.denoise;
            tiny.enable_denoise = args.denoise;
            // libjxl gates gaborish at distance > 0.5 (enc_frame.cc:281).
            // Mirror the LossyConfig::encode wiring at api.rs:3842 so the
            // rate-control CLI path produces the same gaborish state as
            // the default API path for the same distance.
            tiny.enable_gaborish = if args.no_gaborish {
                false
            } else {
                args.effort >= 3 && distance > 0.5
            };
            // EX-J13: adaptive gaborish is silently gated to be a subset
            // of gaborish (no-op when the fixed inverse is disabled).
            tiny.enable_adaptive_gaborish = tiny.enable_gaborish && args.adaptive_gaborish;
            // libjxl `--epf -1..3` override (enc_frame.cc:284-285).
            tiny.epf_level_override = if args.epf < 0 {
                None
            } else {
                Some(args.epf.clamp(0, 3) as u32)
            };
            tiny.error_diffusion = args.error_diffusion;
            tiny.pixel_domain_loss = if args.no_pixel_domain_loss {
                false
            } else {
                args.effort >= 5
            };
            tiny.enable_lz77 = if args.lz77 {
                true
            } else if args.no_lz77 {
                false
            } else {
                args.effort >= 9
            };
            if let Some(method) = lz77_method {
                tiny.lz77_method = method;
            }
            if args.dct8_only {
                tiny.force_strategy = Some(0);
            }
            if let Some(s) = args.force_strategy {
                tiny.force_strategy = Some(s);
            }
            if let Some(max_size) = args.max_strategy_size {
                if max_size < 16 {
                    tiny.profile.try_dct16 = false;
                }
                if max_size < 32 {
                    tiny.profile.try_dct32 = false;
                }
                if max_size < 64 {
                    tiny.profile.try_dct64 = false;
                }
            }
            if args.progressive {
                tiny.progressive = ProgressiveMode::DcVlfLfAc;
            }
            if args.qprogressive {
                tiny.progressive = ProgressiveMode::QuantizedAcFullAc;
            }
            if args.lf_frame {
                tiny.use_lf_frame = true;
            }

            if args.streaming_input && !args.quiet {
                eprintln!("Warning: --streaming-input ignored — --rate-control encodes in memory");
            }
            let linear_rgb = srgb_u8_to_linear_f32(&data);
            let rc_config = jxl_encoder::vardct::RateControlConfig {
                max_iterations: args.rc_iterations,
                ..Default::default()
            };
            let result = tiny.encode_with_rate_control_config(
                width as usize,
                height as usize,
                &linear_rgb,
                &rc_config,
            );
            if !args.quiet
                && let Ok((_, iters)) = &result
            {
                println!("Rate control converged in {} iterations", iters);
            }
            result
                .map(|(data, _)| EncodeOutput::Bytes(data))
                .map_err(|e| jxl_encoder::at(jxl_encoder::EncodeError::from(e)))
        } else if args.streaming_input
            && !args.progressive
            && !args.qprogressive
            && !args.lf_frame
            && args.progressive_dc == 0
        {
            if !args.quiet {
                println!(
                    "Streaming-input: pushing {} chunks of up to {} rows via LossyEncoder",
                    height.div_ceil(STREAM_CHUNK_ROWS),
                    STREAM_CHUNK_ROWS
                );
                if args.streaming_output {
                    println!(
                        "Streaming-output: writing codestream directly to {}",
                        args.output.display()
                    );
                }
            }
            encode_lossy_streaming(
                &cfg,
                width,
                height,
                layout,
                &data,
                metadata.as_ref(),
                source_gamma,
                args.intensity_target,
                args.brotli_effort,
                args.premultiply,
                &args.output,
                args.streaming_output,
            )
        } else {
            if args.streaming_input && !args.quiet {
                eprintln!(
                    "Warning: --streaming-input ignored — lossy path requires no \
                     --progressive / --qprogressive / --lf-frame / --progressive-dc"
                );
            }
            let mut req = cfg.encode_request(width, height, layout);
            if let Some(ref meta) = metadata {
                req = req.with_metadata(meta);
            }
            // cICP outranks gAMA (issue #71); the lossy PQ/HLG scale bug
            // is fixed by the libjxl-parity intensity_target dispatch
            // (issue #73 — SetIntensityTarget: PQ 10,000 / HLG 1,000 nits).
            if let Some(ce) = cicp.and_then(|c| color_encoding_from_cicp(c, layout.is_grayscale()))
            {
                req = req.with_color_encoding(ce);
            } else if let Some(gamma) = source_gamma {
                req = req.with_source_gamma(gamma);
            }
            if let Some(it) = args.intensity_target {
                req = req.with_intensity_target(it);
            }
            if let Some(q) = args.brotli_effort {
                req = req.with_brotli_metadata(q);
            }
            if args.premultiply != 0 && layout.has_alpha() {
                req = req.with_premultiplied_alpha_mode(PremultipliedAlphaMode::from_i8(
                    args.premultiply,
                ));
            }
            req.encode(&data).map(EncodeOutput::Bytes)
        }

        #[cfg(not(feature = "rate-control"))]
        {
            if args.rate_control {
                eprintln!("Warning: --rate-control requires the rate-control feature");
                eprintln!("Rebuild with: cargo build --features rate-control");
            }
            if args.streaming_input
                && !args.progressive
                && !args.qprogressive
                && !args.lf_frame
                && args.progressive_dc == 0
            {
                if !args.quiet {
                    println!(
                        "Streaming-input: pushing {} chunks of up to {} rows via LossyEncoder",
                        height.div_ceil(STREAM_CHUNK_ROWS),
                        STREAM_CHUNK_ROWS
                    );
                    if args.streaming_output {
                        println!(
                            "Streaming-output: writing codestream directly to {}",
                            args.output.display()
                        );
                    }
                }
                encode_lossy_streaming(
                    &cfg,
                    width,
                    height,
                    layout,
                    &data,
                    metadata.as_ref(),
                    source_gamma,
                    args.intensity_target,
                    args.brotli_effort,
                    args.premultiply,
                    &args.output,
                    args.streaming_output,
                )
            } else {
                if args.streaming_input && !args.quiet {
                    eprintln!(
                        "Warning: --streaming-input ignored — lossy path requires no \
                         --progressive / --qprogressive / --lf-frame / --progressive-dc"
                    );
                }
                let mut req = cfg.encode_request(width, height, layout);
                if let Some(ref meta) = metadata {
                    req = req.with_metadata(meta);
                }
                // cICP outranks gAMA (issue #71); the lossy PQ/HLG scale bug
                // is fixed by the libjxl-parity intensity_target dispatch
                // (issue #73 — SetIntensityTarget: PQ 10,000 / HLG 1,000 nits).
                if let Some(ce) =
                    cicp.and_then(|c| color_encoding_from_cicp(c, layout.is_grayscale()))
                {
                    req = req.with_color_encoding(ce);
                } else if let Some(gamma) = source_gamma {
                    req = req.with_source_gamma(gamma);
                }
                if let Some(it) = args.intensity_target {
                    req = req.with_intensity_target(it);
                }
                if let Some(q) = args.brotli_effort {
                    req = req.with_brotli_metadata(q);
                }
                if args.premultiply != 0 && layout.has_alpha() {
                    req = req.with_premultiplied_alpha_mode(PremultipliedAlphaMode::from_i8(
                        args.premultiply,
                    ));
                }
                req.encode(&data).map(EncodeOutput::Bytes)
            }
        }
    } else {
        // Lossless modular path (or lossy RGBA/gray which falls through to modular)
        let mut cfg = LosslessConfig::new()
            .with_effort(args.effort)
            .with_threads(args.threads);
        if args.no_ans {
            cfg = cfg.with_ans(false);
        }
        if args.tree_learning {
            cfg = cfg.with_tree_learning(true).with_ans(true);
        }
        if args.no_tree_learning {
            cfg = cfg.with_tree_learning(false);
        }
        if args.squeeze {
            cfg = cfg.with_squeeze(true);
        }
        if args.no_squeeze {
            cfg = cfg.with_squeeze(false);
        }
        if args.no_patches {
            cfg = cfg.with_patches(false);
        }
        if args.lz77 {
            cfg = cfg.with_lz77(true);
        }
        if args.no_lz77 {
            cfg = cfg.with_lz77(false);
        }
        if let Some(method) = lz77_method {
            cfg = cfg.with_lz77_method(method);
        }
        if args.lossy_palette {
            cfg = cfg.with_lossy_palette(true);
        }
        if let Some(rct) = args.force_rct {
            cfg = cfg.with_force_rct(Some(jxl_encoder::RctType(rct)));
        }
        if let Some(f) = args.tree_learning_sample_fraction {
            cfg = cfg.with_tree_learning_sample_fraction(f);
        }
        if args.smart_fanout {
            cfg = cfg.with_smart_fanout(true);
        }
        if args.small_image_fallback {
            cfg = cfg.with_small_image_fallback_override(Some(true));
        }
        if args.experimental {
            cfg = cfg.with_mode(jxl_encoder::EncoderMode::Experimental);
        }
        // ── A1 passthrough — libjxl cjxl modular knobs ──────────────
        cfg = cfg.with_modular_predictor(args.modular_predictor);
        cfg = cfg.with_modular_palette_colors(args.modular_palette_colors);
        cfg = cfg
            .with_modular_channel_colors_global_percent(args.modular_channel_colors_global_percent);
        cfg = cfg
            .with_modular_channel_colors_group_percent(args.modular_channel_colors_group_percent);
        cfg = cfg.with_modular_nb_prev_channels(args.modular_nb_prev_channels);
        cfg = cfg.with_faster_decoding(args.faster_decoding);
        cfg = cfg.with_modular_group_size(args.modular_group_size);
        cfg = cfg.with_container_mode(container_mode_from_cli(args.container));
        cfg = cfg.with_buffering(Buffering::from_i8(args.buffering));
        let cfg = cfg;

        // `--ec_resampling N` (libjxl parity): pre-downsample the
        // alpha plane on the lossless RGBA path and re-encode RGB
        // + a half/quarter/eighth-res alpha extra channel. Mirrors
        // libjxl `enc_frame.cc:1620` (`DownsampleImage(ec, factor)`).
        let ec_factor = args.ec_resampling;
        let ec_resampling_active = ec_factor > 1
            && matches!(
                layout,
                PixelLayout::Rgba8 | PixelLayout::Bgra8 | PixelLayout::GrayAlpha8
            );
        if ec_factor != 1 && !matches!(ec_factor, 2 | 4 | 8) {
            eprintln!("Error: --ec_resampling must be one of {{1, 2, 4, 8}}, got {ec_factor}");
            std::process::exit(1);
        }
        if ec_factor > 1 && !ec_resampling_active && !args.quiet {
            eprintln!(
                "Warning: --ec_resampling={ec_factor} ignored — only 8-bit \
                 RGBA / BGRA / Gray+Alpha lossless input is wired today \
                 (got {layout:?})."
            );
        }
        // Multi-group writer now propagates `dim_shift > 0` through
        // `extract_region` so downsampled extras (e.g. half-res
        // alpha) crop in channel-local coordinates per libjxl
        // `enc_modular.cc:1400-1407`. Previously the CLI refused
        // multi-group on `--ec_resampling > 1`; the rejection is no
        // longer needed. Lossy still guards `dim_shift > 0` in
        // `vardct/encoder.rs` — only the lossless RGBA / BGRA /
        // GrayAlpha path below exercises the new writer.

        if ec_resampling_active {
            if args.streaming_input && !args.quiet {
                eprintln!(
                    "Warning: --streaming-input ignored — --ec_resampling pre-splits the \
                     alpha plane in memory"
                );
            }
            // Split: bpp = 4 for RGBA/BGRA, bpp = 2 for GrayAlpha.
            let (bpp, color_layout, color_bpp) = match layout {
                PixelLayout::Rgba8 => (4, PixelLayout::Rgb8, 3),
                PixelLayout::Bgra8 => (4, PixelLayout::Bgr8, 3),
                PixelLayout::GrayAlpha8 => (2, PixelLayout::Gray8, 1),
                _ => unreachable!(),
            };
            let n = (width * height) as usize;
            let mut color = Vec::with_capacity(n * color_bpp);
            let mut alpha_full = Vec::with_capacity(n);
            for px in data.chunks_exact(bpp) {
                color.extend_from_slice(&px[..color_bpp]);
                alpha_full.push(px[bpp - 1]);
            }
            let alpha_ds = jxl_encoder::downsample_channel_u8(
                &alpha_full,
                width as usize,
                height as usize,
                ec_factor,
            );
            let log2_factor = ec_factor.trailing_zeros(); // 1→0, 2→1, 4→2, 8→3
            let extras = [
                jxl_encoder::api::ExtraChannel::from_alpha_buf(&alpha_ds, false)
                    .with_dim_shift(log2_factor),
            ];
            let mut req = cfg
                .encode_request(width, height, color_layout)
                .with_extra_channels(&extras);
            if let Some(ref meta) = metadata {
                req = req.with_metadata(meta);
            }
            // cICP outranks gAMA (issue #71); the lossy PQ/HLG scale bug
            // is fixed by the libjxl-parity intensity_target dispatch
            // (issue #73 — SetIntensityTarget: PQ 10,000 / HLG 1,000 nits).
            if let Some(ce) = cicp.and_then(|c| color_encoding_from_cicp(c, layout.is_grayscale()))
            {
                req = req.with_color_encoding(ce);
            } else if let Some(gamma) = source_gamma {
                req = req.with_source_gamma(gamma);
            }
            // ── A1 passthrough — wire intensity_target + brotli_effort ──
            if let Some(it) = args.intensity_target {
                req = req.with_intensity_target(it);
            }
            if let Some(q) = args.brotli_effort {
                req = req.with_brotli_metadata(q);
            }
            req.encode(&color).map(EncodeOutput::Bytes)
        } else if args.streaming_input && !args.lossy_palette {
            if !args.quiet {
                println!(
                    "Streaming-input: pushing {} chunks of up to {} rows via LosslessEncoder",
                    height.div_ceil(STREAM_CHUNK_ROWS),
                    STREAM_CHUNK_ROWS
                );
                if args.streaming_output {
                    println!(
                        "Streaming-output: writing codestream directly to {}",
                        args.output.display()
                    );
                }
            }
            encode_lossless_streaming(
                &cfg,
                width,
                height,
                layout,
                &data,
                metadata.as_ref(),
                source_gamma,
                args.intensity_target,
                args.brotli_effort,
                args.premultiply,
                &args.output,
                args.streaming_output,
            )
        } else {
            if args.streaming_input && !args.quiet {
                eprintln!(
                    "Warning: --streaming-input ignored — lossless path is incompatible \
                     with --lossy-palette (not yet supported by LosslessEncoder)"
                );
            }
            let mut req = cfg.encode_request(width, height, layout);
            if let Some(ref meta) = metadata {
                req = req.with_metadata(meta);
            }
            if let Some(ce) = cicp.and_then(|c| color_encoding_from_cicp(c, layout.is_grayscale()))
            {
                // cICP outranks gAMA (issue #71): HDR PNGs signal PQ/HLG here.
                req = req.with_color_encoding(ce);
            } else if let Some(gamma) = source_gamma {
                req = req.with_source_gamma(gamma);
            }
            // ── A1 passthrough — wire intensity_target + brotli_effort ──
            if let Some(it) = args.intensity_target {
                req = req.with_intensity_target(it);
            }
            if let Some(q) = args.brotli_effort {
                req = req.with_brotli_metadata(q);
            }
            req.encode(&data).map(EncodeOutput::Bytes)
        }
    };

    let encoded = match encoded {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Error encoding: {}", e);
            std::process::exit(1);
        }
    };

    let encode_time = start.elapsed();
    if std::env::var_os("CJXLRS_TIMING").is_some() {
        eprintln!("[cli-timing] encode-arm total: {:?}", _t_enc.elapsed());
    }

    // Write the output unless the encode arm proved it already streamed the
    // bytes to disk. (Never key this off the CLI flags: arms that fall back
    // to in-memory encoding must still produce an output file.)
    let encoded_bytes = match encoded {
        EncodeOutput::Bytes(bytes) => {
            if let Err(e) = write_output(&args.output, &bytes) {
                eprintln!("Error writing output: {}", e);
                std::process::exit(1);
            }
            Some(bytes)
        }
        EncodeOutput::WrittenDirectly => None,
    };

    let input_size = std::fs::metadata(&args.input).map(|m| m.len()).unwrap_or(0);
    let output_size = match &encoded_bytes {
        Some(bytes) => bytes.len() as u64,
        // `finish_to` wrote the bytes directly; ask the FS for the size
        None => std::fs::metadata(&args.output)
            .map(|m| m.len())
            .unwrap_or(0),
    };
    let ratio = if input_size > 0 {
        output_size as f64 / input_size as f64
    } else {
        0.0
    };

    if !args.quiet {
        println!();
        println!("Input size:  {} bytes", input_size);
        println!("Output size: {} bytes", output_size);
        println!("Ratio:       {:.2}x", ratio);
        println!("Time:        {:.2?}", encode_time);
    } else {
        println!("{}", args.output.display());
    }
    // W44-27: emit per-block AdjustQuantBlockAC firing-rate TSV when feature is on.
    #[cfg(feature = "investigate-adjust-quant-block-ac")]
    {
        let tag = std::env::var("JXL_AQBA_DIAG_TAG").unwrap_or_else(|_| {
            args.input
                .file_stem()
                .and_then(|s| s.to_str())
                .map(|s| format!("{s}_d{distance:.1}"))
                .unwrap_or_else(|| "encode".to_string())
        });
        jxl_encoder::vardct::aqba_diag::emit_and_reset(&tag);
    }
}

/// Convert the libjxl `--container -1|0|1` integer CLI form to our
/// [`ContainerMode`] enum. Negative = auto (default), `0` = never wrap,
/// `1` = always wrap. Out-of-range values clamp to `Auto`.
fn container_mode_from_cli(v: i8) -> ContainerMode {
    match v {
        0 => ContainerMode::Never,
        1 => ContainerMode::Always,
        _ => ContainerMode::Auto,
    }
}

/// Drive the lossy encode through [`LossyEncoder`]'s `push_rows()` +
/// `finish_to()` / `finish()` surface for the `--streaming-input` /
/// `--streaming-output` CLI flags.
///
/// The pre-built [`LossyConfig`] is reused verbatim (so distance, effort,
/// gaborish, butteraugli, etc. all flow from the same dispatch as the
/// bulk path). The decoded pixel buffer is sliced into row-groups of
/// [`STREAM_CHUNK_ROWS`] and fed to `push_rows()` to exercise the
/// incremental path end-to-end; bitstreams are bit-identical to the
/// bulk one-shot path on the eligible subset.
///
/// When `streaming_output` is `true` the bytes are streamed directly to
/// `output_path` via [`LossyEncoder::finish_to`] and
/// [`EncodeOutput::WrittenDirectly`] is returned.
/// Outcome of the one-shot encode dispatch: either encoded bytes the caller
/// still has to write to the output path, or proof that a streaming encoder
/// already wrote them there directly via `finish_to`.
///
/// Forcing every encode arm to state which one happened is the point: the
/// old scheme keyed the "skip `write_output`" decision off the raw
/// `--streaming-input`/`--streaming-output` flags, so any arm that fell back
/// to in-memory encoding while both flags were set (progressive modes,
/// `--rate-control`, `--lossy-palette`, `--ec_resampling`) silently produced
/// no output file at all and still exited 0.
enum EncodeOutput {
    /// In-memory codestream; caller must write it to the output path.
    Bytes(Vec<u8>),
    /// A streaming encoder already wrote the output file via `finish_to`.
    WrittenDirectly,
}

#[allow(clippy::too_many_arguments)]
fn encode_lossy_streaming(
    cfg: &LossyConfig,
    width: u32,
    height: u32,
    layout: PixelLayout,
    data: &[u8],
    metadata: Option<&jxl_encoder::ImageMetadata>,
    source_gamma: Option<f32>,
    intensity_target: Option<f32>,
    brotli_effort: Option<u32>,
    premultiply: i8,
    output_path: &std::path::Path,
    streaming_output: bool,
) -> Result<EncodeOutput, jxl_encoder::At<jxl_encoder::EncodeError>> {
    let mut enc = cfg.encoder(width, height, layout)?;
    if let Some(meta) = metadata {
        if let Some(icc) = meta.icc_profile() {
            enc = enc.with_icc_profile(icc);
        }
        if let Some(exif) = meta.exif() {
            enc = enc.with_exif(exif);
        }
        if let Some(xmp) = meta.xmp() {
            enc = enc.with_xmp(xmp);
        }
        if let Some(jumbf) = meta.jumbf() {
            enc = enc.with_jumbf(jumbf);
        }
    }
    if let Some(g) = source_gamma {
        enc = enc.with_source_gamma(g);
    }
    if let Some(it) = intensity_target {
        enc = enc.with_intensity_target(it);
    }
    if let Some(q) = brotli_effort {
        enc = enc.with_brotli_metadata(q);
    }
    if premultiply != 0 && layout.has_alpha() {
        enc = enc.with_premultiplied_alpha(premultiply > 0);
    }

    let bpp = layout.bytes_per_pixel();
    let row_bytes = (width as usize).checked_mul(bpp).ok_or_else(|| {
        jxl_encoder::at(jxl_encoder::EncodeError::InvalidInput {
            message: "row dimensions overflow".into(),
        })
    })?;
    let mut rows_remaining = height;
    let mut offset = 0usize;
    while rows_remaining > 0 {
        let chunk_rows = rows_remaining.min(STREAM_CHUNK_ROWS);
        let chunk_bytes = row_bytes * chunk_rows as usize;
        let end = offset + chunk_bytes;
        if end > data.len() {
            return Err(jxl_encoder::at(jxl_encoder::EncodeError::InvalidInput {
                message: format!(
                    "streaming chunk past pixel buffer end ({end} > {})",
                    data.len()
                ),
            }));
        }
        enc.push_rows(&data[offset..end], chunk_rows)?;
        offset = end;
        rows_remaining -= chunk_rows;
    }

    if streaming_output {
        let file = File::create(output_path).map_err(|e| {
            jxl_encoder::at(jxl_encoder::EncodeError::InvalidInput {
                message: format!("create output {}: {}", output_path.display(), e),
            })
        })?;
        let writer = BufWriter::new(file);
        let _result = enc.finish_to(writer)?;
        Ok(EncodeOutput::WrittenDirectly)
    } else {
        enc.finish().map(EncodeOutput::Bytes)
    }
}

/// Drive the lossless modular encode through [`LosslessEncoder`]'s
/// `push_rows()` + `finish_to()` / `finish()` surface for the
/// `--streaming-input` / `--streaming-output` CLI flags. Counterpart to
/// [`encode_lossy_streaming`]; see that function for the architectural
/// notes.
#[allow(clippy::too_many_arguments)]
fn encode_lossless_streaming(
    cfg: &LosslessConfig,
    width: u32,
    height: u32,
    layout: PixelLayout,
    data: &[u8],
    metadata: Option<&jxl_encoder::ImageMetadata>,
    source_gamma: Option<f32>,
    intensity_target: Option<f32>,
    brotli_effort: Option<u32>,
    premultiply: i8,
    output_path: &std::path::Path,
    streaming_output: bool,
) -> Result<EncodeOutput, jxl_encoder::At<jxl_encoder::EncodeError>> {
    let mut enc = cfg.encoder(width, height, layout)?;
    if let Some(meta) = metadata {
        if let Some(icc) = meta.icc_profile() {
            enc = enc.with_icc_profile(icc);
        }
        if let Some(exif) = meta.exif() {
            enc = enc.with_exif(exif);
        }
        if let Some(xmp) = meta.xmp() {
            enc = enc.with_xmp(xmp);
        }
        if let Some(jumbf) = meta.jumbf() {
            enc = enc.with_jumbf(jumbf);
        }
    }
    if let Some(g) = source_gamma {
        enc = enc.with_source_gamma(g);
    }
    if let Some(it) = intensity_target {
        enc = enc.with_intensity_target(it);
    }
    if let Some(q) = brotli_effort {
        enc = enc.with_brotli_metadata(q);
    }
    if premultiply != 0 && layout.has_alpha() {
        enc = enc.with_premultiplied_alpha(premultiply > 0);
    }

    let bpp = layout.bytes_per_pixel();
    let row_bytes = (width as usize).checked_mul(bpp).ok_or_else(|| {
        jxl_encoder::at(jxl_encoder::EncodeError::InvalidInput {
            message: "row dimensions overflow".into(),
        })
    })?;
    let mut rows_remaining = height;
    let mut offset = 0usize;
    while rows_remaining > 0 {
        let chunk_rows = rows_remaining.min(STREAM_CHUNK_ROWS);
        let chunk_bytes = row_bytes * chunk_rows as usize;
        let end = offset + chunk_bytes;
        if end > data.len() {
            return Err(jxl_encoder::at(jxl_encoder::EncodeError::InvalidInput {
                message: format!(
                    "streaming chunk past pixel buffer end ({end} > {})",
                    data.len()
                ),
            }));
        }
        enc.push_rows(&data[offset..end], chunk_rows)?;
        offset = end;
        rows_remaining -= chunk_rows;
    }

    if streaming_output {
        let file = File::create(output_path).map_err(|e| {
            jxl_encoder::at(jxl_encoder::EncodeError::InvalidInput {
                message: format!("create output {}: {}", output_path.display(), e),
            })
        })?;
        let writer = BufWriter::new(file);
        let _result = enc.finish_to(writer)?;
        Ok(EncodeOutput::WrittenDirectly)
    } else {
        enc.finish().map(EncodeOutput::Bytes)
    }
}

fn quality_to_distance(quality: u32) -> f32 {
    if quality >= 100 {
        0.0
    } else if quality >= 90 {
        (100 - quality) as f32 / 10.0
    } else if quality >= 70 {
        1.0 + (90 - quality) as f32 / 20.0
    } else {
        2.0 + (70 - quality) as f32 / 10.0
    }
}

/// sRGB to linear conversion (exact IEC 61966-2-1 transfer function).
#[cfg(feature = "rate-control")]
fn srgb_to_linear(c: u8) -> f32 {
    let c = c as f32 / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

#[cfg(feature = "rate-control")]
fn srgb_u8_to_linear_f32(data: &[u8]) -> Vec<f32> {
    data.chunks(3)
        .flat_map(|px| {
            [
                srgb_to_linear(px[0]),
                srgb_to_linear(px[1]),
                srgb_to_linear(px[2]),
            ]
        })
        .collect()
}

#[allow(clippy::type_complexity)]
/// Map a PNG cICP chunk (ITU-T H.273 code points) to a JXL
/// [`ColorEncoding`] via the libjxl-parity
/// [`jxl_encoder::headers::ColorEncoding::from_cicp`]. Returns `None`
/// for combinations it rejects (with a stderr note) — the caller then
/// falls back to the gAMA/sRGB logic. Issue #71: without this, 16-bit
/// PQ HDR PNGs were signaled as sRGB transfer in the codestream.
fn color_encoding_from_cicp(
    cicp: png::CodingIndependentCodePoints,
    grayscale: bool,
) -> Option<jxl_encoder::headers::ColorEncoding> {
    use jxl_encoder::headers::color_encoding::ColorSpace;
    match jxl_encoder::headers::ColorEncoding::from_cicp(
        cicp.color_primaries,
        cicp.transfer_function,
        cicp.matrix_coefficients,
        cicp.is_video_full_range_image,
    ) {
        Ok(mut ce) => {
            if grayscale {
                ce.color_space = ColorSpace::Gray;
            }
            Some(ce)
        }
        Err(e) => {
            eprintln!("Warning: PNG cICP ignored ({e}); falling back to gAMA/sRGB");
            None
        }
    }
}

#[allow(clippy::type_complexity)]
fn read_png(
    path: &PathBuf,
) -> Result<
    (
        u32,
        u32,
        png::ColorType,
        png::BitDepth,
        Vec<u8>,
        Option<f32>,
        Option<png::CodingIndependentCodePoints>,
    ),
    Box<dyn std::error::Error>,
> {
    let file = BufReader::new(File::open(path)?);
    let mut decoder = png::Decoder::new(file);
    // Expand palette/indexed PNGs to RGB/RGBA, expand low-bit-depth grayscale to 8-bit
    decoder.set_transformations(png::Transformations::EXPAND);
    let mut reader = decoder.read_info()?;

    // Extract gamma from PNG metadata:
    // - If sRGB chunk is present, use sRGB TF (default, gamma=None)
    // - If only gAMA chunk is present (no sRGB), preserve the gamma value
    let png_info = reader.info();
    let source_gamma = if png_info.srgb.is_some() {
        None // sRGB chunk present → use sRGB TF (default)
    } else {
        png_info.gama_chunk.map(|g| g.into_value())
    };
    // cICP (H.273 code points) outranks gAMA/sRGB when present — HDR
    // PNGs (PQ / HLG) signal their transfer function here. Issue #71:
    // ignoring it tagged PQ input as sRGB in the output codestream.
    let cicp = png_info.coding_independent_code_points;

    let mut buf = vec![
        0;
        reader
            .output_buffer_size()
            .expect("no frame info available")
    ];
    let info = reader.next_frame(&mut buf)?;
    buf.truncate(info.buffer_size());

    // PNG stores 16-bit samples as big-endian. Our encoder expects native-endian u16.
    // On little-endian platforms, swap each u16's bytes.
    if info.bit_depth == png::BitDepth::Sixteen && cfg!(target_endian = "little") {
        for pair in buf.chunks_exact_mut(2) {
            pair.swap(0, 1);
        }
    }

    Ok((
        info.width,
        info.height,
        info.color_type,
        info.bit_depth,
        buf,
        source_gamma,
        cicp,
    ))
}

fn write_output(path: &PathBuf, data: &[u8]) -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    writer.write_all(data)?;
    writer.flush()?;
    Ok(())
}

/// Read a PNM file (P5 = PGM grayscale, P6 = PPM RGB). Supports 8-bit and 16-bit.
#[allow(clippy::type_complexity)]
fn read_pnm(
    path: &PathBuf,
) -> Result<(u32, u32, png::ColorType, png::BitDepth, Vec<u8>), Box<dyn std::error::Error>> {
    use std::io::BufRead;
    let file = BufReader::new(File::open(path)?);
    let mut lines = file.lines();

    // Read magic
    let magic = lines.next().ok_or("Empty PNM file")??;
    let magic = magic.trim();
    let (color_type, channels) = match magic {
        "P5" => (png::ColorType::Grayscale, 1),
        "P6" => (png::ColorType::Rgb, 3),
        _ => return Err(format!("Unsupported PNM magic: {}", magic).into()),
    };

    // Read dimensions and maxval, skipping comments
    let mut tokens: Vec<String> = Vec::new();
    for line in &mut lines {
        let line = line?;
        let line = line.trim().to_string();
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        tokens.extend(line.split_whitespace().map(String::from));
        if tokens.len() >= 3 {
            break;
        }
    }
    if tokens.len() < 3 {
        return Err("PNM header incomplete: need width, height, maxval".into());
    }

    let width: u32 = tokens[0].parse()?;
    let height: u32 = tokens[1].parse()?;
    let maxval: u32 = tokens[2].parse()?;

    let bit_depth = if maxval <= 255 {
        png::BitDepth::Eight
    } else if maxval <= 65535 {
        png::BitDepth::Sixteen
    } else {
        return Err(format!("Unsupported PNM maxval: {}", maxval).into());
    };

    // Reconstruct the reader from remaining buffered data
    // The pixel data starts right after the newline following maxval.
    // Re-open and skip header bytes to get to pixel data.
    let raw = std::fs::read(path)?;
    // Find the pixel data start: after magic line, then after width/height/maxval tokens
    let mut pos = 0;
    // We need to skip: magic line + dimension/maxval lines (skipping comments)
    // Simpler: scan for the third non-comment number, then skip past the next newline/whitespace
    let mut nums_found = 0;
    // Skip magic line
    while pos < raw.len() && raw[pos] != b'\n' {
        pos += 1;
    }
    pos += 1; // skip the newline

    // Parse remaining header (width, height, maxval)
    while pos < raw.len() && nums_found < 3 {
        // Skip whitespace/newlines
        while pos < raw.len()
            && (raw[pos] == b' ' || raw[pos] == b'\n' || raw[pos] == b'\r' || raw[pos] == b'\t')
        {
            pos += 1;
        }
        if pos < raw.len() && raw[pos] == b'#' {
            // Skip comment line
            while pos < raw.len() && raw[pos] != b'\n' {
                pos += 1;
            }
            continue;
        }
        // Skip the number
        while pos < raw.len()
            && raw[pos] != b' '
            && raw[pos] != b'\n'
            && raw[pos] != b'\r'
            && raw[pos] != b'\t'
        {
            pos += 1;
        }
        nums_found += 1;
    }
    // Skip the single whitespace byte after maxval (required by PNM spec)
    if pos < raw.len() {
        pos += 1;
    }

    let pixel_data = &raw[pos..];
    let bytes_per_sample = if maxval <= 255 { 1 } else { 2 };
    let expected = (width as usize) * (height as usize) * channels * bytes_per_sample;

    if pixel_data.len() < expected {
        return Err(format!(
            "PNM pixel data too short: {} bytes, expected {}",
            pixel_data.len(),
            expected
        )
        .into());
    }

    let mut data = pixel_data[..expected].to_vec();

    // PNM 16-bit is big-endian. Convert to native-endian (same as PNG path).
    if bytes_per_sample == 2 && cfg!(target_endian = "little") {
        for pair in data.chunks_exact_mut(2) {
            pair.swap(0, 1);
        }
    }

    Ok((width, height, color_type, bit_depth, data))
}

struct ApngFrameData {
    pixels: Vec<u8>,
    delay_ms: u32,
}

struct ApngResult {
    width: u32,
    height: u32,
    color_type: png::ColorType,
    has_alpha: bool,
    num_loops: u32,
    frames: Vec<ApngFrameData>,
}

/// Read an APNG file, compositing frames according to dispose/blend ops.
/// Returns None if the PNG is not animated.
fn read_apng(path: &PathBuf) -> Result<Option<ApngResult>, Box<dyn std::error::Error>> {
    let file = BufReader::new(File::open(path)?);
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info()?;

    let actl = match reader.info().animation_control {
        Some(actl) => actl,
        None => return Ok(None),
    };

    let num_frames = actl.num_frames;
    let num_loops = actl.num_plays;
    let canvas_width = reader.info().width;
    let canvas_height = reader.info().height;
    let color_type = reader.info().color_type;
    let bit_depth = reader.info().bit_depth;

    if bit_depth != png::BitDepth::Eight {
        return Err(format!("APNG: only 8-bit supported, got {:?}", bit_depth).into());
    }

    let src_channels: usize = match color_type {
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        _ => return Err(format!("APNG: only RGB/RGBA supported, got {:?}", color_type).into()),
    };
    let has_alpha = color_type == png::ColorType::Rgba;

    // Work in RGBA8 for composition
    let canvas_pixels = (canvas_width * canvas_height) as usize;
    let mut canvas = vec![0u8; canvas_pixels * 4];
    let mut prev_canvas = Vec::new(); // saved for DisposeOp::Previous

    let mut frames = Vec::with_capacity(num_frames as usize);
    let mut frame_buf = vec![
        0u8;
        reader
            .output_buffer_size()
            .expect("no frame info available")
    ];

    let mut prev_dispose_op = png::DisposeOp::None;
    let mut prev_region: (u32, u32, u32, u32) = (0, 0, canvas_width, canvas_height);

    for _frame_idx in 0..num_frames {
        let info = reader.next_frame(&mut frame_buf)?;
        let frame_data = &frame_buf[..info.buffer_size()];

        let fc = reader.info().frame_control;

        let (fw, fh, fx, fy, delay_num, delay_den, dispose_op, blend_op) = if let Some(fc) = fc {
            (
                fc.width,
                fc.height,
                fc.x_offset,
                fc.y_offset,
                fc.delay_num,
                fc.delay_den,
                fc.dispose_op,
                fc.blend_op,
            )
        } else {
            // First frame without FrameControl — use full canvas, 100ms default
            (
                canvas_width,
                canvas_height,
                0,
                0,
                100,
                1000,
                png::DisposeOp::None,
                png::BlendOp::Source,
            )
        };

        // Apply previous frame's dispose_op
        if !frames.is_empty() {
            let (px, py, pw, ph) = prev_region;
            match prev_dispose_op {
                png::DisposeOp::None => {}
                png::DisposeOp::Background => {
                    for y in py..(py + ph) {
                        for x in px..(px + pw) {
                            let idx = ((y * canvas_width + x) * 4) as usize;
                            canvas[idx..idx + 4].fill(0);
                        }
                    }
                }
                png::DisposeOp::Previous => {
                    canvas.copy_from_slice(&prev_canvas);
                }
            }
        }

        // Save canvas for potential DisposeOp::Previous
        if dispose_op == png::DisposeOp::Previous {
            prev_canvas = canvas.clone();
        }

        // Composite frame onto canvas
        for y in 0..fh {
            for x in 0..fw {
                let src_idx = ((y * fw + x) * src_channels as u32) as usize;
                let dst_idx = (((fy + y) * canvas_width + (fx + x)) * 4) as usize;

                let (sr, sg, sb, sa) = if has_alpha {
                    (
                        frame_data[src_idx],
                        frame_data[src_idx + 1],
                        frame_data[src_idx + 2],
                        frame_data[src_idx + 3],
                    )
                } else {
                    (
                        frame_data[src_idx],
                        frame_data[src_idx + 1],
                        frame_data[src_idx + 2],
                        255,
                    )
                };

                match blend_op {
                    png::BlendOp::Source => {
                        canvas[dst_idx] = sr;
                        canvas[dst_idx + 1] = sg;
                        canvas[dst_idx + 2] = sb;
                        canvas[dst_idx + 3] = sa;
                    }
                    png::BlendOp::Over => {
                        if sa == 255 {
                            canvas[dst_idx] = sr;
                            canvas[dst_idx + 1] = sg;
                            canvas[dst_idx + 2] = sb;
                            canvas[dst_idx + 3] = 255;
                        } else if sa > 0 {
                            let sa_f = sa as f32 / 255.0;
                            let da_f = canvas[dst_idx + 3] as f32 / 255.0;
                            let out_a = sa_f + da_f * (1.0 - sa_f);
                            if out_a > 0.0 {
                                let inv = 1.0 / out_a;
                                let blend = |s: u8, d: u8| -> u8 {
                                    ((s as f32 * sa_f + d as f32 * da_f * (1.0 - sa_f)) * inv) as u8
                                };
                                canvas[dst_idx] = blend(sr, canvas[dst_idx]);
                                canvas[dst_idx + 1] = blend(sg, canvas[dst_idx + 1]);
                                canvas[dst_idx + 2] = blend(sb, canvas[dst_idx + 2]);
                                canvas[dst_idx + 3] = (out_a * 255.0) as u8;
                            }
                        }
                        // sa == 0: fully transparent source, no change
                    }
                }
            }
        }

        // Compute delay in milliseconds
        let den = if delay_den == 0 {
            100
        } else {
            delay_den as u32
        };
        let delay_ms = (delay_num as u32 * 1000 + den / 2) / den;

        // Extract full canvas as frame pixels
        let frame_pixels = if has_alpha {
            canvas.clone()
        } else {
            // Strip alpha → RGB8
            let mut rgb = Vec::with_capacity(canvas_pixels * 3);
            for px in canvas.chunks_exact(4) {
                rgb.extend_from_slice(&px[..3]);
            }
            rgb
        };

        frames.push(ApngFrameData {
            pixels: frame_pixels,
            delay_ms,
        });

        prev_dispose_op = dispose_op;
        prev_region = (fx, fy, fw, fh);
    }

    Ok(Some(ApngResult {
        width: canvas_width,
        height: canvas_height,
        color_type,
        has_alpha,
        num_loops,
        frames,
    }))
}

#[cfg(test)]
mod cicp_tests {
    use super::color_encoding_from_cicp;
    use jxl_encoder::headers::color_encoding::{ColorSpace, Primaries, TransferFunction};

    fn cicp(cp: u8, tc: u8, mc: u8, full: bool) -> png::CodingIndependentCodePoints {
        png::CodingIndependentCodePoints {
            color_primaries: cp,
            transfer_function: tc,
            matrix_coefficients: mc,
            is_video_full_range_image: full,
        }
    }

    /// Issue #71 regression: the HDR-PNG corpus signature (sRGB
    /// primaries + PQ transfer) must map to a PQ color encoding —
    /// this exact combination was silently dropped to sRGB before.
    #[test]
    fn srgb_primaries_pq_transfer_maps() {
        let ce = color_encoding_from_cicp(cicp(1, 16, 0, true), false).unwrap();
        assert_eq!(ce.transfer_function, TransferFunction::Pq);
        assert_eq!(ce.primaries, Primaries::Srgb);
    }

    #[test]
    fn p3_and_bt2100_and_hlg_map() {
        let ce = color_encoding_from_cicp(cicp(12, 16, 0, true), false).unwrap();
        assert_eq!(ce.primaries, Primaries::P3);
        assert_eq!(ce.transfer_function, TransferFunction::Pq);
        let ce = color_encoding_from_cicp(cicp(9, 18, 0, true), false).unwrap();
        assert_eq!(ce.primaries, Primaries::Bt2100);
        assert_eq!(ce.transfer_function, TransferFunction::Hlg);
    }

    /// Unsupported combos fall back (caller then uses gAMA/sRGB):
    /// YCbCr matrix, limited range, exotic primaries.
    #[test]
    fn unsupported_combos_fall_back() {
        assert!(color_encoding_from_cicp(cicp(1, 16, 1, true), false).is_none());
        assert!(color_encoding_from_cicp(cicp(1, 16, 0, false), false).is_none());
        assert!(color_encoding_from_cicp(cicp(4, 16, 0, true), false).is_none());
    }

    #[test]
    fn grayscale_coerces_color_space() {
        let ce = color_encoding_from_cicp(cicp(1, 16, 0, true), true).unwrap();
        assert_eq!(ce.color_space, ColorSpace::Gray);
    }
}
