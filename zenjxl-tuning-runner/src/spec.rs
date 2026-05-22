// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later.
#![forbid(unsafe_code)]

//! Cell spec (JSON input contract) + cell row (Parquet output schema).

use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use sha2::Digest;

use crate::features::{ExtendedFeatures, compute_extended_features};
use crate::metrics::{MetricBackend, MetricBundle, score_cell};
use crate::params::materialise_params;
use crate::rusage::RUsageDelta;

/// Input contract for one sweep cell. Deserialised from `--cell <json>`
/// (one-line) or `--cell-file <path.json>`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SweepCellSpec {
    /// Sweep identifier (e.g. `W44-XYZ-buttloop-scan`). Lands in the
    /// `sweep_id` Parquet column; downstream uses it to filter rows.
    pub sweep_id: String,

    /// Worker-assigned claim id from the chunk queue (atomic R2 lock).
    /// Lands in the output filename + Parquet `chunk_claim_id` column.
    pub chunk_claim_id: String,

    /// Absolute path to the source PNG on the worker disk.
    pub image_path: PathBuf,

    /// Pre-computed sha256 of the source bytes (for join-on-source
    /// dedup downstream). If absent, the runner computes it from the
    /// loaded bytes.
    #[serde(default)]
    pub image_sha256: Option<String>,

    /// Encoder effort level (1..=12).
    pub effort: u8,

    /// Lossy target distance (Butteraugli units). `0.0` means lossless
    /// — not currently supported by the runner (would route through
    /// LosslessConfig); reject in [`run_cell`].
    pub distance: f32,

    /// Encoder strategy preset. One of: `"zenjxl"` (default),
    /// `"libjxl"`, `"lean-faster"`, `"aggressive"`.
    #[serde(default = "default_strategy")]
    pub strategy: String,

    /// Optional postcard-serialised
    /// [`jxl_encoder::tuning::runtime::RuntimeTuning`] override blob.
    /// If `None`, the encoder runs with all production-default consts.
    #[serde(default)]
    pub params_blob_path: Option<PathBuf>,

    /// Optional override for the number of encoder threads. Defaults
    /// to rayon's default (= num CPUs). Setting `Some(1)` is useful
    /// for fleet workers running 1 cell per process where the OS
    /// scheduler handles cell-level parallelism.
    #[serde(default)]
    pub threads: Option<usize>,

    /// Metric backend preference. `"auto"` (default) tries GPU CLI,
    /// falls back to CPU. `"gpu-cli"` requires the GPU CLI. `"cpu"`
    /// forces CPU. `"skip"` emits null columns with backend
    /// annotation.
    #[serde(default = "default_metric_backend")]
    pub metric_backend: String,
}

fn default_strategy() -> String {
    "zenjxl".to_string()
}
fn default_metric_backend() -> String {
    "auto".to_string()
}

/// Output row schema. One Parquet record per cell. ~40 columns.
///
/// Column groups (logical, not Parquet-physical):
///
/// - **Identity / provenance**: `sweep_id`, `chunk_claim_id`,
///   `image_sha256`, `image_path`, `image_w`, `image_h`,
///   `runner_host`, `gpu_model`, `commit_sha`
/// - **Inputs**: `effort`, `distance`, `strategy`, `params_blob`
/// - **Features**: ~14 zenanalyze-equivalent f32 columns
///   (`feat_mask_p25`, `feat_mask_median`, `feat_m3_colourfulness`,
///   `feat_fcbr`, `feat_edge_density`, `feat_luma_var`, etc.)
/// - **Output bytes**: `encoded_bytes`
/// - **Quality**: `ssim2`, `butter_norm3`, `cvvdp`, `*_backend`
/// - **CPU cost**: `encode_ms`, `encode_user_ms`, `encode_sys_ms`,
///   `encode_peak_rss_mb`, `encode_threads`, `decode_ms`,
///   `decode_peak_rss_mb`
/// - **GPU cost**: `gpu_peak_vram_mb`, `gpu_kernel_ms`
#[derive(Clone, Debug)]
pub struct SweepCellRow {
    // Identity
    pub sweep_id: String,
    pub chunk_claim_id: String,
    pub image_sha256: String,
    pub image_path: String,
    pub image_w: u32,
    pub image_h: u32,

    // Inputs
    pub effort: u8,
    pub distance: f32,
    pub strategy: String,
    pub params_blob: Vec<u8>,

    // Features (W44-91 / W44-96 / W44-164 + extended)
    pub features: ExtendedFeatures,

    // Output
    pub encoded_bytes: u32,

    // Quality (any may be None if the backend skipped or failed)
    pub ssim2: Option<f32>,
    pub ssim2_backend: String,
    pub butter_norm3: Option<f32>,
    pub butter_norm3_backend: String,
    pub cvvdp: Option<f32>,
    pub cvvdp_backend: String,

    // Cost
    pub encode_ms: f64,
    pub encode_user_ms: u64,
    pub encode_sys_ms: u64,
    pub encode_peak_rss_mb: u32,
    pub encode_threads: u8,
    pub decode_ms: f64,
    pub decode_peak_rss_mb: u32,
    pub gpu_peak_vram_mb: u32,
    pub gpu_kernel_ms: f64,

    // Provenance
    pub runner_host: String,
    pub gpu_model: String,
    pub commit_sha: String,
    pub runner_version: String,
}

/// Errors that abort a single cell. The fleet worker treats every
/// variant as "skip this cell, log, move on" — no panic.
#[derive(Debug)]
pub enum CellError {
    /// PNG load / format failed.
    LoadImage(String),
    /// Cell spec violated runtime invariants (e.g. distance ≤ 0).
    InvalidSpec(String),
    /// Encoder returned an error.
    Encode(String),
    /// jxl-rs decode failed on our own output (treat as worker bug —
    /// always means we wrote a bitstream that the primary decoder
    /// can't parse).
    Decode(String),
    /// Parquet write failed.
    Write(String),
}

impl std::fmt::Display for CellError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CellError::LoadImage(s) => write!(f, "load_image: {s}"),
            CellError::InvalidSpec(s) => write!(f, "invalid_spec: {s}"),
            CellError::Encode(s) => write!(f, "encode: {s}"),
            CellError::Decode(s) => write!(f, "decode: {s}"),
            CellError::Write(s) => write!(f, "write: {s}"),
        }
    }
}

impl std::error::Error for CellError {}

/// Run one cell end-to-end. See [`crate`] module docs for the 7-step
/// pipeline.
pub fn run_cell(spec: &SweepCellSpec, output_parquet: &Path) -> Result<SweepCellRow, CellError> {
    // ── 0. Validate ─────────────────────────────────────────────────
    if spec.distance <= 0.0 {
        return Err(CellError::InvalidSpec(format!(
            "distance must be > 0 (got {}); use a separate lossless runner",
            spec.distance
        )));
    }
    if !(1..=12).contains(&spec.effort) {
        return Err(CellError::InvalidSpec(format!(
            "effort must be in 1..=12 (got {})",
            spec.effort
        )));
    }

    // ── 1. Load source ──────────────────────────────────────────────
    //
    // Always strip alpha to RGB8: the canonical sweep grid for
    // VarDCT-tuning rows is opaque images (alpha is a separate
    // modular path with its own tunable; out of scope for W44-212).
    // PNG decoders return RGBA when the source has alpha; we drop
    // the A channel here. The decoder roundtrip mirrors this — we
    // request RGB output from jxl-rs, so dropping alpha at ingest
    // keeps the source/decoded shapes identical for scoring.
    let img = image::open(&spec.image_path)
        .map_err(|e| CellError::LoadImage(format!("{}: {e}", spec.image_path.display())))?;
    let img_rgb = img.into_rgb8();
    let (w, h) = (img_rgb.width(), img_rgb.height());
    let rgb_bytes: Vec<u8> = img_rgb.into_raw();

    // Compute source sha256 from RGB pixel bytes (NOT the PNG file
    // bytes, which differ across re-encodes of the same pixels).
    let image_sha256 = spec.image_sha256.clone().unwrap_or_else(|| {
        let mut h = sha2::Sha256::new();
        h.update(&rgb_bytes);
        hex::encode(h.finalize())
    });

    // ── 2. Compute features (extended ZenanalyzeProxies) ────────────
    let features = compute_extended_features(&rgb_bytes, w as usize, h as usize, 3, 0, 1, 2);

    // ── 3. Materialise params (postcard → RuntimeTuning) ────────────
    let params = materialise_params(spec.params_blob_path.as_deref())
        .map_err(|e| CellError::InvalidSpec(format!("params_blob: {e}")))?;

    // ── 4. Encode + rusage timing ───────────────────────────────────
    let threads = spec.threads.unwrap_or(0);
    let strategy = parse_strategy(&spec.strategy)
        .map_err(|e| CellError::InvalidSpec(format!("strategy: {e}")))?;

    let rusage_pre = RUsageDelta::snapshot();
    let wall_pre = Instant::now();
    let encoded =
        encode_with_strategy(&rgb_bytes, w, h, spec.distance, spec.effort, strategy, threads)
            .map_err(CellError::Encode)?;
    let encode_ms = wall_pre.elapsed().as_secs_f64() * 1000.0;
    let rusage_post = RUsageDelta::snapshot();
    let encode_rusage = rusage_post.diff(&rusage_pre);

    // ── 5. Decode roundtrip via jxl-rs ──────────────────────────────
    let decode_pre = Instant::now();
    let rusage_decode_pre = RUsageDelta::snapshot();
    let decoded_rgb = decode_roundtrip(&encoded, w as usize, h as usize)
        .map_err(CellError::Decode)?;
    let decode_ms = decode_pre.elapsed().as_secs_f64() * 1000.0;
    let rusage_decode_post = RUsageDelta::snapshot();
    let decode_rusage = rusage_decode_post.diff(&rusage_decode_pre);

    // ── 6. Score on GPU (or CPU fallback) ───────────────────────────
    let backend = parse_metric_backend(&spec.metric_backend);
    let MetricBundle {
        ssim2,
        ssim2_backend,
        butter_norm3,
        butter_norm3_backend,
        cvvdp,
        cvvdp_backend,
        gpu_peak_vram_mb,
        gpu_kernel_ms,
    } = score_cell(
        &rgb_bytes,
        &decoded_rgb,
        w as usize,
        h as usize,
        backend,
    );

    // ── 7. Build row ────────────────────────────────────────────────
    let row = SweepCellRow {
        sweep_id: spec.sweep_id.clone(),
        chunk_claim_id: spec.chunk_claim_id.clone(),
        image_sha256,
        image_path: spec.image_path.display().to_string(),
        image_w: w,
        image_h: h,
        effort: spec.effort,
        distance: spec.distance,
        strategy: spec.strategy.clone(),
        params_blob: params.blob.clone(),
        features,
        encoded_bytes: encoded.len() as u32,
        ssim2,
        ssim2_backend,
        butter_norm3,
        butter_norm3_backend,
        cvvdp,
        cvvdp_backend,
        encode_ms,
        encode_user_ms: encode_rusage.user_ms,
        encode_sys_ms: encode_rusage.sys_ms,
        encode_peak_rss_mb: encode_rusage.peak_rss_mb,
        encode_threads: rayon_threads() as u8,
        decode_ms,
        decode_peak_rss_mb: decode_rusage.peak_rss_mb,
        gpu_peak_vram_mb,
        gpu_kernel_ms,
        runner_host: hostname(),
        gpu_model: gpu_model(),
        commit_sha: commit_sha(),
        runner_version: env!("CARGO_PKG_VERSION").to_string(),
    };

    // ── 8. Write Parquet ────────────────────────────────────────────
    crate::parquet_writer::write_single_row_parquet(&row, output_parquet)
        .map_err(CellError::Write)?;

    Ok(row)
}

/// Parse the strategy string. Mirrors the `cjxl-rs` CLI `--strategy`
/// flag.
fn parse_strategy(s: &str) -> Result<jxl_encoder::api::EncoderStrategy, String> {
    use jxl_encoder::api::EncoderStrategy;
    Ok(match s.to_ascii_lowercase().as_str() {
        "" | "zenjxl" => EncoderStrategy::Zenjxl,
        "libjxl" => EncoderStrategy::Libjxl,
        "lean-faster" | "leanfaster" | "lean_faster" => EncoderStrategy::LeanFaster,
        "aggressive" => EncoderStrategy::Aggressive,
        other => return Err(format!("unknown strategy {other:?}; expected one of: zenjxl, libjxl, lean-faster, aggressive")),
    })
}

fn parse_metric_backend(s: &str) -> MetricBackend {
    match s.to_ascii_lowercase().as_str() {
        "auto" => MetricBackend::Auto,
        "gpu-cli" | "gpu" => MetricBackend::GpuCli,
        "cpu" => MetricBackend::Cpu,
        "skip" | "" => MetricBackend::Skip,
        _ => MetricBackend::Auto,
    }
}

fn encode_with_strategy(
    rgb: &[u8],
    w: u32,
    h: u32,
    distance: f32,
    effort: u8,
    strategy: jxl_encoder::api::EncoderStrategy,
    threads: usize,
) -> Result<Vec<u8>, String> {
    use jxl_encoder::{LossyConfig, PixelLayout};
    let cfg = LossyConfig::new(distance)
        .with_effort(effort)
        .with_threads(threads)
        .with_strategy(strategy);
    cfg.encode(rgb, w, h, PixelLayout::Rgb8)
        .map_err(|e| format!("{e}"))
}

/// Decode `bytes` via jxl-rs into the decoder's native sRGB output.
/// Returns `(rgb_bytes, channels)`. Mirrors the pattern in
/// `jxl-encoder/tests/w44_78_decoder_roundtrip.rs:48-108`. We request
/// f32 output then quantise to u8 in the assumed-sRGB space (jxl-rs
/// returns values already in the output color profile, which defaults
/// to the file's signaled encoding = sRGB for our encoder).
fn decode_roundtrip(bytes: &[u8], w: usize, h: usize) -> Result<Vec<u8>, String> {
    use jxl::api::{
        JxlColorType, JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer,
        JxlPixelFormat, ProcessingResult, states,
    };
    use jxl::image::{Image, Rect};

    let mut input = bytes;
    let options = JxlDecoderOptions::default();
    let decoder = JxlDecoder::<states::Initialized>::new(options);

    let mut decoder_init = decoder;
    let mut decoder = loop {
        match decoder_init.process(&mut input) {
            Ok(ProcessingResult::Complete { result }) => break result,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                decoder_init = fallback;
            }
            Err(e) => return Err(format!("jxl-rs header: {e:?}")),
        }
    };
    let basic_info = decoder.basic_info().clone();
    let (width, height) = basic_info.size;
    if width != w || height != h {
        return Err(format!("size mismatch: decoder {width}x{height} != source {w}x{h}"));
    }
    let channels = 3usize;
    decoder.set_pixel_format(JxlPixelFormat {
        color_type: JxlColorType::Rgb,
        color_data_format: Some(JxlDataFormat::f32()),
        extra_channel_format: vec![],
    });

    let mut decoder_frame = loop {
        match decoder.process(&mut input) {
            Ok(ProcessingResult::Complete { result }) => break result,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                decoder = fallback;
            }
            Err(e) => return Err(format!("jxl-rs frame info: {e:?}")),
        }
    };
    let mut output_image = Image::<f32>::new((width * channels, height))
        .map_err(|e| format!("output alloc: {e:?}"))?;
    let mut buffers = vec![JxlOutputBuffer::from_image_rect_mut(
        output_image
            .get_rect_mut(Rect {
                origin: (0, 0),
                size: (width * channels, height),
            })
            .into_raw(),
    )];
    let _ = loop {
        match decoder_frame.process(&mut input, &mut buffers) {
            Ok(ProcessingResult::Complete { result }) => break result,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                decoder_frame = fallback;
            }
            Err(e) => return Err(format!("jxl-rs frame: {e:?}")),
        }
    };
    // Pull bytes back from the Image<f32> via rect read. The image
    // stride is `width * channels`. We quantise to u8 (assume the
    // jxl-rs output is in the sRGB profile signaled by our encoder,
    // values nominally in [0.0, 1.0] BUT jxl-rs returns sample values
    // matching the basic_info bit depth so multiply by 255 then
    // quantise. For the runner the decoded RGBA mirrors the source
    // shape; alpha is filled opaque since we strip-decode RGB only.
    let mut rgb = vec![0u8; w * h * 3];
    for y in 0..h {
        let row = output_image.row(y);
        for x in 0..w {
            let r = row[x * channels];
            let g = row[x * channels + 1];
            let b = row[x * channels + 2];
            let off = (y * w + x) * 3;
            rgb[off] = quantise_srgb_u8(r);
            rgb[off + 1] = quantise_srgb_u8(g);
            rgb[off + 2] = quantise_srgb_u8(b);
        }
    }
    Ok(rgb)
}

#[inline]
fn quantise_srgb_u8(x: f32) -> u8 {
    (x.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn rayon_threads() -> usize {
    // Best-effort: rayon's global pool reflects what the encoder used.
    // Falls back to 1 if rayon isn't initialised.
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

fn hostname() -> String {
    std::env::var("HOSTNAME")
        .or_else(|_| std::env::var("HOST"))
        .unwrap_or_else(|_| "unknown".to_string())
}

fn gpu_model() -> String {
    // Best-effort: read CUDA_VISIBLE_DEVICES or nvidia-smi if present.
    // The fleet onstart script will populate $W44_212_GPU_MODEL.
    std::env::var("W44_212_GPU_MODEL").unwrap_or_else(|_| "unknown".to_string())
}

fn commit_sha() -> String {
    // Build-time: prefer GIT_COMMIT env var (set by Dockerfile build);
    // fall back to a static "unknown" marker.
    option_env!("GIT_COMMIT").unwrap_or("unknown").to_string()
}
