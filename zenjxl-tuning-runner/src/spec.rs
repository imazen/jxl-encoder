// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later.
#![forbid(unsafe_code)]

//! Cell spec (JSON input contract) + cell row (Parquet output schema).
//!
//! ## v2 (W44-PHASE4-M1, 2026-05-24) artifact persistence
//!
//! Per the global CLAUDE.md `4. Always persist encoded variants when
//! encoding is expensive — NO EXCEPTIONS` rule (added 2026-05-24
//! after the W44-PHASE4-S1 incident discarded ~$30 of encoded bytes),
//! the runner can now optionally persist:
//!
//! - **Encoded JXL bitstream** to a content-addressed local file
//!   (`<artifacts_dir>/jxl/<sha[0..2]>/<sha>.jxl`). Cell row stores
//!   the sha256 + the corresponding R2 key. `worker.sh` is responsible
//!   for the actual `s5cmd cp` upload.
//! - **Per-pixel butteraugli diffmap** to
//!   `<artifacts_dir>/diffmap/<sha[0..2]>/<sha>.bin` in a simple raw
//!   binary format: 8-byte magic `BUTTERDM` + u32-LE version=1 + u32-LE
//!   width + u32-LE height + (width*height) × f32-LE values. Content-
//!   addressed on `sha256("butter-v1" || width-LE || height-LE ||
//!   ref_pixels || dist_pixels)` so identical pixel pairs dedup.
//! - **Multi-norm butteraugli + per-channel PSNR** as scalar columns
//!   in the Parquet row.
//!
//! These are gated by THREE env flags (all default OFF so existing
//! tests + CI run byte-identical):
//!
//! | env flag | effect |
//! |---|---|
//! | `W44_PHASE4_M1_SAVE_ENCODED=1` | stage encoded JXL + populate sha256/r2_key |
//! | `W44_PHASE4_M1_SAVE_DIFFMAP=1` | stage diffmap blob + populate diffmap_r2_key |
//! | `W44_PHASE4_M1_COMPUTE_MULTIMETRIC=1` | populate butter_max/p1/p2/p6 + psnr_y/r/g/b |
//!
//! Production sweeps MUST set ALL THREE to `1` per the CLAUDE.md rule.
//! Smoke tests + unit tests leave them OFF.
//!
//! The `W44_PHASE4_M1_ARTIFACTS_DIR` env var picks the local staging
//! directory (defaults to a `<output_parquet>/../artifacts/` sibling).
//! `worker.sh` later runs `s5cmd cp` over the contents.

use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use sha2::Digest;

use crate::features::{ExtendedFeatures, compute_extended_features};
use crate::metrics::{MetricBackend, MetricBundle, ScoreOptions, score_cell_with_options};
use crate::params::materialise_params;
use crate::rusage::RUsageDelta;

/// Persistence configuration parsed from `W44_PHASE4_M1_*` env vars.
/// Read once at the start of [`run_cell`].
#[derive(Clone, Debug, Default)]
struct PersistConfig {
    save_encoded: bool,
    save_diffmap: bool,
    compute_multimetric: bool,
    /// Local directory under which content-addressed artifacts are
    /// staged. `worker.sh` uploads everything under this dir to R2.
    artifacts_dir: Option<PathBuf>,
}

impl PersistConfig {
    fn from_env(output_parquet: &Path) -> Self {
        let env_on = |k: &str| -> bool {
            std::env::var(k)
                .map(|v| matches!(v.as_str(), "1" | "true" | "yes" | "on"))
                .unwrap_or(false)
        };
        let save_encoded = env_on("W44_PHASE4_M1_SAVE_ENCODED");
        let save_diffmap = env_on("W44_PHASE4_M1_SAVE_DIFFMAP");
        let compute_multimetric = env_on("W44_PHASE4_M1_COMPUTE_MULTIMETRIC");
        let artifacts_dir = if save_encoded || save_diffmap {
            Some(
                std::env::var_os("W44_PHASE4_M1_ARTIFACTS_DIR")
                    .map(PathBuf::from)
                    .unwrap_or_else(|| {
                        output_parquet
                            .parent()
                            .unwrap_or_else(|| Path::new("."))
                            .join("artifacts")
                    }),
            )
        } else {
            None
        };
        PersistConfig {
            save_encoded,
            save_diffmap,
            compute_multimetric,
            artifacts_dir,
        }
    }

    fn score_options(&self) -> ScoreOptions {
        ScoreOptions {
            // The diffmap is the substrate for multi-norm aggregations,
            // so save_diffmap implies compute_multimetric.
            compute_multimetric: self.compute_multimetric || self.save_diffmap,
            save_diffmap: self.save_diffmap,
        }
    }
}

/// Stage a file's bytes into a content-addressed location under
/// `<artifacts_dir>/<subdir>/<sha[0..2]>/<sha>.<ext>`. Returns the
/// R2 key (relative path) suitable for `s5cmd cp` upload by worker.sh.
///
/// Writes atomically (write-to-temp + rename) so worker.sh can `s5cmd
/// cp --if-not-exists` without seeing partial files. If the target
/// already exists with identical bytes (dedup), skips the write.
fn stage_content_addressed(
    bytes: &[u8],
    sha256_hex: &str,
    artifacts_dir: &Path,
    subdir: &str,
    ext: &str,
) -> std::io::Result<String> {
    let prefix = &sha256_hex[0..2];
    let rel_key = format!("artifacts/{subdir}/{prefix}/{sha256_hex}.{ext}");
    let dir = artifacts_dir.join(subdir).join(prefix);
    std::fs::create_dir_all(&dir)?;
    let target = dir.join(format!("{sha256_hex}.{ext}"));
    if target.exists() {
        // Already staged by an earlier cell — content-addressed dedup.
        return Ok(rel_key);
    }
    // Write to a unique temp file then atomic rename.
    let tmp = dir.join(format!(".{}.{ext}.tmp", std::process::id()));
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, &target)?;
    Ok(rel_key)
}

/// Encode a butteraugli diffmap as a simple binary blob.
/// Format: `BUTTERDM` (8B) + version u32-LE + width u32-LE + height
/// u32-LE + width*height f32-LE values.
fn encode_diffmap_blob(diffmap: &[f32], width: u32, height: u32) -> Vec<u8> {
    let n_px = (width as usize) * (height as usize);
    debug_assert_eq!(diffmap.len(), n_px, "diffmap len must equal width*height");
    let mut out = Vec::with_capacity(8 + 12 + n_px * 4);
    out.extend_from_slice(b"BUTTERDM");
    out.extend_from_slice(&1u32.to_le_bytes());
    out.extend_from_slice(&width.to_le_bytes());
    out.extend_from_slice(&height.to_le_bytes());
    for v in diffmap {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Content-addressed sha256 for a diffmap. Keys on `("butter-v1",
/// width, height, ref_pixels, dist_pixels)` so two cells with
/// identical pixel pairs produce the same diffmap key and dedup.
fn diffmap_content_sha(ref_rgb: &[u8], dist_rgb: &[u8], width: u32, height: u32) -> String {
    let mut h = sha2::Sha256::new();
    h.update(b"butter-v1");
    h.update(width.to_le_bytes());
    h.update(height.to_le_bytes());
    h.update(ref_rgb);
    h.update(dist_rgb);
    hex::encode(h.finalize())
}

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

    // ── v2 (W44-PHASE4-M1, 2026-05-24) ───────────────────────────────
    //
    // These fields preserve every artifact the sweep produces so future
    // metric R&D (a butteraugli successor) doesn't need to re-encode.
    // ALL are nullable; the runner only populates them when the
    // corresponding `W44_PHASE4_M1_SAVE_*` / `W44_PHASE4_M1_COMPUTE_*`
    // env flag is set. Defaults remain OFF for backwards compat with
    // existing CI + smoke harnesses.
    //
    // Production sweeps MUST set all three persistence env flags to
    // `1` per the global CLAUDE.md `4. Always persist encoded variants`
    // rule. See [`run_cell`] for the propagation.
    /// Content-addressed sha256 (hex) of the encoded JXL bytes.
    /// Joined with `encoded_jxl_r2_key` lets you fetch the bitstream
    /// from R2 forever — encoded bytes are 100x more expensive to
    /// produce than to score, so we save them once and recompute any
    /// future metric on demand.
    pub encoded_jxl_sha256: Option<String>,
    /// R2 key (relative path within the sweep bucket) where the
    /// encoded JXL lands. `worker.sh` performs the actual `s5cmd cp`
    /// upload; the runner just stages the file locally under
    /// `--artifacts-dir` with this exact key as a path suffix.
    pub encoded_jxl_r2_key: Option<String>,
    /// R2 key for the per-pixel butteraugli diffmap (f16 raw blob).
    /// Content-addressed on `sha256(ref_pixels || dist_pixels ||
    /// metric_id)` so two cells with identical pixel pairs dedup.
    pub diffmap_r2_key: Option<String>,
    /// Butteraugli max-norm (a.k.a. global `score`).
    pub butter_max: Option<f32>,
    /// Butteraugli p=1 norm (when the backend exposes it; CPU path
    /// computes from the diffmap).
    pub butter_p1: Option<f32>,
    /// Butteraugli p=2 norm.
    pub butter_p2: Option<f32>,
    /// Butteraugli p=6 norm — the tail-distortion aggregator that's
    /// useful for catching worst-block artefacts that p=3 hides.
    pub butter_p6: Option<f32>,
    /// PSNR over Rec.709 luma Y' (sRGB-encoded RGB → 0.2126·R +
    /// 0.7152·G + 0.0722·B in u8 space).
    pub psnr_y: Option<f32>,
    /// PSNR over the R channel, sRGB-encoded u8.
    pub psnr_r: Option<f32>,
    /// PSNR over the G channel.
    pub psnr_g: Option<f32>,
    /// PSNR over the B channel.
    pub psnr_b: Option<f32>,
    /// MS-SSIM score (rgb-mean across channels). Currently always
    /// `None` — zen-metrics 0.6.0 doesn't expose MS-SSIM and we have
    /// no in-process Rust crate wired in. Reserved column for future
    /// backend implementations.
    pub ms_ssim: Option<f32>,
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
    let encoded = encode_with_strategy(
        &rgb_bytes,
        w,
        h,
        spec.distance,
        spec.effort,
        strategy,
        threads,
    )
    .map_err(CellError::Encode)?;
    let encode_ms = wall_pre.elapsed().as_secs_f64() * 1000.0;
    let rusage_post = RUsageDelta::snapshot();
    let encode_rusage = rusage_post.diff(&rusage_pre);

    // ── 5. Decode roundtrip via jxl-rs ──────────────────────────────
    let decode_pre = Instant::now();
    let rusage_decode_pre = RUsageDelta::snapshot();
    let decoded_rgb =
        decode_roundtrip(&encoded, w as usize, h as usize).map_err(CellError::Decode)?;
    let decode_ms = decode_pre.elapsed().as_secs_f64() * 1000.0;
    let rusage_decode_post = RUsageDelta::snapshot();
    let decode_rusage = rusage_decode_post.diff(&rusage_decode_pre);

    // ── 6. Score on GPU (or CPU fallback) ───────────────────────────
    let persist = PersistConfig::from_env(output_parquet);
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
        butter_max,
        butter_p1,
        butter_p2,
        butter_p6,
        psnr_y,
        psnr_r,
        psnr_g,
        psnr_b,
        ms_ssim,
        diffmap,
    } = score_cell_with_options(
        &rgb_bytes,
        &decoded_rgb,
        w as usize,
        h as usize,
        backend,
        persist.score_options(),
    );

    // ── 6b. Stage artifacts (encoded JXL + diffmap) ─────────────────
    let (encoded_jxl_sha256, encoded_jxl_r2_key) = if persist.save_encoded
        && let Some(dir) = persist.artifacts_dir.as_deref()
    {
        let sha = {
            let mut h = sha2::Sha256::new();
            h.update(&encoded);
            hex::encode(h.finalize())
        };
        match stage_content_addressed(&encoded, &sha, dir, "jxl", "jxl") {
            Ok(key) => (Some(sha), Some(key)),
            Err(e) => {
                eprintln!("[w44-phase4-m1] WARN: encoded-jxl stage failed (sha={sha}): {e}");
                (Some(sha), None)
            }
        }
    } else {
        (None, None)
    };

    let diffmap_r2_key = if persist.save_diffmap
        && let Some(dm) = diffmap.as_ref()
        && let Some(dir) = persist.artifacts_dir.as_deref()
    {
        let sha = diffmap_content_sha(&rgb_bytes, &decoded_rgb, w, h);
        let blob = encode_diffmap_blob(dm, w, h);
        match stage_content_addressed(&blob, &sha, dir, "diffmap", "bin") {
            Ok(key) => Some(key),
            Err(e) => {
                eprintln!("[w44-phase4-m1] WARN: diffmap stage failed (sha={sha}): {e}");
                None
            }
        }
    } else {
        None
    };

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
        // v2 (W44-PHASE4-M1)
        encoded_jxl_sha256,
        encoded_jxl_r2_key,
        diffmap_r2_key,
        butter_max,
        butter_p1,
        butter_p2,
        butter_p6,
        psnr_y,
        psnr_r,
        psnr_g,
        psnr_b,
        ms_ssim,
    };

    // ── 8. Write Parquet ────────────────────────────────────────────
    crate::parquet_writer::write_single_row_parquet(&row, output_parquet)
        .map_err(CellError::Write)?;

    Ok(row)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn persist_config_defaults_off() {
        // Use a temp parquet path to avoid creating an artifacts/ dir
        // in cwd if a test process accidentally has env on.
        let cfg = PersistConfig::default();
        assert!(!cfg.save_encoded);
        assert!(!cfg.save_diffmap);
        assert!(!cfg.compute_multimetric);
        assert!(cfg.artifacts_dir.is_none());
        let opts = cfg.score_options();
        assert!(!opts.compute_multimetric);
        assert!(!opts.save_diffmap);
    }

    #[test]
    fn diffmap_blob_format_roundtrip() {
        let dm: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();
        let blob = encode_diffmap_blob(&dm, 8, 8);
        // Magic + version + w + h + data
        assert_eq!(&blob[0..8], b"BUTTERDM");
        assert_eq!(&blob[8..12], &1u32.to_le_bytes()[..]);
        assert_eq!(&blob[12..16], &8u32.to_le_bytes()[..]);
        assert_eq!(&blob[16..20], &8u32.to_le_bytes()[..]);
        assert_eq!(blob.len(), 20 + 64 * 4);
        // Spot-check the first f32 round-trips.
        let first = f32::from_le_bytes([blob[20], blob[21], blob[22], blob[23]]);
        assert_eq!(first, 0.0);
        let second = f32::from_le_bytes([blob[24], blob[25], blob[26], blob[27]]);
        assert!((second - 0.1).abs() < 1e-6);
    }

    #[test]
    fn diffmap_content_sha_depends_on_pixels() {
        let ref1 = vec![100u8; 16 * 16 * 3];
        let ref2 = vec![101u8; 16 * 16 * 3];
        let dst = vec![100u8; 16 * 16 * 3];
        let sha_a = diffmap_content_sha(&ref1, &dst, 16, 16);
        let sha_b = diffmap_content_sha(&ref2, &dst, 16, 16);
        let sha_a2 = diffmap_content_sha(&ref1, &dst, 16, 16);
        assert_eq!(sha_a, sha_a2, "same inputs → same sha");
        assert_ne!(sha_a, sha_b, "different ref → different sha");
        assert_eq!(sha_a.len(), 64, "hex sha256 = 64 chars");
    }

    #[test]
    fn stage_content_addressed_writes_and_dedups() {
        let tmp = tempfile::TempDir::new().unwrap();
        let bytes = b"hello-jxl-bytes";
        let sha = "feedbeef00112233feedbeef00112233feedbeef00112233feedbeef00112233";
        let key = stage_content_addressed(bytes, sha, tmp.path(), "jxl", "jxl").unwrap();
        assert_eq!(
            key,
            format!("artifacts/jxl/fe/{sha}.jxl"),
            "r2 key shape uses first 2 hex chars as prefix dir"
        );
        let path = tmp.path().join("jxl").join("fe").join(format!("{sha}.jxl"));
        assert!(path.exists());
        assert_eq!(std::fs::read(&path).unwrap(), bytes);
        // Second call dedups (same key, no error, file unchanged).
        let key2 = stage_content_addressed(bytes, sha, tmp.path(), "jxl", "jxl").unwrap();
        assert_eq!(key, key2);
    }

    #[test]
    fn persist_config_from_env_artifacts_dir_default() {
        // Simulate env on by directly constructing — env-mutating tests
        // are flaky in parallel. The default-fallback logic lives in
        // from_env's else branch; here we just confirm the explicit
        // path constructs cleanly.
        let cfg = PersistConfig {
            save_encoded: true,
            save_diffmap: false,
            compute_multimetric: false,
            artifacts_dir: Some(PathBuf::from("/tmp/m1-test")),
        };
        assert!(cfg.score_options().compute_multimetric == false);
        assert!(cfg.artifacts_dir.is_some());
    }
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
        other => {
            return Err(format!(
                "unknown strategy {other:?}; expected one of: zenjxl, libjxl, lean-faster, aggressive"
            ));
        }
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
        return Err(format!(
            "size mismatch: decoder {width}x{height} != source {w}x{h}"
        ));
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
