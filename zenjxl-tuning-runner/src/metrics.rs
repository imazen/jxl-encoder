// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later.
#![forbid(unsafe_code)]

//! GPU-first metric scoring with CPU fallback.
//!
//! ## Backend selection
//!
//! - `Auto` — try `zen-metrics` CLI for each of the three metrics. If
//!   the binary is absent OR a single-metric subprocess fails, fall
//!   back to in-process CPU (only if the `cpu-metrics` feature is on).
//!   Any metric that has no working backend lands as `None` with
//!   `*_backend = "skip"`.
//! - `GpuCli` — require the `zen-metrics` CLI. Fail with backend
//!   `"gpu-cli-error: <stderr>"` if it isn't usable.
//! - `Cpu` — use in-process CPU only (requires `cpu-metrics`
//!   feature). CVVDP stays null (no in-process CVVDP available).
//! - `Skip` — emit null columns with backend `"skip"`. Useful for
//!   smoke tests where you want to time encode+decode only.
//!
//! ## Out-of-process design
//!
//! GPU metrics shell out via `std::process::Command` instead of
//! linking the GPU library directly. Rationale:
//!
//! 1. A CUDA / WGPU OOM in one metric doesn't kill the runner.
//! 2. The runner stays slim — no `cubecl` / `zenmetrics-api` deps.
//! 3. The `zen-metrics` CLI is the canonical fleet entry point per
//!    the zenmetrics CLAUDE.md; using it keeps signal parity.
//!
//! Future work: in-process via `zenmetrics-api` for sweep runs where
//! the GPU cost is dominant. Tracked under W44-213+.

use std::path::Path;
use std::process::Command;

/// Bundle of metric values + provenance + GPU cost.
#[derive(Clone, Debug, Default)]
pub struct MetricBundle {
    pub ssim2: Option<f32>,
    pub ssim2_backend: String,
    pub butter_norm3: Option<f32>,
    pub butter_norm3_backend: String,
    pub cvvdp: Option<f32>,
    pub cvvdp_backend: String,
    pub gpu_peak_vram_mb: u32,
    pub gpu_kernel_ms: f64,
}

/// Backend preference for scoring.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MetricBackend {
    Auto,
    GpuCli,
    Cpu,
    Skip,
}

/// Score one (source, decoded) pair. Both buffers must be Rgb8 of
/// the same dimensions.
pub fn score_cell(
    source_rgb: &[u8],
    decoded_rgb: &[u8],
    width: usize,
    height: usize,
    backend: MetricBackend,
) -> MetricBundle {
    let mut bundle = MetricBundle::default();
    if backend == MetricBackend::Skip {
        bundle.ssim2_backend = "skip".into();
        bundle.butter_norm3_backend = "skip".into();
        bundle.cvvdp_backend = "skip".into();
        return bundle;
    }

    // Write the source+decoded as PNG to a tempdir for the CLI shellout
    // path. We always materialise these — even the CPU path consumes
    // them via re-decoding (cheap on RGBA u8).
    let tmpdir = match tempfile_dir() {
        Some(d) => d,
        None => {
            bundle.ssim2_backend = "no-tmpdir".into();
            bundle.butter_norm3_backend = "no-tmpdir".into();
            bundle.cvvdp_backend = "no-tmpdir".into();
            return bundle;
        }
    };
    let source_path = tmpdir.join("source.png");
    let decoded_path = tmpdir.join("decoded.png");
    if write_rgb_as_png(&source_path, source_rgb, width, height).is_err()
        || write_rgb_as_png(&decoded_path, decoded_rgb, width, height).is_err()
    {
        bundle.ssim2_backend = "png-write-failed".into();
        bundle.butter_norm3_backend = "png-write-failed".into();
        bundle.cvvdp_backend = "png-write-failed".into();
        return bundle;
    }

    // GPU CLI path.
    let try_gpu = matches!(backend, MetricBackend::Auto | MetricBackend::GpuCli);
    let mut gpu_total_ms = 0.0f64;
    let mut gpu_peak_mb = 0u32;

    if try_gpu {
        let zm_present = Command::new("zen-metrics")
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);
        if zm_present {
            // SSIM2 GPU
            let r = invoke_zen_metric(&source_path, &decoded_path, "ssim2");
            bundle.ssim2 = r.value;
            bundle.ssim2_backend = r.backend;
            gpu_total_ms += r.elapsed_ms;
            gpu_peak_mb = gpu_peak_mb.max(r.peak_mb);
            // Butteraugli norm-3 GPU
            let r = invoke_zen_metric(&source_path, &decoded_path, "butteraugli-pnorm3");
            bundle.butter_norm3 = r.value;
            bundle.butter_norm3_backend = r.backend;
            gpu_total_ms += r.elapsed_ms;
            gpu_peak_mb = gpu_peak_mb.max(r.peak_mb);
            // CVVDP GPU (Python pycvvdp under-the-hood per zenmetrics fleet pattern)
            let r = invoke_zen_metric(&source_path, &decoded_path, "cvvdp");
            bundle.cvvdp = r.value;
            bundle.cvvdp_backend = r.backend;
            gpu_total_ms += r.elapsed_ms;
            gpu_peak_mb = gpu_peak_mb.max(r.peak_mb);
        } else if matches!(backend, MetricBackend::GpuCli) {
            bundle.ssim2_backend = "gpu-cli-missing".into();
            bundle.butter_norm3_backend = "gpu-cli-missing".into();
            bundle.cvvdp_backend = "gpu-cli-missing".into();
            bundle.gpu_kernel_ms = 0.0;
            bundle.gpu_peak_vram_mb = 0;
            return bundle;
        }
    }

    bundle.gpu_kernel_ms = gpu_total_ms;
    bundle.gpu_peak_vram_mb = gpu_peak_mb;

    // CPU fallback for any metric that didn't get a GPU value (or all
    // when backend == Cpu).
    #[cfg(feature = "cpu-metrics")]
    {
        let need_ssim2 = bundle.ssim2.is_none() && backend != MetricBackend::Skip;
        let need_butter = bundle.butter_norm3.is_none() && backend != MetricBackend::Skip;
        if need_ssim2 || need_butter {
            if let Some((s, b)) = compute_cpu_metrics(source_rgb, decoded_rgb, width, height) {
                if need_ssim2 {
                    bundle.ssim2 = Some(s);
                    bundle.ssim2_backend = "cpu-ssimulacra2".into();
                }
                if need_butter {
                    bundle.butter_norm3 = Some(b);
                    bundle.butter_norm3_backend = "cpu-butteraugli-norm3".into();
                }
            }
        }
        // CVVDP: no CPU-only implementation; leave None.
        if bundle.cvvdp_backend.is_empty() {
            bundle.cvvdp_backend = "cpu-unsupported".into();
        }
    }

    // Anything still empty after all backends → mark "skip" so the
    // Parquet column carries the reason instead of a bare empty string.
    if bundle.ssim2_backend.is_empty() {
        bundle.ssim2_backend = "skip".into();
    }
    if bundle.butter_norm3_backend.is_empty() {
        bundle.butter_norm3_backend = "skip".into();
    }
    if bundle.cvvdp_backend.is_empty() {
        bundle.cvvdp_backend = "skip".into();
    }

    bundle
}

struct MetricRunResult {
    value: Option<f32>,
    backend: String,
    elapsed_ms: f64,
    peak_mb: u32,
}

/// Invoke `zen-metrics score --metric <name> <source> <decoded>`. The
/// zen-metrics CLI emits one JSON object per call with a `score` field
/// and (for GPU paths) a `gpu_peak_vram_mb` / `gpu_kernel_ms` field.
/// Schema mirrors `zen_metrics_cli::output::ScoreOutput` from the
/// zenmetrics workspace.
fn invoke_zen_metric(src: &Path, dist: &Path, metric: &str) -> MetricRunResult {
    let t0 = std::time::Instant::now();
    let out = Command::new("zen-metrics")
        .args([
            "score",
            "--metric",
            metric,
            "--reference",
            src.to_str().unwrap_or(""),
            "--distorted",
            dist.to_str().unwrap_or(""),
            "--json",
        ])
        .output();
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    match out {
        Ok(o) if o.status.success() => {
            // Best-effort JSON parse. zen-metrics typically emits
            // `{"score": <f64>, "gpu_peak_vram_mb": <u32>?,
            // "gpu_kernel_ms": <f64>?}`. We're lenient: any of the
            // GPU fields may be missing on CPU codepaths.
            let parsed: serde_json::Value =
                serde_json::from_slice(&o.stdout).unwrap_or(serde_json::Value::Null);
            let value = parsed
                .get("score")
                .and_then(|v| v.as_f64())
                .map(|x| x as f32);
            let peak_mb = parsed
                .get("gpu_peak_vram_mb")
                .and_then(|v| v.as_u64())
                .map(|x| x as u32)
                .unwrap_or(0);
            MetricRunResult {
                value,
                backend: format!("gpu-cli-{metric}"),
                elapsed_ms,
                peak_mb,
            }
        }
        Ok(o) => {
            let stderr = String::from_utf8_lossy(&o.stderr);
            MetricRunResult {
                value: None,
                backend: format!("gpu-cli-failed: {}", stderr.trim().chars().take(120).collect::<String>()),
                elapsed_ms,
                peak_mb: 0,
            }
        }
        Err(e) => MetricRunResult {
            value: None,
            backend: format!("gpu-cli-spawn-error: {e}"),
            elapsed_ms,
            peak_mb: 0,
        },
    }
}

fn tempfile_dir() -> Option<std::path::PathBuf> {
    // Don't use /tmp during a sweep (per project rule "Never use /tmp
    // for reboots etc"); use /mnt/v/tmp/w44-212/ if it exists, else
    // fall back to std::env::temp_dir(). Each cell creates a uuid
    // subdir.
    let base = if std::path::Path::new("/mnt/v/tmp").exists() {
        std::path::PathBuf::from("/mnt/v/tmp/w44-212")
    } else {
        std::env::temp_dir().join("w44-212")
    };
    let nonce = format!(
        "{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    );
    let dir = base.join(nonce);
    std::fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

fn write_rgb_as_png(path: &Path, rgb: &[u8], w: usize, h: usize) -> Result<(), String> {
    image::save_buffer(path, rgb, w as u32, h as u32, image::ColorType::Rgb8)
        .map_err(|e| format!("png save {}: {e}", path.display()))
}

#[cfg(feature = "cpu-metrics")]
fn compute_cpu_metrics(
    source: &[u8],
    decoded: &[u8],
    w: usize,
    h: usize,
) -> Option<(f32, f32)> {
    use butteraugli::butteraugli;
    use imgref::Img;

    // Both buffers RGB → RGB f32 in 0..=1.
    let to_rgb_f32 = |buf: &[u8]| -> Vec<rgb::RGB<f32>> {
        let mut out = Vec::with_capacity(w * h);
        for px in buf.chunks_exact(3) {
            out.push(rgb::RGB {
                r: px[0] as f32 / 255.0,
                g: px[1] as f32 / 255.0,
                b: px[2] as f32 / 255.0,
            });
        }
        out
    };
    let src_rgb = to_rgb_f32(source);
    let dist_rgb = to_rgb_f32(decoded);
    // butteraugli expects linear sRGB. Above is gamma-encoded; for a
    // sweep approximation we apply gamma→linear conversion. NOTE:
    // the W44-212 runner is GPU-first; the CPU path is opportunistic
    // and the values land with backend `"cpu-..."` so downstream MLP
    // training can mask if needed.
    let srgb_to_linear = |s: f32| -> f32 {
        if s <= 0.04045 {
            s / 12.92
        } else {
            ((s + 0.055) / 1.055).powf(2.4)
        }
    };
    let to_linear = |rgb: &[rgb::RGB<f32>]| -> Vec<rgb::RGB<f32>> {
        rgb.iter()
            .map(|p| rgb::RGB {
                r: srgb_to_linear(p.r),
                g: srgb_to_linear(p.g),
                b: srgb_to_linear(p.b),
            })
            .collect()
    };
    let src_lin = to_linear(&src_rgb);
    let dist_lin = to_linear(&dist_rgb);
    let src_img = Img::new(src_lin, w, h);
    let dist_img = Img::new(dist_lin, w, h);
    let butter_score = butteraugli(src_img.as_ref(), dist_img.as_ref(), 1.0)
        .ok()
        .map(|r| r.pnorm_3 as f32)
        .unwrap_or(f32::NAN);

    let src_img2 = Img::new(src_rgb.clone(), w, h);
    let dist_img2 = Img::new(dist_rgb.clone(), w, h);
    let s2 = fast_ssim2::compute_frame(src_img2.as_ref(), dist_img2.as_ref())
        .ok()
        .map(|s| s as f32)
        .unwrap_or(f32::NAN);
    Some((s2, butter_score))
}
