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
///
/// The v1 scalars (`ssim2` / `butter_norm3` / `cvvdp`) are always
/// computed (or annotated as `skip`/`failed`). The v2 multimetric
/// fields (`butter_max` / `butter_p1` / `butter_p2` / `butter_p6` /
/// `psnr_*` / `ms_ssim` / `diffmap`) are populated only when
/// `W44_PHASE4_M1_COMPUTE_MULTIMETRIC=1` is set AND the active backend
/// can produce them. PSNR is computed unconditionally when the multi-
/// metric env flag is on (cheap loop over u8 pixels). The diffmap
/// requires either the GPU backend's diffmap export or the CPU
/// butteraugli path with `compute_diffmap=true`.
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
    // ── v2 multimetric (W44-PHASE4-M1) ───────────────────────────────
    pub butter_max: Option<f32>,
    pub butter_p1: Option<f32>,
    pub butter_p2: Option<f32>,
    pub butter_p6: Option<f32>,
    pub psnr_y: Option<f32>,
    pub psnr_r: Option<f32>,
    pub psnr_g: Option<f32>,
    pub psnr_b: Option<f32>,
    pub ms_ssim: Option<f32>,
    /// Per-pixel butteraugli diffmap (rgb-aggregated f32 per pixel),
    /// row-major `width × height`. Populated only when the active
    /// backend can produce it AND `save_diffmap` was requested.
    pub diffmap: Option<Vec<f32>>,
}

/// Backend preference for scoring.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MetricBackend {
    Auto,
    GpuCli,
    Cpu,
    Skip,
}

/// Options for `score_cell` requesting v2 (W44-PHASE4-M1) extras.
///
/// All fields default to `false` so the production-default sweep
/// matches v1 behaviour byte-for-byte until the operator sets the
/// corresponding `W44_PHASE4_M1_*` env flag in `main.rs`.
#[derive(Clone, Copy, Debug, Default)]
pub struct ScoreOptions {
    /// Compute butteraugli max + p1/p2/p6 norms + PSNR per channel.
    /// MS-SSIM is reserved for a future backend; this flag does not
    /// produce it today.
    pub compute_multimetric: bool,
    /// Populate `MetricBundle::diffmap` with the per-pixel butteraugli
    /// diffmap. Requires `compute_multimetric=true` (the diffmap is
    /// the substrate for the multi-norm reductions). On the GPU CLI
    /// backend this is currently a no-op (zen-metrics 0.6.0 doesn't
    /// expose the diffmap); the CPU path produces it directly.
    pub save_diffmap: bool,
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
    score_cell_with_options(
        source_rgb,
        decoded_rgb,
        width,
        height,
        backend,
        ScoreOptions::default(),
    )
}

/// Score one cell with v2 extras controlled by [`ScoreOptions`].
pub fn score_cell_with_options(
    source_rgb: &[u8],
    decoded_rgb: &[u8],
    width: usize,
    height: usize,
    backend: MetricBackend,
    options: ScoreOptions,
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
            // SSIM2 GPU — score field is `scores.ssim2_gpu`
            let r = invoke_zen_metric(&source_path, &decoded_path, "ssim2-gpu", "ssim2_gpu");
            bundle.ssim2 = r.value;
            bundle.ssim2_backend = r.backend;
            gpu_total_ms += r.elapsed_ms;
            gpu_peak_mb = gpu_peak_mb.max(r.peak_mb);
            // Butteraugli norm-3 GPU — score field is `scores.butteraugli_pnorm3_gpu`
            let r = invoke_zen_metric(
                &source_path,
                &decoded_path,
                "butteraugli-gpu",
                "butteraugli_pnorm3_gpu",
            );
            bundle.butter_norm3 = r.value;
            bundle.butter_norm3_backend = r.backend;
            gpu_total_ms += r.elapsed_ms;
            gpu_peak_mb = gpu_peak_mb.max(r.peak_mb);
            // CVVDP — score field is `scores.cvvdp_imazen_v0_0_1`
            let r = invoke_zen_metric(&source_path, &decoded_path, "cvvdp", "cvvdp_imazen_v0_0_1");
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
    // when backend == Cpu). When `options.compute_multimetric` is on,
    // the CPU path ALSO produces the multi-norm butteraugli + PSNR
    // even if the GPU path returned a value for `butter_norm3`.
    #[cfg(feature = "cpu-metrics")]
    {
        let need_ssim2 = bundle.ssim2.is_none() && backend != MetricBackend::Skip;
        let need_butter = bundle.butter_norm3.is_none() && backend != MetricBackend::Skip;
        let need_multimetric =
            options.compute_multimetric && (bundle.butter_max.is_none() || bundle.psnr_y.is_none());
        if need_ssim2 || need_butter || need_multimetric {
            if let Some(cpu) = compute_cpu_metrics(
                source_rgb,
                decoded_rgb,
                width,
                height,
                options.save_diffmap || options.compute_multimetric,
            ) {
                if need_ssim2 {
                    bundle.ssim2 = Some(cpu.ssim2);
                    bundle.ssim2_backend = "cpu-ssimulacra2".into();
                }
                if need_butter {
                    bundle.butter_norm3 = Some(cpu.butter_norm3);
                    bundle.butter_norm3_backend = "cpu-butteraugli-norm3".into();
                }
                if options.compute_multimetric {
                    bundle.butter_max = Some(cpu.butter_max);
                    bundle.butter_p1 = cpu.butter_p1;
                    bundle.butter_p2 = cpu.butter_p2;
                    bundle.butter_p6 = cpu.butter_p6;
                }
                if options.save_diffmap {
                    bundle.diffmap = cpu.diffmap;
                }
            }
        }
        // CVVDP: no CPU-only implementation; leave None.
        if bundle.cvvdp_backend.is_empty() {
            bundle.cvvdp_backend = "cpu-unsupported".into();
        }
    }

    // PSNR is always computable on CPU (cheap u8 loop). Compute on
    // request regardless of backend — even when `MetricBackend::Skip`
    // chose to drop ssim2/butter the operator may still want PSNR for
    // joining against historical sweeps.
    if options.compute_multimetric && bundle.psnr_y.is_none() {
        let (py, pr, pg, pb) = compute_psnr_rgb_and_luma(source_rgb, decoded_rgb);
        bundle.psnr_y = Some(py);
        bundle.psnr_r = Some(pr);
        bundle.psnr_g = Some(pg);
        bundle.psnr_b = Some(pb);
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

/// PSNR per channel + Rec.709 luma. Returns `(y, r, g, b)` in dB.
/// Cheap u8 loop — runs on every call when `compute_multimetric` is
/// requested; ~1 ms for a 1 MP image.
fn compute_psnr_rgb_and_luma(source_rgb: &[u8], decoded_rgb: &[u8]) -> (f32, f32, f32, f32) {
    debug_assert_eq!(source_rgb.len(), decoded_rgb.len());
    debug_assert_eq!(source_rgb.len() % 3, 0);
    let n_px = (source_rgb.len() / 3) as f64;
    if n_px == 0.0 {
        return (f32::NAN, f32::NAN, f32::NAN, f32::NAN);
    }
    let mut sse_r = 0.0_f64;
    let mut sse_g = 0.0_f64;
    let mut sse_b = 0.0_f64;
    let mut sse_y = 0.0_f64;
    for (s, d) in source_rgb
        .as_chunks::<3>()
        .0
        .iter()
        .zip(decoded_rgb.as_chunks::<3>().0)
    {
        let dr = s[0] as f64 - d[0] as f64;
        let dg = s[1] as f64 - d[1] as f64;
        let db = s[2] as f64 - d[2] as f64;
        sse_r += dr * dr;
        sse_g += dg * dg;
        sse_b += db * db;
        // Rec.709 luma weights in sRGB-encoded u8 space (close enough
        // for PSNR — we don't need linearisation for this).
        let y_s = 0.2126 * s[0] as f64 + 0.7152 * s[1] as f64 + 0.0722 * s[2] as f64;
        let y_d = 0.2126 * d[0] as f64 + 0.7152 * d[1] as f64 + 0.0722 * d[2] as f64;
        let dy = y_s - y_d;
        sse_y += dy * dy;
    }
    let psnr_from_mse = |mse: f64| -> f32 {
        if mse <= f64::EPSILON {
            // identical images → "infinite" PSNR; report a finite
            // sentinel (100 dB is far above any real-world PSNR) so
            // downstream training sees a usable number instead of inf.
            100.0
        } else {
            (10.0 * (255.0_f64 * 255.0 / mse).log10()) as f32
        }
    };
    let mse_r = sse_r / n_px;
    let mse_g = sse_g / n_px;
    let mse_b = sse_b / n_px;
    let mse_y = sse_y / n_px;
    (
        psnr_from_mse(mse_y),
        psnr_from_mse(mse_r),
        psnr_from_mse(mse_g),
        psnr_from_mse(mse_b),
    )
}

struct MetricRunResult {
    value: Option<f32>,
    backend: String,
    elapsed_ms: f64,
    peak_mb: u32,
}

/// Invoke `zen-metrics score --metric <name> --reference <src>
/// --distorted <dist> --output json`. The zen-metrics CLI emits one
/// JSON object per call with shape
/// `{"metric":"<metric-name>","scores":{"<score-field>":f64,...}}`
/// where `<score-field>` differs per metric (e.g. `ssim2_gpu`,
/// `butteraugli_pnorm3_gpu`, `cvvdp_imazen_v0_0_1`). W44-214 smoke
/// test established the contract by reading the CLI's actual output;
/// the runner extracts `scores.<score_field>` and falls back to the
/// first numeric field in the `scores` map if the named field is
/// missing.
///
/// GPU resource columns (`gpu_peak_vram_mb` / `gpu_kernel_ms`) are
/// NOT emitted by zen-metrics 0.6.0; the runner tracks wall-clock
/// `elapsed_ms` as a proxy for GPU kernel time, and peak VRAM stays
/// 0 until zen-metrics gains a `--report-resource` flag.
fn invoke_zen_metric(src: &Path, dist: &Path, metric: &str, score_field: &str) -> MetricRunResult {
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
            "--output",
            "json",
        ])
        .output();
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    match out {
        Ok(o) if o.status.success() => {
            let parsed: serde_json::Value =
                serde_json::from_slice(&o.stdout).unwrap_or(serde_json::Value::Null);
            // Primary path: scores.<score_field>
            let value = parsed
                .get("scores")
                .and_then(|s| s.get(score_field))
                .and_then(|v| v.as_f64())
                .map(|x| x as f32)
                // Fallback: scores.<first-numeric-field>
                .or_else(|| {
                    parsed.get("scores").and_then(|s| {
                        s.as_object()
                            .and_then(|m| m.values().find_map(|v| v.as_f64().map(|x| x as f32)))
                    })
                })
                // Legacy fallback: top-level `score` field (kept for
                // forward-compat if zen-metrics adds it).
                .or_else(|| {
                    parsed
                        .get("score")
                        .and_then(|v| v.as_f64())
                        .map(|x| x as f32)
                });
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
                backend: format!(
                    "gpu-cli-failed: {}",
                    stderr.trim().chars().take(120).collect::<String>()
                ),
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

/// CPU-path metric bundle returned by [`compute_cpu_metrics`].
#[cfg(feature = "cpu-metrics")]
struct CpuMetricResult {
    ssim2: f32,
    butter_norm3: f32,
    butter_max: f32,
    butter_p1: Option<f32>,
    butter_p2: Option<f32>,
    butter_p6: Option<f32>,
    diffmap: Option<Vec<f32>>,
}

/// Compute SSIMULACRA2 + butteraugli on a (source, decoded) RGB u8
/// pair. When `with_diffmap` is true, also return the per-pixel
/// diffmap + multi-norm aggregations (max / p1 / p2 / p6).
///
/// Updated W44-PHASE4-M1 to:
/// 1. Fix the pre-existing compile-broken call (was passing `1.0`
///    where a `&ButteraugliParams` was expected, and `RGB<f32>` where
///    `RGB<u8>` was expected). The original CPU path never compiled.
/// 2. Optionally compute the multi-norm aggregations + diffmap in a
///    single butteraugli pass (cheap once `compute_diffmap=true`).
#[cfg(feature = "cpu-metrics")]
fn compute_cpu_metrics(
    source: &[u8],
    decoded: &[u8],
    w: usize,
    h: usize,
    with_diffmap: bool,
) -> Option<CpuMetricResult> {
    use butteraugli::{ButteraugliParams, RGB8, butteraugli};
    use imgref::Img;

    // butteraugli 0.9.2 accepts sRGB-encoded `RGB<u8>` directly and
    // linearises internally with libjxl-correct sRGB TF. No need for
    // our own gamma→linear pre-pass.
    let to_rgb8 = |buf: &[u8]| -> Vec<RGB8> {
        let mut out = Vec::with_capacity(w * h);
        for px in buf.chunks_exact(3) {
            out.push(RGB8 {
                r: px[0],
                g: px[1],
                b: px[2],
            });
        }
        out
    };
    let src_rgb8 = to_rgb8(source);
    let dist_rgb8 = to_rgb8(decoded);
    let src_img = Img::new(src_rgb8, w, h);
    let dist_img = Img::new(dist_rgb8, w, h);

    let params = ButteraugliParams::default().with_compute_diffmap(with_diffmap);
    let result = match butteraugli(src_img.as_ref(), dist_img.as_ref(), &params) {
        Ok(r) => r,
        Err(_) => return None,
    };

    let butter_norm3 = result.pnorm_3 as f32;
    let butter_max = result.score as f32;
    // Multi-norm aggregations only computable when diffmap is present.
    let (butter_p1, butter_p2, butter_p6) = if with_diffmap {
        (
            result.pnorm(1.0).map(|x| x as f32),
            result.pnorm(2.0).map(|x| x as f32),
            result.pnorm(6.0).map(|x| x as f32),
        )
    } else {
        (None, None, None)
    };
    // Extract diffmap as a row-major Vec<f32>. `ImgVec::buf()` returns
    // a contiguous slice (stride == width per butteraugli 0.9.2
    // contract).
    let diffmap = if with_diffmap {
        result.diffmap.as_ref().map(|dm| dm.buf().to_vec())
    } else {
        None
    };

    // fast-ssim2 0.8.0: `compute_ssimulacra2` takes any `ToLinearRgb`
    // input. `ImgRef<[u8; 3]>` is sRGB-encoded by convention. Build a
    // `[u8;3]` planar buffer from our RGB u8 source.
    let to_rgb_arr = |buf: &[u8]| -> Vec<[u8; 3]> {
        let mut out = Vec::with_capacity(w * h);
        for px in buf.chunks_exact(3) {
            out.push([px[0], px[1], px[2]]);
        }
        out
    };
    let src_arr = to_rgb_arr(source);
    let dist_arr = to_rgb_arr(decoded);
    let src_img2 = Img::new(src_arr, w, h);
    let dist_img2 = Img::new(dist_arr, w, h);
    let s2 = fast_ssim2::compute_ssimulacra2(src_img2.as_ref(), dist_img2.as_ref())
        .ok()
        .map(|s| s as f32)
        .unwrap_or(f32::NAN);

    Some(CpuMetricResult {
        ssim2: s2,
        butter_norm3,
        butter_max,
        butter_p1,
        butter_p2,
        butter_p6,
        diffmap,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn psnr_identical_buffers_reports_100db() {
        let buf = vec![128u8; 3 * 64 * 64];
        let (py, pr, pg, pb) = compute_psnr_rgb_and_luma(&buf, &buf);
        assert_eq!(py, 100.0);
        assert_eq!(pr, 100.0);
        assert_eq!(pg, 100.0);
        assert_eq!(pb, 100.0);
    }

    #[test]
    fn psnr_byte_diff_is_finite() {
        let src = vec![100u8; 3 * 64 * 64];
        let mut dst = vec![100u8; 3 * 64 * 64];
        // Flip a few pixels to introduce error.
        for i in 0..32 {
            dst[i * 3] = 110; // R diff = 10
        }
        let (py, pr, pg, pb) = compute_psnr_rgb_and_luma(&src, &dst);
        assert!(py.is_finite() && py < 100.0);
        assert!(pr.is_finite() && pr < 100.0);
        // G and B channels unchanged → 100.
        assert_eq!(pg, 100.0);
        assert_eq!(pb, 100.0);
    }

    #[test]
    fn score_options_defaults_v1_compat() {
        let opts = ScoreOptions::default();
        assert!(!opts.compute_multimetric);
        assert!(!opts.save_diffmap);
    }
}
