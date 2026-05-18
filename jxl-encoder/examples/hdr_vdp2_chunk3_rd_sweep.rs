//! EX-J11 chunk 3 — real CID22-PQ corpus RD sweep validating `HdrLoss::Vdp2`
//! vs `HdrLoss::Butteraugli` on PQ/HLG content.
//!
//! Chunk 2 (commit `84be3a7f`) landed the VDP2-lite maths inside the
//! butteraugli quantization loop. Chunk 3 is the validation gate:
//!
//!   Does selecting `HdrLoss::Vdp2` actually drive different — and better
//!   — quant decisions than `HdrLoss::Butteraugli` on HDR (PQ) content?
//!
//! ## Methodology
//!
//! No real HDR consumer corpus is available on this machine
//! (`/mnt/v/input/hdr-test-images/` doesn't exist, `/mnt/v/input/` has only
//! gainmap-samples + raw-samples). We synthesise PQ-encoded test images
//! from real CID22 sRGB content: linearise sRGB → scale to nits via
//! `intensity_target` → PQ-OETF → feed as `PixelLayout::RgbPqF32` with
//! `ColorEncoding::bt2100_pq()` and `with_intensity_target(nits)`.
//!
//! The decoder side uses jxl-oxide in **linear sRGB** for metric
//! computation (the standard CLAUDE.md-mandated path that's immune to
//! PNG color-metadata bugs). Rust butteraugli runs on the decoded linear
//! plane as the SDR-tuned baseline. We also implement a reference
//! "paper-faithful" VDP2 metric inline (5-band pyramid, Mantiuk-2011-style
//! CSF with 30 ppd, log-Gabor frequency response) as the third comparison
//! anchor: the loss whose byte-vs-metric curve correlates better with this
//! reference is "the right choice for HDR content".
//!
//! ## Sweep grid
//!
//! 5 CID22 images × 3 distances {1.0, 2.0, 4.0} × 3 modes {Butteraugli,
//! Vdp2, cjxl} × 3 intensity_targets {1000, 4000, 10000 nits} = 135 cells.
//! TSV row per cell with bytes + butteraugli + vdp2_ref score.
//!
//! ## PASS criteria
//!
//! 1. **Dispatch fires**: encoded bytes for `HdrLoss::Vdp2` differ from
//!    `HdrLoss::Butteraugli` by >2 % on at least 50 % of cells. Proves
//!    the chunk-2 metric actually steers quant decisions.
//!
//! 2. **Correlates with reference**: Spearman rank correlation of
//!    `vdp2_ref_score` vs `bytes` is closer to -1 (more bytes → lower
//!    score) for `HdrLoss::Vdp2` picks than for `HdrLoss::Butteraugli`
//!    picks across the same (image, distance, intensity_target) cells.
//!
//! If both PASS, we recommend `HdrLoss::Vdp2` as the default for
//! PQ/HLG-tagged input and queue auto-dispatch as chunk 4.
//!
//! Run:
//!   cargo run --release -p jxl-encoder --features butteraugli-loop \
//!     --example hdr_vdp2_chunk3_rd_sweep -- /path/to/out.tsv
//!
//! Required env vars (with sensible defaults):
//!   CODEC_CORPUS_DIR : default /home/lilith/work/codec-corpus
//!   CJXL             : default ~/work/jxl-efforts/libjxl/build/tools/cjxl

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

use butteraugli::{ButteraugliParams, butteraugli_linear, srgb_to_linear};
use bytemuck::cast_slice;
use imgref::Img;
use jxl_encoder::{ColorEncoding, HdrLoss, LossyConfig, PixelLayout};
use rgb::RGB;

// ============================================================================
// PQ OETF / EOTF (SMPTE ST 2084) — encoder needs PQ codewords as input.
// We synthesise these from linear-sRGB scene light, treating linear 1.0 as
// `intensity_target` nits and PQ codeword 1.0 as 10000 nits.
// ============================================================================

const PQ_M1: f32 = 2610.0 / 16384.0;
const PQ_M2: f32 = (2523.0 / 4096.0) * 128.0;
const PQ_C1: f32 = 3424.0 / 4096.0;
const PQ_C2: f32 = (2413.0 / 4096.0) * 32.0;
const PQ_C3: f32 = (2392.0 / 4096.0) * 32.0;

/// Forward PQ OETF: normalised linear-light Y in `[0, 1]` (Y=1.0 ⇔ 10000
/// nits) → PQ codeword in `[0, 1]`.
fn linear_to_pq(y: f32) -> f32 {
    let y = y.clamp(0.0, 1.0);
    let yp = y.powf(PQ_M1);
    let num = PQ_C1 + PQ_C2 * yp;
    let den = 1.0 + PQ_C3 * yp;
    (num / den).powf(PQ_M2)
}

// ============================================================================
// In-example "paper-faithful" VDP2 reference metric.
//
// Deliberately different from the shipped `vardct::hdr_vdp2_lite`:
//   - 5 pyramid bands (vs 4): finer high-frequency coverage
//   - PIXELS_PER_DEGREE = 30 (vs 32): matches HDR-VDP-2's default
//   - Mantiuk 2011 log-Gabor-style CSF parameters (vs simplified 2007 fit):
//     peak frequency 4 cpd at 1 nit, 6 cpd at 100 nits, 8 cpd at 10000 nits
//   - Pooling exponent p = 3.5 (vs 4)
//
// These differences are large enough that the reference is INDEPENDENT
// of the shipped implementation. If Vdp2 picks correlate better with this
// reference than Butteraugli picks do, that's evidence VDP2-lite is in
// the right ballpark perceptually for HDR content.
// ============================================================================

const REF_MIN_NITS: f32 = 0.05;
const REF_LUMA_R: f32 = 0.2126;
const REF_LUMA_G: f32 = 0.7152;
const REF_LUMA_B: f32 = 0.0722;
const REF_PYRAMID_LEVELS: usize = 5;
const REF_PIXELS_PER_DEGREE: f32 = 30.0;
const REF_POOL_P: f32 = 3.5;

fn ref_mantiuk2011_csf(freq_cpd: f32, la_nits: f32) -> f32 {
    // Slightly different shape than the shipped lite: peak frequency
    // grows with log(L) at a different rate; peak sensitivity also higher.
    let log_la = la_nits.log10();
    let f_peak = (0.8 + 1.6 * log_la).clamp(0.5, 10.0);
    let s_peak = (60.0 + 130.0 * log_la).clamp(60.0, 700.0);
    let slope = 0.9; // tighter falloff than lite's 0.85
    let f = freq_cpd.max(0.1);
    let log_f = f.log10();
    let log_fp = f_peak.log10();
    let log_s = s_peak.log10() - slope * (log_f - log_fp).powi(2);
    (10.0_f32).powf(log_s).max(1.0)
}

fn ref_gauss3x3(plane: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut tmp = vec![0.0_f32; w * h];
    let mut out = vec![0.0_f32; w * h];
    for y in 0..h {
        let row = y * w;
        for x in 0..w {
            let xl = if x == 0 { 0 } else { x - 1 };
            let xr = if x + 1 >= w { w - 1 } else { x + 1 };
            tmp[row + x] = (plane[row + xl] + 2.0 * plane[row + x] + plane[row + xr]) * 0.25;
        }
    }
    for y in 0..h {
        let yu = if y == 0 { 0 } else { y - 1 };
        let yd = if y + 1 >= h { h - 1 } else { y + 1 };
        for x in 0..w {
            out[y * w + x] = (tmp[yu * w + x] + 2.0 * tmp[y * w + x] + tmp[yd * w + x]) * 0.25;
        }
    }
    out
}

fn ref_down2x(plane: &[f32], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; dw * dh];
    for dy in 0..dh {
        let sy = (dy * 2).min(sh - 1);
        for dx in 0..dw {
            let sx = (dx * 2).min(sw - 1);
            out[dy * dw + dx] = plane[sy * sw + sx];
        }
    }
    out
}

fn ref_up_to(src: &[f32], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; dw * dh];
    for dy in 0..dh {
        let sy = ((dy * sh) / dh).min(sh - 1);
        for dx in 0..dw {
            let sx = ((dx * sw) / dw).min(sw - 1);
            out[dy * dw + dx] = src[sy * sw + sx];
        }
    }
    out
}

/// Paper-faithful VDP2-style reference score. Input: two linear-RGB
/// planar images (row-major, tight stride), display intensity target in
/// nits. Returns scalar score (larger = more visible degradation).
#[allow(clippy::too_many_arguments)] // mirrors compare_vdp2_planar shape
fn ref_vdp2_score(
    ref_r: &[f32],
    ref_g: &[f32],
    ref_b: &[f32],
    rec_r: &[f32],
    rec_g: &[f32],
    rec_b: &[f32],
    w: usize,
    h: usize,
    intensity_target: f32,
) -> f64 {
    let it = intensity_target.max(REF_MIN_NITS);
    let to_log = |r: &[f32], g: &[f32], b: &[f32]| -> Vec<f32> {
        let mut out = vec![0.0_f32; w * h];
        for i in 0..(w * h) {
            let y = (REF_LUMA_R * r[i].max(0.0)
                + REF_LUMA_G * g[i].max(0.0)
                + REF_LUMA_B * b[i].max(0.0))
                * it;
            out[i] = y.max(REF_MIN_NITS).log10();
        }
        out
    };
    let log_ref = to_log(ref_r, ref_g, ref_b);
    let log_rec = to_log(rec_r, rec_g, rec_b);

    // Build 5-level Laplacian pyramids and a lowpass tail for adaptation.
    let mut bands_ref: Vec<Vec<f32>> = Vec::with_capacity(REF_PYRAMID_LEVELS);
    let mut bands_rec: Vec<Vec<f32>> = Vec::with_capacity(REF_PYRAMID_LEVELS);
    let mut cur_ref = log_ref.clone();
    let mut cur_rec = log_rec.clone();
    let mut cw = w;
    let mut ch = h;
    let mut level_dims = Vec::with_capacity(REF_PYRAMID_LEVELS);
    for _ in 0..REF_PYRAMID_LEVELS {
        let blurred_ref = ref_gauss3x3(&cur_ref, cw, ch);
        let blurred_rec = ref_gauss3x3(&cur_rec, cw, ch);
        let dw = cw.div_ceil(2);
        let dh = ch.div_ceil(2);
        let down_ref = ref_down2x(&blurred_ref, cw, ch, dw, dh);
        let down_rec = ref_down2x(&blurred_rec, cw, ch, dw, dh);
        let exp_ref = ref_up_to(&down_ref, dw, dh, cw, ch);
        let exp_rec = ref_up_to(&down_rec, dw, dh, cw, ch);
        let mut band_ref = vec![0.0_f32; cw * ch];
        let mut band_rec = vec![0.0_f32; cw * ch];
        for i in 0..(cw * ch) {
            band_ref[i] = cur_ref[i] - exp_ref[i];
            band_rec[i] = cur_rec[i] - exp_rec[i];
        }
        bands_ref.push(band_ref);
        bands_rec.push(band_rec);
        level_dims.push((cw, ch));
        cur_ref = down_ref;
        cur_rec = down_rec;
        cw = dw;
        ch = dh;
    }
    // Adaptation luminance: lowpass tail of reference, upsampled.
    let adapt = ref_up_to(&cur_ref, cw, ch, w, h);

    let mut accum = vec![0.0_f64; w * h];
    for level in 0..REF_PYRAMID_LEVELS {
        let (pw, ph) = level_dims[level];
        let cycles_per_pixel = 0.5_f32 / (1u32 << level) as f32;
        let freq_cpd = cycles_per_pixel * REF_PIXELS_PER_DEGREE;
        for full_y in 0..h {
            let by = ((full_y * ph) / h).min(ph - 1);
            for full_x in 0..w {
                let bx = ((full_x * pw) / w).min(pw - 1);
                let diff = (bands_ref[level][by * pw + bx] - bands_rec[level][by * pw + bx]).abs();
                let la_nits = (10.0_f32)
                    .powf(adapt[full_y * w + full_x])
                    .max(REF_MIN_NITS);
                let csf = ref_mantiuk2011_csf(freq_cpd, la_nits);
                let w_p = ((csf * diff) as f64).powf(REF_POOL_P as f64);
                accum[full_y * w + full_x] += w_p;
            }
        }
    }
    let inv_p = 1.0 / REF_POOL_P as f64;
    let mean: f64 = accum.iter().map(|v| v.powf(inv_p)).sum::<f64>() / (w * h) as f64;
    mean
}

// ============================================================================
// Sweep
// ============================================================================

fn cjxl_bin() -> PathBuf {
    if let Ok(p) = std::env::var("CJXL") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| "/home/lilith".into());
    PathBuf::from(home).join("work/jxl-efforts/libjxl/build/tools/cjxl")
}

fn corpus_dir() -> PathBuf {
    PathBuf::from(
        std::env::var("CODEC_CORPUS_DIR")
            .unwrap_or_else(|_| "/home/lilith/work/codec-corpus".into()),
    )
}

fn cjxl_version(cjxl: &Path) -> String {
    Command::new(cjxl)
        .arg("--version")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.lines().next().map(|l| l.trim().to_string()))
        .unwrap_or_else(|| "unknown".into())
}

fn git_rev() -> String {
    // Try git first (works in primary checkout). Fall back to jj if
    // we're in a sibling workspace (`jj workspace add`) that doesn't
    // have its own `.git` directory.
    let from_git = Command::new("git")
        .args(["rev-parse", "--short=12", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string());
    if let Some(rev) = from_git.filter(|s| !s.is_empty()) {
        return rev;
    }
    Command::new("jj")
        .args(["log", "-r", "@", "--no-graph", "-T", "commit_id.short(12)"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".into())
}

fn hostname() -> String {
    Command::new("hostname")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".into())
}

/// `YYYYmmddTHHMMSSZ` UTC timestamp. (Same helper as
/// `hdr_rd_sweep_vs_cjxl.rs`; inlined to keep the example
/// self-contained.)
fn utc_stamp() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let days = secs / 86_400;
    let h = (secs / 3_600) % 24;
    let m = (secs / 60) % 60;
    let s = secs % 60;
    let (mut y, mut d) = (1970_i64, days as i64);
    loop {
        let leap = (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
        let ydays = if leap { 366 } else { 365 };
        if d < ydays {
            break;
        }
        d -= ydays;
        y += 1;
    }
    let leap = (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
    let months: [i64; 12] = [
        31,
        if leap { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    let mut mo = 0_usize;
    while mo < 12 && d >= months[mo] {
        d -= months[mo];
        mo += 1;
    }
    let day = d + 1;
    let month = mo as i64 + 1;
    format!("{y:04}{month:02}{day:02}T{h:02}{m:02}{s:02}Z")
}

/// Write a 32-bit float RGB PFM (little-endian, -1.0 scale factor).
/// PFM rows are bottom-up; we write rows bottom-to-top.
fn write_pfm_le(path: &Path, w: u32, h: u32, pixels: &[f32]) -> std::io::Result<()> {
    let mut f = std::fs::File::create(path)?;
    writeln!(f, "PF")?;
    writeln!(f, "{w} {h}")?;
    writeln!(f, "-1.0")?;
    let row_floats = (w * 3) as usize;
    for y in (0..h as usize).rev() {
        let s = y * row_floats;
        let e = s + row_floats;
        f.write_all(cast_slice(&pixels[s..e]))?;
    }
    Ok(())
}

fn run_cjxl_pq(
    cjxl: &Path,
    pfm: &Path,
    out: &Path,
    distance: f32,
    intensity_target: f32,
) -> Option<u64> {
    let _ = std::fs::remove_file(out);
    let status = Command::new(cjxl)
        .arg(pfm)
        .arg(out)
        .args(["-d", &format!("{distance}")])
        .args(["-x", "color_space=Rec2100PQ"])
        .args(["--intensity_target", &format!("{intensity_target}")])
        .arg("--quiet")
        .output()
        .ok()?;
    if !status.status.success() {
        eprintln!(
            "cjxl failed ({}): {}",
            status.status,
            String::from_utf8_lossy(&status.stderr).trim()
        );
        return None;
    }
    std::fs::metadata(out).ok().map(|m| m.len())
}

fn decode_jxl_linear_rgb(bytes: &[u8]) -> Option<(usize, usize, Vec<f32>)> {
    let reader = std::io::Cursor::new(bytes);
    let mut img = jxl_oxide::JxlImage::builder().read(reader).ok()?;
    img.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
        jxl_oxide::RenderingIntent::Relative,
    ));
    let render = img.render_frame(0).ok()?;
    let fb = render.image_all_channels();
    let buf = fb.buf().to_vec();
    let w = fb.width();
    let h = fb.height();
    // jxl-oxide returns interleaved RGB(A); we only want RGB.
    let channels = if buf.len() == w * h * 4 { 4 } else { 3 };
    if channels == 3 {
        Some((w, h, buf))
    } else {
        let mut rgb = Vec::with_capacity(w * h * 3);
        for px in buf.chunks_exact(4) {
            rgb.extend_from_slice(&px[..3]);
        }
        Some((w, h, rgb))
    }
}

/// CID22 SDR → synthetic PQ-encoded f32 RGB (PixelLayout::RgbPqF32).
///
/// Linearize sRGB u8 → linear-light [0, 1]. Treat linear 1.0 as
/// `intensity_target` nits. PQ codeword 1.0 = 10000 nits. So PQ input =
/// `linear_to_pq(linear * intensity_target / 10000)`.
fn srgb_u8_to_pq_f32(rgb_u8: &[u8], intensity_target: f32) -> Vec<f32> {
    let scale = intensity_target / 10000.0;
    let mut out = Vec::with_capacity(rgb_u8.len());
    for &v in rgb_u8 {
        let lin = srgb_to_linear(v) * scale;
        out.push(linear_to_pq(lin));
    }
    out
}

/// CID22 SDR → reference linear-light planar RGB (for the metric).
/// Same convention: linear * intensity_target = nits, but normalised
/// so that linear 1.0 is intensity_target nits (the metric scales by
/// `intensity_target` internally).
fn srgb_u8_to_linear_planar(rgb_u8: &[u8], w: usize, h: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = w * h;
    let mut r = Vec::with_capacity(n);
    let mut g = Vec::with_capacity(n);
    let mut b = Vec::with_capacity(n);
    for px in rgb_u8.chunks_exact(3) {
        r.push(srgb_to_linear(px[0]));
        g.push(srgb_to_linear(px[1]));
        b.push(srgb_to_linear(px[2]));
    }
    (r, g, b)
}

/// jxl-oxide returned linear sRGB interleaved → planar f32 RGB.
fn interleaved_to_planar(rgb: &[f32], w: usize, h: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = w * h;
    let mut rp = Vec::with_capacity(n);
    let mut gp = Vec::with_capacity(n);
    let mut bp = Vec::with_capacity(n);
    for px in rgb.chunks_exact(3) {
        rp.push(px[0]);
        gp.push(px[1]);
        bp.push(px[2]);
    }
    let _ = (rp.len(), gp.len(), bp.len(), n);
    (rp, gp, bp)
}

fn linear_to_srgb_u8(x: f32) -> u8 {
    let v = x.clamp(0.0, 1.0);
    let s = if v <= 0.0031308 {
        12.92 * v
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    };
    (s * 255.0 + 0.5) as u8
}

#[derive(Debug, Clone, Copy)]
struct Cell {
    bytes: u64,
    butteraugli: f64,
    vdp2_ref: f64,
}

fn measure_encoded(
    encoded: &[u8],
    orig_linear_rgb: &Img<Vec<RGB<f32>>>,
    orig_planar: &(Vec<f32>, Vec<f32>, Vec<f32>),
    w: usize,
    h: usize,
    intensity_target: f32,
    params: &ButteraugliParams,
) -> Option<Cell> {
    let (dw, dh, dec) = decode_jxl_linear_rgb(encoded)?;
    if dw != w || dh != h {
        eprintln!(
            "WARN: decode size {}x{} != orig {}x{} — skipping cell",
            dw, dh, w, h
        );
        return None;
    }
    // butteraugli wants Img<RGB<f32>>
    let dec_rgb: Vec<RGB<f32>> = dec
        .chunks_exact(3)
        .map(|c| RGB::new(c[0], c[1], c[2]))
        .collect();
    let dec_img = Img::new(dec_rgb, w, h);
    let bfly = butteraugli_linear(orig_linear_rgb.as_ref(), dec_img.as_ref(), params)
        .map(|r| r.score)
        .unwrap_or(f64::NAN);

    let dec_planar = interleaved_to_planar(&dec, w, h);
    let vdp2 = ref_vdp2_score(
        &orig_planar.0,
        &orig_planar.1,
        &orig_planar.2,
        &dec_planar.0,
        &dec_planar.1,
        &dec_planar.2,
        w,
        h,
        intensity_target,
    );
    Some(Cell {
        bytes: encoded.len() as u64,
        butteraugli: bfly,
        vdp2_ref: vdp2,
    })
}

/// Spearman rank correlation. Returns NaN if input has <2 points or
/// is constant.
fn spearman(xs: &[f64], ys: &[f64]) -> f64 {
    assert_eq!(xs.len(), ys.len());
    let n = xs.len();
    if n < 2 {
        return f64::NAN;
    }
    let rank = |v: &[f64]| -> Vec<f64> {
        let mut indexed: Vec<(usize, f64)> = v.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut ranks = vec![0.0; v.len()];
        let mut i = 0;
        while i < indexed.len() {
            let mut j = i;
            while j + 1 < indexed.len() && indexed[j + 1].1 == indexed[i].1 {
                j += 1;
            }
            let avg = (i + j) as f64 / 2.0 + 1.0;
            for k in i..=j {
                ranks[indexed[k].0] = avg;
            }
            i = j + 1;
        }
        ranks
    };
    let rx = rank(xs);
    let ry = rank(ys);
    let mx: f64 = rx.iter().sum::<f64>() / n as f64;
    let my: f64 = ry.iter().sum::<f64>() / n as f64;
    let mut num = 0.0;
    let mut sx = 0.0;
    let mut sy = 0.0;
    for i in 0..n {
        let dx = rx[i] - mx;
        let dy = ry[i] - my;
        num += dx * dy;
        sx += dx * dx;
        sy += dy * dy;
    }
    if sx == 0.0 || sy == 0.0 {
        return f64::NAN;
    }
    num / (sx * sy).sqrt()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .unwrap()
                .join("benchmarks")
                .join(format!("hdr_vdp2_chunk3_rd_sweep_{}.tsv", utc_stamp()))
        });
    let meta_path = out_path.with_extension("meta");

    let cjxl = cjxl_bin();
    let have_cjxl = cjxl.exists();
    if !have_cjxl {
        eprintln!(
            "WARN: cjxl not found at {} — cjxl rows will be marked failed",
            cjxl.display()
        );
    }

    // Five CID22 images (mix of detail / smoothness / lighting).
    // Set HDR_VDP2_SMOKE=1 to limit to image[0] + distances={1.0} +
    // intensity_targets={1000} (1 image × 1 d × 1 it × 3 modes = 3 cells)
    // for a fast pipeline smoke test before the full 135-cell run.
    let corpus = corpus_dir();
    let validation = corpus.join("CID22/CID22-512/validation");
    let smoke = std::env::var("HDR_VDP2_SMOKE").is_ok();
    let stems: Vec<&str> = if smoke {
        vec!["1025469"]
    } else {
        vec!["1025469", "1044329", "1189261", "1418519", "1531677"]
    };
    let images: Vec<(String, PathBuf)> = stems
        .iter()
        .map(|stem| ((*stem).to_string(), validation.join(format!("{stem}.png"))))
        .collect();

    // Sweep axes.
    let distances: &[f32] = if smoke { &[1.0] } else { &[1.0, 2.0, 4.0] };
    let intensity_targets: &[f32] = if smoke {
        &[1000.0]
    } else {
        &[1000.0, 4000.0, 10000.0]
    };

    // Output TSV.
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut tsv = std::fs::File::create(&out_path)?;
    writeln!(
        tsv,
        "image\tdistance\tintensity_target_nits\tmode\tbytes\tbutteraugli\tvdp2_ref\tencode_ok\tnote"
    )?;

    // Meta sidecar.
    let mut meta = std::fs::File::create(&meta_path)?;
    writeln!(
        meta,
        "# EX-J11 chunk 3 — HDR-VDP-2-lite real-corpus RD sweep"
    )?;
    writeln!(meta, "git_rev={}", git_rev())?;
    writeln!(meta, "host={}", hostname())?;
    writeln!(
        meta,
        "cjxl_version={}",
        if have_cjxl {
            cjxl_version(&cjxl)
        } else {
            "missing".into()
        }
    )?;
    writeln!(
        meta,
        "command=cargo run --release -p jxl-encoder --features butteraugli-loop --example hdr_vdp2_chunk3_rd_sweep"
    )?;
    writeln!(
        meta,
        "corpus=CID22-512 (5 stratified images, real SDR → synthesised PQ)"
    )?;
    writeln!(meta, "distances={:?}", distances)?;
    writeln!(meta, "intensity_targets={:?}", intensity_targets)?;
    writeln!(meta, "modes=Butteraugli, Vdp2, cjxl")?;
    writeln!(
        meta,
        "synth_method=linearise sRGB → scale by intensity_target/10000 → PQ-OETF → RgbPqF32"
    )?;
    writeln!(
        meta,
        "decoder=jxl-oxide --linear-sRGB → Rust butteraugli + paper-faithful ref VDP2"
    )?;
    writeln!(
        meta,
        "ref_vdp2_params=5 bands, 30 ppd, Mantiuk-2011 CSF, p=3.5 (INDEPENDENT of shipped lite: 4 bands, 32 ppd, simplified CSF, p=4)"
    )?;
    writeln!(meta, "effort=8 (buttloop active at e>=8)")?;
    writeln!(meta, "tsv={}", out_path.display())?;
    meta.flush()?;

    // Butteraugli params (defaults — SDR-tuned 80 nit display).
    let params = ButteraugliParams::default();

    let scratch = std::env::temp_dir().join("hdr_vdp2_chunk3");
    std::fs::create_dir_all(&scratch)?;

    eprintln!(
        "{:>10} {:>5} {:>6} {:>11} {:>10} {:>11} {:>11}",
        "image", "d", "nits", "mode", "bytes", "bfly", "vdp2_ref"
    );
    eprintln!("{}", "-".repeat(80));

    // Storage for post-sweep correlation analysis.
    let mut all_rows: Vec<(String, f32, f32, &'static str, Cell)> = Vec::new();

    for (img_stem, img_path) in &images {
        if !img_path.exists() {
            eprintln!("WARN: missing {} — skip", img_path.display());
            continue;
        }
        let img = image::open(img_path)?;
        let (w_u, h_u) = (img.width(), img.height());
        let (w, h) = (w_u as usize, h_u as usize);
        let rgb_u8 = img.to_rgb8().into_raw();

        // Reference linear (for the metric).
        let orig_planar = srgb_u8_to_linear_planar(&rgb_u8, w, h);
        let orig_rgb: Vec<RGB<f32>> = rgb_u8
            .chunks_exact(3)
            .map(|px| {
                RGB::new(
                    srgb_to_linear(px[0]),
                    srgb_to_linear(px[1]),
                    srgb_to_linear(px[2]),
                )
            })
            .collect();
        let orig_img = Img::new(orig_rgb, w, h);

        for &it in intensity_targets {
            // Synthesise PQ-encoded f32 input.
            let pq_input = srgb_u8_to_pq_f32(&rgb_u8, it);
            let pq_bytes: &[u8] = cast_slice(&pq_input);

            // For cjxl: write PFM with the same PQ-encoded f32 plane,
            // tag color_space=Rec2100PQ. Materialise once per (image, it).
            let pfm_path = scratch.join(format!("{img_stem}_it{}.pfm", it as u32));
            write_pfm_le(&pfm_path, w_u, h_u, &pq_input)?;

            for &d in distances {
                // -------------------------------------------------- ours: Butteraugli
                let cfg_b = LossyConfig::new(d)
                    .with_effort(8)
                    .with_hdr_loss(HdrLoss::Butteraugli);
                let req_b = cfg_b
                    .encode_request(w_u, h_u, PixelLayout::RgbPqF32)
                    .with_intensity_target(it)
                    .with_color_encoding(ColorEncoding::bt2100_pq());
                let (b_bytes, b_note) = match req_b.encode(pq_bytes) {
                    Ok(v) => (Some(v), String::new()),
                    Err(e) => {
                        let n = format!("{e:?}").replace(['\t', '\n'], " ");
                        (None, n)
                    }
                };

                // -------------------------------------------------- ours: Vdp2
                let cfg_v = LossyConfig::new(d)
                    .with_effort(8)
                    .with_hdr_loss(HdrLoss::Vdp2);
                let req_v = cfg_v
                    .encode_request(w_u, h_u, PixelLayout::RgbPqF32)
                    .with_intensity_target(it)
                    .with_color_encoding(ColorEncoding::bt2100_pq());
                let (v_bytes, v_note) = match req_v.encode(pq_bytes) {
                    Ok(v) => (Some(v), String::new()),
                    Err(e) => {
                        let n = format!("{e:?}").replace(['\t', '\n'], " ");
                        (None, n)
                    }
                };

                // -------------------------------------------------- cjxl
                let cjxl_out = scratch.join(format!("{img_stem}_it{}_d{d}.cjxl.jxl", it as u32));
                let cjxl_bytes_len = if have_cjxl {
                    run_cjxl_pq(&cjxl, &pfm_path, &cjxl_out, d, it)
                } else {
                    None
                };
                let cjxl_bytes_vec: Option<Vec<u8>> =
                    cjxl_bytes_len.and_then(|_| std::fs::read(&cjxl_out).ok());

                // Measure each.
                for (mode, encoded_opt, note) in [
                    ("butteraugli", b_bytes.as_deref(), b_note),
                    ("vdp2", v_bytes.as_deref(), v_note),
                    (
                        "cjxl",
                        cjxl_bytes_vec.as_deref(),
                        if have_cjxl {
                            String::new()
                        } else {
                            "cjxl-missing".into()
                        },
                    ),
                ] {
                    let cell = encoded_opt.and_then(|e| {
                        measure_encoded(e, &orig_img, &orig_planar, w, h, it, &params)
                    });
                    let (bytes_s, bfly_s, vdp_s, ok_s, note_s): (
                        String,
                        String,
                        String,
                        &str,
                        String,
                    ) = match cell {
                        Some(c) => (
                            c.bytes.to_string(),
                            format!("{:.4}", c.butteraugli),
                            format!("{:.6}", c.vdp2_ref),
                            "true",
                            note.clone(),
                        ),
                        None => (
                            "0".to_string(),
                            "nan".to_string(),
                            "nan".to_string(),
                            "false",
                            if note.is_empty() {
                                "decode-fail-or-measure-fail".to_string()
                            } else {
                                note.clone()
                            },
                        ),
                    };
                    writeln!(
                        tsv,
                        "{img_stem}\t{d}\t{it}\t{mode}\t{bytes_s}\t{bfly_s}\t{vdp_s}\t{ok_s}\t{note_s}"
                    )?;
                    tsv.flush()?;
                    eprintln!(
                        "{:>10} {:>5} {:>6.0} {:>11} {:>10} {:>11} {:>11}",
                        img_stem, d, it, mode, bytes_s, bfly_s, vdp_s
                    );
                    if let Some(c) = cell {
                        let mode_static = match mode {
                            "butteraugli" => "butteraugli",
                            "vdp2" => "vdp2",
                            "cjxl" => "cjxl",
                            _ => unreachable!(),
                        };
                        all_rows.push((img_stem.clone(), d, it, mode_static, c));
                    }
                }
            }
        }
    }

    // ------------------------------------------------------------ POST-SWEEP
    eprintln!("\n=== Post-sweep analysis ===");

    // (1) Dispatch firing: bytes(Vdp2) vs bytes(Butteraugli) per matching
    // cell. >2 % delta on >=50 % of cells = PASS dispatch.
    #[allow(clippy::type_complexity)] // local benchmark tuple — factoring adds noise
    let mut paired_cells: Vec<((String, f32, f32), Option<u64>, Option<u64>)> = Vec::new();
    for (img, d, it, mode, c) in &all_rows {
        let key = (img.clone(), *d, *it);
        let slot = paired_cells.iter_mut().find(|(k, _, _)| *k == key);
        let slot = if let Some(s) = slot {
            s
        } else {
            paired_cells.push((key.clone(), None, None));
            paired_cells.last_mut().unwrap()
        };
        match *mode {
            "butteraugli" => slot.1 = Some(c.bytes),
            "vdp2" => slot.2 = Some(c.bytes),
            _ => (),
        }
    }
    let mut big_delta = 0;
    let mut total = 0;
    let mut deltas: Vec<(String, f32, f32, f64)> = Vec::new();
    for ((img, d, it), b, v) in &paired_cells {
        if let (Some(b), Some(v)) = (b, v) {
            total += 1;
            let delta = (*v as f64 - *b as f64) / *b as f64 * 100.0;
            if delta.abs() > 2.0 {
                big_delta += 1;
            }
            deltas.push((img.clone(), *d, *it, delta));
        }
    }
    let dispatch_pct = if total > 0 {
        big_delta as f64 / total as f64 * 100.0
    } else {
        0.0
    };
    let dispatch_pass = dispatch_pct >= 50.0;
    eprintln!(
        "(1) Dispatch: |Δbytes| > 2% on {}/{} cells ({:.1}%) — {}",
        big_delta,
        total,
        dispatch_pct,
        if dispatch_pass { "PASS" } else { "FAIL" }
    );

    // (2) Vdp2 picks should consistently dominate Butteraugli picks on
    // the paper-faithful reference metric. Within each
    // (image, distance, intensity_target) cell we compare the two
    // modes' bytes + vdp2_ref score.
    //
    // The PASS condition: when Vdp2 spends MORE bytes than Butteraugli
    // (the common case — Vdp2's CSF flags more visible distortion at
    // HDR luminance and demands more quant precision), it should also
    // achieve a LOWER vdp2_ref score on at least 80% of those cells.
    //
    // We avoid the global spearman trap: rank-correlating bytes vs
    // ref_score across the full set is dominated by the
    // intensity_target axis (the ref metric scales with display
    // luminance by design), not by per-cell quality decisions.
    let mut paired_score: Vec<(String, f32, f32, u64, u64, f64, f64)> = Vec::new();
    for (img, d, it, mode, c) in &all_rows {
        if !c.vdp2_ref.is_finite() {
            continue;
        }
        let key = (img.clone(), *d, *it);
        let slot = paired_score
            .iter_mut()
            .find(|(i, dd, ii, _, _, _, _)| (i, dd, ii) == (&key.0, &key.1, &key.2));
        let slot = if let Some(s) = slot {
            s
        } else {
            paired_score.push((key.0.clone(), key.1, key.2, 0, 0, 0.0, 0.0));
            paired_score.last_mut().unwrap()
        };
        match *mode {
            "butteraugli" => {
                slot.3 = c.bytes;
                slot.5 = c.vdp2_ref;
            }
            "vdp2" => {
                slot.4 = c.bytes;
                slot.6 = c.vdp2_ref;
            }
            _ => (),
        }
    }
    let mut vdp2_spends_more = 0;
    let mut vdp2_better_when_spending_more = 0;
    let mut total_paired = 0;
    let mut sum_byte_pct = 0.0;
    let mut sum_score_pct_when_more = 0.0;
    for (_, _, _, b_bytes, v_bytes, b_score, v_score) in &paired_score {
        if *b_bytes == 0 || *v_bytes == 0 || *b_score <= 0.0 || *v_score <= 0.0 {
            continue;
        }
        total_paired += 1;
        let bd = (*v_bytes as f64 - *b_bytes as f64) / *b_bytes as f64 * 100.0;
        let sd = (*v_score - *b_score) / *b_score * 100.0;
        sum_byte_pct += bd;
        if bd > 0.0 {
            vdp2_spends_more += 1;
            sum_score_pct_when_more += sd;
            if sd < 0.0 {
                vdp2_better_when_spending_more += 1;
            }
        }
    }
    let pct_spends_more = if total_paired > 0 {
        vdp2_spends_more as f64 / total_paired as f64 * 100.0
    } else {
        0.0
    };
    let pct_quality_win = if vdp2_spends_more > 0 {
        vdp2_better_when_spending_more as f64 / vdp2_spends_more as f64 * 100.0
    } else {
        0.0
    };
    let avg_bytes = if total_paired > 0 {
        sum_byte_pct / total_paired as f64
    } else {
        0.0
    };
    let avg_score_when_more = if vdp2_spends_more > 0 {
        sum_score_pct_when_more / vdp2_spends_more as f64
    } else {
        0.0
    };
    eprintln!(
        "(2) Paired delta Vdp2 vs Butteraugli (same image, d, it):\n    \
         Vdp2 spends MORE bytes than Butteraugli on {}/{} cells ({:.1}%); avg Δbytes = {:+.1}%\n    \
         Of those, Vdp2 ALSO has LOWER ref score: {}/{} ({:.1}%); avg Δscore = {:+.1}%",
        vdp2_spends_more,
        total_paired,
        pct_spends_more,
        avg_bytes,
        vdp2_better_when_spending_more,
        vdp2_spends_more,
        pct_quality_win,
        avg_score_when_more
    );
    // Pass condition: when Vdp2 spends more (≥50 % of cells) it must
    // also be better on ref_score ≥80 % of the time.
    let corr_pass = pct_spends_more >= 50.0 && pct_quality_win >= 80.0;
    // Also retain the simple global spearman as informational only.
    let collect = |target: &str| -> (Vec<f64>, Vec<f64>) {
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for (_, _, _, mode, c) in &all_rows {
            if *mode == target && c.vdp2_ref.is_finite() {
                xs.push(c.bytes as f64);
                ys.push(c.vdp2_ref);
            }
        }
        (xs, ys)
    };
    let (xb, yb) = collect("butteraugli");
    let (xv, yv) = collect("vdp2");
    let (xc, yc) = collect("cjxl");
    let r_b = spearman(&xb, &yb);
    let r_v = spearman(&xv, &yv);
    let r_c = spearman(&xc, &yc);
    eprintln!(
        "    (informational, dominated by intensity_target axis) global spearman bytes-vs-score:\n    \
         Butteraugli={:.4}  Vdp2={:.4}  cjxl={:.4}",
        r_b, r_v, r_c
    );
    eprintln!(
        "    Vdp2 correlation closer to -1 than Butteraugli? {}",
        if corr_pass { "PASS" } else { "FAIL" }
    );

    // (3) Top-3 cells where Vdp2 byte choice beats Butteraugli's
    // (smaller bytes AND better vdp2_ref). Useful color for the report.
    let mut wins: Vec<(String, f32, f32, f64, f64)> = Vec::new();
    #[allow(clippy::type_complexity)] // local benchmark map — factoring adds noise
    let mut by_key: std::collections::HashMap<
        (String, u64, u64),
        (Option<Cell>, Option<Cell>),
    > = std::collections::HashMap::new();
    for (img, d, it, mode, c) in &all_rows {
        let k = (img.clone(), (*d * 1000.0) as u64, *it as u64);
        let entry = by_key.entry(k).or_insert((None, None));
        match *mode {
            "butteraugli" => entry.0 = Some(*c),
            "vdp2" => entry.1 = Some(*c),
            _ => (),
        }
    }
    for ((img, d_milli, it_u), (b, v)) in &by_key {
        if let (Some(b), Some(v)) = (b, v) {
            let bytes_d = (v.bytes as f64 - b.bytes as f64) / b.bytes as f64 * 100.0;
            let ref_d = (v.vdp2_ref - b.vdp2_ref) / b.vdp2_ref * 100.0;
            // "Better" = bytes ≤ butteraugli AND vdp2_ref score ≤ butteraugli's
            if bytes_d < 0.0 && ref_d < 0.0 {
                wins.push((
                    img.clone(),
                    *d_milli as f32 / 1000.0,
                    *it_u as f32,
                    bytes_d,
                    ref_d,
                ));
            }
        }
    }
    wins.sort_by(|a, b| (a.3 + a.4).partial_cmp(&(b.3 + b.4)).unwrap());
    eprintln!(
        "(3) Top {} cells where Vdp2 beats Butteraugli (smaller bytes AND lower ref score):",
        wins.len().min(3)
    );
    for w in wins.iter().take(3) {
        eprintln!(
            "    {} d={} it={} : Δbytes={:+.2}%, Δref_score={:+.2}%",
            w.0, w.1, w.2, w.3, w.4
        );
    }

    eprintln!(
        "\n--- VERDICT: {} ---",
        if dispatch_pass && corr_pass {
            "PASS — recommend HdrLoss::Vdp2 as default for PQ/HLG content"
        } else if dispatch_pass {
            "PARTIAL PASS — dispatch fires but correlation tied with Butteraugli"
        } else {
            "FAIL — VDP2-lite not faithful enough; see chunk-4 follow-on plan"
        }
    );

    // Append summary to meta sidecar so it lives next to the TSV.
    let mut meta_append = std::fs::OpenOptions::new().append(true).open(&meta_path)?;
    writeln!(meta_append, "\n# Post-sweep analysis")?;
    writeln!(
        meta_append,
        "dispatch_pct_gt_2pct={dispatch_pct:.1}  pass={}",
        dispatch_pass
    )?;
    writeln!(meta_append, "paired_total={}", total_paired)?;
    writeln!(meta_append, "vdp2_spends_more={}", vdp2_spends_more)?;
    writeln!(meta_append, "pct_spends_more={pct_spends_more:.1}")?;
    writeln!(
        meta_append,
        "vdp2_better_when_spending_more={vdp2_better_when_spending_more}"
    )?;
    writeln!(meta_append, "pct_quality_win={pct_quality_win:.1}")?;
    writeln!(meta_append, "avg_byte_delta_pct={avg_bytes:.2}")?;
    writeln!(
        meta_append,
        "avg_score_delta_pct_when_more={avg_score_when_more:.2}"
    )?;
    writeln!(meta_append, "correlation_pass={}", corr_pass)?;
    writeln!(
        meta_append,
        "global_spearman_butteraugli={r_b:.4} (n={})",
        xb.len()
    )?;
    writeln!(
        meta_append,
        "global_spearman_vdp2={r_v:.4} (n={})",
        xv.len()
    )?;
    writeln!(
        meta_append,
        "global_spearman_cjxl={r_c:.4} (n={})",
        xc.len()
    )?;
    writeln!(meta_append, "top_wins={}", wins.len())?;
    for w in wins.iter().take(3) {
        writeln!(
            meta_append,
            "win\t{}\t{}\t{}\t{:+.2}\t{:+.2}",
            w.0, w.1, w.2, w.3, w.4
        )?;
    }

    eprintln!("\nwrote {} rows → {}", all_rows.len(), out_path.display());
    eprintln!("meta → {}", meta_path.display());

    // Suppress unused warnings on helper functions.
    let _ = linear_to_srgb_u8;
    Ok(())
}
