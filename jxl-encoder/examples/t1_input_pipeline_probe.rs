//! T1-new: probe the linear-RGB REFERENCE images fed to butteraugli by the
//! buttloop-internal path vs the post-decode RD path.
//!
//! Hypothesis (paraphrased from the task spec): the buttloop's reference
//! image is built via `srgb_u8 → forward_xyb → inverse_xyb → linear-RGB`,
//! while the post-decode RD path's reference is built via `srgb_u8 →
//! srgb_to_linear → linear-RGB`. Same source pixels but different f32 linear
//! RGB representations, accumulating over millions of pixels.
//!
//! Source-reading audit BEFORE the probe (see memo):
//! - `vardct/butteraugli_loop.rs:1223-1264` builds the reference as a planar
//!   deinterleave of the caller-supplied `linear_rgb: &[f32]` — NO XYB
//!   round-trip. The reference is the encoder-input linear-RGB.
//! - `api.rs:12461-12472` (`srgb_u8_to_linear_f32`) is the only path that
//!   converts sRGB u8 → linear f32 inside the encoder. It uses a const-fn
//!   LUT (Newton's fifth-root in f64 → as f32 truncation).
//! - `tests/quality_compare.rs:175-184` constructs the post-decode REFERENCE
//!   via `butteraugli::srgb_to_linear` (LUT initialised lazily from
//!   `(v as f32 / 255.0).powf(2.4)` in f32 throughout — see
//!   `butteraugli/src/opsin.rs:309-326`).
//!
//! So the hypothesis is partially mis-stated: neither reference goes through
//! XYB round-trip. BUT the two sRGB→linear LUTs are DIFFERENT
//! implementations (jxl-encoder Newton's 5th-root in f64, butteraugli powf
//! in f32). The probe measures whether the LUT precision delta is the
//! W44-111/138 buttloop-vs-postdecode score gap.
//!
//! Outputs: pixel-diff stats + (optional) butteraugli score diff when fed
//! IDENTICAL distorted image with the two different references.
//!
//! Run:
//! ```
//! CARGO_TARGET_DIR=$HOME/work/zen/jxl-encoder-shared-target \
//!   cargo run --release --example t1_input_pipeline_probe \
//!     --features 'butteraugli-loop' \
//!     --manifest-path jxl-encoder/Cargo.toml
//! ```

use butteraugli::{ButteraugliParams, butteraugli_linear, srgb_to_linear as bu_srgb_to_linear};
use rgb::RGB;
use std::path::PathBuf;

/// Mirror the jxl-encoder `SRGB_U8_TO_LINEAR` LUT from `api.rs:12413-12453`.
/// We can't access the private const directly, so we recompute it bit-for-bit
/// using the same algorithm (Newton's 5th root in f64, then truncate to f32).
fn jxl_encoder_srgb_to_linear_lut() -> [f32; 256] {
    let mut table = [0.0f32; 256];
    for i in 0..256u16 {
        let c = i as f64 / 255.0;
        table[i as usize] = if c <= 0.04045 {
            (c / 12.92) as f32
        } else {
            let base = (c + 0.055) / 1.055;
            let x2 = base * base;
            let x4 = x2 * x2;
            let x8 = x4 * x4;
            let x12 = x8 * x4;
            // Newton fifth root for x^(12/5) = x^2.4
            let mut y = base * base;
            for _ in 0..8 {
                let y2 = y * y;
                let y4 = y2 * y2;
                y = (4.0 * y + x12 / y4) / 5.0;
            }
            y as f32
        };
    }
    table
}

/// Path A: jxl-encoder's LUT (Newton's 5th-root in f64).
fn srgb_to_linear_jxl(lut: &[f32; 256], v: u8) -> f32 {
    lut[v as usize]
}

/// Path B: butteraugli crate's LUT (powf(2.4) in f32).
fn srgb_to_linear_butteraugli(v: u8) -> f32 {
    bu_srgb_to_linear(v)
}

/// Stats over (a - b) absolute differences per channel.
struct DiffStats {
    max_abs: f32,
    mean_abs: f32,
    n_above_1em4: usize,
    n_above_1em5: usize,
    n_above_1em6: usize,
    n_total: usize,
}

fn diff_stats(a: &[f32], b: &[f32]) -> DiffStats {
    assert_eq!(a.len(), b.len());
    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f64;
    let mut n_above_1em4 = 0usize;
    let mut n_above_1em5 = 0usize;
    let mut n_above_1em6 = 0usize;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (x - y).abs();
        if d > max_abs {
            max_abs = d;
        }
        sum_abs += d as f64;
        if d > 1e-4 {
            n_above_1em4 += 1;
        }
        if d > 1e-5 {
            n_above_1em5 += 1;
        }
        if d > 1e-6 {
            n_above_1em6 += 1;
        }
    }
    DiffStats {
        max_abs,
        mean_abs: (sum_abs / a.len() as f64) as f32,
        n_above_1em4,
        n_above_1em5,
        n_above_1em6,
        n_total: a.len(),
    }
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from("/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1418519.png")
        });

    eprintln!("# T1-new: input-pipeline probe");
    eprintln!("# source: {}", path.display());

    let img = match image::open(&path) {
        Ok(i) => i.to_rgb8(),
        Err(e) => {
            eprintln!("error: could not open {}: {e:?}", path.display());
            std::process::exit(1);
        }
    };
    let (w, h) = img.dimensions();
    eprintln!("# dimensions: {w}×{h}");

    // Crop to 256×256 from top-left for a more controlled probe (if larger).
    let (probe_w, probe_h) = if w >= 256 && h >= 256 {
        (256u32, 256u32)
    } else {
        (w, h)
    };
    eprintln!("# probe region: {probe_w}×{probe_h}");

    // Extract sRGB u8 from probe region.
    let mut srgb: Vec<u8> = Vec::with_capacity((probe_w * probe_h * 3) as usize);
    for y in 0..probe_h {
        for x in 0..probe_w {
            let p = img.get_pixel(x, y);
            srgb.push(p[0]);
            srgb.push(p[1]);
            srgb.push(p[2]);
        }
    }

    // Build linear-RGB via each path.
    let jxl_lut = jxl_encoder_srgb_to_linear_lut();

    let linear_jxl: Vec<f32> = srgb
        .iter()
        .map(|&v| srgb_to_linear_jxl(&jxl_lut, v))
        .collect();

    let linear_bu: Vec<f32> = srgb
        .iter()
        .map(|&v| srgb_to_linear_butteraugli(v))
        .collect();

    // Per-LUT-entry comparison: 256 entries × 2 LUTs (verify the LUT itself).
    eprintln!("\n# Per-LUT-entry comparison (256 entries)");
    let mut lut_max = 0.0f32;
    let mut lut_n_diff = 0usize;
    for i in 0..=255u8 {
        let a = srgb_to_linear_jxl(&jxl_lut, i);
        let b = srgb_to_linear_butteraugli(i);
        let d = (a - b).abs();
        if d > lut_max {
            lut_max = d;
        }
        if d > 0.0 {
            lut_n_diff += 1;
        }
        if d > 1e-6 {
            eprintln!("#   lut[{i:3}] jxl={a:.10e} bu={b:.10e} d={d:.3e}");
        }
    }
    eprintln!("# LUT max abs diff: {lut_max:.3e}");
    eprintln!("# LUT entries differing at all: {lut_n_diff}/256");

    // Pixel-diff stats over the probe region.
    let stats = diff_stats(&linear_jxl, &linear_bu);
    eprintln!("\n# Pixel-by-pixel (linear-RGB) diff stats");
    eprintln!("#   n_total      : {}", stats.n_total);
    eprintln!("#   max abs diff : {:.3e}", stats.max_abs);
    eprintln!("#   mean abs diff: {:.3e}", stats.mean_abs);
    eprintln!(
        "#   > 1e-4       : {} ({:.4}%)",
        stats.n_above_1em4,
        100.0 * stats.n_above_1em4 as f64 / stats.n_total as f64
    );
    eprintln!(
        "#   > 1e-5       : {} ({:.4}%)",
        stats.n_above_1em5,
        100.0 * stats.n_above_1em5 as f64 / stats.n_total as f64
    );
    eprintln!(
        "#   > 1e-6       : {} ({:.4}%)",
        stats.n_above_1em6,
        100.0 * stats.n_above_1em6 as f64 / stats.n_total as f64
    );

    // Decision threshold per task spec:
    // - If both paths produce CLOSE-TO-IDENTICAL linear-RGB (< 1e-5 max abs
    //   diff): hypothesis falsified, no butteraugli probe needed.
    // - If MEANINGFULLY different (> 1e-3 max abs diff): hypothesis confirmed
    //   at one layer; measure butteraugli-score divergence.
    let cross_falsified = stats.max_abs < 1e-5;
    let cross_significant = stats.max_abs > 1e-3;
    eprintln!("\n# Pixel-level verdict");
    eprintln!("#   max_abs < 1e-5 (falsified): {cross_falsified}");
    eprintln!("#   max_abs > 1e-3 (significant): {cross_significant}");

    // Butteraugli probe: feed the SAME distorted image with the two
    // different reference linear-RGBs and see if scores differ.
    //
    // For a "distorted" image, perturb the linear-RGB by a small amount
    // (~5% multiplicative noise per pixel). This simulates a JXL encode
    // distortion: small uniform error that butteraugli should measure
    // consistently regardless of which precision the reference uses.
    let probe_w_us = probe_w as usize;
    let probe_h_us = probe_h as usize;

    // Build a "distortion" by quantizing each channel to 5-bit (≈ what
    // d=4 JXL would do): floor((v * 32).round()) / 32. This produces a
    // distorted image that is identical regardless of which reference we
    // compare it against.
    let distort = |lin: f32| -> f32 {
        let q = (lin * 32.0).round() / 32.0;
        q.clamp(0.0, 1.0)
    };
    // Distorted image = quantize the IDENTICAL underlying source via either
    // reference's linear domain. We use the jxl path arbitrarily; what
    // matters is the distorted image is identical in both butteraugli calls.
    let distorted: Vec<f32> = linear_jxl.iter().map(|&v| distort(v)).collect();

    // To avoid biasing, also make a butteraugli-path distorted image
    // separately. Then both butteraugli calls compare (ref_jxl vs
    // distorted_jxl) and (ref_bu vs distorted_jxl). The DELTA in butteraugli
    // scores is the impact of the reference precision difference.
    let to_rgb = |v: &[f32]| -> Vec<RGB<f32>> {
        v.chunks_exact(3)
            .map(|c| RGB::new(c[0], c[1], c[2]))
            .collect()
    };

    let ref_jxl_rgb = to_rgb(&linear_jxl);
    let ref_bu_rgb = to_rgb(&linear_bu);
    let dist_rgb = to_rgb(&distorted);

    use imgref::Img;
    let ref_jxl_img = Img::new(ref_jxl_rgb.clone(), probe_w_us, probe_h_us);
    let ref_bu_img = Img::new(ref_bu_rgb.clone(), probe_w_us, probe_h_us);
    let dist_img = Img::new(dist_rgb, probe_w_us, probe_h_us);

    let params = ButteraugliParams::default();

    let score_jxl_ref = butteraugli_linear(ref_jxl_img.as_ref(), dist_img.as_ref(), &params)
        .expect("butteraugli jxl-ref failed");
    let score_bu_ref = butteraugli_linear(ref_bu_img.as_ref(), dist_img.as_ref(), &params)
        .expect("butteraugli bu-ref failed");

    eprintln!("\n# Butteraugli-score probe (SAME distorted, DIFFERENT references)");
    eprintln!(
        "#   score (jxl-LUT ref)         : {:.6}",
        score_jxl_ref.score
    );
    eprintln!(
        "#   score (butteraugli-LUT ref) : {:.6}",
        score_bu_ref.score
    );
    let score_delta = (score_jxl_ref.score - score_bu_ref.score).abs();
    let score_pct = 100.0 * score_delta / score_bu_ref.score.max(1e-9);
    eprintln!("#   absolute delta              : {:.6e}", score_delta);
    eprintln!("#   relative delta              : {:.4}%", score_pct);

    // Final verdict
    eprintln!("\n# === FINAL VERDICT ===");
    if cross_falsified {
        eprintln!("# HYPOTHESIS FALSIFIED at pixel layer.");
        eprintln!("# The two sRGB→linear LUTs produce essentially-identical");
        eprintln!("# linear-RGB (max abs diff {:.3e} < 1e-5).", stats.max_abs);
        eprintln!("# Reference-construction precision is NOT the W44-111/138 gap.");
    } else if score_pct < 1.0 {
        eprintln!("# HYPOTHESIS FALSIFIED at butteraugli-score layer.");
        eprintln!(
            "# Pixel diff is meaningful (max abs {:.3e}) but butteraugli",
            stats.max_abs
        );
        eprintln!("# score delta is < 1% ({score_pct:.4}%). Reference precision");
        eprintln!("# does not move the metric.");
    } else if score_pct >= 1.0 {
        eprintln!("# HYPOTHESIS CONFIRMED.");
        eprintln!("# Score delta of {score_pct:.4}% from reference-LUT precision");
        eprintln!("# alone. Aligning buttloop's reference-construction to use");
        eprintln!("# butteraugli::srgb_to_linear is a candidate fix.");
    }

    // TSV row for downstream tooling.
    println!(
        "src\tprobe_w\tprobe_h\tn\tlut_max_diff\tpx_max_diff\tpx_mean_diff\tpx_above_1em4\tpx_above_1em5\tpx_above_1em6\tscore_jxl_ref\tscore_bu_ref\tscore_delta\tscore_pct"
    );
    println!(
        "{}\t{probe_w}\t{probe_h}\t{}\t{:.3e}\t{:.3e}\t{:.3e}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{:.3e}\t{:.4}",
        path.display(),
        stats.n_total,
        lut_max,
        stats.max_abs,
        stats.mean_abs,
        stats.n_above_1em4,
        stats.n_above_1em5,
        stats.n_above_1em6,
        score_jxl_ref.score,
        score_bu_ref.score,
        score_delta,
        score_pct,
    );
}
