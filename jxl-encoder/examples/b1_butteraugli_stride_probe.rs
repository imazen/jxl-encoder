//! B1 candidate stride-bug probe.
//!
//! Hypothesis (audit Candidate 1, /benchmarks/buttloop_recon_audit_2026-05-23/
//! buttloop_vs_postdecode.md): butteraugli reference is built with
//! `stride = width` at vardct/butteraugli_loop.rs:1257, then compared with
//! `stride = padded_width` at line 2270. When `width % 8 != 0`, padded_width
//! exceeds width by up to 7, allegedly causing the buffer to be misinterpreted.
//!
//! This probe directly tests that hypothesis WITHOUT invoking the encoder.
//! We construct synthetic reference and "distorted" planar buffers in the
//! exact same shape the buttloop uses, then exercise the four cells of the
//! `(ref-stride, compare-stride)` truth table and compare scores.
//!
//! Truth table:
//!   A: ref stride=W, compare stride=W (both unpadded, control)
//!   B: ref stride=W, compare stride=PW (what production does — UNDER TEST)
//!   C: ref stride=PW, compare stride=PW (both padded; "candidate fix")
//!   D: ref stride=W, compare stride=PW with cropped recon (alt fix)
//!
//! If A == B (within float tolerance), the audit's hypothesis is WRONG:
//! the planar stride API is correctly accommodating layout differences,
//! and the bug lives elsewhere.
//!
//! If A != B, the bug is real and we measure its magnitude.

use butteraugli::{ButteraugliParams, ButteraugliReference};

fn make_ramp_planar(
    width: usize,
    height: usize,
    stride: usize,
    bias: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut r = vec![0.0f32; stride * height];
    let mut g = vec![0.0f32; stride * height];
    let mut b = vec![0.0f32; stride * height];
    for y in 0..height {
        for x in 0..width {
            let off = y * stride + x;
            let u = (x as f32 + 0.5) / (width as f32);
            let v = (y as f32 + 0.5) / (height as f32);
            r[off] = (u + bias * 0.01).clamp(0.0, 1.0);
            g[off] = (v + bias * 0.013).clamp(0.0, 1.0);
            b[off] = ((u * v).sqrt() + bias * 0.007).clamp(0.0, 1.0);
        }
        // padded region (x = width..stride): leave as 0.0 — this mimics the
        // production case where padded columns hold whatever recon emitted
        // (often zeros for the first few iters; this experiment is about the
        // METRIC sensitivity to whether the metric LOOKS AT those columns).
    }
    (r, g, b)
}

fn pad_width(width: usize) -> usize {
    width.div_ceil(8) * 8
}

#[derive(Debug)]
struct ScoreRow {
    label: &'static str,
    ref_stride: &'static str,
    cmp_stride: &'static str,
    score: f64,
}

fn run_case(
    width: usize,
    height: usize,
    bias: f32,
) -> Vec<ScoreRow> {
    let pw = pad_width(width);
    let _ph = height; // height is always exact in this probe

    // Reference contiguous buffers (stride = width)
    let (ref_r_w, ref_g_w, ref_b_w) = make_ramp_planar(width, height, width, 0.0);
    // Reference padded buffers (stride = pw); padded columns left as 0.0
    let (ref_r_pw, ref_g_pw, ref_b_pw) = make_ramp_planar(width, height, pw, 0.0);
    // Distorted: same shape as reference with a small bias (so the score
    // is non-zero and stable but not catastrophic). Build both layouts.
    let (dst_r_w, dst_g_w, dst_b_w) = make_ramp_planar(width, height, width, bias);
    let (dst_r_pw, dst_g_pw, dst_b_pw) = make_ramp_planar(width, height, pw, bias);

    let params = || {
        ButteraugliParams::new()
            .with_intensity_target(80.0)
            .with_compute_diffmap(false)
    };

    let mut out = Vec::new();

    // Case A: ref stride=W, compare stride=W (both unpadded). Control.
    let r_a = ButteraugliReference::new_linear_planar(
        &ref_r_w, &ref_g_w, &ref_b_w, width, height, width, params(),
    )
    .expect("A ref");
    let s_a = r_a
        .compare_linear_planar(&dst_r_w, &dst_g_w, &dst_b_w, width)
        .expect("A cmp")
        .score;
    out.push(ScoreRow {
        label: "A_both_unpadded",
        ref_stride: "W",
        cmp_stride: "W",
        score: s_a,
    });

    // Case B: ref stride=W, compare stride=PW. WHAT PRODUCTION DOES.
    // Note: padded columns of the distorted buffer hold 0.0, which differ
    // from the corresponding columns of the reference (which doesn't have
    // those columns at all — its rows end at x=width). Per butteraugli's
    // stride semantics, both sides should ONLY read the first `width`
    // pixels per row. If audit is right, score would diverge from A.
    let r_b = ButteraugliReference::new_linear_planar(
        &ref_r_w, &ref_g_w, &ref_b_w, width, height, width, params(),
    )
    .expect("B ref");
    let s_b = r_b
        .compare_linear_planar(&dst_r_pw, &dst_g_pw, &dst_b_pw, pw)
        .expect("B cmp")
        .score;
    out.push(ScoreRow {
        label: "B_production",
        ref_stride: "W",
        cmp_stride: "PW",
        score: s_b,
    });

    // Case C: ref stride=PW, compare stride=PW (both padded, padded cols 0.0).
    // Symmetric: both sides see same padded-zero columns, but reference also
    // sees them so they cancel. If audit's "build reference with padded_width"
    // alternate fix is right, this should match A more closely than B does.
    let r_c = ButteraugliReference::new_linear_planar(
        &ref_r_pw, &ref_g_pw, &ref_b_pw, width, height, pw, params(),
    )
    .expect("C ref");
    let s_c = r_c
        .compare_linear_planar(&dst_r_pw, &dst_g_pw, &dst_b_pw, pw)
        .expect("C cmp")
        .score;
    out.push(ScoreRow {
        label: "C_both_padded",
        ref_stride: "PW",
        cmp_stride: "PW",
        score: s_c,
    });

    // Case D: ref stride=W, compare stride=W with manually-cropped recon.
    // This is the alt fix that drops the padded columns of the recon before
    // feeding to butteraugli. Should be identical to A if B == A (no-op).
    let mut crop_r = vec![0.0f32; width * height];
    let mut crop_g = vec![0.0f32; width * height];
    let mut crop_b = vec![0.0f32; width * height];
    for y in 0..height {
        for x in 0..width {
            let src = y * pw + x;
            let dst = y * width + x;
            crop_r[dst] = dst_r_pw[src];
            crop_g[dst] = dst_g_pw[src];
            crop_b[dst] = dst_b_pw[src];
        }
    }
    let r_d = ButteraugliReference::new_linear_planar(
        &ref_r_w, &ref_g_w, &ref_b_w, width, height, width, params(),
    )
    .expect("D ref");
    let s_d = r_d
        .compare_linear_planar(&crop_r, &crop_g, &crop_b, width)
        .expect("D cmp")
        .score;
    out.push(ScoreRow {
        label: "D_crop_to_W",
        ref_stride: "W",
        cmp_stride: "W*",
        score: s_d,
    });

    out
}

fn main() {
    // Cells that exercise both width % 8 == 0 (control) and width % 8 != 0
    // (the alleged failure regime).
    let cells: &[(usize, usize, f32)] = &[
        // (width, height, bias)
        (256, 256, 1.0),  // width % 8 == 0 -> padded_width == width (control)
        (250, 200, 1.0),  // width % 8 == 2 -> padded_width = 256, +6 padded cols
        (253, 253, 1.0),  // width % 8 == 5 -> padded_width = 256, +3 padded cols
        (1080, 720, 1.0), // width % 8 == 0 -> padded_width == width (large control)
        (1075, 715, 1.0), // width % 8 == 3 -> padded_width = 1080, +5 padded cols
        (513, 100, 0.5),  // width % 8 == 1 -> padded_width = 520, +7 padded cols (worst)
    ];

    println!("cell\twidth\theight\tpw\tlabel\tref_stride\tcmp_stride\tscore\tdelta_vs_A");

    for &(w, h, bias) in cells {
        let pw = pad_width(w);
        let rows = run_case(w, h, bias);
        let s_a = rows[0].score;
        for r in &rows {
            let d = r.score - s_a;
            println!(
                "{}x{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.6}\t{:+.6}",
                w, h, w, h, pw, r.label, r.ref_stride, r.cmp_stride, r.score, d,
            );
        }
    }

    // Verdict summary
    println!();
    println!("# Interpretation:");
    println!("# - If |B.score - A.score| < 1e-3 on EVERY non-multiple-of-8 cell:");
    println!("#   audit Candidate 1 is FALSIFIED — stride API is correct.");
    println!("# - If B.score != A.score on non-multiple-of-8 cells:");
    println!("#   bug is real, magnitude shown in delta_vs_A column.");
}
