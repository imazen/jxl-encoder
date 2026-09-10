//! How much of the forward XYB transform is the cube root?
//!
//! The e3 profile (`benchmarks/e3_profile_2026-09-10.md`) puts
//! `convert_rows_to_xyb` at 15.3 % of lossy e3 CPU — the single largest
//! consumer, ahead of every entropy-coding symbol. `jxl-encoder-simd`'s
//! `forward_xyb_impl` is nominally SIMD, but its cube root is two Newton
//! iterations in **f64, each with a division**, run three-channels-at-a-time
//! through `to_array()` / `from_array()` round-trips.
//!
//! This measures the ceiling, it does not propose a change. Arms:
//!   shipped  — `jxl_simd::linear_rgb_to_xyb_batch` exactly as it ships.
//!   f32_cbrt — the same matrix+bias, but the cube root done the way libjxl
//!              does it (`enc_xyb.cc` CubeRootAndAdd): bit-hack guess and
//!              Newton entirely in f32. NOT bit-identical to shipped — it is
//!              here to price the f64 choice, nothing more.
//!   matrix   — matrix + bias + clamp only, cube root skipped entirely. The
//!              hard floor.
//!
//! Run: `cargo run --release --example xyb_cbrt_headroom -- [pixels] [reps]`
use std::time::Instant;

const OPSIN: [[f32; 3]; 3] = [
    [0.300_00, 0.622_00, 0.078_00],
    [0.230_00, 0.692_00, 0.078_00],
    [0.243_422_69, 0.204_767_45, 0.551_809_86],
];
const BIAS: f32 = 0.003_793_073_4;
const NEG_CBRT_BIAS: f32 = -0.155_954_2;

#[inline]
fn cbrt_f64_newton(x: f32) -> f32 {
    if x == 0.0 {
        return 0.0;
    }
    const B1: u32 = 709_958_130;
    let ui = x.to_bits();
    let approx = (ui & 0x7FFF_FFFF) / 3 + B1;
    let mut t = f64::from(f32::from_bits((ui & 0x8000_0000) | approx));
    let xf = f64::from(x);
    let r = t * t * t;
    t = t * (xf + xf + r) / (xf + r + r);
    let r = t * t * t;
    t = t * (xf + xf + r) / (xf + r + r);
    t as f32
}

/// libjxl's cube root (`base/fast_math-inl.h::CubeRootAndAdd`), scalar
/// transcription. The point of interest is that it is **division-free**: it
/// Newtons the INVERSE cube root with multiplies and FMAs only, then recovers
/// `x^(1/3)` as `r*r*x`. Ours does two Newton iterations on the cube root
/// directly, each with an f64 division.
#[inline]
fn cbrt_libjxl(x: f32) -> f32 {
    const K_EXP_BIAS: i32 = 0x5480_0000;
    const K_EXP_MUL: i32 = 0x002A_AAAA;
    let xa_3 = x * (1.0 / 3.0);
    let m1 = x.to_bits() as i32;
    let m2 = if m1 == 0 {
        0
    } else {
        K_EXP_BIAS - ((m1 >> 23) * K_EXP_MUL)
    };
    let mut r = f32::from_bits(m2 as u32);
    for _ in 0..3 {
        let r2 = r * r;
        r = (4.0 / 3.0) * r - xa_3 * (r2 * r2);
    }
    let r2 = r * r;
    r = r + (1.0 / 3.0) * (r - x * (r2 * r2));
    let r2 = r * r;
    r2 * x
}

fn local(
    r: &[f32],
    g: &[f32],
    b: &[f32],
    xo: &mut [f32],
    yo: &mut [f32],
    bo: &mut [f32],
    mode: u8,
) {
    // Writes all THREE output planes, like the shipped kernel — an arm that
    // stores one plane instead of three is not a comparable arm.
    for i in 0..r.len() {
        let mut m = [0.0f32; 3];
        for (c, mc) in m.iter_mut().enumerate() {
            *mc = (OPSIN[c][0] * r[i] + OPSIN[c][1] * g[i] + OPSIN[c][2] * b[i] + BIAS).max(0.0);
        }
        let cb = match mode {
            0 => [
                cbrt_f64_newton(m[0]),
                cbrt_f64_newton(m[1]),
                cbrt_f64_newton(m[2]),
            ],
            1 => [cbrt_libjxl(m[0]), cbrt_libjxl(m[1]), cbrt_libjxl(m[2])],
            _ => m,
        };
        // Same shape as the real opsin tail: two differences and a sum.
        xo[i] = (cb[0] - cb[1]) * 0.5;
        yo[i] = (cb[0] + cb[1]) * 0.5 + NEG_CBRT_BIAS;
        bo[i] = cb[2] + NEG_CBRT_BIAS;
    }
}

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1 << 20);
    let reps: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    // Deterministic pseudo-image; the cbrt cost is data-independent apart from
    // the x == 0 early-out, so keep a realistic fraction of exact zeros.
    let mut seed = 0x9E37_79B9u32;
    let mut nextf = || {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let v = (seed >> 8) as f32 / 16_777_215.0;
        if v < 0.02 { 0.0 } else { v }
    };
    let r: Vec<f32> = (0..n).map(|_| nextf()).collect();
    let g: Vec<f32> = (0..n).map(|_| nextf()).collect();
    let b: Vec<f32> = (0..n).map(|_| nextf()).collect();
    let (mut xo, mut yo, mut bo) = (vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n]);
    let (mut sx, mut sy, mut sb) = (vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n]);

    let mut best = [f64::INFINITY; 4];
    let labels = [
        "shipped (jxl_simd, f64 Newton)",
        "scalar f64 Newton",
        "scalar libjxl f32 (div-free)",
        "matrix only (no cbrt)",
    ];
    for rep in 0..reps {
        for k in 0..4 {
            let arm = (k + rep) % 4;
            let t = Instant::now();
            match arm {
                0 => jxl_simd::linear_rgb_to_xyb_batch(&r, &g, &b, &mut xo, &mut yo, &mut bo),
                1 => local(&r, &g, &b, &mut sx, &mut sy, &mut sb, 0),
                2 => local(&r, &g, &b, &mut sx, &mut sy, &mut sb, 1),
                _ => local(&r, &g, &b, &mut sx, &mut sy, &mut sb, 2),
            }
            let ms = t.elapsed().as_secs_f64() * 1000.0;
            best[arm] = best[arm].min(ms);
        }
    }
    println!("{n} px, min of {reps}, single-threaded:");
    for k in 0..4 {
        println!(
            "  {:<32} {:>8.2} ms  ({:>5.2} ns/px)",
            labels[k],
            best[k],
            best[k] * 1e6 / n as f64
        );
    }
    println!(
        "\n  cube root is {:.0}% of the scalar transform.\n  \
         libjxl's division-free f32 cbrt costs {:.0}% of our f64 two-divide cbrt\n  \
         (scalar arms, same loop and the same three output planes).",
        100.0 * (best[1] - best[3]) / best[1],
        100.0 * (best[2] - best[3]) / (best[1] - best[3])
    );
    // Accuracy of the cheap arm against the shipped one, on the real range.
    let mut worst = 0.0f64;
    let mut probe = 1e-6f32;
    while probe < 8.0 {
        let a = cbrt_f64_newton(probe) as f64;
        let b = cbrt_libjxl(probe) as f64;
        if a != 0.0 {
            worst = worst.max(((a - b) / a).abs());
        }
        probe *= 1.0009;
    }
    println!("  max relative disagreement over x in [1e-6, 8]: {worst:.3e}");
    std::hint::black_box((&xo, &yo, &bo, &sx, &sy, &sb));
}
