//! Per-kernel NEON-vs-scalar for jxl-encoder-simd.
//!
//! This crate carries 714 SIMD dispatch sites — the largest such surface in the
//! zen workspace — and the only tier measurement was END-TO-END whole-image
//! encode (2.07x lossy / 1.18x lossless). A whole-pipeline ratio cannot show an
//! individual kernel running BELOW its scalar reference: in zenresize an
//! end-to-end 1.56x hid a kernel at 0.94x, and in zenquant a healthy
//! whole-quantize number hid a palette search at 0.58x.
//!
//! The crate publishes its `_neon` and `_scalar` variants directly, so this
//! compares them head-to-head rather than toggling the dispatch token — a
//! sharper measurement, since dispatch overhead is out of both arms.
//!
//! NEON is BASELINE on aarch64, so the scalar arm is autovectorized too:
//! ~1.00x means LLVM already matched the hand-written kernel; BELOW 1.00 means
//! the hand-written one is a liability.
//!
//! Run: `cargo bench -p jxl-encoder-simd --bench kernel_tiers`

use zenbench::prelude::*;

fn ramp(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (s >> 8) as f32 / 16_777_216.0
        })
        .collect()
}

#[cfg(target_arch = "aarch64")]
fn bench_kernels(suite: &mut Suite) {
    use archmage::SimdToken;
    let Some(t) = archmage::NeonToken::summon() else {
        eprintln!("[kernel_tiers] no NEON token; skipping");
        return;
    };

    macro_rules! dct {
        ($name:expr, $n:expr, $neon:path, $scalar:path) => {
            suite.compare($name, |g| {
                let v = ramp($n, 7);
                let arr: [f32; $n] = v.try_into().unwrap();
                let inp: &'static [f32; $n] = Box::leak(Box::new(arr));
                g.bench("neon", move |b| {
                    let mut out = Box::new([0f32; $n]);
                    b.iter(move || $neon(t, inp, &mut out))
                });
                g.bench("scalar", move |b| {
                    let mut out = Box::new([0f32; $n]);
                    b.iter(move || $scalar(inp, &mut out))
                });
            });
        };
    }

    dct!("dct_8x8", 64, jxl_encoder_simd::dct_8x8_neon, jxl_encoder_simd::dct_8x8_scalar);
    dct!("dct_16x16", 256, jxl_encoder_simd::dct_16x16_neon, jxl_encoder_simd::dct_16x16_scalar);
    // NOTE dct_32x32 / idct_32x32 / dct_64x64 are absent: their `_neon`
    // variants exist but are not re-exported from lib.rs (internal dispatch
    // still reaches them), so a bench cannot name them.

    // transpose_8x8 is omitted: its `_scalar` variant is not re-exported, so
    // there is nothing to compare the NEON one against from outside the crate.

    // Quantize: multiply + threshold + round, per coefficient.
    suite.compare("quantize_dct8", |g| {
        let c: &'static [f32; 64] = Box::leak(Box::new(ramp(64, 5).try_into().unwrap()));
        let w: &'static [f32; 64] = Box::leak(Box::new(ramp(64, 9).try_into().unwrap()));
        let th: &'static [f32; 4] = Box::leak(Box::new([0.6, 0.6, 0.6, 0.6]));
        g.bench("neon", move |b| {
            let mut out = Box::new([0i32; 64]);
            b.iter(move || jxl_encoder_simd::quantize_dct8_neon(t, c, w, 1.0, th, &mut out))
        });
        g.bench("scalar", move |b| {
            let mut out = Box::new([0i32; 64]);
            b.iter(move || jxl_encoder_simd::quantize_dct8_scalar(c, w, 1.0, th, &mut out))
        });
    });

    const W: usize = 512;
    const H: usize = 512;
    let plane: &'static [f32] = Box::leak(ramp(W * H, 11).into_boxed_slice());
    suite.compare("compute_mask1x1/512x512", move |g| {
        g.throughput(Throughput::Bytes((W * H * 4) as u64));
        g.bench("neon", move |b| {
            let mut out = vec![0f32; W * H];
            b.iter(move || jxl_encoder_simd::compute_mask1x1_neon(t, plane, W, H, &mut out))
        });
        g.bench("scalar", move |b| {
            let mut out = vec![0f32; W * H];
            b.iter(move || jxl_encoder_simd::compute_mask1x1_scalar(plane, W, H, &mut out))
        });
    });

    // ---- wider sweep: the rest of the exported neon/scalar pairs ----
    // 739 dispatch sites in this crate; the four above were a sample. Each of
    // these has both a `_neon` and a `_scalar` export, so they can be compared
    // head-to-head without touching the dispatch token.

    // Inverse DCT — the decode-side twin of dct_8x8.
    suite.compare("idct_8x8", |g| {
        let v = ramp(64, 21);
        let arr: [f32; 64] = v.try_into().unwrap();
        let inp: &'static [f32; 64] = Box::leak(Box::new(arr));
        g.bench("neon", move |b| {
            let mut out = Box::new([0f32; 64]);
            b.iter(move || jxl_encoder_simd::idct_8x8_neon(t, inp, &mut out))
        });
        g.bench("scalar", move |b| {
            let mut out = Box::new([0f32; 64]);
            b.iter(move || jxl_encoder_simd::idct_8x8_scalar(inp, &mut out))
        });
    });

    // Gaborish smoothing — a separable 3-tap over a full plane.
    {
        const W: usize = 512;
        const H: usize = 512;
        let plane: &'static [f32] = Box::leak(ramp(W * H, 33).into_boxed_slice());
        suite.compare("gab_smooth/512x512", move |g| {
            g.throughput(Throughput::Bytes((W * H * 4) as u64));
            g.bench("neon", move |b| {
                let mut out = vec![0f32; W * H];
                b.iter(move || {
                    jxl_encoder_simd::gab_smooth_neon(t, &mut out, plane, W, H, 0.7, 0.15, 0.05)
                })
            });
            g.bench("scalar", move |b| {
                let mut out = vec![0f32; W * H];
                b.iter(move || {
                    jxl_encoder_simd::gab_smooth_scalar(&mut out, plane, W, H, 0.7, 0.15, 0.05)
                })
            });
        });
    }

    // Whole-plane predicate — the shape most likely to be autovectorized
    // already, and the one where an early exit can mislead.
    {
        const N: usize = 1 << 20;
        let plane: &'static [f32] = Box::leak(ramp(N, 41).into_boxed_slice());
        suite.compare("is_finite_plane/1MP", move |g| {
            g.throughput(Throughput::Bytes((N * 4) as u64));
            g.bench("neon", move |b| {
                b.iter(move || jxl_encoder_simd::is_finite_plane_neon(t, plane))
            });
            g.bench("scalar", move |b| {
                b.iter(move || jxl_encoder_simd::is_finite_plane_scalar(plane))
            });
        });
    }

    // Entropy: a reduction over histogram counts.
    {
        let counts: &'static [i32] = Box::leak(
            (0..4096).map(|i| ((i * 7919) % 997) as i32).collect::<Vec<i32>>().into_boxed_slice(),
        );
        let total: usize = counts.iter().map(|c| *c as usize).sum();
        suite.compare("shannon_entropy/4096", move |g| {
            g.bench("neon", move |b| {
                b.iter(move || jxl_encoder_simd::shannon_entropy_neon(t, counts, total))
            });
            g.bench("scalar", move |b| {
                b.iter(move || jxl_encoder_simd::shannon_entropy_scalar(counts, total))
            });
        });
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn bench_kernels(_suite: &mut Suite) {
    eprintln!("[kernel_tiers] aarch64-only bench");
}

zenbench::main!(bench_kernels);
