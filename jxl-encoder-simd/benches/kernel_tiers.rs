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

// Used by the arch-gated bench bodies below; dead only in cfgs where every
// kernel group compiles out (keeps `--workspace --all-targets` clippy green).
#[allow(dead_code)]
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

    dct!(
        "dct_8x8",
        64,
        jxl_encoder_simd::dct_8x8_neon,
        jxl_encoder_simd::dct_8x8_scalar
    );
    dct!(
        "dct_16x16",
        256,
        jxl_encoder_simd::dct_16x16_neon,
        jxl_encoder_simd::dct_16x16_scalar
    );
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
            (0..4096)
                .map(|i| ((i * 7919) % 997) as i32)
                .collect::<Vec<i32>>()
                .into_boxed_slice(),
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

    // ---- third wave: rectangular DCTs, inverse 16x16, sanitize ----
    // 739 dispatch sites; 8 measured was ~1% coverage. Each of these has both
    // a `_neon` and a `_scalar` export, so no shim is needed.

    macro_rules! dct2 {
        ($name:expr, $n:expr, $neon:path, $scalar:path) => {
            suite.compare($name, |g| {
                let v = ramp($n, 51);
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
    dct2!(
        "dct_16x8",
        128,
        jxl_encoder_simd::dct_16x8_neon,
        jxl_encoder_simd::dct_16x8_scalar
    );
    dct2!(
        "dct_8x16",
        128,
        jxl_encoder_simd::dct_8x16_neon,
        jxl_encoder_simd::dct_8x16_scalar
    );
    dct2!(
        "idct_16x16",
        256,
        jxl_encoder_simd::idct_16x16_neon,
        jxl_encoder_simd::idct_16x16_scalar
    );

    // In-place plane sanitize: replaces non-finite values. Fed an ALL-FINITE
    // plane on purpose — the scalar form short-circuits nothing, but a plane
    // full of NaN would exercise a different branch mix and is not the common
    // case.
    {
        const N: usize = 1 << 20;
        let clean: &'static [f32] = Box::leak(ramp(N, 61).into_boxed_slice());
        suite.compare("sanitize_finite/1MP", move |g| {
            g.throughput(Throughput::Bytes((N * 4) as u64));
            g.bench("neon", move |b| {
                b.with_input(move || clean.to_vec()).run(move |mut p| {
                    let r = jxl_encoder_simd::sanitize_finite_neon(t, &mut p);
                    (p, r)
                })
            });
            g.bench("scalar", move |b| {
                b.with_input(move || clean.to_vec()).run(move |mut p| {
                    let r = jxl_encoder_simd::sanitize_finite_scalar(&mut p);
                    (p, r)
                })
            });
        });
    }

    // ---- edge-preserving filter (EPF step 1) ----
    // A real encode hot path, three planes at once. Parameters mirror the
    // crate's own epf test so the shapes are known-good rather than guessed:
    // pad = 2, sigma_scale = 1.65, border_sigma_mul = 2/3.
    {
        const W: usize = 512;
        const H: usize = 512;
        const PAD: usize = 2;
        let stride = W + 2 * PAD;
        let xsb = W / 8;

        let base = ramp(W * H, 71);
        let px: &'static [f32] =
            Box::leak(jxl_encoder_simd::pad_plane(&base, W, H, PAD).into_boxed_slice());
        let py: &'static [f32] =
            Box::leak(jxl_encoder_simd::pad_plane(&ramp(W * H, 73), W, H, PAD).into_boxed_slice());
        let pb: &'static [f32] =
            Box::leak(jxl_encoder_simd::pad_plane(&ramp(W * H, 79), W, H, PAD).into_boxed_slice());
        // Positive inv_sigma so the filter actually runs (the test uses -1.0,
        // which is the "skip this block" sentinel — that would measure nothing).
        let isg: &'static [f32] = Box::leak(vec![0.7f32; xsb * (H / 8)].into_boxed_slice());

        suite.compare("epf_step1/512x512", move |g| {
            g.throughput(Throughput::Bytes((W * H * 4 * 3) as u64));
            g.bench("neon", move |b| {
                let (mut ox, mut oy, mut ob) =
                    (vec![0f32; W * H], vec![0f32; W * H], vec![0f32; W * H]);
                b.iter(move || {
                    jxl_encoder_simd::epf_step1_neon(
                        t,
                        px,
                        py,
                        pb,
                        &mut ox,
                        &mut oy,
                        &mut ob,
                        isg,
                        xsb,
                        W,
                        H,
                        stride,
                        PAD,
                        1.65,
                        2.0 / 3.0,
                    )
                })
            });
            g.bench("scalar", move |b| {
                let (mut ox, mut oy, mut ob) =
                    (vec![0f32; W * H], vec![0f32; W * H], vec![0f32; W * H]);
                b.iter(move || {
                    jxl_encoder_simd::epf_step1_scalar(
                        px,
                        py,
                        pb,
                        &mut ox,
                        &mut oy,
                        &mut ob,
                        isg,
                        xsb,
                        W,
                        H,
                        stride,
                        PAD,
                        1.65,
                        2.0 / 3.0,
                    )
                })
            });
        });
    }

    // ---- rectangular inverse DCTs + EPF step 2 + dequant ----
    dct2!(
        "idct_8x16",
        128,
        jxl_encoder_simd::idct_8x16_neon,
        jxl_encoder_simd::idct_8x16_scalar
    );
    dct2!(
        "idct_16x8",
        128,
        jxl_encoder_simd::idct_16x8_neon,
        jxl_encoder_simd::idct_16x8_scalar
    );

    // EPF step 2 — same signature as step 1, same known-good parameters, and
    // again a POSITIVE inv_sigma so the filter runs rather than taking the
    // skip-sentinel early-out.
    {
        const W: usize = 512;
        const H: usize = 512;
        const PAD: usize = 2;
        let stride = W + 2 * PAD;
        let xsb = W / 8;
        let px: &'static [f32] =
            Box::leak(jxl_encoder_simd::pad_plane(&ramp(W * H, 81), W, H, PAD).into_boxed_slice());
        let py: &'static [f32] =
            Box::leak(jxl_encoder_simd::pad_plane(&ramp(W * H, 83), W, H, PAD).into_boxed_slice());
        let pb: &'static [f32] =
            Box::leak(jxl_encoder_simd::pad_plane(&ramp(W * H, 87), W, H, PAD).into_boxed_slice());
        let isg: &'static [f32] = Box::leak(vec![0.7f32; xsb * (H / 8)].into_boxed_slice());

        suite.compare("epf_step2/512x512", move |g| {
            g.throughput(Throughput::Bytes((W * H * 4 * 3) as u64));
            g.bench("neon", move |b| {
                let (mut ox, mut oy, mut ob) =
                    (vec![0f32; W * H], vec![0f32; W * H], vec![0f32; W * H]);
                b.iter(move || {
                    jxl_encoder_simd::epf_step2_neon(
                        t,
                        px,
                        py,
                        pb,
                        &mut ox,
                        &mut oy,
                        &mut ob,
                        isg,
                        xsb,
                        W,
                        H,
                        stride,
                        PAD,
                        1.65,
                        2.0 / 3.0,
                    )
                })
            });
            g.bench("scalar", move |b| {
                let (mut ox, mut oy, mut ob) =
                    (vec![0f32; W * H], vec![0f32; W * H], vec![0f32; W * H]);
                b.iter(move || {
                    jxl_encoder_simd::epf_step2_scalar(
                        px,
                        py,
                        pb,
                        &mut ox,
                        &mut oy,
                        &mut ob,
                        isg,
                        xsb,
                        W,
                        H,
                        stride,
                        PAD,
                        1.65,
                        2.0 / 3.0,
                    )
                })
            });
        });
    }

    // Dequant: three planes of 64 coefficients, i32 -> f32 with per-plane
    // weights. Self-contained, no padding or block grid needed.
    {
        let qx: &'static [i32; 64] =
            Box::leak(Box::new(core::array::from_fn(|i| (i as i32 % 41) - 20)));
        let qy: &'static [i32; 64] =
            Box::leak(Box::new(core::array::from_fn(|i| (i as i32 % 37) - 18)));
        let qb: &'static [i32; 64] =
            Box::leak(Box::new(core::array::from_fn(|i| (i as i32 % 31) - 15)));
        let wx: &'static [f32; 64] =
            Box::leak(Box::new(core::array::from_fn(|i| 1.0 + i as f32 * 0.01)));
        let wy: &'static [f32; 64] =
            Box::leak(Box::new(core::array::from_fn(|i| 1.0 + i as f32 * 0.02)));
        let wb: &'static [f32; 64] =
            Box::leak(Box::new(core::array::from_fn(|i| 1.0 + i as f32 * 0.03)));

        suite.compare("dequant_dct8", move |g| {
            g.bench("neon", move |b| {
                let (mut ox, mut oy, mut ob) = (
                    Box::new([0f32; 64]),
                    Box::new([0f32; 64]),
                    Box::new([0f32; 64]),
                );
                b.iter(move || {
                    jxl_encoder_simd::dequant_dct8_neon(
                        t,
                        qx,
                        qy,
                        qb,
                        wx,
                        wy,
                        wb,
                        [1.0, 1.0, 1.0],
                        0.5,
                        0.5,
                        &mut ox,
                        &mut oy,
                        &mut ob,
                    )
                })
            });
            g.bench("scalar", move |b| {
                let (mut ox, mut oy, mut ob) = (
                    Box::new([0f32; 64]),
                    Box::new([0f32; 64]),
                    Box::new([0f32; 64]),
                );
                b.iter(move || {
                    jxl_encoder_simd::dequant_dct8_scalar(
                        qx,
                        qy,
                        qb,
                        wx,
                        wy,
                        wb,
                        [1.0, 1.0, 1.0],
                        0.5,
                        0.5,
                        &mut ox,
                        &mut oy,
                        &mut ob,
                    )
                })
            });
        });
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn bench_kernels(_suite: &mut Suite) {
    eprintln!("[kernel_tiers] aarch64-only bench");
}

zenbench::main!(bench_kernels);
