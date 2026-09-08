// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Wall-clock harness for the high-bit-depth work (imazen/jxl-encoder#95, #109).
//!
//! Answers two questions that were left as claims:
//!
//! 1. **Did F1's arithmetic cost anything?** F1 replaced the modular predictor
//!    primitives with libjxl's overflow-correct forms — `clamped_gradient`
//!    (wrapping `u32` + two selects) for saturate-then-clamp, `average2`
//!    (`i64`) for an `i32` sum, and a branchless wrapping `pack_signed` for a
//!    branching one. Those are per-pixel, so "byte-identical" was necessary but
//!    not sufficient; this measures them against the frozen previous bodies.
//! 2. **What do the wide and float paths actually cost end to end**, relative
//!    to the 8- and 16-bit paths that already existed?
//!
//! Method notes that matter, from the repo's own hard-won methodology:
//! repeats are **interleaved** across arms rather than blocked (block-ordered
//! repeats once inverted the sign of a sectioned-K result), and each cell
//! reports the **min** of N, which is the least noisy estimator for "how fast
//! can this go" on a machine with other activity.

use std::hint::black_box;
use std::time::Instant;

// ── The frozen pre-F1 bodies, for the A/B ──────────────────────────────────

#[inline]
fn prev_clamped_gradient(n: i32, w: i32, l: i32) -> i32 {
    let gradient = w.saturating_add(n).saturating_sub(l);
    gradient.clamp(w.min(n), w.max(n))
}

#[inline]
fn prev_average2(a: i32, b: i32) -> i32 {
    (a + b) / 2
}

#[inline]
fn prev_pack_signed(value: i32) -> u32 {
    if value >= 0 {
        (value as u32) * 2
    } else {
        ((-value) as u32) * 2 - 1
    }
}

// Current forms, transcribed so this example does not need `__internals`.
#[inline]
fn now_clamped_gradient(n: i32, w: i32, l: i32) -> i32 {
    let m = n.min(w);
    let big_m = n.max(w);
    let grad = (n as u32).wrapping_add(w as u32).wrapping_sub(l as u32) as i32;
    let grad_clamp_m = if l < m { big_m } else { grad };
    if l > big_m { m } else { grad_clamp_m }
}

#[inline]
fn now_average2(a: i32, b: i32) -> i32 {
    ((a as i64 + b as i64) / 2) as i32
}

#[inline]
fn now_pack_signed(value: i32) -> u32 {
    ((value as u32) << 1) ^ (((!value as u32) >> 31).wrapping_sub(1))
}

/// 8/16-bit-shaped samples: the domain that actually ships today, so the A/B
/// measures the cost on real content rather than on the wide values that were
/// unreachable before this work.
fn narrow_samples(n: usize) -> Vec<i32> {
    let mut x: u32 = 0x2545_f491;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            (x & 0xffff) as i32
        })
        .collect()
}

fn bench_primitives(iters: usize, reps: usize) {
    let s = narrow_samples(1 << 16);
    let mut prev_best = [f64::MAX; 3];
    let mut now_best = [f64::MAX; 3];

    for _ in 0..reps {
        // Interleaved: prev then now, per primitive, inside every rep.
        for which in 0..3 {
            let t = Instant::now();
            let mut acc = 0i64;
            for _ in 0..iters {
                for w in s.windows(3) {
                    acc += match which {
                        0 => i64::from(prev_clamped_gradient(w[0], w[1], w[2])),
                        1 => i64::from(prev_average2(w[0], w[1])),
                        _ => i64::from(prev_pack_signed(w[0])),
                    };
                }
            }
            black_box(acc);
            let e = t.elapsed().as_secs_f64();
            if e < prev_best[which] {
                prev_best[which] = e;
            }

            let t = Instant::now();
            let mut acc = 0i64;
            for _ in 0..iters {
                for w in s.windows(3) {
                    acc += match which {
                        0 => i64::from(now_clamped_gradient(w[0], w[1], w[2])),
                        1 => i64::from(now_average2(w[0], w[1])),
                        _ => i64::from(now_pack_signed(w[0])),
                    };
                }
            }
            black_box(acc);
            let e = t.elapsed().as_secs_f64();
            if e < now_best[which] {
                now_best[which] = e;
            }
        }
    }

    let names = ["clamped_gradient", "average2", "pack_signed"];
    let ops = (s.len() - 2) * iters;
    println!("\n## F1 primitive A/B (min of {reps}, {ops} ops per timing)\n");
    println!("| primitive | pre-F1 ns/op | post-F1 ns/op | delta |");
    println!("|---|--:|--:|--:|");
    for i in 0..3 {
        let a = prev_best[i] * 1e9 / ops as f64;
        let b = now_best[i] * 1e9 / ops as f64;
        println!(
            "| {} | {:.3} | {:.3} | {:+.1} % |",
            names[i],
            a,
            b,
            (b - a) / a * 100.0
        );
    }
}

fn bench_end_to_end(w: u32, h: u32, reps: usize) {
    use jxl_encoder::api::{LosslessConfig, PixelLayout};

    let n = (w * h) as usize;
    // One logical image, expressed at each depth, so the comparison is about
    // the path rather than about the content.
    let base: Vec<u32> = (0..n)
        .map(|i| {
            let x = (i % w as usize) as u64;
            let y = (i / w as usize) as u64;
            ((x * 65_537 + y * 4_099) & 0x7fff_ffff) as u32
        })
        .collect();

    let g8: Vec<u8> = base.iter().map(|v| (v >> 23) as u8).collect();
    let g16: Vec<u8> = base
        .iter()
        .flat_map(|v| ((v >> 15) as u16).to_ne_bytes())
        .collect();
    let p17: Vec<u32> = base.iter().map(|v| v >> 14).collect();
    let p24: Vec<u32> = base.iter().map(|v| v >> 7).collect();
    let p31: Vec<u32> = base.clone();
    let f16b: Vec<u8> = base
        .iter()
        .flat_map(|v| ((v >> 16) as u16 & 0x7bff).to_ne_bytes())
        .collect();
    let f32b: Vec<u8> = base
        .iter()
        .flat_map(|v| ((*v as f32) * 1e-9).to_ne_bytes())
        .collect();

    let cfg = LosslessConfig::new();
    /// One benchmark arm: a label and a closure returning the encoded size.
    type Arm<'a> = (&'a str, Box<dyn Fn() -> usize>);
    let arms: Vec<Arm<'_>> = vec![
        ("Gray8 (layout)", {
            let cfg = cfg.clone();
            let d = g8.clone();
            Box::new(move || {
                cfg.encode_request(w, h, PixelLayout::Gray8)
                    .encode(&d)
                    .expect("gray8")
                    .len()
            })
        }),
        ("Gray16 (layout)", {
            let cfg = cfg.clone();
            let d = g16.clone();
            Box::new(move || {
                cfg.encode_request(w, h, PixelLayout::Gray16)
                    .encode(&d)
                    .expect("gray16")
                    .len()
            })
        }),
        ("17-bit (planar)", {
            let cfg = cfg.clone();
            let d = p17.clone();
            Box::new(move || {
                cfg.encode_planar_int(w, h, &[&d], 17, true, false)
                    .expect("17")
                    .len()
            })
        }),
        ("24-bit (planar)", {
            let cfg = cfg.clone();
            let d = p24.clone();
            Box::new(move || {
                cfg.encode_planar_int(w, h, &[&d], 24, true, false)
                    .expect("24")
                    .len()
            })
        }),
        ("31-bit (planar)", {
            let cfg = cfg.clone();
            let d = p31.clone();
            Box::new(move || {
                cfg.encode_planar_int(w, h, &[&d], 31, true, false)
                    .expect("31")
                    .len()
            })
        }),
        ("f16 (layout)", {
            let cfg = cfg.clone();
            let d = f16b.clone();
            Box::new(move || {
                cfg.encode_request(w, h, PixelLayout::GrayLinearF16)
                    .encode(&d)
                    .expect("f16")
                    .len()
            })
        }),
        ("f32 (layout)", {
            let cfg = cfg.clone();
            let d = f32b.clone();
            Box::new(move || {
                cfg.encode_request(w, h, PixelLayout::GrayLinearF32)
                    .encode(&d)
                    .expect("f32")
                    .len()
            })
        }),
    ];

    let mut best = vec![f64::MAX; arms.len()];
    let mut bytes = vec![0usize; arms.len()];
    for _ in 0..reps {
        for (i, (_, f)) in arms.iter().enumerate() {
            let t = Instant::now();
            let b = f();
            let e = t.elapsed().as_secs_f64();
            bytes[i] = b;
            if e < best[i] {
                best[i] = e;
            }
        }
    }

    println!("\n## End-to-end lossless encode, {w}x{h}, min of {reps} interleaved reps\n");
    println!("| path | ms | bytes | vs Gray16 |");
    println!("|---|--:|--:|--:|");
    let g16_ms = best[1];
    for (i, (name, _)) in arms.iter().enumerate() {
        println!(
            "| {} | {:.1} | {} | {:.2}x |",
            name,
            best[i] * 1e3,
            bytes[i],
            best[i] / g16_ms
        );
    }
}

/// Controlled: hold the SAMPLE VALUES fixed and vary only the DECLARED bit
/// depth. Any time difference is then attributable to the path, not to the
/// content — which the headline table cannot separate, because there the
/// fixture's entropy grows with the width it is expressed at.
fn bench_declared_depth(w: u32, h: u32, reps: usize) {
    use jxl_encoder::api::LosslessConfig;

    let n = (w * h) as usize;
    // Values that fit in 17 bits, so the SAME data is legal at 17, 24 and 31.
    let vals: Vec<u32> = (0..n)
        .map(|i| {
            let x = (i % w as usize) as u64;
            let y = (i / w as usize) as u64;
            ((x * 131 + y * 17) & 0x1_ffff) as u32
        })
        .collect();

    let cfg = LosslessConfig::new();
    let depths = [17u32, 24, 31];
    let mut best = vec![f64::MAX; depths.len()];
    let mut bytes = vec![0usize; depths.len()];
    for _ in 0..reps {
        for (i, &bits) in depths.iter().enumerate() {
            let t = Instant::now();
            let b = cfg
                .encode_planar_int(w, h, &[&vals], bits, true, false)
                .expect("encode")
                .len();
            let e = t.elapsed().as_secs_f64();
            bytes[i] = b;
            if e < best[i] {
                best[i] = e;
            }
        }
    }
    println!("\n## Same samples, varying DECLARED depth ({w}x{h}, min of {reps})\n");
    println!("| declared bits | ms | bytes |");
    println!("|---|--:|--:|");
    for (i, &bits) in depths.iter().enumerate() {
        println!("| {} | {:.1} | {} |", bits, best[i] * 1e3, bytes[i]);
    }
}

/// Controlled the other way: hold the DECLARED depth fixed at 31 and vary only
/// the entropy of the samples. If this is what moves the wall clock, the
/// blow-up in the headline table is a tree-learner-on-noisy-content effect
/// rather than anything about wide samples.
fn bench_entropy_at_fixed_depth(w: u32, h: u32, reps: usize) {
    use jxl_encoder::api::LosslessConfig;

    let n = (w * h) as usize;
    let smooth: Vec<u32> = (0..n)
        .map(|i| {
            let x = (i % w as usize) as u64;
            let y = (i / w as usize) as u64;
            ((x * 131 + y * 17) & 0x7fff_ffff) as u32
        })
        .collect();
    let mut x: u32 = 0x2545_f491;
    let noisy: Vec<u32> = (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            x & 0x7fff_ffff
        })
        .collect();

    let cfg = LosslessConfig::new();
    let arms = [("smooth", &smooth), ("noisy", &noisy)];
    let mut best = vec![f64::MAX; arms.len()];
    let mut bytes = vec![0usize; arms.len()];
    for _ in 0..reps {
        for (i, (_, d)) in arms.iter().enumerate() {
            let t = Instant::now();
            let b = cfg
                .encode_planar_int(w, h, &[d.as_slice()], 31, true, false)
                .expect("encode")
                .len();
            let e = t.elapsed().as_secs_f64();
            bytes[i] = b;
            if e < best[i] {
                best[i] = e;
            }
        }
    }
    println!("\n## Fixed 31-bit depth, varying CONTENT entropy ({w}x{h}, min of {reps})\n");
    println!("| content | ms | bytes |");
    println!("|---|--:|--:|");
    for (i, (name, _)) in arms.iter().enumerate() {
        println!("| {} | {:.1} | {} |", name, best[i] * 1e3, bytes[i]);
    }
}

/// The control that decides whether a slow cell is the NEW planar path or a
/// pre-existing property of lossless on that content: run the identical
/// samples through `encode_planar_int` at 17 bits and through the long-shipped
/// `PixelLayout::Gray16` layout path. Values are kept under 2^16 so both are
/// legal. If the two are close, the planar path is not the problem.
fn bench_planar_vs_layout_same_samples(w: u32, h: u32, reps: usize) {
    use jxl_encoder::api::{LosslessConfig, PixelLayout};

    let n = (w * h) as usize;
    // A smooth ramp with MANY distinct values (~57k), which is the shape that
    // ran slowly in the declared-depth experiment.
    let vals: Vec<u32> = (0..n)
        .map(|i| {
            let x = (i % w as usize) as u64;
            let y = (i / w as usize) as u64;
            ((x * 100 + y * 11) & 0xffff) as u32
        })
        .collect();
    let interleaved: Vec<u8> = vals
        .iter()
        .flat_map(|v| (*v as u16).to_ne_bytes())
        .collect();

    let cfg = LosslessConfig::new();
    let mut best = [f64::MAX; 2];
    let mut bytes = [0usize; 2];
    for _ in 0..reps {
        let t = Instant::now();
        let b = cfg
            .encode_planar_int(w, h, &[&vals], 17, true, false)
            .expect("planar")
            .len();
        let e = t.elapsed().as_secs_f64();
        bytes[0] = b;
        if e < best[0] {
            best[0] = e;
        }

        let t = Instant::now();
        let b = cfg
            .encode_request(w, h, PixelLayout::Gray16)
            .encode(&interleaved)
            .expect("layout")
            .len();
        let e = t.elapsed().as_secs_f64();
        bytes[1] = b;
        if e < best[1] {
            best[1] = e;
        }
    }
    println!(
        "\n## Same samples: new planar path vs shipped Gray16 layout ({w}x{h}, min of {reps})\n"
    );
    println!("| path | ms | bytes |");
    println!("|---|--:|--:|");
    println!(
        "| planar 17-bit (new) | {:.1} | {} |",
        best[0] * 1e3,
        bytes[0]
    );
    println!(
        "| Gray16 layout (shipped) | {:.1} | {} |",
        best[1] * 1e3,
        bytes[1]
    );
    println!("\nplanar / layout = {:.2}x wall\n", best[0] / best[1]);
}

fn main() {
    let reps: usize = std::env::var("REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    if std::env::var("SKIP_PRIM").is_err() {
        bench_primitives(4, reps);
    }
    bench_planar_vs_layout_same_samples(512, 512, reps);
    if std::env::var("SKIP_DECLARED").is_err() {
        bench_declared_depth(512, 512, reps);
    }
    bench_entropy_at_fixed_depth(256, 256, reps);
    if std::env::var("SKIP_E2E").is_err() {
        bench_end_to_end(512, 512, reps);
    }
}
