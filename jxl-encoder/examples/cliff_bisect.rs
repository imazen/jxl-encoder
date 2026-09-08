// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Bisect the lossless wall-clock cliff (imazen/jxl-encoder#115).
//!
//! Two smooth 512x512 ramps, same path, same declared depth, differ ~1300x in
//! wall clock. The issue filed "distinct-value count" as the suspect. This
//! tests a sharper one: the slow fixture's maximum sample is 75628, which is
//! **above 2^16**, while the fast fixture's is 56721, below it. A table or
//! alphabet sized `1 << 16` somewhere would produce exactly that shape.
//!
//! Method: hold dimensions, declared depth and content SHAPE fixed (a smooth
//! monotonic raster ramp, maximally predictable), and vary only the value
//! RANGE across the 2^16 boundary. If the cliff sits at 65536 the cause is the
//! range; if it tracks the distinct count smoothly, it is not.

use std::time::Instant;

#[cfg(feature = "profile-phases")]
use jxl_encoder::__test_exports::profile_phases;
use jxl_encoder::api::LosslessConfig;

/// Smooth monotonic raster ramp spanning `0 ..= range-1`.
///
/// Deliberately the most predictable content possible: the gradient predictor
/// should reduce it to near-constant residuals, so anything slow here is not
/// the entropy coder doing real work.
fn ramp(n: usize, range: u32) -> Vec<u32> {
    (0..n)
        .map(|i| ((i as u64 * u64::from(range)) / n as u64) as u32)
        .collect()
}

fn distinct(v: &[u32]) -> usize {
    let mut s: Vec<u32> = v.to_vec();
    s.sort_unstable();
    s.dedup();
    s.len()
}

fn main() {
    let w: u32 = std::env::var("W")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(512);
    let h: u32 = std::env::var("H")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(512);
    let bits: u32 = std::env::var("BITS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(17);
    let reps: usize = std::env::var("REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    let n = (w * h) as usize;

    // Bracket 2^16 tightly, plus anchors far either side.
    let ranges: Vec<u32> = std::env::var("RANGES")
        .ok()
        .map(|s| s.split(',').filter_map(|v| v.trim().parse().ok()).collect())
        .unwrap_or_else(|| {
            vec![
                32_768, 60_000, 65_000, 65_535, 65_536, 65_537, 66_000, 70_000, 76_000, 100_000,
            ]
        });

    println!("# Cliff bisect: {w}x{h}, declared {bits}-bit, min of {reps}\n");
    println!("| value range | distinct | ms | bytes |");
    println!("|---|--:|--:|--:|");
    let cfg = LosslessConfig::new();
    for range in ranges {
        if range > (1u32 << bits) {
            continue;
        }
        // FORMULA=plane switches to the 2D plane `(x*a + y*b) & mask` that
        // produced the original #115 report, as distinct from the monotonic
        // raster ramp. They have similar distinct counts and both are smooth,
        // so any difference between them isolates something other than range.
        let vals = if std::env::var("FORMULA").as_deref() == Ok("plane") {
            let a = u64::from(range) / 512;
            (0..n)
                .map(|i| {
                    let x = (i % w as usize) as u64;
                    let y = (i / w as usize) as u64;
                    ((x * a + y * 17) & 0x1_ffff) as u32
                })
                .collect()
        } else {
            ramp(n, range)
        };
        let d = distinct(&vals);
        let mut best = f64::MAX;
        let mut bytes = 0usize;
        for _ in 0..reps {
            let t = Instant::now();
            bytes = cfg
                .encode_planar_int(w, h, &[&vals], bits, true, false)
                .expect("encode")
                .len();
            let e = t.elapsed().as_secs_f64();
            if e < best {
                best = e;
            }
        }
        println!("| {} | {} | {:.1} | {} |", range, d, best * 1e3, bytes);

        #[cfg(feature = "profile-phases")]
        {
            let mut snap = profile_phases::take_snapshot();
            snap.sort_by_key(|(_, ns)| core::cmp::Reverse(*ns));
            let total: u128 = snap.iter().map(|(_, ns)| *ns).sum();
            if total > 0 {
                println!(
                    "\n  phases for range {range} (total instrumented {:.1} ms):",
                    total as f64 / 1e6
                );
                for (name, ns) in snap.iter().take(8) {
                    println!(
                        "    {:>9.1} ms  {:>5.1}%  {}",
                        *ns as f64 / 1e6,
                        *ns as f64 / total as f64 * 100.0,
                        name
                    );
                }
                println!();
            }
            profile_phases::reset();
        }
    }
}
