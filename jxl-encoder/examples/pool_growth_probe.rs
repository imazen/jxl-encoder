//! Sequential multi-encode memory-growth probe for issue #93
//! (VarDCT butteraugli backend OOMs many-encode sweeps).
//!
//! Encodes many procedurally-generated images of VARIED sizes in ONE
//! process, SEQUENTIALLY (no outer concurrency), with the butteraugli
//! quantization loop active (effort >= 8), printing current VmRSS / VmHWM
//! after each encode.
//!
//! This is the discriminator the issue needs: each jxl-encoder encode
//! builds a *fresh* `ButteraugliReference` with a *fresh* `BufferPool`
//! that is dropped at encode end, so the pool does NOT persist across
//! encodes. If sequential VmRSS grows monotonically here, the retained
//! memory is a true per-encode leak or glibc not returning freed
//! allocations (heaptrack `peak == retained` rules out the latter); if
//! VmRSS stays flat, the sweep's 32 GB peak is concurrency (many encodes
//! live at once), not pool accumulation.
//!
//! Usage:
//!   pool_growth_probe <n_encodes> <effort> <distance> <threads>
//!
//! Emits a TSV stream to stdout (header first):
//!   idx  w  h  bytes  rss_kb  hwm_kb  wall_ms
//! plus a final SUMMARY line with first/last RSS and the per-encode slope.

use std::fs;
use std::time::Instant;

use jxl_encoder::{LossyConfig, PixelLayout};

fn vm_kb(field: &str) -> u64 {
    let s = fs::read_to_string("/proc/self/status").unwrap_or_default();
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix(field) {
            return rest
                .trim()
                .trim_end_matches(" kB")
                .trim()
                .parse()
                .unwrap_or(0);
        }
    }
    0
}

/// Deterministic per-index size in [min, max], spread so consecutive
/// encodes rarely repeat a size (mimics a sweep over varied renditions).
fn size_for(idx: u32, min: u32, max: u32) -> (u32, u32) {
    // Two independent LCG streams so width and height vary independently.
    let span = max - min;
    let wx = idx.wrapping_mul(2_654_435_761).rotate_left(13);
    let hx = idx.wrapping_mul(40_503).wrapping_add(0x9E37).rotate_left(7);
    let w = min + (wx % (span + 1));
    let h = min + (hx % (span + 1));
    (w, h)
}

/// Deterministic high-entropy RGB8 pixels (so the buttloop does real work
/// and patch detection doesn't collapse the image to a trivial encode).
fn gen_pixels(w: u32, h: u32, seed: u32) -> Vec<u8> {
    let n = (w as usize) * (h as usize) * 3;
    let mut v = Vec::with_capacity(n);
    let mut state = seed.wrapping_mul(2_246_822_519).wrapping_add(1);
    for _ in 0..n {
        // xorshift32
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        v.push((state >> 24) as u8);
    }
    v
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let n: u32 = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(200);
    let effort: u8 = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(8);
    let distance: f32 = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(2.0);
    let threads: usize = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(1);
    // Optional 5th arg: fixed square size. When > 0 EVERY encode uses
    // `fixed × fixed` — removes the size-tracking confounder so a rising
    // RSS floor is unambiguously a per-encode leak (not the current
    // image's working set).
    let fixed: u32 = a.get(5).and_then(|s| s.parse().ok()).unwrap_or(0);

    // Varied-size band. Kept modest so the loop is fast yet exercises the
    // multi-res precompute (the >= 64 px floor keeps half-res alive).
    let (min_px, max_px) = (192u32, 768u32);

    eprintln!(
        "pool_growth_probe: n={n} effort={effort} distance={distance} threads={threads} \
         sizes=[{min_px},{max_px}]"
    );
    println!("idx\tw\th\tbytes\trss_kb\thwm_kb\twall_ms");

    let mut first_rss = 0u64;
    let mut last_rss = 0u64;
    for idx in 0..n {
        let (w, h) = if fixed > 0 {
            (fixed, fixed)
        } else {
            size_for(idx, min_px, max_px)
        };
        let pixels = gen_pixels(w, h, idx.wrapping_add(1));

        let t0 = Instant::now();
        let encoded = LossyConfig::new(distance)
            .with_effort(effort)
            .with_threads(threads)
            .encode_request(w, h, PixelLayout::Rgb8)
            .encode(&pixels);
        let wall = t0.elapsed().as_secs_f64() * 1000.0;
        let bytes = encoded.map(|d| d.len()).unwrap_or(0);

        let rss = vm_kb("VmRSS:");
        let hwm = vm_kb("VmHWM:");
        if idx == 0 {
            first_rss = rss;
        }
        last_rss = rss;
        println!("{idx}\t{w}\t{h}\t{bytes}\t{rss}\t{hwm}\t{wall:.1}");
    }

    let grew = last_rss.saturating_sub(first_rss);
    let per_encode = if n > 1 {
        grew as f64 / (n - 1) as f64
    } else {
        0.0
    };
    println!(
        "SUMMARY\tfirst_rss_kb={first_rss}\tlast_rss_kb={last_rss}\tgrowth_kb={grew}\t\
         per_encode_kb={per_encode:.1}\tn={n}"
    );
    eprintln!(
        "SUMMARY first_rss={first_rss}kB last_rss={last_rss}kB growth={grew}kB \
         per_encode={per_encode:.1}kB/encode over {n} encodes"
    );
}
