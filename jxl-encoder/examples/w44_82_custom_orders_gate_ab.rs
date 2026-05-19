//! W44-82 bytes-only sweep with the custom-orders cost-gate REMOVED.
//!
//! Run this with the source-level gate removed (current HEAD) and compare
//! `bytes_new` vs the W44-79 baseline `jxl_bytes` column in
//! `benchmarks/cjxl_parity_ledger_2026-05-19_w44_79.tsv`.
//!
//! Cells covered (W44-79 F-D OPEN cluster + neighbors at e7 d=3-6):
//!   1420710 e7 d=3/4/5/6
//!   1531677 e7 d=3/4/5/6
//!   1189261 e7 d=3/4/5/6
//!
//! Plus the original "gate motivation" hash-lock fixture image is
//! NOT covered here — the regression on `lossy_defaults_rgb_48x48_noise`
//! is +47 B = +1.46 % (3254 vs 3207), measured via the standard
//! `hash_lock_features` test.
//!
//! Usage:
//!   cargo run -p jxl-encoder --release \
//!       --example w44_82_custom_orders_gate_ab \
//!       --features 'parallel butteraugli-loop'
//!       2>&1 | tee /tmp/w44_82_ab.tsv

use jxl_encoder::api::{Limits, LossyConfig, PixelLayout};

const IMAGES: &[(&str, &str)] = &[
    (
        "1420710",
        "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1420710.png",
    ),
    (
        "1531677",
        "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1531677.png",
    ),
    (
        "1189261",
        "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1189261.png",
    ),
    // Photo controls — should not regress when the gate is removed
    (
        "1025469",
        "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1025469.png",
    ),
    (
        "1418519",
        "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1418519.png",
    ),
];

const EFFORT: u8 = 7;
const DISTANCES: &[f32] = &[3.0, 4.0, 5.0, 6.0];

fn load_rgb(path: &str) -> Option<(Vec<u8>, u32, u32)> {
    let img = image::open(path).ok()?.to_rgb8();
    let (w, h) = img.dimensions();
    Some((img.into_raw(), w, h))
}

fn encode(rgb: &[u8], w: u32, h: u32, d: f32, e: u8) -> Option<usize> {
    let cfg = LossyConfig::new(d).with_effort(e);
    let lim = Limits::default().with_max_memory_bytes(8u64 * 1024 * 1024 * 1024);
    let bytes = cfg
        .encode_request(w, h, PixelLayout::Rgb8)
        .with_limits(&lim)
        .encode(rgb)
        .map_err(|e| eprintln!("encode failed: {e:?}"))
        .ok()?;
    Some(bytes.len())
}

fn main() {
    let dump = std::env::var("W44_82_DUMP_DIR").ok();
    println!("image\teffort\tdistance\tbytes_new_gate_off");
    for &(name, path) in IMAGES {
        let Some((rgb, w, h)) = load_rgb(path) else {
            eprintln!("skip {name}: load failed");
            continue;
        };
        for &d in DISTANCES {
            let mut best = usize::MAX;
            for _ in 0..2 {
                if let Some(b) = encode(&rgb, w, h, d, EFFORT) {
                    best = best.min(b);
                }
            }
            if best == usize::MAX {
                continue;
            }
            println!("{}\t{}\t{:.4}\t{}", name, EFFORT, d, best);
            if let Some(ref dir) = dump {
                let cfg = LossyConfig::new(d).with_effort(EFFORT);
                let lim = Limits::default()
                    .with_max_memory_bytes(8u64 * 1024 * 1024 * 1024);
                if let Ok(bytes) = cfg
                    .encode_request(w, h, PixelLayout::Rgb8)
                    .with_limits(&lim)
                    .encode(&rgb)
                {
                    let p = format!("{}/{}_e{}_d{:.1}.jxl", dir, name, EFFORT, d);
                    let _ = std::fs::write(&p, &bytes);
                }
            }
        }
    }
}
