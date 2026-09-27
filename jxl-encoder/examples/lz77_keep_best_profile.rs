//! #110 — wall/instruction profile harness for the LZ77 keep-best selector.
//!
//! Encodes a single PNG at a fixed crop/depth/effort with the typed
//! `lossless_lz77_keep_best` improvement armed or not, so the candidate
//! ladder cost is directly attributable under callgrind or a wall clock.
//! Pairs with `lz77_hash_ab`; this binary exists for profiling, not TSV sweeps.
//!
//! Usage: lz77_keep_best_profile <png> [--size N] [--depth u8|u16|f32]
//!   [--effort N] [--threads N] [--mode global|squeeze|local|hybrid]
//!   [--arm on|off] [--out FILE]

use std::path::PathBuf;

use jxl_encoder::api::{
    EncoderImprovementsCustom, EncoderStrategy, LosslessConfig, PixelLayout, SectionedTrees,
};

fn arg(name: &str, default: &str) -> String {
    let a: Vec<String> = std::env::args().collect();
    a.iter()
        .position(|x| x == name)
        .and_then(|i| a.get(i + 1).cloned())
        .unwrap_or_else(|| default.to_string())
}

fn crop16(src: &[u16], sw: u32, sh: u32, n: u32) -> Vec<u16> {
    assert!(sw >= n && sh >= n, "source smaller than --size");
    let (x0, y0) = ((sw - n) / 2, (sh - n) / 2);
    let mut o = Vec::with_capacity((n * n * 3) as usize);
    for y in 0..n {
        let r = ((y0 + y) * sw + x0) as usize * 3;
        o.extend_from_slice(&src[r..r + (n * 3) as usize]);
    }
    o
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let source = PathBuf::from(
        a.get(1)
            .expect("usage: lz77_keep_best_profile <png> [options]"),
    );
    let n: u32 = arg("--size", "512").parse().unwrap();
    let depth = arg("--depth", "u8");
    let effort: u8 = arg("--effort", "9").parse().unwrap();
    let threads: usize = arg("--threads", "1").parse().unwrap();
    let mode = arg("--mode", "global");
    let (squeeze, sectioned) = match mode.as_str() {
        "global" => (false, SectionedTrees::Off),
        "squeeze" => (true, SectionedTrees::Off),
        "local" => (false, SectionedTrees::On),
        "hybrid" => (false, SectionedTrees::Hybrid),
        _ => panic!("unknown mode {mode}"),
    };
    let arm = arg("--arm", "on");
    let enabled = arm == "on";
    let out = arg("--out", "");

    let img = image::open(&source).expect("decode source PNG");
    // Accept the corpus's HDR photos AND any other 16-bit RGB PNG (the
    // synthetic float line-art set), since LZ77 fires on repetitive
    // content and photo residuals are the case where it barely does.
    let (sw, sh, flat): (u32, u32, Vec<u16>) = if let Some(r) = img.as_rgb16() {
        (
            r.width(),
            r.height(),
            r.pixels().flat_map(|p| [p.0[0], p.0[1], p.0[2]]).collect(),
        )
    } else {
        let r = img.to_rgb8();
        (
            r.width(),
            r.height(),
            r.pixels()
                .flat_map(|p| {
                    [
                        (p.0[0] as u16) << 8,
                        (p.0[1] as u16) << 8,
                        (p.0[2] as u16) << 8,
                    ]
                })
                .collect(),
        )
    };
    let c16 = crop16(&flat, sw, sh, n);
    let (buf, layout) = match depth.as_str() {
        "u8" => (
            c16.iter().map(|v| (*v >> 8) as u8).collect::<Vec<u8>>(),
            PixelLayout::Rgb8,
        ),
        "u16" => (
            c16.iter().flat_map(|v| v.to_ne_bytes()).collect(),
            PixelLayout::Rgb16,
        ),
        "f32" => (
            c16.iter()
                .flat_map(|v| (*v as f32 / 65535.0).to_ne_bytes())
                .collect(),
            PixelLayout::RgbLinearF32,
        ),
        _ => panic!("unknown depth {depth}"),
    };

    let config = LosslessConfig::new()
        .with_effort(effort)
        .with_threads(threads)
        .with_squeeze(squeeze)
        .with_sectioned_trees(sectioned)
        .with_strategy(EncoderStrategy::Custom(Box::new(
            EncoderImprovementsCustom {
                lossless_lz77_keep_best: enabled,
                ..Default::default()
            },
        )));
    let data = config
        .encode_request(n, n, layout)
        .encode(&buf)
        .expect("lossless encode");
    if !out.is_empty() {
        std::fs::write(&out, &data).expect("write encoded output");
    }
    println!(
        "{} {n} {depth} e{effort} {mode} t{threads} arm={arm} bytes={}",
        source.display(),
        data.len()
    );
}
