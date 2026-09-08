// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Does the #115 pre-quantisation pathology reach EXISTING 16-bit content?
//!
//! #115 was found through the new >16-bit and float paths. The open question
//! was whether it also bites the long-shipped `PixelLayout::Rgb16` surface —
//! which decides whether it was a latent bug or one that has been costing
//! users. The mechanism needs a sampled property column whose distinct set
//! approaches 65536, and for 16-bit RGB that needs no wide input at all: a
//! channel alone carries up to 65536 distinct values, and the learner's
//! properties include neighbour and reference-channel DIFFERENCES, which span
//! 17 bits.
//!
//! **Real content, not synthetic.** Uses the imazen-26 16-bit HDR renders
//! (`png-v3/**/*.hdr.png`, 16-bit RGB, gain-map sources) — per CLAUDE.md,
//! synthetic content hides and invents hotspots in equal measure, and a
//! performance claim about "existing 16-bit content" has to be made on some.
//!
//! Run against the fix, then against a build with the `compact()` doubling
//! reverted, and compare.

use std::time::Instant;

use jxl_encoder::api::{LosslessConfig, PixelLayout};

const CORPUS: &str = "/Users/lilith/work/zen/imazen-26/png-v3";

fn find_hdr_images(limit: usize) -> Vec<std::path::PathBuf> {
    fn walk(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>, limit: usize) {
        if out.len() >= limit {
            return;
        }
        let Ok(rd) = std::fs::read_dir(dir) else {
            return;
        };
        let mut entries: Vec<_> = rd.filter_map(Result::ok).map(|e| e.path()).collect();
        entries.sort();
        for p in entries {
            if out.len() >= limit {
                return;
            }
            if p.is_dir() {
                walk(&p, out, limit);
            } else if p.to_string_lossy().ends_with(".hdr.png") {
                out.push(p);
            }
        }
    }
    let mut out = Vec::new();
    walk(std::path::Path::new(CORPUS), &mut out, limit);
    out
}

fn main() {
    let side: u32 = std::env::var("SIDE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1024);
    let reps: usize = std::env::var("REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    let count: usize = std::env::var("COUNT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);

    let images = find_hdr_images(count);
    assert!(
        !images.is_empty(),
        "no 16-bit HDR renders under {CORPUS} — fetch imazen-26 png-v3 first"
    );

    println!("# #115 scope: REAL 16-bit HDR content on the shipped Rgb16 path\n");
    println!("| image | {side}x{side} crop | ms | bytes |");
    println!("|---|---|--:|--:|");

    let cfg = LosslessConfig::new();
    for path in &images {
        let img = image::open(path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
        let rgb16 = img.to_rgb16();
        let (iw, ih) = (rgb16.width(), rgb16.height());
        let cw = side.min(iw);
        let ch = side.min(ih);
        // Centre crop, so we get real detail rather than a border.
        let (x0, y0) = ((iw - cw) / 2, (ih - ch) / 2);
        let mut px: Vec<u8> = Vec::with_capacity((cw * ch * 6) as usize);
        for y in 0..ch {
            for x in 0..cw {
                let p = rgb16.get_pixel(x0 + x, y0 + y);
                for c in 0..3 {
                    px.extend_from_slice(&p.0[c].to_ne_bytes());
                }
            }
        }

        let mut best = f64::MAX;
        let mut bytes = 0usize;
        for _ in 0..reps {
            let t = Instant::now();
            bytes = cfg
                .encode_request(cw, ch, PixelLayout::Rgb16)
                .encode(&px)
                .expect("Rgb16 lossless encode")
                .len();
            let e = t.elapsed().as_secs_f64();
            if e < best {
                best = e;
            }
        }
        let name = path.file_name().unwrap().to_string_lossy();
        let short: String = name.chars().take(46).collect();
        println!("| {short} | {cw}x{ch} | {:.1} | {} |", best * 1e3, bytes);
    }
}
