// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-65 mask1x1 probe (standalone path) — compute `median_mask1x1`
//! on the test corpus using `srgb_to_xyb` + `compute_mask1x1`. This
//! is a STANDALONE probe; for the authoritative encoder-side values
//! see `w44_65_encoder_mask1x1_probe.rs` (the standalone probe
//! diverges from the encoder pipeline by ~17 on the windows95 case
//! due to LUT vs powf and scalar vs SIMD float precision in the
//! sRGB→linear→XYB conversion).
//!
//! Build / run:
//!   cargo run -p jxl-encoder --release --features __pre_quantized \
//!       --example w44_65_mask1x1_probe

use std::path::PathBuf;

use jxl_encoder::__pre_quantized::compute_mask1x1;
use jxl_encoder::__test_exports::xyb::srgb_image_to_xyb;

const CORPUS_BASE: &str = "/home/lilith/work/codec-corpus";

fn corpus_path(name: &str) -> Option<PathBuf> {
    let cid22 = PathBuf::from(CORPUS_BASE)
        .join("CID22/CID22-512/validation")
        .join(name);
    if cid22.exists() {
        return Some(cid22);
    }
    let gb82 = PathBuf::from(CORPUS_BASE).join("gb82-sc").join(name);
    if gb82.exists() {
        return Some(gb82);
    }
    None
}

fn median_value(mut buf: Vec<f32>) -> f32 {
    if buf.is_empty() {
        return 0.0;
    }
    let mid = buf.len() / 2;
    buf.select_nth_unstable_by(mid, |a, b| {
        a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal)
    });
    buf[mid]
}

fn probe(name: &str) {
    let Some(path) = corpus_path(name) else {
        println!("{:>28}  NOT FOUND", name);
        return;
    };
    let img = image::open(&path).unwrap_or_else(|e| panic!("open {:?}: {}", path, e));
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    let raw = rgb.as_raw();
    let n = w * h;
    let mut r_f = Vec::with_capacity(n);
    let mut g_f = Vec::with_capacity(n);
    let mut b_f = Vec::with_capacity(n);
    for px in raw.as_chunks::<3>().0 {
        r_f.push(px[0] as f32);
        g_f.push(px[1] as f32);
        b_f.push(px[2] as f32);
    }
    let mut x = vec![0.0f32; n];
    let mut y = vec![0.0f32; n];
    let mut b_out = vec![0.0f32; n];
    srgb_image_to_xyb(&r_f, &g_f, &b_f, &mut x, &mut y, &mut b_out);
    let mask1x1 = compute_mask1x1(&y, w, h);
    let median = median_value(mask1x1.clone());
    // Also compute per-pixel min/max
    let mut sorted = mask1x1.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    let p25 = sorted[sorted.len() / 4];
    let p75 = sorted[(sorted.len() * 3) / 4];
    let p10 = sorted[sorted.len() / 10];
    let p90 = sorted[(sorted.len() * 9) / 10];
    println!(
        "{:>28}  {}x{}  median={:7.2}  p25={:7.2}  p75={:7.2}  p10={:7.2}  p90={:7.2}  fires(>95)={}",
        name,
        w,
        h,
        median,
        p25,
        p75,
        p10,
        p90,
        median > 95.0
    );
}

fn main() {
    println!("W44-65 mask1x1 probe — median_mask1x1 over CID22 + gb82-sc test corpus");
    println!();
    println!(
        "{:>28}  {:^9}  {:^15}  {:^15}  {:^15}  {:^15}  {:^15}  fires(>95)",
        "image", "dims", "median", "p25", "p75", "p10", "p90"
    );
    println!("{}", "-".repeat(140));

    // Screenshots (need fires=true so default-on suppresses DCT64)
    println!("\n--- Screenshots (gb82-sc) ---");
    for img in &[
        "codec_wiki.png",
        "imac_g3.png",
        "imac_dark.png",
        "terminal.png",
        "windows.png",
        "windows95.png",
        "windowsxp.png",
        "imessage.png",
        "frymire.png",
        "graph.png",
    ] {
        probe(img);
    }

    // Photos (need fires=false so no regression)
    println!("\n--- Photos (CID22-512 validation) ---");
    for img in &[
        "1189261.png",
        "1418519.png",
        "1420710.png",
        "1531677.png",
        "1025469.png",
    ] {
        probe(img);
    }

    // Survey all CID22 validation photos to find false-fire risk
    println!("\n--- ALL CID22 validation photos (search for fires=true) ---");
    let validation =
        std::path::Path::new("/home/lilith/work/codec-corpus/CID22/CID22-512/validation");
    let mut fired = 0usize;
    let mut total = 0usize;
    let mut close: Vec<(String, f32)> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(validation) {
        let mut paths: Vec<_> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("png"))
            .collect();
        paths.sort();
        for p in paths {
            let img = image::open(&p).unwrap_or_else(|e| panic!("open {:?}: {}", p, e));
            let rgb = img.to_rgb8();
            let (w, h) = (rgb.width() as usize, rgb.height() as usize);
            let raw = rgb.as_raw();
            let n = w * h;
            let mut r_f = Vec::with_capacity(n);
            let mut g_f = Vec::with_capacity(n);
            let mut b_f = Vec::with_capacity(n);
            for px in raw.as_chunks::<3>().0 {
                r_f.push(px[0] as f32);
                g_f.push(px[1] as f32);
                b_f.push(px[2] as f32);
            }
            let mut x = vec![0.0f32; n];
            let mut y = vec![0.0f32; n];
            let mut b_out = vec![0.0f32; n];
            srgb_image_to_xyb(&r_f, &g_f, &b_f, &mut x, &mut y, &mut b_out);
            let mask1x1 = compute_mask1x1(&y, w, h);
            let med = median_value(mask1x1.clone());
            let name = p.file_name().unwrap().to_string_lossy().into_owned();
            total += 1;
            if med > 95.0 {
                fired += 1;
                println!("  FIRE  {}  median={:.2}", name, med);
            } else if med > 85.0 {
                close.push((name, med));
            }
        }
    }
    println!();
    println!(
        "Survey: {}/{} CID22 validation photos fired (>95) — {} more were in 85-95 range:",
        fired,
        total,
        close.len()
    );
    for (n, m) in &close {
        println!("  CLOSE {}  median={:.2}", n, m);
    }
}
