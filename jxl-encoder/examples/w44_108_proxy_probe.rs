//! W44-108 zenanalyze proxy probe.
//!
//! Goal: find a sub-discriminator that admits {terminal, imac_g3} (recover
//! W44-105 wins at d=2..3.5 that W44-107 sacrificed) but rejects
//! {codec_wiki} (avoid re-introducing the FIXED->OPEN regression W44-107
//! closed).
//!
//! Probes the existing `ZenanalyzeProxies::compute_srgb_u8` (m3_colourfulness,
//! flat_color_block_ratio, edge_density) plus a mask1x1 median proxy (the
//! W44-105/107 existing `is_screenshot` discriminator value) on:
//!   - WANT-FIRE: terminal.png, imac_g3.png, imac_dark.png (screenshot wins)
//!   - REJECT:    codec_wiki.png (W44-107 regression target)
//!   - REJECT photos: 1418519, 1189261, 1025469 (must not enable on photos)
//!
//! Run:
//!   cargo run --release -p jxl-encoder --example w44_108_proxy_probe \
//!     --features __expert

use jxl_encoder::__pre_quantized::ZenanalyzeProxies;
use std::path::Path;

fn load_rgb8(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
    let img = image::open(path).ok()?;
    let rgb = img.to_rgb8();
    Some((rgb.as_raw().clone(), rgb.width(), rgb.height()))
}

fn main() {
    let corpus = std::env::var("CORPUS_ROOT")
        .unwrap_or_else(|_| format!("{}/work/codec-corpus", std::env::var("HOME").unwrap()));
    let cases: &[(&str, &str, &str)] = &[
        // (class, name, relative path under CORPUS_ROOT)
        ("WANT-FIRE-2-3", "terminal", "gb82-sc/terminal.png"),
        ("WANT-FIRE-2-3", "imac_g3", "gb82-sc/imac_g3.png"),
        ("WANT-FIRE-2-3", "imac_dark", "gb82-sc/imac_dark.png"),
        ("REJECT-2-3", "codec_wiki", "gb82-sc/codec_wiki.png"),
        ("ALREADY-FIRES-3.5+", "windows", "gb82-sc/windows.png"),
        ("ALREADY-FIRES-3.5+", "graph", "gb82-sc/graph.png"),
        ("ALREADY-FIRES-3.5+", "windows95", "gb82-sc/windows95.png"),
        (
            "REJECT-PHOTO",
            "1418519",
            "CID22/CID22-512/validation/1418519.png",
        ),
        (
            "REJECT-PHOTO",
            "1189261",
            "CID22/CID22-512/validation/1189261.png",
        ),
        (
            "REJECT-PHOTO",
            "1025469",
            "CID22/CID22-512/validation/1025469.png",
        ),
        (
            "REJECT-PHOTO",
            "1420710",
            "CID22/CID22-512/validation/1420710.png",
        ),
    ];

    println!(
        "{:<20} {:<14} {:>5} {:>5} {:>10} {:>8} {:>10}",
        "class", "name", "w", "h", "m3", "fcbr", "edge_dens"
    );
    println!("{}", "-".repeat(80));
    for (class, name, rel) in cases {
        let path = Path::new(&corpus).join(rel);
        let (rgb, w, h) = match load_rgb8(&path) {
            Some(v) => v,
            None => {
                eprintln!("MISS: {}", path.display());
                continue;
            }
        };
        let p = ZenanalyzeProxies::compute_srgb_u8(&rgb, w as usize, h as usize, 3, 0, 1, 2);
        println!(
            "{:<20} {:<14} {:>5} {:>5} {:>10.3} {:>8.4} {:>10.5}",
            class, name, w, h, p.m3_colourfulness, p.flat_color_block_ratio, p.edge_density
        );
    }
}
