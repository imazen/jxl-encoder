//! W44-124 zenanalyze proxy probe for dct32_keep_hint auto-discriminator.
//!
//! Goal: find a discriminator that admits codec_wiki (W44-123's only solid win)
//! but rejects the 6 SCREEN cells that regressed in W44-123 (graph, windows,
//! imessage — all distances). Preserves W44-123 terminal d=4 win.
//!
//! Probes the existing `ZenanalyzeProxies::compute_srgb_u8` (m3_colourfulness,
//! flat_color_block_ratio, edge_density) on:
//!   - WANT-FIRE: codec_wiki.png (W44-123 +0.90 to +1.40 SSIM2 win)
//!   - WANT-FIRE: terminal.png (W44-123 preserved +0.47 SSIM2 at e8/e9 d=4)
//!   - REJECT:    graph, windows, imessage (W44-123 regression set)
//!   - SAFE-IF-AUTO-REJECTS: imac_g3, imac_dark, windows95
//!   - REJECT photos: 1418519, 1189261, 1025469, etc (no effect: mask gates already block)
//!
//! Proposed predicate from dispatch task:
//!   `m3_colourfulness > 60 AND edge_density < 0.05`
//!
//! Run:
//!   cargo run --release -p jxl-encoder --example w44_124_proxy_probe \
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
        ("WANT-FIRE", "codec_wiki", "gb82-sc/codec_wiki.png"),
        ("WANT-FIRE", "terminal", "gb82-sc/terminal.png"),
        ("REGRESSED-W44-123", "graph", "gb82-sc/graph.png"),
        ("REGRESSED-W44-123", "windows", "gb82-sc/windows.png"),
        ("REGRESSED-W44-123", "imessage", "gb82-sc/imessage.png"),
        ("BORDERLINE-W44-123", "imac_g3", "gb82-sc/imac_g3.png"),
        ("BORDERLINE-W44-123", "imac_dark", "gb82-sc/imac_dark.png"),
        ("SAFE-NOREGRESS", "windows95", "gb82-sc/windows95.png"),
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
        (
            "REJECT-PHOTO",
            "1531677",
            "CID22/CID22-512/validation/1531677.png",
        ),
    ];

    println!(
        "{:<22} {:<14} {:>5} {:>5} {:>10} {:>8} {:>10}  predicate(m3>60 AND ed<0.05)",
        "class", "name", "w", "h", "m3", "fcbr", "edge_dens"
    );
    println!("{}", "-".repeat(110));
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
        let predicate_fires = p.m3_colourfulness > 60.0 && p.edge_density < 0.05;
        println!(
            "{:<22} {:<14} {:>5} {:>5} {:>10.3} {:>8.4} {:>10.5}  {}",
            class,
            name,
            w,
            h,
            p.m3_colourfulness,
            p.flat_color_block_ratio,
            p.edge_density,
            if predicate_fires { "FIRES" } else { "reject" }
        );
    }
}
