//! W44-124 single-cell smoke: encode codec_wiki + graph at e7 d=3 with all
//! 3 hint values and verify the auto-default behaviour matches predicted
//! mapping (codec_wiki: auto ≡ Some(true); graph: auto ≡ Some(false)).
//!
//! Reads no benchmark TSV — just prints byte counts for visual inspection.

use jxl_encoder::api::{LossyConfig, PixelLayout};
use std::path::Path;

fn enc(path: &Path, effort: u8, d: f32, hint: Option<bool>) -> usize {
    let img = image::open(path).unwrap();
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    let raw = rgb.into_raw();
    let cfg = LossyConfig::new(d)
        .with_effort(effort)
        .with_threads(1)
        .with_dct32_keep_hint(hint);
    let bytes = cfg
        .encode_request(w, h, PixelLayout::Rgb8)
        .encode(&raw)
        .expect("encode");
    bytes.len()
}

fn main() {
    let corpus = "/home/lilith/work/codec-corpus/gb82-sc";

    let codec_wiki = Path::new(corpus).join("codec_wiki.png");
    let graph = Path::new(corpus).join("graph.png");

    println!("==> codec_wiki.png e7 d=3 (auto should FIRE, m3=145.7, ed=0.04)");
    let cw_off = enc(&codec_wiki, 7, 3.0, Some(false));
    let cw_auto = enc(&codec_wiki, 7, 3.0, None);
    let cw_on = enc(&codec_wiki, 7, 3.0, Some(true));
    println!("  Some(false): {} bytes", cw_off);
    println!("  None (auto): {} bytes", cw_auto);
    println!("  Some(true):  {} bytes", cw_on);
    println!(
        "  predicted: auto == Some(true) → {}",
        if cw_auto == cw_on { "YES" } else { "NO" }
    );

    println!();
    println!("==> graph.png e7 d=3 (auto should REJECT, m3=11.8)");
    let g_off = enc(&graph, 7, 3.0, Some(false));
    let g_auto = enc(&graph, 7, 3.0, None);
    let g_on = enc(&graph, 7, 3.0, Some(true));
    println!("  Some(false): {} bytes", g_off);
    println!("  None (auto): {} bytes", g_auto);
    println!("  Some(true):  {} bytes", g_on);
    println!(
        "  predicted: auto == Some(false) → {}",
        if g_auto == g_off { "YES" } else { "NO" }
    );

    println!();
    if cw_auto == cw_on && g_auto == g_off {
        println!("# W44-124 smoke: PASS");
    } else {
        eprintln!("# W44-124 smoke: FAIL");
        std::process::exit(1);
    }
}
