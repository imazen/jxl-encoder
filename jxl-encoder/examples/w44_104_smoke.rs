//! W44-104 smoke test: 1 cell, terminal.png e7 d=4, A vs B,
//! confirm hint changes output bytes.

use jxl_encoder::api::{Limits, LossyConfig, PixelLayout};
use std::path::PathBuf;

fn main() {
    let path = PathBuf::from("/home/lilith/work/codec-corpus/gb82-sc/terminal.png");
    let img = image::open(&path).unwrap();
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    let rgb_u8: Vec<u8> = rgb.as_raw().clone();

    let lim = Limits::default().with_max_memory_bytes(8u64 * 1024 * 1024 * 1024);

    for hint in &[None, Some(false), Some(true)] {
        let label = match hint {
            None => "Auto (default; W44-65/68 fires on terminal)",
            Some(true) => "FORCE suppress (legacy W44-68 explicit)",
            Some(false) => "BYPASS suppress (admit DCT32+DCT64)",
        };
        let cfg = LossyConfig::new(4.0)
            .with_effort(7)
            .with_strategy_overrides(jxl_encoder::api::StrategyOverrides {
                dct_suppress_hint: *hint,
                ..Default::default()
            });
        let bytes = cfg
            .encode_request(w, h, PixelLayout::Rgb8)
            .with_limits(&lim)
            .encode(&rgb_u8)
            .unwrap();
        println!("hint={:>20?} => {} bytes  ({})", hint, bytes.len(), label);
    }
}
