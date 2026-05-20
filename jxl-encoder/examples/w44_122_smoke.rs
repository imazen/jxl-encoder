//! W44-122 smoke: codec_wiki.png d=3 e5/e6/e7, hint=None vs Some(false)
//! Tests whether the W44-65 gate IS firing on codec_wiki d=3 (per W44-104
//! falsification, terminal fires; need empirical check for codec_wiki).
//! If admit changes the bytes by more than measurement noise the gate is
//! firing — same as W44-104 result on terminal.

use jxl_encoder::api::{Limits, LossyConfig, PixelLayout};
use std::path::PathBuf;

fn main() {
    let path = PathBuf::from("/home/lilith/work/codec-corpus/gb82-sc/codec_wiki.png");
    let img = image::open(&path).unwrap();
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    let rgb_u8: Vec<u8> = rgb.as_raw().clone();

    let lim = Limits::default().with_max_memory_bytes(8u64 * 1024 * 1024 * 1024);

    for effort in &[5u8, 6, 7] {
        for hint in &[None, Some(false), Some(true)] {
            let label = match hint {
                None => "AUTO (W44-65 default)",
                Some(true) => "FORCE-SUPPRESS",
                Some(false) => "ADMIT (bypass W44-65)",
            };
            let cfg = LossyConfig::new(3.0)
                .with_effort(*effort)
                .with_strategy_overrides(jxl_encoder::api::StrategyOverrides { dct_suppress_hint: *hint, ..Default::default() });
            let bytes = cfg
                .encode_request(w, h, PixelLayout::Rgb8)
                .with_limits(&lim)
                .encode(&rgb_u8)
                .unwrap();
            println!(
                "e{} hint={:>15?} => {} bytes  ({})",
                effort,
                hint,
                bytes.len(),
                label
            );
        }
    }
}
