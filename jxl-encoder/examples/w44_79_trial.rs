//! W44-79 trial: confirm with_high_d_photo_hint(Some(true)) saves bytes on 1189261.

use jxl_encoder::api::{LossyConfig, PixelLayout};
use std::path::Path;

fn encode(rgb: &[u8], w: u32, h: u32, d: f32, hint: Option<bool>) -> usize {
    let mut cfg = LossyConfig::new(d).with_effort(7).with_threads(1);
    if let Some(h) = hint {
        cfg = cfg.with_strategy_overrides(jxl_encoder::api::StrategyOverrides {
            high_d_photo_hint: Some(h),
            ..Default::default()
        });
    }
    cfg.encode(rgb, w, h, PixelLayout::Rgb8)
        .expect("encode")
        .len()
}

fn main() {
    let path = Path::new("/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1189261.png");
    let img = image::open(path).unwrap();
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    let raw = rgb.into_raw();
    println!("# 1189261.png ({}x{}) — high_d_photo_hint A/B", w, h);
    println!("# distance, auto_bytes, hint_true_bytes, delta_pct");
    for &d in &[3.0_f32, 4.0, 5.0, 6.0] {
        let a = encode(&raw, w, h, d, None);
        let t = encode(&raw, w, h, d, Some(true));
        let delta_pct = (t as f64 - a as f64) / a as f64 * 100.0;
        println!("{}\t{}\t{}\t{:+.2}", d, a, t, delta_pct);
    }
}
