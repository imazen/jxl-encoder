//! W44-77: A/B vs W44-29 on F-D residual cells (with/without high_d_photo_hint=false).

use image::GenericImageView;
use jxl_encoder::{LossyConfig, PixelLayout};
use std::process::Command;

const CELLS: &[(&str, &str, &[f32])] = &[
    (
        "1420710",
        "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1420710.png",
        &[3.0, 4.0, 5.0, 6.0],
    ),
    (
        "1531677",
        "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1531677.png",
        &[3.0, 4.0, 5.0, 6.0],
    ),
    (
        "1189261",
        "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1189261.png",
        &[3.0, 4.0, 5.0, 6.0],
    ),
];

fn encode_with(rgb: &[u8], w: u32, h: u32, d: f32, hint: Option<bool>) -> usize {
    let mut cfg = LossyConfig::new(d).with_effort(7);
    cfg = cfg.with_high_d_photo_hint(hint);
    cfg.encode(rgb, w, h, PixelLayout::Rgb8)
        .expect("encode")
        .len()
}

fn cjxl_size(src: &str, d: f32) -> Option<usize> {
    let tmp = format!("/tmp/w44_77_cjxl_{}.jxl", std::process::id());
    let out = Command::new("cjxl")
        .args(["-d", &d.to_string(), "-e", "7", src, &tmp])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let sz = std::fs::metadata(&tmp).ok()?.len() as usize;
    let _ = std::fs::remove_file(&tmp);
    Some(sz)
}

fn main() {
    println!("cell\td\tours_w29\tours_nohint\tcjxl\tnohint-w29\t%(no_w29-cjxl)/cjxl");
    for (label, path, distances) in CELLS {
        let img = image::open(path).expect("decode png");
        let (w, h) = img.dimensions();
        let rgb = img.to_rgb8().into_raw();
        for &d in *distances {
            let b_def = encode_with(&rgb, w, h, d, None);
            let b_no_w29 = encode_with(&rgb, w, h, d, Some(false));
            let b_cjxl = cjxl_size(path, d).unwrap_or(0);
            println!(
                "{label}\t{d}\t{b_def}\t{b_no_w29}\t{b_cjxl}\t{:+}\t{:+.2}%",
                b_no_w29 as i64 - b_def as i64,
                if b_cjxl > 0 {
                    (b_no_w29 as f64 - b_cjxl as f64) / b_cjxl as f64 * 100.0
                } else {
                    0.0
                }
            );
        }
    }
}
