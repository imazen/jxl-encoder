//! W44-77 tuning search: hold W44-29 `dct32x32=1.34` while varying
//! `dct16x32` upward toward libjxl's 1.49 reference value.  Goal: tip
//! close-race cells back toward DCT32X32 selection (matching libjxl)
//! without losing W44-29's byte savings on F-D residual cells.
//!
//! Construct a custom `EntropyMulTable` via the `__expert` config hook.

use image::GenericImageView;
use jxl_encoder::{EntropyMulTable, LossyConfig, LossyInternalParams, PixelLayout};
use std::process::Command;

const CELLS: &[(&str, &str)] = &[
    ("1420710", "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1420710.png"),
    ("1531677", "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1531677.png"),
    ("1189261", "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1189261.png"),
];
const DISTANCES: &[f32] = &[3.0, 4.0, 5.0, 6.0];

// dct16x32 candidates to sweep (holding dct32x32=1.34 fixed)
const DCT16X32_CANDIDATES: &[f32] = &[1.349, 1.40, 1.45, 1.49, 1.55, 1.60];

fn build_table(dct16x32_val: f32) -> EntropyMulTable {
    let mut t = EntropyMulTable::high_d_photo_smooth_suppressed();
    t.dct16x32 = dct16x32_val;
    t
}

fn encode_with_table(
    rgb: &[u8],
    w: u32,
    h: u32,
    d: f32,
    table: Option<EntropyMulTable>,
) -> usize {
    let mut cfg = LossyConfig::new(d).with_effort(7);
    if let Some(t) = table {
        // Force W44-29 OFF then inject our custom table via internal_params
        cfg = cfg.with_high_d_photo_hint(Some(false));
        let internal = LossyInternalParams {
            entropy_mul_table: Some(t),
            ..Default::default()
        };
        cfg = cfg.with_internal_params(internal);
    }
    cfg.encode(rgb, w, h, PixelLayout::Rgb8)
        .expect("encode")
        .len()
}

fn cjxl_size(src: &str, d: f32) -> Option<usize> {
    let tmp = format!("/tmp/w44_77_tune_cjxl_{}_{}.jxl", std::process::id(), (d * 10.0) as u32);
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
    print!("cell\td\tdefault\t");
    for c in DCT16X32_CANDIDATES {
        print!("dct16x32={c:.3}\t");
    }
    println!("cjxl");

    for (label, path) in CELLS {
        let img = image::open(path).expect("decode png");
        let (w, h) = img.dimensions();
        let rgb = img.to_rgb8().into_raw();
        for &d in DISTANCES {
            let default_bytes = encode_with_table(&rgb, w, h, d, None);
            let cjxl_bytes = cjxl_size(path, d).unwrap_or(0);
            print!("{label}\t{d}\t{default_bytes}\t");
            for &c in DCT16X32_CANDIDATES {
                let t = build_table(c);
                let b = encode_with_table(&rgb, w, h, d, Some(t));
                let delta = b as i64 - default_bytes as i64;
                print!("{b}({delta:+})\t");
            }
            println!("{cjxl_bytes}");
        }
    }
}
