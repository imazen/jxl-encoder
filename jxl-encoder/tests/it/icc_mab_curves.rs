// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Issue #128: profiles whose `mAB `/`mBA ` tags hold `curv` curves must
//! round-trip.
//!
//! The ICC predictor codes each such curve as a 12-byte header of inserted
//! bytes followed by predicted samples. It used to drop the header, so the
//! decoder rebuilt a profile 12 bytes short per curve: djxl v0.12 failed with
//! "Invalid result size" and jxl-rs with `InvalidIccStream`. Real profiles
//! affected include ITU-R BT.2100 PQ (`ITUR_2100_PQ_FULL.ICC`), ICC's
//! `Probev1_ICCv4.icc`, `sRGB_ICC_v4_Appearance.icc`, the RIMM-RGB v4
//! profiles and the CGATS21/CRPC press profiles.

use crate::icc_ratio::{djxl_icc, jxl_rs_icc};
use jxl_encoder::{ImageMetadata, LosslessConfig, LossyConfig, PixelLayout};

fn be32(v: u32) -> [u8; 4] {
    v.to_be_bytes()
}

/// `curv` type with `n` u16 samples of a smooth ramp, padded to 4 bytes.
fn curv(n: u32, seed: u32) -> Vec<u8> {
    let mut t = b"curv\0\0\0\0".to_vec();
    t.extend_from_slice(&be32(n));
    for i in 0..n {
        let x = i as f64 / (n - 1) as f64;
        let y = x.powf(1.8 + seed as f64 * 0.2);
        t.extend_from_slice(&((y * 65535.0).round() as u16).to_be_bytes());
    }
    while !t.len().is_multiple_of(4) {
        t.push(0);
    }
    t
}

/// `mAB ` (lutAtoBType) with three B curves and three A curves, no matrix,
/// M curves or CLUT.
fn mab(sig: &[u8; 4], n: u32) -> Vec<u8> {
    let b: Vec<u8> = (0..3).flat_map(|c| curv(n, c)).collect();
    let a: Vec<u8> = (0..3).flat_map(|c| curv(n, c + 3)).collect();
    let mut t = sig.to_vec();
    t.extend_from_slice(&[0, 0, 0, 0, 3, 3, 0, 0]);
    let b_off = 32u32;
    let a_off = b_off + b.len() as u32;
    t.extend_from_slice(&be32(b_off)); // B curves
    t.extend_from_slice(&be32(0)); // matrix
    t.extend_from_slice(&be32(0)); // M curves
    t.extend_from_slice(&be32(0)); // CLUT
    t.extend_from_slice(&be32(a_off)); // A curves
    t.extend_from_slice(&b);
    t.extend_from_slice(&a);
    t
}

/// Display-class RGB profile: matrix/TRC tags plus A2B0 (`mAB `) and B2A0
/// (`mBA `) LUT tags whose curves have `n` samples.
fn lut_profile(n: u32) -> Vec<u8> {
    let xyz = |x: f64, y: f64, z: f64| {
        let mut t = b"XYZ \0\0\0\0".to_vec();
        for v in [x, y, z] {
            t.extend_from_slice(&((v * 65536.0).round() as i32).to_be_bytes());
        }
        t
    };
    let mut desc = b"desc\0\0\0\0".to_vec();
    desc.extend_from_slice(&be32(5));
    desc.extend_from_slice(b"mab!\0");
    desc.extend_from_slice(&[0u8; 11 + 67]);
    while !desc.len().is_multiple_of(4) {
        desc.push(0);
    }
    let tags: Vec<([u8; 4], Vec<u8>)> = vec![
        (*b"desc", desc),
        (*b"wtpt", xyz(0.9642, 1.0, 0.8249)),
        (*b"rXYZ", xyz(0.4361, 0.2225, 0.0139)),
        (*b"gXYZ", xyz(0.3851, 0.7169, 0.0971)),
        (*b"bXYZ", xyz(0.1431, 0.0606, 0.7141)),
        (*b"rTRC", curv(n, 0)),
        (*b"gTRC", curv(n, 0)),
        (*b"bTRC", curv(n, 0)),
        (*b"A2B0", mab(b"mAB ", n)),
        (*b"B2A0", mab(b"mBA ", n)),
    ];
    let data_start = 128 + 4 + 12 * tags.len();
    let mut table = be32(tags.len() as u32).to_vec();
    let mut body = Vec::new();
    for (sig, data) in &tags {
        table.extend_from_slice(sig);
        table.extend_from_slice(&be32((data_start + body.len()) as u32));
        table.extend_from_slice(&be32(data.len() as u32));
        body.extend_from_slice(data);
    }
    let total = data_start + body.len();
    let mut icc = Vec::with_capacity(total);
    icc.extend_from_slice(&be32(total as u32));
    icc.extend_from_slice(&[0u8; 4]);
    icc.extend_from_slice(&be32(0x0420_0000)); // v4.2
    icc.extend_from_slice(b"mntrRGB XYZ ");
    icc.extend_from_slice(&[0u8; 12]);
    icc.extend_from_slice(b"acsp");
    icc.extend_from_slice(&[0u8; 24]);
    icc.extend_from_slice(&be32(0));
    for v in [0.9642f64, 1.0, 0.8249] {
        icc.extend_from_slice(&((v * 65536.0).round() as i32).to_be_bytes());
    }
    icc.extend_from_slice(&[0u8; 48]);
    icc.extend_from_slice(&table);
    icc.extend_from_slice(&body);
    assert_eq!(icc.len(), total);
    icc
}

#[test]
fn lut_profiles_with_curv_curves_round_trip_in_both_decoders() {
    let pixels: Vec<u8> = (0..300 * 260 * 3).map(|i| (i * 37 % 256) as u8).collect();
    for n in [9u32, 256, 4096] {
        let icc = lut_profile(n);
        let meta = ImageMetadata::new().with_icc_profile(&icc);
        for (w, h) in [(16u32, 16u32), (300, 260)] {
            let px = &pixels[..(w * h * 3) as usize];
            for lossless in [true, false] {
                let label = format!(
                    "mab-n{n}-{w}x{h}-{}",
                    if lossless { "lossless" } else { "lossy" }
                );
                let jxl = if lossless {
                    LosslessConfig::new()
                        .encode_request(w, h, PixelLayout::Rgb8)
                        .with_metadata(&meta)
                        .encode(px)
                } else {
                    LossyConfig::new(1.0)
                        .encode_request(w, h, PixelLayout::Rgb8)
                        .with_metadata(&meta)
                        .encode(px)
                }
                .unwrap_or_else(|e| panic!("encode {label}: {e}"));
                assert!(
                    jxl_rs_icc(&jxl, &label) == icc,
                    "jxl-rs {label}: ICC differs"
                );
                assert!(djxl_icc(&jxl, &label) == icc, "djxl {label}: ICC differs");
            }
        }
    }
}
