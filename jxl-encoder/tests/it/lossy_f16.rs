// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Lossy f16 input with dark values: half-float subnormals (below 6.1e-5)
//! used to be read 8192x too large, so near-black linear content encoded as
//! mid-grey (found by the conformance sweep: a 64x48 crop of a terminal
//! screenshot decoded with mean error 0.24).

use jxl_encoder::{LossyConfig, PixelLayout};

fn f32_to_f16_bits(v: f32) -> u16 {
    // Exact for the values used here: multiples of 2^-24 below 2^-14
    // (subnormal halves) and halves with at most 10 mantissa bits.
    if v < 2f32.powi(-14) {
        return (v * 2f32.powi(24)).round() as u16;
    }
    let b = v.to_bits();
    ((((b >> 23) & 0xff) - 112) << 10 | ((b >> 13) & 0x3ff)) as u16
}

fn read_pfm(path: &std::path::Path) -> (usize, usize, Vec<f32>) {
    let b = std::fs::read(path).unwrap();
    let mut parts = b.splitn(4, |&c| c == b'\n');
    let _ = parts.next();
    let dims = std::str::from_utf8(parts.next().unwrap()).unwrap();
    let (w, h) = dims.split_once(' ').unwrap();
    let (w, h): (usize, usize) = (w.trim().parse().unwrap(), h.trim().parse().unwrap());
    let little = std::str::from_utf8(parts.next().unwrap())
        .unwrap()
        .trim()
        .parse::<f32>()
        .unwrap()
        < 0.0;
    let data = parts.next().unwrap();
    let mut px = vec![0f32; w * h * 3];
    for y in 0..h {
        for i in 0..w * 3 {
            let o = ((h - 1 - y) * w * 3 + i) * 4;
            let raw = [data[o], data[o + 1], data[o + 2], data[o + 3]];
            px[y * w * 3 + i] = if little {
                f32::from_le_bytes(raw)
            } else {
                f32::from_be_bytes(raw)
            };
        }
    }
    (w, h, px)
}

#[test]
fn lossy_f16_dark_values_decode_close_to_source() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target/lossy-f16");
    std::fs::create_dir_all(&dir).unwrap();
    for (w, h) in [(64usize, 48usize), (300, 260)] {
        // Left half: subnormal halves (0 .. 1023 * 2^-24 ~ 6.1e-5);
        // right half: a normal-range ramp.
        let mut vals = Vec::with_capacity(w * h * 3);
        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    let v = if x < w / 2 {
                        ((x * 37 + y * 11 + c * 5) % 1024) as f32 * 2f32.powi(-24)
                    } else {
                        let q = ((x + y + c * 7) % 256) as f32 / 256.0;
                        f32::from_bits(q.to_bits() & 0xffff_e000)
                    };
                    vals.push(v);
                }
            }
        }
        let px: Vec<u8> = vals
            .iter()
            .flat_map(|&v| f32_to_f16_bits(v).to_ne_bytes())
            .collect();
        let label = format!("f16-{w}x{h}");
        let bytes = LossyConfig::new(1.0)
            .with_effort(7)
            .encode(&px, w as u32, h as u32, PixelLayout::RgbLinearF16)
            .expect("encode");
        let input = dir.join(format!("{label}.jxl"));
        let output = dir.join(format!("{label}.pfm"));
        std::fs::write(&input, &bytes).unwrap();
        let status = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
            .arg(&input)
            .arg(&output)
            .arg("--num_threads=1")
            .status()
            .expect("run djxl");
        assert!(status.success(), "djxl {label}");
        let (dw, dh, dec) = read_pfm(&output);
        assert_eq!((dw, dh), (w, h), "{label}");
        // The dark half must stay dark: its source peaks at 6.1e-5.
        let mut dark_max = 0f32;
        for y in 0..h {
            for x in 0..w / 2 {
                for c in 0..3 {
                    dark_max = dark_max.max(dec[(y * w + x) * 3 + c].abs());
                }
            }
        }
        assert!(
            dark_max < 0.01,
            "{label}: dark region decoded up to {dark_max}"
        );
    }
}
