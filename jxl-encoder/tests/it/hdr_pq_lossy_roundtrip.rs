// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Issue #73 regression gate: lossy VarDCT on PQ-tagged integer input.
//!
//! Pre-fix, `with_color_encoding(PQ)` + `Rgb16` lossy produced destroyed
//! pixels (PQ-pipeline butteraugli ~170, CONSTANT across distances —
//! quality did not track the quality knob). Root cause was a scale
//! mismatch: `pq_*_to_linear_f32` normalizes 1.0 = 10,000 nits while the
//! XYB transform and codestream header assumed the 255-nit SDR scale.
//! Fixed by the libjxl-parity pair: `SetIntensityTarget` dispatch
//! (PQ → 10,000 / HLG → 1,000, `luminance.cc`) + the
//! `ComputePremulAbsorb` intensity multiplier (`enc_xyb.cc:217`) applied
//! in `convert_strip`.
//!
//! The gate encodes a PQ HDR ramp at two distances and asserts, via the
//! same PQ-EOTF butteraugli pipeline both ways (jxl-oxide decode in the
//! file's own PQ space):
//!   score(d=1) < score(d=4)  — quality tracks distance again
//!   score(d=1) < 8.0         — sane range, not destruction (~170)
//! plus PQ signaling survives.

use butteraugli::{ButteraugliParams, butteraugli_linear};
use imgref::Img;
use jxl_encoder::TransferFunction;
use jxl_encoder::{LossyConfig, PixelLayout};
use rgb::RGB;

const PEAK_NITS: f32 = 1000.0;

fn pq_eotf(e: f32) -> f32 {
    const M1: f32 = 2610.0 / 16384.0;
    const M2: f32 = 2523.0 / 4096.0 * 128.0;
    const C1: f32 = 3424.0 / 4096.0;
    const C2: f32 = 2413.0 / 4096.0 * 32.0;
    const C3: f32 = 2392.0 / 4096.0 * 32.0;
    let ep = e.max(0.0).powf(1.0 / M2);
    10_000.0 * ((ep - C1).max(0.0) / (C2 - C3 * ep)).powf(1.0 / M1)
}

/// HDR-photo-shaped ramp: vertical luminance sweep into specular peaks,
/// low-byte noise. PQ-coded u16 RGB.
fn pq_ramp_rgb16(w: usize, h: usize) -> Vec<u8> {
    let mut seed = 4242u64;
    let mut lcg = move || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (seed >> 56) as u16
    };
    let mut out = Vec::with_capacity(w * h * 6);
    for y in 0..h {
        // sweep PQ code values 0.05..0.75 (≈ dark to ~1500 nits)
        let base = (3277.0 + (y as f32 / h as f32) * 45875.0) as u16;
        for x in 0..w {
            let sparkle = if (x * 31 + y * 17) % 97 == 0 {
                12000
            } else {
                0
            };
            let r = base.saturating_add(sparkle).saturating_add(lcg() & 0x3f);
            let g = base.saturating_add(lcg() & 0x3f);
            let b = base.saturating_sub(2000).saturating_add(lcg() & 0x3f);
            for v in [r, g, b] {
                out.extend_from_slice(&v.to_ne_bytes());
            }
        }
    }
    out
}

fn pq_linear_img(samples: &[u16], w: usize, h: usize) -> Img<Vec<RGB<f32>>> {
    let px: Vec<RGB<f32>> = samples
        .as_chunks::<3>()
        .0
        .iter()
        .map(|c| {
            let f = |v: u16| (pq_eotf(v as f32 / 65535.0) / PEAK_NITS).min(1.0);
            RGB::new(f(c[0]), f(c[1]), f(c[2]))
        })
        .collect();
    Img::new(px, w, h)
}

fn encode_and_score(pixels: &[u8], w: usize, h: usize, d: f32) -> (f64, Vec<u8>) {
    let ce = jxl_encoder::ColorEncoding::from_cicp(1, 16, 0, true).unwrap();
    let bytes = LossyConfig::new(d)
        .with_effort(5)
        .encode_request(w as u32, h as u32, PixelLayout::Rgb16)
        .with_color_encoding(ce)
        .encode(pixels)
        .expect("encode");

    let mut image = jxl_oxide::JxlImage::builder()
        .read(std::io::Cursor::new(&bytes[..]))
        .expect("jxl-oxide parse");
    // Decode in the file's own (PQ) color encoding.
    image.request_color_encoding(jxl_oxide::EnumColourEncoding {
        colour_space: jxl_oxide::color::ColourSpace::Rgb,
        white_point: jxl_oxide::color::WhitePoint::D65,
        primaries: jxl_oxide::color::Primaries::Srgb,
        tf: jxl_oxide::color::TransferFunction::Pq,
        rendering_intent: jxl_oxide::RenderingIntent::Relative,
    });
    let render = image.render_frame(0).expect("render");
    let fb = render.image_all_channels();
    let dec: Vec<u16> = fb
        .buf()
        .iter()
        .map(|&v| (v.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16)
        .collect();

    let src_u16: Vec<u16> = pixels
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| u16::from_ne_bytes([c[0], c[1]]))
        .collect();
    let r = pq_linear_img(&src_u16, w, h);
    let dimg = pq_linear_img(&dec, w, h);
    let params = ButteraugliParams::default().with_intensity_target(PEAK_NITS);
    let res = butteraugli_linear(r.as_ref(), dimg.as_ref(), &params).expect("butteraugli");
    (res.score, bytes)
}

#[test]
fn pq_lossy_quality_tracks_distance() {
    let (w, h) = (256usize, 256usize);
    let pixels = pq_ramp_rgb16(w, h);

    let (s1, b1) = encode_and_score(&pixels, w, h, 1.0);
    let (s4, b4) = encode_and_score(&pixels, w, h, 4.0);

    assert!(
        s1 < 8.0,
        "d=1 PQ-butteraugli {s1:.2} — destruction regime (~170) means the \
         intensity-scale fix regressed (issue #73)"
    );
    assert!(
        s1 < s4,
        "quality must track distance: d=1 score {s1:.2} !< d=4 score {s4:.2}"
    );
    assert!(
        b4.len() < b1.len(),
        "bytes must shrink with distance: d=4 {} !< d=1 {}",
        b4.len(),
        b1.len(),
    );

    // Signaling survives (debug-format the colour encoding; enum shape
    // differs across jxl-oxide versions, the string pin is stable).
    let image = jxl_oxide::JxlImage::builder()
        .read(std::io::Cursor::new(&b1[..]))
        .unwrap();
    let ce = format!("{:?}", image.image_header().metadata.colour_encoding);
    assert!(ce.contains("Pq"), "TF must stay PQ, got {ce}");
    let _ = TransferFunction::Pq;
}
