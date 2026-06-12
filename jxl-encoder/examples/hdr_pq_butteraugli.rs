// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! PQ-aware butteraugli for the HDR lossy parity bench (issue #72).
//!
//! Takes two 16-bit PQ-encoded PNGs (reference + distorted — e.g. a
//! source HDR crop and a djxl-decoded lossy encode), applies the SMPTE
//! ST 2084 EOTF to BOTH through the identical pipeline, scales to
//! relative luminance with a 1000-nit peak window, and prints the
//! butteraugli distance (`intensity_target = 1000`).
//!
//! Both arms of any A/B go through this exact path, so relative
//! comparisons are fair; absolute values are NOT comparable to SDR
//! butteraugli scores.
//!
//! usage: hdr_pq_butteraugli <ref.png> <dist.png>

use butteraugli::{ButteraugliParams, butteraugli_linear};
use imgref::Img;
use rgb::RGB;

const PEAK_NITS: f32 = 1000.0;

fn pq_eotf(e: f32) -> f32 {
    // SMPTE ST 2084, output in absolute nits (10_000 * Y).
    const M1: f32 = 2610.0 / 16384.0;
    const M2: f32 = 2523.0 / 4096.0 * 128.0;
    const C1: f32 = 3424.0 / 4096.0;
    const C2: f32 = 2413.0 / 4096.0 * 32.0;
    const C3: f32 = 2392.0 / 4096.0 * 32.0;
    let ep = e.max(0.0).powf(1.0 / M2);
    let num = (ep - C1).max(0.0);
    let den = C2 - C3 * ep;
    10_000.0 * (num / den).powf(1.0 / M1)
}

fn load_pq_linear(path: &str) -> Img<Vec<RGB<f32>>> {
    let img = image::open(path).expect("open png").to_rgb16();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let px: Vec<RGB<f32>> = img
        .as_raw()
        .chunks_exact(3)
        .map(|c| {
            let f = |v: u16| (pq_eotf(v as f32 / 65535.0) / PEAK_NITS).min(1.0);
            RGB::new(f(c[0]), f(c[1]), f(c[2]))
        })
        .collect();
    Img::new(px, w, h)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let r = load_pq_linear(&args.next().expect("ref png"));
    let d = load_pq_linear(&args.next().expect("dist png"));
    let params = ButteraugliParams::default().with_intensity_target(PEAK_NITS);
    let res = butteraugli_linear(r.as_ref(), d.as_ref(), &params).expect("butteraugli");
    println!("{:.4}", res.score);
}
