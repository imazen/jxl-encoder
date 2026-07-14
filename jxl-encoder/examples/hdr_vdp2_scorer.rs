// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! File-based VDP2-lite scorer for the HDR scoreboard (issue #74).
//!
//! Takes two 16-bit PQ-encoded PNGs (reference + distorted — e.g. a
//! source HDR crop and a djxl-decoded lossy encode), applies the SMPTE
//! ST 2084 EOTF to BOTH through the identical pipeline, scales to
//! relative luminance with a 1000-nit peak window, and prints the
//! **shipped** VDP2-lite score (`intensity_target = 1000`).
//!
//! This deliberately calls `jxl_encoder::__internals::compare_vdp2_planar`
//! — the SAME VDP2-lite the buttloop steers on for PQ/HLG encodes (default
//! `HdrLoss`) — so the scoreboard judges HDR quality with the exact metric
//! the encoder optimizes against. The independent reimplementation in
//! `hdr_vdp2_chunk3_rd_sweep` is a validation counterfactual (different
//! bands/ppd/CSF/pooling on purpose); this one is the production metric.
//!
//! VDP2-lite is LUMINANCE-ONLY: it is blind to chroma error near the
//! primaries, which is exactly the failure mode cvvdp catches. The
//! scoreboard therefore pairs this with pq_butteraugli (and, where built,
//! cvvdp) into a multi-metric HDR guard — see run_scoreboard.py.
//!
//! Both arms of any A/B go through this exact path, so relative
//! comparisons are fair; absolute values are NOT comparable to SDR
//! scores or to pq_butteraugli.
//!
//! usage: hdr_vdp2_scorer <ref.png> <dist.png>

use jxl_encoder::__internals::compare_vdp2_planar;

const PEAK_NITS: f32 = 1000.0;

fn pq_eotf(e: f32) -> f32 {
    // SMPTE ST 2084, output in absolute nits (10_000 * Y). Identical to
    // hdr_pq_butteraugli's EOTF so the two HDR metrics linearize the same.
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

/// Load a 16-bit PQ PNG as three tightly-packed planar linear-RGB planes
/// in `[0, 1]` referenced to a 1000-nit peak (the convention VDP2-lite +
/// butteraugli both take). Returns `(r, g, b, width, height)`.
fn load_pq_planar(path: &str) -> (Vec<f32>, Vec<f32>, Vec<f32>, usize, usize) {
    let img = image::open(path).expect("open png").to_rgb16();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let n = w * h;
    let mut r = vec![0.0f32; n];
    let mut g = vec![0.0f32; n];
    let mut b = vec![0.0f32; n];
    let f = |v: u16| (pq_eotf(v as f32 / 65535.0) / PEAK_NITS).min(1.0);
    for (i, c) in img.as_raw().chunks_exact(3).enumerate() {
        r[i] = f(c[0]);
        g[i] = f(c[1]);
        b[i] = f(c[2]);
    }
    (r, g, b, w, h)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let ref_path = args
        .next()
        .expect("usage: hdr_vdp2_scorer <ref.png> <dist.png>");
    let dist_path = args
        .next()
        .expect("usage: hdr_vdp2_scorer <ref.png> <dist.png>");
    let (rr, rg, rb, w, h) = load_pq_planar(&ref_path);
    let (dr, dg, db, dw, dh) = load_pq_planar(&dist_path);
    assert_eq!((w, h), (dw, dh), "reference/distorted dimensions differ");
    // stride == width (tightly packed); intensity_target = 1000 nits.
    let res = compare_vdp2_planar(&rr, &rg, &rb, &dr, &dg, &db, w, h, w, PEAK_NITS)
        .expect("vdp2-lite compare");
    println!("{:.6}", res.score);
}
