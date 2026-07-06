// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Conformance regression test for **imazen/jxl-encoder#94**: true VarDCT
//! near-lossless below distance 0.03 must produce a **spec-conformant**
//! bitstream that a strict reference decoder accepts — not just one our own
//! decoder can read.
//!
//! History: a prior fix (008499e1) widened quantized DC `i16 -> i32` so the
//! reconstruction stopped saturating, but capped sub-0.03 lossy behind a
//! `VARDCT_MIN_LOSSY_DISTANCE = 0.03` floor because sub-floor encodes still
//! produced a stream `jxl-oxide` rejected with "ANS stream verification
//! failed". #94 fixes the ROOT CAUSE and removes that floor.
//!
//! Root cause: at distance < ~0.025 the fine DC quantiser produces quantized
//! DC exceeding the `i16` range (`|DC| > 32767`), while the file header
//! unconditionally signalled `modular_16bit_buffer_sufficient = true`. A
//! conformant decoder honouring that promise (jxl-oxide's `narrow_modular`
//! path) reconstructs the LF/DC modular image into `i16` sample buffers; the
//! oversized DC wraps there, corrupting the neighbours fed back into the DC
//! Weighted-Predictor, which diverges the modular context derivation and
//! desynchronises the DC ANS stream — the final-state check (`0x130000`) then
//! fails. The fix signals `modular_16bit_buffer_sufficient = false` whenever
//! the DC overflows `i16`, so the decoder uses `i32` buffers. (The Huffman
//! entropy path was also sized to the actual max token so large-DC symbols
//! `>= 64` no longer panic.)
//!
//! This test drives the exact reported failure: encode `frymire-srgb`
//! (high-contrast screen content — large Y DC) at distances **below** the old
//! floor, decode with BOTH the pure-Rust `zenjxl-decoder` AND the strict
//! `jxl-oxide` reference decoder, and assert:
//!   1. jxl-oxide ACCEPTS the frame (the spec-conformance gate — this is what
//!      failed pre-fix and is the core of #94);
//!   2. both decoders reconstruct near-lossless (PSNR >= 40 dB);
//!   3. a finer distance never decodes dramatically worse than a coarser one.
//!
//! Pre-fix (root cause unfixed, floor bypassed) jxl-oxide REJECTS distance
//! 0.01 / 0.02 → assertion (1) fails. Post-fix all three distances pass.
//!
//! No runtime skip: the fixture is a committed file (`tests/images/`) and a
//! load failure is a hard panic, never a silent early return.

use jxl_encoder::{LossyConfig, PixelLayout};

/// Load the committed sRGB fixture and crop to a bounded, high-contrast region.
/// `frymire-srgb.png` is a vivid screen illustration; bright saturated blocks
/// give large Y-channel DC — the content class whose quantized DC overflows
/// `i16` at fine distances.
fn load_fixture() -> (Vec<u8>, u32, u32) {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/images/frymire-srgb.png");
    let img = image::open(path)
        .unwrap_or_else(|e| panic!("committed fixture {path} failed to load: {e}"))
        .to_rgb8();
    let (w, h) = img.dimensions();
    let (cw, ch) = (w.min(384), h.min(384));
    let mut out = Vec::with_capacity((cw * ch * 3) as usize);
    for y in 0..ch {
        for x in 0..cw {
            out.extend_from_slice(&img.get_pixel(x, y).0);
        }
    }
    (out, cw, ch)
}

fn psnr_u8(src_rgb: &[u8], dec_rgb: &[u8]) -> f64 {
    assert_eq!(src_rgb.len(), dec_rgb.len(), "PSNR buffer length mismatch");
    let mut sse = 0.0f64;
    for (s, d) in src_rgb.iter().zip(dec_rgb.iter()) {
        let e = *s as i32 - *d as i32;
        sse += (e * e) as f64;
    }
    let mse = sse / src_rgb.len() as f64;
    if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (255.0 * 255.0 / mse).log10()
    }
}

/// Decode with the pure-Rust `zenjxl-decoder` and return RGB PSNR (dB) vs source.
fn zenjxl_psnr(jxl: &[u8], src_rgb: &[u8], w: u32, h: u32) -> f64 {
    let decoded = zenjxl_decoder::decode(jxl)
        .unwrap_or_else(|e| panic!("zenjxl-decoder rejected the #94 bitstream: {e:?}"));
    assert_eq!(
        (decoded.width as u32, decoded.height as u32),
        (w, h),
        "zenjxl-decoder dimensions differ from source"
    );
    assert_eq!(decoded.channels, 4, "expected RGBA output for a color image");
    // Drop alpha to compare against the RGB source.
    let mut rgb = Vec::with_capacity((w * h * 3) as usize);
    for px in decoded.data.chunks_exact(4) {
        rgb.extend_from_slice(&px[..3]);
    }
    psnr_u8(src_rgb, &rgb)
}

/// Decode with the strict `jxl-oxide` reference decoder (in-process). Returns
/// `Ok(psnr_db)` if the frame is ACCEPTED (spec-conformant), `Err(reason)` if
/// rejected — a non-conformant DC ANS stream fails `render_frame` here.
fn jxl_oxide_conformance_psnr(jxl: &[u8], src_rgb: &[u8], w: u32, h: u32) -> Result<f64, String> {
    let cursor = std::io::Cursor::new(jxl);
    let mut img = jxl_oxide::JxlImage::builder()
        .read(cursor)
        .map_err(|e| format!("jxl-oxide read: {e:?}"))?;
    // Request plain sRGB so the rendered f32 maps directly onto the sRGB u8 source.
    img.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb(
        jxl_oxide::RenderingIntent::Relative,
    ));
    // The spec-conformance gate: a non-conformant stream (#94 pre-fix) errors here.
    let render = img
        .render_frame(0)
        .map_err(|e| format!("jxl-oxide REJECTED frame (spec non-conformance): {e:?}"))?;
    let fb = render.image_all_channels();
    assert_eq!(
        (fb.width() as u32, fb.height() as u32),
        (w, h),
        "jxl-oxide dimensions differ from source"
    );
    assert!(fb.channels() >= 3, "jxl-oxide returned < 3 channels");
    let ch = fb.channels();
    let buf = fb.buf();
    let mut rgb = Vec::with_capacity((w * h * 3) as usize);
    for px in buf.chunks_exact(ch) {
        for c in 0..3 {
            rgb.push((px[c].clamp(0.0, 1.0) * 255.0).round() as u8);
        }
    }
    Ok(psnr_u8(src_rgb, &rgb))
}

#[test]
fn sub_floor_vardct_is_spec_conformant_issue94() {
    let (rgb, w, h) = load_fixture();

    // Distances below the old 0.03 floor (0.01, 0.02) exercise the i16-DC
    // overflow the fix addresses; 0.03 is the boundary control (DC fits i16).
    let distances = [0.01f32, 0.02, 0.03];
    let mut zenjxl_psnrs = Vec::new();
    let mut oxide_psnrs = Vec::new();

    for &d in &distances {
        let jxl = LossyConfig::new(d)
            .encode(&rgb, w, h, PixelLayout::Rgb8)
            .unwrap_or_else(|e| panic!("encode at distance {d} failed: {e:?}"));

        // (1) Spec-conformance gate — jxl-oxide must ACCEPT. This is the #94 bug.
        let oxide = jxl_oxide_conformance_psnr(&jxl, &rgb, w, h).unwrap_or_else(|e| {
            panic!(
                "distance-{d} VarDCT stream is NOT spec-conformant — {e}. \
                 #94 has regressed (sub-0.03 DC overflows i16 while the header \
                 still signals modular_16bit_buffer_sufficient = true)."
            )
        });

        // Both decoders must reconstruct near-lossless (not accept-then-garbage).
        let zx = zenjxl_psnr(&jxl, &rgb, w, h);
        assert!(
            zx >= 40.0,
            "distance-{d} zenjxl-decoder PSNR {zx:.2} dB is not near-lossless (>= 40 dB)"
        );
        assert!(
            oxide >= 40.0,
            "distance-{d} jxl-oxide PSNR {oxide:.2} dB is not near-lossless (>= 40 dB) — \
             accepted but reconstructed garbage"
        );

        zenjxl_psnrs.push(zx);
        oxide_psnrs.push(oxide);
    }

    // (3) Monotonicity: a FINER distance must never decode dramatically worse
    // than a coarser one. distances is coarsening (0.01 -> 0.02 -> 0.03), so
    // each earlier (finer) entry should be >= the next (coarser) minus noise.
    for pair in [&zenjxl_psnrs, &oxide_psnrs] {
        for i in 0..pair.len() - 1 {
            let (finer, coarser) = (pair[i], pair[i + 1]);
            assert!(
                finer >= coarser - 1.5,
                "distance-{} PSNR {finer:.2} dB is worse than distance-{} PSNR {coarser:.2} dB \
                 — finer distance decoded worse (#94 monotonicity inversion)",
                distances[i],
                distances[i + 1]
            );
        }
    }
}
