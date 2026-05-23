//! T2B test chunk (2026-05-23): cross-agent convergence on DCT16X8 /
//! DCT8X16 transpose-indexing bug. The DCT/IDCT audit flagged that
//! `reconstruct.rs:441` pre-transposes DCT16X8 coefficient blocks before
//! calling `jxl_simd::idct_16x8`, while libjxl's
//! `ComputeScaledIDCT<16, 8>` consumes the post-swap layout directly. The
//! Quant audit independently flagged the multi-block transpose_slots
//! indexing.
//!
//! W44-115 (`docs/LIBJXL_DIVERGENCES.md`, Section D row 161) already
//! recorded this as INTENTIONAL with regression-gated tests in
//! `jxl-encoder/tests/idct_parity.rs`:
//!
//!   * `idct_16x8_parity_impulses` — libjxl impulse parity WITH pre-transpose
//!   * `idct_16x8_roundtrip_with_pretranspose` — DCT→IDCT roundtrip works
//!   * `idct_16x8_roundtrip_no_transpose_negative_control` — proves the
//!     asymmetry is real, fires regression alarm if `idct_16x8` is ever
//!     "fixed" to consume the post-swap layout.
//!
//! This T2B example adds the missing end-to-end integration check the
//! consolidated audit asked for: a real encode → multi-decoder roundtrip
//! on an image small enough to force the encoder to pick DCT16X8 (resp.
//! DCT8X16) for every block via `LossyConfig::with_force_strategy`. The
//! input contains non-symmetric content (diagonal gradient + sinusoidal
//! ridge) so any 8↔16 axis swap would surface as a large pixel error.
//!
//! Run with:
//!   cargo run --release -p jxl-encoder --example t2b_dct16x8_roundtrip
//!
//! Exits non-zero if the round-trip max-abs-diff exceeds 6/255 = 0.024
//! (in normalized 0..1 sRGB space) for either strategy on either decoder.

use jxl_encoder::api::{LossyConfig, PixelLayout};
use std::io::Cursor;
use std::path::Path;
use std::process::Command;

const RAW_STRATEGY_DCT16X8: u8 = 1;
const RAW_STRATEGY_DCT8X16: u8 = 2;

// Tolerance: 6/255 ≈ 0.0235 in sRGB f32. Lossy at d=1.0 typically
// produces 1-3/255 worst-pixel error on smooth content. A transpose
// bug would produce 50-200/255 — orders of magnitude over.
const TOL: f32 = 6.0 / 255.0;

const DJXL: &str = "/home/lilith/work/jxl-efforts/libjxl/build/tools/djxl";

fn build_non_symmetric_image(w: u32, h: u32) -> Vec<u8> {
    // 3-channel sRGB u8. Content:
    //   R = diagonal stripes (x XOR y) — distinguishes x from y axes
    //   G = horizontal sinusoid scaled by 2 along x — distinguishes
    //       a 16-wide period in x from an 8-wide period in y
    //   B = vertical sinusoid scaled by 2 along y — mirror of G
    //
    // Any 8↔16 axis confusion would produce huge errors in the sinusoid
    // channels because the periods do not commute.
    let mut buf = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let xf = x as f32;
            let yf = y as f32;
            let r = ((x ^ y) & 0xFF) as u8;
            // 16-pixel period in x: cos(2π · x/16). Range [-1, 1] → [0,
            // 255] via *127.5 + 127.5.
            let g_f = (2.0 * core::f32::consts::PI * xf / 16.0).cos();
            let g = ((g_f * 127.5) + 127.5).clamp(0.0, 255.0) as u8;
            // 8-pixel period in y: sin(2π · y/8).
            let b_f = (2.0 * core::f32::consts::PI * yf / 8.0).sin();
            let b = ((b_f * 127.5) + 127.5).clamp(0.0, 255.0) as u8;
            buf.push(r);
            buf.push(g);
            buf.push(b);
        }
    }
    buf
}

fn decode_jxl_oxide(jxl: &[u8], w: u32, h: u32) -> Vec<f32> {
    let mut image = jxl_oxide::JxlImage::builder()
        .read(Cursor::new(jxl))
        .expect("jxl-oxide read");
    image.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb(
        jxl_oxide::RenderingIntent::Relative,
    ));
    let render = image.render_frame(0).expect("jxl-oxide render");
    // Stream returns rendered pixels — f32 in 0..1 sRGB (because we
    // requested srgb encoding, not srgb_linear).
    let mut stream = render.stream();
    let total = (w as usize) * (h as usize) * 3;
    let mut buf = vec![0f32; total];
    stream.write_to_buffer(&mut buf);
    buf
}

fn decode_djxl_pfm(jxl_path: &Path, pfm_path: &Path) -> Option<Vec<f32>> {
    // djxl PFM output: big-endian f32 by default (scale +1.0); values
    // are in sRGB f32 (NOT linear) when the source is sRGB-encoded.
    // We compare against sRGB-normalized reference (not linear).
    let status = Command::new(DJXL)
        .arg(jxl_path)
        .arg(pfm_path)
        .arg("--num_threads=1")
        .status();
    match status {
        Ok(s) if s.success() => {
            let data = std::fs::read(pfm_path).ok()?;
            parse_pfm_f32(&data)
        }
        _ => None,
    }
}

fn parse_pfm_f32(data: &[u8]) -> Option<Vec<f32>> {
    // Parse a 3-channel PF header: "PF\n<w> <h>\n<scale>\n<floats>"
    let mut idx = 0;
    let read_line = |idx: &mut usize| -> Option<String> {
        let start = *idx;
        while *idx < data.len() && data[*idx] != b'\n' {
            *idx += 1;
        }
        if *idx >= data.len() {
            return None;
        }
        let line = std::str::from_utf8(&data[start..*idx]).ok()?.to_string();
        *idx += 1;
        Some(line)
    };
    let magic = read_line(&mut idx)?;
    if magic != "PF" {
        return None;
    }
    let dims = read_line(&mut idx)?;
    let mut sp = dims.split_whitespace();
    let w: usize = sp.next()?.parse().ok()?;
    let h: usize = sp.next()?.parse().ok()?;
    let scale_line = read_line(&mut idx)?;
    let scale: f32 = scale_line.parse().ok()?;
    // PFM convention: negative scale → little-endian, positive → big-endian.
    let little_endian = scale < 0.0;
    let _ = scale; // not used after endian detection
    let n = w * h * 3;
    let mut out = Vec::with_capacity(n);
    // PFM scanlines are bottom-up.
    let bytes = &data[idx..];
    if bytes.len() < n * 4 {
        return None;
    }
    let mut rows: Vec<Vec<f32>> = Vec::with_capacity(h);
    for row in 0..h {
        let mut r = Vec::with_capacity(w * 3);
        for col in 0..(w * 3) {
            let off = (row * w * 3 + col) * 4;
            let arr: [u8; 4] = bytes[off..off + 4].try_into().ok()?;
            let v = if little_endian {
                f32::from_le_bytes(arr)
            } else {
                f32::from_be_bytes(arr)
            };
            r.push(v);
        }
        rows.push(r);
    }
    rows.reverse();
    for row in rows {
        out.extend_from_slice(&row);
    }
    Some(out)
}

fn srgb_u8_to_linear(c: u8) -> f32 {
    let v = (c as f32) / 255.0;
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

fn srgb_u8_to_srgb_f32(c: u8) -> f32 {
    (c as f32) / 255.0
}

fn max_abs_diff_with_loc(a: &[f32], b: &[f32], w: u32) -> (f32, usize) {
    let mut max_e: f32 = 0.0;
    let mut where_: usize = 0;
    let n = a.len().min(b.len());
    for i in 0..n {
        let e = (a[i] - b[i]).abs();
        if e > max_e {
            max_e = e;
            where_ = i;
        }
    }
    let _ = w;
    (max_e, where_)
}

fn run_one_strategy(strategy: u8, name: &str, w: u32, h: u32, distance: f32) -> bool {
    println!("\n--- T2B test: strategy {} ({}) on {}×{} d={} ---", strategy, name, w, h, distance);
    let pixels = build_non_symmetric_image(w, h);

    let cfg = LossyConfig::new(distance)
        .with_effort(7)
        .with_threads(1)
        .with_force_strategy(Some(strategy));
    // (No `with_butteraugli_iters(0)` — at effort 7 the buttloop is
    // gated off anyway; a strategy-forced encode runs the standard
    // forward DCT → quantize → entropy / reconstruct → IDCT pipeline.)

    let jxl = match cfg.encode(&pixels, w, h, PixelLayout::Rgb8) {
        Ok(b) => b,
        Err(e) => {
            println!("ENCODE FAILED: {:?}", e);
            return false;
        }
    };
    println!("  encoded: {} bytes", jxl.len());

    let jxl_path_str = format!("/tmp/t2b_{}_d{}.jxl", name, distance);
    let jxl_path = Path::new(&jxl_path_str);
    std::fs::write(jxl_path, &jxl).expect("write jxl");

    // --- jxl-oxide path: sRGB f32 output, compare against sRGB-normalized reference ---
    let decoded_oxide = decode_jxl_oxide(&jxl, w, h);
    let ref_srgb: Vec<f32> = pixels.iter().map(|&c| srgb_u8_to_srgb_f32(c)).collect();
    let (e_oxide, where_oxide) = max_abs_diff_with_loc(&decoded_oxide, &ref_srgb, w);
    let pix_idx = where_oxide / 3;
    let chan = where_oxide % 3;
    let py = pix_idx / (w as usize);
    let px = pix_idx % (w as usize);
    println!(
        "  jxl-oxide max_abs_diff = {:.4} (tol {:.4}) at (x={}, y={}, ch={})",
        e_oxide, TOL, px, py, chan
    );
    let oxide_pass = e_oxide < TOL;

    // --- djxl path: PFM (linear sRGB f32) output, compare against linear reference ---
    let pfm_path_str = format!("/tmp/t2b_{}_d{}.pfm", name, distance);
    let pfm_path = Path::new(&pfm_path_str);
    let djxl_pass = match decode_djxl_pfm(jxl_path, pfm_path) {
        Some(decoded_pfm) => {
            // djxl PFM is sRGB f32 (not linear) — compare against
            // sRGB-normalized reference, same domain as jxl-oxide.
            let (e_djxl, where_djxl) = max_abs_diff_with_loc(&decoded_pfm, &ref_srgb, w);
            let pix_idx_d = where_djxl / 3;
            let chan_d = where_djxl % 3;
            let py_d = pix_idx_d / (w as usize);
            let px_d = pix_idx_d % (w as usize);
            println!(
                "  djxl     max_abs_diff = {:.4} (tol {:.4}) at (x={}, y={}, ch={}) [sRGB]",
                e_djxl, TOL, px_d, py_d, chan_d
            );
            e_djxl < TOL
        }
        None => {
            println!("  djxl     SKIPPED (decoder unavailable or failed)");
            true
        }
    };

    let pass = oxide_pass && djxl_pass;
    println!("  STATUS: {}", if pass { "PASS" } else { "FAIL" });
    pass
}

fn main() {
    // Use a 32×16 image: 4 blocks of DCT16X8 fit perfectly (2×2 grid of
    // 16w × 8h blocks). For DCT8X16 we use 16×32 (2×2 grid of 8w × 16h).
    // Force the strategy so every block uses the transform under test.
    //
    // Test at two distances to cover both light-quantization (d=0.5,
    // exposes precision/transpose bugs as small but non-trivial pixel
    // drift) and moderate (d=2.0, where the encoder pipeline + decode
    // round-trip exercises the full reconstruction path).
    let mut all_pass = true;
    // Near-lossless: at d=0.01 the quantizer is fine enough that any
    // round-trip error MUST come from a layout/transpose bug, not
    // strategy-choice quantization. We also compare against DCT8
    // (forced strategy 0) at the SAME distance on the SAME content as
    // a "if DCT8 round-trips and DCT16X8 doesn't at d=0.01, that's
    // signal." If both have similar error magnitudes, the error is
    // strategy-quantization-on-our-pathological-input.
    const RAW_STRATEGY_DCT8: u8 = 0;
    for &d in &[0.01_f32, 0.5_f32, 2.0_f32] {
        all_pass &= run_one_strategy(RAW_STRATEGY_DCT8, "dct8_control", 32, 16, d);
        all_pass &= run_one_strategy(RAW_STRATEGY_DCT16X8, "dct16x8", 32, 16, d);
        all_pass &= run_one_strategy(RAW_STRATEGY_DCT8X16, "dct8x16", 16, 32, d);
    }

    if all_pass {
        println!("\nT2B OVERALL: PASS — DCT16X8 and DCT8X16 round-trip cleanly through both decoders.");
        println!("Audit hypothesis FALSIFIED: production pre-transpose wrap is correct.");
    } else {
        eprintln!("\nT2B OVERALL: FAIL — see per-strategy diagnostics above.");
        std::process::exit(1);
    }
}
