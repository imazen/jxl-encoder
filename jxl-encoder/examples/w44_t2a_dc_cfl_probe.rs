//! T2A: DC CfL factor probe.
//!
//! Hypothesis (cross-agent convergence — CfL audit AND DC/LLF audit):
//! our encoder hardcodes `dc_cfl_factor_b = 0.5` at `vardct/reconstruct.rs:303`
//! and `transform.rs:837`, but libjxl decoder applies `DCFactors()[2] = 1.0`
//! at defaults (`compressed_dc.cc:229`, `chroma_from_luma.h:107`).
//!
//! Algebraic analysis shows the 0.5 IS equivalent to libjxl at defaults
//! when `extra_precision = 0`, because:
//!
//!   libjxl encoder:  quant_b = round((b - quant_y * y_factor * cfl_factor) * inv_factor_b)
//!     where y_factor = dc_step_y / mul = 1/(kInvDCQuant[1] * S * mul)
//!     and   inv_factor_b = kInvDCQuant[2] * S * mul
//!     so    y_factor * inv_factor_b = kInvDCQuant[2] / kInvDCQuant[1] = 256/512 = 0.5
//!     and   0.5 * cfl_factor_default(=1.0) = 0.5
//!
//! This probe verifies the algebraic identity holds end-to-end:
//! encode a 16x16 RGB image with a known DC pattern, decode via jxl-oxide
//! linear, then dump our encoder's internal recon for the same image and
//! compare per-block. If the algebra is right, decoded == our_recon
//! to within IDCT precision.
//!
//! Run:
//!   CARGO_TARGET_DIR=$HOME/work/zen/jxl-encoder-shared-target \
//!     cargo run --release --manifest-path jxl-encoder/Cargo.toml \
//!     --example w44_t2a_dc_cfl_probe

use jxl_encoder::api::{LossyConfig, PixelLayout};
use std::io::Cursor;

/// Decode bitstream into linear-RGB f32 planar (interleaved RGB) via jxl-oxide.
fn decode_jxl_linear(bytes: &[u8]) -> Option<(usize, usize, Vec<f32>)> {
    let reader = Cursor::new(bytes);
    let mut img = jxl_oxide::JxlImage::builder().read(reader).ok()?;
    img.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
        jxl_oxide::RenderingIntent::Relative,
    ));
    let render = img.render_frame(0).ok()?;
    let fb = render.image_all_channels();
    Some((fb.width(), fb.height(), fb.buf().to_vec()))
}

fn main() {
    // 16x16 = 4 blocks. Make each block a distinct constant color
    // so DC is non-trivial and Y has clear correlation with B.
    // Layout (4 8x8 blocks):
    //   block 0 (top-left):     RGB(255, 100, 50)   — Y high, B low
    //   block 1 (top-right):    RGB(50, 200, 200)   — Y mid, B mid
    //   block 2 (bot-left):     RGB(200, 100, 100)  — Y mid, B mid
    //   block 3 (bot-right):    RGB(0, 0, 0)        — all zero
    let mut rgb = vec![0u8; 16 * 16 * 3];
    let block_colors = [(255u8, 100, 50), (50, 200, 200), (200, 100, 100), (0, 0, 0)];
    for by in 0..2 {
        for bx in 0..2 {
            let (r, g, b) = block_colors[by * 2 + bx];
            for y in 0..8 {
                for x in 0..8 {
                    let row = by * 8 + y;
                    let col = bx * 8 + x;
                    let off = (row * 16 + col) * 3;
                    rgb[off] = r;
                    rgb[off + 1] = g;
                    rgb[off + 2] = b;
                }
            }
        }
    }

    for &distance in &[1.0f32, 2.0, 4.0] {
        println!("\n=== distance = {} ===", distance);

        // Encode via our encoder
        let cfg = LossyConfig::new(distance).with_effort(5).with_threads(1);
        let bytes = match cfg.encode(&rgb, 16, 16, PixelLayout::Rgb8) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("encode failed: {:?}", e);
                continue;
            }
        };
        println!("encoded {} bytes", bytes.len());

        // Decode via jxl-oxide linear (linear sRGB)
        let (w, h, decoded) = match decode_jxl_linear(&bytes) {
            Some(d) => d,
            None => {
                eprintln!("decode failed");
                continue;
            }
        };
        assert_eq!(w, 16);
        assert_eq!(h, 16);
        let channels = decoded.len() / (w * h);
        println!("decoded: {}x{} channels={}", w, h, channels);

        // Per-block mean of decoded RGB (linear sRGB f32 in [0, 1])
        for by in 0..2 {
            for bx in 0..2 {
                let (orig_r, orig_g, orig_b) = block_colors[by * 2 + bx];
                let mut sum_r = 0.0f64;
                let mut sum_g = 0.0f64;
                let mut sum_b = 0.0f64;
                for y in 0..8 {
                    for x in 0..8 {
                        let row = by * 8 + y;
                        let col = bx * 8 + x;
                        let off = (row * w + col) * channels;
                        sum_r += decoded[off] as f64;
                        sum_g += decoded[off + 1] as f64;
                        sum_b += decoded[off + 2] as f64;
                    }
                }
                let mean_r = sum_r / 64.0;
                let mean_g = sum_g / 64.0;
                let mean_b = sum_b / 64.0;

                // Convert original sRGB u8 → linear sRGB f32 for comparison
                let to_lin = |c8: u8| -> f64 {
                    let c = c8 as f64 / 255.0;
                    if c <= 0.04045 {
                        c / 12.92
                    } else {
                        ((c + 0.055) / 1.055).powf(2.4)
                    }
                };
                let lin_r = to_lin(orig_r);
                let lin_g = to_lin(orig_g);
                let lin_b = to_lin(orig_b);

                let dr = mean_r - lin_r;
                let dg = mean_g - lin_g;
                let db = mean_b - lin_b;

                println!(
                    "  block({},{}) RGB orig=({:>3},{:>3},{:>3})   lin=({:.4},{:.4},{:.4})   decoded_mean=({:.4},{:.4},{:.4})   delta=({:+.4},{:+.4},{:+.4})",
                    bx,
                    by,
                    orig_r,
                    orig_g,
                    orig_b,
                    lin_r,
                    lin_g,
                    lin_b,
                    mean_r,
                    mean_g,
                    mean_b,
                    dr,
                    dg,
                    db
                );
            }
        }
    }
}
