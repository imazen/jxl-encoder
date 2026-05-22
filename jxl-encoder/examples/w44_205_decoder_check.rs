//! W44-205 decoder roundtrip check: confirm that the production W44-205
//! gate (buckets 2 + 4 disabled on top of W44-201 buckets 3 + 6) emits
//! bitstreams that decode cleanly via jxl-oxide on the 5 spot cells
//! from the Phase-1 probe + 1 PROTECT cell. Since the gate only changes
//! coefficient SCAN ORDER (Lehmer permutation header), not the
//! coefficient values themselves, decoded pixels are guaranteed
//! bit-identical between W44-201 baseline and W44-205 production.
//!
//! jxl-rs roundtrip is covered by the existing `tests/w44_*_decoder_roundtrip.rs`
//! pattern; this example focuses on jxl-oxide as a quick smoke test.
//!
//! Run:
//!   cargo run -p jxl-encoder --release \
//!     --features '__expert __internal_recon_hook butteraugli-loop ssim2-loop parallel' \
//!     --example w44_205_decoder_check

use jxl_encoder::api::{EncoderStrategy, LossyConfig, PixelLayout};
use std::path::Path;

const SPOT_CELLS: &[(&str, &str, f32, u8)] = &[
    (
        "LOSER_3637739_d4",
        "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/3637739.png",
        4.0,
        7,
    ),
    (
        "LOSER_297394_d5",
        "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/297394.png",
        5.0,
        7,
    ),
    (
        "LOSER_7062219_d4",
        "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/7062219.png",
        4.0,
        7,
    ),
    (
        "LOSER_1475938_d4",
        "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1475938.png",
        4.0,
        7,
    ),
    (
        "PROTECT_1189261_d4",
        "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1189261.png",
        4.0,
        7,
    ),
    (
        "SCRN_codec_wiki_d4",
        "/home/lilith/work/codec-corpus/gb82-sc/codec_wiki.png",
        4.0,
        7,
    ),
];

fn load_png(path: &Path) -> (Vec<u8>, u32, u32) {
    let img = image::open(path).expect("png decode");
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    (rgb.into_raw(), w, h)
}

fn encode(label: &str, path: &str, distance: f32, effort: u8) -> Vec<u8> {
    let (pixels, w, h) = load_png(Path::new(path));
    eprintln!(
        "  encoding {} ({}x{}) d={} e={}",
        label, w, h, distance, effort
    );
    let cfg = LossyConfig::new(distance)
        .with_effort(effort)
        .with_strategy(EncoderStrategy::Zenjxl);
    cfg.encode_request(w, h, PixelLayout::Rgb8)
        .encode(&pixels)
        .expect("encode")
}

fn decode_jxl_oxide(bytes: &[u8]) -> Result<(u32, u32), String> {
    use std::io::Cursor;
    let cursor = Cursor::new(bytes.to_vec());
    let mut image = jxl_oxide::JxlImage::builder()
        .read(cursor)
        .map_err(|e| format!("jxl_oxide parse: {}", e))?;
    image.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
        jxl_oxide::RenderingIntent::Relative,
    ));
    let render = image
        .render_frame(0)
        .map_err(|e| format!("jxl_oxide render: {}", e))?;
    let stream = render.stream();
    Ok((stream.width(), stream.height()))
}

fn main() {
    let mut total_pass = 0;
    let mut total_fail = 0;
    for &(label, path, distance, effort) in SPOT_CELLS {
        eprintln!("\n=== {} ===", label);
        let buf = encode(label, path, distance, effort);
        eprintln!("  encoded {} bytes", buf.len());
        let (pixels, w, h) = load_png(Path::new(path));
        let _ = pixels;
        match decode_jxl_oxide(&buf) {
            Ok((dw, dh)) if dw == w && dh == h => {
                eprintln!("  ✓ jxl-oxide PASS ({}x{})", dw, dh);
                total_pass += 1;
            }
            Ok((dw, dh)) => {
                eprintln!(
                    "  ✗ jxl-oxide FAIL DIM (got {}x{}, expected {}x{})",
                    dw, dh, w, h
                );
                total_fail += 1;
            }
            Err(e) => {
                eprintln!("  ✗ jxl-oxide FAIL: {}", e);
                total_fail += 1;
            }
        }
    }
    eprintln!(
        "\nTotal: {} pass, {} fail ({} cells)",
        total_pass,
        total_fail,
        SPOT_CELLS.len()
    );
    std::process::exit(if total_fail == 0 { 0 } else { 1 });
}
