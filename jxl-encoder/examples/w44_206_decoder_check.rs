//! W44-206 multi-decoder sanity check: even though W44-206 is HONEST-STOP
//! (no production change), verify that bitstreams produced under the
//! various multiplier values DO decode cleanly via jxl-oxide. This is a
//! belt-and-suspenders check: scan-order changes via the cost-benefit
//! gate are spec-compliant, but a regression in encoder framing would
//! show up here.

use jxl_encoder::api::{EncoderStrategy, LossyConfig, PixelLayout};
use std::path::Path;

fn load_png(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
    let img = image::open(path).ok()?;
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    Some((rgb.into_raw(), w, h))
}

fn encode_with(
    pixels: &[u8],
    w: u32,
    h: u32,
    distance: f32,
    savings_factor: f32,
    disable_w44_201: bool,
    disable_w44_205: bool,
) -> Vec<u8> {
    // SAFETY: Single-threaded probe. No other threads access env vars
    // concurrently. Race-free in this main()-driven sequential bench.
    unsafe {
        std::env::set_var("JXL_W44_201_SAVINGS_FACTOR", format!("{}", savings_factor));
        if disable_w44_201 {
            std::env::set_var("JXL_W44_201_FORCE_LEGACY_LARGE_BUCKETS", "1");
        } else {
            std::env::remove_var("JXL_W44_201_FORCE_LEGACY_LARGE_BUCKETS");
        }
        if disable_w44_205 {
            std::env::set_var("JXL_W44_205_FORCE_LEGACY_MEDIUM_BUCKETS", "1");
        } else {
            std::env::remove_var("JXL_W44_205_FORCE_LEGACY_MEDIUM_BUCKETS");
        }
    }
    let cfg = LossyConfig::new(distance)
        .with_effort(7)
        .with_strategy(EncoderStrategy::Zenjxl);
    cfg.encode_request(w, h, PixelLayout::Rgb8)
        .encode(pixels)
        .expect("encode")
}

fn try_oxide(buf: &[u8]) -> Result<(u32, u32), String> {
    let reader = std::io::Cursor::new(buf);
    let mut image = jxl_oxide::JxlImage::builder()
        .read(reader)
        .map_err(|e| format!("read: {e}"))?;
    image.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
        jxl_oxide::RenderingIntent::Relative,
    ));
    let render = image.render_frame(0).map_err(|e| format!("render: {e}"))?;
    let img = render.image_all_channels();
    Ok((img.width() as u32, img.height() as u32))
}

fn main() {
    // 5 spot cells × (baseline 1.0 + f=0.3 isolated + f=0.3 additive)
    let cells: Vec<(&str, &str, f32)> = vec![
        (
            "LOSER_3637739_d4",
            "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/3637739.png",
            4.0,
        ),
        (
            "PROTECT_1531677_d4",
            "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1531677.png",
            4.0,
        ),
        (
            "PROTECT_1418519_d4",
            "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1418519.png",
            4.0,
        ),
        (
            "SCRN_terminal_d4",
            "/home/lilith/work/codec-corpus/gb82-sc/terminal.png",
            4.0,
        ),
        (
            "SCRN_windows95_d4",
            "/home/lilith/work/codec-corpus/gb82-sc/windows95.png",
            4.0,
        ),
    ];

    let modes: Vec<(&str, f32, bool, bool)> = vec![
        ("baseline_f1.0_gates_on", 1.0, false, false),
        ("isolated_f0.3_gates_off", 0.3, true, true),
        ("additive_f0.3_gates_on", 0.3, false, false),
    ];

    let mut pass = 0;
    let mut fail = 0;
    for (lbl, path, d) in &cells {
        let Some((pixels, w, h)) = load_png(Path::new(path)) else {
            eprintln!("missing: {}", path);
            continue;
        };
        for (mode_name, f, dis201, dis205) in &modes {
            let buf = encode_with(&pixels, w, h, *d, *f, *dis201, *dis205);
            print!("{}/{} ({} bytes): ", lbl, mode_name, buf.len());
            match try_oxide(&buf) {
                Ok((dw, dh)) => {
                    if dw == w && dh == h {
                        println!("PASS ({}x{})", dw, dh);
                        pass += 1;
                    } else {
                        println!("DIM_MISMATCH (got {}x{}, want {}x{})", dw, dh, w, h);
                        fail += 1;
                    }
                }
                Err(e) => {
                    println!("FAIL: {}", e);
                    fail += 1;
                }
            }
        }
    }

    println!(
        "\nTotal: {} pass, {} fail ({} cells × {} modes)",
        pass,
        fail,
        cells.len(),
        modes.len()
    );
    if fail > 0 {
        std::process::exit(1);
    }
}
