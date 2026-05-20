//! W44-29 multi-decoder smoke test for the high_d_photo_hint gate.
//!
//! Encodes 3 test cells with both `auto` (gate may fire), `forced_on`
//! and `forced_off`. For each output, confirms jxl-oxide decodes
//! cleanly. djxl is exercised separately via the existing CLI tests.
//!
//! Cells: F-D photo (gate fires) + screenshot (gate stays off) + photo
//! control at d=1.0 (gate stays off).
//!
//! Run:
//!   cargo run -p jxl-encoder --release --example w44_29_decoder_smoke

use image::GenericImageView;
use jxl_encoder::{LossyConfig, PixelLayout};
use std::io::Cursor;
use std::path::PathBuf;

const CELLS: &[(&str, &str, u8, f32)] = &[
    (
        "FD photo (gate fires)",
        "CID22/CID22-512/validation/1531677.png",
        5,
        4.0,
    ),
    ("screenshot (gate off)", "gb82-sc/imac_g3.png", 7, 4.0),
    (
        "photo @ d=1.0 (gate off)",
        "CID22/CID22-512/validation/1531677.png",
        7,
        1.0,
    ),
];

fn try_decode_jxl_oxide(bytes: &[u8]) -> Result<(usize, usize), String> {
    let reader = Cursor::new(bytes);
    let mut img = jxl_oxide::JxlImage::builder()
        .read(reader)
        .map_err(|e| format!("jxl-oxide read: {e:?}"))?;
    img.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
        jxl_oxide::RenderingIntent::Relative,
    ));
    let render = img
        .render_frame(0)
        .map_err(|e| format!("jxl-oxide render: {e:?}"))?;
    let fb = render.image_all_channels();
    Ok((fb.width(), fb.height()))
}

fn main() {
    let corpus = PathBuf::from(
        std::env::var("CODEC_CORPUS_DIR")
            .unwrap_or_else(|_| String::from("/home/lilith/work/codec-corpus")),
    );
    let mut all_pass = true;
    let out_dir = PathBuf::from("/tmp/w44_29_smoke");
    std::fs::create_dir_all(&out_dir).ok();

    for &(label, rel, effort, d) in CELLS {
        let path = corpus.join(rel);
        if !path.exists() {
            eprintln!("MISS {}", path.display());
            continue;
        }
        let img = image::open(&path).unwrap();
        let (w, h) = img.dimensions();
        let rgb = img.to_rgb8();

        for (mode_label, hint) in &[
            ("auto", None::<Option<bool>>),
            ("forced_on", Some(Some(true))),
            ("forced_off", Some(Some(false))),
        ] {
            let cfg = LossyConfig::new(d).with_effort(effort);
            let cfg = match hint {
                Some(h) => cfg.with_strategy_overrides(jxl_encoder::api::StrategyOverrides { high_d_photo_hint: *h, ..Default::default() }),
                None => cfg,
            };
            let bytes = cfg.encode(rgb.as_raw(), w, h, PixelLayout::Rgb8).unwrap();

            // Persist to disk so djxl can be checked manually if needed.
            let fname = format!(
                "{}_e{}_d{:.0}_{}.jxl",
                rel.replace('/', "_").replace(".png", ""),
                effort,
                d * 10.0,
                mode_label
            );
            std::fs::write(out_dir.join(&fname), &bytes).ok();

            let ox = try_decode_jxl_oxide(&bytes);
            let ox_ok = ox.is_ok();
            let pass = ox_ok;
            if !pass {
                all_pass = false;
            }
            println!(
                "{:<28} mode={:<10} e={} d={:.1} bytes={:>8} | jxl-oxide={}",
                label,
                mode_label,
                effort,
                d,
                bytes.len(),
                if ox_ok { "OK" } else { "FAIL" }
            );
            if !ox_ok {
                eprintln!("  jxl-oxide error: {:?}", ox.unwrap_err());
            }
        }
    }

    println!("\nOutputs written to {}", out_dir.display());
    if all_pass {
        println!("ALL DECODES PASS");
    } else {
        eprintln!("\nFAILURE: at least one decode failed");
        std::process::exit(1);
    }
}
