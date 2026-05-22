//! W44-198 forced-lift probe (measurement-only).
//!
//! Asks: if we forced `high_d_photo_hint = Some(true)` on 3637739 +
//! other LOSER cells, would the W44-29 entropy_mul lift help close the
//! SSIM2 deficit and reduce bytes?
//!
//! Sweep: 5 images × 4 distances × 2 efforts × {AUTO, FORCE_ON, cjxl}.
//! Verifies whether the LIFT MECHANISM is even helpful on 3637739-class,
//! before we invest in a discriminator port.
//!
//! NO production code change.

use butteraugli::srgb_to_linear;
use image::GenericImageView;
use imgref::Img;
use jxl_encoder::api::StrategyOverrides;
use jxl_encoder::{LossyConfig, PixelLayout};
use rgb::RGB;
use std::io::Cursor;
use std::process::Command;

const CELLS: &[(&str, &str, &str)] = &[
    // (label, mask_med_class, path)
    // LOSER_DOMINANT: 3637739 mask_med=75.83 — no W44-29 gate fires
    (
        "3637739",
        "mask76_NOGATE",
        "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/3637739.png",
    ),
    // LOSER_DOMINANT: 7062219 mask_med=47.79 — W44-29 gate FIRES via mask<50
    (
        "7062219",
        "mask48_FIRES",
        "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/7062219.png",
    ),
    // LOSER_HEAVY: 1420710 mask_med=39.55 — W44-29 gate FIRES
    (
        "1420710",
        "mask40_FIRES",
        "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1420710.png",
    ),
    // LOSER_HEAVY: 297394 mask_med=62 — no gate fires
    (
        "297394",
        "mask62_NOGATE",
        "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/297394.png",
    ),
    // LOSER_HEAVY: 1475938 mask_med=89 — no gate fires
    (
        "1475938",
        "mask89_NOGATE",
        "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1475938.png",
    ),
];

const DISTANCES: &[f32] = &[2.0, 3.0, 4.0, 5.0];
const EFFORTS: &[u8] = &[5, 7];

fn encode_with_hint(rgb: &[u8], w: u32, h: u32, d: f32, effort: u8, force: Option<bool>) -> Vec<u8> {
    let mut cfg = LossyConfig::new(d).with_effort(effort);
    cfg = cfg.with_strategy_overrides(StrategyOverrides {
        high_d_photo_hint: force,
        ..Default::default()
    });
    cfg.encode(rgb, w, h, PixelLayout::Rgb8).expect("encode")
}

fn decode_linear(bytes: &[u8]) -> Option<(usize, usize, Vec<f32>)> {
    let reader = Cursor::new(bytes);
    let mut img = jxl_oxide::JxlImage::builder().read(reader).ok()?;
    img.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
        jxl_oxide::RenderingIntent::Relative,
    ));
    let render = img.render_frame(0).ok()?;
    let fb = render.image_all_channels();
    Some((fb.width(), fb.height(), fb.buf().to_vec()))
}

fn linear_to_srgb_u8(v: f32) -> u8 {
    let c = v.clamp(0.0, 1.0);
    let s = if c <= 0.003_130_8 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (s * 255.0).round() as u8
}

fn cjxl_bin() -> std::path::PathBuf {
    if let Ok(p) = std::env::var("CJXL") {
        return std::path::PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| "/home/lilith".into());
    std::path::PathBuf::from(home).join("work/jxl-efforts/libjxl/build/tools/cjxl")
}

fn cjxl_encode(src: &str, d: f32, effort: u8) -> Vec<u8> {
    let tmp = format!("/tmp/w44_198_cjxl_{}.jxl", std::process::id());
    let _out = Command::new(cjxl_bin())
        .args(["-d", &d.to_string(), "-e", &effort.to_string(), src, &tmp])
        .output()
        .expect("cjxl");
    let data = std::fs::read(&tmp).expect("cjxl output read");
    let _ = std::fs::remove_file(&tmp);
    data
}

fn compute_ssim2(
    bytes: &[u8],
    orig_srgb: &Img<Vec<[u8; 3]>>,
) -> (usize, f64) {
    let (dw, dh, dec_linear) = decode_linear(bytes).expect("decode failed");
    let dec_srgb_pix: Vec<[u8; 3]> = dec_linear
        .chunks(3)
        .map(|c| {
            [
                linear_to_srgb_u8(c[0]),
                linear_to_srgb_u8(c[1]),
                linear_to_srgb_u8(c[2]),
            ]
        })
        .collect();
    let dec_srgb_img = Img::new(dec_srgb_pix, dw, dh);
    let ssim2 = fast_ssim2::compute_ssimulacra2(orig_srgb.as_ref(), dec_srgb_img.as_ref())
        .unwrap_or(f64::NAN);
    (bytes.len(), ssim2)
}

fn main() {
    eprintln!("# W44-198 forced-lift probe");
    eprintln!(
        "# CELLS: {}, DISTANCES: {:?}, EFFORTS: {:?}",
        CELLS.len(),
        DISTANCES,
        EFFORTS
    );
    println!(
        "image\tmask_class\teffort\tdistance\tmode\tbytes\tssim2\t\
delta_bytes_pct_vs_cjxl\tdelta_ssim2_vs_cjxl"
    );

    for (label, mask_class, path) in CELLS {
        let img = image::open(path).expect("decode png");
        let (w, h) = img.dimensions();
        let rgb_u8 = img.to_rgb8().into_raw();
        // Build orig_srgb Img<Vec<[u8;3]>>
        let orig_srgb_pix: Vec<[u8; 3]> = rgb_u8
            .chunks(3)
            .map(|c| [c[0], c[1], c[2]])
            .collect();
        let orig_srgb_img = Img::new(orig_srgb_pix, w as usize, h as usize);
        let _orig_linear_for_butt: Vec<RGB<f32>> = rgb_u8
            .chunks(3)
            .map(|c| RGB::new(srgb_to_linear(c[0]), srgb_to_linear(c[1]), srgb_to_linear(c[2])))
            .collect();

        for &effort in EFFORTS {
            for &d in DISTANCES {
                let bytes_auto = encode_with_hint(&rgb_u8, w, h, d, effort, None);
                let (len_auto, ssim_auto) = compute_ssim2(&bytes_auto, &orig_srgb_img);

                let bytes_force = encode_with_hint(&rgb_u8, w, h, d, effort, Some(true));
                let (len_force, ssim_force) = compute_ssim2(&bytes_force, &orig_srgb_img);

                let bytes_cjxl = cjxl_encode(path, d, effort);
                let (len_cjxl, ssim_cjxl) = compute_ssim2(&bytes_cjxl, &orig_srgb_img);

                let dba = (len_auto as f64 - len_cjxl as f64) / len_cjxl as f64 * 100.0;
                let dbf = (len_force as f64 - len_cjxl as f64) / len_cjxl as f64 * 100.0;
                let dbf_vs_auto = (len_force as f64 - len_auto as f64) / len_auto as f64 * 100.0;

                println!(
                    "{}\t{}\te{}\td={:.2}\tAUTO\t{}\t{:.4}\t{:+.3}\t{:+.4}",
                    label, mask_class, effort, d, len_auto, ssim_auto, dba, ssim_auto - ssim_cjxl
                );
                println!(
                    "{}\t{}\te{}\td={:.2}\tFORCE_ON\t{}\t{:.4}\t{:+.3}\t{:+.4}",
                    label, mask_class, effort, d, len_force, ssim_force, dbf, ssim_force - ssim_cjxl
                );
                println!(
                    "{}\t{}\te{}\td={:.2}\tcjxl\t{}\t{:.4}\t{:+.3}\t{:+.4}",
                    label, mask_class, effort, d, len_cjxl, ssim_cjxl, 0.0, 0.0
                );
                eprintln!(
                    "{}/{} e{} d={:.2}: AUTO {}B SSIM2 {:.3} ({:+.3}vsCJXL) | FORCE {}B SSIM2 {:.3} (Δb={:+.2}%vsAUTO,Δss={:+.3}vsAUTO) | cjxl {}B SSIM2 {:.3}",
                    label, mask_class, effort, d,
                    len_auto, ssim_auto, ssim_auto - ssim_cjxl,
                    len_force, ssim_force, dbf_vs_auto, ssim_force - ssim_auto,
                    len_cjxl, ssim_cjxl
                );
            }
        }
    }
}
