// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later.
//
//! Phase 2 multi-decoder spot-check (acceptance gate (k)): encodes
//! a small set of cells under each `DisplayConfig` and pipes the
//! output through djxl + jxl-rs. Pass criteria: every cell decodes
//! cleanly through every decoder.
//!
//! Run via:
//!   cargo run --release -p jxl-encoder --features '__expert butteraugli-loop cvvdp-loop ssim2-loop parallel' --example cvvdp_display_phase2_spotcheck

use jxl_encoder::api::{
    DisplayConfig, LossyConfig, PerceptualDevice, PerceptualMetric, PixelLayout,
};
use std::io::Write;
use std::path::Path;
use std::process::Command;
use std::time::Instant;

const DJXL_PATH: &str = "/home/lilith/work/jxl-efforts/libjxl/build/tools/djxl";
const JXL_RS_PATH: &str = "/home/lilith/work/third-party/jxl-rs/target/release/jxl_cli";

fn load_rgb(path: &Path) -> (Vec<u8>, u32, u32) {
    let img = image::open(path).expect("open png");
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    (rgb.as_raw().clone(), w, h)
}

fn decode_via_subprocess(cmd: &str, jxl: &[u8], tag: &str) -> Result<(), String> {
    let unique = format!("/tmp/cvvdp_p2_spotcheck_{}.jxl", std::process::id());
    let out_png = format!("/tmp/cvvdp_p2_spotcheck_{}.png", std::process::id());
    std::fs::write(&unique, jxl).map_err(|e| format!("write tmp jxl: {e}"))?;
    let out = Command::new(cmd)
        .arg(&unique)
        .arg(&out_png)
        .output()
        .map_err(|e| format!("{tag} spawn: {e}"))?;
    let _ = std::fs::remove_file(&unique);
    let _ = std::fs::remove_file(&out_png);
    if !out.status.success() {
        return Err(format!(
            "{tag} exit={:?} stderr={}",
            out.status.code(),
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    Ok(())
}

fn decode_via_oxide(jxl: &[u8]) -> Result<(usize, usize), String> {
    let reader = std::io::Cursor::new(jxl);
    let mut img = jxl_oxide::JxlImage::builder()
        .read(reader)
        .map_err(|e| format!("oxide read: {e:?}"))?;
    img.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
        jxl_oxide::RenderingIntent::Relative,
    ));
    let render = img
        .render_frame(0)
        .map_err(|e| format!("oxide render: {e:?}"))?;
    let fb = render.image_all_channels();
    Ok((fb.width(), fb.height()))
}

fn main() {
    // 3 cells × 3 displays = 9 spot-check cells. Photo cells only —
    // small enough to finish quickly but exercise the cvvdp dispatch.
    // Effort 8 to ensure the cvvdp buttloop actually fires (gated at
    // speed_tier <= kKitten = effort >= 8); at e < 8 the encoder runs
    // butteraugli regardless of metric setting and display dispatch
    // has no observable effect on bytes.
    let cells: &[(&str, &str, f32, u8)] = &[
        (
            "CID22",
            "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1025469.png",
            1.0,
            8,
        ),
        (
            "CID22",
            "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1189261.png",
            3.0,
            8,
        ),
        (
            "CID22",
            "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1418519.png",
            2.0,
            8,
        ),
    ];

    let out_path = "benchmarks/cvvdp_display_phase2_spotcheck_2026-05-26.tsv";
    let _ = std::fs::create_dir_all(Path::new(out_path).parent().unwrap());
    let mut f = std::fs::File::create(out_path).expect("create tsv");
    writeln!(
        f,
        "image\tdistance\teffort\tdisplay\tbytes\tdecode_oxide\tdecode_djxl\tdecode_jxlrs\twall_ms"
    )
    .unwrap();

    let mut total = 0usize;
    let mut passed = 0usize;
    for &(corpus, path, distance, effort) in cells {
        let (pixels, w, h) = load_rgb(Path::new(path));
        let basename = Path::new(path)
            .file_name()
            .unwrap()
            .to_string_lossy()
            .to_string();
        for &display in &[
            DisplayConfig::WebSdr80,
            DisplayConfig::Phone,
            DisplayConfig::Tv,
        ] {
            let cfg = LossyConfig::new(distance)
                .with_effort(effort)
                .with_perceptual_metric(PerceptualMetric::Cvvdp)
                .with_perceptual_device(PerceptualDevice::Auto)
                .with_target_display(display);

            let t0 = Instant::now();
            let jxl = match cfg.encode(&pixels, w, h, PixelLayout::Rgb8) {
                Ok(b) => b,
                Err(e) => {
                    eprintln!(
                        "[{basename} d={distance} e={effort} {display:?}] encode failed: {e:?}"
                    );
                    continue;
                }
            };
            let wall = t0.elapsed().as_secs_f64() * 1000.0;

            let oxide_tag = match decode_via_oxide(&jxl) {
                Ok((dw, dh)) if dw as u32 == w && dh as u32 == h => "PASS".to_string(),
                Ok((dw, dh)) => format!("FAIL_DIMS_{dw}x{dh}"),
                Err(e) => format!("FAIL:{}", e.split(':').next().unwrap_or("err")),
            };
            let djxl_tag = match decode_via_subprocess(DJXL_PATH, &jxl, "djxl") {
                Ok(()) => "PASS".to_string(),
                Err(e) => format!("FAIL:{}", e.split('\n').next().unwrap_or("")),
            };
            let jxlrs_tag = match decode_via_subprocess(JXL_RS_PATH, &jxl, "jxlrs") {
                Ok(()) => "PASS".to_string(),
                Err(e) => format!("FAIL:{}", e.split('\n').next().unwrap_or("")),
            };

            total += 3;
            if oxide_tag == "PASS" {
                passed += 1;
            }
            if djxl_tag == "PASS" {
                passed += 1;
            }
            if jxlrs_tag == "PASS" {
                passed += 1;
            }

            writeln!(
                f,
                "{}\t{:.1}\t{}\t{:?}\t{}\t{}\t{}\t{}\t{:.1}",
                basename,
                distance,
                effort,
                display,
                jxl.len(),
                oxide_tag,
                djxl_tag,
                jxlrs_tag,
                wall
            )
            .unwrap();
            eprintln!(
                "[{basename} d={distance} e={effort} {display:?}] {} bytes — oxide={oxide_tag} djxl={djxl_tag} jxlrs={jxlrs_tag}",
                jxl.len(),
            );
            let _ = corpus;
        }
    }

    eprintln!(
        "\n=== Phase 2 multi-decoder spot-check: {}/{} PASS ===",
        passed, total
    );
    std::process::exit(if passed == total { 0 } else { 1 });
}
