// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing
//
// Ignored unless an explicit env var or fixture path is provided. Drive
// from the harness via `cargo test ... -- --ignored --nocapture` to
// surface any sanitize_xyb_planes / butteraugli-loop / splines / patches
// assert! fires across the v09/v11 sweep config grid. Runs in both
// debug and release (asserts now fire in both).

use jxl_encoder::{LossyConfig, PixelLayout};
use std::path::Path;

fn try_load(path: &str) -> Option<(Vec<u8>, u32, u32)> {
    if !Path::new(path).exists() {
        return None;
    }
    let img = image::open(path).ok()?;
    let w = img.width();
    let h = img.height();
    let rgb = img.to_rgb8();
    Some((rgb.into_raw(), w, h))
}

/// Run the v11 sweep config grid against `pixels`. Returns (panics, errors).
fn run_grid(pixels: &[u8], w: u32, h: u32, label: &str) -> (u32, u32) {
    use std::panic;
    let mut panics = 0u32;
    let mut errors = 0u32;
    for &distance in &[0.5f32, 1.0, 2.0, 4.0, 8.0] {
        for &effort in &[5u8, 7, 9] {
            for &biters in &[0u32, 1, 2] {
                let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
                    LossyConfig::new(distance)
                        .with_effort(effort)
                        .with_butteraugli_iters(biters)
                        .encode(pixels, w, h, PixelLayout::Rgb8)
                }));
                match result {
                    Ok(Ok(_)) => {}
                    Ok(Err(e)) => {
                        errors += 1;
                        eprintln!("[{label}] err d={distance} e={effort} b={biters}: {e:?}");
                    }
                    Err(_) => {
                        panics += 1;
                        eprintln!("[{label}] PANIC d={distance} e={effort} b={biters}");
                    }
                }
            }
        }
    }
    (panics, errors)
}

fn report(label: &str, panics: u32, errors: u32) {
    let total = 5 * 3 * 3;
    let ok = total - panics - errors;
    println!("[{label}] {ok}/{total} ok, {panics} panics, {errors} errors");
}

#[test]
#[ignore = "requires zentrain-corpus fixtures"]
fn repro_v11_sz1280() {
    let path = "/home/lilith/work/zentrain-corpus/mlp-tune/size-dense-renders/4cd6910a0b7b39365fda5df87618d091__sz1280.png";
    let Some((pixels, w, h)) = try_load(path) else {
        eprintln!("skip — fixture not present at {path}");
        return;
    };
    println!("loaded {w}x{h}");
    let (p, e) = run_grid(&pixels, w, h, "v11_sz1280");
    report("v11_sz1280", p, e);
    assert_eq!(
        p, 0,
        "panics found — see eprintln above for triggering configs"
    );
}

#[test]
#[ignore = "requires zentrain-corpus fixtures"]
fn repro_v09_sz96_via_corpus() {
    let path = "/home/lilith/work/zentrain-corpus/mlp-tune/size-dense-renders/4cd6910a0b7b39365fda5df87618d091__sz96.png";
    let Some((pixels, w, h)) = try_load(path) else {
        eprintln!("skip — fixture not present at {path}");
        return;
    };
    let (p, e) = run_grid(&pixels, w, h, "v09_sz96_corpus");
    report("v09_sz96_corpus", p, e);
    assert_eq!(p, 0);
}

#[test]
#[ignore = "requires codec-corpus"]
fn repro_clic2025_1024_4cd() {
    let path = "/home/lilith/work/codec-corpus/clic2025-1024/4cd6910a0b7b39365fda5df87618d091.png";
    let Some((pixels, w, h)) = try_load(path) else {
        eprintln!("skip — fixture not present at {path}");
        return;
    };
    let (p, e) = run_grid(&pixels, w, h, "clic_4cd");
    report("clic_4cd", p, e);
    assert_eq!(p, 0);
}

#[test]
#[ignore = "requires zentrain-corpus fixtures"]
fn repro_other_sz1280_siblings() {
    let paths = [
        "/home/lilith/work/zentrain-corpus/mlp-tune/size-dense-renders/b939ac34faa94b5d0e753d570edc7048__sz1280.png",
        "/home/lilith/work/zentrain-corpus/mlp-tune/size-dense-renders/22ea12c903e41583b7c469cb86040157__sz1280.png",
        "/home/lilith/work/zentrain-corpus/mlp-tune/size-dense-renders/1e2f9d41529197f10d32bfa68a1e0bcc__sz1280.png",
    ];
    let mut total_panics = 0u32;
    for path in paths {
        let Some((pixels, w, h)) = try_load(path) else {
            eprintln!("skip {path}");
            continue;
        };
        let label = path.rsplit('/').next().unwrap_or(path);
        let (p, e) = run_grid(&pixels, w, h, label);
        report(label, p, e);
        total_panics += p;
    }
    assert_eq!(total_panics, 0);
}
