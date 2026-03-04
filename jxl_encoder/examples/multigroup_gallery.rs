//! Encode several multi-group CLIC images and create visual comparisons.
use std::io::Cursor;
use std::path::Path;

use jxl_encoder::{LossyConfig, PixelLayout};

fn encode_and_compare(img_path: &str, label: &str, out_dir: &std::path::Path) {
    let path = Path::new(img_path);
    if !path.exists() {
        eprintln!("SKIP: {} not found", img_path);
        return;
    }

    eprintln!("=== {} ===", label);
    let img = image::open(img_path).expect("Could not open image");
    let rgb = img.to_rgb8();
    let (width, height) = (rgb.width(), rgb.height());
    eprintln!(
        "  Size: {}x{} ({} groups)",
        width,
        height,
        (width as usize).div_ceil(256) * (height as usize).div_ceil(256)
    );

    // Encode using public API
    let bytes = LossyConfig::new(1.0)
        .encode(rgb.as_raw(), width, height, PixelLayout::Rgb8)
        .expect("Encoding failed");
    let orig_bytes = width as usize * height as usize * 3;
    eprintln!(
        "  Encoded: {} bytes ({:.1}:1 ratio, {:.2} bpp)",
        bytes.len(),
        orig_bytes as f64 / bytes.len() as f64,
        bytes.len() as f64 * 8.0 / (width as usize * height as usize) as f64
    );

    // Save JXL
    let jxl_path = format!("{}/{}.jxl", out_dir.display(), label);
    std::fs::write(&jxl_path, &bytes).expect("Failed to write JXL");

    // Decode with jxl-oxide
    let reader = Cursor::new(&bytes);
    let image_dec = jxl_oxide::JxlImage::builder()
        .read(reader)
        .expect("Parse failed");
    let render = image_dec.render_frame(0).expect("Render failed");
    let fb = render.image_all_channels();
    let decoded = fb.buf();

    // jxl_oxide returns sRGB nonlinear floats (no request_color_encoding call)
    let mut output_img = image::RgbImage::new(width, height);
    for (i, pixel) in output_img.pixels_mut().enumerate() {
        let r = (decoded[i * 3].clamp(0.0, 1.0) * 255.0).round() as u8;
        let g = (decoded[i * 3 + 1].clamp(0.0, 1.0) * 255.0).round() as u8;
        let b = (decoded[i * 3 + 2].clamp(0.0, 1.0) * 255.0).round() as u8;
        *pixel = image::Rgb([r, g, b]);
    }

    // Save decoded
    let decoded_path = format!("{}/{}_decoded.png", out_dir.display(), label);
    output_img
        .save(&decoded_path)
        .expect("Failed to save decoded");

    // Save original (downscaled for montage if needed)
    let orig_path = format!("{}/{}_original.png", out_dir.display(), label);
    img.save(&orig_path).expect("Failed to save original");

    // Create side-by-side montage (scale down for reasonable display)
    let montage_path = format!("{}/{}_compare.png", out_dir.display(), label);
    let status = std::process::Command::new("montage")
        .args([
            &orig_path,
            &decoded_path,
            "-tile",
            "2x1",
            "-geometry",
            "800x800+4+4",
            "-label",
            "",
            &montage_path,
        ])
        .status();
    match status {
        Ok(s) if s.success() => eprintln!("  Montage: {}", montage_path),
        _ => eprintln!("  montage failed"),
    }

    // Annotate the montage
    let annotated_path = format!("{}/{}_annotated.png", out_dir.display(), label);
    let status = std::process::Command::new("convert")
        .args([
            &montage_path,
            "-gravity",
            "North",
            "-pointsize",
            "24",
            "-fill",
            "white",
            "-stroke",
            "black",
            "-strokewidth",
            "1",
            "-annotate",
            "+0+10",
            &format!(
                "{} ({}x{}) — Original vs Decoded ({} bytes, {:.2} bpp)",
                label,
                width,
                height,
                bytes.len(),
                bytes.len() as f64 * 8.0 / (width as usize * height as usize) as f64
            ),
            &annotated_path,
        ])
        .status();
    match status {
        Ok(s) if s.success() => eprintln!("  Annotated: {}", annotated_path),
        _ => {
            // Fall back to unannotated
            std::fs::copy(&montage_path, &annotated_path).ok();
        }
    }

    eprintln!("  Done: {}", label);
}

fn main() {
    let out_dir = jxl_encoder::test_helpers::output_dir("gallery");

    let corpus = jxl_encoder::test_helpers::corpus_dir().join("clic2025/final-test");

    // Pick 4 diverse images
    let images: Vec<(String, &str)> = vec![
        (
            corpus
                .join("07b9f93f170a0381836bdf301280a5b80b2c4be6e66f793a3c335dc200fb4e5b.png")
                .to_string_lossy()
                .to_string(),
            "landscape",
        ),
        (
            corpus
                .join("02809272b4ca9b08af45771501b741296187c7e26907efb44abbbfcb6cd804f7.png")
                .to_string_lossy()
                .to_string(),
            "portrait1",
        ),
        (
            corpus
                .join("0369d229ba4c9965d5caeb38c359a027a810968eee930b81520b604e76b4df14.png")
                .to_string_lossy()
                .to_string(),
            "portrait2",
        ),
        (
            corpus
                .join("1b4ad095795ac552b38a21d51be7bfaee8e7d0a70619d84767814321df4ed062.png")
                .to_string_lossy()
                .to_string(),
            "wide",
        ),
    ];

    for (path, label) in &images {
        encode_and_compare(path, label, &out_dir);
    }

    // Create final 2x2 grid of all comparisons
    let grid_path = format!("{}/gallery_grid.png", out_dir.display());
    let annotated: Vec<String> = images
        .iter()
        .map(|(_, label)| format!("{}/{}_annotated.png", out_dir.display(), label))
        .collect();

    let mut cmd = std::process::Command::new("montage");
    for p in &annotated {
        cmd.arg(p);
    }
    cmd.args(["-tile", "1x4", "-geometry", "1600x+0+8", &grid_path]);
    match cmd.status() {
        Ok(s) if s.success() => eprintln!("\nFinal grid: {}", grid_path),
        _ => eprintln!("\nGrid montage failed"),
    }

    eprintln!("\nAll outputs in: {}", out_dir.display());
}
