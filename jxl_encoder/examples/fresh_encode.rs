//! Encode a fresh image and save original + decoded for visual comparison.
use std::io::Cursor;

use jxl_encoder::{LossyConfig, PixelLayout};

fn main() {
    let out_dir = "/mnt/v/output/jxl-encoder-rs/clic2025";
    std::fs::create_dir_all(out_dir).ok();

    // Load and crop a CLIC image to 256x256 (single group)
    let base_dir = std::env::var("HOME").unwrap_or_else(|_| String::from("/home/lilith"));
    let img_path = format!(
        "{}/work/codec-corpus/clic2025/validation/a36713f1943dac6bc74dea50cadaee6f.png",
        base_dir
    );

    eprintln!("Loading: {}", img_path);
    let img = image::open(&img_path).expect("Could not open image");

    // Crop to 256x256 for single-group test - pick an interesting region
    let cropped = img.crop_imm(400, 600, 256, 256);
    let rgb = cropped.to_rgb8();
    let (width, height) = (256u32, 256u32);

    // Save original crop
    let orig_path = format!("{}/FRESH_original.png", out_dir);
    cropped.save(&orig_path).expect("Failed to save original");
    eprintln!("Saved original: {}", orig_path);

    // Encode using public API (handles sRGB→linear internally)
    eprintln!("Encoding {}x{}...", width, height);
    let bytes = LossyConfig::new(1.0)
        .encode(rgb.as_raw(), width, height, PixelLayout::Rgb8)
        .expect("Encoding failed");
    eprintln!(
        "Encoded to {} bytes ({:.1}x compression)",
        bytes.len(),
        (width * height * 3) as f64 / bytes.len() as f64
    );

    // Save JXL
    let jxl_path = format!("{}/FRESH.jxl", out_dir);
    std::fs::write(&jxl_path, &bytes).expect("Failed to write JXL");
    eprintln!("Saved JXL: {}", jxl_path);

    // Decode with jxl-oxide
    let reader = Cursor::new(&bytes);
    let image = jxl_oxide::JxlImage::builder()
        .read(reader)
        .expect("Parse failed");
    let render = image.render_frame(0).expect("Render failed");
    let fb = render.image_all_channels();
    let decoded = fb.buf();

    // Convert to sRGB u8 and save as PNG.
    // jxl_oxide already returns sRGB nonlinear floats (no request_color_encoding call).
    let mut output_img = image::RgbImage::new(width, height);
    for (i, pixel) in output_img.pixels_mut().enumerate() {
        let r = (decoded[i * 3].clamp(0.0, 1.0) * 255.0).round() as u8;
        let g = (decoded[i * 3 + 1].clamp(0.0, 1.0) * 255.0).round() as u8;
        let b = (decoded[i * 3 + 2].clamp(0.0, 1.0) * 255.0).round() as u8;
        *pixel = image::Rgb([r, g, b]);
    }

    let oxide_path = format!("{}/FRESH_decoded_oxide.png", out_dir);
    output_img
        .save(&oxide_path)
        .expect("Failed to save decoded PNG");
    eprintln!("Saved jxl-oxide decoded: {}", oxide_path);

    // Also decode with djxl for comparison
    let djxl_path = format!("{}/FRESH_decoded_djxl.png", out_dir);
    let status = std::process::Command::new("djxl")
        .arg(&jxl_path)
        .arg(&djxl_path)
        .status();

    match status {
        Ok(s) if s.success() => eprintln!("Saved djxl decoded: {}", djxl_path),
        _ => eprintln!("djxl decode failed or not available"),
    }

    // Create side-by-side comparison
    eprintln!("\nTo compare:");
    eprintln!(
        "  montage {} {} -tile 2x1 -geometry +4+4 {}/FRESH_compare.png",
        orig_path, oxide_path, out_dir
    );
    eprintln!("  display {}/FRESH_compare.png", out_dir);
}
