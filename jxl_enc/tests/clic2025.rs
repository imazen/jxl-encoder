//! Test tiny encoder against CLIC 2025 validation images.

use image::GenericImageView;
use std::io::Cursor;

/// Test encoding and decoding a single CLIC 2025 image.
fn test_clic_image(path: &str) {
    let img = match image::open(path) {
        Ok(img) => img,
        Err(e) => {
            eprintln!("Could not open {}: {}", path, e);
            return;
        }
    };

    let (width, height) = img.dimensions();
    eprintln!("Testing {} ({}x{})", path.rsplit('/').next().unwrap_or(path), width, height);

    // Convert to linear RGB f32
    let rgb = img.to_rgb8();
    let linear_rgb: Vec<f32> = rgb.pixels()
        .flat_map(|p| {
            // Simple sRGB to linear conversion
            let r = (p[0] as f32 / 255.0).powf(2.2);
            let g = (p[1] as f32 / 255.0).powf(2.2);
            let b = (p[2] as f32 / 255.0).powf(2.2);
            [r, g, b]
        })
        .collect();

    // Encode
    let encoder = jxl_enc::tiny::TinyEncoder::new(1.0);
    match encoder.encode(width as usize, height as usize, &linear_rgb) {
        Ok(bytes) => {
            eprintln!("  Encoded to {} bytes ({:.2}x compression)",
                bytes.len(),
                (width * height * 3) as f32 / bytes.len() as f32);

            // Try to decode with jxl-oxide
            let reader = Cursor::new(&bytes);
            match jxl_oxide::JxlImage::builder().read(reader) {
                Ok(image) => {
                    eprintln!("  Parsed OK: {}x{}", image.width(), image.height());
                    match image.render_frame(0) {
                        Ok(_render) => {
                            eprintln!("  Decoded OK");
                            // TODO: compute SSIM2 here
                        }
                        Err(e) => {
                            eprintln!("  DECODE ERROR: {:?}", e);
                        }
                    }
                }
                Err(e) => {
                    eprintln!("  PARSE ERROR: {:?}", e);
                }
            }
        }
        Err(e) => {
            eprintln!("  ENCODE ERROR: {:?}", e);
        }
    }
}

#[test]
#[ignore] // Run with: cargo test --test clic2025 -- --ignored --nocapture
fn test_clic2025_first_5() {
    let base_dir = std::env::var("HOME").unwrap_or_else(|_| String::from("/home/lilith"));
    let validation_dir = format!("{}/work/codec-corpus/clic2025/validation", base_dir);
    
    let entries: Vec<_> = std::fs::read_dir(&validation_dir)
        .expect("Could not read clic2025 validation directory")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "png"))
        .take(5)
        .collect();
    
    for entry in entries {
        test_clic_image(&entry.path().to_string_lossy());
    }
}

#[test]
#[ignore] // Run with: cargo test --test clic2025 -- --ignored --nocapture
fn test_clic2025_small_crop() {
    // Test with a small 200x200 crop to verify single-group encoding works
    let base_dir = std::env::var("HOME").unwrap_or_else(|_| String::from("/home/lilith"));
    let validation_dir = format!("{}/work/codec-corpus/clic2025/validation", base_dir);
    
    let first_png = std::fs::read_dir(&validation_dir)
        .expect("Could not read clic2025 validation directory")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "png"))
        .next()
        .expect("No PNG files found");
    
    let img = image::open(first_png.path()).expect("Could not open image");
    let (width, height) = img.dimensions();
    eprintln!("Original image: {}x{}", width, height);
    
    // Crop to 200x200 (single-group)
    let crop_size = 200u32;
    let cropped = img.crop_imm(0, 0, crop_size.min(width), crop_size.min(height));
    let (cw, ch) = cropped.dimensions();
    eprintln!("Cropped to: {}x{}", cw, ch);
    
    // Convert to linear RGB
    let rgb = cropped.to_rgb8();
    let linear_rgb: Vec<f32> = rgb.pixels()
        .flat_map(|p| {
            let r = (p[0] as f32 / 255.0).powf(2.2);
            let g = (p[1] as f32 / 255.0).powf(2.2);
            let b = (p[2] as f32 / 255.0).powf(2.2);
            [r, g, b]
        })
        .collect();
    
    // Encode
    let encoder = jxl_enc::tiny::TinyEncoder::new(1.0);
    let bytes = encoder.encode(cw as usize, ch as usize, &linear_rgb)
        .expect("Encoding failed");
    eprintln!("Encoded to {} bytes", bytes.len());
    
    // Decode
    let reader = Cursor::new(&bytes);
    let image = jxl_oxide::JxlImage::builder().read(reader).expect("Parse failed");
    eprintln!("Parsed: {}x{}", image.width(), image.height());

    let _render = image.render_frame(0).expect("Render failed");
    eprintln!("Decoded successfully");

    assert_eq!(image.width(), cw);
    assert_eq!(image.height(), ch);
}
