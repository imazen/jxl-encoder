//! Test tiny encoder against CLIC 2025 validation images with SSIM2 quality measurement.

use image::GenericImageView;
use std::io::Cursor;

/// Test encoding and decoding a single CLIC 2025 image, returning SSIM2 score.
fn test_clic_image_with_ssim2(path: &str) -> Option<f64> {
    let img = match image::open(path) {
        Ok(img) => img,
        Err(e) => {
            eprintln!("Could not open {}: {}", path, e);
            return None;
        }
    };

    let (width, height) = img.dimensions();
    let filename = path.rsplit('/').next().unwrap_or(path);

    // Get original sRGB pixels for SSIM2 comparison
    let rgb = img.to_rgb8();
    let original_srgb: Vec<[u8; 3]> = rgb.pixels()
        .map(|p| [p[0], p[1], p[2]])
        .collect();

    // Convert to linear RGB f32 for encoding
    let linear_rgb: Vec<f32> = rgb.pixels()
        .flat_map(|p| {
            // sRGB to linear conversion
            let r = (p[0] as f32 / 255.0).powf(2.2);
            let g = (p[1] as f32 / 255.0).powf(2.2);
            let b = (p[2] as f32 / 255.0).powf(2.2);
            [r, g, b]
        })
        .collect();

    // Encode
    let encoder = jxl_enc::tiny::TinyEncoder::new(1.0);
    let bytes = match encoder.encode(width as usize, height as usize, &linear_rgb) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("{}: ENCODE ERROR: {:?}", filename, e);
            return None;
        }
    };

    let compression = (width * height * 3) as f64 / bytes.len() as f64;

    // Decode with jxl-oxide
    let reader = Cursor::new(&bytes);
    let image = match jxl_oxide::JxlImage::builder().read(reader) {
        Ok(img) => img,
        Err(e) => {
            eprintln!("{}: PARSE ERROR: {:?}", filename, e);
            return None;
        }
    };

    let render = match image.render_frame(0) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("{}: DECODE ERROR: {:?}", filename, e);
            return None;
        }
    };

    // Extract decoded pixels (linear f32)
    let fb = render.image_all_channels();
    let decoded_linear = fb.buf();

    // Convert decoded linear to sRGB u8 for SSIM2
    let decoded_srgb: Vec<[u8; 3]> = decoded_linear
        .chunks(3)
        .map(|rgb| {
            // Linear to sRGB
            let r = (rgb[0].clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0).round() as u8;
            let g = (rgb[1].clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0).round() as u8;
            let b = (rgb[2].clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0).round() as u8;
            [r, g, b]
        })
        .collect();

    // Compute SSIM2 using imgref
    let w = width as usize;
    let h = height as usize;
    let original_img = imgref::Img::new(original_srgb, w, h);
    let decoded_img = imgref::Img::new(decoded_srgb, w, h);

    let ssim2 = match fast_ssim2::compute_ssimulacra2(original_img.as_ref(), decoded_img.as_ref()) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("{}: SSIM2 ERROR: {:?}", filename, e);
            return None;
        }
    };

    eprintln!(
        "{}: {}x{}, {} bytes ({:.1}x), SSIM2 = {:.1}",
        filename, width, height, bytes.len(), compression, ssim2
    );

    Some(ssim2)
}

#[test]
#[ignore] // Run with: cargo test --test clic2025 test_clic2025_first_5 -- --ignored --nocapture
fn test_clic2025_first_5() {
    eprintln!("\n=== CLIC 2025 Multi-Group Quality Test ===\n");

    let base_dir = std::env::var("HOME").unwrap_or_else(|_| String::from("/home/lilith"));
    let validation_dir = format!("{}/work/codec-corpus/clic2025/validation", base_dir);

    let entries: Vec<_> = std::fs::read_dir(&validation_dir)
        .expect("Could not read clic2025 validation directory")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "png"))
        .take(5)
        .collect();

    let mut scores: Vec<f64> = Vec::new();
    for entry in entries {
        if let Some(score) = test_clic_image_with_ssim2(&entry.path().to_string_lossy()) {
            scores.push(score);
        }
    }

    if !scores.is_empty() {
        let avg_ssim2 = scores.iter().sum::<f64>() / scores.len() as f64;
        let min_ssim2 = scores.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_ssim2 = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        eprintln!("\n--- Summary ---");
        eprintln!("Images tested: {}", scores.len());
        eprintln!("SSIM2: avg={:.1}, min={:.1}, max={:.1}", avg_ssim2, min_ssim2, max_ssim2);
        eprintln!("(90+ = imperceptible, 70-90 = subtle, 50-70 = noticeable)\n");

        // Assert quality threshold
        assert!(min_ssim2 > 50.0, "Quality too low! Min SSIM2 = {:.1}", min_ssim2);
    }
}

#[test]
#[ignore] // Run with: cargo test --test clic2025 test_clic2025_small_crop -- --ignored --nocapture
fn test_clic2025_small_crop() {
    eprintln!("\n=== CLIC 2025 Single-Group Quality Test (200x200 crop) ===\n");

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

    // Get original sRGB pixels
    let rgb = cropped.to_rgb8();
    let original_srgb: Vec<[u8; 3]> = rgb.pixels()
        .map(|p| [p[0], p[1], p[2]])
        .collect();

    // Convert to linear RGB
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
    eprintln!("Encoded to {} bytes ({:.1}x compression)",
        bytes.len(),
        (cw * ch * 3) as f64 / bytes.len() as f64);

    // Decode
    let reader = Cursor::new(&bytes);
    let image = jxl_oxide::JxlImage::builder().read(reader).expect("Parse failed");
    let render = image.render_frame(0).expect("Render failed");

    // Extract decoded pixels
    let fb = render.image_all_channels();
    let decoded_linear = fb.buf();

    // Convert to sRGB u8
    let decoded_srgb: Vec<[u8; 3]> = decoded_linear
        .chunks(3)
        .map(|rgb| {
            let r = (rgb[0].clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0).round() as u8;
            let g = (rgb[1].clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0).round() as u8;
            let b = (rgb[2].clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0).round() as u8;
            [r, g, b]
        })
        .collect();

    // Compute SSIM2
    let w = cw as usize;
    let h = ch as usize;
    let original_img = imgref::Img::new(original_srgb, w, h);
    let decoded_img = imgref::Img::new(decoded_srgb, w, h);

    let ssim2 = fast_ssim2::compute_ssimulacra2(original_img.as_ref(), decoded_img.as_ref())
        .expect("SSIM2 computation failed");

    eprintln!("\nSSIM2 = {:.1}", ssim2);
    eprintln!("(90+ = imperceptible, 70-90 = subtle, 50-70 = noticeable)\n");

    assert_eq!(image.width(), cw);
    assert_eq!(image.height(), ch);
    assert!(ssim2 > 50.0, "Quality too low! SSIM2 = {:.1}", ssim2);
}
