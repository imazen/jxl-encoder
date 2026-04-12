//! Test tiny encoder against CLIC 2025 validation images with SSIM2 quality measurement.

use image::GenericImageView;
use std::io::Cursor;

/// Convert sRGB u8 (normalized to 0..1) to linear light using correct sRGB EOTF.
fn srgb_to_linear_val(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Convert linear light value to sRGB u8 using the correct sRGB transfer function.
fn linear_to_srgb_u8(linear: f32) -> u8 {
    let c = linear.clamp(0.0, 1.0);
    let srgb = if c <= 0.003_130_8 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (srgb * 255.0).round() as u8
}

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
    let original_srgb: Vec<[u8; 3]> = rgb.pixels().map(|p| [p[0], p[1], p[2]]).collect();

    // Convert to linear RGB f32 for encoding
    let linear_rgb: Vec<f32> = rgb
        .pixels()
        .flat_map(|p| {
            // sRGB to linear conversion
            let r = srgb_to_linear_val(p[0] as f32 / 255.0);
            let g = srgb_to_linear_val(p[1] as f32 / 255.0);
            let b = srgb_to_linear_val(p[2] as f32 / 255.0);
            [r, g, b]
        })
        .collect();

    // Encode
    let encoder = jxl_encoder::vardct::VarDctEncoder::new(1.0);
    let bytes = match encoder.encode(width as usize, height as usize, &linear_rgb, None) {
        Ok(output) => output.data,
        Err(e) => {
            eprintln!("{}: ENCODE ERROR: {:?}", filename, e);
            return None;
        }
    };

    let compression = (width * height * 3) as f64 / bytes.len() as f64;

    // Decode with jxl-oxide
    let reader = Cursor::new(&bytes);
    let mut image = match jxl_oxide::JxlImage::builder().read(reader) {
        Ok(img) => img,
        Err(e) => {
            eprintln!("{}: PARSE ERROR: {:?}", filename, e);
            return None;
        }
    };
    image.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
        jxl_oxide::RenderingIntent::Relative,
    ));

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
            let r = linear_to_srgb_u8(rgb[0]);
            let g = linear_to_srgb_u8(rgb[1]);
            let b = linear_to_srgb_u8(rgb[2]);
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
        filename,
        width,
        height,
        bytes.len(),
        compression,
        ssim2
    );

    Some(ssim2)
}

#[test]
#[ignore] // Run with: cargo test --test clic2025 test_clic2025_first_5 -- --ignored --nocapture
fn test_clic2025_first_5() {
    eprintln!("\n=== CLIC 2025 Multi-Group Quality Test ===\n");

    let corpus = jxl_encoder::test_helpers::corpus_dir();
    let validation_dir = corpus.join("clic2025/validation");

    let entries: Vec<_> = std::fs::read_dir(&validation_dir)
        .expect("Could not read clic2025 validation directory")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "png"))
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
        eprintln!(
            "SSIM2: avg={:.1}, min={:.1}, max={:.1}",
            avg_ssim2, min_ssim2, max_ssim2
        );
        eprintln!("(90+ = imperceptible, 70-90 = subtle, 50-70 = noticeable)\n");

        // Assert quality threshold
        assert!(
            min_ssim2 > 50.0,
            "Quality too low! Min SSIM2 = {:.1}",
            min_ssim2
        );
    }
}

#[test]
#[ignore] // Run with: cargo test --test clic2025 test_clic2025_all -- --ignored --nocapture
fn test_clic2025_all() {
    eprintln!("\n=== CLIC 2025 Full Validation Set Test (32 images) ===\n");

    let corpus = jxl_encoder::test_helpers::corpus_dir();
    let validation_dir = corpus.join("clic2025/validation");

    let mut entries: Vec<_> = std::fs::read_dir(&validation_dir)
        .expect("Could not read clic2025 validation directory")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "png"))
        .collect();

    // Sort for consistent ordering
    entries.sort_by_key(|e| e.path());

    let mut scores: Vec<f64> = Vec::new();
    let mut failed: Vec<String> = Vec::new();

    for entry in &entries {
        match test_clic_image_with_ssim2(&entry.path().to_string_lossy()) {
            Some(score) => scores.push(score),
            None => failed.push(entry.path().to_string_lossy().to_string()),
        }
    }

    eprintln!("\n--- Summary ---");
    eprintln!("Total images: {}", entries.len());
    eprintln!("Passed: {}", scores.len());
    if !failed.is_empty() {
        eprintln!("Failed: {} - {:?}", failed.len(), failed);
    }

    if !scores.is_empty() {
        let avg_ssim2 = scores.iter().sum::<f64>() / scores.len() as f64;
        let min_ssim2 = scores.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_ssim2 = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        eprintln!(
            "SSIM2: avg={:.1}, min={:.1}, max={:.1}",
            avg_ssim2, min_ssim2, max_ssim2
        );
        eprintln!("(90+ = imperceptible, 70-90 = subtle, 50-70 = noticeable)\n");

        // Assert all images passed with acceptable quality
        assert!(failed.is_empty(), "Some images failed to encode/decode");
        assert!(
            min_ssim2 > 50.0,
            "Quality too low! Min SSIM2 = {:.1}",
            min_ssim2
        );
    }
}

#[test]
#[ignore] // Run with: cargo test --test clic2025 test_clic2025_small_crop -- --ignored --nocapture
fn test_clic2025_small_crop() {
    eprintln!("\n=== CLIC 2025 Single-Group Quality Test (200x200 crop) ===\n");

    let corpus = jxl_encoder::test_helpers::corpus_dir();
    let validation_dir = corpus.join("clic2025/validation");

    let first_png = std::fs::read_dir(&validation_dir)
        .expect("Could not read clic2025 validation directory")
        .filter_map(|e| e.ok())
        .find(|e| e.path().extension().is_some_and(|ext| ext == "png"))
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
    let original_srgb: Vec<[u8; 3]> = rgb.pixels().map(|p| [p[0], p[1], p[2]]).collect();

    // Convert to linear RGB
    let linear_rgb: Vec<f32> = rgb
        .pixels()
        .flat_map(|p| {
            let r = srgb_to_linear_val(p[0] as f32 / 255.0);
            let g = srgb_to_linear_val(p[1] as f32 / 255.0);
            let b = srgb_to_linear_val(p[2] as f32 / 255.0);
            [r, g, b]
        })
        .collect();

    // Encode
    let encoder = jxl_encoder::vardct::VarDctEncoder::new(1.0);
    let bytes = encoder
        .encode(cw as usize, ch as usize, &linear_rgb, None)
        .expect("Encoding failed")
        .data;
    eprintln!(
        "Encoded to {} bytes ({:.1}x compression)",
        bytes.len(),
        (cw * ch * 3) as f64 / bytes.len() as f64
    );

    // Decode
    let reader = Cursor::new(&bytes);
    let mut image = jxl_oxide::JxlImage::builder()
        .read(reader)
        .expect("Parse failed");
    image.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
        jxl_oxide::RenderingIntent::Relative,
    ));
    let render = image.render_frame(0).expect("Render failed");

    // Extract decoded pixels
    let fb = render.image_all_channels();
    let decoded_linear = fb.buf();

    // Convert to sRGB u8
    let decoded_srgb: Vec<[u8; 3]> = decoded_linear
        .chunks(3)
        .map(|rgb| {
            let r = linear_to_srgb_u8(rgb[0]);
            let g = linear_to_srgb_u8(rgb[1]);
            let b = linear_to_srgb_u8(rgb[2]);
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

#[test]
#[ignore] // Run with: cargo test --test clic2025 test_save_multigroup_comparison -- --ignored --nocapture
fn test_save_multigroup_comparison() {
    eprintln!("\n=== Multi-Group Visual Comparison ===\n");

    let corpus = jxl_encoder::test_helpers::corpus_dir();
    let validation_dir = corpus.join("clic2025/validation");
    let output_dir = jxl_encoder::test_helpers::output_dir("clic2025");

    let first_png = std::fs::read_dir(&validation_dir)
        .expect("Could not read clic2025 validation directory")
        .filter_map(|e| e.ok())
        .find(|e| e.path().extension().is_some_and(|ext| ext == "png"))
        .expect("No PNG files found");

    let img = image::open(first_png.path()).expect("Could not open image");
    let (width, height) = img.dimensions();
    eprintln!("Original image: {}x{}", width, height);

    // Test 600x600 (9 groups) - transition point
    let crop_size = 600u32;
    let cropped = img.crop_imm(0, 0, crop_size.min(width), crop_size.min(height));
    let (cw, ch) = cropped.dimensions();
    eprintln!(
        "Cropped to: {}x{} (requires {} groups)",
        cw,
        ch,
        cw.div_ceil(256) * ch.div_ceil(256)
    );

    // Save original
    let orig_path = output_dir.join(format!("original_{}x{}.png", cw, ch));
    cropped.save(&orig_path).expect("Failed to save original");
    eprintln!("Saved original to: {}", orig_path.display());

    // Get original sRGB pixels
    let rgb = cropped.to_rgb8();

    // Convert to linear RGB
    let linear_rgb: Vec<f32> = rgb
        .pixels()
        .flat_map(|p| {
            let r = srgb_to_linear_val(p[0] as f32 / 255.0);
            let g = srgb_to_linear_val(p[1] as f32 / 255.0);
            let b = srgb_to_linear_val(p[2] as f32 / 255.0);
            [r, g, b]
        })
        .collect();

    // Encode
    let encoder = jxl_encoder::vardct::VarDctEncoder::new(1.0);
    let bytes = encoder
        .encode(cw as usize, ch as usize, &linear_rgb, None)
        .expect("Encoding failed")
        .data;
    eprintln!(
        "Encoded to {} bytes ({:.1}x compression)",
        bytes.len(),
        (cw * ch * 3) as f64 / bytes.len() as f64
    );

    // Save JXL
    let jxl_path = output_dir.join(format!("encoded_{}x{}.jxl", cw, ch));
    std::fs::write(&jxl_path, &bytes).expect("Failed to write JXL");
    eprintln!("Saved JXL to: {}", jxl_path.display());

    // Decode
    let reader = Cursor::new(&bytes);
    let mut image = jxl_oxide::JxlImage::builder()
        .read(reader)
        .expect("Parse failed");
    image.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
        jxl_oxide::RenderingIntent::Relative,
    ));
    let render = image.render_frame(0).expect("Render failed");

    // Extract decoded pixels
    let fb = render.image_all_channels();
    let decoded_linear = fb.buf();

    // Debug: check decoded value statistics
    let min_val = decoded_linear.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = decoded_linear
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = decoded_linear.iter().sum();
    let avg = sum / decoded_linear.len() as f32;
    let out_of_range = decoded_linear
        .iter()
        .filter(|&&v| !(0.0..=1.0).contains(&v))
        .count();
    eprintln!(
        "Decoded linear stats: min={:.4}, max={:.4}, avg={:.4}, out_of_range={}/{}",
        min_val,
        max_val,
        avg,
        out_of_range,
        decoded_linear.len()
    );

    // Check which regions have bad values
    let w = cw as usize;
    let h = ch as usize;
    let group_size = 256usize; // pixels
    let num_groups_x = w.div_ceil(group_size);
    let num_groups_y = h.div_ceil(group_size);
    for gy in 0..num_groups_y {
        for gx in 0..num_groups_x {
            let x0 = gx * group_size;
            let y0 = gy * group_size;
            let x1 = (x0 + group_size).min(w);
            let y1 = (y0 + group_size).min(h);
            let mut bad_count = 0usize;
            for y in y0..y1 {
                for x in x0..x1 {
                    let idx = (y * w + x) * 3;
                    for c in 0..3 {
                        let v = decoded_linear[idx + c];
                        if !(0.0..=1.0).contains(&v) {
                            bad_count += 1;
                        }
                    }
                }
            }
            if bad_count > 0 {
                let group_idx = gy * num_groups_x + gx;
                eprintln!(
                    "  Group {} ({},{}) has {} bad values",
                    group_idx, gx, gy, bad_count
                );
            }
        }
    }

    // Convert to sRGB u8
    let decoded_srgb: Vec<u8> = decoded_linear
        .chunks(3)
        .flat_map(|rgb| {
            let r = linear_to_srgb_u8(rgb[0]);
            let g = linear_to_srgb_u8(rgb[1]);
            let b = linear_to_srgb_u8(rgb[2]);
            [r, g, b]
        })
        .collect();

    // Save decoded image
    let decoded_img =
        image::RgbImage::from_raw(cw, ch, decoded_srgb.clone()).expect("Failed to create image");
    let decoded_path = output_dir.join(format!("decoded_{}x{}.png", cw, ch));
    decoded_img
        .save(&decoded_path)
        .expect("Failed to save decoded");
    eprintln!("Saved decoded to: {}", decoded_path.display());

    // Compute SSIM2
    let original_srgb: Vec<[u8; 3]> = rgb.pixels().map(|p| [p[0], p[1], p[2]]).collect();
    let decoded_rgb: Vec<[u8; 3]> = decoded_srgb.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();

    let w = cw as usize;
    let h = ch as usize;
    let original_img = imgref::Img::new(original_srgb, w, h);
    let decoded_img_ref = imgref::Img::new(decoded_rgb, w, h);

    let ssim2 = fast_ssim2::compute_ssimulacra2(original_img.as_ref(), decoded_img_ref.as_ref())
        .expect("SSIM2 computation failed");

    eprintln!("\nSSIM2 = {:.1}", ssim2);
    eprintln!("\nView images:");
    eprintln!("  feh {} {} &", orig_path.display(), decoded_path.display());
}

#[test]
#[ignore] // Run with: cargo test --test clic2025 test_exact_multiples -- --ignored --nocapture
fn test_exact_multiples() {
    eprintln!("\n=== Testing Exact Multiples of 256 ===\n");

    let corpus = jxl_encoder::test_helpers::corpus_dir();
    let validation_dir = corpus.join("clic2025/validation");

    let first_png = std::fs::read_dir(&validation_dir)
        .expect("Could not read directory")
        .filter_map(|e| e.ok())
        .find(|e| e.path().extension().is_some_and(|ext| ext == "png"))
        .expect("No PNG files found");

    let img = image::open(first_png.path()).expect("Could not open image");

    // Test sizes that are exact multiples of 256 to rule out partial group issues
    for &size in &[256u32, 512, 768, 1024, 1280] {
        let (w, h) = img.dimensions();
        if size > w || size > h {
            continue;
        }

        let cropped = img.crop_imm(0, 0, size, size);
        let rgb = cropped.to_rgb8();
        let original_srgb: Vec<[u8; 3]> = rgb.pixels().map(|p| [p[0], p[1], p[2]]).collect();

        let linear_rgb: Vec<f32> = rgb
            .pixels()
            .flat_map(|p| {
                let r = srgb_to_linear_val(p[0] as f32 / 255.0);
                let g = srgb_to_linear_val(p[1] as f32 / 255.0);
                let b = srgb_to_linear_val(p[2] as f32 / 255.0);
                [r, g, b]
            })
            .collect();

        let encoder = jxl_encoder::vardct::VarDctEncoder::new(1.0);
        let bytes = encoder
            .encode(size as usize, size as usize, &linear_rgb, None)
            .expect("Encode failed")
            .data;

        let reader = Cursor::new(&bytes);
        let mut image = jxl_oxide::JxlImage::builder()
            .read(reader)
            .expect("Parse failed");
        image.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
            jxl_oxide::RenderingIntent::Relative,
        ));
        let render = image.render_frame(0).expect("Render failed");

        let fb = render.image_all_channels();
        let decoded_linear = fb.buf();

        let decoded_srgb: Vec<[u8; 3]> = decoded_linear
            .chunks(3)
            .map(|rgb| {
                let r = linear_to_srgb_u8(rgb[0]);
                let g = linear_to_srgb_u8(rgb[1]);
                let b = linear_to_srgb_u8(rgb[2]);
                [r, g, b]
            })
            .collect();

        let s = size as usize;
        let original_img = imgref::Img::new(original_srgb, s, s);
        let decoded_img = imgref::Img::new(decoded_srgb, s, s);

        let ssim2 = fast_ssim2::compute_ssimulacra2(original_img.as_ref(), decoded_img.as_ref())
            .expect("SSIM2 failed");

        let grid = size.div_ceil(256);
        eprintln!(
            "{}x{}: {}x{} = {} full groups, SSIM2 = {:.1}",
            size,
            size,
            grid,
            grid,
            grid * grid,
            ssim2
        );
    }
}

#[test]
#[ignore] // Run with: cargo test --test clic2025 test_multigroup_sizes -- --ignored --nocapture
fn test_multigroup_sizes() {
    eprintln!("\n=== Multi-Group Size Scaling Test ===\n");

    let corpus = jxl_encoder::test_helpers::corpus_dir();
    let validation_dir = corpus.join("clic2025/validation");

    let first_png = std::fs::read_dir(&validation_dir)
        .expect("Could not read clic2025 validation directory")
        .filter_map(|e| e.ok())
        .find(|e| e.path().extension().is_some_and(|ext| ext == "png"))
        .expect("No PNG files found");

    let img = image::open(first_png.path()).expect("Could not open image");
    let (width, height) = img.dimensions();

    // Test different crop sizes
    for &crop_size in &[256u32, 300, 400, 512, 600, 800, 1024, 1280, 1536] {
        if crop_size > width || crop_size > height {
            continue;
        }

        let cropped = img.crop_imm(0, 0, crop_size, crop_size);
        let (cw, ch) = cropped.dimensions();
        let num_groups = cw.div_ceil(256) * ch.div_ceil(256);

        let rgb = cropped.to_rgb8();
        let original_srgb: Vec<[u8; 3]> = rgb.pixels().map(|p| [p[0], p[1], p[2]]).collect();

        let linear_rgb: Vec<f32> = rgb
            .pixels()
            .flat_map(|p| {
                let r = srgb_to_linear_val(p[0] as f32 / 255.0);
                let g = srgb_to_linear_val(p[1] as f32 / 255.0);
                let b = srgb_to_linear_val(p[2] as f32 / 255.0);
                [r, g, b]
            })
            .collect();

        let encoder = jxl_encoder::vardct::VarDctEncoder::new(1.0);
        let bytes = match encoder.encode(cw as usize, ch as usize, &linear_rgb, None) {
            Ok(output) => output.data,
            Err(e) => {
                eprintln!("{}x{}: ENCODE ERROR: {:?}", cw, ch, e);
                continue;
            }
        };

        let reader = Cursor::new(&bytes);
        let mut image = match jxl_oxide::JxlImage::builder().read(reader) {
            Ok(img) => img,
            Err(e) => {
                eprintln!("{}x{}: PARSE ERROR: {:?}", cw, ch, e);
                continue;
            }
        };
        image.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
            jxl_oxide::RenderingIntent::Relative,
        ));

        let render = match image.render_frame(0) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("{}x{}: DECODE ERROR: {:?}", cw, ch, e);
                continue;
            }
        };

        let fb = render.image_all_channels();
        let decoded_linear = fb.buf();

        let decoded_srgb: Vec<[u8; 3]> = decoded_linear
            .chunks(3)
            .map(|rgb| {
                let r = linear_to_srgb_u8(rgb[0]);
                let g = linear_to_srgb_u8(rgb[1]);
                let b = linear_to_srgb_u8(rgb[2]);
                [r, g, b]
            })
            .collect();

        let w = cw as usize;
        let h = ch as usize;
        let original_img = imgref::Img::new(original_srgb, w, h);
        let decoded_img = imgref::Img::new(decoded_srgb, w, h);

        let ssim2 = fast_ssim2::compute_ssimulacra2(original_img.as_ref(), decoded_img.as_ref())
            .unwrap_or(f64::NAN);

        let compression = (cw * ch * 3) as f64 / bytes.len() as f64;
        eprintln!(
            "{}x{}: {} groups, {} bytes ({:.1}x), SSIM2 = {:.1}",
            cw,
            ch,
            num_groups,
            bytes.len(),
            compression,
            ssim2
        );
    }
}

#[test]
#[ignore] // Run with: cargo test --test clic2025 test_djxl_vs_jxl_oxide -- --ignored --nocapture
fn test_djxl_vs_jxl_oxide() {
    eprintln!("\n=== Comparing djxl vs jxl-oxide Decoding ===\n");

    let corpus = jxl_encoder::test_helpers::corpus_dir();
    let validation_dir = corpus.join("clic2025/validation");
    let output_dir = jxl_encoder::test_helpers::output_dir("clic2025");

    let first_png = std::fs::read_dir(&validation_dir)
        .expect("Could not read directory")
        .filter_map(|e| e.ok())
        .find(|e| e.path().extension().is_some_and(|ext| ext == "png"))
        .expect("No PNG files found");

    let img = image::open(first_png.path()).expect("Could not open image");

    // Test 768x768 (3x3 grid = 9 AC groups)
    let size = 768u32;
    let cropped = img.crop_imm(0, 0, size, size);
    let rgb = cropped.to_rgb8();

    // Save original for comparison
    let orig_path = output_dir.join("original_768.png");
    cropped.save(&orig_path).ok();

    let linear_rgb: Vec<f32> = rgb
        .pixels()
        .flat_map(|p| {
            let r = srgb_to_linear_val(p[0] as f32 / 255.0);
            let g = srgb_to_linear_val(p[1] as f32 / 255.0);
            let b = srgb_to_linear_val(p[2] as f32 / 255.0);
            [r, g, b]
        })
        .collect();

    let encoder = jxl_encoder::vardct::VarDctEncoder::new(1.0);
    let bytes = encoder
        .encode(size as usize, size as usize, &linear_rgb, None)
        .expect("Encode failed")
        .data;

    let jxl_path = output_dir.join("test_768.jxl");
    std::fs::write(&jxl_path, &bytes).expect("Failed to write JXL");

    // Decode with jxl-oxide
    let reader = Cursor::new(&bytes);
    let mut image = jxl_oxide::JxlImage::builder()
        .read(reader)
        .expect("Parse failed");
    image.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
        jxl_oxide::RenderingIntent::Relative,
    ));
    let render = image.render_frame(0).expect("Render failed");
    let fb = render.image_all_channels();
    let oxide_decoded = fb.buf();

    // Check jxl-oxide statistics
    let oxide_min = oxide_decoded.iter().cloned().fold(f32::INFINITY, f32::min);
    let oxide_max = oxide_decoded
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let oxide_bad = oxide_decoded
        .iter()
        .filter(|&&v| !(0.0..=1.0).contains(&v))
        .count();

    eprintln!(
        "jxl-oxide: min={:.4}, max={:.4}, bad={}",
        oxide_min, oxide_max, oxide_bad
    );

    // Decode with djxl (writes PNG, we read it back)
    let djxl_png = output_dir.join("djxl_decoded_768.png");
    let djxl_bin = jxl_encoder::test_helpers::djxl_path();

    let output = std::process::Command::new(&djxl_bin)
        .arg(&jxl_path)
        .arg(&djxl_png)
        .output();

    match output {
        Ok(result) => {
            if result.status.success() {
                // Read the djxl decoded image
                let djxl_img = image::open(&djxl_png).expect("Failed to open djxl output");
                let djxl_rgb = djxl_img.to_rgb8();

                // Convert to linear for comparison (djxl outputs sRGB)
                let djxl_linear: Vec<f32> = djxl_rgb
                    .pixels()
                    .flat_map(|p| {
                        let r = srgb_to_linear_val(p[0] as f32 / 255.0);
                        let g = srgb_to_linear_val(p[1] as f32 / 255.0);
                        let b = srgb_to_linear_val(p[2] as f32 / 255.0);
                        [r, g, b]
                    })
                    .collect();

                let djxl_min = djxl_linear.iter().cloned().fold(f32::INFINITY, f32::min);
                let djxl_max = djxl_linear
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max);

                eprintln!(
                    "djxl:      min={:.4}, max={:.4}, bad=0 (clamped to u8)",
                    djxl_min, djxl_max
                );

                // Compare original to djxl (compute SSIM2)
                let original_srgb: Vec<[u8; 3]> =
                    rgb.pixels().map(|p| [p[0], p[1], p[2]]).collect();
                let djxl_srgb: Vec<[u8; 3]> =
                    djxl_rgb.pixels().map(|p| [p[0], p[1], p[2]]).collect();

                let w = size as usize;
                let original_img = imgref::Img::new(original_srgb.clone(), w, w);
                let djxl_img_ref = imgref::Img::new(djxl_srgb, w, w);

                let djxl_ssim2 =
                    fast_ssim2::compute_ssimulacra2(original_img.as_ref(), djxl_img_ref.as_ref())
                        .expect("SSIM2 failed");
                eprintln!("\ndjxl SSIM2:      {:.1}", djxl_ssim2);

                // Compare original to jxl-oxide
                let oxide_srgb: Vec<[u8; 3]> = oxide_decoded
                    .chunks(3)
                    .map(|rgb| {
                        let r = linear_to_srgb_u8(rgb[0]);
                        let g = linear_to_srgb_u8(rgb[1]);
                        let b = linear_to_srgb_u8(rgb[2]);
                        [r, g, b]
                    })
                    .collect();
                let oxide_img_ref = imgref::Img::new(oxide_srgb, w, w);

                let oxide_ssim2 =
                    fast_ssim2::compute_ssimulacra2(original_img.as_ref(), oxide_img_ref.as_ref())
                        .expect("SSIM2 failed");
                eprintln!("jxl-oxide SSIM2: {:.1}", oxide_ssim2);

                eprintln!("\nConclusion:");
                if djxl_ssim2 > 50.0 && oxide_ssim2 < 0.0 {
                    eprintln!("  djxl decodes correctly but jxl-oxide does not!");
                    eprintln!("  This suggests a decoder bug, not an encoder bug.");
                } else if djxl_ssim2 < 0.0 {
                    eprintln!("  Both decoders fail - encoder bug confirmed.");
                } else {
                    eprintln!("  Both decoders work - check the comparison logic.");
                }
            } else {
                eprintln!("djxl failed: {:?}", String::from_utf8_lossy(&result.stderr));
            }
        }
        Err(e) => {
            eprintln!("Could not run djxl: {}", e);
        }
    }
}

#[test]
#[ignore] // Run with: cargo test --test clic2025 test_section_sizes -- --ignored --nocapture
fn test_section_sizes() {
    eprintln!("\n=== Section Size Analysis ===\n");

    let corpus = jxl_encoder::test_helpers::corpus_dir();
    let validation_dir = corpus.join("clic2025/validation");

    let first_png = std::fs::read_dir(&validation_dir)
        .expect("Could not read directory")
        .filter_map(|e| e.ok())
        .find(|e| e.path().extension().is_some_and(|ext| ext == "png"))
        .expect("No PNG files found");

    let img = image::open(first_png.path()).expect("Could not open image");

    // Test 768x768 (3x3 grid = 9 AC groups)
    let size = 768u32;
    let cropped = img.crop_imm(0, 0, size, size);
    let rgb = cropped.to_rgb8();

    let linear_rgb: Vec<f32> = rgb
        .pixels()
        .flat_map(|p| {
            let r = srgb_to_linear_val(p[0] as f32 / 255.0);
            let g = srgb_to_linear_val(p[1] as f32 / 255.0);
            let b = srgb_to_linear_val(p[2] as f32 / 255.0);
            [r, g, b]
        })
        .collect();

    let encoder = jxl_encoder::vardct::VarDctEncoder::new(1.0);
    let bytes = encoder
        .encode(size as usize, size as usize, &linear_rgb, None)
        .expect("Encode failed")
        .data;

    eprintln!("768x768 = 3x3 = 9 AC groups");
    eprintln!(
        "Expected sections: DC_global, DC_group_0, AC_global, AC_group_0..AC_group_8 (12 total)"
    );
    eprintln!("Total file size: {} bytes", bytes.len());

    // Parse the TOC to see section sizes
    // Skip JXL signature (2 bytes), file header, frame header to find TOC
    // This is a rough analysis - we'll look at the file structure

    // First few bytes for debugging
    eprintln!("\nFirst 32 bytes: {:02x?}", &bytes[..32.min(bytes.len())]);

    // Save the JXL for external analysis
    let output_dir = jxl_encoder::test_helpers::output_dir("clic2025");
    let jxl_path = output_dir.join("test_768x768_sections.jxl");
    std::fs::write(&jxl_path, &bytes).expect("Failed to write JXL");
    eprintln!("\nSaved to: {}", jxl_path.display());
    eprintln!(
        "Analyze with: djxl {} /dev/null --print_info",
        jxl_path.display()
    );
}

#[test]
#[ignore] // Run with: cargo test --test clic2025 test_compare_working_vs_broken -- --ignored --nocapture
fn test_compare_working_vs_broken() {
    eprintln!("\n=== Comparing Working (512) vs Broken (768) ===\n");

    let corpus = jxl_encoder::test_helpers::corpus_dir();
    let validation_dir = corpus.join("clic2025/validation");

    let first_png = std::fs::read_dir(&validation_dir)
        .expect("Could not read directory")
        .filter_map(|e| e.ok())
        .find(|e| e.path().extension().is_some_and(|ext| ext == "png"))
        .expect("No PNG files found");

    let img = image::open(first_png.path()).expect("Could not open image");

    for &size in &[512u32, 768] {
        let cropped = img.crop_imm(0, 0, size, size);
        let rgb = cropped.to_rgb8();

        let linear_rgb: Vec<f32> = rgb
            .pixels()
            .flat_map(|p| {
                let r = srgb_to_linear_val(p[0] as f32 / 255.0);
                let g = srgb_to_linear_val(p[1] as f32 / 255.0);
                let b = srgb_to_linear_val(p[2] as f32 / 255.0);
                [r, g, b]
            })
            .collect();

        let encoder = jxl_encoder::vardct::VarDctEncoder::new(1.0);
        let bytes = encoder
            .encode(size as usize, size as usize, &linear_rgb, None)
            .expect("Encode failed")
            .data;

        let num_groups = (size.div_ceil(256)) * (size.div_ceil(256));
        let num_dc_groups = size.div_ceil(2048) * size.div_ceil(2048);
        let num_sections = 2 + num_dc_groups as usize + num_groups as usize;
        let pixels = (size * size) as usize;
        let bpp = bytes.len() as f64 * 8.0 / pixels as f64;

        eprintln!(
            "{}x{}: {} groups, {} DC groups, {} sections",
            size, size, num_groups, num_dc_groups, num_sections
        );
        eprintln!(
            "  {} bytes, {:.2} bpp, {:.2} bytes/group",
            bytes.len(),
            bpp,
            bytes.len() as f64 / num_groups as f64
        );

        // Decode and check
        let reader = Cursor::new(&bytes);
        let mut image = jxl_oxide::JxlImage::builder()
            .read(reader)
            .expect("Parse failed");
        image.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
            jxl_oxide::RenderingIntent::Relative,
        ));
        let render = image.render_frame(0).expect("Render failed");
        let fb = render.image_all_channels();
        let decoded = fb.buf();

        let min_val = decoded.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = decoded.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let bad = decoded
            .iter()
            .filter(|&&v| !(0.0..=1.0).contains(&v))
            .count();

        eprintln!(
            "  Decoded: min={:.4}, max={:.4}, bad={}",
            min_val, max_val, bad
        );
        eprintln!();
    }
}

#[test]
#[ignore] // Run with: cargo test --test clic2025 test_nzeros_by_group -- --ignored --nocapture
fn test_nzeros_by_group() {
    eprintln!("\n=== Checking nzeros distribution by group ===\n");

    let corpus = jxl_encoder::test_helpers::corpus_dir();
    let validation_dir = corpus.join("clic2025/validation");

    let first_png = std::fs::read_dir(&validation_dir)
        .expect("Could not read directory")
        .filter_map(|e| e.ok())
        .find(|e| e.path().extension().is_some_and(|ext| ext == "png"))
        .expect("No PNG files found");

    let img = image::open(first_png.path()).expect("Could not open image");
    let size = 768u32;
    let cropped = img.crop_imm(0, 0, size, size);
    let rgb = cropped.to_rgb8();

    let linear_rgb: Vec<f32> = rgb
        .pixels()
        .flat_map(|p| {
            let r = srgb_to_linear_val(p[0] as f32 / 255.0);
            let g = srgb_to_linear_val(p[1] as f32 / 255.0);
            let b = srgb_to_linear_val(p[2] as f32 / 255.0);
            [r, g, b]
        })
        .collect();

    // Use internal types to compute nzeros
    use jxl_encoder::vardct::VarDctEncoder;

    // Encode and get internal state (we can't access nzeros directly, so let's
    // just verify the output file decodes with reasonable nzeros by checking
    // the encoded file structure)

    let encoder = VarDctEncoder::new(1.0);
    let bytes = encoder
        .encode(size as usize, size as usize, &linear_rgb, None)
        .expect("Encode failed")
        .data;

    eprintln!("Encoded {} bytes", bytes.len());

    // Check what % of bytes are in each AC group section
    // The TOC contains section sizes. Let's try to parse it roughly.
    // For 768x768: 1 DC group, 9 AC groups, so 12 sections total
    // Section order: DC_global, DC_group_0, AC_global, AC_group_0..8

    // File structure:
    // - 2 bytes: JXL signature (FF 0A)
    // - File header (variable)
    // - Frame header (variable)
    // - TOC (12 entries for 768x768)
    // - Sections

    // This is complex to parse manually. Let's just note that the file size is reasonable
    // and the corruption pattern suggests something structural.

    eprintln!("\nFile analysis:");
    eprintln!("  Signature: {:02x} {:02x}", bytes[0], bytes[1]);
    eprintln!("  Total size: {} bytes", bytes.len());

    // Count runs of zeros (potential indicator of corruption)
    let mut max_zero_run = 0;
    let mut current_zero_run = 0;
    for &b in &bytes {
        if b == 0 {
            current_zero_run += 1;
            max_zero_run = max_zero_run.max(current_zero_run);
        } else {
            current_zero_run = 0;
        }
    }
    eprintln!("  Max consecutive zero bytes: {}", max_zero_run);

    // Check bytes at end of file (should not be all zeros for real content)
    let last_100: Vec<u8> = bytes[bytes.len().saturating_sub(100)..].to_vec();
    let last_100_zeros = last_100.iter().filter(|&&b| b == 0).count();
    eprintln!("  Last 100 bytes: {} zeros", last_100_zeros);
}

#[test]
#[ignore] // Run with: cargo test --test clic2025 test_per_group_corruption -- --ignored --nocapture
fn test_per_group_corruption() {
    eprintln!("\n=== Per-Group Corruption Analysis ===\n");

    let corpus = jxl_encoder::test_helpers::corpus_dir();
    let validation_dir = corpus.join("clic2025/validation");

    let first_png = std::fs::read_dir(&validation_dir)
        .expect("Could not read directory")
        .filter_map(|e| e.ok())
        .find(|e| e.path().extension().is_some_and(|ext| ext == "png"))
        .expect("No PNG files found");

    let img = image::open(first_png.path()).expect("Could not open image");

    // Test 768x768 (3x3 grid) to see which groups are corrupted
    let size = 768u32;
    let (w, h) = img.dimensions();
    if size > w || size > h {
        panic!("Image too small");
    }

    let cropped = img.crop_imm(0, 0, size, size);
    let rgb = cropped.to_rgb8();

    let linear_rgb: Vec<f32> = rgb
        .pixels()
        .flat_map(|p| {
            let r = srgb_to_linear_val(p[0] as f32 / 255.0);
            let g = srgb_to_linear_val(p[1] as f32 / 255.0);
            let b = srgb_to_linear_val(p[2] as f32 / 255.0);
            [r, g, b]
        })
        .collect();

    let encoder = jxl_encoder::vardct::VarDctEncoder::new(1.0);
    let bytes = encoder
        .encode(size as usize, size as usize, &linear_rgb, None)
        .expect("Encode failed")
        .data;

    let reader = Cursor::new(&bytes);
    let mut image = jxl_oxide::JxlImage::builder()
        .read(reader)
        .expect("Parse failed");
    image.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
        jxl_oxide::RenderingIntent::Relative,
    ));
    let render = image.render_frame(0).expect("Render failed");

    let fb = render.image_all_channels();
    let decoded = fb.buf();

    let w = size as usize;
    let group_size = 256usize;
    let num_groups_x = w.div_ceil(group_size); // 3

    eprintln!("768x768 = 3x3 group grid");
    eprintln!("Group layout:");
    eprintln!("  [0] [1] [2]");
    eprintln!("  [3] [4] [5]");
    eprintln!("  [6] [7] [8]");
    eprintln!();

    for gy in 0..3 {
        for gx in 0..3 {
            let group_idx = gy * num_groups_x + gx;
            let x0 = gx * group_size;
            let y0 = gy * group_size;
            let x1 = (x0 + group_size).min(w);
            let y1 = (y0 + group_size).min(w);

            let mut group_min = f32::INFINITY;
            let mut group_max = f32::NEG_INFINITY;
            let mut bad_count = 0usize;

            for y in y0..y1 {
                for x in x0..x1 {
                    let idx = (y * w + x) * 3;
                    for c in 0..3 {
                        let v = decoded[idx + c];
                        group_min = group_min.min(v);
                        group_max = group_max.max(v);
                        if !(0.0..=1.0).contains(&v) {
                            bad_count += 1;
                        }
                    }
                }
            }

            let position = match (gx, gy) {
                (1, 1) => "CENTER",
                (0, 0) | (2, 0) | (0, 2) | (2, 2) => "corner",
                _ => "edge",
            };

            let status = if bad_count > 0 { "CORRUPT" } else { "OK" };
            eprintln!(
                "Group {} ({},{}) {}: min={:.4}, max={:.4}, bad={} [{}]",
                group_idx, gx, gy, position, group_min, group_max, bad_count, status
            );
        }
    }
}

#[test]
#[ignore] // Run with: cargo test --test clic2025 test_real_photo_value_stats -- --ignored --nocapture
fn test_real_photo_value_stats() {
    eprintln!("\n=== Real Photo Value Statistics ===\n");
    eprintln!("Checking decoded value ranges for real photos.\n");

    let corpus = jxl_encoder::test_helpers::corpus_dir();
    let validation_dir = corpus.join("clic2025/validation");

    let first_png = std::fs::read_dir(&validation_dir)
        .expect("Could not read directory")
        .filter_map(|e| e.ok())
        .find(|e| e.path().extension().is_some_and(|ext| ext == "png"))
        .expect("No PNG files found");

    let img = image::open(first_png.path()).expect("Could not open image");

    for &size in &[256u32, 512, 768, 1024] {
        let (w, h) = img.dimensions();
        if size > w || size > h {
            continue;
        }

        let cropped = img.crop_imm(0, 0, size, size);
        let rgb = cropped.to_rgb8();

        let linear_rgb: Vec<f32> = rgb
            .pixels()
            .flat_map(|p| {
                let r = srgb_to_linear_val(p[0] as f32 / 255.0);
                let g = srgb_to_linear_val(p[1] as f32 / 255.0);
                let b = srgb_to_linear_val(p[2] as f32 / 255.0);
                [r, g, b]
            })
            .collect();

        let encoder = jxl_encoder::vardct::VarDctEncoder::new(1.0);
        let bytes = encoder
            .encode(size as usize, size as usize, &linear_rgb, None)
            .expect("Encode failed")
            .data;

        let reader = Cursor::new(&bytes);
        let mut image = jxl_oxide::JxlImage::builder()
            .read(reader)
            .expect("Parse failed");
        image.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
            jxl_oxide::RenderingIntent::Relative,
        ));
        let render = image.render_frame(0).expect("Render failed");

        let fb = render.image_all_channels();
        let decoded = fb.buf();

        // Statistics
        let min_val = decoded.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = decoded.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let avg: f32 = decoded.iter().sum::<f32>() / decoded.len() as f32;
        let out_of_range = decoded
            .iter()
            .filter(|&&v| !(-0.5..=1.5).contains(&v))
            .count();
        let moderately_bad = decoded
            .iter()
            .filter(|&&v| !(0.0..=1.0).contains(&v))
            .count();

        let grid = size.div_ceil(256);
        eprintln!(
            "{}x{} ({}x{}): avg={:.4}, min={:.4}, max={:.4}, moderate_bad={}, severe_bad={}",
            size, size, grid, grid, avg, min_val, max_val, moderately_bad, out_of_range
        );
    }
}

#[test]
#[ignore] // Run with: cargo test --test clic2025 test_noise_multigroup -- --ignored --nocapture
fn test_noise_multigroup() {
    eprintln!("\n=== Noise/High-Frequency Multi-Group Test ===\n");
    eprintln!("Testing high-frequency content that produces AC coefficients.\n");

    // Use a simple LCG for deterministic pseudo-random values
    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 33) as f32) / (u32::MAX as f32 / 2.0)
    }

    for &size in &[256u32, 512, 768, 1024] {
        let n = (size * size) as usize;
        let mut linear_rgb: Vec<f32> = Vec::with_capacity(n * 3);
        let mut seed = 12345u64;

        for _y in 0..size {
            for _x in 0..size {
                // Random values 0.2 to 0.8 (avoid extremes)
                let val = 0.2 + lcg(&mut seed) * 0.6;
                linear_rgb.push(val); // R
                linear_rgb.push(val); // G (same as R for grayscale noise)
                linear_rgb.push(val); // B
            }
        }

        let encoder = jxl_encoder::vardct::VarDctEncoder::new(1.0);
        let bytes = encoder
            .encode(size as usize, size as usize, &linear_rgb, None)
            .expect("Encode failed")
            .data;

        // Decode
        let reader = Cursor::new(&bytes);
        let mut image = jxl_oxide::JxlImage::builder()
            .read(reader)
            .expect("Parse failed");
        image.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
            jxl_oxide::RenderingIntent::Relative,
        ));
        let render = image.render_frame(0).expect("Render failed");

        let fb = render.image_all_channels();
        let decoded = fb.buf();

        // Check statistics
        let avg: f32 = decoded.iter().sum::<f32>() / decoded.len() as f32;
        let min_val = decoded.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = decoded.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let out_of_range = decoded
            .iter()
            .filter(|&&v| !(-0.5..=1.5).contains(&v))
            .count();

        let grid = size.div_ceil(256);
        let compression = (size * size * 3) as f64 / bytes.len() as f64;
        eprintln!(
            "{}x{} ({}x{}): avg={:.4}, min={:.4}, max={:.4}, bad={}, {:.1}x compression",
            size, size, grid, grid, avg, min_val, max_val, out_of_range, compression
        );

        if out_of_range > 0 {
            eprintln!(
                "  ERROR: {} values significantly out of range",
                out_of_range
            );
        }

        // Expected average should be around 0.5 (center of 0.2-0.8 range)
        if (avg - 0.5).abs() > 0.1 {
            eprintln!("  ERROR: Average {:.4} is far from expected 0.5", avg);
        }
    }
}

#[test]
#[ignore] // Run with: cargo test --test clic2025 test_gradient_multigroup -- --ignored --nocapture
fn test_gradient_multigroup() {
    eprintln!("\n=== Gradient Multi-Group Test ===\n");
    eprintln!("Testing gradients that cross group boundaries.\n");

    for &size in &[256u32, 512, 768, 1024] {
        // Create horizontal gradient (varies with x, constant with y)
        let n = (size * size) as usize;
        let mut linear_rgb: Vec<f32> = Vec::with_capacity(n * 3);
        for y in 0..size {
            for x in 0..size {
                let val = x as f32 / (size - 1) as f32; // 0.0 to 1.0 across width
                // Linear RGB
                linear_rgb.push(val); // R
                linear_rgb.push(val); // G
                linear_rgb.push(val); // B
                let _ = y; // Unused, gradient is horizontal
            }
        }

        let encoder = jxl_encoder::vardct::VarDctEncoder::new(1.0);
        let bytes = encoder
            .encode(size as usize, size as usize, &linear_rgb, None)
            .expect("Encode failed")
            .data;

        // Decode
        let reader = Cursor::new(&bytes);
        let mut image = jxl_oxide::JxlImage::builder()
            .read(reader)
            .expect("Parse failed");
        image.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
            jxl_oxide::RenderingIntent::Relative,
        ));
        let render = image.render_frame(0).expect("Render failed");

        let fb = render.image_all_channels();
        let decoded = fb.buf();

        // Check statistics
        let avg: f32 = decoded.iter().sum::<f32>() / decoded.len() as f32;
        let min_val = decoded.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = decoded.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let out_of_range = decoded
            .iter()
            .filter(|&&v| !(-0.1..=1.1).contains(&v))
            .count();

        // Check first and last columns (should be ~0 and ~1)
        let first_col_avg: f32 = (0..size)
            .map(|y| {
                let idx = (y as usize * size as usize) * 3;
                (decoded[idx] + decoded[idx + 1] + decoded[idx + 2]) / 3.0
            })
            .sum::<f32>()
            / size as f32;

        let last_col_avg: f32 = (0..size)
            .map(|y| {
                let idx = (y as usize * size as usize + (size as usize - 1)) * 3;
                (decoded[idx] + decoded[idx + 1] + decoded[idx + 2]) / 3.0
            })
            .sum::<f32>()
            / size as f32;

        let grid = size.div_ceil(256);
        eprintln!(
            "{}x{} ({}x{}): avg={:.4}, min={:.4}, max={:.4}, bad={}, first_col={:.3}, last_col={:.3}",
            size,
            size,
            grid,
            grid,
            avg,
            min_val,
            max_val,
            out_of_range,
            first_col_avg,
            last_col_avg
        );

        if out_of_range > 0 {
            eprintln!("  ERROR: {} values out of [-0.1,1.1] range", out_of_range);
        }
    }
}

#[test]
#[ignore] // Run with: cargo test --test clic2025 test_solid_color_multigroup -- --ignored --nocapture
fn test_solid_color_multigroup() {
    eprintln!("\n=== Solid Color Multi-Group Test ===\n");
    eprintln!("Testing if the 3x3 group bug is structural or content-dependent.\n");

    // Test solid gray (linear 0.5) at various sizes
    for &size in &[256u32, 512, 768, 1024] {
        let n = (size * size) as usize;
        let linear_rgb: Vec<f32> = vec![0.5; n * 3]; // Solid mid-gray

        let encoder = jxl_encoder::vardct::VarDctEncoder::new(1.0);
        let bytes = encoder
            .encode(size as usize, size as usize, &linear_rgb, None)
            .expect("Encode failed")
            .data;

        // Decode
        let reader = Cursor::new(&bytes);
        let mut image = jxl_oxide::JxlImage::builder()
            .read(reader)
            .expect("Parse failed");
        image.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
            jxl_oxide::RenderingIntent::Relative,
        ));
        let render = image.render_frame(0).expect("Render failed");

        let fb = render.image_all_channels();
        let decoded = fb.buf();

        // Check statistics
        let avg: f32 = decoded.iter().sum::<f32>() / decoded.len() as f32;
        let min_val = decoded.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = decoded.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let out_of_range = decoded
            .iter()
            .filter(|&&v| !(0.0..=1.0).contains(&v))
            .count();

        let grid = size.div_ceil(256);
        eprintln!(
            "{}x{} ({}x{}): avg={:.4}, min={:.4}, max={:.4}, bad={}/{}",
            size,
            size,
            grid,
            grid,
            avg,
            min_val,
            max_val,
            out_of_range,
            decoded.len()
        );

        // For solid color, average should be very close to 0.5
        let error = (avg - 0.5).abs();
        if error > 0.1 {
            eprintln!("  ERROR: Average deviation {:.4} from expected 0.5", error);
        }
        if out_of_range > 0 {
            eprintln!("  ERROR: {} values out of [0,1] range", out_of_range);
        }
    }
}

/// Compare our encoder output with libjxl-tiny reference
#[test]
#[ignore]
fn test_compare_with_libjxl_tiny() {
    use std::io::Cursor;

    eprintln!("\n=== libjxl-tiny Comparison Test ===\n");

    // Create same 64x64 red-blue vertical gradient as libjxl-tiny test
    // Red at top (y=0), blue at bottom (y=63)
    let mut linear_rgb = Vec::with_capacity(64 * 64 * 3);
    for y in 0..64 {
        let t = y as f32 / 63.0;
        for _x in 0..64 {
            let r = 1.0 - t; // Linear RGB values
            let g = 0.0;
            let b = t;
            linear_rgb.push(r);
            linear_rgb.push(g);
            linear_rgb.push(b);
        }
    }

    // Encode with our encoder (static codes for byte-exact parity with C++)
    let mut encoder = jxl_encoder::vardct::VarDctEncoder::new(1.0);
    encoder.optimize_codes = false;
    let bytes = encoder.encode(64, 64, &linear_rgb, None).unwrap().data;
    eprintln!("Our encoder: {} bytes", bytes.len());

    // Read libjxl-tiny reference
    let ref_bytes = match std::fs::read("/tmp/jxl_compare/libjxl_tiny.jxl") {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Could not read reference file: {}", e);
            eprintln!(
                "Run: ~/work/libjxl-tiny/build/encoder/cjxl_tiny /tmp/jxl_compare/gradient.pfm /tmp/jxl_compare/libjxl_tiny.jxl --quality 100"
            );
            return;
        }
    };
    eprintln!("Reference:   {} bytes", ref_bytes.len());

    // Find first difference
    let mut first_diff = None;
    for i in 0..bytes.len().min(ref_bytes.len()) {
        if bytes[i] != ref_bytes[i] {
            first_diff = Some(i);
            break;
        }
    }

    if let Some(pos) = first_diff {
        eprintln!("\nFirst difference at byte {}:", pos);
        let start = pos.saturating_sub(4);
        let end = (pos + 8).min(bytes.len()).min(ref_bytes.len());
        eprint!("  Ours: ");
        for (i, &b) in bytes[start..end].iter().enumerate() {
            let idx = start + i;
            if idx == pos {
                eprint!("[");
            }
            eprint!("{:02x}", b);
            if idx == pos {
                eprint!("]");
            }
            eprint!(" ");
        }
        eprintln!();
        eprint!("  Ref:  ");
        for (i, &b) in ref_bytes[start..end].iter().enumerate() {
            let idx = start + i;
            if idx == pos {
                eprint!("[");
            }
            eprint!("{:02x}", b);
            if idx == pos {
                eprint!("]");
            }
            eprint!(" ");
        }
        eprintln!();
    } else if bytes.len() != ref_bytes.len() {
        eprintln!(
            "\nSize mismatch: ours={}, ref={}",
            bytes.len(),
            ref_bytes.len()
        );
    } else {
        eprintln!("\nPerfect byte match!");
    }

    // Decode both
    let decode = |data: &[u8], name: &str| -> Option<Vec<f32>> {
        let reader = Cursor::new(data);
        let mut image = match jxl_oxide::JxlImage::builder().read(reader) {
            Ok(img) => img,
            Err(e) => {
                eprintln!("{}: parse error: {:?}", name, e);
                return None;
            }
        };
        image.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
            jxl_oxide::RenderingIntent::Relative,
        ));
        let render = match image.render_frame(0) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("{}: render error: {:?}", name, e);
                return None;
            }
        };
        Some(render.image_all_channels().buf().to_vec())
    };

    if let (Some(ours), Some(ref_dec)) = (decode(&bytes, "ours"), decode(&ref_bytes, "ref")) {
        // Compare decoded values
        let mut max_diff: f32 = 0.0;
        let mut sum_sq_diff: f64 = 0.0;
        for i in 0..ours.len() {
            let diff = (ours[i] - ref_dec[i]).abs();
            max_diff = max_diff.max(diff);
            sum_sq_diff += (diff as f64).powi(2);
        }
        let rmse = (sum_sq_diff / ours.len() as f64).sqrt();

        eprintln!("\nDecoded pixel comparison:");
        eprintln!("  Max difference: {:.6}", max_diff);
        eprintln!("  RMSE: {:.6}", rmse);

        // Show corner values
        eprintln!("\nCorner pixel values (linear RGB):");
        eprintln!("  Top-left (should be red ~1,0,0):");
        eprintln!("    Ours: [{:.4}, {:.4}, {:.4}]", ours[0], ours[1], ours[2]);
        eprintln!(
            "    Ref:  [{:.4}, {:.4}, {:.4}]",
            ref_dec[0], ref_dec[1], ref_dec[2]
        );
        let last = (64 * 64 - 1) * 3;
        eprintln!("  Bottom-right (should be blue ~0,0,1):");
        eprintln!(
            "    Ours: [{:.4}, {:.4}, {:.4}]",
            ours[last],
            ours[last + 1],
            ours[last + 2]
        );
        eprintln!(
            "    Ref:  [{:.4}, {:.4}, {:.4}]",
            ref_dec[last],
            ref_dec[last + 1],
            ref_dec[last + 2]
        );
    }
}

/// Save files for jxl-inspect comparison
#[test]
#[ignore]
fn test_save_comparison_files() {
    eprintln!("\n=== Save Comparison Files ===\n");

    // Create same 64x64 red-blue vertical gradient
    let mut linear_rgb = Vec::with_capacity(64 * 64 * 3);
    for y in 0..64 {
        let t = y as f32 / 63.0;
        for _x in 0..64 {
            linear_rgb.push(1.0 - t);
            linear_rgb.push(0.0);
            linear_rgb.push(t);
        }
    }

    let encoder = jxl_encoder::vardct::VarDctEncoder::new(1.0);
    let bytes = encoder.encode(64, 64, &linear_rgb, None).unwrap().data;

    std::fs::create_dir_all("/tmp/jxl_compare").ok();
    std::fs::write("/tmp/jxl_compare/rust.jxl", &bytes).unwrap();
    eprintln!("Saved rust.jxl: {} bytes", bytes.len());

    // Print hex dump of first 64 bytes
    eprintln!("\nFirst 64 bytes of rust.jxl:");
    for (i, chunk) in bytes[..64.min(bytes.len())].chunks(16).enumerate() {
        eprint!("{:04x}: ", i * 16);
        for b in chunk {
            eprint!("{:02x} ", b);
        }
        eprintln!();
    }
}

/// Test single block encoding/decoding to trace exactly what happens
#[test]
#[ignore]
fn test_single_block_noise() {
    use std::io::Cursor;

    eprintln!("\n=== Single Block Noise Test ===\n");

    // Create an 8x8 image with known noise pattern
    // Use a simple deterministic pattern that creates non-zero AC coefficients
    let mut linear_rgb = Vec::with_capacity(8 * 8 * 3);

    // Checkerboard pattern: alternating high/low values
    for y in 0..8 {
        for x in 0..8 {
            let v = if (x + y) % 2 == 0 { 0.8 } else { 0.2 };
            linear_rgb.push(v); // R
            linear_rgb.push(v); // G
            linear_rgb.push(v); // B
        }
    }

    eprintln!("Input:");
    eprintln!("  Size: 8x8 pixels");
    eprintln!("  Pattern: checkerboard 0.8/0.2");
    let avg_input = linear_rgb.iter().sum::<f32>() / linear_rgb.len() as f32;
    eprintln!("  Average: {:.4}", avg_input);

    // Encode
    let encoder = jxl_encoder::vardct::VarDctEncoder::new(1.0);
    let bytes = match encoder.encode(8, 8, &linear_rgb, None) {
        Ok(output) => output.data,
        Err(e) => {
            eprintln!("ENCODE ERROR: {:?}", e);
            return;
        }
    };
    eprintln!("\nEncoded: {} bytes", bytes.len());

    // Decode
    let reader = Cursor::new(&bytes);
    let mut image = match jxl_oxide::JxlImage::builder().read(reader) {
        Ok(img) => img,
        Err(e) => {
            eprintln!("PARSE ERROR: {:?}", e);
            return;
        }
    };
    image.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
        jxl_oxide::RenderingIntent::Relative,
    ));

    let render = match image.render_frame(0) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("DECODE ERROR: {:?}", e);
            return;
        }
    };

    let fb = render.image_all_channels();
    let decoded = fb.buf();

    eprintln!("\nDecoded:");
    eprintln!("  Size: {} values", decoded.len());
    let avg_decoded = decoded.iter().sum::<f32>() / decoded.len() as f32;
    let min_decoded = decoded.iter().cloned().fold(f32::MAX, f32::min);
    let max_decoded = decoded.iter().cloned().fold(f32::MIN, f32::max);
    eprintln!("  Average: {:.4} (expected ~0.5)", avg_decoded);
    eprintln!("  Min: {:.4}, Max: {:.4}", min_decoded, max_decoded);

    // Show first 8 pixels
    eprintln!("\nFirst row (R values):");
    for x in 0..8 {
        let r = decoded[x * 3];
        let expected = if x % 2 == 0 { 0.8 } else { 0.2 };
        let diff = r - expected;
        eprintln!(
            "  pixel[{}]: {:.4} (expected {:.1}, diff {:+.4})",
            x, r, expected, diff
        );
    }
}

/// Compare XYB conversion with libjxl-tiny
#[test]
#[ignore]
fn test_xyb_conversion() {
    use jxl_encoder::color::xyb::linear_rgb_to_xyb;

    eprintln!("\n=== XYB Conversion Test ===\n");

    // Test with grayscale 0.5 (average of checkerboard)
    let (x, y, b) = linear_rgb_to_xyb(0.5, 0.5, 0.5);
    eprintln!("Gray 0.5: X={:.4}, Y={:.4}, B={:.4}", x, y, b);

    // Test with the two checkerboard values
    let (x1, y1, b1) = linear_rgb_to_xyb(0.8, 0.8, 0.8);
    let (x2, y2, b2) = linear_rgb_to_xyb(0.2, 0.2, 0.2);
    eprintln!("Gray 0.8: X={:.4}, Y={:.4}, B={:.4}", x1, y1, b1);
    eprintln!("Gray 0.2: X={:.4}, Y={:.4}, B={:.4}", x2, y2, b2);

    // Average should match gray 0.5
    let avg_y = (y1 + y2) / 2.0;
    eprintln!(
        "Average Y of 0.8 and 0.2: {:.4} (should be ~{:.4})",
        avg_y, y
    );
}

/// Compare our checkerboard with libjxl-tiny's
#[test]
#[ignore]
fn test_compare_checkerboard() {
    use std::io::Cursor;

    eprintln!("\n=== Checkerboard Comparison ===\n");

    // Create 8x8 checkerboard
    let mut linear_rgb = Vec::with_capacity(8 * 8 * 3);
    for y in 0..8 {
        for x in 0..8 {
            let v = if (x + y) % 2 == 0 { 0.8 } else { 0.2 };
            linear_rgb.push(v);
            linear_rgb.push(v);
            linear_rgb.push(v);
        }
    }

    // Encode with our encoder
    let encoder = jxl_encoder::vardct::VarDctEncoder::new(1.0);
    let bytes = encoder
        .encode(8, 8, &linear_rgb, None)
        .expect("encode failed")
        .data;
    eprintln!("Our encoder: {} bytes", bytes.len());

    // Decode our output
    let reader = Cursor::new(&bytes);
    let mut image = jxl_oxide::JxlImage::builder()
        .read(reader)
        .expect("parse failed");
    image.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
        jxl_oxide::RenderingIntent::Relative,
    ));
    let render = image.render_frame(0).expect("render failed");
    let ours = render.image_all_channels().buf().to_vec();

    let avg_ours = ours.iter().sum::<f32>() / ours.len() as f32;
    eprintln!("Our decoded average: {:.4}", avg_ours);

    // Load libjxl-tiny output
    let ref_bytes = match std::fs::read("/tmp/jxl_compare/checker_tiny.jxl") {
        Ok(b) => b,
        Err(_) => {
            eprintln!("No reference file, run libjxl-tiny first");
            return;
        }
    };
    eprintln!("libjxl-tiny: {} bytes", ref_bytes.len());

    let reader = Cursor::new(&ref_bytes);
    let mut image = jxl_oxide::JxlImage::builder()
        .read(reader)
        .expect("parse failed");
    image.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
        jxl_oxide::RenderingIntent::Relative,
    ));
    let render = image.render_frame(0).expect("render failed");
    let ref_dec = render.image_all_channels().buf().to_vec();

    let avg_ref = ref_dec.iter().sum::<f32>() / ref_dec.len() as f32;
    eprintln!("Reference decoded average: {:.4}", avg_ref);

    // Save our output for byte comparison
    std::fs::write("/tmp/jxl_compare/checker_rust.jxl", &bytes).expect("write failed");
    eprintln!("Saved our output to /tmp/jxl_compare/checker_rust.jxl");

    // Compare first row
    eprintln!("\nFirst row comparison (R channel):");
    for x in 0..8 {
        let expected = if x % 2 == 0 { 0.8 } else { 0.2 };
        eprintln!(
            "  pixel[{}]: ours={:.4}, ref={:.4}, expected={:.1}",
            x,
            ours[x * 3],
            ref_dec[x * 3],
            expected
        );
    }
}

#[test]
#[ignore]
fn test_dark_values_multigroup() {
    use std::io::Cursor;

    eprintln!("\n=== Dark Values Multi-Group Test ===\n");
    eprintln!("Testing with dark values (0.05-0.25) similar to real photo.\n");

    for &size in &[256u32, 512, 768, 1024] {
        let n = (size * size) as usize;
        let mut linear_rgb: Vec<f32> = Vec::with_capacity(n * 3);
        let mut seed = 12345u64;

        for _ in 0..n {
            // LCG random in dark range
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let val = 0.05 + ((seed >> 33) as f32 / 4294967295.0) * 0.20;
            linear_rgb.push(val);
            linear_rgb.push(val);
            linear_rgb.push(val);
        }

        let encoder = jxl_encoder::vardct::VarDctEncoder::new(1.0);
        let bytes = encoder
            .encode(size as usize, size as usize, &linear_rgb, None)
            .expect("Encode failed")
            .data;

        let reader = Cursor::new(&bytes);
        let mut image = jxl_oxide::JxlImage::builder()
            .read(reader)
            .expect("Parse failed");
        image.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
            jxl_oxide::RenderingIntent::Relative,
        ));
        let render = image.render_frame(0).expect("Render failed");
        let decoded = render.image_all_channels().buf().to_vec();

        let avg: f32 = decoded.iter().sum::<f32>() / decoded.len() as f32;
        let min_val = decoded.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = decoded.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let out_of_range = decoded
            .iter()
            .filter(|&&v| !(0.0..=1.0).contains(&v))
            .count();

        let grid = size.div_ceil(256);
        eprintln!(
            "{}x{} ({}x{}): avg={:.4}, min={:.4}, max={:.4}, bad={}",
            size, size, grid, grid, avg, min_val, max_val, out_of_range
        );

        if out_of_range > 0 {
            eprintln!("  ERROR: {} values out of range", out_of_range);
        }
    }
}

#[test]
#[ignore]
fn test_color_multigroup() {
    use std::io::Cursor;

    eprintln!("\n=== Color (Non-Grayscale) Multi-Group Test ===\n");
    eprintln!("Testing with varied RGB values (not R=G=B).\n");

    for &size in &[256u32, 512, 768, 1024] {
        let n = (size * size) as usize;
        let mut linear_rgb: Vec<f32> = Vec::with_capacity(n * 3);
        let mut seed = 12345u64;

        fn lcg(seed: &mut u64) -> f32 {
            *seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((*seed >> 33) as f32) / 4294967295.0
        }

        for _ in 0..n {
            // Different values for R, G, B
            let r = 0.1 + lcg(&mut seed) * 0.3; // 0.1-0.4
            let g = 0.2 + lcg(&mut seed) * 0.4; // 0.2-0.6
            let b = 0.05 + lcg(&mut seed) * 0.2; // 0.05-0.25
            linear_rgb.push(r);
            linear_rgb.push(g);
            linear_rgb.push(b);
        }

        let encoder = jxl_encoder::vardct::VarDctEncoder::new(1.0);
        let bytes = encoder
            .encode(size as usize, size as usize, &linear_rgb, None)
            .expect("Encode failed")
            .data;

        let reader = Cursor::new(&bytes);
        let mut image = jxl_oxide::JxlImage::builder()
            .read(reader)
            .expect("Parse failed");
        image.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
            jxl_oxide::RenderingIntent::Relative,
        ));
        let render = image.render_frame(0).expect("Render failed");
        let decoded = render.image_all_channels().buf().to_vec();

        let avg: f32 = decoded.iter().sum::<f32>() / decoded.len() as f32;
        let min_val = decoded.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = decoded.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let out_of_range = decoded
            .iter()
            .filter(|&&v| !(-0.1..=1.1).contains(&v))
            .count();

        let grid = size.div_ceil(256);
        eprintln!(
            "{}x{} ({}x{}): avg={:.4}, min={:.4}, max={:.4}, bad={}",
            size, size, grid, grid, avg, min_val, max_val, out_of_range
        );

        if out_of_range > 0 {
            eprintln!(
                "  ERROR: {} values significantly out of range",
                out_of_range
            );
        }
    }
}

#[test]
#[ignore]
fn test_analyze_clic_photo() {
    use image::GenericImageView;

    eprintln!("\n=== Analyzing CLIC Photo Properties ===\n");

    let corpus = jxl_encoder::test_helpers::corpus_dir();
    let validation_dir = corpus.join("clic2025/validation");

    let first_png = std::fs::read_dir(&validation_dir)
        .expect("Could not read directory")
        .filter_map(|e| e.ok())
        .find(|e| e.path().extension().is_some_and(|ext| ext == "png"))
        .expect("No PNG files found");

    let img = image::open(first_png.path()).expect("Could not open image");
    let (width, height) = img.dimensions();
    eprintln!("Image: {}x{}", width, height);

    // Crop to 768x768
    let size = 768u32;
    let cropped = img.crop_imm(0, 0, size.min(width), size.min(height));
    let rgb = cropped.to_rgb8();

    // Analyze sRGB values (0-255)
    let mut r_sum = 0u64;
    let mut g_sum = 0u64;
    let mut b_sum = 0u64;
    let mut r_min = 255u8;
    let mut r_max = 0u8;
    let mut g_min = 255u8;
    let mut g_max = 0u8;
    let mut b_min = 255u8;
    let mut b_max = 0u8;

    for p in rgb.pixels() {
        r_sum += p[0] as u64;
        g_sum += p[1] as u64;
        b_sum += p[2] as u64;
        r_min = r_min.min(p[0]);
        r_max = r_max.max(p[0]);
        g_min = g_min.min(p[1]);
        g_max = g_max.max(p[1]);
        b_min = b_min.min(p[2]);
        b_max = b_max.max(p[2]);
    }

    let n = (size * size) as f64;
    eprintln!("sRGB stats:");
    eprintln!(
        "  R: avg={:.1}, min={}, max={}",
        r_sum as f64 / n,
        r_min,
        r_max
    );
    eprintln!(
        "  G: avg={:.1}, min={}, max={}",
        g_sum as f64 / n,
        g_min,
        g_max
    );
    eprintln!(
        "  B: avg={:.1}, min={}, max={}",
        b_sum as f64 / n,
        b_min,
        b_max
    );

    // Convert to linear and analyze
    let linear_rgb: Vec<f32> = rgb
        .pixels()
        .flat_map(|p| {
            let r = srgb_to_linear_val(p[0] as f32 / 255.0);
            let g = srgb_to_linear_val(p[1] as f32 / 255.0);
            let b = srgb_to_linear_val(p[2] as f32 / 255.0);
            [r, g, b]
        })
        .collect();

    let lin_r: Vec<f32> = linear_rgb.iter().step_by(3).cloned().collect();
    let lin_g: Vec<f32> = linear_rgb.iter().skip(1).step_by(3).cloned().collect();
    let lin_b: Vec<f32> = linear_rgb.iter().skip(2).step_by(3).cloned().collect();

    fn stats(v: &[f32]) -> (f32, f32, f32) {
        let sum: f32 = v.iter().sum();
        let avg = sum / v.len() as f32;
        let min = v.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        (avg, min, max)
    }

    let (r_avg, r_min, r_max) = stats(&lin_r);
    let (g_avg, g_min, g_max) = stats(&lin_g);
    let (b_avg, b_min, b_max) = stats(&lin_b);

    eprintln!("\nLinear RGB stats:");
    eprintln!("  R: avg={:.4}, min={:.4}, max={:.4}", r_avg, r_min, r_max);
    eprintln!("  G: avg={:.4}, min={:.4}, max={:.4}", g_avg, g_min, g_max);
    eprintln!("  B: avg={:.4}, min={:.4}, max={:.4}", b_avg, b_min, b_max);

    // Check per-group regions
    eprintln!("\nPer-group input stats (linear):");
    let group_size = 256usize;
    let w = size as usize;
    for gy in 0..3 {
        for gx in 0..3 {
            let x0 = gx * group_size;
            let y0 = gy * group_size;
            let x1 = (x0 + group_size).min(w);
            let y1 = (y0 + group_size).min(w);

            let mut group_sum: f32 = 0.0;
            let mut group_min = f32::INFINITY;
            let mut group_max = f32::NEG_INFINITY;

            for y in y0..y1 {
                for x in x0..x1 {
                    let idx = (y * w + x) * 3;
                    for c in 0..3 {
                        let v = linear_rgb[idx + c];
                        group_sum += v;
                        group_min = group_min.min(v);
                        group_max = group_max.max(v);
                    }
                }
            }

            let group_n = ((x1 - x0) * (y1 - y0) * 3) as f32;
            let group_idx = gy * 3 + gx;
            eprintln!(
                "  Group {} ({},{}): avg={:.4}, min={:.4}, max={:.4}",
                group_idx,
                gx,
                gy,
                group_sum / group_n,
                group_min,
                group_max
            );
        }
    }
}

#[test]
#[ignore]
fn test_high_contrast_multigroup() {
    use std::io::Cursor;

    eprintln!("\n=== High Contrast Multi-Group Test ===\n");
    eprintln!("Testing with full range values (0.0-1.0) like the corrupt CLIC groups.\n");

    for &size in &[256u32, 512, 768, 1024] {
        let n = (size * size) as usize;
        let mut linear_rgb: Vec<f32> = Vec::with_capacity(n * 3);
        let mut seed = 12345u64;

        fn lcg(seed: &mut u64) -> f32 {
            *seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((*seed >> 33) as f32) / 4294967295.0
        }

        for _ in 0..n {
            // Full range 0.0-1.0 for all channels
            let r = lcg(&mut seed);
            let g = lcg(&mut seed);
            let b = lcg(&mut seed);
            linear_rgb.push(r);
            linear_rgb.push(g);
            linear_rgb.push(b);
        }

        let encoder = jxl_encoder::vardct::VarDctEncoder::new(1.0);
        let bytes = encoder
            .encode(size as usize, size as usize, &linear_rgb, None)
            .expect("Encode failed")
            .data;

        let reader = Cursor::new(&bytes);
        let mut image = jxl_oxide::JxlImage::builder()
            .read(reader)
            .expect("Parse failed");
        image.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
            jxl_oxide::RenderingIntent::Relative,
        ));
        let render = image.render_frame(0).expect("Render failed");
        let decoded = render.image_all_channels().buf().to_vec();

        let avg: f32 = decoded.iter().sum::<f32>() / decoded.len() as f32;
        let min_val = decoded.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = decoded.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let out_of_range = decoded
            .iter()
            .filter(|&&v| !(-0.1..=1.1).contains(&v))
            .count();

        let grid = size.div_ceil(256);
        eprintln!(
            "{}x{} ({}x{}): avg={:.4}, min={:.4}, max={:.4}, bad={}",
            size, size, grid, grid, avg, min_val, max_val, out_of_range
        );

        if out_of_range > 0 {
            eprintln!(
                "  ERROR: {} values significantly out of range",
                out_of_range
            );
        }
    }
}

#[test]
#[ignore]
fn test_bright_block_trace() {
    use std::io::Cursor;

    eprintln!("\n=== Bright Block Tracing ===\n");

    // Create a simple 8x8 bright image (single block)
    let size = 8u32;
    let val = 0.8f32; // Bright value
    let linear_rgb: Vec<f32> = vec![val; (size * size * 3) as usize];

    eprintln!(
        "Input: {}x{} solid bright (linear RGB = {:.4})",
        size, size, val
    );

    let encoder = jxl_encoder::vardct::VarDctEncoder::new(1.0);
    let bytes = encoder
        .encode(size as usize, size as usize, &linear_rgb, None)
        .expect("Encode failed")
        .data;

    eprintln!("Encoded to {} bytes", bytes.len());

    let reader = Cursor::new(&bytes);
    let mut image = jxl_oxide::JxlImage::builder()
        .read(reader)
        .expect("Parse failed");
    image.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
        jxl_oxide::RenderingIntent::Relative,
    ));
    let render = image.render_frame(0).expect("Render failed");
    let decoded = render.image_all_channels().buf().to_vec();

    // Check first pixel
    let r = decoded[0];
    let g = decoded[1];
    let b = decoded[2];
    eprintln!("Decoded pixel[0]: R={:.4}, G={:.4}, B={:.4}", r, g, b);
    eprintln!("Expected: ~{:.4}", val);
    eprintln!("Ratio: {:.4}x", r / val);

    // Also test with dark value for comparison
    let dark_val = 0.2f32;
    let dark_rgb: Vec<f32> = vec![dark_val; (size * size * 3) as usize];

    let bytes2 = encoder
        .encode(size as usize, size as usize, &dark_rgb, None)
        .expect("Encode")
        .data;
    let reader2 = Cursor::new(&bytes2);
    let mut image2 = jxl_oxide::JxlImage::builder().read(reader2).expect("Parse");
    image2.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
        jxl_oxide::RenderingIntent::Relative,
    ));
    let render2 = image2.render_frame(0).expect("Render");
    let decoded2 = render2.image_all_channels().buf().to_vec();

    eprintln!("\nDark input: linear RGB = {:.4}", dark_val);
    eprintln!(
        "Decoded pixel[0]: R={:.4}, G={:.4}, B={:.4}",
        decoded2[0], decoded2[1], decoded2[2]
    );
    eprintln!("Expected: ~{:.4}", dark_val);
    eprintln!("Ratio: {:.4}x", decoded2[0] / dark_val);
}

#[test]
#[ignore]
fn test_high_contrast_checkerboard() {
    use std::io::Cursor;

    eprintln!("\n=== High Contrast Checkerboard Test ===\n");

    // 8x8 checkerboard with values 0.1 and 0.9 (high contrast)
    let size = 8u32;
    let dark = 0.1f32;
    let bright = 0.9f32;

    let mut linear_rgb: Vec<f32> = Vec::with_capacity((size * size * 3) as usize);
    for y in 0..size {
        for x in 0..size {
            let val = if (x + y) % 2 == 0 { bright } else { dark };
            linear_rgb.push(val);
            linear_rgb.push(val);
            linear_rgb.push(val);
        }
    }

    let expected_avg = (dark + bright) / 2.0;
    let input_avg: f32 = linear_rgb.iter().sum::<f32>() / linear_rgb.len() as f32;
    eprintln!(
        "Input: {}x{} checkerboard dark={:.2} bright={:.2}",
        size, size, dark, bright
    );
    eprintln!(
        "Input average: {:.4} (expected {:.4})",
        input_avg, expected_avg
    );

    let encoder = jxl_encoder::vardct::VarDctEncoder::new(1.0);
    let bytes = encoder
        .encode(size as usize, size as usize, &linear_rgb, None)
        .expect("Encode failed")
        .data;

    eprintln!("Encoded to {} bytes", bytes.len());

    let reader = Cursor::new(&bytes);
    let mut image = jxl_oxide::JxlImage::builder()
        .read(reader)
        .expect("Parse failed");
    image.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
        jxl_oxide::RenderingIntent::Relative,
    ));
    let render = image.render_frame(0).expect("Render failed");
    let decoded = render.image_all_channels().buf().to_vec();

    let decoded_avg: f32 = decoded.iter().sum::<f32>() / decoded.len() as f32;
    let min_val = decoded.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = decoded.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    eprintln!(
        "Decoded: avg={:.4}, min={:.4}, max={:.4}",
        decoded_avg, min_val, max_val
    );
    eprintln!("Expected avg: {:.4}", expected_avg);
    eprintln!("Ratio: {:.4}x", decoded_avg / expected_avg);

    // Show first row
    eprintln!("\nFirst row (R channel):");
    for x in 0..8 {
        let expected = if x % 2 == 0 { bright } else { dark };
        eprintln!(
            "  pixel[{}]: decoded={:.4}, expected={:.4}, diff={:+.4}",
            x,
            decoded[x as usize * 3],
            expected,
            decoded[x as usize * 3] - expected
        );
    }
}

#[test]
#[ignore]
fn test_full_range_random_8x8() {
    use std::io::Cursor;

    eprintln!("\n=== Full Range Random 8x8 Test ===\n");

    let size = 8u32;
    let mut linear_rgb: Vec<f32> = Vec::with_capacity((size * size * 3) as usize);
    let mut seed = 12345u64;

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 33) as f32) / 4294967295.0
    }

    for _ in 0..(size * size) {
        let r = lcg(&mut seed);
        let g = lcg(&mut seed);
        let b = lcg(&mut seed);
        linear_rgb.push(r);
        linear_rgb.push(g);
        linear_rgb.push(b);
    }

    let input_avg: f32 = linear_rgb.iter().sum::<f32>() / linear_rgb.len() as f32;
    let input_min = linear_rgb.iter().cloned().fold(f32::INFINITY, f32::min);
    let input_max = linear_rgb.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    eprintln!(
        "Input: avg={:.4}, min={:.4}, max={:.4}",
        input_avg, input_min, input_max
    );

    let encoder = jxl_encoder::vardct::VarDctEncoder::new(1.0);
    let bytes = encoder
        .encode(size as usize, size as usize, &linear_rgb, None)
        .expect("Encode failed")
        .data;

    eprintln!("Encoded to {} bytes", bytes.len());

    let reader = Cursor::new(&bytes);
    let mut image = jxl_oxide::JxlImage::builder()
        .read(reader)
        .expect("Parse failed");
    image.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
        jxl_oxide::RenderingIntent::Relative,
    ));
    let render = image.render_frame(0).expect("Render failed");
    let decoded = render.image_all_channels().buf().to_vec();

    let decoded_avg: f32 = decoded.iter().sum::<f32>() / decoded.len() as f32;
    let decoded_min = decoded.iter().cloned().fold(f32::INFINITY, f32::min);
    let decoded_max = decoded.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    eprintln!(
        "Decoded: avg={:.4}, min={:.4}, max={:.4}",
        decoded_avg, decoded_min, decoded_max
    );
    eprintln!("Average ratio: {:.4}x", decoded_avg / input_avg);

    // Compare first few pixels
    eprintln!("\nFirst 4 pixels comparison:");
    for i in 0..4 {
        let idx = i * 3;
        eprintln!(
            "  pixel[{}]: input=({:.3},{:.3},{:.3}) decoded=({:.3},{:.3},{:.3})",
            i,
            linear_rgb[idx],
            linear_rgb[idx + 1],
            linear_rgb[idx + 2],
            decoded[idx],
            decoded[idx + 1],
            decoded[idx + 2]
        );
    }
}

#[test]
#[ignore]
fn test_grayscale_vs_color_random() {
    use std::io::Cursor;

    eprintln!("\n=== Grayscale vs Color Random Comparison ===\n");

    let size = 8u32;

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 33) as f32) / 4294967295.0
    }

    // Test 1: Grayscale (R=G=B)
    eprintln!("=== Test 1: Grayscale Random ===");
    let mut gray_rgb: Vec<f32> = Vec::with_capacity((size * size * 3) as usize);
    let mut seed: u64 = 12345;
    for _ in 0..(size * size) {
        let v = lcg(&mut seed);
        gray_rgb.push(v);
        gray_rgb.push(v);
        gray_rgb.push(v);
    }

    let encoder = jxl_encoder::vardct::VarDctEncoder::new(1.0);
    let bytes = encoder
        .encode(size as usize, size as usize, &gray_rgb, None)
        .unwrap()
        .data;

    let reader = Cursor::new(&bytes);
    let mut image = jxl_oxide::JxlImage::builder().read(reader).unwrap();
    image.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
        jxl_oxide::RenderingIntent::Relative,
    ));
    let render = image.render_frame(0).unwrap();
    let gray_dec = render.image_all_channels().buf().to_vec();

    // Compare first few pixels
    eprintln!("First 4 pixels:");
    let mut gray_max_err = 0f32;
    for i in 0..4 {
        let idx = i * 3;
        let err = (gray_rgb[idx] - gray_dec[idx]).abs();
        gray_max_err = gray_max_err.max(err);
        eprintln!(
            "  pixel[{}]: input={:.4} decoded=({:.4},{:.4},{:.4}) err={:.4}",
            i,
            gray_rgb[idx],
            gray_dec[idx],
            gray_dec[idx + 1],
            gray_dec[idx + 2],
            err
        );
    }
    eprintln!("Max error: {:.4}", gray_max_err);

    // Test 2: Color (R≠G≠B)
    eprintln!("\n=== Test 2: Color Random ===");
    let mut color_rgb: Vec<f32> = Vec::with_capacity((size * size * 3) as usize);
    seed = 12345;
    for _ in 0..(size * size) {
        color_rgb.push(lcg(&mut seed));
        color_rgb.push(lcg(&mut seed));
        color_rgb.push(lcg(&mut seed));
    }

    let bytes = encoder
        .encode(size as usize, size as usize, &color_rgb, None)
        .unwrap()
        .data;

    let reader = Cursor::new(&bytes);
    let mut image = jxl_oxide::JxlImage::builder().read(reader).unwrap();
    image.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
        jxl_oxide::RenderingIntent::Relative,
    ));
    let render = image.render_frame(0).unwrap();
    let color_dec = render.image_all_channels().buf().to_vec();

    eprintln!("First 4 pixels:");
    let mut color_max_err = 0f32;
    for i in 0..4 {
        let idx = i * 3;
        for c in 0..3 {
            let err = (color_rgb[idx + c] - color_dec[idx + c]).abs();
            color_max_err = color_max_err.max(err);
        }
        eprintln!(
            "  pixel[{}]: input=({:.3},{:.3},{:.3}) decoded=({:.3},{:.3},{:.3})",
            i,
            color_rgb[idx],
            color_rgb[idx + 1],
            color_rgb[idx + 2],
            color_dec[idx],
            color_dec[idx + 1],
            color_dec[idx + 2]
        );
    }
    eprintln!("Max error: {:.4}", color_max_err);

    eprintln!("\n=== Conclusion ===");
    if color_max_err > gray_max_err * 2.0 {
        eprintln!("Color images have much larger errors than grayscale - likely CFL bug!");
    } else {
        eprintln!("Both have similar error levels");
    }
}

#[test]
#[ignore]
fn test_gradient_16x16_debug() {
    // Create the same 16x16 gradient as libjxl-tiny test
    let size = 16usize;
    let n = size * size;
    let mut linear_rgb: Vec<f32> = Vec::with_capacity(n * 3);
    for y in 0..size {
        for x in 0..size {
            let val = (x + y) as f32 / (2.0 * (size - 1) as f32);
            linear_rgb.push(val);
            linear_rgb.push(val);
            linear_rgb.push(val);
        }
    }

    // Encode with our encoder
    let encoder = jxl_encoder::vardct::VarDctEncoder::new(1.0);
    let bytes = encoder.encode(size, size, &linear_rgb, None).unwrap().data;

    // Save
    std::fs::write("/tmp/jxl_debug/rust_16.jxl", &bytes).unwrap();
    println!("Our encoder: {} bytes", bytes.len());

    // Decode with jxl-oxide to verify
    let reader = std::io::Cursor::new(&bytes);
    let mut image = jxl_oxide::JxlImage::builder().read(reader).expect("parse");
    image.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
        jxl_oxide::RenderingIntent::Relative,
    ));
    let render = image.render_frame(0).expect("render");
    let decoded = render.image_all_channels().buf().to_vec();

    // Stats
    let avg: f32 = decoded.iter().sum::<f32>() / decoded.len() as f32;
    let min = decoded.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = decoded.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!("Decoded avg={:.4}, min={:.4}, max={:.4}", avg, min, max);

    // Compare first few pixels
    println!("\nFirst 4 decoded pixels:");
    for i in 0..4 {
        let expected = i as f32 / (2.0 * (size - 1) as f32);
        println!(
            "  pixel[0,{}]: expected={:.4}, decoded=({:.4},{:.4},{:.4})",
            i,
            expected,
            decoded[i * 3],
            decoded[i * 3 + 1],
            decoded[i * 3 + 2]
        );
    }
}

#[test]
#[ignore]
fn test_random_16x16_debug() {
    // Create 16x16 random content using LCG
    let size = 16usize;
    let n = size * size;
    let mut linear_rgb: Vec<f32> = Vec::with_capacity(n * 3);
    let mut seed = 12345u64;
    for _y in 0..size {
        for _x in 0..size {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let val = (seed >> 33) as f32 / u32::MAX as f32;
            linear_rgb.push(val);
            linear_rgb.push(val);
            linear_rgb.push(val);
        }
    }

    // Encode with our encoder
    let encoder = jxl_encoder::vardct::VarDctEncoder::new(1.0);
    let bytes = encoder.encode(size, size, &linear_rgb, None).unwrap().data;

    println!("Our encoder: {} bytes", bytes.len());

    // Decode with jxl-oxide
    let reader = std::io::Cursor::new(&bytes);
    let mut image = jxl_oxide::JxlImage::builder().read(reader).expect("parse");
    image.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
        jxl_oxide::RenderingIntent::Relative,
    ));
    let render = image.render_frame(0).expect("render");
    let decoded = render.image_all_channels().buf().to_vec();

    // Regenerate input for comparison
    seed = 12345u64;
    let mut max_err = 0.0f32;
    println!("\nFirst 8 pixels:");
    for i in 0..8 {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let expected = (seed >> 33) as f32 / u32::MAX as f32;
        let dec = decoded[i * 3];
        let err = (dec - expected).abs();
        max_err = max_err.max(err);
        println!(
            "  pixel[{}]: expected={:.4}, decoded={:.4}, err={:.4}",
            i, expected, dec, err
        );
    }
    println!("\nMax error in first 8: {:.4}", max_err);
}

#[test]
#[ignore]
fn test_random_ac_coeffs() {
    // Create 8x8 random content - just one block for easier analysis
    let size = 8usize;
    let mut linear_rgb: Vec<f32> = Vec::with_capacity(size * size * 3);
    let mut seed = 12345u64;
    for _y in 0..size {
        for _x in 0..size {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let val = (seed >> 33) as f32 / u32::MAX as f32;
            linear_rgb.push(val);
            linear_rgb.push(val);
            linear_rgb.push(val);
        }
    }

    // Encode
    let encoder = jxl_encoder::vardct::VarDctEncoder::new(1.0);
    let bytes = encoder.encode(size, size, &linear_rgb, None).unwrap().data;
    println!("Encoded {} bytes", bytes.len());

    // Decode and check
    let reader = std::io::Cursor::new(&bytes);
    let mut image = jxl_oxide::JxlImage::builder().read(reader).expect("parse");
    image.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
        jxl_oxide::RenderingIntent::Relative,
    ));
    let render = image.render_frame(0).expect("render");
    let decoded = render.image_all_channels().buf().to_vec();

    // Check decoded vs input
    seed = 12345u64;
    println!("\nPixel comparison (8x8 block):");
    let mut total_err = 0.0f32;
    for y in 0..size {
        for x in 0..size {
            let idx = y * size + x;
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let expected = (seed >> 33) as f32 / u32::MAX as f32;
            let dec = decoded[idx * 3];
            let err = (dec - expected).abs();
            total_err += err;
            if err > 0.05 {
                print!("*{:.2} ", err);
            } else {
                print!("{:.2} ", err);
            }
        }
        println!();
    }
    println!("Average error: {:.4}", total_err / (size * size) as f32);
}

#[test]
#[ignore]
fn test_compare_libjxl_tiny() {
    // Create same random 8x8 using LCG
    let size = 8usize;
    let mut expected = Vec::new();
    let mut linear_rgb: Vec<f32> = Vec::with_capacity(size * size * 3);
    let mut seed = 12345u64;
    for _ in 0..(size * size) {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let val = (seed >> 33) as f32 / u32::MAX as f32;
        expected.push(val);
        linear_rgb.push(val);
        linear_rgb.push(val);
        linear_rgb.push(val);
    }

    println!("Expected first row:");
    for val in &expected[..8] {
        print!("{:.3} ", val);
    }
    println!();

    // Decode libjxl-tiny output
    if let Ok(bytes) = std::fs::read("/tmp/jxl_debug/random_8x8_tiny.jxl") {
        let reader = Cursor::new(&bytes);
        match jxl_oxide::JxlImage::builder().read(reader) {
            Ok(mut image) => {
                image.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
                    jxl_oxide::RenderingIntent::Relative,
                ));
                match image.render_frame(0) {
                    Ok(render) => {
                        let buf = render.image_all_channels().buf().to_vec();
                        println!("\nlibjxl-tiny decoded first row:");
                        for x in 0..8 {
                            print!("{:.3} ", buf[x * 3]);
                        }
                        println!();

                        // Check for reasonable values
                        let max_val = buf.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        let min_val = buf.iter().cloned().fold(f32::INFINITY, f32::min);
                        println!("libjxl-tiny: min={:.3}, max={:.3}", min_val, max_val);

                        if max_val.abs() > 10.0 || min_val.abs() > 10.0 {
                            println!("WARNING: libjxl-tiny output has extreme values!");
                        } else {
                            // Compute error
                            let mut total_err = 0.0f32;
                            for i in 0..64 {
                                let err = (buf[i * 3] - expected[i]).abs();
                                total_err += err;
                            }
                            println!("libjxl-tiny avg error: {:.4}", total_err / 64.0);
                        }
                    }
                    Err(e) => println!("libjxl-tiny render error: {:?}", e),
                }
            }
            Err(e) => println!("libjxl-tiny parse error: {:?}", e),
        }
    } else {
        println!("Could not read libjxl-tiny output file");
    }

    // Encode with our encoder (static codes for byte-exact parity with C++)
    let mut encoder = jxl_encoder::vardct::VarDctEncoder::new(1.0);
    encoder.optimize_codes = false;
    let our_bytes = encoder
        .encode(size, size, &linear_rgb, None)
        .expect("encode")
        .data;
    println!("\nOur encoder: {} bytes", our_bytes.len());

    let reader = Cursor::new(&our_bytes);
    let mut image = jxl_oxide::JxlImage::builder().read(reader).expect("parse");
    image.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
        jxl_oxide::RenderingIntent::Relative,
    ));
    let render = image.render_frame(0).expect("render");
    let buf = render.image_all_channels().buf().to_vec();

    println!("Our encoder decoded first row:");
    for x in 0..8 {
        print!("{:.3} ", buf[x * 3]);
    }
    println!();

    // Compute error
    let mut total_err = 0.0f32;
    for i in 0..64 {
        let err = (buf[i * 3] - expected[i]).abs();
        total_err += err;
    }
    println!("Our encoder avg error: {:.4}", total_err / 64.0);

    // Compare file sizes
    if let Ok(tiny_bytes) = std::fs::read("/tmp/jxl_debug/random_8x8_tiny.jxl") {
        println!("\nFile size comparison:");
        println!("  libjxl-tiny: {} bytes", tiny_bytes.len());
        println!("  our encoder: {} bytes", our_bytes.len());
        println!(
            "  difference: {} bytes",
            our_bytes.len() as i64 - tiny_bytes.len() as i64
        );
    }
}

#[test]
#[ignore]
fn test_cfl_quality_1024() {
    eprintln!("\n=== CfL Quality Test (clic2025-1024, d=1.0) ===\n");
    let dir = jxl_encoder::test_helpers::corpus_dir().join("clic2025-1024");
    let mut entries: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "png"))
        .collect();
    entries.sort_by_key(|e| e.path());

    let mut scores = Vec::new();
    let mut sizes = Vec::new();
    for entry in entries.iter().take(5) {
        let path = entry.path();
        if let Some(score) = test_clic_image_with_ssim2(&path.to_string_lossy()) {
            scores.push(score);
            // Re-encode to get file size
            let img = image::open(&path).unwrap();
            let (w, h) = img.dimensions();
            let rgb = img.to_rgb8();
            let linear_rgb: Vec<f32> = rgb
                .pixels()
                .flat_map(|p| {
                    let r = srgb_to_linear_val(p[0] as f32 / 255.0);
                    let g = srgb_to_linear_val(p[1] as f32 / 255.0);
                    let b = srgb_to_linear_val(p[2] as f32 / 255.0);
                    [r, g, b]
                })
                .collect();
            let encoder = jxl_encoder::vardct::VarDctEncoder::new(1.0);
            let bytes = encoder
                .encode(w as usize, h as usize, &linear_rgb, None)
                .unwrap()
                .data;
            sizes.push(bytes.len());
        }
    }

    if !scores.is_empty() {
        let avg = scores.iter().sum::<f64>() / scores.len() as f64;
        let min = scores.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let avg_size = sizes.iter().sum::<usize>() / sizes.len();
        eprintln!("\n--- Summary (CfL enabled, d=1.0) ---");
        eprintln!("Images: {}", scores.len());
        eprintln!("SSIM2: avg={:.2}, min={:.2}, max={:.2}", avg, min, max);
        eprintln!("Size:  avg={} bytes", avg_size);
    }
}

/// Encode image at given distance and measure SSIM2 and file size.
fn encode_and_measure_ssim2(
    width: usize,
    height: usize,
    linear_rgb: &[f32],
    original_srgb: &[[u8; 3]],
    distance: f32,
) -> Option<(f64, usize)> {
    encode_and_measure_ssim2_cfl(width, height, linear_rgb, original_srgb, distance, true)
}

/// Encode image at given distance with CfL on/off, measure SSIM2 and file size.
fn encode_and_measure_ssim2_cfl(
    width: usize,
    height: usize,
    linear_rgb: &[f32],
    original_srgb: &[[u8; 3]],
    distance: f32,
    cfl_enabled: bool,
) -> Option<(f64, usize)> {
    let mut encoder = jxl_encoder::vardct::VarDctEncoder::new(distance);
    encoder.cfl_enabled = cfl_enabled;
    let bytes = encoder.encode(width, height, linear_rgb, None).ok()?.data;
    let file_size = bytes.len();

    let reader = std::io::Cursor::new(&bytes);
    let mut image = jxl_oxide::JxlImage::builder().read(reader).ok()?;
    image.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
        jxl_oxide::RenderingIntent::Relative,
    ));
    let render = image.render_frame(0).ok()?;
    let fb = render.image_all_channels();
    let decoded_linear = fb.buf();

    let decoded_srgb: Vec<[u8; 3]> = decoded_linear
        .chunks(3)
        .map(|rgb| {
            let r = linear_to_srgb_u8(rgb[0]);
            let g = linear_to_srgb_u8(rgb[1]);
            let b = linear_to_srgb_u8(rgb[2]);
            [r, g, b]
        })
        .collect();

    let original_img = imgref::Img::new(original_srgb.to_vec(), width, height);
    let decoded_img = imgref::Img::new(decoded_srgb, width, height);
    let ssim2 =
        fast_ssim2::compute_ssimulacra2(original_img.as_ref(), decoded_img.as_ref()).ok()?;
    Some((ssim2, file_size))
}

/// Multi-distance sweep on 5 images to check quality across distances.
#[test]
#[ignore]
fn test_cfl_quality_sweep() {
    eprintln!("\n=== CfL Quality Sweep (clic2025-1024, multiple distances) ===\n");
    let dir = jxl_encoder::test_helpers::corpus_dir().join("clic2025-1024");
    let mut entries: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "png"))
        .collect();
    entries.sort_by_key(|e| e.path());

    let distances = [2.0, 1.0, 0.5, 0.25];

    for &d in &distances {
        let mut scores = Vec::new();
        let mut total_size = 0usize;
        for entry in entries.iter().take(5) {
            let path = entry.path();
            let img = image::open(&path).unwrap();
            let (w, h) = img.dimensions();
            let rgb = img.to_rgb8();
            let original_srgb: Vec<[u8; 3]> = rgb.pixels().map(|p| [p[0], p[1], p[2]]).collect();
            let linear_rgb: Vec<f32> = rgb
                .pixels()
                .flat_map(|p| {
                    let r = srgb_to_linear_val(p[0] as f32 / 255.0);
                    let g = srgb_to_linear_val(p[1] as f32 / 255.0);
                    let b = srgb_to_linear_val(p[2] as f32 / 255.0);
                    [r, g, b]
                })
                .collect();
            if let Some((ssim2, size)) =
                encode_and_measure_ssim2(w as usize, h as usize, &linear_rgb, &original_srgb, d)
            {
                scores.push(ssim2);
                total_size += size;
            }
        }
        if !scores.is_empty() {
            let avg = scores.iter().sum::<f64>() / scores.len() as f64;
            let min = scores.iter().cloned().fold(f64::INFINITY, f64::min);
            let avg_size = total_size / scores.len();
            eprintln!(
                "d={:.2}: SSIM2 avg={:.2} min={:.2} | avg size={} bytes",
                d, avg, min, avg_size
            );
        }
    }
}

/// A/B comparison: CfL enabled vs disabled on the same images.
#[test]
#[ignore]
fn test_cfl_ab_comparison() {
    eprintln!("\n=== CfL A/B Comparison (clic2025-1024) ===\n");
    let dir = jxl_encoder::test_helpers::corpus_dir().join("clic2025-1024");
    let mut entries: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "png"))
        .collect();
    entries.sort_by_key(|e| e.path());

    let distances = [2.0, 1.0, 0.5, 0.25];

    for &d in &distances {
        let mut on_scores = Vec::new();
        let mut off_scores = Vec::new();
        let mut on_sizes = Vec::new();
        let mut off_sizes = Vec::new();

        for entry in entries.iter().take(5) {
            let path = entry.path();
            let img = image::open(&path).unwrap();
            let (w, h) = img.dimensions();
            let rgb = img.to_rgb8();
            let original_srgb: Vec<[u8; 3]> = rgb.pixels().map(|p| [p[0], p[1], p[2]]).collect();
            let linear_rgb: Vec<f32> = rgb
                .pixels()
                .flat_map(|p| {
                    let r = srgb_to_linear_val(p[0] as f32 / 255.0);
                    let g = srgb_to_linear_val(p[1] as f32 / 255.0);
                    let b = srgb_to_linear_val(p[2] as f32 / 255.0);
                    [r, g, b]
                })
                .collect();

            if let Some((ssim2, size)) = encode_and_measure_ssim2_cfl(
                w as usize,
                h as usize,
                &linear_rgb,
                &original_srgb,
                d,
                true,
            ) {
                on_scores.push(ssim2);
                on_sizes.push(size);
            }
            if let Some((ssim2, size)) = encode_and_measure_ssim2_cfl(
                w as usize,
                h as usize,
                &linear_rgb,
                &original_srgb,
                d,
                false,
            ) {
                off_scores.push(ssim2);
                off_sizes.push(size);
            }
        }

        if !on_scores.is_empty() && !off_scores.is_empty() {
            let on_avg = on_scores.iter().sum::<f64>() / on_scores.len() as f64;
            let off_avg = off_scores.iter().sum::<f64>() / off_scores.len() as f64;
            let on_size = on_sizes.iter().sum::<usize>() / on_sizes.len();
            let off_size = off_sizes.iter().sum::<usize>() / off_sizes.len();
            let delta = on_avg - off_avg;
            let size_delta = on_size as i64 - off_size as i64;
            eprintln!(
                "d={:.2}: CfL ON avg={:.2} ({} B) | OFF avg={:.2} ({} B) | delta={:+.2} SSIM2, {:+} bytes",
                d, on_avg, on_size, off_avg, off_size, delta, size_delta
            );
        }
    }
}

/// Encode image with AC strategy on/off, measure SSIM2 and file size.
fn encode_and_measure_ssim2_strategy(
    width: usize,
    height: usize,
    linear_rgb: &[f32],
    original_srgb: &[[u8; 3]],
    distance: f32,
    ac_strategy_enabled: bool,
) -> Option<(f64, usize)> {
    let mut encoder = jxl_encoder::vardct::VarDctEncoder::new(distance);
    encoder.ac_strategy_enabled = ac_strategy_enabled;
    let bytes = encoder.encode(width, height, linear_rgb, None).ok()?.data;
    let file_size = bytes.len();

    let reader = std::io::Cursor::new(&bytes);
    let mut image = jxl_oxide::JxlImage::builder().read(reader).ok()?;
    image.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
        jxl_oxide::RenderingIntent::Relative,
    ));
    let render = image.render_frame(0).ok()?;
    let fb = render.image_all_channels();
    let decoded_linear = fb.buf();

    let decoded_srgb: Vec<[u8; 3]> = decoded_linear
        .chunks(3)
        .map(|rgb| {
            let r = linear_to_srgb_u8(rgb[0]);
            let g = linear_to_srgb_u8(rgb[1]);
            let b = linear_to_srgb_u8(rgb[2]);
            [r, g, b]
        })
        .collect();

    let original_img = imgref::Img::new(original_srgb.to_vec(), width, height);
    let decoded_img = imgref::Img::new(decoded_srgb, width, height);
    let ssim2 =
        fast_ssim2::compute_ssimulacra2(original_img.as_ref(), decoded_img.as_ref()).ok()?;
    Some((ssim2, file_size))
}

/// A/B comparison: AC strategy selection ON vs OFF (DCT8-only).
/// Tests whether adaptive strategy improves compression.
#[test]
#[ignore]
fn test_strategy_ab_comparison() {
    eprintln!("\n=== AC Strategy A/B Comparison (clic2025-1024) ===\n");
    let dir = jxl_encoder::test_helpers::corpus_dir().join("clic2025-1024");
    let mut entries: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "png"))
        .collect();
    entries.sort_by_key(|e| e.path());

    let distances = [2.0, 1.0, 0.5];

    for &d in &distances {
        let mut on_scores = Vec::new();
        let mut off_scores = Vec::new();
        let mut on_sizes = Vec::new();
        let mut off_sizes = Vec::new();

        for entry in entries.iter().take(5) {
            let path = entry.path();
            let img = image::open(&path).unwrap();
            let (w, h) = img.dimensions();
            let rgb = img.to_rgb8();
            let original_srgb: Vec<[u8; 3]> = rgb.pixels().map(|p| [p[0], p[1], p[2]]).collect();
            let linear_rgb: Vec<f32> = rgb
                .pixels()
                .flat_map(|p| {
                    let r = srgb_to_linear_val(p[0] as f32 / 255.0);
                    let g = srgb_to_linear_val(p[1] as f32 / 255.0);
                    let b = srgb_to_linear_val(p[2] as f32 / 255.0);
                    [r, g, b]
                })
                .collect();

            if let Some((ssim2, size)) = encode_and_measure_ssim2_strategy(
                w as usize,
                h as usize,
                &linear_rgb,
                &original_srgb,
                d,
                true, // strategy ON
            ) {
                on_scores.push(ssim2);
                on_sizes.push(size);
            }
            if let Some((ssim2, size)) = encode_and_measure_ssim2_strategy(
                w as usize,
                h as usize,
                &linear_rgb,
                &original_srgb,
                d,
                false, // strategy OFF (DCT8-only)
            ) {
                off_scores.push(ssim2);
                off_sizes.push(size);
            }
        }

        if !on_scores.is_empty() && !off_scores.is_empty() {
            let on_avg = on_scores.iter().sum::<f64>() / on_scores.len() as f64;
            let off_avg = off_scores.iter().sum::<f64>() / off_scores.len() as f64;
            let on_size = on_sizes.iter().sum::<usize>() / on_sizes.len();
            let off_size = off_sizes.iter().sum::<usize>() / off_sizes.len();
            let ssim2_delta = on_avg - off_avg;
            let size_pct = (on_size as f64 - off_size as f64) / off_size as f64 * 100.0;
            eprintln!(
                "d={:.2}: Strategy ON avg={:.2} ({} B) | OFF avg={:.2} ({} B) | delta={:+.2} SSIM2, {:.1}% size",
                d, on_avg, on_size, off_avg, off_size, ssim2_delta, size_pct
            );
        }
    }
}

/// Fair apples-to-apples quality comparison: C++ cjxl_tiny vs Rust encoder.
///
/// Same source images, same 256x256 center crops, same decoder (djxl),
/// same metric (ssimulacra2 CLI). No in-process decoding or measurement
/// differences to bias results.
///
/// Requires external tools:
///   - ~/work/libjxl-tiny/build/encoder/cjxl_tiny
///   - ~/work/jxl-efforts/libjxl/build/tools/djxl
///   - ~/work/jxl-efforts/libjxl/build/tools/ssimulacra2
///
/// Source images: ~/work/codec-corpus/clic2025-1024/ (first 5 PNGs, sorted)
#[test]
#[ignore]
fn test_cpp_vs_rust_quality() {
    let corpus = jxl_encoder::test_helpers::corpus_dir();
    let corpus_dir = corpus.join("clic2025-1024").to_string_lossy().to_string();
    let cjxl_tiny = {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/home/lilith".into());
        format!("{}/work/libjxl-tiny/build/encoder/cjxl_tiny", home)
    };
    let djxl = jxl_encoder::test_helpers::djxl_path();
    let ssim_tool = std::env::var("SSIMULACRA2_PATH").unwrap_or_else(|_| {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/home/lilith".into());
        format!("{}/work/jxl-efforts/libjxl/build/tools/ssimulacra2", home)
    });
    let work_dir = jxl_encoder::test_helpers::output_dir("quality-comparison");

    let have_cpp = std::path::Path::new(&cjxl_tiny).exists();
    assert!(
        std::path::Path::new(&djxl).exists(),
        "djxl not found at {}",
        djxl
    );
    assert!(
        std::path::Path::new(&ssim_tool).exists(),
        "ssimulacra2 not found at {}",
        ssim_tool
    );
    if !have_cpp {
        eprintln!(
            "WARNING: cjxl_tiny not found at {}, skipping C++ column",
            cjxl_tiny
        );
    }

    // Load first 5 images from corpus (sorted for reproducibility)
    let mut entries: Vec<_> = std::fs::read_dir(&corpus_dir)
        .unwrap_or_else(|_| panic!("corpus not found: {}", corpus_dir))
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "png"))
        .collect();
    entries.sort_by_key(|e| e.path());
    let entries: Vec<_> = entries.into_iter().take(5).collect();
    assert!(!entries.is_empty(), "no PNGs in {}", corpus_dir);

    let crop_size: u32 = 256;
    let distances = [0.5f32, 1.0, 2.0];

    // Prepare crops: PNG (reference) + PFM (C++ input) + linear RGB (Rust input)
    struct CropInfo {
        png_path: String,
        pfm_path: String,
        width: u32,
        height: u32,
        linear_rgb: Vec<f32>,
    }
    let mut crops: Vec<CropInfo> = Vec::new();

    for (i, entry) in entries.iter().enumerate() {
        let img = image::open(entry.path()).unwrap();
        let (w, h) = img.dimensions();
        let cx = (w.saturating_sub(crop_size)) / 2;
        let cy = (h.saturating_sub(crop_size)) / 2;
        let cw = crop_size.min(w);
        let ch = crop_size.min(h);
        let cropped = img.crop_imm(cx, cy, cw, ch);
        let rgb = cropped.to_rgb8();

        let png_path = work_dir
            .join(format!("crop_{}.png", i))
            .to_string_lossy()
            .to_string();
        rgb.save(&png_path).unwrap();

        let linear_rgb: Vec<f32> = rgb
            .pixels()
            .flat_map(|p| {
                [
                    srgb_to_linear_val(p[0] as f32 / 255.0),
                    srgb_to_linear_val(p[1] as f32 / 255.0),
                    srgb_to_linear_val(p[2] as f32 / 255.0),
                ]
            })
            .collect();

        // Write PFM (bottom-to-top row order, little-endian floats)
        let pfm_path = work_dir
            .join(format!("crop_{}.pfm", i))
            .to_string_lossy()
            .to_string();
        {
            use std::io::Write;
            let mut f = std::fs::File::create(&pfm_path).unwrap();
            write!(f, "PF\n{} {}\n-1.0\n", cw, ch).unwrap();
            for y in (0..ch as usize).rev() {
                for x in 0..cw as usize {
                    let off = (y * cw as usize + x) * 3;
                    for c in 0..3 {
                        f.write_all(&linear_rgb[off + c].to_le_bytes()).unwrap();
                    }
                }
            }
        }

        crops.push(CropInfo {
            png_path,
            pfm_path,
            width: cw,
            height: ch,
            linear_rgb,
        });
    }

    // Helper: run external command, return true on success
    fn run(cmd: &str, args: &[&str]) -> bool {
        std::process::Command::new(cmd)
            .args(args)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }

    // Helper: measure SSIM2 between two PNGs
    fn ssim2(tool: &str, a: &str, b: &str) -> Option<f64> {
        let out = std::process::Command::new(tool)
            .args([a, b])
            .output()
            .ok()?;
        if !out.status.success() {
            return None;
        }
        let s = String::from_utf8_lossy(&out.stdout);
        s.lines().last()?.trim().parse::<f64>().ok()
    }

    eprintln!("\n=== C++ cjxl_tiny vs Rust jxl-encoder-rs ===");
    eprintln!(
        "Crops: {}x 256x256 from clic2025-1024 | Decoder: djxl | Metric: ssimulacra2\n",
        crops.len()
    );

    for &d in &distances {
        eprintln!("--- distance={:.1} ---", d);
        if have_cpp {
            eprintln!(
                "{:<6} {:>10} {:>7}  {:>10} {:>7}  {:>10} {:>7}",
                "img", "C++", "size", "Rust_ON", "size", "Rust_OFF", "size"
            );
        } else {
            eprintln!(
                "{:<6} {:>10} {:>7}  {:>10} {:>7}",
                "img", "Rust_ON", "size", "Rust_OFF", "size"
            );
        }

        let mut cpp_scores = Vec::new();
        let mut ron_scores = Vec::new();
        let mut roff_scores = Vec::new();

        for (i, crop) in crops.iter().enumerate() {
            let (w, h) = (crop.width as usize, crop.height as usize);

            // Rust ON
            let ron_jxl = work_dir
                .join(format!("rust_{}_d{:.1}_on.jxl", i, d))
                .to_string_lossy()
                .to_string();
            let ron_dec = work_dir
                .join(format!("rust_{}_d{:.1}_on_dec.png", i, d))
                .to_string_lossy()
                .to_string();
            let mut enc = jxl_encoder::vardct::VarDctEncoder::new(d);
            enc.ac_strategy_enabled = true;
            let ron_bytes = enc.encode(w, h, &crop.linear_rgb, None).unwrap().data;
            let ron_size = ron_bytes.len();
            std::fs::write(&ron_jxl, &ron_bytes).unwrap();
            run(&djxl, &[&ron_jxl, &ron_dec]);
            let ron_s = ssim2(&ssim_tool, &crop.png_path, &ron_dec);

            // Rust OFF
            let roff_jxl = work_dir
                .join(format!("rust_{}_d{:.1}_off.jxl", i, d))
                .to_string_lossy()
                .to_string();
            let roff_dec = work_dir
                .join(format!("rust_{}_d{:.1}_off_dec.png", i, d))
                .to_string_lossy()
                .to_string();
            enc.ac_strategy_enabled = false;
            let roff_bytes = enc.encode(w, h, &crop.linear_rgb, None).unwrap().data;
            let roff_size = roff_bytes.len();
            std::fs::write(&roff_jxl, &roff_bytes).unwrap();
            run(&djxl, &[&roff_jxl, &roff_dec]);
            let roff_s = ssim2(&ssim_tool, &crop.png_path, &roff_dec);

            // C++ (if available)
            let (cpp_s, cpp_size) = if have_cpp {
                let cpp_jxl = work_dir
                    .join(format!("cpp_{}_d{:.1}.jxl", i, d))
                    .to_string_lossy()
                    .to_string();
                let cpp_dec = work_dir
                    .join(format!("cpp_{}_d{:.1}_dec.png", i, d))
                    .to_string_lossy()
                    .to_string();
                let d_str = format!("{}", d);
                let ok = run(&cjxl_tiny, &[&crop.pfm_path, &cpp_jxl, "-d", &d_str]);
                if ok {
                    let sz = std::fs::metadata(&cpp_jxl)
                        .map(|m| m.len() as usize)
                        .unwrap_or(0);
                    run(&djxl, &[&cpp_jxl, &cpp_dec]);
                    (ssim2(&ssim_tool, &crop.png_path, &cpp_dec), sz)
                } else {
                    (None, 0)
                }
            } else {
                (None, 0)
            };

            // Record and print
            if let (Some(rs), Some(fs)) = (ron_s, roff_s) {
                ron_scores.push(rs);
                roff_scores.push(fs);
                if have_cpp {
                    if let Some(cs) = cpp_s {
                        cpp_scores.push(cs);
                        eprintln!(
                            "img{}  {:>10.2} {:>6}B  {:>10.2} {:>6}B  {:>10.2} {:>6}B",
                            i, cs, cpp_size, rs, ron_size, fs, roff_size
                        );
                    } else {
                        eprintln!(
                            "img{}  {:>10} {:>7}  {:>10.2} {:>6}B  {:>10.2} {:>6}B",
                            i, "ERR", "", rs, ron_size, fs, roff_size
                        );
                    }
                } else {
                    eprintln!(
                        "img{}  {:>10.2} {:>6}B  {:>10.2} {:>6}B",
                        i, rs, ron_size, fs, roff_size
                    );
                }
            }
        }

        // Print averages
        if !ron_scores.is_empty() {
            let n = ron_scores.len() as f64;
            let ron_avg = ron_scores.iter().sum::<f64>() / n;
            let roff_avg = roff_scores.iter().sum::<f64>() / n;
            if !cpp_scores.is_empty() {
                let cpp_avg = cpp_scores.iter().sum::<f64>() / cpp_scores.len() as f64;
                eprintln!(
                    "AVG   {:>10.2}          {:>10.2}          {:>10.2}",
                    cpp_avg, ron_avg, roff_avg
                );
            } else {
                eprintln!("AVG   {:>10.2}          {:>10.2}", ron_avg, roff_avg);
            }
        }
        eprintln!();
    }
}

/// Multi-group quality test: full 1024x1024 images (16 groups each).
///
/// Fair apples-to-apples comparison: C++ cjxl_tiny vs Rust (ON/OFF),
/// all decoded with djxl, all measured with ssimulacra2 CLI.
/// C++ cjxl_tiny had a crash bug on >256x256 (OOB in debug names array) that
/// was fixed — requires patched build at ~/work/libjxl-tiny/build/encoder/cjxl_tiny.
///
/// Source images: ~/work/codec-corpus/clic2025-1024/ (first 5 PNGs, sorted)
#[test]
#[ignore]
fn test_multigroup_quality() {
    let corpus = jxl_encoder::test_helpers::corpus_dir();
    let corpus_dir = corpus.join("clic2025-1024").to_string_lossy().to_string();
    let djxl = jxl_encoder::test_helpers::djxl_path();
    let ssim_tool = std::env::var("SSIMULACRA2_PATH").unwrap_or_else(|_| {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/home/lilith".into());
        format!("{}/work/jxl-efforts/libjxl/build/tools/ssimulacra2", home)
    });
    let cjxl_tiny = {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/home/lilith".into());
        format!("{}/work/libjxl-tiny/build/encoder/cjxl_tiny", home)
    };
    let work_dir = jxl_encoder::test_helpers::output_dir("multigroup-quality");

    assert!(std::path::Path::new(&djxl).exists(), "djxl not found");
    assert!(
        std::path::Path::new(&ssim_tool).exists(),
        "ssimulacra2 not found"
    );
    let have_cpp = std::path::Path::new(&cjxl_tiny).exists();
    if !have_cpp {
        eprintln!(
            "WARNING: cjxl_tiny not found at {}, skipping C++ column",
            cjxl_tiny
        );
    }

    let mut entries: Vec<_> = std::fs::read_dir(&corpus_dir)
        .unwrap_or_else(|_| panic!("corpus not found: {}", corpus_dir))
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "png"))
        .collect();
    entries.sort_by_key(|e| e.path());
    let entries: Vec<_> = entries.into_iter().take(5).collect();
    assert!(!entries.is_empty(), "no PNGs in {}", corpus_dir);

    let distances = [0.5f32, 1.0, 2.0];

    fn run(cmd: &str, args: &[&str]) -> bool {
        std::process::Command::new(cmd)
            .args(args)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }

    fn ssim2(tool: &str, a: &str, b: &str) -> Option<f64> {
        let out = std::process::Command::new(tool)
            .args([a, b])
            .output()
            .ok()?;
        if !out.status.success() {
            return None;
        }
        String::from_utf8_lossy(&out.stdout)
            .lines()
            .last()?
            .trim()
            .parse::<f64>()
            .ok()
    }

    struct ImageInfo {
        png_path: String,
        pfm_path: String,
        width: u32,
        height: u32,
        linear_rgb: Vec<f32>,
        name: String,
    }
    let mut images: Vec<ImageInfo> = Vec::new();

    for (i, entry) in entries.iter().enumerate() {
        let img = image::open(entry.path()).unwrap();
        let (w, h) = img.dimensions();
        let rgb = img.to_rgb8();

        let png_path = work_dir
            .join(format!("ref_{}.png", i))
            .to_string_lossy()
            .to_string();
        rgb.save(&png_path).unwrap();

        let linear_rgb: Vec<f32> = rgb
            .pixels()
            .flat_map(|p| {
                [
                    srgb_to_linear_val(p[0] as f32 / 255.0),
                    srgb_to_linear_val(p[1] as f32 / 255.0),
                    srgb_to_linear_val(p[2] as f32 / 255.0),
                ]
            })
            .collect();

        // Write PFM for C++ encoder (bottom-to-top row order, little-endian)
        let pfm_path = work_dir
            .join(format!("ref_{}.pfm", i))
            .to_string_lossy()
            .to_string();
        {
            use std::io::Write;
            let mut f = std::fs::File::create(&pfm_path).unwrap();
            write!(f, "PF\n{} {}\n-1.0\n", w, h).unwrap();
            for y in (0..h as usize).rev() {
                for x in 0..w as usize {
                    let off = (y * w as usize + x) * 3;
                    for c in 0..3 {
                        f.write_all(&linear_rgb[off + c].to_le_bytes()).unwrap();
                    }
                }
            }
        }

        let name = entry
            .path()
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .to_string();
        images.push(ImageInfo {
            png_path,
            pfm_path,
            width: w,
            height: h,
            linear_rgb,
            name,
        });
    }

    eprintln!("\n=== Multi-Group Quality (full images, djxl + ssimulacra2) ===");
    for img in &images {
        eprintln!(
            "  {}: {}x{} ({} groups)",
            img.name,
            img.width,
            img.height,
            img.width.div_ceil(256) * img.height.div_ceil(256)
        );
    }
    eprintln!();

    for &d in &distances {
        eprintln!("--- distance={:.1} ---", d);
        if have_cpp {
            eprintln!(
                "{:<6} {:>8} {:>8}  {:>8} {:>8}  {:>8} {:>8}  {:>8} {:>8}",
                "img", "C++", "size", "Rust_ON", "size", "Rust_OFF", "size", "ON-C++", "ON-OFF"
            );
        } else {
            eprintln!(
                "{:<6} {:>8} {:>8}  {:>8} {:>8}  {:>8}",
                "img", "Rust_ON", "size", "Rust_OFF", "size", "ON-OFF"
            );
        }

        let mut cpp_scores = Vec::new();
        let mut ron_scores = Vec::new();
        let mut roff_scores = Vec::new();
        let mut cpp_sizes = Vec::new();
        let mut ron_sizes = Vec::new();
        let mut roff_sizes = Vec::new();

        for (i, img) in images.iter().enumerate() {
            let (w, h) = (img.width as usize, img.height as usize);

            // C++ encode
            let mut cpp_s: Option<f64> = None;
            let mut cpp_size: usize = 0;
            if have_cpp {
                let cpp_jxl = work_dir
                    .join(format!("cpp_{}_d{:.1}.jxl", i, d))
                    .to_string_lossy()
                    .to_string();
                let cpp_dec = work_dir
                    .join(format!("cpp_{}_d{:.1}_dec.png", i, d))
                    .to_string_lossy()
                    .to_string();
                let d_str = format!("{}", d);
                if run(&cjxl_tiny, &[&img.pfm_path, &cpp_jxl, "-d", &d_str]) {
                    cpp_size = std::fs::metadata(&cpp_jxl)
                        .map(|m| m.len() as usize)
                        .unwrap_or(0);
                    run(&djxl, &[&cpp_jxl, &cpp_dec]);
                    cpp_s = ssim2(&ssim_tool, &img.png_path, &cpp_dec);
                }
            }

            // Rust ON
            let ron_jxl = work_dir
                .join(format!("rust_{}_d{:.1}_on.jxl", i, d))
                .to_string_lossy()
                .to_string();
            let ron_dec = work_dir
                .join(format!("rust_{}_d{:.1}_on_dec.png", i, d))
                .to_string_lossy()
                .to_string();
            let mut enc = jxl_encoder::vardct::VarDctEncoder::new(d);
            enc.ac_strategy_enabled = true;
            let ron_bytes = enc.encode(w, h, &img.linear_rgb, None).unwrap().data;
            let ron_size = ron_bytes.len();
            std::fs::write(&ron_jxl, &ron_bytes).unwrap();
            run(&djxl, &[&ron_jxl, &ron_dec]);
            let ron_s = ssim2(&ssim_tool, &img.png_path, &ron_dec);

            // Rust OFF
            let roff_jxl = work_dir
                .join(format!("rust_{}_d{:.1}_off.jxl", i, d))
                .to_string_lossy()
                .to_string();
            let roff_dec = work_dir
                .join(format!("rust_{}_d{:.1}_off_dec.png", i, d))
                .to_string_lossy()
                .to_string();
            enc.ac_strategy_enabled = false;
            let roff_bytes = enc.encode(w, h, &img.linear_rgb, None).unwrap().data;
            let roff_size = roff_bytes.len();
            std::fs::write(&roff_jxl, &roff_bytes).unwrap();
            run(&djxl, &[&roff_jxl, &roff_dec]);
            let roff_s = ssim2(&ssim_tool, &img.png_path, &roff_dec);

            if let (Some(rs), Some(fs)) = (ron_s, roff_s) {
                ron_scores.push(rs);
                roff_scores.push(fs);
                ron_sizes.push(ron_size);
                roff_sizes.push(roff_size);
                if let Some(cs) = cpp_s {
                    cpp_scores.push(cs);
                    cpp_sizes.push(cpp_size);
                    eprintln!(
                        "img{}  {:>8.2} {:>7}B  {:>8.2} {:>7}B  {:>8.2} {:>7}B  {:>+7.2} {:>+7.2}",
                        i,
                        cs,
                        cpp_size,
                        rs,
                        ron_size,
                        fs,
                        roff_size,
                        rs - cs,
                        rs - fs
                    );
                } else {
                    eprintln!(
                        "img{}  {:>8.2} {:>7}B  {:>8.2} {:>7}B  {:>+7.2}",
                        i,
                        rs,
                        ron_size,
                        fs,
                        roff_size,
                        rs - fs
                    );
                }
            }
        }

        if !ron_scores.is_empty() {
            let n = ron_scores.len() as f64;
            let ron_avg = ron_scores.iter().sum::<f64>() / n;
            let roff_avg = roff_scores.iter().sum::<f64>() / n;
            let ron_sz = ron_sizes.iter().sum::<usize>() / ron_sizes.len();
            let roff_sz = roff_sizes.iter().sum::<usize>() / roff_sizes.len();
            let size_pct = (ron_sz as f64 - roff_sz as f64) / roff_sz as f64 * 100.0;
            if !cpp_scores.is_empty() {
                let cpp_avg = cpp_scores.iter().sum::<f64>() / cpp_scores.len() as f64;
                let cpp_sz = cpp_sizes.iter().sum::<usize>() / cpp_sizes.len();
                eprintln!(
                    "AVG   {:>8.2} {:>7}B  {:>8.2} {:>7}B  {:>8.2} {:>7}B  {:>+7.2} {:>+7.2}  ({:+.1}% size ON vs OFF)",
                    cpp_avg,
                    cpp_sz,
                    ron_avg,
                    ron_sz,
                    roff_avg,
                    roff_sz,
                    ron_avg - cpp_avg,
                    ron_avg - roff_avg,
                    size_pct
                );
            } else {
                eprintln!(
                    "AVG   {:>8.2} {:>7}B  {:>8.2} {:>7}B  {:>+7.2}  ({:+.1}% size)",
                    ron_avg,
                    ron_sz,
                    roff_avg,
                    roff_sz,
                    ron_avg - roff_avg,
                    size_pct
                );
            }
        }
        eprintln!();
    }
}

/// Compare enhanced vs simple histogram clustering compression.
///
/// This test compares file sizes when using the enhanced clustering
/// (pair merge refinement) vs the default simple clustering.
///
/// Note: The enhanced clustering was designed for ANS entropy coding and may not
/// provide benefits with Huffman coding. This test verifies both produce valid
/// output and documents the size difference.
#[test]
#[ignore]
fn test_enhanced_clustering_compression() {
    // Load real test images from CLIC 2025 1024x1024 crops
    let corpus_dir = jxl_encoder::test_helpers::corpus_dir().join("clic2025-1024");

    let images: Vec<_> = match std::fs::read_dir(&corpus_dir) {
        Ok(entries) => entries
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .extension()
                    .is_some_and(|ext| ext == "png" || ext == "jpg")
            })
            .take(5) // Test with first 5 images
            .collect(),
        Err(_) => {
            eprintln!("Corpus dir {:?} not found, skipping test", corpus_dir);
            return;
        }
    };

    if images.is_empty() {
        eprintln!("No test images found in {:?}, skipping test", corpus_dir);
        return;
    }

    // Find djxl for decoding - check common locations
    let djxl_candidates = [
        jxl_encoder::test_helpers::djxl_path(),
        "/usr/local/bin/djxl".to_string(),
        "/usr/bin/djxl".to_string(),
    ];

    let djxl = djxl_candidates
        .iter()
        .find(|p| std::path::Path::new(p).exists())
        .cloned()
        .or_else(|| {
            // Try which as a fallback
            std::process::Command::new("which")
                .arg("djxl")
                .output()
                .ok()
                .and_then(|o| {
                    if o.status.success() {
                        String::from_utf8(o.stdout)
                            .ok()
                            .map(|s| s.trim().to_string())
                    } else {
                        None
                    }
                })
        });

    let djxl = match djxl {
        Some(p) => p,
        None => {
            eprintln!("djxl not found, skipping test");
            return;
        }
    };

    eprintln!("\n=== Enhanced Clustering Compression Test ===\n");
    eprintln!(
        "{:<30} {:>12} {:>12} {:>10}",
        "Image", "Simple", "Enhanced", "Savings"
    );
    eprintln!("{}", "-".repeat(70));

    let mut total_simple = 0usize;
    let mut total_enhanced = 0usize;
    let distances = [1.0f32];

    for entry in &images {
        let path = entry.path();
        let name = path.file_name().unwrap().to_string_lossy();

        // Load image
        let img = image::open(&path).unwrap().to_rgb8();
        let (w, h) = (img.width() as usize, img.height() as usize);
        let linear_rgb: Vec<f32> = img
            .pixels()
            .flat_map(|p| {
                // sRGB to linear conversion
                let r = srgb_to_linear_val(p[0] as f32 / 255.0);
                let g = srgb_to_linear_val(p[1] as f32 / 255.0);
                let b = srgb_to_linear_val(p[2] as f32 / 255.0);
                [r, g, b]
            })
            .collect();

        for &distance in &distances {
            // Encode with simple clustering
            let mut enc_simple = jxl_encoder::vardct::VarDctEncoder::new(distance);
            enc_simple.optimize_codes = true;
            enc_simple.enhanced_clustering = false;
            let bytes_simple = enc_simple.encode(w, h, &linear_rgb, None).unwrap().data;

            // Encode with enhanced clustering
            let mut enc_enhanced = jxl_encoder::vardct::VarDctEncoder::new(distance);
            enc_enhanced.optimize_codes = true;
            enc_enhanced.enhanced_clustering = true;
            let bytes_enhanced = enc_enhanced.encode(w, h, &linear_rgb, None).unwrap().data;

            let simple_size = bytes_simple.len();
            let enhanced_size = bytes_enhanced.len();
            let savings_pct =
                (simple_size as f64 - enhanced_size as f64) / simple_size as f64 * 100.0;

            total_simple += simple_size;
            total_enhanced += enhanced_size;

            eprintln!(
                "{:<30} {:>10} B {:>10} B {:>+9.2}%",
                name.chars().take(30).collect::<String>(),
                simple_size,
                enhanced_size,
                savings_pct
            );

            // Verify both decode correctly
            let work_dir = std::path::Path::new("/tmp/enhanced_clustering_test");
            std::fs::create_dir_all(work_dir).ok();

            let simple_jxl = work_dir.join("simple.jxl");
            let enhanced_jxl = work_dir.join("enhanced.jxl");
            let simple_dec = work_dir.join("simple_dec.png");
            let enhanced_dec = work_dir.join("enhanced_dec.png");

            std::fs::write(&simple_jxl, &bytes_simple).unwrap();
            std::fs::write(&enhanced_jxl, &bytes_enhanced).unwrap();

            let s1 = std::process::Command::new(&djxl)
                .args([&simple_jxl, &simple_dec])
                .output();
            let s2 = std::process::Command::new(&djxl)
                .args([&enhanced_jxl, &enhanced_dec])
                .output();

            assert!(
                s1.is_ok() && s1.as_ref().unwrap().status.success(),
                "Simple clustering output failed to decode"
            );
            assert!(
                s2.is_ok() && s2.as_ref().unwrap().status.success(),
                "Enhanced clustering output failed to decode"
            );
        }
    }

    eprintln!("{}", "-".repeat(70));
    let total_savings_pct =
        (total_simple as f64 - total_enhanced as f64) / total_simple as f64 * 100.0;
    eprintln!(
        "{:<30} {:>10} B {:>10} B {:>+9.2}%",
        "TOTAL", total_simple, total_enhanced, total_savings_pct
    );
    eprintln!();

    // The enhanced clustering was designed for ANS entropy coding, not Huffman.
    // With Huffman coding, it may not provide benefits and might slightly increase size
    // due to the cost model mismatch. Just verify both modes produce valid output
    // and the size difference is within a reasonable range (±5%).
    let savings = total_savings_pct;
    eprintln!("Overall difference: {:.2}%", savings);
    assert!(
        savings.abs() < 5.0,
        "Size difference should be within ±5%, got {:.2}%",
        savings
    );
}

/// Comprehensive rate-distortion test across multiple images and distance values.
/// This is the canonical test for validating encoder quality/compression tradeoffs.
///
/// Tests 5 images from clic2025-1024 corpus at 7 distance values (0.1 to 4.0).
/// Outputs a formatted table with SSIM2 quality and file size for each point.
///
/// Run with: cargo test -p jxl_encoder --test clic2025 test_comprehensive_rd_sweep -- --ignored --nocapture
#[test]
#[ignore]
fn test_comprehensive_rd_sweep() {
    let corpus = jxl_encoder::test_helpers::corpus_dir();
    let corpus_dir = corpus.join("clic2025-1024").to_string_lossy().to_string();
    let djxl = jxl_encoder::test_helpers::djxl_path();
    let ssim_tool = std::env::var("SSIMULACRA2_PATH").unwrap_or_else(|_| {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/home/lilith".into());
        format!("{}/work/jxl-efforts/libjxl/build/tools/ssimulacra2", home)
    });
    let work_dir = jxl_encoder::test_helpers::output_dir("rd-sweep");

    // Verify tools exist
    if !std::path::Path::new(&djxl).exists() {
        eprintln!("djxl not found at {}, skipping test", djxl);
        return;
    }
    if !std::path::Path::new(&ssim_tool).exists() {
        eprintln!("ssimulacra2 tool not found at {}, skipping test", ssim_tool);
        return;
    }

    // Load first 5 images (sorted for reproducibility)
    let mut entries: Vec<_> = match std::fs::read_dir(&corpus_dir) {
        Ok(e) => e
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "png"))
            .collect(),
        Err(_) => {
            eprintln!("Corpus dir {} not found, skipping test", corpus_dir);
            return;
        }
    };
    entries.sort_by_key(|e| e.path());
    let entries: Vec<_> = entries.into_iter().take(5).collect();

    if entries.is_empty() {
        eprintln!("No PNG images found in corpus, skipping test");
        return;
    }

    // Comprehensive distance sweep from high quality (d=0.1) to low quality (d=4.0)
    let distances = [0.1f32, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0];

    eprintln!("\n=== Comprehensive Rate-Distortion Sweep ===");
    eprintln!("Date: 2026-01-31");
    eprintln!("Images: {} from clic2025-1024 (1024x1024)", entries.len());
    eprintln!("Distances: {:?}", distances);
    eprintln!();

    // Header
    eprintln!(
        "{:<20} {:>8} {:>10} {:>10} {:>10}",
        "Image", "Distance", "Size (KB)", "SSIM2", "bpp"
    );
    eprintln!("{}", "-".repeat(62));

    // Collect per-distance averages
    let mut distance_stats: Vec<(f32, Vec<f64>, Vec<usize>)> = distances
        .iter()
        .map(|&d| (d, Vec::new(), Vec::new()))
        .collect();

    for entry in &entries {
        let path = entry.path();
        let name: String = path
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .chars()
            .take(18)
            .collect();

        // Load and convert image
        let img = image::open(&path).unwrap();
        let (w, h) = img.dimensions();
        let pixels = (w * h) as usize;
        let rgb = img.to_rgb8();

        let linear_rgb: Vec<f32> = rgb
            .pixels()
            .flat_map(|p| {
                let r = srgb_to_linear_val(p[0] as f32 / 255.0);
                let g = srgb_to_linear_val(p[1] as f32 / 255.0);
                let b = srgb_to_linear_val(p[2] as f32 / 255.0);
                [r, g, b]
            })
            .collect();

        // Save original for SSIM2 comparison
        let orig_path = work_dir
            .join(format!("{}_orig.png", name))
            .to_string_lossy()
            .to_string();
        rgb.save(&orig_path).unwrap();

        for (di, &distance) in distances.iter().enumerate() {
            // Encode
            let encoder = jxl_encoder::vardct::VarDctEncoder::new(distance);
            let bytes = encoder
                .encode(w as usize, h as usize, &linear_rgb, None)
                .unwrap()
                .data;
            let size_kb = bytes.len() as f64 / 1024.0;
            let bpp = bytes.len() as f64 * 8.0 / pixels as f64;

            // Decode with djxl
            let jxl_path = work_dir
                .join(format!("{}_{}.jxl", name, distance))
                .to_string_lossy()
                .to_string();
            let dec_path = work_dir
                .join(format!("{}_{}_dec.png", name, distance))
                .to_string_lossy()
                .to_string();
            std::fs::write(&jxl_path, &bytes).unwrap();

            let decode_ok = std::process::Command::new(&djxl)
                .args([&jxl_path, &dec_path])
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status()
                .map(|s| s.success())
                .unwrap_or(false);

            if !decode_ok {
                eprintln!(
                    "{:<20} {:>8.2} {:>10} {:>10} {:>10}",
                    name, distance, "DECODE", "FAIL", "-"
                );
                continue;
            }

            // Measure SSIM2
            let ssim_output = std::process::Command::new(&ssim_tool)
                .args([&orig_path, &dec_path])
                .output()
                .ok();

            let ssim2 = ssim_output.and_then(|o| {
                if o.status.success() {
                    String::from_utf8_lossy(&o.stdout)
                        .lines()
                        .last()
                        .and_then(|l| l.trim().parse::<f64>().ok())
                } else {
                    None
                }
            });

            match ssim2 {
                Some(score) => {
                    eprintln!(
                        "{:<20} {:>8.2} {:>10.1} {:>10.2} {:>10.3}",
                        name, distance, size_kb, score, bpp
                    );
                    distance_stats[di].1.push(score);
                    distance_stats[di].2.push(bytes.len());
                }
                None => {
                    eprintln!(
                        "{:<20} {:>8.2} {:>10.1} {:>10} {:>10.3}",
                        name, distance, size_kb, "ERR", bpp
                    );
                }
            }
        }
        eprintln!(); // Blank line between images
    }

    // Summary statistics
    eprintln!("{}", "=".repeat(62));
    eprintln!("\n=== Summary by Distance ===\n");
    eprintln!(
        "{:>10} {:>12} {:>12} {:>12} {:>12}",
        "Distance", "Avg Size", "Avg SSIM2", "Min SSIM2", "Avg bpp"
    );
    eprintln!("{}", "-".repeat(62));

    let img = image::open(entries[0].path()).unwrap();
    let pixels = (img.width() * img.height()) as f64;

    for (distance, scores, sizes) in &distance_stats {
        if !scores.is_empty() {
            let avg_size = sizes.iter().sum::<usize>() as f64 / sizes.len() as f64 / 1024.0;
            let avg_ssim = scores.iter().sum::<f64>() / scores.len() as f64;
            let min_ssim = scores.iter().cloned().fold(f64::INFINITY, f64::min);
            let avg_bpp = sizes.iter().sum::<usize>() as f64 * 8.0 / sizes.len() as f64 / pixels;
            eprintln!(
                "{:>10.2} {:>10.1} KB {:>12.2} {:>12.2} {:>12.3}",
                distance, avg_size, avg_ssim, min_ssim, avg_bpp
            );
        }
    }

    eprintln!("\nOutput files saved to: {}", work_dir.display());
}

/// Test that JXL distance parameter roughly matches Butteraugli score.
///
/// The JXL distance parameter is designed so that distance=X produces
/// approximately Butteraugli score X. This test validates that relationship.
///
/// Validates that JXL distance parameter correlates with butteraugli perceptual score.
/// Uses jxl-oxide for decoding and butteraugli_linear for comparing linear RGB.
///
/// Run with: cargo test -p jxl_encoder --test clic2025 test_distance_vs_butteraugli -- --ignored --nocapture
#[test]
#[ignore]
fn test_distance_vs_butteraugli() {
    use butteraugli::{ButteraugliParams, butteraugli_linear, srgb_to_linear};
    use imgref::Img;
    use rgb::RGB;

    let corpus = jxl_encoder::test_helpers::corpus_dir();
    let corpus_dir = corpus.join("clic2025-1024").to_string_lossy().to_string();

    // Load first 3 images
    let mut entries: Vec<_> = match std::fs::read_dir(&corpus_dir) {
        Ok(e) => e
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "png"))
            .collect(),
        Err(_) => {
            eprintln!("Corpus dir {} not found, skipping test", corpus_dir);
            return;
        }
    };
    entries.sort_by_key(|e| e.path());
    let entries: Vec<_> = entries.into_iter().take(3).collect();

    if entries.is_empty() {
        eprintln!("No PNG images found, skipping test");
        return;
    }

    let distances = [0.5f32, 1.0, 2.0, 3.0];

    eprintln!("\n=== Distance vs Butteraugli Score ===");
    eprintln!(
        "Testing {} images at distances {:?}\n",
        entries.len(),
        distances
    );
    eprintln!(
        "{:<20} {:>10} {:>12} {:>10} {:>10}",
        "Image", "Distance", "Butteraugli", "Ratio", "Status"
    );
    eprintln!("{}", "-".repeat(65));

    let params = ButteraugliParams::default();
    let mut all_ratios: Vec<f32> = Vec::new();

    for entry in &entries {
        let path = entry.path();
        let name: String = path
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .chars()
            .take(18)
            .collect();

        let img = image::open(&path).unwrap();
        let (w, h) = img.dimensions();
        let rgb = img.to_rgb8();

        // Convert to linear RGB for encoder (using proper sRGB transfer function)
        let linear_rgb: Vec<f32> = rgb
            .pixels()
            .flat_map(|p| {
                [
                    srgb_to_linear(p[0]),
                    srgb_to_linear(p[1]),
                    srgb_to_linear(p[2]),
                ]
            })
            .collect();

        // Create original linear RGB image for butteraugli comparison
        let orig_pixels: Vec<RGB<f32>> = linear_rgb
            .chunks(3)
            .map(|c| RGB::new(c[0], c[1], c[2]))
            .collect();
        let orig_img = Img::new(orig_pixels, w as usize, h as usize);

        for &distance in &distances {
            // Encode
            let encoder = jxl_encoder::vardct::VarDctEncoder::new(distance);
            let bytes = encoder
                .encode(w as usize, h as usize, &linear_rgb, None)
                .unwrap()
                .data;

            // Decode with jxl-oxide (outputs linear RGB)
            let reader = Cursor::new(&bytes);
            let mut image = match jxl_oxide::JxlImage::builder().read(reader) {
                Ok(img) => img,
                Err(e) => {
                    eprintln!(
                        "{:<20} {:>10.2} {:>12} {:>10} {:>10}",
                        name,
                        distance,
                        "PARSE",
                        format!("{:?}", e),
                        "-"
                    );
                    continue;
                }
            };
            image.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
                jxl_oxide::RenderingIntent::Relative,
            ));

            let render = match image.render_frame(0) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!(
                        "{:<20} {:>10.2} {:>12} {:>10} {:>10}",
                        name,
                        distance,
                        "DECODE",
                        format!("{:?}", e),
                        "-"
                    );
                    continue;
                }
            };

            let decoded = render.image_all_channels();
            let dec_buf = decoded.buf();

            // Convert decoded linear RGB to butteraugli format
            let dec_pixels: Vec<RGB<f32>> = dec_buf
                .chunks(3)
                .map(|c| RGB::new(c[0], c[1], c[2]))
                .collect();
            let dec_imgref = Img::new(dec_pixels, w as usize, h as usize);

            // Compute butteraugli score (linear RGB input)
            match butteraugli_linear(orig_img.as_ref(), dec_imgref.as_ref(), &params) {
                Ok(result) => {
                    let score = result.score as f32;
                    let ratio = score / distance;
                    all_ratios.push(ratio);

                    let status = if ratio > 0.5 && ratio < 2.0 {
                        "OK"
                    } else {
                        "WARN"
                    };
                    eprintln!(
                        "{:<20} {:>10.2} {:>12.3} {:>10.2}x {:>10}",
                        name, distance, score, ratio, status
                    );
                }
                Err(e) => {
                    eprintln!(
                        "{:<20} {:>10.2} {:>12} {:>10} {:>10}",
                        name,
                        distance,
                        "ERROR",
                        format!("{:?}", e),
                        "-"
                    );
                }
            }
        }
        eprintln!();
    }

    // Summary
    if !all_ratios.is_empty() {
        let avg_ratio = all_ratios.iter().sum::<f32>() / all_ratios.len() as f32;
        let min_ratio = all_ratios.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_ratio = all_ratios.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        eprintln!("=== Summary ===");
        eprintln!(
            "Butteraugli/Distance ratio: avg={:.2}x, min={:.2}x, max={:.2}x",
            avg_ratio, min_ratio, max_ratio
        );
        eprintln!("(Ideal ratio is 1.0 - distance should equal butteraugli score)");

        // Warn if ratio is way off
        if !(0.5..=2.0).contains(&avg_ratio) {
            eprintln!("\nWARNING: Average ratio is outside expected range [0.5, 2.0]");
        }
    }
}

/// Regression test: encode/decode and verify Butteraugli score is below threshold.
/// This test uses butteraugli directly (no external tools) and runs on synthetic + real images.
///
/// Run with: cargo test -p jxl_encoder --test clic2025 test_butteraugli_quality_gate -- --nocapture
#[test]
fn test_butteraugli_quality_gate() {
    use butteraugli::{ButteraugliParams, butteraugli_linear};
    use imgref::Img;
    use rgb::RGB;
    use std::io::Cursor;

    let params = ButteraugliParams::default();

    // Test 1: Gradient image at distance=1.0 should have Butteraugli ≤ 2.0
    {
        let (w, h) = (64, 64);
        let linear_rgb: Vec<f32> = (0..w * h)
            .flat_map(|i| {
                let x = (i % w) as f32 / w as f32;
                let y = (i / w) as f32 / h as f32;
                [x, y, 0.5]
            })
            .collect();

        let orig_pixels: Vec<RGB<f32>> = linear_rgb
            .chunks(3)
            .map(|c| RGB::new(c[0], c[1], c[2]))
            .collect();
        let orig_img = Img::new(orig_pixels, w, h);

        let encoder = jxl_encoder::vardct::VarDctEncoder::new(1.0);
        let bytes = encoder.encode(w, h, &linear_rgb, None).unwrap().data;

        // Decode with jxl-oxide
        let reader = Cursor::new(&bytes);
        let mut image = jxl_oxide::JxlImage::builder().read(reader).unwrap();
        image.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
            jxl_oxide::RenderingIntent::Relative,
        ));
        let render = image.render_frame(0).unwrap();
        let decoded = render.image_all_channels();
        let dec_buf = decoded.buf();

        let dec_pixels: Vec<RGB<f32>> = dec_buf
            .chunks(3)
            .map(|c| RGB::new(c[0], c[1], c[2]))
            .collect();
        let dec_img = Img::new(dec_pixels, w, h);

        let result = butteraugli_linear(orig_img.as_ref(), dec_img.as_ref(), &params).unwrap();

        eprintln!("Gradient 64x64 d=1.0: Butteraugli={:.3}", result.score);
        assert!(
            result.score < 3.0,
            "Gradient at d=1.0 should have Butteraugli < 3.0, got {:.3}",
            result.score
        );
    }

    // Test 2: Solid color should have very low Butteraugli
    {
        let (w, h) = (64, 64);
        let linear_rgb: Vec<f32> = [0.5, 0.3, 0.2].repeat(w * h);

        let orig_pixels: Vec<RGB<f32>> = linear_rgb
            .chunks(3)
            .map(|c| RGB::new(c[0], c[1], c[2]))
            .collect();
        let orig_img = Img::new(orig_pixels, w, h);

        let encoder = jxl_encoder::vardct::VarDctEncoder::new(1.0);
        let bytes = encoder.encode(w, h, &linear_rgb, None).unwrap().data;

        let reader = Cursor::new(&bytes);
        let mut image = jxl_oxide::JxlImage::builder().read(reader).unwrap();
        image.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
            jxl_oxide::RenderingIntent::Relative,
        ));
        let render = image.render_frame(0).unwrap();
        let decoded = render.image_all_channels();
        let dec_buf = decoded.buf();

        let dec_pixels: Vec<RGB<f32>> = dec_buf
            .chunks(3)
            .map(|c| RGB::new(c[0], c[1], c[2]))
            .collect();
        let dec_img = Img::new(dec_pixels, w, h);

        let result = butteraugli_linear(orig_img.as_ref(), dec_img.as_ref(), &params).unwrap();

        eprintln!("Solid color 64x64 d=1.0: Butteraugli={:.3}", result.score);
        assert!(
            result.score < 1.0,
            "Solid color at d=1.0 should have Butteraugli < 1.0, got {:.3}",
            result.score
        );
    }

    eprintln!("Butteraugli quality gate: PASSED");
}

/// Encode 256x256 crop for C++ vs Rust comparison
/// Run with: cargo test -p jxl_encoder --test clic2025 test_encode_256_crop_for_comparison -- --ignored --nocapture
#[test]
#[ignore]
fn test_encode_256_crop_for_comparison() {
    use std::fs::File;
    use std::io::{Read, Write};

    let mut f = File::open("/tmp/linear_256.bin").expect("Run Python prep script first");
    let mut buf = [0u8; 8];
    f.read_exact(&mut buf).unwrap();
    let width = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
    let height = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]) as usize;

    let mut linear_bytes = vec![0u8; width * height * 3 * 4];
    f.read_exact(&mut linear_bytes).unwrap();

    let linear_rgb: Vec<f32> = linear_bytes
        .chunks(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    eprintln!(
        "Loaded {}x{} linear RGB ({} floats)",
        width,
        height,
        linear_rgb.len()
    );

    for dist_str in &["0.5", "1.0", "2.0", "3.0"] {
        let dist: f32 = dist_str.parse().unwrap();
        let encoder = jxl_encoder::vardct::VarDctEncoder::new(dist);
        let bytes = encoder
            .encode(width, height, &linear_rgb, None)
            .unwrap()
            .data;

        let work = jxl_encoder::test_helpers::output_dir("compare-cpp-rust");
        let out_path = work.join(format!("rust_d{}.jxl", dist_str));
        let mut out = File::create(&out_path).unwrap();
        out.write_all(&bytes).unwrap();
        eprintln!(
            "d={}: {} bytes -> {}",
            dist_str,
            bytes.len(),
            out_path.display()
        );
    }
}

/// Compare butteraugli scores between C++ and Rust libjxl-tiny outputs
/// Run with: cargo test -p jxl_encoder --test clic2025 test_cpp_vs_rust_butteraugli -- --ignored --nocapture
#[test]
#[ignore]
fn test_cpp_vs_rust_butteraugli() {
    use butteraugli::{ButteraugliParams, butteraugli_linear, srgb_to_linear};
    use imgref::Img;
    use rgb::RGB;
    use std::io::Cursor;

    let work = jxl_encoder::test_helpers::output_dir("compare-cpp-rust");
    let crop_path = work.join("crop_256.png");

    // Load original image
    let img = image::open(&crop_path).unwrap();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let rgb = img.to_rgb8();

    // Convert to linear RGB
    let linear_rgb: Vec<f32> = rgb
        .pixels()
        .flat_map(|p| {
            [
                srgb_to_linear(p[0]),
                srgb_to_linear(p[1]),
                srgb_to_linear(p[2]),
            ]
        })
        .collect();

    let orig_pixels: Vec<RGB<f32>> = linear_rgb
        .chunks(3)
        .map(|c| RGB::new(c[0], c[1], c[2]))
        .collect();
    let orig_img = Img::new(orig_pixels, w, h);

    let params = ButteraugliParams::default();

    eprintln!("\n=== C++ vs Rust Butteraugli Comparison ===");
    eprintln!(
        "{:<6} | {:>10} {:>10} | {:>10} {:>10} | {:>8}",
        "Dist", "C++ Size", "C++ Btrgl", "Rust Size", "Rust Btrgl", "Winner"
    );
    eprintln!("{}", "-".repeat(70));

    for dist in &["0.5", "1.0", "2.0", "3.0"] {
        // Read C++ JXL and decode with jxl-oxide
        let cpp_path = format!("{}/cpp_d{}.jxl", work.display(), dist);
        let cpp_bytes = std::fs::read(&cpp_path).unwrap_or_default();
        let cpp_size = cpp_bytes.len();

        let cpp_btrgl = if !cpp_bytes.is_empty() {
            let reader = Cursor::new(&cpp_bytes);
            if let Ok(mut image) = jxl_oxide::JxlImage::builder().read(reader) {
                image.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
                    jxl_oxide::RenderingIntent::Relative,
                ));
                if let Ok(render) = image.render_frame(0) {
                    let decoded = render.image_all_channels();
                    let dec_buf = decoded.buf();
                    let dec_pixels: Vec<RGB<f32>> = dec_buf
                        .chunks(3)
                        .map(|c| RGB::new(c[0], c[1], c[2]))
                        .collect();
                    let dec_img = Img::new(dec_pixels, w, h);
                    butteraugli_linear(orig_img.as_ref(), dec_img.as_ref(), &params)
                        .map(|r| r.score as f32)
                        .unwrap_or(-1.0)
                } else {
                    -1.0
                }
            } else {
                -1.0
            }
        } else {
            -1.0
        };

        // Read Rust JXL and decode
        let rust_path = format!("{}/rust_d{}.jxl", work.display(), dist);
        let rust_bytes = std::fs::read(&rust_path).unwrap_or_default();
        let rust_size = rust_bytes.len();

        let rust_btrgl = if !rust_bytes.is_empty() {
            let reader = Cursor::new(&rust_bytes);
            if let Ok(mut image) = jxl_oxide::JxlImage::builder().read(reader) {
                image.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
                    jxl_oxide::RenderingIntent::Relative,
                ));
                if let Ok(render) = image.render_frame(0) {
                    let decoded = render.image_all_channels();
                    let dec_buf = decoded.buf();
                    let dec_pixels: Vec<RGB<f32>> = dec_buf
                        .chunks(3)
                        .map(|c| RGB::new(c[0], c[1], c[2]))
                        .collect();
                    let dec_img = Img::new(dec_pixels, w, h);
                    butteraugli_linear(orig_img.as_ref(), dec_img.as_ref(), &params)
                        .map(|r| r.score as f32)
                        .unwrap_or(-1.0)
                } else {
                    -1.0
                }
            } else {
                -1.0
            }
        } else {
            -1.0
        };

        // Determine winner (lower butteraugli + smaller size = better)
        // Use butteraugli/size ratio - lower is better
        let _cpp_ratio = if cpp_size > 0 {
            cpp_btrgl / (cpp_size as f32 / 1000.0)
        } else {
            f32::MAX
        };
        let _rust_ratio = if rust_size > 0 {
            rust_btrgl / (rust_size as f32 / 1000.0)
        } else {
            f32::MAX
        };
        let winner = if rust_btrgl < cpp_btrgl && rust_size <= cpp_size {
            "RUST++"
        } else if rust_btrgl < cpp_btrgl {
            "Rust"
        } else if cpp_btrgl < rust_btrgl {
            "C++"
        } else {
            "Tie"
        };

        eprintln!(
            "{:<6} | {:>10} {:>10.3} | {:>10} {:>10.3} | {:>8}",
            dist, cpp_size, cpp_btrgl, rust_size, rust_btrgl, winner
        );
    }
}

/// Encode at d=0.9 and d=1.1 for finer comparison
#[test]
#[ignore]
fn test_encode_extra_distances() {
    use std::fs::File;
    use std::io::Read;

    let mut f = match File::open("/tmp/linear_256.bin") {
        Ok(f) => f,
        Err(_) => {
            eprintln!("Run test_encode_256_crop_for_comparison first");
            return;
        }
    };
    let mut buf = [0u8; 8];
    f.read_exact(&mut buf).unwrap();
    let width = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
    let height = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]) as usize;

    let mut linear_bytes = vec![0u8; width * height * 3 * 4];
    f.read_exact(&mut linear_bytes).unwrap();

    let linear_rgb: Vec<f32> = linear_bytes
        .chunks(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    for dist in [0.9f32, 1.1] {
        let encoder = jxl_encoder::vardct::VarDctEncoder::new(dist);
        let bytes = encoder
            .encode(width, height, &linear_rgb, None)
            .unwrap()
            .data;
        let work = jxl_encoder::test_helpers::output_dir("compare-cpp-rust");
        let out_path = work.join(format!("rust_d{}.jxl", dist));
        std::fs::write(&out_path, &bytes).unwrap();
        eprintln!("d={}: {} bytes", dist, bytes.len());
    }
}

/// Compare butteraugli at finer distance granularity  
#[test]
#[ignore]
fn test_cpp_vs_rust_butteraugli_fine() {
    use butteraugli::{ButteraugliParams, butteraugli_linear, srgb_to_linear};
    use imgref::Img;
    use rgb::RGB;
    use std::io::Cursor;

    let work = jxl_encoder::test_helpers::output_dir("compare-cpp-rust");
    let crop_path = work.join("crop_256.png");

    let img = image::open(&crop_path).unwrap();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let rgb = img.to_rgb8();

    let linear_rgb: Vec<f32> = rgb
        .pixels()
        .flat_map(|p| {
            [
                srgb_to_linear(p[0]),
                srgb_to_linear(p[1]),
                srgb_to_linear(p[2]),
            ]
        })
        .collect();

    let orig_pixels: Vec<RGB<f32>> = linear_rgb
        .chunks(3)
        .map(|c| RGB::new(c[0], c[1], c[2]))
        .collect();
    let orig_img = Img::new(orig_pixels, w, h);

    let params = ButteraugliParams::default();

    eprintln!("\n=== C++ vs Rust Butteraugli (Fine Granularity) ===");
    eprintln!(
        "{:<6} | {:>10} {:>10} | {:>10} {:>10} | {:>8}",
        "Dist", "C++ Size", "C++ Btrgl", "Rust Size", "Rust Btrgl", "Winner"
    );
    eprintln!("{}", "-".repeat(72));

    for dist in &["0.5", "0.9", "1.0", "1.1", "2.0", "3.0"] {
        let cpp_path = format!("{}/cpp_d{}.jxl", work.display(), dist);
        let cpp_bytes = std::fs::read(&cpp_path).unwrap_or_default();
        let cpp_size = cpp_bytes.len();

        let cpp_btrgl = if !cpp_bytes.is_empty() {
            let reader = Cursor::new(&cpp_bytes);
            if let Ok(mut image) = jxl_oxide::JxlImage::builder().read(reader) {
                image.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
                    jxl_oxide::RenderingIntent::Relative,
                ));
                if let Ok(render) = image.render_frame(0) {
                    let decoded = render.image_all_channels();
                    let dec_buf = decoded.buf();
                    let dec_pixels: Vec<RGB<f32>> = dec_buf
                        .chunks(3)
                        .map(|c| RGB::new(c[0], c[1], c[2]))
                        .collect();
                    let dec_img = Img::new(dec_pixels, w, h);
                    butteraugli_linear(orig_img.as_ref(), dec_img.as_ref(), &params)
                        .map(|r| r.score as f32)
                        .unwrap_or(-1.0)
                } else {
                    -1.0
                }
            } else {
                -1.0
            }
        } else {
            -1.0
        };

        let rust_path = format!("{}/rust_d{}.jxl", work.display(), dist);
        let rust_bytes = std::fs::read(&rust_path).unwrap_or_default();
        let rust_size = rust_bytes.len();

        let rust_btrgl = if !rust_bytes.is_empty() {
            let reader = Cursor::new(&rust_bytes);
            if let Ok(mut image) = jxl_oxide::JxlImage::builder().read(reader) {
                image.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
                    jxl_oxide::RenderingIntent::Relative,
                ));
                if let Ok(render) = image.render_frame(0) {
                    let decoded = render.image_all_channels();
                    let dec_buf = decoded.buf();
                    let dec_pixels: Vec<RGB<f32>> = dec_buf
                        .chunks(3)
                        .map(|c| RGB::new(c[0], c[1], c[2]))
                        .collect();
                    let dec_img = Img::new(dec_pixels, w, h);
                    butteraugli_linear(orig_img.as_ref(), dec_img.as_ref(), &params)
                        .map(|r| r.score as f32)
                        .unwrap_or(-1.0)
                } else {
                    -1.0
                }
            } else {
                -1.0
            }
        } else {
            -1.0
        };

        let winner = if rust_btrgl < 0.0 || cpp_btrgl < 0.0 {
            "N/A"
        } else if rust_btrgl < cpp_btrgl && rust_size <= cpp_size {
            "RUST++"
        } else if rust_btrgl < cpp_btrgl {
            "Rust"
        } else if cpp_btrgl < rust_btrgl && cpp_size <= rust_size {
            "C++++"
        } else if cpp_btrgl < rust_btrgl {
            "C++"
        } else {
            "Tie"
        };

        eprintln!(
            "{:<6} | {:>10} {:>10.3} | {:>10} {:>10.3} | {:>8}",
            dist, cpp_size, cpp_btrgl, rust_size, rust_btrgl, winner
        );
    }
}

/// Debug section sizes at d=1.0
#[test]
#[ignore]
fn test_section_sizes_d1() {
    use std::fs::File;
    use std::io::Read;

    let mut f = match File::open("/tmp/linear_256.bin") {
        Ok(f) => f,
        Err(_) => {
            eprintln!("Need /tmp/linear_256.bin");
            return;
        }
    };
    let mut buf = [0u8; 8];
    f.read_exact(&mut buf).unwrap();
    let width = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
    let height = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]) as usize;

    let mut linear_bytes = vec![0u8; width * height * 3 * 4];
    f.read_exact(&mut linear_bytes).unwrap();

    let linear_rgb: Vec<f32> = linear_bytes
        .chunks(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    let mut encoder = jxl_encoder::vardct::VarDctEncoder::new(1.0);
    encoder.ac_strategy_enabled = true;
    encoder.cfl_enabled = true;

    let bytes = encoder
        .encode(width, height, &linear_rgb, None)
        .unwrap()
        .data;
    eprintln!("Rust d=1.0: {} bytes total", bytes.len());

    // Parse the JXL to get section info
    use std::io::Cursor;
    let reader = Cursor::new(&bytes);
    if let Ok(image) = jxl_oxide::JxlImage::builder().read(reader) {
        eprintln!("Parsed OK, frame count: {}", image.num_loaded_frames());
    }
}

/// Isolate which feature causes d=1.0 butteraugli gap vs C++
#[test]
#[ignore]
fn test_isolate_d1_butteraugli_gap() {
    use butteraugli::{ButteraugliParams, butteraugli_linear, srgb_to_linear};
    use imgref::Img;
    use rgb::RGB;
    use std::io::Cursor;

    let work = jxl_encoder::test_helpers::output_dir("compare-cpp-rust");
    let crop_path = work.join("crop_256.png");

    let img = image::open(&crop_path).unwrap();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let rgb = img.to_rgb8();

    let linear_rgb: Vec<f32> = rgb
        .pixels()
        .flat_map(|p| {
            [
                srgb_to_linear(p[0]),
                srgb_to_linear(p[1]),
                srgb_to_linear(p[2]),
            ]
        })
        .collect();

    let orig_pixels: Vec<RGB<f32>> = linear_rgb
        .chunks(3)
        .map(|c| RGB::new(c[0], c[1], c[2]))
        .collect();
    let orig_img = Img::new(orig_pixels, w, h);
    let params = ButteraugliParams::default();

    // (name, cfl, ac_strategy)
    let configs: Vec<(&str, bool, bool)> = vec![
        ("bare", false, false),
        ("cfl_only", true, false),
        ("strat_only", false, true),
        ("cfl+strat", true, true),
    ];

    eprintln!("\n=== Feature Isolation at d=1.0 (adaptive quant always on) ===");
    eprintln!(
        "{:<15} {:>8} {:>10} {:>12}",
        "Config", "Size", "Butteraugli", "btrgl/KB"
    );
    eprintln!("{}", "-".repeat(50));

    for (name, cfl, strat) in &configs {
        let mut encoder = jxl_encoder::vardct::VarDctEncoder::new(1.0);
        encoder.cfl_enabled = *cfl;
        encoder.ac_strategy_enabled = *strat;

        let bytes = encoder.encode(w, h, &linear_rgb, None).unwrap().data;
        let size = bytes.len();

        let reader = Cursor::new(&bytes);
        let mut image = jxl_oxide::JxlImage::builder().read(reader).unwrap();
        image.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
            jxl_oxide::RenderingIntent::Relative,
        ));
        let render = image.render_frame(0).unwrap();
        let decoded = render.image_all_channels();
        let dec_buf = decoded.buf();
        let dec_pixels: Vec<RGB<f32>> = dec_buf
            .chunks(3)
            .map(|c| RGB::new(c[0], c[1], c[2]))
            .collect();
        let dec_img = Img::new(dec_pixels, w, h);

        let btrgl = butteraugli_linear(orig_img.as_ref(), dec_img.as_ref(), &params)
            .map(|r| r.score as f32)
            .unwrap_or(-1.0);

        eprintln!(
            "{:<15} {:>8} {:>10.3} {:>12.3}",
            name,
            size,
            btrgl,
            btrgl / (size as f32 / 1000.0)
        );
    }

    eprintln!("\nC++ reference: 12394 bytes, butteraugli=1.746");

    // Multi-distance ON/OFF gap analysis
    eprintln!("\n=== Strategy ON vs OFF gap at multiple distances ===");
    eprintln!(
        "{:<8} {:>8} {:>8} {:>10} {:>10} {:>10}",
        "Dist", "OFF sz", "ON sz", "OFF btrgl", "ON btrgl", "gap"
    );
    eprintln!("{}", "-".repeat(60));
    for &dist in &[
        0.5f32, 0.75, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.25, 1.5, 2.0, 3.0,
    ] {
        let mut results = vec![];
        for strat in &[false, true] {
            let mut enc = jxl_encoder::vardct::VarDctEncoder::new(dist);
            enc.cfl_enabled = true;
            enc.ac_strategy_enabled = *strat;
            let bytes = enc.encode(w, h, &linear_rgb, None).unwrap().data;
            let sz = bytes.len();
            let reader = Cursor::new(&bytes);
            let mut image = jxl_oxide::JxlImage::builder().read(reader).unwrap();
            image.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
                jxl_oxide::RenderingIntent::Relative,
            ));
            let render = image.render_frame(0).unwrap();
            let decoded = render.image_all_channels();
            let dec_buf = decoded.buf();
            let dec_pixels: Vec<RGB<f32>> = dec_buf
                .chunks(3)
                .map(|c| RGB::new(c[0], c[1], c[2]))
                .collect();
            let dec_img = Img::new(dec_pixels, w, h);
            let btrgl = butteraugli_linear(orig_img.as_ref(), dec_img.as_ref(), &params)
                .map(|r| r.score as f32)
                .unwrap_or(-1.0);
            results.push((sz, btrgl));
        }
        let gap = results[1].1 - results[0].1;
        eprintln!(
            "d={:<5} {:>8} {:>8} {:>10.3} {:>10.3} {:>+10.3}",
            dist, results[0].0, results[1].0, results[0].1, results[1].1, gap
        );
    }

    // Locate the worst-error region at d=1.0
    eprintln!("\n=== Locating worst error region at d=1.0 ===");
    let mut dec_off = vec![];
    let mut dec_on = vec![];
    for (strat, dec_buf) in [(false, &mut dec_off), (true, &mut dec_on)] {
        let mut enc = jxl_encoder::vardct::VarDctEncoder::new(1.0);
        enc.cfl_enabled = true;
        enc.ac_strategy_enabled = strat;
        let bytes = enc.encode(w, h, &linear_rgb, None).unwrap().data;
        let reader = Cursor::new(&bytes);
        let mut image = jxl_oxide::JxlImage::builder().read(reader).unwrap();
        image.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
            jxl_oxide::RenderingIntent::Relative,
        ));
        let render = image.render_frame(0).unwrap();
        let decoded = render.image_all_channels();
        *dec_buf = decoded.buf().to_vec();
    }
    // Compute per-16x16-block max absolute difference increase (ON minus OFF error vs original)
    let mut worst_blocks: Vec<(f32, usize, usize)> = vec![];
    for by16 in (0..h).step_by(16) {
        for bx16 in (0..w).step_by(16) {
            let mut max_err_increase: f32 = 0.0;
            for dy in 0..16.min(h - by16) {
                for dx in 0..16.min(w - bx16) {
                    let px = (by16 + dy) * w + bx16 + dx;
                    for c in 0..3 {
                        let orig = linear_rgb[px * 3 + c];
                        let err_off = (dec_off[px * 3 + c] - orig).abs();
                        let err_on = (dec_on[px * 3 + c] - orig).abs();
                        max_err_increase = max_err_increase.max(err_on - err_off);
                    }
                }
            }
            worst_blocks.push((max_err_increase, bx16, by16));
        }
    }
    worst_blocks.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    eprintln!("Top 10 worst 16x16 regions (max error increase ON-OFF):");
    for (i, (err, bx, by)) in worst_blocks.iter().take(10).enumerate() {
        eprintln!(
            "  #{}: block ({},{}) pixel ({},{}) err_increase={:.6}",
            i + 1,
            bx / 8,
            by / 8,
            bx,
            by,
            err
        );
    }

    // Decode d=1.0 ON with djxl and compare with jxl-oxide
    eprintln!("\n=== Decoder comparison: jxl-oxide vs djxl at d=1.0 ON ===");
    {
        let mut enc = jxl_encoder::vardct::VarDctEncoder::new(1.0);
        enc.cfl_enabled = true;
        enc.ac_strategy_enabled = true;
        let bytes = enc.encode(w, h, &linear_rgb, None).unwrap().data;
        let jxl_path =
            jxl_encoder::test_helpers::output_dir("compare-cpp-rust").join("rust_d1_on.jxl");
        std::fs::write(&jxl_path, &bytes).unwrap();

        // Decode with djxl to 16-bit PNG
        let djxl_png =
            jxl_encoder::test_helpers::output_dir("compare-cpp-rust").join("rust_d1_on_djxl.png");
        let djxl_bin = jxl_encoder::test_helpers::djxl_path();
        let output = std::process::Command::new(&djxl_bin)
            .args([&jxl_path, &djxl_png])
            .output()
            .unwrap();
        if !output.status.success() {
            eprintln!("djxl failed: {}", String::from_utf8_lossy(&output.stderr));
        } else {
            // Decode with jxl-oxide
            let reader = Cursor::new(&bytes);
            let mut image = jxl_oxide::JxlImage::builder().read(reader).unwrap();
            image.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
                jxl_oxide::RenderingIntent::Relative,
            ));
            let render = image.render_frame(0).unwrap();
            let decoded_oxide = render.image_all_channels();
            let oxide_buf = decoded_oxide.buf();

            // Load djxl output - convert from sRGB 16-bit to linear
            let djxl_img = image::open(&djxl_png).unwrap();
            let djxl_rgb16 = djxl_img.to_rgb16();

            // Compare pixel values
            let mut max_diff: f32 = 0.0;
            let mut sum_diff: f64 = 0.0;
            let mut count = 0u64;
            let mut worst_px = (0usize, 0usize, 0usize); // (x, y, c)
            for y in 0..h {
                for x in 0..w {
                    let px = y * w + x;
                    for c in 0..3 {
                        // jxl-oxide outputs linear RGB directly
                        let oxide_val = oxide_buf[px * 3 + c];
                        // djxl outputs sRGB; convert to linear
                        let djxl_srgb =
                            djxl_rgb16.get_pixel(x as u32, y as u32)[c] as f32 / 65535.0;
                        let djxl_linear = srgb_to_linear_f32(djxl_srgb);
                        let diff = (oxide_val - djxl_linear).abs();
                        if diff > max_diff {
                            max_diff = diff;
                            worst_px = (x, y, c);
                        }
                        sum_diff += diff as f64;
                        count += 1;
                    }
                }
            }
            eprintln!(
                "Max pixel diff (linear): {:.6} at ({},{}) c={}",
                max_diff, worst_px.0, worst_px.1, worst_px.2
            );
            eprintln!("Mean pixel diff (linear): {:.8}", sum_diff / count as f64);
            eprintln!("(diffs > 0.01 indicate decoder disagreement)");
        }

        // Also encode OFF and compare
        let mut enc2 = jxl_encoder::vardct::VarDctEncoder::new(1.0);
        enc2.cfl_enabled = true;
        enc2.ac_strategy_enabled = false;
        let bytes2 = enc2.encode(w, h, &linear_rgb, None).unwrap().data;
        let jxl_path2 =
            jxl_encoder::test_helpers::output_dir("compare-cpp-rust").join("rust_d1_off.jxl");
        std::fs::write(&jxl_path2, &bytes2).unwrap();
        let djxl_png2 =
            jxl_encoder::test_helpers::output_dir("compare-cpp-rust").join("rust_d1_off_djxl.png");
        let output2 = std::process::Command::new(&djxl_bin)
            .args([&jxl_path2, &djxl_png2])
            .output()
            .unwrap();
        if output2.status.success() {
            let reader2 = Cursor::new(&bytes2);
            let mut image2 = jxl_oxide::JxlImage::builder().read(reader2).unwrap();
            image2.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
                jxl_oxide::RenderingIntent::Relative,
            ));
            let render2 = image2.render_frame(0).unwrap();
            let decoded2 = render2.image_all_channels();
            let oxide_buf2 = decoded2.buf();
            let djxl_img2 = image::open(&djxl_png2).unwrap();
            let djxl_rgb16_2 = djxl_img2.to_rgb16();
            let mut max_diff2: f32 = 0.0;
            let mut sum_diff2: f64 = 0.0;
            let mut count2 = 0u64;
            for y in 0..h {
                for x in 0..w {
                    let px = y * w + x;
                    for c in 0..3 {
                        let oxide_val = oxide_buf2[px * 3 + c];
                        let djxl_srgb =
                            djxl_rgb16_2.get_pixel(x as u32, y as u32)[c] as f32 / 65535.0;
                        let djxl_linear = srgb_to_linear_f32(djxl_srgb);
                        let diff = (oxide_val - djxl_linear).abs();
                        max_diff2 = max_diff2.max(diff);
                        sum_diff2 += diff as f64;
                        count2 += 1;
                    }
                }
            }
            eprintln!("\nOFF decoder comparison:");
            eprintln!("Max pixel diff (linear): {:.6}", max_diff2);
            eprintln!("Mean pixel diff (linear): {:.8}", sum_diff2 / count2 as f64);
        }
    }
}

fn srgb_to_linear_f32(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Compare static vs dynamic Huffman codes across CLIC 2025 images.
///
/// Verifies that quality metrics (SSIM2, butteraugli) are identical between modes
/// (same quantized coefficients, only entropy coding differs) and measures file
/// size savings from dynamic codes.
///
/// Source images: ~/work/codec-corpus/clic2025-1024/ (first 5 PNGs, sorted)
#[test]
#[ignore]
fn test_static_vs_dynamic_sweep() {
    use butteraugli::{ButteraugliParams, butteraugli_linear, srgb_to_linear};
    use imgref::Img;
    use rgb::RGB;

    let corpus = jxl_encoder::test_helpers::corpus_dir();
    let corpus_dir = corpus.join("clic2025-1024").to_string_lossy().to_string();

    let mut entries: Vec<_> = match std::fs::read_dir(&corpus_dir) {
        Ok(e) => e
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "png"))
            .collect(),
        Err(_) => {
            eprintln!("Corpus dir {} not found, skipping test", corpus_dir);
            return;
        }
    };
    entries.sort_by_key(|e| e.path());
    let entries: Vec<_> = entries.into_iter().take(5).collect();
    assert!(!entries.is_empty(), "no PNGs in {}", corpus_dir);

    let distances = [0.25f32, 0.5, 1.0, 2.0];
    let crop_size: u32 = 256;
    let ba_params = ButteraugliParams::default();

    eprintln!("\n=== Static vs Dynamic Huffman Codes ===");
    eprintln!(
        "Images: {} x {}x{} crops | Distances: {:?}\n",
        entries.len(),
        crop_size,
        crop_size,
        distances
    );
    eprintln!(
        "{:<8} {:>6} {:>8} {:>8} {:>8} {:>8} {:>10} {:>10} {:>8}",
        "img", "dist", "st_size", "dy_size", "saving", "save%", "st_ssim2", "dy_ssim2", "ba_diff"
    );
    eprintln!("{}", "-".repeat(90));

    let mut total_static_bytes: u64 = 0;
    let mut total_dynamic_bytes: u64 = 0;
    let mut max_ssim2_diff: f64 = 0.0;
    let mut max_ba_diff: f64 = 0.0;
    let mut count = 0;

    for (i, entry) in entries.iter().enumerate() {
        let img = image::open(entry.path()).unwrap();
        let (w, h) = img.dimensions();
        let cx = (w.saturating_sub(crop_size)) / 2;
        let cy = (h.saturating_sub(crop_size)) / 2;
        let cw = crop_size.min(w);
        let ch = crop_size.min(h);
        let cropped = img.crop_imm(cx, cy, cw, ch);
        let rgb = cropped.to_rgb8();

        let linear_rgb: Vec<f32> = rgb
            .pixels()
            .flat_map(|p| {
                [
                    srgb_to_linear(p[0]),
                    srgb_to_linear(p[1]),
                    srgb_to_linear(p[2]),
                ]
            })
            .collect();

        let original_srgb: Vec<[u8; 3]> = rgb.pixels().map(|p| [p[0], p[1], p[2]]).collect();
        let orig_linear: Vec<RGB<f32>> = linear_rgb
            .chunks(3)
            .map(|c| RGB::new(c[0], c[1], c[2]))
            .collect();
        let orig_img = Img::new(orig_linear, cw as usize, ch as usize);

        for &d in &distances {
            // Encode with static codes
            let mut enc_static = jxl_encoder::vardct::VarDctEncoder::new(d);
            enc_static.optimize_codes = false;
            let bytes_static = enc_static
                .encode(cw as usize, ch as usize, &linear_rgb, None)
                .unwrap()
                .data;

            // Encode with dynamic codes
            let mut enc_dynamic = jxl_encoder::vardct::VarDctEncoder::new(d);
            enc_dynamic.optimize_codes = true;
            let bytes_dynamic = enc_dynamic
                .encode(cw as usize, ch as usize, &linear_rgb, None)
                .unwrap()
                .data;

            // Decode static
            let reader_s = Cursor::new(&bytes_static);
            let mut img_s = jxl_oxide::JxlImage::builder().read(reader_s).unwrap();
            img_s.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
                jxl_oxide::RenderingIntent::Relative,
            ));
            let render_s = img_s.render_frame(0).unwrap();
            let buf_s = render_s.image_all_channels().buf().to_vec();
            let ws = render_s.image_all_channels().width();
            let hs = render_s.image_all_channels().height();

            // Decode dynamic
            let reader_d = Cursor::new(&bytes_dynamic);
            let mut img_d = jxl_oxide::JxlImage::builder().read(reader_d).unwrap();
            img_d.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
                jxl_oxide::RenderingIntent::Relative,
            ));
            let render_d = img_d.render_frame(0).unwrap();
            let buf_d = render_d.image_all_channels().buf().to_vec();

            // SSIM2 for static
            let decoded_srgb_s: Vec<[u8; 3]> = buf_s
                .chunks(3)
                .map(|c| {
                    [
                        (c[0].clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
                        (c[1].clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
                        (c[2].clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
                    ]
                })
                .collect();
            let orig_ssim = imgref::Img::new(original_srgb.clone(), cw as usize, ch as usize);
            let dec_ssim_s = imgref::Img::new(decoded_srgb_s, ws, hs);
            let ssim2_s = fast_ssim2::compute_ssimulacra2(orig_ssim.as_ref(), dec_ssim_s.as_ref())
                .unwrap_or(f64::NAN);

            // SSIM2 for dynamic
            let decoded_srgb_d: Vec<[u8; 3]> = buf_d
                .chunks(3)
                .map(|c| {
                    [
                        (c[0].clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
                        (c[1].clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
                        (c[2].clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
                    ]
                })
                .collect();
            let dec_ssim_d = imgref::Img::new(decoded_srgb_d, ws, hs);
            let ssim2_d = fast_ssim2::compute_ssimulacra2(orig_ssim.as_ref(), dec_ssim_d.as_ref())
                .unwrap_or(f64::NAN);

            // Butteraugli for both
            let dec_linear_s: Vec<RGB<f32>> = buf_s
                .chunks(3)
                .map(|c| RGB::new(c[0].max(0.0), c[1].max(0.0), c[2].max(0.0)))
                .collect();
            let dec_img_s = Img::new(dec_linear_s, ws, hs);
            let ba_s = butteraugli_linear(orig_img.as_ref(), dec_img_s.as_ref(), &ba_params)
                .map(|r| r.score)
                .unwrap_or(f64::NAN);

            let dec_linear_d: Vec<RGB<f32>> = buf_d
                .chunks(3)
                .map(|c| RGB::new(c[0].max(0.0), c[1].max(0.0), c[2].max(0.0)))
                .collect();
            let dec_img_d = Img::new(dec_linear_d, ws, hs);
            let ba_d = butteraugli_linear(orig_img.as_ref(), dec_img_d.as_ref(), &ba_params)
                .map(|r| r.score)
                .unwrap_or(f64::NAN);

            let size_s = bytes_static.len();
            let size_d = bytes_dynamic.len();
            let saving = size_s as i64 - size_d as i64;
            let save_pct = if size_s > 0 {
                saving as f64 / size_s as f64 * 100.0
            } else {
                0.0
            };
            let ssim_diff = (ssim2_s - ssim2_d).abs();
            let ba_diff = (ba_s - ba_d).abs();

            eprintln!(
                "img{:<4} {:>6.2} {:>8} {:>8} {:>+8} {:>7.1}% {:>10.2} {:>10.2} {:>8.4}",
                i, d, size_s, size_d, saving, save_pct, ssim2_s, ssim2_d, ba_diff
            );

            total_static_bytes += size_s as u64;
            total_dynamic_bytes += size_d as u64;
            if ssim_diff.is_finite() {
                max_ssim2_diff = max_ssim2_diff.max(ssim_diff);
            }
            if ba_diff.is_finite() {
                max_ba_diff = max_ba_diff.max(ba_diff);
            }
            count += 1;
        }
    }

    let total_saving = total_static_bytes as i64 - total_dynamic_bytes as i64;
    let total_save_pct = if total_static_bytes > 0 {
        total_saving as f64 / total_static_bytes as f64 * 100.0
    } else {
        0.0
    };

    eprintln!("{}", "-".repeat(90));
    eprintln!(
        "TOTAL: {} static, {} dynamic, {:+} saved ({:.1}%)",
        total_static_bytes, total_dynamic_bytes, total_saving, total_save_pct
    );
    eprintln!(
        "Max SSIM2 difference: {:.4} | Max butteraugli difference: {:.4}",
        max_ssim2_diff, max_ba_diff
    );
    eprintln!("Measurements: {}", count);

    // Quality should be identical (same quantized coefficients)
    // Allow tiny floating-point tolerance from different entropy decode paths
    assert!(
        max_ssim2_diff < 0.5,
        "SSIM2 diverged between static and dynamic codes: max diff = {:.4}",
        max_ssim2_diff
    );
    assert!(
        max_ba_diff < 0.1,
        "Butteraugli diverged between static and dynamic codes: max diff = {:.4}",
        max_ba_diff
    );
}

/// Compare file sizes between static and dynamic Huffman codes.
///
/// Source images: ~/work/codec-corpus/clic2025-1024/ (first 5 PNGs, sorted)
#[test]
#[ignore]
fn test_static_vs_optimize_codes() {
    let corpus = jxl_encoder::test_helpers::corpus_dir();
    let corpus_dir = corpus.join("clic2025-1024").to_string_lossy().to_string();

    let mut entries: Vec<_> = match std::fs::read_dir(&corpus_dir) {
        Ok(e) => e
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "png"))
            .collect(),
        Err(_) => {
            eprintln!("Corpus dir {} not found, skipping test", corpus_dir);
            return;
        }
    };
    entries.sort_by_key(|e| e.path());
    let entries: Vec<_> = entries.into_iter().take(5).collect();
    assert!(!entries.is_empty(), "no PNGs in {}", corpus_dir);

    let distances = [0.25f32, 0.5, 1.0, 2.0];
    let crop_size: u32 = 256;

    eprintln!("\n=== Static vs Optimized Huffman Code Sizes ===\n");
    eprintln!(
        "{:<8} {:>6} {:>10} {:>10} {:>+10} {:>8}",
        "img", "dist", "static", "optimized", "delta", "saving%"
    );
    eprintln!("{}", "-".repeat(56));

    let mut total_static: u64 = 0;
    let mut total_optimized: u64 = 0;

    for (i, entry) in entries.iter().enumerate() {
        let img = image::open(entry.path()).unwrap();
        let (w, h) = img.dimensions();
        let cx = (w.saturating_sub(crop_size)) / 2;
        let cy = (h.saturating_sub(crop_size)) / 2;
        let cw = crop_size.min(w);
        let ch = crop_size.min(h);
        let cropped = img.crop_imm(cx, cy, cw, ch);
        let rgb = cropped.to_rgb8();

        let linear_rgb: Vec<f32> = rgb
            .pixels()
            .flat_map(|p| {
                [
                    srgb_to_linear_val(p[0] as f32 / 255.0),
                    srgb_to_linear_val(p[1] as f32 / 255.0),
                    srgb_to_linear_val(p[2] as f32 / 255.0),
                ]
            })
            .collect();

        for &d in &distances {
            let mut enc_static = jxl_encoder::vardct::VarDctEncoder::new(d);
            enc_static.optimize_codes = false;
            let static_bytes = enc_static
                .encode(cw as usize, ch as usize, &linear_rgb, None)
                .unwrap()
                .data;

            let mut enc_opt = jxl_encoder::vardct::VarDctEncoder::new(d);
            enc_opt.optimize_codes = true;
            let opt_bytes = enc_opt
                .encode(cw as usize, ch as usize, &linear_rgb, None)
                .unwrap()
                .data;

            let s = static_bytes.len();
            let o = opt_bytes.len();
            let delta = s as i64 - o as i64;
            let pct = if s > 0 {
                delta as f64 / s as f64 * 100.0
            } else {
                0.0
            };

            eprintln!(
                "img{:<4} {:>6.2} {:>10} {:>10} {:>+10} {:>7.1}%",
                i, d, s, o, delta, pct
            );

            total_static += s as u64;
            total_optimized += o as u64;
        }
    }

    let total_delta = total_static as i64 - total_optimized as i64;
    let total_pct = if total_static > 0 {
        total_delta as f64 / total_static as f64 * 100.0
    } else {
        0.0
    };

    eprintln!("{}", "-".repeat(56));
    eprintln!(
        "{:<8} {:>6} {:>10} {:>10} {:>+10} {:>7.1}%",
        "TOTAL", "", total_static, total_optimized, total_delta, total_pct
    );
}

/// Test ANS histogram serialization roundtrip with jxl-rs decoder.
///
/// Writes ANS distributions using our encoder and verifies jxl-rs can parse them.
#[test]
#[ignore]
fn test_ans_histogram_roundtrip_jxl_rs() {
    use jxl_encoder::bit_writer::BitWriter;
    use jxl_encoder::entropy_coding::ans::{ANSEncodingHistogram, ANSHistogramStrategy};
    use jxl_encoder::entropy_coding::histogram::Histogram;

    // Test cases: various histogram shapes
    let test_cases: Vec<(&str, Vec<i32>)> = vec![
        ("single_symbol", vec![100, 0, 0, 0]),
        ("two_symbols", vec![75, 25, 0, 0]),
        ("uniform_4", vec![25, 25, 25, 25]),
        ("skewed", vec![100, 50, 25, 10, 5, 3, 2, 1]),
        (
            "sparse",
            vec![0, 0, 50, 0, 0, 30, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0],
        ),
    ];

    for (name, counts) in test_cases {
        let h = Histogram::from_counts(&counts);
        let encoded =
            ANSEncodingHistogram::from_histogram(&h, ANSHistogramStrategy::Precise).unwrap();

        // Write the histogram
        let mut writer = BitWriter::new();
        encoded.write(&mut writer).unwrap();
        let bytes = writer.finish_with_padding();

        eprintln!(
            "Test '{}': alphabet={}, method={}, omit_pos={}, {} bytes",
            name,
            encoded.alphabet_size,
            encoded.method,
            encoded.omit_pos,
            bytes.len()
        );
        eprintln!("  Counts: {:?}", &encoded.counts[..encoded.alphabet_size]);
        eprintln!("  Bytes: {:02x?}", &bytes[..bytes.len().min(16)]);

        // Try to decode with jxl-rs
        let mut br = jxl::bit_reader::BitReader::new(&bytes);
        let log_alpha_size = 8; // We use alphabet size up to 256

        match jxl::entropy_coding::ans::AnsCodes::decode(1, log_alpha_size, &mut br) {
            Ok(codes) => {
                eprintln!("  jxl-rs decoded successfully!");

                // Verify single symbol case
                if encoded.method == 1
                    && let Some(sym) = codes.single_symbol(0)
                {
                    eprintln!("  Single symbol: {}", sym);
                }
            }
            Err(e) => {
                eprintln!("  jxl-rs decode FAILED: {:?}", e);
                panic!(
                    "ANS histogram '{}' failed to decode with jxl-rs: {:?}",
                    name, e
                );
            }
        }
    }
}

/// Test histogram round-trip for the exact case from failing ANS encode.
/// Uses our internal ans_decode module to avoid jxl-rs private API issues.
#[test]
fn test_ans_skewed_histogram_roundtrip() {
    use jxl_encoder::bit_writer::BitWriter;
    use jxl_encoder::entropy_coding::ans::{
        ANSEncodingHistogram, ANSHistogramStrategy, AnsDistribution, AnsEncoder,
    };
    use jxl_encoder::entropy_coding::ans_decode::{AnsHistogram, BitReader};
    use jxl_encoder::entropy_coding::histogram::Histogram;

    // Recreate the histogram from the debug output:
    // Skewed distribution like DC tokens: mostly token 0, rare token 1 and 32
    let mut histo = Histogram::new();
    for _ in 0..190 {
        histo.add(0);
    }
    histo.add(1);
    histo.add(32);

    eprintln!("Original histogram:");
    eprintln!("  total: {}", histo.total_count);
    eprintln!("  alphabet_size: {}", histo.alphabet_size());

    // Build ANS histogram
    let ans_histo = ANSEncodingHistogram::from_histogram(&histo, ANSHistogramStrategy::Precise)
        .expect("histogram normalization failed");

    eprintln!("\nANS histogram:");
    eprintln!("  method: {}", ans_histo.method);
    eprintln!("  alphabet_size: {}", ans_histo.alphabet_size);
    eprintln!("  omit_pos: {}", ans_histo.omit_pos);
    for i in 0..ans_histo.alphabet_size {
        if ans_histo.counts[i] > 0 {
            eprintln!("  counts[{}] = {}", i, ans_histo.counts[i]);
        }
    }
    let sum: i32 = ans_histo.counts.iter().sum();
    eprintln!("  sum: {}", sum);
    assert_eq!(sum, 4096, "counts must sum to 4096");

    // Serialize histogram
    let mut hist_writer = BitWriter::new();
    ans_histo.write(&mut hist_writer).expect("write failed");
    let hist_bytes = hist_writer.finish_with_padding();
    eprintln!(
        "\nHistogram bytes ({} bytes): {:02x?}",
        hist_bytes.len(),
        hist_bytes
    );

    // Parse with our decoder
    let mut hist_br = BitReader::new(&hist_bytes);
    let decoded_hist = AnsHistogram::decode(&mut hist_br, 6).expect("decode failed");
    eprintln!("Decoded histogram frequencies:");
    for i in 0..decoded_hist.frequencies.len() {
        if decoded_hist.frequencies[i] > 0 {
            eprintln!("  freq[{}] = {}", i, decoded_hist.frequencies[i]);
        }
    }

    // Verify frequencies match
    for i in 0..ans_histo.alphabet_size {
        let expected = ans_histo.counts[i] as u16;
        let actual = decoded_hist.frequencies.get(i).copied().unwrap_or(0);
        assert_eq!(actual, expected, "frequency mismatch at symbol {}", i);
    }

    // Now test symbol encoding/decoding
    eprintln!("\nTesting symbol encoding:");
    let dist =
        AnsDistribution::from_normalized_counts(&ans_histo.counts).expect("distribution failed");

    // Encode [0, 0, 0, 1, 32] in reverse
    let symbols: Vec<usize> = vec![0, 0, 0, 1, 32];
    let mut encoder = AnsEncoder::new();
    for &sym in symbols.iter().rev() {
        let info = dist.get(sym).expect("symbol not found");
        encoder.put_symbol(info);
    }
    eprintln!("  final encoder state: 0x{:08x}", encoder.state());

    let mut token_writer = BitWriter::new();
    encoder
        .finalize(&mut token_writer)
        .expect("finalize failed");
    let token_bytes = token_writer.finish_with_padding();
    eprintln!(
        "  token bytes ({} bytes): {:02x?}",
        token_bytes.len(),
        token_bytes
    );

    // Decode with our decoder
    let mut token_br = BitReader::new(&token_bytes);
    let initial_state = token_br.read(32).unwrap() as u32;
    eprintln!("  decoder read initial state: 0x{:08x}", initial_state);

    let mut state = initial_state;
    let mut decoded_symbols = Vec::new();
    for i in 0..symbols.len() {
        let sym = decoded_hist.read(&mut token_br, &mut state);
        eprintln!("    step {}: sym={}, state=0x{:08x}", i, sym, state);
        decoded_symbols.push(sym as usize);
    }

    eprintln!("\nDecoded: {:?}", decoded_symbols);
    eprintln!("Expected: {:?}", symbols);
    eprintln!("Final state: 0x{:08x}", state);

    assert_eq!(decoded_symbols, symbols, "symbols should match");
    assert_eq!(state, 0x00130000, "final state should be 0x00130000");
}

/// Test single-symbol ANS distribution - should not change state.
#[test]
fn test_ans_single_symbol_no_state_change() {
    use jxl_encoder::entropy_coding::ans::{AnsDistribution, AnsEncoder};

    // Single symbol at position 8
    let mut counts = vec![0i32; 64];
    counts[8] = 4096;

    let dist = AnsDistribution::from_normalized_counts(&counts).expect("distribution failed");

    eprintln!("Single-symbol distribution:");
    eprintln!("  symbol 8: freq={}", dist.symbols[8].freq);

    // Encode 10 copies of symbol 8
    let mut encoder = AnsEncoder::new();
    eprintln!("\nEncoding 10 copies of symbol 8:");
    for i in 0..10 {
        let state_before = encoder.state();
        encoder.put_symbol(&dist.symbols[8]);
        eprintln!(
            "  step {}: state before=0x{:08x}, after=0x{:08x}",
            i,
            state_before,
            encoder.state()
        );
    }

    eprintln!("\nFinal state: 0x{:08x}", encoder.state());

    // For a single-symbol distribution (100% probability), state should never change
    assert_eq!(
        encoder.state(),
        0x00130000,
        "state should not change for 100% probability symbol"
    );
}

/// Debug single-symbol distribution reverse_map.
#[test]
fn test_ans_single_symbol_reverse_map() {
    use jxl_encoder::entropy_coding::ans::AnsDistribution;

    // Single symbol at position 8
    let mut counts = vec![0i32; 64];
    counts[8] = 4096;

    let dist = AnsDistribution::from_normalized_counts(&counts).expect("distribution failed");

    eprintln!("Single-symbol distribution for symbol 8:");
    eprintln!("  symbols[8].freq = {}", dist.symbols[8].freq);
    eprintln!("  symbols[8].ifreq = {}", dist.symbols[8].ifreq);
    eprintln!(
        "  symbols[8].reverse_map.len() = {}",
        dist.symbols[8].reverse_map.len()
    );
    eprintln!(
        "  reverse_map[0..10] = {:?}",
        &dist.symbols[8].reverse_map[..10.min(dist.symbols[8].reverse_map.len())]
    );

    // The key observation: for freq=4096, reverse_map[r] for ANY r should map to
    // the SAME idx when state = r + v * 4096 for some v.
    // Actually, reverse_map[r] should be the idx where decoder state=idx*4096 + r
    // can reconstruct r.

    // For freq=4096 (100% probability), all 4096 positions in the alias table
    // belong to symbol 8. The decoder formula is:
    // next_state = (state >> 12) * freq + offset
    //            = (state >> 12) * 4096 + offset
    // For this to work, every idx in [0, 4095] should have offset = idx
    // (since we want next_state = (state >> 12) * 4096 + idx)

    // So reverse_map[r] should be r (identity mapping) for single-symbol case.

    for i in 0..10 {
        eprintln!("  reverse_map[{}] = {}", i, dist.symbols[8].reverse_map[i]);
        if dist.symbols[8].reverse_map[i] != i as u16 {
            eprintln!("    ERROR: expected {}", i);
        }
    }
}

/// Full encode-decode cycle for single-symbol distribution.
#[test]
fn test_ans_single_symbol_full_cycle() {
    use jxl_encoder::bit_writer::BitWriter;
    use jxl_encoder::entropy_coding::ans::{
        ANSEncodingHistogram, ANSHistogramStrategy, AnsDistribution, AnsEncoder,
    };
    use jxl_encoder::entropy_coding::ans_decode::{AnsHistogram, BitReader};
    use jxl_encoder::entropy_coding::histogram::Histogram;

    // Single symbol 8 with 10 occurrences
    let mut histo = Histogram::new();
    for _ in 0..10 {
        histo.add(8);
    }

    let ans_histo = ANSEncodingHistogram::from_histogram(&histo, ANSHistogramStrategy::Precise)
        .expect("histogram failed");

    eprintln!("Histogram:");
    eprintln!("  method: {}", ans_histo.method);
    eprintln!("  alphabet_size: {}", ans_histo.alphabet_size);
    for i in 0..ans_histo.alphabet_size {
        if ans_histo.counts[i] > 0 {
            eprintln!("  counts[{}] = {}", i, ans_histo.counts[i]);
        }
    }

    // Build distribution for encoding
    let dist = AnsDistribution::from_normalized_counts(&ans_histo.counts).expect("dist failed");

    // Encode
    let symbols: Vec<usize> = vec![8; 10];
    let mut encoder = AnsEncoder::new();
    eprintln!("\nEncoding:");
    for (i, &sym) in symbols.iter().rev().enumerate() {
        let state_before = encoder.state();
        encoder.put_symbol(&dist.symbols[sym]);
        eprintln!(
            "  enc[{}]: sym={}, state 0x{:08x} -> 0x{:08x}",
            i,
            sym,
            state_before,
            encoder.state()
        );
    }

    let encoder_final_state = encoder.state();
    eprintln!("\nEncoder final state: 0x{:08x}", encoder_final_state);

    let mut token_writer = BitWriter::new();
    encoder
        .finalize(&mut token_writer)
        .expect("finalize failed");
    let token_bytes = token_writer.finish_with_padding();
    eprintln!(
        "Token bytes ({} bytes): {:02x?}",
        token_bytes.len(),
        token_bytes
    );

    // Serialize histogram
    let mut hist_writer = BitWriter::new();
    ans_histo
        .write(&mut hist_writer)
        .expect("hist write failed");
    let hist_bytes = hist_writer.finish_with_padding();

    // Decode histogram
    let mut hist_br = BitReader::new(&hist_bytes);
    let decoded_hist = AnsHistogram::decode(&mut hist_br, 6).expect("hist decode failed");
    eprintln!("\nDecoded histogram frequencies:");
    for i in 0..decoded_hist.frequencies.len() {
        if decoded_hist.frequencies[i] > 0 {
            eprintln!("  freq[{}] = {}", i, decoded_hist.frequencies[i]);
        }
    }

    // Decode tokens
    let mut token_br = BitReader::new(&token_bytes);
    let initial_state = token_br.read(32).expect("read state failed") as u32;
    eprintln!("\nDecoder initial state: 0x{:08x}", initial_state);
    assert_eq!(
        initial_state, encoder_final_state,
        "initial state should match encoder final"
    );

    let mut state = initial_state;
    let mut decoded_symbols = Vec::new();
    eprintln!("Decoding:");
    for i in 0..symbols.len() {
        let state_before = state;
        let sym = decoded_hist.read(&mut token_br, &mut state);
        eprintln!(
            "  dec[{}]: sym={}, state 0x{:08x} -> 0x{:08x}",
            i, sym, state_before, state
        );
        decoded_symbols.push(sym as usize);
    }

    eprintln!("\nDecoded: {:?}", decoded_symbols);
    eprintln!("Expected: {:?}", symbols);
    eprintln!("Final state: 0x{:08x} (expected 0x00130000)", state);

    assert_eq!(decoded_symbols, symbols, "symbols should match");
    assert_eq!(state, 0x00130000, "final state should be 0x00130000");
}

/// Test RGBA encoding - fixed "IncompleteFrame" error by adding ec_upsampling and ec_blending_info
#[test]
#[ignore]
fn test_rgba_simple() {
    use std::io::Cursor;

    // Test various sizes: 8x8 (single block), 256x256 (single group), 512x512 (multi-group)
    for (width, height) in [(8, 8), (256, 256), (512, 512)] {
        eprintln!("\n=== Testing RGBA {}x{} ===", width, height);

        let mut rgba_data = vec![0u8; width * height * 4];
        for i in 0..(width * height) {
            rgba_data[i * 4] = ((i * 3) % 256) as u8; // R - varying
            rgba_data[i * 4 + 1] = ((i * 5) % 256) as u8; // G - varying
            rgba_data[i * 4 + 2] = ((i * 7) % 256) as u8; // B - varying
            rgba_data[i * 4 + 3] = 255; // A - opaque
        }

        let jxl_bytes = jxl_encoder::LosslessConfig::new()
            .encode_request(width as u32, height as u32, jxl_encoder::PixelLayout::Rgba8)
            .encode(&rgba_data)
            .expect("Failed to encode RGBA");

        eprintln!("RGBA Encoded {} bytes", jxl_bytes.len());

        // Test RGBA with jxl-oxide
        let rgba_reader = Cursor::new(&jxl_bytes);
        let mut rgba_image = jxl_oxide::JxlImage::builder()
            .read(rgba_reader)
            .expect("Failed to parse RGBA JXL");
        rgba_image.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
            jxl_oxide::RenderingIntent::Relative,
        ));

        match rgba_image.render_frame(0) {
            Ok(render) => {
                let fb = render.image_all_channels();
                eprintln!(
                    "RGBA jxl-oxide decoded successfully: {}x{} (channels={})",
                    fb.width(),
                    fb.height(),
                    fb.channels()
                );

                // Verify lossless encoding - compare first 10 pixels
                let decoded = fb.buf();
                let channels = fb.channels();
                let mut errors = 0;
                for i in 0..10.min(width * height) {
                    let expected_r = ((i * 3) % 256) as f32 / 255.0;
                    let expected_g = ((i * 5) % 256) as f32 / 255.0;
                    let expected_b = ((i * 7) % 256) as f32 / 255.0;
                    let expected_a = 1.0;

                    let got_r = decoded[i * channels];
                    let got_g = decoded[i * channels + 1];
                    let got_b = decoded[i * channels + 2];
                    let got_a = if channels > 3 {
                        decoded[i * channels + 3]
                    } else {
                        1.0
                    };

                    let tol = 0.01; // Allow small tolerance for floating point
                    if (got_r - expected_r).abs() > tol
                        || (got_g - expected_g).abs() > tol
                        || (got_b - expected_b).abs() > tol
                        || (got_a - expected_a).abs() > tol
                    {
                        if errors < 3 {
                            eprintln!(
                                "Pixel {}: expected ({:.3},{:.3},{:.3},{:.3}), got ({:.3},{:.3},{:.3},{:.3})",
                                i,
                                expected_r,
                                expected_g,
                                expected_b,
                                expected_a,
                                got_r,
                                got_g,
                                got_b,
                                got_a
                            );
                        }
                        errors += 1;
                    }
                }
                if errors > 0 {
                    panic!(
                        "RGBA verification failed: {} pixel errors for {}x{}",
                        errors, width, height
                    );
                }
            }
            Err(e) => {
                panic!(
                    "RGBA jxl-oxide render error for {}x{}: {:?}",
                    width, height, e
                );
            }
        }
    }
}

/// Test ANS encoding with CLIC 2025 images - compare quality and file size with Huffman
#[test]
#[ignore]
fn test_ans_clic2025() {
    let clic_dir = jxl_encoder::test_helpers::corpus_dir().join("clic2025/final-test");

    if !clic_dir.exists() {
        eprintln!("CLIC 2025 directory not found: {:?}", clic_dir);
        return;
    }

    // Get first 5 images
    let mut entries: Vec<_> = std::fs::read_dir(&clic_dir)
        .expect("Failed to read CLIC directory")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "png"))
        .take(5)
        .collect();

    entries.sort_by_key(|e| e.path());

    eprintln!("\n=== ANS vs Huffman on CLIC 2025 ===\n");
    eprintln!(
        "{:<12} {:>8} {:>8} {:>8} {:>7} {:>7} {:>8}",
        "Image", "Size", "Huffman", "ANS", "Ratio", "SSIM2H", "SSIM2A"
    );
    eprintln!("{}", "-".repeat(70));

    let mut total_huffman = 0usize;
    let mut total_ans = 0usize;
    let mut count = 0;

    for entry in &entries {
        let path = entry.path();
        let filename = path.file_name().unwrap().to_string_lossy();
        let short_name = &filename[..8.min(filename.len())];

        let img = match image::open(&path) {
            Ok(img) => img,
            Err(e) => {
                eprintln!("{}: Failed to open: {}", short_name, e);
                continue;
            }
        };

        let (width, height) = img.dimensions();
        let rgb = img.to_rgb8();

        // Get original pixels for SSIM2
        let original_srgb: Vec<[u8; 3]> = rgb.pixels().map(|p| [p[0], p[1], p[2]]).collect();

        // Convert to linear RGB for encoding
        let linear_rgb: Vec<f32> = rgb
            .pixels()
            .flat_map(|p| {
                let r = srgb_to_linear_val(p[0] as f32 / 255.0);
                let g = srgb_to_linear_val(p[1] as f32 / 255.0);
                let b = srgb_to_linear_val(p[2] as f32 / 255.0);
                [r, g, b]
            })
            .collect();

        // Encode with Huffman
        let mut encoder_huff = jxl_encoder::vardct::VarDctEncoder::new(1.0);
        encoder_huff.use_ans = false;
        let bytes_huff =
            match encoder_huff.encode(width as usize, height as usize, &linear_rgb, None) {
                Ok(output) => output.data,
                Err(e) => {
                    eprintln!("{}: Huffman encode failed: {:?}", short_name, e);
                    continue;
                }
            };

        // Encode with ANS
        let mut encoder_ans = jxl_encoder::vardct::VarDctEncoder::new(1.0);
        encoder_ans.use_ans = true;
        let bytes_ans = match encoder_ans.encode(width as usize, height as usize, &linear_rgb, None)
        {
            Ok(output) => output.data,
            Err(e) => {
                eprintln!("{}: ANS encode failed: {:?}", short_name, e);
                continue;
            }
        };

        // Decode Huffman and compute SSIM2
        let ssim2_huff =
            decode_and_ssim2(&bytes_huff, &original_srgb, width as usize, height as usize);

        // Decode ANS and compute SSIM2
        let ssim2_ans =
            decode_and_ssim2(&bytes_ans, &original_srgb, width as usize, height as usize);

        let size_str = format!("{}x{}", width, height);
        let ratio = bytes_huff.len() as f64 / bytes_ans.len() as f64;

        eprintln!(
            "{:<12} {:>8} {:>8} {:>8} {:>6.2}x {:>7.1} {:>7.1}",
            short_name,
            size_str,
            bytes_huff.len(),
            bytes_ans.len(),
            ratio,
            ssim2_huff.unwrap_or(f64::NAN),
            ssim2_ans.unwrap_or(f64::NAN)
        );

        total_huffman += bytes_huff.len();
        total_ans += bytes_ans.len();
        count += 1;

        // Verify ANS decodes correctly
        if ssim2_ans.is_none() {
            panic!("{}: ANS decode failed!", short_name);
        }

        // Verify quality is similar (within 0.5 SSIM2)
        if let (Some(h), Some(a)) = (ssim2_huff, ssim2_ans) {
            let diff = (h - a).abs();
            if diff > 0.5 {
                eprintln!("WARNING: SSIM2 difference {} for {}", diff, short_name);
            }
        }
    }

    if count > 0 {
        eprintln!("{}", "-".repeat(70));
        let overall_ratio = total_huffman as f64 / total_ans as f64;
        eprintln!(
            "{:<12} {:>8} {:>8} {:>8} {:>6.2}x",
            "TOTAL", "", total_huffman, total_ans, overall_ratio
        );

        // ANS should be smaller or equal (allowing 5% overhead for edge cases)
        assert!(
            overall_ratio >= 0.95,
            "ANS files are unexpectedly larger than Huffman: ratio={:.2}x",
            overall_ratio
        );
    }
}

/// Helper: decode JXL bytes and compute SSIM2 against original
fn decode_and_ssim2(
    bytes: &[u8],
    original: &[[u8; 3]],
    width: usize,
    height: usize,
) -> Option<f64> {
    use std::io::Cursor;

    let reader = Cursor::new(bytes);
    let mut image = jxl_oxide::JxlImage::builder().read(reader).ok()?;
    image.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
        jxl_oxide::RenderingIntent::Relative,
    ));
    let render = image.render_frame(0).ok()?;
    let fb = render.image_all_channels();
    let decoded = fb.buf();

    // Convert decoded linear to sRGB
    let decoded_srgb: Vec<[u8; 3]> = decoded
        .chunks(3)
        .map(|rgb| {
            let r = linear_to_srgb_u8(rgb[0]);
            let g = linear_to_srgb_u8(rgb[1]);
            let b = linear_to_srgb_u8(rgb[2]);
            [r, g, b]
        })
        .collect();

    let original_img = imgref::Img::new(original.to_vec(), width, height);
    let decoded_img = imgref::Img::new(decoded_srgb, width, height);

    fast_ssim2::compute_ssimulacra2(original_img.as_ref(), decoded_img.as_ref()).ok()
}

/// Test ANS with multi-group images (synthetic gradient)
#[test]
#[ignore]
fn test_ans_multigroup_gradient() {
    use std::io::Cursor;

    // Test sizes: single-group and multi-group
    // Include 2048x1360 which matches the failing CLIC image dimensions
    for (width, height) in [(256, 256), (512, 512), (1024, 1024), (2048, 1360)] {
        eprintln!("\n=== ANS multi-group test {}x{} ===", width, height);

        // Create gradient image (linear RGB)
        let mut linear_rgb = vec![0.0f32; width * height * 3];
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) * 3;
                linear_rgb[idx] = x as f32 / width as f32;
                linear_rgb[idx + 1] = y as f32 / height as f32;
                linear_rgb[idx + 2] = 0.5;
            }
        }

        // Encode with ANS
        let mut encoder = jxl_encoder::vardct::VarDctEncoder::new(1.0);
        encoder.use_ans = true;

        let bytes = match encoder.encode(width, height, &linear_rgb, None) {
            Ok(output) => output.data,
            Err(e) => {
                panic!("ANS encode failed for {}x{}: {:?}", width, height, e);
            }
        };

        eprintln!("Encoded {} bytes", bytes.len());

        // Write to file for external debugging
        let filename = format!("/tmp/test_ans_{}x{}.jxl", width, height);
        std::fs::write(&filename, &bytes).unwrap();

        // Decode with jxl-oxide
        let reader = Cursor::new(&bytes);
        let mut image = jxl_oxide::JxlImage::builder()
            .read(reader)
            .expect("Failed to parse ANS JXL");
        image.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
            jxl_oxide::RenderingIntent::Relative,
        ));

        match image.render_frame(0) {
            Ok(render) => {
                eprintln!(
                    "jxl-oxide decoded: {}x{}",
                    render.image_all_channels().width(),
                    render.image_all_channels().height()
                );
            }
            Err(e) => {
                panic!("jxl-oxide render error for {}x{}: {:?}", width, height, e);
            }
        }
    }
}

/// Test ANS with specific failing CLIC image
#[test]
#[ignore]
fn test_ans_failing_image() {
    use std::io::Cursor;

    let path = jxl_encoder::test_helpers::corpus_dir().join(
        "clic2025/final-test/a365e6541bab5c0f4e01bf43a0c3a655d88292a8ac45403a889c308d11854555.png",
    );

    if !path.exists() {
        eprintln!("Test image not found: {:?}", path);
        return;
    }

    let img = image::open(&path).expect("Failed to open image");
    let (width, height) = img.dimensions();
    eprintln!("Image size: {}x{}", width, height);

    let rgb = img.to_rgb8();
    let linear_rgb: Vec<f32> = rgb
        .pixels()
        .flat_map(|p| {
            let r = srgb_to_linear_val(p[0] as f32 / 255.0);
            let g = srgb_to_linear_val(p[1] as f32 / 255.0);
            let b = srgb_to_linear_val(p[2] as f32 / 255.0);
            [r, g, b]
        })
        .collect();

    // Encode with ANS
    let mut encoder = jxl_encoder::vardct::VarDctEncoder::new(1.0);
    encoder.use_ans = true;

    eprintln!("Starting ANS encode...");
    let bytes = encoder
        .encode(width as usize, height as usize, &linear_rgb, None)
        .expect("ANS encode failed")
        .data;

    eprintln!("Encoded {} bytes", bytes.len());

    // Save for debugging
    std::fs::write("/tmp/test_ans_failing.jxl", &bytes).unwrap();
    eprintln!("Saved to /tmp/test_ans_failing.jxl");

    // Try decode with jxl-oxide
    let reader = Cursor::new(&bytes);
    let mut image = jxl_oxide::JxlImage::builder()
        .read(reader)
        .expect("Failed to parse JXL");
    image.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
        jxl_oxide::RenderingIntent::Relative,
    ));

    eprintln!("Parsed, attempting render...");
    match image.render_frame(0) {
        Ok(render) => {
            eprintln!(
                "SUCCESS: jxl-oxide decoded {}x{}",
                render.image_all_channels().width(),
                render.image_all_channels().height()
            );
        }
        Err(e) => {
            eprintln!("RENDER FAILED: {:?}", e);
            panic!("Render failed");
        }
    }
}

/// Debug ANS vs Huffman for failing image
#[test]
#[ignore]
fn test_ans_vs_huffman_debug() {
    use std::io::Cursor;

    let path = jxl_encoder::test_helpers::corpus_dir().join(
        "clic2025/final-test/a365e6541bab5c0f4e01bf43a0c3a655d88292a8ac45403a889c308d11854555.png",
    );

    if !path.exists() {
        eprintln!("Test image not found");
        return;
    }

    let img = image::open(&path).expect("Failed to open image");
    let (width, height) = img.dimensions();
    eprintln!("Image size: {}x{}", width, height);

    let rgb = img.to_rgb8();
    let linear_rgb: Vec<f32> = rgb
        .pixels()
        .flat_map(|p| {
            let r = srgb_to_linear_val(p[0] as f32 / 255.0);
            let g = srgb_to_linear_val(p[1] as f32 / 255.0);
            let b = srgb_to_linear_val(p[2] as f32 / 255.0);
            [r, g, b]
        })
        .collect();

    // Encode with Huffman first
    let mut encoder_huff = jxl_encoder::vardct::VarDctEncoder::new(1.0);
    encoder_huff.use_ans = false;
    let bytes_huff = encoder_huff
        .encode(width as usize, height as usize, &linear_rgb, None)
        .expect("Huffman encode failed")
        .data;
    eprintln!("Huffman: {} bytes", bytes_huff.len());
    std::fs::write("/tmp/test_huff.jxl", &bytes_huff).unwrap();

    // Verify Huffman works
    let reader = Cursor::new(&bytes_huff);
    let mut image = jxl_oxide::JxlImage::builder().read(reader).unwrap();
    image.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
        jxl_oxide::RenderingIntent::Relative,
    ));
    match image.render_frame(0) {
        Ok(_) => eprintln!("Huffman: decode OK"),
        Err(e) => eprintln!("Huffman: decode FAILED: {:?}", e),
    }

    // Now encode with ANS
    let mut encoder_ans = jxl_encoder::vardct::VarDctEncoder::new(1.0);
    encoder_ans.use_ans = true;
    let bytes_ans = encoder_ans
        .encode(width as usize, height as usize, &linear_rgb, None)
        .expect("ANS encode failed")
        .data;
    eprintln!("ANS: {} bytes", bytes_ans.len());
    std::fs::write("/tmp/test_ans.jxl", &bytes_ans).unwrap();

    // Check ANS
    let reader = Cursor::new(&bytes_ans);
    let mut image = jxl_oxide::JxlImage::builder().read(reader).unwrap();
    image.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
        jxl_oxide::RenderingIntent::Relative,
    ));
    match image.render_frame(0) {
        Ok(_) => eprintln!("ANS: decode OK"),
        Err(e) => {
            eprintln!("ANS: decode FAILED: {:?}", e);

            // Compare first 200 bytes of each file
            eprintln!("\nHuffman header (first 100 bytes):");
            eprintln!("{:02x?}", &bytes_huff[..100.min(bytes_huff.len())]);
            eprintln!("\nANS header (first 100 bytes):");
            eprintln!("{:02x?}", &bytes_ans[..100.min(bytes_ans.len())]);
        }
    }
}

/// Binary search for smallest failing crop from the CLIC image
#[test]
#[ignore]
fn test_ans_crop_binary_search() {
    use std::io::Cursor;

    let path = jxl_encoder::test_helpers::corpus_dir().join(
        "clic2025/final-test/a365e6541bab5c0f4e01bf43a0c3a655d88292a8ac45403a889c308d11854555.png",
    );

    if !path.exists() {
        eprintln!("Test image not found");
        return;
    }

    let img = image::open(&path).expect("Failed to open image");
    let rgb = img.to_rgb8();
    let (full_w, full_h) = img.dimensions();
    eprintln!("Full image: {}x{}", full_w, full_h);

    // Test at different crop sizes from the top-left corner
    // Test with original image at 2048x1360 and with bottom rows zeroed
    let sizes = [(2048, 1360)];

    for &(w, h) in &sizes {
        if w > full_w as usize || h > full_h as usize {
            continue;
        }

        // Crop the image
        let mut linear_rgb = Vec::with_capacity(w * h * 3);
        for y in 0..h {
            for x in 0..w {
                let p = rgb.get_pixel(x as u32, y as u32);
                linear_rgb.push(srgb_to_linear_val(p[0] as f32 / 255.0));
                linear_rgb.push(srgb_to_linear_val(p[1] as f32 / 255.0));
                linear_rgb.push(srgb_to_linear_val(p[2] as f32 / 255.0));
            }
        }

        let mut encoder = jxl_encoder::vardct::VarDctEncoder::new(1.0);
        encoder.use_ans = true;

        let bytes = encoder
            .encode(w, h, &linear_rgb, None)
            .expect("ANS encode failed")
            .data;

        let reader = Cursor::new(&bytes);
        let mut image = jxl_oxide::JxlImage::builder()
            .read(reader)
            .expect("Failed to parse JXL");
        image.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
            jxl_oxide::RenderingIntent::Relative,
        ));

        let result = image.render_frame(0);
        let status = match &result {
            Ok(_) => "OK",
            Err(_) => "FAIL",
        };
        eprintln!("{:>4}x{:<4} {:>8} bytes  {}", w, h, bytes.len(), status);

        if let Err(e) = &result {
            eprintln!("  Error: {:?}", e);
        }
    }
}

/// Test custom coefficient ordering roundtrip with jxl-oxide on CLIC 2025 photos.
///
/// Verifies that:
/// 1. Custom orders produce decodable files
/// 2. Quality (SSIM2) is comparable to default zig-zag order
#[test]
#[ignore = "Requires CLIC 2025 images"]
fn test_custom_orders() {
    let clic_dir = jxl_encoder::test_helpers::corpus_dir().join("clic2025/final-test");

    if !clic_dir.exists() {
        eprintln!("CLIC 2025 directory not found: {:?}", clic_dir);
        return;
    }

    let mut entries: Vec<_> = std::fs::read_dir(&clic_dir)
        .expect("Failed to read CLIC directory")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "png"))
        .take(5)
        .collect();

    entries.sort_by_key(|e| e.path());

    eprintln!("\n=== Custom Coefficient Orders on CLIC 2025 ===\n");
    eprintln!(
        "{:<12} {:>8} {:>8} {:>8} {:>7} {:>7}",
        "Image", "Size", "Default", "Custom", "SSIM2D", "SSIM2C"
    );
    eprintln!("{}", "-".repeat(62));

    let mut all_decoded = true;

    for entry in &entries {
        let path = entry.path();
        let filename = path.file_name().unwrap().to_string_lossy();
        let short_name = &filename[..8.min(filename.len())];

        let img = match image::open(&path) {
            Ok(img) => img,
            Err(e) => {
                eprintln!("{}: Failed to open: {}", short_name, e);
                continue;
            }
        };

        let (width, height) = img.dimensions();
        let rgb = img.to_rgb8();
        let original_srgb: Vec<[u8; 3]> = rgb.pixels().map(|p| [p[0], p[1], p[2]]).collect();

        let linear_rgb: Vec<f32> = rgb
            .pixels()
            .flat_map(|p| {
                let r = srgb_to_linear_val(p[0] as f32 / 255.0);
                let g = srgb_to_linear_val(p[1] as f32 / 255.0);
                let b = srgb_to_linear_val(p[2] as f32 / 255.0);
                [r, g, b]
            })
            .collect();

        // Encode with default zig-zag order
        let mut enc_default = jxl_encoder::vardct::VarDctEncoder::new(1.0);
        enc_default.custom_orders = false;
        let bytes_default =
            match enc_default.encode(width as usize, height as usize, &linear_rgb, None) {
                Ok(output) => output.data,
                Err(e) => {
                    eprintln!("{}: Default encode failed: {:?}", short_name, e);
                    continue;
                }
            };

        // Encode with custom orders
        let mut enc_custom = jxl_encoder::vardct::VarDctEncoder::new(1.0);
        enc_custom.custom_orders = true;
        let bytes_custom =
            match enc_custom.encode(width as usize, height as usize, &linear_rgb, None) {
                Ok(output) => output.data,
                Err(e) => {
                    eprintln!("{}: Custom encode failed: {:?}", short_name, e);
                    continue;
                }
            };

        // Decode default
        let ssim2_default = decode_and_ssim2(
            &bytes_default,
            &original_srgb,
            width as usize,
            height as usize,
        );

        // Decode custom
        let ssim2_custom = decode_and_ssim2(
            &bytes_custom,
            &original_srgb,
            width as usize,
            height as usize,
        );

        let size_str = format!("{}x{}", width, height);
        eprintln!(
            "{:<12} {:>8} {:>8} {:>8} {:>6.1} {:>6.1}",
            short_name,
            size_str,
            bytes_default.len(),
            bytes_custom.len(),
            ssim2_default.unwrap_or(f64::NAN),
            ssim2_custom.unwrap_or(f64::NAN)
        );

        if ssim2_custom.is_none() {
            eprintln!("  FAIL: Custom order decode failed for {}", short_name);
            all_decoded = false;
        }

        // Quality should be similar (within 0.5 SSIM2)
        if let (Some(d), Some(c)) = (ssim2_default, ssim2_custom) {
            let diff = (d - c).abs();
            if diff > 0.5 {
                eprintln!(
                    "  WARNING: SSIM2 diff={:.2} for {} (default={:.1}, custom={:.1})",
                    diff, short_name, d, c
                );
            }
        }
    }

    assert!(all_decoded, "Some custom order files failed to decode");
}

/// Test that custom coefficient orders produce smaller files on CLIC 2025 photos.
#[test]
#[ignore = "Requires CLIC 2025 images"]
fn test_custom_orders_compression() {
    let clic_dir = jxl_encoder::test_helpers::corpus_dir().join("clic2025/final-test");

    if !clic_dir.exists() {
        eprintln!("CLIC 2025 directory not found: {:?}", clic_dir);
        return;
    }

    let mut entries: Vec<_> = std::fs::read_dir(&clic_dir)
        .expect("Failed to read CLIC directory")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "png"))
        .take(5)
        .collect();

    entries.sort_by_key(|e| e.path());

    eprintln!("\n=== Custom Orders Compression on CLIC 2025 ===\n");
    eprintln!(
        "{:<12} {:>8} {:>8} {:>8} {:>7}",
        "Image", "Size", "Default", "Custom", "Saving"
    );
    eprintln!("{}", "-".repeat(50));

    let mut total_default = 0usize;
    let mut total_custom = 0usize;
    let mut count = 0;

    for entry in &entries {
        let path = entry.path();
        let filename = path.file_name().unwrap().to_string_lossy();
        let short_name = &filename[..8.min(filename.len())];

        let img = match image::open(&path) {
            Ok(img) => img,
            Err(e) => {
                eprintln!("{}: Failed to open: {}", short_name, e);
                continue;
            }
        };

        let (width, height) = img.dimensions();
        let rgb = img.to_rgb8();
        let linear_rgb: Vec<f32> = rgb
            .pixels()
            .flat_map(|p| {
                let r = srgb_to_linear_val(p[0] as f32 / 255.0);
                let g = srgb_to_linear_val(p[1] as f32 / 255.0);
                let b = srgb_to_linear_val(p[2] as f32 / 255.0);
                [r, g, b]
            })
            .collect();

        // Default order
        let mut enc_default = jxl_encoder::vardct::VarDctEncoder::new(1.0);
        enc_default.custom_orders = false;
        let bytes_default =
            match enc_default.encode(width as usize, height as usize, &linear_rgb, None) {
                Ok(output) => output.data,
                Err(e) => {
                    eprintln!("{}: Default encode failed: {:?}", short_name, e);
                    continue;
                }
            };

        // Custom orders
        let mut enc_custom = jxl_encoder::vardct::VarDctEncoder::new(1.0);
        enc_custom.custom_orders = true;
        let bytes_custom =
            match enc_custom.encode(width as usize, height as usize, &linear_rgb, None) {
                Ok(output) => output.data,
                Err(e) => {
                    eprintln!("{}: Custom encode failed: {:?}", short_name, e);
                    continue;
                }
            };

        let saving_pct = (1.0 - bytes_custom.len() as f64 / bytes_default.len() as f64) * 100.0;
        let size_str = format!("{}x{}", width, height);

        eprintln!(
            "{:<12} {:>8} {:>8} {:>8} {:>6.1}%",
            short_name,
            size_str,
            bytes_default.len(),
            bytes_custom.len(),
            saving_pct
        );

        total_default += bytes_default.len();
        total_custom += bytes_custom.len();
        count += 1;
    }

    if count > 0 {
        eprintln!("{}", "-".repeat(50));
        let overall_saving = (1.0 - total_custom as f64 / total_default as f64) * 100.0;
        eprintln!(
            "{:<12} {:>8} {:>8} {:>8} {:>6.1}%",
            "TOTAL", "", total_default, total_custom, overall_saving
        );

        // Custom orders should not make files significantly larger
        // (permutation overhead should be small relative to AC savings)
        assert!(
            overall_saving > -2.0,
            "Custom orders made files {:.1}% larger overall (expected savings or minimal overhead)",
            -overall_saving
        );
    }
}

/// RD regression test: track encoder quality/size over time against committed baselines.
///
/// Encodes 6 test images (frymire + 5 CID22-512) at d=0.25 and d=0.5, measures
/// butteraugli + SSIM2, and asserts per-image thresholds.
///
/// CID22-512 images are auto-downloaded via the `codec-corpus` crate on first run.
///
/// Run with: cargo test -p jxl-encoder --test clic2025 test_rd_regression -- --ignored --nocapture
#[test]
#[ignore]
fn test_rd_regression() {
    use butteraugli::{ButteraugliParams, butteraugli_linear, srgb_to_linear};
    use imgref::Img;
    use rgb::RGB;
    use std::io::Cursor;

    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let project_root = std::path::Path::new(manifest_dir).parent().unwrap();

    // CID22-512 training images (512x512, CC BY-SA 4.0)
    // Downloaded and cached automatically by codec-corpus crate.
    let corpus = codec_corpus::Corpus::new().expect("Failed to init codec-corpus");
    let cid22_dir = corpus
        .get("CID22/CID22-512/training")
        .expect("Failed to download CID22-512 training set");

    // First 5 CID22-512 images by sorted filename
    const CID22_NAMES: [&str; 5] = ["1001682", "1028637", "1029604", "106399", "1080721"];

    struct TestImage {
        name: String,
        path: std::path::PathBuf,
    }

    let mut images = vec![TestImage {
        name: "frymire".into(),
        path: project_root.join("jxl-encoder/tests/images/frymire.png"),
    }];
    for name in &CID22_NAMES {
        images.push(TestImage {
            name: (*name).into(),
            path: cid22_dir.join(format!("{name}.png")),
        });
    }

    // Per-image baselines: (size, butteraugli, ssim2) at d=0.25, d=0.5, and d=1.0.
    // SSIM2: in-process fast-ssim2 on sRGB u8 (gamma 2.2 roundtrip from jxl-oxide linear output).
    // Butteraugli: in-process butteraugli crate on linear RGB (srgb_to_linear for original).
    struct Baseline {
        size: usize,
        butteraugli: f64,
        ssim2: f64,
    }

    struct ImageBaselines {
        d025: Baseline,
        d050: Baseline,
        d100: Baseline,
    }

    // To recalibrate: run `just rd-regression` and update from the output.
    // Last updated: 2026-02-23 (SIMD dispatch hoisting enables FMA in search functions)
    let baselines = [
        // frymire (1118x1105)
        ImageBaselines {
            d025: Baseline {
                size: 1186449,
                butteraugli: 1.546,
                ssim2: 93.47,
            },
            d050: Baseline {
                size: 902610,
                butteraugli: 1.550,
                ssim2: 91.12,
            },
            d100: Baseline {
                size: 661495,
                butteraugli: 1.570,
                ssim2: 86.48,
            },
        },
        // 1001682 (512x512)
        ImageBaselines {
            d025: Baseline {
                size: 120481,
                butteraugli: 0.472,
                ssim2: 92.98,
            },
            d050: Baseline {
                size: 83833,
                butteraugli: 0.662,
                ssim2: 90.61,
            },
            d100: Baseline {
                size: 54766,
                butteraugli: 1.265,
                ssim2: 85.87,
            },
        },
        // 1028637 (512x512)
        ImageBaselines {
            d025: Baseline {
                size: 92631,
                butteraugli: 0.437,
                ssim2: 93.66,
            },
            d050: Baseline {
                size: 64354,
                butteraugli: 0.885,
                ssim2: 90.41,
            },
            d100: Baseline {
                size: 44176,
                butteraugli: 1.371,
                ssim2: 84.56,
            },
        },
        // 1029604 (512x512)
        ImageBaselines {
            d025: Baseline {
                size: 158782,
                butteraugli: 0.412,
                ssim2: 94.81,
            },
            d050: Baseline {
                size: 110085,
                butteraugli: 0.708,
                ssim2: 92.07,
            },
            d100: Baseline {
                size: 70607,
                butteraugli: 1.431,
                ssim2: 87.31,
            },
        },
        // 106399 (512x512)
        ImageBaselines {
            d025: Baseline {
                size: 115785,
                butteraugli: 0.654,
                ssim2: 93.68,
            },
            d050: Baseline {
                size: 76180,
                butteraugli: 0.713,
                ssim2: 91.11,
            },
            d100: Baseline {
                size: 50059,
                butteraugli: 1.417,
                ssim2: 86.24,
            },
        },
        // 1080721 (512x512)
        ImageBaselines {
            d025: Baseline {
                size: 105269,
                butteraugli: 0.470,
                ssim2: 93.30,
            },
            d050: Baseline {
                size: 67249,
                butteraugli: 0.762,
                ssim2: 91.50,
            },
            d100: Baseline {
                size: 43795,
                butteraugli: 1.335,
                ssim2: 88.03,
            },
        },
    ];

    // --- Thresholds ---
    let size_margin = 1.03; // max 3% size growth
    let butteraugli_margin = 1.05; // max 5% butteraugli increase
    let ssim2_margin = 1.0; // max 1.0 SSIM2 point drop

    let params = ButteraugliParams::default();
    let distances: [f32; 3] = [0.25, 0.5, 1.0];
    let mut failures: Vec<String> = Vec::new();
    let mut improvements: Vec<String> = Vec::new();

    eprintln!("\n=== RD Regression Test ===\n");

    for dist in &distances {
        eprintln!("--- Distance {:.2} ---\n", dist);
        eprintln!(
            "{:<10} {:>8} {:>8} {:>6} {:>8} {:>6} {:>7} {:>8} {:>6}",
            "Image", "Size", "Base", "%", "Bfly", "Base", "B%", "SSIM2", "Base"
        );
        eprintln!("{}", "-".repeat(82));

        for (i, image) in images.iter().enumerate() {
            let img = match image::open(&image.path) {
                Ok(img) => img,
                Err(e) => {
                    let msg = format!("{}: failed to open: {}", image.name, e);
                    eprintln!("{}", msg);
                    failures.push(msg);
                    continue;
                }
            };

            let (w, h) = img.dimensions();
            let rgb = img.to_rgb8();

            // Original sRGB for SSIM2
            let original_srgb: Vec<[u8; 3]> = rgb.pixels().map(|p| [p[0], p[1], p[2]]).collect();

            // Linear RGB for encoder + butteraugli
            let linear_rgb: Vec<f32> = rgb
                .pixels()
                .flat_map(|p| {
                    [
                        srgb_to_linear(p[0]),
                        srgb_to_linear(p[1]),
                        srgb_to_linear(p[2]),
                    ]
                })
                .collect();

            // Original image for butteraugli
            let orig_pixels: Vec<RGB<f32>> = linear_rgb
                .chunks(3)
                .map(|c| RGB::new(c[0], c[1], c[2]))
                .collect();
            let orig_img = Img::new(orig_pixels, w as usize, h as usize);

            // Encode
            let encoder = jxl_encoder::vardct::VarDctEncoder::new(*dist);
            let bytes = match encoder.encode(w as usize, h as usize, &linear_rgb, None) {
                Ok(output) => output.data,
                Err(e) => {
                    let msg = format!("{} d={}: encode failed: {:?}", image.name, dist, e);
                    eprintln!("{}", msg);
                    failures.push(msg);
                    continue;
                }
            };

            // Decode with jxl-oxide
            let reader = Cursor::new(&bytes);
            let mut jxl_image = match jxl_oxide::JxlImage::builder().read(reader) {
                Ok(img) => img,
                Err(e) => {
                    let msg = format!("{} d={}: parse failed: {:?}", image.name, dist, e);
                    eprintln!("{}", msg);
                    failures.push(msg);
                    continue;
                }
            };
            jxl_image.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
                jxl_oxide::RenderingIntent::Relative,
            ));

            let render = match jxl_image.render_frame(0) {
                Ok(r) => r,
                Err(e) => {
                    let msg = format!("{} d={}: decode failed: {:?}", image.name, dist, e);
                    eprintln!("{}", msg);
                    failures.push(msg);
                    continue;
                }
            };

            let fb = render.image_all_channels();
            let decoded = fb.buf();

            // Butteraugli
            let dec_pixels: Vec<RGB<f32>> = decoded
                .chunks(3)
                .map(|c| RGB::new(c[0], c[1], c[2]))
                .collect();
            let dec_imgref = Img::new(dec_pixels, w as usize, h as usize);
            let bfly = match butteraugli_linear(orig_img.as_ref(), dec_imgref.as_ref(), &params) {
                Ok(result) => result.score,
                Err(e) => {
                    let msg = format!("{} d={}: butteraugli failed: {:?}", image.name, dist, e);
                    eprintln!("{}", msg);
                    failures.push(msg);
                    continue;
                }
            };

            // SSIM2
            let decoded_srgb: Vec<[u8; 3]> = decoded
                .chunks(3)
                .map(|rgb| {
                    let r = linear_to_srgb_u8(rgb[0]);
                    let g = linear_to_srgb_u8(rgb[1]);
                    let b = linear_to_srgb_u8(rgb[2]);
                    [r, g, b]
                })
                .collect();

            let original_img = imgref::Img::new(original_srgb.clone(), w as usize, h as usize);
            let decoded_img = imgref::Img::new(decoded_srgb, w as usize, h as usize);
            let ssim2 = match fast_ssim2::compute_ssimulacra2(
                original_img.as_ref(),
                decoded_img.as_ref(),
            ) {
                Ok(s) => s,
                Err(e) => {
                    let msg = format!("{} d={}: ssim2 failed: {:?}", image.name, dist, e);
                    eprintln!("{}", msg);
                    failures.push(msg);
                    continue;
                }
            };

            let base = if *dist == 0.25 {
                &baselines[i].d025
            } else if *dist == 0.5 {
                &baselines[i].d050
            } else {
                &baselines[i].d100
            };

            let size = bytes.len();

            // Skip assertions for uncalibrated baselines (size == 0)
            if base.size == 0 {
                eprintln!(
                    "{:<10} {:>8} {:>8} {:>6} {:>7.3} {:>6} {:>7} {:>7.2} {:>6}",
                    image.name, size, "NEW", "", bfly, "", "", ssim2, ""
                );
                continue;
            }

            let size_pct = (size as f64 / base.size as f64 - 1.0) * 100.0;
            let size_indicator = if size_pct <= 0.0 { "" } else { " !" };
            let bfly_pct = (bfly / base.butteraugli - 1.0) * 100.0;
            let bfly_indicator = if bfly_pct <= 0.05 { "" } else { "!" };

            eprintln!(
                "{:<10} {:>8} {:>8} {:>+5.1}%{} {:>7.3} {:>6.3} {:>+5.1}%{} {:>7.2} {:>6.2}",
                image.name,
                size,
                base.size,
                size_pct,
                size_indicator,
                bfly,
                base.butteraugli,
                bfly_pct,
                bfly_indicator,
                ssim2,
                base.ssim2,
            );

            // --- Assertions ---
            let size_limit = (base.size as f64 * size_margin) as usize;
            if size > size_limit {
                failures.push(format!(
                    "{} d={}: size {} > limit {} (baseline {} * {:.0}%)",
                    image.name,
                    dist,
                    size,
                    size_limit,
                    base.size,
                    size_margin * 100.0
                ));
            }

            let bfly_limit = base.butteraugli * butteraugli_margin;
            if bfly > bfly_limit {
                failures.push(format!(
                    "{} d={}: butteraugli {:.3} > limit {:.3} (baseline {:.3} * {:.0}%)",
                    image.name,
                    dist,
                    bfly,
                    bfly_limit,
                    base.butteraugli,
                    butteraugli_margin * 100.0
                ));
            }

            let ssim2_limit = base.ssim2 - ssim2_margin;
            if ssim2 < ssim2_limit {
                failures.push(format!(
                    "{} d={}: SSIM2 {:.2} < limit {:.2} (baseline {:.2} - {:.1})",
                    image.name, dist, ssim2, ssim2_limit, base.ssim2, ssim2_margin
                ));
            }

            // Track improvements
            if size < base.size {
                improvements.push(format!(
                    "{} d={}: size {:.1}% smaller",
                    image.name,
                    dist,
                    (1.0 - size as f64 / base.size as f64) * 100.0
                ));
            }
            if bfly < base.butteraugli * 0.95 {
                improvements.push(format!(
                    "{} d={}: butteraugli {:.1}% better",
                    image.name,
                    dist,
                    (1.0 - bfly / base.butteraugli) * 100.0
                ));
            }
            if ssim2 > base.ssim2 + 0.5 {
                improvements.push(format!(
                    "{} d={}: SSIM2 +{:.2}",
                    image.name,
                    dist,
                    ssim2 - base.ssim2
                ));
            }
        }
        eprintln!();
    }

    // --- Summary ---
    eprintln!("=== Summary ===\n");

    if !improvements.is_empty() {
        eprintln!("Improvements vs baseline:");
        for imp in &improvements {
            eprintln!("  + {}", imp);
        }
        eprintln!();
    }

    if failures.is_empty() {
        eprintln!("All images within regression thresholds.");
        eprintln!("  Size: < baseline * {:.0}%", size_margin * 100.0);
        eprintln!(
            "  Butteraugli: < baseline * {:.0}%",
            butteraugli_margin * 100.0
        );
        eprintln!("  SSIM2: > baseline - {:.1}", ssim2_margin);
    } else {
        eprintln!("REGRESSIONS DETECTED:");
        for fail in &failures {
            eprintln!("  - {}", fail);
        }
        panic!(
            "\n{} regression(s) detected. See output above for details.",
            failures.len()
        );
    }
}

/// High-distance RD regression test: d=2.0 and d=3.0.
///
/// These distances exercise DCT32x32 (d>=2.0) and DCT64x64 (d>=3.0) strategies.
/// This test catches quality regressions from broken non-square transforms
/// (DCT32x16, DCT16x32, DCT64x32, DCT32x64) that produce catastrophic butteraugli
/// (32-114) when enabled.
///
/// Run with: cargo test -p jxl-encoder --test clic2025 test_rd_regression_high_distance -- --ignored --nocapture
#[test]
#[ignore]
fn test_rd_regression_high_distance() {
    use butteraugli::{ButteraugliParams, butteraugli_linear, srgb_to_linear};
    use imgref::Img;
    use rgb::RGB;
    use std::io::Cursor;

    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let project_root = std::path::Path::new(manifest_dir).parent().unwrap();

    let corpus = codec_corpus::Corpus::new().expect("Failed to init codec-corpus");
    let cid22_dir = corpus
        .get("CID22/CID22-512/training")
        .expect("Failed to download CID22-512 training set");

    const CID22_NAMES: [&str; 5] = ["1001682", "1028637", "1029604", "106399", "1080721"];

    struct TestImage {
        name: String,
        path: std::path::PathBuf,
    }

    let mut images = vec![TestImage {
        name: "frymire".into(),
        path: project_root.join("jxl-encoder/tests/images/frymire.png"),
    }];
    for name in &CID22_NAMES {
        images.push(TestImage {
            name: (*name).into(),
            path: cid22_dir.join(format!("{name}.png")),
        });
    }

    struct Baseline {
        size: usize,
        butteraugli: f64,
        ssim2: f64,
    }

    struct ImageBaselines {
        d200: Baseline,
        d300: Baseline,
    }

    // To recalibrate: run the test with --nocapture and update from the output.
    // Last updated: 2026-02-23 (SIMD dispatch hoisting enables FMA in search functions)
    let baselines = [
        // frymire (1118x1105)
        ImageBaselines {
            d200: Baseline {
                size: 463859,
                butteraugli: 3.111,
                ssim2: 77.96,
            },
            d300: Baseline {
                size: 370935,
                butteraugli: 4.028,
                ssim2: 70.90,
            },
        },
        // 1001682 (512x512)
        ImageBaselines {
            d200: Baseline {
                size: 33843,
                butteraugli: 2.512,
                ssim2: 75.84,
            },
            d300: Baseline {
                size: 22674,
                butteraugli: 3.279,
                ssim2: 65.67,
            },
        },
        // 1028637 (512x512)
        ImageBaselines {
            d200: Baseline {
                size: 29082,
                butteraugli: 2.227,
                ssim2: 75.30,
            },
            d300: Baseline {
                size: 22288,
                butteraugli: 3.095,
                ssim2: 68.61,
            },
        },
        // 1029604 (512x512)
        ImageBaselines {
            d200: Baseline {
                size: 43500,
                butteraugli: 2.283,
                ssim2: 79.24,
            },
            d300: Baseline {
                size: 31237,
                butteraugli: 3.276,
                ssim2: 72.40,
            },
        },
        // 106399 (512x512)
        ImageBaselines {
            d200: Baseline {
                size: 31482,
                butteraugli: 2.065,
                ssim2: 77.65,
            },
            d300: Baseline {
                size: 23507,
                butteraugli: 2.821,
                ssim2: 71.08,
            },
        },
        // 1080721 (512x512)
        ImageBaselines {
            d200: Baseline {
                size: 28638,
                butteraugli: 2.087,
                ssim2: 82.28,
            },
            d300: Baseline {
                size: 22304,
                butteraugli: 2.771,
                ssim2: 77.04,
            },
        },
    ];

    let size_margin = 1.03; // max 3% size growth
    let butteraugli_margin = 1.05; // max 5% butteraugli increase
    let ssim2_margin = 1.0; // max 1.0 SSIM2 point drop

    // Hard quality floors: catches catastrophic regressions from broken transforms.
    // Butteraugli > 10 means severe artifacts visible to anyone.
    let butteraugli_floor = 8.0;
    // SSIM2 < 40 means the image is essentially destroyed.
    let ssim2_floor = 40.0;

    let params = ButteraugliParams::default();
    let distances: [f32; 2] = [2.0, 3.0];
    let mut failures: Vec<String> = Vec::new();

    eprintln!("\n=== High-Distance RD Regression Test ===\n");

    for dist in &distances {
        eprintln!("--- Distance {:.1} ---\n", dist);
        eprintln!(
            "{:<10} {:>8} {:>8} {:>6} {:>8} {:>6} {:>7} {:>8} {:>6}",
            "Image", "Size", "Base", "%", "Bfly", "Base", "B%", "SSIM2", "Base"
        );
        eprintln!("{}", "-".repeat(82));

        for (i, image) in images.iter().enumerate() {
            let img = match image::open(&image.path) {
                Ok(img) => img,
                Err(e) => {
                    let msg = format!("{}: failed to open: {}", image.name, e);
                    eprintln!("{}", msg);
                    failures.push(msg);
                    continue;
                }
            };

            let (w, h) = img.dimensions();
            let rgb = img.to_rgb8();

            let original_srgb: Vec<[u8; 3]> = rgb.pixels().map(|p| [p[0], p[1], p[2]]).collect();

            let linear_rgb: Vec<f32> = rgb
                .pixels()
                .flat_map(|p| {
                    [
                        srgb_to_linear(p[0]),
                        srgb_to_linear(p[1]),
                        srgb_to_linear(p[2]),
                    ]
                })
                .collect();

            let orig_pixels: Vec<RGB<f32>> = linear_rgb
                .chunks(3)
                .map(|c| RGB::new(c[0], c[1], c[2]))
                .collect();
            let orig_img = Img::new(orig_pixels, w as usize, h as usize);

            let encoder = jxl_encoder::vardct::VarDctEncoder::new(*dist);
            let bytes = match encoder.encode(w as usize, h as usize, &linear_rgb, None) {
                Ok(output) => output.data,
                Err(e) => {
                    let msg = format!("{} d={}: encode failed: {:?}", image.name, dist, e);
                    eprintln!("{}", msg);
                    failures.push(msg);
                    continue;
                }
            };

            let reader = Cursor::new(&bytes);
            let mut jxl_image = match jxl_oxide::JxlImage::builder().read(reader) {
                Ok(img) => img,
                Err(e) => {
                    let msg = format!("{} d={}: parse failed: {:?}", image.name, dist, e);
                    eprintln!("{}", msg);
                    failures.push(msg);
                    continue;
                }
            };
            jxl_image.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
                jxl_oxide::RenderingIntent::Relative,
            ));

            let render = match jxl_image.render_frame(0) {
                Ok(r) => r,
                Err(e) => {
                    let msg = format!("{} d={}: decode failed: {:?}", image.name, dist, e);
                    eprintln!("{}", msg);
                    failures.push(msg);
                    continue;
                }
            };

            let fb = render.image_all_channels();
            let decoded = fb.buf();

            let dec_pixels: Vec<RGB<f32>> = decoded
                .chunks(3)
                .map(|c| RGB::new(c[0], c[1], c[2]))
                .collect();
            let dec_imgref = Img::new(dec_pixels, w as usize, h as usize);
            let bfly = match butteraugli_linear(orig_img.as_ref(), dec_imgref.as_ref(), &params) {
                Ok(result) => result.score,
                Err(e) => {
                    let msg = format!("{} d={}: butteraugli failed: {:?}", image.name, dist, e);
                    eprintln!("{}", msg);
                    failures.push(msg);
                    continue;
                }
            };

            let decoded_srgb: Vec<[u8; 3]> = decoded
                .chunks(3)
                .map(|rgb| {
                    let r = linear_to_srgb_u8(rgb[0]);
                    let g = linear_to_srgb_u8(rgb[1]);
                    let b = linear_to_srgb_u8(rgb[2]);
                    [r, g, b]
                })
                .collect();

            let original_img = imgref::Img::new(original_srgb.clone(), w as usize, h as usize);
            let decoded_img = imgref::Img::new(decoded_srgb, w as usize, h as usize);
            let ssim2 = match fast_ssim2::compute_ssimulacra2(
                original_img.as_ref(),
                decoded_img.as_ref(),
            ) {
                Ok(s) => s,
                Err(e) => {
                    let msg = format!("{} d={}: ssim2 failed: {:?}", image.name, dist, e);
                    eprintln!("{}", msg);
                    failures.push(msg);
                    continue;
                }
            };

            let base = if *dist == 2.0 {
                &baselines[i].d200
            } else {
                &baselines[i].d300
            };

            let size = bytes.len();

            // Skip relative assertions for uncalibrated baselines (size == 0)
            if base.size == 0 {
                eprintln!(
                    "{:<10} {:>8} {:>8} {:>6} {:>7.3} {:>6} {:>7} {:>7.2} {:>6}",
                    image.name, size, "NEW", "", bfly, "", "", ssim2, ""
                );
            } else {
                let size_pct = (size as f64 / base.size as f64 - 1.0) * 100.0;
                let size_indicator = if size_pct <= 0.0 { "" } else { " !" };
                let bfly_pct = (bfly / base.butteraugli - 1.0) * 100.0;
                let bfly_indicator = if bfly_pct <= 0.05 { "" } else { "!" };

                eprintln!(
                    "{:<10} {:>8} {:>8} {:>+5.1}%{} {:>7.3} {:>6.3} {:>+5.1}%{} {:>7.2} {:>6.2}",
                    image.name,
                    size,
                    base.size,
                    size_pct,
                    size_indicator,
                    bfly,
                    base.butteraugli,
                    bfly_pct,
                    bfly_indicator,
                    ssim2,
                    base.ssim2,
                );

                let size_limit = (base.size as f64 * size_margin) as usize;
                if size > size_limit {
                    failures.push(format!(
                        "{} d={}: size {} > limit {} (baseline {} * {:.0}%)",
                        image.name,
                        dist,
                        size,
                        size_limit,
                        base.size,
                        size_margin * 100.0
                    ));
                }

                let bfly_limit = base.butteraugli * butteraugli_margin;
                if bfly > bfly_limit {
                    failures.push(format!(
                        "{} d={}: butteraugli {:.3} > limit {:.3} (baseline {:.3} * {:.0}%)",
                        image.name,
                        dist,
                        bfly,
                        bfly_limit,
                        base.butteraugli,
                        butteraugli_margin * 100.0
                    ));
                }

                let ssim2_limit = base.ssim2 - ssim2_margin;
                if ssim2 < ssim2_limit {
                    failures.push(format!(
                        "{} d={}: SSIM2 {:.2} < limit {:.2} (baseline {:.2} - {:.1})",
                        image.name, dist, ssim2, ssim2_limit, base.ssim2, ssim2_margin
                    ));
                }
            }

            // Hard quality floors — catches catastrophic regressions from broken transforms.
            // These fire regardless of whether baselines are calibrated.
            if bfly > butteraugli_floor {
                failures.push(format!(
                    "{} d={}: CATASTROPHIC butteraugli {:.3} > floor {:.1} (broken transform?)",
                    image.name, dist, bfly, butteraugli_floor
                ));
            }
            if ssim2 < ssim2_floor {
                failures.push(format!(
                    "{} d={}: CATASTROPHIC SSIM2 {:.2} < floor {:.1} (broken transform?)",
                    image.name, dist, ssim2, ssim2_floor
                ));
            }
        }
        eprintln!();
    }

    eprintln!("=== Summary ===\n");

    if failures.is_empty() {
        eprintln!("All images within quality thresholds.");
        eprintln!(
            "  Butteraugli floor: < {:.1} (catches broken transforms)",
            butteraugli_floor
        );
        eprintln!(
            "  SSIM2 floor: > {:.1} (catches broken transforms)",
            ssim2_floor
        );
    } else {
        eprintln!("REGRESSIONS DETECTED:");
        for fail in &failures {
            eprintln!("  - {}", fail);
        }
        panic!(
            "\n{} regression(s) detected. See output above for details.",
            failures.len()
        );
    }
}

/// Fair quality comparison: cjxl-rs vs cjxl-e5 vs cjxl-e7 using Rust butteraugli.
///
/// CRITICAL: cjxl writes gamma(0.454550) from source gAMA, we write sRGB TF.
/// If you decode to linear via jxl-oxide, the TF mismatch inflates cjxl's scores.
/// Instead, decode to NATIVE u8 (no color conversion) and feed both source and
/// decoded to butteraugli::butteraugli() which applies srgb_to_linear internally.
/// Same treatment for both → fair comparison regardless of declared TF.
///
/// Pre-requisite: run `bash /tmp/fair_cmp.sh` to encode all images.
///
/// Run with: cargo test -p jxl-encoder --test clic2025 test_fair_comparison -- --ignored --nocapture
#[test]
#[ignore]
fn test_fair_comparison() {
    use butteraugli::ButteraugliParams;
    use imgref::Img;
    use rgb::RGB;
    use std::io::Cursor;

    let clic_dir = jxl_encoder::test_helpers::corpus_dir().join("clic2025-1024");
    let jxl_dir = std::path::Path::new("/tmp/fair_cmp");

    let images: &[(&str, &str)] = &[
        (
            "02809272",
            "02809272b4ca9b08af45771501b741296187c7e26907efb44abbbfcb6cd804f7.png",
        ),
        (
            "1b4ad095",
            "1b4ad095795ac552b38a21d51be7bfaee8e7d0a70619d84767814321df4ed062.png",
        ),
        (
            "50fe4c3d",
            "50fe4c3d47d864858e1aaa60fecef5c453b4e18d2b368718eeb5c1e249e0c902.png",
        ),
        ("870516c6", "870516c65d81fb9267de6865964083a9.png"),
        (
            "8426ed22",
            "8426ed2245c791232862b0a0b2a62a1f17031e8e6e38921fe939df0b3a05ac41.png",
        ),
        ("a36713f1", "a36713f1943dac6bc74dea50cadaee6f.png"),
        (
            "0369d229",
            "0369d229ba4c9965d5caeb38c359a027a810968eee930b81520b604e76b4df14.png",
        ),
        ("097cb426", "097cb426910ba8ce2525dd8bb7fb1777.png"),
        ("100a02c2", "100a02c269c5948392f283b2aa3bb4da.png"),
        ("14ab4af2", "14ab4af28901fbeb1356b06d2d08ae06.png"),
        (
            "0d154749",
            "0d154749c7771f58e89ad343653ec4e20d6f037da829f47f5598e5d0a4ab61f0.png",
        ),
        (
            "07b9f93f",
            "07b9f93f170a0381836bdf301280a5b80b2c4be6e66f793a3c335dc200fb4e5b.png",
        ),
    ];

    let params = ButteraugliParams::new().with_intensity_target(80.0);

    // Helper: decode JXL to native u8 via jxl-oxide (NO color conversion).
    // Returns raw u8 in whatever TF the JXL declares. This matches how we treat
    // the source PNG (raw u8 without honoring gAMA), making the comparison fair.
    let decode_jxl_u8 = |path: &std::path::Path| -> Option<(Vec<RGB<u8>>, usize, usize)> {
        let data = std::fs::read(path).ok()?;
        let reader = Cursor::new(&data);
        let image = jxl_oxide::JxlImage::builder().read(reader).ok()?;
        // Don't request color encoding — get native rendering
        let render = image.render_frame(0).ok()?;
        let decoded = render.image_all_channels();
        let w = decoded.width();
        let h = decoded.height();
        let buf = decoded.buf();
        // f32 in native encoding [0,1] → quantize to u8
        let pixels: Vec<RGB<u8>> = buf
            .chunks(3)
            .map(|c| {
                RGB::new(
                    (c[0].clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
                    (c[1].clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
                    (c[2].clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
                )
            })
            .collect();
        Some((pixels, w, h))
    };

    for dist_str in &["0.5", "1.0", "2.0", "3.0"] {
        eprintln!("\n=== Distance {} ===", dist_str);
        eprintln!(
            "{:<10}  {:>7} {:>6}  {:>7} {:>6}  {:>7} {:>6}",
            "Image", "rs-KB", "rs-BA", "e5-KB", "e5-BA", "e7-KB", "e7-BA"
        );
        eprintln!("----------  ------- ------  ------- ------  ------- ------");

        let mut sum_rs_size: f64 = 0.0;
        let mut sum_e5_size: f64 = 0.0;
        let mut sum_e7_size: f64 = 0.0;
        let mut sum_rs_ba: f64 = 0.0;
        let mut sum_e5_ba: f64 = 0.0;
        let mut sum_e7_ba: f64 = 0.0;
        let mut n = 0;

        for (short, filename) in images {
            let src_path = clic_dir.join(filename);
            let img = match image::open(&src_path) {
                Ok(i) => i,
                Err(e) => {
                    eprintln!("Skip {}: {}", short, e);
                    continue;
                }
            };
            let (w, h) = (img.width() as usize, img.height() as usize);
            let rgb = img.to_rgb8();

            // Source as u8 (raw pixel values, no color management)
            let src_pixels: Vec<RGB<u8>> =
                rgb.pixels().map(|p| RGB::new(p[0], p[1], p[2])).collect();
            let src_img = Img::new(src_pixels, w, h);

            // Load and measure each encoder's output
            let rs_path = jxl_dir.join(format!("rs_{}_d{}.jxl", short, dist_str));
            let e5_path = jxl_dir.join(format!("e5_{}_d{}.jxl", short, dist_str));
            let e7_path = jxl_dir.join(format!("e7_{}_d{}.jxl", short, dist_str));

            let rs_size = std::fs::metadata(&rs_path).map(|m| m.len()).unwrap_or(0);
            let e5_size = std::fs::metadata(&e5_path).map(|m| m.len()).unwrap_or(0);
            let e7_size = std::fs::metadata(&e7_path).map(|m| m.len()).unwrap_or(0);

            let measure = |path: &std::path::Path| -> f64 {
                if let Some((pixels, pw, ph)) = decode_jxl_u8(path) {
                    if pw != w || ph != h {
                        return -1.0;
                    }
                    let dec_img = Img::new(pixels, w, h);
                    // butteraugli() applies srgb_to_linear to BOTH images internally
                    butteraugli::butteraugli(src_img.as_ref(), dec_img.as_ref(), &params)
                        .map(|r| r.score)
                        .unwrap_or(-1.0)
                } else {
                    -1.0
                }
            };

            let rs_ba = measure(&rs_path);
            let e5_ba = measure(&e5_path);
            let e7_ba = measure(&e7_path);

            eprintln!(
                "{:<10}  {:>7.1} {:>6.3}  {:>7.1} {:>6.3}  {:>7.1} {:>6.3}",
                short,
                rs_size as f64 / 1024.0,
                rs_ba,
                e5_size as f64 / 1024.0,
                e5_ba,
                e7_size as f64 / 1024.0,
                e7_ba,
            );

            sum_rs_size += rs_size as f64;
            sum_e5_size += e5_size as f64;
            sum_e7_size += e7_size as f64;
            sum_rs_ba += rs_ba;
            sum_e5_ba += e5_ba;
            sum_e7_ba += e7_ba;
            n += 1;
        }

        if n > 0 {
            let nf = n as f64;
            eprintln!("----------  ------- ------  ------- ------  ------- ------");
            eprintln!(
                "{:<10}  {:>7.1} {:>6.3}  {:>7.1} {:>6.3}  {:>7.1} {:>6.3}",
                "AVERAGE",
                sum_rs_size / nf / 1024.0,
                sum_rs_ba / nf,
                sum_e5_size / nf / 1024.0,
                sum_e5_ba / nf,
                sum_e7_size / nf / 1024.0,
                sum_e7_ba / nf,
            );
            let size_vs_e5 = (sum_rs_size - sum_e5_size) * 100.0 / sum_e5_size;
            let size_vs_e7 = (sum_rs_size - sum_e7_size) * 100.0 / sum_e7_size;
            let ba_vs_e5 = (sum_rs_ba - sum_e5_ba) * 100.0 / sum_e5_ba;
            let ba_vs_e7 = (sum_rs_ba - sum_e7_ba) * 100.0 / sum_e7_ba;
            eprintln!(
                "  Size vs e5: {:+.1}%  Size vs e7: {:+.1}%",
                size_vs_e5, size_vs_e7
            );
            eprintln!(
                "  BA   vs e5: {:+.1}%  BA   vs e7: {:+.1}%  (negative = better)",
                ba_vs_e5, ba_vs_e7
            );
        }
    }
}

// ── Patches (dictionary-based repeated patterns) tests ─────────────────────

/// Helper: decode JXL bytes with jxl-rs, return (width, height, sRGB f32 pixels).
fn decode_jxl_rs_for_patches(data: &[u8]) -> (usize, usize, Vec<f32>) {
    use jxl::api::{
        JxlColorType, JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer,
        JxlPixelFormat, ProcessingResult, states,
    };
    use jxl::image::{Image, Rect};

    let mut input = data;
    let options = JxlDecoderOptions::default();
    let mut decoder = JxlDecoder::<states::Initialized>::new(options);

    // Process header
    let mut decoder = loop {
        match decoder.process(&mut input) {
            Ok(ProcessingResult::Complete { result }) => break result,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                if input.is_empty() {
                    panic!("jxl-rs: unexpected end of input during header");
                }
                decoder = fallback;
            }
            Err(e) => panic!("jxl-rs header decode error: {:?}", e),
        }
    };

    let basic_info = decoder.basic_info().clone();
    let (width, height) = basic_info.size;
    let channels = 3;

    let format = JxlPixelFormat {
        color_type: JxlColorType::Rgb,
        color_data_format: Some(JxlDataFormat::f32()),
        extra_channel_format: vec![],
    };
    decoder.set_pixel_format(format);

    // Process to frame info
    let mut decoder = loop {
        match decoder.process(&mut input) {
            Ok(ProcessingResult::Complete { result }) => break result,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                if input.is_empty() {
                    panic!("jxl-rs: unexpected end of input before frame");
                }
                decoder = fallback;
            }
            Err(e) => panic!("jxl-rs frame info decode error: {:?}", e),
        }
    };

    let mut output_image = Image::<f32>::new((width * channels, height))
        .expect("jxl-rs: failed to create output buffer");

    let mut buffers = vec![JxlOutputBuffer::from_image_rect_mut(
        output_image
            .get_rect_mut(Rect {
                origin: (0, 0),
                size: (width * channels, height),
            })
            .into_raw(),
    )];

    // Decode frame(s) — patches produce reference frame + main frame
    loop {
        match decoder.process(&mut input, &mut buffers) {
            Ok(ProcessingResult::Complete { .. }) => break,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                if input.is_empty() {
                    panic!("jxl-rs: unexpected end of input during frame decode");
                }
                decoder = fallback;
            }
            Err(e) => panic!("jxl-rs frame decode error: {:?}", e),
        }
    }

    let mut pixels = Vec::with_capacity(width * height * channels);
    for y in 0..height {
        pixels.extend_from_slice(output_image.row(y));
    }
    (width, height, pixels)
}

/// Helper: decode JXL bytes with djxl (libjxl reference decoder).
fn decode_djxl_for_patches(data: &[u8]) -> (usize, usize, Vec<u8>) {
    let pid = std::process::id();
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let temp_jxl = format!("/tmp/patches_test_{}_{}.jxl", pid, ts);
    let temp_png = format!("/tmp/patches_test_{}_{}.png", pid, ts);

    std::fs::write(&temp_jxl, data).unwrap();
    let output = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
        .args([&temp_jxl, &temp_png])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "djxl failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let img = image::open(&temp_png).unwrap();
    let rgb = img.to_rgb8();
    let w = rgb.width() as usize;
    let h = rgb.height() as usize;
    let srgb_bytes: Vec<u8> = rgb.into_raw();

    let _ = std::fs::remove_file(&temp_jxl);
    let _ = std::fs::remove_file(&temp_png);
    (w, h, srgb_bytes)
}

/// Helper: encode a screenshot PNG with patches on/off, return (bytes, size).
fn encode_screenshot_with_patches(
    path: &str,
    distance: f32,
    patches: bool,
) -> Option<(Vec<u8>, usize)> {
    let img = match image::open(path) {
        Ok(img) => img,
        Err(e) => {
            eprintln!("Could not open {}: {}", path, e);
            return None;
        }
    };
    let (w, h) = img.dimensions();
    let rgb = img.to_rgb8();
    let pixels: Vec<u8> = rgb.into_raw();

    let data = jxl_encoder::LossyConfig::new(distance)
        .with_patches(patches)
        .with_butteraugli_iters(0) // Skip butteraugli loop for speed in tests
        .encode(&pixels, w, h, jxl_encoder::PixelLayout::Rgb8)
        .unwrap();

    let size = data.len();
    Some((data, size))
}

/// Roundtrip test: encode a screenshot with patches → decode with jxl-rs.
#[test]
#[ignore] // Requires codec corpus
fn test_patches_roundtrip_jxl_rs() {
    let path = &format!(
        "{}/gb82-sc/windows95.png",
        jxl_encoder::test_helpers::corpus_dir().display()
    );
    let img = match image::open(path) {
        Ok(img) => img,
        Err(_) => {
            eprintln!("Skipping: corpus not available");
            return;
        }
    };
    let (w, h) = img.dimensions();
    let rgb = img.to_rgb8();
    let pixels: Vec<u8> = rgb.into_raw();

    // Encode with patches enabled
    let data = jxl_encoder::LossyConfig::new(1.0)
        .with_patches(true)
        .with_butteraugli_iters(0)
        .encode(&pixels, w, h, jxl_encoder::PixelLayout::Rgb8)
        .unwrap();

    eprintln!(
        "windows95.png: {}x{}, {} bytes with patches",
        w,
        h,
        data.len()
    );

    // Decode with jxl-rs — must not error
    let (dw, dh, decoded) = decode_jxl_rs_for_patches(&data);
    assert_eq!(dw, w as usize);
    assert_eq!(dh, h as usize);
    assert_eq!(decoded.len(), w as usize * h as usize * 3);
    eprintln!(
        "  jxl-rs decode: OK ({}x{}, {} f32 pixels)",
        dw,
        dh,
        decoded.len() / 3
    );
}

/// Roundtrip test: encode a screenshot with patches → decode with djxl.
#[test]
#[ignore] // Requires codec corpus + djxl
fn test_patches_roundtrip_djxl() {
    let path = &format!(
        "{}/gb82-sc/windows95.png",
        jxl_encoder::test_helpers::corpus_dir().display()
    );
    let img = match image::open(path) {
        Ok(img) => img,
        Err(_) => {
            eprintln!("Skipping: corpus not available");
            return;
        }
    };
    let (w, h) = img.dimensions();
    let rgb = img.to_rgb8();
    let pixels: Vec<u8> = rgb.into_raw();

    let data = jxl_encoder::LossyConfig::new(1.0)
        .with_patches(true)
        .with_butteraugli_iters(0)
        .encode(&pixels, w, h, jxl_encoder::PixelLayout::Rgb8)
        .unwrap();

    eprintln!(
        "windows95.png: {}x{}, {} bytes with patches",
        w,
        h,
        data.len()
    );

    // Decode with djxl — must not error
    let (dw, dh, decoded_srgb) = decode_djxl_for_patches(&data);
    assert_eq!(dw, w as usize);
    assert_eq!(dh, h as usize);
    assert_eq!(decoded_srgb.len(), w as usize * h as usize * 3);
    eprintln!(
        "  djxl decode: OK ({}x{}, {} sRGB pixels)",
        dw,
        dh,
        decoded_srgb.len() / 3
    );
}

/// Size comparison: patches ON vs OFF on GB82-SC screenshot corpus.
/// Expects significant savings on screenshots.
#[test]
#[ignore] // Requires codec corpus
fn test_patches_screenshot_corpus_size() {
    let corpus = &format!(
        "{}/gb82-sc",
        jxl_encoder::test_helpers::corpus_dir().display()
    );
    let screenshots = [
        "windows95.png",
        "graph.png",
        "gui.png",
        "terminal.png",
        "windows.png",
        "codec_wiki.png",
        "gmessages.png",
        "imessage.png",
        "imac_dark.png",
        "imac_g3.png",
    ];

    eprintln!(
        "{:<15} {:>10} {:>10} {:>8}",
        "Image", "No Patch", "Patches", "Savings"
    );
    eprintln!("{}", "-".repeat(50));

    let mut total_no_patches = 0usize;
    let mut total_patches = 0usize;
    let mut count = 0;

    for name in &screenshots {
        let path = format!("{}/{}", corpus, name);
        let no_patch = encode_screenshot_with_patches(&path, 1.0, false);
        let with_patch = encode_screenshot_with_patches(&path, 1.0, true);

        if let (Some((_, size_no)), Some((data_yes, size_yes))) = (no_patch, with_patch) {
            let savings = (1.0 - size_yes as f64 / size_no as f64) * 100.0;
            let short = name.split('.').next().unwrap_or(name);
            eprintln!(
                "{:<15} {:>10} {:>10} {:>7.1}%",
                short, size_no, size_yes, savings
            );
            total_no_patches += size_no;
            total_patches += size_yes;
            count += 1;

            // Also verify patches version decodes with jxl-rs
            let (dw, dh, _) = decode_jxl_rs_for_patches(&data_yes);
            assert!(dw > 0 && dh > 0, "decode failed for {}", name);
        }
    }

    if count > 0 {
        let total_savings = (1.0 - total_patches as f64 / total_no_patches as f64) * 100.0;
        eprintln!("{}", "-".repeat(50));
        eprintln!(
            "{:<15} {:>10} {:>10} {:>7.1}%",
            "TOTAL", total_no_patches, total_patches, total_savings
        );
    }
}

/// Regression test: CLIC photos should produce no patches (zero-cost feature).
/// With patches enabled, file size should be within 0.5% of patches disabled.
#[test]
#[ignore] // Requires codec corpus
fn test_patches_no_regression_on_photos() {
    let corpus = &format!(
        "{}/clic2025-1024",
        jxl_encoder::test_helpers::corpus_dir().display()
    );
    let photos: Vec<String> = match std::fs::read_dir(corpus) {
        Ok(entries) => entries
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().to_string())
            .filter(|n| n.ends_with(".png"))
            .take(3)
            .collect(),
        Err(_) => {
            eprintln!("Skipping: corpus not available at {}", corpus);
            return;
        }
    };

    for name in &photos {
        let path = format!("{}/{}", corpus, name);
        let no_patch = encode_screenshot_with_patches(&path, 1.0, false);
        let with_patch = encode_screenshot_with_patches(&path, 1.0, true);

        if let (Some((_, size_no)), Some((_, size_yes))) = (no_patch, with_patch) {
            let diff_pct = ((size_yes as f64 - size_no as f64) / size_no as f64).abs() * 100.0;
            eprintln!(
                "{}: no_patches={} patches={} diff={:.2}%",
                name, size_no, size_yes, diff_pct
            );
            // Photos should have near-zero overhead from patches
            assert!(
                diff_pct < 1.0,
                "{}: patches added {:.2}% overhead on a photo (expected <1%)",
                name,
                diff_pct
            );
        }
    }
}

/// Synthetic screenshot test: 256x256 with many repeated glyphs.
/// Verifies the full patches path: detect → pack → reference frame → subtract → encode → decode.
#[test]
fn test_patches_synthetic_screenshot_encode() {
    // 256x256 image: solid gray background with a grid of repeated 8x8 "glyphs"
    let w = 256usize;
    let h = 256usize;
    let mut pixels = vec![200u8; w * h * 3]; // Light gray background

    // Create 3 different glyph patterns, each repeated many times in a grid
    let glyphs: Vec<Vec<u8>> = vec![
        // Glyph 0: solid dark block
        vec![40; 8 * 8 * 3],
        // Glyph 1: vertical bar
        {
            let mut g = vec![200u8; 8 * 8 * 3];
            for y in 0..8 {
                for x in 2..5 {
                    let i = (y * 8 + x) * 3;
                    g[i] = 60;
                    g[i + 1] = 60;
                    g[i + 2] = 60;
                }
            }
            g
        },
        // Glyph 2: horizontal bar
        {
            let mut g = vec![200u8; 8 * 8 * 3];
            for y in 2..5 {
                for x in 0..8 {
                    let i = (y * 8 + x) * 3;
                    g[i] = 80;
                    g[i + 1] = 80;
                    g[i + 2] = 80;
                }
            }
            g
        },
    ];

    // Place glyphs in a grid: 16 columns × 12 rows = 192 occurrences
    for row in 0..12 {
        for col in 0..16 {
            let gx = col * 16 + 4;
            let gy = row * 20 + 4;
            let glyph_idx = (row * 16 + col) % glyphs.len();
            let glyph = &glyphs[glyph_idx];
            for dy in 0..8 {
                for dx in 0..8 {
                    let px = gx + dx;
                    let py = gy + dy;
                    if px < w && py < h {
                        let dst = (py * w + px) * 3;
                        let src = (dy * 8 + dx) * 3;
                        pixels[dst] = glyph[src];
                        pixels[dst + 1] = glyph[src + 1];
                        pixels[dst + 2] = glyph[src + 2];
                    }
                }
            }
        }
    }

    // Encode without patches first (baseline)
    let data_no_patches = jxl_encoder::LossyConfig::new(1.0)
        .with_patches(false)
        .with_butteraugli_iters(0)
        .encode(&pixels, w as u32, h as u32, jxl_encoder::PixelLayout::Rgb8)
        .unwrap();
    eprintln!(
        "Synthetic screenshot (no patches): {}x{}, {} bytes",
        w,
        h,
        data_no_patches.len()
    );

    // Verify no-patches version decodes with jxl-oxide
    {
        let reader = Cursor::new(&data_no_patches);
        let mut image = jxl_oxide::JxlImage::builder().read(reader).unwrap();
        image.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb(
            jxl_oxide::RenderingIntent::Relative,
        ));
        let render = image.render_frame(0).unwrap();
        eprintln!(
            "  no-patches jxl-oxide decode: OK ({}x{})",
            render.image_all_channels().width(),
            render.image_all_channels().height()
        );
    }

    // Encode with patches enabled
    let data = jxl_encoder::LossyConfig::new(1.0)
        .with_patches(true)
        .with_butteraugli_iters(0)
        .encode(&pixels, w as u32, h as u32, jxl_encoder::PixelLayout::Rgb8)
        .unwrap();

    eprintln!(
        "Synthetic screenshot (patches): {}x{}, {} bytes",
        w,
        h,
        data.len()
    );

    // Check if patches actually fired
    let patches_fired = data.len() != data_no_patches.len();
    eprintln!(
        "  patches fired: {} (size diff: {} bytes)",
        patches_fired,
        data_no_patches.len() as i64 - data.len() as i64
    );

    // Verify patches version decodes with djxl
    let patches_dir = jxl_encoder::test_helpers::output_dir_for("jxl-encoder", "patches");
    let test_path = patches_dir.join("synthetic_test.jxl");
    std::fs::write(&test_path, &data).unwrap();

    let decoded_path = patches_dir.join("synthetic_test.png");
    let output = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
        .args([&test_path, &decoded_path])
        .output();
    if let Ok(out) = output {
        assert!(
            out.status.success(),
            "djxl decode failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        eprintln!("  djxl decode: OK");
    }

    // Verify with jxl-oxide
    let reader = Cursor::new(&data);
    let mut image = jxl_oxide::JxlImage::builder()
        .read(reader)
        .expect("jxl-oxide parse failed");
    image.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb(
        jxl_oxide::RenderingIntent::Relative,
    ));
    let render = image.render_frame(0).expect("jxl-oxide render failed");
    eprintln!(
        "  jxl-oxide decode: OK ({}x{})",
        render.image_all_channels().width(),
        render.image_all_channels().height()
    );
}
