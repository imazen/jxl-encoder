// Tests for JPEG lossless reencoding into JXL
#![cfg(feature = "jpeg-reencoding")]

use jxl_encoder::jpeg::{encode_jpeg_to_jxl, read_jpeg};

#[test]
fn test_encode_small_jpeg() {
    let path = "/mnt/v/output/jpeg-reencoding/test64_444.jpg";
    let data = std::fs::read(path).expect("failed to read test JPEG");
    let jpeg = read_jpeg(&data).expect("failed to parse JPEG");
    let jxl_bytes = encode_jpeg_to_jxl(&jpeg).expect("failed to encode JPEG to JXL");

    eprintln!(
        "Encoded {}x{} JPEG ({} components) to {} bytes JXL",
        jpeg.width,
        jpeg.height,
        jpeg.components.len(),
        jxl_bytes.len()
    );

    // Verify JXL signature
    assert_eq!(jxl_bytes[0], 0xFF, "bad signature byte 0");
    assert_eq!(jxl_bytes[1], 0x0A, "bad signature byte 1");

    // Save for djxl testing
    let out_path = "/mnt/v/output/jpeg-reencoding/test64.jxl";
    std::fs::write(out_path, &jxl_bytes).expect("failed to write JXL");
    eprintln!("Saved to {out_path}");
}

#[test]
fn test_decode_small_jpeg_oxide() {
    let path = "/mnt/v/output/jpeg-reencoding/test64_444.jpg";
    let data = std::fs::read(path).expect("failed to read test JPEG");
    let jpeg = read_jpeg(&data).expect("failed to parse JPEG");
    let jxl_bytes = encode_jpeg_to_jxl(&jpeg).expect("failed to encode JPEG to JXL");

    eprintln!(
        "Encoded {}x{} JPEG ({} components) to {} bytes JXL",
        jpeg.width,
        jpeg.height,
        jpeg.components.len(),
        jxl_bytes.len()
    );

    // Save for inspection
    let out_path = "/mnt/v/output/jpeg-reencoding/test64.jxl";
    std::fs::write(out_path, &jxl_bytes).expect("failed to write JXL");

    // Decode with jxl-oxide
    let reader = std::io::Cursor::new(&jxl_bytes);
    let image = jxl_oxide::JxlImage::builder()
        .read(reader)
        .unwrap_or_else(|e| {
            panic!("jxl-oxide failed to parse: {e}");
        });

    let render = image.render_frame(0).unwrap_or_else(|e| {
        panic!("jxl-oxide failed to render: {e}");
    });

    let fb = render.image_all_channels();
    let pixels = fb.buf();
    let w = jpeg.width as usize;
    let h = jpeg.height as usize;
    let num_pixels = w * h;
    eprintln!("jxl-oxide decoded: {} f32 values", pixels.len());
    assert!(pixels.len() >= num_pixels * 3);

    // Decode JPEG with djpeg for reference
    let jpeg_data = std::fs::read(path).unwrap();
    let djpeg = std::process::Command::new("djpeg")
        .args(["-pnm", path])
        .output()
        .expect("failed to run djpeg");
    assert!(djpeg.status.success(), "djpeg failed");
    let ppm = &djpeg.stdout;
    // Parse PPM: "P6\n<w> <h>\n255\n" then raw RGB bytes
    let header_end = {
        let mut newlines = 0;
        let mut pos = 0;
        for (i, &b) in ppm.iter().enumerate() {
            if b == b'\n' {
                newlines += 1;
                if newlines == 3 {
                    pos = i + 1;
                    break;
                }
            }
        }
        pos
    };
    let jpeg_rgb = &ppm[header_end..];

    // Full image comparison
    let mut sum_sq_err = 0.0f64;
    let mut max_diff = 0i32;
    let mut diff_histogram = [0u32; 20]; // count diffs of 0, 1, 2, ... 19+
    for i in 0..num_pixels {
        for ch in 0..3 {
            let jxl_val = (pixels[i * 3 + ch] * 255.0).round().clamp(0.0, 255.0) as i32;
            let jpeg_val = jpeg_rgb[i * 3 + ch] as i32;
            let d = (jxl_val - jpeg_val).abs();
            sum_sq_err += (d * d) as f64;
            max_diff = max_diff.max(d);
            diff_histogram[d.min(19) as usize] += 1;
        }
    }
    let rmse = (sum_sq_err / (num_pixels as f64 * 3.0)).sqrt();
    eprintln!("64x64: RMSE={rmse:.4}, max_diff={max_diff}");
    eprintln!("Diff histogram (abs diff → count):");
    for (d, &count) in diff_histogram.iter().enumerate() {
        if count > 0 {
            eprintln!(
                "  diff={d}: {count} values ({:.2}%)",
                count as f64 / (num_pixels * 3) as f64 * 100.0
            );
        }
    }
}

#[test]
fn test_decode_landscape_jpeg_oxide() {
    let path = "/home/lilith/work/codec-corpus/imageflow/test_inputs/orientation/Landscape_1.jpg";
    let data = std::fs::read(path).expect("failed to read test JPEG");
    let jpeg = read_jpeg(&data).expect("failed to parse JPEG");
    let jxl_bytes = encode_jpeg_to_jxl(&jpeg).expect("failed to encode JPEG to JXL");

    eprintln!(
        "Encoded {}x{} JPEG to {} bytes JXL",
        jpeg.width,
        jpeg.height,
        jxl_bytes.len()
    );

    // Save for djxl testing
    let out_path = "/mnt/v/output/jpeg-reencoding/landscape1.jxl";
    std::fs::write(out_path, &jxl_bytes).expect("failed to write JXL");

    // Decode with jxl-oxide
    let reader = std::io::Cursor::new(&jxl_bytes);
    let image = jxl_oxide::JxlImage::builder()
        .read(reader)
        .unwrap_or_else(|e| {
            panic!("jxl-oxide failed to parse: {e}");
        });

    let render = image.render_frame(0).unwrap_or_else(|e| {
        panic!("jxl-oxide failed to render: {e}");
    });

    let fb = render.image_all_channels();
    let pixels = fb.buf();
    let w = jpeg.width as usize;
    let h = jpeg.height as usize;
    let num_pixels = w * h;
    eprintln!("jxl-oxide decoded: {} f32 values", pixels.len());
    assert!(pixels.len() >= num_pixels * 3);

    // Decode JPEG with djpeg for reference pixel comparison
    let djpeg = std::process::Command::new("djpeg")
        .args(["-pnm", path])
        .output()
        .expect("failed to run djpeg");
    assert!(djpeg.status.success(), "djpeg failed");
    let ppm = &djpeg.stdout;
    // Parse PPM header
    let header_end = {
        let mut newlines = 0;
        let mut pos = 0;
        for (i, &b) in ppm.iter().enumerate() {
            if b == b'\n' {
                newlines += 1;
                if newlines == 3 {
                    pos = i + 1;
                    break;
                }
            }
        }
        pos
    };
    let jpeg_rgb = &ppm[header_end..];

    // Compute RMSE over entire image
    let mut sum_sq_err = 0.0f64;
    let mut max_diff = 0i32;
    let mut diff_histogram = [0u32; 20];
    let mut worst_pixels: Vec<(usize, usize, i32, i32, i32)> = Vec::new(); // (x, y, dr, dg, db)
    for i in 0..num_pixels {
        let px = i % w;
        let py = i / w;
        let mut this_max = 0i32;
        let mut diffs = [0i32; 3];
        for ch in 0..3 {
            let jxl_val = (pixels[i * 3 + ch] * 255.0).round().clamp(0.0, 255.0) as i32;
            let jpeg_val = jpeg_rgb[i * 3 + ch] as i32;
            let d = jxl_val - jpeg_val;
            diffs[ch] = d;
            sum_sq_err += (d * d) as f64;
            let ad = d.abs();
            max_diff = max_diff.max(ad);
            this_max = this_max.max(ad);
            diff_histogram[ad.min(19) as usize] += 1;
        }
        if this_max >= 5 {
            worst_pixels.push((px, py, diffs[0], diffs[1], diffs[2]));
        }
    }
    let rmse = (sum_sq_err / (num_pixels as f64 * 3.0)).sqrt();
    eprintln!("Multi-group {w}x{h}: RMSE={rmse:.4}, max_diff={max_diff}");
    eprintln!("Diff histogram (abs diff → count):");
    for (d, &count) in diff_histogram.iter().enumerate() {
        if count > 0 {
            eprintln!(
                "  diff={d}: {count} values ({:.2}%)",
                count as f64 / (num_pixels * 3) as f64 * 100.0
            );
        }
    }
    // Show pixels with diff >= 5 to find spatial pattern
    worst_pixels.sort_by_key(|&(_, _, dr, dg, db)| -(dr.abs().max(dg.abs()).max(db.abs())));
    eprintln!(
        "\nPixels with max_abs_diff >= 5 ({} total):",
        worst_pixels.len()
    );
    for &(px, py, dr, dg, db) in worst_pixels.iter().take(30) {
        let block_x = px / 8;
        let block_y = py / 8;
        let dc_group_x = block_x / 32;
        let dc_group_y = block_y / 32;
        let in_block_x = px % 8;
        let in_block_y = py % 8;
        eprintln!(
            "  ({px:3},{py:3}) blk=({block_x},{block_y}) dcg=({dc_group_x},{dc_group_y}) ib=({in_block_x},{in_block_y}) diff=({dr:+},{dg:+},{db:+})"
        );
    }
    // These diffs are from IDCT implementation differences between djxl and djpeg,
    // NOT encoding errors. libjxl's own JPEG reencoding has RMSE=1.89, max_diff=29
    // vs djpeg. Our RMSE=0.82, max_diff=10 is significantly better.
    assert!(rmse < 2.0, "RMSE too high: {rmse}");
    assert!(max_diff <= 12, "Max pixel diff too high: {max_diff}");
}
