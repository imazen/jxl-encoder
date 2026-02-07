fn main() {
    // Test vertical gradient at various sizes
    let sizes = [(64, 64), (100, 100), (33, 47), (200, 200), (256, 256)];

    for (w, h) in sizes {
        eprintln!("\n=== Testing {}x{} vertical gradient ===", w, h);

        // Generate vertical gradient (same as test)
        let mut data = vec![0u8; w * h * 3];
        for y in 0..h {
            let val = (y * 255 / h.max(1)) as u8;
            for x in 0..w {
                let idx = (y * w + x) * 3;
                data[idx] = val;
                data[idx + 1] = val;
                data[idx + 2] = val;
            }
        }

        // Try encoding
        let encoded = match jxl_enc::encoder::encode_lossy_rgb8(&data, w, h, 1.0) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("  ENCODE ERROR: {:?}", e);
                continue;
            }
        };
        eprintln!("  Encoded: {} bytes", encoded.len());

        // Save for debugging
        let filename = format!("/tmp/v_grad_{}x{}.jxl", w, h);
        std::fs::write(&filename, &encoded).unwrap();
        eprintln!("  Saved to: {}", filename);

        // Try decoding with jxl-oxide
        let result = jxl_oxide::JxlImage::builder()
            .read(std::io::Cursor::new(&encoded))
            .and_then(|img| img.render_frame(0));

        match result {
            Ok(_) => eprintln!("  DECODE: OK"),
            Err(e) => eprintln!("  DECODE ERROR: {:?}", e),
        }

        // Also try djxl if available
        let djxl_result = std::process::Command::new("djxl")
            .args([&filename, "/tmp/test_out.png"])
            .output();

        match djxl_result {
            Ok(output) => {
                if output.status.success() {
                    eprintln!("  djxl: OK");
                } else {
                    eprintln!("  djxl: FAIL - {}", String::from_utf8_lossy(&output.stderr));
                }
            }
            Err(_) => eprintln!("  djxl: not available"),
        }
    }
}
