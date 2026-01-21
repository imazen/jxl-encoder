use jxl_enc::encoder::encode_lossy_rgb8;
/// Compare vertical (failing) vs horizontal (working) encoding in detail
///
/// Focus on finding the difference that causes vertical to fail
use std::io::Cursor;

fn generate_vertical(size: usize) -> Vec<u8> {
    let mut data = vec![0u8; size * size * 3];
    for y in 0..size {
        let val = (y * 255 / size.max(1)) as u8;
        for x in 0..size {
            let idx = (y * size + x) * 3;
            data[idx] = val;
            data[idx + 1] = val;
            data[idx + 2] = val;
        }
    }
    data
}

fn generate_horizontal(size: usize) -> Vec<u8> {
    let mut data = vec![0u8; size * size * 3];
    for y in 0..size {
        for x in 0..size {
            let val = (x * 255 / size.max(1)) as u8;
            let idx = (y * size + x) * 3;
            data[idx] = val;
            data[idx + 1] = val;
            data[idx + 2] = val;
        }
    }
    data
}

fn try_decode(jxl_data: &[u8], label: &str) -> bool {
    let result = jxl_oxide::JxlImage::builder()
        .read(Cursor::new(jxl_data))
        .and_then(|img| img.render_frame(0));

    match result {
        Ok(_) => {
            eprintln!("{}: DECODE OK", label);
            true
        }
        Err(e) => {
            eprintln!("{}: DECODE FAIL - {}", label, e);
            false
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Test different sizes to find where the bug appears
    for size in [16, 24, 32, 33, 40, 48] {
        eprintln!("\n{}", "=".repeat(60));
        eprintln!("Testing size {}x{}", size, size);
        eprintln!("{}", "=".repeat(60));

        eprintln!("\n--- Horizontal {} ---", size);
        let h_data = generate_horizontal(size);
        let h_jxl = encode_lossy_rgb8(&h_data, size, size, 85.0)?;
        eprintln!("File size: {} bytes", h_jxl.len());
        let h_ok = try_decode(&h_jxl, &format!("Horizontal {}", size));

        eprintln!("\n--- Vertical {} ---", size);
        let v_data = generate_vertical(size);
        let v_jxl = encode_lossy_rgb8(&v_data, size, size, 85.0)?;
        eprintln!("File size: {} bytes", v_jxl.len());
        let v_ok = try_decode(&v_jxl, &format!("Vertical {}", size));

        if h_ok && !v_ok {
            eprintln!(
                "\n*** BUG FOUND at size {}x{}: horizontal works, vertical fails ***",
                size, size
            );
        }
    }

    // Now test 33x33 with detailed token output
    eprintln!("\n\n{}", "=".repeat(60));
    eprintln!("DETAILED 33x33 COMPARISON");
    eprintln!("{}", "=".repeat(60));

    eprintln!("\n--- 33x33 Horizontal (should work) ---");
    let h33 = generate_horizontal(33);
    let _ = encode_lossy_rgb8(&h33, 33, 33, 85.0)?;

    eprintln!("\n--- 33x33 Vertical (should fail) ---");
    let v33 = generate_vertical(33);
    let _ = encode_lossy_rgb8(&v33, 33, 33, 85.0)?;

    Ok(())
}
