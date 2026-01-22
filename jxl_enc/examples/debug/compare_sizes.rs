/// Compare 8x8 vs 33x33 vertical gradient to find the difference
use jxl_enc::encoder::encode_lossy_rgb8;
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

fn test_size(size: usize) {
    eprintln!("\n{}", "=".repeat(60));
    eprintln!("Testing {}x{} vertical gradient", size, size);
    eprintln!("{}\n", "=".repeat(60));

    let data = generate_vertical(size);
    match encode_lossy_rgb8(&data, size, size, 85.0) {
        Ok(jxl) => {
            eprintln!("Encoded {} bytes", jxl.len());

            // Try to decode
            let result = jxl_oxide::JxlImage::builder().read(Cursor::new(&jxl));
            match result {
                Ok(img) => {
                    eprintln!("Parse OK, attempting render...");
                    match img.render_frame(0) {
                        Ok(_) => eprintln!(">>> RESULT: OK <<<"),
                        Err(e) => eprintln!(">>> RESULT: RENDER FAIL: {:?} <<<", e),
                    }
                }
                Err(e) => eprintln!(">>> RESULT: PARSE FAIL: {:?} <<<", e),
            }
        }
        Err(e) => eprintln!(">>> RESULT: ENCODE FAIL: {:?} <<<", e),
    }
}

fn main() {
    // Test 8x8 (works with alphabet=54) vs 33x33 (fails with alphabet=54)
    // to see what tokens are emitted
    test_size(8);
    test_size(33);
}
