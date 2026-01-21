use jxl_enc::encoder::encode_lossy_rgb8;
/// Test with jxl-rs decoder
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("=== Testing 33x33 vertical gradient ===\n");

    let size = 33;
    let data = generate_vertical(size);
    let jxl = encode_lossy_rgb8(&data, size, size, 85.0)?;

    eprintln!("Encoded {} bytes", jxl.len());
    std::fs::write("/tmp/v33_test.jxl", &jxl)?;

    // Test with jxl-oxide
    eprintln!("\n--- jxl-oxide decode ---");
    let result = jxl_oxide::JxlImage::builder().read(Cursor::new(&jxl));

    match result {
        Ok(img) => {
            eprintln!("Parsed successfully, attempting render...");
            match img.render_frame(0) {
                Ok(_) => eprintln!("Render OK"),
                Err(e) => eprintln!("Render failed: {:?}", e),
            }
        }
        Err(e) => eprintln!("Parse failed: {:?}", e),
    }

    // Save hex dump of first 200 bytes
    eprintln!("\n--- First 200 bytes hex dump ---");
    for (i, chunk) in jxl.chunks(16).take(13).enumerate() {
        eprint!("{:04x}: ", i * 16);
        for b in chunk {
            eprint!("{:02x} ", b);
        }
        eprintln!();
    }

    // Try to parse manually to find the exact failure point
    eprintln!("\n--- Manual parsing analysis ---");
    // JXL signature is 0xFF 0x0A
    if jxl[0] == 0xFF && jxl[1] == 0x0A {
        eprintln!("Signature: OK (FF 0A)");
    } else {
        eprintln!("Signature: WRONG ({:02X} {:02X})", jxl[0], jxl[1]);
    }

    Ok(())
}
