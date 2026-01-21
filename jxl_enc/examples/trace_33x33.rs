/// Trace 33x33 encoding in detail to find the bug
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("=== Tracing 33x33 vertical gradient encoding ===\n");

    let size = 33;
    let data = generate_vertical(size);

    eprintln!("Encoding...");
    let jxl = encode_lossy_rgb8(&data, size, size, 85.0)?;

    eprintln!("\nEncoded {} bytes", jxl.len());
    std::fs::write("/tmp/trace_33x33.jxl", &jxl)?;

    // Try to decode
    eprintln!("\n=== Attempting decode with jxl-oxide ===\n");
    let result = jxl_oxide::JxlImage::builder().read(Cursor::new(&jxl));

    match result {
        Ok(img) => {
            eprintln!("Parse OK, attempting render...");
            match img.render_frame(0) {
                Ok(_) => eprintln!("Render OK"),
                Err(e) => eprintln!("Render FAIL: {:?}", e),
            }
        }
        Err(e) => eprintln!("Parse FAIL: {:?}", e),
    }

    // Dump hex for analysis
    eprintln!("\n=== First 300 bytes hex dump ===");
    for (i, chunk) in jxl.chunks(16).take(19).enumerate() {
        eprint!("{:04x}: ", i * 16);
        for b in chunk {
            eprint!("{:02x} ", b);
        }
        eprintln!();
    }

    Ok(())
}
