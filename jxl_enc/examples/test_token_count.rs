/// Test relationship between token count and alphabet size
use jxl_enc::encoder::encode_lossy_rgb8;
use std::io::Cursor;

fn generate_gradient(size: usize, steepness: f32) -> Vec<u8> {
    let mut data = vec![0u8; size * size * 3];
    for y in 0..size {
        let val = ((y as f32 * steepness) as u32).min(255) as u8;
        for x in 0..size {
            let idx = (y * size + x) * 3;
            data[idx] = val;
            data[idx + 1] = val;
            data[idx + 2] = val;
        }
    }
    data
}

fn try_decode(jxl_data: &[u8]) -> String {
    match jxl_oxide::JxlImage::builder().read(Cursor::new(jxl_data)) {
        Ok(img) => match img.render_frame(0) {
            Ok(_) => "OK".to_string(),
            Err(e) => format!("FAIL"),
        },
        Err(e) => format!("Parse FAIL"),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Test different block counts
    // 2x2 blocks = 16x16 pixels, 3x3 blocks = 24x24 pixels, etc.

    eprintln!("Testing different image sizes and gradients to find failure pattern...\n");

    for &blocks_dim in &[2, 3, 4, 5] {
        let size = blocks_dim * 8;
        let block_count = blocks_dim * blocks_dim;
        eprintln!("=== {} blocks ({}x{}) ===", block_count, size, size);

        for &steepness in &[4.0, 8.0, 12.0, 16.0, 20.0, 30.0] {
            let data = generate_gradient(size, steepness);
            // Suppress the eprintln by redirecting stderr temporarily
            let jxl = encode_lossy_rgb8(&data, size, size, 85.0)?;
            let result = try_decode(&jxl);
            eprintln!("  steep={}: {}", steepness, result);
        }
        eprintln!();
    }

    Ok(())
}
