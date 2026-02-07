use jxl_encoder::encoder::encode_lossy_rgb8;
/// Test specific alphabet sizes with single-cluster encoding
///
/// Since vertical 33x33 (alphabet_size=54) fails but similar sizes work,
/// this test isolates whether the issue is with specific alphabet sizes.
use std::io::Cursor;

fn generate_test_image(width: usize, height: usize, seed: u8) -> Vec<u8> {
    let mut data = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            // Create a gradient that will produce different token distributions
            // based on the seed value
            let val = ((y.wrapping_mul(seed as usize).wrapping_add(x) * 255 / (width + height))
                % 256) as u8;
            let idx = (y * width + x) * 3;
            data[idx] = val;
            data[idx + 1] = val;
            data[idx + 2] = val;
        }
    }
    data
}

fn try_decode(jxl_data: &[u8]) -> Result<(), String> {
    let result = jxl_oxide::JxlImage::builder()
        .read(Cursor::new(jxl_data))
        .and_then(|img| img.render_frame(0));

    match result {
        Ok(_) => Ok(()),
        Err(e) => Err(format!("{}", e)),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("=== Testing different image patterns to target alphabet sizes ===\n");

    // Test many different seeds at 33x33 to find more failing cases
    let mut results = Vec::new();

    for seed in 1u8..=20 {
        let data = generate_test_image(33, 33, seed);
        let jxl = encode_lossy_rgb8(&data, 33, 33, 85.0)?;
        let ok = try_decode(&jxl).is_ok();
        results.push((seed, ok));

        eprintln!("Seed {}: {}", seed, if ok { "OK" } else { "FAIL" });
    }

    eprintln!("\n=== Summary ===");
    let fails: Vec<_> = results.iter().filter(|(_, ok)| !ok).collect();
    eprintln!(
        "Failures: {:?}",
        fails.iter().map(|(s, _)| s).collect::<Vec<_>>()
    );

    // Now test specific sizes that should trigger alphabet_size=54
    eprintln!("\n=== Testing specific sizes to find alphabet_size=54 cases ===");

    // Pure vertical gradient at different sizes
    for size in [30, 31, 32, 33, 34, 35, 36, 40, 48, 64] {
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

        let jxl = encode_lossy_rgb8(&data, size, size, 85.0)?;
        let result = try_decode(&jxl);

        eprintln!(
            "{}x{} vertical: {} ({} bytes)",
            size,
            size,
            if result.is_ok() { "OK" } else { "FAIL" },
            jxl.len()
        );
    }

    Ok(())
}
