use std::process::Command;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let djxl_path = "/home/lilith/work/jxl-efforts/libjxl/build/tools/djxl";

    for h in 32..=42 {
        let w = 32;
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

        let encoded = jxl_encoder::encoder::encode_lossy_rgb8(&data, w, h, 1.0)?;
        let out_path = format!("/tmp/test_32x{}.jxl", h);
        std::fs::write(&out_path, &encoded)?;

        let djxl_out = format!("/tmp/decoded_32x{}.ppm", h);
        let status = Command::new(djxl_path)
            .args([&out_path, &djxl_out])
            .output()?;

        let result = if status.status.success() {
            "OK"
        } else {
            "FAIL"
        };

        println!("32x{}: {} ({} bytes)", h, result, encoded.len());
    }

    Ok(())
}
