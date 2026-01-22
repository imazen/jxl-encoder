// Compare jxl-encoder-rs vs libjpegli quality/size

use std::io::Write;
use std::process::Command;

fn main() {
    // Test image sizes
    let sizes = [(64, 64), (128, 128), (256, 256)];

    println!("Format      Size      Bytes    SSIM2    Bits/px");
    println!("==============================================");

    for (width, height) in sizes {
        // Create a smooth gradient test image (good for lossy compression)
        let mut data = vec![0u8; width * height * 3];
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) * 3;
                // Diagonal gradient with some color variation
                let r = ((x + y) * 255 / (width + height - 2)) as u8;
                let g = (x * 255 / (width - 1)) as u8;
                let b = (y * 255 / (height - 1)) as u8;
                data[idx] = r;
                data[idx + 1] = g;
                data[idx + 2] = b;
            }
        }

        // Save as PNG for cjpegli input
        let png_path = format!("/tmp/test_{}x{}.png", width, height);
        save_png(&png_path, &data, width, height);

        // Encode with our JXL encoder
        let jxl_result = encode_jxl(&data, width, height, 1.0);
        let jxl_ssim = compute_ssim(&data, &jxl_result, width, height, "jxl");
        let jxl_bpp = (jxl_result.len() * 8) as f64 / (width * height) as f64;
        println!(
            "JXL d=1.0   {}x{}   {:5}    {:.2}    {:.3}",
            width,
            height,
            jxl_result.len(),
            jxl_ssim.unwrap_or(-1.0),
            jxl_bpp
        );

        // Encode with cjpegli at quality 90
        let jpeg_path = format!("/tmp/test_{}x{}_q90.jpg", width, height);
        let status = Command::new("cjpegli")
            .args([&png_path, &jpeg_path, "-q", "90"])
            .status()
            .expect("cjpegli failed");

        if status.success() {
            let jpeg_bytes = std::fs::read(&jpeg_path).unwrap();
            let jpeg_ssim = compute_ssim_jpeg(&data, &jpeg_path, width, height);
            let jpeg_bpp = (jpeg_bytes.len() * 8) as f64 / (width * height) as f64;
            println!(
                "JPEGLI q90  {}x{}   {:5}    {:.2}    {:.3}",
                width,
                height,
                jpeg_bytes.len(),
                jpeg_ssim.unwrap_or(-1.0),
                jpeg_bpp
            );
        }

        // Encode with cjpegli at quality 80
        let jpeg_path = format!("/tmp/test_{}x{}_q80.jpg", width, height);
        let status = Command::new("cjpegli")
            .args([&png_path, &jpeg_path, "-q", "80"])
            .status()
            .expect("cjpegli failed");

        if status.success() {
            let jpeg_bytes = std::fs::read(&jpeg_path).unwrap();
            let jpeg_ssim = compute_ssim_jpeg(&data, &jpeg_path, width, height);
            let jpeg_bpp = (jpeg_bytes.len() * 8) as f64 / (width * height) as f64;
            println!(
                "JPEGLI q80  {}x{}   {:5}    {:.2}    {:.3}",
                width,
                height,
                jpeg_bytes.len(),
                jpeg_ssim.unwrap_or(-1.0),
                jpeg_bpp
            );
        }

        println!();
    }
}

fn save_png(path: &str, data: &[u8], width: usize, height: usize) {
    let file = std::fs::File::create(path).unwrap();
    let mut encoder = png::Encoder::new(file, width as u32, height as u32);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().unwrap();
    writer.write_image_data(data).unwrap();
}

fn encode_jxl(data: &[u8], width: usize, height: usize, distance: f32) -> Vec<u8> {
    jxl_enc::encoder::encode_lossy_rgb8(data, width, height, distance).unwrap()
}

fn compute_ssim(
    original: &[u8],
    encoded: &[u8],
    width: usize,
    height: usize,
    ext: &str,
) -> Option<f64> {
    use ssimulacra2::{ColorPrimaries, Rgb, TransferCharacteristic, compute_frame_ssimulacra2};

    // Decode JXL
    let decoded = if ext == "jxl" {
        let img = jxl_oxide::JxlImage::builder()
            .read(std::io::Cursor::new(encoded))
            .ok()?;
        let frame = img.render_frame(0).ok()?;
        let fb = frame.image_all_channels();
        let channels = fb.channels();
        let buf = fb.buf();
        (0..(width * height))
            .map(|i| {
                let idx = i * channels;
                [buf[idx], buf[idx + 1], buf[idx + 2]]
            })
            .collect::<Vec<_>>()
    } else {
        return None;
    };

    // Convert original to f32
    let original_f32: Vec<[f32; 3]> = (0..(width * height))
        .map(|i| {
            [
                original[i * 3] as f32 / 255.0,
                original[i * 3 + 1] as f32 / 255.0,
                original[i * 3 + 2] as f32 / 255.0,
            ]
        })
        .collect();

    let source = Rgb::new(
        original_f32,
        width,
        height,
        TransferCharacteristic::SRGB,
        ColorPrimaries::BT709,
    )
    .ok()?;

    let distorted = Rgb::new(
        decoded,
        width,
        height,
        TransferCharacteristic::SRGB,
        ColorPrimaries::BT709,
    )
    .ok()?;

    compute_frame_ssimulacra2(source, distorted).ok()
}

fn compute_ssim_jpeg(original: &[u8], jpeg_path: &str, width: usize, height: usize) -> Option<f64> {
    use ssimulacra2::{ColorPrimaries, Rgb, TransferCharacteristic, compute_frame_ssimulacra2};

    // Decode JPEG using image crate
    let img = image::open(jpeg_path).ok()?;
    let rgb = img.to_rgb8();

    let decoded: Vec<[f32; 3]> = rgb
        .pixels()
        .map(|p| {
            [
                p[0] as f32 / 255.0,
                p[1] as f32 / 255.0,
                p[2] as f32 / 255.0,
            ]
        })
        .collect();

    // Convert original to f32
    let original_f32: Vec<[f32; 3]> = (0..(width * height))
        .map(|i| {
            [
                original[i * 3] as f32 / 255.0,
                original[i * 3 + 1] as f32 / 255.0,
                original[i * 3 + 2] as f32 / 255.0,
            ]
        })
        .collect();

    let source = Rgb::new(
        original_f32,
        width,
        height,
        TransferCharacteristic::SRGB,
        ColorPrimaries::BT709,
    )
    .ok()?;

    let distorted = Rgb::new(
        decoded,
        width,
        height,
        TransferCharacteristic::SRGB,
        ColorPrimaries::BT709,
    )
    .ok()?;

    compute_frame_ssimulacra2(source, distorted).ok()
}
