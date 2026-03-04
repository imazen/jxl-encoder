// Quick quality comparison: LfFrame (lossy modular) vs no-LfFrame
// Run with: cargo test -p jxl-encoder --test lf_quality --release -- --ignored --nocapture

use butteraugli::{ButteraugliParams, butteraugli_linear, srgb_to_linear};
use image::GenericImageView;
use imgref::Img;
use rgb::RGB;
use std::io::Cursor;
use std::path::PathBuf;

/// Convert linear light value to sRGB u8 using the correct sRGB transfer function.
fn linear_to_srgb_u8(linear: f32) -> u8 {
    let c = linear.clamp(0.0, 1.0);
    let srgb = if c <= 0.003_130_8 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (srgb * 255.0).round() as u8
}

fn decode_jxl_linear(bytes: &[u8]) -> Option<(usize, usize, Vec<f32>)> {
    let reader = Cursor::new(bytes);
    let mut jxl_image = jxl_oxide::JxlImage::builder().read(reader).ok()?;
    jxl_image.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
        jxl_oxide::RenderingIntent::Relative,
    ));
    let render = jxl_image.render_frame(0).ok()?;
    let fb = render.image_all_channels();
    Some((fb.width(), fb.height(), fb.buf().to_vec()))
}

fn compute_metrics(
    orig_linear: &Img<Vec<RGB<f32>>>,
    orig_srgb: &Img<Vec<[u8; 3]>>,
    jxl_bytes: &[u8],
    w: usize,
    h: usize,
) -> (f64, f64) {
    let (_, _, decoded) = decode_jxl_linear(jxl_bytes).expect("decode failed");

    let dec_pixels: Vec<RGB<f32>> = decoded
        .chunks(3)
        .map(|c| RGB::new(c[0], c[1], c[2]))
        .collect();
    let dec_img = Img::new(dec_pixels, w, h);
    let bfly = butteraugli_linear(
        orig_linear.as_ref(),
        dec_img.as_ref(),
        &ButteraugliParams::default(),
    )
    .expect("butteraugli failed")
    .score;

    let decoded_srgb: Vec<[u8; 3]> = decoded
        .chunks(3)
        .map(|c| {
            [
                linear_to_srgb_u8(c[0]),
                linear_to_srgb_u8(c[1]),
                linear_to_srgb_u8(c[2]),
            ]
        })
        .collect();
    let dec_srgb_img = Img::new(decoded_srgb, w, h);
    let ssim2 = fast_ssim2::compute_ssimulacra2(orig_srgb.as_ref(), dec_srgb_img.as_ref())
        .expect("ssim2 failed");

    (bfly, ssim2)
}

#[test]
#[ignore]
fn test_lf_frame_quality() {
    use jxl_encoder::api::{LossyConfig, PixelLayout};

    let images = [
        &format!(
            "{}/CID22/CID22-512/validation/1025469.png",
            jxl_encoder::test_helpers::corpus_dir().display()
        ),
        &format!(
            "{}/CID22/CID22-512/validation/1044329.png",
            jxl_encoder::test_helpers::corpus_dir().display()
        ),
        &format!(
            "{}/CID22/CID22-512/validation/1189261.png",
            jxl_encoder::test_helpers::corpus_dir().display()
        ),
    ];

    println!(
        "\n{:<10} {:>8} {:>8} {:>8} | {:>8} {:>8} {:>8} | {:>8} {:>8}",
        "Image", "NoLF sz", "bfly", "ssim2", "LF sz", "bfly", "ssim2", "sz %", "bfly %"
    );
    println!("{}", "-".repeat(100));

    for img_path in &images {
        let path = PathBuf::from(img_path);
        if !path.exists() {
            eprintln!("Skipping {}: not found", img_path);
            continue;
        }
        let name = path.file_stem().unwrap().to_string_lossy();

        let img = image::open(&path).expect("open failed");
        let (w, h) = img.dimensions();
        let (w, h) = (w as usize, h as usize);
        let rgb = img.to_rgb8();

        let original_srgb: Vec<[u8; 3]> = rgb.pixels().map(|p| [p[0], p[1], p[2]]).collect();
        let orig_srgb_img = Img::new(original_srgb, w, h);

        let linear_rgb: Vec<f32> = rgb
            .pixels()
            .flat_map(|p| {
                [
                    srgb_to_linear(p[0]),
                    srgb_to_linear(p[1]),
                    srgb_to_linear(p[2]),
                ]
            })
            .collect();
        let orig_pixels: Vec<RGB<f32>> = linear_rgb
            .chunks(3)
            .map(|c| RGB::new(c[0], c[1], c[2]))
            .collect();
        let orig_linear_img = Img::new(orig_pixels, w, h);

        let pixels_u8: Vec<u8> = rgb.pixels().flat_map(|p| [p[0], p[1], p[2]]).collect();

        // Encode without LfFrame
        let cfg_no_lf = LossyConfig::new(2.0);
        let no_lf_bytes = cfg_no_lf
            .encode(&pixels_u8, w as u32, h as u32, PixelLayout::Rgb8)
            .expect("encode no-lf");
        let (no_lf_bfly, no_lf_ssim2) =
            compute_metrics(&orig_linear_img, &orig_srgb_img, &no_lf_bytes, w, h);

        // Encode with LfFrame (lossy modular)
        let cfg_lf = LossyConfig::new(2.0).with_lf_frame(true);
        let lf_bytes = cfg_lf
            .encode(&pixels_u8, w as u32, h as u32, PixelLayout::Rgb8)
            .expect("encode lf");
        let (lf_bfly, lf_ssim2) =
            compute_metrics(&orig_linear_img, &orig_srgb_img, &lf_bytes, w, h);

        let sz_pct = (lf_bytes.len() as f64 / no_lf_bytes.len() as f64 - 1.0) * 100.0;
        let bfly_pct = (lf_bfly / no_lf_bfly - 1.0) * 100.0;

        println!(
            "{:<10} {:>8} {:>8.3} {:>8.2} | {:>8} {:>8.3} {:>8.2} | {:>7.1}% {:>7.1}%",
            &name[..name.len().min(10)],
            no_lf_bytes.len(),
            no_lf_bfly,
            no_lf_ssim2,
            lf_bytes.len(),
            lf_bfly,
            lf_ssim2,
            sz_pct,
            bfly_pct
        );
    }
}
