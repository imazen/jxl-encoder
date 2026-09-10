//! Why is e5 strictly dominated by e3 at d=1 on some photos?
//!
//! Design B's gate found two cells where e5 costs MORE bytes AND scores WORSE
//! SSIM2 than e3 — worse on both axes, which no effort level should ever be.
//! Six knobs turn on between e3 and e5: `custom_orders` at effort 4, and
//! `gaborish`, `pixel_domain_loss`, `ac_strategy_enabled`, `try_dct16` and
//! `try_dct32` at effort 5. This isolates which one is responsible by disabling
//! the two that have public setters, and reports the rest as unreachable from
//! here. It also runs cjxl v0.12 over the same crop, because "is this ours or
//! inherited?" changes the disposition completely.
use butteraugli::{ButteraugliParams, butteraugli_linear, srgb_to_linear};
use imgref::Img;
use jxl_encoder::api::{LossyConfig, PixelLayout};
use rgb::RGB;

/// Delivered butteraugli (LINEAR input, lower = better).
///
/// Gaborish is calibrated against BUTTERAUGLI, not SSIM2, so "gaborish costs
/// SSIM2" cannot be read as a quality loss without checking the metric it was
/// actually tuned for. If butteraugli improves while SSIM2 falls, the filter is
/// doing its job and the SSIM2 delta is a metric disagreement.
fn bfly_of(encoded: &[u8], orig_linear: &Img<Vec<RGB<f32>>>) -> f64 {
    let Ok(mut img) = jxl_oxide::JxlImage::builder().read(std::io::Cursor::new(encoded)) else {
        return f64::NAN;
    };
    img.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
        jxl_oxide::RenderingIntent::Relative,
    ));
    let Ok(render) = img.render_frame(0) else {
        return f64::NAN;
    };
    let fb = render.image_all_channels();
    let (buf, ch) = (fb.buf(), fb.channels());
    if ch < 3 {
        return f64::NAN;
    }
    let px: Vec<RGB<f32>> = (0..fb.width() * fb.height())
        .map(|i| RGB::new(buf[i * ch], buf[i * ch + 1], buf[i * ch + 2]))
        .collect();
    butteraugli_linear(
        orig_linear.as_ref(),
        Img::new(px, fb.width(), fb.height()).as_ref(),
        &ButteraugliParams::default(),
    )
    .map(|r| r.score)
    .unwrap_or(f64::NAN)
}

fn score(encoded: &[u8], src: &[u8], n: u32) -> f64 {
    use fast_ssim2::compute_ssimulacra2;
    use imgref::ImgVec;
    let Ok(image) = jxl_oxide::JxlImage::builder().read(std::io::Cursor::new(encoded)) else {
        return f64::NAN;
    };
    let Ok(render) = image.render_frame(0) else {
        return f64::NAN;
    };
    let fb = render.image_all_channels();
    let (buf, ch) = (fb.buf(), fb.channels());
    let px = (n * n) as usize;
    if ch < 3 || buf.len() < px * ch {
        return f64::NAN;
    }
    let dec: Vec<[u8; 3]> = (0..px)
        .map(|i| {
            let o = i * ch;
            [
                (buf[o] * 255.0).clamp(0.0, 255.0) as u8,
                (buf[o + 1] * 255.0).clamp(0.0, 255.0) as u8,
                (buf[o + 2] * 255.0).clamp(0.0, 255.0) as u8,
            ]
        })
        .collect();
    let orig: Vec<[u8; 3]> = src.as_chunks::<3>().0.to_vec();
    compute_ssimulacra2(
        ImgVec::new(orig, n as usize, n as usize).as_ref(),
        ImgVec::new(dec, n as usize, n as usize).as_ref(),
    )
    .unwrap_or(f64::NAN)
}

fn main() {
    let path = std::env::args().nth(1).expect("usage: probe <png> [size]");
    let n: u32 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(512);
    let rgb = image::open(&path).expect("open").to_rgb8();
    let (x0, y0) = ((rgb.width() - n) / 2, (rgb.height() - n) / 2);
    let src: Vec<u8> = image::imageops::crop_imm(&rgb, x0, y0, n, n)
        .to_image()
        .as_raw()
        .clone();

    let orig_lin_px: Vec<RGB<f32>> = src
        .as_chunks::<3>()
        .0
        .iter()
        .map(|c| {
            RGB::new(
                srgb_to_linear(c[0]),
                srgb_to_linear(c[1]),
                srgb_to_linear(c[2]),
            )
        })
        .collect();
    let orig_lin: Img<Vec<RGB<f32>>> = Img::new(orig_lin_px, n as usize, n as usize);

    let run = |label: &str, cfg: LossyConfig| {
        let enc = cfg
            .encode_request(n, n, PixelLayout::Rgb8)
            .encode(&src)
            .expect("encode");
        println!(
            "  {label:<34} {:>8} B   ssim2 {:>7.3}   bfly {:>7.4}",
            enc.len(),
            score(&enc, &src, n),
            bfly_of(&enc, &orig_lin)
        );
    };

    println!("{} @ {n}x{n}, d=1.0", path.rsplit('/').next().unwrap());
    run("e3 (baseline)", LossyConfig::new(1.0).with_effort(3));
    run("e5 (dominated)", LossyConfig::new(1.0).with_effort(5));
    run(
        "e5, gaborish OFF",
        LossyConfig::new(1.0).with_effort(5).with_gaborish(false),
    );
    run(
        "e5, pixel_domain_loss OFF",
        LossyConfig::new(1.0)
            .with_effort(5)
            .with_pixel_domain_loss(false),
    );
    run(
        "e5, both OFF",
        LossyConfig::new(1.0)
            .with_effort(5)
            .with_gaborish(false)
            .with_pixel_domain_loss(false),
    );
    run(
        "e4 (custom_orders on, rest off)",
        LossyConfig::new(1.0).with_effort(4),
    );
    run(
        "e5, adaptive gaborish ON",
        LossyConfig::new(1.0)
            .with_effort(5)
            .with_adaptive_gaborish(true),
    );

    // Is the e4 -> e5 domination OURS or inherited from libjxl? Same crop,
    // same distance, same efforts, through the pinned v0.12 reference.
    let png = std::env::var("HOME").unwrap() + "/tmp/e3e5_src.png";
    image::RgbImage::from_raw(n, n, src.clone())
        .unwrap()
        .save(&png)
        .unwrap();
    let cjxl = jxl_encoder::test_helpers::cjxl_path();
    println!("  --- cjxl v0.12, same crop ---");
    for e in [3u8, 4, 5, 7] {
        let out = format!("{}/tmp/e3e5_cj.jxl", std::env::var("HOME").unwrap());
        let st = std::process::Command::new(&cjxl)
            .args([
                png.as_str(),
                out.as_str(),
                "-d",
                "1.0",
                "-e",
                &e.to_string(),
                "--quiet",
            ])
            .output()
            .expect("cjxl");
        if st.status.success() {
            let b = std::fs::read(&out).unwrap();
            println!(
                "  {:<34} {:>8} B   ssim2 {:>8.3}",
                format!("cjxl e{e}"),
                b.len(),
                score(&b, &src, n)
            );
        }
    }
    println!("  (ac_strategy_enabled / try_dct16 / try_dct32 have no public setter)");
}
