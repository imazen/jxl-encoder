//! Design C: why does our SSIM2 DIP at exactly d=3.0 on 5058 while cjxl's
//! ladder is monotone through the same points?
//!
//! Measured shape (e8, 1024 crop): ours 89.403 @ d=2.5 -> **86.655 @ d=3.0** ->
//! 88.168 @ d=3.5, while cjxl runs 89.650 -> 89.326 -> 88.342 with no dip. Ours
//! also spends MORE bytes at d=3 (39,322) than at d=3.5 (36,982) for LESS
//! quality, so this is an efficiency failure, not just a targeting offset.
//!
//! Prime suspect: dot detection, gated at `distance >= 3.0`
//! (`kMinButteraugliForDots`) and effort >= 7 — the boundary coincides exactly.
use butteraugli::{ButteraugliParams, butteraugli_linear, srgb_to_linear};
use imgref::Img;
use jxl_encoder::api::{LossyConfig, PixelLayout};
use rgb::RGB;

/// Decode to LINEAR sRGB — butteraugli's required input domain, and the
/// opposite of what SSIM2 wants (which linearises internally from sRGB).
/// Feeding the wrong one is the documented way to get garbage from either.
fn decode_linear(bytes: &[u8]) -> Option<(usize, usize, Vec<f32>)> {
    let mut img = jxl_oxide::JxlImage::builder()
        .read(std::io::Cursor::new(bytes))
        .ok()?;
    img.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
        jxl_oxide::RenderingIntent::Relative,
    ));
    let render = img.render_frame(0).ok()?;
    let fb = render.image_all_channels();
    Some((fb.width(), fb.height(), fb.buf().to_vec()))
}

fn butteraugli_of(bytes: &[u8], orig_linear: &Img<Vec<RGB<f32>>>) -> f64 {
    let Some((dw, dh, lin)) = decode_linear(bytes) else {
        return f64::NAN;
    };
    let px: Vec<RGB<f32>> = lin.chunks(3).map(|c| RGB::new(c[0], c[1], c[2])).collect();
    let img: Img<Vec<RGB<f32>>> = Img::new(px, dw, dh);
    butteraugli_linear(
        orig_linear.as_ref(),
        img.as_ref(),
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
    let path = std::env::args()
        .nth(1)
        .expect("usage: d3_dip_probe <png> [size] [effort]");
    let n: u32 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1024);
    let e: u8 = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let rgb = image::open(&path).expect("open").to_rgb8();
    let (x0, y0) = (
        (rgb.width().saturating_sub(n)) / 2,
        (rgb.height().saturating_sub(n)) / 2,
    );
    let src: Vec<u8> = image::imageops::crop_imm(&rgb, x0, y0, n, n)
        .to_image()
        .as_raw()
        .clone();

    println!("{} @ {n}x{n} e{e}", path.rsplit('/').next().unwrap());
    // butteraugli wants LINEAR input; ssim2 wants sRGB and linearises
    // internally. Feeding either the wrong domain is the documented way to get
    // garbage scores, so each reference is built in its own domain.
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

    println!(
        "{:>6}{:>17}{:>17}{:>17}{:>17}   (bytes/ssim2/butteraugli; the loop optimises butteraugli)",
        "d", "it=0", "it=2", "it=3", "it=6"
    );
    let dists: Vec<f32> = std::env::args()
        .nth(4)
        .map(|s| s.split(',').filter_map(|x| x.parse().ok()).collect())
        .unwrap_or_else(|| vec![2.5, 2.75, 3.0, 3.25, 3.5]);
    for &d in &dists {
        print!("{d:>6}");
        for iters in [0u32, 2, 3, 6] {
            let enc = LossyConfig::new(d)
                .with_effort(e)
                .with_butteraugli_iters(iters)
                .encode_request(n, n, PixelLayout::Rgb8)
                .encode(&src)
                .expect("encode");
            print!(
                "  {:>6}/{:>5.1}/{:>5.3}",
                enc.len(),
                score(&enc, &src, n),
                butteraugli_of(&enc, &orig_lin)
            );
        }
        println!();
    }
}
