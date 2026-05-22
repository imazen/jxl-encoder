//! W44-201 SSIM2 verification: confirm bucket disable doesn't affect decoded
//! quality. The scan order only affects WHERE the bits go (encoded
//! Lehmer codes), not the coefficient values themselves. Decoded pixels
//! should be byte-identical between A/B/C variants for the SAME image.

use jxl_encoder::api::{EncoderStrategy, LossyConfig, PixelLayout};
use std::path::Path;

fn load_png(path: &Path) -> (Vec<u8>, u32, u32) {
    let img = image::open(path).expect("png decode");
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    (rgb.into_raw(), w, h)
}

fn encode(pixels: &[u8], w: u32, h: u32, disable_spec: Option<&str>, distance: f32) -> Vec<u8> {
    unsafe {
        match disable_spec {
            Some(spec) => std::env::set_var("JXL_W44_201_DISABLE_BUCKETS", spec),
            None => std::env::remove_var("JXL_W44_201_DISABLE_BUCKETS"),
        }
    }
    let cfg = LossyConfig::new(distance)
        .with_effort(7)
        .with_strategy(EncoderStrategy::Zenjxl);
    cfg.encode_request(w, h, PixelLayout::Rgb8)
        .encode(pixels)
        .expect("encode")
}

fn decode_to_rgb8(bytes: &[u8]) -> (Vec<u8>, u32, u32) {
    use std::io::Cursor;
    let cursor = Cursor::new(bytes.to_vec());
    let image = jxl_oxide::JxlImage::builder().read(cursor).expect("oxide read");
    let render = image.render_frame(0).expect("oxide render");
    let stream = render.stream();
    let w = stream.width();
    let h = stream.height();
    // stream is RGB f32; need to convert to u8
    let frame = render.image_all_channels();
    let buf = frame.buf();
    let mut rgb = Vec::with_capacity((w * h * 3) as usize);
    for &v in buf {
        rgb.push((v.clamp(0.0, 1.0) * 255.0).round() as u8);
    }
    (rgb, w, h)
}

fn main() {
    let imgs = &[
        ("3637739", "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/3637739.png"),
        ("1420710", "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1420710.png"),
    ];
    for (label, path) in imgs {
        let (pixels, w, h) = load_png(Path::new(path));
        for d in &[2.0_f32, 4.0, 6.0] {
            let buf_a = encode(&pixels, w, h, None, *d);
            let buf_c = encode(&pixels, w, h, Some("3,6"), *d);
            let (dec_a, _, _) = decode_to_rgb8(&buf_a);
            let (dec_c, _, _) = decode_to_rgb8(&buf_c);
            let identical = dec_a == dec_c;
            let bytes_a = buf_a.len();
            let bytes_c = buf_c.len();
            // Compute MSE/SSE if not identical
            let mut max_diff: u8 = 0;
            let mut sum_sq_diff: u64 = 0;
            for (a, c) in dec_a.iter().zip(dec_c.iter()) {
                let d = a.abs_diff(*c);
                max_diff = max_diff.max(d);
                sum_sq_diff += (d as u64) * (d as u64);
            }
            let mse = sum_sq_diff as f64 / dec_a.len() as f64;
            println!(
                "{}\td={}\tA={} C={} pixel_identical={} max_diff={} mse={:.4}",
                label, d, bytes_a, bytes_c, identical, max_diff, mse
            );
        }
    }
}
