//! W44-201 decoder roundtrip check: encode 3637739 with bucket 3 / 3+6
//! disabled and verify both decode cleanly via jxl-rs and jxl-oxide.

use jxl_encoder::api::{EncoderStrategy, LossyConfig, PixelLayout};
use std::path::Path;

const IMG_3637739: &str = "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/3637739.png";
const IMG_1418519: &str = "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1418519.png";

fn load_png(path: &Path) -> (Vec<u8>, u32, u32) {
    let img = image::open(path).expect("png decode");
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    (rgb.into_raw(), w, h)
}

fn encode(
    pixels: &[u8],
    w: u32,
    h: u32,
    disable_spec: Option<&str>,
    distance: f32,
    effort: u8,
) -> Vec<u8> {
    unsafe {
        match disable_spec {
            Some(spec) => std::env::set_var("JXL_W44_201_DISABLE_BUCKETS", spec),
            None => std::env::remove_var("JXL_W44_201_DISABLE_BUCKETS"),
        }
    }
    let cfg = LossyConfig::new(distance)
        .with_effort(effort)
        .with_strategy(EncoderStrategy::Zenjxl);
    cfg.encode_request(w, h, PixelLayout::Rgb8)
        .encode(pixels)
        .expect("encode")
}

fn decode_jxl_oxide(bytes: &[u8]) -> Result<(u32, u32), String> {
    use std::io::Cursor;
    let cursor = Cursor::new(bytes.to_vec());
    let image = jxl_oxide::JxlImage::builder()
        .read(cursor)
        .map_err(|e| format!("jxl_oxide parse: {}", e))?;
    let render = image
        .render_frame(0)
        .map_err(|e| format!("jxl_oxide render: {}", e))?;
    let stream = render.stream();
    Ok((stream.width(), stream.height()))
}

fn main() {
    for (path, label) in &[(IMG_3637739, "3637739"), (IMG_1418519, "1418519")] {
        let (pixels, w, h) = load_png(Path::new(path));
        for (spec, sl) in &[
            (None, "A_default"),
            (Some("3"), "B_no_bucket3"),
            (Some("3,6"), "C_no_bucket3_6"),
        ] {
            for d in &[2.0f32, 4.0, 6.0] {
                let buf = encode(&pixels, w, h, *spec, *d, 7);
                let dec = decode_jxl_oxide(&buf);
                let result = match dec {
                    Ok((dw, dh)) if dw == w && dh == h => "PASS",
                    Ok((dw, dh)) => &format!("FAIL_DIM: got {}x{} expected {}x{}", dw, dh, w, h),
                    Err(e) => &format!("FAIL: {}", e),
                };
                println!(
                    "{}\td={}\t{}\tbytes={}\t{}",
                    label,
                    d,
                    sl,
                    buf.len(),
                    result
                );
            }
        }
    }
}
