//! scratch codegen-profiling harness: lossy/lossless, default (Zenjxl) or strict.
//! usage: cg_profile <png-or-ppm> <lossy-effort|lossless> <distance-or-effort>
use jxl_encoder::api::EncoderStrategy;
use jxl_encoder::{LosslessConfig, LossyConfig, PixelLayout};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let input = &args[1];
    let effort: u8 = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(7);
    let distance: f32 = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(1.0);
    let strict = std::env::var("STRICT").is_ok();
    let lossless = std::env::var("LOSSLESS").is_ok();

    let (pixels, w, h): (Vec<u8>, u32, u32) = if input.ends_with(".ppm") {
        let bytes = std::fs::read(input).unwrap();
        let mut idx = 0;
        let mut fields = 0;
        while fields < 4 {
            while bytes[idx].is_ascii_whitespace() {
                idx += 1;
            }
            if bytes[idx] == 35 {
                while bytes[idx..idx + 1] != *b"\n" {
                    idx += 1;
                }
                continue;
            }
            while !bytes[idx].is_ascii_whitespace() {
                idx += 1;
            }
            fields += 1;
        }
        idx += 1;
        let text = std::str::from_utf8(&bytes[..idx]).unwrap();
        let mut toks = text.split_ascii_whitespace();
        assert_eq!(toks.next().unwrap(), "P6");
        let w: u32 = toks.next().unwrap().parse().unwrap();
        let h: u32 = toks.next().unwrap().parse().unwrap();
        toks.next().unwrap();
        (bytes[idx..].to_vec(), w, h)
    } else {
        let img = image::open(input).unwrap().to_rgb8();
        let (w, h) = img.dimensions();
        (img.into_raw(), w, h)
    };

    let _t0 = std::time::Instant::now();
    let out = if lossless {
        LosslessConfig::new()
            .with_effort(effort)
            .encode(&pixels, w, h, PixelLayout::Rgb8)
            .expect("encode failed")
    } else {
        let mut cfg = LossyConfig::new(distance).with_effort(effort);
        if strict {
            cfg = cfg.with_strategy(EncoderStrategy::Libjxl);
        }
        cfg.encode(&pixels, w, h, PixelLayout::Rgb8)
            .expect("encode failed")
    };
    eprintln!("encode={:?} {} bytes", _t0.elapsed(), out.len());
    if let Ok(p) = std::env::var("CG_OUT") {
        std::fs::write(p, &out).unwrap();
    }
}
