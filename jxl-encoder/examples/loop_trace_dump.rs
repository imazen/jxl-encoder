//! Encode an image with the perceptual quantization loop and write its
//! per-iteration trace ([`jxl_encoder::LoopTrace`]) as one `.ltrace` file,
//! for the zenmetrics browser viewer's loop player.
//!
//! ```text
//! cargo run --release --example loop_trace_dump -- in.png out.ltrace [distance] [effort] [iters]
//! ```
//!
//! Defaults: distance 1.0, effort 7, 4 butteraugli iterations.
//!
//! Format (all little-endian): the magic `JXLTRACE1\n`; a `u32` header
//! length and that many bytes of UTF-8 JSON
//! (`width`, `height`, `distance`, `effort`, `iters`, `bytes`,
//! `xsize_blocks`, `ysize_blocks` and `frames`: one `{seed, iter, metric,
//! score, target, has_next}` per frame); the source as packed sRGB8 RGB,
//! `width·height·3` bytes; then per frame, as `f32` arrays: `quant_field`
//! (`xsize_blocks·ysize_blocks`), `next_quant_field` (same size, only when
//! `has_next`), `tile_dist` (same size) and `diffmap` (`width·height`), then
//! the reconstruction as packed sRGB8 RGB (`width·height·3` bytes, the
//! linear planes through the sRGB transfer, for display).

use std::io::Write;

use jxl_encoder::{LoopTrace, LossyConfig, PixelLayout};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: loop_trace_dump in.png out.ltrace [distance] [effort] [iters]");
        std::process::exit(2);
    }
    let distance: f32 = args.get(3).map_or(1.0, |s| s.parse().expect("distance"));
    let effort: u8 = args.get(4).map_or(7, |s| s.parse().expect("effort"));
    let iters: u32 = args.get(5).map_or(4, |s| s.parse().expect("iters"));
    let img = image::open(&args[1]).expect("decode input").to_rgb8();
    let (w, h) = (img.width(), img.height());
    let trace = LoopTrace::new();
    let bytes = LossyConfig::new(distance)
        .with_effort(effort)
        .with_butteraugli_iters(iters)
        .encode_request(w, h, PixelLayout::Rgb8)
        .with_loop_trace(&trace)
        .encode(img.as_raw())
        .expect("encode");
    let frames = trace.take();
    let first = frames.first().expect("the loop recorded no iterations");
    let header = serde_json::json!({
        "width": w,
        "height": h,
        "distance": distance,
        "effort": effort,
        "iters": iters,
        "bytes": bytes.len(),
        "xsize_blocks": first.xsize_blocks,
        "ysize_blocks": first.ysize_blocks,
        "frames": frames.iter().map(|f| serde_json::json!({
            "seed": f.seed,
            "iter": f.iter,
            "metric": f.metric,
            "score": f.score,
            "target": f.target,
            "has_next": f.next_quant_field.is_some(),
        })).collect::<Vec<_>>(),
    })
    .to_string();
    let mut out = std::io::BufWriter::new(std::fs::File::create(&args[2]).expect("create output"));
    let f32s = |out: &mut std::io::BufWriter<std::fs::File>, v: &[f32]| {
        for x in v {
            out.write_all(&x.to_le_bytes()).expect("write");
        }
    };
    out.write_all(b"JXLTRACE1\n").expect("write");
    out.write_all(&(header.len() as u32).to_le_bytes())
        .expect("write");
    out.write_all(header.as_bytes()).expect("write");
    out.write_all(img.as_raw()).expect("write");
    for f in &frames {
        f32s(&mut out, &f.quant_field);
        if let Some(next) = &f.next_quant_field {
            f32s(&mut out, next);
        }
        f32s(&mut out, &f.tile_dist);
        f32s(&mut out, &f.diffmap);
        let srgb = |v: f32| {
            let v = v.clamp(0.0, 1.0);
            let e = if v <= 0.003_130_8 {
                12.92 * v
            } else {
                1.055 * v.powf(1.0 / 2.4) - 0.055
            };
            (e * 255.0 + 0.5) as u8
        };
        let [r, g, b] = &f.recon;
        let packed: Vec<u8> = (0..r.len())
            .flat_map(|i| [srgb(r[i]), srgb(g[i]), srgb(b[i])])
            .collect();
        out.write_all(&packed).expect("write");
    }
    out.flush().expect("flush");
    println!(
        "{w}x{h} d{distance} e{effort}: {} frames, {} bytes encoded -> {}",
        frames.len(),
        bytes.len(),
        args[2]
    );
}
