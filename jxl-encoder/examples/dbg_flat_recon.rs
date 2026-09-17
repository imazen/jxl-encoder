// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Debug probe: capture the buttloop's internal final-iter recon for a strict
//! libjxl encode, then decode the shipped bytes via jxl-oxide, and compare
//! per-channel stats. Usage: dbg_flat_recon <in.png> <out.jxl>

#![cfg(all(feature = "__internal_recon_hook", feature = "butteraugli-loop"))]

use jxl_encoder::api::EncoderStrategy;
use jxl_encoder::vardct::__recon_hook;
use jxl_encoder::{LossyConfig, PixelLayout};
use std::io::Cursor;

fn stats(name: &str, v: &[f32]) {
    let n = v.len() as f64;
    let sum: f64 = v.iter().map(|&x| x as f64).sum();
    let min = v.iter().copied().fold(f32::INFINITY, f32::min);
    let max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let uniq: std::collections::BTreeSet<u32> = v.iter().map(|&x| x.to_bits()).collect();
    eprintln!(
        "{name}: mean={:.6} min={:.6} max={:.6} uniq={}",
        sum / n,
        min,
        max,
        uniq.len()
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let src_path = &args[1];
    let out_path = &args[2];

    let src = image::open(src_path).expect("open src").to_rgb8();
    let (w, h) = (src.width() as usize, src.height() as usize);
    let raw = src.into_raw();

    __recon_hook::set_capture_enabled(true);

    let cfg = LossyConfig::new(1.0)
        .with_effort(8)
        .with_strategy(EncoderStrategy::Libjxl);
    let bytes = cfg
        .encode(&raw, w as u32, h as u32, PixelLayout::Rgb8)
        .expect("encode");
    std::fs::write(out_path, &bytes).unwrap();
    eprintln!("wrote {} bytes to {out_path}", bytes.len());

    let recon = __recon_hook::take_last().expect("no recon captured");
    stats("internal_recon.r", &recon.r);
    stats("internal_recon.g", &recon.g);
    stats("internal_recon.b", &recon.b);
    eprintln!(
        "internal: gs={} scale={:.6} inv_scale={:.6} iter={}/{}",
        recon.final_global_scale, recon.final_scale, recon.final_inv_scale, recon.iter, recon.iters
    );

    // decode shipped bytes via jxl-oxide → linear srgb f32
    let mut img = jxl_oxide::JxlImage::builder()
        .read(Cursor::new(&bytes))
        .expect("decode header");
    img.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
        jxl_oxide::RenderingIntent::Relative,
    ));
    let render = img.render_frame(0).expect("render");
    let fb = render.image_all_channels();
    let buf = fb.buf();
    let chans = fb.channels() as usize;
    let r: Vec<f32> = buf.iter().step_by(chans).copied().collect();
    let g: Vec<f32> = buf.iter().skip(1).step_by(chans).copied().collect();
    let b: Vec<f32> = buf.iter().skip(2).step_by(chans).copied().collect();
    stats("decoded.r", &r);
    stats("decoded.g", &g);
    stats("decoded.b", &b);

    // source in linear
    let lin: Vec<f32> = raw
        .iter()
        .map(|&c| {
            let v = c as f32 / 255.0;
            if v <= 0.040_45 {
                v / 12.92
            } else {
                ((v + 0.055) / 1.055).powf(2.4)
            }
        })
        .collect();
    stats("src_linear", &lin);

    // max abs diff internal vs decoded
    let mut maxd = 0.0f32;
    let mut argmax = 0usize;
    for (i, (a, b)) in recon.g.iter().zip(g.iter()).enumerate() {
        let d = (a - b).abs();
        if d > maxd {
            maxd = d;
            argmax = i;
        }
    }
    eprintln!(
        "max |internal_g - decoded_g| = {maxd:.6} at px {argmax} (internal={:.6} decoded={:.6})",
        recon.g[argmax], g[argmax]
    );

    // Optional extra files: decode-only stats (arg 3+ = paths to .jxl)
    for extra in &args[3..] {
        let eb = std::fs::read(extra).unwrap();
        let mut eimg = jxl_oxide::JxlImage::builder()
            .read(Cursor::new(&eb))
            .expect("decode header");
        eimg.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
            jxl_oxide::RenderingIntent::Relative,
        ));
        let er = eimg.render_frame(0).expect("render");
        let efb = er.image_all_channels();
        let ebuf = efb.buf();
        let ech = efb.channels() as usize;
        let eg: Vec<f32> = ebuf.iter().skip(1).step_by(ech).copied().collect();
        stats(&format!("{extra}.g"), &eg);
    }
}
