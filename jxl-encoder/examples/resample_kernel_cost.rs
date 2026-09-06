//! What does the decoder-adjoint 2× downsampler actually cost?
//!
//! libjxl restricts `DownsampleImage2_Iterative` to effort 10-11 with the note
//! "it's an 80% slowdown and downsampling is only active at high distances by
//! default anyway" (`enc_frame.cc`). The second half of that rationale does not
//! apply to us — we never auto-resample, so `with_resampling(N)` is always a
//! deliberate request — which makes the first half the whole decision. This
//! measures it on our code rather than inheriting libjxl's figure.
//!
//! Both round trips do one downsample plus one identical upsample, so the
//! DIFFERENCE between them is exactly the extra downsample cost. Reported
//! against a real encode's wall time at the same settings, because a slowdown
//! only matters relative to what it is a slowdown of.
//!
//! Env: `IMGS` (comma-separated paths; default: four imazen-26 pick-list
//! images), `CROP` (default 1024), `REPS` (default 5), `EFFORT` (default 7).
//!
//! Reproducer:
//!   cargo run -p jxl-encoder --release --features __internals \
//!     --example resample_kernel_cost

use std::time::Instant;

use jxl_encoder::__internals::{Downsample2xKernel, resample_roundtrip_2x_rgb};
use jxl_encoder::api::{Limits, LossyConfig, PixelLayout};

fn srgb_to_linear(s: u8) -> f32 {
    let c = s as f32 / 255.0;
    if c <= 0.040_45 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn main() {
    let default_imgs = concat!(
        "/home/lilith/work/codec-corpus/imazen-26/9226-lilith-ai-products/beauty/",
        "9291_gen_products-beauty_bryn-birch-beard-oil-back_ingredients_p0062_1024x1536.png"
    );
    let imgs: Vec<String> = std::env::var("IMGS")
        .unwrap_or_else(|_| default_imgs.to_string())
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    let crop: u32 = std::env::var("CROP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1024);
    let reps: usize = std::env::var("REPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let effort: u8 = std::env::var("EFFORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(7);

    println!("image\tw\th\tmp\tsharper_ms\titerative_ms\tdelta_ms\tencode_ms\tdelta_pct_of_encode");
    for path in &imgs {
        let Ok(img) = image::open(path) else {
            eprintln!("unreadable: {path}");
            continue;
        };
        let rgb8 = img.to_rgb8();
        let (mut w, mut h) = (rgb8.width(), rgb8.height());
        let mut raw = rgb8.as_raw().clone();
        let (cw, ch) = (w.min(crop), h.min(crop));
        if (cw, ch) != (w, h) {
            let (x0, y0) = ((w - cw) / 2, (h - ch) / 2);
            let mut out = Vec::with_capacity(cw as usize * ch as usize * 3);
            for y in y0..y0 + ch {
                let start = ((y * w + x0) * 3) as usize;
                out.extend_from_slice(&raw[start..start + cw as usize * 3]);
            }
            raw = out;
            w = cw;
            h = ch;
        }
        let lin: Vec<f32> = raw.iter().map(|&b| srgb_to_linear(b)).collect();

        // Warm once so neither kernel pays the first-touch page faults.
        let _ =
            resample_roundtrip_2x_rgb(&lin, w as usize, h as usize, Downsample2xKernel::Sharper);
        let mut best = |k: Downsample2xKernel| -> f64 {
            let mut ms = f64::MAX;
            for _ in 0..reps {
                let t = Instant::now();
                let out = resample_roundtrip_2x_rgb(&lin, w as usize, h as usize, k);
                let e = t.elapsed().as_secs_f64() * 1000.0;
                std::hint::black_box(&out);
                ms = ms.min(e);
            }
            ms
        };
        let sharper = best(Downsample2xKernel::Sharper);
        let iterative = best(Downsample2xKernel::Iterative);

        let lim = Limits::default().with_max_memory_bytes(8u64 << 30);
        let t = Instant::now();
        let bytes = LossyConfig::new(2.0)
            .with_effort(effort)
            .with_resampling(2)
            .encode_request(w, h, PixelLayout::Rgb8)
            .with_limits(&lim)
            .encode(&raw)
            .expect("encode");
        let enc_ms = t.elapsed().as_secs_f64() * 1000.0;
        std::hint::black_box(&bytes);

        let name = std::path::Path::new(path)
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default();
        let mp = w as f64 * h as f64 / 1e6;
        let delta = iterative - sharper;
        println!(
            "{}\t{w}\t{h}\t{mp:.2}\t{sharper:.1}\t{iterative:.1}\t{delta:.1}\t{enc_ms:.1}\t{:.1}",
            &name[..name.len().min(40)],
            delta / enc_ms * 100.0
        );
    }
}
