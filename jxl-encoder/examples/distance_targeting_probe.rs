//! Requested vs DELIVERED butteraugli distance, per image × effort × distance.
//!
//! The encoder's distance knob is a promise: ask for `d` and the decoded
//! result should sit near butteraugli `d`. This probe measures the promise
//! directly, which is what separates the two ways a gate can misbehave:
//!
//! - **off the RD curve** — more bytes for the same delivered quality (waste);
//! - **on the curve but mis-targeted** — more bytes AND proportionally better
//!   quality than asked for (not waste, but not what the caller requested,
//!   and a monotonicity break where the gate switches on).
//!
//! Built for the d≥3.5 screenshot qf-seed lift (#101 follow-up, 2026-09-06),
//! but the question is general, so the probe takes any image and grid.
//!
//! Env: `IMG` (required), `DISTANCES` (default a ladder across the 3.5 gate),
//! `EFFORTS` (default `8`), `CROP` (centre-crop cap, default none),
//! plus whatever runtime overrides you want to A/B — notably
//! `JXL_BUTTLOOP_INITIAL_QF_SCALE=1.0` to turn the qf-seed lift off
//! (`4.0` is the default scale; `1.0` ≡ off).
//!
//! Prints a TSV to stdout: `effort d_req bytes bfly ssim2 delivered_ratio`,
//! where `delivered_ratio = bfly / d_req` (1.0 = promise kept, <1 = finer
//! than requested, >1 = coarser).
//!
//! Reproducer:
//!   IMG=<png> cargo run -p jxl-encoder --release --example distance_targeting_probe

use std::io::Cursor;

use butteraugli::{ButteraugliParams, butteraugli_linear};
use imgref::Img;
use jxl_encoder::api::{Limits, LossyConfig, PixelLayout};
use rgb::RGB;

fn srgb_to_linear_f32(s: u8) -> f32 {
    let c = s as f32 / 255.0;
    if c <= 0.040_45 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb_u8(l: f32) -> u8 {
    let c = l.clamp(0.0, 1.0);
    let s = if c <= 0.003_130_8 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (s * 255.0).round() as u8
}

fn main() {
    let path = std::env::var("IMG").expect("set IMG=<png path>");
    let distances: Vec<f32> = std::env::var("DISTANCES")
        .unwrap_or_else(|_| "1,2,3,3.4,3.6,4,5,6,8".into())
        .split(',')
        .map(|s| s.trim().parse().expect("DISTANCES: f32 list"))
        .collect();
    let efforts: Vec<u8> = std::env::var("EFFORTS")
        .unwrap_or_else(|_| "8".into())
        .split(',')
        .map(|s| s.trim().parse().expect("EFFORTS: u8 list"))
        .collect();
    let crop: Option<u32> = std::env::var("CROP")
        .ok()
        .map(|s| s.parse().expect("CROP: u32"));

    let img = image::open(&path).expect("open IMG").to_rgb8();
    let (mut w, mut h) = (img.width(), img.height());
    let mut rgb = img.as_raw().clone();
    if let Some(cap) = crop {
        let (cw, ch) = (w.min(cap), h.min(cap));
        if (cw, ch) != (w, h) {
            let (x0, y0) = ((w - cw) / 2, (h - ch) / 2);
            let mut out = Vec::with_capacity(cw as usize * ch as usize * 3);
            for y in y0..y0 + ch {
                let start = ((y * w + x0) * 3) as usize;
                out.extend_from_slice(&rgb[start..start + cw as usize * 3]);
            }
            rgb = out;
            w = cw;
            h = ch;
        }
    }

    let lin: Vec<f32> = rgb.iter().map(|&b| srgb_to_linear_f32(b)).collect();
    let orig_lin: Img<Vec<RGB<f32>>> = Img::new(
        lin.chunks(3).map(|c| RGB::new(c[0], c[1], c[2])).collect(),
        w as usize,
        h as usize,
    );
    let orig_srgb: Img<Vec<[u8; 3]>> = Img::new(
        rgb.chunks(3)
            .map(|c| [c[0], c[1], c[2]])
            .collect::<Vec<_>>(),
        w as usize,
        h as usize,
    );

    let lim = Limits::default().with_max_memory_bytes(8u64 << 30);
    eprintln!(
        "# {path} {w}x{h}  qf_scale_env={:?}",
        std::env::var("JXL_BUTTLOOP_INITIAL_QF_SCALE").ok()
    );
    println!("effort\td_req\tbytes\tbfly\tssim2\tdelivered_ratio");
    for &e in &efforts {
        for &d in &distances {
            let bytes = LossyConfig::new(d)
                .with_effort(e)
                .encode_request(w, h, PixelLayout::Rgb8)
                .with_limits(&lim)
                .encode(&rgb)
                .unwrap_or_else(|err| panic!("encode d={d} e={e}: {err:?}"));
            let mut dec = jxl_oxide::JxlImage::builder()
                .read(Cursor::new(&bytes))
                .expect("decode");
            dec.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
                jxl_oxide::RenderingIntent::Relative,
            ));
            let fb = dec.render_frame(0).expect("render").image_all_channels();
            let buf = fb.buf();
            let dist_lin: Img<Vec<RGB<f32>>> = Img::new(
                buf.chunks(3).map(|c| RGB::new(c[0], c[1], c[2])).collect(),
                fb.width(),
                fb.height(),
            );
            let bfly = butteraugli_linear(
                orig_lin.as_ref(),
                dist_lin.as_ref(),
                &ButteraugliParams::default(),
            )
            .expect("butteraugli")
            .score as f64;
            let dist_srgb: Img<Vec<[u8; 3]>> = Img::new(
                buf.chunks(3)
                    .map(|c| {
                        [
                            linear_to_srgb_u8(c[0]),
                            linear_to_srgb_u8(c[1]),
                            linear_to_srgb_u8(c[2]),
                        ]
                    })
                    .collect::<Vec<_>>(),
                fb.width(),
                fb.height(),
            );
            let ssim2 = fast_ssim2::compute_ssimulacra2(orig_srgb.as_ref(), dist_srgb.as_ref())
                .expect("ssim2");
            println!(
                "{e}\t{d}\t{}\t{bfly:.4}\t{ssim2:.3}\t{:.3}",
                bytes.len(),
                bfly / d as f64
            );
        }
    }
}
