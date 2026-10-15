//! Where the perceptual loop lands against its butteraugli target, and
//! what the bytes past the target buy.
//!
//! Each cell encodes one image at one effort and distance twice: with the
//! effort's butteraugli loop, and with `with_butteraugli_iters(0)` (same
//! effort, no loop). Both outputs are decoded with jxl-oxide to linear sRGB
//! and scored against the source: butteraugli max (the loop's own score),
//! its 3-norm, and SSIMULACRA2. The loop run is traced, so each row also
//! carries the encoder's internal score per iteration and where the last
//! frame's blocks sit against the target.
//!
//! The rows answer two different questions (jxl-encoder #114 keeps them
//! apart): `delivered / requested` says whether the loop kept the distance
//! it was asked for; comparing the loop and no-loop curves at the same
//! delivered quality says whether its extra bytes are waste.
//! `scripts/loop_overshoot_summary.py` does both.
//!
//! Usage:
//!   cargo run --release -p jxl-encoder --features butteraugli-loop,__internals \
//!     --example loop_overshoot -- out.tsv [--efforts 9] \
//!     [--distances 0.5,1,2] [--crop 512] [--cur-pow 0.5] [--late-cur-pow 0.2] [--label loop] \
//!     [--no-baseline] <image.png>...
//!
//! `--cur-pow` overrides the good-block exponent of iterations 0 and 1 at
//! every distance (the `CUR_POW_X1000_{LOW,HIGH}` sweep slots);
//! `--late-cur-pow` sets it for iterations 2 on (`LATE_CUR_POW_X1000`;
//! libjxl and the default use 0 there); `--label`
//! names the loop rows' mode; `--no-baseline` skips the no-loop encodes, so
//! a variant run can be concatenated with a baseline run.

use std::io::{Cursor, Write};

use butteraugli::{ButteraugliParams, butteraugli_linear};
use imgref::Img;
use jxl_encoder::__internals::ZenanalyzeProxies;
use jxl_encoder::vardct::__buttloop_overrides::{
    CUR_POW_X1000_HIGH, CUR_POW_X1000_LOW, LATE_CUR_POW_X1000,
};
use jxl_encoder::{LoopTrace, LossyConfig, PixelLayout};
use rgb::RGB;

struct Image {
    name: String,
    w: u32,
    h: u32,
    rgb: Vec<u8>,
    lin: Img<Vec<RGB<f32>>>,
    srgb: Img<Vec<[u8; 3]>>,
}

fn srgb_to_linear(v: u8) -> f32 {
    let c = f32::from(v) / 255.0;
    if c <= 0.04045 {
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
    (s * 255.0 + 0.5).clamp(0.0, 255.0) as u8
}

fn load(path: &str, crop: u32) -> Image {
    let img = image::open(path)
        .unwrap_or_else(|e| panic!("{path}: {e}"))
        .to_rgb8();
    let (w, h) = (img.width().min(crop), img.height().min(crop));
    let (x0, y0) = ((img.width() - w) / 2, (img.height() - h) / 2);
    let rgb = image::imageops::crop_imm(&img, x0, y0, w, h)
        .to_image()
        .into_raw();
    let lin = Img::new(
        rgb.chunks(3)
            .map(|c| {
                RGB::new(
                    srgb_to_linear(c[0]),
                    srgb_to_linear(c[1]),
                    srgb_to_linear(c[2]),
                )
            })
            .collect(),
        w as usize,
        h as usize,
    );
    let srgb = Img::new(
        rgb.chunks(3).map(|c| [c[0], c[1], c[2]]).collect(),
        w as usize,
        h as usize,
    );
    Image {
        name: path.to_string(),
        w,
        h,
        rgb,
        lin,
        srgb,
    }
}

fn list<T: std::str::FromStr>(s: &str) -> Vec<T>
where
    T::Err: std::fmt::Debug,
{
    s.split(',')
        .map(|v| v.parse().expect("list value"))
        .collect()
}

/// Decoded butteraugli max, butteraugli 3-norm, SSIMULACRA2.
fn score(img: &Image, bytes: &[u8]) -> (f64, f64, f64) {
    let mut dec = jxl_oxide::JxlImage::builder()
        .read(Cursor::new(bytes))
        .expect("decode");
    dec.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
        jxl_oxide::RenderingIntent::Relative,
    ));
    let fb = dec.render_frame(0).expect("render").image_all_channels();
    assert_eq!((fb.width(), fb.height()), (img.w as usize, img.h as usize));
    assert_eq!(fb.channels(), 3);
    let buf = fb.buf();
    let lin = Img::new(
        buf.chunks(3)
            .map(|c| RGB::new(c[0], c[1], c[2]))
            .collect::<Vec<_>>(),
        fb.width(),
        fb.height(),
    );
    let b = butteraugli_linear(
        img.lin.as_ref(),
        lin.as_ref(),
        &ButteraugliParams::default(),
    )
    .expect("butteraugli");
    let srgb = Img::new(
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
    let ssim2 =
        fast_ssim2::compute_ssimulacra2(img.srgb.as_ref(), srgb.as_ref()).expect("ssimulacra2");
    (b.score, b.pnorm_3, ssim2)
}

fn quantile(sorted: &[f32], q: f64) -> f32 {
    sorted[((sorted.len() - 1) as f64 * q).round() as usize]
}

fn main() {
    let mut args = std::env::args().skip(1);
    let out_path = args
        .next()
        .expect("usage: loop_overshoot out.tsv [opts] images...");
    let mut efforts = vec![9u8];
    let mut distances = vec![
        0.5f32, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0,
    ];
    let (mut crop, mut paths) = (512u32, Vec::new());
    let (mut label, mut baseline) = ("loop".to_string(), true);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--efforts" => efforts = list(&args.next().expect("--efforts")),
            "--distances" => distances = list(&args.next().expect("--distances")),
            "--crop" => crop = args.next().expect("--crop").parse().expect("crop"),
            "--cur-pow" => {
                let v: f64 = args.next().expect("--cur-pow").parse().expect("cur-pow");
                let x1000 = (v * 1000.0).round() as i32;
                CUR_POW_X1000_LOW.store(x1000, std::sync::atomic::Ordering::Relaxed);
                CUR_POW_X1000_HIGH.store(x1000, std::sync::atomic::Ordering::Relaxed);
            }
            "--late-cur-pow" => {
                let v: f64 = args.next().expect("--late-cur-pow").parse().expect("late");
                LATE_CUR_POW_X1000.store(
                    (v * 1000.0).round() as i32,
                    std::sync::atomic::Ordering::Relaxed,
                );
            }
            "--label" => label = args.next().expect("--label"),
            "--no-baseline" => baseline = false,
            _ => paths.push(a),
        }
    }
    let mut out = std::fs::File::create(&out_path).expect("create output");
    writeln!(
        out,
        "image\twidth\theight\teffort\tdistance\tmode\tbytes\tbfly_max\tbfly_p3\tssim2\
         \tseeds\titers_run\tinternal_scores\tfinal_target\tblocks_above\tblock_p50\tblock_p90\
         \tblocks_below_half\tm3\tfcbr\tedge_density\tluma_var"
    )
    .unwrap();
    let total = paths.len() * efforts.len() * distances.len();
    let mut k = 0;
    for path in &paths {
        let img = load(path, crop);
        // The proxies the encoder's content gates read, on the cropped
        // pixels the encoder sees.
        let p = ZenanalyzeProxies::compute_srgb_u8(
            &img.rgb,
            img.w as usize,
            img.h as usize,
            3,
            0,
            1,
            2,
        );
        let proxies = format!(
            "{:.3}\t{:.4}\t{:.4}\t{:.1}",
            p.m3_colourfulness, p.flat_color_block_ratio, p.edge_density, p.luma_var
        );
        for &effort in &efforts {
            for &distance in &distances {
                k += 1;
                let config = LossyConfig::new(distance).with_effort(effort);
                let plain = baseline.then(|| {
                    config
                        .clone()
                        .with_butteraugli_iters(0)
                        .encode_request(img.w, img.h, PixelLayout::Rgb8)
                        .encode(&img.rgb)
                        .expect("encode without loop")
                });
                let trace = LoopTrace::new();
                let looped = config
                    .encode_request(img.w, img.h, PixelLayout::Rgb8)
                    .with_loop_trace(&trace)
                    .encode(&img.rgb)
                    .expect("encode with loop");
                let frames = trace.take();
                let row = |mode: &str, bytes: &[u8], extra: String| {
                    let (bmax, bp3, ssim2) = score(&img, bytes);
                    format!(
                        "{}\t{}\t{}\t{effort}\t{distance}\t{mode}\t{}\t{bmax:.4}\t{bp3:.4}\t{ssim2:.3}\t{extra}\t{proxies}",
                        img.name,
                        img.w,
                        img.h,
                        bytes.len()
                    )
                };
                let none = "0\t0\t-\t-\t-\t-\t-\t-".to_string();
                let plain_row = plain.as_deref().map(|p| row("noloop", p, none.clone()));
                // Per seed, the internal score of each scored iteration;
                // seeds separated by `|`. The block summary is the last
                // frame of the last seed (e9 and below run one seed).
                let seeds: Vec<&[jxl_encoder::LoopFrame]> = frames
                    .chunk_by(|a, b| a.seed.to_bits() == b.seed.to_bits())
                    .collect();
                let extra = match frames.last() {
                    None => none,
                    Some(last) => {
                        let scores = seeds
                            .iter()
                            .map(|s| {
                                s.iter()
                                    .map(|f| format!("{:.4}", f.score))
                                    .collect::<Vec<_>>()
                                    .join(",")
                            })
                            .collect::<Vec<_>>()
                            .join("|");
                        let target = last.target as f32;
                        let mut ratio: Vec<f32> =
                            last.tile_dist.iter().map(|&t| t / target).collect();
                        ratio.sort_by(f32::total_cmp);
                        let above = ratio.iter().filter(|&&r| r > 1.0).count();
                        let below_half = ratio.iter().filter(|&&r| r < 0.5).count();
                        format!(
                            "{}\t{}\t{scores}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}",
                            seeds.len(),
                            frames.len(),
                            last.target,
                            above as f64 / ratio.len() as f64,
                            quantile(&ratio, 0.5),
                            quantile(&ratio, 0.9),
                            below_half as f64 / ratio.len() as f64,
                        )
                    }
                };
                if let Some(plain_row) = plain_row {
                    writeln!(out, "{plain_row}").unwrap();
                }
                writeln!(out, "{}", row(&label, &looped, extra)).unwrap();
                out.flush().unwrap();
                eprintln!(
                    "[{k}/{total}] {} e{effort} d{distance}: {:?} -> {} B",
                    img.name.rsplit('/').next().unwrap_or(""),
                    plain.as_ref().map(Vec::len),
                    looped.len()
                );
            }
        }
    }
}
