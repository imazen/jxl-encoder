//! W44-183 spot-check: measure CfL Newton parameter port impact across
//! 5 CID22 photos + 3 GB82-SC screenshots + clic_097cb426 at e7 d=2/3/5.
//!
//! W44-182 (`3c2026b2`) identified our `find_best_multiplier_newton`
//! diverged from libjxl in 4 parameters (eps 1.0 vs 100, max_iters 10 vs
//! 20, starting x ls_x vs 0, no-LS-fallback). W44-183 (this commit) ports
//! all 4 bit-exactly per libjxl `enc_chroma_from_luma.cc:152-167`.
//!
//! Outputs:
//!   benchmarks/w44_183_cfl_newton_spotcheck_2026-05-22.tsv
//!   benchmarks/w44_183_cfl_newton_spotcheck_2026-05-22.meta
//!
//! Run:
//!   CARGO_TARGET_DIR=$HOME/work/zen/jxl-encoder-shared-target \
//!     cargo run --release --features 'parallel butteraugli-loop ssim2-loop' \
//!       --manifest-path jxl-encoder/Cargo.toml \
//!       --example w44_183_cfl_newton_spotcheck

use butteraugli::{ButteraugliParams, butteraugli_linear};
use imgref::{Img, ImgVec};
use jxl_encoder::api::{LossyConfig, PixelLayout};
use rgb::RGB;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::process::Command;

const EFFORT: u8 = 7;
const DISTANCES: &[f32] = &[2.0, 3.0, 5.0];

struct Cell {
    short: &'static str,
    path: &'static str,
    class: &'static str,
}

const CELLS: &[Cell] = &[
    Cell {
        short: "cid22_1025469",
        path: "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1025469.png",
        class: "photo",
    },
    Cell {
        short: "cid22_1189261",
        path: "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1189261.png",
        class: "photo",
    },
    Cell {
        short: "cid22_1418519",
        path: "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1418519.png",
        class: "photo",
    },
    Cell {
        short: "cid22_1420710",
        path: "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1420710.png",
        class: "photo",
    },
    Cell {
        short: "cid22_1531677",
        path: "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1531677.png",
        class: "photo",
    },
    Cell {
        short: "gb82_codec_wiki",
        path: "/home/lilith/work/codec-corpus/gb82-sc/codec_wiki.png",
        class: "screenshot",
    },
    Cell {
        short: "gb82_terminal",
        path: "/home/lilith/work/codec-corpus/gb82-sc/terminal.png",
        class: "screenshot",
    },
    Cell {
        short: "gb82_imac_g3",
        path: "/home/lilith/work/codec-corpus/gb82-sc/imac_g3.png",
        class: "screenshot",
    },
    Cell {
        short: "clic_097cb426",
        path: "/home/lilith/work/codec-corpus/clic2025-1024/097cb426910ba8ce2525dd8bb7fb1777.png",
        class: "photo_wedge",
    },
];

fn cjxl_bin() -> PathBuf {
    if let Ok(p) = std::env::var("CJXL") {
        return PathBuf::from(p);
    }
    PathBuf::from("/home/lilith/work/jxl-efforts/libjxl/build/tools/cjxl")
}

fn srgb_to_linear_f32(s: u8) -> f32 {
    let c = s as f32 / 255.0;
    if c <= 0.040_45 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb_u8(linear: f32) -> u8 {
    let c = linear.clamp(0.0, 1.0);
    let srgb = if c <= 0.003_130_8 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (srgb * 255.0 + 0.5) as u8
}

fn load_png(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
    let img = image::open(path).ok()?;
    let rgb = img.to_rgb8();
    Some((rgb.as_raw().clone(), rgb.width(), rgb.height()))
}

fn decode_jxl_linear(bytes: &[u8]) -> Option<(u32, u32, Vec<f32>)> {
    let mut img = jxl_oxide::JxlImage::builder().read(Cursor::new(bytes)).ok()?;
    img.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
        jxl_oxide::RenderingIntent::Relative,
    ));
    let render = img.render_frame(0).ok()?;
    let fb = render.image_all_channels();
    Some((fb.width() as u32, fb.height() as u32, fb.buf().to_vec()))
}

fn rgb_u8_to_linear_img(rgb: &[u8], w: u32, h: u32) -> ImgVec<RGB<f32>> {
    let n = (w * h) as usize;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(RGB {
            r: srgb_to_linear_f32(rgb[i * 3]),
            g: srgb_to_linear_f32(rgb[i * 3 + 1]),
            b: srgb_to_linear_f32(rgb[i * 3 + 2]),
        });
    }
    Img::new(out, w as usize, h as usize)
}

fn linear_planar_to_img(lin: &[f32], w: u32, h: u32) -> ImgVec<RGB<f32>> {
    let n = (w * h) as usize;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(RGB {
            r: lin[i * 3].clamp(0.0, 1.0),
            g: lin[i * 3 + 1].clamp(0.0, 1.0),
            b: lin[i * 3 + 2].clamp(0.0, 1.0),
        });
    }
    Img::new(out, w as usize, h as usize)
}

fn linear_planar_to_srgb_arr3(lin: &[f32], w: u32, h: u32) -> ImgVec<[u8; 3]> {
    let n = (w * h) as usize;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push([
            linear_to_srgb_u8(lin[i * 3]),
            linear_to_srgb_u8(lin[i * 3 + 1]),
            linear_to_srgb_u8(lin[i * 3 + 2]),
        ]);
    }
    Img::new(out, w as usize, h as usize)
}

fn rgb_u8_to_arr3_img(rgb: &[u8], w: u32, h: u32) -> ImgVec<[u8; 3]> {
    let n = (w * h) as usize;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push([rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2]]);
    }
    Img::new(out, w as usize, h as usize)
}

fn encode_ours(rgb: &[u8], w: u32, h: u32, distance: f32) -> Vec<u8> {
    LossyConfig::new(distance)
        .with_effort(EFFORT)
        .with_threads(1)
        .encode(rgb, w, h, PixelLayout::Rgb8)
        .expect("ours encode")
}

fn encode_cjxl(src_png: &Path, distance: f32) -> Option<Vec<u8>> {
    let stem = src_png.file_stem()?.to_string_lossy().into_owned();
    let out_path =
        std::env::temp_dir().join(format!("w44_183_cjxl_{stem}_d{distance}.jxl"));
    let status = Command::new(cjxl_bin())
        .arg(src_png)
        .arg(&out_path)
        .arg("-d")
        .arg(format!("{distance}"))
        .arg("-e")
        .arg(format!("{EFFORT}"))
        .args(["--num_threads", "1"])
        .arg("--quiet")
        .status()
        .ok()?;
    if !status.success() {
        return None;
    }
    let bytes = std::fs::read(&out_path).ok()?;
    let _ = std::fs::remove_file(&out_path);
    Some(bytes)
}

fn main() {
    let out_tsv = "benchmarks/w44_183_cfl_newton_spotcheck_2026-05-22.tsv";
    let out_meta = "benchmarks/w44_183_cfl_newton_spotcheck_2026-05-22.meta";

    let mut rows: Vec<String> = Vec::new();
    rows.push(
        "image\tclass\tw\th\teffort\tdistance\
        \tours_bytes\tcjxl_bytes\tbytes_delta_pct\
        \tours_ssim2\tcjxl_ssim2\tssim2_delta\
        \tours_bfly\tcjxl_bfly\tbfly_delta_pct"
            .to_string(),
    );

    for cell in CELLS {
        let path = Path::new(cell.path);
        let (rgb, w, h) = match load_png(path) {
            Some(v) => v,
            None => {
                eprintln!("  skip (load failed): {}", cell.short);
                continue;
            }
        };
        let src_lin = rgb_u8_to_linear_img(&rgb, w, h);
        let src_srgb = rgb_u8_to_arr3_img(&rgb, w, h);

        for &d in DISTANCES {
            eprintln!("[w44-183] {} d={d} ...", cell.short);

            let ours = encode_ours(&rgb, w, h, d);
            let cjxl = match encode_cjxl(path, d) {
                Some(b) => b,
                None => {
                    eprintln!("  cjxl encode failed");
                    continue;
                }
            };

            let (ow, oh, ours_lin) = match decode_jxl_linear(&ours) {
                Some(v) => v,
                None => {
                    eprintln!("  ours decode failed");
                    continue;
                }
            };
            let (cw, ch, cjxl_lin) = match decode_jxl_linear(&cjxl) {
                Some(v) => v,
                None => {
                    eprintln!("  cjxl decode failed");
                    continue;
                }
            };
            if (ow, oh) != (w, h) || (cw, ch) != (w, h) {
                eprintln!("  decode dimensions mismatch");
                continue;
            }

            let ours_img = linear_planar_to_img(&ours_lin, w, h);
            let cjxl_img = linear_planar_to_img(&cjxl_lin, w, h);
            let ours_srgb = linear_planar_to_srgb_arr3(&ours_lin, w, h);
            let cjxl_srgb = linear_planar_to_srgb_arr3(&cjxl_lin, w, h);

            let ours_ssim2 =
                fast_ssim2::compute_ssimulacra2(src_srgb.as_ref(), ours_srgb.as_ref())
                    .unwrap_or(0.0);
            let cjxl_ssim2 =
                fast_ssim2::compute_ssimulacra2(src_srgb.as_ref(), cjxl_srgb.as_ref())
                    .unwrap_or(0.0);
            let ours_bfly = butteraugli_linear(
                src_lin.as_ref(),
                ours_img.as_ref(),
                &ButteraugliParams::default(),
            )
            .map(|s| s.score as f32)
            .unwrap_or(99.0);
            let cjxl_bfly = butteraugli_linear(
                src_lin.as_ref(),
                cjxl_img.as_ref(),
                &ButteraugliParams::default(),
            )
            .map(|s| s.score as f32)
            .unwrap_or(99.0);

            let bytes_delta =
                (ours.len() as f32 - cjxl.len() as f32) / cjxl.len() as f32 * 100.0;
            let ssim2_delta = ours_ssim2 as f32 - cjxl_ssim2 as f32;
            let bfly_delta = (ours_bfly - cjxl_bfly) / cjxl_bfly * 100.0;

            rows.push(format!(
                "{}\t{}\t{w}\t{h}\t{EFFORT}\t{d}\
                \t{}\t{}\t{:.3}\
                \t{:.4}\t{:.4}\t{:.4}\
                \t{:.4}\t{:.4}\t{:.3}",
                cell.short,
                cell.class,
                ours.len(),
                cjxl.len(),
                bytes_delta,
                ours_ssim2,
                cjxl_ssim2,
                ssim2_delta,
                ours_bfly,
                cjxl_bfly,
                bfly_delta,
            ));
        }
    }

    std::fs::write(out_tsv, rows.join("\n") + "\n").expect("write tsv");
    eprintln!("[w44-183] wrote {out_tsv}");

    let meta = "# W44-183 CfL Newton parameter port spot-check\n\
                # Date: 2026-05-22\n\
                # Cells: 9 images x 3 distances = 27 cells\n\
                # Effort: e7 (Newton gate fires at effort >= 7)\n\
                #\n\
                # W44-182 (3c2026b2) found find_best_multiplier_newton diverged\n\
                # from libjxl in 4 parameters (eps, max_iters, start x, fallback).\n\
                # W44-183 ports all 4 bit-exactly per\n\
                # enc_chroma_from_luma.cc:152-167.\n\
                #\n\
                # Reproducer:\n\
                #   CARGO_TARGET_DIR=$HOME/work/zen/jxl-encoder-shared-target \\\n\
                #     cargo run --release --features 'parallel butteraugli-loop ssim2-loop' \\\n\
                #     --manifest-path jxl-encoder/Cargo.toml \\\n\
                #     --example w44_183_cfl_newton_spotcheck\n";
    std::fs::write(out_meta, meta).expect("write meta");
    eprintln!("[w44-183] wrote {out_meta}");
}
