//! W44-AUDIT-9 / SA-G Fix C: A/B bench for `cfl_zero_for_search` gate.
//!
//! This bench measures the impact of the SA-G Fix C `cfl_zero_for_search`
//! gate (force `CflMap::zeros` during AC strategy SEARCH only — emitted
//! cmap stays Newton-derived):
//!
//! - **SMOKE** cell `clic_22ea12 e9 d=4 --strategy libjxl`: verifies the
//!   SA-G report's prediction (bytes ~-0.6% on Libjxl strategy).
//! - **WIDER_LIBJXL**: per-cell bytes / SSIM2 / butteraugli on a manageable
//!   corpus of 9 photos × 3 distances at effort 7 under `--strategy libjxl`.
//!   Documents wins/losses for the OPT-IN escape hatch (NOT a default-flip).
//! - **PROTECT_ZENJXL**: Zenjxl strategy bytes — gate is OFF, so output is
//!   structurally byte-identical to pre-Fix-C. Hash-locks 36/36 confirm
//!   byte-identity on synthetic fixtures; this bench just reports Zenjxl
//!   bytes as a sanity check.
//!
//! Outputs:
//!   benchmarks/sa_g_fix_c_ab_2026-05-25.tsv
//!   benchmarks/sa_g_fix_c_ab_2026-05-25.meta
//!
//! Run:
//!   CARGO_TARGET_DIR=$HOME/work/zen/jxl-encoder-shared-target \
//!     cargo run --release --features 'parallel butteraugli-loop ssim2-loop' \
//!       --manifest-path jxl-encoder/Cargo.toml \
//!       --example sa_g_fix_c_ab

use butteraugli::{ButteraugliParams, butteraugli_linear};
use imgref::{Img, ImgVec};
use jxl_encoder::api::{EncoderStrategy, LossyConfig, PixelLayout};
use rgb::RGB;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::process::Command;

const SMOKE_DISTANCE: f32 = 4.0;
const SMOKE_EFFORT: u8 = 9;
const WIDER_EFFORT: u8 = 7;
const WIDER_DISTANCES: &[f32] = &[2.0, 3.0, 5.0];

struct Cell {
    short: &'static str,
    path: &'static str,
    class: &'static str,
}

const SMOKE_CELL: Cell = Cell {
    short: "clic_22ea12",
    path: "/home/lilith/work/codec-corpus/clic2025-1024/22ea12c903e41583b7c469cb86040157.png",
    class: "smoke",
};

const WIDER_CELLS: &[Cell] = &[
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

fn srgb_to_linear_f32(s: u8) -> f32 {
    let s = s as f32 / 255.0;
    if s <= 0.04045 {
        s / 12.92
    } else {
        ((s + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb_u8(lin: f32) -> u8 {
    let srgb = if lin <= 0.0031308 {
        lin * 12.92
    } else {
        1.055 * lin.max(0.0).powf(1.0 / 2.4) - 0.055
    };
    (srgb.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

fn load_png(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
    let img = image::open(path).ok()?;
    let rgb = img.to_rgb8();
    Some((rgb.as_raw().clone(), rgb.width(), rgb.height()))
}

fn decode_jxl_linear(bytes: &[u8]) -> Option<(u32, u32, Vec<f32>)> {
    let mut img = jxl_oxide::JxlImage::builder()
        .read(Cursor::new(bytes))
        .ok()?;
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

fn encode_with_strategy(
    rgb: &[u8],
    w: u32,
    h: u32,
    distance: f32,
    effort: u8,
    strategy: EncoderStrategy,
) -> Vec<u8> {
    LossyConfig::new(distance)
        .with_effort(effort)
        .with_threads(1)
        .with_strategy(strategy)
        .encode(rgb, w, h, PixelLayout::Rgb8)
        .expect("encode")
}

fn cjxl_bin() -> PathBuf {
    if let Ok(p) = std::env::var("CJXL") {
        return PathBuf::from(p);
    }
    PathBuf::from("/home/lilith/work/jxl-efforts/libjxl/build/tools/cjxl")
}

fn encode_cjxl(src_png: &Path, distance: f32, effort: u8) -> Option<Vec<u8>> {
    let stem = src_png.file_stem()?.to_string_lossy().into_owned();
    let out_path =
        std::env::temp_dir().join(format!("sa_g_fix_c_cjxl_{stem}_d{distance}_e{effort}.jxl"));
    let status = Command::new(cjxl_bin())
        .arg(src_png)
        .arg(&out_path)
        .arg("-d")
        .arg(format!("{distance}"))
        .arg("-e")
        .arg(format!("{effort}"))
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

fn ssim2_metric(src: &ImgVec<[u8; 3]>, decoded: &ImgVec<[u8; 3]>) -> f32 {
    fast_ssim2::compute_ssimulacra2(src.as_ref(), decoded.as_ref()).unwrap_or(0.0) as f32
}

fn butteraugli_metric(src: &ImgVec<RGB<f32>>, decoded: &ImgVec<RGB<f32>>) -> f32 {
    let params = ButteraugliParams::default();
    butteraugli_linear(src.as_ref(), decoded.as_ref(), &params)
        .map(|s| s.score as f32)
        .unwrap_or(99.0)
}

#[allow(clippy::too_many_arguments)]
fn run_cell(
    rows: &mut Vec<String>,
    cell: &Cell,
    distance: f32,
    effort: u8,
    band: &str,
    zenjxl_libjxl_differs_count: &mut usize,
    total_cells: &mut usize,
) {
    let path = Path::new(cell.path);
    let (rgb, w, h) = match load_png(path) {
        Some(v) => v,
        None => {
            eprintln!("  skip (load failed): {} d={distance}", cell.short);
            return;
        }
    };
    let src_lin = rgb_u8_to_linear_img(&rgb, w, h);
    let src_srgb = rgb_u8_to_arr3_img(&rgb, w, h);

    let zenjxl_bytes = encode_with_strategy(&rgb, w, h, distance, effort, EncoderStrategy::Zenjxl);
    let libjxl_bytes = encode_with_strategy(&rgb, w, h, distance, effort, EncoderStrategy::Libjxl);
    let cjxl_bytes = encode_cjxl(path, distance, effort);

    // Decode all 3 outputs via jxl-oxide for metric scoring
    let zenjxl_dec_lin = match decode_jxl_linear(&zenjxl_bytes) {
        Some(v) => v,
        None => {
            eprintln!("  ERR zenjxl decode failed: {} d={distance}", cell.short);
            return;
        }
    };
    let libjxl_dec_lin = match decode_jxl_linear(&libjxl_bytes) {
        Some(v) => v,
        None => {
            eprintln!("  ERR libjxl decode failed: {} d={distance}", cell.short);
            return;
        }
    };
    let cjxl_dec_lin = cjxl_bytes.as_ref().and_then(|b| decode_jxl_linear(b));

    let zenjxl_dec_img = linear_planar_to_img(&zenjxl_dec_lin.2, w, h);
    let libjxl_dec_img = linear_planar_to_img(&libjxl_dec_lin.2, w, h);
    let cjxl_dec_img = cjxl_dec_lin
        .as_ref()
        .map(|(dw, dh, buf)| linear_planar_to_img(buf, *dw, *dh));

    let zenjxl_dec_srgb = linear_planar_to_srgb_arr3(&zenjxl_dec_lin.2, w, h);
    let libjxl_dec_srgb = linear_planar_to_srgb_arr3(&libjxl_dec_lin.2, w, h);
    let cjxl_dec_srgb = cjxl_dec_lin
        .as_ref()
        .map(|(dw, dh, buf)| linear_planar_to_srgb_arr3(buf, *dw, *dh));

    let zenjxl_ssim2 = ssim2_metric(&src_srgb, &zenjxl_dec_srgb);
    let libjxl_ssim2 = ssim2_metric(&src_srgb, &libjxl_dec_srgb);
    let cjxl_ssim2 = cjxl_dec_srgb
        .as_ref()
        .map(|d| ssim2_metric(&src_srgb, d))
        .unwrap_or(f32::NAN);

    let zenjxl_bfly = butteraugli_metric(&src_lin, &zenjxl_dec_img);
    let libjxl_bfly = butteraugli_metric(&src_lin, &libjxl_dec_img);
    let cjxl_bfly = cjxl_dec_img
        .as_ref()
        .map(|d| butteraugli_metric(&src_lin, d))
        .unwrap_or(f32::NAN);

    let zenjxl_libjxl_differ = zenjxl_bytes != libjxl_bytes;
    if zenjxl_libjxl_differ {
        *zenjxl_libjxl_differs_count += 1;
    }
    *total_cells += 1;

    let cjxl_size = cjxl_bytes.as_ref().map(|b| b.len()).unwrap_or(0);
    let libjxl_minus_cjxl_pct = if cjxl_size > 0 {
        (libjxl_bytes.len() as f32 - cjxl_size as f32) / cjxl_size as f32 * 100.0
    } else {
        f32::NAN
    };
    let libjxl_minus_zenjxl_pct = if zenjxl_bytes.is_empty() {
        f32::NAN
    } else {
        (libjxl_bytes.len() as f32 - zenjxl_bytes.len() as f32) / zenjxl_bytes.len() as f32 * 100.0
    };

    rows.push(format!(
        "{band}\t{}\t{}\t{w}\t{h}\t{effort}\t{distance}\t{}\t{}\t{}\t{libjxl_minus_zenjxl_pct:.3}\t{}\t{libjxl_minus_cjxl_pct:.3}\t{zenjxl_ssim2:.4}\t{libjxl_ssim2:.4}\t{cjxl_ssim2:.4}\t{zenjxl_bfly:.4}\t{libjxl_bfly:.4}\t{cjxl_bfly:.4}\t{zenjxl_libjxl_differ}",
        cell.short, cell.class, zenjxl_bytes.len(), libjxl_bytes.len(),
        (libjxl_bytes.len() as i64 - zenjxl_bytes.len() as i64),
        cjxl_size,
    ));
}

fn main() {
    let out_tsv = "benchmarks/sa_g_fix_c_ab_2026-05-25.tsv";
    let out_meta = "benchmarks/sa_g_fix_c_ab_2026-05-25.meta";

    let mut rows: Vec<String> = Vec::new();
    rows.push(
        "band\timage\tclass\tw\th\teffort\tdistance\
        \tzenjxl_bytes\tlibjxl_bytes\tlibjxl_minus_zenjxl_bytes\tlibjxl_minus_zenjxl_pct\
        \tcjxl_bytes\tlibjxl_minus_cjxl_pct\
        \tzenjxl_ssim2\tlibjxl_ssim2\tcjxl_ssim2\
        \tzenjxl_bfly\tlibjxl_bfly\tcjxl_bfly\
        \tzenjxl_vs_libjxl_differ"
            .to_string(),
    );

    let mut zenjxl_libjxl_differs_count = 0usize;
    let mut total_cells = 0usize;

    eprintln!(
        "→ SMOKE: {} e{} d={}",
        SMOKE_CELL.short, SMOKE_EFFORT, SMOKE_DISTANCE
    );
    run_cell(
        &mut rows,
        &SMOKE_CELL,
        SMOKE_DISTANCE,
        SMOKE_EFFORT,
        "SMOKE",
        &mut zenjxl_libjxl_differs_count,
        &mut total_cells,
    );

    eprintln!(
        "→ WIDER_LIBJXL: {} cells × {} distances at e{}",
        WIDER_CELLS.len(),
        WIDER_DISTANCES.len(),
        WIDER_EFFORT
    );
    for cell in WIDER_CELLS {
        for &d in WIDER_DISTANCES {
            eprintln!("  {} d={d}", cell.short);
            run_cell(
                &mut rows,
                cell,
                d,
                WIDER_EFFORT,
                "WIDER_LIBJXL",
                &mut zenjxl_libjxl_differs_count,
                &mut total_cells,
            );
        }
    }

    std::fs::create_dir_all("benchmarks").ok();
    std::fs::write(out_tsv, rows.join("\n") + "\n").expect("write tsv");

    let zenjxl_libjxl_differ_pct = if total_cells > 0 {
        zenjxl_libjxl_differs_count as f32 / total_cells as f32 * 100.0
    } else {
        0.0
    };

    let meta = format!(
        "W44-AUDIT-9 / SA-G Fix C: A/B bench for `cfl_zero_for_search` gate\n\
        Date: 2026-05-25\n\
        Commit: (run before commit; see git log for SHA)\n\
        Host: {}\n\
        \n\
        Bands:\n\
        - SMOKE: 1 cell ({} e{} d={}) — SA-G report prediction: ~-0.6% bytes on Libjxl strategy.\n\
        - WIDER_LIBJXL: {} cells × {} distances at e{} — documents Libjxl-strategy bytes/quality \
          deltas with the gate ON by default. NOT a default-flip for Zenjxl.\n\
        \n\
        Aggregate:\n\
        - Total cells: {}\n\
        - Zenjxl ≠ Libjxl byte differ count: {} ({:.1}%) — the two strategies SHOULD differ \
          on most cells because Fix C + Section A/B/D flips together change the Libjxl bitstream. \
          Cells where they don't differ are 'gate doesn't fire' (synthetic / too-small input).\n\
        \n\
        Acceptance gates per task spec:\n\
        - (f) SMOKE: bytes drop ~-0.6% on Libjxl strategy: SEE band=SMOKE row libjxl_minus_zenjxl_bytes.\n\
        - (f) SMOKE: SSIM2 regression ≤ 0.5 on Libjxl: SEE band=SMOKE row libjxl_ssim2 vs zenjxl_ssim2.\n\
        - (g) WIDER_LIBJXL: documented in TSV — count wins/losses on Libjxl strategy bytes. \
          Catastrophic threshold: 10+ cells lose >2 SSIM2 → HONEST-STOP and revert.\n\
        \n\
        DO NOT (binding for future agents):\n\
        - DO NOT default-flip Fix C on Zenjxl/Aggressive/LeanFaster. Cost model (W44-29..W44-172) \
          is calibrated against the current cmap baseline; W44-183 honest-stop pattern likely \
          to repeat.\n\
        - DO NOT cite 'FMA precision' for ANY byte movement (per W44-66 user-mandated rule). \
          The bytes change because the AC strategy search picks different transforms when fed \
          a zero cmap.\n\
        - DO NOT touch CfL APPLY behavior — the emitted cmap_x/cmap_b in the bitstream stays \
          Newton-derived. ONLY the consumption-during-search is zeroed.\n\
        - DO NOT widen scope to 'fix' the Newton kernel — that's Fix B (parallel sibling).\n\
        \n\
        Repro:\n\
        CARGO_TARGET_DIR=$HOME/work/zen/jxl-encoder-shared-target \\\n  \
          cargo run --release --features 'parallel butteraugli-loop ssim2-loop' \\\n  \
            --manifest-path jxl-encoder/Cargo.toml \\\n  \
            --example sa_g_fix_c_ab\n",
        hostname(),
        SMOKE_CELL.short,
        SMOKE_EFFORT,
        SMOKE_DISTANCE,
        WIDER_CELLS.len(),
        WIDER_DISTANCES.len(),
        WIDER_EFFORT,
        total_cells,
        zenjxl_libjxl_differs_count,
        zenjxl_libjxl_differ_pct,
    );
    std::fs::write(out_meta, meta).expect("write meta");

    eprintln!("\n✓ wrote {out_tsv}");
    eprintln!("✓ wrote {out_meta}");
    eprintln!(
        "  cells: {total_cells}, zenjxl≠libjxl: {zenjxl_libjxl_differs_count} ({zenjxl_libjxl_differ_pct:.1}%)"
    );
}

fn hostname() -> String {
    std::env::var("HOSTNAME").unwrap_or_else(|_| {
        std::process::Command::new("hostname")
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|| "unknown".to_string())
    })
}
