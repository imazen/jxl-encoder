//! Auto-resample monotonicity baseline (issue #101 follow-up, 2026-09-05).
//!
//! libjxl (and we, at parity) switch to 2× resampling at `distance >= 10`
//! and remap the internal distance to `d * 0.25 + 0.25`
//! (`enc_frame.cc:108-114`). Issue #101 measured a byte INCREASE across
//! that boundary (d=10 larger than d=8) and a 36-point SSIM2 cliff between
//! d=9.9 and d=10.0 on an even-dimension image. This harness quantifies
//! both on real content and records the data a principled switch rule
//! needs: the full-resolution RD curve, the 2× RD curve parametrised by
//! its internal distance, and each image's resampling floor.
//!
//! Grid, per image × effort:
//! - `full`: auto-resampling OFF, d ∈ D_FULL (6 … 25) — the single-regime
//!   reference ladder.
//! - `auto`: the default path, d ∈ D_AUTO (10 … 25) — engages 2× with the
//!   libjxl remap. Below 10 it is byte-identical to `full`, so not re-run.
//! - `res2`: explicit `with_resampling(2)` at internal distance t ∈ T_RES2.
//!   T_RES2 contains every remapped value the `auto` cells use, so each
//!   `auto` row has a `res2` twin that must be BYTE-IDENTICAL (sha16
//!   column) — a self-check that the harness measures what it claims.
//!   t = 0.5 doubles as the resampling-floor proxy (2× down→up loss plus
//!   near-lossless quantisation).
//!
//! Metrics: jxl-oxide `srgb_linear` decode + Rust `butteraugli_linear`
//! (full resolution, i.e. AFTER the decoder's upsampling) +
//! `fast_ssim2::compute_ssimulacra2`. CLAUDE.md compliant — no
//! `butteraugli_main`, no PNG-metadata linearisation trap.
//!
//! Corpus (20 real images, stratified; big screenshots centre-cropped to
//! ≤ `MAX_PIXELS`): 8 CID22-512 training photos, 6 gb82-sc screenshots,
//! 3 imazen-26 aliased line-art / grid plots, 3 imazen-26 375×667 web
//! captures (odd height — exercises the #101 header fix too).
//!
//! Env knobs: `CODEC_CORPUS_DIR` (default `~/work/codec-corpus`),
//! `EFFORTS` (default `5,8`), `OUT_DIR` (default `benchmarks`),
//! `MAX_PIXELS` (default 1100000), `TAG` (default UTC date).
//!
//! Output: `<OUT_DIR>/auto_resample_monotonicity_<TAG>.tsv` + `.meta`.
//! Reproducer: `cargo run -p jxl-encoder --release --example auto_resample_monotonicity`
//!
//! `MODE=cjxl` cross-check (reference encoder, differential-validation
//! use only): the same corpus through the installed `cjxl` (`CJXL` env
//! overrides the binary) at effort 7, d ∈ D_CJXL, twice per d — default
//! flags (`cjxl_auto`: whatever auto-resample rule that cjxl build has)
//! and `--resampling=1` (`cjxl_full`). Identical bytes ⇒ that cjxl did
//! not switch at that d; a difference shows its switch and, through the
//! same decode+score path, whether the switch pays under butteraugli.
//! Writes `auto_resample_monotonicity_cjxl_<TAG>.tsv` + `.meta`.

use std::fs::{self, OpenOptions};
use std::io::{Cursor, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use butteraugli::{ButteraugliParams, butteraugli_linear};
use imgref::Img;
use jxl_encoder::api::{Limits, LossyConfig, PixelLayout};
use rgb::RGB;
use sha2::{Digest, Sha256};

const D_FULL: &[f32] = &[
    6.0, 8.0, 9.0, 9.5, 9.9, 10.0, 10.5, 11.0, 12.0, 13.0, 15.0, 17.0, 20.0, 25.0,
];
const D_AUTO: &[f32] = &[10.0, 10.5, 11.0, 12.0, 13.0, 15.0, 17.0, 20.0, 25.0];
const D_CJXL: &[f32] = &[8.0, 9.9, 10.0, 12.0, 15.0, 17.0, 20.0, 22.0, 25.0];
const T_RES2: &[f32] = &[
    0.5, 1.0, 1.5, 2.0, 2.75, 2.875, 3.0, 3.25, 3.5, 4.0, 4.5, 5.25, 6.0, 6.5, 8.0, 10.0, 12.0,
];

struct Pick {
    class: &'static str,
    rel: &'static str,
}

const PICKS: &[Pick] = &[
    Pick {
        class: "photo",
        rel: "CID22/CID22-512/training/1001682.png",
    },
    Pick {
        class: "photo",
        rel: "CID22/CID22-512/training/1028637.png",
    },
    Pick {
        class: "photo",
        rel: "CID22/CID22-512/training/1029604.png",
    },
    Pick {
        class: "photo",
        rel: "CID22/CID22-512/training/106399.png",
    },
    Pick {
        class: "photo",
        rel: "CID22/CID22-512/training/1080721.png",
    },
    Pick {
        class: "photo",
        rel: "CID22/CID22-512/training/1082342.png",
    },
    Pick {
        class: "photo",
        rel: "CID22/CID22-512/training/1089930.png",
    },
    Pick {
        class: "photo",
        rel: "CID22/CID22-512/training/110472.png",
    },
    Pick {
        class: "screenshot",
        rel: "gb82-sc/codec_wiki.png",
    },
    Pick {
        class: "screenshot",
        rel: "gb82-sc/gmessages.png",
    },
    Pick {
        class: "screenshot",
        rel: "gb82-sc/graph.png",
    },
    Pick {
        class: "screenshot",
        rel: "gb82-sc/gui.png",
    },
    Pick {
        class: "screenshot",
        rel: "gb82-sc/terminal.png",
    },
    Pick {
        class: "screenshot",
        rel: "gb82-sc/windows95.png",
    },
    Pick {
        class: "lineart",
        rel: "imazen-26/7000-lilith-plots/aliased-lines/7006_plots_line-00012-s2be0c08d_1024x1024.png",
    },
    Pick {
        class: "lineart",
        rel: "imazen-26/7000-lilith-plots/aliased-lines/7007_plots_line-00020-s1aac7045_1024x1024.png",
    },
    Pick {
        class: "lineart",
        rel: "imazen-26/7000-lilith-plots/grids/7037_plots_chart-heatmap-01-corporate-1024sq-mpl_1024x1024.png",
    },
    Pick {
        class: "web",
        rel: "imazen-26/8100-lilith-web-screenshots/375x667/8271_web-screenshots_archive-wayback-search_dpr1_page1_375x667.png",
    },
    Pick {
        class: "web",
        rel: "imazen-26/8100-lilith-web-screenshots/375x667/8272_web-screenshots_archives-exhibits_dpr1_page1_375x667.png",
    },
    Pick {
        class: "web",
        rel: "imazen-26/8100-lilith-web-screenshots/375x667/8273_web-screenshots_archives-exhibits_dpr1_page2_375x667.png",
    },
];

fn corpus_dir() -> PathBuf {
    std::env::var("CODEC_CORPUS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap_or_else(|_| "/home/lilith".into());
            PathBuf::from(format!("{home}/work/codec-corpus"))
        })
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
    (srgb * 255.0).round() as u8
}

fn rgb_to_linear_img(rgb: &[u8], w: u32, h: u32) -> Img<Vec<RGB<f32>>> {
    let pixels: Vec<RGB<f32>> = rgb
        .chunks(3)
        .map(|c| {
            RGB::new(
                srgb_to_linear_f32(c[0]),
                srgb_to_linear_f32(c[1]),
                srgb_to_linear_f32(c[2]),
            )
        })
        .collect();
    Img::new(pixels, w as usize, h as usize)
}

fn rgb_to_srgb_arr3(rgb: &[u8], w: u32, h: u32) -> Img<Vec<[u8; 3]>> {
    let pixels: Vec<[u8; 3]> = rgb.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();
    Img::new(pixels, w as usize, h as usize)
}

fn load_png(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
    let img = image::open(path).ok()?;
    let rgb = img.to_rgb8();
    Some((rgb.as_raw().clone(), rgb.width(), rgb.height()))
}

/// Centre-crop to at most `max_pixels` (square-ish, keeps the source's
/// odd/even parity out of the decision — crops are ≤ 1024 per axis).
fn center_crop(rgb: &[u8], w: u32, h: u32, max_pixels: u64) -> (Vec<u8>, u32, u32, bool) {
    if u64::from(w) * u64::from(h) <= max_pixels {
        return (rgb.to_vec(), w, h, false);
    }
    let cw = w.min(1024);
    let ch = h.min(1024);
    let x0 = (w - cw) / 2;
    let y0 = (h - ch) / 2;
    let mut out = Vec::with_capacity(cw as usize * ch as usize * 3);
    for y in y0..y0 + ch {
        let start = ((y * w + x0) * 3) as usize;
        out.extend_from_slice(&rgb[start..start + cw as usize * 3]);
    }
    (out, cw, ch, true)
}

fn decode_jxl_linear(bytes: &[u8]) -> Option<(usize, usize, Vec<f32>)> {
    let mut img = jxl_oxide::JxlImage::builder()
        .read(Cursor::new(bytes))
        .ok()?;
    img.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
        jxl_oxide::RenderingIntent::Relative,
    ));
    let render = img.render_frame(0).ok()?;
    let fb = render.image_all_channels();
    Some((fb.width(), fb.height(), fb.buf().to_vec()))
}

fn score(
    bytes: &[u8],
    orig_linear: &Img<Vec<RGB<f32>>>,
    orig_srgb: &Img<Vec<[u8; 3]>>,
    w: u32,
    h: u32,
) -> Option<(f64, f64)> {
    let (dw, dh, dec_lin) = decode_jxl_linear(bytes)?;
    if dw != w as usize || dh != h as usize {
        eprintln!("  decoded {dw}x{dh} != source {w}x{h} (dimension bug!)");
        return None;
    }
    let dec_pixels: Vec<RGB<f32>> = dec_lin
        .chunks(3)
        .map(|c| RGB::new(c[0], c[1], c[2]))
        .collect();
    let dec_lin_img: Img<Vec<RGB<f32>>> = Img::new(dec_pixels, dw, dh);
    let bfly = butteraugli_linear(
        orig_linear.as_ref(),
        dec_lin_img.as_ref(),
        &ButteraugliParams::default(),
    )
    .ok()?
    .score as f64;
    let dec_srgb_pixels: Vec<[u8; 3]> = dec_lin
        .chunks(3)
        .map(|c| {
            [
                linear_to_srgb_u8(c[0]),
                linear_to_srgb_u8(c[1]),
                linear_to_srgb_u8(c[2]),
            ]
        })
        .collect();
    let dec_srgb_img: Img<Vec<[u8; 3]>> = Img::new(dec_srgb_pixels, dw, dh);
    let ssim2 = fast_ssim2::compute_ssimulacra2(orig_srgb.as_ref(), dec_srgb_img.as_ref()).ok()?;
    Some((bfly, ssim2))
}

fn sha16(bytes: &[u8]) -> String {
    let d = Sha256::digest(bytes);
    d.iter().take(8).map(|b| format!("{b:02x}")).collect()
}

struct Cell {
    mode: &'static str,
    d_req: f32,
    cfg: LossyConfig,
}

fn cells_for(effort: u8) -> Vec<Cell> {
    let mut v = Vec::new();
    for &d in D_FULL {
        v.push(Cell {
            mode: "full",
            d_req: d,
            cfg: LossyConfig::new(d)
                .with_effort(effort)
                .with_auto_resampling(false),
        });
    }
    for &d in D_AUTO {
        v.push(Cell {
            mode: "auto",
            d_req: d,
            cfg: LossyConfig::new(d).with_effort(effort),
        });
    }
    for &t in T_RES2 {
        v.push(Cell {
            mode: "res2",
            d_req: t,
            cfg: LossyConfig::new(t).with_effort(effort).with_resampling(2),
        });
    }
    v
}

fn git_head() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "--short=12", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".into())
}

/// `MODE=cjxl`: reference-encoder cross-check (see module docs).
fn run_cjxl(out_dir: &Path, tag: &str, max_pixels: u64) {
    // libjxl v0.12 only — the packaged binary is v0.11.x and switches 2x
    // downsamplers at a different effort, which is exactly the trap that
    // produced a bogus differential in issue #102.
    let cjxl = jxl_encoder::test_helpers::cjxl_path();
    let version = std::process::Command::new(&cjxl)
        .arg("--version")
        .output()
        .ok()
        .map(|o| {
            String::from_utf8_lossy(&o.stdout)
                .lines()
                .next()
                .unwrap_or("")
                .to_string()
        })
        .unwrap_or_else(|| panic!("cannot run {cjxl} --version"));
    let tmp = out_dir.join(".arm_cjxl_tmp");
    fs::create_dir_all(&tmp).expect("tmp dir");
    let final_tsv = out_dir.join(format!("auto_resample_monotonicity_cjxl_{tag}.tsv"));
    let partial = out_dir.join(format!("auto_resample_monotonicity_cjxl_{tag}.tsv.partial"));
    let meta = out_dir.join(format!("auto_resample_monotonicity_cjxl_{tag}.meta"));
    let mut tsv = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&partial)
        .expect("open partial");
    writeln!(
        tsv,
        "image\tclass\tw\th\tcropped\teffort\tmode\td_req\td_internal\tresampling\tbytes\tbfly\tssim2\tenc_ms\tsha16"
    )
    .unwrap();
    let corpus = corpus_dir();
    let t_all = Instant::now();
    let (mut n_cells, mut n_fail) = (0usize, 0usize);
    for (pi, pick) in PICKS.iter().enumerate() {
        let path = corpus.join(pick.rel);
        let Some((rgb0, w0, h0)) = load_png(&path) else {
            eprintln!("SKIP (unreadable): {}", path.display());
            n_fail += 1;
            continue;
        };
        let (rgb, w, h, cropped) = center_crop(&rgb0, w0, h0, max_pixels);
        let name = Path::new(pick.rel)
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .to_string();
        // cjxl reads PNG from disk; crops go through a temp PNG.
        let input = if cropped {
            let p = tmp.join(format!("{name}_crop.png"));
            image::save_buffer(&p, &rgb, w, h, image::ColorType::Rgb8).expect("write crop png");
            p
        } else {
            path.clone()
        };
        let orig_lin = rgb_to_linear_img(&rgb, w, h);
        let orig_srgb = rgb_to_srgb_arr3(&rgb, w, h);
        let t_img = Instant::now();
        for &d in D_CJXL {
            for (mode, extra) in [("cjxl_auto", None), ("cjxl_full", Some("--resampling=1"))] {
                let out = tmp.join(format!("{name}_{mode}_{d}.jxl"));
                let mut cmd = std::process::Command::new(&cjxl);
                cmd.arg(&input)
                    .arg(&out)
                    .arg("-d")
                    .arg(format!("{d}"))
                    .arg("-e")
                    .arg("7")
                    .arg("--quiet");
                if let Some(x) = extra {
                    cmd.arg(x);
                }
                let t0 = Instant::now();
                let st = cmd.output();
                let enc_ms = t0.elapsed().as_secs_f64() * 1000.0;
                let ok = st.as_ref().map(|o| o.status.success()).unwrap_or(false);
                if !ok {
                    eprintln!(
                        "  cjxl failed {name} {mode} d={d}: {}",
                        st.map(|o| String::from_utf8_lossy(&o.stderr).to_string())
                            .unwrap_or_default()
                    );
                    n_fail += 1;
                    continue;
                }
                let bytes = fs::read(&out).expect("read cjxl output");
                let _ = fs::remove_file(&out);
                let (bfly, ssim2) = match score(&bytes, &orig_lin, &orig_srgb, w, h) {
                    Some(v) => v,
                    None => {
                        n_fail += 1;
                        (f64::NAN, f64::NAN)
                    }
                };
                writeln!(
                    tsv,
                    "{name}\t{}\t{w}\t{h}\t{}\t7\t{mode}\t{d:.3}\t{d:.4}\t0\t{}\t{bfly:.4}\t{ssim2:.3}\t{enc_ms:.1}\t{}",
                    pick.class,
                    u8::from(cropped),
                    bytes.len(),
                    sha16(&bytes),
                )
                .unwrap();
                n_cells += 1;
            }
        }
        if cropped {
            let _ = fs::remove_file(&input);
        }
        tsv.flush().unwrap();
        eprintln!(
            "[{}/{}] cjxl {name} ({w}x{h}{}): {} cells in {:.1}s (total {:.0}s)",
            pi + 1,
            PICKS.len(),
            if cropped { " crop" } else { "" },
            D_CJXL.len() * 2,
            t_img.elapsed().as_secs_f64(),
            t_all.elapsed().as_secs_f64()
        );
    }
    drop(tsv);
    let _ = fs::remove_dir(&tmp);
    fs::rename(&partial, &final_tsv).expect("rename partial -> final");
    let hostname = fs::read_to_string("/etc/hostname")
        .map(|s| s.trim().to_string())
        .unwrap_or_default();
    let mut m = fs::File::create(&meta).expect("meta");
    writeln!(
        m,
        "harness: jxl-encoder/examples/auto_resample_monotonicity.rs (MODE=cjxl)"
    )
    .unwrap();
    writeln!(m, "commit: {}", git_head()).unwrap();
    writeln!(m, "cjxl: {version} ({cjxl})").unwrap();
    writeln!(m, "host: {hostname}").unwrap();
    writeln!(m, "tag: {tag}").unwrap();
    writeln!(m, "effort: 7 (cjxl -e 7)").unwrap();
    writeln!(
        m,
        "D_CJXL: {D_CJXL:?}; modes: cjxl_auto (default flags) vs cjxl_full (--resampling=1)"
    )
    .unwrap();
    writeln!(
        m,
        "max_pixels (centre-crop above, to <=1024 per axis; crops fed as temp PNG): {max_pixels}"
    )
    .unwrap();
    writeln!(m, "metrics: jxl-oxide srgb_linear decode; butteraugli_linear (ButteraugliParams::default) at full res; fast_ssim2 compute_ssimulacra2 on sRGB u8").unwrap();
    writeln!(
        m,
        "cells: {n_cells}  failures: {n_fail}  wall_s: {:.0}",
        t_all.elapsed().as_secs_f64()
    )
    .unwrap();
    for p in PICKS {
        writeln!(m, "pick: {}\t{}", p.class, p.rel).unwrap();
    }
    writeln!(
        m,
        "command: MODE=cjxl cargo run -p jxl-encoder --release --example auto_resample_monotonicity"
    )
    .unwrap();
    eprintln!(
        "wrote {} and {} ({n_cells} cells, {n_fail} failures)",
        final_tsv.display(),
        meta.display()
    );
}

fn main() {
    let efforts: Vec<u8> = std::env::var("EFFORTS")
        .unwrap_or_else(|_| "5,8".into())
        .split(',')
        .map(|s| s.trim().parse().expect("EFFORTS: u8 list"))
        .collect();
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap_or_else(|_| "benchmarks".into()));
    let max_pixels: u64 = std::env::var("MAX_PIXELS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1_100_000);
    let tag = std::env::var("TAG").unwrap_or_else(|_| {
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        // YYYY-MM-DD from epoch (UTC), no chrono dep.
        let days = secs / 86_400;
        let (y, m, d) = civil_from_days(days as i64);
        format!("{y:04}-{m:02}-{d:02}")
    });
    fs::create_dir_all(&out_dir).expect("create OUT_DIR");
    if std::env::var("MODE").map(|m| m == "cjxl").unwrap_or(false) {
        run_cjxl(&out_dir, &tag, max_pixels);
        return;
    }
    let final_tsv = out_dir.join(format!("auto_resample_monotonicity_{tag}.tsv"));
    let partial = out_dir.join(format!("auto_resample_monotonicity_{tag}.tsv.partial"));
    let meta = out_dir.join(format!("auto_resample_monotonicity_{tag}.meta"));

    let mut tsv = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&partial)
        .expect("open partial");
    writeln!(
        tsv,
        "image\tclass\tw\th\tcropped\teffort\tmode\td_req\td_internal\tresampling\tbytes\tbfly\tssim2\tenc_ms\tsha16"
    )
    .unwrap();

    let corpus = corpus_dir();
    let lim = Limits::default().with_max_memory_bytes(8u64 << 30);
    let t_all = Instant::now();
    let mut n_cells = 0usize;
    let mut n_fail = 0usize;
    for (pi, pick) in PICKS.iter().enumerate() {
        let path = corpus.join(pick.rel);
        let Some((rgb0, w0, h0)) = load_png(&path) else {
            eprintln!("SKIP (unreadable): {}", path.display());
            n_fail += 1;
            continue;
        };
        let (rgb, w, h, cropped) = center_crop(&rgb0, w0, h0, max_pixels);
        let orig_lin = rgb_to_linear_img(&rgb, w, h);
        let orig_srgb = rgb_to_srgb_arr3(&rgb, w, h);
        let name = Path::new(pick.rel)
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .to_string();
        for &e in &efforts {
            let cells = cells_for(e);
            let t_img = Instant::now();
            for cell in &cells {
                let t0 = Instant::now();
                let bytes = match cell
                    .cfg
                    .clone()
                    .encode_request(w, h, PixelLayout::Rgb8)
                    .with_limits(&lim)
                    .encode(&rgb)
                {
                    Ok(b) => b,
                    Err(err) => {
                        eprintln!(
                            "  encode failed {name} e{e} {} d={}: {err:?}",
                            cell.mode, cell.d_req
                        );
                        n_fail += 1;
                        continue;
                    }
                };
                let enc_ms = t0.elapsed().as_secs_f64() * 1000.0;
                let (bfly, ssim2) = match score(&bytes, &orig_lin, &orig_srgb, w, h) {
                    Some(s) => s,
                    None => {
                        n_fail += 1;
                        (f64::NAN, f64::NAN)
                    }
                };
                writeln!(
                    tsv,
                    "{name}\t{}\t{w}\t{h}\t{}\t{e}\t{}\t{:.3}\t{:.4}\t{}\t{}\t{bfly:.4}\t{ssim2:.3}\t{enc_ms:.1}\t{}",
                    pick.class,
                    u8::from(cropped),
                    cell.mode,
                    cell.d_req,
                    cell.cfg.effective_distance(),
                    cell.cfg.effective_resampling(),
                    bytes.len(),
                    sha16(&bytes),
                )
                .unwrap();
                n_cells += 1;
            }
            tsv.flush().unwrap();
            eprintln!(
                "[{}/{}] {name} ({}x{}{}) e{e}: {} cells in {:.1}s (total {:.0}s)",
                pi + 1,
                PICKS.len(),
                w,
                h,
                if cropped { " crop" } else { "" },
                cells.len(),
                t_img.elapsed().as_secs_f64(),
                t_all.elapsed().as_secs_f64()
            );
        }
    }
    drop(tsv);
    fs::rename(&partial, &final_tsv).expect("rename partial -> final");

    let hostname = fs::read_to_string("/etc/hostname")
        .map(|s| s.trim().to_string())
        .unwrap_or_default();
    let mut m = fs::File::create(&meta).expect("meta");
    writeln!(
        m,
        "harness: jxl-encoder/examples/auto_resample_monotonicity.rs"
    )
    .unwrap();
    writeln!(m, "commit: {}", git_head()).unwrap();
    writeln!(m, "jxl-encoder: {}", env!("CARGO_PKG_VERSION")).unwrap();
    writeln!(m, "host: {hostname}").unwrap();
    writeln!(m, "tag: {tag}").unwrap();
    writeln!(m, "efforts: {efforts:?}").unwrap();
    writeln!(
        m,
        "max_pixels (centre-crop above, to <=1024 per axis): {max_pixels}"
    )
    .unwrap();
    writeln!(m, "D_FULL (auto off): {D_FULL:?}").unwrap();
    writeln!(
        m,
        "D_AUTO (default path, 2x + libjxl remap d*0.25+0.25): {D_AUTO:?}"
    )
    .unwrap();
    writeln!(
        m,
        "T_RES2 (explicit with_resampling(2), internal distance): {T_RES2:?}"
    )
    .unwrap();
    writeln!(m, "metrics: jxl-oxide srgb_linear decode; butteraugli_linear (ButteraugliParams::default) at full res; fast_ssim2 compute_ssimulacra2 on sRGB u8").unwrap();
    writeln!(m, "self-check: every auto row must be byte-identical (sha16) to the res2 row at t = d*0.25+0.25").unwrap();
    writeln!(
        m,
        "cells: {n_cells}  failures: {n_fail}  wall_s: {:.0}",
        t_all.elapsed().as_secs_f64()
    )
    .unwrap();
    writeln!(m, "corpus_dir: {}", corpus.display()).unwrap();
    for p in PICKS {
        writeln!(m, "pick: {}\t{}", p.class, p.rel).unwrap();
    }
    writeln!(m, "command: EFFORTS={} cargo run -p jxl-encoder --release --example auto_resample_monotonicity",
        efforts.iter().map(|e| e.to_string()).collect::<Vec<_>>().join(",")).unwrap();
    eprintln!(
        "wrote {} and {} ({n_cells} cells, {n_fail} failures)",
        final_tsv.display(),
        meta.display()
    );
}

/// Howard Hinnant's days-from-civil inverse (UTC date from epoch days).
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}
