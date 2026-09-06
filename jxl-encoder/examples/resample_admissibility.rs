//! Is 2× resampling ever the cheaper regime on imazen-26, and can a sound
//! rule decide it per image? (issue #101 follow-up, 2026-09-05.)
//!
//! The monotonicity baseline (`examples/auto_resample_monotonicity`) sampled
//! only 6 imazen-26 images, all graphics classes. This harness sweeps the
//! committed 43-image stratified pick list — all 23 strata, photos through
//! patents — and adds the quantity a *sound* rule needs.
//!
//! ## The admissibility bound
//!
//! `resample_roundtrip_2x_rgb` (down→up in the opsin domain, no quantisation)
//! is what a decoder reconstructs from a PERFECTLY coded 2× frame. No bitrate
//! can beat it. Scoring it against the source gives the **resampling floor**:
//! if `floor > requested distance`, the 2× regime provably cannot hit the
//! target, whatever the encoder does. That is a bound, not a fitted threshold,
//! and it costs one downsample + one upsample + one metric call — no encode.
//! Both kernels are measured (sharper = e ≤ 9, iterative = e ≥ 10) because the
//! iterative one is decoder-adjoint and may lower the floor enough to change
//! the answer at high effort.
//!
//! ## Ground truth per cell
//!
//! For each full-resolution ladder point `d` we interpolate, on the 2× regime's
//! own distance curve, the bytes it needs to reach the SAME butteraugli. The
//! ratio `bytes_2x / bytes_full` at matched quality is the only honest
//! comparison; a matched-*distance* comparison confuses "spent fewer bytes"
//! with "delivered less quality" (which is what makes libjxl's fixed threshold
//! look reasonable on graphics). `admissible = ratio < 1`.
//!
//! ## Axes
//!
//! 43 images × size caps {512, 1024} (centre crop; duplicates deduped) ×
//! efforts {5, 8} × (13 full-res distances + 14 forced-2× internal distances).
//! Per image × size: both floors + every zenanalyze feature the default build
//! exposes, so a rule can be fit and cross-validated by stratum.
//!
//! Env: `PICKS_TSV`, `CODEC_CORPUS_DIR`, `EFFORTS` (default `5,8`),
//! `SIZE_CAPS` (default `512,1024`), `MAX_IMAGES`, `OUT_DIR`, `TAG`.
//!
//! Output: `<OUT_DIR>/resample_admissibility_<TAG>.{cells.tsv,images.tsv,meta}`
//! Reproducer: `cargo run -p jxl-encoder --release --features __internals --example resample_admissibility`

use std::fs::{self, OpenOptions};
use std::io::{Cursor, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use butteraugli::{ButteraugliParams, butteraugli_linear};
use imgref::Img;
use jxl_encoder::__internals::{Downsample2xKernel, resample_roundtrip_2x_rgb};
use jxl_encoder::api::{Limits, LossyConfig, PixelLayout};
use rgb::RGB;

/// Full-resolution ladder. Spans low distances (where the floor should make 2×
/// hopeless) through the aggressive end, so the crossing is bracketed.
const D_FULL: &[f32] = &[
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 18.0, 21.0, 25.0,
];
/// Forced-2× ladder, parametrised by the encoder's INTERNAL distance.
const T_RES2: &[f32] = &[
    0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0,
];

fn env_list_f(name: &str, default: &str) -> Vec<u8> {
    std::env::var(name)
        .unwrap_or_else(|_| default.into())
        .split(',')
        .map(|s| {
            s.trim()
                .parse()
                .unwrap_or_else(|_| panic!("{name}: u8 list"))
        })
        .collect()
}

fn env_list_u32(name: &str, default: &str) -> Vec<u32> {
    std::env::var(name)
        .unwrap_or_else(|_| default.into())
        .split(',')
        .map(|s| {
            s.trim()
                .parse()
                .unwrap_or_else(|_| panic!("{name}: u32 list"))
        })
        .collect()
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

fn rgb_to_linear_planar_vec(rgb: &[u8]) -> Vec<f32> {
    rgb.iter().map(|&b| srgb_to_linear_f32(b)).collect()
}

fn linear_to_img(linear: &[f32], w: u32, h: u32) -> Img<Vec<RGB<f32>>> {
    let px: Vec<RGB<f32>> = linear
        .chunks(3)
        .map(|c| RGB::new(c[0], c[1], c[2]))
        .collect();
    Img::new(px, w as usize, h as usize)
}

fn linear_to_srgb_img(linear: &[f32], w: u32, h: u32) -> Img<Vec<[u8; 3]>> {
    let px: Vec<[u8; 3]> = linear
        .chunks(3)
        .map(|c| {
            [
                linear_to_srgb_u8(c[0]),
                linear_to_srgb_u8(c[1]),
                linear_to_srgb_u8(c[2]),
            ]
        })
        .collect();
    Img::new(px, w as usize, h as usize)
}

fn load_rgb8(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
    let img = image::open(path).ok()?;
    let rgb = img.to_rgb8();
    Some((rgb.as_raw().clone(), rgb.width(), rgb.height()))
}

/// Centre crop to at most `cap` per axis (no upscaling, no resampling — the
/// study must not pre-filter its own input).
fn center_crop(rgb: &[u8], w: u32, h: u32, cap: u32) -> (Vec<u8>, u32, u32) {
    let cw = w.min(cap);
    let ch = h.min(cap);
    if cw == w && ch == h {
        return (rgb.to_vec(), w, h);
    }
    let (x0, y0) = ((w - cw) / 2, (h - ch) / 2);
    let mut out = Vec::with_capacity(cw as usize * ch as usize * 3);
    for y in y0..y0 + ch {
        let start = ((y * w + x0) * 3) as usize;
        out.extend_from_slice(&rgb[start..start + cw as usize * 3]);
    }
    (out, cw, ch)
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

fn score_linear(
    dist_linear: &[f32],
    w: u32,
    h: u32,
    orig_lin: &Img<Vec<RGB<f32>>>,
    orig_srgb: &Img<Vec<[u8; 3]>>,
) -> Option<(f64, f64)> {
    let dist_img = linear_to_img(dist_linear, w, h);
    let bfly = butteraugli_linear(
        orig_lin.as_ref(),
        dist_img.as_ref(),
        &ButteraugliParams::default(),
    )
    .ok()?
    .score as f64;
    let dist_srgb = linear_to_srgb_img(dist_linear, w, h);
    let ssim2 = fast_ssim2::compute_ssimulacra2(orig_srgb.as_ref(), dist_srgb.as_ref()).ok()?;
    Some((bfly, ssim2))
}

/// Every feature the default zenanalyze build exposes, as `(name, value)`.
fn zenanalyze_features(rgb: &[u8], w: u32, h: u32) -> Vec<(&'static str, f32)> {
    use zenanalyze::feature::{AnalysisFeature, AnalysisQuery, FeatureSet};
    let mut set = FeatureSet::new();
    let mut wanted: Vec<AnalysisFeature> = Vec::new();
    for id in 0u16..256 {
        if let Some(f) = AnalysisFeature::from_u16(id) {
            set = set.with(f);
            wanted.push(f);
        }
    }
    let query = AnalysisQuery::new(set);
    let Ok(res) = zenanalyze::try_analyze_features_rgb8(rgb, w, h, &query) else {
        return Vec::new();
    };
    let mut out = Vec::with_capacity(wanted.len());
    for f in wanted {
        let v = res.get_f32(f).or_else(|| res.get(f).map(|fv| fv.to_f32()));
        if let Some(v) = v
            && v.is_finite()
        {
            out.push((f.name(), v));
        }
    }
    out
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

fn utc_date_tag() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let z = (secs / 86_400) as i64 + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}")
}

struct Pick {
    stratum: String,
    path: PathBuf,
    name: String,
}

fn read_picks() -> Vec<Pick> {
    let tsv = std::env::var("PICKS_TSV")
        .unwrap_or_else(|_| "benchmarks/lossless_bench_set_2026-06-10.tsv".into());
    let text = fs::read_to_string(&tsv).unwrap_or_else(|e| panic!("read {tsv}: {e}"));
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().expect("header").split('\t').collect();
    let col = |n: &str| {
        header
            .iter()
            .position(|h| *h == n)
            .unwrap_or_else(|| panic!("column {n}"))
    };
    let (c_str, c_in, c_desc) = (col("stratum"), col("bench_input"), col("descriptor"));
    let mut out = Vec::new();
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        out.push(Pick {
            stratum: f[c_str].to_string(),
            path: PathBuf::from(f[c_in]),
            name: f[c_desc].to_string(),
        });
    }
    if let Ok(n) = std::env::var("MAX_IMAGES") {
        out.truncate(n.parse().expect("MAX_IMAGES"));
    }
    out
}

fn main() {
    let efforts = env_list_f("EFFORTS", "5,8");
    let size_caps = env_list_u32("SIZE_CAPS", "512,1024");
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap_or_else(|_| "benchmarks".into()));
    let tag = std::env::var("TAG").unwrap_or_else(|_| utc_date_tag());
    fs::create_dir_all(&out_dir).expect("create OUT_DIR");

    let cells_final = out_dir.join(format!("resample_admissibility_{tag}.cells.tsv"));
    let images_final = out_dir.join(format!("resample_admissibility_{tag}.images.tsv"));
    let meta_final = out_dir.join(format!("resample_admissibility_{tag}.meta"));
    let cells_partial = cells_final.with_extension("tsv.partial");
    let images_partial = images_final.with_extension("tsv.partial");

    let mut cells = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&cells_partial)
        .expect("open cells");
    writeln!(
        cells,
        "image\tstratum\tw\th\tsize_cap\teffort\tmode\td_req\td_internal\tresampling\tbytes\tbfly\tssim2\tenc_ms"
    )
    .unwrap();
    let mut images = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&images_partial)
        .expect("open images");
    writeln!(
        images,
        "image\tstratum\tw\th\tsize_cap\tfloor_kernel\tfloor_bfly\tfloor_ssim2\tfeature\tvalue"
    )
    .unwrap();

    let picks = read_picks();
    let lim = Limits::default().with_max_memory_bytes(8u64 << 30);
    let t_all = Instant::now();
    let (mut n_cells, mut n_fail, mut n_img) = (0usize, 0usize, 0usize);

    for (pi, pick) in picks.iter().enumerate() {
        let Some((rgb0, w0, h0)) = load_rgb8(&pick.path) else {
            eprintln!("SKIP (unreadable): {}", pick.path.display());
            n_fail += 1;
            continue;
        };
        let mut seen_dims: Vec<(u32, u32)> = Vec::new();
        for &cap in &size_caps {
            let (rgb, w, h) = center_crop(&rgb0, w0, h0, cap);
            if seen_dims.contains(&(w, h)) {
                // Same crop as a smaller cap (image is below this cap): the
                // cell would be a duplicate, not a skipped measurement.
                continue;
            }
            seen_dims.push((w, h));
            n_img += 1;
            let t_img = Instant::now();
            let orig_linear = rgb_to_linear_planar_vec(&rgb);
            let orig_lin_img = linear_to_img(&orig_linear, w, h);
            let orig_srgb_img = linear_to_srgb_img(&orig_linear, w, h);

            // ── the admissibility bound: floors, no encode ──
            for (kname, kernel) in [
                ("sharper", Downsample2xKernel::Sharper),
                ("iterative", Downsample2xKernel::Iterative),
            ] {
                let round = resample_roundtrip_2x_rgb(&orig_linear, w as usize, h as usize, kernel);
                let (fb, fs) =
                    score_linear(&round, w, h, &orig_lin_img, &orig_srgb_img).expect("floor score");
                writeln!(
                    images,
                    "{}\t{}\t{w}\t{h}\t{cap}\t{kname}\t{fb:.4}\t{fs:.3}\t\t",
                    pick.name, pick.stratum
                )
                .unwrap();
            }
            // ── features (once per image × size) ──
            for (fname, fval) in zenanalyze_features(&rgb, w, h) {
                writeln!(
                    images,
                    "{}\t{}\t{w}\t{h}\t{cap}\t\t\t\t{fname}\t{fval}",
                    pick.name, pick.stratum
                )
                .unwrap();
            }
            images.flush().unwrap();

            // ── the two regimes ──
            for &e in &efforts {
                let mut plan: Vec<(&str, f32, LossyConfig)> = Vec::new();
                for &d in D_FULL {
                    plan.push((
                        "full",
                        d,
                        LossyConfig::new(d)
                            .with_effort(e)
                            .with_auto_resampling(false),
                    ));
                }
                for &t in T_RES2 {
                    plan.push((
                        "res2",
                        t,
                        LossyConfig::new(t).with_effort(e).with_resampling(2),
                    ));
                }
                for (mode, d_req, cfg) in plan {
                    let t0 = Instant::now();
                    let bytes = match cfg
                        .clone()
                        .encode_request(w, h, PixelLayout::Rgb8)
                        .with_limits(&lim)
                        .encode(&rgb)
                    {
                        Ok(b) => b,
                        Err(err) => {
                            eprintln!(
                                "  encode failed {} e{e} {mode} d={d_req}: {err:?}",
                                pick.name
                            );
                            n_fail += 1;
                            continue;
                        }
                    };
                    let enc_ms = t0.elapsed().as_secs_f64() * 1000.0;
                    let (bfly, ssim2) = match decode_jxl_linear(&bytes) {
                        Some((dw, dh, lin)) if dw == w as usize && dh == h as usize => {
                            score_linear(&lin, w, h, &orig_lin_img, &orig_srgb_img)
                                .unwrap_or((f64::NAN, f64::NAN))
                        }
                        Some((dw, dh, _)) => {
                            eprintln!(
                                "  decoded {dw}x{dh} != {w}x{h} for {} {mode} d={d_req}",
                                pick.name
                            );
                            n_fail += 1;
                            (f64::NAN, f64::NAN)
                        }
                        None => {
                            n_fail += 1;
                            (f64::NAN, f64::NAN)
                        }
                    };
                    writeln!(
                        cells,
                        "{}\t{}\t{w}\t{h}\t{cap}\t{e}\t{mode}\t{d_req:.3}\t{:.4}\t{}\t{}\t{bfly:.4}\t{ssim2:.3}\t{enc_ms:.1}",
                        pick.name,
                        pick.stratum,
                        cfg.effective_distance(),
                        cfg.effective_resampling(),
                        bytes.len(),
                    )
                    .unwrap();
                    n_cells += 1;
                }
            }
            cells.flush().unwrap();
            eprintln!(
                "[{}/{}] {} ({}) {w}x{h} cap={cap}: {} cells in {:.1}s (total {:.0}s)",
                pi + 1,
                picks.len(),
                pick.name,
                pick.stratum,
                efforts.len() * (D_FULL.len() + T_RES2.len()),
                t_img.elapsed().as_secs_f64(),
                t_all.elapsed().as_secs_f64()
            );
        }
    }
    drop(cells);
    drop(images);
    fs::rename(&cells_partial, &cells_final).expect("rename cells");
    fs::rename(&images_partial, &images_final).expect("rename images");

    let hostname = fs::read_to_string("/etc/hostname")
        .map(|s| s.trim().to_string())
        .unwrap_or_default();
    let mut m = fs::File::create(&meta_final).expect("meta");
    writeln!(m, "harness: jxl-encoder/examples/resample_admissibility.rs").unwrap();
    writeln!(m, "commit: {}", git_head()).unwrap();
    writeln!(m, "jxl-encoder: {}", env!("CARGO_PKG_VERSION")).unwrap();
    writeln!(m, "host: {hostname}").unwrap();
    writeln!(m, "tag: {tag}").unwrap();
    writeln!(m, "picks: benchmarks/lossless_bench_set_2026-06-10.tsv ({} images, imazen-26 k-means stratified)", picks.len()).unwrap();
    writeln!(
        m,
        "size_caps (centre crop, no upscale, duplicates deduped): {size_caps:?}"
    )
    .unwrap();
    writeln!(m, "efforts: {efforts:?}").unwrap();
    writeln!(m, "D_FULL (auto off): {D_FULL:?}").unwrap();
    writeln!(
        m,
        "T_RES2 (explicit with_resampling(2), internal distance): {T_RES2:?}"
    )
    .unwrap();
    writeln!(m, "floor: __internals::resample_roundtrip_2x_rgb (opsin down->up, NO quantisation) scored vs source; kernels sharper (e<=9) + iterative (e>=10)").unwrap();
    writeln!(m, "metrics: jxl-oxide srgb_linear decode; butteraugli_linear (ButteraugliParams::default) at FULL resolution; fast_ssim2 compute_ssimulacra2 on sRGB u8").unwrap();
    writeln!(m, "ground truth: bytes_2x interpolated on the res2 bfly curve at each full-res point's bfly; admissible = ratio < 1 (matched QUALITY, never matched distance)").unwrap();
    writeln!(
        m,
        "image_cells: {n_img}  encode_cells: {n_cells}  failures: {n_fail}  wall_s: {:.0}",
        t_all.elapsed().as_secs_f64()
    )
    .unwrap();
    writeln!(m, "command: EFFORTS={} SIZE_CAPS={} cargo run -p jxl-encoder --release --features __internals --example resample_admissibility",
        efforts.iter().map(|e| e.to_string()).collect::<Vec<_>>().join(","),
        size_caps.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(",")).unwrap();
    eprintln!(
        "wrote {} / {} / {} ({n_cells} cells, {n_img} image-size rows, {n_fail} failures)",
        cells_final.display(),
        images_final.display(),
        meta_final.display()
    );
}
