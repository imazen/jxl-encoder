// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Diffmap RD harness (#38, 2026-05-27): encode a small multi-content corpus
//! with a chosen perceptual-loop metric/diffmap, save the decoded output, and
//! emit a manifest TSV. An EXTERNAL independent panel (`zen-metrics compare`:
//! butteraugli + dssim + ssim2) then scores the decoded outputs vs the clean
//! reference — the CIRCULARITY guard: a zensim/cvvdp-driven encode is NEVER
//! judged by the metric that drove it.
//!
//! Compares the three diffmap avenues for "which diffmap drives jxl-encoder
//! best" (user 2026-05-27):
//!   1. `--metric zensim` (+ ZENSIM_* env DiffmapOptions sweep)
//!   2. `--metric cvvdp`  (cvvdp's own diffmap)
//!   3. `--metric butteraugli` (the default baseline reference)
//!    (Avenue 2 — the #33 structural diffmap — is a separate code path added
//!    to the zensim loop; run it via `--metric zensim` once that env knob
//!    exists.)
//!
//! ## Run
//!
//! ```bash
//! cargo run --release -p jxl-encoder \
//!   --features "__expert butteraugli-loop zensim-loop cvvdp-loop-cpu ssim2-loop parallel" \
//!   --example zensim_diffmap_rd -- \
//!   --metric zensim --label zensim_v47_default \
//!   --out-dir /mnt/v/output/zensim/diffmap-rd-2026-05-27
//! ```
//!
//! Then score (independent panel) + build RD with the companion script.

#![cfg(feature = "zensim-loop")]

use std::fs;
use std::io::Cursor;
use std::path::PathBuf;
use std::time::Instant;

use jxl_encoder::api::{EncoderStrategy, PerceptualMetric};
use jxl_encoder::{LossyConfig, PixelLayout};

const CID22_VAL_DIR: &str = "/home/lilith/work/codec-corpus/CID22/CID22-512/validation";
const GB82_SC_DIR: &str = "/home/lilith/work/codec-corpus/gb82-sc";

/// (name, content-class, absolute path). Default = the 3 calibration-seed
/// images; override with `--corpus-file <tsv>` (cols: path, name, class).
fn default_corpus() -> Vec<(String, String, String)> {
    vec![
        (
            "1418519".into(),
            "photo".into(),
            format!("{CID22_VAL_DIR}/1418519.png"),
        ),
        (
            "1025469".into(),
            "photo".into(),
            format!("{CID22_VAL_DIR}/1025469.png"),
        ),
        (
            "codec_wiki".into(),
            "screen".into(),
            format!("{GB82_SC_DIR}/codec_wiki.png"),
        ),
    ]
}

/// Parse a corpus TSV: each line `abs_path \t name \t class` (class optional,
/// defaults to "image"). Blank lines + `#` comments skipped.
fn corpus_from_file(path: &str) -> Vec<(String, String, String)> {
    let mut out = Vec::new();
    for line in std::fs::read_to_string(path).expect("corpus file").lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut f = line.split('\t');
        let p = f.next().unwrap_or("").to_string();
        let name = f.next().map(|s| s.to_string()).unwrap_or_else(|| {
            std::path::Path::new(&p)
                .file_stem()
                .unwrap()
                .to_string_lossy()
                .into_owned()
        });
        let class = f.next().unwrap_or("image").to_string();
        if !p.is_empty() {
            out.push((name, class, p));
        }
    }
    out
}

const EFFORT: u8 = 8;

fn load_rgb8(path: &str) -> Result<(Vec<u8>, u32, u32), Box<dyn std::error::Error + Send + Sync>> {
    let img = image::open(path)?.to_rgb8();
    let (w, h) = (img.width(), img.height());
    Ok((img.into_raw(), w, h))
}

fn decode_jxl_srgb_u8(
    encoded: &[u8],
    w: u32,
    h: u32,
) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
    let decoder = jxl_oxide::JxlImage::builder().read(Cursor::new(encoded))?;
    let frame = decoder.render_frame(0)?;
    let stream = frame.stream();
    let (dec_w, dec_h) = (stream.width(), stream.height());
    if dec_w != w || dec_h != h {
        return Err(format!("decoded dims {dec_w}x{dec_h} != {w}x{h}").into());
    }
    let ch = stream.channels() as usize;
    let mut f32buf: Vec<f32> = vec![0.0; (dec_w as usize) * (dec_h as usize) * ch];
    let mut s = stream;
    let _ = s.write_to_buffer(&mut f32buf);
    let n_px = (dec_w as usize) * (dec_h as usize);
    let mut rgb = Vec::with_capacity(n_px * 3);
    for i in 0..n_px {
        let r = (f32buf[i * ch].clamp(0.0, 1.0) * 255.0).round() as u8;
        let g = if ch >= 2 {
            (f32buf[i * ch + 1].clamp(0.0, 1.0) * 255.0).round() as u8
        } else {
            r
        };
        let b = if ch >= 3 {
            (f32buf[i * ch + 2].clamp(0.0, 1.0) * 255.0).round() as u8
        } else {
            r
        };
        rgb.extend_from_slice(&[r, g, b]);
    }
    Ok(rgb)
}

fn parse_metric(s: &str) -> PerceptualMetric {
    match s {
        "zensim" => PerceptualMetric::Zensim,
        #[cfg(any(feature = "cvvdp-loop", feature = "cvvdp-loop-cpu"))]
        "cvvdp" => PerceptualMetric::Cvvdp,
        "butteraugli" => PerceptualMetric::Butteraugli,
        other => {
            eprintln!("unknown/feature-gated metric '{other}', falling back to butteraugli");
            PerceptualMetric::Butteraugli
        }
    }
}

/// C3b (zensim task #67): judge profile — the SAME bake that drives the
/// loop, scoring decoded-vs-ref (native zensim, higher = better). One bake
/// per process (OnceLock; the loop's own RD_PROFILE cache has the same
/// constraint — the outer script runs one process per bake).
static JUDGE_BYTES: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
fn judge_bytes() -> &'static [u8] {
    JUDGE_BYTES.get().expect("judge bake set").as_slice()
}
static JUDGE: std::sync::OnceLock<zensim::ZensimProfile> = std::sync::OnceLock::new();

fn judge_profile(bake_path: &str) -> zensim::ZensimProfile {
    *JUDGE.get_or_init(|| {
        // Metric-matrix study (2026-07-31): `profile:<a|b|latest>` judges
        // with the NAMED profile (the loop scorer is set to the same name
        // by `set_rd_profile_env`) — the same-scorer judging contract,
        // without mounting bytes.
        if let Some(name) = bake_path.strip_prefix("profile:") {
            return match name {
                "b" => zensim::ZensimProfile::B,
                "latest" => zensim::ZensimProfile::latest_preview(),
                "a" =>
                {
                    #[allow(deprecated)]
                    zensim::ZensimProfile::A
                }
                other => panic!("unsupported judge profile:{other}"),
            };
        }
        let bytes = std::fs::read(bake_path).expect("judge bake read");
        JUDGE_BYTES.set(bytes).expect("judge bytes once");
        let params = zensim::profile::ProfileParams::builder()
            .mlp(judge_bytes)
            .skip_score_mapping(true)
            .extrapolate_score(true)
            .extended_features(true)
            .compute_iw_features(true)
            .build();
        let params: &'static zensim::profile::ProfileParams = Box::leak(Box::new(params));
        zensim::ZensimProfile::Custom {
            params,
            name: "judge-bake",
        }
    })
}

fn judge_score(bake_path: &str, ref_rgb: &[u8], dec_rgb: &[u8], w: u32, h: u32) -> f64 {
    let profile = judge_profile(bake_path);
    let z = zensim::Zensim::new(profile).with_parallel(false);
    let rp: Vec<[u8; 3]> = ref_rgb
        .chunks_exact(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect();
    let dp: Vec<[u8; 3]> = dec_rgb
        .chunks_exact(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect();
    let rs = zensim::RgbSlice::new(&rp, w as usize, h as usize);
    let ds = zensim::RgbSlice::new(&dp, w as usize, h as usize);
    z.compute(&rs, &ds).map(|r| r.score()).unwrap_or(f64::NAN)
}

/// Metric-matrix study (2026-07-31): one place maps `--bake` to the loop's
/// `JXL_ZENSIM_RD_PROFILE` — `profile:<a|b|latest>` selects a NAMED profile,
/// anything else mounts bake bytes (`bake:<path>`, the prior studies' path).
fn set_rd_profile_env(bake: &str) {
    // SAFETY: edition-2024 set_var — single-threaded driver; all encodes run
    // sequentially after this write.
    unsafe {
        if let Some(name) = bake.strip_prefix("profile:") {
            std::env::set_var("JXL_ZENSIM_RD_PROFILE", name);
        } else {
            std::env::set_var("JXL_ZENSIM_RD_PROFILE", format!("bake:{bake}"));
        }
    }
}

/// Metric-matrix study (2026-07-31): SSIMULACRA2 via the canonical owner —
/// shell zenmetrics `batch --metric ssim2` on (ref, dist) PNGs. Returns NaN
/// on any failure (a recorded null per the protocol, never a silent skip).
fn ssim2_score_via_zenmetrics(
    bin: &str,
    ref_png: &std::path::Path,
    dist_png: &std::path::Path,
    scratch_dir: &std::path::Path,
) -> f64 {
    let pairs = scratch_dir.join(".ssim2_pairs_tmp.tsv");
    let out = scratch_dir.join(".ssim2_out_tmp.tsv");
    let _ = fs::remove_file(&out);
    if fs::write(
        &pairs,
        format!(
            "ref_path\tdist_path\n{}\t{}\n",
            ref_png.display(),
            dist_png.display()
        ),
    )
    .is_err()
    {
        return f64::NAN;
    }
    let status = std::process::Command::new(bin)
        .args(["batch", "--metric", "ssim2", "--pairs"])
        .arg(&pairs)
        .arg("--output")
        .arg(&out)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();
    if !status.map(|s| s.success()).unwrap_or(false) {
        return f64::NAN;
    }
    fs::read_to_string(&out)
        .ok()
        .and_then(|s| {
            s.lines()
                .nth(1)
                .and_then(|l| l.split('\t').nth(2))
                .and_then(|v| v.parse().ok())
        })
        .unwrap_or(f64::NAN)
}

/// Seed distance for the target controller (rough; the controller owns
/// convergence — the seed only sets iteration-1's operating point).
fn seed_distance_for_target(t: f64) -> f32 {
    if t >= 90.0 {
        0.9
    } else if t >= 80.0 {
        1.5
    } else {
        2.5
    }
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut metric_s = "zensim".to_string();
    let mut label = "zensim_v47_default".to_string();
    let mut out_dir = PathBuf::from("/mnt/v/output/zensim/diffmap-rd-2026-05-27");
    // Diffmap-driven per-tile redistribution only runs when iters >= 1 (iter 0
    // is compare-only). With iters=0 the diffmap is computed but NEVER used, so
    // every DiffmapOptions/metric variant produces byte-identical output —
    // useless for a diffmap comparison. Default to 3 redistribution iters.
    let mut iters: u32 = 3;
    let mut corpus_file: Option<String> = None;
    let mut distances: Vec<f32> = vec![1.0, 2.0, 3.0];
    // C3b target-A/B mode (activated by --zensim-targets): per
    // (fixture × target × arm) encode with the shared target controller and
    // judge decoded output with the SAME bake.
    let mut zensim_targets: Vec<f64> = Vec::new();
    let mut arms: Vec<String> = vec![
        "baseline".into(),
        "abs".into(),
        "attr".into(),
        "attr-stale".into(),
    ];
    let mut bake: Option<String> = None;
    // Efficiency study E7 (2026-07-31): bytes-target outer-loop mode.
    let mut bytes_targets_file: Option<String> = None;
    // Metric-matrix study (2026-07-31): SCORE-target outer-loop mode —
    // `--score-targets-outer <zensim|ssim2>` (targets from --zensim-targets,
    // in the arm metric's own units).
    let mut score_targets_outer: Option<String> = None;
    let mut ssim2_bin = "/home/lilith/work/zen/zenmetrics/target/release/zenmetrics".to_string();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--bytes-targets-file" => bytes_targets_file = args.next(),
            "--score-targets-outer" => score_targets_outer = args.next(),
            "--ssim2-bin" => ssim2_bin = args.next().unwrap_or(ssim2_bin),
            "--metric" => metric_s = args.next().unwrap_or(metric_s),
            "--label" => label = args.next().unwrap_or(label),
            "--out-dir" => out_dir = PathBuf::from(args.next().unwrap_or_default()),
            "--iters" => iters = args.next().and_then(|s| s.parse().ok()).unwrap_or(iters),
            "--corpus-file" => corpus_file = args.next(),
            "--distances" => {
                if let Some(s) = args.next() {
                    distances = s.split(',').filter_map(|x| x.trim().parse().ok()).collect();
                }
            }
            "--zensim-targets" => {
                if let Some(s) = args.next() {
                    zensim_targets = s.split(',').filter_map(|x| x.trim().parse().ok()).collect();
                }
            }
            "--arms" => {
                if let Some(s) = args.next() {
                    arms = s.split(',').map(|x| x.trim().to_string()).collect();
                }
            }
            "--bake" => bake = args.next(),
            _ => {}
        }
    }
    let corpus = match &corpus_file {
        Some(f) => corpus_from_file(f),
        None => default_corpus(),
    };
    let metric = parse_metric(&metric_s);

    if let Some(btf) = &bytes_targets_file {
        return run_bytes_target(
            btf,
            bake.as_deref()
                .expect("--bake required for --bytes-targets-file"),
            iters,
            &label,
            &out_dir,
        );
    }

    if let Some(metric_arm) = &score_targets_outer {
        return run_score_target_outer(
            &corpus,
            &zensim_targets,
            metric_arm,
            bake.as_deref()
                .expect("--bake required for --score-targets-outer"),
            iters,
            &label,
            &out_dir,
            &ssim2_bin,
        );
    }

    if !zensim_targets.is_empty() {
        return run_target_ab(
            &corpus,
            &zensim_targets,
            &arms,
            bake.as_deref()
                .expect("--bake required for --zensim-targets"),
            iters,
            &label,
            &out_dir,
        );
    }
    let decoded_dir = out_dir.join("decoded");
    let ref_dir = out_dir.join("ref");
    fs::create_dir_all(&decoded_dir)?;
    fs::create_dir_all(&ref_dir)?;

    let manifest_path = out_dir.join(format!("manifest_{label}.tsv"));
    let mut manifest =
        String::from("ref_path\tdist_path\tlabel\timage\tcorpus\tdistance\tbytes\tencode_ms\n");
    eprintln!(
        "[diffmap_rd] metric={metric_s} label={label} | ZENSIM_MASKING={} ZENSIM_SQRT={} ZENSIM_HF={} ZENSIM_EDGE_MSE={}",
        std::env::var("ZENSIM_MASKING").unwrap_or_else(|_| "default(8)".into()),
        std::env::var("ZENSIM_SQRT").unwrap_or_else(|_| "default(0)".into()),
        std::env::var("ZENSIM_HF").unwrap_or_else(|_| "default(1)".into()),
        std::env::var("ZENSIM_EDGE_MSE").unwrap_or_else(|_| "default(1)".into()),
    );

    eprintln!(
        "[diffmap_rd] corpus={} images, distances={:?}, iters={iters}",
        corpus.len(),
        distances
    );
    for (name, corpus_name, path) in &corpus {
        let (name, corpus_name) = (name.as_str(), corpus_name.as_str());
        let (pixels, w, h) = load_rgb8(path)?;
        // Save the clean reference once (as PNG) for the external scorer.
        let ref_png = ref_dir.join(format!("{name}.png"));
        if !ref_png.exists() {
            image::RgbImage::from_raw(w, h, pixels.clone())
                .ok_or("ref from_raw")?
                .save(&ref_png)?;
        }
        for &d in &distances {
            let t = Instant::now();
            let mut cfg = LossyConfig::new(d)
                .with_strategy(EncoderStrategy::Zenjxl)
                .with_effort(EFFORT)
                .with_perceptual_metric(metric);
            // Engage the diffmap-driven per-tile redistribution loop for the
            // active metric (iter 0 alone is compare-only → diffmap unused).
            // The quality loops are mutually exclusive — zero butteraugli's
            // default iters when driving with zensim/cvvdp.
            // zensim has a SEPARATE zensim_iters (and butteraugli_iters must be
            // 0 — mutually exclusive). cvvdp + butteraugli both drive the shared
            // buttloop via butteraugli_iters (perceptual_loop resolves
            // iters_override.unwrap_or(self.butteraugli_iters)); cvvdp is
            // selected by the perceptual-metric, not a separate iters field.
            cfg = match metric_s.as_str() {
                "zensim" => cfg.with_butteraugli_iters(0).with_zensim_iters(iters),
                _ => cfg.with_butteraugli_iters(iters),
            };
            let encoded = cfg.encode(&pixels, w, h, PixelLayout::Rgb8)?;
            let encode_ms = t.elapsed().as_secs_f64() * 1000.0;
            let bytes = encoded.len();
            let decoded = decode_jxl_srgb_u8(&encoded, w, h)?;
            let dist_png = decoded_dir.join(format!("{label}__{name}__d{d:.2}.png"));
            image::RgbImage::from_raw(w, h, decoded)
                .ok_or("dec from_raw")?
                .save(&dist_png)?;
            manifest.push_str(&format!(
                "{}\t{}\t{label}\t{name}\t{corpus_name}\t{d:.2}\t{bytes}\t{encode_ms:.1}\n",
                ref_png.display(),
                dist_png.display(),
            ));
            eprintln!("  {label} {name} d={d:.2} bytes={bytes} encode={encode_ms:.0}ms");
        }
    }
    fs::write(&manifest_path, &manifest)?;
    eprintln!("[diffmap_rd] wrote manifest {}", manifest_path.display());
    Ok(())
}

/// C3b target-hitting A/B (zensim task #67): per (fixture × target × arm),
/// encode with the shared target controller, decode, judge with the SAME
/// bake, and emit one TSV row. Arms differ ONLY in `JXL_ZENSIM_MODEL_MAP`.
#[allow(clippy::too_many_arguments)]
fn run_target_ab(
    corpus: &[(String, String, String)],
    targets: &[f64],
    arms: &[String],
    bake: &str,
    iters: u32,
    label: &str,
    out_dir: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use jxl_encoder::api::{EncoderStrategy, PerceptualMetric};
    use jxl_encoder::{LossyConfig, PixelLayout};

    let decoded_dir = out_dir.join("decoded");
    let ref_dir = out_dir.join("ref");
    fs::create_dir_all(&decoded_dir)?;
    fs::create_dir_all(&ref_dir)?;
    let stats_tmp = out_dir.join(format!(".stats_tmp_{label}.tsv"));

    // One bake per process (the loop's RD_PROFILE OnceLock caches the first).
    set_rd_profile_env(bake);
    // SAFETY: edition-2024 set_var — single-threaded driver; all encodes run
    // sequentially after these writes.
    unsafe {
        std::env::set_var("JXL_ZENSIM_RD_STATS", &stats_tmp);
    }

    let manifest_path = out_dir.join(format!("target_ab_{label}.tsv"));
    let mut manifest = String::from(
        "image\tclass\ttarget\tarm\tbake\tseed_d\tachieved_inloop\titers_used\tachieved_decoded\tabs_err\tbytes\tencode_ms\tloop_ms\tms_per_compare\n",
    );
    for (name, class, path) in corpus {
        let (pixels, w, h) = load_rgb8(path)?;
        let ref_png = ref_dir.join(format!("{name}.png"));
        if !ref_png.exists() {
            image::RgbImage::from_raw(w, h, pixels.clone())
                .ok_or("ref from_raw")?
                .save(&ref_png)?;
        }
        for &t in targets {
            for arm in arms {
                // SAFETY: sequential driver, no concurrent env readers here.
                unsafe {
                    match arm.as_str() {
                        "baseline" => std::env::remove_var("JXL_ZENSIM_MODEL_MAP"),
                        other => std::env::set_var("JXL_ZENSIM_MODEL_MAP", other),
                    }
                    std::env::set_var("JXL_ZENSIM_TARGET_SCORE", format!("{t}"));
                    // Efficiency study: per-encode trace id (the loop only
                    // reads it when JXL_ZENSIM_TRACE is set by the runner).
                    std::env::set_var(
                        "JXL_ZENSIM_TRACE_ID",
                        format!("{label}|{name}|{class}|{t:.0}|{arm}"),
                    );
                }
                let _ = fs::remove_file(&stats_tmp);
                let seed_d = seed_distance_for_target(t);
                let t0 = Instant::now();
                let cfg = LossyConfig::new(seed_d)
                    .with_strategy(EncoderStrategy::Zenjxl)
                    .with_effort(EFFORT)
                    .with_perceptual_metric(PerceptualMetric::Zensim)
                    .with_butteraugli_iters(0)
                    .with_zensim_iters(iters);
                let encoded = cfg.encode(&pixels, w, h, PixelLayout::Rgb8)?;
                let encode_ms = t0.elapsed().as_secs_f64() * 1000.0;
                let bytes = encoded.len();
                // Loop stats: last line of the per-encode stats file.
                let (iters_used, inloop, loop_ms, per_iter) = fs::read_to_string(&stats_tmp)
                    .ok()
                    .and_then(|s| {
                        let l = s.lines().last()?.to_string();
                        let mut f = l.split('\t');
                        Some((
                            f.next()?.parse::<u32>().ok()?,
                            f.next()?.parse::<f64>().ok()?,
                            f.next()?.parse::<f64>().ok()?,
                            f.next().unwrap_or("").to_string(),
                        ))
                    })
                    .unwrap_or((0, f64::NAN, f64::NAN, String::new()));
                let ms_per_compare = if iters_used > 0 {
                    loop_ms / iters_used as f64
                } else {
                    f64::NAN
                };
                // Efficiency study R0 identity gate: save the encoded
                // bitstream itself when JXL_SAVE_BITSTREAM=1 (byte-identity
                // across instrument-env conditions is checked on the .jxl).
                if std::env::var("JXL_SAVE_BITSTREAM").as_deref() == Ok("1") {
                    let bs = decoded_dir.join(format!("{label}__{name}__t{t:.0}__{arm}.jxl"));
                    fs::write(&bs, &encoded)?;
                }
                let decoded = decode_jxl_srgb_u8(&encoded, w, h)?;
                let dist_png = decoded_dir.join(format!("{label}__{name}__t{t:.0}__{arm}.png"));
                image::RgbImage::from_raw(w, h, decoded.clone())
                    .ok_or("dec from_raw")?
                    .save(&dist_png)?;
                let achieved = judge_score(bake, &pixels, &decoded, w, h);
                let err = (achieved - t).abs();
                manifest.push_str(&format!(
                    "{name}\t{class}\t{t:.0}\t{arm}\t{bake}\t{seed_d:.2}\t{inloop:.3}\t{iters_used}\t{achieved:.3}\t{err:.3}\t{bytes}\t{encode_ms:.1}\t{loop_ms:.1}\t{ms_per_compare:.1}\n",
                ));
                eprintln!(
                    "  [{label}] {name} t={t:.0} {arm}: inloop={inloop:.2} decoded={achieved:.2} err={err:.2} iters={iters_used} bytes={bytes} {encode_ms:.0}ms ({per_iter} ms/iter)"
                );
            }
        }
    }
    let _ = fs::remove_file(&stats_tmp);
    fs::write(&manifest_path, &manifest)?;
    eprintln!("[target_ab] wrote {}", manifest_path.display());
    Ok(())
}

/// Efficiency study E7 (2026-07-31): SIZE (bytes) targeting via an OUTER
/// loop of full encodes. The zensim loop never entropy-codes, so per-iterate
/// bytes do not exist inside it (registered limitation); an E7 "iteration"
/// is one full encode with REAL bytes. The existing damped controller
/// formula (ratio^0.6, per-step clamp [1/1.35, 1.35]) drives the env-gated
/// `JXL_ZENSIM_QF_GLOBAL_SCALE` actuator — the same actuator (global
/// multiplicative qf step) the in-loop score controller uses. Baseline arm
/// only; `JXL_ZENSIM_TARGET_SCORE` unset (inner loop = redistribution only).
///
/// Input TSV rows: `path \t name \t class \t quality_target \t target_bytes`
/// (quality_target only sets the seed distance + labels the cell; the
/// controller sees bytes alone). Fixed outer budget of 8 encodes per cell —
/// no early acceptance, so the full convergence curve is recorded.
fn run_bytes_target(
    targets_file: &str,
    bake: &str,
    iters: u32,
    label: &str,
    out_dir: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    const OUTER_BUDGET: u32 = 8;
    fs::create_dir_all(out_dir)?;
    // SAFETY: edition-2024 set_var — single-threaded driver; all encodes run
    // sequentially after these writes.
    unsafe {
        std::env::set_var("JXL_ZENSIM_RD_PROFILE", format!("bake:{bake}"));
        std::env::remove_var("JXL_ZENSIM_TARGET_SCORE");
        std::env::remove_var("JXL_ZENSIM_MODEL_MAP");
    }
    let manifest_path = out_dir.join(format!("bytes_target_{label}.tsv"));
    let mut manifest = String::from(
        "image\tclass\tqtarget\ttarget_bytes\touter_iter\tqf_scale\tbytes\trel_err\tjudged\tencode_ms\n",
    );
    for line in std::fs::read_to_string(targets_file)?.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() < 5 {
            return Err(format!("bytes-targets row needs 5 cols: {line}").into());
        }
        let (path, name, class) = (f[0], f[1], f[2]);
        let qtarget: f64 = f[3].parse()?;
        let target_bytes: f64 = f[4].parse::<u64>()? as f64;
        let (pixels, w, h) = load_rgb8(path)?;
        let seed_d = seed_distance_for_target(qtarget);
        let mut g: f64 = 1.0;
        for j in 0..OUTER_BUDGET {
            // SAFETY: sequential driver (as above).
            unsafe {
                std::env::set_var("JXL_ZENSIM_QF_GLOBAL_SCALE", format!("{g}"));
                std::env::set_var(
                    "JXL_ZENSIM_TRACE_ID",
                    format!("{label}|{name}|{class}|{qtarget:.0}|bytes{j}"),
                );
            }
            let t0 = Instant::now();
            let cfg = LossyConfig::new(seed_d)
                .with_strategy(EncoderStrategy::Zenjxl)
                .with_effort(EFFORT)
                .with_perceptual_metric(PerceptualMetric::Zensim)
                .with_butteraugli_iters(0)
                .with_zensim_iters(iters);
            let encoded = cfg.encode(&pixels, w, h, PixelLayout::Rgb8)?;
            let encode_ms = t0.elapsed().as_secs_f64() * 1000.0;
            let bytes = encoded.len() as f64;
            let rel_err = (bytes - target_bytes) / target_bytes;
            let decoded = decode_jxl_srgb_u8(&encoded, w, h)?;
            let judged = judge_score(bake, &pixels, &decoded, w, h);
            manifest.push_str(&format!(
                "{name}\t{class}\t{qtarget:.0}\t{target_bytes:.0}\t{j}\t{g:.6}\t{bytes:.0}\t{rel_err:.5}\t{judged:.3}\t{encode_ms:.1}\n",
            ));
            eprintln!(
                "  [bytes {label}] {name} q{qtarget:.0} j={j} g={g:.4} bytes={bytes:.0} rel_err={:+.2}% judged={judged:.2} {encode_ms:.0}ms",
                rel_err * 100.0
            );
            // The registered damped update (qf up ⇒ more bytes).
            g *= ((target_bytes / bytes).powf(0.6)).clamp(1.0 / 1.35, 1.35);
        }
        // SAFETY: sequential driver (as above).
        unsafe {
            std::env::remove_var("JXL_ZENSIM_QF_GLOBAL_SCALE");
        }
    }
    fs::write(&manifest_path, &manifest)?;
    eprintln!("[bytes_target] wrote {}", manifest_path.display());
    Ok(())
}

/// Metric-matrix study (2026-07-31): SCORE targeting via an OUTER loop of
/// full encodes — the mechanism arm of the model × mechanism matrix. The
/// in-loop damped controller formula (loss-ratio^0.6, per-step clamp
/// [1/1.35, 1.35], loss floors 0.05) is transplanted AS-IS onto the
/// env-gated `JXL_ZENSIM_QF_GLOBAL_SCALE` actuator; the controller reads
/// the DECODED-judged score of each full encode in the arm's metric
/// (`zensim` = the mounted judge bake, in-process; `ssim2` = zenmetrics
/// shelled on the decoded PNG). The inner loop runs redistribution-only
/// (`JXL_ZENSIM_TARGET_SCORE` unset) at the caller's `--iters` (the study
/// registers 1 — the minimum engagement that applies the actuator, since
/// the scale lives inside the zensim loop gated on `zensim_iters > 0`).
/// Encodes j = 0..=3: budget 3 = three controller steps after the seed
/// encode, the step-count parallel of inner k=3. Both arms record BOTH
/// metrics per iterate (feeds the F5 cross-metric endpoint).
#[allow(clippy::too_many_arguments)]
fn run_score_target_outer(
    corpus: &[(String, String, String)],
    targets: &[f64],
    metric_arm: &str,
    bake: &str,
    iters: u32,
    label: &str,
    out_dir: &std::path::Path,
    ssim2_bin: &str,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use jxl_encoder::api::{EncoderStrategy, PerceptualMetric};
    use jxl_encoder::{LossyConfig, PixelLayout};
    const OUTER_ENCODES: u32 = 4; // j = 0..=3 — seed + 3 controller steps

    assert!(
        matches!(metric_arm, "zensim" | "ssim2"),
        "--score-targets-outer must be zensim|ssim2"
    );
    let decoded_dir = out_dir.join("decoded");
    let ref_dir = out_dir.join("ref");
    fs::create_dir_all(&decoded_dir)?;
    fs::create_dir_all(&ref_dir)?;
    set_rd_profile_env(bake);
    // SAFETY: edition-2024 set_var — single-threaded driver; all encodes run
    // sequentially after these writes.
    unsafe {
        std::env::remove_var("JXL_ZENSIM_TARGET_SCORE");
        std::env::remove_var("JXL_ZENSIM_MODEL_MAP");
    }
    let manifest_path = out_dir.join(format!("score_outer_{label}.tsv"));
    let mut manifest = String::from(
        "image\tclass\ttarget\tmetric\touter_iter\tqf_scale\tbytes\tjudged\tzensimA\tssim2\tencode_ms\tjudge_ms\tssim2_ms\n",
    );
    for (name, class, path) in corpus {
        let (pixels, w, h) = load_rgb8(path)?;
        let ref_png = ref_dir.join(format!("{name}.png"));
        if !ref_png.exists() {
            image::RgbImage::from_raw(w, h, pixels.clone())
                .ok_or("ref from_raw")?
                .save(&ref_png)?;
        }
        for &t in targets {
            let seed_d = seed_distance_for_target(t);
            let mut g: f64 = 1.0;
            for j in 0..OUTER_ENCODES {
                // SAFETY: sequential driver (as above).
                unsafe {
                    std::env::set_var("JXL_ZENSIM_QF_GLOBAL_SCALE", format!("{g}"));
                    std::env::set_var(
                        "JXL_ZENSIM_TRACE_ID",
                        format!("{label}|{name}|{class}|{t:.0}|outer{j}"),
                    );
                }
                let t0 = Instant::now();
                let cfg = LossyConfig::new(seed_d)
                    .with_strategy(EncoderStrategy::Zenjxl)
                    .with_effort(EFFORT)
                    .with_perceptual_metric(PerceptualMetric::Zensim)
                    .with_butteraugli_iters(0)
                    .with_zensim_iters(iters);
                let encoded = cfg.encode(&pixels, w, h, PixelLayout::Rgb8)?;
                let encode_ms = t0.elapsed().as_secs_f64() * 1000.0;
                let bytes = encoded.len();
                let decoded = decode_jxl_srgb_u8(&encoded, w, h)?;
                let dist_png = decoded_dir.join(format!("{label}__{name}__t{t:.0}__outer{j}.png"));
                image::RgbImage::from_raw(w, h, decoded.clone())
                    .ok_or("dec from_raw")?
                    .save(&dist_png)?;
                let tj = Instant::now();
                let zensim_a = judge_score(bake, &pixels, &decoded, w, h);
                let judge_ms = tj.elapsed().as_secs_f64() * 1000.0;
                let ts = Instant::now();
                let ssim2 = ssim2_score_via_zenmetrics(ssim2_bin, &ref_png, &dist_png, out_dir);
                let ssim2_ms = ts.elapsed().as_secs_f64() * 1000.0;
                let judged = if metric_arm == "zensim" {
                    zensim_a
                } else {
                    ssim2
                };
                manifest.push_str(&format!(
                    "{name}\t{class}\t{t:.0}\t{metric_arm}\t{j}\t{g:.6}\t{bytes}\t{judged:.3}\t{zensim_a:.3}\t{ssim2:.3}\t{encode_ms:.1}\t{judge_ms:.1}\t{ssim2_ms:.1}\n",
                ));
                eprintln!(
                    "  [outer {label}] {name} t={t:.0} j={j} g={g:.4} bytes={bytes} judged={judged:.2} (zA={zensim_a:.2} s2={ssim2:.2}) {encode_ms:.0}ms"
                );
                // The transplanted damped update (loss above target ⇒ too
                // lossy ⇒ scale the field up ⇒ more bits). A non-finite
                // judged score (failed scorer call) leaves g unchanged — the
                // row's NaN is the recorded null.
                if judged.is_finite() {
                    let achieved_loss = (100.0 - judged).max(0.05);
                    let target_loss = (100.0 - t).max(0.05);
                    g *= ((achieved_loss / target_loss).powf(0.6)).clamp(1.0 / 1.35, 1.35);
                }
            }
            // SAFETY: sequential driver (as above).
            unsafe {
                std::env::remove_var("JXL_ZENSIM_QF_GLOBAL_SCALE");
            }
        }
    }
    fs::write(&manifest_path, &manifest)?;
    eprintln!("[score_outer] wrote {}", manifest_path.display());
    Ok(())
}
