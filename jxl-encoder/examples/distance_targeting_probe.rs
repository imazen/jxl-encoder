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
//! `JXL_BUTTLOOP_INITIAL_QF_SCALE=1.0` at e >= 8 (default scale 4.0) and
//! `JXL_W44_109_ADAPTIVE_QUANT_QF_SCALE=1.0` at e in [5,7] (defaults 2.0 at
//! e5/e6, 3.0 at e7). Those are DIFFERENT gates: both must be set to turn the
//! lift off across an effort ladder. `RESAMPLING` (1/2/4/8, default 1) drives
//! the `with_resampling(N)` path so our 2x regime can be scored against a
//! reference encoder's.
//!
//! Prints a TSV to stdout: `effort d_req bytes bfly ssim2 delivered_ratio`,
//! where `delivered_ratio = bfly / d_req` (1.0 = promise kept, <1 = finer
//! than requested, >1 = coarser).
//!
//! `ARTIFACT_DIR` is required. Each cell persists its JXL and full f32
//! butteraugli diffmap (`BFMAPF32`, little-endian u32 width/height, then
//! row-major little-endian f32 pixels), plus max and p1/p2/p3/p6 norms.
//! Per-run TSV and metadata record the source hash and build commit.
//! Every cell fully decodes through jxl-rs, djxl v0.12, and jxl-oxide.
//! `TARGETING_POLICY` selects `default`, `libjxl`, `legacy` (explicit old
//! seed lifts and iteration skip), or `unlifted` (both seed lifts and both
//! adaptive iteration flags disabled; other Zenjxl choices retained).
//! `ITERS` optionally overrides the existing quant-loop iteration setting.
//!
//! Reproducer (sets compile-time source provenance and the local djxl path):
//!   IMG=<png> ARTIFACT_DIR=<dir> just distance-targeting-probe

use std::io::{Cursor, Write};
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use sha2::{Digest, Sha256};

#[path = "distance_targeting_probe/decode.rs"]
mod decode;

use butteraugli::{ButteraugliParams, butteraugli_linear};
use imgref::Img;
use jxl_encoder::api::{
    AdaptiveQuantQfSeedPolicy, ButtloopQfSeedPolicy, EncoderImprovementsCustom, EncoderStrategy,
    Limits, LossyConfig, PixelLayout,
};
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
    let artifacts = PathBuf::from(std::env::var("ARTIFACT_DIR").expect("set ARTIFACT_DIR"));
    std::fs::create_dir_all(&artifacts).expect("create ARTIFACT_DIR");
    let build_commit = match option_env!("JXL_PROBE_BUILD_COMMIT") {
        Some(commit) => commit,
        None => {
            eprintln!("build via just distance-targeting-probe to record source provenance");
            std::process::exit(2);
        }
    };
    let targeting_policy = std::env::var("TARGETING_POLICY").unwrap_or_else(|_| "default".into());
    let strategy = match targeting_policy.as_str() {
        "default" => EncoderStrategy::Zenjxl,
        "libjxl" => EncoderStrategy::Libjxl,
        "legacy" | "unlifted" => {
            let legacy = targeting_policy == "legacy";
            let mut custom = EncoderImprovementsCustom::default();
            custom.buttloop_qf_seed = if legacy {
                ButtloopQfSeedPolicy::AutoScale4
            } else {
                ButtloopQfSeedPolicy::Off
            };
            custom.adaptive_quant_qf_seed = if legacy {
                AdaptiveQuantQfSeedPolicy::AutoScalePerEffort
            } else {
                AdaptiveQuantQfSeedPolicy::Off
            };
            custom.adaptive_buttloop_iters = legacy;
            custom.adaptive_buttloop_iters_narrow = legacy;
            EncoderStrategy::Custom(Box::new(custom))
        }
        other => {
            panic!("unknown TARGETING_POLICY {other:?}; use default, legacy, unlifted or libjxl")
        }
    };
    let iters: Option<u32> = std::env::var("ITERS")
        .ok()
        .map(|v| v.parse().expect("ITERS: u32"));
    #[cfg(not(feature = "butteraugli-loop"))]
    assert!(iters.is_none(), "ITERS requires butteraugli-loop");
    let forced_strategy: Option<u8> = std::env::var("FORCE_STRATEGY")
        .ok()
        .map(|v| v.parse().expect("FORCE_STRATEGY: u8"));
    let epf: Option<i8> = std::env::var("EPF")
        .ok()
        .map(|v| v.parse().expect("EPF: i8"));
    let gaborish: Option<bool> = std::env::var("GABORISH")
        .ok()
        .map(|v| v.parse().expect("GABORISH: true or false"));
    let patches: Option<bool> = std::env::var("PATCHES")
        .ok()
        .map(|v| v.parse().expect("PATCHES: true or false"));
    let djxl = jxl_encoder::test_helpers::djxl_path();
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

    // Forced resampling factor (1/2/4/8). `1` (default) leaves the regime
    // alone; `2` drives the same path `LossyConfig::with_resampling(2)` takes,
    // so this probe can score OUR 2x regime against a reference encoder's.
    let resampling: u32 = std::env::var("RESAMPLING")
        .ok()
        .map(|s| s.parse().expect("RESAMPLING: u32"))
        .unwrap_or(1);

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
    let source_hash = sha256(&rgb);
    let run_id = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let stem = artifacts.join(format!("{source_hash}-{run_id}"));
    std::fs::write(stem.with_extension("meta"), format!(
        "build_commit\t{build_commit}\nsource\t{path:?}\nsource_rgb8_sha256\t{source_hash}\nwidth\t{w}\nheight\t{h}\nresampling\t{resampling}\ntargeting_policy\t{targeting_policy}\niters\t{iters:?}\nforced_strategy\t{forced_strategy:?}\nepf\t{epf:?}\ngaborish\t{gaborish:?}\npatches\t{patches:?}\nbuttloop_scale\t{:?}\nadaptive_scale\t{:?}\nepf_seed_disable\t{:?}\nepf_per_iter\t{:?}\n",
        std::env::var("JXL_BUTTLOOP_INITIAL_QF_SCALE").ok(),
        std::env::var("JXL_W44_109_ADAPTIVE_QUANT_QF_SCALE").ok(),
        std::env::var("JXL_W44_117_DISABLE").ok(),
        std::env::var("JXL_W44_118_PER_ITER_SHARPNESS").ok(),
    )).expect("persist run metadata");
    let mut records = std::fs::File::create(stem.with_extension("tsv")).expect("create run TSV");
    eprintln!(
        "# {path} {w}x{h}  qf_scale_env={:?}",
        std::env::var("JXL_BUTTLOOP_INITIAL_QF_SCALE").ok()
    );
    let header = "effort\td_req\tbytes\tbfly\tssim2\tdelivered_ratio\tencode_ms\tpnorm1\tpnorm2\tpnorm3\tpnorm6\tencoded_sha256\tdiffmap_sha256";
    println!("{header}");
    writeln!(records, "{header}").unwrap();
    for &e in &efforts {
        for &d in &distances {
            let mut config = LossyConfig::new(d)
                .with_strategy(strategy.clone())
                .with_effort(e)
                .with_resampling(resampling);
            if let Some(patches) = patches {
                config = config.with_patches(patches);
            }
            if let Some(strategy) = forced_strategy {
                config = config.with_force_strategy(Some(strategy));
            }
            if let Some(epf) = epf {
                config = config.with_epf_level(epf);
            }
            if let Some(gaborish) = gaborish {
                config = config.with_gaborish(gaborish);
            }
            #[cfg(feature = "butteraugli-loop")]
            if let Some(iters) = iters {
                config = config.with_butteraugli_iters(iters);
            }
            #[cfg(feature = "__internal_recon_hook")]
            {
                use jxl_encoder::vardct::__recon_hook;
                let _ = __recon_hook::take_last();
                __recon_hook::set_capture_enabled(true);
                let _ = __recon_hook::take_last_production_qf();
                __recon_hook::set_production_qf_capture_enabled(true);
            }
            let started = Instant::now();
            let bytes = config
                .encode_request(w, h, PixelLayout::Rgb8)
                .with_limits(&lim)
                .encode(&rgb)
                .unwrap_or_else(|err| panic!("encode d={d} e={e}: {err:?}"));
            let encode_ms = started.elapsed().as_secs_f64() * 1000.0;
            let encoded_hash = sha256(&bytes);
            #[cfg(feature = "__internal_recon_hook")]
            {
                use jxl_encoder::vardct::__recon_hook;
                __recon_hook::set_capture_enabled(false);
                __recon_hook::set_production_qf_capture_enabled(false);
                if let Some(recon) = __recon_hook::take_last() {
                    std::fs::write(
                        artifacts.join(format!("{encoded_hash}.internal-strategy")),
                        &recon.raw_strategy,
                    )
                    .expect("persist strategy map");
                    std::fs::write(
                        artifacts.join(format!("{encoded_hash}.internal-qf")),
                        &recon.quant_field_u8,
                    )
                    .expect("persist quant field");
                    let production = __recon_hook::take_last_production_qf()
                        .expect("production quant-field capture");
                    let changed = recon
                        .quant_field_u8
                        .iter()
                        .zip(&production.quant_field_u8)
                        .filter(|(a, b)| a != b)
                        .count();
                    eprintln!(
                        "internal vs production: changed_qf={changed} global_scale={} vs {}",
                        recon.final_global_scale, production.global_scale
                    );
                    let pixels: Vec<RGB<f32>> = recon
                        .r
                        .iter()
                        .zip(&recon.g)
                        .zip(&recon.b)
                        .map(|((&r, &g), &b)| RGB::new(r, g, b))
                        .collect();
                    let mut raw = Vec::with_capacity(pixels.len() * 12);
                    for p in &pixels {
                        for value in [p.r, p.g, p.b] {
                            raw.extend_from_slice(&value.to_le_bytes());
                        }
                    }
                    std::fs::write(
                        artifacts.join(format!("{encoded_hash}.internal-rgb-f32le")),
                        raw,
                    )
                    .expect("persist internal reconstruction");
                    let internal = Img::new(pixels, recon.width, recon.height);
                    let score = butteraugli_linear(
                        orig_lin.as_ref(),
                        internal.as_ref(),
                        &ButteraugliParams::default(),
                    )
                    .expect("score internal reconstruction");
                    eprintln!(
                        "internal e{e} d{d}: score={} iteration={} spatial_iters={} global_scale={}",
                        score.score, recon.iter, recon.iters, recon.final_global_scale
                    );
                }
            }
            let encoded_path = artifacts.join(format!("{encoded_hash}.jxl"));
            std::fs::write(&encoded_path, &bytes).expect("persist JXL before validation");
            let primary_pixels = decode::verify_jxl_rs(&bytes, w as usize, h as usize);
            let reference = std::process::Command::new(&djxl)
                .arg(&encoded_path)
                .args(["--disable_output", "--num_threads=1"])
                .output()
                .expect("run djxl");
            let mut reference_log = reference.stdout;
            reference_log.extend_from_slice(&reference.stderr);
            std::fs::write(
                artifacts.join(format!("{encoded_hash}.djxl.log")),
                &reference_log,
            )
            .expect("persist djxl output");
            assert!(
                reference.status.success(),
                "djxl rejected {encoded_hash}: {}",
                String::from_utf8_lossy(&reference_log)
            );
            let mut dec = jxl_oxide::JxlImage::builder()
                .read(Cursor::new(&bytes))
                .expect("decode");
            dec.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
                jxl_oxide::RenderingIntent::Relative,
            ));
            let fb = dec.render_frame(0).expect("render").image_all_channels();
            assert_eq!((fb.width(), fb.height()), (w as usize, h as usize));
            let buf = fb.buf();
            #[cfg(feature = "__internal_recon_hook")]
            {
                let linear: Vec<RGB<f32>> = primary_pixels
                    .as_chunks::<3>()
                    .0
                    .iter()
                    .map(|p| {
                        let convert = decode::srgb_to_linear;
                        RGB::new(convert(p[0]), convert(p[1]), convert(p[2]))
                    })
                    .collect();
                let primary = Img::new(linear, w as usize, h as usize);
                let metric = butteraugli_linear(
                    orig_lin.as_ref(),
                    primary.as_ref(),
                    &ButteraugliParams::default(),
                )
                .expect("score jxl-rs pixels");
                let internal_path = artifacts.join(format!("{encoded_hash}.internal-rgb-f32le"));
                if internal_path.is_file() {
                    let raw = std::fs::read(internal_path).unwrap();
                    assert_eq!(raw.len(), buf.len() * 4);
                    let mut max_diff = (0.0f32, 0usize, 0.0f32, 0.0f32);
                    let mut sum_diff = 0.0f64;
                    for (i, (&decoded, raw)) in
                        buf.iter().zip(raw.as_chunks::<4>().0.iter()).enumerate()
                    {
                        let internal = f32::from_le_bytes(*raw);
                        let diff = (internal - decoded).abs();
                        sum_diff += f64::from(diff);
                        if diff > max_diff.0 {
                            max_diff = (diff, i, internal, decoded);
                        }
                    }
                    eprintln!(
                        "decoder drift: jxl_rs_score={} max={:?} mean_abs={}",
                        metric.score,
                        max_diff,
                        sum_diff / buf.len() as f64
                    );
                }
            }
            drop(primary_pixels);
            let dist_lin: Img<Vec<RGB<f32>>> = Img::new(
                buf.chunks(3).map(|c| RGB::new(c[0], c[1], c[2])).collect(),
                fb.width(),
                fb.height(),
            );
            let metric = butteraugli_linear(
                orig_lin.as_ref(),
                dist_lin.as_ref(),
                &ButteraugliParams::default().with_compute_diffmap(true),
            )
            .expect("butteraugli");
            let bfly = metric.score;
            let diffmap = metric.diffmap.as_ref().expect("requested diffmap");
            let mut map_bytes = b"BFMAPF32".to_vec();
            map_bytes.extend_from_slice(&w.to_le_bytes());
            map_bytes.extend_from_slice(&h.to_le_bytes());
            for &value in diffmap.buf() {
                map_bytes.extend_from_slice(&value.to_le_bytes());
            }
            let diffmap_hash = sha256(&map_bytes);
            std::fs::write(artifacts.join(format!("{diffmap_hash}.bfmap")), map_bytes)
                .expect("persist f32 butteraugli diffmap");
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
            let row = format!(
                "{e}\t{d}\t{}\t{bfly:.4}\t{ssim2:.3}\t{:.3}\t{encode_ms:.3}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{encoded_hash}\t{diffmap_hash}",
                bytes.len(),
                bfly / d as f64,
                metric.pnorm(1.0).unwrap(),
                metric.pnorm(2.0).unwrap(),
                metric.pnorm_3,
                metric.pnorm(6.0).unwrap(),
            );
            println!("{row}");
            writeln!(records, "{row}").unwrap();
            records.flush().unwrap();
            eprintln!(
                "saved {w}x{h} e{e} d{d}: {encoded_hash}, ratio={:.3}",
                bfly / d as f64
            );
        }
    }
}

fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
