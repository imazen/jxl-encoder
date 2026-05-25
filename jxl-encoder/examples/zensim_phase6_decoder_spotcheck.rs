//! zensim-fork Phase 6 multi-decoder roundtrip spot-check.
//!
//! Encodes a small set of cells with each zensim-driven backend
//! (Z_GPU, Z_CPU) and decodes each output through jxl-oxide, external
//! djxl, and external jxl-rs CLI. Reports PASS/FAIL per (cell,
//! backend, decoder). Any FAIL is a structural bitstream bug per the
//! Phase 6 acceptance gate (e): STOP + revert.
//!
//! Fixtures: 10 cells = 5 corpus images × 1 distance {3.0} × 2
//! backends at effort 8 (the buttloop fires at e>=8, so the zensim
//! backend actually drives the quant loop). Distance 3.0 was chosen
//! over 1.0 because the Phase 4 smoke observation showed zensim's
//! large encoded-bytes deviations from B mostly land at d>=2.0 where
//! the per-distance calibration table targets lower zensim scores.
//!
//! Decoder paths (hard-coded; adjust if env differs):
//! - djxl: `~/work/jxl-efforts/libjxl/build/tools/djxl`
//! - jxl-rs: `~/work/third-party/jxl-rs/target/release/jxl_cli`
//! - jxl-oxide: in-process (existing dep)
//!
//! Run:
//!
//!     CUDA_PATH=/usr/local/cuda cargo run --release -p jxl-encoder \
//!       --features '__expert butteraugli-loop zensim-loop zensim-loop-gpu ssim2-loop parallel' \
//!       --example zensim_phase6_decoder_spotcheck

use jxl_encoder::api::{LossyConfig, PixelLayout};
use std::io::{Cursor, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

const DJXL_PATH: &str = "/home/lilith/work/jxl-efforts/libjxl/build/tools/djxl";
const JXL_RS_PATH: &str = "/home/lilith/work/third-party/jxl-rs/target/release/jxl_cli";

const CID22_DIR: &str = "/home/lilith/work/codec-corpus/CID22/CID22-512/validation";
const GB82_SC_DIR: &str = "/home/lilith/work/codec-corpus/gb82-sc";

// 5 fixtures × 1 distance × 2 backends = 10 encodes × 3 decoders = 30 checks.
const FIXTURES: &[(&str, &str)] = &[
    ("CID22", "1025469.png"),
    ("CID22", "1418519.png"),
    ("CID22", "1189261.png"),
    ("CID22", "297394.png"),
    ("GB82-SC", "terminal.png"),
];
const DISTANCES: &[f32] = &[3.0];

#[derive(Debug, Clone, Copy)]
enum Backend {
    ZGpu,
    ZCpu,
}

impl Backend {
    fn tag(self) -> &'static str {
        match self {
            Backend::ZGpu => "Z_GPU",
            Backend::ZCpu => "Z_CPU",
        }
    }
    fn apply(self, cfg: LossyConfig) -> LossyConfig {
        match self {
            Backend::ZGpu => cfg
                .with_perceptual_metric(jxl_encoder::api::PerceptualMetric::Zensim)
                .with_perceptual_device(jxl_encoder::api::PerceptualDevice::Gpu),
            Backend::ZCpu => cfg
                .with_perceptual_metric(jxl_encoder::api::PerceptualMetric::Zensim)
                .with_perceptual_device(jxl_encoder::api::PerceptualDevice::Cpu),
        }
    }
}

fn load_source(corpus: &str, name: &str) -> (Vec<u8>, u32, u32) {
    let p = match corpus {
        "CID22" => Path::new(CID22_DIR).join(name),
        "GB82-SC" => Path::new(GB82_SC_DIR).join(name),
        _ => panic!("unknown corpus {corpus}"),
    };
    let img = image::open(&p).unwrap_or_else(|e| panic!("open {}: {e}", p.display()));
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    (rgb.as_raw().clone(), w, h)
}

fn decode_jxl_oxide(bytes: &[u8]) -> Result<(usize, usize), String> {
    let mut img = jxl_oxide::JxlImage::builder()
        .read(Cursor::new(bytes))
        .map_err(|e| format!("read: {e:?}"))?;
    img.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
        jxl_oxide::RenderingIntent::Relative,
    ));
    let render = img.render_frame(0).map_err(|e| format!("render: {e:?}"))?;
    let fb = render.image_all_channels();
    Ok((fb.width(), fb.height()))
}

fn decode_via_subprocess(cmd: &str, jxl_bytes: &[u8], tag: &str) -> Result<u64, String> {
    if !Path::new(cmd).is_file() {
        return Err(format!("{tag} binary missing at {cmd}"));
    }
    let unique = format!("/tmp/zensim_phase6_spotcheck_{tag}_{:x}.jxl", rand_id());
    let out_png = format!("/tmp/zensim_phase6_spotcheck_{tag}_{:x}.png", rand_id());
    std::fs::write(&unique, jxl_bytes).map_err(|e| format!("write tmp: {e}"))?;
    let res = Command::new(cmd).arg(&unique).arg(&out_png).output();
    let _ = std::fs::remove_file(&unique);
    let output = res.map_err(|e| format!("spawn: {e}"))?;
    if !output.status.success() {
        let _ = std::fs::remove_file(&out_png);
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        return Err(format!(
            "nonzero ({}): {}",
            output.status,
            stderr.lines().last().unwrap_or("")
        ));
    }
    let meta = std::fs::metadata(&out_png).map_err(|e| format!("metadata: {e}"))?;
    let sz = meta.len();
    let _ = std::fs::remove_file(&out_png);
    Ok(sz)
}

fn rand_id() -> u64 {
    use std::time::SystemTime;
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64
}

fn main() {
    let out_tsv = PathBuf::from("benchmarks/zensim_phase6_decoder_spotcheck_2026-05-25.tsv");
    let mut f = std::fs::File::create(&out_tsv).expect("create tsv");
    writeln!(
        f,
        "corpus\timage\tdistance\teffort\tbackend\tencoded_bytes\tdecode_oxide\tdecode_djxl\tdecode_jxl_rs\tdecode_notes"
    )
    .unwrap();

    let mut total = 0usize;
    let mut total_fail = 0usize;
    for (corpus, name) in FIXTURES {
        let (src_u8, w, h) = load_source(corpus, name);
        eprintln!("loaded {} ({} bytes, {w}x{h})", name, src_u8.len());
        for &distance in DISTANCES {
            for backend in [Backend::ZGpu, Backend::ZCpu] {
                total += 1;
                let cfg = backend.apply(LossyConfig::new(distance).with_effort(8));
                let t0 = Instant::now();
                let encode_res = cfg.encode(&src_u8, w, h, PixelLayout::Rgb8);
                let encode_ms = t0.elapsed().as_secs_f64() * 1000.0;
                let jxl_bytes = match encode_res {
                    Ok(b) => b,
                    Err(e) => {
                        writeln!(
                            f,
                            "{corpus}\t{name}\t{distance:.1}\t8\t{}\t0\tENCODE_FAIL\tENCODE_FAIL\tENCODE_FAIL\t{e:?}",
                            backend.tag()
                        )
                        .unwrap();
                        eprintln!(
                            "ENCODE FAIL {} {} d={distance} {} {e:?}",
                            corpus,
                            name,
                            backend.tag()
                        );
                        total_fail += 1;
                        continue;
                    }
                };
                let oxide = decode_jxl_oxide(&jxl_bytes);
                let djxl = decode_via_subprocess(DJXL_PATH, &jxl_bytes, "djxl");
                let jxlrs = decode_via_subprocess(JXL_RS_PATH, &jxl_bytes, "jxlrs");
                let oxide_tag = match &oxide {
                    Ok((dw, dh)) if *dw as u32 == w && *dh as u32 == h => "PASS".to_string(),
                    Ok((dw, dh)) => format!("BAD_DIMS({}x{})", dw, dh),
                    Err(e) => format!("FAIL:{e}"),
                };
                let djxl_tag = match &djxl {
                    Ok(_) => "PASS".to_string(),
                    Err(e) => format!("FAIL:{e}"),
                };
                let jxlrs_tag = match &jxlrs {
                    Ok(_) => "PASS".to_string(),
                    Err(e) => format!("FAIL:{e}"),
                };
                let fail = !oxide_tag.starts_with("PASS")
                    || !djxl_tag.starts_with("PASS")
                    || !jxlrs_tag.starts_with("PASS");
                if fail {
                    total_fail += 1;
                }
                writeln!(
                    f,
                    "{corpus}\t{name}\t{distance:.1}\t8\t{}\t{}\t{}\t{}\t{}\tencode_ms={:.1}",
                    backend.tag(),
                    jxl_bytes.len(),
                    oxide_tag,
                    djxl_tag,
                    jxlrs_tag,
                    encode_ms
                )
                .unwrap();
                eprintln!(
                    "[{} {} d={} {}] bytes={} oxide={} djxl={} jxlrs={} ({:.1}ms)",
                    corpus,
                    name,
                    distance,
                    backend.tag(),
                    jxl_bytes.len(),
                    oxide_tag,
                    djxl_tag,
                    jxlrs_tag,
                    encode_ms
                );
            }
        }
    }

    eprintln!(
        "\n[zensim_phase6_decoder_spotcheck] DONE: {total} cells, {} pass, {} fail",
        total - total_fail,
        total_fail
    );
    if total_fail > 0 {
        eprintln!(
            "HARD ACCEPTANCE GATE (e) FAILED — see TSV at {}",
            out_tsv.display()
        );
        std::process::exit(1);
    }
    eprintln!("OK — TSV: {}", out_tsv.display());
}
