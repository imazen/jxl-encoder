// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! cvvdp-fork Phase 8d (2026-05-25) multi-decoder roundtrip spot-check.
//!
//! Encodes 6 cells with the cvvdp loop AND the Phase 8d bytes-tighten
//! exit pass active, then decodes each output through:
//!   - jxl-oxide (in-process; existing dep)
//!   - djxl (libjxl CLI subprocess)
//!   - jxl-rs (third-party CLI subprocess)
//!
//! Any FAIL signals a structural bitstream regression introduced by the
//! Phase 8d tighten path. The hard acceptance gate (f) per the Phase 8d
//! brief: ANY decoder failure → revert + STOP.
//!
//! Output: `benchmarks/cvvdp_phase8d_decoder_roundtrip_2026-05-25.tsv`.

#![cfg(all(
    feature = "cvvdp-loop",
    feature = "cvvdp-loop-tighten",
    feature = "butteraugli-loop"
))]

use jxl_encoder::api::{LossyConfig, PixelLayout};
use std::io::{Cursor, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

const DJXL_PATH: &str = "/home/lilith/work/jxl-efforts/libjxl/build/tools/djxl";
const JXL_RS_PATH: &str = "/home/lilith/work/third-party/jxl-rs/target/release/jxl_cli";

const CID22_DIR: &str = "/home/lilith/work/codec-corpus/CID22/CID22-512/validation";
const GB82_SC_DIR: &str = "/home/lilith/work/codec-corpus/gb82-sc";

/// 5 fixtures × 1 distance = 5 cells at e8 (the buttloop + tighten
/// fires at e>=8). One extra GB82-SC at d=3.0 to cover the
/// screenshot-class + high-distance combo where the renorm scale
/// historically had the worst headroom.
const FIXTURES: &[(&str, &str, f32)] = &[
    ("CID22", "1025469.png", 1.0),
    ("CID22", "1418519.png", 1.0),
    ("CID22", "1189261.png", 1.0),
    ("GB82-SC", "terminal.png", 1.0),
    ("GB82-SC", "terminal.png", 3.0),
    ("GB82-SC", "imac_g3.png", 2.0),
];

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
    let unique = format!("/tmp/cvvdp_phase8d_rt_{tag}_{:x}.jxl", rand_id());
    let out_png = format!("/tmp/cvvdp_phase8d_rt_{tag}_{:x}.png", rand_id());
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
    let out_tsv = PathBuf::from("benchmarks/cvvdp_phase8d_decoder_roundtrip_2026-05-25.tsv");
    let mut f = std::fs::File::create(&out_tsv).expect("create tsv");
    writeln!(
        f,
        "corpus\timage\tdistance\teffort\tbackend\tencoded_bytes\tdecode_oxide\tdecode_djxl\tdecode_jxl_rs\tdecode_notes"
    )
    .unwrap();

    let limits =
        jxl_encoder::api::Limits::default().with_max_memory_bytes(8 * 1024 * 1024 * 1024);
    let mut total = 0usize;
    let mut total_fail = 0usize;
    for (corpus, name, distance) in FIXTURES {
        let (src_u8, w, h) = load_source(corpus, name);
        eprintln!("loaded {} ({} bytes, {w}x{h})", name, src_u8.len());
        total += 1;
        let cfg = LossyConfig::new(*distance)
            .with_effort(8)
            .with_cvvdp_loop(Some(true))
            .with_cvvdp_bytes_tighten(Some(true));
        let t0 = Instant::now();
        let encode_res = cfg
            .encode_request(w, h, PixelLayout::Rgb8)
            .with_limits(&limits)
            .encode(&src_u8);
        let encode_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let jxl_bytes = match encode_res {
            Ok(b) => b,
            Err(e) => {
                writeln!(
                    f,
                    "{corpus}\t{name}\t{distance:.1}\t8\tC_GPU_v3\t0\tENCODE_FAIL\tENCODE_FAIL\tENCODE_FAIL\t{e:?}"
                )
                .unwrap();
                eprintln!("ENCODE FAIL {corpus} {name} d={distance} {e:?}");
                total_fail += 1;
                continue;
            }
        };
        let oxide = decode_jxl_oxide(&jxl_bytes);
        let djxl = decode_via_subprocess(DJXL_PATH, &jxl_bytes, "djxl");
        let jxlrs = decode_via_subprocess(JXL_RS_PATH, &jxl_bytes, "jxlrs");
        let oxide_tag = match &oxide {
            Ok((dw, dh)) if *dw as u32 == w && *dh as u32 == h => "PASS".to_string(),
            Ok((dw, dh)) => format!("BAD_DIMS({dw}x{dh})"),
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
            "{corpus}\t{name}\t{distance:.1}\t8\tC_GPU_v3\t{}\t{}\t{}\t{}\tencode_ms={:.1}",
            jxl_bytes.len(),
            oxide_tag,
            djxl_tag,
            jxlrs_tag,
            encode_ms
        )
        .unwrap();
        eprintln!(
            "[{corpus} {name} d={distance}] bytes={} oxide={oxide_tag} djxl={djxl_tag} jxlrs={jxlrs_tag} ({encode_ms:.1}ms)",
            jxl_bytes.len(),
        );
    }

    eprintln!(
        "\n[cvvdp_phase8d_decoder_roundtrip] DONE: {total} cells, {} pass, {} fail",
        total - total_fail,
        total_fail
    );
    if total_fail > 0 {
        eprintln!(
            "HARD ACCEPTANCE GATE (f) FAILED — see TSV at {}",
            out_tsv.display()
        );
        std::process::exit(1);
    }
}
