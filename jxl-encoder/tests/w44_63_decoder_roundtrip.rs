//! W44-63 decoder roundtrip — verify the new `with_dct_suppress_hint`
//! produces decoder-valid JXL via djxl and jxl_cli (jxl-rs) on the two
//! cells with largest measured wins (codec_wiki + imac_g3 at e7 d=5).
//!
//! Skipped if the corpus or external decoders aren't available; this is
//! a manual-validation harness, NOT a regression gate. Run with:
//!     cargo test --release -p jxl-encoder --test w44_63_decoder_roundtrip \
//!         --features '__expert butteraugli-loop ssim2-loop parallel' -- --nocapture --ignored

use std::path::{Path, PathBuf};
use std::process::Command;

use jxl_encoder::api::{LossyConfig, PixelLayout};

const DJXL: &str = "/home/lilith/work/jxl-efforts/libjxl/build/tools/djxl";
const JXLRS: &str = "/home/lilith/work/third-party/jxl-rs/target/release/jxl_cli";

fn corpus_path(name: &str) -> Option<PathBuf> {
    let p = PathBuf::from("/home/lilith/work/codec-corpus/gb82-sc").join(name);
    if p.exists() { Some(p) } else { None }
}

fn run_decoder(name: &str, prog: &str, jxl: &Path, png: &Path) -> bool {
    if !Path::new(prog).exists() {
        eprintln!("    [skip] {name} not found at {prog}");
        return true;
    }
    let out = Command::new(prog)
        .arg(jxl)
        .arg(png)
        .output()
        .unwrap_or_else(|e| panic!("{name} exec failed: {e}"));
    let status = out.status;
    if !status.success() {
        eprintln!("    [FAIL] {name} exit={:?}", status.code());
        eprintln!("    stderr: {}", String::from_utf8_lossy(&out.stderr));
        return false;
    }
    eprintln!("    [ok] {name} -> {}", png.display());
    true
}

#[test]
#[ignore]
fn w44_63_codec_wiki_e7_d5_decoder_roundtrip() {
    let Some(path) = corpus_path("codec_wiki.png") else {
        eprintln!("corpus missing — skipping");
        return;
    };
    decode_cell("codec_wiki", &path, 7, 5.0);
}

#[test]
#[ignore]
fn w44_63_imac_g3_e7_d5_decoder_roundtrip() {
    let Some(path) = corpus_path("imac_g3.png") else {
        eprintln!("corpus missing — skipping");
        return;
    };
    decode_cell("imac_g3", &path, 7, 5.0);
}

fn decode_cell(label: &str, src: &Path, effort: u8, distance: f32) {
    let pic = image::open(src).expect("open source");
    let rgb = pic.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    let out_dir = PathBuf::from("/tmp/w44_63_rt");
    std::fs::create_dir_all(&out_dir).expect("mkdir tmp");
    for (variant, cfg) in [
        (
            "def",
            LossyConfig::new(distance)
                .with_effort(effort)
                .with_threads(1),
        ),
        (
            "hint_none",
            LossyConfig::new(distance)
                .with_effort(effort)
                .with_threads(1)
                .with_content_aware_entropy_mul(true),
        ),
        (
            "hint_some",
            LossyConfig::new(distance)
                .with_effort(effort)
                .with_threads(1)
                .with_content_aware_entropy_mul(true)
                .with_strategy_overrides(jxl_encoder::api::StrategyOverrides { dct_suppress_hint: Some(true), ..Default::default() }),
        ),
    ] {
        let bytes = cfg
            .encode(rgb.as_raw(), w, h, PixelLayout::Rgb8)
            .expect("encode");
        let jxl = out_dir.join(format!("{label}_e{effort}_d{distance}_{variant}.jxl"));
        let png_djxl = out_dir.join(format!("{label}_e{effort}_d{distance}_{variant}_djxl.png"));
        let png_jxlrs = out_dir.join(format!("{label}_e{effort}_d{distance}_{variant}_jxlrs.png"));
        std::fs::write(&jxl, &bytes).expect("write jxl");
        eprintln!("{label} variant={variant} bytes={}", bytes.len());
        let ok_djxl = run_decoder("djxl", DJXL, &jxl, &png_djxl);
        let ok_jxlrs = run_decoder("jxl_cli", JXLRS, &jxl, &png_jxlrs);
        assert!(ok_djxl, "{label} {variant} djxl decoder failed");
        assert!(ok_jxlrs, "{label} {variant} jxl_cli decoder failed");
    }
}
