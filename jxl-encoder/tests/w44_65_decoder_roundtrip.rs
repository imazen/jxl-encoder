//! W44-65 decoder roundtrip — verify the W44-65 default-on
//! `dct_suppress_hint` auto-detection produces decoder-valid JXL via
//! djxl and jxl_cli (jxl-rs) on the three target codec_wiki cells +
//! windows95 pixel-art (must stay byte-identical) + a photo
//! regression probe.
//!
//! Mirrors `w44_63_decoder_roundtrip.rs` shape but tests the new
//! default path (`with_dct_suppress_hint(None)`) explicitly, plus
//! `Some(false)` to pin pre-W44-65 main bitstream for direct
//! byte-comparison.
//!
//! Skipped if the corpus or external decoders aren't available; this
//! is a manual-validation harness, NOT a regression gate. Run with:
//!     cargo test --release -p jxl-encoder --test w44_65_decoder_roundtrip \
//!         --features 'parallel' -- --nocapture --ignored

use std::path::{Path, PathBuf};
use std::process::Command;

use jxl_encoder::api::{LossyConfig, PixelLayout};

const DJXL: &str = "/home/lilith/work/jxl-efforts/libjxl/build/tools/djxl";
const JXLRS: &str = "/home/lilith/work/third-party/jxl-rs/target/release/jxl_cli";

fn gb82_path(name: &str) -> Option<PathBuf> {
    let p = PathBuf::from("/home/lilith/work/codec-corpus/gb82-sc").join(name);
    if p.exists() { Some(p) } else { None }
}

fn cid22_path(name: &str) -> Option<PathBuf> {
    let p = PathBuf::from("/home/lilith/work/codec-corpus/CID22/CID22-512/validation").join(name);
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
fn w44_65_codec_wiki_e7_d5_decoder_roundtrip() {
    let Some(path) = gb82_path("codec_wiki.png") else {
        eprintln!("corpus missing — skipping");
        return;
    };
    // expect_change=true: codec_wiki e7 d=5 should flip OPEN→FIXED
    decode_cell("codec_wiki", &path, 7, 5.0, true);
}

#[test]
#[ignore]
fn w44_65_codec_wiki_e7_d4_decoder_roundtrip() {
    let Some(path) = gb82_path("codec_wiki.png") else {
        eprintln!("corpus missing — skipping");
        return;
    };
    decode_cell("codec_wiki", &path, 7, 4.0, true);
}

#[test]
#[ignore]
fn w44_65_codec_wiki_e7_d6_decoder_roundtrip() {
    let Some(path) = gb82_path("codec_wiki.png") else {
        eprintln!("corpus missing — skipping");
        return;
    };
    decode_cell("codec_wiki", &path, 7, 6.0, true);
}

#[test]
#[ignore]
fn w44_65_windows95_pixel_art_invariant() {
    // windows95.png mask1x1_median = 99.06 — must NOT fire the
    // W44-65 default gate (threshold 99.5). The default-path
    // (`with_dct_suppress_hint(None)`) MUST produce bytes
    // byte-identical to `with_dct_suppress_hint(Some(false))`.
    let Some(path) = gb82_path("windows95.png") else {
        eprintln!("corpus missing — skipping");
        return;
    };
    assert_pixel_art_invariant("windows95", &path, 7, 2.0);
}

#[test]
#[ignore]
fn w44_65_photo_invariant_1418519() {
    // 1418519.png is the photo with the highest mask1x1_median (92.34) —
    // still below the 99.5 W44-65 threshold. Default-path MUST be
    // byte-identical to pre-W44-65.
    let Some(path) = cid22_path("1418519.png") else {
        eprintln!("corpus missing — skipping");
        return;
    };
    assert_pixel_art_invariant("1418519", &path, 7, 3.0);
}

fn decode_cell(label: &str, src: &Path, effort: u8, distance: f32, expect_change: bool) {
    let pic = image::open(src).expect("open source");
    let rgb = pic.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    let out_dir = PathBuf::from("/tmp/w44_65_rt");
    std::fs::create_dir_all(&out_dir).expect("mkdir tmp");
    let pinned_bytes = LossyConfig::new(distance)
        .with_effort(effort)
        .with_threads(1)
        .with_dct_suppress_hint(Some(false))
        .encode(rgb.as_raw(), w, h, PixelLayout::Rgb8)
        .expect("encode pinned");
    let default_bytes = LossyConfig::new(distance)
        .with_effort(effort)
        .with_threads(1)
        .encode(rgb.as_raw(), w, h, PixelLayout::Rgb8)
        .expect("encode default");
    let force_bytes = LossyConfig::new(distance)
        .with_effort(effort)
        .with_threads(1)
        .with_dct_suppress_hint(Some(true))
        .encode(rgb.as_raw(), w, h, PixelLayout::Rgb8)
        .expect("encode force");
    eprintln!(
        "{label} e{effort} d={distance}: pinned={} default={} force={}",
        pinned_bytes.len(),
        default_bytes.len(),
        force_bytes.len()
    );
    if expect_change {
        assert_ne!(
            pinned_bytes.len(),
            default_bytes.len(),
            "{label} e{effort} d={distance}: expected default-path to differ from pinned (W44-65 gate should fire)"
        );
        assert!(
            default_bytes.len() <= pinned_bytes.len(),
            "{label} e{effort} d={distance}: default should be ≤ pinned (we suppress DCT64 to save bytes)"
        );
    }
    assert_eq!(
        default_bytes.len(),
        force_bytes.len(),
        "{label} e{effort} d={distance}: default-path auto and forced suppress should agree (both pick try_dct64=false)"
    );
    for (variant, bytes) in [
        ("pinned", &pinned_bytes),
        ("default", &default_bytes),
        ("force", &force_bytes),
    ] {
        let jxl = out_dir.join(format!("{label}_e{effort}_d{distance}_{variant}.jxl"));
        let png_djxl = out_dir.join(format!("{label}_e{effort}_d{distance}_{variant}_djxl.png"));
        let png_jxlrs = out_dir.join(format!("{label}_e{effort}_d{distance}_{variant}_jxlrs.png"));
        std::fs::write(&jxl, bytes).expect("write jxl");
        let ok_djxl = run_decoder("djxl", DJXL, &jxl, &png_djxl);
        let ok_jxlrs = run_decoder("jxl_cli", JXLRS, &jxl, &png_jxlrs);
        assert!(ok_djxl, "{label} {variant} djxl decoder failed");
        assert!(ok_jxlrs, "{label} {variant} jxl_cli decoder failed");
    }
}

fn assert_pixel_art_invariant(label: &str, src: &Path, effort: u8, distance: f32) {
    let pic = image::open(src).expect("open source");
    let rgb = pic.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    let pinned = LossyConfig::new(distance)
        .with_effort(effort)
        .with_threads(1)
        .with_dct_suppress_hint(Some(false))
        .encode(rgb.as_raw(), w, h, PixelLayout::Rgb8)
        .expect("encode pinned");
    let default = LossyConfig::new(distance)
        .with_effort(effort)
        .with_threads(1)
        .encode(rgb.as_raw(), w, h, PixelLayout::Rgb8)
        .expect("encode default");
    assert_eq!(
        pinned.len(),
        default.len(),
        "{label} e{effort} d={distance}: W44-65 default should produce byte-identical output (mask1x1_median < threshold). pinned={} default={}",
        pinned.len(),
        default.len()
    );
    // Strong invariant: bytes themselves must match.
    assert_eq!(
        pinned, default,
        "{label} e{effort} d={distance}: W44-65 default not byte-identical to pinned"
    );
    eprintln!(
        "{label} e{effort} d={distance}: BYTE-IDENTICAL pinned/default ({} bytes)",
        pinned.len()
    );
}
