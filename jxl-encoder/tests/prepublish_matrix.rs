//! Changed-path release checks. The caller supplies a natural image and an
//! artifact directory; every encoded stream and decoder diagnostic is retained.
#![cfg(all(feature = "__expert", feature = "corpus-tests"))]

use jxl_encoder::api::{EncoderStrategy, SectionedTrees};
use jxl_encoder::{
    LosslessConfig, LosslessInternalParams, LossyConfig, PixelLayout, ProgressiveMode,
};
use sha2::{Digest, Sha256};
use std::io::Write;
use std::path::{Path, PathBuf};

fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn inputs() -> (image::RgbImage, image::RgbImage, PathBuf) {
    let photo = image::open(std::env::var_os("JXL_AUDIT_PHOTO").expect("JXL_AUDIT_PHOTO"))
        .expect("natural image")
        .to_rgb8();
    let graphic = image::open(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/images/frymire-srgb.png"
    ))
    .unwrap()
    .to_rgb8();
    let dir = PathBuf::from(std::env::var_os("JXL_AUDIT_ARTIFACTS").expect("JXL_AUDIT_ARTIFACTS"));
    std::fs::create_dir_all(&dir).unwrap();
    (photo, graphic, dir)
}

fn pixels(
    w: u32,
    h: u32,
    pattern: usize,
    photo: &image::RgbImage,
    graphic: &image::RgbImage,
) -> Vec<u8> {
    let mut data = Vec::new();
    let mut rng = 0x91e10da5u32;
    for y in 0..h {
        for x in 0..w {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            let p = match pattern {
                0 => [0, 0, 0, 0],
                1 => [255, 255, 255, 255],
                2 => {
                    let v = if (x + y) % 2 == 0 { 0 } else { 255 };
                    [v, 255 - v, v, v]
                }
                3 => {
                    let v = if x == w - 1 && y == h - 1 { 255 } else { 0 };
                    [v, v, v, 255]
                }
                4 => [x as u8, x as u8, x as u8, y.wrapping_add(x) as u8],
                5 => [
                    rng as u8,
                    (rng >> 8) as u8,
                    (rng >> 16) as u8,
                    (rng >> 24) as u8,
                ],
                6 => {
                    let v = if y % 16 < 2 && x % 8 < 6 { 0 } else { 255 };
                    [v, v, v, 255]
                }
                7 => [0, (rng >> 8) as u8, 0, if x % 257 == 0 { 1 } else { 255 }],
                8 | 9 => {
                    let src = if pattern == 8 { photo } else { graphic };
                    let p = src.get_pixel(x % src.width(), y % src.height()).0;
                    [
                        p[0],
                        p[1],
                        p[2],
                        (x.wrapping_mul(13) + y.wrapping_mul(7)) as u8,
                    ]
                }
                _ => unreachable!(),
            };
            data.extend_from_slice(&p);
        }
    }
    data
}

fn verify(dir: &Path, label: &str, bytes: &[u8], source: &[u8], w: u32, h: u32, lossless: bool) {
    let sha = sha256(bytes);
    let jxl = dir.join(format!("{sha}.jxl"));
    std::fs::write(&jxl, bytes).unwrap();
    let decoded = zenjxl_decoder::decode(bytes).expect("Rust full decode");
    assert_eq!(
        (decoded.width, decoded.height, decoded.channels),
        (w as usize, h as usize, 4)
    );
    let png = dir.join(format!("{sha}.png"));
    let output = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
        .arg(&jxl)
        .arg(&png)
        .arg("--num_threads=1")
        .output()
        .unwrap();
    std::fs::write(dir.join(format!("{sha}.djxl.log")), &output.stderr).unwrap();
    assert!(
        output.status.success(),
        "{label}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let reference = image::open(png).unwrap().to_rgba8();
    assert_eq!(reference.dimensions(), (w, h));
    if lossless {
        assert_eq!(
            decoded.data.as_slice(),
            source,
            "{label}: Rust lossless pixels"
        );
        assert_eq!(
            reference.as_raw(),
            source,
            "{label}: libjxl lossless pixels"
        );
    } else {
        // Color is lossy and native decoders can use different output rounding.
        // Alpha is coded losslessly and must remain exact in BOTH decoders.
        for (i, p) in source.as_chunks::<4>().0.iter().enumerate() {
            assert_eq!(decoded.data[4 * i + 3], p[3], "{label}: Rust alpha at {i}");
            assert_eq!(
                reference.as_raw()[4 * i + 3],
                p[3],
                "{label}: libjxl alpha at {i}"
            );
        }
    }
}

fn run(lossless: bool) {
    let (photo, graphic, root) = inputs();
    let kind = if lossless { "lossless" } else { "lossy" };
    let dir = root.join(kind);
    std::fs::create_dir_all(&dir).unwrap();
    let mut log = std::fs::File::create(dir.join("cells.tsv")).unwrap();
    writeln!(log, "cell\tbytes\tsha256\tstatus").unwrap();
    let selected = std::env::var("JXL_AUDIT_CASE").ok();
    let mut failures = Vec::new();
    let mut count = 0;
    for (w, h) in [
        (1, 1),
        (1, 257),
        (257, 1),
        (7, 9),
        (255, 17),
        (256, 17),
        (257, 17),
        (259, 133),
        (2049, 9),
    ] {
        for pattern in 0..10 {
            let source = pixels(w, h, pattern, &photo, &graphic);
            let stride = w as usize * 4 + 13;
            let mut padded = vec![0xa5; stride * h as usize];
            for (src, dst) in source
                .chunks_exact(w as usize * 4)
                .zip(padded.chunks_exact_mut(stride))
            {
                dst[..src.len()].copy_from_slice(src);
            }
            for variant in 0..12 {
                let label = format!("{kind}-{w}x{h}-p{pattern}-v{variant}");
                if selected.as_ref().is_some_and(|s| s != &label) {
                    continue;
                }
                count += 1;
                eprintln!("AUDIT {label}");
                let result = std::panic::catch_unwind(|| {
                    let strategy = if variant % 2 == 0 {
                        EncoderStrategy::Zenjxl
                    } else {
                        EncoderStrategy::Libjxl
                    };
                    let bytes = if lossless {
                        let mut params = LosslessInternalParams::default();
                        params.forced_wp_mode = Some((variant % 5) as u8);
                        let cfg = LosslessConfig::new()
                            .with_effort([3, 5, 7, 8, 9, 10][variant / 2])
                            .with_strategy(strategy)
                            .with_threads(1)
                            .with_ans(variant % 4 != 0)
                            .with_squeeze(variant % 3 == 0)
                            .with_sectioned_trees(
                                [
                                    SectionedTrees::Off,
                                    SectionedTrees::On,
                                    SectionedTrees::Hybrid,
                                ][variant % 3],
                            )
                            .with_internal_params(params);
                        let bytes = cfg.encode(&source, w, h, PixelLayout::Rgba8).unwrap();
                        std::fs::write(dir.join(format!("{label}.jxl")), &bytes).unwrap();
                        let strided = cfg
                            .encode_request(w, h, PixelLayout::Rgba8)
                            .with_row_stride(stride)
                            .encode(&padded)
                            .unwrap();
                        assert_eq!(bytes, strided, "{label}: stride");
                        let mut stream = cfg.encoder(w, h, PixelLayout::Rgba8).unwrap();
                        for rows in source.chunks(w as usize * 4 * 7) {
                            stream
                                .push_rows(rows, (rows.len() / (w as usize * 4)) as u32)
                                .unwrap();
                        }
                        assert_eq!(bytes, stream.finish().unwrap(), "{label}: streaming");
                        bytes
                    } else {
                        let cfg = LossyConfig::new([0.1, 0.5, 1.5, 4.0, 10.0, 15.0][variant / 2])
                            .with_effort([3, 4, 7, 8, 9, 10][variant / 2])
                            .with_strategy(strategy)
                            .with_threads(1)
                            .with_auto_resampling(false)
                            .with_progressive(
                                [
                                    ProgressiveMode::Single,
                                    ProgressiveMode::QuantizedAcFullAc,
                                    ProgressiveMode::DcVlfLfAc,
                                ][variant % 3],
                            );
                        let bytes = cfg.encode(&source, w, h, PixelLayout::Rgba8).unwrap();
                        std::fs::write(dir.join(format!("{label}.jxl")), &bytes).unwrap();
                        let strided = cfg
                            .encode_request(w, h, PixelLayout::Rgba8)
                            .with_row_stride(stride)
                            .encode(&padded)
                            .unwrap();
                        assert_eq!(bytes, strided, "{label}: stride");
                        bytes
                    };
                    let sha = sha256(&bytes);
                    verify(&dir, &label, &bytes, &source, w, h, lossless);
                    (bytes.len(), sha)
                });
                match result {
                    Ok((len, sha)) => writeln!(log, "{label}\t{len}\t{sha}\tpass").unwrap(),
                    Err(_) => {
                        writeln!(log, "{label}\t0\t\tfail").unwrap();
                        failures.push(label);
                    }
                }
                log.flush().unwrap();
            }
        }
    }
    assert!(count > 0, "case selector matched no cells");
    assert!(failures.is_empty(), "{count} cells; failures: {failures:?}");
    eprintln!("AUDIT {kind}: {count} cells passed");
}

#[test]
fn lossless_changed_settings() {
    run(true);
}

#[test]
fn strict_extras_and_progressive() {
    run(false);
}
