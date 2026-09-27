//! Process-isolated, full-output regression gate for #110.
use crate::api::{EncoderStrategy, LosslessConfig, LossyConfig, PixelLayout, SectionedTrees};
use std::{
    collections::BTreeMap,
    io::Write,
    path::{Path, PathBuf},
    process::Command,
};

const TEST: &str = "modular::lz77_keep_best_tests::production_keep_best_regressions";

#[test]
fn production_keep_best_regressions() {
    if let Ok(arm) = std::env::var("JXL_KEEP_BEST_TEST_ARM") {
        #[cfg(feature = "parallel")]
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| run_arm(&arm));
        #[cfg(not(feature = "parallel"))]
        run_arm(&arm);
        return;
    }
    let output = PathBuf::from(
        std::env::var_os("LZ77_KEEP_BEST_OUTPUT").expect("caller must set LZ77_KEEP_BEST_OUTPUT"),
    );
    std::fs::create_dir_all(&output).unwrap();
    for arm in ["off", "on"] {
        let dir = output.join(arm);
        std::fs::create_dir(&dir).expect("refuse to overwrite an earlier run");
        let log = std::fs::File::create(dir.join("test.log")).unwrap();
        let status = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", TEST, "--nocapture"])
            .env("JXL_KEEP_BEST_TEST_ARM", arm)
            .env("JXL_LZ77_KEEP_BEST", if arm == "on" { "1" } else { "0" })
            .stdout(log.try_clone().unwrap())
            .stderr(log)
            .status()
            .unwrap();
        assert!(status.success(), "{arm} failed; see {}", dir.display());
    }
    let read = |arm: &str| -> BTreeMap<String, (usize, String)> {
        std::fs::read_to_string(output.join(arm).join("cells.tsv"))
            .unwrap()
            .lines()
            .skip(1)
            .map(|line| {
                let f: Vec<_> = line.split('\t').collect();
                (f[0].to_owned(), (f[1].parse().unwrap(), f[2].to_owned()))
            })
            .collect()
    };
    let off = read("off");
    let on = read("on");
    assert_eq!(
        off.keys().collect::<Vec<_>>(),
        on.keys().collect::<Vec<_>>()
    );
    assert_eq!(off.len(), 70, "the complete selected grid must run");
    let mut wins = 0;
    for (label, (size, hash)) in &off {
        let selected = &on[label];
        assert!(selected.0 <= *size, "{label}: {size} -> {}", selected.0);
        if label.contains("strict") || label.contains("lossy") {
            assert_eq!(selected.1, *hash, "{label}: opt-in escaped lossless Zen");
        }
        wins += usize::from(selected.0 < *size);
        eprintln!("KEEP-BEST {label}: {size} -> {}", selected.0);
    }
    assert!(wins > 0, "selection must reach a winning candidate");
    let fry = "frymire-256-e8-f32-global";
    assert!(on[fry].0 < off[fry].0, "retain the verified bucket win");
    let terminal = "terminal-256-e8-f32-global";
    assert!(
        on[terminal].0 <= off[terminal].0,
        "reject the verified terminal regression"
    );
}

fn hash(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn run_arm(arm: &str) {
    let enabled = LosslessConfig::new().with_effort(8);
    assert_eq!(enabled.effective_profile().lz77_keep_best, arm == "on");
    for disabled in [
        enabled.clone().with_lz77(false),
        enabled
            .clone()
            .with_lz77_method(crate::entropy_coding::lz77::Lz77Method::Rle),
        enabled.clone().with_ans(false),
        enabled.clone().with_faster_decoding(1),
        enabled.with_strategy(EncoderStrategy::Libjxl),
    ] {
        assert!(!disabled.effective_profile().lz77_keep_best);
    }
    let dir = PathBuf::from(std::env::var_os("LZ77_KEEP_BEST_OUTPUT").unwrap()).join(arm);
    let corpus = PathBuf::from(
        std::env::var_os("CODEC_CORPUS_DIR").expect("caller must provision codec-corpus"),
    );
    let sources = [
        (
            "frymire",
            PathBuf::from(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/images/frymire-srgb.png"
            )),
        ),
        ("terminal", corpus.join("gb82-sc/terminal.png")),
    ];
    let mut table = std::fs::File::create(dir.join("cells.tsv")).unwrap();
    writeln!(table, "cell\tbytes\tsha256\tencode_ms\tsource_sha256").unwrap();
    for (name, path) in sources {
        let source_hash = hash(&std::fs::read(&path).unwrap());
        let source = image::open(&path).unwrap().to_rgb8();
        for n in [64, 256, 259] {
            let rgb = image::imageops::crop_imm(
                &source,
                (source.width() - n) / 2,
                (source.height() - n) / 2,
                n,
                n,
            )
            .to_image();
            let floats: Vec<_> = rgb
                .as_raw()
                .iter()
                .map(|&v| (u16::from(v) << 8) as f32 / 65535.0)
                .collect();
            let raw: Vec<_> = floats.iter().flat_map(|v| v.to_ne_bytes()).collect();
            for e in [8, 9] {
                let base = LosslessConfig::new()
                    .with_effort(e)
                    .with_threads(0)
                    .with_sectioned_trees(SectionedTrees::Off);
                for strict in [false, true] {
                    let label = format!(
                        "{name}-{n}-e{e}-f32-{}",
                        if strict { "strict" } else { "global" }
                    );
                    let cfg = base.clone().with_strategy(if strict {
                        EncoderStrategy::Libjxl
                    } else {
                        EncoderStrategy::Zenjxl
                    });
                    super::lz77_keep_best::begin_observation();
                    let now = std::time::Instant::now();
                    let data = cfg.encode(&raw, n, n, PixelLayout::RgbLinearF32).unwrap();
                    let costs = super::lz77_keep_best::take_observation();
                    if arm == "on" && !strict {
                        assert!(!costs.is_empty(), "{label}: selector was not reached");
                        std::fs::write(
                            dir.join(format!("{label}.costs.txt")),
                            format!("{costs:?}\n"),
                        )
                        .unwrap();
                        if n == 256 && e == 8 {
                            let bucket = costs.iter().find(|(p, _)| *p == "bucket").unwrap().1;
                            let incumbent = costs[0].1;
                            if name == "terminal" {
                                assert!(
                                    bucket > incumbent,
                                    "terminal must exercise candidate rejection"
                                );
                            }
                            if name == "frymire" {
                                assert!(
                                    bucket < incumbent,
                                    "frymire must exercise a candidate win"
                                );
                            }
                        }
                    } else {
                        assert!(costs.is_empty(), "{label}: selector unexpectedly active");
                    }

                    let ms = now.elapsed().as_secs_f64() * 1000.0;
                    let mut streaming = cfg.encoder(n, n, PixelLayout::RgbLinearF32).unwrap();
                    let row = n as usize * 12;
                    for chunk in raw.chunks(row * 7) {
                        streaming
                            .push_rows(chunk, (chunk.len() / row) as u32)
                            .unwrap();
                    }
                    assert_eq!(streaming.finish().unwrap(), data, "{label}: streaming");
                    check_float(&dir, &label, &data, n as usize, &floats);
                    writeln!(
                        table,
                        "{label}\t{}\t{}\t{ms:.3}\t{source_hash}",
                        data.len(),
                        hash(&data)
                    )
                    .unwrap();
                    table.flush().unwrap();
                    eprintln!("{arm} {label}: {} bytes", data.len());
                }
                if n == 256 {
                    continue;
                } // caller's explicit interaction grid
                let wide: Vec<u16> = rgb.as_raw().iter().map(|&v| u16::from(v) << 8).collect();
                let wide_raw: Vec<u8> = wide.iter().flat_map(|v| v.to_ne_bytes()).collect();
                let label = format!("{name}-{n}-e{e}-u16-global");
                let now = std::time::Instant::now();
                let data = base.encode(&wide_raw, n, n, PixelLayout::Rgb16).unwrap();
                let ms = now.elapsed().as_secs_f64() * 1000.0;
                let decoded = crate::test_helpers::decode_with_jxl_rs(&data).unwrap();
                assert_eq!(
                    (decoded.width, decoded.height, decoded.channels),
                    (n as usize, n as usize, 3)
                );
                let samples: Vec<u16> = decoded
                    .pixels
                    .iter()
                    .map(|v| (v * 65535.0).round() as u16)
                    .collect();
                assert_eq!(samples, wide, "{label}: Rust");
                let jxl = dir.join(format!("{label}.jxl"));
                let png = dir.join(format!("{label}.png"));
                std::fs::write(&jxl, &data).unwrap();
                run_djxl(&dir, &label, &jxl, &png);
                assert_eq!(
                    *image::open(png).unwrap().to_rgb16().as_raw(),
                    wide,
                    "{label}: djxl"
                );
                writeln!(
                    table,
                    "{label}\t{}\t{}\t{ms:.3}\t{source_hash}",
                    data.len(),
                    hash(&data)
                )
                .unwrap();
                table.flush().unwrap();
                for (mode, squeeze, sectioned) in [
                    ("global", false, SectionedTrees::Off),
                    ("squeeze", true, SectionedTrees::Off),
                    ("local", false, SectionedTrees::On),
                    ("hybrid", false, SectionedTrees::Hybrid),
                ] {
                    let cfg = base
                        .clone()
                        .with_squeeze(squeeze)
                        .with_sectioned_trees(sectioned);
                    let label = format!("{name}-{n}-e{e}-u8-{mode}");
                    let now = std::time::Instant::now();
                    let data = cfg.encode(rgb.as_raw(), n, n, PixelLayout::Rgb8).unwrap();
                    let ms = now.elapsed().as_secs_f64() * 1000.0;
                    check_u8(&dir, &label, &data, &rgb);
                    writeln!(
                        table,
                        "{label}\t{}\t{}\t{ms:.3}\t{source_hash}",
                        data.len(),
                        hash(&data)
                    )
                    .unwrap();
                    table.flush().unwrap();
                    eprintln!("{arm} {label}: {} bytes", data.len());
                }
            }
            // Shared matcher users must remain byte-identical when the switch is set.
            let label = format!("{name}-{n}-lossy");
            let data = LossyConfig::new(2.0)
                .with_effort(9)
                .with_threads(0)
                .encode(rgb.as_raw(), n, n, PixelLayout::Rgb8)
                .unwrap();
            crate::test_helpers::decode_with_jxl_rs(&data).unwrap();
            let jxl = dir.join(format!("{label}.jxl"));
            std::fs::write(&jxl, &data).unwrap();
            run_djxl(&dir, &label, &jxl, &dir.join(format!("{label}.png")));
            writeln!(
                table,
                "{label}\t{}\t{}\t0\t{source_hash}",
                data.len(),
                hash(&data)
            )
            .unwrap();
            table.flush().unwrap();
        }
    }
}

fn run_djxl(dir: &Path, label: &str, jxl: &Path, out: &Path) {
    let result = Command::new(crate::test_helpers::djxl_path())
        .arg(jxl)
        .arg(out)
        .arg("--num_threads=1")
        .output()
        .unwrap();
    std::fs::write(dir.join(format!("{label}.djxl.log")), &result.stderr).unwrap();
    assert!(
        result.status.success(),
        "{label}: {}",
        String::from_utf8_lossy(&result.stderr)
    );
}

fn check_u8(dir: &Path, label: &str, data: &[u8], rgb: &image::RgbImage) {
    let decoded = crate::test_helpers::decode_with_jxl_rs(data).unwrap();
    assert_eq!(
        (decoded.width, decoded.height, decoded.channels),
        (rgb.width() as usize, rgb.height() as usize, 3)
    );
    let samples: Vec<_> = decoded
        .pixels
        .iter()
        .map(|v| (v * 255.0).round() as u8)
        .collect();
    assert_eq!(samples, *rgb.as_raw(), "{label}: Rust");
    let jxl = dir.join(format!("{label}.jxl"));
    std::fs::write(&jxl, data).unwrap();
    let png = dir.join(format!("{label}.png"));
    run_djxl(dir, label, &jxl, &png);
    assert_eq!(image::open(png).unwrap().to_rgb8(), *rgb, "{label}: djxl");
}

fn check_float(dir: &Path, label: &str, data: &[u8], n: usize, values: &[f32]) {
    use std::io::BufRead;
    let decoded = crate::test_helpers::decode_with_jxl_rs(data).unwrap();
    assert_eq!((decoded.width, decoded.height, decoded.channels), (n, n, 3));
    let expected: Vec<_> = values.iter().map(|v| v.to_bits()).collect();
    assert_eq!(
        decoded
            .pixels
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        expected,
        "{label}: Rust"
    );
    let jxl = dir.join(format!("{label}.jxl"));
    std::fs::write(&jxl, data).unwrap();
    let pfm = dir.join(format!("{label}.pfm"));
    run_djxl(dir, label, &jxl, &pfm);
    let mut reader = std::io::BufReader::new(std::fs::File::open(pfm).unwrap());
    let mut line = String::new();
    reader.read_line(&mut line).unwrap();
    assert_eq!(line.trim(), "PF");
    line.clear();
    reader.read_line(&mut line).unwrap();
    assert_eq!(line.trim(), format!("{n} {n}"));
    line.clear();
    reader.read_line(&mut line).unwrap();
    let scale: f32 = line.trim().parse().unwrap();
    assert_eq!(scale.abs(), 1.0);
    let mut bytes = Vec::new();
    std::io::Read::read_to_end(&mut reader, &mut bytes).unwrap();
    assert_eq!(bytes.len(), n * n * 12);
    let actual: Vec<_> = bytes
        .chunks_exact(n * 12)
        .rev()
        .flat_map(|row| {
            row.as_chunks::<4>().0.iter().map(|b| {
                if scale < 0.0 {
                    u32::from_le_bytes(*b)
                } else {
                    u32::from_be_bytes(*b)
                }
            })
        })
        .collect();
    assert_eq!(actual, expected, "{label}: djxl");
}
