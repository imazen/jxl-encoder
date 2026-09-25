//! Decode both arms of persisted process-wall measurements, outside timing.
#![cfg(feature = "corpus-tests")]

use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::path::PathBuf;

#[test]
fn retained_measurements_fully_decode() {
    let tables =
        std::env::var("JXL_AUDIT_PERF_TSVS").expect("JXL_AUDIT_PERF_TSVS (newline-separated)");
    let side_filter = std::env::var("JXL_AUDIT_PERF_SIDE").unwrap_or_else(|_| "both".into());
    assert!(["both", "base_sha256", "ours_sha256"].contains(&side_filter.as_str()));
    let mut count = 0;
    let mut failures = Vec::new();
    for table in tables.lines() {
        let table = PathBuf::from(table);
        let meta: serde_json::Value = serde_json::from_slice(
            &std::fs::read(format!("{}.meta.json", table.display())).unwrap(),
        )
        .unwrap();
        let command = meta["command"]
            .as_array()
            .expect("recorded benchmark command");
        let lossless = !command.iter().any(|arg| arg == "--lossy-distance");
        let artifacts = table.with_extension("artifacts");
        let text = std::fs::read_to_string(&table).unwrap();
        let mut lines = text.lines();
        let columns: Vec<_> = lines.next().unwrap().split('\t').collect();
        let col = |name| columns.iter().position(|s| *s == name).unwrap();
        let mut seen = BTreeSet::new();
        for line in lines {
            let fields: Vec<_> = line.split('\t').collect();
            let source = image::open(fields[col("image")]).unwrap().to_rgba8();
            for side in ["base_sha256", "ours_sha256"] {
                if side_filter != "both" && side_filter != side {
                    continue;
                }
                for sha in fields[col(side)].split(',') {
                    if !seen.insert((fields[col("image")].to_string(), sha.to_string())) {
                        continue;
                    }
                    let result = std::panic::catch_unwind(|| {
                        eprintln!("PERF-DECODE START {} {side} {sha}", fields[col("cell")]);
                        let jxl = artifacts.join(format!("{sha}.jxl"));
                        let bytes = std::fs::read(&jxl).unwrap();
                        let actual: String = Sha256::digest(&bytes)
                            .iter()
                            .map(|b| format!("{b:02x}"))
                            .collect();
                        assert_eq!(actual, sha, "artifact digest");
                        let decoded =
                            zenjxl_decoder::decode(&bytes).expect("primary Rust full decode");
                        assert_eq!(
                            (decoded.width, decoded.height),
                            (source.width() as usize, source.height() as usize)
                        );
                        // The convenience decoder returns gray+alpha for gray input.
                        // Expand channels without changing any sample value.
                        let rgba = match decoded.channels {
                            4 => decoded.data,
                            2 => {
                                let (pixels, tail) = decoded.data.as_chunks::<2>();
                                assert!(tail.is_empty());
                                pixels
                                    .iter()
                                    .flat_map(|p| [p[0], p[0], p[0], p[1]])
                                    .collect()
                            }
                            channels => panic!("unexpected channel count: {channels}"),
                        };
                        assert_eq!(rgba.len(), source.as_raw().len());
                        let png = artifacts.join(format!("{sha}.png"));
                        let output =
                            std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
                                .arg(&jxl)
                                .arg(&png)
                                .arg("--num_threads=1")
                                .output()
                                .unwrap();
                        std::fs::write(artifacts.join(format!("{sha}.djxl.log")), &output.stderr)
                            .unwrap();
                        assert!(
                            output.status.success(),
                            "{}: {}",
                            jxl.display(),
                            String::from_utf8_lossy(&output.stderr)
                        );
                        let reference = image::open(png).unwrap().to_rgba8();
                        assert_eq!(reference.dimensions(), source.dimensions());
                        if lossless {
                            assert!(
                                rgba.as_slice() == source.as_raw(),
                                "{}: Rust lossless mismatch",
                                jxl.display()
                            );
                            assert!(
                                reference == source,
                                "{}: libjxl lossless mismatch",
                                jxl.display()
                            );
                        }
                        eprintln!(
                            "PERF-DECODE {} {side} {sha} lossless={lossless}",
                            fields[col("cell")]
                        );
                    });
                    count += 1;
                    if result.is_err() {
                        failures.push(format!("{} {side} {sha}", fields[col("cell")]));
                    }
                }
            }
        }
    }
    assert!(count > 0, "no measured artifacts verified");
    assert!(
        failures.is_empty(),
        "{count} streams checked; failures: {failures:?}"
    );
    eprintln!("PERF-DECODE {count} unique source/stream pairs passed");
}
