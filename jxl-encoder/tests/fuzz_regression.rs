//! Stable replay of the bounded fuzz entry points, including full rendering.
#[path = "../../fuzz/entry.rs"]
mod entry;

#[test]
fn encoder_fuzz_entry_point_regressions() {
    for (width, height) in [(19u16, 13u16), (259, 33)] {
        for lossless in [0, 1] {
            for effort in [1, 5, 8] {
                let mut seed = Vec::from((width - 1).to_le_bytes());
                seed.extend_from_slice(&(height - 1).to_le_bytes());
                seed.extend_from_slice(&[effort - 1, 37, lossless, 6, 3, 127, 251, 0, 65]);
                let encoded =
                    entry::streaming_roundtrip(&seed).expect("regression cell must encode");
                // The fuzz entry already fully renders with jxl-rs. The
                // stable compatibility leg also runs the pinned C++ decoder.
                let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("target/fuzz-regression")
                    .join(std::process::id().to_string());
                std::fs::create_dir_all(&dir).unwrap();
                let path = dir.join(format!("{width}-{height}-{lossless}-{effort}.jxl"));
                std::fs::write(&path, encoded).unwrap();
                let decoded = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
                    .arg(path)
                    .args(["--disable_output", "--num_threads=1"])
                    .output()
                    .unwrap();
                assert!(
                    decoded.status.success(),
                    "djxl: {}",
                    String::from_utf8_lossy(&decoded.stderr)
                );
            }
        }
    }
    for width in [0u32, 1, 514, 1 << 30, u32::MAX] {
        let mut seed = Vec::from(width.to_le_bytes());
        seed.extend_from_slice(&u32::MAX.to_le_bytes());
        seed.extend_from_slice(&[4, 255, 255, 255, 127, 255, 1, 0]);
        entry::request_limits(&seed);
    }
}
