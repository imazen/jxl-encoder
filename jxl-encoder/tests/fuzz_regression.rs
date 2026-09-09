//! Stable replay of the bounded fuzz entry points, including full rendering.
#[path = "support/fuzz_entry.rs"]
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

    // #95 chunk 4 / #109: the wide-integer and lossless-float surfaces, which
    // had no fuzz coverage at all. Walk the sample widths that bracket the
    // modular limits (0 and 32 are invalid, 1 and 31 are the ends of the legal
    // range) against every plane-count/flag combination the entry derives.
    for bits in [0u8, 1, 8, 16, 31, 32, 255] {
        for flags in 0u8..8 {
            for planes in 0u8..6 {
                let mut seed = Vec::new();
                seed.extend_from_slice(&17u16.to_le_bytes()); // width
                seed.extend_from_slice(&13u16.to_le_bytes()); // height
                seed.push(bits);
                seed.push(flags);
                seed.push(7); // effort
                seed.push(planes);
                seed.extend_from_slice(&0xA5A5_1234u32.to_le_bytes());
                seed.extend_from_slice(&[0x11, 0x22, 0x33, 0x44]);
                entry::wide_lossless(&seed);
            }
        }
    }
}
