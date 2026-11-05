//! Retain every measured stream and require independent full decodes.
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

pub(super) fn hash(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

pub(super) struct Evidence {
    dir: PathBuf,
    djxl: String,
}

impl Evidence {
    pub fn new(output: &Path, args: &[String]) -> Self {
        assert!(!output.exists(), "refusing existing output");
        let dir = PathBuf::from(format!("{}.artifacts", output.display()));
        std::fs::create_dir(&dir).expect("create new artifact directory");
        let djxl = jxl_encoder::test_helpers::djxl_path();
        let manifest = serde_json::json!({
            "schema": "effort-envelope-evidence-v1",
            "build_commit": option_env!("JXL_PROBE_BUILD_COMMIT"),
            "binary_sha256": hash(&std::fs::read(std::env::current_exe().unwrap()).unwrap()),
            "harness_sha256": hash(include_bytes!("../effort_monotonicity_gate.rs")),
            "cargo_lock_sha256": hash(include_bytes!("../../../Cargo.lock")),
            "djxl": djxl, "args": args,
            "timing": "encode wall; validation and artifact IO excluded"
        });
        std::fs::write(
            dir.join("manifest.json"),
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();
        Self { dir, djxl }
    }

    pub fn retain(&self, bytes: &[u8], extension: &str) -> String {
        let key = hash(bytes);
        let path = self.dir.join(format!("{key}.{extension}"));
        if path.exists() {
            assert_eq!(std::fs::read(path).unwrap(), bytes, "artifact collision");
        } else {
            std::fs::write(path, bytes).expect("retain artifact");
        }
        key
    }

    pub fn verify(&self, encoded: &[u8], src: &[u8], n: u32, lossless: bool) -> String {
        let key = self.retain(encoded, "jxl");
        let rust = super::decode::verify_jxl_rs(encoded, n as usize, n as usize);
        assert_eq!(rust.len(), src.len(), "Rust channel count");
        if lossless {
            for (&sample, &expected) in rust.iter().zip(src) {
                assert_eq!(
                    (sample * 255.0).round() as i32,
                    i32::from(expected),
                    "Rust lossless sample"
                );
            }
        }
        // Decode each distinct stream once, including repeated timing samples.
        let png = self.dir.join(format!("{key}.png"));
        if !png.exists() {
            let result = std::process::Command::new(&self.djxl)
                .arg(self.dir.join(format!("{key}.jxl")))
                .arg(&png)
                .args(["--bits_per_sample=8", "--num_threads=1"])
                .output()
                .expect("run djxl");
            std::fs::write(
                self.dir.join(format!("{key}.djxl.log")),
                [&result.stdout[..], &result.stderr[..]].concat(),
            )
            .unwrap();
            assert!(
                result.status.success(),
                "djxl failed: {}",
                String::from_utf8_lossy(&result.stderr)
            );
        }
        let decoded = image::open(png).expect("read djxl pixels").to_rgb8();
        assert_eq!(decoded.dimensions(), (n, n), "djxl dimensions");
        if lossless {
            assert_eq!(decoded.as_raw(), src, "djxl lossless samples");
        }
        key
    }
}
