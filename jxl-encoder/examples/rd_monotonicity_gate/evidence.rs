//! Retained evidence for the distance gate. No files are reused across runs.
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

pub(super) fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

pub(super) struct Evidence {
    pub dir: PathBuf,
}

impl Evidence {
    pub fn new(output: &Path, args: &[String]) -> Self {
        assert!(
            !output.exists(),
            "refusing existing output: {}",
            output.display()
        );
        let mut dir = output.as_os_str().to_os_string();
        dir.push(".artifacts");
        let dir = PathBuf::from(dir);
        std::fs::create_dir(&dir).expect("create new artifact directory");
        let binary = std::env::current_exe().expect("executable path");
        let metadata = serde_json::json!({
            "schema": "rd-monotonicity-evidence-v1",
            "build_commit": option_env!("JXL_PROBE_BUILD_COMMIT"),
            "binary_sha256": sha256(&std::fs::read(binary).expect("executable bytes")),
            "harness_sha256": sha256(include_bytes!("../rd_monotonicity_gate.rs")),
            "cargo_lock_sha256": sha256(include_bytes!("../../../Cargo.lock")),
            "args": args,
            "timing": "diagnostic only; Rust encode wall versus cjxl process wall",
        });
        std::fs::write(
            dir.join("manifest.json"),
            serde_json::to_vec_pretty(&metadata).unwrap(),
        )
        .unwrap();
        Self { dir }
    }

    pub fn retain(&self, bytes: &[u8], extension: &str) -> String {
        let hash = sha256(bytes);
        let path = self.dir.join(format!("{hash}.{extension}"));
        if path.exists() {
            assert_eq!(
                std::fs::read(path).unwrap(),
                bytes,
                "artifact hash collision"
            );
        } else {
            std::fs::write(path, bytes).expect("persist artifact");
        }
        hash
    }

    pub fn verify(&self, bytes: &[u8], hash: &str, size: u32, djxl: &str) {
        super::decode::verify_jxl_rs(bytes, size as usize, size as usize);
        let output = std::process::Command::new(djxl)
            .arg(self.dir.join(format!("{hash}.jxl")))
            .args(["--disable_output", "--num_threads=1"])
            .output()
            .expect("run djxl");
        std::fs::write(
            self.dir.join(format!("{hash}.djxl.log")),
            [&output.stdout[..], &output.stderr[..]].concat(),
        )
        .unwrap();
        assert!(
            output.status.success(),
            "djxl failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    pub fn metric(
        &self,
        result: &butteraugli::ButteraugliResult,
        width: usize,
        height: usize,
    ) -> String {
        let map = result.diffmap.as_ref().expect("requested diffmap");
        let mut bytes = b"BFMAPF32".to_vec();
        bytes.extend_from_slice(&(width as u32).to_le_bytes());
        bytes.extend_from_slice(&(height as u32).to_le_bytes());
        for value in map.buf() {
            assert!(value.is_finite(), "non-finite diffmap");
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let hash = self.retain(&bytes, "bfmap");
        let summary = serde_json::json!({"score": result.score,
            "pnorm1": result.pnorm(1.0).unwrap(), "pnorm2": result.pnorm(2.0).unwrap(),
            "pnorm3": result.pnorm_3, "pnorm6": result.pnorm(6.0).unwrap()});
        std::fs::write(
            self.dir.join(format!("{hash}.metrics.json")),
            serde_json::to_vec(&summary).unwrap(),
        )
        .unwrap();
        hash
    }
}
