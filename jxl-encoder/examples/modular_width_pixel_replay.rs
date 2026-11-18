// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Compare retained float-lock streams through the primary and pinned native decoders.
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
#[path = "../tests/support/decoded_frames.rs"]
mod decoded_frames;
#[path = "../tests/support/pfm.rs"]
mod pfm;

fn same(left: &[f32], right: &[f32]) {
    assert_eq!(left.len(), right.len());
    assert!(left.iter().chain(right).all(|value| value.is_finite()));
    assert!(
        left.iter()
            .zip(right)
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );
}
fn sha(path: &Path) -> String {
    Sha256::digest(std::fs::read(path).unwrap())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}
fn main() {
    let args: Vec<_> = std::env::args_os().collect();
    assert_eq!(
        args.len(),
        3,
        "modular_width_pixel_replay <pairs.json> <new-output>"
    );
    let pairs: Vec<serde_json::Value> =
        serde_json::from_slice(&std::fs::read(&args[1]).unwrap()).unwrap();
    assert!(!pairs.is_empty());
    let output = PathBuf::from(&args[2]);
    std::fs::create_dir(&output).unwrap();
    let mut records = Vec::new();
    for row in pairs {
        let name = row["name"].as_str().unwrap();
        let width = row["width"].as_u64().unwrap() as usize;
        let height = row["height"].as_u64().unwrap() as usize;
        let directory = output.join(name);
        std::fs::create_dir(&directory).unwrap();
        let mut primary = Vec::new();
        let mut native = Vec::new();
        for phase in ["before", "after"] {
            let input = Path::new(row[phase].as_str().unwrap());
            let bytes = std::fs::read(input).unwrap();
            let frames = decoded_frames::decode_current_frames(&bytes, name);
            assert_eq!(frames.len(), 1);
            assert_eq!(frames[0].0.len(), width * height * 3);
            assert!(frames[0].1.is_empty());
            let samples = &frames[0].0;
            std::fs::write(
                directory.join(format!("primary-{phase}.f32le")),
                samples
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<u8>>(),
            )
            .unwrap();
            primary.push(samples.clone());
            let path = directory.join(format!("native-{phase}.pfm"));
            let rendered = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
                .arg(input)
                .arg(&path)
                .args(["--num_threads=1", "--color_space=RGB_D65_SRG_Rel_Lin"])
                .output()
                .unwrap();
            std::fs::write(
                directory.join(format!("native-{phase}.log")),
                [rendered.stdout, rendered.stderr].concat(),
            )
            .unwrap();
            assert!(rendered.status.success());
            native.push(pfm::read_pfm_samples(&path, (width, height)));
        }
        same(&primary[0], &primary[1]);
        same(&native[0], &native[1]);
        records.push(
            serde_json::json!({"name":name,"width":width,"height":height,
            "primary_before_sha256":sha(&directory.join("primary-before.f32le")),
            "primary_after_sha256":sha(&directory.join("primary-after.f32le")),
            "native_before_sha256":sha(&directory.join("native-before.pfm")),
            "native_after_sha256":sha(&directory.join("native-after.pfm")),
            "primary_bit_exact":true,"native_bit_exact":true}),
        );
        std::fs::write(
            output.join("results.partial.json"),
            serde_json::to_vec_pretty(&records).unwrap(),
        )
        .unwrap();
        println!("{name}: complete primary and native pixels are bit-identical");
    }
    std::fs::write(
        output.join("results.json"),
        serde_json::to_vec_pretty(&records).unwrap(),
    )
    .unwrap();
}
