//! Pinned v0.12 production controls for the geometry and finite-float triggers.
#![cfg(feature = "__expert")]
use sha2::{Digest, Sha256};
#[path = "support/decoded_extras.rs"]
mod decoded_extras;
#[path = "support/pfm.rs"]
mod pfm;

#[test]
fn pinned_responsive_controls_preserve_geometry_and_float_trigger_bits() {
    let root = jxl_encoder::test_helpers::output_dir("squeeze-native-controls");
    std::fs::create_dir_all(&root).unwrap();
    let out = (0..)
        .find_map(|i| {
            let p = root.join(format!("run-{i}"));
            match std::fs::create_dir(&p) {
                Ok(()) => Some(p),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => None,
                Err(e) => panic!("{e}"),
            }
        })
        .unwrap();
    for (w, h, float) in [(8, 65665, false), (65665, 8, false), (16, 1, true)] {
        let label = format!("{w}x{h}-float{float}");
        let expected: Vec<u32> = if float {
            (0..16)
                .flat_map(|x| {
                    if x < 2 {
                        [0x0800_0000, 0, 0xf800_0000]
                    } else {
                        [0xf800_0000, 0, 0x0800_0000]
                    }
                })
                .collect()
        } else {
            (0..h)
                .flat_map(|y| {
                    (0..w).flat_map(move |x| {
                        [
                            ((x * 17 + y * 3) % 256) as u32,
                            ((x * 5 + y * 13) % 256) as u32,
                            ((x * 29 + y * 7) % 256) as u32,
                        ]
                    })
                })
                .collect()
        };
        let mut file = if float {
            format!("PF\n{w} {h}\n-1.0\n").into_bytes()
        } else {
            format!("P6\n{w} {h}\n255\n").into_bytes()
        };
        if float {
            file.extend(expected.iter().flat_map(|v| v.to_le_bytes()));
        } else {
            file.extend(expected.iter().map(|v| *v as u8));
        }
        let source = out.join(format!("{label}.{}", if float { "pfm" } else { "ppm" }));
        std::fs::write(&source, &file).unwrap();
        let source_hash: String = Sha256::digest(&file)
            .iter()
            .map(|v| format!("{v:02x}"))
            .collect();
        std::fs::write(out.join(format!("{label}.source.sha256")), source_hash).unwrap();
        let stream = out.join(format!("{label}.jxl"));
        let result = std::process::Command::new(jxl_encoder::test_helpers::cjxl_path())
            .arg(&source)
            .arg(&stream)
            .args([
                "--distance=0",
                "--effort=1",
                "--modular=1",
                "--responsive=1",
                "--modular_group_size=1",
                "--num_threads=1",
            ])
            .output()
            .unwrap();
        std::fs::write(out.join(format!("{label}.encode.log")), &result.stderr).unwrap();
        assert!(result.status.success(), "{label}: native encoding");
        let bytes = std::fs::read(&stream).unwrap();
        let stream_hash: String = Sha256::digest(&bytes)
            .iter()
            .map(|v| format!("{v:02x}"))
            .collect();
        std::fs::write(out.join(format!("{label}.jxl.sha256")), stream_hash).unwrap();
        let (color, extras) = decoded_extras::decode_current_planes(&bytes, &label);
        assert!(extras.is_empty());
        let codes = |samples: &[f32]| -> Vec<u32> {
            samples
                .iter()
                .map(|v| {
                    if float {
                        v.to_bits()
                    } else {
                        (v * 255.0).round() as u32
                    }
                })
                .collect()
        };
        assert_eq!(
            codes(&color),
            expected,
            "{label}: native stream in primary decoder"
        );
        let rendered = out.join(format!("{label}.decoded.pfm"));
        let result = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
            .arg(&stream)
            .arg(&rendered)
            .arg("--num_threads=1")
            .output()
            .unwrap();
        std::fs::write(out.join(format!("{label}.decode.log")), &result.stderr).unwrap();
        assert!(result.status.success(), "{label}: native rendering");
        assert_eq!(
            codes(&pfm::read_pfm_samples(&rendered, (w, h))),
            expected,
            "{label}: native reference samples"
        );
    }
}
