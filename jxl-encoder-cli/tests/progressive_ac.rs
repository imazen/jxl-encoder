//! CLI progressive spellings preserve the final color coefficients.
use std::{
    path::{Path, PathBuf},
    process::Command,
};
#[path = "../../jxl-encoder/tests/support/decoded_extras.rs"]
mod decoded_extras;
#[path = "../../jxl-encoder/tests/support/pfm.rs"]
mod pfm;

fn fresh_directory() -> PathBuf {
    let root = jxl_encoder::test_helpers::output_dir("cli-progressive-ac");
    std::fs::create_dir_all(&root).unwrap();
    (0..)
        .find_map(|i| {
            let path = root.join(format!("run-{i}"));
            match std::fs::create_dir(&path) {
                Ok(()) => Some(path),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => None,
                Err(e) => panic!("{e}"),
            }
        })
        .unwrap()
}
fn render(path: &Path, width: usize, height: usize) -> [Vec<u32>; 2] {
    let bytes = std::fs::read(path).unwrap();
    let (primary, extras) = decoded_extras::decode_current_planes(&bytes, "CLI progressive");
    assert!(extras.is_empty());
    assert_eq!(primary.len(), width * height * 3);
    let output = path.with_extension("pfm");
    let result = Command::new(jxl_encoder::test_helpers::djxl_path())
        .arg(path)
        .arg(&output)
        .arg("--num_threads=1")
        .output()
        .unwrap();
    std::fs::write(path.with_extension("decode.log"), &result.stderr).unwrap();
    assert!(result.status.success());
    let reference = pfm::read_pfm_samples(&output, (width, height));
    assert_eq!(reference.len(), primary.len());
    assert!(primary.iter().chain(&reference).all(|v| v.is_finite()));
    [primary, reference].map(|v| v.into_iter().map(f32::to_bits).collect())
}
#[test]
fn cli_progressive_spellings_preserve_final_pixels() {
    let root = fresh_directory();
    let fixture =
        image::load_from_memory(include_bytes!("../../jxl-encoder/tests/images/frymire.png"))
            .unwrap()
            .to_rgb8();
    for (width, height) in [(64, 64), (259, 133)] {
        let input = root.join(format!("input-{width}x{height}.png"));
        image::imageops::crop_imm(&fixture, 0, 0, width, height)
            .to_image()
            .save(&input)
            .unwrap();
        for policy in [
            "libjxl",
            "libjxl-exact",
            "libjxl-strict",
            "lean-faster",
            "zenjxl",
            "aggressive",
        ] {
            for experimental in [false, true] {
                for stream in [false, true] {
                    let case = root.join(format!(
                        "{width}x{height}-{policy}-experimental{experimental}-stream{stream}"
                    ));
                    std::fs::create_dir(&case).unwrap();
                    let mut baseline = None;
                    for progressive in ["single", "qprogressive", "progressive"] {
                        let output = case.join(format!("{progressive}.jxl"));
                        let mut command = Command::new(env!("CARGO_BIN_EXE_cjxl-rs"));
                        command.arg(&input).arg(&output).args([
                            "--quiet",
                            "--effort",
                            "4",
                            "--distance",
                            "1",
                            "--force-strategy",
                            "3",
                            "--strategy",
                            policy,
                        ]);
                        if experimental {
                            command.arg("--experimental");
                        }
                        if stream {
                            command.args(["--streaming-input", "--streaming-output"]);
                        }
                        if progressive != "single" {
                            command.arg(format!("--{progressive}"));
                        }
                        std::fs::write(
                            output.with_extension("command.txt"),
                            format!("{command:?}\n"),
                        )
                        .unwrap();
                        command.env(
                            "LLVM_PROFILE_FILE",
                            case.join(format!("{progressive}-%p.profraw")),
                        );
                        let result = command.output().unwrap();
                        std::fs::write(output.with_extension("encode.log"), &result.stderr)
                            .unwrap();
                        assert!(
                            result.status.success(),
                            "{}: {}",
                            output.display(),
                            String::from_utf8_lossy(&result.stderr)
                        );
                        let actual = render(&output, width as usize, height as usize);
                        if let Some(expected) = &baseline {
                            assert_eq!(&actual, expected, "{}", output.display());
                        } else {
                            baseline = Some(actual);
                        }
                    }
                }
            }
        }
    }
}
