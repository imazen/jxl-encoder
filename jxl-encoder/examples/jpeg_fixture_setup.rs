//! Provision the legacy JPEG tests before invoking them; no test-body skips.
use std::path::{Path, PathBuf};
use std::process::Command;

fn preserve_or_write(path: &Path, bytes: &[u8]) {
    if path.exists() {
        assert!(
            std::fs::read(path).unwrap() == bytes,
            "fixture differs: preserve it and select a fresh JXL_ENCODER_OUTPUT_DIR: {}",
            path.display()
        );
    } else {
        std::fs::write(path, bytes).unwrap();
    }
}

fn main() {
    let root_file = PathBuf::from(
        std::env::args_os()
            .nth(1)
            .expect("usage: jpeg_fixture_setup <corpus-root-path-file>"),
    );
    let cjpeg = std::env::var_os("CJPEG_PATH").unwrap_or_else(|| "cjpeg".into());
    let version = Command::new(&cjpeg)
        .arg("-version")
        .output()
        .expect("cjpeg is required");
    assert!(version.status.success(), "cjpeg -version failed");
    eprintln!("{}", String::from_utf8_lossy(&version.stderr));
    let corpus = codec_corpus::Corpus::new().unwrap();
    eprintln!("Provisioning imageflow through codec-corpus");
    let imageflow = corpus.get("imageflow").unwrap();
    eprintln!("Provisioning jpeg-conformance through codec-corpus");
    let valid = corpus.get("jpeg-conformance/valid").unwrap();
    for name in ["Landscape_1.jpg", "Landscape_2.jpg"] {
        assert!(
            imageflow
                .join("test_inputs/orientation")
                .join(name)
                .is_file(),
            "missing {name}"
        );
    }
    assert!(
        valid.join("cmyk_logo.jpg").is_file(),
        "missing cmyk_logo.jpg"
    );
    let root = imageflow.parent().unwrap();
    assert_eq!(valid.parent().unwrap().parent().unwrap(), root);
    std::fs::write(root_file, root.as_os_str().as_encoded_bytes()).unwrap();

    let out = jxl_encoder::test_helpers::output_dir_for("jpeg-reencoding", "");
    let source = image::load_from_memory(include_bytes!("../tests/images/frymire-srgb.png"))
        .unwrap()
        .to_rgb8();
    for (name, w, h, sampling) in [
        ("test64_444", 64, 64, "1x1,1x1,1x1"),
        ("test128_444", 128, 128, "1x1,1x1,1x1"),
        ("test64_420", 64, 64, "2x2,1x1,1x1"),
        ("test128_420", 128, 128, "2x2,1x1,1x1"),
        ("test512_420", 512, 512, "2x2,1x1,1x1"),
        ("test_odd_420", 100, 75, "2x2,1x1,1x1"),
        ("test64_422", 64, 64, "2x1,1x1,1x1"),
        ("test128_422", 128, 128, "2x1,1x1,1x1"),
        ("test64_440", 64, 64, "1x2,1x1,1x1"),
        ("test128_440", 128, 128, "1x2,1x1,1x1"),
        ("test128_gray", 128, 128, "gray"),
    ] {
        let crop = image::imageops::crop_imm(&source, 0, 0, w, h).to_image();
        let mut ppm = format!("P6\n{w} {h}\n255\n").into_bytes();
        ppm.extend_from_slice(crop.as_raw());
        let input = out.join(format!("{name}.ppm"));
        preserve_or_write(&input, &ppm);
        let mut command = Command::new(&cjpeg);
        command.args(["-baseline", "-quality", "90"]);
        if sampling == "gray" {
            command.arg("-grayscale");
        } else {
            command.args(["-sample", sampling]);
        }
        let result = command.arg(input).output().unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        // Pin shape and sampling so fixture setup cannot weaken the test matrix.
        let parsed = jxl_encoder::jpeg::read_jpeg(&result.stdout, None, None).unwrap();
        assert_eq!((parsed.width, parsed.height), (w, h));
        let expected = match sampling {
            "2x2,1x1,1x1" => (2, 2),
            "2x1,1x1,1x1" => (2, 1),
            "1x2,1x1,1x1" => (1, 2),
            _ => (1, 1),
        };
        assert_eq!(
            (
                parsed.components[0].h_samp_factor,
                parsed.components[0].v_samp_factor
            ),
            expected
        );
        assert_eq!(
            parsed.components.len(),
            if sampling == "gray" { 1 } else { 3 }
        );
        preserve_or_write(&out.join(format!("{name}.jpg")), &result.stdout);
        eprintln!("Prepared {name}: {} bytes", result.stdout.len());
    }
    // Strip only APP/COM markers, retaining the original photo's entropy bytes.
    let photo = std::fs::read(imageflow.join("test_inputs/orientation/Landscape_2.jpg")).unwrap();
    let mut stripped = Vec::new();
    let mut cursor = 0;
    for span in zenjpeg::container::marker::iter(&photo) {
        if matches!(
            span.kind,
            zenjpeg::container::marker::MarkerKind::App(_)
                | zenjpeg::container::marker::MarkerKind::Com
        ) {
            stripped.extend_from_slice(&photo[cursor..span.offset]);
            cursor = span.offset + span.length;
        }
    }
    stripped.extend_from_slice(&photo[cursor..]);
    preserve_or_write(&out.join("test_real_420_stripped.jpg"), &stripped);
    eprintln!(
        "Prepared stripped Landscape_2 and corpus root {}",
        root.display()
    );
}
