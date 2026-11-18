//! CLI rate-control is a container-owning caller of raw prepared emission.
use super::*;
#[test]
#[ignore = "large resource probe; caller must provide the freshly pinned CLI binary"]
fn large_cli_rate_control_lf_level_signaling() {
    let root = directory();
    let cli = std::env::var_os("JXL_FRAME_LEVEL_CLI")
        .expect("caller must select the verified CLI runtime");
    let mut journal = std::fs::File::create(root.join("levels.tsv")).unwrap();
    let mut failures = Vec::new();
    for (width, height) in [(32_u32, 17_u32), (16384, 16384)] {
        let input = root.join(format!("source-{width}x{height}.ppm"));
        let mut source = std::io::BufWriter::new(std::fs::File::create(&input).unwrap());
        writeln!(source, "P6\n{width} {height}\n255").unwrap();
        let row = vec![128_u8; width as usize * 3];
        for _ in 0..height {
            source.write_all(&row).unwrap();
        }
        source.flush().unwrap();
        drop(source);
        for lf in [false, true] {
            let out = root.join(format!("{width}x{height}-lf{lf}"));
            std::fs::create_dir(&out).unwrap();
            let encoded = out.join("input.jxl");
            writeln!(journal, "{width}x{height} lf={lf} stage=CLI-encoding").unwrap();
            journal.flush().unwrap();
            let mut command = std::process::Command::new(&cli);
            command.arg(&input).arg(&encoded).args([
                "--rate-control",
                "--rc-iterations",
                "0",
                "--effort",
                "1",
                "--distance",
                "1",
                "--dct8-only",
                "--no-gaborish",
                "--epf",
                "0",
                "--no-pixel-domain-loss",
                "--quiet",
            ]);
            if lf {
                command.arg("--lf-frame");
            }
            apply_cli_policy(&mut command);
            command.env("LLVM_PROFILE_FILE", out.join("cli-%p-%m.profraw"));
            let result = command.output().unwrap();
            std::fs::write(out.join("encode.stdout"), result.stdout).unwrap();
            std::fs::write(out.join("encode.stderr"), result.stderr).unwrap();
            assert!(result.status.success(), "{}", out.display());
            let bytes = std::fs::read(&encoded).unwrap();
            let parsed = jxl_oxide::JxlImage::builder()
                .read(std::io::Cursor::new(&bytes))
                .unwrap();
            assert_eq!((parsed.width(), parsed.height()), (width, height));
            assert!(
                parsed.image_header().metadata.modular_16bit_buffers,
                "isolate actual frame area"
            );
            assert_eq!(parsed.num_loaded_frames(), if lf { 2 } else { 1 });
            let mut area = u64::from(width) * u64::from(height);
            if lf {
                let header = parsed.frame(0).unwrap().header();
                assert_eq!(format!("{:?}", header.frame_type), "LfFrame");
                assert_eq!(header.lf_level, 1);
                let divisor = 1u32 << (3 * header.lf_level);
                area += u64::from(header.width.div_ceil(divisor))
                    * u64::from(header.height.div_ceil(divisor));
            }
            let required = if area > (1 << 28) { 10 } else { 5 };
            writeln!(journal,"{width}x{height} lf={lf} observed_level={} required={required} actual_zero_duration_area={area}",level(&bytes)).unwrap();
            journal.flush().unwrap();
            if level(&bytes) != required {
                failures.push(format!(
                    "{width}x{height} lf={lf}: Level{} expected{required}",
                    level(&bytes)
                ));
            }
            let primary =
                decoded_frames::decode_current_frames(&bytes, "CLI rate control LF level");
            assert_eq!(primary.len(), 1);
            assert!(primary[0].1.is_empty());
            assert_eq!(primary[0].0.len(), width as usize * height as usize * 3);
            let native = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
                .arg(&encoded)
                .arg(out.join("native.png"))
                .args(["--num_threads=1", "--bits_per_sample=16"])
                .output()
                .unwrap();
            std::fs::write(out.join("native.stdout"), native.stdout).unwrap();
            std::fs::write(out.join("native.log"), native.stderr).unwrap();
            assert!(native.status.success());
            let mut reader = image::ImageReader::open(out.join("native.png")).unwrap();
            let mut limits = image::Limits::default();
            limits.max_alloc = Some(4u64 << 30);
            reader.limits(limits);
            let actual = reader.decode().unwrap().to_rgb16();
            assert_eq!(actual.dimensions(), (width, height));
            assert!(
                primary[0]
                    .0
                    .iter()
                    .zip(actual.as_raw())
                    .all(|(&a, &b)| (a.clamp(0.0, 1.0) - f32::from(b) / 65535.0).abs() <= 0.001)
            );
            writeln!(journal, "{width}x{height} lf={lf} stage=complete-render").unwrap();
            journal.flush().unwrap();
        }
    }
    assert!(
        failures.is_empty(),
        "{}; {}",
        failures.join("; "),
        root.display()
    );
}
