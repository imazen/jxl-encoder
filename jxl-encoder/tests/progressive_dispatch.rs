//! An explicit progressive request must reach the emitted frame pass schedule.
#![cfg(feature = "__expert")]
use jxl_encoder::{LossyConfig, PixelLayout, ProgressiveMode};
use std::{
    io::Write,
    path::{Path, PathBuf},
};
#[path = "support/decoded_extras.rs"]
mod decoded_extras;
#[path = "support/encoder_policies.rs"]
mod encoder_policies;
#[path = "support/pfm.rs"]
mod pfm;
fn directory() -> PathBuf {
    let root = jxl_encoder::test_helpers::output_dir("progressive-dispatch");
    std::fs::create_dir_all(&root).unwrap();
    (0..)
        .find_map(|i| {
            let p = root.join(format!("run-{i}"));
            match std::fs::create_dir(&p) {
                Ok(()) => Some(p),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => None,
                Err(e) => panic!("{e}"),
            }
        })
        .unwrap()
}
fn encode(
    config: &LossyConfig,
    pixels: &[u8],
    width: usize,
    height: usize,
    entry: &str,
) -> Vec<u8> {
    match entry {
        "legacy" => config
            .encode(pixels, width as u32, height as u32, PixelLayout::Rgb8)
            .unwrap(),
        "request" => config
            .encode_request(width as u32, height as u32, PixelLayout::Rgb8)
            .encode(pixels)
            .unwrap(),
        "writer" => {
            let mut out = Vec::new();
            config
                .encode_request(width as u32, height as u32, PixelLayout::Rgb8)
                .encode_to(pixels, &mut out)
                .unwrap();
            out
        }
        "strided" => {
            let stride = width * 3 + 11;
            let mut padded = vec![0xAB; stride * height];
            for (s, d) in pixels.chunks(width * 3).zip(padded.chunks_mut(stride)) {
                d[..s.len()].copy_from_slice(s);
            }
            config
                .encode_request(width as u32, height as u32, PixelLayout::Rgb8)
                .with_row_stride(stride)
                .encode(&padded)
                .unwrap()
        }
        "stream" => {
            let mut encoder = config
                .encoder(width as u32, height as u32, PixelLayout::Rgb8)
                .unwrap();
            for rows in pixels.chunks(width * 3 * 7) {
                encoder
                    .push_rows(rows, (rows.len() / (width * 3)) as u32)
                    .unwrap();
            }
            encoder.finish().unwrap()
        }
        _ => panic!("unknown entry"),
    }
}
fn render(out: &Path, label: &str, bytes: &[u8], dims: (usize, usize)) -> [Vec<u32>; 2] {
    let input = out.join(format!("{label}.jxl"));
    let output = out.join(format!("{label}.pfm"));
    std::fs::write(&input, bytes).unwrap();
    let (primary, extras) = decoded_extras::decode_current_planes(bytes, label);
    assert!(extras.is_empty());
    let result = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
        .arg(&input)
        .arg(&output)
        .arg("--num_threads=1")
        .output()
        .unwrap();
    std::fs::write(out.join(format!("{label}.decode.log")), &result.stderr).unwrap();
    assert!(result.status.success(), "{}", input.display());
    let native = pfm::read_pfm_samples(&output, dims);
    assert_eq!(primary.len(), dims.0 * dims.1 * 3);
    assert_eq!(native.len(), primary.len());
    assert!(primary.iter().chain(&native).all(|v| v.is_finite()));
    [primary, native].map(|v| v.into_iter().map(f32::to_bits).collect())
}
#[test]
fn requested_progressive_schedule_reaches_low_effort_frames() {
    let root = directory();
    let fixture = image::load_from_memory(include_bytes!("images/frymire.png"))
        .unwrap()
        .to_rgb8();
    let mut journal = std::fs::File::create(root.join("completed.tsv")).unwrap();
    writeln!(journal, "case\tmode\texpected_passes\tobserved_passes").unwrap();
    let mut failures = Vec::new();
    for (policy, strategy, mode) in encoder_policies::policies() {
        for (width, height) in [(64, 64), (259, 133)] {
            let pixels: Vec<u8> = (0..height)
                .flat_map(|y| {
                    let fixture = &fixture;
                    (0..width).flat_map(move |x| {
                        fixture
                            .get_pixel(x as u32 % fixture.width(), y as u32 % fixture.height())
                            .0
                    })
                })
                .collect();
            for effort in [1, 2] {
                for entry in ["legacy", "request", "writer", "strided", "stream"] {
                    let label = format!("{policy}-{mode:?}-{width}x{height}-e{effort}-{entry}");
                    let out = root.join(&label);
                    std::fs::create_dir(&out).unwrap();
                    std::fs::write(out.join("input.rgb8"), &pixels).unwrap();
                    let base = LossyConfig::new(1.0)
                        .with_effort(effort)
                        .with_threads(1)
                        .with_strategy(strategy.clone())
                        .with_mode(mode)
                        .with_force_strategy(Some(3));
                    let mut baseline = None;
                    for (progressive, wanted) in [
                        (ProgressiveMode::Single, 1),
                        (ProgressiveMode::QuantizedAcFullAc, 2),
                        (ProgressiveMode::DcVlfLfAc, 3),
                    ] {
                        let bytes = encode(
                            &base.clone().with_progressive(progressive),
                            &pixels,
                            width,
                            height,
                            entry,
                        );
                        let image = jxl_oxide::JxlImage::builder()
                            .read(std::io::Cursor::new(&bytes))
                            .unwrap();
                        let count = image.frame(0).unwrap().header().passes.num_passes;
                        let actual =
                            render(&out, &format!("{progressive:?}"), &bytes, (width, height));
                        if let Some(expected) = &baseline {
                            assert_eq!(&actual, expected, "{label} {progressive:?}");
                        } else {
                            baseline = Some(actual);
                        }
                        writeln!(journal, "{label}\t{progressive:?}\t{wanted}\t{count}").unwrap();
                        journal.flush().unwrap();
                        if count != wanted {
                            failures.push(format!("{label} {progressive:?}: {count} != {wanted}"));
                        }
                    }
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{}\n{}",
        root.display(),
        failures.join("\n")
    );
}
