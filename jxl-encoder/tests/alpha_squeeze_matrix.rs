//! Alpha-band samples against independently encoded pinned libjxl controls.
#![cfg(feature = "__expert")]
use jxl_encoder::api::{EncoderMode, EncoderStrategy};
use jxl_encoder::{Buffering, LossyConfig, PixelLayout, ProgressiveMode};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    io::Write,
    path::{Path, PathBuf},
};
#[path = "support/decoded_extras.rs"]
mod decoded_extras;
#[path = "support/encoder_policies.rs"]
mod encoder_policies;
#[path = "support/pfm.rs"]
mod pfm;

fn digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|v| format!("{v:02x}"))
        .collect()
}
fn directory() -> PathBuf {
    let root = jxl_encoder::test_helpers::output_dir("alpha-squeeze-matrix");
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
#[derive(Clone)]
struct Cell {
    policy: String,
    strategy: EncoderStrategy,
    mode: EncoderMode,
    layout: PixelLayout,
    width: usize,
    height: usize,
    distance: f32,
    entry: &'static str,
    buffering: Buffering,
    threads: usize,
    associated: bool,
    progressive: ProgressiveMode,
    ans: Option<bool>,
    canonicalize: bool,
    effort: u8,
    progressive_dc: u8,
    bits: Option<u32>,
}
impl Cell {
    fn label(&self) -> String {
        format!(
            "{}-{:?}-{:?}-{}x{}-ad{}-{}-{:?}-t{}-assoc{}-{:?}-ans{:?}-canon{}-e{}-pdc{}-bits{:?}",
            self.policy,
            self.mode,
            self.layout,
            self.width,
            self.height,
            self.distance,
            self.entry,
            self.buffering,
            self.threads,
            self.associated,
            self.progressive,
            self.ans,
            self.canonicalize,
            self.effort,
            self.progressive_dc,
            self.bits
        )
    }
}
fn strict(width: usize, height: usize) -> Cell {
    Cell {
        policy: "strict".into(),
        strategy: EncoderStrategy::Libjxl,
        mode: EncoderMode::Reference,
        layout: PixelLayout::Rgba8,
        width,
        height,
        distance: 2.0,
        entry: "request",
        buffering: Buffering::FullBuffered,
        threads: 1,
        associated: false,
        progressive: ProgressiveMode::Single,
        ans: None,
        canonicalize: false,
        effort: 3,
        progressive_dc: 0,
        bits: None,
    }
}
fn input(cell: &Cell) -> (Vec<u8>, Vec<u16>, u16) {
    let fixture = image::load_from_memory(include_bytes!("images/frymire.png"))
        .unwrap()
        .to_rgb8();
    let float = cell.layout.is_f32() || cell.layout.is_f16();
    let max = if cell.layout.is_16bit() {
        ((1_u32 << cell.bits.unwrap_or(16)) - 1) as u16
    } else if float {
        65535
    } else {
        255
    };
    let mut pixels = Vec::with_capacity(cell.width * cell.height * cell.layout.bytes_per_pixel());
    let mut alpha = Vec::with_capacity(cell.width * cell.height);
    for y in 0..cell.height {
        for x in 0..cell.width {
            let rgb = fixture
                .get_pixel(x as u32 % fixture.width(), y as u32 % fixture.height())
                .0;
            let selector = (x * 17 + y * 29) % 256;
            let alpha_code = if float {
                [0, 32768, 65535][selector * 3 / 256]
            } else if max == 65535 {
                (selector * 257 + (x + y) % 257).min(65535) as u16
            } else if cell.layout.is_16bit() {
                ((selector * usize::from(max) + 127) / 255
                    + (x + y) % (usize::from(max) / 255).max(1))
                .min(usize::from(max)) as u16
            } else {
                selector as u16
            };
            alpha.push(alpha_code);
            let channels = if cell.layout.is_grayscale() { 1 } else { 3 };
            let mut color = rgb[..channels].to_vec();
            if cell.canonicalize {
                color.fill(rgb[0]);
            }
            if cell.associated {
                for value in &mut color {
                    *value = ((u32::from(*value) * u32::from(alpha_code) + u32::from(max) / 2)
                        / u32::from(max)) as u8;
                }
            }
            if cell.layout == PixelLayout::Bgra8 {
                color.reverse();
            }
            if cell.layout.is_f16() {
                for value in color {
                    let bits = [0x3000_u16, 0x3400, 0x3800][usize::from(value) * 3 / 256];
                    pixels.extend(bits.to_ne_bytes());
                }
                let bits = match alpha_code {
                    0 => 0_u16,
                    32768 => 0x3800,
                    _ => 0x3c00,
                };
                pixels.extend(bits.to_ne_bytes());
            } else if cell.layout.is_f32() {
                for value in color {
                    pixels.extend((f32::from(value) / 255.0).to_ne_bytes());
                }
                let value = match alpha_code {
                    0 => 0.0_f32,
                    32768 => 0.5,
                    _ => 1.0,
                };
                pixels.extend(value.to_ne_bytes());
            } else if cell.layout.is_16bit() {
                for value in color {
                    let value = ((u32::from(value) * u32::from(max) + 127) / 255) as u16;
                    pixels.extend(value.to_ne_bytes());
                }
                pixels.extend(alpha_code.to_ne_bytes());
            } else {
                pixels.extend(color);
                pixels.push(alpha_code as u8);
            }
        }
    }
    (pixels, alpha, max)
}
fn reference(
    root: &Path,
    cell: &Cell,
    alpha: &[u16],
    max: u16,
    cache: &mut BTreeMap<String, Vec<i64>>,
) -> Vec<i64> {
    let mut pam = format!(
        "P7\nWIDTH {}\nHEIGHT {}\nDEPTH 4\nMAXVAL {max}\nTUPLTYPE RGB_ALPHA\nENDHDR\n",
        cell.width, cell.height
    )
    .into_bytes();
    for &a in alpha {
        if max <= 255 {
            for value in [64_u32, 128, 192] {
                pam.push(((value * u32::from(max) + 127) / 255) as u8);
            }
            pam.push(a as u8);
        } else {
            for value in [64_u32, 128, 192] {
                let value = ((value * u32::from(max) + 127) / 255) as u16;
                pam.extend(value.to_be_bytes());
            }
            pam.extend(a.to_be_bytes());
        }
    }
    let key = format!("{}-{:08x}", digest(&pam), cell.distance.to_bits());
    if let Some(samples) = cache.get(&key) {
        return samples.clone();
    }
    let out = root.join("native").join(&key);
    std::fs::create_dir_all(&out).unwrap();
    std::fs::write(out.join("input.pam"), pam).unwrap();
    if cell.bits.is_some() {
        // Verify the independent PAM adapter's significant-bit interpretation
        // with lossless alpha before using its lossy quantization as an oracle.
        let control = std::process::Command::new(jxl_encoder::test_helpers::cjxl_path())
            .arg(out.join("input.pam"))
            .arg(out.join("source-control.jxl"))
            .args([
                "-d",
                "1",
                "-e",
                "3",
                "--responsive=1",
                "--alpha_distance=0",
                "--num_threads=1",
            ])
            .output()
            .unwrap();
        std::fs::write(out.join("source-control-encode.log"), &control.stderr).unwrap();
        assert!(control.status.success(), "native precision source encode");
        let control = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
            .arg(out.join("source-control.jxl"))
            .arg(out.join("source-control.pfm"))
            .args(["--output_extra_channels", "--num_threads=1"])
            .output()
            .unwrap();
        std::fs::write(out.join("source-control-decode.log"), &control.stderr).unwrap();
        assert!(
            control.status.success(),
            "native precision source rendering"
        );
        let control = pfm::read_pfm_samples(
            &out.join("source-control.pfm-ec1.pfm"),
            (cell.width, cell.height),
        );
        assert_eq!(control.len(), alpha.len());
        for (sample, &expected) in control.iter().zip(alpha) {
            assert!(sample.is_finite());
            assert_eq!(
                (sample * f32::from(max)).round() as i64,
                i64::from(expected),
                "native PAM source precision: {}",
                cell.label()
            );
        }
    }
    let result = std::process::Command::new(jxl_encoder::test_helpers::cjxl_path())
        .arg(out.join("input.pam"))
        .arg(out.join("reference.jxl"))
        .args(["-d", "1", "-e", "3", "--responsive=1", "--num_threads=1"])
        .arg(format!("--alpha_distance={}", cell.distance))
        .output()
        .unwrap();
    std::fs::write(out.join("encode.log"), &result.stderr).unwrap();
    assert!(
        result.status.success(),
        "native encode {}: {}",
        cell.label(),
        String::from_utf8_lossy(&result.stderr)
    );
    let bytes = std::fs::read(out.join("reference.jxl")).unwrap();
    std::fs::write(out.join("reference.sha256"), digest(&bytes)).unwrap();
    let result = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
        .arg(out.join("reference.jxl"))
        .arg(out.join("reference.pfm"))
        .args(["--output_extra_channels", "--num_threads=1"])
        .output()
        .unwrap();
    std::fs::write(out.join("decode.log"), &result.stderr).unwrap();
    assert!(result.status.success(), "native reference rendering");
    let samples = pfm::read_pfm_samples(
        &out.join("reference.pfm-ec1.pfm"),
        (cell.width, cell.height),
    );
    let samples: Vec<_> = samples
        .into_iter()
        .map(|v| (v * f32::from(max)).round() as i64)
        .collect();
    cache.insert(key, samples.clone());
    samples
}
fn check(root: &Path, cell: &Cell, cache: &mut BTreeMap<String, Vec<i64>>) -> String {
    let label = cell.label();
    let out = root.join(&label);
    std::fs::create_dir(&out).unwrap();
    let (pixels, alpha, max) = input(cell);
    std::fs::write(out.join("input.bin"), &pixels).unwrap();
    std::fs::write(out.join("source.sha256"), digest(&pixels)).unwrap();
    let expected = reference(root, cell, &alpha, max, cache);
    let mut config = LossyConfig::new(1.0)
        .with_effort(cell.effort)
        .with_threads(cell.threads)
        .with_mode(cell.mode)
        .with_strategy(cell.strategy.clone())
        .with_force_strategy(Some(0))
        .with_progressive(cell.progressive)
        .with_progressive_dc(cell.progressive_dc)
        .with_alpha_squeeze(true)
        .with_alpha_distance(Some(cell.distance))
        .with_buffering(cell.buffering);
    config = config.with_canonicalize_input(cell.canonicalize);
    if let Some(ans) = cell.ans {
        config = config.with_ans(ans);
    }
    let request = || {
        let mut request = config
            .encode_request(cell.width as u32, cell.height as u32, cell.layout)
            .with_premultiplied_alpha(cell.associated);
        if let Some(bits) = cell.bits {
            request = request.with_bits_per_sample(bits);
        }
        request
    };
    let bytes = match cell.entry {
        "request" => {
            let result = request().encode_with_stats(&pixels).unwrap();
            if cell.threads > 1 {
                assert_eq!(result.stats().threads_used(), cell.threads as u32);
            }
            result.data().unwrap().to_vec()
        }
        "strided" => {
            let row = cell.width * cell.layout.bytes_per_pixel();
            let stride = row + 3;
            let mut padded = vec![0xa5; stride * cell.height];
            for (source, dest) in pixels
                .chunks_exact(row)
                .zip(padded.chunks_exact_mut(stride))
            {
                dest[..row].copy_from_slice(source);
            }
            let bytes = request().with_row_stride(stride).encode(&padded).unwrap();
            assert_eq!(
                bytes,
                request().encode(&pixels).unwrap(),
                "packed/strided {label}"
            );
            bytes
        }
        "writer" => {
            let mut dest = Vec::new();
            request().encode_to(&pixels, &mut dest).unwrap();
            assert_eq!(
                dest,
                request().encode(&pixels).unwrap(),
                "writer/request {label}"
            );
            dest
        }
        "legacy" => {
            assert!(!cell.associated);
            assert!(
                cell.bits.is_none(),
                "precision override belongs to request/streaming"
            );
            let bytes = config
                .encode(&pixels, cell.width as u32, cell.height as u32, cell.layout)
                .unwrap();
            assert_eq!(
                bytes,
                request().encode(&pixels).unwrap(),
                "legacy/request {label}"
            );
            bytes
        }
        "stream" => {
            let row = cell.width * cell.layout.bytes_per_pixel();
            let mut encoder = config
                .encoder(cell.width as u32, cell.height as u32, cell.layout)
                .unwrap()
                .with_premultiplied_alpha(cell.associated);
            if let Some(bits) = cell.bits {
                encoder = encoder.with_bits_per_sample(bits);
            }
            for rows in pixels.chunks(row * 7) {
                encoder.push_rows(rows, (rows.len() / row) as u32).unwrap();
            }
            let bytes = encoder.finish().unwrap();
            assert_eq!(
                bytes,
                request().encode(&pixels).unwrap(),
                "stream/request {label}"
            );
            bytes
        }
        #[cfg(all(feature = "__pre_quantized", feature = "__internals"))]
        "precomputed" | "prequantized" => {
            use jxl_encoder::__pre_quantized::{DistanceParams, EncoderPrecomputed, VarDctEncoder};
            assert_eq!(cell.policy, "engine-default");
            assert_eq!(cell.layout, PixelLayout::Rgba8);
            // These exported engine entries have no named-strategy setter.
            // Exercise their actual default engine policy in both encoder modes.
            let mut encoder = VarDctEncoder::new(1.0);
            encoder.profile = config.resolved_profile();
            encoder.encoder_mode = cell.mode;
            encoder.effort = cell.effort;
            encoder.use_ans = config.ans();
            encoder.force_strategy = Some(0);
            encoder.enable_patches = false;
            encoder.alpha_squeeze = true;
            encoder.alpha_distance = Some(cell.distance);
            encoder.progressive = cell.progressive;
            let linear: Vec<_> = pixels
                .as_chunks::<4>()
                .0
                .iter()
                .flat_map(|pixel| pixel[..3].iter().map(|v| f32::from(*v) / 255.0))
                .collect();
            let precomputed = EncoderPrecomputed::compute(
                cell.width,
                cell.height,
                &linear,
                encoder.distance,
                encoder.cfl_enabled,
                encoder.ac_strategy_enabled,
                encoder.pixel_domain_loss,
                encoder.enable_noise,
                encoder.enable_denoise,
                encoder.enable_gaborish,
                encoder.force_strategy,
                &encoder.profile,
                encoder.color_encoding.as_ref(),
            )
            .unwrap();
            let qf = vec![1; precomputed.xsize_blocks * precomputed.ysize_blocks];
            let alpha: Vec<_> = alpha.iter().map(|v| *v as u8).collect();
            let extras = [jxl_encoder::api::ExtraChannel::from_alpha_buf(
                &alpha,
                cell.associated,
            )];
            if cell.entry == "precomputed" {
                encoder
                    .encode_from_precomputed_with_extras(&precomputed, &qf, &extras)
                    .unwrap()
            } else {
                let params =
                    DistanceParams::compute_for_profile(encoder.distance, &encoder.profile);
                let mut transform_qf = qf.clone();
                let transform = encoder
                    .transform_and_quantize_for_test(&precomputed, &mut transform_qf, &params)
                    .unwrap();
                encoder
                    .encode_from_pre_quantized_ac_with_extras(
                        &precomputed,
                        &qf,
                        &transform.quant_dc,
                        &transform.quant_ac,
                        &transform.nzeros,
                        &transform.raw_nzeros,
                        &extras,
                    )
                    .unwrap()
            }
        }
        _ => panic!("unknown entry"),
    };
    if cell.canonicalize {
        assert_eq!(cell.layout, PixelLayout::Rgba8);
        let reduced: Vec<u8> = pixels
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(|pixel| [pixel[0], pixel[3]])
            .collect();
        let explicit = config
            .clone()
            .with_canonicalize_input(false)
            .encode(
                &reduced,
                cell.width as u32,
                cell.height as u32,
                PixelLayout::GrayAlpha8,
            )
            .unwrap();
        assert_eq!(
            bytes, explicit,
            "canonicalization must actually select GrayAlpha8: {label}"
        );
    }
    std::fs::write(out.join("input.jxl"), &bytes).unwrap();
    let hash = digest(&bytes);
    std::fs::write(out.join("input.sha256"), &hash).unwrap();
    let primary =
        std::panic::catch_unwind(|| decoded_extras::decode_current_planes(&bytes, &label));
    let result = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
        .arg(out.join("input.jxl"))
        .arg(out.join("output.pfm"))
        .args(["--output_extra_channels", "--num_threads=1"])
        .output()
        .unwrap();
    std::fs::write(out.join("decode.log"), &result.stderr).unwrap();
    assert!(result.status.success(), "pinned rendering {label}");
    let (_, planes) = primary.expect("primary decoder rendering");
    assert_eq!(planes.len(), 1);
    let reference =
        pfm::read_pfm_samples(&out.join("output.pfm-ec1.pfm"), (cell.width, cell.height));
    for (decoder, samples) in [("jxl-rs", &planes[0]), ("djxl", &reference)] {
        assert_eq!(samples.len(), expected.len());
        for (i, (value, expected)) in samples.iter().zip(&expected).enumerate() {
            assert!(value.is_finite());
            assert_eq!(
                (value * f32::from(max)).round() as i64,
                *expected,
                "{label} {decoder} sample {i}"
            );
        }
    }
    hash
}
fn run(cells: Vec<Cell>) {
    let root = directory();
    let mut cache = BTreeMap::new();
    let mut journal = std::fs::File::create(root.join("completed.tsv")).unwrap();
    writeln!(journal, "label\tstatus\tstream_sha256_or_error").unwrap();
    let mut failures = Vec::new();
    for cell in cells.into_iter().flat_map(|cell| {
        [
            ProgressiveMode::Single,
            ProgressiveMode::QuantizedAcFullAc,
            ProgressiveMode::DcVlfLfAc,
        ]
        .map(|progressive| Cell {
            progressive,
            ..cell.clone()
        })
    }) {
        let label = cell.label();
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            check(&root, &cell, &mut cache)
        })) {
            Ok(hash) => writeln!(journal, "{label}\tPASS\t{hash}").unwrap(),
            Err(e) => {
                let error = e
                    .downcast_ref::<String>()
                    .map(String::as_str)
                    .or_else(|| e.downcast_ref::<&str>().copied())
                    .unwrap_or("panic")
                    .replace(['\n', '\t'], " ");
                writeln!(journal, "{label}\tFAIL\t{error}").unwrap();
                failures.push(label);
            }
        }
        journal.flush().unwrap();
    }
    assert!(
        failures.is_empty(),
        "failed alpha cells {failures:?}; {}",
        root.display()
    );
}
#[test]
fn alpha_layouts_match_native_quantized_samples() {
    let mut cells = Vec::new();
    for layout in [
        PixelLayout::Rgba8,
        PixelLayout::Bgra8,
        PixelLayout::GrayAlpha8,
        PixelLayout::Rgba16,
        PixelLayout::GrayAlpha16,
        PixelLayout::RgbaLinearF16,
        PixelLayout::GrayAlphaLinearF16,
        PixelLayout::RgbaLinearF32,
        PixelLayout::GrayAlphaLinearF32,
        PixelLayout::RgbaPqF32,
        PixelLayout::RgbaHlgF32,
        PixelLayout::RgbaBt709F32,
    ] {
        for (width, height) in [(257, 257), (1025, 1024)] {
            cells.push(Cell {
                layout,
                ..strict(width, height)
            });
        }
    }
    run(cells);
}
#[test]
fn policies_and_public_entries_keep_alpha_band_samples() {
    let mut cells = Vec::new();
    for (policy, strategy, mode) in encoder_policies::policies() {
        for entry in ["request", "strided", "writer", "legacy", "stream"] {
            for (width, height) in [(128, 64), (1025, 1024)] {
                cells.push(Cell {
                    policy: policy.clone(),
                    strategy: strategy.clone(),
                    mode,
                    entry,
                    ..strict(width, height)
                });
            }
        }
    }
    run(cells);
}
#[test]
fn buffering_and_lf_groups_keep_native_alpha_samples() {
    let mut cells = Vec::new();
    for buffering in [
        Buffering::Auto,
        Buffering::FullBuffered,
        Buffering::Threshold2048,
        Buffering::BufferedOutput,
        Buffering::FullStreaming,
    ] {
        for distance in [2.0, 25.0] {
            cells.push(Cell {
                buffering,
                distance,
                ..strict(2049, 2048)
            });
        }
    }
    run(cells);
}

#[cfg(all(feature = "__pre_quantized", feature = "__internals"))]
#[test]
fn precomputed_and_prequantized_alpha_entries_keep_native_samples() {
    let mut cells = Vec::new();
    for mode in [EncoderMode::Reference, EncoderMode::Experimental] {
        for entry in ["precomputed", "prequantized"] {
            for (width, height) in [(128, 64), (257, 257), (1025, 1024), (2049, 2048)] {
                cells.push(Cell {
                    policy: "engine-default".into(),
                    mode,
                    entry,
                    ..strict(width, height)
                });
            }
        }
    }
    run(cells);
}

#[test]
fn alpha_shift_brackets_and_association_keep_native_samples() {
    let mut cells = Vec::new();
    for distance in [0.125, 1.0, 2.0, 10.0, 25.0] {
        for (width, height) in [
            (2, 2),
            (1, 17),
            (17, 1),
            (256, 257),
            (257, 256),
            (1024, 1025),
        ] {
            cells.push(Cell {
                distance,
                ..strict(width, height)
            });
        }
    }
    for layout in [PixelLayout::Rgba8, PixelLayout::GrayAlpha16] {
        for entry in ["request", "strided", "writer", "stream"] {
            cells.push(Cell {
                layout,
                entry,
                associated: true,
                ..strict(1025, 1024)
            });
        }
    }
    run(cells);
}

#[cfg(feature = "parallel")]
#[test]
fn alpha_band_writers_use_two_requested_workers() {
    let mut cells = Vec::new();
    for mode in [EncoderMode::Reference, EncoderMode::Experimental] {
        cells.push(Cell {
            policy: "zen-workers".into(),
            strategy: EncoderStrategy::Zenjxl,
            mode,
            threads: 2,
            ..strict(1025, 1024)
        });
    }
    run(cells);
}

#[test]
fn raw_lossless_alpha_stays_in_the_final_modular_pass() {
    let mut cells = Vec::new();
    for layout in [
        PixelLayout::Rgba8,
        PixelLayout::Bgra8,
        PixelLayout::GrayAlpha8,
        PixelLayout::Rgba16,
        PixelLayout::GrayAlpha16,
    ] {
        for entry in ["request", "stream"] {
            for (width, height) in [(128, 64), (257, 257), (1025, 1024)] {
                cells.push(Cell {
                    layout,
                    entry,
                    distance: 0.0,
                    ..strict(width, height)
                });
            }
        }
    }
    run(cells);
}

#[test]
fn progressive_alpha_preserves_explicit_entropy_selection() {
    let mut cells = Vec::new();
    for ans in [false, true] {
        for (policy, strategy, mode) in [
            (
                "strict-entropy",
                EncoderStrategy::Libjxl,
                EncoderMode::Reference,
            ),
            (
                "zen-entropy",
                EncoderStrategy::Zenjxl,
                EncoderMode::Experimental,
            ),
        ] {
            for entry in ["request", "writer", "stream"] {
                for (width, height) in [(257, 257), (1025, 1024)] {
                    cells.push(Cell {
                        ans: Some(ans),
                        policy: policy.into(),
                        strategy: strategy.clone(),
                        mode,
                        entry,
                        ..strict(width, height)
                    });
                }
            }
        }
    }
    run(cells);
}

#[test]
fn canonicalized_gray_alpha_traverses_each_progressive_bracket() {
    let mut cells = Vec::new();
    for entry in ["request", "strided", "writer", "legacy"] {
        for (width, height) in [(128, 64), (257, 257), (1025, 1024)] {
            cells.push(Cell {
                canonicalize: true,
                entry,
                ..strict(width, height)
            });
        }
    }
    run(cells);
}

#[test]
fn progressive_alpha_effort_or_lf_boundaries_keep_native_bands() {
    let mut cells = Vec::new();
    for effort in [1, 2, 4, 7, 8, 9] {
        for (policy, strategy, mode) in [
            (
                "strict-effort",
                EncoderStrategy::Libjxl,
                EncoderMode::Reference,
            ),
            (
                "zen-effort",
                EncoderStrategy::Zenjxl,
                EncoderMode::Experimental,
            ),
        ] {
            for (width, height) in [(1025, 17), (17, 1025)] {
                cells.push(Cell {
                    effort,
                    policy: policy.into(),
                    strategy: strategy.clone(),
                    mode,
                    ..strict(width, height)
                });
            }
        }
    }
    for progressive_dc in [1, 2] {
        for layout in [PixelLayout::Rgba8, PixelLayout::GrayAlpha16] {
            for entry in ["request", "stream"] {
                for (width, height) in [(128, 64), (1025, 17)] {
                    for (policy, strategy, mode) in [
                        ("strict-lf", EncoderStrategy::Libjxl, EncoderMode::Reference),
                        ("zen-lf", EncoderStrategy::Zenjxl, EncoderMode::Experimental),
                    ] {
                        cells.push(Cell {
                            progressive_dc,
                            layout,
                            entry,
                            policy: policy.into(),
                            strategy,
                            mode,
                            ..strict(width, height)
                        });
                    }
                }
            }
        }
    }
    run(cells);
}

#[test]
fn progressive_alpha_reduced_integer_precision_keeps_native_codes() {
    let mut cells = Vec::new();
    for bits in 1..=16 {
        for layout in [PixelLayout::Rgba16, PixelLayout::GrayAlpha16] {
            for entry in ["request", "strided", "writer", "stream"] {
                for (width, height) in [(128, 64), (1025, 17)] {
                    cells.push(Cell {
                        bits: Some(bits),
                        layout,
                        entry,
                        ..strict(width, height)
                    });
                }
            }
        }
    }
    run(cells);
}

#[test]
fn elongated_alpha_paths_and_layouts_keep_native_bands() {
    let mut cells = Vec::new();
    let shapes = [(131585, 17), (17, 131585)];
    for (i, layout) in [
        PixelLayout::Rgba8,
        PixelLayout::Bgra8,
        PixelLayout::GrayAlpha8,
        PixelLayout::Rgba16,
        PixelLayout::GrayAlpha16,
        PixelLayout::RgbaLinearF16,
        PixelLayout::GrayAlphaLinearF16,
        PixelLayout::RgbaLinearF32,
        PixelLayout::GrayAlphaLinearF32,
        PixelLayout::RgbaPqF32,
        PixelLayout::RgbaHlgF32,
        PixelLayout::RgbaBt709F32,
    ]
    .into_iter()
    .enumerate()
    {
        let (width, height) = shapes[i % 2];
        cells.push(Cell {
            layout,
            ..strict(width, height)
        });
    }
    for (i, (policy, strategy, mode)) in encoder_policies::policies().into_iter().enumerate() {
        let (width, height) = shapes[i % 2];
        cells.push(Cell {
            policy,
            strategy,
            mode,
            ..strict(width, height)
        });
    }
    for (i, entry) in ["request", "strided", "writer", "legacy", "stream"]
        .into_iter()
        .enumerate()
    {
        for mode in [EncoderMode::Reference, EncoderMode::Experimental] {
            // The first layout cell already selects this exact request input.
            if entry == "request" && mode == EncoderMode::Reference {
                continue;
            }
            let (width, height) = shapes[i % 2];
            cells.push(Cell {
                entry,
                mode,
                ..strict(width, height)
            });
        }
    }
    for (i, layout) in [PixelLayout::Rgba8, PixelLayout::Rgba16]
        .into_iter()
        .enumerate()
    {
        let (width, height) = shapes[i % 2];
        cells.push(Cell {
            layout,
            associated: true,
            ..strict(width, height)
        });
    }
    #[cfg(all(feature = "__pre_quantized", feature = "__internals"))]
    for (i, entry) in ["precomputed", "prequantized"].into_iter().enumerate() {
        for mode in [EncoderMode::Reference, EncoderMode::Experimental] {
            let (width, height) = shapes[i % 2];
            cells.push(Cell {
                policy: "engine-default".into(),
                entry,
                mode,
                ..strict(width, height)
            });
        }
    }
    let mut identities = std::collections::BTreeSet::new();
    assert!(
        cells.iter().all(|cell| identities.insert(cell.label())),
        "elongated fixture identities must be unique before encoding"
    );
    run(cells);
}

#[cfg(feature = "parallel")]
#[test]
fn elongated_alpha_two_worker_bands_keep_native_samples() {
    run([(131585, 17), (17, 131585)]
        .into_iter()
        .map(|(width, height)| Cell {
            policy: "wide-workers".into(),
            mode: EncoderMode::Experimental,
            threads: 2,
            ..strict(width, height)
        })
        .collect());
}
