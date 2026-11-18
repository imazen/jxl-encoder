//! Observe actual palette winners and compare against a declaration-only control.
use super::channel::Channel;
use super::contract_decoded_frames as decoded_frames;
use crate::{LosslessConfig, PixelLayout, RctType};
use std::io::Write;
use std::sync::Mutex;
#[derive(Clone)]
struct Record {
    width: usize,
    height: usize,
    nb_deltas: usize,
    predictor: u8,
    quantized: Vec<[i32; 3]>,
    indices: Vec<i32>,
}
struct Capture {
    force: Option<u8>,
    records: Vec<Record>,
}
static CAPTURE: Mutex<Option<Capture>> = Mutex::new(None);
fn with_capture<R>(f: impl FnOnce(&mut Option<Capture>) -> R) -> R {
    f(&mut CAPTURE.lock().unwrap())
}
pub(super) fn record(
    quantized: &[[i32; 3]],
    index: &Channel,
    width: usize,
    height: usize,
    nb_deltas: usize,
    predictor: u8,
) -> u8 {
    with_capture(|capture| {
        let Some(capture) = capture else {
            return predictor;
        };
        let declared = if predictor == 0 {
            0
        } else {
            capture.force.unwrap_or(predictor)
        };
        let indices = (0..height)
            .flat_map(|y| (0..width).map(move |x| index.get(x, y)))
            .collect();
        capture.records.push(Record {
            width,
            height,
            nb_deltas,
            predictor: declared,
            quantized: quantized.to_vec(),
            indices,
        });
        declared
    })
}
fn distinguishing_delta(record: &Record) -> Option<(usize, usize, usize, i32, i32, i32)> {
    for y in 1..record.height {
        for x in 1..record.width {
            let index = record.indices[y * record.width + x];
            if index >= record.nb_deltas as i32 {
                continue;
            }
            for c in 0..3 {
                let top = record.quantized[(y - 1) * record.width + x][c];
                let left = record.quantized[y * record.width + x - 1][c];
                let northwest = record.quantized[(y - 1) * record.width + x - 1][c];
                let gradient = (i64::from(top) + i64::from(left) - i64::from(northwest))
                    .clamp(i64::from(top.min(left)), i64::from(top.max(left)))
                    as i32;
                let select = if top.abs_diff(northwest) < left.abs_diff(northwest) {
                    left
                } else {
                    top
                };
                if gradient != select {
                    return Some((x, y, c, index, gradient, select));
                }
            }
        }
    }
    None
}
pub(super) type Policy = (String, crate::api::EncoderStrategy, crate::api::EncoderMode);

pub(super) fn policies() -> Vec<Policy> {
    use crate::api::{EncoderImprovementsCustom, EncoderMode, EncoderStrategy};
    let mut policies = vec![
        (
            "libjxl".into(),
            EncoderStrategy::Libjxl,
            EncoderMode::Reference,
        ),
        (
            "lean".into(),
            EncoderStrategy::LeanFaster,
            EncoderMode::Reference,
        ),
        (
            "zen".into(),
            EncoderStrategy::Zenjxl,
            EncoderMode::Reference,
        ),
        (
            "aggressive".into(),
            EncoderStrategy::Aggressive,
            EncoderMode::Reference,
        ),
        (
            "experimental".into(),
            EncoderStrategy::Zenjxl,
            EncoderMode::Experimental,
        ),
    ];
    for compact in [false, true] {
        for smoothing in [false, true] {
            policies.push((
                format!("custom-{compact}-{smoothing}"),
                EncoderStrategy::Custom(Box::new(EncoderImprovementsCustom {
                    header_all_default_fast_paths: compact,
                    dc_adaptive_smoothing: smoothing,
                    ..Default::default()
                })),
                EncoderMode::Reference,
            ));
        }
    }
    let experimental: Vec<_> = policies
        .iter()
        .filter(|(_, _, m)| *m == EncoderMode::Reference)
        .map(|(n, s, _)| {
            (
                format!("{n}-experimental"),
                s.clone(),
                EncoderMode::Experimental,
            )
        })
        .collect();
    policies.extend(experimental);
    policies
}

fn encode(
    pixels: &[u8],
    width: u32,
    height: u32,
    force: Option<u8>,
    policy: Option<&Policy>,
    use_ans: bool,
    route: &str,
) -> (Vec<u8>, Record) {
    let mut config = LosslessConfig::new()
        .with_effort(7)
        .with_threads(1)
        .with_patches(false)
        .with_squeeze(false)
        .with_tree_learning(false)
        .with_lz77(false)
        .with_ans(use_ans)
        .with_modular_predictor(Some(0))
        .with_force_rct(Some(RctType::NONE))
        .with_modular_palette_colors(Some(16))
        .with_lossy_palette(true)
        .with_strategy(crate::api::EncoderStrategy::Libjxl);
    if let Some((_, strategy, mode)) = policy {
        config = config.with_strategy(strategy.clone()).with_mode(*mode);
    }
    with_capture(|state| {
        *state = Some(Capture {
            force,
            records: Vec::new(),
        })
    });
    let bytes = match route {
        "streaming" => {
            let mut encoder = config.encoder(width, height, PixelLayout::Rgb8).unwrap();
            for rows in pixels.chunks(width as usize * 3 * 7) {
                encoder
                    .push_rows(rows, (rows.len() / (width as usize * 3)) as u32)
                    .unwrap();
            }
            encoder.finish().unwrap()
        }
        "legacy" => config
            .encode(pixels, width, height, PixelLayout::Rgb8)
            .unwrap(),
        "request" => config
            .encode_request(width, height, PixelLayout::Rgb8)
            .encode(pixels)
            .unwrap(),
        "strided" => {
            let stride = width as usize * 3 + 7;
            let mut padded = vec![0xd1; stride * height as usize];
            for (row, source) in padded
                .chunks_exact_mut(stride)
                .zip(pixels.chunks_exact(width as usize * 3))
            {
                row[..source.len()].copy_from_slice(source);
            }
            config
                .encode_request(width, height, PixelLayout::Rgb8)
                .with_row_stride(stride)
                .encode(&padded)
                .unwrap()
        }
        "writer" => {
            let mut bytes = Vec::new();
            config
                .encode_request(width, height, PixelLayout::Rgb8)
                .encode_to(pixels, &mut bytes)
                .unwrap();
            bytes
        }
        "animation" => config
            .encode_animation(
                width,
                height,
                PixelLayout::Rgb8,
                &crate::AnimationParams::default(),
                &[crate::AnimationFrame::new(pixels, 10)],
            )
            .unwrap(),
        "animation-request" => config
            .encode_request(width, height, PixelLayout::Rgb8)
            .encode_animation(
                &crate::AnimationParams::default(),
                &[crate::AnimationFrame::new(pixels, 10)],
            )
            .unwrap(),
        _ => panic!("unknown explicit route"),
    };
    let mut capture = with_capture(|state| state.take().unwrap());
    assert_eq!(
        capture.records.len(),
        1,
        "one actual global palette application"
    );
    (bytes, capture.records.remove(0))
}
fn decoded_codes(bytes: &[u8], out: &std::path::Path) -> (Vec<u8>, Vec<u8>) {
    let path = out.join("input.jxl");
    std::fs::write(&path, bytes).unwrap();
    let primary = decoded_frames::decode_current_frames(bytes, "selected delta palette");
    assert_eq!(primary.len(), 1);
    assert!(primary[0].1.is_empty());
    let primary = primary[0]
        .0
        .iter()
        .map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect();
    let native = std::process::Command::new(crate::test_helpers::djxl_path())
        .arg(&path)
        .arg(out.join("native.png"))
        .args(["--num_threads=1", "--bits_per_sample=8"])
        .output()
        .unwrap();
    std::fs::write(
        out.join("native.log"),
        [&native.stdout[..], &native.stderr[..]].concat(),
    )
    .unwrap();
    assert!(
        native.status.success(),
        "{}",
        String::from_utf8_lossy(&native.stderr)
    );
    let image = image::open(out.join("native.png")).unwrap().to_rgb8();
    (primary, image.into_raw())
}
#[test]
#[ignore = "conditional winning-delta investigation; explicit caller required"]
fn selected_delta_palette_decodes_its_actual_quantized_values() {
    #[cfg(feature = "parallel")]
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(run);
    #[cfg(not(feature = "parallel"))]
    run();
}
fn run() {
    let root = crate::test_helpers::output_dir("delta-palette-contract");
    std::fs::create_dir_all(&root).unwrap();
    let root = (0..)
        .find_map(|i| {
            let path = root.join(format!("run-{i}"));
            match std::fs::create_dir(&path) {
                Ok(()) => Some(path),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => None,
                Err(e) => panic!("{e}"),
            }
        })
        .unwrap();
    let mut journal = std::fs::File::create(root.join("selection.tsv")).unwrap();
    let mut failures = Vec::new();
    for (width, height) in [(32u32, 32u32), (259, 17)] {
        let pixels: Vec<u8> = (0..height)
            .flat_map(|y| {
                (0..width).flat_map(move |x| {
                    let (xx, yy) = ((x % 32) as i32, (y % 32) as i32);
                    if xx == 0 {
                        [128, 120, 136]
                    } else {
                        [
                            128 + 4 * xx - 4 * yy,
                            120 + 4 * xx - 3 * yy,
                            136 + 3 * xx - 4 * yy,
                        ]
                    }
                    .map(|v| u8::try_from(v).unwrap())
                })
            })
            .collect();
        let (ordinary, record) = encode(&pixels, width, height, None, None, false, "legacy");
        let reached = distinguishing_delta(&record);
        writeln!(
            journal,
            "{width}x{height}\tdeclared={}\tdeltas={}\twitness={reached:?}",
            record.predictor, record.nb_deltas
        )
        .unwrap();
        journal.flush().unwrap();
        assert!(
            reached.is_some(),
            "a distinguishing delta must actually win"
        );
        let target: Vec<u8> = record
            .quantized
            .iter()
            .flatten()
            .map(|v| (*v).clamp(0, 255) as u8)
            .collect();
        let (control, control_record) =
            encode(&pixels, width, height, Some(5), None, false, "legacy");
        assert_eq!(control_record.predictor, 5);
        assert_eq!(record.quantized, control_record.quantized);
        assert_eq!(record.indices, control_record.indices);
        assert_eq!(ordinary.len(), control.len());
        let differing_bits: u32 = ordinary
            .iter()
            .zip(&control)
            .map(|(a, b)| (a ^ b).count_ones())
            .sum();
        assert_eq!(
            differing_bits,
            if record.predictor == 5 { 0 } else { 1 },
            "only the declared predictor changes"
        );
        for (name, bytes, expected) in [
            ("ordinary", &ordinary, false),
            ("gradient-control", &control, true),
        ] {
            let out = root.join(format!("{width}x{height}-{name}"));
            std::fs::create_dir(&out).unwrap();
            let (primary, native) = decoded_codes(bytes, &out);
            let primary_exact = primary == target;
            let native_exact = native == target;
            writeln!(journal,"{width}x{height}\t{name}\tprimary_target_exact={primary_exact}\tnative_target_exact={native_exact}\tdiffering_bits={differing_bits}").unwrap();
            journal.flush().unwrap();
            if expected {
                assert!(
                    primary_exact && native_exact,
                    "declaration-only native control"
                );
            } else if !primary_exact || !native_exact {
                failures.push(format!(
                    "{width}x{height}: primary={primary_exact},native={native_exact}"
                ));
            }
        }
        let (streaming, stream_record) =
            encode(&pixels, width, height, None, None, false, "streaming");
        assert_eq!(streaming, ordinary);
        assert_eq!(stream_record.indices, record.indices);
    }
    assert!(
        failures.is_empty(),
        "{}; {}",
        failures.join("; "),
        root.display()
    );
}

#[test]
#[ignore = "explicit full-decoder caller matrix"]
fn delta_palette_all_offered_policies_and_public_callers() {
    let root = crate::test_helpers::output_dir("delta-palette-callers");
    std::fs::create_dir_all(&root).unwrap();
    let root = (0..)
        .find_map(|i| {
            let p = root.join(format!("run-{i}"));
            match std::fs::create_dir(&p) {
                Ok(()) => Some(p),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => None,
                Err(e) => panic!("{e}"),
            }
        })
        .unwrap();
    let mut journal = std::fs::File::create(root.join("callers.tsv")).unwrap();
    for policy in policies() {
        for use_ans in [false, true] {
            for (width, height) in [(32u32, 32u32), (259, 17)] {
                let pixels: Vec<u8> = (0..height)
                    .flat_map(|y| {
                        (0..width).flat_map(move |x| {
                            let (xx, yy) = ((x % 32) as i32, (y % 32) as i32);
                            if xx == 0 {
                                [128, 120, 136]
                            } else {
                                [
                                    128 + 4 * xx - 4 * yy,
                                    120 + 4 * xx - 3 * yy,
                                    136 + 3 * xx - 4 * yy,
                                ]
                            }
                            .map(|v| u8::try_from(v).unwrap())
                        })
                    })
                    .collect();

                let mut ordinary = None;
                for route in [
                    "legacy",
                    "request",
                    "strided",
                    "writer",
                    "streaming",
                    "animation",
                    "animation-request",
                ] {
                    let (bytes, record) =
                        encode(&pixels, width, height, None, Some(&policy), use_ans, route);
                    assert_eq!(record.predictor, 5, "selected deltas use Gradient");
                    assert!(
                        distinguishing_delta(&record).is_some(),
                        "actual distinguishing winning delta"
                    );
                    let target: Vec<u8> = record
                        .quantized
                        .iter()
                        .flatten()
                        .map(|v| (*v).clamp(0, 255) as u8)
                        .collect();
                    let out = root.join(format!(
                        "{}-ans-{use_ans}-{width}x{height}-{route}",
                        policy.0
                    ));
                    std::fs::create_dir(&out).unwrap();
                    let (primary, native) = decoded_codes(&bytes, &out);
                    assert_eq!(
                        primary, target,
                        "{} {route}: complete primary samples",
                        policy.0
                    );
                    assert_eq!(
                        native, target,
                        "{} {route}: complete native samples",
                        policy.0
                    );
                    if !route.starts_with("animation") {
                        if let Some(ref ordinary) = ordinary {
                            assert_eq!(&bytes, ordinary, "equivalent still entries");
                        } else {
                            ordinary = Some(bytes);
                        }
                    }
                    writeln!(journal,"{}\tans={use_ans}\t{width}x{height}\t{route}\tpredictor={}\tdeltas={}\texact=true",policy.0,record.predictor,record.nb_deltas).unwrap();
                    journal.flush().unwrap();
                }
            }
        }
    }
}

#[test]
#[ignore = "explicit no-delta and precision controls"]
fn delta_palette_no_delta_and_integer_extra_channels() {
    let root = crate::test_helpers::output_dir("delta-palette-precision");
    std::fs::create_dir_all(&root).unwrap();
    let root = (0..)
        .find_map(|i| {
            let p = root.join(format!("run-{i}"));
            match std::fs::create_dir(&p) {
                Ok(()) => Some(p),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => None,
                Err(e) => panic!("{e}"),
            }
        })
        .unwrap();
    let checker: Vec<u8> = (0..32 * 32)
        .flat_map(|i| {
            if (i / 32 + i % 32) % 2 == 0 {
                [255, 0, 0]
            } else {
                [0, 255, 0]
            }
        })
        .collect();
    let (bytes, record) = encode(&checker, 32, 32, None, None, false, "legacy");
    assert_eq!(record.predictor, 0, "actual no-delta outcome");
    let out = root.join("no-delta");
    std::fs::create_dir(&out).unwrap();
    let (primary, native) = decoded_codes(&bytes, &out);
    assert_eq!(primary, checker);
    assert_eq!(native, checker);
    for (bits, alpha, layout) in [
        (8, false, PixelLayout::Rgb8),
        (8, true, PixelLayout::Rgba8),
        (16, false, PixelLayout::Rgb16),
        (16, true, PixelLayout::Rgba16),
    ] {
        for (width, height) in [(32u32, 32u32), (259, 17)] {
            let max = if bits == 8 { 255u32 } else { 65535 };
            let scale = max / 255;
            let mut input = Vec::new();
            let mut expected_alpha = Vec::new();
            for y in 0..height {
                for x in 0..width {
                    let (xx, yy) = ((x % 32) as i32, (y % 32) as i32);
                    let rgb = if xx == 0 {
                        [128, 120, 136]
                    } else {
                        [
                            128 + 4 * xx - 4 * yy,
                            120 + 4 * xx - 3 * yy,
                            136 + 3 * xx - 4 * yy,
                        ]
                    };
                    for v in rgb {
                        let v = u16::try_from(v).unwrap() * (scale as u16);
                        if bits == 8 {
                            input.push(v as u8);
                        } else {
                            input.extend(v.to_ne_bytes());
                        }
                    }
                    if alpha {
                        let a = 1 + (x * 17 + y * 31) % (max - 1);
                        expected_alpha.push(a as u16);
                        if bits == 8 {
                            input.push(a as u8);
                        } else {
                            input.extend((a as u16).to_ne_bytes());
                        }
                    }
                }
            }
            for route in ["request", "strided", "streaming", "animation"] {
                let cfg = LosslessConfig::new()
                    .with_effort(7)
                    .with_threads(1)
                    .with_patches(false)
                    .with_squeeze(false)
                    .with_tree_learning(false)
                    .with_lz77(false)
                    .with_ans(true)
                    .with_force_rct(Some(RctType::NONE))
                    .with_modular_predictor(Some(0))
                    .with_modular_palette_colors(Some(16))
                    .with_lossy_palette(true)
                    .with_strategy(crate::api::EncoderStrategy::Libjxl);
                with_capture(|s| {
                    *s = Some(Capture {
                        force: None,
                        records: Vec::new(),
                    })
                });
                let bytes = match route {
                    "request" => cfg
                        .encode_request(width, height, layout)
                        .encode(&input)
                        .unwrap(),
                    "strided" => {
                        let row = width as usize * layout.bytes_per_pixel();
                        let stride = row + 7;
                        let mut padded = vec![0xd1; stride * height as usize];
                        for (a, b) in padded.chunks_exact_mut(stride).zip(input.chunks_exact(row)) {
                            a[..row].copy_from_slice(b);
                        }
                        cfg.encode_request(width, height, layout)
                            .with_row_stride(stride)
                            .encode(&padded)
                            .unwrap()
                    }
                    "streaming" => {
                        let mut enc = cfg.encoder(width, height, layout).unwrap();
                        for rows in input.chunks(width as usize * layout.bytes_per_pixel() * 7) {
                            enc.push_rows(
                                rows,
                                (rows.len() / (width as usize * layout.bytes_per_pixel())) as u32,
                            )
                            .unwrap();
                        }
                        enc.finish().unwrap()
                    }
                    "animation" => cfg
                        .encode_animation(
                            width,
                            height,
                            layout,
                            &crate::AnimationParams::default(),
                            &[crate::AnimationFrame::new(&input, 10)],
                        )
                        .unwrap(),
                    _ => unreachable!(),
                };
                let capture = with_capture(|s| s.take().unwrap());
                assert_eq!(capture.records.len(), 1);
                let record = &capture.records[0];
                let target: Vec<u16> = record
                    .quantized
                    .iter()
                    .flatten()
                    .map(|v| (*v).clamp(0, max as i32) as u16)
                    .collect();
                let primary = decoded_frames::decode_current_frames(
                    &bytes,
                    "palette precision and extra channels",
                );
                assert_eq!(primary.len(), 1);
                let raw = &primary[0].0;
                let channels = if alpha { 4 } else { 3 };
                let mut rgb = Vec::new();
                let mut a = Vec::new();
                if raw.len() == width as usize * height as usize * channels {
                    for p in raw.chunks_exact(channels) {
                        rgb.extend(p[..3].iter().map(|v| (*v * max as f32).round() as u16));
                        if alpha {
                            a.push((p[3] * max as f32).round() as u16);
                        }
                    }
                } else {
                    assert!(alpha);
                    assert_eq!(raw.len(), target.len());
                    rgb = raw
                        .iter()
                        .map(|v| (*v * max as f32).round() as u16)
                        .collect();
                    assert_eq!(primary[0].1.len(), 1);
                    a = primary[0].1[0]
                        .iter()
                        .map(|v| (*v * max as f32).round() as u16)
                        .collect();
                }
                assert_eq!(rgb, target, "complete primary quantized color");
                if alpha {
                    assert_eq!(a, expected_alpha, "exact primary alpha");
                }
                let out = root.join(format!("{layout:?}-{width}x{height}-{route}"));
                std::fs::create_dir(&out).unwrap();
                let jxl = out.join("input.jxl");
                let png = out.join("native.png");
                std::fs::write(&jxl, &bytes).unwrap();
                let native = std::process::Command::new(crate::test_helpers::djxl_path())
                    .arg(&jxl)
                    .arg(&png)
                    .arg("--num_threads=1")
                    .arg(format!("--bits_per_sample={bits}"))
                    .output()
                    .unwrap();
                std::fs::write(
                    out.join("native.log"),
                    [native.stdout.as_slice(), native.stderr.as_slice()].concat(),
                )
                .unwrap();
                assert!(native.status.success());
                let image = image::open(png).unwrap();
                let native: Vec<u16> = if bits == 8 {
                    (if alpha {
                        image.to_rgba8().into_raw()
                    } else {
                        image.to_rgb8().into_raw()
                    })
                    .into_iter()
                    .map(u16::from)
                    .collect()
                } else if alpha {
                    image.to_rgba16().into_raw()
                } else {
                    image.to_rgb16().into_raw()
                };
                let mut rgb = Vec::new();
                let mut a = Vec::new();
                for p in native.chunks_exact(channels) {
                    rgb.extend_from_slice(&p[..3]);
                    if alpha {
                        a.push(p[3]);
                    }
                }
                assert_eq!(rgb, target, "complete native quantized color");
                if alpha {
                    assert_eq!(a, expected_alpha, "exact native alpha");
                }
            }
        }
    }
}

#[test]
#[ignore = "explicit nonempty patch-reference palette writer"]
fn delta_palette_with_nonempty_patch_dictionary_preserves_selected_target() {
    use super::channel::ModularImage;
    use super::frame::{FrameEncoder, FrameEncoderOptions};
    use crate::bit_writer::BitWriter;
    use crate::headers::{color_encoding::ColorEncoding, file_header::FileHeader};
    let root = crate::test_helpers::output_dir("delta-palette-patches");
    std::fs::create_dir_all(&root).unwrap();
    let root = (0..)
        .find_map(|i| {
            let p = root.join(format!("run-{i}"));
            match std::fs::create_dir(&p) {
                Ok(()) => Some(p),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => None,
                Err(e) => panic!("{e}"),
            }
        })
        .unwrap();
    for (width, height) in [(32usize, 32usize), (259, 17)] {
        let pixels: Vec<u8> = (0..height)
            .flat_map(|y| {
                (0..width).flat_map(move |x| {
                    let (xx, yy) = ((x % 32) as i32, (y % 32) as i32);
                    (if xx == 0 {
                        [128, 120, 136]
                    } else {
                        [
                            128 + 4 * xx - 4 * yy,
                            120 + 4 * xx - 3 * yy,
                            136 + 3 * xx - 4 * yy,
                        ]
                    })
                    .map(|v| u8::try_from(v).unwrap())
                })
            })
            .collect();
        for use_ans in [false, true] {
            let mut image = ModularImage::from_rgb8(&pixels, width, height).unwrap();
            for (x, y) in [(3usize, 4usize), (7, 9)] {
                for (c, v) in [10, 20, 30].into_iter().enumerate() {
                    let previous = image.channels[c].get(x, y);
                    image.channels[c].set(x, y, previous - v);
                }
            }
            let patches = crate::vardct::patches::palette_contract_reference();
            assert_eq!(patches.positions.len(), 2);
            assert_eq!(patches.ref_positions.len(), 1);
            let mut writer = BitWriter::new();
            FileHeader::new_rgb(width as u32, height as u32)
                .write(&mut writer)
                .unwrap();
            writer.zero_pad_to_byte();
            crate::vardct::patches::encode_reference_frame_rgb(
                &patches,
                8,
                use_ans,
                false,
                &[],
                &mut writer,
                None,
            )
            .unwrap();
            let options = FrameEncoderOptions {
                use_ans,
                lossy_palette: true,
                use_tree_learning: false,
                use_squeeze: false,
                modular_knobs: super::palette::ModularKnobs {
                    modular_predictor: Some(0),
                    palette_colors: Some(16),
                    ..Default::default()
                },
                ..Default::default()
            };
            with_capture(|s| {
                *s = Some(Capture {
                    force: None,
                    records: Vec::new(),
                })
            });
            FrameEncoder::new(width, height, options)
                .encode_modular_with_patches(
                    &image,
                    &ColorEncoding::srgb(),
                    &mut writer,
                    Some(&patches),
                    None,
                )
                .unwrap();
            let capture = with_capture(|s| s.take().unwrap());
            let expected = if width <= 256 {
                assert_eq!(capture.records.len(), 1);
                let record = &capture.records[0];
                assert_eq!(record.predictor, 5);
                assert!(distinguishing_delta(record).is_some());
                let mut target = record.quantized.clone();
                for (x, y) in [(3usize, 4usize), (7, 9)] {
                    for (c, v) in [10, 20, 30].into_iter().enumerate() {
                        target[y * width + x][c] += v;
                    }
                }
                target
                    .into_iter()
                    .flatten()
                    .map(|v| v.clamp(0, 255) as u8)
                    .collect::<Vec<_>>()
            } else {
                // The multi-group patched writer dispatches directly to its
                // ordinary inner writer, so lossy_palette is not applied.
                assert!(
                    capture.records.is_empty(),
                    "observed multi-group patch bypass"
                );
                pixels.clone()
            };
            let out = root.join(format!("{width}x{height}-ans-{use_ans}"));
            std::fs::create_dir(&out).unwrap();
            let bytes = writer.finish_with_padding();
            let (primary, native) = decoded_codes(&bytes, &out);
            assert_eq!(
                primary, expected,
                "primary residual plus independent RGB patch"
            );
            assert_eq!(
                native, expected,
                "native residual plus independent RGB patch"
            );
        }
    }
}
