//! Caller-selected CMYK ICC controls for final progressive color and Black.
use jxl_encoder::{ImageMetadata, LossyConfig, PixelLayout, ProgressiveMode};
use sha2::{Digest, Sha256};
use std::{
    io::Write,
    path::{Path, PathBuf},
};
#[path = "../tests/support/decoded_extras.rs"]
mod decoded_extras;
#[path = "../tests/support/encoder_policies.rs"]
mod encoder_policies;
#[path = "../tests/support/pfm.rs"]
mod pfm;
fn hash(data: &[u8]) -> String {
    Sha256::digest(data)
        .iter()
        .map(|v| format!("{v:02x}"))
        .collect()
}
fn render(out: &Path, label: &str, bytes: &[u8], dims: (usize, usize)) -> [Vec<u32>; 2] {
    let path = out.join(format!("{label}.jxl"));
    std::fs::write(&path, bytes).unwrap();
    let (mut primary, extras) = decoded_extras::decode_current_planes(bytes, label);
    assert_eq!(extras.len(), 1, "Black");
    primary.extend_from_slice(&extras[0]);
    let output = out.join(format!("{label}.pfm"));
    let result = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
        .arg(path)
        .arg(&output)
        .args(["--output_extra_channels", "--num_threads=1"])
        .output()
        .unwrap();
    std::fs::write(out.join(format!("{label}.decode.log")), &result.stderr).unwrap();
    assert!(
        result.status.success(),
        "{label}: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    let mut native = pfm::read_pfm_samples(&output, dims);
    native.extend(pfm::read_pfm_samples(
        &out.join(format!("{label}.pfm-ec1.pfm")),
        dims,
    ));
    assert_eq!(primary.len(), native.len());
    assert!(primary.iter().chain(&native).all(|v| v.is_finite()));
    [primary, native].map(|v| v.into_iter().map(f32::to_bits).collect())
}
fn main() {
    let mut args = std::env::args_os().skip(1);
    let profile = std::fs::read(args.next().expect("CMYK ICC path")).unwrap();
    let out = PathBuf::from(args.next().expect("new evidence directory"));
    let selection = args.next();
    let animation_replace = selection.as_deref() == Some(std::ffi::OsStr::new("--animation-replace"));
    let animation_probe = animation_replace || selection.as_deref() == Some(std::ffi::OsStr::new("--animation"));
    let physical_control = if selection.as_deref() == Some(std::ffi::OsStr::new("--physical-control")) {
        Some(PathBuf::from(args.next().expect("native LCMS table")))
    } else { None };
    let source_control = if selection.as_deref() == Some(std::ffi::OsStr::new("--source-control")) {
        Some(PathBuf::from(
            args.next().expect("native reference directory"),
        ))
    } else {
        assert!(selection.is_none() || animation_probe || physical_control.is_some());
        None
    };
    assert!(args.next().is_none());
    assert_eq!(profile.get(16..20), Some(b"CMYK".as_slice()));
    std::fs::create_dir(&out).unwrap();
    std::fs::write(out.join("profile.icc"), &profile).unwrap();
    std::fs::write(out.join("profile.sha256"), hash(&profile)).unwrap();
    std::fs::write(
        out.join("binary.sha256"),
        hash(&std::fs::read(std::env::current_exe().unwrap()).unwrap()),
    )
    .unwrap();
    let metadata = ImageMetadata::new().with_icc_profile(&profile);
    if let Some(table) = physical_control { return probe_physical(&out, &metadata, &table); }
    if animation_probe {
        return probe_animation(&out, &metadata, animation_replace);
    }
    if let Some(reference) = source_control {
        return probe_source(&out, &metadata, &reference);
    }
    let fixture = image::load_from_memory(include_bytes!("../tests/images/frymire.png"))
        .unwrap()
        .to_rgb8();
    let mut journal = std::fs::File::create(out.join("completed.tsv")).unwrap();
    writeln!(journal, "case\tmode\tsha256").unwrap();
    for (policy, strategy, mode) in encoder_policies::policies() {
        for (width, height) in [(64_usize, 64_usize), (259, 133)] {
            for layout in [PixelLayout::Cmyk8, PixelLayout::Cmyk16] {
                let pixels: Vec<u8> = (0..height)
                    .flat_map(|y| {
                        let fixture = &fixture;
                        (0..width).flat_map(move |x| {
                            let p = fixture
                                .get_pixel(x as u32 % fixture.width(), y as u32 % fixture.height())
                                .0;
                            [p[0], p[1], p[2], p[1] / 2].into_iter().flat_map(move |v| {
                                if layout.is_16bit() {
                                    (u16::from(v) * 257).to_ne_bytes().to_vec()
                                } else {
                                    vec![v]
                                }
                            })
                        })
                    })
                    .collect();
                let label = format!("{policy}-{mode:?}-{width}x{height}-{layout:?}");
                let case = out.join(&label);
                std::fs::create_dir(&case).unwrap();
                std::fs::write(case.join("input.bin"), &pixels).unwrap();
                let mut baseline = None;
                for progressive in [
                    ProgressiveMode::Single,
                    ProgressiveMode::QuantizedAcFullAc,
                    ProgressiveMode::DcVlfLfAc,
                ] {
                    let config = LossyConfig::new(1.0)
                        .with_effort(4)
                        .with_threads(1)
                        .with_strategy(strategy.clone())
                        .with_mode(mode)
                        .with_force_strategy(Some(3))
                        .with_progressive(progressive);
                    let bytes = config
                        .encode_request(width as u32, height as u32, layout)
                        .with_metadata(&metadata)
                        .encode(&pixels)
                        .unwrap();
                    let actual =
                        render(&case, &format!("{progressive:?}"), &bytes, (width, height));
                    if let Some(expected) = &baseline {
                        assert_eq!(&actual, expected, "{label} {progressive:?}");
                    } else {
                        baseline = Some(actual);
                    }
                    writeln!(journal, "{label}\t{progressive:?}\t{}", hash(&bytes)).unwrap();
                    journal.flush().unwrap();
                }
            }
        }
    }
}

fn probe_animation(out: &Path, metadata: &ImageMetadata<'_>, explicit_replace: bool) {
    use jxl_encoder::{AnimationFrame, AnimationParams};
    let mut journal = std::fs::File::create(out.join("animation.tsv")).unwrap();
    writeln!(journal, "policy\tlayout\tprogressive\tstage\tresult").unwrap();
    let physical:Vec<[f32;3]>=if explicit_replace {
        let table=std::env::var_os("CMYK_NATIVE_GRADIENT").expect("native LCMS K-gradient table");
        let rows=std::fs::read_to_string(table).unwrap();let values:Vec<[f32;3]>=rows.lines().skip(1).map(|row|row.split('\t').skip(1).map(|x|x.parse::<f32>().unwrap()).collect::<Vec<_>>().try_into().unwrap()).collect();assert_eq!(values.len(),251);values
    }else{Vec::new()};
    let mut failures = Vec::new();
    for (width,height) in [(32_usize,17_usize),(259,17)] {
    let source_bound=if explicit_replace {
        let reference=std::env::var_os("CMYK_NATIVE_RGB_GRADIENT").expect("pinned v0.12 linear RGB gradient control");
        let reference=pfm::read_pfm_samples(&Path::new(&reference).join(format!("{width}x17.pfm")),(width,height));
        assert_eq!(reference.len(),width*height*3);assert!(reference.iter().all(|v|v.is_finite()));
        reference.iter().enumerate().map(|(i,&actual)|(actual-physical[(i/3)%251][i%3]).abs()).fold(0.005_f32,f32::max)
    }else{0.005};
    for layout in [PixelLayout::Cmyk8, PixelLayout::Cmyk16] {
        let pixels: Vec<u8> = (0..width * height)
            .flat_map(|i| {
                [64_u8, 128, 192, (i % 251) as u8]
                    .into_iter()
                    .flat_map(move |v| {
                        if layout.is_16bit() {
                            (u16::from(v) * 257).to_ne_bytes().to_vec()
                        } else {
                            vec![v]
                        }
                    })
            })
            .collect();
        // Prove that this selected profile is accepted and renders on the
        // existing still path before interpreting animation outcomes.
        let control = LossyConfig::new(if explicit_replace {0.01}else{1.0})
            .with_effort(3)
            .with_threads(1)
            .with_strategy(jxl_encoder::api::EncoderStrategy::Libjxl)
            .encode_request(width as u32, height as u32, layout)
            .with_metadata(metadata)
            .encode(&pixels)
            .unwrap();
        render(out, &format!("still-{layout:?}-{width}x{height}"), &control, (width, height));
        for (policy, strategy, mode) in encoder_policies::policies() {
            for progressive in [
                ProgressiveMode::Single,
                ProgressiveMode::QuantizedAcFullAc,
                ProgressiveMode::DcVlfLfAc,
            ] {
                let case = out.join(format!("animation-{layout:?}-{policy}-{progressive:?}-{width}x{height}"));
                std::fs::create_dir(&case).unwrap();
                std::fs::write(case.join("input.bin"), &pixels).unwrap();
                let config = LossyConfig::new(if explicit_replace {0.01}else{1.0})
                    .with_effort(3)
                    .with_threads(1)
                    .with_strategy(strategy.clone())
                    .with_mode(mode)
                    .with_progressive(progressive)
                    .with_auto_delta_frames(false);
                let frames = [
                    if explicit_replace { AnimationFrame::new(&pixels,10).with_blend_mode(jxl_encoder::BlendMode::Replace) } else { AnimationFrame::new(&pixels,10) },
                    if explicit_replace { AnimationFrame::new(&pixels,20).with_blend_mode(jxl_encoder::BlendMode::Replace) } else { AnimationFrame::new(&pixels,20) },
                ];
                let bytes = match config
                    .encode_request(width as u32, height as u32, layout)
                    .with_metadata(metadata)
                    .encode_animation(&AnimationParams::default(), &frames)
                {
                    Ok(bytes) => bytes,
                    Err(error) => {
                        writeln!(
                            journal,
                            "{policy}\t{layout:?}\t{progressive:?}\tencode\t{error:?}"
                        )
                        .unwrap();
                        journal.flush().unwrap();
                        failures.push(format!("{}: {error:?}", case.display()));
                        continue;
                    }
                };
                std::fs::write(case.join("animation.jxl"), &bytes).unwrap();
                let primary = std::panic::catch_unwind(|| {
                    decoded_extras::decode_current_frames(&bytes, &policy)
                });
                let native = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
                    .arg(case.join("animation.jxl"))
                    .arg(case.join("animation.pfm"))
                    .args([
                        "--output_frames",
                        "--output_extra_channels",
                        "--num_threads=1",
                    ])
                    .output()
                    .unwrap();
                std::fs::write(case.join("native.log"), &native.stderr).unwrap();
                let primary_good = primary.as_ref().is_ok_and(|frames| {
                    frames.len() == 2 && frames.iter().all(|(_, extras)| extras.len() == 1)
                });
                writeln!(
                    journal,
                    "{policy}\t{layout:?}\t{progressive:?}\tprimary\t{primary_good}"
                )
                .unwrap();
                writeln!(
                    journal,
                    "{policy}\t{layout:?}\t{progressive:?}\tnative\t{}",
                    native.status
                )
                .unwrap();
                journal.flush().unwrap();
                if explicit_replace && primary_good && native.status.success() {
                    let oracle=std::env::var_os("CMYK_RAW_FRAME_ORACLE").expect("caller-selected pinned native layer oracle");
                    let version=std::process::Command::new(&oracle).arg("--version").output().unwrap();assert!(version.status.success());
                    assert_eq!(String::from_utf8(version.stdout).unwrap().trim(),"libjxl CMYK raw-frame oracle 12000");
                    let observed=std::process::Command::new(oracle).arg(case.join("animation.jxl")).arg(case.join("native-layer")).arg("linear").output().unwrap();
                    std::fs::write(case.join("native-layer-headers.tsv"),&observed.stdout).unwrap();std::fs::write(case.join("native-layer.log"),&observed.stderr).unwrap();
                    assert!(observed.status.success(),"full-canvas Replace color and Black layer headers");
                    assert_eq!(String::from_utf8(observed.stdout).unwrap().lines().count(),2);
                    let frames=primary.as_ref().unwrap();
                    let maximum=if layout.is_16bit(){65535.0}else{255.0};
                    for (index,(color,extras)) in frames.iter().enumerate() {
                        let black_expected=|i:usize|if layout.is_16bit(){((i%251)*257)as u32}else{(i%251)as u32};
                        assert_eq!(color.len(),width*height*3);assert_eq!(extras[0].len(),width*height);
                        assert!(extras[0].iter().enumerate().all(|(i,&v)|v.is_finite() && (v*maximum).round() as u32==black_expected(i)),"primary Black codes");
                        let native_color=pfm::read_pfm_samples(&case.join(format!("native-layer-{index}.pfm")),(width,height));
                        let native_black=pfm::read_pfm_samples(&case.join(format!("native-layer-{index}-black.pfm")),(width,height));
                        assert!(native_black.iter().enumerate().all(|(i,&v)|v.is_finite() && (v*maximum).round() as u32==black_expected(i)),"native Black codes");
                        for (decoder,values) in [color.as_slice(),native_color.as_slice()].into_iter().enumerate() {
                            assert_eq!(values.len(),width*height*3);assert!(values.iter().all(|v|v.is_finite()));
                            let error=values.iter().enumerate().map(|(i,&actual)|(actual-physical[(i/3)%251][i%3]).abs()).fold(0_f32,f32::max);
                            assert!(error<=source_bound,"{policy}/{layout:?}/{progressive:?}/frame{index}/decoder{decoder}: {error} > native source control {source_bound}");
                            writeln!(journal,"{policy}\t{layout:?}\t{progressive:?}\tframe{index}-decoder{decoder}-source-error\t{error}/{source_bound}").unwrap();
                            journal.flush().unwrap();
                        }
                    }

                }
                if !primary_good || !native.status.success() {
                    failures.push(format!(
                        "{}: primary Black/render {primary_good}, native {}",
                        case.display(),
                        native.status
                    ));
                }
            }
        }
    }
    }
    std::fs::write(out.join("animation-failures.txt"), failures.join("\n")).unwrap();
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

fn probe_source(out: &Path, metadata: &ImageMetadata<'_>, reference: &Path) {
    let swatches = [
        [255, 255, 255, 255],
        [0, 255, 255, 255],
        [255, 0, 255, 255],
        [255, 255, 0, 255],
        [255, 255, 255, 0],
        [64, 128, 192, 224],
    ];
    let mut journal = std::fs::File::create(out.join("source.tsv")).unwrap();
    writeln!(
        journal,
        "policy\tcase\tlossless\tdecoder\tmax_color_error\tfirst_ours\tfirst_reference"
    )
    .unwrap();
    let mut failures = Vec::new();
    for (index, pixel) in swatches.into_iter().enumerate() {
        let pixels = pixel.repeat(32 * 17);
        for lossless in [false, true] {
            let name = format!("{index}-{}", if lossless { "lossless" } else { "lossy" });
            let reference = render(
                out,
                &format!("reference-{name}"),
                &std::fs::read(reference.join(format!("{name}.jxl"))).unwrap(),
                (32, 17),
            );
            for (policy, strategy, mode) in encoder_policies::policies() {
                let lossless_cfg = jxl_encoder::LosslessConfig::new()
                    .with_effort(3)
                    .with_threads(1)
                    .with_strategy(strategy.clone())
                    .with_mode(mode);
                let lossy_cfg = LossyConfig::new(0.01)
                    .with_effort(3)
                    .with_threads(1)
                    .with_strategy(strategy)
                    .with_mode(mode);
                let request = if lossless {
                    lossless_cfg.encode_request(32, 17, PixelLayout::Cmyk8)
                } else {
                    lossy_cfg.encode_request(32, 17, PixelLayout::Cmyk8)
                };
                let bytes = request.with_metadata(metadata).encode(&pixels).unwrap();
                let actual = render(out, &format!("{policy}-{name}"), &bytes, (32, 17));
                for decoder in 0..2 {
                    assert_eq!(actual[decoder].len(), reference[decoder].len());
                    let error = actual[decoder][..32 * 17 * 3]
                        .iter()
                        .zip(&reference[decoder])
                        .map(|(&a, &b)| (f32::from_bits(a) - f32::from_bits(b)).abs())
                        .fold(0_f32, f32::max);
                    writeln!(
                        journal,
                        "{policy}\t{index}\t{lossless}\t{decoder}\t{error}\t{:?}\t{:?}",
                        actual[decoder][..3]
                            .iter()
                            .copied()
                            .map(f32::from_bits)
                            .collect::<Vec<_>>(),
                        reference[decoder][..3]
                            .iter()
                            .copied()
                            .map(f32::from_bits)
                            .collect::<Vec<_>>()
                    )
                    .unwrap();
                    journal.flush().unwrap();
                    if error > 0.005 {
                        failures.push(format!("{policy} {name} decoder{decoder}: {error}"));
                    }
                    assert_eq!(
                        &actual[decoder][32 * 17 * 3..],
                        &reference[decoder][32 * 17 * 3..],
                        "exact Black samples"
                    );
                }
            }
        }
    }
    std::fs::write(out.join("source-failures.txt"), failures.join("\n")).unwrap();
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

fn probe_physical(out: &Path, metadata: &ImageMetadata<'_>, table: &Path) {
    let vectors: Vec<[f32;3]> = std::fs::read_to_string(table).unwrap().lines().skip(1).map(|line| {
        let values: Vec<f32> = line.split('\t').skip(1).map(|value| value.parse().unwrap()).collect();
        values.try_into().unwrap()
    }).collect();
    let swatches = [[255_u8,255,255,255],[0,255,255,255],[255,0,255,255],[255,255,0,255],[255,255,255,0],[64,128,192,224]];
    assert_eq!(vectors.len(),swatches.len());
    let mut journal = std::fs::File::create(out.join("physical.tsv")).unwrap();
    writeln!(journal,"case\tdecoder\tmaximum_error").unwrap();
    let mut failures=Vec::new();
    for (case,(sample,expected)) in swatches.into_iter().zip(vectors).enumerate() {
        for (policy,strategy,mode) in encoder_policies::policies() {
            for wide in [false,true] {
                for (w,h) in [(32_usize,17_usize),(259,17)] {
                    let layout=if wide {PixelLayout::Cmyk16} else {PixelLayout::Cmyk8};
                    let pixel:Vec<u8>=if wide {sample.iter().flat_map(|&value| (u16::from(value)*257).to_ne_bytes()).collect()} else {sample.to_vec()};
                    let pixels=pixel.repeat(w*h);
                    let config=LossyConfig::new(0.01).with_effort(3).with_threads(1).with_strategy(strategy.clone()).with_mode(mode);
                    let request=||config.encode_request(w as u32,h as u32,layout).with_metadata(metadata);
                    let bytes=request().encode(&pixels).unwrap();
                    let stride=w*pixel.len()+11;let mut padded=vec![0xA5;stride*h];
                    for row in 0..h {padded[row*stride..][..w*pixel.len()].copy_from_slice(&pixels[row*w*pixel.len()..][..w*pixel.len()]);}
                    assert_eq!(request().with_row_stride(stride).encode(&padded).unwrap(),bytes);
                    let mut destination=Vec::new();request().encode_to(&pixels,&mut destination).unwrap();assert_eq!(destination,bytes);
                    let mut stream=config.encoder(w as u32,h as u32,layout).unwrap().with_icc_profile(metadata.icc_profile().unwrap());
                    let error=stream.push_rows(&pixels[..w*pixel.len()],1).unwrap_err();
                    assert!(format!("{error:?}").contains("UnsupportedPixelLayout"),"{error:?}");
                    let mut destination=b"existing streaming destination".to_vec();let original=destination.clone();
                    assert!(stream.finish_into(&mut destination).is_err());assert_eq!(destination,original);
                    let label=format!("physical-{case}-{policy}-{wide}-{w}x{h}");
                    let path=out.join(format!("{label}.jxl"));std::fs::write(&path,&bytes).unwrap();
                    let (primary,extra)=decoded_extras::decode_current_planes(&bytes,&label);
                    assert_eq!(extra.len(),1);assert_eq!(primary.len(),w*h*3);
                    let maximum=if wide {65535.0} else {255.0};
                    let black_code=if wide {u32::from(sample[3])*257} else {u32::from(sample[3])};
                    assert!(extra[0].iter().all(|&value| value.is_finite() && (value*maximum).round() as u32==black_code),"exact Black codes");
                    let pfm=out.join(format!("{label}.pfm"));let native=std::process::Command::new(jxl_encoder::test_helpers::djxl_path()).arg(&path).arg(&pfm).args(["--num_threads=1","--color_space=RGB_D65_SRG_Rel_Lin","--output_extra_channels"]).output().unwrap();
                    std::fs::write(out.join(format!("{label}.decode.log")),native.stderr).unwrap();assert!(native.status.success());
                    let native_black=pfm::read_pfm_samples(&out.join(format!("{label}.pfm-ec1.pfm")),(w,h));
                    assert_eq!(native_black.len(),w*h);
                    assert!(native_black.iter().all(|&value| value.is_finite() && (value*maximum).round() as u32==black_code),"native exact Black codes");
                    for (decoder,values) in [primary,pfm::read_pfm_samples(&pfm,(w,h))].into_iter().enumerate() {
                        assert!(values.iter().all(|v|v.is_finite()));
                        let error=values.iter().zip(expected.iter().cycle()).map(|(a,b)|(a-b).abs()).fold(0_f32,f32::max);
                        writeln!(journal,"{label}\t{decoder}\t{error}").unwrap();journal.flush().unwrap();
                        if error>0.005 {failures.push(format!("{label} decoder{decoder}: {error}"));}
                    }
                }
            }
        }
    }
    std::fs::write(out.join("physical-failures.txt"),failures.join("\n")).unwrap();assert!(failures.is_empty(),"{failures:?}");
}
