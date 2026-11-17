//! Explicit caller-supplied CMYK ICC controls for shifted native depth grids.
use jxl_encoder::api::{ExtraChannel, SectionedTrees};
use jxl_encoder::{ImageMetadata, LosslessConfig, PixelLayout, RctType};
use sha2::{Digest, Sha256};
use std::{io::Write, path::PathBuf};
#[path = "../tests/support/decoded_extras.rs"]
mod decoded_extras;
#[path = "../tests/support/encoder_policies.rs"]
mod encoder_policies;
#[path = "../tests/support/pfm.rs"]
mod pfm;

fn main() {
    let mut args = std::env::args_os().skip(1);
    let profile_path = PathBuf::from(args.next().expect("CMYK ICC path"));
    let out = PathBuf::from(args.next().expect("new evidence directory"));
    let extra_count: usize = args
        .next()
        .map_or(1, |v| v.to_str().unwrap().parse().unwrap());
    assert!((1..=18).contains(&extra_count));
    let squeeze = args.next().is_some_and(|v| {
        assert_eq!(v, "--squeeze");
        true
    });
    assert!(args.next().is_none(), "unexpected argument");
    std::fs::create_dir(&out).expect("new evidence directory");
    let profile = std::fs::read(&profile_path).expect("required CMYK ICC");
    assert_eq!(profile.get(16..20), Some(b"CMYK".as_slice()));
    assert_eq!(profile.get(36..40), Some(b"acsp".as_slice()));
    std::fs::write(
        out.join("profile.sha256"),
        Sha256::digest(&profile)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>()
            + "\n",
    )
    .unwrap();
    let metadata = ImageMetadata::new().with_icc_profile(&profile);
    let binary = std::fs::read(std::env::current_exe().unwrap()).unwrap();
    let binary_hash = Sha256::digest(&binary)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>();
    std::fs::write(out.join("binary.sha256"), binary_hash + "\n").unwrap();
    let mut journal =
        std::io::BufWriter::new(std::fs::File::create(out.join("completed.tsv")).unwrap());
    writeln!(
        journal,
        "label\twidth\theight\tlayout\tdim_shift\tencoded_sha256"
    )
    .unwrap();
    journal.flush().unwrap();
    for (name, strategy, mode) in encoder_policies::policies() {
        for wide in [false, true] {
            let layout = if wide {
                PixelLayout::Cmyk16
            } else {
                PixelLayout::Cmyk8
            };
            for &shift in if squeeze { &[0, 1, 3][..] } else { &[1, 3][..] } {
                for (w, h) in [(17_usize, 9_usize), (259, 17)] {
                    for sectioned in [SectionedTrees::Off, SectionedTrees::On] {
                        let config = LosslessConfig::new()
                            .with_effort(7)
                            .with_strategy(strategy.clone())
                            .with_mode(mode)
                            .with_threads(1)
                            .with_modular_group_size(Some(if squeeze { 0 } else { 1 }))
                            .with_squeeze(squeeze)
                            .with_patches(false)
                            .with_tree_learning(true)
                            .with_ans(true)
                            .with_sectioned_trees(sectioned)
                            .with_modular_palette_colors(Some(0))
                            .with_force_rct(Some(RctType(0)));
                        let native = vec![173; w.div_ceil(1 << shift) * h.div_ceil(1 << shift)];
                        let native16 = vec![0xad5f; native.len()];
                        let extras = vec![
                            if wide {
                                ExtraChannel::depth_u16(&native16)
                            } else {
                                ExtraChannel::depth(&native)
                            }
                            .with_dim_shift(shift);
                            extra_count
                        ];
                        let pixels = vec![255; w * h * layout.bytes_per_pixel()];
                        let request = || {
                            config
                                .encode_request(w as u32, h as u32, layout)
                                .with_metadata(&metadata)
                                .with_extra_channels(&extras)
                        };
                        let bytes = request().encode(&pixels).unwrap();
                        let row = w * layout.bytes_per_pixel();
                        let stride = row + 3;
                        let mut padded = vec![0xa5; stride * h];
                        for (src, dst) in pixels
                            .chunks_exact(row)
                            .zip(padded.chunks_exact_mut(stride))
                        {
                            dst[..row].copy_from_slice(src);
                        }
                        assert_eq!(
                            request().with_row_stride(stride).encode(&padded).unwrap(),
                            bytes
                        );
                        let label = format!("{name}-wide{wide}-s{shift}-{w}x{h}-{sectioned:?}");
                        std::fs::write(out.join(format!("{label}.jxl")), &bytes).unwrap();
                        let (_, planes) = decoded_extras::decode_current_planes(&bytes, &label);
                        assert_eq!(planes.len(), extra_count + 1, "Black and Depth");
                        let result =
                            std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
                                .arg(out.join(format!("{label}.jxl")))
                                .arg(out.join(format!("{label}.pfm")))
                                .args(["--output_extra_channels", "--num_threads=1"])
                                .output()
                                .unwrap();
                        std::fs::write(out.join(format!("{label}.decode.log")), &result.stderr)
                            .unwrap();
                        assert!(
                            result.status.success(),
                            "{label}: {}",
                            String::from_utf8_lossy(&result.stderr)
                        );
                        let digits = (extra_count + 1).to_string().len();
                        for (index, decoded) in planes.iter().enumerate().skip(1) {
                            let reference = pfm::read_pfm_samples(
                                &out.join(format!("{label}.pfm-ec{:0digits$}.pfm", index + 1)),
                                (w, h),
                            );
                            for plane in [decoded, &reference] {
                                assert_eq!(plane.len(), w * h);
                                for v in plane {
                                    if wide {
                                        assert_eq!(
                                            (*v * 65535.0).round() as u16,
                                            0xad5f,
                                            "{label}: {index}"
                                        );
                                    } else {
                                        assert_eq!(
                                            (*v * 255.0).round() as u8,
                                            173,
                                            "{label}: {index}"
                                        );
                                    }
                                }
                            }
                        }
                        let hash = Sha256::digest(&bytes)
                            .iter()
                            .map(|b| format!("{b:02x}"))
                            .collect::<String>();
                        writeln!(journal, "{label}\t{w}\t{h}\t{layout:?}\t{shift}\t{hash}")
                            .unwrap();
                        journal.flush().unwrap();
                    }
                }
            }
        }
    }
}
