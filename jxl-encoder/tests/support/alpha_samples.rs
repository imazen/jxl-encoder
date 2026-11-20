//! Pinned native alpha planes and explicit input sample representations.
use super::pfm;
use jxl_encoder::PixelLayout;
use std::path::Path;

pub(crate) fn source(
    layout: PixelLayout,
    width: usize,
    height: usize,
    phase: usize,
) -> (Vec<u8>, Vec<u16>, u16) {
    let float = layout.is_f16() || layout.is_f32();
    let max = if layout.is_16bit() || float {
        65535
    } else {
        255
    };
    let mut pixels = Vec::new();
    let mut alpha = Vec::new();
    for y in 0..height {
        for x in 0..width {
            let selector = (17 * x + 29 * y + phase) % 256;
            let code = if float {
                [0, 32768, 65535][selector * 3 / 256]
            } else if max == 65535 {
                (selector * 257 + (x + y) % 257).min(65535) as u16
            } else {
                selector as u16
            };
            alpha.push(code);
            let channels = if layout.is_grayscale() { 1 } else { 3 };
            if layout.is_f16() {
                for &value in &[0x3400_u16, 0x3800, 0x3a00][..channels] {
                    pixels.extend(value.to_ne_bytes());
                }
                pixels.extend(
                    match code {
                        0 => 0_u16,
                        32768 => 0x3800,
                        _ => 0x3c00,
                    }
                    .to_ne_bytes(),
                );
            } else if layout.is_f32() {
                for &value in &[0.25_f32, 0.5, 0.75][..channels] {
                    pixels.extend(value.to_ne_bytes());
                }
                pixels.extend(
                    match code {
                        0 => 0.0_f32,
                        32768 => 0.5,
                        _ => 1.0,
                    }
                    .to_ne_bytes(),
                );
            } else if layout.is_16bit() {
                for &value in &[16384_u16, 32768, 49152][..channels] {
                    pixels.extend(value.to_ne_bytes());
                }
                pixels.extend(code.to_ne_bytes());
            } else {
                pixels.extend_from_slice(&[64, 128, 192][..channels]);
                pixels.push(code as u8);
            }
        }
    }
    (pixels, alpha, max)
}

pub(crate) fn native_alpha(
    out: &Path,
    dims: (usize, usize),
    alpha: &[u16],
    max: u16,
    distance: f32,
    squeeze: bool,
) -> Vec<i64> {
    std::fs::create_dir(out).unwrap();
    let (width, height) = dims;
    let mut pam = format!(
        "P7\nWIDTH {width}\nHEIGHT {height}\nDEPTH 4\nMAXVAL {max}\nTUPLTYPE RGB_ALPHA\nENDHDR\n"
    )
    .into_bytes();
    for &a in alpha {
        for sample in [max / 4, max / 2, (u32::from(max) * 3 / 4) as u16, a] {
            if max == 255 {
                pam.push(sample as u8);
            } else {
                pam.extend(sample.to_be_bytes());
            }
        }
    }
    std::fs::write(out.join("input.pam"), pam).unwrap();
    let encoded = std::process::Command::new(jxl_encoder::test_helpers::cjxl_path())
        .arg(out.join("input.pam"))
        .arg(out.join("reference.jxl"))
        .args(["-d", "1", "-e", "3", "--num_threads=1"])
        .arg(if squeeze {
            "--responsive=1"
        } else {
            "--responsive=0"
        })
        .arg(format!("--alpha_distance={distance}"))
        .output()
        .unwrap();
    std::fs::write(out.join("encode.log"), &encoded.stderr).unwrap();
    assert!(
        encoded.status.success(),
        "native alpha reference encode: {out:?}"
    );
    let decoded = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
        .arg(out.join("reference.jxl"))
        .arg(out.join("reference.pfm"))
        .args(["--output_extra_channels", "--num_threads=1"])
        .output()
        .unwrap();
    std::fs::write(out.join("decode.log"), &decoded.stderr).unwrap();
    assert!(decoded.status.success());
    pfm::read_pfm_samples(&out.join("reference.pfm-ec1.pfm"), dims)
        .into_iter()
        .map(|v| (v * f32::from(max)).round() as i64)
        .collect()
}
