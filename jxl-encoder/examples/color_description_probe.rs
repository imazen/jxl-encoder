//! Explicit full-decoder probe for Part 1 E.2 descriptions.
//! Decoder refusals are retained per cell and fail the command after the matrix.
use jxl_encoder::api::{EncoderImprovementsCustom, EncoderMode, EncoderStrategy};
use jxl_encoder::{
    AnimationFrame, AnimationParams, ColorEncoding, ColorSpace, LosslessConfig, PixelLayout,
    RenderingIntent,
};
use sha2::{Digest, Sha256};
use std::{io::Write, path::Path};

#[path = "distance_targeting_probe/decode.rs"]
mod decode;

fn strategies() -> Vec<(&'static str, EncoderStrategy)> {
    let mut all = vec![
        ("libjxl", EncoderStrategy::Libjxl),
        ("lean", EncoderStrategy::LeanFaster),
        ("zen", EncoderStrategy::Zenjxl),
        ("aggressive", EncoderStrategy::Aggressive),
    ];
    for (name, compact) in [("custom-long", false), ("custom-compact", true)] {
        all.push((
            name,
            EncoderStrategy::Custom(Box::new(EncoderImprovementsCustom {
                header_all_default_fast_paths: compact,
                ..Default::default()
            })),
        ));
    }
    all
}

fn render(
    bytes: &[u8],
    directory: &Path,
    pixels: &[u8],
    wide: bool,
    dims: (usize, usize),
    space: ColorSpace,
) -> (bool, bool) {
    let (w, h) = dims;
    let rust = std::panic::catch_unwind(|| decode::verify_jxl_rs(bytes, w, h));
    let max = if wide { 65535.0 } else { 255.0 };
    let expected: Vec<u16> = if wide {
        pixels
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| u16::from_ne_bytes(*b))
            .collect()
    } else {
        pixels.iter().map(|v| u16::from(*v)).collect()
    };
    if let Ok(samples) = &rust {
        assert_eq!(samples.len(), w * h * 3);
        assert!(samples.iter().all(|sample| sample.is_finite()));
        if space != ColorSpace::Xyb {
            let samples: Vec<_> = samples.iter().map(|v| (v * max).round() as u16).collect();
            assert_eq!(samples, expected);
        }
    }
    let output = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
        .arg(directory.join("encoded.jxl"))
        .arg(directory.join("decoded.png"))
        .args(["--num_threads=1", "--bits_per_sample=16"])
        .output()
        .unwrap();
    std::fs::write(directory.join("djxl.log"), &output.stderr).unwrap();
    if output.status.success() {
        let decoded = image::open(directory.join("decoded.png"))
            .unwrap()
            .to_rgb16();
        assert_eq!(decoded.dimensions(), (w as u32, h as u32));
        if space != ColorSpace::Xyb {
            let expected: Vec<_> = expected
                .iter()
                .map(|v| if wide { *v } else { *v * 257 })
                .collect();
            assert_eq!(decoded.as_raw(), &expected);
        }
    }
    (rust.is_ok(), output.status.success())
}

fn main() {
    let mut args = std::env::args().skip(1);
    let space = match args.next().as_deref() {
        Some("xyb") => ColorSpace::Xyb,
        Some("unknown") => ColorSpace::Unknown,
        Some("rgb") => ColorSpace::Rgb,
        _ => panic!("usage: color_description_probe <rgb|xyb|unknown> <new-output-directory>"),
    };
    let directory = std::path::PathBuf::from(args.next().expect("new output directory"));
    assert!(args.next().is_none());
    std::fs::create_dir(&directory).expect("output directory must be new");
    let mut table = std::fs::File::create_new(directory.join("results.tsv")).unwrap();
    writeln!(table, "cell\tbytes\tsha256\tjxl_rs\tdjxl").unwrap();
    let color = ColorEncoding {
        color_space: space,
        rendering_intent: RenderingIntent::Perceptual,
        ..ColorEncoding::srgb()
    };
    let mut failed = false;
    for (name, strategy) in strategies() {
        for mode in [EncoderMode::Reference, EncoderMode::Experimental] {
            let cfg = LosslessConfig::new()
                .with_effort(1)
                .with_strategy(strategy.clone())
                .with_mode(mode);
            for (w, h) in [(64_usize, 32_usize), (259, 17)] {
                for wide in [false, true] {
                    let layout = if wide {
                        PixelLayout::Rgb16
                    } else {
                        PixelLayout::Rgb8
                    };
                    let pixels: Vec<u8> = (0..w * h * 3)
                        .flat_map(|i| {
                            let value = ((i * 37 + i / 7) % 256) as u8;
                            if wide {
                                (u16::from(value) * 257).to_ne_bytes().to_vec()
                            } else {
                                vec![value]
                            }
                        })
                        .collect();
                    let request = cfg
                        .encode_request(w as u32, h as u32, layout)
                        .with_color_encoding(color.clone());
                    let still = request.encode(&pixels).unwrap();
                    for route in ["still", "strided", "streaming", "animation"] {
                        let bytes = match route {
                            "strided" => {
                                let row_bytes = w * layout.bytes_per_pixel();
                                let stride = row_bytes + 3;
                                let mut padded = vec![0xa5; stride * h];
                                for (source, dest) in pixels
                                    .chunks_exact(row_bytes)
                                    .zip(padded.chunks_exact_mut(stride))
                                {
                                    dest[..row_bytes].copy_from_slice(source);
                                }
                                let bytes = cfg
                                    .encode_request(w as u32, h as u32, layout)
                                    .with_color_encoding(color.clone())
                                    .with_row_stride(stride)
                                    .encode(&padded)
                                    .unwrap();
                                assert_eq!(bytes, still);
                                bytes
                            }
                            "streaming" => {
                                let mut stream = cfg
                                    .encoder(w as u32, h as u32, layout)
                                    .unwrap()
                                    .with_color_encoding(color.clone());
                                for chunk in pixels.chunks(w * layout.bytes_per_pixel() * 7) {
                                    stream
                                        .push_rows(
                                            chunk,
                                            (chunk.len() / (w * layout.bytes_per_pixel())) as u32,
                                        )
                                        .unwrap();
                                }
                                let bytes = stream.finish().unwrap();
                                assert_eq!(bytes, still);
                                bytes
                            }
                            "animation" => cfg
                                .encode_request(w as u32, h as u32, layout)
                                .with_color_encoding(color.clone())
                                .encode_animation(
                                    &AnimationParams::default(),
                                    &[AnimationFrame::new(&pixels, 10)],
                                )
                                .unwrap(),
                            _ => still.clone(),
                        };
                        let cell = format!("{name}-{mode:?}-{w}x{h}-{wide}-{route}");
                        let cell_dir = directory.join(&cell);
                        std::fs::create_dir(&cell_dir).unwrap();
                        std::fs::write(cell_dir.join("encoded.jxl"), &bytes).unwrap();
                        let (rust, reference) =
                            render(&bytes, &cell_dir, &pixels, wide, (w, h), space);
                        let sha = Sha256::digest(&bytes)
                            .iter()
                            .map(|byte| format!("{byte:02x}"))
                            .collect::<String>();
                        writeln!(table, "{cell}\t{}\t{sha}\t{rust}\t{reference}", bytes.len())
                            .unwrap();
                        table.flush().unwrap();
                        failed |= !rust || !reference;
                    }
                }
            }
        }
    }
    assert!(
        !failed,
        "one or more decoders refused retained descriptions; see results.tsv"
    );
}
