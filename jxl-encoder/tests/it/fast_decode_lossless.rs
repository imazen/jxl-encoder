// Copyright (c) Imazen LLC.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! `LosslessConfig::with_fast_decode` must stay an ordinary conformant
//! lossless codestream: four independent decoders (jxl-rs, jxl-oxide, libjxl
//! v0.12 `djxl`, zenjxl-decoder) reproduce every source sample exactly, for
//! gray, gray+alpha, RGB and RGBA at 8 and 16 bits, single- and multi-group
//! sizes, at efforts 1, 4 and 9.
//!
//! The released (unpatched) decoder crates are checked separately by
//! `tools/decoder-conformance`; see `just fast-decode-conformance`.

use jxl_encoder::api::{LosslessConfig, PixelLayout};

const EFFORTS: [u8; 3] = [1, 4, 9];

/// Interleaved source samples, widened to u16 with their bit depth.
struct Source {
    width: usize,
    height: usize,
    /// Color channels (1 or 3).
    color: usize,
    alpha: bool,
    bits: u32,
    samples: Vec<u16>,
}

impl Source {
    fn channels(&self) -> usize {
        self.color + usize::from(self.alpha)
    }

    fn layout(&self) -> PixelLayout {
        match (self.color, self.alpha, self.bits) {
            (1, false, 8) => PixelLayout::Gray8,
            (1, true, 8) => PixelLayout::GrayAlpha8,
            (3, false, 8) => PixelLayout::Rgb8,
            (3, true, 8) => PixelLayout::Rgba8,
            (1, false, 16) => PixelLayout::Gray16,
            (1, true, 16) => PixelLayout::GrayAlpha16,
            (3, false, 16) => PixelLayout::Rgb16,
            (3, true, 16) => PixelLayout::Rgba16,
            other => unreachable!("{other:?}"),
        }
    }

    fn bytes(&self) -> Vec<u8> {
        if self.bits == 8 {
            self.samples.iter().map(|&s| s as u8).collect()
        } else {
            self.samples.iter().flat_map(|s| s.to_ne_bytes()).collect()
        }
    }

    /// Samples as RGBA at the source depth, for decoders that expand gray.
    fn rgba(&self) -> Vec<u16> {
        let max = ((1u32 << self.bits) - 1) as u16;
        self.samples
            .chunks_exact(self.channels())
            .flat_map(|p| {
                let (c, a) = (&p[..self.color], p.get(self.color).copied().unwrap_or(max));
                if self.color == 1 {
                    [c[0], c[0], c[0], a]
                } else {
                    [c[0], c[1], c[2], a]
                }
            })
            .collect()
    }
}

/// Crops of the repository's photo-like test image, with a non-trivial
/// alpha plane and genuine 16-bit low bits.
fn sources() -> Vec<(String, Source)> {
    let img = image::open(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/images/frymire-srgb.png"
    ))
    .expect("open frymire-srgb.png");
    let mut out = Vec::new();
    for (w, h) in [(40u32, 30u32), (300, 260)] {
        let rgb = img.crop_imm(37, 41, w, h).to_rgb8();
        for color in [1usize, 3] {
            for alpha in [false, true] {
                for bits in [8u32, 16] {
                    let mut samples = Vec::new();
                    for (x, y, p) in rgb.enumerate_pixels() {
                        let [r, g, b] = p.0;
                        let mut px: Vec<u32> = if color == 1 {
                            vec![(u32::from(r) * 77 + u32::from(g) * 150 + u32::from(b) * 29) >> 8]
                        } else {
                            vec![r.into(), g.into(), b.into()]
                        };
                        if alpha {
                            px.push((x * 7 + y * 3) % 256);
                        }
                        for v in px {
                            let s = if bits == 8 {
                                v
                            } else {
                                // Distinct low byte so 16-bit is not 8-bit x 257.
                                (v << 8) | ((v * 37 + x + y * 5) & 0xff)
                            };
                            samples.push(s as u16);
                        }
                    }
                    let name = format!(
                        "{}{}{bits}_{w}x{h}",
                        if color == 1 { "gray" } else { "rgb" },
                        if alpha { "a" } else { "" }
                    );
                    out.push((
                        name,
                        Source {
                            width: w as usize,
                            height: h as usize,
                            color,
                            alpha,
                            bits,
                            samples,
                        },
                    ));
                }
            }
        }
    }
    out
}

fn decode_jxl_rs(data: &[u8], src: &Source) -> Vec<u16> {
    use jxl::api::{
        JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer, JxlPixelFormat,
        ProcessingResult, states,
    };
    use jxl::image::{Image, Rect};

    let mut input = data;
    let mut init = JxlDecoder::<states::Initialized>::new(JxlDecoderOptions::default());
    let mut decoder = loop {
        match init.process(&mut input) {
            Ok(ProcessingResult::Complete { result }) => break result,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => init = fallback,
            Err(e) => panic!("jxl-rs header: {e:?}"),
        }
    };
    let (w, h) = decoder.basic_info().size;
    assert_eq!((w, h), (src.width, src.height), "jxl-rs size");
    let format = decoder.current_pixel_format().clone();
    let color = format.color_type.samples_per_pixel();
    let extra = format.extra_channel_format.len();
    assert_eq!(
        (color, extra),
        (src.color, usize::from(src.alpha)),
        "jxl-rs channels"
    );
    decoder.set_pixel_format(JxlPixelFormat {
        color_type: format.color_type,
        color_data_format: Some(JxlDataFormat::f32()),
        extra_channel_format: vec![Some(JxlDataFormat::f32()); extra],
    });
    let mut frame = loop {
        match decoder.process(&mut input) {
            Ok(ProcessingResult::Complete { result }) => break result,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => decoder = fallback,
            Err(e) => panic!("jxl-rs frame info: {e:?}"),
        }
    };
    let mut color_img = Image::<f32>::new((w * color, h)).expect("alloc");
    let mut extra_imgs: Vec<Image<f32>> = (0..extra)
        .map(|_| Image::<f32>::new((w, h)).expect("alloc"))
        .collect();
    {
        let mut buffers = vec![JxlOutputBuffer::from_image_rect_mut(
            color_img
                .get_rect_mut(Rect {
                    origin: (0, 0),
                    size: (w * color, h),
                })
                .into_raw(),
        )];
        for e in &mut extra_imgs {
            buffers.push(JxlOutputBuffer::from_image_rect_mut(
                e.get_rect_mut(Rect {
                    origin: (0, 0),
                    size: (w, h),
                })
                .into_raw(),
            ));
        }
        loop {
            match frame.process(&mut input, &mut buffers) {
                Ok(ProcessingResult::Complete { .. }) => break,
                Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => frame = fallback,
                Err(e) => panic!("jxl-rs decode: {e:?}"),
            }
        }
    }
    let max = ((1u32 << src.bits) - 1) as f32;
    let q = |v: f32| (v * max).round() as u16;
    let mut out = Vec::with_capacity(w * h * (color + extra));
    for y in 0..h {
        for x in 0..w {
            for c in 0..color {
                out.push(q(color_img.row(y)[x * color + c]));
            }
            for e in &extra_imgs {
                out.push(q(e.row(y)[x]));
            }
        }
    }
    out
}

fn decode_jxl_oxide(data: &[u8], src: &Source) -> Vec<u16> {
    let image = jxl_oxide::JxlImage::builder()
        .read(std::io::Cursor::new(data))
        .expect("jxl-oxide header");
    assert_eq!(
        (image.width() as usize, image.height() as usize),
        (src.width, src.height),
        "jxl-oxide size"
    );
    let render = image.render_frame(0).expect("jxl-oxide render");
    let mut stream = render.stream();
    assert_eq!(
        stream.channels() as usize,
        src.channels(),
        "jxl-oxide channels"
    );
    let mut buf = vec![0u16; src.samples.len()];
    assert_eq!(
        stream.write_to_buffer(&mut buf),
        buf.len(),
        "jxl-oxide samples"
    );
    // u16 output is scaled to the full 16-bit range.
    if src.bits == 8 {
        buf.iter()
            .map(|&v| ((u32::from(v) + 128) / 257) as u16)
            .collect()
    } else {
        buf
    }
}

fn decode_djxl(data: &[u8], src: &Source, name: &str) -> Vec<u16> {
    let dir = jxl_encoder::test_helpers::output_dir_for("jxl-encoder", "fast_decode_lossless");
    let jxl = dir.join(format!("{name}.jxl"));
    let png = dir.join(format!("{name}.png"));
    std::fs::write(&jxl, data).expect("write");
    let out = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
        .arg(&jxl)
        .arg(&png)
        .arg("--num_threads=1")
        .output()
        .expect("run djxl v0.12");
    assert!(
        out.status.success(),
        "djxl rejected {name}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let img = image::open(&png).expect("open djxl output").to_rgba16();
    assert_eq!(
        (img.width() as usize, img.height() as usize),
        (src.width, src.height)
    );
    let shift = 16 - src.bits;
    img.into_raw()
        .into_iter()
        .map(|v| {
            if shift == 0 {
                v
            } else {
                ((u32::from(v) + 128) / 257) as u16
            }
        })
        .collect()
}

#[test]
fn fast_decode_preset_is_exact_in_four_decoders() {
    for (name, src) in sources() {
        for effort in EFFORTS {
            let label = format!("{name}_e{effort}");
            let data = LosslessConfig::new()
                .with_effort(effort)
                .with_threads(1)
                .with_fast_decode()
                .encode(
                    &src.bytes(),
                    src.width as u32,
                    src.height as u32,
                    src.layout(),
                )
                .unwrap_or_else(|e| panic!("{label}: encode {e:?}"));

            assert_eq!(decode_jxl_rs(&data, &src), src.samples, "{label}: jxl-rs");
            assert_eq!(
                decode_jxl_oxide(&data, &src),
                src.samples,
                "{label}: jxl-oxide"
            );
            assert_eq!(
                decode_djxl(&data, &src, &label),
                src.rgba(),
                "{label}: djxl"
            );
            if src.bits == 8 {
                let z = zenjxl_decoder::decode(&data).expect("zenjxl decode");
                let rgba: Vec<u16> = if z.channels == 4 {
                    z.data.iter().map(|&v| v.into()).collect()
                } else {
                    z.data
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .flat_map(|&[g, a]| [g, g, g, a].map(u16::from))
                        .collect()
                };
                assert_eq!(rgba, src.rgba(), "{label}: zenjxl-decoder");
            }
        }
    }
}
