// Decode-only (pixel) parity for JPEG->JXL transcode, plus orientation / ICC
// preservation. For each JPEG argument:
//   - decode the original JPEG to sRGB8 (image crate, raw pixels)
//   - transcode JPEG -> JXL and decode the JXL to sRGB8 via jxl-rs
//   - report max / mean per-channel pixel diff (the "decode perfectly" number)
//   - reconstruct the original JPEG byte-exactly (proves EXIF orientation + ICC
//     + all metadata survive the transcode)
//
//   cargo run -p jxl-encoder --features jpeg-reencoding --example jpeg_pixel_parity -- a.jpg b.jpg ...

#[cfg(feature = "jpeg-reencoding")]
fn decode_jxl_rs_rgb8(jxl: &[u8]) -> (u32, u32, Vec<u8>) {
    use jxl::api::{
        JxlColorType, JxlDataFormat, JxlDecoder, JxlOutputBuffer, JxlPixelFormat, ProcessingResult,
        states,
    };
    use jxl::image::{Image, Rect};

    let mut input: &[u8] = jxl;
    let mut decoder = JxlDecoder::<states::Initialized>::new(Default::default());
    let mut decoder = loop {
        match decoder.process(&mut input).expect("jxl-rs header") {
            ProcessingResult::Complete { result } => break result,
            ProcessingResult::NeedsMoreInput { fallback, .. } => decoder = fallback,
        }
    };
    let (w, h) = decoder.basic_info().size;
    decoder.set_pixel_format(JxlPixelFormat {
        color_type: JxlColorType::Rgb,
        color_data_format: Some(JxlDataFormat::f32()),
        extra_channel_format: vec![],
    });
    let mut decoder = loop {
        match decoder.process(&mut input).expect("jxl-rs frame") {
            ProcessingResult::Complete { result } => break result,
            ProcessingResult::NeedsMoreInput { fallback, .. } => decoder = fallback,
        }
    };
    let mut img = Image::<f32>::new((w * 3, h)).expect("alloc");
    let mut buffers = vec![JxlOutputBuffer::from_image_rect_mut(
        img.get_rect_mut(Rect {
            origin: (0, 0),
            size: (w * 3, h),
        })
        .into_raw(),
    )];
    loop {
        match decoder
            .process(&mut input, &mut buffers)
            .expect("jxl-rs decode")
        {
            ProcessingResult::Complete { .. } => break,
            ProcessingResult::NeedsMoreInput { fallback, .. } => decoder = fallback,
        }
    }
    // jxl-rs returns the image in the file's (sRGB) encoding as gamma-encoded
    // f32 in [0, 1]; map to sRGB8 the same way the JPEG decode produces it.
    let mut out = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for &p in img.row(y) {
            out.push((p.clamp(0.0, 1.0) * 255.0).round() as u8);
        }
    }
    (w as u32, h as u32, out)
}

#[cfg(feature = "jpeg-reencoding")]
fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: jpeg_pixel_parity <file.jpg> [more.jpg ...]");
        return;
    }
    let cfg = jxl_encoder::LosslessConfig::new();
    for path in &args {
        let jpeg = std::fs::read(path).expect("read jpeg");

        // Byte-exact reconstruction => EXIF orientation + ICC + all metadata preserved.
        let container = cfg
            .encode_jpeg_transcode(&jpeg)
            .expect("transcode container");
        let recon = zenjxl_decoder::reconstruct_jpeg(&container).expect("reconstruct");
        let byte_exact = recon.as_deref() == Some(jpeg.as_slice());

        // Reference: original JPEG decoded to raw sRGB8 (image crate).
        let jref = image::load_from_memory(&jpeg)
            .expect("decode jpeg")
            .to_rgb8();
        let (jw, jh) = jref.dimensions();

        // Decode-only: transcoded JXL codestream decoded to sRGB8 via jxl-rs.
        let codestream = cfg
            .encode_jpeg_transcode_codestream(&jpeg)
            .expect("transcode codestream");
        let (xw, xh, xrgb) = decode_jxl_rs_rgb8(&codestream);

        let name = std::path::Path::new(path)
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.clone());
        print!(
            "{name:<26} jpeg {jw}x{jh} / jxl {xw}x{xh}  recon={}  ",
            if byte_exact { "BYTE-EXACT" } else { "DIFF" }
        );
        if (jw, jh) != (xw, xh) {
            println!("DIMENSION MISMATCH (orientation applied?)");
            continue;
        }
        let (mut maxd, mut sum, mut ndiff) = (0u8, 0u64, 0u64);
        for (&a, &b) in jref.as_raw().iter().zip(xrgb.iter()) {
            let d = a.abs_diff(b);
            maxd = maxd.max(d);
            sum += d as u64;
            if d != 0 {
                ndiff += 1;
            }
        }
        let n = jref.as_raw().len() as f64;
        println!(
            "pixel: max={maxd} mean={:.4} diff={:.2}%",
            sum as f64 / n,
            100.0 * ndiff as f64 / n
        );
    }
}

#[cfg(not(feature = "jpeg-reencoding"))]
fn main() {
    eprintln!("requires --features jpeg-reencoding");
}
