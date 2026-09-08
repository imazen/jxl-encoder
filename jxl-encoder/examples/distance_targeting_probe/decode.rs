//! Full jxl-rs frame decode, including extra channels; returns RGB samples.

pub(super) fn verify_jxl_rs(data: &[u8], width: usize, height: usize) -> Vec<f32> {
    use jxl::api::{
        JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer, JxlPixelFormat,
        ProcessingResult, states,
    };
    use jxl::image::{Image, Rect};

    let mut input = data;
    let mut init = JxlDecoder::<states::Initialized>::new(JxlDecoderOptions::default());
    let mut decoder = loop {
        match init.process(&mut input).expect("jxl-rs image header") {
            ProcessingResult::Complete { result } => break result,
            ProcessingResult::NeedsMoreInput { fallback, .. } => {
                assert!(!input.is_empty(), "truncated jxl-rs image header");
                init = fallback;
            }
        }
    };
    assert_eq!(decoder.basic_info().size, (width, height));
    let format = decoder.current_pixel_format();
    assert_eq!(format.color_type.samples_per_pixel(), 3);
    let num_extra = format.extra_channel_format.len();
    decoder.set_pixel_format(JxlPixelFormat {
        color_type: format.color_type,
        color_data_format: Some(JxlDataFormat::f32()),
        extra_channel_format: vec![Some(JxlDataFormat::f32()); num_extra],
    });
    let mut decoder = loop {
        match decoder.process(&mut input).expect("jxl-rs frame header") {
            ProcessingResult::Complete { result } => break result,
            ProcessingResult::NeedsMoreInput { fallback, .. } => {
                assert!(!input.is_empty(), "truncated jxl-rs frame header");
                decoder = fallback;
            }
        }
    };
    let mut pixels = Image::<f32>::new((width * 3, height)).expect("jxl-rs pixel buffer");
    let mut extras: Vec<_> = (0..num_extra)
        .map(|_| Image::<f32>::new((width, height)).expect("jxl-rs extra buffer"))
        .collect();
    let mut buffers = vec![JxlOutputBuffer::from_image_rect_mut(
        pixels
            .get_rect_mut(Rect {
                origin: (0, 0),
                size: (width * 3, height),
            })
            .into_raw(),
    )];
    for extra in &mut extras {
        buffers.push(JxlOutputBuffer::from_image_rect_mut(
            extra
                .get_rect_mut(Rect {
                    origin: (0, 0),
                    size: (width, height),
                })
                .into_raw(),
        ));
    }
    loop {
        match decoder
            .process(&mut input, &mut buffers)
            .expect("jxl-rs frame pixels")
        {
            ProcessingResult::Complete { .. } => break,
            ProcessingResult::NeedsMoreInput { fallback, .. } => {
                assert!(!input.is_empty(), "truncated jxl-rs frame pixels");
                decoder = fallback;
            }
        }
    }
    for extra in &extras {
        for y in 0..height {
            assert!(
                extra.row(y).iter().all(|v| v.is_finite()),
                "non-finite extra channel"
            );
        }
    }
    for y in 0..height {
        assert!(
            pixels.row(y).iter().all(|v| v.is_finite()),
            "non-finite decoded pixel"
        );
    }
    (0..height)
        .flat_map(|y| pixels.row(y).iter().copied())
        .collect()
}

/// The decoder extends sRGB symmetrically outside the display gamut.
/// Match jxl-rs color/tf.rs: transform the magnitude, then restore its sign.
#[cfg(feature = "__internal_recon_hook")]
pub(super) fn srgb_to_linear(value: f32) -> f32 {
    let magnitude = value.abs();
    let linear = if magnitude <= 0.04045 {
        magnitude / 12.92
    } else {
        ((magnitude + 0.055) / 1.055).powf(2.4)
    };
    linear.copysign(value)
}
