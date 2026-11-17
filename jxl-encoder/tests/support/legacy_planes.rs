//! Supplemental rendering through the separately pinned jxl-rs 0.4 API.
pub(crate) fn decode(data: &[u8]) -> (Vec<f32>, Vec<Vec<f32>>) {
    use jxl::api::{
        JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer, JxlPixelFormat,
        ProcessingResult, states,
    };
    use jxl::image::{Image, Rect};
    let mut input = data;
    let initial = JxlDecoder::<states::Initialized>::new(JxlDecoderOptions::default());
    let ProcessingResult::Complete {
        result: mut decoder,
    } = initial.process(&mut input).unwrap()
    else {
        panic!("incomplete image header")
    };
    let (width, height) = decoder.basic_info().size;
    let format = decoder.current_pixel_format();
    let colors = format.color_type.samples_per_pixel();
    let count = format.extra_channel_format.len();
    decoder.set_pixel_format(JxlPixelFormat {
        color_type: format.color_type,
        color_data_format: Some(JxlDataFormat::f32()),
        extra_channel_format: vec![Some(JxlDataFormat::f32()); count],
    });
    let ProcessingResult::Complete { result: frame } = decoder.process(&mut input).unwrap() else {
        panic!("incomplete frame header")
    };
    let mut color = Image::<f32>::new((width * colors, height)).unwrap();
    let mut extras: Vec<_> = (0..count)
        .map(|_| Image::<f32>::new((width, height)).unwrap())
        .collect();
    let mut buffers = vec![JxlOutputBuffer::from_image_rect_mut(
        color
            .get_rect_mut(Rect {
                origin: (0, 0),
                size: (width * colors, height),
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
    let ProcessingResult::Complete { result: next } =
        frame.process(&mut input, &mut buffers).unwrap()
    else {
        panic!("incomplete frame pixels")
    };
    assert!(!next.has_more_frames());
    drop(buffers);
    let color = (0..height).flat_map(|y| color.row(y).to_vec()).collect();
    let extras = extras
        .iter()
        .map(|p| (0..height).flat_map(|y| p.row(y).to_vec()).collect())
        .collect();
    (color, extras)
}
