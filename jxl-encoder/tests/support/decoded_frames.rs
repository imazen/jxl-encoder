//! Full multi-frame rendering through the pinned current jxl-rs API.
pub(crate) fn decode_current_frames(data: &[u8], label: &str) -> Vec<(Vec<f32>, Vec<Vec<f32>>)> {
    use jxl_current::api::{
        Event, JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer, JxlPixelFormat,
        ProfileLevel,
    };
    use jxl_current::image::{Image, Rect};
    let mut options = JxlDecoderOptions::default();
    options.max_profile_level = ProfileLevel::Main10;
    let mut decoder = JxlDecoder::new(options);
    let mut input = data;
    let mut color: Option<Image<f32>> = None;
    let mut extras: Vec<Image<f32>> = vec![];
    let mut frames = Vec::new();
    loop {
        let mut output = color.as_mut().map(|color| {
            let size = color.size();
            let mut output = vec![JxlOutputBuffer::from_image_rect_mut(
                color
                    .get_rect_mut(Rect {
                        origin: (0, 0),
                        size,
                    })
                    .into_raw(),
            )];
            for extra in &mut extras {
                let size = extra.size();
                output.push(JxlOutputBuffer::from_image_rect_mut(
                    extra
                        .get_rect_mut(Rect {
                            origin: (0, 0),
                            size,
                        })
                        .into_raw(),
                ));
            }
            output
        });
        let event = decoder
            .process(&mut input, output.as_deref_mut(), None)
            .unwrap_or_else(|e| panic!("jxl-rs 0.8 {label}: {e:?}"));
        drop(output);
        match event {
            Event::BasicInfo => {
                let (width, height) = decoder.basic_info().unwrap().size;
                let format = decoder.current_pixel_format().unwrap().clone();
                let channels = format.color_type.samples_per_pixel();
                let count = format.extra_channel_format.len();
                decoder
                    .set_pixel_format(JxlPixelFormat {
                        color_type: format.color_type,
                        color_data_format: Some(JxlDataFormat::f32()),
                        extra_channel_format: vec![Some(JxlDataFormat::f32()); count],
                    })
                    .unwrap();
                color = Some(Image::new((width * channels, height)).unwrap());
                extras = (0..count)
                    .map(|_| Image::new((width, height)).unwrap())
                    .collect();
            }
            Event::FrameHeader => {}
            Event::FrameComplete { .. } => {
                let color = color.as_ref().unwrap();
                let pixels: Vec<f32> = (0..color.size().1)
                    .flat_map(|y| color.row(y).to_vec())
                    .collect();
                let planes: Vec<Vec<f32>> = extras
                    .iter()
                    .map(|extra| {
                        (0..extra.size().1)
                            .flat_map(|y| extra.row(y).to_vec())
                            .collect()
                    })
                    .collect();
                assert!(
                    pixels.iter().all(|v| v.is_finite()),
                    "non-finite color: {label}"
                );
                assert!(
                    planes.iter().flatten().all(|v| v.is_finite()),
                    "non-finite extra: {label}"
                );
                frames.push((pixels, planes));
            }
            Event::Complete => return frames,
            Event::NeedMoreInput { .. } => assert!(!input.is_empty(), "truncated {label}"),
        }
    }
}
