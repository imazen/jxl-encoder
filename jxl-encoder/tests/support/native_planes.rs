//! Full rendering through the pinned native decoder's public buffers.
pub(crate) fn decode(data: &[u8]) -> (Vec<f32>, Vec<Vec<f32>>) {
    let mut frames = decode_frames(data);
    assert_eq!(frames.len(), 1, "one displayed frame");
    frames.pop().unwrap()
}

pub(crate) fn decode_frames(data: &[u8]) -> Vec<(Vec<f32>, Vec<Vec<f32>>)> {
    use zensim_decoder::api::{
        JxlCodestreamLevel, JxlDataFormat, JxlDecoder, JxlDecoderLimits, JxlDecoderOptions,
        JxlOutputBuffer, JxlPixelFormat, ProcessingResult, states,
    };
    let mut input = data;
    let initial = JxlDecoder::<states::Initialized>::new(JxlDecoderOptions::default().with_limits(
        JxlDecoderLimits::default().with_max_codestream_level(JxlCodestreamLevel::Level10),
    ));
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
    let mut frames = Vec::new();
    loop {
        let ProcessingResult::Complete { result: frame } = decoder.process(&mut input).unwrap()
        else {
            panic!("incomplete frame header")
        };
        let mut color = vec![0_u8; width * height * colors * 4];
        let mut extras = vec![vec![0_u8; width * height * 4]; count];
        let mut buffers = vec![JxlOutputBuffer::new(&mut color, height, width * colors * 4)];
        for extra in &mut extras {
            buffers.push(JxlOutputBuffer::new(extra, height, width * 4));
        }
        let ProcessingResult::Complete { result: next } =
            frame.process(&mut input, &mut buffers).unwrap()
        else {
            panic!("incomplete frame pixels")
        };
        drop(buffers);
        let unpack = |bytes: &[u8]| -> Vec<f32> {
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|v| f32::from_ne_bytes(*v))
                .collect()
        };
        let color = unpack(&color);
        let extras: Vec<_> = extras.iter().map(|p| unpack(p)).collect();
        assert!(color.iter().all(|v| v.is_finite()));
        assert!(extras.iter().flatten().all(|v| v.is_finite()));
        frames.push((color, extras));
        decoder = next;
        if !decoder.has_more_frames() {
            return frames;
        }
    }
}
