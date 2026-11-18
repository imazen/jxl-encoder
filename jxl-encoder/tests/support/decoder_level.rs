//! Keep reference-decoder conformance admission equal to the stream's declaration.
//! This changes no pixel/metadata expectation or decoder resource budget.
pub(crate) fn declared_level(bytes: &[u8]) -> u8 {
    if bytes.starts_with(&[0xff, 0x0a]) {
        return 5;
    }
    assert!(bytes.starts_with(&[0, 0, 0, 12, b'J', b'X', b'L', b' ', 13, 10, 135, 10]));
    let mut offset = 0;
    let mut level = 5;
    let mut found = false;
    while offset < bytes.len() {
        assert!(bytes.len() - offset >= 8);
        let short = u32::from_be_bytes(bytes[offset..offset + 4].try_into().unwrap());
        let (length, header) = match short {
            0 => (bytes.len() - offset, 8),
            1 => {
                assert!(bytes.len() - offset >= 16);
                (
                    usize::try_from(u64::from_be_bytes(
                        bytes[offset + 8..offset + 16].try_into().unwrap(),
                    ))
                    .unwrap(),
                    16,
                )
            }
            value => (value as usize, 8),
        };
        assert!(length >= header && length <= bytes.len() - offset);
        if &bytes[offset + 4..offset + 8] == b"jxll" {
            assert!(!found && length == header + 1);
            level = bytes[offset + header];
            assert!(matches!(level, 5 | 10));
            found = true;
        }
        offset += length;
    }
    level
}

pub(crate) fn options(bytes: &[u8]) -> zensim_decoder::api::JxlDecoderOptions {
    use zensim_decoder::api::{JxlCodestreamLevel, JxlDecoderLimits, JxlDecoderOptions};
    let level = match declared_level(bytes) {
        5 => JxlCodestreamLevel::Level5,
        10 => JxlCodestreamLevel::Level10,
        _ => unreachable!(),
    };
    JxlDecoderOptions::default()
        .with_limits(JxlDecoderLimits::default().with_max_codestream_level(level))
}
