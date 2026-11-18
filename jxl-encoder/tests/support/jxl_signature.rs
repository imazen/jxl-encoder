//! Format assertions shared by roundtrip tests and the CLI level regression.

/// Canonical signature (12), ftyp (20), and third-position Level 10 box (9).
pub(crate) const LEVEL_TEN_PREFIX: &[u8] =
    b"\0\0\0\x0cJXL \r\n\x87\n\0\0\0\x14ftypjxl \0\0\0\0jxl \0\0\0\x09jxll\x0a";

pub(crate) fn assert_signature(bytes: &[u8]) {
    assert!(
        bytes.starts_with(&[0xff, 0x0a]) || bytes.starts_with(LEVEL_TEN_PREFIX),
        "missing bare-codestream or Level 10 container signature"
    );
}
