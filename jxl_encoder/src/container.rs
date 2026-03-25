// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! JXL container format (ISOBMFF boxes) for metadata embedding.
//!
//! When EXIF/XMP metadata is present, the codestream must be wrapped in a
//! container format. The container uses ISOBMFF-style boxes.
//!
//! Standard layout (single `jxlc` codestream box):
//! ```text
//! [JXL signature] [ftyp] [jxlc] [Exif?] [xml?]
//! ```
//!
//! JPEG lossless reencoding layout (split `jxlp` boxes with `jbrd` between):
//! ```text
//! [JXL signature] [ftyp] [jxlp part 0] [jbrd] [jxlp part 1 (last)] [Exif?] [xml?]
//! ```

/// JXL container signature box (12 bytes).
const JXL_CONTAINER_SIGNATURE: [u8; 12] = [
    0x00, 0x00, 0x00, 0x0C, // box size = 12
    b'J', b'X', b'L', b' ', // box type
    0x0D, 0x0A, 0x87, 0x0A, // JXL container magic
];

/// ftyp box (20 bytes): brand = "jxl ", minor_version = 0, compatible = "jxl ".
const FTYP_BOX: [u8; 20] = [
    0x00, 0x00, 0x00, 0x14, // box size = 20
    b'f', b't', b'y', b'p', // box type
    b'j', b'x', b'l', b' ', // major brand
    0x00, 0x00, 0x00, 0x00, // minor version
    b'j', b'x', b'l', b' ', // compatible brand
];

/// Wraps a JXL codestream in a container with optional EXIF and XMP metadata.
///
/// Returns bare codestream bytes wrapped in ISOBMFF boxes. The codestream
/// goes into a `jxlc` box, EXIF into an `Exif` box (with 4-byte Tiff offset
/// prefix), and XMP into an `xml ` box.
pub fn wrap_in_container(codestream: &[u8], exif: Option<&[u8]>, xmp: Option<&[u8]>) -> Vec<u8> {
    wrap_in_container_with_jbrd(codestream, None, exif, xmp)
}

/// Wraps a JXL codestream in a container with optional JBRD, EXIF, and XMP.
///
/// The `jbrd` box contains JPEG Bitstream Reconstruction Data needed for
/// byte-exact JPEG reconstruction from the JXL file. When present, the box
/// order is: signature, ftyp, jxlc, jbrd, Exif (optional), xml (optional).
pub fn wrap_in_container_with_jbrd(
    codestream: &[u8],
    jbrd: Option<&[u8]>,
    exif: Option<&[u8]>,
    xmp: Option<&[u8]>,
) -> Vec<u8> {
    // Calculate total size for pre-allocation
    let header_size = JXL_CONTAINER_SIGNATURE.len() + FTYP_BOX.len(); // 32
    let jxlc_size = 8 + codestream.len();
    let jbrd_size = jbrd.map_or(0, |j| 8 + j.len());
    let exif_size = exif.map_or(0, |e| 8 + 4 + e.len()); // 4-byte Tiff header offset prefix
    let xmp_size = xmp.map_or(0, |x| 8 + x.len());
    let total = header_size + jxlc_size + jbrd_size + exif_size + xmp_size;

    let mut out = Vec::with_capacity(total);

    // Container header
    out.extend_from_slice(&JXL_CONTAINER_SIGNATURE);
    out.extend_from_slice(&FTYP_BOX);

    // jxlc box (codestream)
    write_box(&mut out, b"jxlc", codestream);

    // jbrd box (JPEG Bitstream Reconstruction Data)
    if let Some(jbrd_data) = jbrd {
        write_box(&mut out, b"jbrd", jbrd_data);
    }

    // Exif box (with 4-byte Tiff header offset prefix, always 0)
    if let Some(exif_data) = exif {
        write_exif_box(&mut out, exif_data);
    }

    // xml box (raw XMP data)
    if let Some(xmp_data) = xmp {
        write_box(&mut out, b"xml ", xmp_data);
    }

    out
}

/// Wraps a split codestream in a container with jxlp boxes, jbrd, and metadata.
///
/// The codestream is split into two `jxlp` boxes with the `jbrd` box between
/// them. This matches libjxl's JPEG lossless transcoding container format:
/// ```text
/// [JXL signature] [ftyp] [jxlp part 0] [jbrd] [jxlp part 1 (last)] [Exif?] [xml?]
/// ```
/// Each `jxlp` box has a 4-byte BE counter where bit 31 marks the last part.
#[cfg(feature = "jpeg-reencoding")]
pub fn wrap_in_container_jxlp(
    cs_part1: &[u8],
    cs_part2: &[u8],
    jbrd: &[u8],
    exif: Option<&[u8]>,
    xmp: Option<&[u8]>,
) -> Vec<u8> {
    let header_size = JXL_CONTAINER_SIGNATURE.len() + FTYP_BOX.len(); // 32
    // jxlp box: 8-byte box header + 4-byte counter + payload
    let jxlp1_size = 8 + 4 + cs_part1.len();
    let jxlp2_size = 8 + 4 + cs_part2.len();
    let jbrd_size = 8 + jbrd.len();
    let exif_size = exif.map_or(0, |e| 8 + 4 + e.len());
    let xmp_size = xmp.map_or(0, |x| 8 + x.len());
    let total = header_size + jxlp1_size + jbrd_size + jxlp2_size + exif_size + xmp_size;

    let mut out = Vec::with_capacity(total);

    // Container header
    out.extend_from_slice(&JXL_CONTAINER_SIGNATURE);
    out.extend_from_slice(&FTYP_BOX);

    // First jxlp box: file header (counter = 0, not last)
    write_jxlp_box(&mut out, 0, false, cs_part1);

    // jbrd box
    write_box(&mut out, b"jbrd", jbrd);

    // Second jxlp box: frame data (counter = 1, last)
    write_jxlp_box(&mut out, 1, true, cs_part2);

    // Exif box
    if let Some(exif_data) = exif {
        write_exif_box(&mut out, exif_data);
    }

    // xml box
    if let Some(xmp_data) = xmp {
        write_box(&mut out, b"xml ", xmp_data);
    }

    out
}

/// Write a jxlp (partial codestream) box.
/// Counter format: bits 0-30 = sequence number, bit 31 = last part flag.
#[allow(dead_code)]
fn write_jxlp_box(out: &mut Vec<u8>, sequence: u32, is_last: bool, data: &[u8]) {
    let total_size = 8u64 + 4 + data.len() as u64;
    if total_size <= u32::MAX as u64 {
        out.extend_from_slice(&(total_size as u32).to_be_bytes());
        out.extend_from_slice(b"jxlp");
    } else {
        let extended_size = 16u64 + 4 + data.len() as u64;
        out.extend_from_slice(&1u32.to_be_bytes());
        out.extend_from_slice(b"jxlp");
        out.extend_from_slice(&extended_size.to_be_bytes());
    }
    let counter = if is_last {
        sequence | 0x8000_0000
    } else {
        sequence
    };
    out.extend_from_slice(&counter.to_be_bytes());
    out.extend_from_slice(data);
}

/// Write an Exif box: box header + 4-byte Tiff offset (always 0) + EXIF data.
fn write_exif_box(out: &mut Vec<u8>, exif_data: &[u8]) {
    // Exif box payload = 4-byte Tiff offset + EXIF data
    let payload_size = 4u64 + exif_data.len() as u64;
    let total_size = 8u64 + payload_size;
    if total_size <= u32::MAX as u64 {
        out.extend_from_slice(&(total_size as u32).to_be_bytes());
        out.extend_from_slice(b"Exif");
    } else {
        let extended_size = 16u64 + payload_size;
        out.extend_from_slice(&1u32.to_be_bytes());
        out.extend_from_slice(b"Exif");
        out.extend_from_slice(&extended_size.to_be_bytes());
    }
    out.extend_from_slice(&[0u8; 4]); // Tiff header offset (always 0)
    out.extend_from_slice(exif_data);
}

/// Write an ISOBMFF box: 4-byte big-endian size + 4-byte type + payload.
///
/// For payloads > ~4GB, uses extended 64-bit box header (size field = 1,
/// followed by 8-byte extended size). Matches libjxl's box format.
fn write_box(out: &mut Vec<u8>, box_type: &[u8; 4], payload: &[u8]) {
    let total_size = 8u64 + payload.len() as u64;
    if total_size <= u32::MAX as u64 {
        out.extend_from_slice(&(total_size as u32).to_be_bytes());
        out.extend_from_slice(box_type);
    } else {
        // Extended box header: size=1 signals 64-bit extended size follows
        let extended_size = 16u64 + payload.len() as u64;
        out.extend_from_slice(&1u32.to_be_bytes()); // size=1 means extended
        out.extend_from_slice(box_type);
        out.extend_from_slice(&extended_size.to_be_bytes());
    }
    out.extend_from_slice(payload);
}

/// Returns `true` if `data` begins with the JXL container signature.
///
/// JXL files may be either bare codestreams (starting with `0xFF0A`) or
/// container-format files (starting with the 12-byte ISOBMFF signature box).
#[must_use]
pub fn is_container(data: &[u8]) -> bool {
    data.len() >= 12 && data[..12] == JXL_CONTAINER_SIGNATURE
}

/// Returns `true` if `data` begins with the bare JXL codestream signature (`0xFF0A`).
#[must_use]
pub fn is_bare_codestream(data: &[u8]) -> bool {
    data.len() >= 2 && data[0] == 0xFF && data[1] == 0x0A
}

/// Append a `jhgm` box to JXL data (container or bare codestream).
///
/// `jhgm_payload` is the raw gain map bundle payload (ISO 21496-1).
///
/// If `jxl_data` is already a container (starts with JXL signature), the
/// `jhgm` box is appended at the end. If `jxl_data` is a bare codestream,
/// it is first wrapped in a container (signature + ftyp + jxlc), then the
/// jhgm box is appended.
#[must_use]
pub fn append_gain_map_box(jxl_data: &[u8], jhgm_payload: &[u8]) -> Vec<u8> {
    if is_container(jxl_data) {
        // Already a container — append jhgm box at the end.
        let jhgm_box_size = 8 + jhgm_payload.len();
        let mut out = Vec::with_capacity(jxl_data.len() + jhgm_box_size);
        out.extend_from_slice(jxl_data);
        write_box(&mut out, b"jhgm", jhgm_payload);
        out
    } else {
        // Bare codestream — wrap in container first, then append jhgm.
        let header_size = JXL_CONTAINER_SIGNATURE.len() + FTYP_BOX.len();
        let jxlc_size = 8 + jxl_data.len();
        let jhgm_box_size = 8 + jhgm_payload.len();
        let total = header_size + jxlc_size + jhgm_box_size;

        let mut out = Vec::with_capacity(total);
        out.extend_from_slice(&JXL_CONTAINER_SIGNATURE);
        out.extend_from_slice(&FTYP_BOX);
        write_box(&mut out, b"jxlc", jxl_data);
        write_box(&mut out, b"jhgm", jhgm_payload);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_container_no_metadata() {
        let codestream = b"\xFF\x0A\x00\x00"; // fake codestream
        let result = wrap_in_container(codestream, None, None);
        // Header (32) + jxlc box (8 + 4)
        assert_eq!(result.len(), 32 + 8 + 4);
        // Starts with container signature
        assert_eq!(&result[..4], &[0, 0, 0, 0x0C]);
        assert_eq!(&result[4..8], b"JXL ");
    }

    #[test]
    fn test_container_with_exif() {
        let codestream = b"\xFF\x0A";
        let exif = b"Exif\x00\x00MM\x00\x2a"; // minimal EXIF
        let result = wrap_in_container(codestream, Some(exif), None);
        // Header (32) + jxlc (8+2) + Exif (8+4+10)
        assert_eq!(result.len(), 32 + 10 + 22);
        // Find Exif box
        let exif_start = 32 + 10; // after header + jxlc
        assert_eq!(&result[exif_start + 4..exif_start + 8], b"Exif");
        // Tiff offset prefix (4 zero bytes)
        assert_eq!(&result[exif_start + 8..exif_start + 12], &[0, 0, 0, 0]);
    }

    #[test]
    fn test_container_with_xmp() {
        let codestream = b"\xFF\x0A";
        let xmp = b"<?xpacket begin='...'?>";
        let result = wrap_in_container(codestream, None, Some(xmp));
        // Header (32) + jxlc (8+2) + xml (8+23)
        assert_eq!(result.len(), 32 + 10 + 31);
        let xml_start = 32 + 10;
        assert_eq!(&result[xml_start + 4..xml_start + 8], b"xml ");
    }

    #[test]
    fn test_container_with_both() {
        let codestream = b"\xFF\x0A";
        let exif = b"EX";
        let xmp = b"XM";
        let result = wrap_in_container(codestream, Some(exif), Some(xmp));
        // Header (32) + jxlc (8+2) + Exif (8+4+2) + xml (8+2)
        assert_eq!(result.len(), 32 + 10 + 14 + 10);
    }

    #[test]
    fn test_is_container() {
        assert!(is_container(&JXL_CONTAINER_SIGNATURE));
        assert!(!is_container(&[0xFF, 0x0A, 0x00]));
        assert!(!is_container(&[0x00; 4]));
    }

    #[test]
    fn test_is_bare_codestream() {
        assert!(is_bare_codestream(&[0xFF, 0x0A]));
        assert!(is_bare_codestream(&[0xFF, 0x0A, 0x00, 0x01]));
        assert!(!is_bare_codestream(&[0xFF, 0x0B]));
        assert!(!is_bare_codestream(&[0x00]));
    }

    #[test]
    fn test_append_gain_map_to_bare_codestream() {
        let codestream = b"\xFF\x0A\x00\x01\x02\x03";
        let jhgm_payload = b"\x00\x00\x03\x01\x02\x03\x00\x00\x00\x00\x00\xFF\x0A";

        let result = append_gain_map_box(codestream, jhgm_payload);

        // Should start with container signature
        assert!(is_container(&result));
        assert_eq!(&result[..12], &JXL_CONTAINER_SIGNATURE);
        assert_eq!(&result[12..32], &FTYP_BOX);

        // jxlc box at offset 32
        let jxlc_size = u32::from_be_bytes([result[32], result[33], result[34], result[35]]);
        assert_eq!(jxlc_size as usize, 8 + codestream.len());
        assert_eq!(&result[36..40], b"jxlc");
        assert_eq!(&result[40..40 + codestream.len()], codestream);

        // jhgm box follows jxlc
        let jhgm_offset = 40 + codestream.len();
        let jhgm_size = u32::from_be_bytes([
            result[jhgm_offset],
            result[jhgm_offset + 1],
            result[jhgm_offset + 2],
            result[jhgm_offset + 3],
        ]);
        assert_eq!(jhgm_size as usize, 8 + jhgm_payload.len());
        assert_eq!(&result[jhgm_offset + 4..jhgm_offset + 8], b"jhgm");
        assert_eq!(&result[jhgm_offset + 8..], jhgm_payload);
    }

    #[test]
    fn test_append_gain_map_to_existing_container() {
        // Build a minimal container
        let codestream = b"\xFF\x0A\x00";
        let mut container = Vec::new();
        container.extend_from_slice(&JXL_CONTAINER_SIGNATURE);
        container.extend_from_slice(&FTYP_BOX);
        write_box(&mut container, b"jxlc", codestream);

        let jhgm_payload = b"\x00\x00\x01\xAA\x00\x00\x00\x00\x00\xFF\x0A";
        let result = append_gain_map_box(&container, jhgm_payload);

        // Original container bytes preserved
        assert_eq!(&result[..container.len()], container.as_slice());

        // jhgm box appended at the end
        let jhgm_offset = container.len();
        assert_eq!(&result[jhgm_offset + 4..jhgm_offset + 8], b"jhgm");
        assert_eq!(&result[jhgm_offset + 8..], jhgm_payload);
    }

    #[test]
    fn test_write_box_small() {
        let mut out = Vec::new();
        write_box(&mut out, b"test", b"hello");
        assert_eq!(out.len(), 8 + 5);
        let size = u32::from_be_bytes([out[0], out[1], out[2], out[3]]);
        assert_eq!(size, 13);
        assert_eq!(&out[4..8], b"test");
        assert_eq!(&out[8..], b"hello");
    }

    #[test]
    fn test_write_box_empty_payload() {
        let mut out = Vec::new();
        write_box(&mut out, b"emty", b"");
        assert_eq!(out.len(), 8);
        let size = u32::from_be_bytes([out[0], out[1], out[2], out[3]]);
        assert_eq!(size, 8);
        assert_eq!(&out[4..8], b"emty");
    }
}
