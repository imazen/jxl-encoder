// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! JXL container format (ISOBMFF boxes) for metadata embedding.
//!
//! When EXIF or XMP metadata is present, the codestream must be wrapped in a
//! container format. The container uses ISOBMFF-style boxes:
//!
//! ```text
//! [JXL signature box - 12 bytes]
//! [ftyp box - 20 bytes]
//! [jxlc box - 8 + codestream_len bytes]
//! [Exif box - 8 + 4 + exif_len bytes] (optional)
//! [xml  box - 8 + xmp_len bytes] (optional)
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
    // Calculate total size for pre-allocation
    let header_size = JXL_CONTAINER_SIGNATURE.len() + FTYP_BOX.len(); // 32
    let jxlc_size = 8 + codestream.len();
    let exif_size = exif.map_or(0, |e| 8 + 4 + e.len()); // 4-byte Tiff header offset prefix
    let xmp_size = xmp.map_or(0, |x| 8 + x.len());
    let total = header_size + jxlc_size + exif_size + xmp_size;

    let mut out = Vec::with_capacity(total);

    // Container header
    out.extend_from_slice(&JXL_CONTAINER_SIGNATURE);
    out.extend_from_slice(&FTYP_BOX);

    // jxlc box (codestream)
    write_box(&mut out, b"jxlc", codestream);

    // Exif box (with 4-byte Tiff header offset prefix, always 0)
    if let Some(exif_data) = exif {
        let box_size = (8 + 4 + exif_data.len()) as u32;
        out.extend_from_slice(&box_size.to_be_bytes());
        out.extend_from_slice(b"Exif");
        out.extend_from_slice(&[0u8; 4]); // Tiff header offset (always 0)
        out.extend_from_slice(exif_data);
    }

    // xml box (raw XMP data)
    if let Some(xmp_data) = xmp {
        write_box(&mut out, b"xml ", xmp_data);
    }

    out
}

/// Write an ISOBMFF box: 4-byte big-endian size + 4-byte type + payload.
fn write_box(out: &mut Vec<u8>, box_type: &[u8; 4], payload: &[u8]) {
    let box_size = (8 + payload.len()) as u32;
    out.extend_from_slice(&box_size.to_be_bytes());
    out.extend_from_slice(box_type);
    out.extend_from_slice(payload);
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
}
