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

/// `jxll` (level) box — fixed 9 bytes: 4-byte BE size (0x09) +
/// 4-byte type (`jxll`) + 1 byte payload (the codestream level).
/// Mirrors `kLevelBoxHeader` in libjxl `lib/jxl/encode_internal.h:150`.
const JXLL_BOX_HEADER: [u8; 8] = [
    0x00, 0x00, 0x00, 0x09, // box size = 9 (header + 1 byte payload)
    b'j', b'x', b'l', b'l',
];

/// JPEG XL codestream level required by an image with the given
/// properties. Returns `5` for baseline-compatible inputs, `10` for
/// inputs that exceed any level-5 cap but stay within the level-10
/// caps, and `None` for inputs that overflow level 10 (currently
/// unencodable).
///
/// Mirrors libjxl `VerifyLevelSettings` in `lib/jxl/encode.cc:550`.
/// `has_black_channel` is `true` when one of the extra channels is a
/// CMYK `Black` plane (level-5 forbids it). The encoder does not yet
/// surface 32-bit modular buffers, so the corresponding level-5 cap
/// is left to the encoder.
#[must_use]
pub fn compute_codestream_level(
    width: u32,
    height: u32,
    num_extra_channels: u32,
    has_black_channel: bool,
    icc_size: u64,
) -> Option<u8> {
    let w = u64::from(width);
    let h = u64::from(height);

    // Level 10 hard caps (encode.cc:563-575).
    if w > (1u64 << 30) || h > (1u64 << 30) {
        return None;
    }
    if w.saturating_mul(h) > (1u64 << 40) {
        return None;
    }
    if icc_size > (1u64 << 28) {
        return None;
    }
    if num_extra_channels > 256 {
        return None;
    }

    // Level 5 caps (encode.cc:579-601). Any single violation bumps
    // to level 10.
    if w > (1u64 << 18) || h > (1u64 << 18) {
        return Some(10);
    }
    if w.saturating_mul(h) > (1u64 << 28) {
        return Some(10);
    }
    if icc_size > (1u64 << 22) {
        return Some(10);
    }
    if num_extra_channels > 4 {
        return Some(10);
    }
    if has_black_channel {
        return Some(10);
    }

    Some(5)
}

/// True when the level forces a container wrapper (level != 5).
/// Mirrors the `codestream_level` clause of libjxl `MustUseContainer`
/// in `encode_internal.h:662-665`.
#[must_use]
pub fn level_requires_container(level: u8) -> bool {
    level != 5
}

/// Push the `jxll` (level) box (9 bytes) onto `out` when `level != 5`.
/// Level 5 is the baseline — no box is emitted so the byte layout for
/// the overwhelming majority of images is unchanged.
fn write_level_box(out: &mut Vec<u8>, level: u8) {
    if level == 5 {
        return;
    }
    out.extend_from_slice(&JXLL_BOX_HEADER);
    out.push(level);
}

/// Wraps a JXL codestream in a container with optional EXIF and XMP metadata.
///
/// Returns bare codestream bytes wrapped in ISOBMFF boxes. The codestream
/// goes into a `jxlc` box, EXIF into an `Exif` box (with 4-byte Tiff offset
/// prefix), and XMP into an `xml ` box.
pub fn wrap_in_container(codestream: &[u8], exif: Option<&[u8]>, xmp: Option<&[u8]>) -> Vec<u8> {
    wrap_in_container_with_jbrd_and_level_and_jumbf(codestream, None, exif, xmp, 5, None)
}

/// Like [`wrap_in_container`] but emits a `jxll` (codestream level) box
/// directly after `ftyp` when `level != 5`. Use [`compute_codestream_level`]
/// to derive `level` from image dimensions, ICC size, and extra channels.
pub fn wrap_in_container_with_level(
    codestream: &[u8],
    exif: Option<&[u8]>,
    xmp: Option<&[u8]>,
    level: u8,
) -> Vec<u8> {
    wrap_in_container_with_jbrd_and_level_and_jumbf(codestream, None, exif, xmp, level, None)
}

/// Wraps a JXL codestream in a container with optional EXIF, XMP, and JUMBF
/// metadata. The JUMBF payload (ISO 19566-5, used by C2PA / Content
/// Authenticity Initiative) lands in a `jumb` box appended after the
/// `Exif`/`xml ` boxes. Caller supplies the JUMBF superbox contents
/// verbatim — no ISO BMFF header included.
pub fn wrap_in_container_with_jumbf(
    codestream: &[u8],
    exif: Option<&[u8]>,
    xmp: Option<&[u8]>,
    jumbf: Option<&[u8]>,
) -> Vec<u8> {
    wrap_in_container_with_jbrd_and_level_and_jumbf(codestream, None, exif, xmp, 5, jumbf)
}

/// Like [`wrap_in_container_with_jumbf`] but also emits a `jxll`
/// codestream-level box when `level != 5`.
pub fn wrap_in_container_with_level_and_jumbf(
    codestream: &[u8],
    exif: Option<&[u8]>,
    xmp: Option<&[u8]>,
    level: u8,
    jumbf: Option<&[u8]>,
) -> Vec<u8> {
    wrap_in_container_with_jbrd_and_level_and_jumbf(codestream, None, exif, xmp, level, jumbf)
}

/// Wraps a JXL codestream like [`wrap_in_container`], but Brotli-compresses
/// EXIF / XMP metadata into `brob` boxes when it saves bytes (closes #15).
///
/// `brotli_quality` (0-11; libjxl default 4) controls the Brotli encoder
/// effort. Each metadata blob is independently evaluated: if the brob
/// (compressed) version would be smaller, it lands in a `brob` box;
/// otherwise the function falls back to the uncompressed Exif/xml box.
/// Sub-500-byte metadata typically falls back due to Brotli framing
/// overhead.
///
/// The codestream itself is never compressed (always written as `jxlc`).
/// Compression for `jbrd` is handled separately by the
/// `jpeg-reencoding` feature.
#[cfg(feature = "brotli-metadata")]
pub fn wrap_in_container_with_brob(
    codestream: &[u8],
    exif: Option<&[u8]>,
    xmp: Option<&[u8]>,
    brotli_quality: u32,
) -> Vec<u8> {
    wrap_in_container_with_brob_and_level_and_jumbf(codestream, exif, xmp, None, brotli_quality, 5)
}

/// Like [`wrap_in_container_with_brob`] but emits a `jxll` (codestream
/// level) box directly after `ftyp` when `level != 5`.
#[cfg(feature = "brotli-metadata")]
pub fn wrap_in_container_with_brob_and_level(
    codestream: &[u8],
    exif: Option<&[u8]>,
    xmp: Option<&[u8]>,
    brotli_quality: u32,
    level: u8,
) -> Vec<u8> {
    wrap_in_container_with_brob_and_level_and_jumbf(
        codestream,
        exif,
        xmp,
        None,
        brotli_quality,
        level,
    )
}

/// Like [`wrap_in_container_with_brob`] but also routes an optional JUMBF
/// (`jumb`) box through the same brob compress-if-smaller path. JUMBF
/// payloads carrying C2PA manifests can be sizable (XMP-shaped XML +
/// embedded cose signatures), so brob savings are worth attempting.
#[cfg(feature = "brotli-metadata")]
pub fn wrap_in_container_with_brob_and_jumbf(
    codestream: &[u8],
    exif: Option<&[u8]>,
    xmp: Option<&[u8]>,
    jumbf: Option<&[u8]>,
    brotli_quality: u32,
) -> Vec<u8> {
    wrap_in_container_with_brob_and_level_and_jumbf(codestream, exif, xmp, jumbf, brotli_quality, 5)
}

/// Full-fat brob path: accepts JUMBF + a non-default codestream level.
/// All other `wrap_in_container_with_brob*` entry points delegate here.
#[cfg(feature = "brotli-metadata")]
pub fn wrap_in_container_with_brob_and_level_and_jumbf(
    codestream: &[u8],
    exif: Option<&[u8]>,
    xmp: Option<&[u8]>,
    jumbf: Option<&[u8]>,
    brotli_quality: u32,
    level: u8,
) -> Vec<u8> {
    let header_size = JXL_CONTAINER_SIGNATURE.len()
        + FTYP_BOX.len()
        + if level == 5 {
            0
        } else {
            JXLL_BOX_HEADER.len() + 1
        };
    let jxlc_size = 8 + codestream.len();
    // Worst-case sizes for pre-allocation: assume each metadata blob
    // either compresses (8 + 4 + payload.len) or stays uncompressed
    // (8 + 4 + payload.len for exif, 8 + payload.len for xmp/jumb).
    // The uncompressed Exif size already includes the 4-byte Tiff prefix.
    let exif_size = exif.map_or(0, |e| 8 + 4 + e.len());
    let xmp_size = xmp.map_or(0, |x| 8 + x.len());
    let jumbf_size = jumbf.map_or(0, |j| 8 + j.len());
    let total = header_size + jxlc_size + exif_size + xmp_size + jumbf_size;
    let mut out = Vec::with_capacity(total);

    out.extend_from_slice(&JXL_CONTAINER_SIGNATURE);
    out.extend_from_slice(&FTYP_BOX);
    write_level_box(&mut out, level);
    write_box(&mut out, b"jxlc", codestream);

    // Exif: build the Exif payload (4-byte Tiff offset + EXIF data),
    // then try brob. The brob's "original type" is the box type that
    // would have been used, INCLUDING the 4-byte Tiff prefix in the
    // payload — matches libjxl's brob-Exif convention.
    if let Some(exif_data) = exif {
        let mut exif_payload = Vec::with_capacity(4 + exif_data.len());
        exif_payload.extend_from_slice(&[0u8; 4]); // Tiff offset
        exif_payload.extend_from_slice(exif_data);
        if !try_write_brob_box(&mut out, b"Exif", &exif_payload, brotli_quality) {
            write_exif_box(&mut out, exif_data);
        }
    }
    // XMP: payload is the raw XML; type is `xml ` (note trailing space).
    if let Some(xmp_data) = xmp
        && !try_write_brob_box(&mut out, b"xml ", xmp_data, brotli_quality)
    {
        write_box(&mut out, b"xml ", xmp_data);
    }
    // JUMBF: payload is the raw JUMBF superbox bytes; original type is
    // `jumb`. Nested brob inside jumb is forbidden per spec, but the
    // outer wrap of jumb in brob is allowed (libjxl emits it via the
    // same JxlEncoderAddBox/compress_box path).
    if let Some(jumbf_data) = jumbf
        && !try_write_brob_box(&mut out, b"jumb", jumbf_data, brotli_quality)
    {
        write_box(&mut out, b"jumb", jumbf_data);
    }
    out
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
    wrap_in_container_with_jbrd_and_level_and_jumbf(codestream, jbrd, exif, xmp, 5, None)
}

/// Like [`wrap_in_container_with_jbrd`] but emits a `jxll` (codestream
/// level) box directly after `ftyp` when `level != 5`.
pub fn wrap_in_container_with_jbrd_and_level(
    codestream: &[u8],
    jbrd: Option<&[u8]>,
    exif: Option<&[u8]>,
    xmp: Option<&[u8]>,
    level: u8,
) -> Vec<u8> {
    wrap_in_container_with_jbrd_and_level_and_jumbf(codestream, jbrd, exif, xmp, level, None)
}

/// Wraps a JXL codestream with optional JBRD, EXIF, XMP, and JUMBF
/// boxes, plus a `jxll` codestream-level box when `level != 5`.
///
/// Box order: signature, ftyp, jxll? (level != 5), jxlc, jbrd?, Exif?,
/// xml?, jumb?. Matches libjxl's ordering — JUMBF (C2PA / Content
/// Authenticity Initiative, ISO 19566-5) is appended after the standard
/// metadata boxes so legacy readers that stop at the first unknown box
/// still see the codestream.
pub fn wrap_in_container_with_jbrd_and_level_and_jumbf(
    codestream: &[u8],
    jbrd: Option<&[u8]>,
    exif: Option<&[u8]>,
    xmp: Option<&[u8]>,
    level: u8,
    jumbf: Option<&[u8]>,
) -> Vec<u8> {
    // Calculate total size for pre-allocation
    let level_size = if level == 5 {
        0
    } else {
        JXLL_BOX_HEADER.len() + 1
    };
    let header_size = JXL_CONTAINER_SIGNATURE.len() + FTYP_BOX.len() + level_size; // 32 (+9)
    let jxlc_size = 8 + codestream.len();
    let jbrd_size = jbrd.map_or(0, |j| 8 + j.len());
    let exif_size = exif.map_or(0, |e| 8 + 4 + e.len()); // 4-byte Tiff header offset prefix
    let xmp_size = xmp.map_or(0, |x| 8 + x.len());
    let jumbf_size = jumbf.map_or(0, |j| 8 + j.len());
    let total = header_size + jxlc_size + jbrd_size + exif_size + xmp_size + jumbf_size;

    let mut out = Vec::with_capacity(total);

    // Container header
    out.extend_from_slice(&JXL_CONTAINER_SIGNATURE);
    out.extend_from_slice(&FTYP_BOX);
    write_level_box(&mut out, level);

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

    // jumb box (JUMBF / C2PA metadata, ISO 19566-5). Caller-provided
    // payload is the JUMBF superbox contents verbatim; we do not
    // validate or interpret it (third-party libraries like c2pa-rs
    // produce conformant payloads).
    if let Some(jumbf_data) = jumbf {
        write_box(&mut out, b"jumb", jumbf_data);
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
    wrap_in_container_jxlp_with_level(cs_part1, cs_part2, jbrd, exif, xmp, 5)
}

/// Like [`wrap_in_container_jxlp`] but emits a `jxll` (codestream
/// level) box directly after `ftyp` when `level != 5`.
#[cfg(feature = "jpeg-reencoding")]
pub fn wrap_in_container_jxlp_with_level(
    cs_part1: &[u8],
    cs_part2: &[u8],
    jbrd: &[u8],
    exif: Option<&[u8]>,
    xmp: Option<&[u8]>,
    level: u8,
) -> Vec<u8> {
    let level_size = if level == 5 {
        0
    } else {
        JXLL_BOX_HEADER.len() + 1
    };
    let header_size = JXL_CONTAINER_SIGNATURE.len() + FTYP_BOX.len() + level_size;
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
    write_level_box(&mut out, level);

    // First jxlp box: file header (counter = 0, not last)
    write_jxlp_box(&mut out, 0, false, cs_part1);

    // jbrd box
    write_box(&mut out, b"jbrd", jbrd);

    // Second jxlp box: frame data (counter = 1, last)
    write_jxlp_box(&mut out, 1, true, cs_part2);

    // Exif box — try brob first, fall back to raw if brotli doesn't help.
    //
    // For XMP this is usually a big win (2-6× smaller — XMP is XML, very
    // redundant). For Exif it's a marginal win because Exif is binary and
    // already fairly compact, but `try_write_brob_box` only succeeds when
    // brob is strictly smaller, so the worst case is "no change".
    //
    // libjxl's default `jpeg_compress_boxes = true` uses brotli quality 4
    // — good ratio at low CPU. Matched here.
    const JPEG_BROB_QUALITY: u32 = 4;
    if let Some(exif_data) = exif {
        // Exif box on the JXL wire carries a 4-byte big-endian "TIFF offset"
        // prefix before the actual TIFF payload (= 0 when the TIFF starts
        // right after the prefix). We build that prefixed payload here so
        // the brob's `inner` payload, after brotli-decompression, exactly
        // matches what a non-brob `Exif` box would have stored — a clean
        // round-trip through djxl.
        let mut prefixed_exif = Vec::with_capacity(4 + exif_data.len());
        prefixed_exif.extend_from_slice(&[0, 0, 0, 0]);
        prefixed_exif.extend_from_slice(exif_data);
        if !try_write_brob_box(&mut out, b"Exif", &prefixed_exif, JPEG_BROB_QUALITY) {
            write_exif_box(&mut out, exif_data);
        }
    }

    // xml box (XMP) — try brob first, fall back to raw if brotli is bigger.
    if let Some(xmp_data) = xmp {
        if !try_write_brob_box(&mut out, b"xml ", xmp_data, JPEG_BROB_QUALITY) {
            write_box(&mut out, b"xml ", xmp_data);
        }
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

/// Brotli-compress `payload` and emit a `brob` box (closes #15).
///
/// Format: `[box_size] [b"brob"] [original_type: 4 bytes]
/// [brotli_compressed_payload]`. Decoders read the first 4 bytes of
/// the payload to learn the original box type, then Brotli-decompress
/// the rest. libjxl quality default is 4 (good balance for typical
/// XMP/EXIF blobs).
///
/// Skips compression when the brob output would be >= the
/// uncompressed alternative (e.g. tiny / already-compressed payloads):
/// the caller's intent is "save bytes if possible". When skipping
/// returns `false`; the caller falls back to the uncompressed box.
/// When compressing returns `true`.
///
/// Per spec, `original_type` MUST NOT be one of `jxl*`, `jbrd`, or
/// `brob` (nested brob is forbidden). The function does not validate
/// — caller is responsible for routing only metadata boxes
/// (Exif/xml/jumb).
#[cfg(feature = "brotli-metadata")]
#[must_use]
pub(crate) fn try_write_brob_box(
    out: &mut Vec<u8>,
    original_type: &[u8; 4],
    payload: &[u8],
    quality: u32,
) -> bool {
    use brotli::enc::BrotliEncoderParams;
    use std::io::Write;

    // Brotli params: quality 0-11, libjxl default 4. lgwin 22 (4 MiB
    // window) matches Brotli default; mode = generic.
    let mut params = BrotliEncoderParams::default();
    params.quality = quality.min(11) as i32;
    let mut compressed: Vec<u8> = Vec::with_capacity(payload.len() / 2);
    {
        let mut writer = brotli::CompressorWriter::with_params(&mut compressed, 4096, &params);
        if writer.write_all(payload).is_err() {
            return false;
        }
        if writer.flush().is_err() {
            return false;
        }
    }
    // brob payload = 4-byte original_type + brotli stream.
    let brob_payload_size = 4 + compressed.len();
    let uncompressed_box_size = 8 + payload.len();
    let brob_box_size = 8 + brob_payload_size;
    if brob_box_size >= uncompressed_box_size {
        // Brotli overhead beat the savings — skip and let the caller
        // emit the uncompressed box.
        return false;
    }
    let total_size = brob_box_size as u64;
    if total_size <= u32::MAX as u64 {
        out.extend_from_slice(&(total_size as u32).to_be_bytes());
        out.extend_from_slice(b"brob");
    } else {
        let extended_size = 16u64 + brob_payload_size as u64;
        out.extend_from_slice(&1u32.to_be_bytes());
        out.extend_from_slice(b"brob");
        out.extend_from_slice(&extended_size.to_be_bytes());
    }
    out.extend_from_slice(original_type);
    out.extend_from_slice(&compressed);
    true
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

/// Append a `jumb` box (JUMBF / C2PA superbox, ISO 19566-5) to JXL data.
///
/// `jumbf_payload` is the raw JUMBF superbox bytes provided by the caller
/// (e.g. produced by the `c2pa` crate). The encoder does not validate or
/// interpret the payload — it is the caller's responsibility to provide
/// a conformant JUMBF blob.
///
/// If `jxl_data` is already a container (starts with the 12-byte JXL
/// signature), the `jumb` box is appended at the end. If `jxl_data` is a
/// bare codestream (starts with `0xFF 0x0A`), it is first wrapped in a
/// container (signature + ftyp + jxlc), then the `jumb` box is appended.
/// Matches libjxl's box ordering when JUMBF is added via
/// `JxlEncoderAddBox(enc, "jumb", ...)` (see libjxl `encode.cc:2214`).
#[must_use]
pub fn append_jumbf_box(jxl_data: &[u8], jumbf_payload: &[u8]) -> Vec<u8> {
    if is_container(jxl_data) {
        // Already a container — append jumb box at the end.
        let jumb_box_size = 8 + jumbf_payload.len();
        let mut out = Vec::with_capacity(jxl_data.len() + jumb_box_size);
        out.extend_from_slice(jxl_data);
        write_box(&mut out, b"jumb", jumbf_payload);
        out
    } else {
        // Bare codestream — wrap in container first, then append jumb.
        let header_size = JXL_CONTAINER_SIGNATURE.len() + FTYP_BOX.len();
        let jxlc_size = 8 + jxl_data.len();
        let jumb_box_size = 8 + jumbf_payload.len();
        let total = header_size + jxlc_size + jumb_box_size;

        let mut out = Vec::with_capacity(total);
        out.extend_from_slice(&JXL_CONTAINER_SIGNATURE);
        out.extend_from_slice(&FTYP_BOX);
        write_box(&mut out, b"jxlc", jxl_data);
        write_box(&mut out, b"jumb", jumbf_payload);
        out
    }
}

/// Build a `colr` box payload using the `nclx` (named-colour, on-screen)
/// subtype defined by ISO/IEC 14496-12.
///
/// Returns the 11-byte payload that should be passed to
/// [`append_colr_box`]:
/// ```text
/// colour_type        : 4 bytes ASCII "nclx"
/// colour_primaries   : u16 BE  (CICP, ITU-T H.273 §8.1)
/// transfer_chars     : u16 BE  (CICP, ITU-T H.273 §8.2)
/// matrix_coefficients: u16 BE  (CICP, ITU-T H.273 §8.3)
/// full_range_flag    : 1 bit (MSB of last byte)
/// reserved           : 7 bits (= 0)
/// ```
///
/// JPEG XL signals its primary colour information in-codestream via
/// [`crate::headers::color_encoding::ColourEncoding`]. This box is an
/// **alternative** descriptor that ISOBMFF-aware tools (HEIF/AVIF
/// readers, container-level metadata extractors) can read without
/// parsing the JXL codestream. Per JPEG XL spec clause 5, decoders MUST
/// ignore boxes with unrecognised types — so emitting this box never
/// affects in-band decode, it only enriches container-level inspection.
///
/// CICP enum values (subset, ITU-T H.273):
/// - `colour_primaries = 1` BT.709, `9` BT.2020, `12` Display-P3
/// - `transfer_chars   = 1` BT.709, `13` sRGB, `16` PQ (BT.2100), `18` HLG (BT.2100)
/// - `matrix_coefficients = 0` Identity (RGB), `1` BT.709, `9` BT.2020-NCL
/// - `full_range = true` for full-swing / sYCC; `false` for studio-swing
#[must_use]
pub fn colr_nclx_payload(
    colour_primaries: u16,
    transfer_characteristics: u16,
    matrix_coefficients: u16,
    full_range: bool,
) -> [u8; 11] {
    let cp = colour_primaries.to_be_bytes();
    let tc = transfer_characteristics.to_be_bytes();
    let mc = matrix_coefficients.to_be_bytes();
    let flags = if full_range { 0x80 } else { 0x00 };
    [
        b'n', b'c', b'l', b'x', cp[0], cp[1], tc[0], tc[1], mc[0], mc[1], flags,
    ]
}

/// Append a `colr` box (ISOBMFF ColourInformationBox, ISO/IEC 14496-12) to
/// JXL data.
///
/// `colr_payload` is the raw box contents — the first 4 bytes are the
/// colour_type FourCC (`nclx`, `rICC`, `prof`, …) and the remainder is
/// the subtype-specific payload. Use [`colr_nclx_payload`] to build a
/// conformant nclx payload from CICP enum values, or supply raw ICC
/// profile bytes prefixed with `b"rICC"` / `b"prof"` for ICC variants.
///
/// JXL's in-codestream
/// [`crate::headers::color_encoding::ColourEncoding`] is the primary
/// colour source; the `colr` box here is an **alternative descriptor**
/// that ISOBMFF-aware tooling (HEIF/AVIF inspectors, metadata
/// extractors) can read without parsing the codestream. Per JPEG XL spec
/// clause 5, decoders MUST ignore boxes with unrecognised types — so
/// emitting this box never alters decoded pixels.
///
/// If `jxl_data` is already a container (starts with the 12-byte JXL
/// signature), the `colr` box is appended at the end. If `jxl_data` is a
/// bare codestream (starts with `0xFF 0x0A`), it is first wrapped in a
/// container (signature + ftyp + jxlc), then the `colr` box is
/// appended.
#[must_use]
pub fn append_colr_box(jxl_data: &[u8], colr_payload: &[u8]) -> Vec<u8> {
    append_aux_box(jxl_data, b"colr", colr_payload)
}

/// Append an `hCdR` box (HDR content-description metadata) to JXL data.
///
/// `hcdr_payload` is the raw box contents — a typical layout (matching
/// the ITU-T / SMPTE ST 2086 + CTA-861.3 MaxCLL/MaxFALL conventions
/// used by other ISOBMFF carriers) is:
/// ```text
/// display_primaries[3][2] : 6 × u16 BE (Rxy, Gxy, Bxy in 0.00002 units)
/// white_point[2]          : 2 × u16 BE (Wxy in 0.00002 units)
/// max_display_luminance   : u32 BE     (in 0.0001 cd/m²)
/// min_display_luminance   : u32 BE     (in 0.0001 cd/m²)
/// max_content_light       : u16 BE     (MaxCLL, cd/m²)
/// max_pic_average_light   : u16 BE     (MaxFALL, cd/m²)
/// ```
/// The encoder does not validate or interpret the payload — callers
/// that know which HDR metadata schema their downstream tooling expects
/// supply the bytes verbatim.
///
/// JXL signals peak / min display luminance in the in-codestream
/// [`crate::headers::color_encoding::ToneMapping`] (`intensity_target`,
/// `min_nits`). This box is an **alternative descriptor** for
/// ISOBMFF-aware HDR tooling. Per JPEG XL spec clause 5, decoders MUST
/// ignore boxes with unrecognised types — so emitting this box never
/// alters decoded pixels.
///
/// If `jxl_data` is already a container, the `hCdR` box is appended at
/// the end. If `jxl_data` is a bare codestream, it is first wrapped in
/// a container (signature + ftyp + jxlc), then the `hCdR` box is
/// appended.
#[must_use]
pub fn append_hcdr_box(jxl_data: &[u8], hcdr_payload: &[u8]) -> Vec<u8> {
    append_aux_box(jxl_data, b"hCdR", hcdr_payload)
}

/// Shared implementation for appending an auxiliary metadata box to
/// either an existing container or a bare codestream. Used by
/// [`append_colr_box`] and [`append_hcdr_box`] — same control flow as
/// [`append_jumbf_box`] / [`append_gain_map_box`], factored out to
/// avoid copy-paste drift.
fn append_aux_box(jxl_data: &[u8], box_type: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    if is_container(jxl_data) {
        let aux_box_size = 8 + payload.len();
        let mut out = Vec::with_capacity(jxl_data.len() + aux_box_size);
        out.extend_from_slice(jxl_data);
        write_box(&mut out, box_type, payload);
        out
    } else {
        let header_size = JXL_CONTAINER_SIGNATURE.len() + FTYP_BOX.len();
        let jxlc_size = 8 + jxl_data.len();
        let aux_box_size = 8 + payload.len();
        let total = header_size + jxlc_size + aux_box_size;

        let mut out = Vec::with_capacity(total);
        out.extend_from_slice(&JXL_CONTAINER_SIGNATURE);
        out.extend_from_slice(&FTYP_BOX);
        write_box(&mut out, b"jxlc", jxl_data);
        write_box(&mut out, box_type, payload);
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

    // ─── #15 brotli-metadata (brob boxes) ─────────────────────────

    #[cfg(feature = "brotli-metadata")]
    #[test]
    fn test_brob_compresses_large_xmp() {
        // Synthetic 4 KB of repeated XMP-like XML — Brotli compresses
        // this trivially (>10x). Verifies a brob box lands.
        let codestream = b"\xFF\x0A";
        let xmp_blob =
            "<rdf:Description><exif:CreatorTool>Test</exif:CreatorTool></rdf:Description>"
                .repeat(64);
        let xmp = xmp_blob.as_bytes();
        let result = wrap_in_container_with_brob(codestream, None, Some(xmp), 4);
        // Find a brob box header somewhere in the result.
        let mut found_brob = false;
        for i in 0..result.len().saturating_sub(8) {
            if &result[i..i + 4] == b"brob" {
                found_brob = true;
                // Next 4 bytes should be the original type ("xml ").
                assert_eq!(
                    &result[i + 4..i + 8],
                    b"xml ",
                    "brob original_type should be `xml ` for XMP",
                );
                break;
            }
        }
        assert!(found_brob, "expected a brob box for compressible XMP");
        // Result should be smaller than the uncompressed equivalent.
        let uncompressed = wrap_in_container(codestream, None, Some(xmp));
        assert!(
            result.len() < uncompressed.len(),
            "brob version {} should be smaller than uncompressed {}",
            result.len(),
            uncompressed.len(),
        );
    }

    #[cfg(feature = "brotli-metadata")]
    #[test]
    fn test_brob_skips_tiny_xmp() {
        // Small payload where Brotli framing overhead would beat the
        // savings. Should fall back to the uncompressed `xml ` box.
        let codestream = b"\xFF\x0A";
        let xmp = b"<x>1</x>";
        let result = wrap_in_container_with_brob(codestream, None, Some(xmp), 4);
        let mut found_brob = false;
        let mut found_xml = false;
        for i in 0..result.len().saturating_sub(4) {
            if &result[i..i + 4] == b"brob" {
                found_brob = true;
            }
            if &result[i..i + 4] == b"xml " {
                found_xml = true;
            }
        }
        assert!(!found_brob, "tiny XMP should NOT use brob");
        assert!(found_xml, "tiny XMP should fall back to plain `xml ` box");
    }

    // ─── JUMBF (jumb) box pass-through ───────────────────────────

    #[test]
    fn test_wrap_in_container_with_jumbf_only() {
        let codestream = b"\xFF\x0A\x00\x01";
        // Minimal synthetic JUMBF: outer box header (8 bytes) +
        // payload. The encoder doesn't interpret this — pass-through
        // only.
        let jumbf = b"\x00\x00\x00\x10jumdc2pa\x00\x00\x00\x00";
        let result = wrap_in_container_with_jumbf(codestream, None, None, Some(jumbf));
        // Header (32) + jxlc (8 + 4) + jumb (8 + 16)
        assert_eq!(result.len(), 32 + 12 + 24);
        let jumb_start = 32 + 12;
        let jumb_size = u32::from_be_bytes([
            result[jumb_start],
            result[jumb_start + 1],
            result[jumb_start + 2],
            result[jumb_start + 3],
        ]);
        assert_eq!(jumb_size as usize, 8 + jumbf.len());
        assert_eq!(&result[jumb_start + 4..jumb_start + 8], b"jumb");
        assert_eq!(&result[jumb_start + 8..jumb_start + 8 + jumbf.len()], jumbf);
    }

    #[test]
    fn test_wrap_in_container_with_exif_and_jumbf() {
        let codestream = b"\xFF\x0A";
        let exif = b"MM\x00\x2a\x00\x00\x00\x08";
        let jumbf = b"\x00\x00\x00\x0Cjumdc2pa";
        let result = wrap_in_container_with_jumbf(codestream, Some(exif), None, Some(jumbf));
        // Order: header (32) + jxlc (8+2) + Exif (8+4+exif.len) + jumb (8+jumbf.len)
        let expected = 32 + 10 + (8 + 4 + exif.len()) + (8 + jumbf.len());
        assert_eq!(result.len(), expected);
        // Find jumb at the tail.
        let jumb_offset = 32 + 10 + 8 + 4 + exif.len();
        assert_eq!(&result[jumb_offset + 4..jumb_offset + 8], b"jumb");
        assert_eq!(&result[jumb_offset + 8..], jumbf);
    }

    #[test]
    fn test_wrap_in_container_jumbf_none_is_no_op() {
        // Sanity: wrap_in_container_with_jumbf with jumbf=None must
        // produce byte-identical output to plain wrap_in_container.
        // Pins the default-None hash-lock invariant.
        let codestream = b"\xFF\x0A\x12\x34";
        let exif = b"Exif data";
        let xmp = b"<x:xmpmeta/>";
        let baseline = wrap_in_container(codestream, Some(exif), Some(xmp));
        let with_no_jumbf = wrap_in_container_with_jumbf(codestream, Some(exif), Some(xmp), None);
        assert_eq!(baseline, with_no_jumbf);
    }

    #[test]
    fn test_append_jumbf_box_to_bare_codestream() {
        let codestream = b"\xFF\x0A\x00\x01\x02\x03";
        let jumbf = b"\x00\x00\x00\x10jumdc2pa\xDE\xAD\xBE\xEF";
        let result = append_jumbf_box(codestream, jumbf);
        // Result is a container starting with the signature + ftyp.
        assert!(is_container(&result));
        assert_eq!(&result[12..32], &FTYP_BOX);
        // jxlc box at offset 32.
        let jxlc_size = u32::from_be_bytes([result[32], result[33], result[34], result[35]]);
        assert_eq!(jxlc_size as usize, 8 + codestream.len());
        assert_eq!(&result[36..40], b"jxlc");
        assert_eq!(&result[40..40 + codestream.len()], codestream);
        // jumb box follows jxlc.
        let jumb_offset = 40 + codestream.len();
        let jumb_size = u32::from_be_bytes([
            result[jumb_offset],
            result[jumb_offset + 1],
            result[jumb_offset + 2],
            result[jumb_offset + 3],
        ]);
        assert_eq!(jumb_size as usize, 8 + jumbf.len());
        assert_eq!(&result[jumb_offset + 4..jumb_offset + 8], b"jumb");
        assert_eq!(&result[jumb_offset + 8..], jumbf);
    }

    #[test]
    fn test_append_jumbf_box_to_existing_container() {
        // Build a minimal container.
        let codestream = b"\xFF\x0A\x00";
        let mut container = Vec::new();
        container.extend_from_slice(&JXL_CONTAINER_SIGNATURE);
        container.extend_from_slice(&FTYP_BOX);
        write_box(&mut container, b"jxlc", codestream);

        let jumbf = b"\x00\x00\x00\x0Cjumdc2pa";
        let result = append_jumbf_box(&container, jumbf);

        // Original container bytes preserved.
        assert_eq!(&result[..container.len()], container.as_slice());
        // jumb box appended at the end.
        let jumb_offset = container.len();
        assert_eq!(&result[jumb_offset + 4..jumb_offset + 8], b"jumb");
        assert_eq!(&result[jumb_offset + 8..], jumbf);
    }

    #[cfg(feature = "brotli-metadata")]
    #[test]
    fn test_brob_compresses_large_jumbf() {
        // C2PA JUMBF blobs in the wild are typically several KB of
        // CBOR + XML — brob should be a clear win on synthetic data
        // of that shape.
        let codestream = b"\xFF\x0A";
        let jumbf_blob = b"\x00\x00\x00\x20jumdc2pa\x00\x00\x00\x00manifest-bytes--".repeat(64);
        let result =
            wrap_in_container_with_brob_and_jumbf(codestream, None, None, Some(&jumbf_blob), 4);
        let mut found_brob_jumb = false;
        for i in 0..result.len().saturating_sub(8) {
            if &result[i..i + 4] == b"brob" && &result[i + 4..i + 8] == b"jumb" {
                found_brob_jumb = true;
                break;
            }
        }
        assert!(
            found_brob_jumb,
            "expected brob box with original_type=jumb for repetitive JUMBF payload"
        );
    }

    #[cfg(feature = "brotli-metadata")]
    #[test]
    fn test_brob_exif_with_tiff_prefix() {
        // EXIF brob includes the 4-byte Tiff prefix in the compressed
        // payload (matches libjxl convention). Verify the brob's
        // original_type is `Exif`.
        let codestream = b"\xFF\x0A";
        let exif = vec![0x4D, 0x4D, 0x00, 0x2A].repeat(64); // 256-byte TIFF-MM-tagged blob
        let result = wrap_in_container_with_brob(codestream, Some(&exif), None, 4);
        let mut found_brob_exif = false;
        for i in 0..result.len().saturating_sub(8) {
            if &result[i..i + 4] == b"brob" && &result[i + 4..i + 8] == b"Exif" {
                found_brob_exif = true;
                break;
            }
        }
        assert!(found_brob_exif, "expected brob box with original_type=Exif");
    }

    // ─── Codestream level (jxll box) ──────────────────────────────

    #[test]
    fn test_compute_level_baseline_fits_level5() {
        // Typical web-sized images (≤ 262 144 per axis, ≤ 2²⁸ pixels,
        // small ICC, ≤ 4 extras, no CMYK) → level 5.
        assert_eq!(compute_codestream_level(1024, 1024, 0, false, 0), Some(5));
        assert_eq!(
            compute_codestream_level(1024, 1024, 4, false, 1 << 22),
            Some(5)
        );
        assert_eq!(compute_codestream_level(262_144, 1, 0, false, 0), Some(5));
        // 16384 × 16384 = 2²⁸ pixels, exactly the level-5 ceiling.
        assert_eq!(
            compute_codestream_level(16_384, 16_384, 0, false, 0),
            Some(5)
        );
    }

    #[test]
    fn test_compute_level_exceeding_level5_bumps_to_10() {
        // > 262 144 per axis bumps to level 10.
        assert_eq!(compute_codestream_level(262_145, 1, 0, false, 0), Some(10));
        // > 2²⁸ total pixels bumps to level 10.
        assert_eq!(
            compute_codestream_level(16_385, 16_385, 0, false, 0),
            Some(10)
        );
        // ICC > 2²² bumps to level 10.
        assert_eq!(
            compute_codestream_level(1024, 1024, 0, false, (1 << 22) + 1),
            Some(10)
        );
        // 5+ extra channels bumps to level 10.
        assert_eq!(compute_codestream_level(1024, 1024, 5, false, 0), Some(10));
        // CMYK (`Black`) extra channel bumps to level 10.
        assert_eq!(compute_codestream_level(1024, 1024, 1, true, 0), Some(10));
    }

    #[test]
    fn test_compute_level_overflow_returns_none() {
        // Beyond the level-10 caps: not encodable.
        assert_eq!(compute_codestream_level(u32::MAX, 1, 0, false, 0), None);
        assert_eq!(
            compute_codestream_level(1, 1, 0, false, (1 << 28) + 1),
            None
        );
        assert_eq!(compute_codestream_level(1, 1, 257, false, 0), None);
    }

    #[test]
    fn test_level_requires_container() {
        assert!(!level_requires_container(5));
        assert!(level_requires_container(10));
    }

    #[test]
    fn test_wrap_with_level5_byte_identical() {
        // Level 5 ≡ no jxll box ≡ existing baseline byte layout.
        let codestream = b"\xFF\x0A\x00\x00";
        let baseline = wrap_in_container(codestream, None, None);
        let with_level = wrap_in_container_with_level(codestream, None, None, 5);
        assert_eq!(baseline, with_level);
    }

    #[test]
    fn test_wrap_with_level10_emits_jxll_box() {
        let codestream = b"\xFF\x0A\x00\x00";
        let result = wrap_in_container_with_level(codestream, None, None, 10);
        // 32 (signature + ftyp) + 9 (jxll) + 12 (jxlc 8 + 4 payload).
        assert_eq!(result.len(), 32 + 9 + 12);
        // jxll box header sits directly after ftyp (offset 32).
        assert_eq!(&result[32..36], &[0, 0, 0, 0x09]); // size = 9
        assert_eq!(&result[36..40], b"jxll");
        assert_eq!(result[40], 10); // level byte
        // jxlc box follows.
        assert_eq!(&result[41..45], &[0, 0, 0, 0x0C]); // size = 12
        assert_eq!(&result[45..49], b"jxlc");
        assert_eq!(&result[49..53], codestream);
    }

    #[test]
    fn test_wrap_with_level10_and_metadata() {
        // jxll box still sits between ftyp and jxlc; Exif follows jxlc.
        let codestream = b"\xFF\x0A";
        let exif = b"MM\x00\x2a\x00\x00";
        let result = wrap_in_container_with_level(codestream, Some(exif), None, 10);
        // 32 + 9 + (8 + 2) jxlc + (8 + 4 + 6) Exif
        assert_eq!(result.len(), 32 + 9 + 10 + 18);
        assert_eq!(&result[32..36], &[0, 0, 0, 0x09]);
        assert_eq!(&result[36..40], b"jxll");
        assert_eq!(result[40], 10);
        assert_eq!(&result[41..45], &[0, 0, 0, 0x0A]); // jxlc size 10
        assert_eq!(&result[45..49], b"jxlc");
        // Exif starts at 32 + 9 + 10 = 51.
        assert_eq!(&result[51 + 4..51 + 8], b"Exif");
    }

    #[test]
    fn test_wrap_with_jbrd_and_level10() {
        let codestream = b"\xFF\x0A\x00";
        let jbrd = b"jbrdpayload";
        let result = wrap_in_container_with_jbrd_and_level(codestream, Some(jbrd), None, None, 10);
        // signature(12) + ftyp(20) + jxll(9) + jxlc(8+3) + jbrd(8+11)
        assert_eq!(result.len(), 12 + 20 + 9 + 11 + 19);
        assert_eq!(&result[32..36], &[0, 0, 0, 0x09]);
        assert_eq!(&result[36..40], b"jxll");
        assert_eq!(result[40], 10);
    }

    // ─── `colr` / `hCdR` alternative descriptor boxes ─────────────

    #[test]
    fn test_colr_nclx_payload_layout() {
        // BT.2100 PQ, full range: CP=9, TC=16, MC=9, full=true.
        let payload = colr_nclx_payload(9, 16, 9, true);
        // 4-byte "nclx" + 3 × u16 BE + 1 byte flags = 11 bytes
        assert_eq!(payload.len(), 11);
        assert_eq!(&payload[0..4], b"nclx");
        // u16 BE: 0x0009 → 0x00 0x09
        assert_eq!(&payload[4..6], &[0x00, 0x09]); // primaries
        assert_eq!(&payload[6..8], &[0x00, 0x10]); // transfer = 16
        assert_eq!(&payload[8..10], &[0x00, 0x09]); // matrix
        assert_eq!(payload[10], 0x80); // full_range flag set (MSB)

        // sRGB studio range: CP=1, TC=13, MC=1, full=false.
        let payload2 = colr_nclx_payload(1, 13, 1, false);
        assert_eq!(&payload2[4..6], &[0x00, 0x01]);
        assert_eq!(&payload2[6..8], &[0x00, 0x0D]);
        assert_eq!(&payload2[8..10], &[0x00, 0x01]);
        assert_eq!(payload2[10], 0x00);
    }

    #[test]
    fn test_append_colr_box_to_bare_codestream() {
        let codestream = b"\xFF\x0A\x00\x01\x02\x03";
        let payload = colr_nclx_payload(9, 16, 9, true); // BT.2100 PQ, full range
        let result = append_colr_box(codestream, &payload);

        // Result is a container.
        assert!(is_container(&result));
        assert_eq!(&result[12..32], &FTYP_BOX);
        // jxlc box at offset 32.
        let jxlc_size = u32::from_be_bytes([result[32], result[33], result[34], result[35]]);
        assert_eq!(jxlc_size as usize, 8 + codestream.len());
        assert_eq!(&result[36..40], b"jxlc");
        assert_eq!(&result[40..40 + codestream.len()], codestream);
        // colr box follows jxlc.
        let colr_offset = 40 + codestream.len();
        let colr_size = u32::from_be_bytes([
            result[colr_offset],
            result[colr_offset + 1],
            result[colr_offset + 2],
            result[colr_offset + 3],
        ]);
        assert_eq!(colr_size as usize, 8 + payload.len());
        assert_eq!(&result[colr_offset + 4..colr_offset + 8], b"colr");
        assert_eq!(&result[colr_offset + 8..], &payload);
    }

    #[test]
    fn test_append_colr_box_to_existing_container() {
        // Build a minimal container.
        let codestream = b"\xFF\x0A\x00";
        let mut container = Vec::new();
        container.extend_from_slice(&JXL_CONTAINER_SIGNATURE);
        container.extend_from_slice(&FTYP_BOX);
        write_box(&mut container, b"jxlc", codestream);

        // Raw ICC variant: 4-byte "prof" + payload.
        let mut payload = Vec::from(*b"prof");
        payload.extend_from_slice(b"fake-icc-bytes");
        let result = append_colr_box(&container, &payload);

        assert_eq!(&result[..container.len()], container.as_slice());
        let colr_offset = container.len();
        assert_eq!(&result[colr_offset + 4..colr_offset + 8], b"colr");
        assert_eq!(&result[colr_offset + 8..], payload.as_slice());
    }

    #[test]
    fn test_append_hcdr_box_to_bare_codestream() {
        let codestream = b"\xFF\x0A\x12\x34";
        // Synthetic HDR metadata blob: 6×u16 primaries + 2×u16 white +
        // 2×u32 luminance + 2×u16 light = 28 bytes.
        let payload: [u8; 28] = [
            0x80, 0x00, 0x33, 0x33, // Rxy
            0x1F, 0x40, 0xC3, 0x50, // Gxy
            0x21, 0x34, 0x08, 0x6A, // Bxy
            0x3D, 0x13, 0x40, 0x42, // Wxy
            0x00, 0x98, 0x96, 0x80, // max_lum = 10000000 (1000 nits in 0.0001)
            0x00, 0x00, 0x00, 0x32, // min_lum = 50
            0x03, 0xE8, 0x01, 0x90, // MaxCLL=1000, MaxFALL=400
        ];
        let result = append_hcdr_box(codestream, &payload);

        assert!(is_container(&result));
        let jxlc_size = u32::from_be_bytes([result[32], result[33], result[34], result[35]]);
        assert_eq!(jxlc_size as usize, 8 + codestream.len());
        let hcdr_offset = 40 + codestream.len();
        let hcdr_size = u32::from_be_bytes([
            result[hcdr_offset],
            result[hcdr_offset + 1],
            result[hcdr_offset + 2],
            result[hcdr_offset + 3],
        ]);
        assert_eq!(hcdr_size as usize, 8 + payload.len());
        // Case-sensitive FourCC: capital C and R.
        assert_eq!(&result[hcdr_offset + 4..hcdr_offset + 8], b"hCdR");
        assert_eq!(&result[hcdr_offset + 8..], &payload);
    }

    #[test]
    fn test_append_hcdr_box_to_existing_container() {
        let codestream = b"\xFF\x0A";
        let mut container = Vec::new();
        container.extend_from_slice(&JXL_CONTAINER_SIGNATURE);
        container.extend_from_slice(&FTYP_BOX);
        write_box(&mut container, b"jxlc", codestream);

        let payload = b"\x00\x01\x02\x03\x04\x05\x06\x07";
        let result = append_hcdr_box(&container, payload);

        assert_eq!(&result[..container.len()], container.as_slice());
        let hcdr_offset = container.len();
        assert_eq!(&result[hcdr_offset + 4..hcdr_offset + 8], b"hCdR");
        assert_eq!(&result[hcdr_offset + 8..], payload);
    }

    #[test]
    fn test_colr_then_hcdr_both_present_after_chained_append() {
        // Chained appends survive — both boxes findable in the output.
        let codestream = b"\xFF\x0A\x00\x00";
        let colr = colr_nclx_payload(9, 16, 9, true);
        let hcdr = b"\x12\x34\x56\x78";

        let step1 = append_colr_box(codestream, &colr);
        let step2 = append_hcdr_box(&step1, hcdr);

        // Both FourCCs present.
        let mut found_colr = false;
        let mut found_hcdr = false;
        for i in 0..step2.len().saturating_sub(4) {
            if &step2[i..i + 4] == b"colr" {
                found_colr = true;
            }
            if &step2[i..i + 4] == b"hCdR" {
                found_hcdr = true;
            }
        }
        assert!(found_colr, "expected colr box in chained output");
        assert!(found_hcdr, "expected hCdR box in chained output");
        // hCdR appears strictly after colr (append order preserved).
        let colr_pos = step2.windows(4).position(|w| w == b"colr").unwrap();
        let hcdr_pos = step2.windows(4).position(|w| w == b"hCdR").unwrap();
        assert!(colr_pos < hcdr_pos);
    }
}
