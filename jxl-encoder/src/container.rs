// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! JXL container format (ISOBMFF boxes) for metadata embedding.
//!
//! When EXIF/XMP metadata is present, the codestream must be wrapped in a
//! container format. The container uses ISOBMFF-style boxes.
//!
//! The box order is libjxl's (`encode.cc`, `ProcessOneEnqueuedInput`, v0.12):
//! boxes queued before the frames go between the codestream header and the
//! frames, which split into two `jxlp` boxes, and no box follows the last
//! codestream box. cjxl queues Exif, XMP, JUMBF and the `jhgm` gain map that
//! way (`lib/extras/enc/jxl.cc`), and `JxlEncoderAddJPEGFrame` queues the
//! JPEG's Exif, XMP and JUMBF ahead of its frame.
//! ```text
//! no boxes, level 5:   bare codestream
//! no boxes, level 10:  [JXL signature] [ftyp] [jxll] [jxlc]
//! boxes:               [JXL signature] [ftyp] [jxll?] [jxlp 0: header] [Exif?] [xml ?] [jumb?] [colr?] [hCdR?] [jhgm?] [jxlp 1 (last): frames]
//! JPEG reconstruction: [JXL signature] [ftyp] [jxll?] [jxlp 0: header] [jbrd] [Exif?] [xml ?] [jhgm?] [jxlp 1 (last): frames]
//! ```
//! The header part is the codestream's signature, SizeHeader, ImageMetadata,
//! CustomTransformData and ICC profile, zero-padded to a byte: the bytes libjxl
//! writes into its first `jxlp` box "so that it comes before metadata or jpeg
//! reconstruction boxes". [`codestream_header_len`] finds where they end.
//!
//! A box after the last codestream box is valid, but Ubuntu 24.04's ffmpeg 6.1
//! (libjxl 0.7) loops on one until it runs out of memory. Files in libjxl's
//! layout decode there exactly (imazen-26 reader census, 2026-10-17).

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
/// CMYK `Black` plane (level-5 forbids it).
///
/// `modular_16bit_buffer_sufficient` is the value of the header field of the
/// same name — compute it with
/// [`crate::headers::file_header::ImageMetadata::modular_16bit_buffer_sufficient`]
/// so the two cannot disagree. libjxl checks it FIRST among the level-5 caps
/// (`encode.cc:585`) and returns 10 when it is false.
///
/// Before 2026-09-08 this function did not take the parameter and the doc here
/// claimed the corresponding cap was "left to the encoder". It was not left to
/// anything: a 16-bit lossless encode wrote the header field as `false` and
/// then signalled level 5 by omitting the `jxll` box — exactly the combination
/// level 5 forbids. Verified against cjxl v0.12 on the same 16-bit source:
/// cjxl emits a container with `jxll = 10`; we emitted a bare codestream.
#[must_use]
pub fn compute_codestream_level(
    width: u32,
    height: u32,
    num_extra_channels: u32,
    has_black_channel: bool,
    icc_size: u64,
    modular_16bit_buffer_sufficient: bool,
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

    // libjxl makes this the FIRST level-5 check (`encode.cc:585`): a stream
    // that needs 32-bit modular buffers cannot be level 5.
    if !modular_16bit_buffer_sufficient {
        return Some(10);
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
#[allow(dead_code)] // #76: thin delegating wrapper; unit tests exercise the family
pub fn wrap_in_container(codestream: &[u8], exif: Option<&[u8]>, xmp: Option<&[u8]>) -> Vec<u8> {
    wrap_in_container_with_jbrd_and_level_and_jumbf(codestream, None, exif, xmp, 5, None)
}

/// Like [`wrap_in_container`] but emits a `jxll` (codestream level) box
/// directly after `ftyp` when `level != 5`. Use [`compute_codestream_level`]
/// to derive `level` from image dimensions, ICC size, and extra channels.
#[allow(dead_code)] // #76: thin delegating wrapper; unit tests exercise the family
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
#[allow(dead_code)] // #76: thin delegating wrapper; unit tests exercise the family
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
#[allow(dead_code)] // #76: thin delegating wrapper; unit tests exercise the family
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
#[allow(dead_code)] // #76: thin delegating wrapper; unit tests exercise the family
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
#[allow(dead_code)] // #76: thin delegating wrapper; unit tests exercise the family
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
#[allow(dead_code)] // #76: thin delegating wrapper; unit tests exercise the family
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
#[allow(dead_code)] // #76: thin delegating wrapper; unit tests exercise the family
#[cfg(feature = "brotli-metadata")]
pub fn wrap_in_container_with_brob_and_level_and_jumbf(
    codestream: &[u8],
    exif: Option<&[u8]>,
    xmp: Option<&[u8]>,
    jumbf: Option<&[u8]>,
    brotli_quality: u32,
    level: u8,
) -> Vec<u8> {
    let mut boxes = Vec::new();
    write_metadata_boxes(&mut boxes, exif, xmp, jumbf, Some(brotli_quality));
    assemble(codestream, None, level, &boxes)
}

/// Wraps a JXL codestream in a container with optional JBRD, EXIF, and XMP.
///
/// The `jbrd` box contains JPEG Bitstream Reconstruction Data needed for
/// byte-exact JPEG reconstruction from the JXL file. When present, it comes
/// first among the boxes between the codestream header and the frames, as in
/// libjxl.
#[allow(dead_code)] // #76: thin delegating wrapper; unit tests exercise the family
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
#[allow(dead_code)] // #76: thin delegating wrapper; unit tests exercise the family
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
/// Box order (libjxl's): signature, ftyp, jxll? (level != 5), then the
/// codestream split after its header: `jxlp` 0, jbrd?, Exif?, xml?, jumb?,
/// `jxlp` 1 (last). Without any box the codestream stays one `jxlc` box. The
/// JUMBF payload (C2PA / Content Authenticity Initiative, ISO 19566-5) is the
/// superbox contents verbatim; it is not validated or interpreted.
///
/// The header's end comes from [`codestream_header_len`]. This function does
/// not know the ICC profile, so a codestream that embeds one (or a preview,
/// or one it can't parse) keeps a single `jxlc` box, after the other boxes.
pub fn wrap_in_container_with_jbrd_and_level_and_jumbf(
    codestream: &[u8],
    jbrd: Option<&[u8]>,
    exif: Option<&[u8]>,
    xmp: Option<&[u8]>,
    level: u8,
    jumbf: Option<&[u8]>,
) -> Vec<u8> {
    let mut boxes = Vec::new();
    if let Some(jbrd_data) = jbrd {
        write_box(&mut boxes, b"jbrd", jbrd_data);
    }
    write_metadata_boxes(&mut boxes, exif, xmp, jumbf, None);
    assemble(codestream, None, level, &boxes)
}

/// Boxes an encode adds to its codestream, in libjxl's queue order (cjxl's
/// `lib/extras/enc/jxl.cc`: Exif, XMP, JUMBF), then this crate's `colr` and
/// `hCdR` descriptor boxes.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct EncodedBoxes<'a> {
    pub(crate) exif: Option<&'a [u8]>,
    pub(crate) xmp: Option<&'a [u8]>,
    pub(crate) jumbf: Option<&'a [u8]>,
    pub(crate) colr: Option<&'a [u8]>,
    pub(crate) hcdr: Option<&'a [u8]>,
}

impl EncodedBoxes<'_> {
    fn is_empty(&self) -> bool {
        self.exif.is_none()
            && self.xmp.is_none()
            && self.jumbf.is_none()
            && self.colr.is_none()
            && self.hcdr.is_none()
    }
}

/// The container of a codestream this crate's encoder wrote, in libjxl's
/// layout. `icc` is the profile the encoder embedded, if any: the header's end
/// is found by re-encoding it and checking the codestream holds those bits.
/// `brotli_quality` puts Exif, XMP and JUMBF in `brob` boxes where that is
/// smaller (with the `brotli-metadata` feature).
///
/// Returns the bare codestream when there is no box and `level` is 5.
pub(crate) fn wrap_encoded(
    codestream: Vec<u8>,
    icc: Option<&[u8]>,
    level: u8,
    boxes: &EncodedBoxes<'_>,
    brotli_quality: Option<u32>,
) -> Vec<u8> {
    if boxes.is_empty() && !level_requires_container(level) {
        return codestream;
    }
    let mut meta = Vec::new();
    write_metadata_boxes(
        &mut meta,
        boxes.exif,
        boxes.xmp,
        boxes.jumbf,
        brotli_quality,
    );
    if let Some(payload) = boxes.colr {
        write_box(&mut meta, b"colr", payload);
    }
    if let Some(payload) = boxes.hcdr {
        write_box(&mut meta, b"hCdR", payload);
    }
    assemble(&codestream, icc, level, &meta)
}

/// Exif (with its 4-byte TIFF offset), XMP and JUMBF boxes in that order,
/// each in a `brob` box when `brotli_quality` is set, the `brotli-metadata`
/// feature is on, and compression makes it smaller.
fn write_metadata_boxes(
    out: &mut Vec<u8>,
    exif: Option<&[u8]>,
    xmp: Option<&[u8]>,
    jumbf: Option<&[u8]>,
    brotli_quality: Option<u32>,
) {
    if let Some(exif_data) = exif {
        // The brob's payload includes the TIFF offset, so it decompresses to
        // what a plain `Exif` box holds (libjxl's convention).
        let mut payload = Vec::with_capacity(4 + exif_data.len());
        payload.extend_from_slice(&[0u8; 4]);
        payload.extend_from_slice(exif_data);
        write_metadata_box(out, b"Exif", &payload, brotli_quality);
    }
    if let Some(xmp_data) = xmp {
        write_metadata_box(out, b"xml ", xmp_data, brotli_quality);
    }
    // Nested brob inside jumb is forbidden per spec, but the outer wrap of
    // jumb in brob is allowed (libjxl emits it via the same
    // JxlEncoderAddBox/compress_box path).
    if let Some(jumbf_data) = jumbf {
        write_metadata_box(out, b"jumb", jumbf_data, brotli_quality);
    }
}

fn write_metadata_box(
    out: &mut Vec<u8>,
    box_type: &[u8; 4],
    payload: &[u8],
    brotli_quality: Option<u32>,
) {
    #[cfg(feature = "brotli-metadata")]
    if let Some(quality) = brotli_quality
        && try_write_brob_box(out, box_type, payload, quality)
    {
        return;
    }
    #[cfg(not(feature = "brotli-metadata"))]
    let _ = brotli_quality;
    write_box(out, box_type, payload);
}

/// Signature, `ftyp`, the `jxll` box when `level != 5`, and the codestream with
/// the serialized `boxes` where libjxl puts boxes queued before the frames.
///
/// With boxes, the codestream splits after its header (see
/// [`codestream_header_len`]; `icc` as there) into `jxlp` 0, the boxes, and
/// `jxlp` 1 (last). When the header's end can't be found, the boxes go ahead
/// of a single `jxlc` box instead, so still no box follows the codestream.
fn assemble(codestream: &[u8], icc: Option<&[u8]>, level: u8, boxes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(64 + boxes.len() + codestream.len());
    out.extend_from_slice(&JXL_CONTAINER_SIGNATURE);
    out.extend_from_slice(&FTYP_BOX);
    write_level_box(&mut out, level);
    if boxes.is_empty() {
        write_box(&mut out, b"jxlc", codestream);
        return out;
    }
    match codestream_header_len(codestream, icc) {
        Some(header) => {
            write_jxlp_box(&mut out, 0, false, &codestream[..header]);
            out.extend_from_slice(boxes);
            write_jxlp_box(&mut out, 1, true, &codestream[header..]);
        }
        None => {
            out.extend_from_slice(boxes);
            write_box(&mut out, b"jxlc", codestream);
        }
    }
    out
}

/// Wraps a split codestream in a container with jxlp boxes, jbrd, and metadata.
///
/// The codestream is split into two `jxlp` boxes with the `jbrd` box and the
/// JPEG's metadata between them. This matches libjxl's JPEG lossless
/// transcoding container format (`JxlEncoderAddJPEGFrame` queues the Exif and
/// XMP boxes ahead of the frame):
/// ```text
/// [JXL signature] [ftyp] [jxlp part 0] [jbrd] [Exif?] [xml?] [jxlp part 1 (last)]
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

    // Exif box — try brob first, fall back to raw if brotli doesn't help.
    //
    // For XMP this is usually a big win (2-6× smaller — XMP is XML, very
    // redundant). For Exif it's a marginal win because Exif is binary and
    // already fairly compact, but `try_write_brob_box` only succeeds when
    // brob is strictly smaller, so the worst case is "no change".
    //
    // libjxl's PUBLIC ENC API default is brotli quality 4 (`box.cc:1094`),
    // but the cjxl CLI overrides this with `--brotli_effort=9` (the user-
    // facing default per `cjxl --help`). Quality 11 is the maximum and
    // achieves ~0.05% better metadata compression than 9 on typical
    // EXIF + XMP payloads, at a small one-shot CPU cost (~5-50 ms per
    // file for typical 5-50 KB metadata, dwarfed by the codestream
    // encode time). For a lossless transcode encoder we treat brotli
    // quality as a pay-once cost; lifting from 4 → 11 saves ~300-600
    // bytes per file with brob payloads on the 200-file bench.
    const JPEG_BROB_QUALITY: u32 = 11;
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
    if let Some(xmp_data) = xmp
        && !try_write_brob_box(&mut out, b"xml ", xmp_data, JPEG_BROB_QUALITY)
    {
        write_box(&mut out, b"xml ", xmp_data);
    }

    // Second jxlp box: frame data (counter = 1, last)
    write_jxlp_box(&mut out, 1, true, cs_part2);

    out
}

/// Write a jxlp (partial codestream) box.
/// Counter format: bits 0-30 = sequence number, bit 31 = last part flag.
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
#[cfg_attr(not(feature = "jpeg-reencoding"), allow(dead_code))]
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
    let params = BrotliEncoderParams {
        quality: quality.min(11) as i32,
        ..Default::default()
    };
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

/// Add a `jhgm` box to JXL data (container or bare codestream).
///
/// `jhgm_payload` is the raw gain map bundle payload (ISO 21496-1).
///
/// The box goes where libjxl puts a box queued before the frames (cjxl queues
/// `jhgm` after Exif, XMP and JUMBF): see [`insert_box`]. A bare codestream
/// becomes a container first.
#[must_use]
pub fn append_gain_map_box(jxl_data: &[u8], jhgm_payload: &[u8]) -> Vec<u8> {
    insert_box(jxl_data, b"jhgm", jhgm_payload)
}

/// Add a `jumb` box (JUMBF / C2PA superbox, ISO 19566-5) to JXL data.
///
/// `jumbf_payload` is the raw JUMBF superbox bytes provided by the caller
/// (e.g. produced by the `c2pa` crate). The encoder does not validate or
/// interpret the payload — it is the caller's responsibility to provide
/// a conformant JUMBF blob.
///
/// The box goes where libjxl puts a box queued with
/// `JxlEncoderAddBox(enc, "jumb", ...)` before the frames: see
/// [`insert_box`]. A bare codestream becomes a container first.
#[must_use]
#[allow(dead_code)] // #76: thin delegating wrapper; unit tests exercise the family
pub fn append_jumbf_box(jxl_data: &[u8], jumbf_payload: &[u8]) -> Vec<u8> {
    insert_box(jxl_data, b"jumb", jumbf_payload)
}

/// Put one box where libjxl puts a box queued before the frames: after the
/// codestream header and the boxes already there, ahead of the frames.
///
/// - A container whose codestream is split into `jxlp` boxes gets the box
///   right before the second codestream box (the first that holds frames).
/// - A container with one codestream box (`jxlc`, or a single `jxlp`), or a
///   bare codestream (which becomes a container), has the codestream split
///   after its header into `jxlp` 0 and `jxlp` 1 with the box between. When
///   [`codestream_header_len`] can't find the header's end (an embedded ICC
///   profile, which this function can't verify, a preview, or a codestream
///   it can't parse), the box goes ahead of the codestream box instead.
///
/// Boxes already after the codestream stay where they are. A container whose
/// top-level boxes don't parse gets the box at the end, as before.
fn insert_box(jxl_data: &[u8], box_type: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut new_box = Vec::with_capacity(16 + payload.len());
    write_box(&mut new_box, box_type, payload);
    if !is_container(jxl_data) {
        return assemble(jxl_data, None, 5, &new_box);
    }
    let mut out = Vec::with_capacity(jxl_data.len() + new_box.len() + 12);
    let codestream_boxes: Vec<TopBox> = match top_level_boxes(jxl_data) {
        Some(boxes) => boxes
            .into_iter()
            .filter(|b| &b.box_type == b"jxlc" || &b.box_type == b"jxlp")
            .collect(),
        None => Vec::new(),
    };
    match codestream_boxes.as_slice() {
        [] => {
            out.extend_from_slice(jxl_data);
            out.extend_from_slice(&new_box);
        }
        [only] => {
            let counter = if &only.box_type == b"jxlp" { 4 } else { 0 };
            let start = only.offset + only.header + counter;
            let end = only.offset + only.size;
            let codestream = jxl_data.get(start..end).unwrap_or_default();
            out.extend_from_slice(&jxl_data[..only.offset]);
            match codestream_header_len(codestream, None) {
                Some(header) => {
                    write_jxlp_box(&mut out, 0, false, &codestream[..header]);
                    out.extend_from_slice(&new_box);
                    write_jxlp_box(&mut out, 1, true, &codestream[header..]);
                }
                _ => {
                    out.extend_from_slice(&new_box);
                    out.extend_from_slice(&jxl_data[only.offset..end]);
                }
            }
            out.extend_from_slice(&jxl_data[end..]);
        }
        [_, frames, ..] => {
            out.extend_from_slice(&jxl_data[..frames.offset]);
            out.extend_from_slice(&new_box);
            out.extend_from_slice(&jxl_data[frames.offset..]);
        }
    }
    out
}

/// Offset of the first codestream box that holds frames in a container whose
/// codestream is split into `jxlp` boxes: where libjxl puts a box queued
/// before the frames, after the ones already there. `None` when the container
/// has fewer than two codestream boxes or doesn't parse.
#[cfg_attr(not(feature = "jpeg-reencoding"), allow(dead_code))]
pub(crate) fn frames_box_offset(container: &[u8]) -> Option<usize> {
    top_level_boxes(container)?
        .into_iter()
        .filter(|b| &b.box_type == b"jxlc" || &b.box_type == b"jxlp")
        .nth(1)
        .map(|b| b.offset)
}

/// Top-level box types of a container, for tests elsewhere in the crate.
#[cfg(test)]
pub(crate) fn box_types(data: &[u8]) -> Vec<[u8; 4]> {
    top_level_boxes(data)
        .expect("top-level boxes parse")
        .iter()
        .map(|b| b.box_type)
        .collect()
}

/// One top-level box of a container.
struct TopBox {
    offset: usize,
    /// Header length: 8, or 16 with a 64-bit size.
    header: usize,
    /// Total size, header included.
    size: usize,
    box_type: [u8; 4],
}

/// The top-level boxes of a container, or `None` if they don't parse.
fn top_level_boxes(data: &[u8]) -> Option<Vec<TopBox>> {
    let mut boxes = Vec::new();
    let mut offset = 0usize;
    while offset < data.len() {
        let rest = &data[offset..];
        let size32 = u32::from_be_bytes(rest.get(..4)?.try_into().ok()?);
        let box_type: [u8; 4] = rest.get(4..8)?.try_into().ok()?;
        let (header, size) = match size32 {
            0 => (8, rest.len()),
            1 => (
                16,
                usize::try_from(u64::from_be_bytes(rest.get(8..16)?.try_into().ok()?)).ok()?,
            ),
            n => (8, n as usize),
        };
        if size < header || size > rest.len() {
            return None;
        }
        boxes.push(TopBox {
            offset,
            header,
            size,
            box_type,
        });
        offset += size;
    }
    Some(boxes)
}

// ── Codestream header length ────────────────────────────────────────────

/// One choice of a JPEG XL `U32` field: a value, or `n` bits plus an offset.
#[derive(Clone, Copy)]
enum U32Dist {
    Val(u32),
    Bits(u32, u32),
}

/// LSB-first reader over a codestream for [`codestream_header_len`]. Every
/// read returns `None` past the end of the data.
struct HeaderReader<'a> {
    data: &'a [u8],
    /// Position in bits.
    pos: usize,
}

impl HeaderReader<'_> {
    fn bits(&mut self, n: u32) -> Option<u64> {
        let mut value = 0u64;
        for i in 0..n {
            let byte = *self.data.get(self.pos / 8)?;
            value |= u64::from((byte >> (self.pos % 8)) & 1) << i;
            self.pos += 1;
        }
        Some(value)
    }

    fn bool(&mut self) -> Option<bool> {
        Some(self.bits(1)? == 1)
    }

    fn skip(&mut self, n: u64) -> Option<()> {
        let end = (self.pos as u64).checked_add(n)?;
        if end > (self.data.len() as u64) * 8 {
            return None;
        }
        self.pos = usize::try_from(end).ok()?;
        Some(())
    }

    fn u32(&mut self, dist: [U32Dist; 4]) -> Option<u32> {
        match dist[self.bits(2)? as usize] {
            U32Dist::Val(v) => Some(v),
            U32Dist::Bits(n, offset) => u32::try_from(self.bits(n)?).ok()?.checked_add(offset),
        }
    }

    /// `U64` (libjxl `U64Coder::Read`).
    fn u64(&mut self) -> Option<u64> {
        match self.bits(2)? {
            0 => Some(0),
            1 => Some(1 + self.bits(4)?),
            2 => Some(17 + self.bits(8)?),
            _ => {
                let mut value = self.bits(12)?;
                let mut shift = 12;
                while self.bool()? {
                    if shift == 60 {
                        value |= self.bits(4)? << shift;
                        break;
                    }
                    value |= self.bits(8)? << shift;
                    shift += 8;
                }
                Some(value)
            }
        }
    }

    /// `Enum`: U32(Val(0), Val(1), BitsOffset(4, 2), BitsOffset(6, 18)).
    fn enumeration(&mut self) -> Option<u32> {
        use U32Dist::{Bits, Val};
        self.u32([Val(0), Val(1), Bits(4, 2), Bits(6, 18)])
    }

    /// A bundle's extensions: the mask, one size per set bit, then those
    /// bits, which a reader skips (libjxl `ReadVisitor::EndExtensions`).
    fn extensions(&mut self) -> Option<()> {
        let mask = self.u64()?;
        let mut total = 0u64;
        for _ in 0..mask.count_ones() {
            total = total.checked_add(self.u64()?)?;
        }
        self.skip(total)
    }

    /// `SizeHeader`; the intrinsic size is one too.
    fn size_header(&mut self) -> Option<()> {
        use U32Dist::Bits;
        let dist = [Bits(9, 1), Bits(13, 1), Bits(18, 1), Bits(30, 1)];
        let small = self.bool()?;
        if small {
            self.bits(5)?;
        } else {
            self.u32(dist)?;
        }
        if self.bits(3)? == 0 {
            if small {
                self.bits(5)?;
            } else {
                self.u32(dist)?;
            }
        }
        Some(())
    }

    fn preview_header(&mut self) -> Option<()> {
        use U32Dist::{Bits, Val};
        let div8 = self.bool()?;
        let dist = if div8 {
            [Val(16), Val(32), Bits(5, 1), Bits(9, 33)]
        } else {
            [Bits(6, 1), Bits(8, 65), Bits(10, 321), Bits(12, 1345)]
        };
        self.u32(dist)?;
        if self.bits(3)? == 0 {
            self.u32(dist)?;
        }
        Some(())
    }

    fn bit_depth(&mut self) -> Option<()> {
        use U32Dist::{Bits, Val};
        if self.bool()? {
            self.u32([Val(32), Val(16), Val(24), Bits(6, 1)])?;
            self.bits(4)?;
        } else {
            self.u32([Val(8), Val(10), Val(12), Bits(6, 1)])?;
        }
        Some(())
    }

    /// `Customxy`: two `U32`s.
    fn custom_xy(&mut self) -> Option<()> {
        use U32Dist::Bits;
        let dist = [
            Bits(19, 0),
            Bits(19, 524_288),
            Bits(20, 1_048_576),
            Bits(21, 2_097_152),
        ];
        self.u32(dist)?;
        self.u32(dist)?;
        Some(())
    }
}

/// What the header bundles say about the bytes after them.
struct HeaderBundles {
    /// Bit position after `CustomTransformData`.
    end_bits: usize,
    want_icc: bool,
    have_preview: bool,
}

/// Read the signature, `SizeHeader`, `ImageMetadata` and `CustomTransformData`
/// (libjxl `VisitFields` of each, v0.12).
fn read_header_bundles(codestream: &[u8]) -> Option<HeaderBundles> {
    use U32Dist::{Bits, Val};
    const COLOUR_SPACE_GREY: u32 = 1;
    const COLOUR_SPACE_XYB: u32 = 2;
    const WHITE_POINT_CUSTOM: u32 = 2;
    const PRIMARIES_CUSTOM: u32 = 2;
    const EXTRA_ALPHA: u32 = 0;
    const EXTRA_SPOT_COLOUR: u32 = 2;
    const EXTRA_CFA: u32 = 5;

    let mut r = HeaderReader {
        data: codestream,
        pos: 0,
    };
    if r.bits(16)? != 0x0AFF {
        return None;
    }
    r.size_header()?;

    // ImageMetadata. Its all-default form is 8-bit XYB with no ICC profile.
    let mut want_icc = false;
    let mut have_preview = false;
    let mut xyb_encoded = true;
    if !r.bool()? {
        let extra_fields = r.bool()?;
        if extra_fields {
            r.bits(3)?; // orientation - 1
            if r.bool()? {
                r.size_header()?; // intrinsic size
            }
            have_preview = r.bool()?;
            if have_preview {
                r.preview_header()?;
            }
            if r.bool()? {
                // AnimationHeader
                r.u32([Val(100), Val(1000), Bits(10, 1), Bits(30, 1)])?;
                r.u32([Val(1), Val(1001), Bits(8, 1), Bits(10, 1)])?;
                r.u32([Val(0), Bits(3, 0), Bits(16, 0), Bits(32, 0)])?;
                r.bool()?; // have_timecodes
            }
        }
        r.bit_depth()?;
        r.bool()?; // modular_16_bit_buffer_sufficient
        let num_extra_channels = r.u32([Val(0), Val(1), Bits(4, 2), Bits(12, 1)])?;
        for _ in 0..num_extra_channels {
            if r.bool()? {
                continue; // all_default: an 8-bit alpha channel
            }
            let kind = r.enumeration()?;
            r.bit_depth()?;
            r.u32([Val(0), Val(3), Val(4), Bits(3, 1)])?; // dim_shift
            let name_len = r.u32([Val(0), Bits(4, 0), Bits(5, 16), Bits(10, 48)])?;
            r.skip(u64::from(name_len) * 8)?;
            match kind {
                EXTRA_ALPHA => {
                    r.bool()?; // alpha_associated
                }
                EXTRA_SPOT_COLOUR => r.skip(4 * 16)?,
                EXTRA_CFA => {
                    r.u32([Val(1), Bits(2, 0), Bits(4, 3), Bits(8, 19)])?;
                }
                _ => {}
            }
        }
        xyb_encoded = r.bool()?;
        // ColourEncoding
        if !r.bool()? {
            want_icc = r.bool()?;
            let colour_space = r.enumeration()?;
            if !want_icc {
                if colour_space != COLOUR_SPACE_XYB && r.enumeration()? == WHITE_POINT_CUSTOM {
                    r.custom_xy()?;
                }
                if colour_space != COLOUR_SPACE_GREY
                    && colour_space != COLOUR_SPACE_XYB
                    && r.enumeration()? == PRIMARIES_CUSTOM
                {
                    r.custom_xy()?;
                    r.custom_xy()?;
                    r.custom_xy()?;
                }
                // CustomTransferFunction, implicit for XYB.
                if colour_space != COLOUR_SPACE_XYB {
                    if r.bool()? {
                        r.bits(24)?; // gamma
                    } else {
                        r.enumeration()?;
                    }
                }
                r.enumeration()?; // rendering_intent
            }
        }
        if extra_fields && !r.bool()? {
            // ToneMapping: intensity_target, min_nits, relative_to_max_display,
            // linear_below.
            r.skip(32)?;
            r.bool()?;
            r.skip(16)?;
        }
        r.extensions()?;
    }

    // CustomTransformData
    if !r.bool()? {
        if xyb_encoded && !r.bool()? {
            r.skip(16 * 16)?; // OpsinInverseMatrix: 9 + 3 + 4 F16
        }
        let mask = r.bits(3)?;
        for (bit, weights) in [(1, 15u64), (2, 55), (4, 210)] {
            if mask & bit != 0 {
                r.skip(weights * 16)?;
            }
        }
    }
    Some(HeaderBundles {
        end_bits: r.pos,
        want_icc,
        have_preview,
    })
}

/// Length in bytes of `codestream`'s header: the signature, `SizeHeader`,
/// `ImageMetadata`, `CustomTransformData` and ICC profile, zero-padded to a
/// byte. libjxl writes exactly these bytes as `header_bytes` and puts them in
/// the first `jxlp` box when boxes come before the frames (`encode.cc`).
///
/// `icc` is the profile the encoder embedded. The ICC stream is entropy-coded,
/// so its end is found by encoding `icc` again and checking the codestream
/// holds those bits; a header that wants a profile when `icc` is `None` or
/// different returns `None`. So do a preview (its frame comes before the
/// first frame proper), padding bits that aren't zero, and anything that
/// doesn't parse or leaves no frame data.
pub fn codestream_header_len(codestream: &[u8], icc: Option<&[u8]>) -> Option<usize> {
    let bundles = read_header_bundles(codestream)?;
    if bundles.have_preview {
        return None;
    }
    let mut end = bundles.end_bits;
    if bundles.want_icc {
        let mut writer = crate::bit_writer::BitWriter::new();
        crate::icc::write_icc(icc?, &mut writer).ok()?;
        let n = writer.bits_written();
        let encoded = writer.finish_with_padding();
        for i in 0..n {
            let ours = (encoded[i / 8] >> (i % 8)) & 1;
            let at = end + i;
            if (*codestream.get(at / 8)? >> (at % 8)) & 1 != ours {
                return None;
            }
        }
        end += n;
    }
    let len = end.div_ceil(8);
    if len >= codestream.len() || (end % 8 != 0 && codestream[len - 1] >> (end % 8) != 0) {
        return None;
    }
    Some(len)
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
/// The box goes where libjxl puts a box queued before the frames: see
/// [`insert_box`]. A bare codestream (starts with `0xFF 0x0A`) becomes a
/// container first.
#[allow(dead_code)] // #76: thin delegating wrapper; unit tests exercise the family
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
/// The box goes where libjxl puts a box queued before the frames: see
/// [`insert_box`]. A bare codestream becomes a container first.
#[allow(dead_code)] // #76: thin delegating wrapper; unit tests exercise the family
#[must_use]
pub fn append_hcdr_box(jxl_data: &[u8], hcdr_payload: &[u8]) -> Vec<u8> {
    append_aux_box(jxl_data, b"hCdR", hcdr_payload)
}

/// Shared implementation for adding an auxiliary metadata box to either an
/// existing container or a bare codestream. Used by [`append_colr_box`] and
/// [`append_hcdr_box`]; the box goes where [`insert_box`] puts it.
#[allow(dead_code)] // #76: reached through append_colr_box / append_hcdr_box
fn append_aux_box(jxl_data: &[u8], box_type: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    insert_box(jxl_data, box_type, payload)
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
        // Header (32) + Exif (8+4+10) + jxlc (8+2)
        assert_eq!(result.len(), 32 + 22 + 10);
        // A codestream without a header to parse: the box goes ahead of it.
        let exif_start = 32;
        assert_eq!(&result[exif_start + 4..exif_start + 8], b"Exif");
        // Tiff offset prefix (4 zero bytes)
        assert_eq!(&result[exif_start + 8..exif_start + 12], &[0, 0, 0, 0]);
        assert_eq!(&result[32 + 22 + 4..32 + 22 + 8], b"jxlc");
    }

    #[test]
    fn test_container_with_xmp() {
        let codestream = b"\xFF\x0A";
        let xmp = b"<?xpacket begin='...'?>";
        let result = wrap_in_container(codestream, None, Some(xmp));
        // Header (32) + xml (8+23) + jxlc (8+2)
        assert_eq!(result.len(), 32 + 31 + 10);
        let xml_start = 32;
        assert_eq!(&result[xml_start + 4..xml_start + 8], b"xml ");
        assert_eq!(&result[32 + 31 + 4..32 + 31 + 8], b"jxlc");
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

        // A codestream without a header to parse: jhgm goes ahead of it.
        let jhgm_size = u32::from_be_bytes([result[32], result[33], result[34], result[35]]);
        assert_eq!(jhgm_size as usize, 8 + jhgm_payload.len());
        assert_eq!(&result[36..40], b"jhgm");
        assert_eq!(&result[40..40 + jhgm_payload.len()], jhgm_payload);

        let jxlc_offset = 40 + jhgm_payload.len();
        assert_eq!(&result[jxlc_offset + 4..jxlc_offset + 8], b"jxlc");
        assert_eq!(&result[jxlc_offset + 8..], codestream);
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

        // A codestream without a header to parse: jhgm goes ahead of the
        // jxlc box, which follows unchanged.
        assert_eq!(&result[..32], &container[..32]);
        assert_eq!(&result[36..40], b"jhgm");
        assert_eq!(&result[40..40 + jhgm_payload.len()], jhgm_payload);
        assert_eq!(&result[40 + jhgm_payload.len()..], &container[32..]);
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
        // Header (32) + jumb (8 + 16) + jxlc (8 + 4)
        assert_eq!(result.len(), 32 + 24 + 12);
        let jumb_start = 32;
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
        // Order: header (32) + Exif (8+4+exif.len) + jumb (8+jumbf.len) + jxlc (8+2)
        let expected = 32 + (8 + 4 + exif.len()) + (8 + jumbf.len()) + 10;
        assert_eq!(result.len(), expected);
        let jumb_offset = 32 + 8 + 4 + exif.len();
        assert_eq!(&result[jumb_offset + 4..jumb_offset + 8], b"jumb");
        assert_eq!(
            &result[jumb_offset + 8..jumb_offset + 8 + jumbf.len()],
            jumbf
        );
        assert_eq!(&result[expected - 10 + 4..expected - 10 + 8], b"jxlc");
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
        // A codestream without a header to parse: jumb goes ahead of it.
        let jumb_size = u32::from_be_bytes([result[32], result[33], result[34], result[35]]);
        assert_eq!(jumb_size as usize, 8 + jumbf.len());
        assert_eq!(&result[36..40], b"jumb");
        assert_eq!(&result[40..40 + jumbf.len()], jumbf);
        let jxlc_offset = 40 + jumbf.len();
        assert_eq!(&result[jxlc_offset + 4..jxlc_offset + 8], b"jxlc");
        assert_eq!(&result[jxlc_offset + 8..], codestream);
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

        // A codestream without a header to parse: jumb goes ahead of the jxlc
        // box, which follows unchanged.
        assert_eq!(&result[..32], &container[..32]);
        assert_eq!(&result[36..40], b"jumb");
        assert_eq!(&result[40..40 + jumbf.len()], jumbf);
        assert_eq!(&result[40 + jumbf.len()..], &container[32..]);
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
        let exif = [0x4D, 0x4D, 0x00, 0x2A].repeat(64); // 256-byte TIFF-MM-tagged blob
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
        assert_eq!(
            compute_codestream_level(1024, 1024, 0, false, 0, true),
            Some(5)
        );
        assert_eq!(
            compute_codestream_level(1024, 1024, 4, false, 1 << 22, true),
            Some(5)
        );
        assert_eq!(
            compute_codestream_level(262_144, 1, 0, false, 0, true),
            Some(5)
        );
        // 16384 × 16384 = 2²⁸ pixels, exactly the level-5 ceiling.
        assert_eq!(
            compute_codestream_level(16_384, 16_384, 0, false, 0, true),
            Some(5)
        );
    }

    #[test]
    fn test_compute_level_exceeding_level5_bumps_to_10() {
        // > 262 144 per axis bumps to level 10.
        assert_eq!(
            compute_codestream_level(262_145, 1, 0, false, 0, true),
            Some(10)
        );
        // > 2²⁸ total pixels bumps to level 10.
        assert_eq!(
            compute_codestream_level(16_385, 16_385, 0, false, 0, true),
            Some(10)
        );
        // ICC > 2²² bumps to level 10.
        assert_eq!(
            compute_codestream_level(1024, 1024, 0, false, (1 << 22) + 1, true),
            Some(10)
        );
        // 5+ extra channels bumps to level 10.
        assert_eq!(
            compute_codestream_level(1024, 1024, 5, false, 0, true),
            Some(10)
        );
        // CMYK (`Black`) extra channel bumps to level 10.
        assert_eq!(
            compute_codestream_level(1024, 1024, 1, true, 0, true),
            Some(10)
        );
    }

    #[test]
    fn test_compute_level_overflow_returns_none() {
        // Beyond the level-10 caps: not encodable.
        assert_eq!(
            compute_codestream_level(u32::MAX, 1, 0, false, 0, true),
            None
        );
        assert_eq!(
            compute_codestream_level(1, 1, 0, false, (1 << 28) + 1, true),
            None
        );
        assert_eq!(compute_codestream_level(1, 1, 257, false, 0, true), None);
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
        // jxll box still sits directly after ftyp; with a codestream whose
        // header doesn't parse, Exif goes ahead of the jxlc box.
        let codestream = b"\xFF\x0A";
        let exif = b"MM\x00\x2a\x00\x00";
        let result = wrap_in_container_with_level(codestream, Some(exif), None, 10);
        // 32 + 9 + (8 + 4 + 6) Exif + (8 + 2) jxlc
        assert_eq!(result.len(), 32 + 9 + 18 + 10);
        assert_eq!(&result[32..36], &[0, 0, 0, 0x09]);
        assert_eq!(&result[36..40], b"jxll");
        assert_eq!(result[40], 10);
        assert_eq!(&result[41 + 4..41 + 8], b"Exif");
        // jxlc starts at 32 + 9 + 18 = 59.
        assert_eq!(&result[59..63], &[0, 0, 0, 0x0A]); // jxlc size 10
        assert_eq!(&result[63..67], b"jxlc");
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
        // A codestream without a header to parse: colr goes ahead of it.
        let colr_size = u32::from_be_bytes([result[32], result[33], result[34], result[35]]);
        assert_eq!(colr_size as usize, 8 + payload.len());
        assert_eq!(&result[36..40], b"colr");
        assert_eq!(&result[40..40 + payload.len()], &payload);
        let jxlc_offset = 40 + payload.len();
        assert_eq!(&result[jxlc_offset + 4..jxlc_offset + 8], b"jxlc");
        assert_eq!(&result[jxlc_offset + 8..], codestream);
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

        assert_eq!(&result[..32], &container[..32]);
        assert_eq!(&result[36..40], b"colr");
        assert_eq!(&result[40..40 + payload.len()], payload.as_slice());
        assert_eq!(&result[40 + payload.len()..], &container[32..]);
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
        // A codestream without a header to parse: hCdR goes ahead of it.
        let hcdr_size = u32::from_be_bytes([result[32], result[33], result[34], result[35]]);
        assert_eq!(hcdr_size as usize, 8 + payload.len());
        // Case-sensitive FourCC: capital C and R.
        assert_eq!(&result[36..40], b"hCdR");
        assert_eq!(&result[40..40 + payload.len()], &payload);
        let jxlc_offset = 40 + payload.len();
        assert_eq!(&result[jxlc_offset + 4..jxlc_offset + 8], b"jxlc");
        assert_eq!(&result[jxlc_offset + 8..], codestream);
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

        assert_eq!(&result[..32], &container[..32]);
        assert_eq!(&result[36..40], b"hCdR");
        assert_eq!(&result[40..40 + payload.len()], payload);
        assert_eq!(&result[40 + payload.len()..], &container[32..]);
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

    // ─── libjxl's box order on real codestream headers ─────────────

    use crate::bit_writer::BitWriter;
    use crate::headers::color_encoding::{CIExy, CustomPrimaries, RenderingIntent};
    use crate::headers::extra_channels::{ExtraChannelInfo, ExtraChannelType};
    use crate::headers::file_header::{AnimationHeader, BitDepth, Orientation};
    use crate::headers::{ColorEncoding, FileHeader};

    /// A codestream with `header` (and `icc`) followed by stand-in frame bytes,
    /// and the header's length as the writer produced it.
    fn codestream_with(header: &FileHeader, icc: Option<&[u8]>) -> (Vec<u8>, usize) {
        let mut writer = BitWriter::new();
        header.write(&mut writer).unwrap();
        if let Some(icc) = icc {
            crate::icc::write_icc(icc, &mut writer).unwrap();
        }
        writer.zero_pad_to_byte();
        let len = writer.bytes_written();
        let mut codestream = writer.finish_with_padding();
        codestream.extend_from_slice(&[0x5A; 16]);
        (codestream, len)
    }

    /// The 256-byte profile `icc::tests` builds, with one tag of type `tag`.
    fn test_icc(tag: &[u8; 4]) -> Vec<u8> {
        let mut icc = vec![0u8; 256];
        icc[..4].copy_from_slice(&256u32.to_be_bytes());
        icc[12..16].copy_from_slice(b"mntr");
        icc[16..20].copy_from_slice(b"RGB ");
        icc[20..24].copy_from_slice(b"XYZ ");
        icc[36..40].copy_from_slice(b"acsp");
        icc[128..132].copy_from_slice(&1u32.to_be_bytes());
        icc[132..136].copy_from_slice(tag);
        icc[136..140].copy_from_slice(&144u32.to_be_bytes());
        icc[140..144].copy_from_slice(&12u32.to_be_bytes());
        icc[144..148].copy_from_slice(b"text");
        icc
    }

    /// Headers that reach every branch of the header parse this crate writes.
    fn header_variants() -> Vec<(String, FileHeader)> {
        let mut out: Vec<(String, FileHeader)> = vec![
            ("rgb 64x64".into(), FileHeader::new_rgb(64, 64)),
            ("gray 37x1000".into(), FileHeader::new_gray(37, 1000)),
            ("rgba 300x200".into(), FileHeader::new_rgba(300, 200)),
            ("gray+alpha 1x1".into(), FileHeader::new_gray_alpha(1, 1)),
            ("rgb 2320x3408".into(), FileHeader::new_rgb(2320, 3408)),
            ("rgb 300000x3".into(), FileHeader::new_rgb(300_000, 3)),
        ];
        let mut h = FileHeader::new_rgb(256, 256);
        h.metadata.xyb_encoded = true;
        h.metadata.color_encoding.rendering_intent = RenderingIntent::Relative;
        h.header_all_default_fast_paths = true;
        out.push(("all-default ImageMetadata".into(), h));
        let mut h = FileHeader::new_rgba(640, 480);
        h.metadata.bit_depth = BitDepth::uint16();
        for ec in &mut h.metadata.extra_channels {
            ec.bit_depth = BitDepth::uint16();
        }
        out.push(("rgba 16-bit".into(), h));
        let mut h = FileHeader::new_rgb(8, 8);
        h.metadata.bit_depth = BitDepth::float32();
        out.push(("float32".into(), h));
        let mut h = FileHeader::new_gray(2320, 3408);
        h.metadata.bit_depth.bits_per_sample = 1;
        h.metadata.color_encoding = ColorEncoding::gray();
        out.push(("gray 1-bit".into(), h));
        let mut h = FileHeader::new_rgba(64, 48);
        h.metadata.animation = Some(AnimationHeader {
            tps_numerator: 1000,
            tps_denominator: 1,
            num_loops: 0,
            have_timecodes: false,
        });
        out.push(("animation".into(), h));
        let mut h = FileHeader::new_rgb(64, 48);
        h.metadata.animation = Some(AnimationHeader {
            tps_numerator: 30_000,
            tps_denominator: 1001,
            num_loops: 70_000,
            have_timecodes: true,
        });
        out.push(("animation with timecodes".into(), h));
        let mut h = FileHeader::new_rgb(4000, 3000);
        h.metadata.orientation = Orientation::Rotate90CW;
        h.metadata.intensity_target = 1000.0;
        h.metadata.min_nits = 0.005;
        h.metadata.relative_to_max_display = true;
        h.metadata.linear_below = 0.25;
        out.push(("orientation and tone mapping".into(), h));
        let mut h = FileHeader::new_rgb(64, 64);
        h.metadata.extra_channels = vec![
            ExtraChannelInfo {
                name: "depth map".into(),
                dim_shift: 3,
                ..ExtraChannelInfo::depth()
            },
            ExtraChannelInfo::spot_color([0.1, 0.2, 0.3, 0.9]),
            ExtraChannelInfo {
                ec_type: ExtraChannelType::Cfa,
                cfa_channel: 2,
                ..Default::default()
            },
            ExtraChannelInfo {
                name: "x".repeat(70),
                alpha_associated: true,
                ..ExtraChannelInfo::alpha()
            },
        ];
        out.push(("named extra channels".into(), h));
        let primaries = CustomPrimaries {
            red: CIExy { x: 0.68, y: 0.32 },
            green: CIExy { x: 0.265, y: 0.69 },
            blue: CIExy { x: 0.15, y: 0.06 },
        };
        for (name, colour) in [
            ("display p3", ColorEncoding::display_p3()),
            ("gamma", ColorEncoding::with_gamma(0.45455)),
            ("pq", ColorEncoding::bt2100_pq()),
            (
                "custom white point and primaries",
                ColorEncoding::with_custom_white_point_and_primaries(
                    CIExy {
                        x: 0.3127,
                        y: 0.329,
                    },
                    primaries,
                ),
            ),
        ] {
            let mut h = FileHeader::new_rgb(64, 64);
            h.metadata.color_encoding = colour;
            out.push((name.into(), h));
        }
        let mut h = FileHeader::new_rgb(640, 480);
        h.metadata.have_intrinsic_size = true;
        h.metadata.intrinsic_width = 1280;
        h.metadata.intrinsic_height = 960;
        out.push(("intrinsic size".into(), h));
        for factor in [2, 4, 8] {
            let mut h = FileHeader::new_rgb(64, 64);
            h.metadata.xyb_encoded = true;
            h.upsampling_factor = factor;
            h.upsampling_mode = Some(0);
            out.push((format!("upsampling weights x{factor}"), h));
        }
        out
    }

    use super::box_types as types;

    /// A container's codestream: its `jxlc` box, or its `jxlp` boxes joined.
    fn codestream_of(data: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        for b in top_level_boxes(data).unwrap() {
            let skip = match &b.box_type {
                b"jxlc" => 0,
                b"jxlp" => 4,
                _ => continue,
            };
            out.extend_from_slice(&data[b.offset + b.header + skip..b.offset + b.size]);
        }
        out
    }

    #[test]
    fn test_header_len_matches_writer() {
        for (name, header) in header_variants() {
            let (codestream, len) = codestream_with(&header, None);
            assert_eq!(
                codestream_header_len(&codestream, None),
                Some(len),
                "{name}"
            );
        }
    }

    #[test]
    fn test_header_len_with_icc() {
        let icc = test_icc(b"cprt");
        for (name, mut header) in header_variants() {
            header.metadata.color_encoding.want_icc = true;
            let (codestream, len) = codestream_with(&header, Some(&icc));
            assert_eq!(
                codestream_header_len(&codestream, Some(&icc)),
                Some(len),
                "{name}"
            );
            // The profile's end can't be found without it, or with another one.
            assert_eq!(codestream_header_len(&codestream, None), None, "{name}");
            assert_eq!(
                codestream_header_len(&codestream, Some(&test_icc(b"desc"))),
                None,
                "{name}"
            );
        }
    }

    #[test]
    fn test_header_len_rejects_preview_and_bad_input() {
        // Signature, an 8x8 SizeHeader, and ImageMetadata with or without a
        // 128x128 preview; then CustomTransformData.
        let header = |preview: bool| {
            let mut w = BitWriter::new();
            FileHeader::write_signature(&mut w).unwrap();
            for (bits, value) in [(1, 1), (5, 0), (3, 1), (1, 0), (1, 1), (3, 0), (1, 0)] {
                w.write(bits, value).unwrap();
            }
            w.write_bit(preview).unwrap();
            if preview {
                w.write(1, 1).unwrap(); // div8
                w.write(2, 0).unwrap(); // ysize_div8 = 16
                w.write(3, 1).unwrap(); // ratio 1:1
            }
            // have_animation, bit depth (integer, 8), modular_16_bit,
            // no extra channels, xyb_encoded, colour and tone mapping
            // all-default, no extensions, CustomTransformData all-default.
            for (bits, value) in [(1, 0), (1, 0), (2, 0), (1, 1), (2, 0), (1, 1), (1, 1)] {
                w.write(bits, value).unwrap();
            }
            for (bits, value) in [(1, 1), (2, 0), (1, 1)] {
                w.write(bits, value).unwrap();
            }
            w.zero_pad_to_byte();
            let len = w.bytes_written();
            let mut codestream = w.finish_with_padding();
            codestream.extend_from_slice(&[0x5A; 16]);
            (codestream, len)
        };
        let (codestream, len) = header(false);
        assert_eq!(codestream_header_len(&codestream, None), Some(len));
        // A preview's frame follows the header: the frames don't start there.
        let (codestream, _) = header(true);
        assert_eq!(codestream_header_len(&codestream, None), None);

        assert_eq!(codestream_header_len(b"\x00\x00\x00\x00", None), None);
        let (codestream, len) = codestream_with(&FileHeader::new_rgb(64, 64), None);
        // Truncated, or nothing after the header.
        assert_eq!(codestream_header_len(&codestream[..len - 1], None), None);
        assert_eq!(codestream_header_len(&codestream[..len], None), None);
    }

    #[test]
    fn test_wrap_splits_after_header() {
        let (codestream, len) = codestream_with(&FileHeader::new_rgb(64, 64), None);
        let exif = b"MM\x00\x2a\x00\x00\x00\x08";
        let xmp = b"<x:xmpmeta/>";
        let jumbf = b"\x00\x00\x00\x0Cjumdc2pa";
        let result = wrap_in_container_with_jbrd_and_level_and_jumbf(
            &codestream,
            None,
            Some(exif),
            Some(xmp),
            10,
            Some(jumbf),
        );
        assert_eq!(
            types(&result),
            [
                *b"JXL ", *b"ftyp", *b"jxll", *b"jxlp", *b"Exif", *b"xml ", *b"jumb", *b"jxlp"
            ]
        );
        // jxlp 0 holds exactly the header, jxlp 1 (last) the frames.
        let boxes = top_level_boxes(&result).unwrap();
        let payload = |b: &TopBox| &result[b.offset + b.header..b.offset + b.size];
        assert_eq!(
            payload(&boxes[3]),
            [&0u32.to_be_bytes()[..], &codestream[..len]].concat()
        );
        assert_eq!(
            payload(&boxes[7]),
            [&0x8000_0001u32.to_be_bytes()[..], &codestream[len..]].concat()
        );
        // No boxes: one jxlc box, as libjxl writes it.
        assert_eq!(
            types(&wrap_in_container_with_level(&codestream, None, None, 10)),
            [*b"JXL ", *b"ftyp", *b"jxll", *b"jxlc"]
        );
    }

    #[test]
    fn test_wrap_encoded_finds_header_with_icc() {
        let icc = test_icc(b"cprt");
        let mut header = FileHeader::new_rgb(64, 64);
        header.metadata.color_encoding.want_icc = true;
        let (codestream, len) = codestream_with(&header, Some(&icc));
        let boxes = EncodedBoxes {
            xmp: Some(b"<x/>"),
            colr: Some(b"nclx"),
            hcdr: Some(b"hdr"),
            ..Default::default()
        };
        let result = wrap_encoded(codestream.clone(), Some(&icc), 5, &boxes, None);
        assert_eq!(
            types(&result),
            [
                *b"JXL ", *b"ftyp", *b"jxlp", *b"xml ", *b"colr", *b"hCdR", *b"jxlp"
            ]
        );
        let first = &top_level_boxes(&result).unwrap()[2];
        assert_eq!(first.size, 8 + 4 + len);
        assert_eq!(codestream_of(&result), codestream);
        // No boxes at level 5: the bare codestream.
        assert_eq!(
            wrap_encoded(
                codestream.clone(),
                Some(&icc),
                5,
                &EncodedBoxes::default(),
                None
            ),
            codestream
        );
    }

    #[test]
    fn test_insert_box_at_libjxl_position() {
        let (codestream, _) = codestream_with(&FileHeader::new_rgb(64, 64), None);
        // A bare codestream: split after the header.
        let bare = append_gain_map_box(&codestream, b"gain map");
        assert_eq!(
            types(&bare),
            [*b"JXL ", *b"ftyp", *b"jxlp", *b"jhgm", *b"jxlp"]
        );
        // A split container: after the boxes already there, ahead of the frames.
        let split = append_jumbf_box(
            &wrap_in_container(&codestream, None, Some(b"<x/>")),
            b"c2pa",
        );
        assert_eq!(
            types(&split),
            [*b"JXL ", *b"ftyp", *b"jxlp", *b"xml ", *b"jumb", *b"jxlp"]
        );
        // A jxlc container: split, the level box kept.
        let jxlc = append_gain_map_box(
            &wrap_in_container_with_level(&codestream, None, None, 10),
            b"gain map",
        );
        assert_eq!(
            types(&jxlc),
            [*b"JXL ", *b"ftyp", *b"jxll", *b"jxlp", *b"jhgm", *b"jxlp"]
        );
        // A box already after the codestream stays there.
        let mut old = Vec::new();
        old.extend_from_slice(&JXL_CONTAINER_SIGNATURE);
        old.extend_from_slice(&FTYP_BOX);
        write_box(&mut old, b"jxlc", &codestream);
        write_box(&mut old, b"xml ", b"<x/>");
        let old = append_hcdr_box(&old, b"hdr");
        assert_eq!(
            types(&old),
            [*b"JXL ", *b"ftyp", *b"jxlp", *b"hCdR", *b"jxlp", *b"xml "]
        );
        for result in [&bare, &split, &jxlc, &old] {
            assert_eq!(codestream_of(result), codestream);
        }
    }

    #[cfg(feature = "jpeg-reencoding")]
    #[test]
    fn test_jpeg_layout_puts_metadata_before_frames() {
        let exif = [0x4D, 0x4D, 0x00, 0x2A].repeat(64);
        let xmp = b"<x:xmpmeta/>";
        let result =
            wrap_in_container_jxlp(b"\xFF\x0A\x00", b"frames", b"jbrd", Some(&exif), Some(xmp));
        // Exif compresses into a brob box; the short XMP stays plain.
        assert_eq!(
            types(&result),
            [
                *b"JXL ", *b"ftyp", *b"jxlp", *b"jbrd", *b"brob", *b"xml ", *b"jxlp"
            ]
        );
        // A gain map goes after them, ahead of the frames.
        let at = frames_box_offset(&result).unwrap();
        assert_eq!(&result[at + 4..at + 8], b"jxlp");
        assert_eq!(top_level_boxes(&result).unwrap()[6].offset, at);
    }
}
