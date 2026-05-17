// Copyright (c) Imazen LLC.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing
//
// Wire format matches libjxl `gain_map.cc:JxlGainMapWriteBundle` exactly.

//! Typed `GainMapBundle` mirroring libjxl `JxlGainMapBundle`.

use alloc::vec::Vec;

use crate::api::EncodeError;
use crate::bit_writer::BitWriter;
use crate::container;
use crate::headers::color_encoding::ColorEncoding;

/// Helper: build an `EncodeError::InvalidInput` from a `String`.
/// The variant is a struct variant — re-typing `{ message: ... }` at
/// every call site is noise.
#[inline]
fn invalid(message: alloc::string::String) -> EncodeError {
    EncodeError::InvalidInput { message }
}

/// Owned, typed mirror of libjxl `JxlGainMapBundle` (`gain_map.h:38`).
///
/// Field ordering, semantics, and serialized wire format all match
/// libjxl 0.12.0. The `jxl-encoder` version is owned (`Vec<u8>` rather
/// than `*const u8` + `size`) and gets its color-encoding bytes from
/// our own [`ColorEncoding::write`] implementation instead of libjxl's
/// internal `jxl::Bundle::Write`.
///
/// # Wire layout (produced by [`Self::serialize`])
///
/// | field                      | size                                         |
/// |----------------------------|----------------------------------------------|
/// | `jhgm_version`             | 1 byte                                       |
/// | `gain_map_metadata_size`   | 2 bytes (BE u16)                             |
/// | `gain_map_metadata`        | `gain_map_metadata_size` bytes               |
/// | `color_encoding_size`      | 1 byte (0 if no color encoding present)      |
/// | `color_encoding`           | `color_encoding_size` bytes (byte-aligned)   |
/// | `alt_icc_size`             | 4 bytes (BE u32)                             |
/// | `alt_icc`                  | `alt_icc_size` bytes (compressed ICC)        |
/// | `gain_map`                 | remaining bytes (raw JXL codestream)         |
///
/// Cross-validated against `libjxl/lib/extras/gain_map.cc:83-153`.
#[derive(Clone, Debug)]
pub struct GainMapBundle {
    /// Version of the bundle layout. Always `0` for the current
    /// format; reserved for future ISO 21496-1 revisions.
    pub jhgm_version: u8,
    /// Raw ISO 21496-1 metadata payload — the bytes produced by
    /// `ultrahdr_core::serialize_iso21496_fmt(params, Iso21496Format::JxlJhgm)`.
    /// Must fit in a `u16` (≤ 65 535 bytes); the spec-defined metadata
    /// blob is always well under 1 KB so this is a practical no-op.
    pub iso21496_metadata: Vec<u8>,
    /// Color encoding of the gain map itself.
    ///
    /// `None` if the gain map inherits the base image's color encoding
    /// (the typical case for single-channel luminance gain maps —
    /// they're conceptually grayscale).
    pub color_encoding: Option<ColorEncoding>,
    /// Compressed ICC profile bytes for the gain map's alternate color
    /// space. Set when the HDR "alternate" image has an ICC profile
    /// distinct from the SDR base. Empty when not needed.
    pub alt_icc: Vec<u8>,
    /// Bare JXL codestream encoding the gain-map plane (typically a
    /// single-channel modular lossless JXL of the 8-bit gain values).
    /// Decoders read this back and treat it as a normal JXL stream.
    pub gain_map_codestream: Vec<u8>,
}

impl GainMapBundle {
    /// Construct a new bundle from raw components.
    ///
    /// This is a thin convenience over a struct literal; it exists so the
    /// `jhgm_version` default (`0`) lives in one place.
    #[must_use]
    pub fn new(iso21496_metadata: Vec<u8>, gain_map_codestream: Vec<u8>) -> Self {
        Self {
            jhgm_version: 0,
            iso21496_metadata,
            color_encoding: None,
            alt_icc: Vec::new(),
            gain_map_codestream,
        }
    }

    /// Set an explicit color encoding for the gain-map plane. Optional.
    #[must_use]
    pub fn with_color_encoding(mut self, ce: ColorEncoding) -> Self {
        self.color_encoding = Some(ce);
        self
    }

    /// Set the compressed alternate ICC profile. Optional.
    ///
    /// Pass the **compressed** ICC bytes (the output of the JXL
    /// `PredictICC + Huffman` encoder) — libjxl writes the bytes here
    /// verbatim; matching its wire format means matching its choice
    /// of compressed bytes too. Most callers leave this empty.
    #[must_use]
    pub fn with_alt_icc(mut self, alt_icc: Vec<u8>) -> Self {
        self.alt_icc = alt_icc;
        self
    }

    /// Compute the exact number of bytes [`Self::serialize`] will produce.
    ///
    /// Mirrors `JxlGainMapGetBundleSize` (`gain_map.cc:55-81`).
    ///
    /// # Errors
    ///
    /// Returns [`EncodeError::InvalidInput`] if the metadata payload
    /// exceeds the u16 limit, or if the color-encoding bits exceed
    /// 255 bytes after byte-padding.
    pub fn serialized_size(&self) -> Result<usize, EncodeError> {
        if self.iso21496_metadata.len() > u16::MAX as usize {
            return Err(invalid(format!(
                "gain_map_metadata too large: {} bytes (max {})",
                self.iso21496_metadata.len(),
                u16::MAX
            )));
        }
        if self.alt_icc.len() > u32::MAX as usize {
            return Err(invalid(format!(
                "alt_icc too large: {} bytes (max {})",
                self.alt_icc.len(),
                u32::MAX
            )));
        }

        let color_encoding_bytes = self.color_encoding_bytes()?;
        if color_encoding_bytes.len() > u8::MAX as usize {
            return Err(invalid(format!(
                "color_encoding serialized to {} bytes (max {})",
                color_encoding_bytes.len(),
                u8::MAX
            )));
        }

        Ok(
            1                                                    // jhgm_version
            + 2                                                  // gain_map_metadata_size
            + self.iso21496_metadata.len()
            + 1                                                  // color_encoding_size
            + color_encoding_bytes.len()
            + 4                                                  // alt_icc_size
            + self.alt_icc.len()
            + self.gain_map_codestream.len(),
        )
    }

    /// Serialize the bundle to bytes ready for embedding in a `jhgm`
    /// container box.
    ///
    /// Output matches libjxl `JxlGainMapWriteBundle` byte-for-byte.
    ///
    /// # Errors
    ///
    /// Returns [`EncodeError::InvalidInput`] for any of the bounds
    /// violations documented on [`Self::serialized_size`].
    pub fn serialize(&self) -> Result<Vec<u8>, EncodeError> {
        let total = self.serialized_size()?;
        let mut out = Vec::with_capacity(total);

        // 1. jhgm_version (1 byte)
        out.push(self.jhgm_version);

        // 2. gain_map_metadata_size (2 bytes, big-endian)
        let metadata_size = self.iso21496_metadata.len() as u16;
        out.extend_from_slice(&metadata_size.to_be_bytes());

        // 3. gain_map_metadata (metadata_size bytes)
        out.extend_from_slice(&self.iso21496_metadata);

        // 4. color_encoding_size (1 byte) + color_encoding bytes
        let color_encoding_bytes = self.color_encoding_bytes()?;
        out.push(color_encoding_bytes.len() as u8);
        out.extend_from_slice(&color_encoding_bytes);

        // 5. alt_icc_size (4 bytes, big-endian) + alt_icc bytes
        let alt_icc_size = self.alt_icc.len() as u32;
        out.extend_from_slice(&alt_icc_size.to_be_bytes());
        out.extend_from_slice(&self.alt_icc);

        // 6. gain_map_codestream (remaining bytes)
        out.extend_from_slice(&self.gain_map_codestream);

        debug_assert_eq!(
            out.len(),
            total,
            "serialized_size {total} != actual {}",
            out.len()
        );
        Ok(out)
    }

    /// Produce the byte-aligned encoding of `self.color_encoding`, or
    /// an empty vector when no color encoding is set.
    ///
    /// libjxl writes a `BitWriter`, calls `ZeroPadToByte`, and uses the
    /// resulting `Span<const uint8_t>` (`gain_map.cc:96-104, 127-129`).
    /// We do the same with our own [`BitWriter`].
    fn color_encoding_bytes(&self) -> Result<Vec<u8>, EncodeError> {
        let Some(ref ce) = self.color_encoding else {
            return Ok(Vec::new());
        };
        let mut bw = BitWriter::new();
        ce.write(&mut bw)?;
        Ok(bw.finish_with_padding())
    }
}

/// Wrap [`crate::container::append_gain_map_box`] for a typed bundle.
///
/// Serializes `bundle` and appends a `jhgm` box to `jxl_data`. If
/// `jxl_data` is a bare codestream, it is wrapped in a container first.
///
/// # Errors
///
/// Forwards any error from [`GainMapBundle::serialize`].
pub fn append_gain_map_bundle(
    jxl_data: &[u8],
    bundle: &GainMapBundle,
) -> Result<Vec<u8>, EncodeError> {
    let payload = bundle.serialize()?;
    Ok(container::append_gain_map_box(jxl_data, &payload))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::headers::color_encoding::ColorEncoding;

    #[test]
    fn empty_bundle_layout_matches_libjxl_shape() {
        // jhgm_version + meta_size(=0) + cenc_size(=0) + alt_icc_size(=0)
        // = 1 + 2 + 1 + 4 = 8 bytes
        let b = GainMapBundle::new(Vec::new(), Vec::new());
        let bytes = b.serialize().unwrap();
        assert_eq!(bytes.len(), 8);
        assert_eq!(bytes[0], 0); // jhgm_version
        assert_eq!(&bytes[1..3], &[0, 0]); // metadata_size BE
        assert_eq!(bytes[3], 0); // color_encoding_size
        assert_eq!(&bytes[4..8], &[0, 0, 0, 0]); // alt_icc_size BE
    }

    #[test]
    fn serialized_size_matches_serialize_len() {
        let meta = vec![1u8, 2, 3, 4, 5];
        let codestream = vec![0xFFu8, 0x0A, 0x10, 0x20, 0x30];
        let b = GainMapBundle::new(meta.clone(), codestream.clone());
        let expected = b.serialized_size().unwrap();
        let actual = b.serialize().unwrap();
        assert_eq!(actual.len(), expected);
    }

    #[test]
    fn metadata_size_is_big_endian() {
        // 257 = 0x0101 → MSB first per libjxl `StoreBE16`.
        let meta = vec![0u8; 257];
        let b = GainMapBundle::new(meta, Vec::new());
        let bytes = b.serialize().unwrap();
        assert_eq!(&bytes[1..3], &[0x01, 0x01]);
    }

    #[test]
    fn alt_icc_size_is_big_endian() {
        let codestream = vec![0xFFu8, 0x0A];
        let alt_icc = vec![0x42u8; 0x00010203];
        let b = GainMapBundle::new(Vec::new(), codestream.clone()).with_alt_icc(alt_icc);
        let bytes = b.serialize().unwrap();
        // After: jhgm_ver(1) + meta_size(2,=0) + cenc_size(1,=0) = 4 bytes
        // alt_icc_size at offset 4, 4 BE bytes
        assert_eq!(&bytes[4..8], &[0x00, 0x01, 0x02, 0x03]);
    }

    #[test]
    fn gain_map_codestream_appears_at_tail() {
        let codestream = vec![0xDEu8, 0xAD, 0xBE, 0xEF];
        let b = GainMapBundle::new(vec![0x77u8], codestream.clone());
        let bytes = b.serialize().unwrap();
        // jhgm_ver(1) + meta_size(2,=1) + meta(1) + cenc_size(1,=0)
        // + alt_icc_size(4,=0) + codestream(4)
        assert_eq!(bytes.len(), 1 + 2 + 1 + 1 + 4 + 4);
        assert_eq!(&bytes[bytes.len() - 4..], &codestream[..]);
    }

    #[test]
    fn with_color_encoding_serializes_byte_aligned() {
        // sRGB color encoding produces a 1-bit `all_default=1` payload.
        // After zero-padding to byte, that becomes a single 0x01 byte.
        let b =
            GainMapBundle::new(Vec::new(), Vec::new()).with_color_encoding(ColorEncoding::srgb());
        let bytes = b.serialize().unwrap();
        // jhgm_ver(1) + meta_size(2) + cenc_size(1,=1) + cenc(1) + alt_icc_size(4)
        // = 9 bytes
        assert_eq!(bytes.len(), 9);
        assert_eq!(bytes[3], 1, "color_encoding_size for srgb default");
        assert_eq!(bytes[4], 0x01, "srgb all_default bit packed to LSB");
    }

    #[test]
    fn append_gain_map_bundle_wraps_bare_codestream() {
        let bare = b"\xFF\x0A\x00\x01\x02";
        let bundle = GainMapBundle::new(vec![0u8], b"\xFF\x0Again".to_vec());
        let container_bytes = append_gain_map_bundle(bare, &bundle).unwrap();
        // Must start with the JXL container signature.
        assert_eq!(&container_bytes[..4], &[0u8, 0, 0, 0x0C]);
        // Must contain "jhgm" somewhere downstream.
        let needle = b"jhgm";
        assert!(
            container_bytes.windows(needle.len()).any(|w| w == needle),
            "expected jhgm box in output container"
        );
    }

    #[test]
    fn metadata_too_large_is_rejected() {
        let meta = vec![0u8; u16::MAX as usize + 1];
        let b = GainMapBundle::new(meta, Vec::new());
        assert!(b.serialize().is_err());
    }
}
