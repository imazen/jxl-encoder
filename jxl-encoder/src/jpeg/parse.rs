// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! JPEG parsing for lossless reencoding.
//!
//! Uses zenjpeg for reliable coefficient extraction, plus a lightweight marker
//! scanner for the jbrd reconstruction metadata.

use super::data::*;

/// Errors that can occur during JPEG parsing.
#[derive(Debug, thiserror::Error)]
pub enum JpegError {
    #[error("unexpected end of JPEG data")]
    UnexpectedEof,
    #[error("invalid JPEG: {0}")]
    Invalid(String),
    #[error("unsupported JPEG feature: {0}")]
    Unsupported(String),
    #[error("zenjpeg decode error: {0}")]
    Decode(String),
}

type Result<T> = core::result::Result<T, JpegError>;

/// Parse a JPEG file and extract all data needed for lossless reencoding.
///
/// This performs two passes:
/// 1. Marker scan: extracts marker structure, Huffman/quant tables, APP/COM data
/// 2. zenjpeg decode: extracts quantized DCT coefficients reliably
pub fn read_jpeg(data: &[u8]) -> Result<JpegData> {
    // Phase 1: Scan markers for jbrd metadata
    let mut jpeg = scan_markers(data)?;

    // Phase 2: Use zenjpeg for reliable coefficient extraction
    extract_coefficients_zenjpeg(data, &mut jpeg)?;

    Ok(jpeg)
}

/// Lightweight marker scanner that reads JPEG marker structure without
/// decoding entropy data. Extracts everything needed for jbrd box serialization.
fn scan_markers(data: &[u8]) -> Result<JpegData> {
    if data.len() < 2 || data[0] != 0xFF || data[1] != 0xD8 {
        return Err(JpegError::Invalid("missing SOI marker".into()));
    }

    let mut jpeg = JpegData {
        width: 0,
        height: 0,
        restart_interval: 0,
        app_data: Vec::new(),
        app_marker_type: Vec::new(),
        com_data: Vec::new(),
        quant: Vec::new(),
        huffman_code: Vec::new(),
        components: Vec::new(),
        scan_info: Vec::new(),
        marker_order: vec![],
        inter_marker_data: vec![Vec::new()],
        tail_data: Vec::new(),
        has_zero_padding_bit: false,
        padding_bits: Vec::new(),
        component_type: JpegComponentType::YCbCr,
    };

    let mut pos = 2usize; // After SOI
    let mut seen_jfif = false;
    let mut adobe_transform: Option<u8> = None;
    let mut have_exif = false;
    let mut have_xmp = false;

    loop {
        // Skip to next marker (0xFF followed by non-zero byte)
        while pos < data.len() && data[pos] != 0xFF {
            // Inter-marker data
            if let Some(imd) = jpeg.inter_marker_data.last_mut() {
                imd.push(data[pos]);
            }
            pos += 1;
        }

        // Skip fill bytes (consecutive 0xFF)
        while pos < data.len() && data[pos] == 0xFF {
            pos += 1;
        }

        if pos >= data.len() {
            break;
        }

        let marker = data[pos];
        pos += 1;

        if marker == 0x00 {
            continue; // Byte stuffing outside entropy data, skip
        }

        jpeg.marker_order.push(marker);
        jpeg.inter_marker_data.push(Vec::new());

        match marker {
            0xD9 => {
                // EOI
                if pos < data.len() {
                    jpeg.tail_data = data[pos..].to_vec();
                }
                break;
            }
            0xC0 | 0xC1 => {
                // SOF0 (baseline) / SOF1 (extended sequential)
                parse_sof_marker(data, &mut pos, &mut jpeg)?;
            }
            0xC2 => {
                // SOF2: progressive JPEG. zenjpeg's `decode_coefficients()`
                // accepts SOF2 sources and merges per-scan band updates into
                // final 64-coefficient blocks before returning, so the
                // downstream encoder (`encode_jpeg_to_jxl`) and JBRD writer
                // (which already loops `for scan in scan_info`) see exactly
                // the same shape as a baseline (SOF0) source.
                //
                // The two relevant differences from baseline parsing:
                //   1. Multiple SOS markers (one per progressive scan); the
                //      main loop already handles this — `parse_sos_header`
                //      appends to `scan_info` on every visit.
                //   2. The SOS markers carry non-default
                //      `Ss / Se / Ah / Al`; `parse_sos_header` stores those
                //      verbatim into the new scan_info entry, and `jbrd.rs`
                //      already serializes them.
                //
                // `last_needed_pass` (per-scan field in JBRD) is the only
                // value we currently hard-code to 0. For multi-scan inputs
                // this needs the per-scan pass index — handled at the JBRD
                // writer rather than in this branch.
                parse_sof_marker(data, &mut pos, &mut jpeg)?;
            }
            0xDB => {
                // DQT
                parse_dqt_marker(data, &mut pos, &mut jpeg)?;
            }
            0xC4 => {
                // DHT
                parse_dht_marker(data, &mut pos, &mut jpeg)?;
            }
            0xDD => {
                // DRI
                if pos + 3 >= data.len() {
                    return Err(JpegError::UnexpectedEof);
                }
                let _len = u16::from_be_bytes([data[pos], data[pos + 1]]);
                jpeg.restart_interval = u16::from_be_bytes([data[pos + 2], data[pos + 3]]) as u32;
                pos += 4;
            }
            0xDA => {
                // SOS - scan header, then skip entropy-coded data
                parse_sos_header(data, &mut pos, &mut jpeg)?;
                // Skip entropy-coded segment (find next marker)
                skip_entropy_data(data, &mut pos);
            }
            0xE0..=0xEF => {
                // APP markers
                if pos + 1 >= data.len() {
                    return Err(JpegError::UnexpectedEof);
                }
                let len = u16::from_be_bytes([data[pos], data[pos + 1]]) as usize;
                if pos + len > data.len() || len < 2 {
                    return Err(JpegError::Invalid("APP marker length invalid".into()));
                }
                let payload = &data[pos + 2..pos + len];

                let mut marker_type = classify_app_marker(marker, payload);
                // libjxl only marks the FIRST Exif and FIRST XMP marker;
                // duplicates remain Unknown (go in JBRD data stream).
                match marker_type {
                    AppMarkerType::Exif if have_exif => marker_type = AppMarkerType::Unknown,
                    AppMarkerType::Exif => have_exif = true,
                    AppMarkerType::Xmp if have_xmp => marker_type = AppMarkerType::Unknown,
                    AppMarkerType::Xmp => have_xmp = true,
                    _ => {}
                }
                if marker == 0xE0 && payload.starts_with(b"JFIF\0") {
                    seen_jfif = true;
                }
                if marker == 0xEE && payload.len() >= 12 && payload.starts_with(b"Adobe") {
                    adobe_transform = Some(payload[11]);
                }

                // Store raw APP data: marker_type + length_field + payload
                // Format: [marker_byte, len_hi, len_lo, payload...]
                // This matches libjxl's JPEGData format (size = marker_len + 1)
                let mut raw = Vec::with_capacity(1 + len);
                raw.push(marker);
                raw.extend_from_slice(&data[pos..pos + len]);
                jpeg.app_data.push(raw);
                jpeg.app_marker_type.push(marker_type);

                pos += len;
            }
            0xFE => {
                // COM
                if pos + 1 >= data.len() {
                    return Err(JpegError::UnexpectedEof);
                }
                let len = u16::from_be_bytes([data[pos], data[pos + 1]]) as usize;
                if pos + len > data.len() || len < 2 {
                    return Err(JpegError::Invalid("COM marker length invalid".into()));
                }
                // Store raw COM data: marker_type + length_field + payload
                // Format: [0xFE, len_hi, len_lo, payload...]
                // This matches libjxl's JPEGData format (size = marker_len + 1)
                let mut raw = Vec::with_capacity(1 + len);
                raw.push(0xFE);
                raw.extend_from_slice(&data[pos..pos + len]);
                jpeg.com_data.push(raw);
                pos += len;
            }
            0xD0..=0xD7 => {
                // RST markers (no payload)
            }
            _ => {
                // Unknown marker with length prefix
                if pos + 1 < data.len() {
                    let len = u16::from_be_bytes([data[pos], data[pos + 1]]) as usize;
                    if pos + len <= data.len() {
                        pos += len;
                    }
                }
            }
        }
    }

    jpeg.component_type = classify_components(&jpeg, seen_jfif, adobe_transform);
    Ok(jpeg)
}

/// Parse SOF marker (without the marker bytes, starting at length field).
fn parse_sof_marker(data: &[u8], pos: &mut usize, jpeg: &mut JpegData) -> Result<()> {
    let p = *pos;
    if p + 1 >= data.len() {
        return Err(JpegError::UnexpectedEof);
    }
    let len = u16::from_be_bytes([data[p], data[p + 1]]) as usize;
    if p + len > data.len() || len < 8 {
        return Err(JpegError::Invalid("SOF marker too short".into()));
    }

    let precision = data[p + 2];
    if precision != 8 {
        return Err(JpegError::Unsupported(format!("{precision}-bit precision")));
    }
    jpeg.height = u16::from_be_bytes([data[p + 3], data[p + 4]]) as u32;
    jpeg.width = u16::from_be_bytes([data[p + 5], data[p + 6]]) as u32;
    let num_comp = data[p + 7] as usize;

    if num_comp != 1 && num_comp != 3 {
        return Err(JpegError::Unsupported(format!("{num_comp} components")));
    }

    let mut max_h = 1u32;
    let mut max_v = 1u32;
    let mut raw = Vec::new();

    for i in 0..num_comp {
        let base = p + 8 + i * 3;
        if base + 2 >= data.len() {
            return Err(JpegError::UnexpectedEof);
        }
        let id = data[base] as u32;
        let hv = data[base + 1];
        let h = (hv >> 4) as u32;
        let v = (hv & 0x0F) as u32;
        let q = data[base + 2] as u32;
        max_h = max_h.max(h);
        max_v = max_v.max(v);
        raw.push((id, h, v, q));
    }

    let mcu_w = jpeg.width.div_ceil(max_h * 8);
    let mcu_h = jpeg.height.div_ceil(max_v * 8);

    for (id, h, v, q) in raw {
        let wb = mcu_w * h;
        let hb = mcu_h * v;
        jpeg.components.push(JpegComponent {
            id,
            h_samp_factor: h,
            v_samp_factor: v,
            quant_idx: q,
            width_in_blocks: wb,
            height_in_blocks: hb,
            coeffs: Vec::new(), // Filled by zenjpeg
        });
    }

    *pos = p + len;
    Ok(())
}

/// Parse DQT marker.
fn parse_dqt_marker(data: &[u8], pos: &mut usize, jpeg: &mut JpegData) -> Result<()> {
    let p = *pos;
    if p + 1 >= data.len() {
        return Err(JpegError::UnexpectedEof);
    }
    let len = u16::from_be_bytes([data[p], data[p + 1]]) as usize;
    if p + len > data.len() {
        return Err(JpegError::Invalid("DQT truncated".into()));
    }

    let end = p + len;
    let mut cur = p + 2;
    let mut first = true;

    while cur < end {
        let pq_tq = data[cur];
        cur += 1;
        let precision = (pq_tq >> 4) as u32;
        let index = (pq_tq & 0x0F) as u32;

        let mut values = [0i32; 64];
        for &natural_idx in &JPEG_NATURAL_ORDER {
            if precision == 0 {
                if cur >= end {
                    return Err(JpegError::UnexpectedEof);
                }
                values[natural_idx] = data[cur] as i32;
                cur += 1;
            } else if cur + 1 >= end {
                return Err(JpegError::UnexpectedEof);
            } else {
                values[natural_idx] = u16::from_be_bytes([data[cur], data[cur + 1]]) as i32;
                cur += 2;
            }
        }

        if !first && let Some(last) = jpeg.quant.last_mut() {
            last.is_last = false;
        }

        jpeg.quant.push(JpegQuantTable {
            values,
            precision,
            index,
            is_last: true,
        });
        first = false;
    }

    *pos = end;
    Ok(())
}

/// Parse DHT marker.
fn parse_dht_marker(data: &[u8], pos: &mut usize, jpeg: &mut JpegData) -> Result<()> {
    let p = *pos;
    if p + 1 >= data.len() {
        return Err(JpegError::UnexpectedEof);
    }
    let len = u16::from_be_bytes([data[p], data[p + 1]]) as usize;
    if p + len > data.len() {
        return Err(JpegError::Invalid("DHT truncated".into()));
    }

    let end = p + len;
    let mut cur = p + 2;
    let mut first = true;

    while cur < end {
        let tc_th = data[cur];
        cur += 1;
        let is_ac = (tc_th >> 4) != 0;
        let id = (tc_th & 0x0F) as u32;

        let mut counts = [0u32; 16];
        let mut total = 0u32;
        for count in &mut counts {
            if cur >= end {
                return Err(JpegError::UnexpectedEof);
            }
            *count = data[cur] as u32;
            total += *count;
            cur += 1;
        }

        if cur + total as usize > end {
            return Err(JpegError::UnexpectedEof);
        }
        let values = data[cur..cur + total as usize].to_vec();
        cur += total as usize;

        if !first && let Some(last) = jpeg.huffman_code.last_mut() {
            last.is_last = false;
        }

        jpeg.huffman_code.push(JpegHuffmanCode {
            is_ac,
            id,
            is_last: true,
            counts,
            values,
        });
        first = false;
    }

    *pos = end;
    Ok(())
}

/// Parse SOS header (without decoding entropy data).
fn parse_sos_header(data: &[u8], pos: &mut usize, jpeg: &mut JpegData) -> Result<()> {
    let p = *pos;
    if p + 1 >= data.len() {
        return Err(JpegError::UnexpectedEof);
    }
    let len = u16::from_be_bytes([data[p], data[p + 1]]) as usize;
    if p + len > data.len() {
        return Err(JpegError::Invalid("SOS truncated".into()));
    }

    // The SOS marker payload (after the 2-byte length) must contain:
    //   1 byte  Ns
    //   2*Ns    component selectors
    //   3 bytes Ss, Se, Ah|Al
    // Validate `len` against this layout BEFORE indexing into `data` so a
    // crafted truncated SOS marker (e.g. `len = 0`) does not panic via
    // out-of-bounds indexing. Security audit 2026-05-06 H4.
    if len < 3 {
        return Err(JpegError::Invalid("SOS marker too short for Ns".into()));
    }
    let ns = data[p + 2] as u32;
    let required = 3usize
        .checked_add((ns as usize).saturating_mul(2))
        .and_then(|n| n.checked_add(3))
        .ok_or_else(|| JpegError::Invalid("SOS Ns overflow".into()))?;
    if len < required {
        return Err(JpegError::Invalid(format!(
            "SOS marker length {len} too short for Ns={ns} (need {required})"
        )));
    }

    let mut comp_indices = Vec::with_capacity(ns as usize);
    let mut dc_idx = Vec::with_capacity(ns as usize);
    let mut ac_idx = Vec::with_capacity(ns as usize);

    for i in 0..ns as usize {
        let base = p + 3 + i * 2;
        let cs = data[base];
        let td_ta = data[base + 1];

        let ci = jpeg
            .components
            .iter()
            .position(|c| c.id == cs as u32)
            .ok_or_else(|| JpegError::Invalid(format!("unknown component {cs}")))?;

        comp_indices.push(ci as u32);
        dc_idx.push((td_ta >> 4) as u32);
        ac_idx.push((td_ta & 0x0F) as u32);
    }

    let spec_base = p + 3 + ns as usize * 2;
    let ss = data[spec_base] as u32;
    let se = data[spec_base + 1] as u32;
    let ahal = data[spec_base + 2];

    jpeg.scan_info.push(JpegScanInfo {
        num_components: ns,
        component_indices: comp_indices,
        dc_tbl_idx: dc_idx,
        ac_tbl_idx: ac_idx,
        ss,
        se,
        ah: (ahal >> 4) as u32,
        al: (ahal & 0x0F) as u32,
        reset_points: Vec::new(),
        extra_zero_runs: Vec::new(),
    });

    *pos = p + len;
    Ok(())
}

/// Skip entropy-coded data segment (find next real marker).
fn skip_entropy_data(data: &[u8], pos: &mut usize) {
    while *pos < data.len() {
        if data[*pos] == 0xFF {
            if *pos + 1 >= data.len() {
                *pos = data.len();
                return;
            }
            let next = data[*pos + 1];
            if next == 0x00 {
                // Byte stuffing
                *pos += 2;
                continue;
            }
            if (0xD0..=0xD7).contains(&next) {
                // RST marker
                *pos += 2;
                continue;
            }
            // Real marker - stop before it (the main loop will read it)
            return;
        }
        *pos += 1;
    }
}

/// Extract DCT coefficients using zenjpeg's decoder.
fn extract_coefficients_zenjpeg(data: &[u8], jpeg: &mut JpegData) -> Result<()> {
    use zenjpeg::decoder::DecodeConfig;
    use zenjpeg::encoder::Unstoppable;

    let config = DecodeConfig::new();
    let decoded = config
        .decode_coefficients(data, Unstoppable)
        .map_err(|e| JpegError::Decode(format!("{e}")))?;

    if decoded.components.len() != jpeg.components.len() {
        return Err(JpegError::Invalid(format!(
            "component count mismatch: marker scanner found {}, zenjpeg found {}",
            jpeg.components.len(),
            decoded.components.len()
        )));
    }

    // Copy coefficients from zenjpeg's zigzag format to our natural-order format
    for (i, zen_comp) in decoded.components.iter().enumerate() {
        let comp = &mut jpeg.components[i];
        let num_blocks = (comp.width_in_blocks * comp.height_in_blocks) as usize;
        comp.coeffs = vec![0i16; num_blocks * 64];

        for blk in 0..num_blocks {
            let src_base = blk * 64;
            let dst_base = blk * 64;

            // Convert from zigzag to natural order
            for (zigzag_idx, &natural_idx) in JPEG_NATURAL_ORDER.iter().enumerate() {
                comp.coeffs[dst_base + natural_idx] = zen_comp.coeffs[src_base + zigzag_idx];
            }
        }
    }

    Ok(())
}

/// Classify APP marker type.
fn classify_app_marker(marker_byte: u8, payload: &[u8]) -> AppMarkerType {
    match marker_byte {
        0xE1 => {
            if payload.starts_with(b"Exif\0\0") {
                AppMarkerType::Exif
            } else if payload.starts_with(b"http://ns.adobe.com/xap/1.0/\0") {
                AppMarkerType::Xmp
            } else {
                AppMarkerType::Unknown
            }
        }
        0xE2 => {
            if payload.starts_with(b"ICC_PROFILE\0") {
                AppMarkerType::Icc
            } else {
                AppMarkerType::Unknown
            }
        }
        _ => AppMarkerType::Unknown,
    }
}

/// Classify components from IDs and markers.
fn classify_components(
    jpeg: &JpegData,
    _seen_jfif: bool,
    _adobe_transform: Option<u8>,
) -> JpegComponentType {
    // Mirror libjxl `JPEGData::VisitFields` (`lib/jxl/jpeg/jpeg_data.cc:158-167`)
    // exactly. The wire-format `component_type` (2 bits) is a TAG that the
    // decoder uses to RECONSTRUCT component IDs:
    //   - Gray   → id[0] = 1
    //   - YCbCr  → id[0..3] = (1, 2, 3)
    //   - RGB    → id[0..3] = ('R', 'G', 'B') = (82, 71, 66)
    //   - Custom → IDs are stored explicitly in the JBRD payload
    // Picking a non-Custom tag when the source IDs don't match the tag's
    // implied IDs corrupts roundtrip (decoder writes the implied IDs back to
    // the SOF marker). libjxl IGNORES the JFIF / Adobe-APP14 markers when
    // tagging; only literal ID equality matters. Adobe-transform=0 ("this is
    // RGB") is still useful as a SEMANTIC color-space hint elsewhere (e.g.
    // `do_ycbcr` / `color_transform` selection in the frame header), but it
    // does NOT change the JBRD component_type tag.
    let ids: Vec<u32> = jpeg.components.iter().map(|c| c.id).collect();
    if ids.len() == 1 && ids[0] == 1 {
        return JpegComponentType::Gray;
    }
    if ids.len() == 3 && ids == [1, 2, 3] {
        return JpegComponentType::YCbCr;
    }
    if ids.len() == 3 && ids == [b'R' as u32, b'G' as u32, b'B' as u32] {
        return JpegComponentType::Rgb;
    }
    JpegComponentType::Custom
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zigzag_roundtrip() {
        for (i, &natural) in JPEG_NATURAL_ORDER.iter().enumerate() {
            let back = JPEG_ZIGZAG_ORDER[natural];
            assert_eq!(i, back, "zigzag roundtrip failed at {i}");
        }
    }

    /// Security audit 2026-05-06 H4 regression: a truncated SOS marker with
    /// `len < 6 + 2*Ns` previously panicked with index-out-of-bounds when
    /// `parse_sos_header` indexed `data[spec_base..spec_base + 3]` past the
    /// declared marker length. Reachable from the public `read_jpeg` API.
    #[test]
    fn read_jpeg_rejects_truncated_sos_marker_with_zero_length() {
        // SOI + minimal SOF0 (so component table has entries) + truncated SOS
        // marker with len = 0 declared after the FFDA marker bytes.
        let mut data = Vec::new();
        // SOI
        data.extend_from_slice(&[0xFF, 0xD8]);
        // SOF0: marker FFC0, len 17, P=8, Y=8, X=8, Nf=3, then 3*3 component bytes
        data.extend_from_slice(&[0xFF, 0xC0]);
        data.extend_from_slice(&17u16.to_be_bytes());
        data.push(8); // precision
        data.extend_from_slice(&8u16.to_be_bytes()); // height
        data.extend_from_slice(&8u16.to_be_bytes()); // width
        data.push(3); // Nf
        // component 1
        data.extend_from_slice(&[1, 0x22, 0]);
        // component 2
        data.extend_from_slice(&[2, 0x11, 1]);
        // component 3
        data.extend_from_slice(&[3, 0x11, 1]);
        // SOS marker with declared len = 0 (truncated). Without the bounds
        // check this panicked while reading data[spec_base..spec_base+3].
        data.extend_from_slice(&[0xFF, 0xDA]);
        data.extend_from_slice(&0u16.to_be_bytes());

        let result = read_jpeg(&data);
        assert!(matches!(result, Err(JpegError::Invalid(_))));
    }

    /// Security audit 2026-05-06 H4 regression: SOS marker with a body
    /// covering Ns and component selectors but missing the trailing 3-byte
    /// (Ss, Se, Ah/Al) tuple. `len = 3 + 2*Ns` (no Ss/Se/Ah/Al). Should error,
    /// not panic.
    #[test]
    fn read_jpeg_rejects_sos_marker_missing_spec_bytes() {
        let mut data = Vec::new();
        data.extend_from_slice(&[0xFF, 0xD8]); // SOI
        // SOF0
        data.extend_from_slice(&[0xFF, 0xC0]);
        data.extend_from_slice(&17u16.to_be_bytes());
        data.push(8);
        data.extend_from_slice(&8u16.to_be_bytes());
        data.extend_from_slice(&8u16.to_be_bytes());
        data.push(3);
        data.extend_from_slice(&[1, 0x22, 0]);
        data.extend_from_slice(&[2, 0x11, 1]);
        data.extend_from_slice(&[3, 0x11, 1]);
        // SOS: Ns = 1, then 2 bytes selector, but len declared as 5 (covers
        // length-bytes + Ns + 2 selector bytes only — no trailing Ss/Se/Ah).
        // Required: 3 + 2*Ns + 3 = 8. Declared: 5.
        data.extend_from_slice(&[0xFF, 0xDA]);
        data.extend_from_slice(&5u16.to_be_bytes());
        data.push(1); // Ns
        data.extend_from_slice(&[1, 0x00]); // component selector + table

        let result = read_jpeg(&data);
        assert!(matches!(result, Err(JpegError::Invalid(_))));
    }

    #[test]
    #[cfg_attr(
        not(feature = "corpus-tests"),
        ignore = "requires codec corpus; enable `corpus-tests` feature"
    )]
    fn test_parse_real_jpeg() {
        crate::skip_without_corpus!();
        let path = format!(
            "{}/imageflow/test_inputs/orientation/Landscape_1.jpg",
            crate::test_helpers::corpus_dir().display()
        );
        let data = std::fs::read(path).expect("failed to read test JPEG");
        let jpeg = read_jpeg(&data).expect("failed to parse JPEG");

        assert!(jpeg.width > 0, "width should be nonzero");
        assert!(jpeg.height > 0, "height should be nonzero");
        assert!(!jpeg.components.is_empty(), "should have components");
        assert!(!jpeg.quant.is_empty(), "should have quant tables");
        assert!(!jpeg.huffman_code.is_empty(), "should have huffman tables");
        assert!(!jpeg.scan_info.is_empty(), "should have scan info");

        // Verify coefficients were extracted
        for (i, comp) in jpeg.components.iter().enumerate() {
            let expected_coeffs = (comp.width_in_blocks * comp.height_in_blocks * 64) as usize;
            assert_eq!(
                comp.coeffs.len(),
                expected_coeffs,
                "component {i} should have {expected_coeffs} coefficients, got {}",
                comp.coeffs.len()
            );
            // DC coefficients should not all be zero
            let has_nonzero_dc = (0..comp.width_in_blocks * comp.height_in_blocks)
                .any(|b| comp.coeffs[b as usize * 64] != 0);
            assert!(
                has_nonzero_dc,
                "component {i} should have nonzero DC values"
            );
        }

        println!(
            "Parsed: {}x{}, {} components, {} quant tables, {} huffman tables, {} scans",
            jpeg.width,
            jpeg.height,
            jpeg.components.len(),
            jpeg.quant.len(),
            jpeg.huffman_code.len(),
            jpeg.scan_info.len(),
        );
        println!("Component type: {:?}", jpeg.component_type);
        println!("Marker order: {:?}", jpeg.marker_order);
        for (i, comp) in jpeg.components.iter().enumerate() {
            println!(
                "  Component {i}: id={}, {}x{} samp, quant_idx={}, {}x{} blocks, {} coeffs",
                comp.id,
                comp.h_samp_factor,
                comp.v_samp_factor,
                comp.quant_idx,
                comp.width_in_blocks,
                comp.height_in_blocks,
                comp.coeffs.len()
            );
        }
    }
}
