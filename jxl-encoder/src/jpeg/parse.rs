// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! JPEG parsing for lossless reencoding.
//!
//! Uses zenjpeg for reliable coefficient extraction, plus a lightweight marker
//! scanner for the jbrd reconstruction metadata.

use super::data::*;
use crate::budget::MemoryBudget;
use alloc::sync::Arc;
use enough::Stop;

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
    /// The caller's [`enough::Stop`] token requested cancellation. Surfaced
    /// as [`crate::api::EncodeError::Cancelled`] at the transcode boundary.
    #[error("encoding cancelled")]
    Cancelled,
    /// A dimension-driven coefficient-buffer allocation exceeded the
    /// configured [`MemoryBudget`] cap. Surfaced as
    /// [`crate::api::EncodeError::LimitExceeded`] at the transcode boundary.
    #[error("memory budget exceeded: {0}")]
    ResourceLimit(String),
}

type Result<T> = core::result::Result<T, JpegError>;

/// Parse a JPEG file and extract all data needed for lossless reencoding.
///
/// This performs two passes:
/// 1. Marker scan: extracts marker structure, Huffman/quant tables, APP/COM data
/// 2. zenjpeg decode: extracts quantized DCT coefficients reliably
///
/// `max_pixels` is the pre-flight cap applied to the untrusted SOF
/// `width × height` *before* any per-block coefficient buffer is allocated
/// (the transcode path builds no `MemoryBudget`). `None`
/// applies the secure default
/// [`crate::api::Limits::DEFAULT_MAX_JPEG_TRANSCODE_PIXELS`] (120 MP);
/// callers thread their own cap from [`crate::api::Limits::max_pixels`]
/// (e.g. via [`crate::api::LosslessConfig::with_limits`]). Pass
/// `Some(u64::MAX)` to opt out of the cap entirely (trusted input only).
pub fn read_jpeg(data: &[u8], max_pixels: Option<u64>) -> Result<JpegData> {
    read_jpeg_with_stop(data, max_pixels, None, None)
}

/// Like [`read_jpeg`], but polls `stop` and reserves the coefficient buffers
/// against `budget` so a server can abort or bound an oversized untrusted
/// transcode mid-decode.
///
/// `stop` is checked at entry and threaded into the zenjpeg coefficient decode
/// (the dominant time sink for a large frame). `budget` (when `Some`) caps the
/// per-component `i16` coefficient-buffer allocations — the largest
/// attacker-controlled buffers on the decode side — returning
/// [`JpegError::ResourceLimit`] if a reservation exceeds the cap. Both `None`
/// are equivalent to an `Unstoppable` / unbounded run — **byte-identical with
/// zero overhead**. Returns [`JpegError::Cancelled`] on cancellation. Used by
/// the cancellable / budgeted transcode entry points
/// ([`crate::api::LosslessConfig::encode_jpeg_transcode_with_stop`]).
pub(crate) fn read_jpeg_with_stop(
    data: &[u8],
    max_pixels: Option<u64>,
    stop: Option<&dyn Stop>,
    budget: Option<&Arc<MemoryBudget>>,
) -> Result<JpegData> {
    if let Some(s) = stop {
        s.check().map_err(|_| JpegError::Cancelled)?;
    }
    let max_pixels = max_pixels.unwrap_or(crate::api::Limits::DEFAULT_MAX_JPEG_TRANSCODE_PIXELS);

    // Phase 1: Scan markers for jbrd metadata
    let mut jpeg = scan_markers(data, max_pixels)?;

    // Phase 2: Use zenjpeg for reliable coefficient extraction
    extract_coefficients_zenjpeg(data, &mut jpeg, stop, budget)?;

    Ok(jpeg)
}

/// Lightweight marker scanner that reads JPEG marker structure without
/// decoding entropy data. Extracts everything needed for jbrd box serialization.
fn scan_markers(data: &[u8], max_pixels: u64) -> Result<JpegData> {
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
                parse_sof_marker(data, &mut pos, &mut jpeg, max_pixels)?;
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
                parse_sof_marker(data, &mut pos, &mut jpeg, max_pixels)?;
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
///
/// `max_pixels` is the pre-flight cap on `width × height` (see [`read_jpeg`]);
/// `u64::MAX` disables it.
fn parse_sof_marker(
    data: &[u8],
    pos: &mut usize,
    jpeg: &mut JpegData,
    max_pixels: u64,
) -> Result<()> {
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

    // Pre-flight pixel cap on the untrusted SOF dimensions, BEFORE the
    // per-block coefficient buffers are allocated (the transcode path does not
    // build a `MemoryBudget`). JPEG dims are `u16` (≤ 65535 each), so a crafted
    // header can advertise ~4.3 Gpx and force a multi-gigabyte `i16`
    // coefficient allocation downstream. `max_pixels` is the caller's cap
    // (`Limits::max_pixels`, default 120 MP — admits 108 MP phone photos);
    // `u64::MAX` opts out.
    if (jpeg.width as u64) * (jpeg.height as u64) > max_pixels {
        return Err(JpegError::Invalid(format!(
            "JPEG frame {}x{} exceeds the {} MP transcode pixel cap",
            jpeg.width,
            jpeg.height,
            max_pixels / 1_000_000
        )));
    }

    let num_comp = data[p + 7] as usize;

    // Accept 1-4 components (libjxl `kMaxComponents = 4`,
    // `enc_jpeg_data_reader.cc:83`). The JBRD on-wire `num_components`
    // field encodes `Val(1), Val(2), Val(3), Val(4)`, but libjxl rejects
    // any value other than 1 or 3 at `jpeg_data.cc:180-182`. The check
    // is therefore moved to `encode_jpeg_to_jxl_inner` so callers that
    // only need to PARSE a CMYK / 4-component JPEG (e.g. to inspect its
    // metadata) succeed; callers that try to TRANSCODE one fail with a
    // clearer spec-level error message.
    if !(1..=4).contains(&num_comp) {
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

/// Block count and coefficient-buffer length for a JPEG component, computed
/// with checked `usize` arithmetic.
///
/// SOF `*_in_blocks` fields are attacker-controlled (each up to ~983025 for a
/// 65535-px dimension with 1×1 sampling), so `width_in_blocks *
/// height_in_blocks` overflows `u32`, and the `* 64` coefficient scale
/// overflows `usize` on 32-bit targets (i686 / wasm32 — #77 item 3). A wrapped
/// length would size the component coefficient buffer short and make the
/// zigzag copy in [`extract_coefficients_zenjpeg`] read out of bounds, so a
/// malformed SOF must error rather than wrap. Returns `None` on overflow.
fn checked_coeff_len(width_in_blocks: usize, height_in_blocks: usize) -> Option<(usize, usize)> {
    let num_blocks = width_in_blocks.checked_mul(height_in_blocks)?;
    let coeff_len = num_blocks.checked_mul(64)?;
    Some((num_blocks, coeff_len))
}

/// Extract DCT coefficients using zenjpeg's decoder.
///
/// As of zenjpeg 0.8.5 we use [`decode_coefficients_with_jbrd_metadata`]
/// so we get the per-scan JBRD signals (`reset_points` and
/// `extra_zero_runs`) for free in the same single decode pass. These are
/// merged into the marker-scanner-built `scan_info` entries by matching
/// on `(ss, se, ah, al)` in bitstream order — exactly the order both the
/// marker scanner and zenjpeg traverse SOS markers.
///
/// [`decode_coefficients_with_jbrd_metadata`]: zenjpeg::decoder::DecodeConfig::decode_coefficients_with_jbrd_metadata
fn extract_coefficients_zenjpeg(
    data: &[u8],
    jpeg: &mut JpegData,
    stop: Option<&dyn Stop>,
    budget: Option<&Arc<MemoryBudget>>,
) -> Result<()> {
    use enough::Unstoppable;
    use zenjpeg::decoder::DecodeConfig;

    let config = DecodeConfig::new();
    // Thread the caller's Stop into zenjpeg's entropy decode (the dominant
    // time sink for a large untrusted frame). `None` → `Unstoppable`, which
    // zenjpeg never polls — byte-identical to the pre-cancellation path.
    let unstoppable = Unstoppable;
    let stop_ref: &dyn Stop = stop.unwrap_or(&unstoppable);
    let decode_result = config.decode_coefficients_with_jbrd_metadata(data, stop_ref);
    // If the stop fired, surface a clean `Cancelled` instead of zenjpeg's
    // generic decode error (zenjpeg maps `StopReason` into its own error).
    if let Some(s) = stop {
        s.check().map_err(|_| JpegError::Cancelled)?;
    }
    let (decoded, jbrd_meta) = decode_result.map_err(|e| JpegError::Decode(format!("{e}")))?;

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
        // SOF dims are attacker-controlled (each `*_in_blocks` up to ~983025),
        // so `wb * hb` overflows `u32` and `* 64` overflows `usize` on 32-bit
        // (#77 item 3). Compute block / coefficient counts with checked `usize`
        // arithmetic; a malformed SOF errors instead of wrapping (a wrapped
        // count would size `comp.coeffs` short and make the zigzag copy below
        // read out of bounds).
        let (num_blocks, coeff_len) = checked_coeff_len(
            comp.width_in_blocks as usize,
            comp.height_in_blocks as usize,
        )
        .ok_or_else(|| {
            JpegError::Invalid(format!(
                "component {i}: coefficient count overflow ({} × {} blocks)",
                comp.width_in_blocks, comp.height_in_blocks
            ))
        })?;
        // Reconcile the SOF-derived count against what zenjpeg actually decoded
        // — the zigzag copy indexes `zen_comp.coeffs` by block, so a shorter
        // zenjpeg buffer (inconsistent marker scan vs decode) would read OOB.
        if zen_comp.coeffs.len() < coeff_len {
            return Err(JpegError::Invalid(format!(
                "component {i}: zenjpeg decoded {} coefficients, SOF implies {coeff_len}",
                zen_comp.coeffs.len()
            )));
        }
        // Reserve + allocate the per-component i16 coefficient buffer — the
        // largest attacker-controlled allocation on the decode side (#77
        // item 1). The helper reserves `coeff_len * 2` bytes against the budget
        // (with checked byte arithmetic) BEFORE allocating, and honors the
        // budget's runtime fallible-alloc policy (`vec!` fast / `try_reserve`
        // safe — `Limits::fallible_alloc`). `None` budget is a no-op.
        comp.coeffs = crate::budget::try_alloc_zeroed_permanent::<i16>(budget, coeff_len)
            .map_err(|e| JpegError::ResourceLimit(format!("{e}")))?;

        for blk in 0..num_blocks {
            let src_base = blk * 64;
            let dst_base = blk * 64;

            // Convert from zigzag to natural order
            for (zigzag_idx, &natural_idx) in JPEG_NATURAL_ORDER.iter().enumerate() {
                comp.coeffs[dst_base + natural_idx] = zen_comp.coeffs[src_base + zigzag_idx];
            }
        }
    }

    // Merge zenjpeg's JBRD per-scan signals into our marker-scanner-built
    // scan_info. zenjpeg only emits a tracked entry for scans it actually
    // decodes (progressive scans + baseline scan). The marker scanner emits
    // ONE entry per SOS marker in the same bitstream order, so the indexes
    // align 1:1 for valid JPEGs.
    //
    // For baseline JPEGs zenjpeg currently routes the entropy decode through
    // a path that does NOT call `decode_progressive_scan` and therefore does
    // not push a JbrdScanInfo. That's fine — baseline JPEGs never emit EOB
    // runs > 1 or ZRL-chains preceding EOB by construction (no compliant
    // encoder ever does), so `reset_points` and `extra_zero_runs` are
    // intrinsically empty there. The marker-scanner's default empty Vecs
    // are correct as-is.
    //
    // For progressive JPEGs (the case the 2 known DIFF unsplash cases
    // exercise), zenjpeg pushes one JbrdScanInfo per SOS in order. We
    // pair by index; if for any reason counts disagree (e.g. truncated
    // last scan), we simply skip the merge for the unmatched tail —
    // the empty default is still safe.
    for (scan, jbrd_scan) in jpeg.scan_info.iter_mut().zip(jbrd_meta.scans.iter()) {
        // Sanity check: the spectral / approximation parameters must match.
        // If they don't, the bitstream ordering between marker scanner and
        // zenjpeg has diverged — bail with a descriptive error rather than
        // silently emitting wrong JBRD data.
        if scan.ss != jbrd_scan.ss as u32
            || scan.se != jbrd_scan.se as u32
            || scan.ah != jbrd_scan.ah as u32
            || scan.al != jbrd_scan.al as u32
        {
            return Err(JpegError::Invalid(format!(
                "JBRD scan-parameter mismatch: marker scanner saw \
                 ss={}/se={}/ah={}/al={} but zenjpeg saw \
                 ss={}/se={}/ah={}/al={}",
                scan.ss,
                scan.se,
                scan.ah,
                scan.al,
                jbrd_scan.ss,
                jbrd_scan.se,
                jbrd_scan.ah,
                jbrd_scan.al,
            )));
        }
        scan.reset_points = jbrd_scan.reset_points.clone();
        scan.extra_zero_runs = jbrd_scan.extra_zero_runs.clone();
    }

    // Copy entropy-segment padding-bit signals (zenjpeg 0.8.6+,
    // `JbrdMetadata::has_zero_padding_bit` + `padding_bits`). Some JPEG
    // encoders pad the final partial byte of each entropy segment with
    // 0-bits instead of the spec-recommended 1-bits; byte-exact JPEG-XL
    // transcoding (`djxl --reconstruct_jpeg`) needs the explicit bits to
    // reproduce those bytes. `has_zero_padding_bit == false` means the
    // standard 1-bit padding is used and djxl's default emitter produces
    // the right bytes; `padding_bits` carries the explicit bit sequence
    // only when `has_zero_padding_bit == true`. See libjxl
    // `enc_jpeg_data_reader.cc:441-470` `FinishStream` for the source-side
    // semantics that this mirrors.
    jpeg.has_zero_padding_bit = jbrd_meta.has_zero_padding_bit;
    jpeg.padding_bits = jbrd_meta.padding_bits.clone();

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

    /// Build a minimal SOI + SOF0 (3 components) advertising `w`×`h`. The frame
    /// has no SOS/coefficients, so parsing fails AFTER the SOF either at the
    /// pixel cap (oversized) or later (missing scan data).
    fn sof_only(w: u16, h: u16) -> Vec<u8> {
        let mut d = vec![0xFF, 0xD8, 0xFF, 0xC0];
        d.extend_from_slice(&17u16.to_be_bytes()); // len
        d.push(8); // precision
        d.extend_from_slice(&h.to_be_bytes());
        d.extend_from_slice(&w.to_be_bytes());
        d.push(3); // Nf
        d.extend_from_slice(&[1, 0x11, 0]);
        d.extend_from_slice(&[2, 0x11, 1]);
        d.extend_from_slice(&[3, 0x11, 1]);
        d
    }

    /// proderrdoc 2026-06-14 (issue #77): the JPEG-transcode path builds no
    /// `MemoryBudget`, and JPEG dims are `u16` (≤ 65535 each), so a crafted SOF
    /// can advertise ~4.3 Gpx and force a multi-GB coefficient allocation.
    /// `parse_sof_marker` rejects oversized frames before any allocation;
    /// `None` applies the default 120 MP cap.
    #[test]
    fn read_jpeg_rejects_oversized_sof_before_alloc() {
        // 20000 × 20000 = 400 MP, well over the default 120 MP cap.
        let result = read_jpeg(&sof_only(20000, 20000), None);
        assert!(
            matches!(&result, Err(JpegError::Invalid(m)) if m.contains("120 MP")),
            "expected the 120 MP cap rejection, got {result:?}"
        );
    }

    /// Regression guard: a normal-sized frame (16.8 MP) passes the default cap
    /// — it fails later for missing scan data, never with the cap message.
    #[test]
    fn read_jpeg_does_not_cap_normal_dimensions() {
        let result = read_jpeg(&sof_only(4096, 4096), None); // 16.8 MP
        if let Err(JpegError::Invalid(m)) = &result {
            assert!(
                !m.contains("MP transcode pixel cap"),
                "16.8 MP wrongly hit the cap: {m}"
            );
        }
    }

    /// The pixel cap is **configurable** (issue #77): a caller-supplied
    /// `max_pixels` overrides the 120 MP default in both directions, and
    /// `Some(u64::MAX)` opts out entirely.
    #[test]
    fn read_jpeg_pixel_cap_is_configurable() {
        // (a) A tighter cap rejects a frame the default would admit: 4096×4096
        //     (16.8 MP) under a 10 MP cap fails AT the cap, before allocation.
        let tight = read_jpeg(&sof_only(4096, 4096), Some(10_000_000));
        assert!(
            matches!(&tight, Err(JpegError::Invalid(m)) if m.contains("10 MP transcode pixel cap")),
            "expected the 10 MP cap rejection, got {tight:?}"
        );

        // (b) A looser cap admits a frame the default would reject: 20000×20000
        //     (400 MP) under a 500 MP cap passes the cap (then fails later for
        //     missing scan data — never with the cap message).
        let loose = read_jpeg(&sof_only(20000, 20000), Some(500_000_000));
        if let Err(JpegError::Invalid(m)) = &loose {
            assert!(
                !m.contains("transcode pixel cap"),
                "400 MP wrongly hit the 500 MP cap: {m}"
            );
        }

        // (c) `u64::MAX` opts out: even the largest representable JPEG frame
        //     (65535×65535 ≈ 4.29 Gpx) clears the cap.
        let disabled = read_jpeg(&sof_only(65535, 65535), Some(u64::MAX));
        if let Err(JpegError::Invalid(m)) = &disabled {
            assert!(
                !m.contains("transcode pixel cap"),
                "u64::MAX should disable the cap, got: {m}"
            );
        }
    }

    /// #77 item 3: the per-component coefficient-buffer length is computed with
    /// checked `usize` arithmetic so an attacker-controlled SOF can't wrap it to
    /// a short length (which would make the zigzag copy in
    /// `extract_coefficients_zenjpeg` read out of bounds). The wrap only bites
    /// 32-bit targets (`*_in_blocks * 64` overflows `usize` there), so the guard
    /// is verified host-independently by exercising `checked_coeff_len` directly.
    #[test]
    fn checked_coeff_len_guards_overflow() {
        // `wb * hb` overflows `usize` outright → None (never a wrapped small value).
        assert_eq!(checked_coeff_len(usize::MAX, 2), None);
        // `wb * hb` fits but the ×64 coefficient scale overflows → None.
        assert_eq!(checked_coeff_len(usize::MAX / 32, 1), None);
        // A normal component is computed exactly.
        assert_eq!(checked_coeff_len(128, 96), Some((12_288, 786_432)));

        // The largest representable JPEG (65535 px → 8192 blocks/axis) has
        // num_blocks = 8192² = 67_108_864 and coeff_len = ×64 = 2^32 — the exact
        // value that overflows a 32-bit `usize` (where the guard fires and
        // returns None), but is representable on 64-bit (where it is returned).
        #[cfg(target_pointer_width = "64")]
        assert_eq!(
            checked_coeff_len(8192, 8192),
            Some((67_108_864, 4_294_967_296))
        );
        #[cfg(target_pointer_width = "32")]
        assert_eq!(checked_coeff_len(8192, 8192), None);
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

        let result = read_jpeg(&data, None);
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

        let result = read_jpeg(&data, None);
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
        let jpeg = read_jpeg(&data, None).expect("failed to parse JPEG");

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
