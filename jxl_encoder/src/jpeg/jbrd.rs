// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! JBRD (JPEG Bitstream Reconstruction Data) serialization.
//!
//! The JBRD box contains all JPEG metadata needed for byte-exact JPEG
//! reconstruction from a JXL file: Huffman tables, quantization tables,
//! scan headers, APP markers, restart markers, inter-marker data, etc.
//!
//! Format: bit-packed header (via BitWriter) + Brotli-compressed data stream.
//! The data stream contains concatenated APP marker payloads (non-ICC/EXIF/XMP),
//! COM marker payloads, inter-marker data, and tail data.

use super::data::*;
use crate::bit_writer::BitWriter;
use crate::error::Result;

/// Encode the JBRD box content for a parsed JPEG.
///
/// Returns the raw bytes to be placed inside a `jbrd` ISOBMFF box.
/// The format is: bit-packed header + Brotli-compressed data.
pub fn encode_jbrd(jpeg: &JpegData) -> Result<Vec<u8>> {
    let mut writer = BitWriter::new();

    // is_gray
    let is_gray = jpeg.components.len() == 1;
    writer.write(1, is_gray as u64)?;

    // Marker sequence (6 bits each, offset from 0xC0)
    for &marker in &jpeg.marker_order {
        writer.write(6, (marker - 0xC0) as u64)?;
    }
    // marker_order should end with 0xD9 (EOI)

    let num_app_markers = jpeg.app_data.len();

    // APP marker types and lengths
    for i in 0..num_app_markers {
        let app_type = jpeg.app_marker_type[i] as u32;
        // U32(Val(0), Val(1), BitsOffset(1, 2), BitsOffset(2, 4))
        write_u32_jbrd(&mut writer, app_type, &[0, 1], &[(1, 2), (2, 4)])?;

        let len = jpeg.app_data[i].len() as u32 - 1;
        writer.write(16, len as u64)?;
    }

    // COM marker lengths
    for com in &jpeg.com_data {
        let len = com.len() as u32 - 1;
        writer.write(16, len as u64)?;
    }

    // Quantization tables
    let num_quant = jpeg.quant.len() as u32;
    // U32(Val(1), Val(2), Val(3), Val(4)) — stored as num_quant, default 2
    write_u32_jbrd(&mut writer, num_quant, &[1, 2, 3, 4], &[])?;
    for qt in &jpeg.quant {
        writer.write(1, qt.precision as u64)?;
        writer.write(2, qt.index as u64)?;
        writer.write(1, qt.is_last as u64)?;
    }

    // Component type
    let comp_type = jpeg.component_type as u32;
    writer.write(2, comp_type as u64)?;

    // Custom component IDs
    if jpeg.component_type == JpegComponentType::Custom {
        let num_comp = jpeg.components.len() as u32;
        // U32(Val(1), Val(2), Val(3), Val(4))
        write_u32_jbrd(&mut writer, num_comp, &[1, 2, 3, 4], &[])?;
        for comp in &jpeg.components {
            writer.write(8, comp.id as u64)?;
        }
    }

    // Component quant table indices
    for comp in &jpeg.components {
        writer.write(2, comp.quant_idx as u64)?;
    }

    // Huffman codes
    let num_huff = jpeg.huffman_code.len() as u32;
    // U32(Val(4), BitsOffset(3, 2), BitsOffset(4, 10), BitsOffset(6, 26))
    write_u32_jbrd(&mut writer, num_huff, &[4], &[(3, 2), (4, 10), (6, 26)])?;

    for hc in &jpeg.huffman_code {
        writer.write(1, hc.is_ac as u64)?;
        writer.write(2, hc.id as u64)?;
        writer.write(1, hc.is_last as u64)?;

        // libjxl adds a sentinel symbol (value=256) at the deepest bit length.
        // This fills the remaining Huffman code space with an invalid symbol.
        // Our parser stores counts[0..15] for bit lengths 1-16.
        // max_depth_idx = index of deepest non-zero count (0-based, maps to bit length idx+1)
        let max_depth_idx = hc.counts.iter().rposition(|&c| c > 0).unwrap_or(0);

        // 17 count values (bit lengths 0-16)
        // U32(Val(0), Val(1), BitsOffset(3, 2), Bits(8))
        write_u32_jbrd(&mut writer, 0, &[0, 1], &[(3, 2), (8, 0)])?; // counts[0] = 0
        for i in 0..16 {
            let mut count = hc.counts[i];
            if i == max_depth_idx {
                count += 1; // sentinel symbol at max depth
            }
            write_u32_jbrd(&mut writer, count, &[0, 1], &[(3, 2), (8, 0)])?;
        }

        // Symbol values (original values + sentinel value 256)
        let num_symbols: u32 = hc.counts.iter().sum::<u32>() + 1;
        for i in 0..num_symbols as usize {
            let val = if i < hc.values.len() {
                hc.values[i] as u32
            } else {
                256 // kJpegHuffmanAlphabetSize sentinel
            };
            // U32(Bits(2), BitsOffset(2, 4), BitsOffset(4, 8), BitsOffset(8, 1))
            write_u32_jbrd(&mut writer, val, &[], &[(2, 0), (2, 4), (4, 8), (8, 1)])?;
        }
    }

    // Scan info
    for scan in &jpeg.scan_info {
        // num_components: U32(Val(1), Val(2), Val(3), Val(4))
        write_u32_jbrd(&mut writer, scan.num_components, &[1, 2, 3, 4], &[])?;

        writer.write(6, scan.ss as u64)?;
        writer.write(6, scan.se as u64)?;
        writer.write(4, scan.al as u64)?;
        writer.write(4, scan.ah as u64)?;

        for i in 0..scan.num_components as usize {
            writer.write(2, scan.component_indices[i] as u64)?;
            writer.write(2, scan.ac_tbl_idx[i] as u64)?;
            writer.write(2, scan.dc_tbl_idx[i] as u64)?;
        }

        // last_needed_pass: U32(Val(0), Val(1), Val(2), BitsOffset(3, 3))
        // Default value in libjxl is kMaxNumPasses - 1 = 2
        // For baseline JPEG, this is always 0
        write_u32_jbrd(&mut writer, 0, &[0, 1, 2], &[(3, 3)])?;
    }

    // Restart interval (only if DRI marker present)
    let has_dri = jpeg.marker_order.contains(&0xDD);
    if has_dri {
        writer.write(16, jpeg.restart_interval as u64)?;
    }

    // Scan more info (reset points and extra zero runs per scan)
    for scan in &jpeg.scan_info {
        // Reset points
        let num_reset_points = scan.reset_points.len() as u32;
        // U32(Val(0), BitsOffset(2, 1), BitsOffset(4, 4), BitsOffset(16, 20))
        write_u32_jbrd(
            &mut writer,
            num_reset_points,
            &[0],
            &[(2, 1), (4, 4), (16, 20)],
        )?;

        let mut last_block_idx: i64 = -1;
        for &block_idx in &scan.reset_points {
            let diff = (block_idx as i64 - last_block_idx - 1) as u32;
            // U32(Val(0), BitsOffset(3, 1), BitsOffset(5, 9), BitsOffset(28, 41))
            write_u32_jbrd(&mut writer, diff, &[0], &[(3, 1), (5, 9), (28, 41)])?;
            last_block_idx = block_idx as i64;
        }

        // Extra zero runs
        let num_extra = scan.extra_zero_runs.len() as u32;
        write_u32_jbrd(&mut writer, num_extra, &[0], &[(2, 1), (4, 4), (16, 20)])?;

        let mut last_block_idx: i64 = -1;
        for &(block_idx, num_runs) in &scan.extra_zero_runs {
            // num_extra_zero_runs: U32(Val(1), BitsOffset(2, 2), BitsOffset(4, 5), BitsOffset(8, 20))
            write_u32_jbrd(&mut writer, num_runs, &[1], &[(2, 2), (4, 5), (8, 20)])?;

            let diff = (block_idx as i64 - last_block_idx - 1) as u32;
            write_u32_jbrd(&mut writer, diff, &[0], &[(3, 1), (5, 9), (28, 41)])?;
            last_block_idx = block_idx as i64;
        }
    }

    // Inter-marker data lengths
    let num_intermarkers = jpeg.marker_order.iter().filter(|&&m| m == 0xFF).count();
    for i in 0..num_intermarkers {
        let len = jpeg.inter_marker_data[i].len() as u32;
        writer.write(16, len as u64)?;
    }

    // Tail data length
    let tail_len = jpeg.tail_data.len() as u32;
    // U32(Val(0), BitsOffset(8, 1), BitsOffset(16, 257), BitsOffset(22, 65793))
    write_u32_jbrd(
        &mut writer,
        tail_len,
        &[0],
        &[(8, 1), (16, 257), (22, 65793)],
    )?;

    // Padding bits
    writer.write(1, jpeg.has_zero_padding_bit as u64)?;
    if jpeg.has_zero_padding_bit {
        let nbit = jpeg.padding_bits.len() as u32;
        writer.write(24, nbit as u64)?;
        for &bit in &jpeg.padding_bits {
            writer.write(1, bit as u64)?;
        }
    }

    // Zero-pad header to byte boundary
    writer.zero_pad_to_byte();
    let header_bytes = writer.finish();

    // Collect data to Brotli-compress
    let mut data_stream = Vec::new();
    // APP marker data (only type == Unknown, others go in container boxes)
    for i in 0..num_app_markers {
        if jpeg.app_marker_type[i] != AppMarkerType::Unknown {
            continue;
        }
        data_stream.extend_from_slice(&jpeg.app_data[i]);
    }
    // COM marker data
    for com in &jpeg.com_data {
        data_stream.extend_from_slice(com);
    }
    // Inter-marker data
    for imd in &jpeg.inter_marker_data {
        data_stream.extend_from_slice(imd);
    }
    // Tail data
    data_stream.extend_from_slice(&jpeg.tail_data);

    // Brotli-compress the data stream
    let compressed = brotli_compress(&data_stream)?;

    // Combine header + compressed data
    let mut result = Vec::with_capacity(header_bytes.len() + compressed.len());
    result.extend_from_slice(&header_bytes);
    result.extend_from_slice(&compressed);

    Ok(result)
}

/// Write a JXL U32 value.
///
/// `direct_values` are selectors that encode exact values with no extra bits.
/// `bits_offset` are `(num_bits, offset)` pairs for the remaining selectors.
/// The total number of selectors (direct + bits_offset) must be exactly 4.
fn write_u32_jbrd(
    writer: &mut BitWriter,
    value: u32,
    direct_values: &[u32],
    bits_offset: &[(usize, u32)],
) -> Result<()> {
    // Try direct value matches first
    for (selector, &dv) in direct_values.iter().enumerate() {
        if value == dv {
            writer.write(2, selector as u64)?;
            return Ok(());
        }
    }

    // Try bits+offset matches
    let base_selector = direct_values.len();
    for (i, &(bits, offset)) in bits_offset.iter().enumerate() {
        let selector = base_selector + i;
        let is_last = selector >= 3;
        if is_last || (value >= offset && (bits == 0 || (value - offset) < (1 << bits))) {
            writer.write(2, selector as u64)?;
            if bits > 0 {
                writer.write(bits, (value - offset) as u64)?;
            }
            return Ok(());
        }
    }

    unreachable!("No selector matched for value {value}");
}

/// Brotli-compress data. Returns compressed bytes.
fn brotli_compress(data: &[u8]) -> Result<Vec<u8>> {
    use std::io::Write;

    let mut compressed = Vec::new();
    {
        let mut encoder = brotli::CompressorWriter::new(&mut compressed, 4096, 11, 22);
        encoder.write_all(data)?;
        encoder.flush()?;
    }
    Ok(compressed)
}

/// Extract ICC profile from JPEG APP2 markers.
///
/// ICC profiles in JPEG are stored across one or more APP2 markers,
/// each with header: "ICC_PROFILE\0" + chunk_number(1) + total_chunks(1) + data.
/// The raw app_data format is: [marker_byte, len_hi, len_lo, payload...].
/// Returns the concatenated ICC profile bytes, or None.
pub(crate) fn extract_icc(jpeg: &JpegData) -> Option<Vec<u8>> {
    const ICC_TAG: &[u8] = b"ICC_PROFILE\0";
    // Overhead per APP2 marker: marker_byte(1) + length(2) + tag(12) + chunk_num(1) + total_chunks(1) = 17
    const OVERHEAD: usize = 17;

    let mut icc_data = Vec::new();
    for i in 0..jpeg.app_data.len() {
        if jpeg.app_marker_type[i] != AppMarkerType::Icc {
            continue;
        }
        let data = &jpeg.app_data[i];
        if data.len() <= OVERHEAD {
            continue;
        }
        // Verify ICC_PROFILE tag at offset 3 (after marker_byte + length)
        let tag_start = 3;
        if data.len() > tag_start + ICC_TAG.len()
            && &data[tag_start..tag_start + ICC_TAG.len()] == ICC_TAG
        {
            // ICC payload starts after the 17-byte overhead
            icc_data.extend_from_slice(&data[OVERHEAD..]);
        }
    }
    if icc_data.is_empty() {
        None
    } else {
        Some(icc_data)
    }
}

/// Extract EXIF data from JPEG APP markers for the container Exif box.
///
/// Returns the raw EXIF data (after the "Exif\0\0" header), or None.
pub(crate) fn extract_exif(jpeg: &JpegData) -> Option<Vec<u8>> {
    // APP data format: [marker_byte, len_hi, len_lo, payload...]
    // EXIF payload starts with "Exif\0\0" (6 bytes)
    const EXIF_HEADER: &[u8] = b"Exif\0\0";
    for i in 0..jpeg.app_data.len() {
        if jpeg.app_marker_type[i] == AppMarkerType::Exif {
            let data = &jpeg.app_data[i];
            // Skip marker byte (1) + length (2) = 3, then find "Exif\0\0"
            let header_start = 3;
            if data.len() > header_start + EXIF_HEADER.len()
                && &data[header_start..header_start + EXIF_HEADER.len()] == EXIF_HEADER
            {
                return Some(data[header_start + EXIF_HEADER.len()..].to_vec());
            }
        }
    }
    None
}

/// Extract XMP data from JPEG APP markers for the container xml box.
///
/// Returns the raw XMP string, or None.
pub(crate) fn extract_xmp(jpeg: &JpegData) -> Option<Vec<u8>> {
    // APP data format: [marker_byte, len_hi, len_lo, payload...]
    // XMP payload starts with "http://ns.adobe.com/xap/1.0/\0"
    const XMP_HEADER: &[u8] = b"http://ns.adobe.com/xap/1.0/\0";
    for i in 0..jpeg.app_data.len() {
        if jpeg.app_marker_type[i] == AppMarkerType::Xmp {
            let data = &jpeg.app_data[i];
            let header_start = 3; // Skip marker byte + length
            let skip = header_start + XMP_HEADER.len();
            if data.len() > skip && &data[header_start..skip] == XMP_HEADER {
                return Some(data[skip..].to_vec());
            }
        }
    }
    None
}
