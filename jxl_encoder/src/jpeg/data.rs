// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Data structures for JPEG metadata, matching libjxl's JPEGData.
//!
//! These structures hold everything needed to reconstruct the original JPEG
//! bytes bit-exactly from a JPEG XL container.

/// Classification of APP markers for metadata extraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum AppMarkerType {
    /// Unknown APP marker (preserved in jbrd box).
    Unknown = 0,
    /// ICC profile (APP2 with "ICC_PROFILE\0" header).
    Icc = 1,
    /// EXIF data (APP1 with "Exif\0\0" header).
    Exif = 2,
    /// XMP data (APP1 with "http://ns.adobe.com/xap/1.0/\0" header).
    Xmp = 3,
}

/// JPEG quantization table.
#[derive(Debug, Clone)]
pub struct JpegQuantTable {
    /// 64 quantization values in natural (row-major) order.
    pub values: [i32; 64],
    /// Precision: 0 = 8-bit, 1 = 16-bit.
    pub precision: u32,
    /// DQT table index (0-3).
    pub index: u32,
    /// Whether this is the last table in its DQT marker.
    pub is_last: bool,
}

/// JPEG Huffman code table.
#[derive(Debug, Clone)]
pub struct JpegHuffmanCode {
    /// Whether this is an AC table (true) or DC table (false).
    pub is_ac: bool,
    /// Table ID (0-3).
    pub id: u32,
    /// Whether this is the last table in its DHT marker.
    pub is_last: bool,
    /// Number of codes at each bit length (1-16). Index 0 = 1-bit codes.
    pub counts: [u32; 16],
    /// Symbol values, ordered by code length then value.
    pub values: Vec<u8>,
}

/// JPEG image component.
#[derive(Debug, Clone)]
pub struct JpegComponent {
    /// Component ID byte (typically 1/2/3 for YCbCr, 'R'/'G'/'B' for RGB).
    pub id: u32,
    /// Horizontal sampling factor (1-4).
    pub h_samp_factor: u32,
    /// Vertical sampling factor (1-4).
    pub v_samp_factor: u32,
    /// Index into `JpegData::quant` for this component's quant table.
    pub quant_idx: u32,
    /// Width in 8x8 blocks.
    pub width_in_blocks: u32,
    /// Height in 8x8 blocks.
    pub height_in_blocks: u32,
    /// Quantized DCT coefficients, stored block-by-block in raster order.
    /// Each block has 64 coefficients in natural (row-major, NOT zigzag) order.
    /// Values are quantized (divided by quant table values).
    pub coeffs: Vec<i16>,
}

/// JPEG scan information.
#[derive(Debug, Clone)]
pub struct JpegScanInfo {
    /// Number of components in this scan (1-4).
    pub num_components: u32,
    /// Component indices (into JpegData::components).
    pub component_indices: Vec<u32>,
    /// DC Huffman table index per component.
    pub dc_tbl_idx: Vec<u32>,
    /// AC Huffman table index per component.
    pub ac_tbl_idx: Vec<u32>,
    /// Start of spectral selection (Ss).
    pub ss: u32,
    /// End of spectral selection (Se).
    pub se: u32,
    /// Successive approximation high bit (Ah).
    pub ah: u32,
    /// Successive approximation low bit (Al).
    pub al: u32,
    /// Block indices where RST markers occur.
    pub reset_points: Vec<u32>,
    /// (block_index, num_extra_zero_runs) for extra zero runs before EOB.
    pub extra_zero_runs: Vec<(u32, u32)>,
}

/// JPEG component type classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum JpegComponentType {
    Gray = 0,
    YCbCr = 1,
    Rgb = 2,
    Custom = 3,
}

/// Complete JPEG data for lossless reencoding.
///
/// This structure holds all the information needed to reconstruct the original
/// JPEG file bit-exactly from JPEG XL.
#[derive(Debug, Clone)]
pub struct JpegData {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Restart interval (from DRI marker, 0 = none).
    pub restart_interval: u32,
    /// Raw APP marker data (including the APP marker type byte and length).
    pub app_data: Vec<Vec<u8>>,
    /// Classification of each APP marker.
    pub app_marker_type: Vec<AppMarkerType>,
    /// Raw COM marker data.
    pub com_data: Vec<Vec<u8>>,
    /// Quantization tables.
    pub quant: Vec<JpegQuantTable>,
    /// Huffman code tables.
    pub huffman_code: Vec<JpegHuffmanCode>,
    /// Image components.
    pub components: Vec<JpegComponent>,
    /// Scan information.
    pub scan_info: Vec<JpegScanInfo>,
    /// Marker order (sequence of second bytes of 0xFF XX markers).
    pub marker_order: Vec<u8>,
    /// Data between markers (bytes between end of one marker and start of next).
    pub inter_marker_data: Vec<Vec<u8>>,
    /// Data after EOI marker.
    pub tail_data: Vec<u8>,
    /// Whether there are any non-zero padding bits.
    pub has_zero_padding_bit: bool,
    /// Individual padding bits (bit-stuffing values).
    pub padding_bits: Vec<u8>,

    /// Component type classification (derived from component IDs and markers).
    pub component_type: JpegComponentType,
}

/// JPEG zigzag to natural order lookup table.
/// `JPEG_NATURAL_ORDER[zigzag_index]` = natural (row-major) index.
pub const JPEG_NATURAL_ORDER: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

/// Natural order to zigzag order lookup table.
/// `JPEG_ZIGZAG_ORDER[natural_index]` = zigzag index.
pub const JPEG_ZIGZAG_ORDER: [usize; 64] = {
    let mut table = [0usize; 64];
    let mut i = 0;
    while i < 64 {
        table[JPEG_NATURAL_ORDER[i]] = i;
        i += 1;
    }
    table
};
