// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! ICC profile encoding for JPEG XL.
//!
//! Ported from libjxl's `enc_icc_codec.cc` and `icc_codec_common.cc`.
//! The ICC profile is transformed via PredictICC (making it more compressible),
//! then entropy-coded using Huffman with 41 contexts.

use crate::bit_writer::BitWriter;
use crate::entropy_coding::encode::build_entropy_code_with_options;
use crate::entropy_coding::lz77::{Lz77Method, apply_lz77, write_lz77_header};
use crate::entropy_coding::token::Token;
use crate::error::Result;

use alloc::collections::BTreeMap;

// ── Constants ───────────────────────────────────────────────────────────────

const ICC_HEADER_SIZE: usize = 128;
const NUM_ICC_CONTEXTS: usize = 41;

type Tag = [u8; 4];

// Tag names for RGB/GRAY monitor profiles
const TAG_CPRT: Tag = *b"cprt";
const TAG_WTPT: Tag = *b"wtpt";
const TAG_BKPT: Tag = *b"bkpt";
const TAG_RXYZ: Tag = *b"rXYZ";
const TAG_GXYZ: Tag = *b"gXYZ";
const TAG_BXYZ: Tag = *b"bXYZ";
const TAG_KXYZ: Tag = *b"kXYZ";
const TAG_RTRC: Tag = *b"rTRC";
const TAG_GTRC: Tag = *b"gTRC";
const TAG_BTRC: Tag = *b"bTRC";
const TAG_KTRC: Tag = *b"kTRC";
const TAG_CHAD: Tag = *b"chad";
const TAG_DESC: Tag = *b"desc";
const TAG_CHRM: Tag = *b"chrm";
const TAG_DMND: Tag = *b"dmnd";
const TAG_DMDD: Tag = *b"dmdd";
const TAG_LUMI: Tag = *b"lumi";

const TAG_STRINGS: [Tag; 17] = [
    TAG_CPRT, TAG_WTPT, TAG_BKPT, TAG_RXYZ, TAG_GXYZ, TAG_BXYZ, TAG_KXYZ, TAG_RTRC, TAG_GTRC,
    TAG_BTRC, TAG_KTRC, TAG_CHAD, TAG_DESC, TAG_CHRM, TAG_DMND, TAG_DMDD, TAG_LUMI,
];

// Tag types
const TAG_XYZ: Tag = *b"XYZ ";
const TAG_DESC_TYPE: Tag = *b"desc";
const TAG_TEXT: Tag = *b"text";
const TAG_MLUC: Tag = *b"mluc";
const TAG_PARA: Tag = *b"para";
const TAG_CURV: Tag = *b"curv";
const TAG_SF32: Tag = *b"sf32";
const TAG_GBD: Tag = *b"gbd ";

const TYPE_STRINGS: [Tag; 8] = [
    TAG_XYZ,
    TAG_DESC_TYPE,
    TAG_TEXT,
    TAG_MLUC,
    TAG_PARA,
    TAG_CURV,
    TAG_SF32,
    TAG_GBD,
];

// Command constants
const COMMAND_INSERT: u8 = 1;
const COMMAND_SHUFFLE2: u8 = 2;
const COMMAND_PREDICT: u8 = 4;
const COMMAND_XYZ: u8 = 10;
const COMMAND_TYPE_START_FIRST: u8 = 16;

const COMMAND_TAG_UNKNOWN: u8 = 1;
const COMMAND_TAG_TRC: u8 = 2;
const COMMAND_TAG_XYZ: u8 = 3;
const COMMAND_TAG_STRING_FIRST: u8 = 4;

const FLAG_BIT_OFFSET: u8 = 64;
const FLAG_BIT_SIZE: u8 = 128;

const SIZE_LIMIT: usize = u32::MAX as usize >> 2;

// Initial header prediction (matches libjxl's kIccInitialHeaderPrediction)
#[rustfmt::skip]
const ICC_INITIAL_HEADER_PREDICTION: [u8; ICC_HEADER_SIZE] = [
    0,   0,   0,   0,   0,   0,   0,   0,   4,   0,   0,   0,  b'm', b'n', b't', b'r',
    b'R', b'G', b'B', b' ', b'X', b'Y', b'Z', b' ', 0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,  b'a', b'c', b's', b'p',  0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   246, 214, 0,   1,   0,   0,   0,   0,   211, 45,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
];

// ── Helper functions ────────────────────────────────────────────────────────

fn decode_uint32(data: &[u8], pos: usize) -> u32 {
    if pos + 4 > data.len() {
        0
    } else {
        u32::from_be_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]])
    }
}

fn decode_keyword(data: &[u8], pos: usize) -> Tag {
    if pos + 4 > data.len() {
        *b"    "
    } else {
        [data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]
    }
}

fn append_keyword(keyword: &Tag, out: &mut Vec<u8>) {
    out.extend_from_slice(keyword);
}

fn encode_var_int(value: u64, out: &mut Vec<u8>) {
    let mut v = value;
    while v > 127 {
        out.push((v as u8 & 127) | 128);
        v >>= 7;
    }
    out.push(v as u8 & 127);
}

fn icc_initial_header_prediction(size: u32) -> [u8; ICC_HEADER_SIZE] {
    let mut copy = ICC_INITIAL_HEADER_PREDICTION;
    let bytes = size.to_be_bytes();
    copy[0] = bytes[0];
    copy[1] = bytes[1];
    copy[2] = bytes[2];
    copy[3] = bytes[3];
    copy
}

fn icc_predict_header(icc: &[u8], header: &mut [u8; ICC_HEADER_SIZE], pos: usize) {
    if pos == 8 && icc.len() >= 8 {
        header[80] = icc[4];
        header[81] = icc[5];
        header[82] = icc[6];
        header[83] = icc[7];
    }
    if pos == 41 && icc.len() >= 41 {
        if icc[40] == b'A' {
            header[41] = b'P';
            header[42] = b'P';
            header[43] = b'L';
        }
        if icc[40] == b'M' {
            header[41] = b'S';
            header[42] = b'F';
            header[43] = b'T';
        }
    }
    if pos == 42 && icc.len() >= 42 {
        if icc[40] == b'S' && icc[41] == b'G' {
            header[42] = b'I';
            header[43] = b' ';
        }
        if icc[40] == b'S' && icc[41] == b'U' {
            header[42] = b'N';
            header[43] = b'W';
        }
    }
}

fn predict_value_u8(p1: u8, p2: u8, p3: u8, order: i32) -> u8 {
    match order {
        0 => p1,
        1 => (2u16.wrapping_mul(p1 as u16)).wrapping_sub(p2 as u16) as u8,
        2 => (3u16.wrapping_mul(p1 as u16))
            .wrapping_sub(3u16.wrapping_mul(p2 as u16))
            .wrapping_add(p3 as u16) as u8,
        _ => 0,
    }
}

fn predict_value_u16(p1: u16, p2: u16, p3: u16, order: i32) -> u16 {
    match order {
        0 => p1,
        1 => (2u32.wrapping_mul(p1 as u32)).wrapping_sub(p2 as u32) as u16,
        2 => (3u32.wrapping_mul(p1 as u32))
            .wrapping_sub(3u32.wrapping_mul(p2 as u32))
            .wrapping_add(p3 as u32) as u16,
        _ => 0,
    }
}

fn predict_value_u32(p1: u32, p2: u32, p3: u32, order: i32) -> u32 {
    match order {
        0 => p1,
        1 => (2u64.wrapping_mul(p1 as u64)).wrapping_sub(p2 as u64) as u32,
        2 => (3u64.wrapping_mul(p1 as u64))
            .wrapping_sub(3u64.wrapping_mul(p2 as u64))
            .wrapping_add(p3 as u64) as u32,
        _ => 0,
    }
}

fn linear_predict_icc_value(
    data: &[u8],
    start: usize,
    i: usize,
    stride: usize,
    width: usize,
    order: i32,
) -> u8 {
    let pos = start + i;
    if width == 1 {
        let p1 = data[pos - stride];
        let p2 = data[pos - stride * 2];
        let p3 = data[pos - stride * 3];
        predict_value_u8(p1, p2, p3, order)
    } else if width == 2 {
        let p = start + (i & !1);
        let p1 = (data[p - stride] as u16) << 8 | data[p - stride + 1] as u16;
        let p2 = (data[p - stride * 2] as u16) << 8 | data[p - stride * 2 + 1] as u16;
        let p3 = (data[p - stride * 3] as u16) << 8 | data[p - stride * 3 + 1] as u16;
        let pred = predict_value_u16(p1, p2, p3, order);
        if (i & 1) != 0 {
            (pred & 255) as u8
        } else {
            ((pred >> 8) & 255) as u8
        }
    } else {
        let p = start + (i & !3);
        let p1 = decode_uint32(data, p - stride);
        let p2 = decode_uint32(data, p - stride * 2);
        let p3 = decode_uint32(data, p - stride * 3);
        let pred = predict_value_u32(p1, p2, p3, order);
        let shiftbytes = 3 - (i & 3);
        ((pred >> (shiftbytes * 8)) & 255) as u8
    }
}

/// Unshuffle (de-interleave) bytes. With width=2, turns "AaBbCcDd" into "ABCDabcd".
fn unshuffle(data: &mut [u8], width: usize) {
    let size = data.len();
    let height = size.div_ceil(width);
    let mut result = vec![0u8; size];

    let mut s = 0;
    let mut j = 0;
    for &byte in &data[..size] {
        result[j] = byte;
        j += height;
        if j >= size {
            s += 1;
            j = s;
        }
    }
    data.copy_from_slice(&result);
}

/// Predict residuals and unshuffle a section of ICC data.
fn predict_and_shuffle(
    stride: usize,
    width: usize,
    order: i32,
    num: usize,
    data: &[u8],
    pos: &mut usize,
    result: &mut Vec<u8>,
) {
    assert!(*pos + num <= data.len());
    assert!(*pos >= stride * 4);

    let start = result.len();
    for i in 0..num {
        let predicted = linear_predict_icc_value(data, *pos, i, stride, width, order);
        result.push(data[*pos + i].wrapping_sub(predicted));
    }
    *pos += num;
    if width > 1 {
        unshuffle(&mut result[start..], width);
    }
}

// ── Context function ────────────────────────────────────────────────────────

fn byte_kind1(b: u8) -> u8 {
    if b.is_ascii_lowercase() || b.is_ascii_uppercase() {
        return 0;
    }
    if b.is_ascii_digit() || b == b'.' || b == b',' {
        return 1;
    }
    if b == 0 {
        return 2;
    }
    if b == 1 {
        return 3;
    }
    if b < 16 {
        return 4;
    }
    if b == 255 {
        return 6;
    }
    if b > 240 {
        return 5;
    }
    7
}

fn byte_kind2(b: u8) -> u8 {
    if b.is_ascii_lowercase() || b.is_ascii_uppercase() {
        return 0;
    }
    if b.is_ascii_digit() || b == b'.' || b == b',' {
        return 1;
    }
    if b < 16 {
        return 2;
    }
    if b > 240 {
        return 3;
    }
    4
}

fn icc_ans_context(i: usize, b1: u8, b2: u8) -> u32 {
    if i <= 128 {
        0
    } else {
        1 + byte_kind1(b1) as u32 + byte_kind2(b2) as u32 * 8
    }
}

/// Encode predict flags byte: (order << 2) | (width - 1) | (stride != width ? 16 : 0)
fn predict_flags(order: i32, width: usize, stride: usize) -> u8 {
    ((order << 2) as u8) | ((width - 1) as u8) | if stride == width { 0 } else { 16 }
}

// ── PredictICC ──────────────────────────────────────────────────────────────

/// Transform an ICC profile into a more compressible form.
///
/// The output is a varint-encoded size, followed by commands, followed by data.
/// This matches libjxl's `PredictICC()`.
fn predict_icc(icc: &[u8]) -> Vec<u8> {
    let size = icc.len();
    assert!(size <= SIZE_LIMIT, "ICC profile too large");

    let mut result = Vec::new();
    let mut commands = Vec::new();
    let mut data = Vec::new();

    encode_var_int(size as u64, &mut result);

    // Header prediction
    let mut header = icc_initial_header_prediction(size as u32);
    for i in 0..ICC_HEADER_SIZE.min(size) {
        icc_predict_header(icc, &mut header, i);
        data.push(icc[i].wrapping_sub(header[i]));
    }

    if size <= ICC_HEADER_SIZE {
        encode_var_int(0, &mut result); // 0 commands
        result.extend_from_slice(&data);
        return result;
    }

    // Parse tag table
    let mut tags: Vec<Tag> = Vec::new();
    let mut tagstarts: Vec<usize> = Vec::new();
    let mut tagsizes: Vec<usize> = Vec::new();
    let mut tagmap: BTreeMap<usize, usize> = BTreeMap::new();

    let mut pos = ICC_HEADER_SIZE;

    if pos + 4 <= size {
        let numtags = decode_uint32(icc, pos) as u64;
        pos += 4;
        encode_var_int(numtags + 1, &mut commands);

        let mut prevtagstart = ICC_HEADER_SIZE + (numtags as usize) * 12;
        let mut prevtagsize: u32 = 0;
        let mut i = 0usize;
        while i < numtags as usize {
            if pos + 12 > size {
                break;
            }

            let tag = decode_keyword(icc, pos);
            let tagstart = decode_uint32(icc, pos + 4) as usize;
            let tagsize = decode_uint32(icc, pos + 8) as usize;
            pos += 12;

            tags.push(tag);
            tagstarts.push(tagstart);
            tagsizes.push(tagsize);
            tagmap.insert(tagstart, tags.len() - 1);

            let mut tagcode = COMMAND_TAG_UNKNOWN;
            for (j, ts) in TAG_STRINGS.iter().enumerate() {
                if tag == *ts {
                    tagcode = j as u8 + COMMAND_TAG_STRING_FIRST;
                    break;
                }
            }

            // Check for rTRC/gTRC/bTRC triple with identical data
            if tag == TAG_RTRC && pos + 24 < size {
                let mut ok = true;
                ok &= decode_keyword(icc, pos) == TAG_GTRC;
                ok &= decode_keyword(icc, pos + 12) == TAG_BTRC;
                if ok {
                    for kk in 0..8 {
                        if icc[pos - 8 + kk] != icc[pos + 4 + kk] {
                            ok = false;
                        }
                        if icc[pos - 8 + kk] != icc[pos + 16 + kk] {
                            ok = false;
                        }
                    }
                }
                if ok {
                    tagcode = COMMAND_TAG_TRC;
                    pos += 24;
                    i += 2;
                }
            }

            // Check for rXYZ/gXYZ/bXYZ triple with standard layout
            if tag == TAG_RXYZ && pos + 24 < size {
                let mut ok = true;
                ok &= decode_keyword(icc, pos) == TAG_GXYZ;
                ok &= decode_keyword(icc, pos + 12) == TAG_BXYZ;
                let offsetr = tagstart;
                let offsetg = decode_uint32(icc, pos + 4) as usize;
                let offsetb = decode_uint32(icc, pos + 16) as usize;
                let sizer = tagsize as u32;
                let sizeg = decode_uint32(icc, pos + 8);
                let sizeb = decode_uint32(icc, pos + 20);
                ok &= sizer == 20;
                ok &= sizeg == 20;
                ok &= sizeb == 20;
                ok &= offsetg == offsetr + 20;
                ok &= offsetb == offsetr + 40;
                if ok {
                    tagcode = COMMAND_TAG_XYZ;
                    pos += 24;
                    i += 2;
                }
            }

            let mut command = tagcode;
            let predicted_tagstart = prevtagstart + prevtagsize as usize;
            if predicted_tagstart != tagstart {
                command |= FLAG_BIT_OFFSET;
            }
            let mut predicted_tagsize = prevtagsize;
            if tag == TAG_RXYZ
                || tag == TAG_GXYZ
                || tag == TAG_BXYZ
                || tag == TAG_KXYZ
                || tag == TAG_WTPT
                || tag == TAG_BKPT
                || tag == TAG_LUMI
            {
                predicted_tagsize = 20;
            }
            if predicted_tagsize != tagsize as u32 {
                command |= FLAG_BIT_SIZE;
            }
            commands.push(command);
            if tagcode == 1 {
                append_keyword(&tag, &mut data);
            }
            if command & FLAG_BIT_OFFSET != 0 {
                encode_var_int(tagstart as u64, &mut commands);
            }
            if command & FLAG_BIT_SIZE != 0 {
                encode_var_int(tagsize as u64, &mut commands);
            }

            prevtagstart = tagstart;
            prevtagsize = tagsize as u32;
            i += 1;
        }
    }
    // End of tag list
    commands.push(0);

    // Main content processing
    let mut tag = [0u8; 4];
    let mut tagstart: usize = 0;
    let mut tagsize: usize = 0;
    let mut clutstart: usize = 0;

    let tag_sane = |ts: usize| -> bool { ts > 8 && ts < SIZE_LIMIT };

    let mut last0 = pos;

    while pos <= size {
        let last1 = pos;
        let mut commands_add: Vec<u8> = Vec::new();
        let mut data_add: Vec<u8> = Vec::new();

        // Check if position is beyond current tag
        if pos > tagstart + tagsize && tagsize < SIZE_LIMIT {
            tag = [0, 0, 0, 0];
        }

        // Check for start of a new tag
        if commands_add.is_empty()
            && data_add.is_empty()
            && tagmap.contains_key(&pos)
            && pos + 4 <= size
        {
            let index = tagmap[&pos];
            tag = decode_keyword(icc, pos);
            tagstart = tagstarts[index];
            tagsize = tagsizes[index];

            // mluc tag
            if tag == TAG_MLUC
                && tag_sane(tagsize)
                && pos + tagsize <= size
                && icc[pos + 4] == 0
                && icc[pos + 5] == 0
                && icc[pos + 6] == 0
                && icc[pos + 7] == 0
            {
                let num = tagsize - 8;
                commands_add.push(COMMAND_TYPE_START_FIRST + 3); // mluc is index 3
                pos += 8;
                commands_add.push(COMMAND_SHUFFLE2);
                encode_var_int(num as u64, &mut commands_add);
                let start = data_add.len();
                for _ in 0..num {
                    data_add.push(icc[pos]);
                    pos += 1;
                }
                unshuffle(&mut data_add[start..], 2);
            }

            // curv tag
            if tag == TAG_CURV
                && tag_sane(tagsize)
                && pos + tagsize <= size
                && icc[pos + 4] == 0
                && icc[pos + 5] == 0
                && icc[pos + 6] == 0
                && icc[pos + 7] == 0
            {
                let num = tagsize - 8;
                if num > 16 && num < (1 << 28) && pos + num <= size && pos > 0 {
                    commands_add.push(COMMAND_TYPE_START_FIRST + 5); // curv is index 5
                    pos += 8;
                    commands_add.push(COMMAND_PREDICT);
                    let order: i32 = 1;
                    let width: usize = 2;
                    commands_add.push(predict_flags(order, width, width));
                    encode_var_int(num as u64, &mut commands_add);
                    predict_and_shuffle(width, width, order, num, icc, &mut pos, &mut data_add);
                }
            }
        }

        // mAB/mBA sub-tags
        if tag == *b"mAB " || tag == *b"mBA " {
            let sub_tag = decode_keyword(icc, pos);
            if pos + 12 < size
                && (sub_tag == TAG_CURV || sub_tag == *b"vcgt")
                && decode_uint32(icc, pos + 4) == 0
            {
                let num = decode_uint32(icc, pos + 8) as usize * 2;
                if num > 16 && num < (1 << 28) && pos + 12 + num <= size {
                    pos += 12;
                    commands_add.push(COMMAND_PREDICT);
                    let order: i32 = 1;
                    let width: usize = 2;
                    commands_add.push(predict_flags(order, width, width));
                    encode_var_int(num as u64, &mut commands_add);
                    predict_and_shuffle(width, width, order, num, icc, &mut pos, &mut data_add);
                }
            }

            // CLUT offset detection
            if pos == tagstart + 24 && pos + 4 < size {
                clutstart = tagstart + decode_uint32(icc, pos) as usize;
            }

            // CLUT data prediction
            if pos == clutstart && clutstart + 16 < size {
                let numi = icc[tagstart + 8] as usize;
                let numo = icc[tagstart + 9] as usize;
                let width = icc[clutstart + 16] as usize;
                let stride = width * numo;
                let mut num = width * numo;
                for ci in 0..numi {
                    if clutstart + ci < size {
                        num *= icc[clutstart + ci] as usize;
                    }
                }
                if (width == 1 || width == 2)
                    && num > 64
                    && num < (1 << 28)
                    && pos + num <= size
                    && pos > stride * 4
                {
                    let order: i32 = 1;
                    let flags = predict_flags(order, width, stride);
                    commands_add.push(COMMAND_PREDICT);
                    commands_add.push(flags);
                    if flags & 16 != 0 {
                        encode_var_int(stride as u64, &mut commands_add);
                    }
                    encode_var_int(num as u64, &mut commands_add);
                    predict_and_shuffle(stride, width, order, num, icc, &mut pos, &mut data_add);
                }
            }
        }

        // gbd tag
        if commands_add.is_empty()
            && data_add.is_empty()
            && tag == TAG_GBD
            && tag_sane(tagsize)
            && pos == tagstart + 8
            && pos + tagsize - 8 <= size
            && pos > 16
        {
            let width: usize = 4;
            let order: i32 = 0;
            let stride = width;
            let num = tagsize - 8;
            let flags = predict_flags(order, width, stride);
            commands_add.push(COMMAND_PREDICT);
            commands_add.push(flags);
            if flags & 16 != 0 {
                encode_var_int(stride as u64, &mut commands_add);
            }
            encode_var_int(num as u64, &mut commands_add);
            predict_and_shuffle(stride, width, order, num, icc, &mut pos, &mut data_add);
        }

        // XYZ tag type
        if commands_add.is_empty() && data_add.is_empty() && pos + 20 <= size {
            let sub_tag = decode_keyword(icc, pos);
            if sub_tag == TAG_XYZ && decode_uint32(icc, pos + 4) == 0 {
                commands_add.push(COMMAND_XYZ);
                pos += 8;
                for _ in 0..12 {
                    data_add.push(icc[pos]);
                    pos += 1;
                }
            }
        }

        // Generic type tag detection
        if commands_add.is_empty()
            && data_add.is_empty()
            && pos + 8 <= size
            && decode_uint32(icc, pos + 4) == 0
        {
            let sub_tag = decode_keyword(icc, pos);
            for (ti, ts) in TYPE_STRINGS.iter().enumerate() {
                if sub_tag == *ts {
                    commands_add.push(COMMAND_TYPE_START_FIRST + ti as u8);
                    pos += 8;
                    break;
                }
            }
        }

        // Flush pending data
        if !(commands_add.is_empty() && data_add.is_empty()) || pos == size {
            if last0 < last1 {
                commands.push(COMMAND_INSERT);
                encode_var_int((last1 - last0) as u64, &mut commands);
                for b in &icc[last0..last1] {
                    data.push(*b);
                }
            }
            commands.extend_from_slice(&commands_add);
            data.extend_from_slice(&data_add);
            last0 = pos;
        }
        if commands_add.is_empty() && data_add.is_empty() {
            pos += 1;
        }
    }

    // Write output: varint(commands_len) + commands + data
    encode_var_int(commands.len() as u64, &mut result);
    result.extend_from_slice(&commands);
    result.extend_from_slice(&data);
    result
}

// ── U64 Coder ───────────────────────────────────────────────────────────────

/// Write a JXL U64 (varint) value to the bitstream.
fn write_u64(value: u64, writer: &mut BitWriter) -> Result<()> {
    if value == 0 {
        writer.write(2, 0)?;
    } else if value <= 16 {
        writer.write(2, 1)?;
        writer.write(4, value - 1)?;
    } else if value <= 272 {
        writer.write(2, 2)?;
        writer.write(8, value - 17)?;
    } else {
        writer.write(2, 3)?;
        writer.write(12, value & 4095)?;
        let mut v = value >> 12;
        let mut shift = 12;
        while v > 0 && shift < 60 {
            writer.write(1, 1)?; // continue
            writer.write(8, v & 255)?;
            v >>= 8;
            shift += 8;
        }
        if v > 0 {
            writer.write(1, 1)?; // continue
            writer.write(4, v & 15)?;
            // Implicitly closed, no stop bit needed
        } else {
            writer.write(1, 0)?; // stop
        }
    }
    Ok(())
}

// ── WriteICC ────────────────────────────────────────────────────────────────

/// Write an ICC profile to the JXL bitstream.
///
/// The ICC data is transformed via PredictICC for better compressibility, then
/// entropy-coded using Huffman with 41 contexts (matching libjxl's force_huffman=true).
pub fn write_icc(icc: &[u8], writer: &mut BitWriter) -> Result<()> {
    assert!(!icc.is_empty(), "ICC profile must not be empty");

    let enc = predict_icc(icc);

    // Write predicted size as U64
    write_u64(enc.len() as u64, writer)?;

    // Tokenize predicted bytes with ICC contexts
    let mut tokens = Vec::with_capacity(enc.len());
    for i in 0..enc.len() {
        let b1 = if i > 0 { enc[i - 1] } else { 0 };
        let b2 = if i > 1 { enc[i - 2] } else { 0 };
        let ctx = icc_ans_context(i, b1, b2);
        tokens.push(Token::new(ctx, enc[i] as u32));
    }

    // Apply LZ77 to ICC tokens (libjxl enc_icc_codec.cc:455-482).
    // Optimal for small profiles (<16KB), greedy for larger.
    // ICC always uses Huffman (force_huffman=true).
    let lz77_method = if enc.len() < 16384 {
        Lz77Method::Optimal
    } else {
        Lz77Method::Greedy
    };
    let (tokens, lz77_params) = match apply_lz77(
        &tokens,
        NUM_ICC_CONTEXTS,
        true, // force_huffman
        lz77_method,
        0, // no special distance codes for ICC
    ) {
        Some((lz77_tokens, params)) => (lz77_tokens, Some(params)),
        None => (tokens, None),
    };

    // Build Huffman entropy code (libjxl uses force_huffman=true for ICC)
    let num_contexts = if lz77_params.is_some() {
        NUM_ICC_CONTEXTS + 1 // +1 for LZ77 distance context
    } else {
        NUM_ICC_CONTEXTS
    };
    let code = build_entropy_code_with_options(&tokens, num_contexts, false, lz77_params.as_ref());

    // Write LZ77 header (enabled or disabled)
    write_lz77_header(lz77_params.as_ref(), writer)?;

    // Write entropy code header (context map + prefix codes)
    code.write_header(writer)?;

    // Write tokens (with LZ77 if enabled)
    code.write_tokens_owned(&tokens, lz77_params.as_ref(), writer)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_predict_icc_small_profile() {
        // Minimal 128-byte ICC header only
        let mut icc = vec![0u8; 128];
        let size_bytes = 128u32.to_be_bytes();
        icc[..4].copy_from_slice(&size_bytes);

        let predicted = predict_icc(&icc);
        assert!(!predicted.is_empty());
    }

    #[test]
    fn test_predict_icc_roundtrip_structure() {
        let mut icc = vec![0u8; 256];
        let size_bytes = 256u32.to_be_bytes();
        icc[..4].copy_from_slice(&size_bytes);
        icc[12..16].copy_from_slice(b"mntr");
        icc[16..20].copy_from_slice(b"RGB ");
        icc[20..24].copy_from_slice(b"XYZ ");
        icc[36..40].copy_from_slice(b"acsp");
        // Tag count = 1
        icc[128..132].copy_from_slice(&1u32.to_be_bytes());
        icc[132..136].copy_from_slice(b"cprt");
        icc[136..140].copy_from_slice(&144u32.to_be_bytes());
        icc[140..144].copy_from_slice(&12u32.to_be_bytes());
        icc[144..148].copy_from_slice(b"text");

        let predicted = predict_icc(&icc);
        assert!(!predicted.is_empty());
    }

    #[test]
    fn test_write_u64_values() {
        let mut w = BitWriter::new();
        write_u64(0, &mut w).unwrap();
        write_u64(1, &mut w).unwrap();
        write_u64(16, &mut w).unwrap();
        write_u64(17, &mut w).unwrap();
        write_u64(272, &mut w).unwrap();
        write_u64(273, &mut w).unwrap();
        write_u64(1_000_000, &mut w).unwrap();
        assert!(w.bits_written() > 0);
    }

    #[test]
    fn test_icc_ans_context_header_region() {
        for i in 0..=128 {
            assert_eq!(icc_ans_context(i, 42, 42), 0);
        }
        assert!(icc_ans_context(129, 0, 0) > 0);
    }

    #[test]
    fn test_unshuffle() {
        let mut data = vec![b'A', b'a', b'B', b'b', b'C', b'c'];
        unshuffle(&mut data, 2);
        assert_eq!(data, vec![b'A', b'B', b'C', b'a', b'b', b'c']);
    }

    #[test]
    fn test_encode_var_int() {
        let mut out = Vec::new();
        encode_var_int(0, &mut out);
        assert_eq!(out, vec![0]);

        out.clear();
        encode_var_int(127, &mut out);
        assert_eq!(out, vec![127]);

        out.clear();
        encode_var_int(128, &mut out);
        assert_eq!(out, vec![128, 1]);
    }
}
