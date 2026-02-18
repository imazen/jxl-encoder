// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Transform descriptor writing for modular encoding.
//!
//! Contains RCT (Reversible Color Transform), Palette, and Squeeze (Haar wavelet)
//! transform descriptors written to the JXL bitstream.

#![allow(dead_code)]

use crate::bit_writer::BitWriter;
use crate::error::Result;
use crate::modular::rct::RctType;

/// Write the RCT transform descriptor to the bitstream.
///
/// Format (for YCoCg with begin_c=0):
/// - TransformId: 2 bits (selector 0 = RCT)
/// - begin_c: 2 bits selector + 3 bits value = 5 bits for value 0
/// - rct_type: 2 bits (selector 0 = 6 = YCoCg)
pub(crate) fn write_rct_transform(
    writer: &mut BitWriter,
    begin_c: usize,
    rct_type: RctType,
) -> Result<()> {
    // TransformId: U32(Val(0)=RCT, Val(1)=Palette, Val(2)=Squeeze, Val(3)=Invalid)
    // RCT = selector 0 = 2 bits "00"
    writer.write(2, 0)?;

    // begin_c: U32(Bits(3), BitsOffset(6, 8), BitsOffset(10, 72), BitsOffset(13, 1096), 0)
    // For begin_c 0-7: selector 0 = 2 bits + 3 bits value
    if begin_c < 8 {
        writer.write(2, 0)?; // selector 0
        writer.write(3, begin_c as u64)?;
    } else if begin_c < 72 {
        writer.write(2, 1)?; // selector 1 = BitsOffset(6, 8)
        writer.write(6, (begin_c - 8) as u64)?;
    } else if begin_c < 1096 {
        writer.write(2, 2)?; // selector 2 = BitsOffset(10, 72)
        writer.write(10, (begin_c - 72) as u64)?;
    } else {
        writer.write(2, 3)?; // selector 3 = BitsOffset(13, 1096)
        writer.write(13, (begin_c - 1096) as u64)?;
    }

    // rct_type: U32(Val(6), Bits(2), BitsOffset(4, 2), BitsOffset(6, 10), 6)
    // Val(6) = YCoCg at selector 0
    // Bits(2) = 0-3 at selector 1
    // BitsOffset(4, 2) = 2-17 at selector 2
    // BitsOffset(6, 10) = 10-73 at selector 3
    let rct_val = rct_type.0 as u64;
    if rct_val == 6 {
        writer.write(2, 0)?; // selector 0 = Val(6)
    } else if rct_val < 2 {
        // 0-1 encoded as Bits(2) at selector 1, but 0 and 1 need 2 bits
        writer.write(2, 1)?;
        writer.write(2, rct_val)?;
    } else if rct_val < 10 {
        // 2-9 encoded as BitsOffset(4, 2) at selector 2
        writer.write(2, 2)?;
        writer.write(4, rct_val - 2)?;
    } else {
        // 10-73 encoded as BitsOffset(6, 10) at selector 3
        writer.write(2, 3)?;
        writer.write(6, rct_val - 10)?;
    }

    Ok(())
}

/// Write the Palette transform descriptor to the bitstream.
///
/// Format:
/// - TransformId: 2 bits (selector 1 = Palette)
/// - begin_c: U32(Bits(3), BitsOffset(6,8), BitsOffset(10,72), BitsOffset(13,1096))
/// - num_c: U32(Val(1), Val(3), Val(4), BitsOffset(13,1))
/// - nb_colors: U32(Bits(8), BitsOffset(10,256), BitsOffset(12,1280), BitsOffset(16,5376))
/// - nb_deltas: U32(Val(0), BitsOffset(8,1), BitsOffset(10,257), BitsOffset(16,1281))
/// - predictor: 4 bits (0=Zero for lossless)
pub(super) fn write_palette_transform(
    writer: &mut BitWriter,
    begin_c: usize,
    num_c: usize,
    nb_colors: usize,
    nb_deltas: usize,
    predictor: u8,
) -> Result<()> {
    // TransformId: U32(Val(0)=RCT, Val(1)=Palette, Val(2)=Squeeze, Val(3)=Invalid)
    // Palette = selector 1 = 2 bits "01"
    writer.write(2, 1)?;

    // begin_c: U32(Bits(3), BitsOffset(6, 8), BitsOffset(10, 72), BitsOffset(13, 1096))
    if begin_c < 8 {
        writer.write(2, 0)?;
        writer.write(3, begin_c as u64)?;
    } else if begin_c < 72 {
        writer.write(2, 1)?;
        writer.write(6, (begin_c - 8) as u64)?;
    } else if begin_c < 1096 {
        writer.write(2, 2)?;
        writer.write(10, (begin_c - 72) as u64)?;
    } else {
        writer.write(2, 3)?;
        writer.write(13, (begin_c - 1096) as u64)?;
    }

    // num_c: U32(Val(1), Val(3), Val(4), BitsOffset(13, 1))
    match num_c {
        1 => writer.write(2, 0)?, // selector 0 = Val(1)
        3 => writer.write(2, 1)?, // selector 1 = Val(3)
        4 => writer.write(2, 2)?, // selector 2 = Val(4)
        _ => {
            writer.write(2, 3)?; // selector 3 = BitsOffset(13, 1)
            writer.write(13, (num_c - 1) as u64)?;
        }
    }

    // nb_colors: U32(Bits(8), BitsOffset(10, 256), BitsOffset(12, 1280), BitsOffset(16, 5376))
    if nb_colors < 256 {
        writer.write(2, 0)?;
        writer.write(8, nb_colors as u64)?;
    } else if nb_colors < 1280 {
        writer.write(2, 1)?;
        writer.write(10, (nb_colors - 256) as u64)?;
    } else if nb_colors < 5376 {
        writer.write(2, 2)?;
        writer.write(12, (nb_colors - 1280) as u64)?;
    } else {
        writer.write(2, 3)?;
        writer.write(16, (nb_colors - 5376) as u64)?;
    }

    // nb_deltas: U32(Val(0), BitsOffset(8,1), BitsOffset(10,257), BitsOffset(16,1281))
    if nb_deltas == 0 {
        writer.write(2, 0)?; // selector 0 = Val(0)
    } else if nb_deltas <= 256 {
        writer.write(2, 1)?; // selector 1 = BitsOffset(8, 1)
        writer.write(8, (nb_deltas - 1) as u64)?;
    } else if nb_deltas <= 1280 {
        writer.write(2, 2)?; // selector 2 = BitsOffset(10, 257)
        writer.write(10, (nb_deltas - 257) as u64)?;
    } else {
        writer.write(2, 3)?; // selector 3 = BitsOffset(16, 1281)
        writer.write(16, (nb_deltas - 1281) as u64)?;
    }

    // predictor: 4 bits (0=Zero, 4=ClampedGradient, etc.)
    writer.write(4, predictor as u64)?;

    Ok(())
}

/// Write the Squeeze transform descriptor to the bitstream.
///
/// Format:
/// - TransformId: 2 bits (selector 2 = Squeeze)
/// - num_squeezes: U32(Val(0), BitsOffset(4,1), BitsOffset(6,9), BitsOffset(8,41))
///   Val(0) = default squeeze (decoder computes parameters)
/// - For each squeeze: horizontal(1 bit), in_place(1 bit),
///   begin_c(U32), num_c(U32(Val(1),Val(2),Val(3),BitsOffset(4,4)))
pub(crate) fn write_squeeze_transform(
    writer: &mut BitWriter,
    params: &[super::squeeze::SqueezeParams],
) -> Result<()> {
    // TransformId: Val(2) = Squeeze = selector 2 = "10"
    writer.write(2, 2)?;

    if params.is_empty() {
        // num_squeezes = 0: use default squeeze
        writer.write(2, 0)?; // selector 0 = Val(0)
    } else {
        // Encode num_squeezes
        let n = params.len();
        if (1..=16).contains(&n) {
            writer.write(2, 1)?; // selector 1 = BitsOffset(4, 1)
            writer.write(4, (n - 1) as u64)?;
        } else if (9..=72).contains(&n) {
            writer.write(2, 2)?; // selector 2 = BitsOffset(6, 9)
            writer.write(6, (n - 9) as u64)?;
        } else {
            writer.write(2, 3)?; // selector 3 = BitsOffset(8, 41)
            writer.write(8, (n - 41) as u64)?;
        }

        // Write each squeeze parameter
        for sp in params {
            writer.write(1, sp.horizontal as u64)?;
            writer.write(1, sp.in_place as u64)?;

            // begin_c: U32(Bits(3), BitsOffset(6,8), BitsOffset(10,72), BitsOffset(13,1096))
            let bc = sp.begin_c as usize;
            if bc < 8 {
                writer.write(2, 0)?;
                writer.write(3, bc as u64)?;
            } else if bc < 72 {
                writer.write(2, 1)?;
                writer.write(6, (bc - 8) as u64)?;
            } else if bc < 1096 {
                writer.write(2, 2)?;
                writer.write(10, (bc - 72) as u64)?;
            } else {
                writer.write(2, 3)?;
                writer.write(13, (bc - 1096) as u64)?;
            }

            // num_c: U32(Val(1), Val(2), Val(3), BitsOffset(4, 4))
            match sp.num_c {
                1 => writer.write(2, 0)?,
                2 => writer.write(2, 1)?,
                3 => writer.write(2, 2)?,
                _ => {
                    writer.write(2, 3)?;
                    writer.write(4, (sp.num_c - 4) as u64)?;
                }
            }
        }
    }

    Ok(())
}
