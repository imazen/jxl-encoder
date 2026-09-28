// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Channel-mult constants shared by `effort` defaults and
//! `vardct::ac_strategy` (defined here so `jxl-modular` stays a leaf:
//! `effort` must not reach up into `vardct`).

/// Historical default (`20882706.4655936`, 1.0, 1.03^8) — see
/// `vardct::ac_strategy` docs / `EntropyMulTable::channel_loss_mul`.
pub const CHANNEL_MUL: [f64; 3] = [
    20882706.4655936, // X channel: historical value (see note above)
    1.0,              // Y channel: 1.0^8
    1.26677008064,    // B channel: 1.03^8
];

/// libjxl `enc_ac_strategy.cc` `kChannelMul` —
/// `{pow(8.2, 8.0), 1.0, pow(1.03, 8.0)}`. Used by the strict
/// `EncoderStrategy::Libjxl` profile only (W45-RECON part 6).
pub const CHANNEL_MUL_LIBJXL: [f64; 3] = [
    20441408.586549744, // X channel: 8.2^8
    1.0,                // Y channel: 1.0^8
    1.2667700813876164, // B channel: 1.03^8
];

/// JXL codestream signature bytes.
pub const JXL_SIGNATURE: [u8; 2] = [0xFF, 0x0A];
