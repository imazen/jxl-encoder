// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Modular encoding for JPEG XL.
//!
//! The modular mode encodes images using prediction and entropy coding,
//! without DCT transforms. This is the primary mode for lossless encoding.

pub mod channel;
pub mod encode;
mod encode_primitives;
mod encode_transforms;
mod encode_tree;
// Float-sample packing (imazen/jxl-encoder#109). `float_to_int_sample` is the
// production f32 path (`ModularImage::from_float_native`).
// `int_to_float_sample` and `is_valid_float_format` are the decoder-side
// inverse and the format validator: both are exercised by this module's golden
// vectors, and the validator becomes load-bearing when #95 lifts
// `with_bits_per_sample` past 16 and arbitrary widths become reachable.
#[allow(dead_code)]
pub mod float_pack;
pub mod frame;
pub mod fuzz_safety;
pub mod inline_add_sample;
pub mod inline_dedup_table;
pub mod lz77_keep_best;
// Strict `EncoderStrategy::Libjxl` adaptive MA-tree learner — a line-level
// port of libjxl `enc_ma.cc`/`enc_encoding.cc` used only by the strict path.
pub mod ma_libjxl;
pub mod palette;
pub mod predictor;
pub mod predictor_prune;
pub mod quantize;
pub mod rct;
pub mod section;
pub mod squeeze;
pub mod tree;
pub mod tree_learn;
pub mod tree_learn_split;

// #76 (0.4.0): every re-export below except `RctType` is crate-internal
// (`pub use`). `RctType` is the one modular type on the supported
// surface — `LosslessConfig::with_rct_type` takes it, and the crate root
// re-exports it. The 207 pub item lines this module used to leak
// (GlobalModularState, ModularImage, FrameEncoder, the tree/predictor
// machinery, the section writers) are implementation detail; the
// `GlobalModularState` variant-field semver break from
// docs/RELEASE_SEMVER_0.3.1_to_0.3.2.md dies here.
pub use channel::Channel;
pub use predictor::Predictor;
pub use rct::RctType;

// `forced_wp_tests` moved to jxl-encoder/src/forced_wp_tests.rs (crate
// split); the observation hook itself stays here — the BitWriter being
// sampled lives in this crate.
#[cfg(all(feature = "__expert", feature = "std"))]
pub mod wp_observe;

// `lz77_keep_best_tests` moved to jxl-encoder/src/lz77_keep_best_tests.rs (crate split).
