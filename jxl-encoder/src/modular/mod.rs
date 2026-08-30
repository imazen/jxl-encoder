// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Modular encoding for JPEG XL.
//!
//! The modular mode encodes images using prediction and entropy coding,
//! without DCT transforms. This is the primary mode for lossless encoding.

pub(crate) mod channel;
pub(crate) mod encode;
mod encode_primitives;
mod encode_transforms;
mod encode_tree;
pub(crate) mod frame;
pub(crate) mod fuzz_safety;
pub(crate) mod inline_add_sample;
pub(crate) mod inline_dedup_table;
pub(crate) mod palette;
pub(crate) mod predictor;
pub(crate) mod predictor_prune;
pub(crate) mod quantize;
pub(crate) mod rct;
pub(crate) mod section;
pub(crate) mod squeeze;
pub(crate) mod tree;
pub(crate) mod tree_learn;
pub(crate) mod tree_learn_split;

// #76 (0.4.0): every re-export below except `RctType` is crate-internal
// (`pub(crate) use`). `RctType` is the one modular type on the supported
// surface — `LosslessConfig::with_rct_type` takes it, and the crate root
// re-exports it. The 207 pub item lines this module used to leak
// (GlobalModularState, ModularImage, FrameEncoder, the tree/predictor
// machinery, the section writers) are implementation detail; the
// `GlobalModularState` variant-field semver break from
// docs/RELEASE_SEMVER_0.3.1_to_0.3.2.md dies here.
pub(crate) use channel::Channel;
pub(crate) use predictor::Predictor;
pub use rct::RctType;
