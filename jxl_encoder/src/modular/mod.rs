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
pub(crate) mod palette;
pub(crate) mod predictor;
pub(crate) mod quantize;
pub(crate) mod rct;
pub(crate) mod section;
pub(crate) mod squeeze;
pub(crate) mod tree;
pub(crate) mod tree_learn;

pub use channel::{Channel, ModularImage};
pub use encode::{
    build_histogram_from_residuals, collect_all_residuals, write_global_modular_section,
    write_group_modular_section, write_modular_stream_with_rct,
    write_modular_stream_with_rct_weighted, write_modular_stream_with_weighted,
};
pub use frame::{FrameEncoder, FrameEncoderOptions};
pub use predictor::{
    Neighbors, Predictor, WeightedPredictorParams, WeightedPredictorState, pack_signed,
    unpack_signed,
};
pub use rct::{RctType, forward_rct, inverse_rct};
pub use section::GlobalModularState;
pub use tree::{
    PixelProperties, Property, PropertyDecisionNode, Tree, TreeToken,
    adaptive_gradient_weighted_tree, collect_tree_tokens, count_contexts, gradient_tree,
    simple_tree, traverse_tree, weighted_tree,
};
