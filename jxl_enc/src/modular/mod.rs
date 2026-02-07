// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Modular encoding for JPEG XL.
//!
//! The modular mode encodes images using prediction and entropy coding,
//! without DCT transforms. This is the primary mode for lossless encoding.

pub mod channel;
pub mod encoder;
pub mod improved;
pub mod minimal;
pub mod palette;
pub mod predictor;
pub mod rct;
pub mod section;
pub mod squeeze;
pub mod token;
pub mod tree;
pub mod tree_learn;

pub use channel::{Channel, ModularImage};
pub use encoder::{EncodedModularData, ModularEncoder, ModularEncoderOptions};
pub use improved::{
    build_histogram_from_residuals, collect_all_residuals, write_global_modular_section,
    write_group_modular_section, write_modular_stream_with_rct,
    write_modular_stream_with_rct_weighted, write_modular_stream_with_weighted,
};
pub use predictor::{
    Neighbors, Predictor, WeightedPredictorParams, WeightedPredictorState, pack_signed,
    unpack_signed,
};
pub use rct::{RctType, forward_rct, inverse_rct};
pub use section::GlobalModularState;
pub use token::Token;
pub use tree::{
    PixelProperties, Property, PropertyDecisionNode, Tree, TreeToken,
    adaptive_gradient_weighted_tree, collect_tree_tokens, count_contexts, gradient_tree,
    simple_tree, traverse_tree, weighted_tree,
};
