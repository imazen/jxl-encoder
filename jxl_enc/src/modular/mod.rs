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
pub mod minimal;
pub mod predictor;
pub mod token;
pub mod tree;

pub use channel::{Channel, ModularImage};
pub use encoder::{EncodedModularData, ModularEncoder, ModularEncoderOptions};
pub use predictor::Predictor;
pub use token::Token;
pub use tree::{PropertyDecisionNode, Tree};
