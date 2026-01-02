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
pub mod predictor;
pub mod rct;
pub mod token;
pub mod tree;

pub use channel::{Channel, ModularImage};
pub use encoder::{EncodedModularData, ModularEncoder, ModularEncoderOptions};
pub use improved::write_modular_stream_with_rct;
pub use predictor::Predictor;
pub use rct::{RctType, forward_rct, inverse_rct};
pub use token::Token;
pub use tree::{PropertyDecisionNode, Tree};
