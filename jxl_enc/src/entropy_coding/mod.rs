// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Entropy coding for JPEG XL encoder.
//!
//! This module provides ANS (Asymmetric Numeral Systems) and Huffman
//! encoding implementations for compressing symbols in the JXL bitstream.

pub mod ans;
pub mod huffman;
pub mod hybrid_uint;

pub use ans::AnsEncoder;
pub use huffman::HuffmanEncoder;
