//! VarDCT (lossy) encoding module.
//!
//! This module provides the components for JPEG XL lossy encoding using
//! variable-size DCT transforms.

pub mod ac_strategy;
pub mod context;
pub mod enc_coeff;
pub mod quant_weights;
pub mod quantizer;
pub mod tokenize;

pub use ac_strategy::AcStrategy;
pub use context::BlockContextMap;
pub use enc_coeff::{pack_signed, quantize_block_8x8, quantize_block_ac, unpack_signed};
pub use quant_weights::{DequantMatrices, INV_LF_QUANT, LF_QUANT, LfQuantFactors, QuantTable};
pub use quantizer::{Quantizer, QuantizerParams};
pub use tokenize::{Token, TokenCollector};
