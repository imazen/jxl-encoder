// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! VarDCT entropy code types.
//!
//! Contains `BuiltEntropyCode` enum (Huffman/ANS) and the `force_strategy_map` helper.

use super::ac_strategy::AcStrategyMap;
use crate::bit_writer::BitWriter;
use crate::entropy_coding::encode::{
    OwnedAnsEntropyCode, OwnedEntropyCode, write_entropy_code, write_entropy_code_ans,
    write_tokens, write_tokens_ans,
};
use crate::entropy_coding::token::Token;
use crate::error::Result;

/// Create an AC strategy map forcing a specific strategy.
pub(crate) fn force_strategy_map(
    xsize_blocks: usize,
    ysize_blocks: usize,
    raw_strategy: u8,
) -> AcStrategyMap {
    AcStrategyMap::force_strategy(xsize_blocks, ysize_blocks, raw_strategy)
}

/// Entropy code that holds either Huffman or ANS code.
pub enum BuiltEntropyCode<'a> {
    /// Static Huffman prefix codes (borrowed).
    StaticHuffman(crate::entropy_coding::encode::EntropyCode<'a>),
    /// Dynamic Huffman prefix codes (owned).
    Huffman(OwnedEntropyCode),
    /// ANS distributions with context map.
    Ans(OwnedAnsEntropyCode),
}

impl<'a> BuiltEntropyCode<'a> {
    /// Write the entropy code header (context map + codes/distributions).
    pub fn write_header(&self, writer: &mut BitWriter) -> Result<()> {
        match self {
            BuiltEntropyCode::StaticHuffman(code) => write_entropy_code(code, writer),
            BuiltEntropyCode::Huffman(code) => code.write_header(writer),
            BuiltEntropyCode::Ans(code) => write_entropy_code_ans(code, writer),
        }
    }

    /// Write tokens using this entropy code.
    pub fn write_tokens(
        &self,
        tokens: &[Token],
        lz77: Option<&crate::entropy_coding::lz77::Lz77Params>,
        writer: &mut BitWriter,
    ) -> Result<()> {
        match self {
            BuiltEntropyCode::StaticHuffman(code) => write_tokens(tokens, code, lz77, writer),
            BuiltEntropyCode::Huffman(code) => code.write_tokens_owned(tokens, lz77, writer),
            BuiltEntropyCode::Ans(code) => write_tokens_ans(tokens, code, lz77, writer),
        }
    }

    /// Get the underlying Huffman code for streaming token writing.
    ///
    /// Panics if this is an ANS code (streaming with ANS is not supported).
    pub fn as_huffman(&self) -> crate::entropy_coding::encode::EntropyCode<'_> {
        match self {
            BuiltEntropyCode::StaticHuffman(code) => *code,
            BuiltEntropyCode::Huffman(code) => code.as_entropy_code(),
            BuiltEntropyCode::Ans(_) => {
                panic!("ANS codes cannot be used with streaming encoder")
            }
        }
    }

    #[allow(dead_code)]
    /// Returns the number of contexts in this entropy code.
    pub fn num_contexts(&self) -> usize {
        match self {
            BuiltEntropyCode::StaticHuffman(code) => code.num_contexts,
            BuiltEntropyCode::Huffman(code) => code.context_map.len(),
            BuiltEntropyCode::Ans(code) => code.context_map.len(),
        }
    }

    #[allow(dead_code)]
    /// Returns the number of histograms/prefix codes in this entropy code.
    pub fn num_histograms(&self) -> usize {
        match self {
            BuiltEntropyCode::StaticHuffman(code) => code.num_prefix_codes,
            BuiltEntropyCode::Huffman(code) => code.prefix_codes.len(),
            BuiltEntropyCode::Ans(code) => code.histograms.len(),
        }
    }
}
