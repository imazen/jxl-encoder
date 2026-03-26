// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Entropy coding for JPEG XL encoder.
//!
//! This module provides ANS (Asymmetric Numeral Systems) and Huffman
//! encoding implementations for compressing symbols in the JXL bitstream.

pub mod ans;
pub mod ans_decode;
pub(crate) mod cluster;
pub(crate) mod context_map;
pub(crate) mod encode;
mod encode_ans;
mod encode_huffman;
pub mod histogram;
pub(crate) mod huffman_tree;
pub(crate) mod hybrid_uint;
pub(crate) mod lz77;
pub(crate) mod token;

pub use ans::{
    ANS_LOG_TAB_SIZE, ANS_MAX_ALPHABET_SIZE, ANS_SIGNATURE, ANS_TAB_MASK, ANS_TAB_SIZE,
    ANSEncodingHistogram, ANSHistogramStrategy, AnsEncoder, get_population_count_precision,
};
pub use cluster::{
    ClusterResult, ClusteringType, EntropyType, cluster_histograms, fast_cluster_histograms,
};
pub use context_map::{
    encode_context_map, inverse_move_to_front_transform, move_to_front_transform,
};
pub use histogram::{
    DistanceScratch, HISTOGRAM_ROUNDING, Histogram, MIN_DISTANCE_FOR_DISTINCT, histogram_distance,
    histogram_distance_reuse, histogram_kl_divergence,
};
pub use huffman_tree::{
    HuffmanTable, build_and_store_huffman_tree, convert_bit_depths_to_symbols, create_huffman_tree,
    store_huffman_tree, write_huffman_tree,
};
pub use lz77::Lz77Method;
