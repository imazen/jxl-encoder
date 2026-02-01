// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Entropy coding for JPEG XL encoder.
//!
//! This module provides ANS (Asymmetric Numeral Systems) and Huffman
//! encoding implementations for compressing symbols in the JXL bitstream.

pub mod ans;
pub mod cluster;
pub mod context_map;
pub mod histogram;
pub mod huffman;
pub mod huffman_tree;
pub mod hybrid_uint;

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
    HISTOGRAM_ROUNDING, Histogram, MIN_DISTANCE_FOR_DISTINCT, histogram_distance,
    histogram_kl_divergence,
};
pub use huffman::HuffmanEncoder;
pub use huffman_tree::{
    HuffmanTable, build_and_store_huffman_tree, convert_bit_depths_to_symbols, create_huffman_tree,
    store_huffman_tree, write_huffman_tree,
};
