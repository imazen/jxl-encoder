//! Encoder heuristics for quality optimization.
//!
//! This module contains algorithms for making encoding decisions that
//! don't affect bitstream correctness but do affect quality and compression.

pub mod ac_strategy;
pub mod chroma_from_luma;

pub use ac_strategy::{AcStrategyMap, HeuristicLevel, select_ac_strategies};
pub use chroma_from_luma::{ColorCorrelationMap, apply_cfl_decorrelation};
