//! Encoder heuristics for quality optimization.
//!
//! This module contains algorithms for making encoding decisions that
//! don't affect bitstream correctness but do affect quality and compression.

pub mod ac_strategy;

pub use ac_strategy::{AcStrategyMap, HeuristicLevel, select_ac_strategies};
