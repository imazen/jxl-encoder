# jxl-encoder-simd

[![crates.io](https://img.shields.io/crates/v/jxl-encoder-simd.svg)](https://crates.io/crates/jxl-encoder-simd)
[![docs.rs](https://docs.rs/jxl-encoder-simd/badge.svg)](https://docs.rs/jxl-encoder-simd)

SIMD-accelerated primitives for [jxl-encoder](https://crates.io/crates/jxl-encoder). Internal crate — you probably want `jxl-encoder` instead.

`#![no_std]`, `#![forbid(unsafe_code)]`.

Uses [archmage](https://crates.io/crates/archmage) for portable SIMD dispatch across x86-64 (AVX2) and aarch64 (NEON) with scalar fallback.

## What's inside

DCT/IDCT (8x8, 16x16), quantization, dequantization, XYB color transform, gaborish pre-filter, edge-preserving filter (EPF), adaptive quantization masking, entropy estimation, pixel-domain loss computation.

## License

AGPL-3.0-or-later. Commercial licenses at [imazen.io/pricing](https://www.imazen.io/pricing).

Algorithms and constants derived from [libjxl](https://github.com/libjxl/libjxl) (BSD-3-Clause).
