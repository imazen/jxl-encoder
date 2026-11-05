# jxl-encoder-simd

[![CI](https://github.com/imazen/jxl-encoder/actions/workflows/ci.yml/badge.svg)](https://github.com/imazen/jxl-encoder/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/jxl-encoder-simd.svg)](https://crates.io/crates/jxl-encoder-simd)
[![docs.rs](https://docs.rs/jxl-encoder-simd/badge.svg)](https://docs.rs/jxl-encoder-simd)
[![MSRV](https://img.shields.io/badge/MSRV-1.93-blue.svg)](https://blog.rust-lang.org/)

SIMD-accelerated primitives for [jxl-encoder](https://crates.io/crates/jxl-encoder). Internal crate — you probably want `jxl-encoder` instead.

`#![no_std]`, `#![forbid(unsafe_code)]`.

Uses [archmage](https://crates.io/crates/archmage) and magetypes for runtime SIMD dispatch: x86-64 (AVX2 + FMA, AVX-512 for a few kernels), aarch64 (NEON) and wasm32 (SIMD128), with a scalar fallback.

## What's inside

Forward DCTs (4 to 64) and inverse DCTs (16 to 64), quantization, dequantization, XYB color transform, gaborish pre-filter, edge-preserving filter (EPF), adaptive quantization masking, chroma-from-luma, noise, entropy estimation, pixel-domain loss.

## License

Dual-licensed: [AGPL-3.0](https://github.com/imazen/jxl-encoder/blob/main/LICENSE-AGPL3) or [commercial](https://github.com/imazen/jxl-encoder/blob/main/LICENSE-COMMERCIAL).

I've maintained and developed open-source image server software — and the 40+
library ecosystem it depends on — full-time since 2011. Fifteen years of
continual maintenance, backwards compatibility, support, and the (very rare)
security patch. That kind of stability requires sustainable funding, and
dual-licensing is how we make it work without venture capital or rug-pulls.
Support sustainable and secure software; swap patch tuesday for patch leap-year.

[Our open-source products](https://www.imazen.io/open-source)

**Your options:**

- **Startup license** — $1 if your company has under $1M revenue and fewer
  than 5 employees. [Get a key →](https://www.imazen.io/pricing)
- **Commercial subscription** — Governed by the Imazen Site-wide Subscription
  License v1.1 or later. Apache 2.0-like terms, no source-sharing requirement.
  Sliding scale by company size.
  [Pricing & 60-day free trial →](https://www.imazen.io/pricing)
- **AGPL v3** — Free and open. Share your source if you distribute.

See [LICENSE-COMMERCIAL](https://github.com/imazen/jxl-encoder/blob/main/LICENSE-COMMERCIAL) for details.

### Upstream

Upstream code from [libjxl/libjxl](https://github.com/libjxl/libjxl) is licensed under BSD-3-Clause.

Our additions and improvements are dual-licensed (AGPL-3.0 or commercial) as above.

Algorithms and constants derived from [libjxl](https://github.com/libjxl/libjxl) (BSD-3-Clause).

## Calling from feature-enabled code

Ordinary callers use the crate-root runtime-dispatched functions. Code already
compiled with the full CPU feature set can call `target_features` functions with
plain arrays and slices, such as `target_features::dct_8x8_v3(input, output)`.
Rust checks the calling function's `#[target_feature]` attributes. A native baseline
caller gets E0133 and should use the runtime-dispatched entry point.

WASM also permits safe baseline calls, with module features validated by the
engine at load time; SIMD128 requires engine support.

The module documents all required features and provides safe composition
examples. Its `_v3`, `_v4`, `_neon` and `_wasm128` functions call the existing
kernels. The original token-taking functions remain available and are deprecated.
Existing scalar functions continue to work without tokens. There is no new speed
or inlining guarantee. The AVX-512 validation uses magetypes 0.9.31-beta.2, which
provides narrow-vector traits absent from 0.9.30.
