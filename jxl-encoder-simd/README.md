# jxl-encoder-simd

[![CI](https://github.com/imazen/jxl-encoder/actions/workflows/ci.yml/badge.svg)](https://github.com/imazen/jxl-encoder/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/jxl-encoder-simd.svg)](https://crates.io/crates/jxl-encoder-simd)
[![docs.rs](https://docs.rs/jxl-encoder-simd/badge.svg)](https://docs.rs/jxl-encoder-simd)
[![MSRV](https://img.shields.io/badge/MSRV-1.89-blue.svg)](https://blog.rust-lang.org/)

SIMD-accelerated primitives for [jxl-encoder](https://crates.io/crates/jxl-encoder). Internal crate — you probably want `jxl-encoder` instead.

`#![no_std]`, `#![forbid(unsafe_code)]`.

Uses [archmage](https://crates.io/crates/archmage) for portable SIMD dispatch across x86-64 (AVX2) and aarch64 (NEON) with scalar fallback.

## What's inside

DCT/IDCT (8x8, 16x16), quantization, dequantization, XYB color transform, gaborish pre-filter, edge-preserving filter (EPF), adaptive quantization masking, entropy estimation, pixel-domain loss computation.

## License

Dual-licensed: [AGPL-3.0](LICENSE-AGPL3) or [commercial](LICENSE-COMMERCIAL).

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

See [LICENSE-COMMERCIAL](LICENSE-COMMERCIAL) for details.

### Upstream

Upstream code from [libjxl/libjxl](https://github.com/libjxl/libjxl) is licensed under BSD-3-Clause.

Our additions and improvements are dual-licensed (AGPL-3.0 or commercial) as above.

Algorithms and constants derived from [libjxl](https://github.com/libjxl/libjxl) (BSD-3-Clause).
