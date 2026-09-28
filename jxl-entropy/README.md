# jxl-entropy

[![crates.io](https://img.shields.io/crates/v/jxl-entropy.svg)](https://crates.io/crates/jxl-entropy)
[![docs.rs](https://docs.rs/jxl-entropy/badge.svg)](https://docs.rs/jxl-entropy)

JPEG XL entropy-coding primitives, extracted from
[jxl-encoder](https://crates.io/crates/jxl-encoder): a `BitWriter`,
hybrid-uint configs, Huffman/ANS writers, context-map clustering, LZ77
match finding, and the token/histogram plumbing that connects them.

Useful if you're building a JXL-compatible codec or want these pieces
for a custom bitstream — but most users want
[jxl-encoder](https://crates.io/crates/jxl-encoder) directly.

## Features

- `std` *(default)* — std-enabled error/IO paths
- `parallel` — rayon-backed parallel histogram/tree work
- `jpeg-reencoding` — JPEG-transcode coding paths
- `trace-bitstream`, `debug-tokens`, `debug-rect`, `profile-phases` —
  encoder-side instrumentation
