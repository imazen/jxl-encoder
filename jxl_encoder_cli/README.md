# cjxl-rs

[![crates.io](https://img.shields.io/crates/v/jxl-encoder-cli.svg)](https://crates.io/crates/jxl-encoder-cli)
[![CI](https://github.com/imazen/jxl-encoder/actions/workflows/ci.yml/badge.svg)](https://github.com/imazen/jxl-encoder/actions/workflows/ci.yml)
[![MSRV](https://img.shields.io/badge/MSRV-1.89-blue.svg)](https://blog.rust-lang.org/)

Command-line JPEG XL encoder. Pure Rust, `#![forbid(unsafe_code)]`.

Wraps [jxl-encoder](https://crates.io/crates/jxl-encoder) — see that crate for library usage.

## Install

```bash
cargo install jxl-encoder-cli
```

## Usage

```bash
# Lossy (distance 1.0 = visually lossless)
cjxl-rs input.png output.jxl -d 1.0

# Lossless
cjxl-rs input.png output.jxl --lossless

# Quality mode (0-100, maps to distance internally)
cjxl-rs input.png output.jxl -q 90
```

Reads PNG (including APNG animation). Writes bare JXL codestream or container format (when metadata or animation is present).

## Flags

| Flag | Default | Description |
|------|---------|-------------|
| `-d, --distance` | 1.0 | Butteraugli distance (0 = lossless, 1.0 = visually lossless) |
| `-q, --quality` | 90 | Quality 0-100 (alternative to distance) |
| `-e, --effort` | 7 | Effort 1-10 (higher = slower, better compression) |
| `--lossless` | off | Lossless modular encoding |
| `--no-gaborish` | on | Disable gaborish pre-filter |
| `--no-pixel-domain-loss` | on | Disable pixel-domain loss (faster, lower quality) |
| `--no-ans` | ANS on | Use Huffman instead of ANS |
| `--no-optimize-codes` | on | Single-pass static Huffman (streaming) |
| `--dct8-only` | off | Force DCT8 only |
| `--noise` | off | Enable noise synthesis |
| `--denoise` | off | Wiener denoising pre-filter (implies --noise) |
| `--no-error-diffusion` | on | Disable error diffusion |
| `--lz77` | off | Enable LZ77 backward references (ANS two-pass only) |
| `--tree-learning` | off | Content-adaptive MA tree for lossless |
| `--squeeze` | off | Haar wavelet transform for lossless |
| `--no-butteraugli` | on | Disable butteraugli quantization loop |
| `--icc FILE` | none | Embed ICC profile |
| `--exif FILE` | none | Embed EXIF metadata |
| `--xmp FILE` | none | Embed XMP metadata |

## License

AGPL-3.0-or-later. Commercial licenses at [imazen.io/pricing](https://www.imazen.io/pricing).
