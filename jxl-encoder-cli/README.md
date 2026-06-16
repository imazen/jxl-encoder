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

Notes:
- The output file is **overwritten silently** if it already exists — there is no `--force` / overwrite prompt.
- If both `-d` and `-q` are given, `-d` wins (`-q` is only the fallback when `-d` is absent); they are not mutually exclusive.
- `--lossless` and `-d 0` are equivalent — both select the modular path for true mathematical lossless. Either works.

## Flags

| Flag | Default | Description |
|------|---------|-------------|
| `-d, --distance` | 1.0 | Butteraugli distance (0 = lossless, 1.0 = visually lossless) |
| `-q, --quality` | 90 | Quality 0-100 (alternative to distance) |
| `-e, --effort` | 7 | Effort 1-12 (higher = slower, better compression; e10/e11/e12 extends libjxl kTortoise=9 — e12 = 32 butteraugli iters) |
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
| `--buffering MODE` | -1 (auto) | Input/output buffering policy. `-1`=auto (≤ 2048² full-buffer, larger stream-input+buffered-output), `0`=full buffer, `1`=threshold 2048², `2`=stream input + buffered output, `3`=stream input + stream output. Scaffolding only in chunk 1 — no dispatch wired yet; output bytes are identical regardless of value. |

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
