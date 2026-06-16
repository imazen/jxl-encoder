# jxl-encoder

[![crates.io](https://img.shields.io/crates/v/jxl-encoder.svg)](https://crates.io/crates/jxl-encoder)
[![docs.rs](https://docs.rs/jxl-encoder/badge.svg)](https://docs.rs/jxl-encoder)
[![CI](https://github.com/imazen/jxl-encoder/actions/workflows/ci.yml/badge.svg)](https://github.com/imazen/jxl-encoder/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/imazen/jxl-encoder/branch/main/graph/badge.svg)](https://codecov.io/gh/imazen/jxl-encoder)
[![MSRV](https://img.shields.io/badge/MSRV-1.89-blue.svg)](https://blog.rust-lang.org/)

Pure Rust JPEG XL encoder. Lossy (VarDCT) and lossless (Modular) encoding, verified against three independent decoders (jxl-rs, jxl-oxide, djxl). `#![forbid(unsafe_code)]`.

740+ tests passing.

## Install

```toml
[dependencies]
jxl-encoder = "0.3.1"
```

Or `cargo add jxl-encoder` to pull the latest release (the crates.io badge above
tracks the current version). MSRV is Rust 1.89. `std` is on by default; the crate
is `no_std + alloc` capable — disable default features for the `alloc`-only build
(`std` adds the `encode_to()` / `finish_to()` `Write`-target sinks).

## Quick start

```rust
use jxl_encoder::{LossyConfig, LosslessConfig, PixelLayout};

// Lossy — distance 1.0 is visually lossless
let jxl = LossyConfig::new(1.0)
    .encode(&pixels, width, height, PixelLayout::Rgb8)?;

// Lossless
let jxl = LosslessConfig::new()
    .encode(&pixels, width, height, PixelLayout::Rgb8)?;

// Full control — limits, metadata, cancellation
use jxl_encoder::Limits;
let jxl = LossyConfig::new(1.0)
    .with_ans(true)
    .with_gaborish(true)
    .encode_request(width, height, PixelLayout::Rgba8)
    .with_limits(&Limits::default())
    .encode(&pixels)?;
```

## Quality (distance) and effort

**Distance** is the butteraugli target passed to `LossyConfig::new(distance)`.
It is a *perceptual error budget*, so the scale runs opposite to a percent slider:

- **Lower distance = higher quality** (and larger files).
- Valid lossy range is **`0.0 < distance <= 25.0`**. A distance outside that range
  (or non-finite) is rejected at encode time as `EncodeError::InvalidInput`.
- **`1.0` is the visually-lossless anchor** — the libjxl default, indistinguishable
  from the source for most images. Go below `1.0` (e.g. `0.5`) for near-transparent
  quality; raise it (`2.0`, `4.0`, …) to trade quality for size.
- **`0.0` (mathematically lossless) is *not* accepted by `LossyConfig`** — use
  `LosslessConfig` for exact reconstruction instead.

**Effort** trades encode time for compression. `LossyConfig` and `LosslessConfig`
both **default to effort 7**; set it with `with_effort(level)`:

```rust
use jxl_encoder::{LossyConfig, LosslessConfig, PixelLayout};

// Slower, smaller (effort 9 = Viterbi LZ77, 4 butteraugli iterations)
let jxl = LossyConfig::new(1.0)
    .with_effort(9)
    .encode(&pixels, width, height, PixelLayout::Rgb8)?;

// Fast preview (effort 3 = DCT8 only, Huffman, no gaborish/patches)
let jxl = LosslessConfig::new()
    .with_effort(3)
    .encode(&pixels, width, height, PixelLayout::Rgb8)?;
```

Valid effort is **`1..=12`**. `1..=9` mirrors libjxl's `kFalcon..=kTortoise` ladder;
`10..=12` are this crate's extended search budgets (longer butteraugli / tree-learn
seeds, still 100 %-spec-valid bitstreams). Higher effort = slower, better compression.

## Cancellation

Encodes are cooperatively cancellable. Pass a stop token via
`EncodeRequest::with_stop(&dyn Stop)` — the encoder checks it periodically and
returns `EncodeError::Cancelled` if it fires. The `Stop` trait and the no-op
`Unstoppable` token are re-exported from `jxl_encoder` (originally from the
[`enough`](https://crates.io/crates/enough) crate):

```rust
use jxl_encoder::{LossyConfig, PixelLayout, Unstoppable};

// No-op token — zero cost, never cancels (same as not passing one):
let jxl = LossyConfig::new(1.0)
    .encode_request(width, height, PixelLayout::Rgb8)
    .with_stop(&Unstoppable)
    .encode(&pixels)?;
```

For a token you can actually trigger (e.g. from another thread, a timeout, or a
user "cancel" button), add [`almost-enough`](https://crates.io/crates/almost-enough)
and use its `Stopper` — clone it to share, then call `.cancel()`:

```toml
[dependencies]
almost-enough = "0.4.4"
```

```rust
use jxl_encoder::{LossyConfig, PixelLayout};
use almost_enough::Stopper;

let stop = Stopper::new();
let watcher = stop.clone();           // hand a clone to a watchdog / signal handler
// ... watcher.cancel() from elsewhere when the user aborts ...

let result = LossyConfig::new(1.0)
    .encode_request(width, height, PixelLayout::Rgb8)
    .with_stop(&stop)
    .encode(&pixels);
// If `cancel()` fired before the encode finished, `result` is `Err(e)` where
// `matches!(e.error(), EncodeError::Cancelled)` holds (see Errors below).
```

## Errors

`encode` returns `jxl_encoder::Result<Vec<u8>>` = `Result<Vec<u8>, whereat::At<EncodeError>>`.
The `At<…>` wrapper records a source location for logs (`format!("{e}")`); borrow the
inner error with `e.error()` (or own it with `e.decompose().0`) to match. `EncodeError`
is `#[non_exhaustive]`, so keep a wildcard arm:

```rust
use jxl_encoder::{LossyConfig, EncodeError, PixelLayout};

match LossyConfig::new(1.0).encode(&pixels, width, height, PixelLayout::Rgb8) {
    Ok(_jxl) => { /* encoded bytes */ }
    Err(e) => match e.error() {
        EncodeError::Cancelled => { /* a Stop token requested cancellation */ }
        EncodeError::LimitExceeded { message } => eprintln!("limit: {message}"),
        EncodeError::Oom(_) => eprintln!("out of memory"),
        EncodeError::InvalidInput { message }
        | EncodeError::InvalidConfig { message } => eprintln!("bad input/config: {message}"),
        other => eprintln!("encode failed: {other:?}"),
    },
}
```

## Pixel layouts

`Rgb8`, `Rgba8`, `Bgr8`, `Bgra8`, `Gray8`, `GrayAlpha8`, `Rgb16`, `Rgba16`, `Gray16`, `RgbLinearF32`.

Lossy encoding supports all layouts including alpha (VarDCT for RGB + modular for the alpha channel). Lossless supports RGB, RGBA, grayscale, and gray+alpha.

## What works

**Lossy (VarDCT)**: 19/27 AC strategies (all that libjxl evaluates through effort 7), ANS entropy coding, adaptive quantization, chroma-from-luma, gaborish, pixel-domain loss, butteraugli quantization loop, custom coefficient ordering, noise synthesis, error diffusion, EPF sharpness, JPEG re-encoding.

**Lossless (Modular)**: RCT (all 42 variants), ANS + Huffman, LZ77 (RLE + hash chain), histogram clustering, content-adaptive MA tree learning, palette transform, squeeze (Haar wavelet), 14/14 predictors including Weighted.

**Animation**: Both lossy and lossless, per-frame duration, loop count, frame crop detection.

**Input formats**: 8-bit sRGB, 16-bit sRGB, linear f32, grayscale, alpha. BGR/BGRA layouts.

**Lossy quality vs libjxl**: Within 3% of cjxl effort 5 at low distances (d <= 1.0). The gap widens to ~22-26% at higher distances due to missing cost model refinements (iterative rate control, full histogram clustering).

## Features

| Feature | Default | Description |
|---------|---------|-------------|
| `std` | yes | Standard library support; enables `encode_to()` for `Write` targets |
| `butteraugli-loop` | yes | Iterative quant field refinement via butteraugli distmap |
| `rate-control` | no | Iterative encode for precise distance targeting |
| `jpeg-reencoding` | no | JPEG bitstream re-encoding into JXL |
| `trace-bitstream` | no | Zero-cost bitstream tracing for debugging |

## Resource limits

`EncodeRequest::with_limits(&Limits)` bounds an encode against untrusted input.
`Limits` primarily caps **encoder working-set memory** (it also exposes optional
`max_width` / `max_height` / `max_pixels` / `max_quant_loop_iters` setters, all
`None` by default):

```rust
use jxl_encoder::{LossyConfig, Limits, PixelLayout};

let limits = Limits::default()              // no explicit caps set …
    .with_max_memory_bytes(512 * 1024 * 1024); // … 512 MB hard ceiling

let jxl = LossyConfig::new(1.0)
    .encode_request(width, height, PixelLayout::Rgb8)
    .with_limits(&limits)
    .encode(&pixels)?;
```

`Limits::default()` sets **no explicit** memory bound, but the encoder still
applies a *soft default cap* so an unconfigured image proxy can't be OOM'd:
**4 GiB for lossy**, **8 GiB for lossless** (lossless tree-learning is a heavier
memory regime). These defaults are fixed ceilings — they are deliberately **not**
scaled with image dimensions, so an oversized untrusted upload is still bounded.
For trusted batch work, raise the cap with `with_max_memory_bytes(n)` (or pass
`u64::MAX` to opt out of the soft cap entirely). If an encode exceeds the cap it
returns `EncodeError::LimitExceeded`.

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
