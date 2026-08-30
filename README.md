# jxl-encoder [![CI](https://img.shields.io/github/actions/workflow/status/imazen/jxl-encoder/ci.yml?style=flat-square&label=CI)](https://github.com/imazen/jxl-encoder/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/jxl-encoder?style=flat-square)](https://crates.io/crates/jxl-encoder) [![lib.rs](https://img.shields.io/crates/v/jxl-encoder?style=flat-square&label=lib.rs&color=blue)](https://lib.rs/crates/jxl-encoder) [![docs.rs](https://img.shields.io/docsrs/jxl-encoder?style=flat-square)](https://docs.rs/jxl-encoder) [![license](https://img.shields.io/badge/license-AGPL--3.0%20%2F%20Commercial-blue?style=flat-square)](#license) ![MSRV](https://img.shields.io/badge/MSRV-1.89-blue?style=flat-square)

jxl-encoder is a pure-Rust [JPEG XL](https://jpeg.org/jpegxl/) encoder for both
lossy (VarDCT) and lossless (Modular) images, built on the foundation of
[libjxl](https://github.com/libjxl/libjxl) and targeting parity with it.
`#![forbid(unsafe_code)]` by default; `no_std + alloc` capable (disable default
features for the `alloc`-only build — `std` adds the `encode_to()` / `finish_to()`
`Write`-target sinks).

The reference encoder, cjxl (libjxl), is a well-engineered, mature C++ codebase.
This crate exists because we wanted a pure-Rust encoder we could embed in
[Imageflow](https://github.com/imazen/imageflow) with no C FFI, and a place to
experiment with content-aware dispatch and alternative perceptual metrics. Our
algorithms, quantization weights, cost models, and the bitstream format itself
are derived from libjxl. Every encoded file in our test suite is verified against
three independent decoders: [jxl-rs](https://github.com/libjxl/jxl-rs),
[jxl-oxide](https://github.com/tirr-c/jxl-oxide), and djxl (libjxl).

On a strict per-cell Pareto scoreboard cjxl still wins more cells overall and is
measurably faster; this crate leads on 8-bit lossless and a fair slice of SDR
lossy, and is pure-Rust and embeddable. The honest, measured breakdown is in
[the benchmark index](https://github.com/imazen/jxl-encoder/blob/main/benchmarks/README.md)
(and the scoreboard tables below on GitHub).

<!-- crates.io:skip-start -->
## Where it stands vs cjxl (the honest version)

cjxl is the reference, and on a per-scenario basis it still wins more cells than
we do today, and it is measurably faster. We track this with a strict per-cell
Pareto scoreboard (`docs/GOAL_BEAT_CJXL.md`): a cell is "ours" only when we are
no worse on bytes, no worse on perceptual quality, and within the wall budget.
The numbers below are what that scoreboard reports — not aspirations.

**Bytes + quality**, 280 cells across SDR lossy, SDR lossless, HDR lossy, and a
fixed-overhead size axis (measured 2026-06-12, binary `3f025244`; cjxl =
libjxl v0.12.0; reproduce with `scripts/scoreboard/run_scoreboard.py`,
see [`benchmarks/README.md`](benchmarks/README.md)):

| Verdict | Cells | Share |
|---|---|---|
| cjxl dominates | 105 | 38 % |
| mixed (we win one axis, lose another) | 89 | 32 % |
| we dominate | 78 | 28 % |
| tie | 8 | 3 % |

By content family (we-dominate / tie / mixed / cjxl-dominates), from
`benchmarks/scoreboard/scoreboard_2026-06-12_run4_summary.md`:

| Family | We | Tie | Mixed | cjxl |
|---|---|---|---|---|
| SDR lossless | 30 | 4 | 0 | 22 |
| SDR lossy | 33 | 0 | 47 | 24 |
| HDR lossy | 10 | 1 | 30 | 55 |
| Size axis (64²/256²) | 5 | 3 | 12 | 4 |

We lead on lossless graphics and photos at e7, and on a good chunk of SDR lossy
content; HDR lossy is where cjxl wins most cells (small per-cell byte gaps at
tied quality — the per-cell strictness counts every one). The smooth-gradient /
HDR sky class is the historical loss locus and an open wedge.

**Wall time is the weak axis, and there is no way to spin it.** On a 40-cell
quiet-box grid (5 strata × e{5,7} × {1,8} threads × {lossy, lossless}, measured
2026-06-12, binary `a5a9e4d6`, `benchmarks/scoreboard/wall_grid_2026-06-12.tsv`),
**39 of 40 cells are over the ≤1.2× budget** — cjxl is faster on all but one
(plots lossy e7 1T, 1.12×):

| Mode | 1 thread | 8 threads |
|---|---|---|
| lossy | 1.12–2.17× cjxl | 1.57–3.76× cjxl |
| lossless | 1.33–5.90× cjxl | 2.89–10.78× cjxl |

Single-thread lossy is roughly competitive; everything else is slower, and the
8T gap is the largest. cjxl scales ~4.6–5.5× from 1T→8T while we scale ~1.8–2.5×
— that's both a serial-speed gap and a parallel-coverage gap (our AC-tile search
is parallel, but XYB / adaptive-quant / gaborish / transform / tokenize are
still serial, and cjxl parallelizes per 256² group). Closing this is the active
workstream.

### Lossless, in more detail

8-bit lossless is where we are strongest. With the e5/e6 budgeted tree-learn
lift (`benchmarks/lossless_8bit_tree_lift_2026-06-12.tsv`, 43 imazen-26 picks,
djxl-verified pixel-exact on all 56 cells):

- **e5**: vs cjxl mean −2.4 %, median −0.1 % bytes, 20/43 cells smaller (worst
  +36.9 % on a noaa-documents scan).
- **e6**: vs cjxl mean −10.7 %, median −7.0 % bytes, 11/13 cells smaller.

16-bit lossless at low effort (e2/e4) still loses to cjxl on bytes, and the
lossless wall gap is the widest of any mode (see the grid above).

### HDR lossy, in more detail

On 12 PQ/HLG crops × e{5,7} × d{0.5,1,2,4}
(`benchmarks/hdr_lossy_parity_postdispatch_2026-06-12.tsv`, PQ-EOTF butteraugli
@ 1000 nits), median bytes run **+1.2 % to +4.6 % over cjxl** at quality
at-or-better than cjxl on 7 of 8 measured points. The QuantizeWP DC-shaping
dispatch (keyed on the resolved transfer function) closed roughly half the
median HDR byte gap; the smooth-sky residual remains.
<!-- crates.io:skip-end -->

## Quick start

```toml
[dependencies]
jxl-encoder = "0.3.1"
```

```rust
use jxl_encoder::{LossyConfig, LosslessConfig, PixelLayout};

// Lossy — distance 1.0 is visually lossless (lower distance = higher quality).
let jxl = LossyConfig::new(1.0)
    .encode(&pixels, width, height, PixelLayout::Rgb8)?;

// Lossless — exact reconstruction.
let jxl = LosslessConfig::new()
    .encode(&pixels, width, height, PixelLayout::Rgb8)?;

// Full control — per-knob overrides, then a request with limits / cancellation.
use jxl_encoder::Limits;
let jxl = LossyConfig::new(1.0)
    .with_ans(true)
    .with_gaborish(true)
    .encode_request(width, height, PixelLayout::Rgba8)
    .with_limits(&Limits::default())
    .encode(&pixels)?;
```

`std` is on by default; `cargo add jxl-encoder` pulls the latest release. MSRV is
Rust 1.89.

## Quality (distance) and effort

**Distance** is the butteraugli target passed to `LossyConfig::new(distance)`.
It is a *perceptual error budget*, so the scale runs opposite to a percent slider:

- **Lower distance = higher quality** (and larger files).
- Valid lossy range is **`0.0 < distance <= 25.0`**.
- **`1.0` is the visually-lossless anchor** — the libjxl default, indistinguishable
  from the source for most images. Go below `1.0` (e.g. `0.5`) for near-transparent
  quality; raise it (`2.0`, `4.0`, …) to trade quality for size.
- **`0.0` (mathematically lossless) is *not* accepted by `LossyConfig`** — use
  `LosslessConfig` for exact reconstruction instead.

**Effort** trades encode time for compression. `LossyConfig` and `LosslessConfig`
both **default to effort 7**; set it with `with_effort(level)`:

```rust
use jxl_encoder::{LossyConfig, LosslessConfig, PixelLayout};

// Slower, smaller (effort 9 = Viterbi LZ77, 4 butteraugli iterations).
let jxl = LossyConfig::new(1.0)
    .with_effort(9)
    .encode(&pixels, width, height, PixelLayout::Rgb8)?;

// Fast preview.
let jxl = LosslessConfig::new()
    .with_effort(3)
    .encode(&pixels, width, height, PixelLayout::Rgb8)?;
```

Valid effort is **`1..=13`**. `1..=9` mirrors libjxl's `kFalcon..=kTortoise` ladder.
**e10** aligns with libjxl e10 (kGlacier) as a superset — finer non-aligned
AC-strategy search, a lower MA-tree split threshold (75 vs 89) for lossless,
and the iterative 2× downsampler when `with_resampling(2)` is set. **`11..=13`**
are this crate's extended search budgets (8/16/32 butteraugli iterations,
multi-seed butteraugli + tree-learn search), and lossless e11+ supersets libjxl
e11 with the TectonicPlate per-image config trial (~22 whole-frame modular
configurations, smallest wins) — still 100 %-spec-valid bitstreams. Effort
numbering above 9 shifted by +1 on 2026-08-29 to align with libjxl's new
e10/e11 tiers: pre-shift e10/e11/e12 behaviours now live at e11/e12/e13 (see
issue #45). Higher effort = slower, better compression.

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

Integer and HDR/float layouts are accepted directly; the encoder converts to XYB
(lossy) or the modular integer space (lossless) internally:

`Rgb8`, `Rgba8`, `Bgr8`, `Bgra8`, `Gray8`, `GrayAlpha8`, `Rgb16`, `Rgba16`,
`Gray16`, `GrayAlpha16`, `RgbLinearF32`, `RgbaLinearF32`, `GrayLinearF32`,
`GrayAlphaLinearF32`, `RgbLinearF16`, `RgbaLinearF16`, `GrayLinearF16`,
`GrayAlphaLinearF16`, `RgbPqF32`, `RgbaPqF32`, `RgbHlgF32`, `RgbaHlgF32`,
`RgbBt709F32`, `RgbaBt709F32`, `Cmyk8`, `Cmyk16`.

Lossy encoding supports all layouts including alpha (VarDCT for RGB + modular for
the alpha channel). Lossless supports RGB, RGBA, grayscale, and gray+alpha. See
[HDR / wide-gamut](#hdr--wide-gamut) for the PQ / HLG / BT.709 transfer-function
variants.

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
returns `EncodeError::LimitExceeded`. `LossyConfig::estimate_peak_memory_bytes`
(and the `LosslessConfig` equivalent) let callers plan a budget up front.

## HDR / wide-gamut

| Capability | Entry point |
|---|---|
| PQ / HLG / BT.709 f32 input | Pass the matching `PixelLayout` variant (e.g. `RgbPqF32`, `RgbHlgF32`); the encoder inverts the transfer function before XYB. |
| BT.2100 PQ / HLG colour encoding | `ColorEncoding::bt2100_pq()` / `bt2100_hlg()`, via `EncodeRequest::with_color_encoding(...)`. |
| `intensity_target` / `min_nits` | `EncodeRequest::with_intensity_target(nits)` / `with_min_nits(nits)`. |
| HDR-aware perceptual loss in the quant loop | `LossyConfig::with_hdr_loss(HdrLoss::Auto)` — auto-dispatches to a VDP2 path on PQ/HLG content, butteraugli elsewhere. SDR encodes stay byte-identical. Requires the `butteraugli-loop` feature. |

Measured HDR bytes/quality vs cjxl are in
[the benchmark index](https://github.com/imazen/jxl-encoder/blob/main/benchmarks/README.md).

## Feature coverage vs libjxl

We implement all 19 AC strategies that libjxl evaluates through effort 9, all
enabled. The remaining 8 are either commented out in libjxl (DCT32x8, DCT8x32)
or experimental/unused (DCT128+) — cjxl never selects them either. Effort 9 adds
fine-grained strategy search (step=1 for 32×32+ blocks).

### Lossy (VarDCT)

| Feature | libjxl e5 | libjxl e7 | jxl-encoder |
|---------|-----------|-----------|-------------|
| AC strategies | 7 | 19 | 19 |
| ANS entropy coding (default-on) | Yes | Yes | Yes |
| Adaptive quantization | Yes | Yes | Yes |
| Pixel-domain loss (default-on) | Yes | Yes | Yes |
| Chroma-from-luma (per-tile least-squares) | Yes | Yes | Yes |
| Gaborish inverse pre-filter (default-on) | Yes | Yes | Yes |
| Custom coefficient ordering (default-on) | Yes | Yes | Yes |
| Butteraugli quant loop (effort 8+) | Yes | Yes | Yes (2 iters at e8, 4 at e9+) |
| EPF per-block sharpness | Yes | Yes | Yes |
| Content-adaptive block context map | Yes | Yes | Yes |
| Error diffusion in AC quantization | No | No | Yes (opt-in) |
| Noise synthesis | Yes | Yes | Yes (opt-in) |
| Lossy + alpha (VarDCT RGB + modular alpha) | Yes | Yes | Yes |
| JPEG transcode (byte-exact re-encode) | Yes | Yes | Yes (opt-in feature) |
| Animation (lossy + lossless) | Yes | Yes | Yes |
| 16-bit / float input | Yes | Yes | Yes (26 pixel layouts) |
| Patches / dictionary (default-on for screenshots) | No | Yes | Yes |
| Fine-grained AC strategy search | Yes | Yes | Yes (effort 9+) |
| Splines | No | Yes | Yes (opt-in API) |
| Dots detection | No | Yes | Yes (opt-in) |
| Progressive VarDCT (2-pass / 3-pass) | Yes | Yes | Yes |
| Photon-noise simulation | Yes | Yes | Yes (`with_photon_noise_iso`) |
| Forced RCT colorspace (lossless) | Yes | Yes | Yes (`with_force_rct`) |
| Peak-memory estimate helper | No | No | Yes (`estimate_peak_memory_bytes`) |

### Lossless (Modular)

| Feature | libjxl | jxl-encoder |
|---------|--------|-------------|
| RCT (all 42 variants) | Yes | Yes |
| ANS entropy coding (default-on) | Yes | Yes |
| Huffman entropy coding (fallback) | Yes | Yes |
| LZ77 RLE / greedy / optimal Viterbi DP | Yes | Yes (default-on at e7 / e8 / e9+) |
| MA tree learning (14 predictors, 16 properties) | Yes | Yes |
| Weighted predictor | Yes | Yes (bit-exact match) |
| Palette transform (auto-detect) | Yes | Yes |
| Squeeze transform (Haar wavelet) | Yes | Yes |
| Histogram clustering | Full (kDefault) | Pair-merge refinement |
| Multi-group encoding (any image size) | Yes | Yes |
| RGBA / grayscale / alpha | Yes | Yes |
| Lossy palette / delta palette | Yes | Yes (opt-in) |
| 16-bit / float input | Yes | Yes |
| Best/Variable predictors (effort 8+) | Yes | Tree learning is the Variable-mode equivalent |

### Container / metadata

| Feature | libjxl | jxl-encoder |
|---------|--------|-------------|
| ICC profile embedding (PredictICC + entropy coded) | Yes | Yes |
| EXIF / XMP metadata (container box) | Yes | Yes |
| Animation (per-frame duration) | Yes | Yes |
| Multi-group framing (>256×256) | Yes | Yes |
| Cancellation / resource limits | No | Yes (`&dyn Stop`, `Limits`) |

### Honest gaps

| Feature | libjxl | Notes |
|---------|--------|-------|
| Streaming frame encoding | Yes | Current impl buffers the full image; `estimate_peak_memory_bytes` lets callers plan around this. |
| `ec_distance` (per-extra-channel quality) | Yes | Lossy alpha is currently encoded as a lossless modular extra channel. |
| `decoding_speed_tier` | Yes | We expose individual gating knobs that approximate the major effects. |
| Wall-time parity | — | cjxl is faster on 39/40 measured cells; see the scoreboard on GitHub. |

<!-- crates.io:skip-start -->
### AC strategy coverage

| Strategy | Block | Min distance | libjxl effort |
|----------|-------|-------------|---------------|
| DCT8 | 8×8 | any | e1+ |
| DCT4x4 | 8×8 (4 sub) | any | e5+ |
| DCT4x8, DCT8x4 | 8×8 (2 sub) | any | e6+ |
| IDENTITY | 8×8 (pixel domain) | any | e5+ |
| DCT2x2 | 8×8 (4 sub) | any | e5+ |
| AFV0-3 | 8×8 (corner DCT) | any | e6+ |
| DCT16x8, DCT8x16 | 16×8 | any | e5+ |
| DCT16x16 | 16×16 | any | e5+ |
| DCT32x16, DCT16x32 | 32×16 | d ≥ 2.0 | e6+ |
| DCT32x32 | 32×32 | d ≥ 2.0 | e7+ |
| DCT64x32, DCT32x64 | 64×32 | d ≥ 3.0 | e7+ |
| DCT64x64 | 64×64 | d ≥ 3.0 | e7+ |
<!-- crates.io:skip-end -->

## CLI

A command-line wrapper, `cjxl-rs`, ships in the
[`jxl-encoder-cli`](https://crates.io/crates/jxl-encoder-cli) crate:

```bash
cargo install jxl-encoder-cli

# Lossy (distance 1.0 = visually lossless)
cjxl-rs input.png output.jxl -d 1.0

# Lossless
cjxl-rs input.png output.jxl --lossless

# See all options
cjxl-rs --help
```

<!-- crates.io:skip-start -->
## Experimental: CVVDP-driven quantization loop (opt-in)

The quant loop at effort ≥ 8 normally calls butteraugli once per iteration. An
opt-in path drives it with [ColorVideoVDP](https://github.com/gfxdisp/ColorVideoVDP)
(cvvdp, Mantiuk et al. 2024) instead. Default OFF; butteraugli stays the
production default, and `EncoderStrategy::Libjxl` forces cvvdp off regardless so
cjxl-parity byte-locks hold.

```bash
cargo build --release --features cvvdp-loop       # GPU (needs CUDA)
cargo build --release --features cvvdp-loop-cpu    # pure-Rust CPU
```

It is opt-in because cvvdp ships an uncalibrated per-distance target table: at
the same `distance` it converges to a tighter perceptual target than the
`distance` knob currently implies, producing larger files. The full 1,134-cell
tracking sweep is at `benchmarks/cvvdp_vs_buttloop_tracking_2026-05-24.tsv`;
methodology and the ship-rule are in
[`docs/CVVDP_FORK_DECISION.md`](docs/CVVDP_FORK_DECISION.md). zensim is also
available as a third quant-loop metric (`zensim-loop` / `zensim-loop-gpu`).

## Reproducible benchmarks

Every comparison number above traces to a committed file under `benchmarks/`.
[`benchmarks/README.md`](benchmarks/README.md) is the reproduction index: the
exact command for each board, what corpus it uses, what it measures, and the
quiet-box caveat for wall numbers. Bytes and quality are deterministic;
**wall-time numbers are only disposition-grade on a quiet box** — the wall
harness refuses to run under load for that reason.

## Project structure

```
jxl-encoder/                    # workspace root (this repo)
├── jxl-encoder/               # the jxl-encoder library crate
│   └── src/
│       ├── api.rs                 # public API (LossyConfig, LosslessConfig, EncodeRequest)
│       ├── vardct/                # VarDCT (lossy) encoder
│       ├── modular/               # Modular (lossless) encoder
│       ├── entropy_coding/        # ANS, Huffman, HybridUint, LZ77
│       └── headers/               # file / frame headers
├── jxl-encoder-simd/          # SIMD primitives (jxl-encoder-simd on crates.io)
├── jxl-encoder-macros/        # internal proc-macros (jxl-encoder-macros on crates.io)
└── jxl-encoder-cli/           # CLI tool: cjxl-rs (jxl-encoder-cli on crates.io)
```

## Building

```bash
cargo build                                # debug
cargo build --release -p jxl-encoder-cli   # release CLI
cargo test --workspace --lib --tests       # all tests
cargo clippy --workspace -- -D warnings    # lint
```
<!-- crates.io:skip-end -->

## When to reach for cjxl instead

- **When encode speed matters more than the last few percent of bytes.** cjxl is
  faster on nearly every measured cell, especially multi-threaded.
- **When you need streaming / bounded-memory encode of very large images** — not
  yet implemented here; cjxl streams.
- **For HDR / smooth-gradient lossy at the byte minimum** — cjxl wins most of
  those cells today.
- This crate is pure-Rust, `forbid(unsafe_code)`, embeddable with no C FFI, and
  ahead on 8-bit lossless and a fair slice of SDR lossy — reach for it when those
  matter.

## Credits

- **[libjxl](https://github.com/libjxl/libjxl)** (JPEG XL Project Authors,
  BSD-3-Clause) — the reference encoder and a well-engineered, battle-tested
  codebase. Our algorithms, quantization weights, cost models, and bitstream
  format are derived from it.
  [libjxl-tiny](https://github.com/nicoshev/libjxl-tiny) was the initial
  porting target.
- **[zune-jpegxl](https://github.com/etemesi254/zune-image/tree/dev/crates/zune-jpegxl)**
  (Caleb Etemesi, MIT/Apache-2.0/Zlib) — a working pure-Rust JXL lossless
  encoder (~2.5k lines) that was the inspiration to extend into lossy encoding
  and the features above.
- **[jxl-rs](https://github.com/libjxl/jxl-rs)** (BSD-3-Clause) — primary
  roundtrip validation decoder.
- **[jxl-oxide](https://github.com/tirr-c/jxl-oxide)** — secondary validation
  decoder.
- **Claude** (Anthropic) — AI-assisted development. Not all code has been
  manually reviewed; review critical paths before production use.

## License

Dual-licensed:
[AGPL-3.0-or-later](https://github.com/imazen/jxl-encoder/blob/main/LICENSE-AGPL3)
or
[commercial](https://github.com/imazen/jxl-encoder/blob/main/LICENSE-COMMERCIAL).

I've maintained open-source image software — and the 40+ library ecosystem it
depends on — full-time since 2011. Fifteen years of continual maintenance,
backwards compatibility, support, and the (very rare) security patch. That kind
of stability requires sustainable funding, and dual-licensing is how we make it
work without venture capital or rug-pulls.

**Your options:**

- **Startup license** — $1 if your company has under $1M revenue and fewer than
  5 employees. [Get a key →](https://www.imazen.io/pricing)
- **Commercial subscription** — Apache-2.0-like terms, no source-sharing
  requirement. Sliding scale by company size.
  [Pricing & 60-day free trial →](https://www.imazen.io/pricing)
- **AGPL v3** — free and open. Share your source if you distribute.

See
[LICENSE-COMMERCIAL](https://github.com/imazen/jxl-encoder/blob/main/LICENSE-COMMERCIAL)
for details. Upstream code from
[libjxl/libjxl](https://github.com/libjxl/libjxl) is licensed under BSD-3-Clause;
our additions are dual-licensed (AGPL-3.0-or-later or commercial) as above.

## Image tech I maintain

| | |
|:--|:--|
| **Codecs** ¹ | [zenjpeg] · [zenpng] · [zenwebp] · [zengif] · [zenavif] · [zenjxl] · [zenbitmaps] · [heic] · [zentiff] · [zenpdf] · [zensvg] · [zenjp2] · [zenraw] · [ultrahdr] |
| Codec internals | [zenjxl-decoder] · **jxl-encoder** · [zenrav1e] · [rav1d-safe] · [zenavif-parse] · [zenavif-serialize] |
| Compression | [zenflate] · [zenzop] · [zenzstd] |
| Processing | [zenresize] · [zenquant] · [zenblend] · [zenfilters] · [zensally] · [zentone] |
| Pixels & color | [zenpixels] · [zenpixels-convert] · [linear-srgb] · [garb] |
| Pipeline & framework | [zenpipe] · [zencodec] · [zencodecs] · [zenlayout] · [zennode] · [zenwasm] · [zentract] |
| Metrics | [zensim] · [fast-ssim2] · [butteraugli] · [zenmetrics] · [resamplescope-rs] |
| Pickers & ML | [zenanalyze] · [zenpredict] · [zenpicker] |
| Products | [Imageflow] image engine ([.NET][imageflow-dotnet] · [Node][imageflow-node] · [Go][imageflow-go]) · [Imageflow Server] · [ImageResizer] (C#) |

<sub>¹ pure-Rust, `#![forbid(unsafe_code)]` codecs, as of 2026</sub>

### General Rust awesomeness

[zenbench] · [archmage] · [magetypes] · [enough] · [whereat] · [cargo-copter]

[Open source](https://www.imazen.io/open-source) · [@imazen](https://github.com/imazen) · [@lilith](https://github.com/lilith) · [lib.rs/~lilith](https://lib.rs/~lilith)

[zenjpeg]: https://github.com/imazen/zenjpeg
[zenpng]: https://github.com/imazen/zenpng
[zenwebp]: https://github.com/imazen/zenwebp
[zengif]: https://github.com/imazen/zengif
[zenavif]: https://github.com/imazen/zenavif
[zenjxl]: https://github.com/imazen/zenjxl
[zenbitmaps]: https://github.com/imazen/zenbitmaps
[heic]: https://github.com/imazen/heic
[zentiff]: https://github.com/imazen/zentiff
[zenpdf]: https://github.com/imazen/zenpdf
[zensvg]: https://github.com/imazen/zenextras
[zenjp2]: https://github.com/imazen/zenextras
[zenraw]: https://github.com/imazen/zenraw
[ultrahdr]: https://github.com/imazen/ultrahdr
[zenjxl-decoder]: https://github.com/imazen/zenjxl-decoder
[zenrav1e]: https://github.com/imazen/zenrav1e
[rav1d-safe]: https://github.com/imazen/rav1d-safe
[zenavif-parse]: https://github.com/imazen/zenavif-parse
[zenavif-serialize]: https://github.com/imazen/zenavif-serialize
[zenflate]: https://github.com/imazen/zenflate
[zenzop]: https://github.com/imazen/zenzop
[zenzstd]: https://github.com/imazen/zenzstd
[zenresize]: https://github.com/imazen/zenresize
[zenquant]: https://github.com/imazen/zenquant
[zenblend]: https://github.com/imazen/zenblend
[zenfilters]: https://github.com/imazen/zenfilters
[zensally]: https://github.com/imazen/zensally
[zentone]: https://github.com/imazen/zentone
[zenpixels]: https://github.com/imazen/zenpixels
[zenpixels-convert]: https://github.com/imazen/zenpixels
[linear-srgb]: https://github.com/imazen/linear-srgb
[garb]: https://github.com/imazen/garb
[zenpipe]: https://github.com/imazen/zenpipe
[zencodec]: https://github.com/imazen/zencodec
[zencodecs]: https://github.com/imazen/zencodecs
[zenlayout]: https://github.com/imazen/zenlayout
[zennode]: https://github.com/imazen/zennode
[zenwasm]: https://github.com/imazen/zenwasm
[zentract]: https://github.com/imazen/zentract
[zensim]: https://github.com/imazen/zensim
[fast-ssim2]: https://github.com/imazen/fast-ssim2
[butteraugli]: https://github.com/imazen/butteraugli
[zenmetrics]: https://github.com/imazen/zenmetrics
[resamplescope-rs]: https://github.com/imazen/resamplescope-rs
[zenanalyze]: https://github.com/imazen/zenanalyze
[zenpredict]: https://github.com/imazen/zenanalyze
[zenpicker]: https://github.com/imazen/zenanalyze
[zenbench]: https://github.com/imazen/zenbench
[archmage]: https://github.com/imazen/archmage
[magetypes]: https://github.com/imazen/archmage
[enough]: https://github.com/imazen/enough
[whereat]: https://github.com/lilith/whereat
[cargo-copter]: https://github.com/imazen/cargo-copter
[Imageflow]: https://github.com/imazen/imageflow
[Imageflow Server]: https://github.com/imazen/imageflow-dotnet-server
[ImageResizer]: https://github.com/imazen/resizer
[imageflow-dotnet]: https://github.com/imazen/imageflow-dotnet
[imageflow-node]: https://github.com/imazen/imageflow-node
[imageflow-go]: https://github.com/imazen/imageflow-go
