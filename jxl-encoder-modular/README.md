# jxl-encoder-modular

[![crates.io](https://img.shields.io/crates/v/jxl-encoder-modular.svg)](https://crates.io/crates/jxl-encoder-modular)
[![docs.rs](https://docs.rs/jxl-encoder-modular/badge.svg)](https://docs.rs/jxl-encoder-modular)

The JPEG XL modular-mode (lossless) encoder, extracted from
[jxl-encoder](https://crates.io/crates/jxl-encoder): channel/palette
transforms, MA tree learning, sectioned local trees, weighted
predictor, headers/ColorEncoding, and the effort schedule.

Mostly internal — for the full public API use `jxl-encoder`, which
re-exports this crate's modules. Not related to `jxl-modular`
(the decoder half in the jxl-oxide family).
