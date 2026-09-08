# Encoder fuzz regressions

The codec_wiki streaming mismatch is reproduced from the real corpus by
`production_resources::real_screenshot_streaming_keeps_one_shot_content_dispatch`.
Its original 398609-byte input remains outside git (SHA256
`e04fba6274cb62e33c2d175cf639e7ed48a0404bf05cdc62fdb20f9d9fc25844`).
The fixed implementation passes that original input under AddressSanitizer.

Resource-admission cases live in `jxl-encoder/src/api/tests.rs`. Stable replay
in `jxl-encoder/tests/fuzz_regression.rs` exercises both fuzz entry points on
single- and multi-group input and fully renders with jxl-rs and djxl v0.12.
Working corpora and raw crashes stay outside git; storage provenance belongs
in `benchmarks/production_validation_2026-09-08.pointer.md`.
