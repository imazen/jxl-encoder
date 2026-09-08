# Encoder fuzz regressions

No minimized crash inputs have been added yet. Fixed resource-admission cases
live as Rust tests in `jxl-encoder/src/api/tests.rs`. The stable replay target
`jxl-encoder/tests/fuzz_regression.rs` also exercises both fuzz entry points on
single- and multi-group input without requiring a nightly toolchain.

Working corpora and raw crashes stay outside git. New known-fixed crashes must
be minimized, replayed, and added to the stable regression target.
