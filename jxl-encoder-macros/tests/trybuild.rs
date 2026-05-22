//! `trybuild` UI tests for the [`jxl_encoder_macros::strategy_def!`]
//! macro.
//!
//! - `ui_pass/` — well-formed invocations that should compile cleanly.
//! - `ui_fail/` — malformed invocations that should produce
//!   `compile_error!` with the expected message (recorded in `*.stderr`).
//!
//! Run with: `cargo test -p jxl-encoder-macros --test trybuild`.
//! Regenerate `.stderr` golden files: `TRYBUILD=overwrite cargo test ...`.

#[test]
fn ui() {
    let t = trybuild::TestCases::new();
    t.pass("tests/ui_pass/*.rs");
    t.compile_fail("tests/ui_fail/*.rs");
}
