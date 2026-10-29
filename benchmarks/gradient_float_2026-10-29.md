# Full-range modular gradient correction

Source: test-only `1b4dd6a3a9b9a117b06226e763283c2d6f8a4097`,
fixed production `c9584496c6730e15a21a00847dcad31cedd48d03`.

The existing F32 squeeze integration test failed in coverage at `11c6eb0c`
with integer overflow in the private gradient helper. The added test compares
343 extrema triples to an i64 clamped arithmetic oracle and fails before the fix.
The fix reuses the existing full-range predictor implementation.

Serialized commands used `run-heavy --mem 3G --jobs 2`, timeout 900 seconds,
a locked isolated dependency closure, and test workers capped at two.

- Pre-fix boundary: expected failure, rc101, 178s, peak RSS 1.80 GiB.
- Fixed `just gradient-float-check` equivalent commands: boundary and existing
  F32 squeeze bit-exact round-trip pass, rc0, 143s, peak RSS 1.81 GiB.
  The native test also requires libjxl acceptance (official v0.12.0 tools).
- `cargo test --locked -p jxl-encoder --lib --features __expert -- --test-threads=2`
  then `cargo clippy --locked --workspace --all-targets -- -D warnings`:
  1717 passed, 0 failed, 29 existing ignored; Clippy passed.
  Wrapper rc0, 78s, peak RSS 1.03 GiB, minimum available 13039 MiB.

These are correctness and build controls, not speed benchmarks. They do not
qualify every integration test or cross-platform release. Original assertions
and ignored-test applicability are unchanged.
