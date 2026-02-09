# jxl-encoder-rs task runner

# Run RD regression test (encodes 6 images at d=0.25 and d=0.5, ~3 min in debug)
rd-regression:
    cargo test -p jxl-encoder --test clic2025 test_rd_regression -- --ignored --nocapture

# Compare cjxl-rs vs libwebp on CID22 validation set (41 images x 4 quality points)
cid22-vs-webp:
    bash scripts/cid22_vs_webp.sh

# Cross-compile and test for 32-bit x86 (requires cross: cargo install cross --git https://github.com/cross-rs/cross)
test-i686:
    cross test --workspace --no-default-features --features safe-mode --lib --target i686-unknown-linux-gnu

# Cross-compile and test for 32-bit ARM (requires cross)
test-armv7:
    cross test --workspace --no-default-features --features safe-mode --lib --target armv7-unknown-linux-gnueabihf

# Run all cross-compilation targets
test-cross: test-i686 test-armv7
