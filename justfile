# jxl-encoder-rs task runner

# Run RD regression test (encodes 6 images at d=0.25 and d=0.5, ~3 min in debug)
rd-regression:
    cargo test -p jxl-encoder --test clic2025 test_rd_regression -- --ignored --nocapture

# Run high-distance RD regression test (d=2.0 and d=3.0, exercises DCT32x32/DCT64x64)
rd-regression-hd:
    cargo test -p jxl-encoder --test clic2025 test_rd_regression_high_distance -- --ignored --nocapture

# Compare cjxl-rs vs libwebp on CID22 validation set (41 images x 4 quality points)
cid22-vs-webp:
    bash scripts/cid22_vs_webp.sh

# Cross-compile and test for 32-bit x86 (requires cross: cargo install cross --git https://github.com/cross-rs/cross)
test-i686:
    cross test --workspace --no-default-features --features safe-mode --lib --target i686-unknown-linux-gnu

# Cross-compile and test for 32-bit ARM (requires cross)
test-armv7:
    cross test --workspace --no-default-features --features safe-mode --lib --target armv7-unknown-linux-gnueabihf

# Cross-compile and test for AArch64 (requires cross + Docker)
test-aarch64:
    CROSS_CONTAINER_OPTS="--volume /home/lilith/work:/home/lilith/work" cross test --workspace --no-default-features --features safe-mode --lib --target aarch64-unknown-linux-gnu

# Build and test for WASM (requires wasmtime)
test-wasm:
    CARGO_TARGET_WASM32_WASIP1_RUNNER="wasmtime --" cargo test -p jxl-encoder --target wasm32-wasip1 --no-default-features --features safe-mode --lib -- api::tests

# Run encode benchmark on all platforms
bench-platforms:
    @echo "=== x86_64 native ===" && cargo run --example wasm_bench -p jxl-encoder --release --no-default-features --features safe-mode
    @echo "=== WASM (wasmtime) ===" && cargo build --example wasm_bench -p jxl-encoder --release --target wasm32-wasip1 --no-default-features --features safe-mode 2>/dev/null && wasmtime ./target/wasm32-wasip1/release/examples/wasm_bench.wasm
    @echo "=== AArch64 (qemu) ===" && CROSS_CONTAINER_OPTS="--volume /home/lilith/work:/home/lilith/work" cross run --example wasm_bench -p jxl-encoder --release --target aarch64-unknown-linux-gnu --no-default-features --features safe-mode

# Run all cross-compilation targets
test-cross: test-i686 test-armv7
