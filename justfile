# jxl-encoder-rs task runner

# Run RD regression test (encodes 6 images at d=0.25 and d=0.5, ~3 min in debug)
rd-regression:
    cargo test -p jxl_enc --test clic2025 test_rd_regression -- --ignored --nocapture
