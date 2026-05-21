# jxl-encoder-rs task runner

# Run RD regression test (encodes 6 images at d=0.25, d=0.5, d=1.0)
rd-regression:
    cargo test -p jxl-encoder --test clic2025 test_rd_regression -- --ignored --nocapture

# Run high-distance RD regression test (d=2.0 and d=3.0, exercises DCT32x32/DCT64x64)
rd-regression-hd:
    cargo test -p jxl-encoder --test clic2025 test_rd_regression_high_distance -- --ignored --nocapture

# Compare cjxl-rs vs libwebp on CID22 validation set (41 images x 4 quality points)
cid22-vs-webp:
    bash scripts/cid22_vs_webp.sh

# W44-170: full comprehensive cjxl-parity sweep at step 0.25 across 20-image
# varied corpus, 5 efforts, both Libjxl and Zenjxl strategies. ~15-30 min
# wall on the workstation. Writes per-strategy TSVs to benchmarks/, then
# runs Python analysis producing charts + markdown report.
w44-170-sweep:
    cargo run -p jxl-encoder --release --features 'parallel butteraugli-loop ssim2-loop' \
        --example w44_170_cjxl_step025_sweep -- \
        --corpus-manifest benchmarks/corpora/w44_170_varied_corpus.tsv \
        --output-prefix benchmarks/cjxl_step025 \
        --strategies zenjxl,libjxl
    python3 benchmarks/scripts/w44_170_analyze.py \
        --zenjxl $(ls -1t benchmarks/cjxl_step025_zenjxl_*.tsv | head -1) \
        --libjxl $(ls -1t benchmarks/cjxl_step025_libjxl_*.tsv | head -1) \
        --output-md benchmarks/w44_170_analysis_$(date +%Y-%m-%d).md \
        --chart-dir benchmarks/charts \
        --chart-tag w44_170

# Cross-compile and test for 32-bit x86 (requires cross: cargo install cross --git https://github.com/cross-rs/cross)
test-i686:
    cross test --workspace --no-default-features --lib --target i686-unknown-linux-gnu

# Cross-compile and test for 32-bit ARM (requires cross)
test-armv7:
    cross test --workspace --no-default-features --lib --target armv7-unknown-linux-gnueabihf

# Cross-compile and test for AArch64 (requires cross + Docker)
test-aarch64:
    CROSS_CONTAINER_OPTS="--volume $HOME/work:$HOME/work" cross test --workspace --no-default-features --lib --target aarch64-unknown-linux-gnu

# Build and test for WASM (requires wasmtime)
test-wasm:
    CARGO_TARGET_WASM32_WASIP1_RUNNER="wasmtime --" cargo test -p jxl-encoder --target wasm32-wasip1 --no-default-features --lib -- api::tests

# Test jxl-encoder-simd under WASM SIMD128 (requires wasmtime)
test-wasm-simd:
    RUSTFLAGS="-C target-feature=+simd128" CARGO_TARGET_WASM32_WASIP1_RUNNER="wasmtime --" cargo test -p jxl-encoder-simd --target wasm32-wasip1

# Run encode benchmark on all platforms
bench-platforms:
    @echo "=== x86_64 native ===" && cargo run --example wasm_bench -p jxl-encoder --release --no-default-features
    @echo "=== WASM (wasmtime) ===" && cargo build --example wasm_bench -p jxl-encoder --release --target wasm32-wasip1 --no-default-features 2>/dev/null && wasmtime ./target/wasm32-wasip1/release/examples/wasm_bench.wasm
    @echo "=== AArch64 (qemu) ===" && CROSS_CONTAINER_OPTS="--volume $HOME/work:$HOME/work" cross run --example wasm_bench -p jxl-encoder --release --target aarch64-unknown-linux-gnu --no-default-features

# Run all cross-compilation targets
test-cross: test-i686 test-armv7

# Generate cjxl reference CSV (~40 min, run once per cjxl version)
generate-reference:
    bash scripts/generate_cjxl_reference.sh

# Encode with cjxl-rs and compare against reference (~30 min)
rd-compare:
    cargo build --release -p jxl-encoder-cli
    bash scripts/measure_cjxl_rs.sh
    python3 scripts/rd_report.py

# Regenerate hash lock sidecar file after intentional encoding changes
update-hashes:
    rm -f jxl_encoder/tests/hash_lock_expected.txt
    UPDATE_HASHES=1 cargo test --test hash_lock_features -- --test-threads=1

# Compare quality vs cjxl (uses committed reference CSV, ~2 min)
quality-compare:
    cargo test -p jxl-encoder --test quality_compare --release -- --ignored --nocapture

# Quick comparison: 10 CLIC + 5 CID22 + all screenshots (~5 min)
rd-compare-quick:
    cargo build --release -p jxl-encoder-cli
    bash scripts/measure_cjxl_rs.sh --quick
    python3 scripts/rd_report.py

# Compare lossless compression vs cjxl (CSV-backed, ~1 min)
lossless-compare:
    cargo test -p jxl-encoder --test lossless_compare --release -- --ignored --nocapture

# Generate cjxl lossless reference CSV (~10 min)
generate-lossless-reference:
    bash scripts/generate_cjxl_lossless_reference.sh

# Six-panel visual comparison at a given distance
# Layout:  Source info     | Ours d=X metrics        | cjxl d=X metrics (deltas vs ours)
#          Ours-cjxl 20x   | Ours Error 10x          | cjxl Error 10x
# Usage: just compare-visual <source.png> <ours.jxl> <cjxl.jxl> <distance> [output_dir]
# Ours/cjxl args can be .jxl (decoded via djxl) or .png (used directly)
compare-visual source ours cjxl distance outdir="${JXL_ENCODER_OUTPUT_DIR:-/mnt/v/output/jxl-encoder-rs}/compare":
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p "{{outdir}}"
    DJXL="${DJXL_PATH:-$HOME/work/jxl-efforts/libjxl/build/tools/djxl}"
    BFLY="${BUTTERAUGLI_MAIN_PATH:-$HOME/work/jxl-efforts/libjxl/build/tools/butteraugli_main}"
    SS2="${SSIMULACRA2_PATH:-$HOME/work/jxl-efforts/libjxl/build/tools/ssimulacra2}"
    # Source info
    src_size=$(wc -c < "{{source}}")
    src_kb=$(python3 -c "print(f'{${src_size}/1024:.1f}KB')")
    src_info=$(identify -format '%wx%h %[channels]' "{{source}}" 2>/dev/null || echo "?x?")
    # Decode JXL to PNG if needed, get file sizes
    if [[ "{{ours}}" == *.jxl ]]; then
      ours_png="{{outdir}}/ours_decoded.png"
      "$DJXL" "{{ours}}" "$ours_png" 2>/dev/null
      ours_size=$(wc -c < "{{ours}}")
    else
      ours_png="{{ours}}"
      ours_size=$(wc -c < "{{ours}}")
    fi
    if [[ "{{cjxl}}" == *.jxl ]]; then
      cjxl_png="{{outdir}}/cjxl_decoded.png"
      "$DJXL" "{{cjxl}}" "$cjxl_png" 2>/dev/null
      cjxl_size=$(wc -c < "{{cjxl}}")
    else
      cjxl_png="{{cjxl}}"
      cjxl_size=$(wc -c < "{{cjxl}}")
    fi
    ours_kb=$(python3 -c "print(f'{${ours_size}/1024:.1f}KB')")
    cjxl_kb=$(python3 -c "print(f'{${cjxl_size}/1024:.1f}KB')")
    # Strip source PNG metadata to avoid gAMA/sRGB TF mismatch in butteraugli_main
    stripped="{{outdir}}/source_stripped.png"
    convert "{{source}}" -strip "$stripped"
    # Compute butteraugli + ssim2
    ours_bfly=$("$BFLY" "$stripped" "$ours_png" 2>/dev/null | head -1) || ours_bfly="?"
    cjxl_bfly=$("$BFLY" "$stripped" "$cjxl_png" 2>/dev/null | head -1) || cjxl_bfly="?"
    ours_ss2=$("$SS2" "$stripped" "$ours_png" 2>/dev/null | head -1) || ours_ss2="?"
    cjxl_ss2=$("$SS2" "$stripped" "$cjxl_png" 2>/dev/null | head -1) || cjxl_ss2="?"
    rm -f "$stripped"
    # Format labels: ours is baseline, cjxl shows deltas relative to ours
    eval "$(python3 -c "
    ob, cb = float('${ours_bfly}'), float('${cjxl_bfly}')
    os2, cs2 = float('${ours_ss2}'), float('${cjxl_ss2}')
    osz, csz = ${ours_size}, ${cjxl_size}
    def dp(a,b): return f'{(a-b)/b*100:+.1f}%' if b else '?'
    # Shell-safe: no spaces that could break read
    print(f'ours_metrics=\"{ob:.2f}  ss2={os2:.1f}\"')
    print(f'cjxl_metrics=\"{cb:.2f} ({dp(cb,ob)})  ss2={cs2:.1f} ({dp(cs2,os2)})\"')
    print(f'size_delta=\"{dp(csz,osz)}\"')
    " 2>/dev/null)"
    src_label="Source  ${src_info}  ${src_kb}"
    ours_label="Ours d={{distance}}  ${ours_kb}  bfly=${ours_metrics}"
    cjxl_label="cjxl d={{distance}}  ${cjxl_kb} (${size_delta})  bfly=${cjxl_metrics}"
    # Generate diff images
    convert "{{source}}" "$ours_png" -compose difference -composite -evaluate multiply 10 "{{outdir}}/ours_err_10x.png"
    convert "{{source}}" "$cjxl_png" -compose difference -composite -evaluate multiply 10 "{{outdir}}/cjxl_err_10x.png"
    convert "$ours_png" "$cjxl_png" -compose difference -composite -evaluate multiply 20 "{{outdir}}/delta_20x.png"
    montage \
      -label "$src_label" "{{source}}" \
      -label "$ours_label" "$ours_png" \
      -label "$cjxl_label" "$cjxl_png" \
      -label "Ours-cjxl 20x" "{{outdir}}/delta_20x.png" \
      -label "Ours Error 10x" "{{outdir}}/ours_err_10x.png" \
      -label "cjxl Error 10x" "{{outdir}}/cjxl_err_10x.png" \
      -tile 3x2 -geometry +2+2 \
      -font DejaVu-Sans -pointsize 14 \
      "{{outdir}}/six_compare.png"
    feh "{{outdir}}/six_compare.png" &
    echo "Saved to {{outdir}}/six_compare.png"
