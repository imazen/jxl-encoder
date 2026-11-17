# jxl-encoder-rs task runner

# Caller supplies an isolated pinned source closure and resource controls.
frame-header-check manifest="Cargo.toml" features="__expert":
    cargo test --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features {{quote(features)}} --test frame_header_contract -- --nocapture

color-header-check manifest="Cargo.toml":
    cargo test --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --lib headers::color_encoding::tests
    python3 -m unittest discover -s scripts -p 'test_jxl*.py'

color-description-probe space output manifest="Cargo.toml":
    cargo run --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert --example color_description_probe -- {{quote(space)}} {{quote(output)}}

color-contract-check manifest="Cargo.toml":
    cargo test --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert --lib
    cargo test --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert --test frame_header_contract
    cargo test --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert --test it hash_lock
    cargo test --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert --test it strategy_libjxl
    cargo test --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert,__internals --test it divergence_table_drift
    cargo clippy --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert --lib --test frame_header_contract --example color_description_probe -- -D warnings
    python3 -m unittest discover -s scripts -p 'test_jxl*.py'

color-contract-feature-check manifest="Cargo.toml":
    cargo test --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --no-default-features --features std,__expert --test frame_header_contract
    cargo clippy --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --no-default-features --features std,__expert --lib --test frame_header_contract -- -D warnings
    cargo test --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert,jpeg-reencoding,chroma-subsampling --test subsampled_metadata
    cargo clippy --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert,jpeg-reencoding,chroma-subsampling --lib --test subsampled_metadata -- -D warnings

color-payload-controls rgb xyb unknown output:
    python3 scripts/check_color_description_payloads.py {{quote(rgb)}} {{quote(xyb)}} {{quote(unknown)}} {{quote(output)}}

color-original-writer manifest original output:
    python3 scripts/probe_original_color_writer.py {{quote(manifest)}} {{quote(original)}} {{quote(output)}}

color-header-mutations manifest logs:
    python3 scripts/test_color_header_mutations.py {{quote(manifest)}} {{quote(logs)}}

color-header-coverage manifest output:
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert --lib --no-report -- headers::color_encoding::tests
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert --test frame_header_contract --no-report -- explicit_icc_flag
    cargo llvm-cov report --manifest-path {{quote(manifest)}} --json --output-path {{quote(output)}}

color-header-oracle source build output:
    python3 scripts/frame_header_oracle.py --component color_header {{quote(source)}} {{quote(build)}} {{quote(output)}}

frame-header-lint manifest="Cargo.toml":
    cargo clippy --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert --lib --test frame_header_contract -- -D warnings

frame-header-oracle source build output:
    python3 scripts/frame_header_oracle.py {{quote(source)}} {{quote(build)}} {{quote(output)}}

frame-header-mutations manifest logs:
    python3 scripts/test_frame_header_mutations.py {{quote(manifest)}} {{quote(logs)}}

frame-header-coverage manifest output:
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert --lib --no-report -- headers::frame_header::tests
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert --test frame_header_contract --no-report
    cargo llvm-cov report --manifest-path {{quote(manifest)}} --json --output-path {{quote(output)}}

reference-tools-check manifest="Cargo.toml":
    cargo test --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --lib test_helpers::libjxl_version_guard_tests
    cargo clippy --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --lib -- -D warnings

reference-cross-diagnostics:
    bash scripts/reference_cross_diagnostics.sh

icc-tag-mutation manifest log:
    python3 scripts/test_icc_tag_mutation.py {{quote(manifest)}} {{quote(log)}}

icc-tag-coverage manifest output:
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert --lib --no-report -- icc::tests
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert --test icc_tag_contract --no-report
    cargo llvm-cov report --manifest-path {{quote(manifest)}} --json --output-path {{quote(output)}}

icc-tag-check manifest="Cargo.toml":
    cargo test --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert --lib
    cargo test --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert --test icc_tag_contract
    cargo test --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert --test it icc_
    cargo test --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert --test it hash_lock
    cargo test --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert --test it strategy_libjxl
    cargo test --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert,__internals --test it divergence_table_drift
    cargo clippy --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert --lib --test icc_tag_contract -- -D warnings

icc-tag-feature-check manifest="Cargo.toml":
    cargo test --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --no-default-features --features std,__expert --test icc_tag_contract
    cargo test --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert,jpeg-reencoding,chroma-subsampling --test icc_tag_contract
    cargo clippy --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert,jpeg-reencoding,chroma-subsampling --lib --test icc_tag_contract -- -D warnings

# Independent decoder controls require verified CJXL_PATH/DJXL_PATH on macOS.
fixed-point-baseline manifest="Cargo.toml":
    cargo test --locked --manifest-path "{{manifest}}" -p jxl-encoder --lib

# Exact fixed-point transition, full-schedule byte/state parity, callback and
# cancellation behavior. The caller supplies run-heavy/resource controls.
fixed-point-check manifest="Cargo.toml":
    cargo test --locked --manifest-path "{{manifest}}" -p jxl-encoder --lib fixed_point_tests
    cargo test --locked --manifest-path "{{manifest}}" -p jxl-encoder --features __expert,__internals,__external_metric,__internal_recon_hook --test loop_fixpoint
    cargo test --locked --manifest-path "{{manifest}}" -p jxl-encoder --features __expert,__internals,__external_metric,__internal_recon_hook --test it loop_trace
    cargo test --locked --manifest-path "{{manifest}}" -p jxl-encoder --features __expert,__internals,__external_metric,__internal_recon_hook --test external_metric
    cargo test --locked --manifest-path "{{manifest}}" -p jxl-encoder --features __expert,__internals,__external_metric,__internal_recon_hook --test it hash_lock_features
    cargo test --locked --manifest-path "{{manifest}}" -p jxl-encoder --features __expert,__internals,__external_metric,__internal_recon_hook --test it strategy_libjxl
    cargo test --locked --manifest-path "{{manifest}}" -p jxl-encoder --features __expert,__internals,__external_metric,__internal_recon_hook --test it divergence_table_drift

fixed-point-lint manifest="Cargo.toml":
    cargo clippy --locked --manifest-path "{{manifest}}" --workspace --all-targets -- -D warnings
    cargo clippy --locked --manifest-path "{{manifest}}" -p jxl-encoder --features __expert,__internals,__external_metric,__internal_recon_hook --test loop_fixpoint --test external_metric --test it -- -D warnings

# Prove the isolated external oracle needs only its own feature closure.
fixed-point-external-minimal manifest="Cargo.toml":
    cargo test --locked --manifest-path "{{manifest}}" -p jxl-encoder --no-default-features --features __external_metric --test external_metric

# Format workspace members (NOT --all: that recurses into cloned sibling
# path-dep repos) and refresh the public-API surface snapshots.
# The snapshot runner lives in the workspace-excluded apidoc/ package, so it
# is never built or run by plain `cargo test` or any CI job.
fmt:
    cargo fmt -p jxl-encoder -p jxl-encoder-cli -p jxl-encoder-macros -p jxl-encoder-simd -p zenjxl-tuning-runner
    cargo test --manifest-path apidoc/Cargo.toml

# Build upstream parent/rewrite/current sources without changing the v0.12 reference.
lz77-upstream-build root:
    mkdir -p "{{root}}"
    TMPDIR="$HOME/tmp" nice -n 19 python3 scripts/lz77_upstream_build.py "{{root}}" --jobs 2 >> "{{root}}/build-driver.log" 2>&1

# Retain each upstream float encode and require exact samples in both decoders.
lz77-upstream-compare root screen manifest="Cargo.toml":
    #!/usr/bin/env bash
    set -euo pipefail
    export TMPDIR="$HOME/tmp" CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=2
    export CARGO_TARGET_DIR="{{justfile_directory()}}/target"
    export DJXL_PATH="{{justfile_directory()}}/.ci-libjxl/tools/djxl"
    nice -n 19 cargo build --locked --manifest-path "{{manifest}}" -p jxl-encoder --example lossless_float_parity >> "{{root}}/rust-verifier-build.log" 2>&1
    verifier="$CARGO_TARGET_DIR/debug/examples/lossless_float_parity"
    reference=$("$verifier" --reference-tools)
    nice -n 19 uv run scripts/lz77_upstream_compare.py "{{root}}" "{{screen}}" "$reference" >> "{{root}}/compare.log" 2>&1
    nice -n 19 "$verifier" --verify-manifest "{{root}}/results.tsv" >> "{{root}}/rust-verification.log" 2>&1
    mkdir "{{root}}/group256"
    cp "{{root}}/builds.json" "{{root}}/group256/builds.json"
    nice -n 19 uv run scripts/lz77_upstream_compare.py "{{root}}/group256" "{{screen}}" "$reference" --sizes 259 --group-size-shift 1 >> "{{root}}/group256/compare.log" 2>&1
    nice -n 19 "$verifier" --verify-manifest "{{root}}/group256/results.tsv" >> "{{root}}/group256/rust-verification.log" 2>&1

# Regenerate the public-API surface snapshots only
api-doc:
    cargo test --manifest-path apidoc/Cargo.toml

# Verify the committed snapshots are current
api-doc-check:
    ZEN_API_DOC=check cargo test --manifest-path apidoc/Cargo.toml

# Explicit release matrix: natural input, pathological patterns, retained outputs.
prepublish-matrix photo artifacts manifest="Cargo.toml" features="__expert,corpus-tests,parallel" *cargo_args:
    TMPDIR="$HOME/tmp" CARGO_BUILD_JOBS=4 RAYON_NUM_THREADS=4 CARGO_TARGET_DIR="{{justfile_directory()}}/target" CJXL_PATH="{{justfile_directory()}}/.ci-libjxl/tools/cjxl" DJXL_PATH="{{justfile_directory()}}/.ci-libjxl/tools/djxl" JXL_AUDIT_PHOTO="{{photo}}" JXL_AUDIT_ARTIFACTS="{{artifacts}}" nice -n 19 cargo test --manifest-path "{{manifest}}" --locked -p jxl-encoder --features "{{features}}" {{cargo_args}} --test prepublish_matrix -- --test-threads=1 --nocapture

# A selected compile permutation; callers record each feature list and exit status.
prepublish-feature-check features manifest="Cargo.toml":
    TMPDIR="$HOME/tmp" CARGO_BUILD_JOBS=4 CARGO_TARGET_DIR="{{justfile_directory()}}/target" nice -n 19 cargo check --locked --manifest-path "{{manifest}}" -p jxl-encoder --lib --no-default-features --features "{{features}}"

# Validate retained benchmark bitstreams outside the timed process.
prepublish-perf-roundtrip tables manifest="Cargo.toml" side="both":
    TMPDIR="$HOME/tmp" CARGO_BUILD_JOBS=4 RAYON_NUM_THREADS=4 CARGO_TARGET_DIR="{{justfile_directory()}}/target" DJXL_PATH="{{justfile_directory()}}/.ci-libjxl/tools/djxl" JXL_AUDIT_PERF_TSVS="{{tables}}" JXL_AUDIT_PERF_SIDE="{{side}}" nice -n 19 cargo test --locked --manifest-path "{{manifest}}" -p jxl-encoder --features corpus-tests --test prepublish_perf_roundtrip -- --nocapture

# Wide integer bounds, primary/reference decode and admission ordering.
wide-integer-check manifest="Cargo.toml":
    TMPDIR="$HOME/tmp" CARGO_BUILD_JOBS=4 RAYON_NUM_THREADS=4 CARGO_TARGET_DIR="{{justfile_directory()}}/target" DJXL_PATH="{{justfile_directory()}}/.ci-libjxl/tools/djxl" JXL_ENCODER_OUTPUT_DIR="$HOME/tmp/jxl-wide-integer" nice -n 19 cargo test --locked --manifest-path "{{manifest}}" -p jxl-encoder --test it high_bit_depth_int -- --test-threads=1
    TMPDIR="$HOME/tmp" CARGO_BUILD_JOBS=4 CARGO_TARGET_DIR="{{justfile_directory()}}/target" nice -n 19 cargo test --locked --manifest-path "{{manifest}}" -p jxl-encoder --lib planar_admission_tests_95

# Fast-decode lossless preset: exact round trip through jxl-rs, jxl-oxide,
# libjxl v0.12 djxl and zenjxl-decoder (in-tree test, patched oxide fork).
fast-decode-check manifest="Cargo.toml":
    TMPDIR="$HOME/tmp" CARGO_BUILD_JOBS=4 RAYON_NUM_THREADS=4 CARGO_TARGET_DIR="{{justfile_directory()}}/target" DJXL_PATH="{{justfile_directory()}}/.ci-libjxl/tools/djxl" JXL_ENCODER_OUTPUT_DIR="$HOME/tmp/jxl-fast-decode" nice -n 19 cargo test --locked --manifest-path "{{manifest}}" --release -p jxl-encoder --test it fast_decode_lossless

# Released (unpatched) jxl-rs + jxl-oxide crates and djxl v0.12 against a
# source PNG: `just fast-decode-conformance source.png a.jxl b.jxl ...`
fast-decode-conformance source +files:
    CARGO_BUILD_JOBS=4 nice -n 19 cargo build --release --manifest-path "{{justfile_directory()}}/tools/decoder-conformance/Cargo.toml"
    DJXL_PATH="{{justfile_directory()}}/.ci-libjxl/tools/djxl" "{{justfile_directory()}}/tools/decoder-conformance/target/release/decoder-conformance" "{{source}}" {{files}}

# Replay every encoder failure recorded in issue 138 through 2026-11-02.
issue138-replay binary out corpus icc:
    python3 scripts/replay_issue138.py {{quote(binary)}} {{quote(out)}} {{quote(corpus)}} {{quote(icc)}}

issue138-retained-replay binary out corpus icc:
    python3 scripts/replay_issue138.py {{quote(binary)}} {{quote(out)}} {{quote(corpus)}} {{quote(icc)}} --retain

# Exact recorded cells; all source/encoded/reference artifacts remain retained.
conformance-replay binary cases out corpus icc:
    python3 scripts/replay_issue138.py {{quote(binary)}} {{quote(out)}} {{quote(corpus)}} {{quote(icc)}} --cases-tsv {{quote(cases)}} --retain

lossless-effort-ablate encoder decoder source out:
    TMPDIR="$HOME/tmp" uv run --with pillow python scripts/ablate_lossless_effort.py {{quote(encoder)}} {{quote(decoder)}} {{quote(source)}} {{quote(out)}}

compare-float-decodes out *args:
    TMPDIR="$HOME/tmp" uv run --with numpy python scripts/compare_float_decodes.py {{quote(out)}} {{args}}

issue138-ablate binary replay out corpus icc:
    python3 scripts/ablate_issue138.py {{quote(binary)}} {{quote(replay)}} {{quote(out)}} {{quote(corpus)}} {{quote(icc)}}

issue138-dc-ablate binary replay out corpus icc:
    python3 scripts/ablate_issue138.py {{quote(binary)}} {{quote(replay)}} {{quote(out)}} {{quote(corpus)}} {{quote(icc)}} --arms custom,dc_parity

issue138-loop-ablate binary replay out corpus icc:
    python3 scripts/ablate_issue138.py {{quote(binary)}} {{quote(replay)}} {{quote(out)}} {{quote(corpus)}} {{quote(icc)}} --arms original,butteraugli_loop,butteraugli_loop8,dc_parity_loop,dc_parity_loop8

issue138-strict-loop-ablate binary replay out corpus icc:
    python3 scripts/ablate_issue138.py {{quote(binary)}} {{quote(replay)}} {{quote(out)}} {{quote(corpus)}} {{quote(icc)}} --arms original,dc_parity_loop,libjxl_r1_loop,libjxl_r1_loop8

issue138-ablation-table ablations out:
    python3 scripts/summarize_issue138.py {{quote(ablations)}} {{quote(out)}}

issue138-vdp2-loop-ablate binary replay out corpus icc:
    python3 scripts/ablate_issue138.py {{quote(binary)}} {{quote(replay)}} {{quote(out)}} {{quote(corpus)}} {{quote(icc)}} --arms original,vdp2_loop,vdp2_loop8,dc_vdp2_loop

issue138-pq-pairs baseline candidate records out corpus icc:
    python3 scripts/check_issue138_pq.py {{quote(baseline)}} {{quote(candidate)}} {{quote(records)}} {{quote(out)}} {{quote(corpus)}} {{quote(icc)}}

pq-ramp-quality djxl before after out:
    python3 scripts/compare_pq_dc_ramp.py {{quote(djxl)}} {{quote(before)}} {{quote(after)}} {{quote(out)}}

pq-dc-contract manifest="Cargo.toml":
    cargo test --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --test animation_contract saturated_pq_keeps_dark_channels -- --nocapture

# Caller supplies the frozen sibling closure, v0.12 tools, flock and run-heavy.
pq-dc-validation manifest="Cargo.toml":
    cargo test --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --lib
    just animation-hdr-loop-check {{quote(manifest)}}
    just hdr-intensity-check {{quote(manifest)}}
    just hdr-loop-override-check {{quote(manifest)}}
    just pq-existing-hdr-check {{quote(manifest)}}
    cargo test --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert,__internals --test it hash_lock
    cargo test --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert,__internals --test it strategy_libjxl
    cargo test --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert,__internals --test it divergence_table_drift
    cargo clippy --locked --manifest-path {{quote(manifest)}} --workspace --all-targets -- -D warnings
    cargo clippy --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --no-default-features --features std,convenience --lib -- -D warnings
    cargo fmt -p jxl-encoder -- --check
    python3 -m unittest discover -s scripts -p 'test_jxl*.py'

# Existing HDR quality, signaling, float-depth and fine-quantization contracts.
pq-existing-hdr-check manifest="Cargo.toml":
    cargo test --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert,__internals --test it hdr_
    cargo test --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert,__internals --test it lossy_float_bitdepth
    cargo test --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert,__internals --test it low_effort_static_codes

gate-fixtures corpus:
    CODEC_CORPUS_DIR={{quote(corpus)}} bash scripts/hunt/fetch_imazen26_gate_files.sh

conformance-progress root:
    python3 scripts/conformance_progress.py {{quote(root)}}

conformance-journal-check out control="":
    python3 scripts/check_conformance_journal.py {{quote(out)}} {{if control == "" { "" } else { "--control " + quote(control) }}}

conformance-build manifest="tools/conformance-sweep/Cargo.toml":
    cargo build --locked --release --manifest-path {{quote(manifest)}}

conformance-recon-build manifest="tools/conformance-sweep/Cargo.toml":
    cargo build --locked --release --manifest-path {{quote(manifest)}} --features jxl-encoder/__internal_recon_hook

issue138-quant-trace binary out seed index corpus icc opts:
    mkdir {{quote(out)}}
    mkdir {{quote(out + "/quant-trace")}}
    JXL_AQDBG_DUMP={{quote(out + "/quant-trace")}} {{quote(binary)}} --images {{quote(corpus + "/CID22/CID22-512/validation")}} --images {{quote(corpus + "/gb82-sc")}} --icc {{quote(icc)}} --out {{quote(out + "/sweep")}} --seed {{seed}} --only {{index}} --threads 1 --retain --reference-study --study-max-pixels 100000000 --options {{quote(opts)}}

bitstream-parser-check:
    python3 -m unittest discover -s scripts -p 'test_jxl*.py'

issue138-cell binary out seed index corpus icc:
    {{quote(binary)}} --images {{quote(corpus + "/CID22/CID22-512/validation")}} --images {{quote(corpus + "/gb82-sc")}} --icc {{quote(icc)}} --out {{quote(out)}} --seed {{seed}} --only {{index}} --threads 1 --retain --reference-study --study-max-pixels 100000000

# Caller provides resource controls and verified CJXL_PATH/DJXL_PATH.
issue138-reference-check manifest="tools/conformance-sweep/Cargo.toml":
    cargo test --release --manifest-path {{quote(manifest)}}
    cargo clippy --manifest-path {{quote(manifest)}} --all-targets -- -D warnings

# Caller supplies pinned siblings, reference tools, and run-heavy controls.
gain-map-bundle-check manifest="Cargo.toml":
    cargo test --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features hdr-gainmap,jpeg-reencoding --lib
    cargo test --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features hdr-gainmap,jpeg-reencoding --test it jpeg_gainmap::
    cargo clippy --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features hdr-gainmap,jpeg-reencoding --lib --test it -- -D warnings

animation-hdr-loop-check manifest="Cargo.toml":
    cargo test --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features std,butteraugli-loop --test animation_contract
    cargo clippy --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --lib --test animation_contract -- -D warnings

hdr-loop-override-check manifest="Cargo.toml":
    cargo test --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --lib packed_reference_matches_strided_reconstruction
    cargo test --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features std,butteraugli-loop --test hdr_loop_override
    cargo test --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features std,butteraugli-loop --test it hdr_vdp2
    cargo clippy --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --lib --test hdr_loop_override -- -D warnings

hdr-intensity-check manifest="Cargo.toml":
    cargo test --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --test hdr_intensity --test hdr_metadata_validation -- --nocapture

# Caller supplies pinned siblings, v0.12 tools, flock and run-heavy controls.
subsampled-metadata-check manifest="Cargo.toml":
    cargo test --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features jpeg-reencoding,chroma-subsampling --test subsampled_odd_shapes --test subsampled_metadata
    cargo clippy --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features jpeg-reencoding,chroma-subsampling --lib --test subsampled_odd_shapes --test subsampled_metadata -- -D warnings

libjxl-cfl-dc-oracle source build out:
    c++ -std=c++17 -O1 -ffp-contract=off -no-pie -I{{quote(source)}} -I{{quote(build + "/lib/include")}} -I{{quote(source + "/third_party/highway")}} scripts/libjxl_cfl_dc.cc {{quote(build + "/lib/CMakeFiles/jxl_enc-obj.dir/jxl/enc_modular.cc.o")}} {{quote(build + "/lib/CMakeFiles/jxl_dec-obj.dir/jxl/quantizer.cc.o")}} {{quote(build + "/lib/CMakeFiles/jxl_dec-obj.dir/jxl/quant_weights.cc.o")}} {{quote(build + "/lib/CMakeFiles/jxl_dec-obj.dir/jxl/memory_manager_internal.cc.o")}} -Wl,--unresolved-symbols=ignore-all -o {{quote(out)}}

cfl-dc-reference oracle out:
    python3 scripts/generate_cfl_dc_reference.py {{quote(oracle)}} {{quote(out)}}

cfl-dc-check manifest="Cargo.toml":
    cargo test --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --lib dc_cfl_quantization_matches_reference_operation_order

# Linux object oracle from an unmodified libjxl v0.12 build.
libjxl-gradient-dc-oracle include object out:
    c++ -std=c++17 -O1 -no-pie -I{{quote(include)}} scripts/libjxl_gradient_dc.cc {{quote(object)}} -Wl,--unresolved-symbols=ignore-all -o {{quote(out)}}

gradient-dc-reference oracle out:
    python3 scripts/generate_gradient_dc_reference.py {{quote(oracle)}} {{quote(out)}}

gradient-dc-check manifest="Cargo.toml":
    cargo test --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --lib gradient_dc_matches_unmodified_libjxl_object

metric-domain-check manifest="Cargo.toml":
    cargo test --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __external_metric --test metric_color_domain --test external_metric
    cargo test --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --lib
    cargo test --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert,__internals --test it hash_lock
    cargo test --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert,__internals --test it strategy_libjxl
    cargo test --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert,__internals --test it divergence_table_drift
    cargo test --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --test animation_contract

metric-domain-lint manifest="Cargo.toml":
    cargo clippy --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --lib --test metric_color_domain --features __external_metric,ssim2-loop,zensim-loop -- -D warnings
    cargo clippy --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --no-default-features --features std,convenience --lib -- -D warnings

issue128-check manifest="Cargo.toml":
    cargo test --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert,__internals --test it icc_ratio::
    cargo test --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert,__internals --test it icc_mab_curves::

issue128-real-profiles binary out icc:
    {{quote(binary)}} --icc-regression --out {{quote(out)}} --icc {{quote(icc)}}

correctness-closure-check manifest="Cargo.toml": (metric-domain-check manifest) (issue128-check manifest) (metric-domain-lint manifest)

# Weekly conformance sweep (tools/conformance-sweep): random real-image
# encodes checked by djxl v0.12, upstream jxl-rs and zenjxl-decoder.
# `just conformance-sweep <out> <icc-dir> <images-or-dirs>... [--seed N] [--cells N] [--minutes N]`
# ICC profiles: scripts/fetch_icc_profiles.sh <icc-dir>.
conformance-sweep out icc +args:
    CARGO_BUILD_JOBS=4 nice -n 19 cargo build --release --manifest-path "{{justfile_directory()}}/tools/conformance-sweep/Cargo.toml"
    DJXL_PATH="${DJXL_PATH:-{{justfile_directory()}}/.ci-libjxl/tools/djxl}" nice -n 19 "{{justfile_directory()}}/tools/conformance-sweep/target/release/conformance-sweep" --out "{{out}}" --icc "{{icc}}" {{args}}

triage-hook-lint manifest="Cargo.toml":
    TMPDIR="$HOME/tmp" CARGO_BUILD_JOBS=4 CARGO_TARGET_DIR="{{justfile_directory()}}/target" nice -n 19 cargo clippy --locked --manifest-path "{{manifest}}" -p jxl-encoder --features __expert,__internal_recon_hook --test it --example rd_monotonicity_gate --example distance_targeting_probe -- -D warnings

prepublish-float manifest="Cargo.toml":
    TMPDIR="$HOME/tmp" CARGO_BUILD_JOBS=4 RAYON_NUM_THREADS=4 CARGO_TARGET_DIR="{{justfile_directory()}}/target" DJXL_PATH="{{justfile_directory()}}/.ci-libjxl/tools/djxl" nice -n 19 cargo test --locked --manifest-path "{{manifest}}" -p jxl-encoder --features __expert --test it forced_wp_float_extremes_match_rust_and_libjxl -- --nocapture

prepublish-perf base ours out *args:
    TMPDIR="$HOME/tmp" nice -n 19 python3 scripts/bench_lossless_ab.py --base "{{base}}" --ours "{{ours}}" --out "{{out}}" {{args}}

prepublish-perf-driver-check:
    nice -n 19 python3 -m unittest discover -s scripts -p test_bench_lossless_ab.py

# Byte-preserving strict/Zen cleanup checks. These are modules in the `it`
# binary, so nextest selects test names rather than nonexistent binary IDs.
libjxl-exact-cleanup-check label manifest="Cargo.toml":
    #!/usr/bin/env bash
    set -euo pipefail
    export CJXL_PATH="{{justfile_directory()}}/.ci-libjxl/tools/cjxl"
    export DJXL_PATH="{{justfile_directory()}}/.ci-libjxl/tools/djxl"
    export TMPDIR="$HOME/tmp" CARGO_BUILD_JOBS=4 CARGO_TARGET_DIR="{{justfile_directory()}}/target"
    log_dir="$HOME/tmp/jxl-exact-cleanup/{{label}}"
    mkdir -p "$log_dir"
    nice -n 19 cargo nextest run --manifest-path "{{manifest}}" --locked -p jxl-encoder --features __expert,__internals --test it --test-threads 4 -E 'test(strategy_libjxl_byte_lock) | test(hash_lock_features) | test(divergence_table_drift)' > "$log_dir/locks-and-drift.log" 2>&1
    rg 'Summary' "$log_dir/locks-and-drift.log"
    nice -n 19 cargo test --manifest-path "{{manifest}}" --locked -p jxl-encoder --lib -j 4 -- --test-threads=4 > "$log_dir/lib.log" 2>&1
    rg 'test result:' "$log_dir/lib.log"

libjxl-exact-cleanup-lint label manifest="Cargo.toml":
    #!/usr/bin/env bash
    set -euo pipefail
    export TMPDIR="$HOME/tmp" CARGO_BUILD_JOBS=4 CARGO_TARGET_DIR="{{justfile_directory()}}/target"
    log_dir="$HOME/tmp/jxl-exact-cleanup/{{label}}"
    mkdir -p "$log_dir"
    echo "Clippy log: $log_dir/clippy.log"
    nice -n 19 cargo clippy --manifest-path "{{manifest}}" --workspace --all-targets --locked -- -D warnings > "$log_dir/clippy.log" 2>&1

# Verify sectioned benchmark failure handling and persisted lossless samples.
sectioned-harness-check manifest="Cargo.toml":
    python3 -m unittest discover -s scripts -p test_sectioned_k_bar_cell.py
    TMPDIR="$HOME/tmp" CARGO_BUILD_JOBS=4 CARGO_TARGET_DIR="{{justfile_directory()}}/target" DJXL_PATH="{{justfile_directory()}}/.ci-libjxl/tools/djxl" nice -n 19 cargo test --locked --manifest-path "{{manifest}}" -p jxl-encoder --example sectioned_k_corpus --features std,parallel,profile-phases
    TMPDIR="$HOME/tmp" CARGO_BUILD_JOBS=4 CARGO_TARGET_DIR="{{justfile_directory()}}/target" nice -n 19 cargo clippy --locked --manifest-path "{{manifest}}" -p jxl-encoder --example sectioned_k_corpus --features std,parallel,profile-phases -- -D warnings

sectioned-harness-build manifest="Cargo.toml" profile="dev":
    TMPDIR="$HOME/tmp" CARGO_BUILD_JOBS=4 CARGO_TARGET_DIR="{{justfile_directory()}}/target" nice -n 19 cargo build --profile "{{profile}}" --locked --manifest-path "{{manifest}}" -p jxl-encoder --example sectioned_k_corpus --features std,parallel,profile-phases

# Lossless strategy resolution, exact decoded pixels and CLI/API agreement.
lossless-strategy-check manifest="Cargo.toml":
    TMPDIR="$HOME/tmp" CARGO_BUILD_JOBS=4 CARGO_TARGET_DIR="{{justfile_directory()}}/target" CJXL_PATH="{{justfile_directory()}}/.ci-libjxl/tools/cjxl" DJXL_PATH="{{justfile_directory()}}/.ci-libjxl/tools/djxl" nice -n 19 cargo test --locked --manifest-path "{{manifest}}" -p jxl-encoder --features __expert --lib lossless_strategy -- --nocapture
    TMPDIR="$HOME/tmp" CARGO_BUILD_JOBS=4 CARGO_TARGET_DIR="{{justfile_directory()}}/target" nice -n 19 cargo test --locked --manifest-path "{{manifest}}" -p jxl-encoder-cli --bin cjxl-rs --test strategy_flag

# All five WP modes: captured wire headers and real single/multi-group pixels.
forced-wp-check manifest="Cargo.toml" features="__expert":
    TMPDIR="$HOME/tmp" CARGO_BUILD_JOBS=4 CARGO_TARGET_DIR="{{justfile_directory()}}/target" CJXL_PATH="{{justfile_directory()}}/.ci-libjxl/tools/cjxl" DJXL_PATH="{{justfile_directory()}}/.ci-libjxl/tools/djxl" nice -n 19 cargo test --locked --manifest-path "{{manifest}}" -p jxl-encoder --features "{{features}}" --lib forced_wp_tests -- --test-threads=1

# The oracle records a source commit and retains every encoded candidate.
lossless-oracle-build manifest="Cargo.toml" profile="dev":
    JXL_BENCH_COMMIT="$(jj log --no-graph -r @ -T 'commit_id')" TMPDIR="$HOME/tmp" CARGO_BUILD_JOBS=4 CARGO_TARGET_DIR="{{justfile_directory()}}/target" nice -n 19 cargo build --profile "{{profile}}" --locked --manifest-path "{{manifest}}" -p jxl-encoder --example lossless_pareto_calibrate --features __expert,parallel,learned-admission

lossless-oracle-check manifest="Cargo.toml":
    TMPDIR="$HOME/tmp" CARGO_BUILD_JOBS=4 CARGO_TARGET_DIR="{{justfile_directory()}}/target" DJXL_PATH="{{justfile_directory()}}/.ci-libjxl/tools/djxl" nice -n 19 cargo test --locked --manifest-path "{{manifest}}" -p jxl-encoder --example lossless_pareto_calibrate --features __expert,parallel,learned-admission
    TMPDIR="$HOME/tmp" CARGO_BUILD_JOBS=4 CARGO_TARGET_DIR="{{justfile_directory()}}/target" nice -n 19 cargo clippy --locked --manifest-path "{{manifest}}" -p jxl-encoder --example lossless_pareto_calibrate --features __expert,parallel,learned-admission -- -D warnings

lossless-oracle-cli-check binary="target/debug/examples/lossless_pareto_calibrate":
    LOSSLESS_ORACLE_PROBE="{{binary}}" nice -n 19 python3 -m unittest discover -s scripts -p test_lossless_oracle.py

# Reproduce the sectioned wall bar, optionally interleaving a baseline binary.
sectioned-bar out image reps="3" baseline="" efforts="7 9" threads="1 8":
    TMPDIR="$HOME/tmp" CJXL_PATH="{{justfile_directory()}}/.ci-libjxl/tools/cjxl" DJXL_PATH="{{justfile_directory()}}/.ci-libjxl/tools/djxl" SECTIONED_K_BASELINE_PROBE="{{baseline}}" nice -n 19 bash scripts/sectioned_k_bar_cell.sh "{{out}}" "{{image}}" "{{reps}}" "{{efforts}}" "{{threads}}"

# Persist real-input LZ77 harness smoke outputs and verify every recorded hash.
lz77-artifact-check label:
    #!/usr/bin/env bash
    set -euo pipefail
    export TMPDIR="$HOME/tmp" CARGO_BUILD_JOBS=4 RAYON_NUM_THREADS=4
    root="$HOME/tmp/jxl-backlog/{{label}}"
    mkdir -p "$root/input"
    cp jxl-encoder/tests/images/frymire-srgb.png "$root/input/"
    jj log --no-graph -r @ -T 'commit_id ++ "\n"' > "$root/build.meta"
    shasum -a 256 jxl-encoder/examples/lz77_hash_ab.rs scripts/lz77_hash_ab_join.py >> "$root/build.meta"
    nice -n 19 python3 scripts/test_lz77_hash_ab_join.py > "$root/analyzer-tests.log" 2>&1
    nice -n 19 cargo build --locked -p jxl-encoder --release --example lz77_hash_ab -j 4 > "$root/build.log" 2>&1
    for size in 64 259; do
        nice -n 19 target/release/examples/lz77_hash_ab "$root/input" "$root/$size.tsv" --images 1 --size "$size" --efforts 8 > "$root/$size.log" 2>&1
        nice -n 19 python3 scripts/lz77_hash_ab_join.py "$root/$size.tsv" "$root/$size.tsv" --verify-artifacts > "$root/$size-verification.log" 2>&1
        cat "$root/$size-verification.log"
    done

# Run RD regression test (encodes 6 images at d=0.25, d=0.5, d=1.0)
rd-regression:
    cargo test -p jxl-encoder --test it clic2025::test_rd_regression -- --ignored --nocapture

# Run high-distance RD regression test (d=2.0 and d=3.0, exercises DCT32x32/DCT64x64)
rd-regression-hd:
    cargo test -p jxl-encoder --test it clic2025::test_rd_regression_high_distance -- --ignored --nocapture

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

# W44-179: re-run the W44-170 comprehensive sweep with output-prefix tagged
# w44_179. Same corpus, same params — only the output file naming differs so
# the new run can be diffed against the W44-170 baseline TSVs.
w44-179-rerun:
    cargo run -p jxl-encoder --release --features 'parallel butteraugli-loop ssim2-loop' \
        --example w44_170_cjxl_step025_sweep -- \
        --corpus-manifest benchmarks/corpora/w44_170_varied_corpus.tsv \
        --output-prefix benchmarks/cjxl_step025_w44_179 \
        --strategies zenjxl,libjxl
    python3 benchmarks/scripts/w44_170_analyze.py \
        --zenjxl $(ls -1t benchmarks/cjxl_step025_w44_179_zenjxl_*.tsv | head -1) \
        --libjxl $(ls -1t benchmarks/cjxl_step025_w44_179_libjxl_*.tsv | head -1) \
        --output-md benchmarks/w44_179_analysis_$(date +%Y-%m-%d).md \
        --chart-dir benchmarks/charts \
        --chart-tag w44_179
    python3 benchmarks/scripts/w44_179_compare_vs_baseline.py \
        --baseline-zenjxl $(ls -1t benchmarks/cjxl_step025_zenjxl_*.tsv | grep -v w44_179 | head -1) \
        --baseline-libjxl $(ls -1t benchmarks/cjxl_step025_libjxl_*.tsv | grep -v w44_179 | head -1) \
        --new-zenjxl $(ls -1t benchmarks/cjxl_step025_w44_179_zenjxl_*.tsv | head -1) \
        --new-libjxl $(ls -1t benchmarks/cjxl_step025_w44_179_libjxl_*.tsv | head -1) \
        --output-md benchmarks/w44_179_vs_w44_170_comparison_$(date +%Y-%m-%d).md \
        --output-tsv-prefix benchmarks/w44_179_vs_w44_170 \
        --top-n 20

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

# Regenerate hash lock sidecar file after intentional encoding changes.
# NOTE: manual platform-variant lines (name@arch + repeated-name libm
# variants, issue #70) are DESTROYED by the rm — re-add them from git
# after regen (see the sidecar comment in hash_lock_features.rs).
update-hashes:
    rm -f jxl-encoder/tests/hash_lock_expected.txt
    UPDATE_HASHES=1 cargo test --test it hash_lock_features -- --test-threads=1

# Effort-ladder bench (#45, 2026-08-28): bytes + wall per tier e9..=e13 on
# the imazen-26 gate fixtures (fetch: CODEC_CORPUS_DIR=... scripts/hunt/
# fetch_imazen26_gate_files.sh). EFFORTS/SIZES/MODES/FIXTURES/THREADS env
# knobs — see the example's header. 4 MP e11-e13 lossless cells take minutes.
effort-ladder-bench:
    nice -n 19 cargo run --release -p jxl-encoder --features parallel --example effort_ladder_shift

# Compare quality vs cjxl (uses committed reference CSV, ~2 min)
quality-compare:
    cargo test -p jxl-encoder --test it quality_compare --release -- --ignored --nocapture

# Quick comparison: 10 CLIC + 5 CID22 + all screenshots (~5 min)
rd-compare-quick:
    cargo build --release -p jxl-encoder-cli
    bash scripts/measure_cjxl_rs.sh --quick
    python3 scripts/rd_report.py

# Compare lossless compression vs cjxl (CSV-backed, ~1 min)
lossless-compare:
    cargo test -p jxl-encoder --test it lossless_compare --release -- --ignored --nocapture

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

# Native ARM comparisons without target-cpu=native. Preserve full logs.
arm-kernel-tiers-macos group="":
    mkdir -p "$HOME/tmp"
    CARGO_BUILD_JOBS=4 RAYON_NUM_THREADS=4 OMP_NUM_THREADS=4 TMPDIR="$HOME/tmp" nice -n 19 cargo bench --locked -p jxl-encoder-simd --bench kernel_tiers -- --group='{{group}}' --format=llm > "$HOME/tmp/jxl-encoder-kernel-tiers.log" 2>&1

arm-encode-tiers-macos:
    #!/usr/bin/env bash
    set -euo pipefail
    stamp="$(date -u +%Y%m%dT%H%M%SZ)"
    artifact_dir="$HOME/tmp/jxl-encoder-arm-audit-$stamp"
    log="$HOME/tmp/jxl-encoder-full-tiers-$stamp.log"
    mkdir -p "$artifact_dir"
    CODEC_BENCH_ARTIFACT_DIR="$artifact_dir" CARGO_BUILD_JOBS=4 RAYON_NUM_THREADS=4 OMP_NUM_THREADS=4 TMPDIR="$HOME/tmp" \
      nice -n 19 cargo bench -p jxl-encoder --features _dev --bench tier_isolation -- --cpu-policy=unrestricted --format=llm > "$log" 2>&1
    printf 'Benchmark log: %s\nBitstreams: %s\n' "$log" "$artifact_dir"

# Issue 103: local validation with explicit reference tools and caller filters.
issue103-test *args:
    CJXL_PATH="{{justfile_directory()}}/.ci-libjxl/tools/cjxl" DJXL_PATH="{{justfile_directory()}}/.ci-libjxl/tools/djxl" nice -n 19 cargo test -p jxl-encoder -j 4 {{args}}

# Set IMGS/CROP/REPS and ARTIFACT_DIR in the caller; keep the full TSV and stderr.
resample-kernel-cost:
    nice -n 19 cargo run -p jxl-encoder -j 4 --release --features __internals --example resample_kernel_cost

resample-kernel-build:
    nice -n 19 cargo build -p jxl-encoder -j 4 --release --features __internals --example resample_kernel_cost

resample-kernel-check:
    nice -n 19 cargo clippy -p jxl-encoder --features __internals --example resample_kernel_cost -j 4 -- -D warnings

resample-kernel-profile:
    nice -n 19 cargo run -p jxl-encoder -j 4 --release --features '__internals profile-phases' --example resample_kernel_cost

# Run after the default suite; each command is serialized and reference-pinned.
issue103-resampling-gates:
    just issue103-test --features __expert,__internals --test it strategy_libjxl_byte_lock
    just issue103-test --features __expert,__internals --test it divergence_table_drift
    just issue103-test --test it issue_101_auto_resample_odd_dims_decodes_via_djxl_to_source_size -- --ignored
    just issue103-test --test it clic2025::test_rd_regression -- --ignored --nocapture

resample-kernel-ab *args:
    nice -n 19 python3 scripts/bench_resample_kernels.py {{args}}

# IMG, ARTIFACT_DIR, CROP, EFFORTS, DISTANCES, and optional ITERS come from caller.
distance-targeting-probe:
    JXL_PROBE_BUILD_COMMIT="$(jj log --no-graph -r @ -T commit_id)" DJXL_PATH="{{justfile_directory()}}/.ci-libjxl/tools/djxl" nice -n 19 cargo run -p jxl-encoder -j 4 --release --example distance_targeting_probe

distance-targeting-check:
    nice -n 19 cargo clippy -p jxl-encoder --example distance_targeting_probe -j 4 -- -D warnings

# Same probe with captured final internal pixels for decoder-drift diagnosis.
distance-targeting-recon:
    JXL_PROBE_BUILD_COMMIT="$(jj log --no-graph -r @ -T commit_id)" DJXL_PATH="{{justfile_directory()}}/.ci-libjxl/tools/djxl" nice -n 19 cargo run -p jxl-encoder -j 4 --release --features __internal_recon_hook --example distance_targeting_probe

qfseed-lift-ab *args:
    nice -n 19 python3 scripts/qfseed_lift_ab.py {{args}}

# Build distance-targeting-recon first. Each cell persists metadata and all pixels.
distance-recon-strategies:
    #!/usr/bin/env bash
    set -euo pipefail
    for crop in 64 512; do
        for strategy in 0 5 6 7 8 9 12 13 14 15; do
            echo "reconstruction crop=$crop strategy=$strategy"
            CROP="$crop" FORCE_STRATEGY="$strategy" EPF=0 GABORISH=false EFFORTS=8 DISTANCES=4 ITERS=1 nice -n 19 target/release/examples/distance_targeting_probe
        done
    done

# Compare persisted targeting runs; input paths may be individual TSVs or directories.
qfseed-lift-analyze *args:
    python3 scripts/qfseed_lift_ab_analyze.py {{args}}

issue103-fmt:
    nice -n19 cargo fmt -p jxl-encoder -p jxl-encoder-simd

# Process isolation keeps the reconstruction capture from racing other encodes.
issue103-reconstruction-parity:
    just issue103-test --features __internal_recon_hook --test reconstruction_decoder_parity -- --nocapture

# Explicit corpus caller: requires CODEC_CORPUS_DIR and libjxl v0.12 tools.
canonicalization-test:
    TMPDIR="{{env_var('HOME')}}/tmp" DJXL_PATH="{{justfile_directory()}}/.ci-libjxl/tools/djxl" nice -n 19 cargo test -p jxl-encoder --lib --features corpus-tests -j 4 api::canonicalize -- --nocapture

canonicalization-check:
    nice -n 19 cargo clippy --workspace --all-targets -j 4 -- -D warnings

# Admission is checked before streaming image planes exist; no corpus needed.
streaming-admission-test *args:
    TMPDIR="$HOME/tmp" nice -n 19 cargo test -p jxl-encoder -j 4 --lib streaming_ {{args}}

# Caller supplies corpus and artifact directories outside the repository.
fuzz-encoder target corpus artifacts *args:
    TMPDIR="$HOME/tmp" CARGO_BUILD_JOBS=4 nice -n 19 cargo +nightly fuzz run --codegen-units 16 {{target}} {{corpus}} -- -artifact_prefix={{artifacts}}/ -rss_limit_mb=2048 {{args}}

# Linux x86 runtime used by nightly; Rust sanitizer and coverage flags stay intact.
fuzz-encoder-popcnt target corpus artifacts *args:
    grep -qw popcnt /proc/cpuinfo
    CXX="{{justfile_directory()}}/scripts/fuzz_cxx.py" CXXFLAGS="${CXXFLAGS:-} -mpopcnt" just fuzz-encoder {{target}} {{corpus}} {{artifacts}} {{args}}

fuzz-runtime-check:
    python3 -m unittest discover -s scripts -p test_fuzz_cxx.py

fuzz-replay input output manifest="Cargo.toml":
    cargo run --locked --release --manifest-path {{quote(manifest)}} -p jxl-encoder --example streaming_fuzz_replay -- {{quote(input)}} {{quote(output)}}

fuzz-regression:
    DJXL_PATH="{{justfile_directory()}}/.ci-libjxl/tools/djxl" TMPDIR="$HOME/tmp" nice -n 19 cargo test -p jxl-encoder -j 4 --test fuzz_regression

fuzz-build:
    TMPDIR="$HOME/tmp" CARGO_BUILD_JOBS=4 nice -n 19 cargo +nightly fuzz build --codegen-units 16

fuzz-seed manifest output:
    TMPDIR="$HOME/tmp" nice -n 19 uv run --with pillow python scripts/seed_encoder_fuzz.py {{manifest}} {{output}} --build-commit "$(jj log --no-graph -r @ -T commit_id)"

production-resources manifest="Cargo.toml":
    TMPDIR="$HOME/tmp" CARGO_TARGET_DIR="{{justfile_directory()}}/target" RAYON_NUM_THREADS=4 DJXL_PATH="{{justfile_directory()}}/.ci-libjxl/tools/djxl" CARGO_BUILD_JOBS=4 nice -n 19 cargo test --locked --manifest-path "{{manifest}}" -p jxl-encoder --features corpus-tests,parallel --test production_resources -- --test-threads=1 --nocapture

# Local release checks preserve the package error as a release blocker.
release-semver *args:
    TMPDIR="$HOME/tmp" CARGO_BUILD_JOBS=4 nice -n 19 cargo semver-checks -p jxl-encoder --only-explicit-features --features std,parallel,butteraugli-loop {{args}}

# macOS process RSS; keep encoded bytes and full time output per cell.
production-clippy manifest="Cargo.toml":
    TMPDIR="$HOME/tmp" CARGO_BUILD_JOBS=4 CARGO_TARGET_DIR="{{justfile_directory()}}/target" nice -n 19 cargo clippy --locked --manifest-path "{{manifest}}" --workspace --all-targets -- -D warnings

# Validate dependency changes against the source closure prepared for CI.
development-dependency-tests manifest="Cargo.toml":
    TMPDIR="$HOME/tmp" CARGO_BUILD_JOBS=4 RAYON_NUM_THREADS=4 RUST_TEST_THREADS=4 CARGO_TARGET_DIR="{{justfile_directory()}}/target" nice -n 19 cargo test --locked --manifest-path "{{manifest}}" -p zenjxl-tuning-runner --all-targets

# Default workspace validation using the same pinned closure as CI.
production-workspace-tests manifest="Cargo.toml":
    TMPDIR="$HOME/tmp" CARGO_BUILD_JOBS=4 RAYON_NUM_THREADS=4 RUST_TEST_THREADS=4 CJXL_PATH="{{justfile_directory()}}/.ci-libjxl/tools/cjxl" DJXL_PATH="{{justfile_directory()}}/.ci-libjxl/tools/djxl" CARGO_TARGET_DIR="{{justfile_directory()}}/target" nice -n 19 cargo test --locked --manifest-path "{{manifest}}" --workspace --all-targets

# Match the optional zensim CI lane against a pinned source closure.
zensim-dependency-validation manifest="Cargo.toml":
    TMPDIR="$HOME/tmp" CARGO_BUILD_JOBS=4 CARGO_TARGET_DIR="{{justfile_directory()}}/target" nice -n 19 cargo clippy --locked --manifest-path "{{manifest}}" --workspace --all-targets --features zensim-loop -- -D warnings
    TMPDIR="$HOME/tmp" CARGO_BUILD_JOBS=4 RAYON_NUM_THREADS=4 RUST_TEST_THREADS=4 CARGO_TARGET_DIR="{{justfile_directory()}}/target" nice -n 19 cargo test --locked --manifest-path "{{manifest}}" -p jxl-encoder --features zensim-loop,parallel --test zensim_attr_smoke --test zensim_h_arms_smoke

# Caller supplies CODEC_CORPUS_DIR with gb82-sc/imac_g3.png; missing input fails.
empty-modular-compatibility manifest="Cargo.toml":
    python3 -m unittest discover -s scripts -p 'test_jxl*.py'
    TMPDIR="$HOME/tmp" RAYON_NUM_THREADS=4 RUST_TEST_THREADS=4 CARGO_TARGET_DIR="{{justfile_directory()}}/target" just issue103-test --manifest-path "{{manifest}}" --locked --test it empty_modular_section_roundtrip -- --include-ignored --nocapture

resource-build manifest="Cargo.toml":
    TMPDIR="$HOME/tmp" CARGO_BUILD_JOBS=4 CARGO_TARGET_DIR="{{justfile_directory()}}/target" nice -n 19 cargo build --locked --manifest-path "{{manifest}}" -p jxl-encoder --release --features std,parallel --example mem_grid_probe

resource-cell input mode effort threads output budget="default":
    mkdir -p "{{output}}"
    MEM_PROBE_OUT="{{output}}/encoded.jxl" nice -n 19 /usr/bin/time -l target/release/examples/mem_grid_probe "{{input}}" {{mode}} {{effort}} 4 {{threads}} {{budget}} > "{{output}}/encode.log" 2>&1
    nice -n 19 target/release/examples/mem_grid_probe --decode "{{output}}/encoded.jxl" "{{input}}" {{mode}} > "{{output}}/jxl-rs.log" 2>&1
    nice -n 19 .ci-libjxl/tools/djxl "{{output}}/encoded.jxl" --disable_output --num_threads=1 > "{{output}}/djxl.log" 2>&1

release-semver-json current baseline *args:
    TMPDIR="$HOME/tmp" CARGO_BUILD_JOBS=4 nice -n 19 cargo semver-checks --current-rustdoc "{{current}}" --baseline-rustdoc "{{baseline}}" {{args}}

# The baseline supplies the exact cell grid and expected bitstream hashes.
resource-verify baseline inputs output commit:
    nice -n 19 python3 scripts/verify_resource_cells.py "{{baseline}}" "{{inputs}}" "{{output}}" --build-commit "{{commit}}"

# Corroborated IQA inversions fail; bytes and unequal-scope timings are advisory.
# A new output path prevents accidentally replacing historical measurements.
rd-monotonicity corpus out manifest="Cargo.toml" *args:
    JXL_PROBE_BUILD_COMMIT="$(jj log --no-graph -r @ -T 'commit_id')" CJXL_PATH="{{justfile_directory()}}/.ci-libjxl/tools/cjxl" DJXL_PATH="{{justfile_directory()}}/.ci-libjxl/tools/djxl" TMPDIR="$HOME/tmp" CARGO_BUILD_JOBS=4 RAYON_NUM_THREADS=4 CARGO_TARGET_DIR="{{justfile_directory()}}/target" nice -n 19 cargo run --locked --manifest-path "{{manifest}}" -p jxl-encoder --release --example rd_monotonicity_gate -- "{{corpus}}" "{{out}}" {{args}}

rd-monotonicity-check manifest="Cargo.toml":
    TMPDIR="$HOME/tmp" CARGO_BUILD_JOBS=4 CARGO_TARGET_DIR="{{justfile_directory()}}/target" nice -n 19 cargo test --locked --manifest-path "{{manifest}}" -p jxl-encoder --example rd_monotonicity_gate
    TMPDIR="$HOME/tmp" CARGO_BUILD_JOBS=4 CARGO_TARGET_DIR="{{justfile_directory()}}/target" nice -n 19 cargo clippy --locked --manifest-path "{{manifest}}" -p jxl-encoder --example rd_monotonicity_gate -- -D warnings

rd-monotonicity-driver-check binary="target/release/examples/rd_monotonicity_gate":
    RD_MONOTONICITY_PROBE="{{binary}}" CJXL_PATH="{{justfile_directory()}}/.ci-libjxl/tools/cjxl" DJXL_PATH="{{justfile_directory()}}/.ci-libjxl/tools/djxl" RAYON_NUM_THREADS=4 nice -n 19 python3 -m unittest discover -s scripts -p test_rd_monotonicity_gate.py

# Fully render progressive small/global and large/group alpha through both decoders.
progressive-extras-check label:
    #!/usr/bin/env bash
    set -euo pipefail
    export CJXL_PATH="{{justfile_directory()}}/.ci-libjxl/tools/cjxl"
    export DJXL_PATH="{{justfile_directory()}}/.ci-libjxl/tools/djxl"
    export TMPDIR="$HOME/tmp" CARGO_BUILD_JOBS=4 RAYON_NUM_THREADS=4
    mkdir -p "$HOME/tmp/jxl-backlog"
    nice -n 19 cargo test --locked -p jxl-encoder --features __expert --test it progressive_extras_preserve_alpha_in_both_decoders -- --nocapture > "$HOME/tmp/jxl-backlog/progressive-{{label}}.log" 2>&1
    rg 'test result:' "$HOME/tmp/jxl-backlog/progressive-{{label}}.log"

jpeg-orientation-check label:
    #!/usr/bin/env bash
    set -euo pipefail
    export CJXL_PATH="{{justfile_directory()}}/.ci-libjxl/tools/cjxl"
    export DJXL_PATH="{{justfile_directory()}}/.ci-libjxl/tools/djxl"
    export TMPDIR="$HOME/tmp" CARGO_BUILD_JOBS=4 RAYON_NUM_THREADS=4
    mkdir -p "$HOME/tmp/jxl-backlog"
    nice -n 19 cargo test --locked -p jxl-encoder --features jpeg-reencoding --test it jpeg_exif_orientation_preserves_display_and_reconstruction -- --nocapture > "$HOME/tmp/jxl-backlog/jpeg-orientation-{{label}}.log" 2>&1
    rg 'test result:' "$HOME/tmp/jxl-backlog/jpeg-orientation-{{label}}.log"

# Strict AC-metadata contexts must match the tree consumed by all decoders.
libjxl-acmeta-check label:
    #!/usr/bin/env bash
    set -euo pipefail
    export CJXL_PATH="$PWD/.ci-libjxl/tools/cjxl" DJXL_PATH="$PWD/.ci-libjxl/tools/djxl"
    export TMPDIR="$HOME/tmp" CARGO_BUILD_JOBS=4 RAYON_NUM_THREADS=4
    mkdir -p "$HOME/tmp/jxl-backlog"
    nice -n 19 cargo test --locked -p jxl-encoder --lib ac_meta_epf_token_contexts_match_serialized_tree > "$HOME/tmp/jxl-backlog/acmeta-unit-{{label}}.log" 2>&1
    nice -n 19 cargo test --locked -p jxl-encoder --features __expert --test it strict_ac_metadata_contexts_render_in_both_decoders -- --nocapture > "$HOME/tmp/jxl-backlog/acmeta-render-{{label}}.log" 2>&1
    rg 'test result:' "$HOME/tmp/jxl-backlog/acmeta-"*"-{{label}}.log"

# Compile the differential cost oracle against the pinned, unmodified libjxl.
libjxl-estimate-cost-oracle:
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p "$HOME/tmp/jxl-backlog"
    nice -n 19 c++ -std=c++17 -O2 -DNDEBUG -I.ci-libjxl/source -I.ci-libjxl/source/third_party/highway -I.ci-libjxl/source/lib/include -I.ci-libjxl/build/lib/include scripts/libjxl_estimate_cost_oracle/ref.cc .ci-libjxl/build/lib/libjxl.a .ci-libjxl/build/lib/libjxl_cms.a .ci-libjxl/build/third_party/highway/libhwy.a .ci-libjxl/build/third_party/brotli/libbrotlidec.a .ci-libjxl/build/third_party/brotli/libbrotlienc.a .ci-libjxl/build/third_party/brotli/libbrotlicommon.a -o "$HOME/tmp/jxl-backlog/estimate-cost-ref" > "$HOME/tmp/jxl-backlog/estimate-cost-build.log" 2>&1

libjxl-extras-cost-check label:
    #!/usr/bin/env bash
    set -euo pipefail
    export CJXL_PATH="$PWD/.ci-libjxl/tools/cjxl" DJXL_PATH="$PWD/.ci-libjxl/tools/djxl"
    export TMPDIR="$HOME/tmp" CARGO_BUILD_JOBS=4 RAYON_NUM_THREADS=4
    mkdir -p "$HOME/tmp/jxl-backlog"
    nice -n 19 cargo test --locked -p jxl-encoder --features __expert --test it strict_palette_cost_revert_preserves_alpha_in_both_decoders -- --nocapture > "$HOME/tmp/jxl-backlog/part-24-roundtrip-{{label}}.log" 2>&1
    rg 'test result:' "$HOME/tmp/jxl-backlog/part-24-roundtrip-{{label}}.log"

# JPEG restart markers after the final MCU must survive JBRD reconstruction.
jpeg-restart-check label:
    #!/usr/bin/env bash
    set -euo pipefail
    export CJXL_PATH="$PWD/.ci-libjxl/tools/cjxl" DJXL_PATH="$PWD/.ci-libjxl/tools/djxl"
    export TMPDIR="$HOME/tmp" CARGO_BUILD_JOBS=4 RAYON_NUM_THREADS=4
    mkdir -p "$HOME/tmp/jxl-backlog"
    nice -n 19 cargo test --locked -p jxl-encoder --features jpeg-reencoding --lib entropy_scan_preserves_terminal_restart_markers > "$HOME/tmp/jxl-backlog/jpeg120-parser-{{label}}.log" 2>&1
    nice -n 19 cargo test --locked -p jxl-encoder --features jpeg-reencoding --test it jpeg_terminal_restart_markers_roundtrip -- --nocapture > "$HOME/tmp/jxl-backlog/jpeg120-render-{{label}}.log" 2>&1
    rg 'test result:' "$HOME/tmp/jxl-backlog/jpeg120-"*"-{{label}}.log"

# Camera originals from tests/fixtures/jpeg_restart_corpus.tsv plus local fixtures.
jpeg-restart-corpus-check label corpus:
    #!/usr/bin/env bash
    set -euo pipefail
    export DJXL_PATH="$PWD/.ci-libjxl/tools/djxl" JBRD_CONFORMANCE_REFERENCE=1
    export TMPDIR="$HOME/tmp" CARGO_BUILD_JOBS=4 RAYON_NUM_THREADS=4
    export JBRD_CONFORMANCE_CORPUS="{{corpus}}"
    export JBRD_CONFORMANCE_ARTIFACTS="$HOME/tmp/jxl-backlog/jpeg120-{{label}}-artifacts"
    mkdir -p "$HOME/tmp/jxl-backlog"
    nice -n 19 cargo test --locked -p jxl-encoder --features jpeg-reencoding --test it jbrd_roundtrip_conformance -- --nocapture > "$HOME/tmp/jxl-backlog/jpeg120-corpus-{{label}}.log" 2>&1
    rg 'test result:' "$HOME/tmp/jxl-backlog/jpeg120-corpus-{{label}}.log"

# JPEG CfL reference rendering and reconstruction across 64-pixel tile boundaries.
jpeg-cfl-check label:
    #!/usr/bin/env bash
    set -euo pipefail
    export CJXL_PATH="$PWD/.ci-libjxl/tools/cjxl" DJXL_PATH="$PWD/.ci-libjxl/tools/djxl"
    export TMPDIR="$HOME/tmp" CARGO_BUILD_JOBS=4 RAYON_NUM_THREADS=4
    mkdir -p "$HOME/tmp/jxl-backlog"
    nice -n 19 cargo test --locked -p jxl-encoder --features jpeg-reencoding --test it jpeg_cfl_reference_ -- --nocapture > "$HOME/tmp/jxl-backlog/jpeg-cfl-{{label}}.log" 2>&1
    rg 'test result:' "$HOME/tmp/jxl-backlog/jpeg-cfl-{{label}}.log"

# Attribute a recompressed JPEG's JPEG XL render vs its JPEG decode: dumps the
# coefficients, transcodes with ours and libjxl v0.12 (CfL on/off), renders via
# djxl, checks render parity + JBRD, then runs scripts/jpeg_transcode_render_model.py.
# The JPEG decode reference is ImageMagick (libjpeg-turbo). Tools: CJXL_PATH /
# DJXL_PATH override .ci-libjxl/tools.
jpeg-render-attribution label jpeg effort="7":
    #!/usr/bin/env bash
    set -euo pipefail
    cjxl="${CJXL_PATH:-$PWD/.ci-libjxl/tools/cjxl}"
    djxl="${DJXL_PATH:-$PWD/.ci-libjxl/tools/djxl}"
    out="$HOME/tmp/jxl-backlog/jpeg-render-{{label}}"
    mkdir -p "$out"
    export TMPDIR="$HOME/tmp" CARGO_BUILD_JOBS=4
    nice -n 19 cargo run --locked --release -p jxl-encoder --features jpeg-reencoding --example jpeg_coeff_dump -- "{{jpeg}}" "$out/input.coef"
    nice -n 19 cargo run --locked --release -p jxl-encoder-cli --features jpeg-reencoding -- "{{jpeg}}" "$out/ours.jxl" -e {{effort}} --quiet
    "$cjxl" --lossless_jpeg=1 -e {{effort}} --quiet "{{jpeg}}" "$out/libjxl.jxl"
    "$cjxl" --lossless_jpeg=1 --jpeg_reconstruction_cfl=0 -e {{effort}} --quiet "{{jpeg}}" "$out/libjxl-nocfl.jxl"
    for n in ours libjxl libjxl-nocfl; do
        "$djxl" --quiet "$out/$n.jxl" "$out/$n.ppm"
        magick "$out/$n.ppm" -depth 8 "rgb:$out/$n.rgb8"
    done
    cmp "$out/ours.rgb8" "$out/libjxl.rgb8" && echo "render: ours == libjxl v0.12"
    "$djxl" --quiet "$out/ours.jxl" "$out/reconstructed.jpg"
    cmp "$out/reconstructed.jpg" "{{jpeg}}" && echo "djxl JBRD: byte-exact"
    magick "{{jpeg}}" -depth 8 "rgb:$out/jpeg.rgb8"
    python3 scripts/jpeg_transcode_render_model.py "$out/input.coef" "$out/jpeg.rgb8" --render-cfl "$out/ours.rgb8" --render-nocfl "$out/libjxl-nocfl.rgb8" --label "{{label}}" | tee "$out/model.tsv"

# ISO gain-map container, exact JPEG reconstruction, and independent pixel checks.
jpeg-gainmap-check label:
    #!/usr/bin/env bash
    set -euo pipefail
    export CJXL_PATH="$PWD/.ci-libjxl/tools/cjxl" DJXL_PATH="$PWD/.ci-libjxl/tools/djxl"
    export TMPDIR="$HOME/tmp" CARGO_BUILD_JOBS=4 RAYON_NUM_THREADS=4
    mkdir -p "$HOME/tmp/jxl-backlog"
    nice -n 19 cargo test --locked -p jxl-encoder --features jpeg-reencoding --test it iso_gain_map_ -- --nocapture > "$HOME/tmp/jxl-backlog/jpeg122-{{label}}.log" 2>&1
    rg 'test result:' "$HOME/tmp/jxl-backlog/jpeg122-{{label}}.log"

jpeg-gainmap-corpus-check label corpus:
    #!/usr/bin/env bash
    set -euo pipefail
    export CJXL_PATH="$PWD/.ci-libjxl/tools/cjxl" DJXL_PATH="$PWD/.ci-libjxl/tools/djxl"
    export TMPDIR="$HOME/tmp" CARGO_BUILD_JOBS=4 RAYON_NUM_THREADS=4
    export JPEG_GAINMAP_CORPUS="{{corpus}}"
    export JPEG_GAINMAP_ARTIFACTS="$HOME/tmp/jxl-backlog/jpeg122-{{label}}-artifacts"
    mkdir -p "$HOME/tmp/jxl-backlog"
    nice -n 19 cargo test --locked -p jxl-encoder --features jpeg-reencoding,corpus-tests --test it iso_gain_map_camera_corpus -- --nocapture > "$HOME/tmp/jxl-backlog/jpeg122-{{label}}.log" 2>&1
    rg 'test result:' "$HOME/tmp/jxl-backlog/jpeg122-{{label}}.log"

# Provision every legacy JPEG fixture through codec-corpus and libjpeg-turbo.
jpeg-legacy-check label:
    #!/usr/bin/env bash
    set -euo pipefail
    export TMPDIR="$HOME/tmp" CARGO_BUILD_JOBS=4 RAYON_NUM_THREADS=4
    export CJXL_PATH="$PWD/.ci-libjxl/tools/cjxl" DJXL_PATH="$PWD/.ci-libjxl/tools/djxl"
    root="$HOME/tmp/jxl-backlog/jpeg-legacy-{{label}}"
    mkdir -p "$root"
    export JXL_ENCODER_OUTPUT_DIR="$root/encoder"
    nice -n 19 cargo run --locked -p jxl-encoder --features jpeg-reencoding --example jpeg_fixture_setup -- "$root/corpus-root" > "$root/setup.log" 2>&1
    export CODEC_CORPUS_DIR="$(cat "$root/corpus-root")"
    nice -n 19 cargo test --locked -p jxl-encoder --features jpeg-reencoding --test it jpeg_reencoding:: -- --nocapture --test-threads=1 > "$root/tests.log" 2>&1
    rg 'test result:' "$root/tests.log"

jpeg-enabled-check label:
    #!/usr/bin/env bash
    set -euo pipefail
    just jpeg-legacy-check "{{label}}"
    export TMPDIR="$HOME/tmp" CARGO_BUILD_JOBS=4 RAYON_NUM_THREADS=4
    export CJXL_PATH="$PWD/.ci-libjxl/tools/cjxl" DJXL_PATH="$PWD/.ci-libjxl/tools/djxl"
    root="$HOME/tmp/jxl-backlog/jpeg-legacy-{{label}}"
    export JXL_ENCODER_OUTPUT_DIR="$root/encoder"
    export CODEC_CORPUS_DIR="$(cat "$root/corpus-root")"
    export JBRD_CONFORMANCE_REFERENCE=1 JBRD_CONFORMANCE_ARTIFACTS="$root/conformance"
    nice -n 19 cargo test --locked -p jxl-encoder --features jpeg-reencoding --test it -- --test-threads=4 > "$root/full-integration.log" 2>&1
    rg 'test result:' "$root/full-integration.log"

# Complete lossless coding-unit selection; caller provisions gb82-sc/terminal.png.
lz77-keep-best-check label manifest="Cargo.toml" features="corpus-tests,__expert,__internals,parallel":
    #!/usr/bin/env bash
    set -euo pipefail
    root="$HOME/tmp/jxl110-keep-best-{{label}}"
    mkdir -p "$root"
    export TMPDIR="$HOME/tmp" CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=2
    export DJXL_PATH="${DJXL_PATH:-$PWD/.ci-libjxl/tools/djxl}"
    export LZ77_KEEP_BEST_OUTPUT="$root/artifacts"
    nice -n19 cargo test --manifest-path "{{manifest}}" --locked --release -p jxl-encoder --features '{{features}}' --lib production_keep_best_regressions -- --nocapture > "$root/test.log" 2>&1
    rg 'KEEP-BEST|test result:' "$root/test.log"

# Paired encode-only timing; manifest controls content/size/depth/effort/mode/thread axes.
lz77-keep-best-timing binary manifest output repeats="5":
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p "$(dirname '{{output}}')"
    nice -n19 python3 scripts/lz77_keep_best_timing.py '{{binary}}' '{{manifest}}' '{{output}}' --repeats '{{repeats}}' > '{{output}}.log' 2>&1
    cat '{{output}}/summary.tsv'
shipgates-native-check:
    cargo clippy --release --locked -p jxl-encoder --example zensim_diffmap_rd --features __expert,butteraugli-loop,zensim-loop,ssim2-loop,parallel --target-dir target -- -D warnings

shipgates-native-build:
    cargo build --release --locked -p jxl-encoder --example zensim_diffmap_rd --features __expert,butteraugli-loop,zensim-loop,ssim2-loop,parallel --target-dir target

shipgates-map-check:
    cargo test --release --locked -p jxl-encoder --lib --features zensim-loop --target-dir target rd_map_cas_preserves_bits_and_refuses_missing_or_changed_artifacts


shipgates-simd-check:
    cargo test --release --locked -p jxl-encoder-simd --lib --target-dir target

shipgates-ci-sources root:
    python3 scripts/ci_frozen_sources.py .github/sibling-revisions.tsv {{quote(root)}}

shipgates-arm-check:
    cargo clippy --release --locked -p jxl-encoder-simd --all-targets --target aarch64-unknown-linux-gnu -- -D warnings

shipgates-intervention-build:
    cargo build --release --locked -p jxl-encoder --example zensim_diffmap_rd --features __expert,__pre_quantized,__internal_recon_hook,butteraugli-loop,zensim-loop,ssim2-loop,parallel --target-dir target

shipgates-intervention-check:
    cargo clippy --release --locked -p jxl-encoder --example zensim_diffmap_rd --features __expert,__pre_quantized,__internal_recon_hook,butteraugli-loop,zensim-loop,ssim2-loop,parallel --target-dir target -- -D warnings

# LoopTrace is public with std; its mutation helpers need the perceptual backend.
loop-trace-feature-check:
    cargo clippy --locked -p jxl-encoder --lib --no-default-features --features std,convenience -- -D warnings
    cargo clippy --locked -p jxl-encoder --lib -- -D warnings

# Independent color-header readback, lossless sample parity, and constructor serialization.
rendering-intent-check:
    cargo test --locked -p jxl-encoder --lib headers::color_encoding::tests
    cargo clippy --locked -p jxl-encoder --lib -- -D warnings

# Keep existing VarDCT byte locks while fixing constructor intent.
relative-header-byte-check:
    cargo test --locked -p jxl-encoder --lib headers::color_encoding::tests
    cargo test --locked -p jxl-encoder --lib vardct::encoder::tests::test_hash_lock_
    cargo clippy --locked -p jxl-encoder --lib -- -D warnings

# Includes split-header integration locks, beyond the whole-bitstream unit locks.
relative-integration-byte-check:
    cargo test --locked -p jxl-encoder --features __expert --test it hash_lock_features -- --test-threads=4

# Check compatibility after admitting the current archmage/magetypes beta.
beta-cohort-check:
    cargo fmt --check
    cargo clippy --locked --workspace --lib -- -D warnings
    cargo test --locked --workspace --lib --no-fail-fast

# Full-range predictor and original native float/squeeze pixel oracle.
gradient-float-check:
    cargo test --locked -p jxl-encoder --lib --features __expert gradient_full_i32_matches_wide_arithmetic
    cargo test --locked -p jxl-encoder --test it lossless_float::f32_with_squeeze_round_trips_bit_exactly -- --exact

# Uses the same locked source closure as CI; caller provides verified v0.12 tools.
native-orientation-animation-check manifest="Cargo.toml":
    cargo test --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --test native_orientation animation_metadata_preserves_native_orientation

native-orientation-check manifest="Cargo.toml":
    cargo test --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features jpeg-reencoding,chroma-subsampling --test native_orientation
    cargo test --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features jpeg-reencoding --test it jpeg_gainmap::
    cargo clippy --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features jpeg-reencoding,chroma-subsampling --test native_orientation -- -D warnings
    cargo fmt --manifest-path {{quote(manifest)}} -p jxl-encoder -- --check

# Diagnostic v0.12 decoder with failure locations; caller supplies run-heavy controls.
libjxl-debug-oracle source out:
    cmake -S {{quote(source)}} -B {{quote(out)}} -DCMAKE_BUILD_TYPE=Debug "-DCMAKE_CXX_FLAGS_DEBUG=-O1 -g1" "-DCMAKE_C_FLAGS_DEBUG=-O1 -g1" -DJPEGXL_ENABLE_OPENEXR=OFF -DJPEGXL_ENABLE_JPEGLI=OFF -DJPEGXL_ENABLE_BENCHMARK=OFF -DJPEGXL_ENABLE_DEVTOOLS=OFF -DJPEGXL_ENABLE_MANPAGES=OFF -DJPEGXL_ENABLE_SKCMS=OFF -DJPEGXL_ENABLE_SJPEG=OFF -DJPEGXL_FORCE_SYSTEM_LCMS2=ON -DBUILD_TESTING=OFF
    cmake --build {{quote(out)}} --target djxl --parallel "${CARGO_BUILD_JOBS:-4}"

# Explicit caller-selected original-ICC oracle: default djxl cannot coalesce arbitrary-ICC XYB crops.
libjxl-original-profile-oracle source build out:
    c++ -std=c++17 -O1 scripts/libjxl_original_profile.cc -I{{quote(source + "/lib/include")}} -I{{quote(build + "/lib/include")}} -L{{quote(build + "/lib")}} -Wl,-rpath,{{quote(build + "/lib")}} -ljxl -ljxl_cms -o {{quote(out)}}

native-orientation-icc-check manifest oracle:
    LIBJXL_ORIGINAL_PROFILE_ORACLE={{quote(oracle)}} cargo test --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features corpus-tests,butteraugli-loop --test native_orientation original_icc_orientation_control -- --exact

# Preserve the pinned sibling graph while checking zero-allocation budgets.
zeroed-alloc-check:
    cargo metadata --locked --format-version 1 > /dev/null
    cargo test --locked -p jxl-encoder --lib budget::
    cargo clippy --locked -p jxl-encoder --lib -- -D warnings

# Require complete effort ladders and preserve the known inversion unchanged.
effort-gate-evidence-check binary source before out:
    python3 scripts/check_effort_gate.py '{{binary}}' '{{source}}' '{{before}}' '{{out}}'

float-decode-evidence-check:
    uv run --with numpy python scripts/test_compare_float_decodes.py

# Diagnostic copies only; retain the original failed PQ stream unchanged.
pq-linear-control input out *args:
    TMPDIR="$HOME/tmp" uv run --with numpy python scripts/pq_linear_control.py {{quote(input)}} {{quote(out)}} {{args}}

# Caller supplies flock and run-heavy; refuse an existing destination.
nightly-artifact-archive run name out:
    mkdir {{quote(out)}}
    gh run download {{quote(run)}} --repo imazen/jxl-encoder --name {{quote(name)}} --dir {{quote(out)}}


# Run both strict CI lint feature graphs even when one has diagnostics.
ci-lint-check manifest="Cargo.toml":
    #!/usr/bin/env bash
    set -uo pipefail
    cargo clippy --keep-going --locked --manifest-path {{quote(manifest)}} --workspace --all-targets -- -D warnings
    default_status=$?
    cargo clippy --keep-going --locked --manifest-path {{quote(manifest)}} --workspace --all-targets --features zensim-loop -- -D warnings
    zensim_status=$?
    test "$default_status" -eq 0 && test "$zensim_status" -eq 0


# Existing tests affected by the Rust 1.100 iterator maintenance.
ci-iteration-regressions manifest="Cargo.toml":
    cargo test --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --lib -- --test-threads=2
    cargo test --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --test it clic2025::test_ans_skewed_histogram_roundtrip -- --exact --test-threads=1
    cargo test --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --test it clic2025::test_ans_single_symbol -- --test-threads=1

# Profile-model checks include refusal and valid/canonicalized producer controls.
icc-model-check manifest="Cargo.toml":
    cargo test --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert,jpeg-reencoding,chroma-subsampling --lib headers::file_header::tests
    cargo test --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert,jpeg-reencoding,chroma-subsampling --test color_profile_contract --test icc_tag_contract --test frame_header_contract
    cargo clippy --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert,jpeg-reencoding,chroma-subsampling --lib --test color_profile_contract --test icc_tag_contract --test frame_header_contract -- -D warnings

icc-model-feature-check manifest="Cargo.toml":
    cargo test --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --no-default-features --features std,__expert --test color_profile_contract
    cargo test --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert,jpeg-reencoding,chroma-subsampling,__pre_quantized --test color_profile_contract
    cargo clippy --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert,jpeg-reencoding,chroma-subsampling,__pre_quantized --lib --test color_profile_contract -- -D warnings

icc-model-mutations manifest logs:
    python3 scripts/test_icc_model_mutations.py {{quote(manifest)}} {{quote(logs)}}

icc-model-coverage manifest output:
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert,jpeg-reencoding,chroma-subsampling,__pre_quantized --lib --no-report -- headers::file_header::tests
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert,jpeg-reencoding,chroma-subsampling,__pre_quantized --test color_profile_contract --no-report
    cargo llvm-cov report --manifest-path {{quote(manifest)}} --json --output-path {{quote(output)}}

icc-model-cli-check manifest="Cargo.toml":
    cargo test --locked --manifest-path {{quote(manifest)}} -p jxl-encoder-cli --test icc_model -- --nocapture
    cargo clippy --locked --manifest-path {{quote(manifest)}} -p jxl-encoder-cli --test icc_model -- -D warnings

extra-rect-oracle source build output:
    python3 scripts/frame_header_oracle.py --component extra_rect {{quote(source)}} {{quote(build)}} {{quote(output)}}

extra-edge-unit manifest="Cargo.toml":
    cargo test --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --lib shifted_region_oracle_tests -- --nocapture

extra-edge-check manifest="Cargo.toml":
    cargo test --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert --test shifted_extra_edges -- --nocapture

extra-edge-cli-check manifest="Cargo.toml":
    cargo test --locked --manifest-path {{quote(manifest)}} -p jxl-encoder-cli --test shifted_alpha -- --nocapture

extra-edge-validation manifest="Cargo.toml":
    cargo test --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert --lib
    cargo test --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert,__internals --test it hash_lock
    cargo test --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert,__internals --test it strategy_libjxl_byte_lock
    cargo test --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert,__internals --test it divergence_table_drift
    cargo clippy --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert --lib --test shifted_extra_edges -- -D warnings
    cargo clippy --locked --manifest-path {{quote(manifest)}} -p jxl-encoder-cli --test shifted_alpha -- -D warnings

extra-edge-mutations manifest logs:
    python3 scripts/test_shifted_extra_mutations.py {{quote(manifest)}} {{quote(logs)}}

extra-edge-mutation-cli manifest logs:
    python3 scripts/test_shifted_extra_mutations.py {{quote(manifest)}} {{quote(logs)}} --only clipped-cli

extra-edge-coverage manifest output:
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert --lib --no-report -- shifted_region_oracle_tests
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert --test shifted_extra_edges --no-report
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder-cli --features jxl_encoder/__expert --test shifted_alpha --no-report
    cargo llvm-cov report --manifest-path {{quote(manifest)}} --json --output-path {{quote(output)}}

# Completed evidence only; verifies each destination before removing its source.
archive-test-evidence source destination journal:
    python3 scripts/archive_test_evidence.py {{quote(source)}} {{quote(destination)}} {{quote(journal)}}

extra-edge-cli-coverage manifest output:
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder-cli --features jxl_encoder/__expert --test shifted_alpha --no-report
    cargo llvm-cov report --manifest-path {{quote(manifest)}} --json --output-path {{quote(output)}}

extra-edge-gray-mutations manifest logs:
    python3 scripts/test_shifted_extra_mutations.py {{quote(manifest)}} {{quote(logs)}} --only gray --instrumented

coverage-function-regions report *functions:
    python3 scripts/coverage_function_regions.py {{quote(report)}} --file-suffix modular/frame.rs --file-suffix modular/encode.rs {{functions}}

extra-edge-resource-coverage manifest output:
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert --lib --no-report -- shifted_lf_resource_tests
    cargo llvm-cov report --manifest-path {{quote(manifest)}} --json --output-path {{quote(output)}}

ans-cancellation-check manifest="Cargo.toml":
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert --lib --no-report -- cancellation_144

ans-cancellation-coverage manifest output:
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert --lib --no-report -- cancellation_144
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert,parallel,parallel-tree-learning --lib --no-report -- cancellation_144
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert,parallel,parallel-tree-learning --lib --no-report -- shifted_lf_resource_tests
    cargo llvm-cov report --manifest-path {{quote(manifest)}} --json --output-path {{quote(output)}}

ans-cancellation-validation manifest:
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert --lib --no-report
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert,__internals --test it --no-report -- hash_lock
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert,__internals --test it --no-report -- strategy_libjxl_byte_lock
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert,__internals --test it --no-report -- divergence_table_drift
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert,parallel,parallel-tree-learning --lib --no-report -- cancellation_144

squeeze-contract-check manifest:
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert --test squeeze_contract --no-report -- --nocapture --test-threads=1

squeeze-params-oracle source build output:
    python3 scripts/frame_header_oracle.py --component squeeze_params {{quote(source)}} {{quote(build)}} {{quote(output)}}

ans-cancellation-lint manifest coverage_env:
    bash -c 'set -e; source "$1"; cargo clippy --locked --manifest-path "$2" -p jxl-encoder --features __expert --lib -- -D warnings; cargo clippy --locked --manifest-path "$2" -p jxl-encoder --features __expert,parallel,parallel-tree-learning --lib -- -D warnings' _ {{quote(coverage_env)}} {{quote(manifest)}}

coverage-selected-report llvm_cov profile output +objects:
    python3 scripts/export_selected_coverage.py {{quote(llvm_cov)}} {{quote(profile)}} {{quote(output)}} {{objects}}

extra-edge-final-coverage manifest output:
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert --lib --no-report -- request_stop_tests
    just ans-cancellation-coverage {{quote(manifest)}} {{quote(output + ".resource.json")}}
    just extra-edge-coverage {{quote(manifest)}} {{quote(output)}}

request-cancellation-check manifest:
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert --test request_cancellation --no-report
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert,parallel,parallel-tree-learning --test request_cancellation --no-report
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert,jpeg-reencoding --test request_cancellation --no-report

request-cancellation-mutation manifest log only="still":
    python3 scripts/test_request_stop_mutation.py {{quote(manifest)}} {{quote(log)}} --only {{quote(only)}}

extra-edge-cmyk-check manifest profile output:
    cargo llvm-cov run --profile test --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert --example shifted_extra_cmyk --no-report -- {{quote(profile)}} {{quote(output)}}

extra-edge-final-validation manifest:
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert --lib --no-report
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert,__internals --test it --no-report -- hash_lock
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert,__internals --test it --no-report -- strategy_libjxl_byte_lock
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert,__internals --test it --no-report -- divergence_table_drift
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --no-default-features --features std,convenience,__expert --test shifted_extra_edges --no-report
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert,parallel,parallel-tree-learning --test shifted_extra_edges --no-report -- threaded_shifted_routes

animation-cancellation-check manifest:
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert --test request_cancellation --no-report -- animation_requests --nocapture

jpeg-cancellation-check manifest:
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert,jpeg-reencoding --test request_cancellation --no-report -- jpeg_ --nocapture

low-level-cancellation-check manifest:
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert,jpeg-reencoding --test request_cancellation --no-report -- low_level_jpeg_operations --nocapture

request-cancellation-lint manifest coverage_env:
    bash -c 'set -e; source "$1"; export CARGO_TARGET_DIR="$CARGO_LLVM_COV_TARGET_DIR"; cargo clippy --locked --manifest-path "$2" -p jxl-encoder --features __expert,jpeg-reencoding,parallel,parallel-tree-learning --lib --test request_cancellation --test shifted_extra_edges --example shifted_extra_cmyk -- -D warnings; cargo check --locked --manifest-path "$2" -p jxl-encoder --no-default-features --features std,convenience --lib' _ {{quote(coverage_env)}} {{quote(manifest)}}

request-cancellation-unit manifest:
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert,jpeg-reencoding --lib --no-report -- request_stop_tests

coverage-code-window report function suffix first last:
    python3 scripts/coverage_function_regions.py {{quote(report)}} {{quote(function)}} --file-suffix {{quote(suffix)}} --start-line {{first}} --end-line {{last}}

squeeze-tall-contract-check manifest:
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert --test squeeze_contract --no-report -- squeeze_bands --nocapture

squeeze-f32-contract-check manifest:
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert --test squeeze_contract --no-report -- f32_squeeze_admission --nocapture

extra-edge-cli-selected-check manifest coverage_env:
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder-cli --test shifted_alpha --no-report
    bash -c 'set -e; source "$1"; export CARGO_TARGET_DIR="$CARGO_LLVM_COV_TARGET_DIR"; cargo clippy --locked --manifest-path "$2" -p jxl-encoder-cli --bin cjxl-rs --test shifted_alpha -- -D warnings' _ {{quote(coverage_env)}} {{quote(manifest)}}

squeeze-default-vectors oracle output:
    set -C; {{quote(oracle)}} --defaults > {{quote(output)}}

squeeze-transform-counts oracle output:
    set -C; {{quote(oracle)}} --transform-counts > {{quote(output)}}

squeeze-begin-fields oracle output:
    set -C; {{quote(oracle)}} --begin-fields > {{quote(output)}}

squeeze-count-contract-check manifest:
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert --test squeeze_contract --no-report -- twenty_channel --nocapture

squeeze-wire-vectors oracle output:
    set -C; {{quote(oracle)}} --wire > {{quote(output)}}

squeeze-descriptor-unit manifest:
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert --lib --no-report -- default_squeeze_wire_tests
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert --lib --no-report -- squeeze_field_tests

squeeze-count-matrix manifest:
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert --test squeeze_count_matrix --no-report -- every_policy --nocapture --test-threads=1

squeeze-no-debug-control manifest:
    python3 scripts/test_encoder_without_checks.py {{quote(manifest)}}

squeeze-no-debug-f32-control manifest:
    python3 scripts/test_encoder_without_checks.py {{quote(manifest)}} --include-f32

squeeze-count-feature-controls manifest:
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert,parallel,parallel-tree-learning --test squeeze_count_matrix --no-report -- larger_squeeze_defaults --nocapture --test-threads=1
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert --test squeeze_count_matrix --no-report -- unsafe_implicit_dimensions --nocapture --test-threads=1

squeeze-layout-controls manifest:
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert --test squeeze_layouts --no-report -- --nocapture --test-threads=1

squeeze-lf-count-controls manifest:
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert --test squeeze_lf_count --no-report -- --nocapture --test-threads=1

lf-squeeze-color-oracle source build output:
    python3 scripts/frame_header_oracle.py --component lf_squeeze_color {{quote(source)}} {{quote(build)}} {{quote(output)}}

squeeze-cmyk-count-controls manifest icc out:
    cargo llvm-cov run --profile test --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert --example shifted_extra_cmyk --no-report -- {{quote(icc)}} {{quote(out)}} 17 --squeeze

squeeze-descriptor-mutations manifest out:
    python3 scripts/test_squeeze_mutation.py {{quote(manifest)}} {{quote(out)}}

# Current-source E2 checks; E3/E5 regressions retain their separate failing verdicts.
squeeze-descriptor-validation manifest:
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert --lib --no-report -- default_squeeze_wire_tests --test-threads=1
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert --lib --no-report -- squeeze_field_tests --test-threads=1
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert --test squeeze_count_matrix --test squeeze_layouts --test squeeze_lf_count --no-report -- --test-threads=1
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert --test squeeze_contract --no-report -- twenty_channel --test-threads=1
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert --lib --no-report -- --test-threads=1
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert,__internals --test it --no-report -- hash_lock --test-threads=1
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert,__internals --test it --no-report -- strategy_libjxl_byte_lock --test-threads=1
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert,__internals --test it --no-report -- divergence_table_drift --test-threads=1

squeeze-descriptor-minimal manifest:
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --no-default-features --features std,convenience,__expert --test squeeze_count_matrix --test squeeze_layouts --test squeeze_lf_count --no-report -- --test-threads=1

squeeze-descriptor-lint manifest:
    cargo clippy --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert,parallel,parallel-tree-learning --lib --test squeeze_contract --test squeeze_count_matrix --test squeeze_layouts --test squeeze_lf_count --test squeeze_reference --example shifted_extra_cmyk -- -D warnings

squeeze-native-reference-controls manifest:
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert --test squeeze_reference --no-report -- --nocapture --test-threads=1

squeeze-descriptor-additional manifest:
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert --test squeeze_reference --no-report -- --nocapture --test-threads=1
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert --test squeeze_count_matrix --no-report -- maximum_admitted --nocapture --test-threads=1
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert --test squeeze_lf_count --no-report -- --nocapture --test-threads=1
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert --lib --no-report -- default_squeeze_wire_tests --test-threads=1


squeeze-descriptor-final-gates manifest coverage_env:
    bash -c 'set -e; source "$1"; export CARGO_TARGET_DIR="$CARGO_LLVM_COV_TARGET_DIR"; just squeeze-descriptor-lint "$2"' _ {{quote(coverage_env)}} {{quote(manifest)}}
    just squeeze-descriptor-minimal {{quote(manifest)}}
    just squeeze-count-feature-controls {{quote(manifest)}}

coverage-from-archive llvm_directory archive output:
    python3 scripts/coverage_from_archive.py {{quote(llvm_directory)}} {{quote(archive)}} {{quote(output)}}

squeeze-f32-route-controls manifest:
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert --test squeeze_f32_routes --no-report -- --nocapture --test-threads=1

squeeze-generator-edge-controls manifest:
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert --lib --no-report -- default_squeeze_wire_tests --test-threads=1
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert --test squeeze_count_matrix --no-report -- lower_count_one_axis --nocapture --test-threads=1

squeeze-group-geometry-controls manifest:
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert --test squeeze_group_geometry --no-report -- --nocapture --test-threads=1

squeeze-rct-isolated manifest revision output:
    python3 scripts/validate_squeeze_rct_isolated.py {{quote(manifest)}} {{quote(revision)}} {{quote(output)}}

squeeze-rct-isolated-resume manifest revision output start:
    python3 scripts/validate_squeeze_rct_isolated.py {{quote(manifest)}} {{quote(revision)}} {{quote(output)}} --start-at {{start}}

squeeze-rct-unchecked manifest revision output:
    python3 scripts/validate_squeeze_rct_isolated.py {{quote(manifest)}} {{quote(revision)}} {{quote(output)}} --unchecked --end-before 1

issue144-production-contracts manifest profile="test":
    cargo test --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --profile {{quote(profile)}} --features __expert,__internals,jpeg-reencoding,parallel,parallel-tree-learning --test frame_header_contract --test color_profile_contract --test icc_tag_contract --test request_cancellation --test shifted_extra_edges --test squeeze_reference --test squeeze_f32_routes -- --test-threads=1

squeeze-rct-isolated-contracts manifest revision output:
    python3 scripts/validate_squeeze_rct_isolated.py {{quote(manifest)}} {{quote(revision)}} {{quote(output)}} --contracts

squeeze-rct-isolated-lint manifest revision output:
    python3 scripts/validate_squeeze_rct_isolated.py {{quote(manifest)}} {{quote(revision)}} {{quote(output)}} --lint

squeeze-rct-planar-controls manifest revision output:
    python3 scripts/validate_squeeze_rct_isolated.py {{quote(manifest)}} {{quote(revision)}} {{quote(output)}} --planar-only

squeeze-rct-guard-regions report:
    python3 scripts/coverage_function_regions.py {{quote(report)}} 33write_modular_stream_with_squeeze --file-suffix modular/encode.rs --start-line 943 --end-line 946
    python3 scripts/coverage_function_regions.py {{quote(report)}} write_modular_stream_with_squeeze_and_tree_stop --file-suffix modular/encode.rs --start-line 3221 --end-line 3224
    python3 scripts/coverage_function_regions.py {{quote(report)}} encode_modular_multi_group_squeeze_with_tree --file-suffix modular/frame.rs --start-line 2472 --end-line 2475

squeeze-rct-boundary-controls manifest revision output:
    python3 scripts/validate_squeeze_rct_isolated.py {{quote(manifest)}} {{quote(revision)}} {{quote(output)}} --end-before 1

squeeze-group-unit-controls manifest:
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert --lib --no-report -- group_schedule_tests --test-threads=1

squeeze-group-regressions manifest:
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert --test squeeze_contract --test squeeze_group_geometry --no-report -- --nocapture --test-threads=1

squeeze-group-policy-controls manifest:
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert --test squeeze_group_geometry --no-report -- every_policy --nocapture --test-threads=1

squeeze-group-entry-controls manifest:
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert --test squeeze_group_geometry --no-report -- supported_layouts --nocapture --test-threads=1

squeeze-group-size-controls manifest:
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert --test squeeze_group_geometry --no-report -- all_group_sizes --nocapture --test-threads=1

squeeze-gray-band-diagnostic manifest:
    RUST_BACKTRACE=1 cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert --test squeeze_group_geometry --no-report -- grayscale_global_band_diagnostic --nocapture --test-threads=1

squeeze-gray-independent-controls manifest:
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert --test squeeze_group_geometry --no-report -- grayscale_global_bands_in_independent_decoders --nocapture --test-threads=1

squeeze-gray-native-oracle source build output:
    python3 scripts/frame_header_oracle.py --component squeeze_gray_geometry {{quote(source)}} {{quote(build)}} {{quote(output)}}

squeeze-gray-native-controls manifest:
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert --lib --no-report -- gray_global_native_tests --test-threads=1
    just squeeze-gray-independent-controls {{quote(manifest)}}

squeeze-gray-native-decoder manifest:
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert --test squeeze_group_geometry --no-report -- native_decoder_grayscale_global_bands_diagnostic --nocapture --test-threads=1

# Public CMS admission versus internal raw ICC storage, pinned v0.12 objects.
icc-raw-policy-oracle source build output:
    python3 scripts/frame_header_oracle.py --component icc_policy {{quote(source)}} {{quote(build)}} {{quote(output)}}

squeeze-native-decoder-controls manifest:
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert --test squeeze_group_geometry --no-report -- native_decoder_grayscale --nocapture --test-threads=1

squeeze-group-animation-controls manifest:
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert --test squeeze_group_geometry --no-report -- integer_animation_routes --nocapture --test-threads=1

squeeze-repaired-layout-animation-controls manifest:
    just squeeze-group-entry-controls {{quote(manifest)}}
    just squeeze-group-animation-controls {{quote(manifest)}}

squeeze-group-mutations manifest out:
    python3 scripts/test_squeeze_mutation.py {{quote(manifest)}} {{quote(out)}} --case group-admission --case group-implicit-fallback --case group-prefix-cutoff

# Isolate the dependency pin from the pending squeeze production changes.
decoder-pin-baseline manifest revision out:
    python3 scripts/validate_squeeze_rct_isolated.py {{quote(manifest)}} {{quote(revision)}} {{quote(out)}} --start-at 2

squeeze-group-planar-controls manifest:
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert --test squeeze_group_geometry --no-report -- planar_integer_entries --nocapture --test-threads=1

squeeze-group-final-callers manifest:
    cargo fmt --manifest-path {{quote(manifest)}} -p jxl-encoder
    just squeeze-group-animation-controls {{quote(manifest)}}
    just squeeze-group-planar-controls {{quote(manifest)}}
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert --test squeeze_layouts --no-report -- grayscale_with_a_shifted_first_extra --nocapture --test-threads=1

# Supply only completed profiles and their exact executable/source versions.
preserve-test-evidence source_root output args:
    python3 scripts/preserve_test_evidence.py {{quote(source_root)}} {{quote(output)}} {{args}}

squeeze-group-parallel-controls manifest:
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert,parallel,parallel-tree-learning --test squeeze_group_geometry --no-report -- tall_group_routes_observe --nocapture --test-threads=1

squeeze-group-unit-evidence manifest output coverage_env:
    cargo fmt --manifest-path {{quote(manifest)}} -p jxl-encoder
    bash -c 'set -e; source "$1"; export CARGO_TARGET_DIR="$CARGO_LLVM_COV_TARGET_DIR"; python3 scripts/run_selected_test_evidence.py "$2" "$3" group_schedule_tests' _ {{quote(coverage_env)}} {{quote(manifest)}} {{quote(output)}}

squeeze-group-parallel-evidence manifest output coverage_env:
    bash -c 'set -e; source "$1"; export CARGO_TARGET_DIR="$CARGO_LLVM_COV_TARGET_DIR"; python3 scripts/run_selected_test_evidence.py "$2" "$3" tall_group_routes_observe --test-target squeeze_group_geometry --features __expert,parallel,parallel-tree-learning' _ {{quote(coverage_env)}} {{quote(manifest)}} {{quote(output)}}

squeeze-group-final-lint manifest:
    cargo clippy --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert,parallel,parallel-tree-learning --lib --test squeeze_contract --test squeeze_count_matrix --test squeeze_group_geometry --test squeeze_layouts --test squeeze_lf_count --test squeeze_reference --example shifted_extra_cmyk -- -D warnings

squeeze-group-final-mechanics manifest parallel_output mutation_output unit_evidence unit_coverage coverage_env llvm_directory:
    just coverage-from-archive {{quote(llvm_directory)}} {{quote(unit_evidence)}} {{quote(unit_coverage)}}
    just squeeze-group-parallel-evidence {{quote(manifest)}} {{quote(parallel_output)}} {{quote(coverage_env)}}
    just squeeze-group-mutations {{quote(manifest)}} {{quote(mutation_output)}}
    bash -c 'set -e; source "$1"; export CARGO_TARGET_DIR="$CARGO_LLVM_COV_TARGET_DIR"; just squeeze-group-final-lint "$2"' _ {{quote(coverage_env)}} {{quote(manifest)}}

squeeze-group-mechanics-resume manifest output mutation_output coverage_env:
    just squeeze-group-parallel-evidence {{quote(manifest)}} {{quote(output)}} {{quote(coverage_env)}}
    just squeeze-group-mutations {{quote(manifest)}} {{quote(mutation_output)}}
    bash -c 'set -e; source "$1"; export CARGO_TARGET_DIR="$CARGO_LLVM_COV_TARGET_DIR"; just squeeze-group-final-lint "$2"' _ {{quote(coverage_env)}} {{quote(manifest)}}

squeeze-group-one-axis-controls manifest:
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert --test squeeze_group_geometry --no-report -- one_pixel_axes --nocapture --test-threads=1

squeeze-group-cutoff-mutation manifest out:
    python3 scripts/test_squeeze_mutation.py {{quote(manifest)}} {{quote(out)}} --case group-prefix-cutoff

squeeze-group-cutoff-verification manifest unit_out mutation_out coverage_env:
    just squeeze-group-unit-evidence {{quote(manifest)}} {{quote(unit_out)}} {{quote(coverage_env)}}
    just squeeze-group-one-axis-controls {{quote(manifest)}}
    just squeeze-group-cutoff-mutation {{quote(manifest)}} {{quote(mutation_out)}}
    bash -c 'set -e; source "$1"; export CARGO_TARGET_DIR="$CARGO_LLVM_COV_TARGET_DIR"; just squeeze-group-final-lint "$2"' _ {{quote(coverage_env)}} {{quote(manifest)}}

squeeze-group-region-report report:
    python3 scripts/coverage_function_regions.py {{quote(report)}} default_squeeze_params_for_groups squeeze_geometry_step squeeze_group_geometry_fits --file-suffix modular/squeeze.rs

squeeze-patch-caller-controls manifest:
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert --test squeeze_group_geometry --no-report -- winning_patch_hosts --nocapture --test-threads=1

squeeze-final-feature-check manifest:
    cargo fmt --manifest-path {{quote(manifest)}} -p jxl-encoder
    just squeeze-patch-caller-controls {{quote(manifest)}}
    python3 scripts/test_encoder_without_checks.py {{quote(manifest)}} --include-f32
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --no-default-features --features std,convenience,__expert --test squeeze_contract --test squeeze_lf_count --no-report -- --test-threads=1
    cargo check --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --no-default-features

alpha-squeeze-edge-regression manifest:
    cargo fmt --manifest-path {{quote(manifest)}} -p jxl-encoder
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert --test alpha_squeeze_contract --no-report -- --nocapture --test-threads=1

squeeze-rust1100-lint manifest:
    cargo +1.100.0 clippy --locked --manifest-path {{quote(manifest)}} --workspace --all-targets -- -D warnings
    cargo +1.100.0 clippy --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert,__internals,parallel,parallel-tree-learning --lib --test squeeze_contract --test squeeze_count_matrix --test squeeze_group_geometry --test squeeze_layouts --test squeeze_lf_count --test squeeze_reference --example shifted_extra_cmyk -- -D warnings


squeeze-current-caller-evidence manifest no_rct_output axis_output coverage_env:
    bash -c 'set -e; source "$1"; export CARGO_TARGET_DIR="$CARGO_LLVM_COV_TARGET_DIR"; python3 scripts/run_selected_test_evidence.py "$2" "$3" every_policy_reaches_no_rct --test-target squeeze_layouts; python3 scripts/run_selected_test_evidence.py "$2" "$4" lower_count_one_axis --test-target squeeze_count_matrix' _ {{quote(coverage_env)}} {{quote(manifest)}} {{quote(no_rct_output)}} {{quote(axis_output)}}

squeeze-scope-coverage-summary output *pairs:
    python3 scripts/squeeze_coverage_summary.py {{quote(output)}} {{pairs}}

alpha-squeeze-layout-controls manifest:
    cargo fmt --manifest-path {{quote(manifest)}} -p jxl-encoder
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert --test alpha_squeeze_matrix --no-report -- alpha_layouts --nocapture --test-threads=1

alpha-squeeze-caller-controls manifest:
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert --test alpha_squeeze_matrix --no-report -- policies_and_public buffering_and_lf --nocapture --test-threads=1

alpha-squeeze-extra-callers manifest:
    cargo fmt --manifest-path {{quote(manifest)}} -p jxl-encoder
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert,__internals,__pre_quantized,parallel --test alpha_squeeze_matrix --no-report -- precomputed_and alpha_shift_brackets alpha_band_writers --nocapture --test-threads=1

alpha-squeeze-progressive-regression manifest:
    cargo fmt --manifest-path {{quote(manifest)}} -p jxl-encoder
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert --test alpha_squeeze_progressive --no-report -- --nocapture --test-threads=1

alpha-squeeze-mutations manifest output coverage_env:
    bash -c 'set -e; source "$1"; export CARGO_TARGET_DIR="$CARGO_LLVM_COV_TARGET_DIR"; python3 scripts/test_alpha_squeeze_mutation.py "$2" "$3"' _ {{quote(coverage_env)}} {{quote(manifest)}} {{quote(output)}}

alpha-squeeze-broad-baseline manifest:
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --lib --no-report -- --test-threads=1
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert,__internals --test it --no-report -- hash_lock strategy_libjxl_byte_lock divergence_table_drift --test-threads=1

alpha-squeeze-minimal-controls manifest:
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --no-default-features --features std,convenience,__expert --test alpha_squeeze_contract --test alpha_squeeze_matrix --no-report -- --test-threads=1
    cargo check --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --no-default-features

alpha-squeeze-rust1100-lint manifest:
    cargo +1.100.0 clippy --locked --manifest-path {{quote(manifest)}} --workspace --all-targets -- -D warnings
    cargo +1.100.0 clippy --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert,__internals,__pre_quantized,parallel --lib --test alpha_squeeze_contract --test alpha_squeeze_matrix --test alpha_squeeze_progressive -- -D warnings

alpha-squeeze-existing-controls manifest:
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert --test it --no-report -- alpha_squeeze --test-threads=1

progressive-brackets-oracle source build output:
    python3 scripts/frame_header_oracle.py --component progressive_brackets {{quote(source)}} {{quote(build)}} {{quote(output)}}

alpha-squeeze-scope-summary output *pairs:
    python3 scripts/alpha_squeeze_coverage_summary.py {{quote(output)}} {{pairs}}

alpha-squeeze-unit-evidence manifest output coverage_env:
    bash -c 'set -e; source "$1"; export CARGO_TARGET_DIR="$CARGO_LLVM_COV_TARGET_DIR"; python3 scripts/run_selected_test_evidence.py "$2" "$3" empty_group_alpha_section --source jxl-encoder/src/vardct/bitstream.rs' _ {{quote(coverage_env)}} {{quote(manifest)}} {{quote(output)}}

progressive-alpha-matrix-evidence manifest output coverage_env:
    bash -c 'set -e; source "$1"; export CARGO_TARGET_DIR="$CARGO_LLVM_COV_TARGET_DIR"; python3 scripts/run_selected_test_evidence.py "$2" "$3" alpha --test-target alpha_squeeze_matrix --features __expert,__internals,__pre_quantized,parallel --source jxl-encoder/src/vardct/bitstream.rs --source jxl-encoder/src/vardct/encoder.rs --source jxl-encoder/src/vardct/extras.rs --source jxl-encoder/src/vardct/progressive_brackets_v012.tsv' _ {{quote(coverage_env)}} {{quote(manifest)}} {{quote(output)}}

progressive-alpha-mutations manifest output coverage_env:
    bash -c 'set -e; source "$1"; export CARGO_TARGET_DIR="$CARGO_LLVM_COV_TARGET_DIR"; python3 scripts/test_progressive_alpha_mutation.py "$2" "$3"' _ {{quote(coverage_env)}} {{quote(manifest)}} {{quote(output)}}

progressive-alpha-scope-summary output *pairs:
    python3 scripts/alpha_squeeze_coverage_summary.py {{quote(output)}} --scope progressive {{pairs}}

progressive-alpha-additional-evidence manifest matrix_output core_output coverage_env:
    bash -c 'set -e; source "$1"; export CARGO_TARGET_DIR="$CARGO_LLVM_COV_TARGET_DIR"; python3 scripts/run_selected_test_evidence.py "$2" "$3" progressive --test-target alpha_squeeze_matrix --features __expert,__internals,__pre_quantized,parallel --source jxl-encoder/src/vardct/bitstream.rs; python3 scripts/run_selected_test_evidence.py "$2" "$4" progressive --test-target alpha_squeeze_progressive --source jxl-encoder/src/vardct/bitstream.rs' _ {{quote(coverage_env)}} {{quote(manifest)}} {{quote(matrix_output)}} {{quote(core_output)}}

progressive-alpha-minimal-controls manifest:
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --no-default-features --features std,convenience,__expert --test alpha_squeeze_contract --test alpha_squeeze_progressive --no-report -- --test-threads=1
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --no-default-features --features std,convenience,__expert --test alpha_squeeze_matrix --no-report -- alpha_layouts progressive canonicalized raw_lossless --test-threads=1
    cargo check --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --no-default-features

progressive-alpha-header-controls controls output:
    python3 scripts/check_progressive_alpha_headers.py {{quote(controls)}} {{quote(output)}}

progressive-alpha-effort-evidence manifest full_output minimal_output coverage_env:
    bash -c 'set -e; source "$1"; export CARGO_TARGET_DIR="$CARGO_LLVM_COV_TARGET_DIR"; python3 scripts/run_selected_test_evidence.py "$2" "$3" progressive_alpha_effort --test-target alpha_squeeze_matrix --features __expert,__internals,__pre_quantized,parallel --source jxl-encoder/src/vardct/bitstream.rs; python3 scripts/run_selected_test_evidence.py "$2" "$4" progressive_alpha_effort --test-target alpha_squeeze_matrix --no-default-features --features std,convenience,__expert --source jxl-encoder/src/vardct/bitstream.rs' _ {{quote(coverage_env)}} {{quote(manifest)}} {{quote(full_output)}} {{quote(minimal_output)}}

progressive-alpha-merged-extras-controls manifest:
    cargo llvm-cov --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert --test it --no-report -- progressive_extras_preserve_alpha --test-threads=1

thin-alpha-partition-evidence manifest output coverage_env:
    bash -c 'set -e; source "$1"; export CARGO_TARGET_DIR="$CARGO_LLVM_COV_TARGET_DIR"; python3 scripts/run_selected_test_evidence.py "$2" "$3" thin_alpha_global_partition --source jxl-encoder/src/vardct/extras.rs --source jxl-encoder/src/vardct/bitstream.rs --source jxl-encoder/src/vardct/alpha_squeeze_thin_v012.tsv --source jxl-encoder/src/vardct/progressive_brackets_v012.tsv' _ {{quote(coverage_env)}} {{quote(manifest)}} {{quote(output)}}

alpha-partition-scope-summary output *pairs:
    python3 scripts/alpha_squeeze_coverage_summary.py {{quote(output)}} --scope partition {{pairs}}

progressive-alpha-precision-evidence manifest full_output minimal_output coverage_env:
    bash -c 'set -e; source "$1"; export CARGO_TARGET_DIR="$CARGO_LLVM_COV_TARGET_DIR"; python3 scripts/run_selected_test_evidence.py "$2" "$3" progressive_alpha_reduced --test-target alpha_squeeze_matrix --features __expert,__internals,__pre_quantized,parallel --source jxl-encoder/src/vardct/bitstream.rs --source jxl-encoder/src/vardct/extras.rs; python3 scripts/run_selected_test_evidence.py "$2" "$4" progressive_alpha_reduced --test-target alpha_squeeze_matrix --no-default-features --features std,convenience,__expert --source jxl-encoder/src/vardct/bitstream.rs --source jxl-encoder/src/vardct/extras.rs' _ {{quote(coverage_env)}} {{quote(manifest)}} {{quote(full_output)}} {{quote(minimal_output)}}

# Unmodified native coefficient order and progressive split observations.
progressive-ac-native-oracle source build output:
    python3 scripts/frame_header_oracle.py --component progressive_ac {{quote(source)}} {{quote(build)}} {{quote(output)}}

alpha-squeeze-elongated-evidence manifest output coverage_env:
    bash -c 'set -e; source "$1"; export CARGO_TARGET_DIR="$CARGO_LLVM_COV_TARGET_DIR"; python3 scripts/run_selected_test_evidence.py "$2" "$3" zz_elongated --test-target alpha_squeeze_progressive --source jxl-encoder/src/vardct/bitstream.rs --source jxl-encoder/src/vardct/extras.rs' _ {{quote(coverage_env)}} {{quote(manifest)}} {{quote(output)}}

alpha-squeeze-elongated-final-check manifest core_output matrix_output coverage_env:
    cargo fmt --manifest-path {{quote(manifest)}} -p jxl-encoder --check
    just alpha-squeeze-elongated-evidence {{quote(manifest)}} {{quote(core_output)}} {{quote(coverage_env)}}
    just progressive-alpha-matrix-evidence {{quote(manifest)}} {{quote(matrix_output)}} {{quote(coverage_env)}}
    just alpha-squeeze-broad-baseline {{quote(manifest)}}
    just progressive-alpha-minimal-controls {{quote(manifest)}}

alpha-squeeze-admission-scope-summary output *pairs:
    python3 scripts/alpha_squeeze_coverage_summary.py {{quote(output)}} --scope admission {{pairs}}

alpha-squeeze-elongated-matrix-evidence manifest output coverage_env:
    bash -c 'set -e; source "$1"; export CARGO_TARGET_DIR="$CARGO_LLVM_COV_TARGET_DIR"; python3 scripts/run_selected_test_evidence.py "$2" "$3" elongated_alpha_paths --test-target alpha_squeeze_matrix --features __expert,__internals,__pre_quantized,parallel --source jxl-encoder/src/vardct/bitstream.rs --source jxl-encoder/src/vardct/extras.rs' _ {{quote(coverage_env)}} {{quote(manifest)}} {{quote(output)}}

reference-version-diagnostics-check output:
    python3 scripts/test_reference_version_diagnostics.py {{quote(output)}}

alpha-squeeze-geometry-mutation manifest output coverage_env:
    bash -c 'set -e; source "$1"; export CARGO_TARGET_DIR="$CARGO_LLVM_COV_TARGET_DIR"; python3 scripts/test_alpha_squeeze_mutation.py "$2" "$3" --case geometry-admission' _ {{quote(coverage_env)}} {{quote(manifest)}} {{quote(output)}}

alpha-squeeze-elongated-minimal-evidence manifest output coverage_env:
    bash -c 'set -e; source "$1"; export CARGO_TARGET_DIR="$CARGO_LLVM_COV_TARGET_DIR"; python3 scripts/run_selected_test_evidence.py "$2" "$3" elongated_alpha_paths --test-target alpha_squeeze_matrix --no-default-features --features std,convenience,__expert --source jxl-encoder/src/vardct/bitstream.rs --source jxl-encoder/src/vardct/extras.rs' _ {{quote(coverage_env)}} {{quote(manifest)}} {{quote(output)}}

alpha-squeeze-elongated-worker-evidence manifest output coverage_env:
    bash -c 'set -e; source "$1"; export CARGO_TARGET_DIR="$CARGO_LLVM_COV_TARGET_DIR"; python3 scripts/run_selected_test_evidence.py "$2" "$3" elongated_alpha_two_worker --test-target alpha_squeeze_matrix --features __expert,__internals,__pre_quantized,parallel --source jxl-encoder/src/vardct/bitstream.rs --source jxl-encoder/src/vardct/extras.rs' _ {{quote(coverage_env)}} {{quote(manifest)}} {{quote(output)}}

progressive-ac-unit-evidence manifest output coverage_env:
    bash -c 'set -e; source "$1"; export CARGO_TARGET_DIR="$CARGO_LLVM_COV_TARGET_DIR"; python3 scripts/run_selected_test_evidence.py "$2" "$3" progressive_sparse_ac_tokens --source jxl-encoder/src/vardct/bitstream.rs --source jxl-encoder/src/vardct/ac_group.rs --source jxl-encoder/src/vardct/ac_strategy.rs --source jxl-encoder/src/vardct/ac_context.rs --source jxl-encoder/src/vardct/coeff_order.rs --source jxl-encoder/src/vardct/common.rs --source jxl-encoder/src/vardct/progressive_ac_token_tests.rs --source jxl-encoder/src/vardct/progressive_ac_v012.tsv --source jxl-encoder/src/vardct/alpha_squeeze_thin_v012.tsv --source jxl-encoder/src/vardct/progressive_brackets_v012.tsv' _ {{quote(coverage_env)}} {{quote(manifest)}} {{quote(output)}}

progressive-ac-real-evidence manifest output coverage_env:
    bash -c 'set -e; source "$1"; export CARGO_TARGET_DIR="$CARGO_LLVM_COV_TARGET_DIR"; python3 scripts/run_selected_test_evidence.py "$2" "$3" progressive_real_pixels --test-target progressive_ac_contract --features __expert,parallel --source jxl-encoder/src/vardct/bitstream.rs --source jxl-encoder/src/vardct/ac_group.rs --source jxl-encoder/src/vardct/ac_strategy.rs --source jxl-encoder/src/vardct/ac_context.rs --source jxl-encoder/src/vardct/coeff_order.rs --source jxl-encoder/src/vardct/common.rs --source jxl-encoder/src/vardct/progressive_ac_v012.tsv --source jxl-encoder/tests/images/frymire.png' _ {{quote(coverage_env)}} {{quote(manifest)}} {{quote(output)}}

progressive-ac-caller-evidence manifest output coverage_env:
    bash -c 'set -e; source "$1"; export CARGO_TARGET_DIR="$CARGO_LLVM_COV_TARGET_DIR"; python3 scripts/run_selected_test_evidence.py "$2" "$3" progressive --test-target progressive_ac_contract --features __expert,parallel --source jxl-encoder/src/vardct/bitstream.rs --source jxl-encoder/src/vardct/ac_group.rs --source jxl-encoder/src/vardct/ac_strategy.rs --source jxl-encoder/src/vardct/ac_context.rs --source jxl-encoder/src/vardct/coeff_order.rs --source jxl-encoder/src/vardct/common.rs --source jxl-encoder/src/vardct/progressive_ac_v012.tsv --source jxl-encoder/tests/images/frymire.png' _ {{quote(coverage_env)}} {{quote(manifest)}} {{quote(output)}}

progressive-ac-focused-lint manifest:
    cargo +1.100.0 clippy --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert,parallel --lib --test progressive_ac_contract -- -D warnings

progressive-ac-animation-evidence manifest output coverage_env:
    bash -c 'set -e; source "$1"; export CARGO_TARGET_DIR="$CARGO_LLVM_COV_TARGET_DIR"; python3 scripts/run_selected_test_evidence.py "$2" "$3" progressive_animation --test-target progressive_ac_contract --features __expert,parallel --source jxl-encoder/src/vardct/bitstream.rs --source jxl-encoder/src/vardct/ac_group.rs --source jxl-encoder/src/api/animate.rs --source jxl-encoder/src/vardct/progressive_ac_v012.tsv --source jxl-encoder/tests/images/frymire.png' _ {{quote(coverage_env)}} {{quote(manifest)}} {{quote(output)}}

progressive-ac-engine-evidence manifest output coverage_env:
    bash -c 'set -e; source "$1"; export CARGO_TARGET_DIR="$CARGO_LLVM_COV_TARGET_DIR"; python3 scripts/run_selected_test_evidence.py "$2" "$3" progressive_prepared --test-target progressive_ac_contract --features __expert,__internals,__pre_quantized,parallel --source jxl-encoder/src/vardct/bitstream.rs --source jxl-encoder/src/vardct/ac_group.rs --source jxl-encoder/src/vardct/encoder.rs --source jxl-encoder/src/vardct/progressive_ac_v012.tsv --source jxl-encoder/tests/images/frymire.png' _ {{quote(coverage_env)}} {{quote(manifest)}} {{quote(output)}}

progressive-ac-cli-evidence manifest output coverage_env:
    bash -c 'set -e; source "$1"; export CARGO_TARGET_DIR="$CARGO_LLVM_COV_TARGET_DIR"; python3 scripts/run_selected_test_evidence.py "$2" "$3" cli_progressive --package jxl-encoder-cli --test-target progressive_ac --features parallel --source jxl-encoder-cli/src/main.rs --source jxl-encoder/src/vardct/bitstream.rs --source jxl-encoder/src/vardct/ac_group.rs --source jxl-encoder/tests/images/frymire.png' _ {{quote(coverage_env)}} {{quote(manifest)}} {{quote(output)}}

progressive-ac-options-evidence manifest output coverage_env:
    bash -c 'set -e; source "$1"; export CARGO_TARGET_DIR="$CARGO_LLVM_COV_TARGET_DIR"; python3 scripts/run_selected_test_evidence.py "$2" "$3" progressive_producer_options --test-target progressive_ac_contract --features __expert,parallel --source jxl-encoder/src/vardct/bitstream.rs --source jxl-encoder/src/vardct/ac_group.rs --source jxl-encoder/src/vardct/encoder.rs --source jxl-encoder/src/vardct/progressive_ac_v012.tsv --source jxl-encoder/tests/images/frymire.png' _ {{quote(coverage_env)}} {{quote(manifest)}} {{quote(output)}}

progressive-ac-region-export llvm archive output:
    python3 scripts/coverage_from_archive.py {{quote(llvm)}} {{quote(archive)}} {{quote(output)}}

progressive-ac-core-regions llvm artifact_root:
    #!/usr/bin/env bash
    set -euo pipefail
    for name in progressive-ac-unit-first-fix progressive-ac-callers-first-fix progressive-ac-animation-corrected progressive-ac-engines-typed; do
        python3 scripts/coverage_from_archive.py {{quote(llvm)}} {{quote(artifact_root)}}/"$name" {{quote(artifact_root)}}/"$name"-ac-regions
    done
    python3 scripts/alpha_squeeze_coverage_summary.py {{quote(artifact_root)}}/progressive-ac-regions.json --scope ac --pair {{quote(artifact_root)}}/progressive-ac-unit-first-fix {{quote(artifact_root)}}/progressive-ac-unit-first-fix-ac-regions/selected.json --pair {{quote(artifact_root)}}/progressive-ac-callers-first-fix {{quote(artifact_root)}}/progressive-ac-callers-first-fix-ac-regions/selected.json --pair {{quote(artifact_root)}}/progressive-ac-animation-corrected {{quote(artifact_root)}}/progressive-ac-animation-corrected-ac-regions/selected.json --pair {{quote(artifact_root)}}/progressive-ac-engines-typed {{quote(artifact_root)}}/progressive-ac-engines-typed-ac-regions/selected.json

progressive-ac-minimal-check manifest:
    cargo test --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --no-default-features --features std,convenience,__expert --test progressive_ac_contract progressive_real_pixels -- --test-threads=1

progressive-ac-complete-lint manifest:
    cargo +1.100.0 clippy --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert,__internals,__pre_quantized,parallel --lib --test progressive_ac_contract -- -D warnings
    cargo +1.100.0 clippy --locked --manifest-path {{quote(manifest)}} -p jxl-encoder-cli --test progressive_ac -- -D warnings

progressive-ac-original-count-replay manifest output coverage_env:
    bash -c 'set -e; source "$1"; export CARGO_TARGET_DIR="$CARGO_LLVM_COV_TARGET_DIR"; python3 scripts/test_alpha_squeeze_mutation.py "$2" "$3" --case ac-flat-prefix' _ {{quote(coverage_env)}} {{quote(manifest)}} {{quote(output)}}

progressive-ac-minimal-evidence manifest output coverage_env:
    bash -c 'set -e; source "$1"; export CARGO_TARGET_DIR="$CARGO_LLVM_COV_TARGET_DIR"; python3 scripts/run_selected_test_evidence.py "$2" "$3" progressive --test-target progressive_ac_contract --no-default-features --features std,convenience,__expert --source jxl-encoder/src/vardct/bitstream.rs --source jxl-encoder/src/vardct/ac_group.rs --source jxl-encoder/src/vardct/encoder.rs --source jxl-encoder/src/vardct/progressive_ac_v012.tsv --source jxl-encoder/tests/images/frymire.png' _ {{quote(coverage_env)}} {{quote(manifest)}} {{quote(output)}}

progressive-ac-cmyk-check manifest profile output:
    cargo run --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert,parallel --example progressive_ac_cmyk -- {{quote(profile)}} {{quote(output)}}

progressive-dispatch-evidence manifest output coverage_env:
    bash -c 'set -e; source "$1"; export CARGO_TARGET_DIR="$CARGO_LLVM_COV_TARGET_DIR"; python3 scripts/run_selected_test_evidence.py "$2" "$3" requested_progressive --test-target progressive_dispatch --features __expert,parallel --source jxl-encoder/src/vardct/encoder.rs --source jxl-encoder/src/vardct/bitstream.rs --source jxl-encoder/src/vardct/ac_group.rs --source jxl-encoder/tests/images/frymire.png' _ {{quote(coverage_env)}} {{quote(manifest)}} {{quote(output)}}

progressive-dispatch-lint manifest:
    cargo +1.100.0 clippy --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert,__internals,__pre_quantized,parallel --lib --test progressive_ac_contract --test progressive_dispatch --example progressive_ac_cmyk -- -D warnings

progressive-dispatch-original-replay manifest output coverage_env:
    bash -c 'set -e; source "$1"; export CARGO_TARGET_DIR="$CARGO_LLVM_COV_TARGET_DIR"; python3 scripts/test_alpha_squeeze_mutation.py "$2" "$3" --case static-progressive' _ {{quote(coverage_env)}} {{quote(manifest)}} {{quote(output)}}

progressive-ac-specialized-evidence manifest output coverage_env:
    bash -c 'set -e; source "$1"; export CARGO_TARGET_DIR="$CARGO_LLVM_COV_TARGET_DIR"; python3 scripts/run_selected_test_evidence.py "$2" "$3" progressive_multi_dc --test-target progressive_ac_contract --features __expert,parallel --source jxl-encoder/src/vardct/bitstream.rs --source jxl-encoder/src/vardct/ac_group.rs --source jxl-encoder/src/vardct/encoder.rs --source jxl-encoder/tests/images/frymire.png' _ {{quote(coverage_env)}} {{quote(manifest)}} {{quote(output)}}

progressive-ac-patches-evidence manifest corpus output coverage_env:
    bash -c 'set -e; source "$1"; export CARGO_TARGET_DIR="$CARGO_LLVM_COV_TARGET_DIR" CODEC_CORPUS_DIR="$4"; python3 scripts/run_selected_test_evidence.py "$2" "$3" progressive_corpus_patches --test-target progressive_ac_contract --features __expert,parallel,corpus-tests --source jxl-encoder/src/vardct/bitstream.rs --source jxl-encoder/src/vardct/ac_group.rs --source jxl-encoder/src/vardct/encoder.rs --source jxl-encoder/src/vardct/patches.rs' _ {{quote(coverage_env)}} {{quote(manifest)}} {{quote(output)}} {{quote(corpus)}}

progressive-ac-reference-evidence manifest output coverage_env:
    bash -c 'set -e; source "$1"; export CARGO_TARGET_DIR="$CARGO_LLVM_COV_TARGET_DIR"; python3 scripts/run_selected_test_evidence.py "$2" "$3" progressive_reference_only --test-target progressive_ac_contract --features __expert,parallel --source jxl-encoder/src/vardct/bitstream.rs --source jxl-encoder/src/api/animate.rs --source jxl-encoder/src/headers/frame_header.rs --source jxl-encoder/tests/images/frymire.png' _ {{quote(coverage_env)}} {{quote(manifest)}} {{quote(output)}}

reference-progressive-native-oracle source build output:
    python3 scripts/frame_header_oracle.py --component reference_passes {{quote(source)}} {{quote(build)}} {{quote(output)}}

color-matrix-native-oracle source build output:
    python3 scripts/frame_header_oracle.py --component color_matrix {{quote(source)}} {{quote(build)}} {{quote(output)}}

progressive-reference-mutations manifest output coverage_env:
    bash -c 'set -e; source "$1"; export CARGO_TARGET_DIR="$CARGO_LLVM_COV_TARGET_DIR"; python3 scripts/test_alpha_squeeze_mutation.py "$2" "$3" --case reference-requested-passes --case reference-forget-regular' _ {{quote(coverage_env)}} {{quote(manifest)}} {{quote(output)}}

progressive-reference-minimal-evidence manifest output coverage_env:
    bash -c 'set -e; source "$1"; export CARGO_TARGET_DIR="$CARGO_LLVM_COV_TARGET_DIR"; python3 scripts/run_selected_test_evidence.py "$2" "$3" progressive_reference_only --test-target progressive_ac_contract --no-default-features --features std,convenience,__expert --source jxl-encoder/src/vardct/bitstream.rs --source jxl-encoder/src/api/animate.rs --source jxl-encoder/src/headers/frame_header.rs --source jxl-encoder/tests/images/frymire.png' _ {{quote(coverage_env)}} {{quote(manifest)}} {{quote(output)}}

progressive-ac-rate-control-evidence manifest output coverage_env:
    bash -c 'set -e; source "$1"; export CARGO_TARGET_DIR="$CARGO_LLVM_COV_TARGET_DIR"; python3 scripts/run_selected_test_evidence.py "$2" "$3" progressive_rate_control --test-target progressive_ac_contract --features __expert,__pre_quantized,parallel,rate-control --source jxl-encoder/src/vardct/bitstream.rs --source jxl-encoder/src/vardct/encoder.rs --source jxl-encoder/src/vardct/rate_control.rs --source jxl-encoder/src/vardct/precomputed.rs --source jxl-encoder/tests/images/frymire.png' _ {{quote(coverage_env)}} {{quote(manifest)}} {{quote(output)}}

rate-control-cli-build manifest:
    cargo check --locked --manifest-path {{quote(manifest)}} -p jxl-encoder-cli --features rate-control

progressive-ac-rate-cli-evidence manifest output coverage_env:
    bash -c 'set -e; source "$1"; export CARGO_TARGET_DIR="$CARGO_LLVM_COV_TARGET_DIR"; python3 scripts/run_selected_test_evidence.py "$2" "$3" cli_progressive_rate_control --package jxl-encoder-cli --test-target progressive_ac --features parallel,rate-control --source jxl-encoder-cli/src/main.rs --source jxl-encoder-cli/Cargo.toml --source jxl-encoder/src/vardct/bitstream.rs --source jxl-encoder/src/vardct/encoder.rs --source jxl-encoder/src/vardct/rate_control.rs --source jxl-encoder/tests/images/frymire.png' _ {{quote(coverage_env)}} {{quote(manifest)}} {{quote(output)}}

progressive-reference-final-lint manifest:
    cargo +1.100.0 clippy --locked --manifest-path {{quote(manifest)}} -p jxl-encoder --features __expert,__internals,__pre_quantized,parallel,corpus-tests,rate-control --lib --test progressive_ac_contract --test progressive_dispatch --example progressive_ac_cmyk -- -D warnings
    cargo +1.100.0 clippy --locked --manifest-path {{quote(manifest)}} -p jxl-encoder-cli --features rate-control --all-targets -- -D warnings

progressive-ac-final-regions llvm artifacts output:
    python3 scripts/progressive_ac_coverage.py {{quote(llvm)}} {{quote(artifacts)}} {{quote(output)}}

progressive-ac-typed-animation-evidence manifest output coverage_env:
    bash -c 'set -e; source "$1"; export CARGO_TARGET_DIR="$CARGO_LLVM_COV_TARGET_DIR"; python3 scripts/run_selected_test_evidence.py "$2" "$3" progressive_typed_animation --test-target progressive_ac_contract --features __expert,parallel --source jxl-encoder/src/vardct/bitstream.rs --source jxl-encoder/src/api/animate.rs --source jxl-encoder/src/api.rs --source jxl-encoder/tests/images/frymire.png' _ {{quote(coverage_env)}} {{quote(manifest)}} {{quote(output)}}

progressive-ac-typed-minimal-evidence manifest output coverage_env:
    bash -c 'set -e; source "$1"; export CARGO_TARGET_DIR="$CARGO_LLVM_COV_TARGET_DIR"; python3 scripts/run_selected_test_evidence.py "$2" "$3" progressive_typed --test-target progressive_ac_contract --no-default-features --features std,convenience,__expert --source jxl-encoder/src/vardct/bitstream.rs --source jxl-encoder/src/api/animate.rs --source jxl-encoder/src/api.rs --source jxl-encoder/tests/images/frymire.png' _ {{quote(coverage_env)}} {{quote(manifest)}} {{quote(output)}}

animation-alpha-native source build output:
    python3 scripts/frame_header_oracle.py --component animation_alpha {{quote(source)}} {{quote(build)}} {{quote(output)}}

animation-alpha-native-check log djxl output:
    python3 scripts/check_animation_alpha_native.py {{quote(log)}} {{quote(djxl)}} {{quote(output)}}
