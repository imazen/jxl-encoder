#!/usr/bin/env bash
# W44-phase3-B7c paired interleaved A/B reproducer.
#
# Builds the encode bench TWICE, once against butteraugli at the
# B7a+b parent (Mutex pool) and once against a hypothetical B7c
# (TLS pool), then runs them alternately to produce paired walls.
#
# The B7c source change was reverted on butteraugli/main after this
# script measured +3.6 % wall regression. To re-measure / re-attempt,
# point BUTTERAUGLI_B7C_REF at any commit / branch implementing a TLS
# pool variant.
#
# Usage:
#   BUTTERAUGLI_PRE_REF=dd12b90f \
#   BUTTERAUGLI_POST_REF=59aeb1a7 \
#   ./scripts/run_w44_phase3_b7c_ab.sh
#
# Output:
#   /tmp/w44_b7c_ab_rounds/{pre,post}_r{1..3}.tsv
#   per-round per-cell median wall_us, paired across pre/post.
#
# Acceptance gate: median wall improved ≥ 1 % on the post side, OR
# honest-stop if regression observed.

set -euo pipefail

PRE_REF="${BUTTERAUGLI_PRE_REF:-dd12b90f}"   # B7a+b
POST_REF="${BUTTERAUGLI_POST_REF:-59aeb1a7}"  # B7c (reverted on main)
TARGET="${CARGO_TARGET_DIR:-$HOME/work/zen/jxl-encoder-shared-target}"
OUT_DIR="${OUT_DIR:-/tmp/w44_b7c_ab_rounds}"
ROUNDS="${ROUNDS:-3}"

mkdir -p "$OUT_DIR"
rm -f "$OUT_DIR"/*.tsv

# Helper: prepare a butteraugli sibling at the requested git ref.
# Uses jj workspace add for jj-managed parent repo.
prep_butteraugli() {
  local ref="$1"
  local sibling="/home/lilith/work/butteraugli--w44-phase3-b7c-$ref-ws"
  if [ ! -d "$sibling" ]; then
    cd /home/lilith/work/butteraugli
    jj workspace add "$sibling" -r "$ref" --name "w44-b7c-$ref-ws" 2>&1 | tail -3
  fi
  echo "$sibling"
}

build_at_butteraugli() {
  local sibling="$1"
  local bin_name="$2"
  # Temporarily swap the encoder's butteraugli path to the sibling.
  local cargo_toml=/home/lilith/work/zen/jxl-encoder--w44-phase3-b7c/Cargo.toml
  local backup="$cargo_toml.bak"
  cp "$cargo_toml" "$backup"
  sed -i "s|butteraugli = { path = \"../../butteraugli[^\"]*\" }|butteraugli = { path = \"$sibling/butteraugli\" }|" "$cargo_toml"
  cd /home/lilith/work/zen/jxl-encoder--w44-phase3-b7c
  touch "$sibling/butteraugli/src/image.rs"
  CARGO_TARGET_DIR="$TARGET" cargo build --release \
    --example w44_phase3_b7c_tls_pool_ab \
    --features "__expert butteraugli-loop parallel" 2>&1 | tail -3
  cp "$TARGET/release/examples/w44_phase3_b7c_tls_pool_ab" "/tmp/$bin_name"
  mv "$backup" "$cargo_toml"
}

PRE_SIBLING=$(prep_butteraugli "$PRE_REF" | tail -1)
build_at_butteraugli "$PRE_SIBLING" "bench_pre_b7c"

POST_SIBLING=$(prep_butteraugli "$POST_REF" | tail -1)
build_at_butteraugli "$POST_SIBLING" "bench_post_b7c"

for r in $(seq 1 $ROUNDS); do
  if (( r % 2 == 1 )); then
    B7C_LABEL="PRE_B7C_r${r}" /tmp/bench_pre_b7c "$OUT_DIR/pre_r${r}.tsv" 2>&1 | tail -5
    B7C_LABEL="POST_B7C_r${r}" /tmp/bench_post_b7c "$OUT_DIR/post_r${r}.tsv" 2>&1 | tail -5
  else
    B7C_LABEL="POST_B7C_r${r}" /tmp/bench_post_b7c "$OUT_DIR/post_r${r}.tsv" 2>&1 | tail -5
    B7C_LABEL="PRE_B7C_r${r}" /tmp/bench_pre_b7c "$OUT_DIR/pre_r${r}.tsv" 2>&1 | tail -5
  fi
done

echo "DONE: $OUT_DIR"
ls "$OUT_DIR"
