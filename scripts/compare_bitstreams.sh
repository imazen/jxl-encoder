#!/bin/bash
# Compare our encoder's output with libjxl reference
#
# Usage: ./scripts/compare_bitstreams.sh <test_name>
#   test_name: e.g., "lossy_8x8", "lossless_gradient"

set -e

if [ $# -ne 1 ]; then
    echo "Usage: $0 <test_name>"
    echo "Example: $0 lossy_8x8"
    exit 1
fi

TEST_NAME=$1
CJXL=~/work/jxl-efforts/libjxl/build/tools/cjxl
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "=== Bitstream Comparison: $TEST_NAME ==="
echo ""

# Step 1: Generate reference file with libjxl
echo "Step 1: Generating reference with libjxl..."
if [ "$TEST_NAME" = "lossy_8x8" ]; then
    convert -size 8x8 xc:red /tmp/test_${TEST_NAME}.png
    $CJXL -d 1.0 /tmp/test_${TEST_NAME}.png /tmp/ref_${TEST_NAME}.jxl
elif [ "$TEST_NAME" = "lossless_gradient" ]; then
    convert -size 16x16 gradient:red-blue /tmp/test_${TEST_NAME}.png
    $CJXL -q 100 /tmp/test_${TEST_NAME}.png /tmp/ref_${TEST_NAME}.jxl
else
    echo "Unknown test: $TEST_NAME"
    echo "Add test generation logic for this test name"
    exit 1
fi

echo "  Reference: /tmp/ref_${TEST_NAME}.jxl ($(stat -c%s /tmp/ref_${TEST_NAME}.jxl) bytes)"
echo ""

# Step 2: Generate our encoder's output
echo "Step 2: Generating output with our encoder..."
echo "  (Run your test to generate /tmp/our_${TEST_NAME}.jxl)"
echo ""

# Step 3: Dump reference bitstream
echo "Step 3: Dumping reference bitstream..."
cargo run --example dump_bitstream /tmp/ref_${TEST_NAME}.jxl > /tmp/ref_${TEST_NAME}_dump.txt 2>&1
echo "  Saved to: /tmp/ref_${TEST_NAME}_dump.txt"
echo ""

# Step 4: Hex comparison
echo "Step 4: Hex comparison..."
xxd /tmp/ref_${TEST_NAME}.jxl > /tmp/ref_${TEST_NAME}.hex
echo "  Reference hex: /tmp/ref_${TEST_NAME}.hex"

if [ -f "/tmp/our_${TEST_NAME}.jxl" ]; then
    xxd /tmp/our_${TEST_NAME}.jxl > /tmp/our_${TEST_NAME}.hex
    echo "  Our hex: /tmp/our_${TEST_NAME}.hex"
    echo ""
    echo "Byte-by-byte comparison:"
    diff -y --width=200 /tmp/ref_${TEST_NAME}.hex /tmp/our_${TEST_NAME}.hex | head -50 || true
else
    echo "  Our file not found: /tmp/our_${TEST_NAME}.jxl"
    echo "  Generate it by running the appropriate test"
fi

echo ""
echo "=== Next Steps ==="
echo "1. View reference bitstream dump: less /tmp/ref_${TEST_NAME}_dump.txt"
echo "2. Run your test with tracing: cargo test --features trace-bitstream <test_name> -- --nocapture"
echo "3. Compare hex dumps to find divergence point"
