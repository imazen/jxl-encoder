#!/usr/bin/env bash
# build_and_push_image.sh — stage zenjxl-tuning-runner + zen-metrics
# binaries from the local shared-target dirs, build the fleet image,
# push to ghcr.io/lilith/zenjxl-tuning-sweep:v3-schema-v2-<8sha>.
#
# Tag scheme history:
#   v1                   (early W44-212, pre-CUDA fleet image)
#   v2-<sha>             (W44-215+, CUDA NVRTC + dlsym shim,
#                         parquet-v1 runner — 43 cols, no artifacts)
#   v3-schema-v2-<8sha>  (W44-PHASE4-S1g, 2026-05-24, runner emits
#                         parquet-v2 — 55 cols + stages encoded JXL
#                         + diffmap blobs for worker.sh to upload)
#
# The v3 tag bump signals two coupled binary changes:
#   1. zenjxl-tuning-runner builds the schema-v2 Parquet (12 new cols:
#      encoded_jxl_sha256/r2_key, diffmap_r2_key, butter_max/p1/p2/p6,
#      psnr_y/r/g/b, ms_ssim — see W44-PHASE4-M1 commit d2129886)
#   2. worker.sh exports W44_PHASE4_M1_SAVE_{ENCODED,DIFFMAP} +
#      W44_PHASE4_M1_COMPUTE_MULTIMETRIC default-ON and uploads the
#      staged artifacts to s3://<bucket>/<sweep>/artifacts/{jxl,diffmap}/
#
# Sweeps using the old v2-<sha> image keep working but silently lose
# encoded bytes (the very problem CLAUDE.md §4 made a hard rule).
#
# Assumes:
#   - $PWD == /home/lilith/work/zen/jxl-encoder--<sibling>
#     (or any sibling jxl-encoder workspace where ./scripts is present)
#   - shared cargo target dirs exist:
#       /home/lilith/work/zen/jxl-encoder-shared-target/release/zenjxl-tuning-runner
#       /home/lilith/work/zen/zenmetrics/target/release/zen-metrics
#     (rebuild via the scripts/build_release.sh in those workspaces if missing)
#   - gh auth token is valid (write:packages scope)
#
# Usage:
#   ./scripts/zenjxl-tuning-sweep/build_and_push_image.sh [COMMIT-SHA-OR-TAG]
#
# If COMMIT is omitted, derives from the parent repo's git HEAD as an
# 8-char short SHA. The tags emitted are:
#   ghcr.io/lilith/zenjxl-tuning-sweep:v3-schema-v2-<8charsha>
#   ghcr.io/lilith/zenjxl-tuning-sweep:v3-schema-v2   (floating)

set -euo pipefail

COMMIT="${1:-}"
if [[ -z "$COMMIT" ]]; then
    # try parent repo (sibling workspaces don't have .git)
    if [[ -d /home/lilith/work/zen/jxl-encoder/.git ]]; then
        COMMIT="$(git -C /home/lilith/work/zen/jxl-encoder rev-parse --short=8 HEAD)"
    else
        echo "ERROR: pass commit-sha explicitly OR run from a workspace with .git" >&2
        exit 1
    fi
fi

IMG_TAG="ghcr.io/lilith/zenjxl-tuning-sweep:v3-schema-v2-${COMMIT}"
IMG_FLOAT="ghcr.io/lilith/zenjxl-tuning-sweep:v3-schema-v2"

RUNNER_SRC="/home/lilith/work/zen/jxl-encoder-shared-target/release/zenjxl-tuning-runner"
METRICS_SRC="/home/lilith/work/zen/zenmetrics/target/release/zen-metrics"

[[ -x "$RUNNER_SRC" ]] || { echo "ERROR: $RUNNER_SRC missing — rebuild zenjxl-tuning-runner" >&2; exit 1; }
[[ -x "$METRICS_SRC" ]] || { echo "ERROR: $METRICS_SRC missing — rebuild zen-metrics" >&2; exit 1; }

echo "[build] staging binaries into $(pwd)"
cp "$RUNNER_SRC" ./zenjxl-tuning-runner-bin
cp "$METRICS_SRC" ./zen-metrics-bin
ls -lah ./zenjxl-tuning-runner-bin ./zen-metrics-bin

echo "[build] docker build -> $IMG_TAG"
docker build \
    -f scripts/zenjxl-tuning-sweep/Dockerfile.zenjxl-tuning-sweep.v2 \
    --build-arg ZEN_METRICS_BINARY=./zen-metrics-bin \
    --build-arg RUNNER_BINARY=./zenjxl-tuning-runner-bin \
    -t "$IMG_TAG" \
    -t "$IMG_FLOAT" \
    . > /tmp/zenjxl-tuning-sweep-build.log 2>&1
RC=$?
if (( RC != 0 )); then
    echo "[build] FAILED rc=$RC — last 40 lines:" >&2
    tail -40 /tmp/zenjxl-tuning-sweep-build.log >&2
    exit "$RC"
fi
echo "[build] OK (full log /tmp/zenjxl-tuning-sweep-build.log)"

echo "[push] docker push $IMG_TAG"
docker push "$IMG_TAG"
echo "[push] docker push $IMG_FLOAT"
docker push "$IMG_FLOAT"

echo
echo "DONE. Image: $IMG_TAG"
echo "      Float: $IMG_FLOAT"
echo
echo "Pull on vast.ai requires private-image auth via --login:"
echo "  -u lilith -p \$(gh auth token) ghcr.io"
