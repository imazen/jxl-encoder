#!/usr/bin/env bash
# sectioned_k_bar_cell.sh — the #99 item-1 WALL BAR cell, measured with
# repeats, for the sectioned content-adaptive predictor selector.
#
# The bar is "sectioned lossless wall <= 1.3x cjxl v0.12 on the same input".
# The reference cell (issue #99 / `benchmarks/jxl_sectioned_prune_k_2026-08-28`)
# is the imazen-26 1403 photo cropped TOP-LEFT to 3840x2160, at e7 and e9,
# threads 1 and 8.
#
# Two wall conventions are reported because they differ materially at t=8:
#   cjxl_ms  = whole process, INCLUDING PNG decode (the convention #99 used)
#   ours_ms  = encode only (the probe harness times `encode()`)
# At t=1 the PNG load is noise; at t=8 it is ~10 % of cjxl's number, so the
# process-wall ratio flatters us. Both are in the output.
#
# Repeats matter: t=8 cells vary run to run with worker scheduling (measured
# up to 15 % on this cell), t=1 cells are stable to ~2 %. Every repetition
# is retained; calculate summary statistics from the TSV.
#
# Usage: sectioned_k_bar_cell.sh <out.tsv> <bar_crop.png> [reps]
set -euo pipefail
OUT="${1:?usage: sectioned_k_bar_cell.sh <out.tsv> <bar.png> [reps]}"
IMG="${2:?bar crop png}"
REPS="${3:-5}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROBE="${SECTIONED_K_PROBE:-$ROOT/target/release/examples/sectioned_k_corpus}"
[ -x "$PROBE" ] || { echo "build sectioned_k_corpus with std,parallel,profile-phases" >&2; exit 2; }
[[ "$REPS" =~ ^[1-9][0-9]*$ ]] || { echo "reps must be positive" >&2; exit 2; }
[ ! -e "$OUT" ] || { echo "refusing to overwrite $OUT" >&2; exit 2; }
mkdir -p "$(dirname "$OUT")"
# The Rust helper checks the version and refuses packaged v0.11 tools.
CJXL=$("$PROBE" reference-tools)
ARTIFACT_DIR="${ARTIFACT_DIR:-${OUT%.tsv}.artifacts}"
export ARTIFACT_DIR
mkdir -p "$ARTIFACT_DIR"
LOG_DIR="${OUT%.tsv}.logs"
mkdir -p "$LOG_DIR"
"$CJXL" --version > "$LOG_DIR/cjxl-version.log" 2>&1
{
  printf '# sectioned_k_bar_cell.sh commit=%s host=%s date=%s reps=%s\n' \
    "$(jj -R "$ROOT" log --no-graph -r @ -T commit_id)" "$(hostname)" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$REPS"
  printf '# img=%s sha256=%s cjxl=%s probe_sha256=%s artifacts=%s\n' "$IMG" \
    "$(shasum -a 256 "$IMG" | cut -d' ' -f1)" "$CJXL" \
    "$(shasum -a 256 "$PROBE" | cut -d' ' -f1)" "$ARTIFACT_DIR"
  cat "$LOG_DIR/cjxl-version.log"
} > "${OUT}.meta"
printf 'encoder\teffort\tthreads\tarm\trep\tbytes\twall_ms\tencoded_sha256\n' > "$OUT"

reference_cell() {
  local cell="e${E}-t${T}-r${r}"
  local output="$ARTIFACT_DIR/cjxl-${cell}.jxl"
  local ct="$T"
  [ "$T" != 1 ] || ct=0
  # Unique output names plus set -e prevent a failed encode from reusing bytes.
  [ ! -e "$output" ] || { echo "artifact already exists: $output" >&2; exit 2; }
  /usr/bin/time -p nice -n 19 "$CJXL" -d 0 -e "$E" --num_threads="$ct" \
    "$IMG" "$output" > "$LOG_DIR/cjxl-${cell}.log" 2>&1
  local seconds bytes sha
  seconds=$(awk '/^real / {s=$2; n++} END {if(n != 1) exit 1; print s}' "$LOG_DIR/cjxl-${cell}.log")
  bytes=$(wc -c < "$output" | tr -d ' ')
  [ "$bytes" -gt 0 ]
  sha=$(shasum -a 256 "$output" | cut -d' ' -f1)
  cp "$output" "$ARTIFACT_DIR/$sha.jxl"
  printf 'cjxl\t%s\t%s\tprocess\t%s\t%s\t%s\t%s\n' "$E" "$T" "$r" "$bytes" \
    "$(awk -v s="$seconds" 'BEGIN{printf "%.1f", s*1000}')" "$sha" >> "$OUT"
}

ours_cell() {
  local log="$LOG_DIR/ours-e${E}-t${T}-r${r}.log"
  local arms=(k8 default)
  if (( r % 2 == 0 )); then arms=(default k8); fi
  nice -n 19 "$PROBE" phases "$IMG" "$E" "$T" "${arms[@]}" > "$log" 2>&1
  awk -v e="$E" -v t="$T" -v r="$r" '
    /^== / {a=$2; sub(":$", "", a); bytes[a]=$3; wall[a]=$5}
    /^artifact / {
      a=$2
      if (!(a in bytes) || bytes[a] <= 0 || length($3) != 64) exit 1
      printf "ours\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", e,t,a,r,bytes[a],wall[a],$3
      seen[a]++
    }
    END {if(seen["k8"] != 1 || seen["default"] != 1) exit 1}
  ' "$log" >> "$OUT"
}

for E in 7 9; do
  for T in 1 8; do
    for ((r=1; r<=REPS; r++)); do
      # Alternate the reference/process order and the in-process arm order.
      if (( r % 2 == 1 )); then reference_cell; ours_cell
      else ours_cell; reference_cell; fi
      echo "completed e$E t$T repeat $r/$REPS" | tee -a "$LOG_DIR/progress.log"
    done
  done
done
echo "wrote $OUT"
