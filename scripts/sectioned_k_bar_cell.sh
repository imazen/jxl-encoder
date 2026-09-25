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
# Set SECTIONED_K_BASELINE_PROBE for an interleaved binary A/B of the default
# policy instead of the within-binary k8/default comparison.
# Usage: sectioned_k_bar_cell.sh <out.tsv> <bar_crop.png> [reps] [efforts] [threads]
set -euo pipefail
OUT="${1:?usage: sectioned_k_bar_cell.sh <out.tsv> <bar.png> [reps]}"
IMG="${2:?bar crop png}"
REPS="${3:-5}"
EFFORTS="${4:-7 9}"
THREADS="${5:-1 8}"
[[ "$EFFORTS" =~ ^[1-9][0-9\ ]*$ && "$THREADS" =~ ^[1-9][0-9\ ]*$ ]] || {
  echo "efforts and threads must be space-separated positive integers" >&2; exit 2;
}
for value in $EFFORTS $THREADS; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "invalid grid value: $value" >&2; exit 2; }
done
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROBE="${SECTIONED_K_PROBE:-$ROOT/target/release/examples/sectioned_k_corpus}"
BASE_PROBE="${SECTIONED_K_BASELINE_PROBE:-}"
if [ -n "$BASE_PROBE" ]; then
  [ -x "$BASE_PROBE" ] || { echo "baseline probe is not executable: $BASE_PROBE" >&2; exit 2; }
fi
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
  if [ -n "$BASE_PROBE" ]; then
    printf '# baseline_probe=%s sha256=%s\n' "$BASE_PROBE" "$(shasum -a 256 "$BASE_PROBE" | cut -d' ' -f1)"
  fi
  printf '# efforts=%s threads=%s\n' "$EFFORTS" "$THREADS"
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

probe_cell() {
  local binary="$1" encoder="$2"
  shift 2
  local arms=("$@")
  local log="$LOG_DIR/${encoder}-e${E}-t${T}-r${r}.log"
  nice -n 19 "$binary" phases "$IMG" "$E" "$T" "${arms[@]}" > "$log" 2>&1
  awk -v enc="$encoder" -v expected="${arms[*]}" -v e="$E" -v t="$T" -v r="$r" '
    /^== / {a=$2; sub(":$", "", a); bytes[a]=$3; wall[a]=$5}
    /^artifact / {
      a=$2
      if (!(a in bytes) || bytes[a] <= 0 || length($3) != 64) exit 1
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", enc,e,t,a,r,bytes[a],wall[a],$3
      seen[a]++
    }
    END {
      n=split(expected, arms, " ")
      for(i=1;i<=n;i++) if(seen[arms[i]] != 1) exit 1
    }
  ' "$log" >> "$OUT"
}

ours_cell() {
  if [ -n "$BASE_PROBE" ]; then
    if (( r % 2 == 1 )); then
      probe_cell "$BASE_PROBE" baseline default
      probe_cell "$PROBE" ours default
    else
      probe_cell "$PROBE" ours default
      probe_cell "$BASE_PROBE" baseline default
    fi
  else
    local arms=(k8 default)
    if (( r % 2 == 0 )); then arms=(default k8); fi
    probe_cell "$PROBE" ours "${arms[@]}"
  fi
}

for E in $EFFORTS; do
  for T in $THREADS; do
    for ((r=1; r<=REPS; r++)); do
      # Alternate the reference/process order and the in-process arm order.
      if (( r % 2 == 1 )); then reference_cell; ours_cell
      else ours_cell; reference_cell; fi
      echo "completed e$E t$T repeat $r/$REPS" | tee -a "$LOG_DIR/progress.log"
    done
  done
done
echo "wrote $OUT"
