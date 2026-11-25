#!/usr/bin/env bash
# T4 libjxl-mimic byte-parity sweep.
#
# Encodes each fixture with BOTH cjxl and our `--strategy libjxl` at matched
# effort/distance, then emits one TSV row per TOC section with the per-section
# byte delta, plus one row per diverging header field. That is the unit that
# makes a residual actionable: "HfGlobal is +33 %" points at the ANS/dequant
# global structures, where "byte 4711 onward" points at nothing.
#
#   scripts/libjxl_parity_sweep.sh <outdir> [fixture-dir]
#
# Env:
#   CJXL     path to reference cjxl              (default: `command -v cjxl`)
#   CJXL_VERSION           expected first cjxl version (default: v0.12.0)
#   OURS                   path to our CLI (default: target/release/cjxl-rs)
#   COMPARE_CJXL           compare a second cjxl in place of our encoder
#   COMPARE_CJXL_VERSION   expected second cjxl version (default: v0.13.0)
#   KEEP_STREAMS           1 retains successful encodes for decoder checks
#   EFFORTS  space-separated effort list         (default: "3 5 7")
#   DISTS    space-separated distance list       (default: "0.5 1.0 4.0")
#
# Writes <outdir>/libjxl_parity_<date>.tsv and a .meta sidecar recording the
# host, both encoder versions and the git commit — a parity number without the
# reference build it was taken against is not comparable to anything.
set -euo pipefail

OUT=${1:?usage: libjxl_parity_sweep.sh <outdir> [fixture-dir]}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
FIX=${2:-$OUT/fixtures}
CJXL=${CJXL:-$(command -v cjxl || true)}
OURS=${OURS:-$ROOT/target/release/cjxl-rs}
EFFORTS=${EFFORTS:-"3 5 7"}
DISTS=${DISTS:-"0.5 1.0 4.0"}

cjxl_version() {
  local tool=$1 expected=$2 output name version rest
  if ! output=$("$tool" --version 2>&1); then
    printf '%s --version failed: %s\n' "$tool" "$output" >&2
    return 2
  fi
  read -r name version rest <<< "$output"
  if [[ "$name" != cjxl || "$version" != "$expected" ]]; then
    printf '%s must report cjxl %s. Got: %s\n' "$tool" "$expected" "$output" >&2
    return 2
  fi
  printf '%s\n' "${output%%$'\n'*}"
}

[ -x "$CJXL" ] || { echo "cjxl not found (set CJXL=)" >&2; exit 2; }
ref_version=$(cjxl_version "$CJXL" "${CJXL_VERSION:-v0.12.0}")

if [[ -n ${COMPARE_CJXL:-} ]]; then
  compare=$COMPARE_CJXL
  compare_version=$(cjxl_version "$compare" "${COMPARE_CJXL_VERSION:-v0.13.0}")
  a_label=${CJXL_VERSION:-v0.12.0}
  b_label=${COMPARE_CJXL_VERSION:-v0.13.0}
  compare_args=(--num_threads=0)
else
  compare=$OURS
  [ -x "$compare" ] || { echo "our CLI not built at $compare" >&2; exit 2; }
  compare_version=$("$compare" --version)
  a_label=cjxl
  b_label=ours
  compare_args=(--strategy libjxl)
fi

DATE=$(date -u +%Y-%m-%d)
TSV="$OUT/libjxl_parity_$DATE.tsv"
META="$OUT/libjxl_parity_$DATE.meta"
if [[ -e "$TSV" || -e "$META" ]]; then
  printf 'Results already exist in %s\n' "$OUT" >&2
  exit 2
fi

mkdir -p "$OUT"
if [[ $# -lt 2 ]]; then
  mkdir -p "$FIX"
  python3 "$HERE/make_parity_fixtures.py" "$FIX" >/dev/null
fi
shopt -s nullglob
fixtures=("$FIX"/*.png)
if [[ ${#fixtures[@]} -eq 0 ]]; then
  printf 'No PNG fixtures in %s\n' "$FIX" >&2
  exit 2
fi

{
  echo "# libjxl envelope parity sweep"
  echo "date_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host: $(uname -srm) / $(hostname)"
  echo "git_commit: $(git -C "$ROOT" rev-parse HEAD)"
  echo "cjxl_path: $CJXL"
  echo "cjxl_version: $ref_version"
  echo "comparison_path: $compare"
  echo "comparison_version: $compare_version"
  echo "efforts: $EFFORTS"
  echo "distances: $DISTS"
  printf 'comparison_args: %s\n' "${compare_args[*]}"
} > "$META"

printf 'image\twidth\theight\teffort\tdistance\tkind\tname\t%s\t%s\tdelta\n' "$a_label" "$b_label" > "$TSV"

for png in "${fixtures[@]}"; do
  base=$(basename "$png" .png)
  dims=$(python3 - "$png" <<'PY'
import struct, sys
d = open(sys.argv[1], 'rb').read()
w, h = struct.unpack('>II', d[16:24])
print(f"{w}\t{h}")
PY
)
  for e in $EFFORTS; do
    for d in $DISTS; do
      a="$OUT/.a_${base}_e${e}_d${d}.jxl"
      b="$OUT/.b_${base}_e${e}_d${d}.jxl"
      log="$OUT/${base}_e${e}_d${d}.log"
      if ! { "$CJXL" -d "$d" -e "$e" --num_threads=0 "$png" "$a" &&
             "$compare" -d "$d" -e "$e" "${compare_args[@]}" "$png" "$b"; } >"$log" 2>&1; then
        printf 'Encode failed for %s e%s d%s. See %s\n' "$base" "$e" "$d" "$log" >&2
        exit 1
      fi
      if ! python3 "$HERE/jxl_bitstream_diff.py" tsv "$a" "$b" 2>>"$log" \
        | python3 -c '
import csv, sys
reader = csv.reader(sys.stdin, delimiter="\t")
writer = csv.writer(sys.stdout, delimiter="\t", lineterminator="\n")
image, dims, effort, distance = sys.argv[1:]
w, h = dims.split("\t")
next(reader)
for kind, name, a, b in reader:
    delta = int(b) - int(a) if kind in ("size", "section") and a.isdecimal() and b.isdecimal() else ""
    writer.writerow((image, w, h, effort, distance, kind, name, a, b, delta))
' "$base" "$dims" "$e" "$d" >> "$TSV"; then
        printf 'Comparison failed for %s e%s d%s. See %s\n' "$base" "$e" "$d" "$log" >&2
        exit 1
      fi
      if [[ ${KEEP_STREAMS:-0} == 0 ]]; then rm -f "$a" "$b"; fi
    done
  done
done

echo "wrote $TSV"
echo "wrote $META"
python3 - "$TSV" "$a_label" "$b_label" <<'PY'
import collections, sys
rows = [l.rstrip('\n').split('\t') for l in open(sys.argv[1])][1:]
agg = collections.defaultdict(lambda: [0, 0])
for r in rows:
    if r[5] != 'section':
        continue
    name = r[6].split('[')[0]
    try:
        a, b = int(r[7]), int(r[8])
    except ValueError:
        continue
    agg[name][0] += a
    agg[name][1] += b
print()
print(f"  {'section':22s} {sys.argv[2]:>10s} {sys.argv[3]:>10s} {'delta':>10s} {'pct':>8s}")
for k, (a, b) in sorted(agg.items(), key=lambda kv: -(kv[1][1] - kv[1][0])):
    pct = (b - a) / a * 100 if a else 0.0
    print(f"  {k:22s} {a:10d} {b:10d} {b - a:+10d} {pct:+7.1f}%")
ta = sum(v[0] for v in agg.values())
tb = sum(v[1] for v in agg.values())
print(f"  {'ALL':22s} {ta:10d} {tb:10d} {tb - ta:+10d} "
      f"{((tb - ta) / ta * 100 if ta else 0):+7.1f}%")
PY
