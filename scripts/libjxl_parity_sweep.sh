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
#   OURS     path to our CLI                     (default: target/release/cjxl-rs)
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

[ -x "$CJXL" ] || { echo "cjxl not found (set CJXL=)"; exit 2; }
[ -x "$OURS" ] || { echo "our CLI not built at $OURS (cargo build --release -p jxl-encoder-cli)"; exit 2; }

mkdir -p "$OUT" "$FIX"
python3 "$HERE/make_parity_fixtures.py" "$FIX" >/dev/null

DATE=$(date -u +%Y-%m-%d)
TSV="$OUT/libjxl_parity_$DATE.tsv"
META="$OUT/libjxl_parity_$DATE.meta"

{
  echo "# T4 libjxl mimic byte-parity sweep"
  echo "date_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host: $(uname -srm) / $(hostname)"
  echo "git_commit: $(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "cjxl_path: $CJXL"
  echo "cjxl_version: $("$CJXL" --version 2>&1 | head -1)"
  echo "ours_path: $OURS"
  echo "ours_version: $("$OURS" --version 2>&1 | head -1)"
  echo "efforts: $EFFORTS"
  echo "distances: $DISTS"
  echo "note: ours always runs --strategy libjxl (the mimic); zen mode is a"
  echo "note: different bundle and is NOT what this sweep measures."
} > "$META"

printf 'image\twidth\theight\teffort\tdistance\tkind\tname\tcjxl\tours\tdelta\n' > "$TSV"

for png in "$FIX"/*.png; do
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
      a="$OUT/.cj_${base}_e${e}_d${d}.jxl"
      b="$OUT/.our_${base}_e${e}_d${d}.jxl"
      "$CJXL" -d "$d" -e "$e" --num_threads=0 "$png" "$a" >/dev/null 2>&1 || continue
      "$OURS" -d "$d" -e "$e" --strategy libjxl "$png" "$b" >/dev/null 2>&1 || continue
      python3 "$HERE/jxl_bitstream_diff.py" tsv "$a" "$b" 2>/dev/null \
        | tail -n +2 \
        | while IFS=$'\t' read -r kind name va vb; do
            if [ -n "$va" ] && [ -n "$vb" ] && [ "$kind" != "field" ]; then
              delta=$((vb - va))
            else
              delta=""
            fi
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
              "$base" "$dims" "$e" "$d" "$kind" "$name" "$va" "$vb" "$delta"
          done >> "$TSV"
      rm -f "$a" "$b"
    done
  done
done

echo "wrote $TSV"
echo "wrote $META"
python3 - "$TSV" <<'PY'
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
print(f"  {'section':22s} {'cjxl':>10s} {'ours':>10s} {'delta':>10s} {'pct':>8s}")
for k, (a, b) in sorted(agg.items(), key=lambda kv: -(kv[1][1] - kv[1][0])):
    pct = (b - a) / a * 100 if a else 0.0
    print(f"  {k:22s} {a:10d} {b:10d} {b - a:+10d} {pct:+7.1f}%")
ta = sum(v[0] for v in agg.values())
tb = sum(v[1] for v in agg.values())
print(f"  {'ALL':22s} {ta:10d} {tb:10d} {tb - ta:+10d} "
      f"{((tb - ta) / ta * 100 if ta else 0):+7.1f}%")
PY
