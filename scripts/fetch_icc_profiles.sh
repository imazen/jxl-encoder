#!/usr/bin/env bash
# Fetch the real-world ICC profiles the conformance sweep embeds:
# Ubuntu's icc-profiles and icc-profiles-free packages plus W3C png-hdr-pq's
# ITU-R BT.2100 PQ profile. Usage: scripts/fetch_icc_profiles.sh <out-dir>
# Needs apt-get (Debian/Ubuntu) for the package downloads.
set -euo pipefail
out="${1:?usage: $0 <out-dir>}"
base="${TMPDIR:-$HOME/tmp}"
mkdir -p "$base"
work="$(mktemp -d "$base/icc-fetch.XXXXXX")"
trap 'rm -rf "$work"' EXIT
mkdir -p "$out"
(cd "$work" && apt-get download icc-profiles icc-profiles-free >/dev/null)
for deb in "$work"/*.deb; do
    dpkg-deb -x "$deb" "$work/x"
done
find "$work/x/usr/share/color/icc" -type f -exec cp --update=none {} "$out/" \;
pq="$out/ITUR_2100_PQ_FULL.ICC"
curl -fsSL -o "$pq" https://raw.githubusercontent.com/w3c/png-hdr-pq/main/icc/ITUR_2100_PQ_FULL.ICC
echo "8a66e7e397f1a833f34b584681857589144a1da01918e2b52447c56bdebd5e4f  $pq" | sha256sum --check --quiet
# Keep only files that are ICC profiles ('acsp' at offset 36).
for f in "$out"/*; do
    [ "$(dd if="$f" bs=1 skip=36 count=4 2>/dev/null)" = "acsp" ] || rm -f "$f"
done
echo "$(ls "$out" | wc -l) ICC profiles in $out"
