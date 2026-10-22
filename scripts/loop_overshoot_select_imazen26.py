#!/usr/bin/env python3
"""Stratified imazen-26 selection for examples/loop_overshoot.rs.

Usage: loop_overshoot_select_imazen26.py <imazen-26-png-v3 checkout> <split> <per_class>

Prints `png-v3/...sdr.png` paths (relative to the checkout), `per_class`
per content class, evenly spaced over the class's ids in
`manifests/<split>.tsv` (sorted by id), so the pick is deterministic and
covers each class's id range rather than its first few files. Renders
whose published name differs from the manifest (rotated photos) are
resolved through `RENDER-NAME-MAP.tsv`.
"""

import csv
import sys
from collections import defaultdict

PREFIX = "https://codec-corpus.r2.imazen.org/imazen-26-png-v3/"


def main():
    root, split, k = sys.argv[1], sys.argv[2], int(sys.argv[3])
    renamed = {
        r["manifest_sdr_url"]: r["actual_sdr_url"]
        for r in csv.DictReader(open(f"{root}/RENDER-NAME-MAP.tsv"), delimiter="\t")
        if r["actual_sdr_url"]
    }
    rows = csv.DictReader(open(f"{root}/manifests/{split}.tsv"), delimiter="\t")
    by_class = defaultdict(list)
    for r in rows:
        url = renamed.get(r["png_v3_sdr_url"], r["png_v3_sdr_url"])
        if url.startswith(PREFIX):
            by_class[r["content_class"]].append((int(r["id"]), "png-v3/" + url[len(PREFIX):]))
    for cls in sorted(by_class):
        items = sorted(by_class[cls])
        n = min(k, len(items))
        for i in range(n):
            print(items[(2 * i + 1) * len(items) // (2 * n)][1])


if __name__ == "__main__":
    main()
