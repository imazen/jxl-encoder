#!/usr/bin/env python3
"""Flatten complete issue-138 ablation records and verify retained artifacts."""
import argparse
import csv
import hashlib
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ablations", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.ablations.open(), delimiter="\t"))
    if not rows:
        parser.error("empty ablation table")
    keys = [(r["seed"], r["index"], r["arm"]) for r in rows]
    if len(set(keys)) != len(keys):
        parser.error("duplicate ablation cell")
    arms = {r["arm"] for r in rows}
    cells = {(r["seed"], r["index"]) for r in rows}
    if len(rows) != len(arms) * len(cells):
        parser.error("incomplete cell/arm Cartesian product")
    merged = []
    inputs = {}
    for row in rows:
        study = Path(row["study"])
        records = list(csv.DictReader(
            (line for line in study.open() if not line.startswith("#")), delimiter="\t"))
        if len(records) != 1:
            parser.error(f"incomplete study: {study}")
        record = records[0]
        if record["index"] != row["index"] or row["exit"] not in ("0", "1"):
            parser.error(f"invalid result: {row}")
        stem = study.parent / f"cell{row['index']}"
        encoded = stem.with_suffix(".jxl").read_bytes()
        source = stem.with_suffix(".input").read_bytes()
        if len(encoded) != int(record["ours_bytes"]):
            parser.error(f"encoded size mismatch: {stem}")
        source_hash = hashlib.sha256(source).hexdigest()
        cell = (row["seed"], row["index"])
        if inputs.setdefault(cell, source_hash) != source_hash:
            parser.error(f"source differs across arms: {cell}")
        merged.append({"seed": row["seed"], "index": row["index"],
                       "arm": row["arm"], "exit": row["exit"],
                       **{k: v for k, v in record.items() if k != "index"},
                       "encoded_sha256": hashlib.sha256(encoded).hexdigest(),
                       "input_sha256": source_hash})
    with args.out.open("x", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(merged[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(merged)
    print(f"verified {len(cells)} cells x {len(arms)} arms = {len(merged)} records")


if __name__ == "__main__":
    main()
