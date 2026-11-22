#!/usr/bin/env python3

"""Compare measured encodes against each tier-0 rung at matched quality."""

import argparse
from collections import defaultdict
import csv
from pathlib import Path

from measure_lossy import quality_passes


def load_ladders(directories):
    groups = defaultdict(lambda: defaultdict(list))
    seen = set()
    for directory in directories:
        if not (directory / "complete").is_file():
            raise ValueError(f"incomplete ladder: {directory}")
        with (directory / "quality.tsv").open() as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                shape = row["id"], row["size"]
                policy = row["policy"]
                distance = float(row["distance"])
                key = (*shape, policy, distance)
                if key in seen:
                    raise ValueError(f"duplicate rendition: {key}")
                seen.add(key)
                groups[shape][policy].append({
                    **row, "distance": distance, "bytes": int(row["bytes"]),
                    "ssim2": float(row["ssim2"]), "butteraugli": float(row["butteraugli"]),
                })
    if not groups:
        raise ValueError("empty ladders")
    return groups


def matched_rows(groups):
    yield ("id", "size", "baseline_distance", "policy", "same_distance_quality_pass",
           "matched_distance", "byte_ratio", "ssim2_delta", "butteraugli_delta",
           "stream_sha256")
    for (image, size), policies in sorted(groups.items()):
        baseline = policies["base"]
        distances = {base["distance"] for base in baseline}
        for policy, rungs in sorted(policies.items()):
            if policy == "base":
                continue
            by_distance = {rung["distance"]: rung for rung in rungs}
            if set(by_distance) != distances:
                raise ValueError(f"unequal distance ladders: {image} {size} {policy}")
            for base in sorted(baseline, key=lambda row: row["distance"]):
                distance = base["distance"]
                same = int(quality_passes(base, by_distance[distance]))
                fields = (image, size, distance, policy, same)
                feasible = [rung for rung in rungs if quality_passes(base, rung)]
                if not feasible:
                    yield (*fields, "", "", "", "", "")
                    continue
                best = min(feasible, key=lambda row: (row["bytes"], abs(row["distance"] - distance)))
                yield (*fields, best["distance"], f"{best['bytes'] / base['bytes']:.8f}",
                       f"{best['ssim2'] - base['ssim2']:.8f}",
                       f"{best['butteraugli'] - base['butteraugli']:.8f}", best["stream_sha256"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("ladders", nargs="+", type=Path)
    args = parser.parse_args()
    rows = list(matched_rows(load_ladders(args.ladders)))
    with args.output.open("x", newline="") as report:
        csv.writer(report, delimiter="\t", lineterminator="\n").writerows(rows)


if __name__ == "__main__":
    main()
