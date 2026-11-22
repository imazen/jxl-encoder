#!/usr/bin/env python3

"""Replay rate-first tier-1 filter selection using both rendered quality endpoints."""

import argparse
from collections import defaultdict
import csv
import math
from pathlib import Path

from measure_lossy import quality_passes

STAGES = {
    "native": ("native", "tier4-native", "tier4-native-refine"),
    "1024": ("match", "refine"),
}
SCORES = {
    "native": "native-verify",
    "match": "match-target",
    "refine": "refine-target",
}


def read_rows(directory, name):
    if not (directory / "complete").is_file():
        raise ValueError(f"incomplete measurement stage: {directory}")
    with (directory / name).open() as table:
        return list(csv.DictReader(table, delimiter="\t"))


def rendered_scores(row, prefix=""):
    scores = {key: float(row[prefix + key]) for key in ("ssim2", "butteraugli")}
    if not all(math.isfinite(value) for value in scores.values()):
        raise ValueError("non-finite rendered score")
    return scores


def priority(row):
    passes = int(row["policy"].removeprefix("tier1-epf").split("-")[0])
    return int(row["bytes"]), passes, "gaboff" not in row["policy"], row["file"]


def selected_rows(root):
    yield ("id", "size", "band", "stage", "file", "bytes", "byte_ratio", "policy", "distance",
           "jxl_rs_ssim2_delta", "jxl_rs_butteraugli_delta", "zen_ssim2_delta",
           "zen_butteraugli_delta", "sha256")
    for size, stages in STAGES.items():
        candidates = defaultdict(list)
        baselines = {}
        for stage in stages:
            scores = (
                {row["file"]: row for row in read_rows(root / SCORES[stage], "scores.tsv")}
                if stage in SCORES else {}
            )
            for row in read_rows(root / stage, "quality.tsv"):
                if row["size"] != size:
                    continue
                band = 1 if float(row["distance"]) < 2 else 4
                cell = row["id"], band
                zen = row if row.get("target_ssim2") else scores.get(row["file"])
                if zen is None:
                    continue
                scores_pair = rendered_scores(row), rendered_scores(zen, "target_")
                rendition = row, scores_pair, stage
                if row["policy"] == "base":
                    previous = baselines.setdefault(cell, rendition)
                    if previous[0]["stream_sha256"] != row["stream_sha256"]:
                        raise ValueError(f"changed baseline for {cell}")
                elif row["policy"].startswith("tier1-epf") and (
                    band != 1 or row["policy"].startswith("tier1-epf0")
                ):
                    candidates[cell].append(rendition)
        if not baselines:
            raise ValueError(f"missing quality baselines for {size}")
        for (image, band), baseline in sorted(baselines.items()):
            base, base_scores, _ = baseline
            eligible = [
                rendition for rendition in candidates[image, band]
                if 10 * int(rendition[0]["bytes"]) <= 11 * int(base["bytes"])
                and all(quality_passes(reference, measured)
                        for reference, measured in zip(base_scores, rendition[1], strict=True))
            ]
            row, scores, stage = min(eligible, key=lambda rendition: priority(rendition[0]), default=baseline)
            errors = [
                measured[key] - reference[key]
                for reference, measured in zip(base_scores, scores, strict=True)
                for key in ("ssim2", "butteraugli")
            ]
            yield (image, size, band, stage, row["file"], row["bytes"],
                   f"{int(row['bytes']) / int(base['bytes']):.8f}", row["policy"], row["distance"],
                   *[f"{value:.8f}" for value in errors], row["stream_sha256"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurements", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = list(selected_rows(args.measurements))
    with args.output.open("x", newline="") as report:
        csv.writer(report, delimiter="\t", lineterminator="\n").writerows(rows)


if __name__ == "__main__":
    main()
