#!/usr/bin/env python3

"""Summarize paired decode timings and the measured JPEG XL streams."""

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
from statistics import median

from measure_lossy import PINS, quality_passes


def p95(values):
    return sorted(values)[math.ceil(0.95 * len(values)) - 1]


def summary(measured):
    finalists = json.loads((measured / "finalists.json").read_text())
    manifest = json.loads((measured / "manifest.json").read_text())
    blocks = manifest.get("timing_blocks", 3)
    paired = defaultdict(dict)
    with (measured / "timings.tsv").open() as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            key = tuple(row[k] for k in ("id", "size", "band", "workers", "block", "sample"))
            if row["arm"] in paired[key]:
                raise ValueError(f"duplicate sample: {key} {row['arm']}")
            paired[key][row["arm"]] = row
    lossless = manifest["stage"] == "palette-grid"
    fields = ["id", "size", "band", "arm", "bytes", "byte_ratio", "sha256"]
    if not lossless:
        fields = ["id", "size", "band", "arm", "distance", "bytes", "byte_ratio",
                  "ssim2_delta", "butteraugli_delta", "quality_pass", "sha256"]
    timing_fields = ["throughput", "paired_throughput_ratio", "p95_latency_ratio", "p95_us"]
    fields.extend(f"{name}_{workers}w" for workers in PINS for name in timing_fields)
    yield fields
    for key, candidates in finalists.items():
        source, size, band = key.split("-")
        base = candidates[0][1]
        expected = {name: row for name, row in candidates}
        expected.update(aa=base, old=base)
        by_arm = defaultdict(list)
        for workers in PINS:
            samples = [paired.pop((source, size, band, str(workers), str(block), str(sample)))
                       for block in range(1, blocks + 1) for sample in range(1, 6)]
            if any(set(sample) != set(expected) for sample in samples):
                raise ValueError(f"incomplete timing for {key} at {workers} workers")
            if any(r["file"] != expected[arm]["file"] for s in samples for arm, r in s.items()):
                raise ValueError(f"wrong measured stream for {key} at {workers} workers")

            def rate(sample, arm):
                return float(sample[arm]["mp_s"])

            pixels = base["width"] * base["height"]
            base_p95 = p95([pixels / rate(s, "base") for s in samples])
            for arm in expected:
                rates = [rate(s, arm) for s in samples]
                tail = p95([pixels / r for r in rates])
                items = (median(rates), median(r / rate(s, "base") for r, s in zip(rates, samples)),
                         tail / base_p95, tail)
                by_arm[arm].extend(f"{value:.6f}" for value in items)
        for arm, row in expected.items():
            values = [source, size, band, arm]
            if not lossless:
                values.append(row["distance"])
            values.extend((row["bytes"], f"{row['bytes'] / base['bytes']:.6f}"))
            if not lossless:
                quality = ("", "", "") if "fallback" in base else (
                    f"{row['ssim2'] - base['ssim2']:.6f}",
                    f"{row['butteraugli'] - base['butteraugli']:.6f}",
                    int(quality_passes(base, row) and (
                        "target_ssim2" not in base or quality_passes(base, row, "target_")
                    )),
                )
                values.extend(quality)
            yield [*values, row["stream_sha256"], *by_arm[arm]]
    if paired:
        raise ValueError("unmatched timing rows")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measured", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if not (args.measured / "complete").is_file():
        raise ValueError("timing run is incomplete")
    rows = list(summary(args.measured))
    with args.output.open("x", newline="") as report:
        csv.writer(report, delimiter="\t", lineterminator="\n").writerows(rows)


if __name__ == "__main__":
    main()
