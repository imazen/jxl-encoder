#!/usr/bin/env python3

"""Summarize matched-quality profile timing, per-rung and tail results."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from statistics import median
from measure_lossy import PINS
from summarize_lossy import p95



def read_tsv(path):
    with path.open() as stream:
        yield from csv.DictReader(stream, delimiter="\t")


def write_tsv(path, fields, records):
    with path.open("x", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(fields)
        writer.writerows([f"{x:.6f}" if isinstance(x, float) else x for x in row] for row in records)


def endpoint_rows(run, endpoints, warm):
    manifest = json.loads((endpoints / "reuse-manifest.json").read_text())
    selected = {(r["id"], r["q"], r["profile"]): r for r in read_tsv(run / "selected.tsv")}
    cold = defaultdict(dict)
    for r in read_tsv(endpoints / "cold" / "timings.tsv"):
        key = (r["id"], float(r["band"]), int(r["workers"]), int(r["block"]), int(r["sample"]))
        if r["arm"] in cold[key] or r["mode"] != "cold":
            raise ValueError(f"invalid cold sample: {key}")
        cold[key][r["arm"]] = float(r["mp_s"])
    fields = ("endpoint", "id", "q", "profile", "workers", "speedup", "aa_ratio",
              "p95_ratio", "aa_p95_ratio")
    yield fields
    expected_samples = {(b, s) for b in range(1, 4) for s in range(1, 6)}
    for profile, pairs in manifest["pairs"].items():
        for workers in PINS:
            reused = defaultdict(dict)
            path = endpoints / f"reused-{profile}-{workers}.tsv"
            for r in read_tsv(path):
                key = (Path(r["input"]).name, int(r["block"]), int(r["sample"]))
                if r["arm"] in reused[key]:
                    raise ValueError(f"duplicate reuse sample: {key}")
                reused[key][r["arm"]] = 1 / int(r["ns"])
            if set(reused) != {(name, b, s) for _, _, name, _ in pairs for b, s in expected_samples}:
                raise ValueError(f"incomplete reuse samples: {path}")
            for image, q, base_file, _ in pairs:
                choice = selected[image, q, profile]
                d = float(choice["distance"])
                for endpoint, rows, base_arm, candidate_arm in (
                    ("cold", [cold[image, d, workers, b, s] for b, s in sorted(expected_samples)],
                     "base", choice["timed_as"]),
                    ("warm_allocating",
                     [warm[image, choice["size"], d, workers, b, s] for b, s in sorted(expected_samples)],
                     "base", choice["timed_as"]),
                    ("warm_reused", [reused[base_file, b, s] for b, s in sorted(expected_samples)],
                     "baseline", "candidate"),
                ):
                    base = [r[base_arm] for r in rows]
                    candidate = [r[candidate_arm] for r in rows]
                    aa = [r["aa"] for r in rows]
                    baseline_p95 = p95([1 / rate for rate in base])
                    yield (endpoint, image, q, profile, workers,
                           median(c / b for c, b in zip(candidate, base, strict=True)),
                           median(a / b for a, b in zip(aa, base, strict=True)),
                           p95([1 / rate for rate in candidate]) / baseline_p95,
                           p95([1 / rate for rate in aa]) / baseline_p95)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--endpoints", type=Path)
    args = parser.parse_args()
    run, out = args.run, args.output
    if not (run / "complete").is_file():
        raise ValueError(f"incomplete warm replay: {run}")
    out.mkdir()

    raw = defaultdict(dict)
    for r in read_tsv(run / "timings.tsv"):
        key = (r["id"], r["size"], float(r["band"]), int(r["workers"]), int(r["block"]), int(r["sample"]))
        cell = raw[key]
        if r["arm"] in cell:
            raise ValueError(f"duplicate sample: {key} {r['arm']}")
        cell[r["arm"]] = float(r["mp_s"])
    views = defaultdict(list)
    rungs = defaultdict(list)
    tails = []
    for r in read_tsv(run / "selected.tsv"):
        q = int(r["q"])
        is_photo = r["id"] in {"1007", "1019"}
        cohort = "photo" if is_photo else f"{r['id']}-{r['size']}"
        band = "aggressive" if q <= 60 else "higher"
        fixture = (r["id"], r["size"], float(r["distance"]))
        ratio = int(r["bytes"]) / int(r["base_bytes"])
        fast = int("epf" in r["policy"] or r["policy"] == "tier4")
        fallback = int(r["timed_as"] == "base")
        for n in PINS:
            samples = [raw[(*fixture, n, b, s)] for b in range(1, 4) for s in range(1, 6)]
            rates = [sample[r["timed_as"]] for sample in samples]
            base = [sample["base"] for sample in samples]
            aa = [sample["aa"] for sample in samples]
            old = [sample["old"] for sample in samples]
            chosen = median(x/y for x, y in zip(rates, base))
            aa_ratio = median(x/y for x, y in zip(aa, base))
            old_ratio = median(x/y for x, y in zip(base, old))
            system_ratio = median(x/y for x, y in zip(rates, old))
            norm = median(base)
            row = (ratio, fast, fallback, chosen, aa_ratio, old_ratio, system_ratio,
                   [norm/x for x in rates], [norm/x for x in base], [norm/x for x in aa])
            for subset in ("all", band):
                views[cohort, subset, r["profile"], n].append(row)
            if is_photo:
                rungs[q, r["profile"], n].append(row)
            by_arm = [p95(x) for x in row[7:]]
            if by_arm[0] / by_arm[1] > 1.10:
                tails.append((*fixture, q, r["profile"], n, by_arm[0] / by_arm[1], by_arm[2] / by_arm[1]))

    tail_fields = ("id", "size", "distance", "q", "profile", "workers", "p95_ratio", "aa_p95_ratio")
    stat_fields = ("cells", "median_rate", "p90_rate", "fast_count", "tier0_count", "profile_speedup",
                   "aa_ratio", "same_stream_decoder_speedup", "whole_system_speedup", "p95_ratio", "aa_p95_ratio")
    def p90(values):
        return sorted(values)[(9 * len(values) + 9) // 10 - 1]
    def stats(group):
        rate, fast, fallback, speed, aa, decoder, system, perf, base, aa_times = map(list, zip(*group))
        baseline_p95 = p95([x for s in base for x in s])
        return [len(group), median(rate), p90(rate), sum(fast), sum(fallback), median(speed), median(aa),
                median(decoder), median(system), p95([x for s in perf for x in s]) / baseline_p95,
                p95([x for s in aa_times for x in s]) / baseline_p95]
    write_tsv(out / "ladder-cohorts.tsv", ("cohort", "q_band", "profile", "workers", *stat_fields),
              ((*key, *stats(values)) for key, values in sorted(views.items())))
    write_tsv(out / "ladder-photo-rungs.tsv", ("q", "profile", "workers", *stat_fields),
              ((*key, *stats(values)) for key, values in sorted(rungs.items())))
    write_tsv(out / "ladder-outliers.tsv", tail_fields, tails)
    if args.endpoints:
        if not (args.endpoints / "complete").is_file():
            raise ValueError(f"incomplete endpoint replay: {args.endpoints}")
        report = endpoint_rows(run, args.endpoints, raw)
        write_tsv(out / "ladder-photo-endpoints.tsv", next(report), report)


if __name__ == "__main__":
    main()
