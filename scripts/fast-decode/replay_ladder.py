#!/usr/bin/env python3

"""Verify the frozen lossy ladder and time matched-quality tradeoffs on i270."""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
from types import SimpleNamespace

from measure_lossy import (
    PINS, QUALITY_LADDER, Study, digest, quality_passes, quality_to_distance, score,
)


FILTER_ORDER = ("tier1-epf0-gaboff", "tier1-epf0", "tier1-epf1", "tier1-epf2", "tier1")
PROFILES = {
    "balanced": (3, FILTER_ORDER),
    "throughput": (10, FILTER_ORDER),
    "tier4": (10, ("tier4",)),
}


def load_inputs(study, directories):
    inputs = {}
    keys = set()
    groups = defaultdict(list)
    provenance = {}
    for directory in directories:
        manifest_file = directory / "manifest.json"
        quality_file = directory / "quality.tsv"
        manifest = json.loads(manifest_file.read_text())
        if (not (directory / "complete").is_file() or manifest["effort"] != 7
                or tuple(manifest["quality_ladder"]) != QUALITY_LADDER):
            raise ValueError(f"incomplete or unexpected ladder: {directory}")
        for name in ("encoder", "jxl_rs", "scorer", "djxl"):
            if manifest["binaries"][name]["sha256"] != study.manifest["binaries"][name]["sha256"]:
                raise ValueError(f"binary mismatch: {directory} {name}")
        provenance[str(directory)] = {
            "manifest_sha256": digest(manifest_file), "quality_sha256": digest(quality_file),
        }
        for name, row in manifest["inputs"].items():
            raw = directory / f"{name}.rgb"
            if digest(raw) != row["rgb_sha256"]:
                raise ValueError(f"reference raster hash mismatch: {raw}")
            previous = inputs.setdefault(name, (raw, row))
            for field in ("crop", "source_sha256", "rgb_sha256"):
                if previous[1][field] != row[field]:
                    raise ValueError(f"inconsistent reference raster: {raw} {field}")
        with quality_file.open() as table:
            for row in csv.DictReader(table, delimiter="\t"):
                shape = (row["id"], row["size"])
                name = "-".join(shape)
                for field in ("width", "height", "bytes"):
                    row[field] = int(row[field])
                for field in ("distance", "ssim2", "butteraugli"):
                    row[field] = float(row[field])
                    if not math.isfinite(row[field]):
                        raise ValueError(f"invalid {field}: {row['file']}")
                key = (*shape, row["policy"], row["distance"])
                source = directory / "streams" / row["file"]
                if (key in keys or Path(row["file"]).name != row["file"]
                        or row["bytes"] <= 0 or source.stat().st_size != row["bytes"]
                        or digest(source) != row["stream_sha256"]
                        or inputs[name][1]["crop"][2:] != [row["width"], row["height"]]):
                    raise ValueError(f"unexpected stream: {source}")
                keys.add(key)
                (study.streams / row["file"]).symlink_to(source.resolve())
                groups[shape].append(row)
    if not groups or {f"{image}-{size}" for image, size in groups} != inputs.keys():
        raise ValueError("missing ladder inputs")
    expected_distances = {quality_to_distance(q) for q in QUALITY_LADDER}
    for shape, rows in groups.items():
        distances = defaultdict(set)
        for row in rows:
            distances[row["policy"]].add(row["distance"])
        if "base" not in distances or any(grid != expected_distances for grid in distances.values()):
            raise ValueError(f"incomplete policy ladder: {shape}")
    study.manifest.update(
        frozen_ladders=provenance,
        inputs={name: dict(row, raw=str(raw)) for name, (raw, row) in inputs.items()},
        profiles=PROFILES,
    )
    return groups, inputs


def rescore(study, groups, inputs):
    with (study.out / "quality.tsv").open("x", newline="", buffering=1) as table:
        fields = (*next(iter(groups.values()))[0], "target_ssim2", "target_butteraugli", "target_rgb_sha256")
        writer = csv.DictWriter(table, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for shape, rows in sorted(groups.items()):
            raw = inputs["-".join(shape)][0]
            w, h = rows[0]["width"], rows[0]["height"]
            cmd = study.pinned("2", [str(study.args.scorer), str(w), str(h), str(raw)])
            print(json.dumps(cmd), file=study.commands)
            with subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=study.commands) as scorer:
                try:
                    for row in rows:
                        stream = study.streams / row["file"]
                        oracle = study.read_rgb(stream, w, h)
                        if hashlib.sha256(oracle).hexdigest() != row["rgb_sha256"]:
                            raise ValueError(f"frozen jxl-rs pixels changed: {stream}")
                        rgb = study.read_rgb(stream, w, h, study.args.decoder)
                        row["target_ssim2"], row["target_butteraugli"] = score(scorer, rgb, row["file"])
                        row["target_rgb_sha256"] = hashlib.sha256(rgb).hexdigest()
                        writer.writerow(row)
                finally:
                    scorer.stdin.close()
                    if code := scorer.wait():
                        raise subprocess.CalledProcessError(code, cmd)
            print("quality verified", *shape, len(rows), flush=True)


def select(study, groups):
    selected = {}
    with (study.out / "selected.tsv").open("x", newline="", buffering=1) as out:
        writer = csv.writer(out, delimiter="\t", lineterminator="\n")
        writer.writerow(("id", "size", "q", "distance", "profile", "policy", "candidate_distance",
                         "base_sha256", "candidate_sha256", "timed_as", "base_bytes", "bytes",
                         "ssim2_delta", "butteraugli_delta", "target_ssim2_delta", "target_butteraugli_delta"))
        for shape, rows in sorted(groups.items()):
            baseline = {r["distance"]: r for r in rows if r["policy"] == "base"}
            if set(baseline) != {quality_to_distance(q) for q in QUALITY_LADDER}:
                raise ValueError(f"incomplete baseline ladder: {shape}")
            for q in QUALITY_LADDER:
                distance = quality_to_distance(q)
                base = baseline[distance]
                arms = {"base": base}
                by_hash = {base["stream_sha256"]: "base"}
                for name, (growth, policies) in PROFILES.items():
                    eligible = [r for r in rows if r["policy"] in policies
                                and 100 * r["bytes"] <= (100 + growth) * base["bytes"]
                                and quality_passes(base, r) and quality_passes(base, r, "target_")]
                    row = min(eligible, default=base, key=lambda r: (
                        policies.index(r["policy"]), r["bytes"], abs(r["distance"] - distance), r["distance"]))
                    arm = by_hash.setdefault(row["stream_sha256"], name)
                    arms.setdefault(arm, row)
                    writer.writerow((*shape, q, distance, name, row["policy"], row["distance"],
                                     base["stream_sha256"], row["stream_sha256"], arm, base["bytes"], row["bytes"],
                                     *(row[key] - base[key] for key in
                                       ("ssim2", "butteraugli", "target_ssim2", "target_butteraugli"))))
                selected[*shape, distance] = list(arms.items())
    return selected


def check_workers(study, selected):
    by_hash = {row["stream_sha256"]: row for arms in selected.values() for _, row in arms}
    with (study.out / "readbacks.tsv").open("x", newline="", buffering=1) as out:
        writer = csv.writer(out, delimiter="\t", lineterminator="\n")
        writer.writerow(("file", "workers", "baseline_rgb_sha256", "current_rgb_sha256"))
        for row in by_hash.values():
            stream = study.streams / row["file"]
            for workers in PINS:
                values = [hashlib.sha256(study.read_rgb(
                    stream, row["width"], row["height"], decoder, workers)).hexdigest()
                    for decoder in (study.args.baseline_decoder, study.args.decoder)]
                if values != [row["target_rgb_sha256"]] * 2:
                    raise ValueError(f"decoder/worker pixels changed: {stream} at {workers}")
                writer.writerow((row["file"], workers, *values))
    print("same-stream readbacks complete", len(by_hash) * len(PINS), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=lambda p: Path(p).resolve())
    parser.add_argument("ladders", nargs="+", type=lambda p: Path(p).resolve())
    for name in ("encoder", "decoder", "baseline-decoder", "djxl", "jxl-rs", "scorer"):
        parser.add_argument(f"--{name}", required=True, type=lambda p: Path(p).resolve())
    args = parser.parse_args()
    study = Study(SimpleNamespace(**vars(args), stage="replay-ladder", blocks=3), cohort={})
    groups, inputs = load_inputs(study, args.ladders)
    (study.out / "manifest.json").write_text(json.dumps(study.manifest, indent=2) + "\n")
    rescore(study, groups, inputs)
    selected = select(study, groups)
    check_workers(study, selected)
    finalists = {f"{image}-{size}-{distance}": arms
                 for (image, size, distance), arms in selected.items()}
    (study.out / "finalists.json").write_text(json.dumps(finalists, indent=2) + "\n")
    study.measure_time(selected)
    (study.out / "complete").touch(exist_ok=False)


if __name__ == "__main__":
    main()
