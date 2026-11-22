#!/usr/bin/env python3

"""Measure VarDCT filter candidates on pinned imazen-26 SDR inputs."""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess

from PIL import Image

BANDS = (1, 4)
PINS = {1: "2", 4: "2,3,0,1", 8: "0-7"}
CLI_RATE = re.compile(r"Decoded (\d+) pixels in [\d.]+ seconds: ([\d.]+) MP/s")
POLICIES = {
    "base": (),
    "tier1": ("--faster-decoding", "1"),
    "tier1-epf0": ("--faster-decoding", "1", "--epf", "0"),
    "tier1-epf1": ("--faster-decoding", "1", "--epf", "1"),
    "tier1-epf2": ("--faster-decoding", "1", "--epf", "2"),
    "tier1-gaboff": ("--faster-decoding", "1", "--no-gaborish"),
    "tier1-epf0-gaboff": ("--faster-decoding", "1", "--epf", "0", "--no-gaborish"),
    "tier1-epf1-gaboff": ("--faster-decoding", "1", "--epf", "1", "--no-gaborish"),
    "tier0-epf0": ("--epf", "0"),
    "tier0-epf1": ("--epf", "1"),
    "tier4": ("--faster-decoding", "4"),
}
FAST_POLICIES = ("tier1-epf0", "tier1-epf1", "tier1-epf2")
GABORISH_POLICIES = ("tier1-gaboff", "tier1-epf0-gaboff", "tier1-epf1-gaboff")
QUALITY_LADDER = tuple(range(5, 100, 5)) + (99,)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def quality_to_distance(q):
    if q >= 90:
        return (100 - q) / 10
    if q >= 70:
        return 1 + (90 - q) / 20
    return 2 + (70 - q) / 10


def score(scorer, rgb, name):
    scorer.stdin.write(rgb)
    scorer.stdin.flush()
    values = scorer.stdout.readline().decode().strip().split("\t")
    if len(values) != 2 or not all(math.isfinite(float(x)) for x in values):
        raise ValueError(f"missing quality scores: {name}")
    return tuple(map(float, values))


class Study:
    def __init__(self, args):
        self.args = args
        self.out = args.output
        self.out.mkdir()
        self.streams = self.out / "streams"
        self.streams.mkdir()
        self.commands = (self.out / "commands.log").open("x", buffering=1)
        self.key = {}
        with args.cohort.open() as stream:
            self.cohort = {row["id"]: row for row in csv.DictReader(stream, delimiter="\t")}
        names = ("encoder", "decoder", "baseline_decoder", "djxl", "jxl_rs", "scorer")
        self.manifest = {
            "stage": args.stage,
            "inputs": {},
            "binaries": {k: {"path": str(getattr(args, k)), "sha256": digest(getattr(args, k))}
                         for k in names},
            "policies": POLICIES,
            "pins": PINS,
            "effort": 7,
            "quality_ladder": QUALITY_LADDER,
            "timing_blocks": args.blocks,
        }
        version = self.run([args.djxl, "--version"])
        if "v0.12.0 a7a9c78" not in version:
            raise ValueError(f"wrong libjxl reference: {version}")
        self.manifest["djxl_version"] = version.strip()

    def run(self, command):
        cmd = list(map(str, command))
        print(json.dumps(cmd), file=self.commands)
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout, result.stderr, file=self.commands)
        result.check_returncode()
        return result.stdout + result.stderr

    def read_rgb(self, stream, w, h, decoder=None, workers=1):
        path = self.out / "readback.ppm"
        if decoder is None:
            decoder = self.args.jxl_rs
            flags = ("--num-threads", "1")
        else:
            flags = ("--no-cms", "--num-threads", str(workers if workers > 1 else 0))
        read_fd, write_fd = os.pipe()
        with os.fdopen(read_fd, "rb") as reader, os.fdopen(write_fd, "wb") as writer:
            path.symlink_to(f"/dev/fd/{write_fd}")
            try:
                cmd = ["taskset", "-c", PINS[workers], str(decoder), str(stream), str(path),
                       *flags, "--data-type", "u8"]
                print(json.dumps(cmd), file=self.commands)
                with subprocess.Popen(cmd, pass_fds=(write_fd,), stdout=self.commands,
                                      stderr=self.commands) as process:
                    writer.close()
                    ppm = reader.read()
                    if code := process.wait():
                        raise subprocess.CalledProcessError(code, cmd)
            finally:
                path.unlink()
        header = re.match(rb"P6\s+(\d+)\s+(\d+)\s+255\r?\n", ppm)
        if not header or tuple(map(int, header.groups())) != (w, h):
            raise ValueError(f"invalid RGB8 readback for {stream}")
        rgb = ppm[header.end():]
        if len(rgb) != 3 * w * h:
            raise ValueError(f"wrong sample count for {stream}")
        return rgb

    def source(self, image_id, size):
        row = self.cohort[image_id]
        path, = self.args.corpus.rglob(row["url"].rsplit("/", 1)[1])
        if digest(path) != row["sha256"]:
            raise ValueError(f"source hash mismatch: {path}")
        with Image.open(path) as image:
            if image.mode != "RGB":
                raise ValueError(f"RGB8 input required: {path}")
            if size == "native":
                w, h = image.size
                x = y = 0
            else:
                w = h = 256 if size == "text256" else int(size)
                if min(image.size) < w:
                    raise ValueError(f"requested crop exceeds source: {path}")
                if size == "text256":
                    x, y = 0, 64
                else:
                    x, y = (image.width - w) // 2, (image.height - h) // 2
                image = image.crop((x, y, x + w, y + h))
                path = self.out / f"{image_id}-{size}.png"
                image.save(path)
            raw = self.out / f"{image_id}-{size}.rgb"
            raw.write_bytes(image.tobytes())
        self.manifest["inputs"][f"{image_id}-{size}"] = {
            "image_id": image_id, "size": size, "path": str(path), "crop": [x, y, w, h],
            "source_sha256": row["sha256"], "input_sha256": digest(path), "rgb_sha256": digest(raw),
        }
        return path, raw, w, h

    def encode(self, source, stream, distance, policy):
        self.run(["taskset", "-c", "0-7", self.args.encoder, source, stream,
                  "--threads", 8, "-e", 7, "-d", distance, *POLICIES[policy]])

    def measure_quality(self, cases):
        fields = ("id", "size", "width", "height", "policy", "distance", "bytes",
                  "ssim2", "butteraugli", "stream_sha256", "rgb_sha256", "file")
        with (self.out / "quality.tsv").open("x", newline="", buffering=1) as evidence:
            writer = csv.DictWriter(evidence, fieldnames=fields, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            for image_id, size, arms in cases:
                source, raw, w, h = self.source(image_id, size)
                score_cmd = ["taskset", "-c", "2", str(self.args.scorer), str(w), str(h), str(raw)]
                print(json.dumps(score_cmd), file=self.commands)
                with subprocess.Popen(score_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      stderr=self.commands) as scorer:
                    try:
                        for policy, d in dict.fromkeys(arms):
                            name = f"{image_id}-{size}-{policy}-d{d:g}.jxl"
                            stream = self.streams / name
                            self.encode(source, stream, f"{d:g}", policy)
                            rgb = self.read_rgb(stream, w, h)
                            scores = score(scorer, rgb, name)
                            row = dict(zip(fields, (image_id, size, w, h, policy, d, stream.stat().st_size,
                                                    *scores, digest(stream),
                                                    hashlib.sha256(rgb).hexdigest(), name)))
                            writer.writerow(row)
                            self.key[image_id, size, policy, d] = row
                    finally:
                        scorer.stdin.close()
                        if code := scorer.wait():
                            raise subprocess.CalledProcessError(code, score_cmd)
                print("quality complete", image_id, size, len(arms), flush=True)
                (self.out / "manifest.json").write_text(json.dumps(self.manifest, indent=2) + "\n")

    def prepare_timing(self, bands, policies):
        selected = {}
        for image_id, size in dict.fromkeys((image, size) for image, size, _, _ in self.key):
            for band in bands:
                base = self.key[image_id, size, "base", band]
                candidates = [("base", base), ("tier1", self.key[image_id, size, "tier1", band])]
                for policy in policies:
                    arms = [r for (i, s, p, d), r in self.key.items()
                            if (i, s, p) == (image_id, size, policy)
                            and abs(d - band) <= (0.25 if band == 1 else 0.81)]
                    feasible = [r for r in arms if quality_passes(base, r) and r["bytes"] <= 1.1 * base["bytes"]]
                    if feasible:
                        candidates.append((policy, min(feasible, key=lambda r: (r["bytes"], abs(r["distance"] - band)))))
                selected[image_id, size, band] = candidates
        return selected

    def measure_time(self, selected):
        with (self.out / "timings.tsv").open("x", newline="", buffering=1) as output:
            writer = csv.writer(output, delimiter="\t", lineterminator="\n")
            writer.writerow(("id", "size", "band", "mode", "workers", "block", "sample",
                             "arm", "file", "decoder", "pixels", "mp_s"))
            for (image_id, size, band), arms in selected.items():
                base = arms[0][1]
                all_arms = [(name, row, "current") for name, row in arms]
                all_arms.extend([("aa", base, "current"), ("old", base, "baseline")])
                for _, row in arms:
                    self.run(["taskset", "-c", "2", self.args.djxl, self.streams / row["file"],
                              "--disable_output", "--num_threads=0"])
                for workers, cpus in PINS.items():
                    for block in range(self.args.blocks):
                        position = (block % 3) * len(all_arms) // 3
                        order = all_arms[position:] + all_arms[:position]
                        if block % 3 == 1:
                            order = order[::-1]
                        for sample in range(5):
                            for name, row, kind in order:
                                decoder = self.args.decoder if kind == "current" else self.args.baseline_decoder
                                measured = self.run(["taskset", "-c", cpus, decoder, self.streams / row["file"],
                                                     "--speedtest", "--num-reps", "1", "--warmup-reps", "1",
                                                     "--no-cms", "--data-type", "u8",
                                                     "--num-threads", 0 if workers == 1 else workers])
                                parsed = CLI_RATE.search(measured)
                                pixels = row["width"] * row["height"]
                                if not parsed or int(parsed[1]) != pixels or float(parsed[2]) <= 0:
                                    raise ValueError(f"wrong timing output: {row['file']}")
                                writer.writerow((image_id, size, band, "warm", workers, block + 1, sample + 1,
                                                 name, row["file"], kind, pixels, parsed[2]))
                print("timing complete", image_id, size, band, flush=True)

    def confirm_photos(self, measured):
        self.streams = measured / "streams"
        candidates = json.loads((measured / "finalists.json").read_text())
        selected = {}
        for key, rows in candidates.items():
            image_id, size, band = key.split("-")
            if image_id not in ("1007", "1019"):
                continue
            wanted = ("base", "tier1-epf0", "tier1-epf0-gaboff")
            arms = [(name, row) for name, row in rows if name in wanted]
            if any(digest(self.streams / row["file"]) != row["stream_sha256"] for _, row in arms):
                raise ValueError(f"changed input stream: {key}")
            selected[image_id, size, int(band)] = arms
        if not selected:
            raise ValueError(f"no photo timing cells in {measured}")
        self.manifest["measured"] = str(measured)
        (self.out / "manifest.json").write_text(json.dumps(self.manifest, indent=2) + "\n")
        (self.out / "finalists.json").write_text(json.dumps(
            {f"{i}-{s}-{d}": rows for (i, s, d), rows in selected.items()}, indent=2) + "\n")
        self.measure_time(selected)
        return len(selected)

    def verify_readbacks(self, measured):
        finalists = json.loads((measured / "finalists.json").read_text())
        unique = {}
        for arms in finalists.values():
            for _, row in arms:
                unique[row["file"]] = row
        fields = ("file", "workers", "baseline_rgb_sha256", "candidate_rgb_sha256")
        with (
            (self.out / "readbacks.tsv").open("x", newline="", buffering=1) as output,
            (self.out / "scores.tsv").open("x", newline="", buffering=1) as scores,
        ):
            writer = csv.writer(output, delimiter="\t", lineterminator="\n")
            writer.writerow(fields)
            quality = csv.writer(scores, delimiter="\t", lineterminator="\n")
            quality.writerow(("file", "target_ssim2", "target_butteraugli"))
            for name, row in unique.items():
                stream = measured / "streams" / name
                if digest(stream) != row["stream_sha256"]:
                    raise ValueError(f"changed stream: {name}")
                expected = None
                for workers in PINS:
                    rgb = [self.read_rgb(
                        stream, row["width"], row["height"], decoder, workers
                    ) for decoder in (self.args.baseline_decoder, self.args.decoder)]
                    hashes = [hashlib.sha256(pixels).hexdigest() for pixels in rgb]
                    expected = expected or hashes[0]
                    if hashes != [expected, expected]:
                        raise ValueError(f"different decoder or worker-count pixels: {name}")
                    writer.writerow((name, workers, *hashes))
                    if workers == 1:
                        reference = measured / f"{row['id']}-{row['size']}.rgb"
                        cmd = ["taskset", "-c", "2", str(self.args.scorer), str(row["width"]),
                               str(row["height"]), str(reference)]
                        print(json.dumps(cmd), file=self.commands)
                        with subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                              stderr=self.commands) as scorer:
                            values = score(scorer, rgb[1], name)
                            scorer.stdin.close()
                            if code := scorer.wait():
                                raise subprocess.CalledProcessError(code, cmd)
                        quality.writerow((name, *values))
        return len(unique) * len(PINS)


def quality_passes(base, arm):
    return (arm["ssim2"] - base["ssim2"] >= -0.25 and
            arm["butteraugli"] - base["butteraugli"] <= max(0.01, 0.02 * base["butteraugli"]))


def match_cases():
    controls = [(str(i), "native" if i != 6824 else "1024") for i in (8457, 7026, 8106, 6824)]
    anchors = [(p, d) for d in BANDS for p in POLICIES if p not in GABORISH_POLICIES]
    refine = ([(f"tier1-epf{e}", d / 100) for e in (0, 1) for d in (85, 90, 95, 97, 99, 102, 105)] +
              [(p, d / 100) for p in FAST_POLICIES for d in range(320, 421, 10)])
    for image_id in ("1007", "1019"):
        yield image_id, "1024", anchors + refine
    for image_id, size in controls:
        yield image_id, size, anchors


def refine_cases():
    anchors = [("base", 1), ("tier1", 1)]
    grid = (0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1, 1.01, 1.02)
    for image_id in ("1007", "1019"):
        arms = [(p, d) for p in ("tier1-epf0", *GABORISH_POLICIES) for d in grid]
        yield image_id, "1024", anchors + arms + [("tier1-epf1", 1), ("tier1-epf2", 1)]


def native_cases():
    policies = ("base", "tier1", *FAST_POLICIES, "tier1-epf0-gaboff")
    anchors = [(p, d) for d in BANDS for p in policies]
    refinements = (
        [(p, d) for p in ("tier1-epf0", "tier1-epf0-gaboff") for d in (0.90, 0.95, 0.98, 1.02)]
        + [("tier1-epf0", d) for d in (3.6, 3.75, 3.9, 4.08)]
        + [(p, d) for p in ("tier1-epf1", "tier1-epf2") for d in (3.9, 4.08)]
    )
    for image_id in ("1007", "1019"):
        yield image_id, "native", anchors + refinements


def tier4_cases():
    anchors = [(p, d) for d in BANDS for p in ("base", "tier1")]
    distances = [d / 100 for d in (*range(55, 106, 5), *range(350, 421, 10))]
    for image_id in ("1007", "1019"):
        yield image_id, "1024", anchors + [("tier4", d) for d in distances]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "cohort", "corpus", "encoder", "decoder", "baseline_decoder", "djxl", "jxl_rs", "scorer"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--stage", choices=("match", "refine", "native", "ladder",
                                            "ladder-gaboff", "ladder-text", "ladder-tier4",
                                            "tier4-match", "verify", "confirm"),
                        default="match")
    parser.add_argument("--measured", type=Path)
    parser.add_argument("--blocks", type=int, default=3)
    args = parser.parse_args()
    if args.stage in ("verify", "confirm") and args.measured is None:
        parser.error(f"--stage {args.stage} requires --measured")
    if args.blocks < 3:
        parser.error("--blocks requires at least three interleaved blocks")
    study = Study(args)
    if args.stage in ("match", "refine", "native", "tier4-match"):
        if args.stage == "match":
            study.measure_quality(match_cases())
            selected = study.prepare_timing(BANDS, FAST_POLICIES)
        elif args.stage == "refine":
            study.measure_quality(refine_cases())
            selected = study.prepare_timing((1,), FAST_POLICIES + GABORISH_POLICIES)
        elif args.stage == "tier4-match":
            study.measure_quality(tier4_cases())
            selected = study.prepare_timing(BANDS, ("tier4",))
        else:
            study.measure_quality(native_cases())
            selected = study.prepare_timing(BANDS, FAST_POLICIES + GABORISH_POLICIES)
        (study.out / "finalists.json").write_text(json.dumps(
            {f"{i}-{s}-{d}": rows for (i, s, d), rows in selected.items()}, indent=2) + "\n")
        study.measure_time(selected)
    elif args.stage in ("ladder", "ladder-gaboff", "ladder-text", "ladder-tier4"):
        policies = ("base", "tier1", *FAST_POLICIES)
        sizes = [("1007", "1024"), ("1019", "1024"), ("8457", "256"), ("7026", "64")]
        if args.stage == "ladder-gaboff":
            policies = ("tier1-epf0-gaboff",)
        elif args.stage == "ladder-text":
            policies += ("tier1-epf0-gaboff",)
            sizes = [("8457", "text256")]
        elif args.stage == "ladder-tier4":
            policies = ("tier4",)
            sizes += [("8457", "text256")]
        ladder = [(policy, quality_to_distance(q)) for q in QUALITY_LADDER for policy in policies]
        study.measure_quality((i, s, ladder) for i, s in sizes)
    elif args.stage == "confirm":
        count = study.confirm_photos(args.measured)
        (study.out / "complete").write_text(f"{count} photo timing cells\n")
        return
    else:
        count = study.verify_readbacks(args.measured)
        (study.out / "complete").write_text(f"{count} exact decoder and worker-count comparisons\n")
        return
    (study.out / "complete").write_text(f"{len(study.key)} scored streams\n")


if __name__ == "__main__":
    main()
