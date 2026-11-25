#!/usr/bin/env python3

"""Measure JPEG XL decoding profiles on pinned imazen-26 SDR inputs."""

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

REFERENCE_DJXL = {
    ("djxl", "v0.12.0", "a7a9c78"),
    ("djxl", "v0.13.0", "5a7da83"),
}
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
PHOTO_FILTER_POLICIES = {
    "tier0-epf0-gaboff": ("--epf", "0", "--no-gaborish"),
    "tier1-epf0-adaptive": ("--faster-decoding", "1", "--epf", "0", "--adaptive-gaborish"),
    "tier0-cap16-epf0": ("--max-strategy-size", "16", "--epf", "0"),
    "tier0-cap16-epf0-gaboff": ("--max-strategy-size", "16", "--epf", "0", "--no-gaborish"),
}
QUALITY_LADDER = tuple(range(5, 100, 5)) + (99,)
MODULAR_128 = ("--faster-decoding", "1", "-g", "0")
MODULAR_PALETTE = {
    "base": (),
    "tier0-g128": ("-g", "0"),
    "tier1-g128": MODULAR_128,
    "tier1-g256": ("--faster-decoding", "1", "-g", "1"),
    "tier1-g512": ("--faster-decoding", "1", "-g", "2"),
    "palette-off": (*MODULAR_128, "--modular-palette-colors", "0"),
    "palette64": (*MODULAR_128, "--modular-palette-colors", "64"),
    "palette256": (*MODULAR_128, "--modular-palette-colors", "256"),
    "palette4096": (*MODULAR_128, "--modular-palette-colors", "4096"),
    "compact-off": (*MODULAR_128, "--modular-channel-colors-global-percent", "0",
                    "--modular-channel-colors-group-percent", "0"),
}


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
    def __init__(self, args, *, cohort=None, effort=7, threads=8, affinity=True):
        self.args = args
        self.effort = effort
        self.threads = threads
        self.affinity = affinity
        self.out = args.output
        self.out.mkdir()
        self.streams = self.out / "streams"
        self.streams.mkdir()
        self.commands = (self.out / "commands.log").open("x", buffering=1)
        self.key = {}
        if cohort is None:
            with args.cohort.open() as stream:
                cohort = {row["id"]: row for row in csv.DictReader(stream, delimiter="\t")}
        self.cohort = cohort
        names = ("encoder", "decoder", "baseline_decoder", "djxl", "jxl_rs", "scorer")
        self.manifest = {
            "stage": args.stage,
            "inputs": {},
            "binaries": {k: {"path": str(getattr(args, k)), "sha256": digest(getattr(args, k))}
                         for k in names},
            "policies": POLICIES,
            "pins": PINS if affinity else {},
            "effort": effort,
            "encode_threads": threads,
            "quality_ladder": QUALITY_LADDER,
            "timing_blocks": args.blocks,
        }
        version = self.run([args.djxl, "--version"])
        if tuple(version.split()[:3]) not in REFERENCE_DJXL:
            raise ValueError(f"wrong libjxl reference: {version}")
        self.manifest["djxl_version"] = version.strip()

    def pinned(self, cpus, command):
        return ["taskset", "-c", cpus, *command] if self.affinity else command

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
                cmd = self.pinned(PINS[workers], [
                    str(decoder), str(stream), str(path), *flags, "--data-type", "u8",
                ])
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

    def input_file(self, image_id):
        row = self.cohort[image_id]
        if "path" in row:
            path = Path(row["path"])
        else:
            path, = self.args.corpus.rglob(row["url"].rsplit("/", 1)[1])
        if digest(path) != row["sha256"]:
            raise ValueError(f"source hash mismatch: {path}")
        return path, row["sha256"]

    def source(self, image_id, size):
        path, source_sha256 = self.input_file(image_id)
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
            "source_sha256": source_sha256, "input_sha256": digest(path), "rgb_sha256": digest(raw),
        }
        return path, raw, w, h

    def encode(self, source, stream, distance, policy):
        self.run(self.pinned("0-7", [
            self.args.encoder, source, stream, "--threads", self.threads,
            "-e", self.effort, "-d", distance, *POLICIES[policy],
        ]))

    def score_target(self, reference, row, rgb):
        cmd = self.pinned("2", [
            str(self.args.scorer), str(row["width"]), str(row["height"]), str(reference),
        ])
        print(json.dumps(cmd), file=self.commands)
        with subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                              stderr=self.commands) as scorer:
            try:
                return score(scorer, rgb, row["file"])
            finally:
                scorer.stdin.close()
                if code := scorer.wait():
                    raise subprocess.CalledProcessError(code, cmd)

    def measure_quality(self, cases, check_target=False):
        fields = ("id", "size", "width", "height", "policy", "distance", "bytes",
                  "ssim2", "butteraugli", "stream_sha256", "rgb_sha256", "file")
        if check_target:
            fields += ("target_ssim2", "target_butteraugli")
        self.manifest["match_target_decoder"] = check_target
        with (self.out / "quality.tsv").open("x", newline="", buffering=1) as evidence:
            writer = csv.DictWriter(evidence, fieldnames=fields, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            for image_id, size, arms in cases:
                source, raw, w, h = self.source(image_id, size)
                score_cmd = self.pinned("2", [
                    str(self.args.scorer), str(w), str(h), str(raw),
                ])
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
                            if check_target:
                                target = self.read_rgb(stream, w, h, self.args.decoder)
                                row["target_ssim2"], row["target_butteraugli"] = self.score_target(raw, row, target)
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
                    feasible = [r for r in arms if quality_passes(base, r) and r["bytes"] <= 1.1 * base["bytes"]
                                and ("target_ssim2" not in base or quality_passes(base, r, "target_"))]
                    if feasible:
                        candidates.append((policy, min(feasible, key=lambda r: (r["bytes"], abs(r["distance"] - band)))))
                selected[image_id, size, band] = candidates
        return selected

    def measure_time(self, selected, *, warmup=1):
        mode = "warm" if warmup else "cold"
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
                                                     "--speedtest", "--num-reps", "1", "--warmup-reps", warmup,
                                                     "--no-cms", "--data-type", "u8",
                                                     "--num-threads", 0 if workers == 1 else workers])
                                parsed = CLI_RATE.search(measured)
                                pixels = row["width"] * row["height"]
                                if not parsed or int(parsed[1]) != pixels or float(parsed[2]) <= 0:
                                    raise ValueError(f"wrong timing output: {row['file']}")
                                writer.writerow((image_id, size, band, mode, workers, block + 1, sample + 1,
                                                 name, row["file"], kind, pixels, parsed[2]))
                print("timing complete", image_id, size, band, flush=True)

    def exact_readbacks(self, stream, reference, width, height):
        for decoder in (self.args.decoder, self.args.jxl_rs):
            workers_to_check = PINS if decoder == self.args.decoder else (1,)
            for workers in workers_to_check:
                actual = self.read_rgb(stream, width, height,
                                       decoder if decoder == self.args.decoder else None, workers)
                if actual != reference:
                    raise ValueError(f"source RGB8 mismatch: {stream} via {decoder} at {workers} workers")
        cmd = ["taskset", "-c", "2", str(self.args.djxl), str(stream), "-",
               "--output_format", "ppm", "--num_threads=0"]
        print(json.dumps(cmd), file=self.commands)
        result = subprocess.run(cmd, capture_output=True)
        print(result.stderr.decode(errors="replace"), file=self.commands)
        result.check_returncode()
        header = re.match(rb"P6\s+(\d+)\s+(\d+)\s+(255|65535)\r?\n", result.stdout)
        if not header or tuple(map(int, header.groups()[:2])) != (width, height):
            raise ValueError(f"bad djxl readback: {stream}")
        raster = result.stdout[header.end():]
        if header[3] == b"65535":
            exact = raster[::2] == raster[1::2] == reference
        else:
            exact = raster == reference
        if not exact:
            raise ValueError(f"djxl source RGB8 mismatch: {stream}")

    def measure_palette_grid(self):
        self.manifest["policies"] = MODULAR_PALETTE
        fields = ("id", "size", "width", "height", "policy", "bytes", "stream_sha256", "file")
        selected = {}
        with (self.out / "palette.tsv").open("x", newline="", buffering=1) as output:
            writer = csv.DictWriter(output, fieldnames=fields, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            for image_id in ("8457", "7026", "8106"):
                for size in ("64", "256", "native"):
                    rows = {}
                    src, raw, w, h = self.source(image_id, size)
                    reference = raw.read_bytes()
                    for policy, flags in MODULAR_PALETTE.items():
                        name = f"{image_id}-{size}-{policy}.jxl"
                        stream = self.streams / name
                        self.run(["taskset", "-c", "0-7", self.args.encoder, src, stream,
                                  "--threads", 8, "-e", 7, "-d", 0, "--lossless", *flags])
                        self.exact_readbacks(stream, reference, w, h)
                        row = dict(zip(fields, (image_id, size, w, h, policy, stream.stat().st_size,
                                                digest(stream), name)))
                        writer.writerow(row)
                        rows[policy] = row
                    best_palette = min((name for name in rows if name.startswith("palette") or name == "compact-off"),
                                       key=lambda name: rows[name]["bytes"])
                    policies = ("base", "tier0-g128", "tier1-g128", "tier1-g256", "tier1-g512", best_palette)
                    arms = []
                    hashes = set()
                    for policy in policies:
                        row = rows[policy]
                        if row["stream_sha256"] not in hashes:
                            arms.append((policy, row))
                            hashes.add(row["stream_sha256"])
                    selected[image_id, size, 0] = arms
                    print("lossless grid exact", image_id, size, len(rows), flush=True)
                    (self.out / "manifest.json").write_text(json.dumps(self.manifest, indent=2) + "\n")
        return selected

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
                if "fallback" in row:
                    self.read_rgb(stream, row["width"], row["height"])
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
                    if workers == 1 and "fallback" not in row:
                        reference = measured / f"{row['id']}-{row['size']}.rgb"
                        values = self.score_target(reference, row, rgb[1])
                        if "target_ssim2" in row and values != (row["target_ssim2"], row["target_butteraugli"]):
                            raise ValueError(f"target quality scores changed during readback: {name}")
                        quality.writerow((name, *values))
        return len(unique) * len(PINS)


def quality_passes(base, arm, prefix=""):
    ssim2 = prefix + "ssim2"
    butteraugli = prefix + "butteraugli"
    return (arm[ssim2] - base[ssim2] >= -0.25 and
            arm[butteraugli] - base[butteraugli] <= max(0.01, 0.02 * base[butteraugli]))


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


def native_tier4_cases(refine=False):
    if refine:
        images = ("1019",)
        distances = (4.4, 4.42, 4.44, 4.46, 4.48, 4.5, 4.55, 4.6, 4.7, 4.8)
    else:
        images = ("1007", "1019")
        distances = (3.8, 3.9, 4, 4.05, 4.1, 4.15, 4.2, 4.3, 4.4)
    for image_id in images:
        yield image_id, "native", [
            ("base", 4), ("tier1", 4), ("tier1-epf2", 4),
            ("tier1-epf2", 4.08), ("tier1-epf0", 4.08),
            *(("tier4", d) for d in distances),
        ]


def photo_filter_cases(stage):
    if stage == "photo-transform-dc-bracket":
        return [("1619", "native", [
            ("base", 1),
            *(("tier0-epf0-gaboff", d) for d in (0.94, 0.945)),
            *(("tier0-cap16-epf0-gaboff", d) for d in (0.96, 0.98, 1)),
            *(("tier0-cap16-epf0", d) for d in (0.99, 1.01)),
            *(("tier1-epf0", d) for d in (0.98, 1, 1.02)),
            *(("tier1-epf0-gaboff", d) for d in (0.955, 0.98, 1)),
        ])]
    if stage == "photo-filter-saturated-holdout":
        return [("1619", "native", [
            ("base", 1),
            *(("tier0-epf0-gaboff", d) for d in (1, 0.98, 0.95)),
            ("tier0-epf0", 1), ("tier0-epf0", 0.98),
            ("base", 4),
            *(("tier1-epf0", d) for d in (4.10, 4.08, 4)),
            ("tier1-epf2", 4.08), ("tier1-epf2", 4),
        ])]
    if stage == "photo-filter-backlit-bracket":
        return [("1237", "native", [
            ("base", 1), ("tier0-epf0", 1),
            *(("tier0-epf0-gaboff", d) for d in (0.92, 0.94, 0.95, 0.96, 0.97, 0.975)),
        ])]
    if stage == "photo-filter-holdout":
        return [("1237", "native", [
            ("base", 1), ("tier0-epf0", 1),
            ("tier0-epf0-gaboff", 0.98), ("tier0-epf0-gaboff", 1),
            ("base", 4), ("tier1-epf0", 4.10),
            ("tier1-epf2", 4), ("tier1-epf2", 4.08),
        ])]
    return [(
        image_id, "native", [
            ("base", 1), ("tier1-epf0", 1), ("tier1-epf0-gaboff", 1),
            *((p, d) for p in ("tier0-epf0", "tier0-epf0-gaboff") for d in (0.98, 1, 1.02)),
            *(("tier1-epf0-adaptive", d) for d in (1, 1.02, 1.04)),
        ],
    ) for image_id in ("1007", "1019", "1069")]


def photo_filter_matches(
    study, cases, bands, ranking, *, compare_filter_free=False, max_growth=10, fast_dc_growth=None,
):
    selected = {}
    filters = {
        "tier0-epf0-gaboff": 0,
        "tier1-epf0-gaboff": 0,
        "tier0-epf0": 1,
        "tier1-epf0": 1,
        "tier1-epf2": 3,
    }
    for image_id, size, arms in cases:
        for band in bands:
            base = study.key[image_id, size, "base", band]
            candidates = [study.key[image_id, size, policy, distance]
                          for policy, distance in arms
                          if policy != "base" and (distance < 2) == (band == 1)]
            feasible = [row for row in candidates
                        if 100 * row["bytes"] <= (100 + max_growth) * base["bytes"]
                        and quality_passes(base, row) and quality_passes(base, row, "target_")]
            if ranking == "rate-then-filters":
                # The per-file 3% preference precedes filter count.
                choice = min(feasible, key=lambda row: (
                    100 * row["bytes"] > 103 * base["bytes"], filters[row["policy"]],
                    row["bytes"], row["file"],
                ), default=base)
            elif ranking == "filters":
                def rank(row):
                    fast_dc = (
                        fast_dc_growth is not None
                        and row["policy"] in ("tier1-epf0", "tier1-epf0-gaboff")
                        and 100 * row["bytes"] <= (100 + fast_dc_growth) * base["bytes"]
                    )
                    return (filters[row["policy"]], not fast_dc, row["bytes"], row["file"])

                choice = min(feasible, key=rank, default=base)
            else:
                choice = min(feasible, key=lambda row: (row["bytes"], row["file"]), default=base)
            finalists = [("base", base), ("selected", choice)]
            if ranking == "compare-configurations":
                finalists = [("base", base)]
                seen = {base["stream_sha256"]}
                for policy in sorted({row["policy"] for row in feasible}):
                    best = min((row for row in feasible if row["policy"] == policy),
                               key=lambda row: (row["bytes"], row["file"]))
                    if best["stream_sha256"] not in seen:
                        finalists.append((policy, best))
                        seen.add(best["stream_sha256"])
            if compare_filter_free:
                filter_free = (row for row in feasible if row["policy"] == "tier0-epf0-gaboff")
                alternate = min(filter_free, key=lambda row: row["bytes"], default=choice)
                if alternate["stream_sha256"] != choice["stream_sha256"]:
                    finalists.append(("filter-free", alternate))
            selected[image_id, size, band] = finalists
    return selected


def measure_photo_tiers(study):
    # Reuse the exact search and input domain of the user-facing tool.
    from encode_photo import BYTE_GROWTH, candidates, photo_input, tier_choices

    grid = [arm for band in BANDS for arm in candidates(band)]
    cases = []
    fallback = {}
    for image_id in study.cohort:
        path, source_sha256 = study.input_file(image_id)
        (width, height), reason = photo_input(path)
        if reason is None:
            cases.append((image_id, "native", grid))
            continue
        study.manifest["inputs"][f"{image_id}-native"] = {
            "path": str(path), "source_sha256": source_sha256, "crop": [0, 0, width, height],
        }
        for band in BANDS:
            stream = study.streams / f"{image_id}-native-base-d{band:g}.jxl"
            study.encode(path, stream, band, "base")
            fallback[image_id, "native", band] = {
                "id": image_id, "size": "native", "width": width, "height": height,
                "policy": "base", "distance": band, "file": stream.name,
                "bytes": stream.stat().st_size, "stream_sha256": digest(stream),
                "fallback": reason,
            }

    study.measure_quality(cases, check_target=True)
    all_choices = {
        name: tier_choices(study, cases, BANDS, name)
        for name in BYTE_GROWTH
    }
    arms = {}
    for image_id in study.cohort:
        for band in BANDS:
            key = image_id, "native", band
            base = fallback.get(key, study.key.get((image_id, "native", "base", band)))
            arms[key] = [("base", base)]
            for name, choices in all_choices.items():
                choice = choices[key][1][1] if key in choices else base
                arms[key].append((name, choice))
    study.manifest["photo_tier_byte_growth"] = BYTE_GROWTH
    (study.out / "manifest.json").write_text(json.dumps(study.manifest, indent=2) + "\n")
    return arms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "cohort", "corpus", "encoder", "decoder", "baseline_decoder", "djxl", "jxl_rs", "scorer"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--stage", choices=("match", "refine", "native", "ladder",
                                            "ladder-gaboff", "ladder-text", "ladder-tier4",
                                            "tier4-match", "tier4-native", "tier4-native-refine",
                                            "d1-independent-filter", "photo-filter-holdout",
                                            "photo-filter-backlit-bracket", "photo-filter-saturated-holdout",
                                            "photo-transform-dc-bracket", "photo-tiers-native",
                                            "palette-grid", "verify", "confirm"),
                        default="match")
    parser.add_argument("--measured", type=Path)
    parser.add_argument("--blocks", type=int, default=3)
    args = parser.parse_args()
    if args.stage in ("verify", "confirm") and args.measured is None:
        parser.error(f"--stage {args.stage} requires --measured")
    if args.blocks < 3:
        parser.error("--blocks requires at least three interleaved blocks")
    photo_filter_stages = {
        "d1-independent-filter": ((1,), "rate"),
        "photo-filter-holdout": (BANDS, "rate-then-filters"),
        "photo-filter-backlit-bracket": ((1,), "rate-then-filters"),
        "photo-filter-saturated-holdout": (BANDS, "filters"),
        "photo-transform-dc-bracket": ((1,), "compare-configurations"),
    }
    photo_filter_stage = args.stage in photo_filter_stages
    if photo_filter_stage or args.stage == "photo-tiers-native":
        POLICIES.update(PHOTO_FILTER_POLICIES)
    study = Study(args)
    if photo_filter_stage or args.stage in (
            "match", "refine", "native", "tier4-match", "tier4-native", "tier4-native-refine",
            "photo-tiers-native"):
        if args.stage == "photo-tiers-native":
            selected = measure_photo_tiers(study)
        elif photo_filter_stage:
            cases = photo_filter_cases(args.stage)
            study.measure_quality(cases, check_target=True)
            selected = photo_filter_matches(
                study, cases, *photo_filter_stages[args.stage],
                compare_filter_free=args.stage == "photo-filter-backlit-bracket",
            )
        elif args.stage == "match":
            study.measure_quality(match_cases())
            selected = study.prepare_timing(BANDS, FAST_POLICIES)
        elif args.stage == "refine":
            study.measure_quality(refine_cases())
            selected = study.prepare_timing((1,), FAST_POLICIES + GABORISH_POLICIES)
        elif args.stage == "tier4-match":
            study.measure_quality(tier4_cases())
            selected = study.prepare_timing(BANDS, ("tier4",))
        elif args.stage in ("tier4-native", "tier4-native-refine"):
            study.measure_quality(
                native_tier4_cases(refine=args.stage == "tier4-native-refine"),
                check_target=True,
            )
            selected = study.prepare_timing((4,), ("tier1-epf2", "tier1-epf0", "tier4"))
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
    elif args.stage == "palette-grid":
        selected = study.measure_palette_grid()
        (study.out / "finalists.json").write_text(json.dumps(
            {f"{i}-{s}-{d}": rows for (i, s, d), rows in selected.items()}, indent=2) + "\n")
        study.measure_time(selected)
        (study.out / "complete").write_text(f"{len(selected) * len(MODULAR_PALETTE)} exact streams\n")
        return
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
