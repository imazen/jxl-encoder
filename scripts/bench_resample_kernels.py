#!/usr/bin/env python3
"""Interleave two resample_kernel_cost binaries and require identical hashes.

Each binary persists encoded artifacts through ARTIFACT_DIR. Full invocation
output and progress survive in ARTIFACT_DIR, including failed runs.
"""

import argparse
import csv
import io
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--image", type=Path, action="append", required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[64, 256, 1024, 4096])
    parser.add_argument("--effort", type=int, default=7)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    args = parser.parse_args()
    assert args.repeats > 0 and all(size > 0 for size in args.sizes)
    for path in [args.before, args.after, *args.image]:
        assert path.is_file(), path
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.artifacts.mkdir(parents=True, exist_ok=True)
    log_path = args.artifacts / (args.output.stem + ".log")
    meta = {
        "log": str(log_path),
        "command": sys.argv,
        "hostname": platform.node(),
        "platform": platform.platform(),
        "commit": subprocess.check_output(["jj", "log", "--no-graph", "-r", "@",
                                           "-T", "commit_id"], text=True).strip(),
        "binary_sha256": {},
        "timing": "One warmed measurement per arm per invocation; alternate binary order each repeat.",
    }
    import hashlib
    for name in ["before", "after"]:
        with getattr(args, name).open("rb") as source:
            meta["binary_sha256"][name] = hashlib.file_digest(source, "sha256").hexdigest()
    args.output.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    with args.output.open("w") as output, log_path.open("w") as log:
        writer = None
        for image in args.image:
            for size in args.sizes:
                hashes = None
                for repeat in range(args.repeats):
                    arms = ["before", "after"] if repeat % 2 == 0 else ["after", "before"]
                    for arm in arms:
                        progress = f"{image.name} crop={size} repeat={repeat + 1}/{args.repeats} {arm}"
                        print(progress, flush=True)
                        log.write(progress + "\n")
                        log.flush()
                        env = dict(os.environ, IMGS=str(image), CROP=str(size), REPS="1",
                                   EFFORT=str(args.effort), ARTIFACT_DIR=str(args.artifacts),
                                   RAYON_NUM_THREADS="4")
                        run = subprocess.run(["nice", "-n", "19", str(getattr(args, arm).resolve())],
                                             env=env, stdout=subprocess.PIPE, stderr=log, text=True)
                        log.write(run.stdout)
                        log.flush()
                        run.check_returncode()
                        rows = list(csv.DictReader(io.StringIO(run.stdout), delimiter="\t"))
                        assert len(rows) == 1, rows
                        row = rows[0]
                        actual = tuple(row[key] for key in ["sharper_sha256", "iterative_sha256", "encoded_sha256"])
                        if hashes is None:
                            hashes = actual
                        assert actual == hashes, (progress, hashes, actual)
                        row = dict(arm=arm, repeat=repeat, crop=size, effort=args.effort,
                                   source=str(image), **row)
                        if writer is None:
                            writer = csv.DictWriter(output, fieldnames=list(row), delimiter="\t", lineterminator="\n")
                            writer.writeheader()
                        writer.writerow(row)
                        output.flush()


if __name__ == "__main__":
    main()
