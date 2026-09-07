#!/usr/bin/env python3
"""Run the persisted distance-targeting probe on an explicit image manifest.

The manifest is TSV with image, class, path columns. Work is serial. Each
image gets a result TSV compatible with qfseed_lift_ab_analyze.py; complete
probe output, encodes, diffmaps, and build provenance remain in artifacts.
"""

import argparse
import csv
import os
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--djxl", type=Path, required=True)
    parser.add_argument("--efforts", default="5,7,8")
    parser.add_argument("--distances", default="2,3,3.4,3.6,4,5,6,8")
    parser.add_argument("--off-distances", default="0.75,1,1.25,1.5,2,2.5,3,3.4,3.6,4,5,6,8")
    parser.add_argument("--crop", type=int, default=1024)
    parser.add_argument("--iters", type=int)
    parser.add_argument("--modes", nargs="+", choices=["on", "off"], default=["on", "off"])
    args = parser.parse_args()
    with args.manifest.open() as source:
        images = list(csv.DictReader(source, delimiter="\t"))
    assert images
    for image in images:
        assert Path(image["path"]).is_file(), image
    for path in [args.probe, args.djxl]:
        assert path.is_file(), path
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.artifacts.mkdir(parents=True, exist_ok=True)
    for image in images:
        output_path = args.output_dir / (image["image"] + ".tsv")
        # A different run gets a different directory; never overwrite a sweep.
        with output_path.open("x") as output:
            writer = None
            for mode in args.modes:
                print(f"{image['image']} {image['class']} lift={mode}", flush=True)
                env = dict(os.environ, IMG=image["path"], CROP=str(args.crop),
                           EFFORTS=args.efforts, ARTIFACT_DIR=str(args.artifacts),
                           DJXL_PATH=str(args.djxl.resolve()), RAYON_NUM_THREADS="4",
                           DISTANCES=args.distances if mode == "on" else args.off_distances)
                for key in ["JXL_BUTTLOOP_INITIAL_QF_SCALE", "JXL_W44_109_ADAPTIVE_QUANT_QF_SCALE"]:
                    env.pop(key, None)
                    if mode == "off":
                        env[key] = "1.0"
                env.pop("ITERS", None)
                if args.iters is not None:
                    env["ITERS"] = str(args.iters)
                stem = args.artifacts / (image["image"] + "-" + mode)
                with stem.with_suffix(".log").open("x") as log:
                    with subprocess.Popen(["nice", "-n", "19", str(args.probe.resolve())],
                                          env=env, stdout=subprocess.PIPE, stderr=log,
                                          text=True) as run:
                        reader = csv.DictReader(run.stdout, delimiter="\t")
                        count = 0
                        for row in reader:
                            row = dict(image=image["image"], **{"class": image["class"]}, mode=mode, **row)
                            if writer is None:
                                writer = csv.DictWriter(output, fieldnames=list(row), delimiter="\t",
                                                        lineterminator="\n")
                                writer.writeheader()
                            writer.writerow(row)
                            output.flush()
                            count += 1
                            print(f"  e{row['effort']} d{row['d_req']} ratio={row['delivered_ratio']}", flush=True)
                        assert run.wait() == 0, f"probe failed; see {stem.with_suffix('.log')}"
                        expected = len(args.efforts.split(",")) * len(env["DISTANCES"].split(","))
                        assert count == expected, (count, expected)


if __name__ == "__main__":
    main()
