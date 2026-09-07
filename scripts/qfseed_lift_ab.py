#!/usr/bin/env python3
"""Run the persisted distance-targeting probe on an explicit image manifest.

The manifest is TSV with image, class, path columns. Work is serial. Each
image gets a result TSV compatible with qfseed_lift_ab_analyze.py; complete
probe output, encodes, diffmaps, and build provenance remain in artifacts.
"""

import argparse
import csv
import math
import json
import hashlib
import os
import subprocess
from pathlib import Path


def feedback_rows(args, image, mode):
    """Measure the full encoder's distance response; never interpolate scores."""
    for effort in args.efforts.split(","):
        for target_text in args.distances.split(","):
            target = float(target_text)
            assert target > 0, "feedback targeting requires positive distance"
            trial = target
            fine = coarse = None
            observed = []
            seen = set()
            for step in range(12):
                trial = min(25.0, max(0.01, trial))
                if trial in seen:
                    break
                seen.add(trial)
                stem = f"{image['image']}-{mode}-e{effort}-t{target_text}-step{step}"
                env = dict(os.environ, IMG=image["path"], CROP=str(args.crop),
                           EFFORTS=effort, DISTANCES=str(trial),
                           ARTIFACT_DIR=str(args.artifacts),
                           DJXL_PATH=str(args.djxl.resolve()), RAYON_NUM_THREADS="4")
                for key in ["JXL_BUTTLOOP_INITIAL_QF_SCALE", "JXL_W44_109_ADAPTIVE_QUANT_QF_SCALE", "ITERS"]:
                    env.pop(key, None)
                if mode == "off":
                    env["JXL_BUTTLOOP_INITIAL_QF_SCALE"] = "1.0"
                    env["JXL_W44_109_ADAPTIVE_QUANT_QF_SCALE"] = "1.0"
                if args.iters is not None:
                    env["ITERS"] = str(args.iters)
                with (args.artifacts / (stem + ".log")).open("x") as log:
                    result = subprocess.run(["nice", "-n", "19", str(args.probe.resolve())],
                                            env=env, stdout=subprocess.PIPE, stderr=log,
                                            text=True)
                (args.artifacts / (stem + ".tsv")).write_text(result.stdout)
                assert result.returncode == 0, stem
                candidates = list(csv.DictReader(result.stdout.splitlines(), delimiter="\t"))
                assert len(candidates) == 1, stem
                row = candidates[0]
                score = float(row["bfly"])
                assert math.isfinite(score) and score >= 0, stem
                observed.append(row)
                print(f"  e{effort} target={target:g} step={step} internal={trial:.5g} delivered={score:.5g}", flush=True)
                if target <= score <= target * 1.1:
                    break
                if score < target:
                    fine = trial
                else:
                    coarse = trial
                if fine is not None and coarse is not None:
                    trial = math.sqrt(fine * coarse)
                elif fine is not None:
                    trial = fine * 2
                else:
                    trial = coarse / 2
            in_band = [r for r in observed if target <= float(r["bfly"]) <= target * 1.1]
            best = min(in_band, key=lambda r: int(r["bytes"])) if in_band else min(
                observed, key=lambda r: abs(float(r["bfly"]) / target - 1))
            yield dict(image=image["image"], **{"class": image["class"]}, mode=mode,
                       **{**best, "d_req": target_text,
                          "delivered_ratio": float(best["bfly"]) / target},
                       internal_distance=best["d_req"], measurements=len(observed),
                       search_encode_ms=sum(float(r["encode_ms"]) for r in observed),
                       targeting_status="in_band" if in_band else "outside_band")


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
    parser.add_argument("--target-feedback", action="store_true",
                        help="Adjust the full encoder distance against measured decoded quality.")
    parser.add_argument("--modes", nargs="+", choices=["on", "off"], default=["on", "off"])
    args = parser.parse_args()
    # LossyConfig validates this range; fail before starting a partial sweep.
    for grid in [args.distances, args.off_distances]:
        assert all(0.0 <= float(d) <= 25.0 for d in grid.split(",")), "distance must be in 0..=25"
    with args.manifest.open() as source:
        images = list(csv.DictReader(source, delimiter="\t"))
    assert images
    for image in images:
        assert Path(image["path"]).is_file(), image
    for path in [args.probe, args.djxl]:
        assert path.is_file(), path
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.artifacts.mkdir(parents=True, exist_ok=True)
    if args.target_feedback:
        (args.output_dir / "meta.json").write_text(json.dumps(dict(
            probe=str(args.probe.resolve()),
            probe_sha256=hashlib.sha256(args.probe.read_bytes()).hexdigest(),
            manifest=str(args.manifest.resolve()), crop=args.crop,
            efforts=args.efforts, distances=args.distances, modes=args.modes,
            artifacts=str(args.artifacts.resolve()), max_measurements=12,
            ratio_band=[1,1.1]), indent=2) + "\n")
    for image in images:
        output_path = args.output_dir / (image["image"] + ".tsv")
        # A different run gets a different directory; never overwrite a sweep.
        with output_path.open("x") as output:
            writer = None
            for mode in args.modes:
                print(f"{image['image']} {image['class']} lift={mode}", flush=True)
                if args.target_feedback:
                    for row in feedback_rows(args, image, mode):
                        if writer is None:
                            writer = csv.DictWriter(output, fieldnames=list(row), delimiter="\t",
                                                    lineterminator="\n")
                            writer.writeheader()
                        writer.writerow(row)
                        output.flush()
                    continue
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
