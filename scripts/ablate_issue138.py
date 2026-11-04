#!/usr/bin/env python3
"""Attribute failing issue-138 cells without changing sweep tolerances.

Reads an existing replay directory. All variants retain the source seed,
layout, distance and dimensions; only the explicitly named options differ.
Every original failure remains in the results, regardless of other arms.
"""
import argparse
import csv
import re
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("binary", type=Path)
    p.add_argument("replay", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("corpus", type=Path)
    p.add_argument("icc", type=Path)
    p.add_argument("--arms", help="explicit comma-separated arm subset")
    args = p.parse_args()
    binary = args.binary.resolve(strict=True)
    photos = args.corpus / "CID22/CID22-512/validation"
    screens = args.corpus / "gb82-sc"
    for directory in [photos, screens, args.icc]:
        if not directory.is_dir():
            p.error(f"missing input directory: {directory}")
    cases = list(csv.DictReader((args.replay / "replays.tsv").open(), delimiter="\t"))
    args.out.mkdir(parents=True, exist_ok=False)
    with (args.out / "ablations.tsv").open("x") as table:
        table.write("seed\tindex\tarm\texit\tstudy\n")
        for case in cases:
            if case["exit"] == "0":
                continue
            seed, index = case["seed"], case["index"]
            original = args.replay / f"{seed}-{index}"
            description = (original / "failures" / f"cell{index}.txt").read_text().splitlines()[0]
            match = re.fullmatch(r".* d[\d.]+ e\d+(.*)", description)
            if not match:
                raise ValueError(f"not a lossy cell: {description}")
            opts = match[1].split()
            # Streaming is a caller shape, not a lossy-config option.
            opts = [o for o in opts if not o.startswith("stream") and o != "exif+xmp"]
            arms = {
                "original": opts,
                "custom": [o for o in opts if not o.startswith("strategy:")] + ["strategy:custom"],
                "dc_parity": [o for o in opts if not o.startswith("strategy:")] + ["strategy:dc-parity"],
                "libjxl": [o for o in opts if not o.startswith("strategy:")] + ["strategy:libjxl"],
                "libjxl_r1": [o for o in opts if not o.startswith(("strategy:", "resampling:"))] + ["strategy:libjxl", "resampling:1"],
                "libjxl_r1_loop": [o for o in opts if not o.startswith(("strategy:", "resampling:"))]
                    + ["strategy:libjxl", "resampling:1", "hdr_loss:butteraugli", "butteraugli_iters:2"],
                "libjxl_r1_loop8": [o for o in opts if not o.startswith(("strategy:", "resampling:"))]
                    + ["strategy:libjxl", "resampling:1", "hdr_loss:butteraugli", "butteraugli_iters:8"],
                "vdp2_loop": opts + ["hdr_loss:vdp2", "butteraugli_iters:2"],
                "vdp2_loop8": opts + ["hdr_loss:vdp2", "butteraugli_iters:8"],
                "dc_vdp2_loop": [o for o in opts if not o.startswith("strategy:")]
                    + ["strategy:dc-parity", "hdr_loss:vdp2", "butteraugli_iters:2"],
                "no_options": [],
                "no_faster": [o for o in opts if not o.startswith("faster_decoding:")],
                "epf0": [o for o in opts if not o.startswith("epf:")] + ["epf:0"],
                "no_gaborish": [o for o in opts if o != "no_gaborish"] + ["no_gaborish"],
                "butteraugli_loop": opts + ["hdr_loss:butteraugli", "butteraugli_iters:2"],
                "butteraugli_loop8": opts + ["hdr_loss:butteraugli", "butteraugli_iters:8"],
                "dc_parity_loop": [o for o in opts if not o.startswith("strategy:")]
                    + ["strategy:dc-parity", "hdr_loss:butteraugli", "butteraugli_iters:2"],
                "dc_parity_loop8": [o for o in opts if not o.startswith("strategy:")]
                    + ["strategy:dc-parity", "hdr_loss:butteraugli", "butteraugli_iters:8"],
            }
            if args.arms:
                requested = args.arms.split(",")
                if len(set(requested)) != len(requested) or any(a not in arms for a in requested):
                    p.error("unknown or duplicate arm")
                arms = {a: arms[a] for a in requested}
            for arm, selected in arms.items():
                out = args.out / f"{seed}-{index}-{arm}"
                command = [str(binary), "--images", str(photos), "--images", str(screens),
                           "--icc", str(args.icc), "--out", str(out), "--seed", seed,
                           "--only", index, "--threads", "1", "--options", ",".join(selected),
                           "--reference-study", "--study-max-pixels", "100000000", "--retain"]
                with out.with_suffix(".log").open("x") as log:
                    log.write(f"command: {command!r}\n")
                    log.flush()
                    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
                study = out / "reference_study.tsv"
                if not study.is_file():
                    raise RuntimeError(f"missing reference study: {out}")
                records = list(csv.DictReader((line for line in study.open() if not line.startswith("#")), delimiter="\t"))
                if len(records) != 1:
                    raise RuntimeError(f"incomplete reference study: {out}")
                # Failed variants retain their exact source bytes. They must
                # agree with the original independently captured input.
                saved = out / f"cell{index}.input"
                if saved.read_bytes() != (original / "failures" / f"cell{index}.input").read_bytes():
                    raise RuntimeError(f"input changed: {out}")
                table.write(f"{seed}\t{index}\t{arm}\t{result.returncode}\t{study}\n")
                table.flush()
                print(f"{seed}/{index} {arm}: exit {result.returncode}, {records[0]}", flush=True)


if __name__ == "__main__":
    main()
