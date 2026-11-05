#!/usr/bin/env python3
"""Retain an effort-3/5 option ablation and require exact independent decodes."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess

from PIL import Image, __version__ as pillow_version


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("encoder", type=Path)
    parser.add_argument("decoder_check", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()
    encoder = args.encoder.resolve(strict=True)
    decoder = args.decoder_check.resolve(strict=True)
    source = args.source.resolve(strict=True)
    with Image.open(source) as opened:
        rgb = opened.convert("RGB")
    n = args.size
    if n <= 0 or min(rgb.size) < n:
        parser.error("source must contain the requested positive square crop")
    x, y = (rgb.width - n) // 2, (rgb.height - n) // 2
    crop = rgb.crop((x, y, x + n, y + n))
    args.out.mkdir(parents=True, exist_ok=False)
    crop_path = args.out / "source.png"
    crop.save(crop_path)
    metadata = dict(source=str(source), source_sha256=digest(source), crop_origin=[x, y],
                    size=n, crop_pixels_sha256=hashlib.sha256(crop.tobytes()).hexdigest(),
                    crop_png_sha256=digest(crop_path), encoder_sha256=digest(encoder),
                    decoder_check_sha256=digest(decoder), pillow_version=pillow_version)
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    arms = [("e3", 3, []), ("e5", 5, []),
            ("e5-no-patches", 5, ["--no-patches"]),
            ("e5-no-palette", 5, ["--modular-palette-colors", "0"]),
            ("e5-no-patches-no-palette", 5, ["--no-patches", "--modular-palette-colors", "0"])]
    for effort in [3, 5]:
        for rct in [0, 5, 6, 10]:
            arms.append((f"e{effort}-rct{rct}", effort, ["--force-rct", str(rct)]))
    arms.extend([
        ("e5-rct10-no-patches", 5, ["--force-rct", "10", "--no-patches"]),
        ("e5-rct10-no-palette", 5, ["--force-rct", "10", "--modular-palette-colors", "0"]),
        ("e5-rct10-no-patches-no-palette", 5,
         ["--force-rct", "10", "--no-patches", "--modular-palette-colors", "0"]),
    ])
    for effort in [3, 5]:
        arms.extend([
            (f"e{effort}-no-squeeze", effort, ["--no-squeeze"]),
            (f"e{effort}-no-tree", effort, ["--no-tree-learning"]),
            (f"e{effort}-explicit-tree", effort, ["--tree-learning"]),
        ])
        for fraction in ["0.05", "0.15", "0.25"]:
            arms.append((f"e{effort}-tree-f{fraction}", effort,
                         ["--tree-learning", "--tree-learning-sample-fraction", fraction]))
    files = []
    with (args.out / "cells.tsv").open("x") as table:
        writer = csv.writer(table, delimiter="\t")
        writer.writerow(["arm", "effort", "bytes", "sha256", "options"])
        for name, effort, options in arms:
            output = args.out / f"{name}.jxl"
            command = [str(encoder), str(crop_path), str(output), "--lossless",
                       "--effort", str(effort), "--threads", "1", *options]
            with (args.out / f"{name}.log").open("x") as log:
                log.write(f"command: {command!r}\n")
                log.flush()
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
            writer.writerow([name, effort, output.stat().st_size, digest(output), json.dumps(options)])
            table.flush()
            files.append(str(output))
            print(f"{name}: {output.stat().st_size} bytes", flush=True)
    with (args.out / "decoders.log").open("x") as log:
        subprocess.run([str(decoder), str(crop_path), *files], check=True,
                       stdout=log, stderr=subprocess.STDOUT)
    print(f"all {len(files)} streams reproduce the crop in every decoder", flush=True)


if __name__ == "__main__":
    main()
