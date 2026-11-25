#!/usr/bin/env python3
"""Fully render retained envelope-sweep streams in two independent decoders."""

import argparse
import csv
import hashlib
import subprocess
from pathlib import Path

from PIL import Image


def input_rgb(path):
    with Image.open(path) as image:
        if image.mode != "RGB":
            raise ValueError(f"{path}: this probe expects an RGB8 fixture")
        return image.size, image.tobytes()


def readbacks(args, writer, log):
    for sweep in args.sweeps:
        tables = list(sweep.glob("libjxl_parity_*.tsv"))
        if len(tables) != 1:
            raise ValueError(f"{sweep}: expected one envelope table")
        with tables[0].open() as table:
            sizes = [row for row in csv.DictReader(table, delimiter="\t") if row["kind"] == "size"]
        if not sizes or len(list(sweep.glob(".*.jxl"))) != 2 * len(sizes):
            raise ValueError(f"{sweep}: each cell must have two retained streams")
        for row in sizes:
            image, effort, distance = (row[key] for key in ("image", "effort", "distance"))
            for side in ("a", "b"):
                stream = sweep / f".{side}_{image}_e{effort}_d{distance}.jxl"
                data = stream.read_bytes()
                for decoder, executable, suffix, flags in (
                    ("djxl", args.djxl, "png", ["--num_threads=0", "--bits_per_sample=8",
                        "--color_space=RGB_D65_SRG_Rel_SRG"]),
                    ("jxl-rs", args.jxl_cli, "ppm", ["--num-threads", "1", "--data-type", "u8"]),
                ):
                    path = args.output / f"readback.{suffix}"
                    command = [executable, stream, path, *flags]
                    print(subprocess.list2cmdline(map(str, command)), file=log, flush=True)
                    subprocess.run(command, check=True, stdout=log, stderr=log)
                    size, samples = input_rgb(path)
                    if size != (int(row["width"]), int(row["height"])):
                        raise ValueError(f"{decoder}: unexpected dimensions for {stream}")
                    exact = float(distance) == 0
                    if exact and samples != input_rgb(args.fixtures / f"{image}.png")[1]:
                        raise ValueError(f"{decoder}: lossless mismatch in {stream}")
                    writer.writerow((sweep.name, image, effort, distance, side, decoder, len(data),
                                     hashlib.sha256(data).hexdigest(), hashlib.sha256(samples).hexdigest(),
                                     int(exact)))
                    path.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixtures", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("sweeps", type=Path, nargs="+")
    parser.add_argument("--djxl", type=Path, required=True)
    parser.add_argument("--jxl-cli", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir()
    with (args.output / "readbacks.tsv").open("x") as output, (args.output / "commands.log").open("x") as log:
        for executable in (args.djxl, args.jxl_cli):
            print(f"{hashlib.sha256(executable.read_bytes()).hexdigest()}  {executable}", file=log)
        writer = csv.writer(output, delimiter="\t", lineterminator="\n")
        writer.writerow(("stage", "image", "effort", "distance", "arm", "decoder", "bytes",
                         "stream_sha256", "rgb_sha256", "lossless_checked"))
        readbacks(args, writer, log)


if __name__ == "__main__":
    main()
