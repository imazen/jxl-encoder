#!/usr/bin/env python3
"""Seed the two encoder fuzz boundaries from an explicit real-image manifest.

Writes outside git, records source paths and SHA256, and never overwrites a seed.
The input TSV must have image and path columns, as used by qfseed_lift_ab.py.
"""
import argparse
import csv
import hashlib
import json
import struct
from pathlib import Path
from PIL import Image, __version__ as pillow_version

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("manifest", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--build-commit", required=True)
args = parser.parse_args()
records = []
for source in csv.DictReader(args.manifest.open(), delimiter="\t"):
    with Image.open(source["path"]) as opened:
        rgb = opened.convert("RGB")
    for cap_w, cap_h in [(32, 32), (513, 259)]:
        width, height = min(cap_w, rgb.width), min(cap_h, rgb.height)
        x, y = (rgb.width - width) // 2, (rgb.height - height) // 2
        pixels = rgb.crop((x, y, x + width, y + height)).tobytes()
        for target in ["request_limits", "streaming_roundtrip"]:
            for effort in [5, 8]:
                for lossless in ([0, 1] if target == "streaming_roundtrip" else [0]):
                    if target == "request_limits":
                        header = struct.pack("<IIBfBBB", width, height, 0, 4.0, effort, 0, 0)
                    else:
                        header = struct.pack("<HHBBBB", width - 1, height - 1, effort - 1, 40, lossless, 6)
                    data = header + pixels
                    digest = hashlib.sha256(data).hexdigest()
                    path = args.output / target / digest
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if path.exists():
                        assert path.read_bytes() == data
                    else:
                        with path.open("xb") as output:
                            output.write(data)
                    records.append(dict(target=target, seed_sha256=digest,
                                        source=source["path"], source_image=source["image"],
                                        width=width, height=height, effort=effort, lossless=lossless))
        print(f"seeded {source['image']} {width}x{height}", flush=True)
with (args.output / "_MANIFEST.json").open("x") as output:
    json.dump(dict(build_commit=args.build_commit, pillow_version=pillow_version, inputs=str(args.manifest), seeds=records), output, indent=2)
