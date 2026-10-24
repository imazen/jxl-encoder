#!/usr/bin/env python3
"""Check native animation artifacts against the pinned libjxl 0.12 decoder.

Generate input with JXL_ANIMATION_ARTIFACTS=<dir> cargo test --release -p
jxl-encoder --test animation_contract lossless_animation_preserves.
This is differential validation of this port, not production decoding.
"""
import argparse
import ast
import json
import math
from pathlib import Path
import re
import struct
import subprocess
import tempfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--djxl", required=True, type=Path)
parser.add_argument("artifacts", type=Path)
args = parser.parse_args()
version = subprocess.check_output([str(args.djxl), "--version"], stderr=subprocess.STDOUT, text=True)
if not re.search(r"v0\.12\.", version):
    raise SystemExit(f"libjxl 0.12 required, got {version}")
files = sorted(args.artifacts.glob("*.jxl"))
if len(files) != 24:
    raise SystemExit(f"expected all 24 native precision/color/size fixtures, got {len(files)}")
results = []
with tempfile.TemporaryDirectory(prefix="animation-libjxl-") as temporary:
    for path in files:
        match = re.fullmatch(r"(\d+)x(\d+)-(\d+)-(srgb|linear|pq|hlg)\.jxl", path.name)
        if not match:
            raise SystemExit(f"unexpected fixture {path}")
        width, height, bits = map(int, match.groups()[:3])
        output = Path(temporary) / (path.stem + ".npy")
        subprocess.run([str(args.djxl), str(path), str(output), "--output_frames", "--num_threads=0", "--quiet"], check=True)
        raw = output.read_bytes()
        if raw[:8] != b"\x93NUMPY\x01\x00":
            raise SystemExit(f"unexpected npy header: {raw[:8]!r}")
        length = struct.unpack_from("<H", raw, 8)[0]
        header = ast.literal_eval(raw[10:10 + length].decode("ascii").strip())
        if header != {"descr": "<f4", "fortran_order": False, "shape": (3, height, width, 3)}:
            raise SystemExit(f"unexpected output layout: {header}")
        pixels = struct.iter_unpack("<f", raw[10 + length:])
        maximum = (1 << bits) - 1
        error = 0.0
        samples = 0
        for index, (actual,) in enumerate(pixels):
            frame, sample = divmod(index, width * height * 3)
            expected = ((sample * 73 + sample // 17 * 113 + min(frame, 1) * 29) & maximum) / maximum
            delta = abs(actual - expected)
            if not math.isfinite(actual) or delta >= 0.5 / maximum:
                raise SystemExit(f"{path.name} frame {frame} sample {sample}: {actual} != {expected}")
            error = max(error, delta)
            samples += 1
        if samples != 3 * height * width * 3:
            raise SystemExit(f"short output: {samples}")
        results.append({"fixture": path.name, "samples": samples, "max_absolute_error": error})
print(json.dumps({"decoder": version.strip(), "results": results}, indent=2))
