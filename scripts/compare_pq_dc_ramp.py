#!/usr/bin/env python3
"""Measure the retained PQ byte-lock ramp with an independent v0.12 decoder."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("djxl", type=Path)
    p.add_argument("before", type=Path)
    p.add_argument("after", type=Path)
    p.add_argument("out", type=Path)
    args = p.parse_args()
    decoder = args.djxl.resolve(strict=True)
    version = subprocess.run([str(decoder), "--version"], capture_output=True,
                             text=True, check=True)
    identity = version.stdout + version.stderr
    if not re.search(r"\bdjxl v0\.12\.0(?:\s|$)", identity):
        p.error(f"requires the project v0.12 reference: {identity}")
    if args.before.joinpath("source.rgb16").read_bytes() != args.after.joinpath("source.rgb16").read_bytes():
        p.error("ramp source bytes differ")
    args.out.mkdir(parents=True, exist_ok=False)
    source = np.fromfile(args.before / "source.rgb16", dtype="<u2").reshape(512, 512, 3) / 65535.0
    result = {"decoder_version": identity.strip(), "decoder_sha256": sha(decoder),
              "source_sha256": sha(args.before / "source.rgb16"), "outputs": {}}
    for arm, directory in [("before", args.before), ("after", args.after)]:
        encoded = directory / "encoded.jxl"
        output = args.out / f"{arm}.pfm"
        subprocess.run([str(decoder), str(encoded), str(output), "--num_threads=1",
                        "--color_space=RGB_D65_202_Rel_PeQ"], check=True, capture_output=True)
        with output.open("rb") as f:
            assert f.readline().strip() == b"PF"
            assert tuple(map(int, f.readline().split())) == (512, 512)
            scale = float(f.readline())
            assert abs(scale) == 1.0
            pixels = np.frombuffer(f.read(), dtype=">f4" if scale > 0 else "<f4").reshape(512, 512, 3)[::-1]
        assert np.isfinite(pixels).all()
        result["outputs"][arm] = {"bytes": encoded.stat().st_size, "sha256": sha(encoded),
                                  "pfm_sha256": sha(output),
                                  "mean_pq_signal_error": float(np.abs(pixels.clip(0, 1).astype(float) - source).mean())}
    (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
