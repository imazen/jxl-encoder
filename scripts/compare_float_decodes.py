#!/usr/bin/env python3
"""Attribute retained RGB float decoder differences without changing a gate."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_pixels(path):
    if path.suffix == ".npy":
        pixels = np.load(path, allow_pickle=False)
        if pixels.ndim == 4:
            assert pixels.shape[0] == 1, "requires exactly one displayed frame"
            pixels = pixels[0]
    elif path.suffix == ".f32le":
        info = json.loads(path.with_suffix(".json").read_text())
        assert info["format"] == "f32le"
        pixels = np.fromfile(path, dtype="<f4")
        assert pixels.size == info["samples"]
        pixels = pixels.reshape(info["height"], info["width"], info["channels"])
    elif path.suffix == ".pfm":
        with path.open("rb") as stream:
            assert stream.readline().strip() == b"PF", "requires RGB PFM"
            width, height = map(int, stream.readline().split())
            scale = float(stream.readline())
            assert abs(scale) == 1.0, "unexpected PFM scale"
            pixels = np.frombuffer(stream.read(), dtype=">f4" if scale > 0 else "<f4")
            pixels = pixels.reshape(height, width, 3)[::-1]
    else:
        raise ValueError(f"unsupported decoded format: {path}")
    assert pixels.dtype.kind == "f" and pixels.dtype.itemsize == 4, "requires f32"
    assert pixels.ndim == 3 and pixels.shape[2] >= 3, "requires RGB channels"
    assert np.isfinite(pixels).all(), f"non-finite decoder output: {path}"
    return pixels.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--decode", action="append", required=True, help="label=file")
    parser.add_argument("--source-f32", type=Path, help="original RGB/RGBA f32 input")
    args = parser.parse_args()
    arrays, identities = {}, {}
    for value in args.decode:
        label, name = value.split("=", 1)
        if not label or label in arrays:
            parser.error("decoder labels must be nonempty and unique")
        path = Path(name).resolve(strict=True)
        arrays[label] = read_pixels(path)
        identities[label] = {"path": str(path), "sha256": digest(path),
                             "shape": list(arrays[label].shape)}
    if len(arrays) < 2:
        parser.error("requires at least two decoded outputs")
    height, width = next(iter(arrays.values())).shape[:2]
    assert all(a.shape[:2] == (height, width) for a in arrays.values()), "shape mismatch"
    source = None
    if args.source_f32:
        source = np.fromfile(args.source_f32, dtype="<f4").reshape(height, width, -1)
        assert source.shape[2] in [3, 4] and np.isfinite(source).all()
    # Match the sweep: clamp each f32 sample, then subtract in f64.
    clipped = {name: value[:, :, :3].clip(0, 1).astype(np.float64)
               for name, value in arrays.items()}
    result = {"outputs": identities, "comparisons": []}
    if source is not None:
        visible = source[:, :, 3] != 0 if source.shape[2] == 4 else np.ones((height, width), dtype=bool)
        assert visible.any()
        result["source_sha256"] = digest(args.source_f32)
        result["visible_color_mean_signal_error"] = {
            name: float(np.abs(value - source[:, :, :3].astype(np.float64))[visible].mean())
            for name, value in clipped.items()}
    for left, right in itertools.combinations(arrays, 2):
        difference = np.abs(clipped[left] - clipped[right])
        y, x, channel = map(int, np.unravel_index(difference.argmax(), difference.shape))
        row = {"left": left, "right": right, "maximum_clamped_rgb_difference": float(difference.max()),
               "samples_above_0.0200": int((difference > 0.0200).sum()),
               "worst_xy_channel": [x, y, channel],
               "left_raw": float(arrays[left][y, x, channel]),
               "right_raw": float(arrays[right][y, x, channel])}
        if source is not None:
            row["source_at_worst"] = source[y, x].astype(float).tolist()
        result["comparisons"].append(row)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
