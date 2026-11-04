#!/usr/bin/env python3
"""Freeze DC CFL boundary outputs from the independent v0.12 object oracle."""
import argparse
import struct
import subprocess
from pathlib import Path


def bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("oracle", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    cases = [
        (3, 13, 1, 1.6410256624221802, 3.6758973598480225),
        (3, 64, 1, 0.3333333432674408, 2.3333332538604736),
        (17, 3, 1, 1.2549020051956177, 2.8109803199768066),
        (3, 64, 1, 0.6666666865348816, 1.6666666269302368),
        (3, 64, 1, 1.3333333730697632, 4.666666507720947),
        (1024, 1, 1, 0.125, 0.125),
        (53, 7, 0, 0.4, 0.55),
        (127, 13, 2, 0.75, 1.1),
        (1103, 3, 1, 0.01, 0.02),
        (8192, 64, 3, 0.001, 0.0014),
    ]
    inputs = "".join(" ".join(map(str, case)) + "\n" for case in cases)
    result = subprocess.run([str(args.oracle.resolve(strict=True))], input=inputs,
                            text=True, capture_output=True, check=True)
    rows = [line.split(",") for line in result.stdout.splitlines()]
    if len(rows) != len(cases) or any(len(row) != 5 for row in rows):
        raise RuntimeError(f"incomplete oracle output: {result.stdout!r}")
    with args.out.open("x") as out:
        out.write("# libjxl v0.12 Quantizer + QuantizeGradient; default DC correlation\n")
        out.write("# global_scale quant_dc precision Ybits Bbits qY qB legacyB invYbits yFactorBits\n")
        for case, row in zip(cases, rows, strict=True):
            global_scale, quant_dc, precision, y, b = case
            qy, qb, legacy = map(int, row[:3])
            inv_y, y_factor = map(float, row[3:])
            out.write(f"{global_scale} {quant_dc} {precision} {bits(y):08x} {bits(b):08x} "
                      f"{qy} {qb} {legacy} {bits(inv_y):08x} {bits(y_factor):08x}\n")
    print(f"{len(rows)} CFL records written to {args.out}")


if __name__ == "__main__":
    main()
