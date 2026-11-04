#!/usr/bin/env python3
"""Generate small frozen QuantizeGradient vectors from the libjxl object oracle."""
import argparse
from pathlib import Path
import struct
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("oracle", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    oracle = args.oracle.resolve(strict=True)
    boundaries = [0.61999995, 0.62, 0.62000006, -0.61999995, -0.62,
                  -0.62000006, 2.5, -2.5, 3.5, -3.5]
    with args.out.open("x") as output:
        output.write("# libjxl v0.12 QuantizeGradient: width height input-f32-bits quantized\n")
        for width, height in [(1, 17), (17, 1), (2, 17), (3, 9), (8, 8), (33, 5)]:
            values = [boundaries[i % len(boundaries)] if i % 7 == 0
                      else ((i * 73 + i // width * 29) % 257 - 128) / 8
                      for i in range(width * height)]
            bits = [struct.unpack("<I", struct.pack("<f", value))[0] for value in values]
            values = [struct.unpack("<f", struct.pack("<I", value))[0] for value in bits]
            source = f"{width} {height}\n" + " ".join(map(str, values)) + "\n"
            result = subprocess.run([str(oracle)], input=source, text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            quantized = [int(value) for value in result.stdout.strip().split(",")]
            if len(quantized) != len(values):
                raise RuntimeError(f"incomplete oracle output: {width}x{height}")
            output.write(f"{width} {height} " + ",".join(f"{value:08x}" for value in bits)
                         + " " + ",".join(map(str, quantized)) + "\n")


if __name__ == "__main__":
    main()
