#!/usr/bin/env python3
"""Deterministic PNG fixtures for the libjxl byte-parity harness (T4).

Dependency-free (stdlib `zlib` + `struct` only) so the harness runs on a bare
machine, and byte-reproducible so a parity delta measured today is comparable to
one measured next month.

Size ladder follows the project sweep rule (tiny / small / medium): 32² and 64²
expose fixed per-file overhead, which is where a 27-bit header divergence is
4 % of the file and where it is otherwise invisible; 512² is the smallest size
that yields more than one AC group (group_dim = 256), which is what makes the
TOC per-section attribution meaningful.

    make_parity_fixtures.py <outdir>
"""

import os
import struct
import sys
import zlib


def write_png(path, w, h, px, alpha=False):
    ch = 4 if alpha else 3
    raw = bytearray()
    for y in range(h):
        raw.append(0)  # filter type 0 (None) on every row
        for x in range(w):
            raw.extend(px(x, y)[:ch])
    comp = zlib.compress(bytes(raw), 9)

    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8,
                                           6 if alpha else 2, 0, 0, 0)))
        f.write(chunk(b"IDAT", comp))
        f.write(chunk(b"IEND", b""))


class LCG:
    """Same generator as the in-tree hash-lock fixtures
    (`tests/it/strategy_libjxl_byte_lock.rs::noise_rgb_48x48`), so a cell here
    is comparable with a cell there."""

    def __init__(self, seed=0x12345678):
        self.s = seed

    def byte(self):
        self.s = (self.s * 1103515245 + 12345) & 0xFFFFFFFF
        return (self.s >> 16) & 0xFF


def gradient(w, h):
    return lambda x, y: (x * 255 // (w - 1), y * 255 // (h - 1), 128, 255)


def noise(w, h, seed):
    g = LCG(seed)
    rows = [[(g.byte(), g.byte(), g.byte(), 255) for _ in range(w)]
            for _ in range(h)]
    return lambda x, y: rows[y][x]


def photo(x, y):
    """Smooth low-frequency base + one hard edge + mild high-frequency dither.
    Stands in for photo-class content: an all-smooth fixture exercises neither
    the AC-strategy search nor the entropy clustering."""
    import math
    r = int(127 + 100 * math.sin(x / 57.0) * math.cos(y / 83.0))
    g = int(127 + 90 * math.sin((x + y) / 71.0))
    b = int(127 + 80 * math.cos((x - y) / 47.0))
    if 200 < x < 260:
        r += 60
    t = ((x * 2654435761) ^ (y * 40503)) & 0xF
    clamp = lambda v: max(0, min(255, v))
    return (clamp(r + t - 8), clamp(g), clamp(b), 255)


def rgba_strip(w, h):
    def f(x, y):
        return (x * 255 // (w - 1), y * 255 // (h - 1), 64,
                128 if (w // 3) <= x < (2 * w // 3) else 255)
    return f


def main(outdir):
    os.makedirs(outdir, exist_ok=True)
    j = lambda n: os.path.join(outdir, n)
    write_png(j("gradient_32x32.png"), 32, 32, gradient(32, 32))
    write_png(j("gradient_512x512.png"), 512, 512, gradient(512, 512))
    write_png(j("noise_48x48.png"), 48, 48, noise(48, 48, 0x12345678))
    write_png(j("noise_512x512.png"), 512, 512, noise(512, 512, 0xA5A5A5A5))
    write_png(j("photo_512x512.png"), 512, 512, photo)
    write_png(j("flat_64x64.png"), 64, 64, lambda x, y: (128, 128, 128, 255))
    write_png(j("rgba_64x32.png"), 64, 32, rgba_strip(64, 32), alpha=True)
    print(f"wrote 7 fixtures to {outdir}")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
