# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0"]
# ///
"""Deterministic generator for float / high-bit-depth synthetic line-art,
graphics and plot content.

Why this exists
---------------
The jxl-encoder #109 F5 lossless-float parity measurement
(`benchmarks/lossless_float_parity_2026-09-08.md`) is **photographic only** --
imazen-26's float set (`.hdr.png`) exists in just four photo strata, and its
mantissas are a 16-bit lattice because it is derived from 16-bit integer
sources.  There is no float line-art / plot / graphics content anywhere in the
corpus, so the lossless float path has never been measured on the content class
where the predictor / entropy balance is most likely to differ.

This tool fills that gap.  It mirrors the *categories* and *conventions* of
imazen-26's `7000-lilith-plots` stratum (including its load-bearing
aliased / anti-aliased pairing at the same seed) but emits two high-precision
arms of identical content instead of 8-bit PNG:

  * ``png16``  -- 16-bit integer RGB PNG  (round(v * 65535))
  * ``pfm32``  -- IEEE binary32 RGB, little-endian PFM (scale -1.0), rows
                  bottom-up, which is what ``cjxl`` reads natively.

Design decisions that matter
----------------------------
* **Anti-aliasing is analytic, not supersampled.**  Coverage comes from an
  exact signed-distance field, so edge pixels carry genuinely continuous
  fractional coverage.  A supersampled AA arm would put every sample on a
  1/N lattice and the float arm would just re-measure the integer path --
  which is the one failure mode that would make this whole set worthless.
* **The aliased arm is a hard 0/1 coverage test on the same geometry and the
  same seed.**  The pair is the discriminator the cfl_two_pass line-art
  investigation needed; keep it.
* **Every size is rendered natively.**  Nothing is resampled from a larger
  render: resampling changes the local statistics predictors key on, so a
  resampled 256 would not be the same content class as its 1024 sibling.
* **The PRNG is a local splitmix64 over Python ints**, not ``numpy.random``,
  so geometry is byte-reproducible independent of the numpy version.  numpy is
  used only for IEEE raster arithmetic.

Usage
-----
    uv run scripts/gen_float_lineart_corpus.py --selftest
    uv run scripts/gen_float_lineart_corpus.py --out ~/work/zen/imazen26-float-synth

Per the imazen-26 VARIANTS-SPEC section 6 ("one committed tool per pipeline
generation; ad-hoc ImageMagick/PIL invocations are banned"), this file is the
generator of record for the set it produces, and its sha256 is recorded in the
emitted manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import struct
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# deterministic PRNG (splitmix64 over Python ints -- no numpy, no version drift)
# --------------------------------------------------------------------------

_M64 = (1 << 64) - 1


class Rng:
    """splitmix64. Reproducible across Python builds and numpy versions."""

    __slots__ = ("s",)

    def __init__(self, seed: int) -> None:
        self.s = seed & _M64

    def next_u64(self) -> int:
        self.s = (self.s + 0x9E3779B97F4A7C15) & _M64
        z = self.s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _M64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _M64
        return z ^ (z >> 31)

    def uniform(self, lo: float = 0.0, hi: float = 1.0) -> float:
        # 53 significant bits -> a real in [0,1)
        return lo + (hi - lo) * ((self.next_u64() >> 11) * (1.0 / (1 << 53)))

    def randint(self, lo: int, hi: int) -> int:
        """inclusive both ends"""
        return lo + int(self.next_u64() % (hi - lo + 1))

    def choice(self, seq):
        return seq[self.randint(0, len(seq) - 1)]


# --------------------------------------------------------------------------
# PFM  (little-endian, scale -1.0, rows bottom-up)
# --------------------------------------------------------------------------


def write_pfm(path: Path, img: np.ndarray) -> None:
    """img: (h, w, 3) float32, top-down. PFM stores rows bottom-up."""
    assert img.dtype == np.float32 and img.ndim == 3 and img.shape[2] == 3
    h, w, _ = img.shape
    body = np.ascontiguousarray(img[::-1], dtype="<f4")
    with open(path, "wb") as f:
        f.write(b"PF\n")
        f.write(f"{w} {h}\n".encode("ascii"))
        f.write(b"-1.0\n")  # negative scale == little-endian
        f.write(body.tobytes())


def read_pfm(path: Path) -> np.ndarray:
    """Returns (h, w, 3) float32, top-down. Honours the scale sign:
    djxl v0.12 *writes* PFM with scale +1.0 (BIG-endian), while we write -1.0
    (little-endian). Both must read back identically."""
    with open(path, "rb") as f:
        magic = f.readline().strip()
        if magic not in (b"PF", b"Pf"):
            raise ValueError(f"{path}: not a PFM ({magic!r})")
        nchan = 3 if magic == b"PF" else 1
        line = f.readline()
        while line.startswith(b"#"):
            line = f.readline()
        w, h = (int(t) for t in line.split())
        scale = float(f.readline().strip())
        dt = "<f4" if scale < 0 else ">f4"
        data = np.frombuffer(f.read(w * h * nchan * 4), dtype=dt)
    img = data.reshape(h, w, nchan).astype(np.float32)
    return img[::-1].copy()  # bottom-up on disk -> top-down in memory


# --------------------------------------------------------------------------
# 16-bit PNG  (stdlib zlib; numpy-vectorised filter heuristic)
# --------------------------------------------------------------------------


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def write_png16(path: Path, img_u16: np.ndarray) -> None:
    """img_u16: (h, w, 3) uint16."""
    assert img_u16.dtype == np.uint16 and img_u16.shape[2] == 3
    h, w, _ = img_u16.shape
    rows = np.ascontiguousarray(img_u16.astype(">u2")).view(np.uint8).reshape(h, w * 6)
    bpp = 6

    prev = np.zeros(w * 6, dtype=np.uint8)
    out = bytearray()
    for y in range(h):
        cur = rows[y]
        # filter 0 (None)
        f0 = cur
        # filter 1 (Sub)
        shifted = np.empty_like(cur)
        shifted[:bpp] = 0
        shifted[bpp:] = cur[:-bpp]
        f1 = (cur.astype(np.int16) - shifted.astype(np.int16)).astype(np.uint8)
        # filter 2 (Up)
        f2 = (cur.astype(np.int16) - prev.astype(np.int16)).astype(np.uint8)

        best, bestscore, besttag = f0, None, 0
        for tag, cand in ((0, f0), (1, f1), (2, f2)):
            v = cand.astype(np.int32)
            score = int(np.minimum(v, 256 - v).sum())
            if bestscore is None or score < bestscore:
                best, bestscore, besttag = cand, score, tag
        out.append(besttag)
        out += best.tobytes()
        prev = cur

    ihdr = struct.pack(">IIBBBBB", w, h, 16, 2, 0, 0, 0)  # 16-bit, truecolour
    payload = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"sRGB", bytes([0]))
        + _png_chunk(b"IDAT", zlib.compress(bytes(out), 9))
        + _png_chunk(b"IEND", b"")
    )
    path.write_bytes(payload)


# --------------------------------------------------------------------------
# analytic SDF raster
# --------------------------------------------------------------------------


class Canvas:
    """Normalised square domain [0,1]^2. Pixel centre of (x,y) is
    ((x+0.5)/n, (y+0.5)/n) where n is the side length in pixels."""

    def __init__(self, n: int, aa: bool) -> None:
        self.n = n
        self.aa = aa
        self.px = 1.0 / n
        self.img = np.zeros((n, n, 3), dtype=np.float32)

    def fill(self, color) -> None:
        self.img[:] = np.asarray(color, dtype=np.float32)

    # -- coverage ---------------------------------------------------------
    def _coverage(self, sd: np.ndarray) -> np.ndarray:
        """sd: signed distance in normalised units, negative == inside."""
        if self.aa:
            # continuous fractional coverage -> genuinely continuous floats
            return np.clip(0.5 - sd / np.float32(self.px), 0.0, 1.0).astype(np.float32)
        return (sd <= 0.0).astype(np.float32)

    def _grid(self, box):
        x0, y0, x1, y1 = box
        xs = (np.arange(x0, x1, dtype=np.float32) + np.float32(0.5)) * np.float32(self.px)
        ys = (np.arange(y0, y1, dtype=np.float32) + np.float32(0.5)) * np.float32(self.px)
        return xs[None, :], ys[:, None]

    def _clipbox(self, xmin, ymin, xmax, ymax, pad):
        n = self.n
        x0 = max(0, int(math.floor((xmin - pad) * n)))
        y0 = max(0, int(math.floor((ymin - pad) * n)))
        x1 = min(n, int(math.ceil((xmax + pad) * n)) + 1)
        y1 = min(n, int(math.ceil((ymax + pad) * n)) + 1)
        if x1 <= x0 or y1 <= y0:
            return None
        return (x0, y0, x1, y1)

    def _composite(self, box, cov, color) -> None:
        x0, y0, x1, y1 = box
        sub = self.img[y0:y1, x0:x1, :]
        c = cov[:, :, None]
        col = np.asarray(color, dtype=np.float32)[None, None, :]
        sub *= 1.0 - c
        sub += col * c

    # -- primitives -------------------------------------------------------
    def segment(self, a, b, halfwidth, color) -> None:
        pad = halfwidth + 3.0 * self.px
        box = self._clipbox(min(a[0], b[0]), min(a[1], b[1]),
                            max(a[0], b[0]), max(a[1], b[1]), pad)
        if box is None:
            return
        U, V = self._grid(box)
        ax, ay = np.float32(a[0]), np.float32(a[1])
        bax, bay = np.float32(b[0] - a[0]), np.float32(b[1] - a[1])
        denom = np.float32(max(bax * bax + bay * bay, 1e-20))
        pax = U - ax
        pay = V - ay
        t = np.clip((pax * bax + pay * bay) / denom, 0.0, 1.0)
        dx = pax - bax * t
        dy = pay - bay * t
        d = np.sqrt(dx * dx + dy * dy)
        self._composite(box, self._coverage(d - np.float32(halfwidth)), color)

    def ring(self, c, r, halfwidth, color) -> None:
        pad = halfwidth + 3.0 * self.px
        box = self._clipbox(c[0] - r, c[1] - r, c[0] + r, c[1] + r, pad)
        if box is None:
            return
        U, V = self._grid(box)
        dx = U - np.float32(c[0])
        dy = V - np.float32(c[1])
        d = np.abs(np.sqrt(dx * dx + dy * dy) - np.float32(r))
        self._composite(box, self._coverage(d - np.float32(halfwidth)), color)

    def disc(self, c, r, color) -> None:
        box = self._clipbox(c[0] - r, c[1] - r, c[0] + r, c[1] + r, 3.0 * self.px)
        if box is None:
            return
        U, V = self._grid(box)
        dx = U - np.float32(c[0])
        dy = V - np.float32(c[1])
        d = np.sqrt(dx * dx + dy * dy) - np.float32(r)
        self._composite(box, self._coverage(d), color)

    def polygon(self, pts, color, halfwidth=None) -> None:
        """Filled when halfwidth is None, otherwise a stroked outline."""
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        pad = (halfwidth or 0.0) + 3.0 * self.px
        box = self._clipbox(min(xs), min(ys), max(xs), max(ys), pad)
        if box is None:
            return
        U, V = self._grid(box)
        U = np.broadcast_to(U, (V.shape[0], U.shape[1]))
        V = np.broadcast_to(V, (V.shape[0], U.shape[1]))
        m = len(pts)
        d2 = None
        sign = np.ones(U.shape, dtype=np.float32)
        for i in range(m):
            vi = pts[i]
            vj = pts[(i + 1) % m]
            ex, ey = np.float32(vj[0] - vi[0]), np.float32(vj[1] - vi[1])
            wx = U - np.float32(vi[0])
            wy = V - np.float32(vi[1])
            denom = np.float32(max(ex * ex + ey * ey, 1e-20))
            t = np.clip((wx * ex + wy * ey) / denom, 0.0, 1.0)
            bx = wx - ex * t
            by = wy - ey * t
            cand = bx * bx + by * by
            d2 = cand if d2 is None else np.minimum(d2, cand)
            c1 = V >= np.float32(vi[1])
            c2 = V < np.float32(vj[1])
            c3 = (ex * wy) > (ey * wx)
            flip = (c1 & c2 & c3) | (~c1 & ~c2 & ~c3)
            sign = np.where(flip, -sign, sign)
        dist = np.sqrt(d2)
        if halfwidth is None:
            sd = sign * dist
        else:
            sd = dist - np.float32(halfwidth)
        self._composite(box, self._coverage(sd), color)


# --------------------------------------------------------------------------
# palette helpers -- continuous float colours (never a 1/255 or 1/65535 lattice)
# --------------------------------------------------------------------------


def rand_color(rng: Rng, lo=0.02, hi=0.98) -> tuple:
    return (rng.uniform(lo, hi), rng.uniform(lo, hi), rng.uniform(lo, hi))


def rand_bg(rng: Rng) -> tuple:
    base = rng.uniform(0.80, 0.99)
    return (
        min(1.0, base + rng.uniform(-0.03, 0.03)),
        min(1.0, base + rng.uniform(-0.03, 0.03)),
        min(1.0, base + rng.uniform(-0.03, 0.03)),
    )


# --------------------------------------------------------------------------
# content categories -- geometry depends ONLY on the seed, never on n or aa
# --------------------------------------------------------------------------


def scene_lines(rng: Rng, cv: Canvas) -> None:
    cv.fill(rand_bg(rng))
    px = 1.0 / 1024.0  # widths quoted in "reference pixels" at 1024
    nring = rng.randint(2, 6)
    for _ in range(nring):
        c = (rng.uniform(0.15, 0.85), rng.uniform(0.15, 0.85))
        r0 = rng.uniform(0.04, 0.34)
        k = rng.randint(3, 9)
        col = rand_color(rng)
        for i in range(k):
            cv.ring(c, r0 + i * rng.uniform(0.006, 0.02),
                    0.5 * rng.uniform(0.7, 3.2) * px, col)
    nseg = rng.randint(28, 64)
    for _ in range(nseg):
        a = (rng.uniform(-0.05, 1.05), rng.uniform(-0.05, 1.05))
        ang = rng.uniform(0.0, 2.0 * math.pi)
        ln = rng.uniform(0.08, 0.9)
        b = (a[0] + ln * math.cos(ang), a[1] + ln * math.sin(ang))
        cv.segment(a, b, 0.5 * rng.uniform(0.7, 3.5) * px, rand_color(rng))


def scene_polygons(rng: Rng, cv: Canvas) -> None:
    cv.fill(rand_bg(rng))
    px = 1.0 / 1024.0
    npoly = rng.randint(7, 14)
    for _ in range(npoly):
        cx, cy = rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)
        k = rng.randint(3, 9)
        rr = rng.uniform(0.05, 0.28)
        star = rng.uniform(0.0, 1.0) < 0.45
        phase = rng.uniform(0.0, 2.0 * math.pi)
        pts = []
        for i in range(k * (2 if star else 1)):
            ang = phase + 2.0 * math.pi * i / (k * (2 if star else 1))
            r = rr * (0.42 if (star and i % 2) else 1.0) * rng.uniform(0.85, 1.15)
            pts.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
        cv.polygon(pts, rand_color(rng))
        if rng.uniform(0.0, 1.0) < 0.6:
            cv.polygon(pts, rand_color(rng), halfwidth=0.5 * rng.uniform(0.8, 3.0) * px)
    for _ in range(rng.randint(4, 12)):
        a = (rng.uniform(0.0, 1.0), rng.uniform(0.0, 1.0))
        ang = rng.uniform(0.0, 2.0 * math.pi)
        ln = rng.uniform(0.15, 0.7)
        cv.segment(a, (a[0] + ln * math.cos(ang), a[1] + ln * math.sin(ang)),
                   0.5 * rng.uniform(0.7, 2.4) * px, rand_color(rng))


def scene_line_patterns(rng: Rng, cv: Canvas) -> None:
    """A tiled motif -- the corpus's `line-tiling` shape."""
    cv.fill(rand_bg(rng))
    px = 1.0 / 1024.0
    m = rng.randint(4, 14)
    step = 1.0 / m
    nmotif = rng.randint(3, 7)
    motif = []
    for _ in range(nmotif):
        motif.append((
            (rng.uniform(0.02, 0.98), rng.uniform(0.02, 0.98)),
            (rng.uniform(0.02, 0.98), rng.uniform(0.02, 0.98)),
            0.5 * rng.uniform(0.7, 2.6) * px,
            rand_color(rng),
        ))
    rot = rng.uniform(0.0, 1.0) < 0.5
    for ty in range(m):
        for tx in range(m):
            th = (0.37 * (tx + 2 * ty)) if rot else 0.0
            ca, sa = math.cos(th), math.sin(th)
            ox, oy = tx * step, ty * step
            for (a, b, hw, col) in motif:
                def xf(p):
                    u, v = p[0] - 0.5, p[1] - 0.5
                    return (ox + (0.5 + u * ca - v * sa) * step,
                            oy + (0.5 + u * sa + v * ca) * step)
                cv.segment(xf(a), xf(b), hw, col)


def scene_grids(rng: Rng, cv: Canvas) -> None:
    """Chart-like: major/minor rules, ticks, filled bars, polylines."""
    cv.fill(rand_bg(rng))
    px = 1.0 / 1024.0
    l, r = 0.10, 0.96
    t, bt = 0.06, 0.90
    axis = rand_color(rng, 0.02, 0.35)
    minor = rand_color(rng, 0.55, 0.85)
    nmaj = rng.randint(4, 11)
    nmin = rng.randint(2, 6)
    for i in range(nmaj + 1):
        y = t + (bt - t) * i / nmaj
        cv.segment((l, y), (r, y), 0.5 * rng.uniform(0.9, 1.9) * px, axis)
        if i < nmaj:
            for j in range(1, nmin):
                yy = y + (bt - t) / nmaj * j / nmin
                cv.segment((l, yy), (r, yy), 0.5 * rng.uniform(0.6, 1.1) * px, minor)
    ncol = rng.randint(5, 16)
    for i in range(ncol + 1):
        x = l + (r - l) * i / ncol
        cv.segment((x, t), (x, bt), 0.5 * rng.uniform(0.7, 1.6) * px, minor)
        cv.segment((x, bt), (x, bt + 0.014), 0.5 * rng.uniform(1.0, 2.2) * px, axis)
    cv.segment((l, bt), (r, bt), 0.5 * rng.uniform(1.6, 3.2) * px, axis)
    cv.segment((l, t), (l, bt), 0.5 * rng.uniform(1.6, 3.2) * px, axis)
    if rng.uniform(0.0, 1.0) < 0.5:
        w = (r - l) / ncol * rng.uniform(0.45, 0.8)
        for i in range(ncol):
            x = l + (r - l) * (i + 0.5) / ncol
            hgt = rng.uniform(0.05, 0.78) * (bt - t)
            cv.polygon([(x - w / 2, bt - hgt), (x + w / 2, bt - hgt),
                        (x + w / 2, bt), (x - w / 2, bt)], rand_color(rng))
    for _ in range(rng.randint(1, 4)):
        col = rand_color(rng, 0.05, 0.7)
        hw = 0.5 * rng.uniform(1.2, 3.4) * px
        k = rng.randint(6, 40)
        prev = None
        for i in range(k + 1):
            x = l + (r - l) * i / k
            y = t + (bt - t) * rng.uniform(0.05, 0.95)
            if prev is not None:
                cv.segment(prev, (x, y), hw, col)
            prev = (x, y)


def scene_gradients_smooth(rng: Rng, cv: Canvas) -> None:
    """Smooth continuous ramps -- the case where 8-bit bands and float is
    genuinely needed. No edges, so this category has no aliased arm."""
    n = cv.n
    xs = (np.arange(n, dtype=np.float32) + np.float32(0.5)) / np.float32(n)
    U = np.broadcast_to(xs[None, :], (n, n))
    V = np.broadcast_to(xs[:, None], (n, n))
    out = np.empty((n, n, 3), dtype=np.float32)
    for ch in range(3):
        base = np.float32(rng.uniform(0.15, 0.55))
        ang = rng.uniform(0.0, 2.0 * math.pi)
        gx, gy = np.float32(math.cos(ang)), np.float32(math.sin(ang))
        acc = base + np.float32(rng.uniform(0.1, 0.4)) * (U * gx + V * gy)
        cx, cy = np.float32(rng.uniform(0.2, 0.8)), np.float32(rng.uniform(0.2, 0.8))
        rad = np.sqrt((U - cx) ** 2 + (V - cy) ** 2)
        acc = acc + np.float32(rng.uniform(0.05, 0.30)) * np.exp(
            -rad * np.float32(rng.uniform(1.2, 5.0)))
        for _ in range(rng.randint(2, 4)):
            fu = np.float32(rng.uniform(0.7, 4.5))
            fv = np.float32(rng.uniform(0.7, 4.5))
            ph = np.float32(rng.uniform(0.0, 6.283185307179586))
            amp = np.float32(rng.uniform(0.01, 0.09))
            acc = acc + amp * np.sin(fu * U * np.float32(6.283185307179586)
                                     + fv * V * np.float32(6.283185307179586) + ph)
        out[:, :, ch] = acc
    np.clip(out, 0.0, 1.0, out=out)
    cv.img[:] = out


SCENES = {
    "lines": scene_lines,
    "polygons": scene_polygons,
    "line-patterns": scene_line_patterns,
    "grids": scene_grids,
    "gradients-smooth": scene_gradients_smooth,
}
# categories that get an aliased twin at the same seed
PAIRED = ("lines", "polygons", "line-patterns", "grids")


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def render(kind: str, seed: int, n: int, aa: bool) -> np.ndarray:
    cv = Canvas(n, aa)
    SCENES[kind](Rng(seed), cv)
    np.clip(cv.img, 0.0, 1.0, out=cv.img)
    return cv.img


def to_u16(img: np.ndarray) -> np.ndarray:
    return np.rint(img.astype(np.float64) * 65535.0).astype(np.uint16)


def selftest(tmpdir: Path) -> None:
    """PFM write -> read must be bit-exact, for both endiannesses."""
    tmpdir.mkdir(parents=True, exist_ok=True)
    rng = Rng(0xDEADBEEF)
    for (h, w) in ((1, 1), (3, 5), (17, 4), (64, 64)):
        a = np.array([[[rng.uniform(-4.0, 4.0) for _ in range(3)]
                       for _ in range(w)] for _ in range(h)], dtype=np.float32)
        p = tmpdir / f"st_{h}x{w}.pfm"
        write_pfm(p, a)
        b = read_pfm(p)
        if b.shape != a.shape or not np.array_equal(a.view(np.uint32), b.view(np.uint32)):
            raise SystemExit(f"PFM self-test FAILED at {h}x{w}")
        # big-endian variant (what djxl emits): same pixels, scale +1.0
        pbe = tmpdir / f"st_{h}x{w}_be.pfm"
        with open(pbe, "wb") as f:
            f.write(b"PF\n"); f.write(f"{w} {h}\n".encode()); f.write(b"1.0\n")
            f.write(np.ascontiguousarray(a[::-1], dtype=">f4").tobytes())
        c = read_pfm(pbe)
        if not np.array_equal(a.view(np.uint32), c.view(np.uint32)):
            raise SystemExit(f"PFM big-endian self-test FAILED at {h}x{w}")
    print("PFM self-test OK (little-endian write/read + big-endian read), 4 shapes")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path.home() / "work/zen/imazen26-float-synth"))
    ap.add_argument("--sizes", default="64,256,1024,4096")
    ap.add_argument("--seeds-small", type=int, default=3,
                    help="seeds per category at sizes < 4096")
    ap.add_argument("--seeds-large", type=int, default=2,
                    help="seeds per category at 4096 (disk-bounded)")
    ap.add_argument("--master-seed", type=lambda s: int(s, 0), default=0x7000_F10A7)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--selftest-only", action="store_true")
    ap.add_argument("--tmp", default=str(Path.home() / "tmp/floatsynth"))
    args = ap.parse_args()

    tmp = Path(args.tmp)
    selftest(tmp)
    if args.selftest_only:
        return 0

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    gen_path = Path(__file__).resolve()
    gen_sha = hashlib.sha256(gen_path.read_bytes()).hexdigest()
    try:
        repo_commit = subprocess.run(
            ["git", "-C", str(gen_path.parent), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        repo_commit = "unknown"
    generator = "scripts/gen_float_lineart_corpus.py"
    generator_commit = f"uncommitted@{repo_commit}+sha256:{gen_sha[:16]}"

    sizes = [int(s) for s in args.sizes.split(",")]
    rows = []
    seq = 0
    for n in sizes:
        nseeds = args.seeds_large if n >= 4096 else args.seeds_small
        for kind in SCENES:
            arms = [("anti-aliased", kind, True)]
            if kind in PAIRED:
                arms.append(("aliased", f"aliased-{kind}", False))
            for idx in range(nseeds):
                # seed depends on (kind, idx) ONLY -- identical geometry across
                # sizes and across the aliased/AA arms.
                seed = Rng(args.master_seed
                           ^ (hash_kind(kind) << 8) ^ (idx * 0x9E3779B1)).next_u64() & 0xFFFFFFFF
                desc = f"{kind}-{idx:05d}-s{seed:08x}"
                for (arm, cat, aa) in arms:
                    img = render(kind, seed, n, aa)
                    (out / cat).mkdir(parents=True, exist_ok=True)
                    seq += 1
                    stem = f"{seq:04d}_floatsynth_{desc}_{n}x{n}"
                    ppfm = out / cat / f"{stem}.pfm"
                    ppng = out / cat / f"{stem}.png"
                    write_pfm(ppfm, img)
                    write_png16(ppng, to_u16(img))
                    for (p, fmt) in ((ppng, "png16"), (ppfm, "pfm32")):
                        rows.append([
                            f"{cat}/{p.name}", str(seq), cat, desc, str(n), str(n),
                            fmt, str(p.stat().st_size), sha256_file(p),
                            f"0x{seed:08x}", arm, generator, generator_commit,
                        ])
                    print(f"  {cat}/{stem}  ({arm})", flush=True)

    hdr = ["filename", "number", "category", "descriptor", "width", "height",
           "format", "bytes", "sha256", "seed", "arm", "generator",
           "generator_commit"]
    with open(out / "MANIFEST.tsv", "w") as f:
        f.write("\t".join(hdr) + "\n")
        for r in rows:
            f.write("\t".join(r) + "\n")
    print(f"\nwrote {len(rows)} rows to {out/'MANIFEST.tsv'}")
    return 0


def hash_kind(kind: str) -> int:
    """Stable (non-PYTHONHASHSEED-dependent) small integer per category."""
    return int.from_bytes(hashlib.sha256(kind.encode()).digest()[:4], "big")


if __name__ == "__main__":
    sys.exit(main())
