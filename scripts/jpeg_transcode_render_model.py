#!/usr/bin/env python3
"""Attribute the gap between a JPEG XL render of a recompressed JPEG and the
JPEG's own decode.

Renders the JPEG's quantized coefficients (from `examples/jpeg_coeff_dump`)
under several dequantization models with a float IDCT, and compares each to
(a) a libjpeg-style decode of the JPEG (raw interleaved RGB8) and (b) optional
JPEG XL renders (raw RGB8, e.g. djxl output converted with
`magick x.ppm -depth 8 rgb:x.rgb8`).

Models:
  jpeg-float      exact dequant (q*Q), float IDCT, no sample clamp
  jpeg-clamp      as above, Y/Cb/Cr samples rounded+clamped to [0,255]
                  before YCbCr->RGB (what libjpeg-style decoders do)
  jxl-nocfl       JPEG XL decode semantics without chroma-from-luma:
                  default AC quant bias (AdjustQuantBias with the
                  kDefaultQuantBias values, which a JPEG frame cannot
                  override because opsin_inverse_matrix is only signalled
                  when xyb_encoded), float IDCT, no sample clamp; for 4:2:0
                  libjxl's 3/4,1/4 chroma upsampling on floats
  jxl-cfl         jxl-nocfl plus floating-point chroma-from-luma using the
                  CfL map libjxl's JPEG search selects (4:4:4 only; ported
                  from enc_frame.cc / our jpeg_cfl_search)

Usage:
  jpeg_transcode_render_model.py COEF JPEG_RGB8 [--render-cfl RGB8]
                                 [--render-nocfl RGB8] [--label NAME]
Prints one TSV row per model: label, model, compared-to, max, mean, frac>=3.
"""
import argparse
import struct

import numpy as np

BIAS = {0: 1 - 0.05465, 1: 1 - 0.07005, 2: 1 - 0.049935}  # kDefaultQuantBias
BIAS_NUM = 0.145
JXL_CHANNEL = [1, 0, 2]  # JPEG component (Y, Cb, Cr) -> JXL channel


def load(path):
    b = open(path, "rb").read()
    o = 0

    def u32():
        nonlocal o
        v = struct.unpack_from("<I", b, o)[0]
        o += 4
        return v

    w, h, n = u32(), u32(), u32()
    comps = [dict(zip(["id", "h", "v", "q", "wb", "hb"], [u32() for _ in range(6)])) for _ in range(n)]
    tables = {}
    for _ in range(u32()):
        idx = u32()
        tables[idx] = np.array(struct.unpack_from("<64i", b, o), dtype=np.int64)
        o += 256
    for c in comps:
        count = c["wb"] * c["hb"] * 64
        c["coef"] = np.frombuffer(b, dtype="<i2", count=count, offset=o).astype(np.int64)
        c["coef"] = c["coef"].reshape(c["hb"], c["wb"], 64)
        o += count * 2
    return w, h, comps, tables


_u = np.arange(8)
_C = np.where(_u == 0, 1 / np.sqrt(2), 1.0)
IDCT = (_C[None, :] / 2) * np.cos((2 * _u[:, None] + 1) * _u[None, :] * np.pi / 16)


def plane(F, hb, wb, w, h):
    px = np.einsum("yv,...vu,xu->...yx", IDCT, F.reshape(hb, wb, 8, 8), IDCT)
    return px.transpose(0, 2, 1, 3).reshape(hb * 8, wb * 8)[:h, :w] + 128.0


def biased(q, jxl_c):
    out = q.astype(np.float64).copy()
    a = np.abs(q)
    out[a == 1] = q[a == 1] * BIAS[jxl_c]
    m = a >= 2
    out[m] = q[m] - BIAS_NUM / q[m]
    out[..., 0] = q[..., 0]  # DC is not biased
    return out


def up2(p, axis):
    p = np.moveaxis(p, axis, 0)
    prev = np.concatenate([p[:1], p[:-1]])
    nxt = np.concatenate([p[1:], p[-1:]])
    out = np.empty((p.shape[0] * 2,) + p.shape[1:])
    out[0::2] = 0.25 * prev + 0.75 * p
    out[1::2] = 0.75 * p + 0.25 * nxt
    return np.moveaxis(out, 0, axis)


def to_rgb(Y, Cb, Cr, clamp):
    if clamp:
        Y, Cb, Cr = [np.clip(np.round(p), 0, 255) for p in (Y, Cb, Cr)]
    R = Y + 1.402 * (Cr - 128)
    G = Y - 0.344136 * (Cb - 128) - 0.714136 * (Cr - 128)
    B = Y + 1.772 * (Cb - 128)
    return np.clip(np.round(np.stack([R, G, B], -1)), 0, 255).astype(np.int64)


def cfl_search(Yq, Cq, sq, zero_bias=0.5):
    """libjxl JPEG CfL search (FindAvgIndexOfSumMaximum), natural order."""
    hb, wb, _ = Yq.shape
    th, tw = -(-hb // 8), -(-wb // 8)
    k_scale = np.float32(84.0)
    k_zero = np.float32(k_scale * np.float32(zero_bias) * np.float32(0.9999))
    inv_fp = np.float32(1.0 / 2048)
    out = np.zeros((th, tw), np.int64)
    for ty in range(th):
        for tx in range(tw):
            y = Yq[ty * 8:(ty + 1) * 8, tx * 8:(tx + 1) * 8, 1:].reshape(-1, 63).astype(np.float32)
            c = Cq[ty * 8:(ty + 1) * 8, tx * 8:(tx + 1) * 8, 1:].reshape(-1, 63).astype(np.float32)
            m = (y * sq[1:].astype(np.float32)) * inv_fp
            s = k_scale * c + np.float32(127.0) * m
            ok = np.abs(m) > np.float32(1e-8)
            m, s = m[ok], s[ok]
            a, b = (s - k_zero) / m, (s + k_zero) / m
            lo_f = np.maximum(np.where(m > 0, a, b), np.float32(0))
            hi_f = np.minimum(np.where(m > 0, b, a), np.float32(255))
            v = lo_f <= hi_f
            lo = np.ceil(lo_f[v]).astype(np.int64)
            hi = np.floor(hi_f[v] + np.float32(1)).astype(np.int64)
            d = np.zeros(258, np.int64)
            np.add.at(d, lo[(lo >= 0) & (lo <= 256)], 1)
            np.add.at(d, hi[(hi >= 0) & (hi <= 256)], -1)
            run = np.cumsum(d[:256])
            best = run.max()
            if best > 0 and best > run[127] + 1:
                idx = np.nonzero(run == best)[0]
                out[ty, tx] = ((idx[0] + idx[-1] + 1) >> 1) - 127
    return out


def cfl_residual(Yq, Cq, sq, cmap):
    hb, wb, _ = Yq.shape
    v = np.repeat(np.repeat(cmap, 8, 0), 8, 1)[:hb, :wb]
    num = v * 2048
    scale = np.trunc(num / 84).astype(np.int64)  # C++ integer division
    coeff_scale = (scale[..., None] * sq[None, None, :] + 1024) >> 11
    R = Cq - ((Yq * coeff_scale + 1024) >> 11)
    R[..., 0] = Cq[..., 0]
    return R, v


def render(w, h, comps, tables, model):
    subsampled = comps[0]["h"] == 2
    if model == "jxl-cfl" and subsampled:
        return None
    planes = []
    Y = comps[0]
    Ydeq = (biased(Y["coef"], 1) if model.startswith("jxl") else Y["coef"]) * tables[Y["q"]]
    planes.append(plane(Ydeq, Y["hb"], Y["wb"], w, h))
    for i in (1, 2):
        c = comps[i]
        Q = tables[c["q"]]
        if model == "jxl-cfl":
            QY = tables[Y["q"]]
            sq = (2048 * QY) // Q
            R, v = cfl_residual(Y["coef"], c["coef"], sq, cfl_search(Y["coef"], c["coef"], sq))
            F = biased(R, JXL_CHANNEL[i]) * Q
            F[..., 1:] += (v.astype(np.float32) / np.float32(84.0))[..., None] * Ydeq[..., 1:]
        elif model.startswith("jxl"):
            F = biased(c["coef"], JXL_CHANNEL[i]) * Q
        else:
            F = c["coef"] * Q
        if subsampled:
            p = plane(F, c["hb"], c["wb"], -(-w // 2), -(-h // 2))
            planes.append(up2(up2(p, 0), 1)[:h, :w])
        else:
            planes.append(plane(F, c["hb"], c["wb"], w, h))
    return to_rgb(*planes, clamp=(model == "jpeg-clamp"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("coef")
    ap.add_argument("jpeg_rgb8")
    ap.add_argument("--render-cfl")
    ap.add_argument("--render-nocfl")
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    w, h, comps, tables = load(args.coef)
    refs = {"jpeg-decode": args.jpeg_rgb8, "jxl-render": args.render_cfl,
            "jxl-render-nocfl": args.render_nocfl}
    refs = {k: np.fromfile(v, dtype=np.uint8).reshape(h, w, 3).astype(np.int64)
            for k, v in refs.items() if v}
    print("label\tmodel\tcompared_to\tmax\tmean\tfrac_ge3")
    for model in ("jpeg-float", "jpeg-clamp", "jxl-nocfl", "jxl-cfl"):
        r = render(w, h, comps, tables, model)
        if r is None:
            continue
        for name, ref in refs.items():
            d = np.abs(r - ref)
            print(f"{args.label}\t{model}\t{name}\t{d.max()}\t{d.mean():.4f}\t{np.mean(d >= 3):.4f}")


if __name__ == "__main__":
    main()
