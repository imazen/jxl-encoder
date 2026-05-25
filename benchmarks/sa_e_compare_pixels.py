#!/usr/bin/env python3
"""SA-E pixel-divergence comparison: ours vs cjxl decoded outputs at multiple pipeline stages."""
import numpy as np
import struct
import sys
from pathlib import Path

def load_pfm(path):
    """Load 3-channel PFM. Returns f32 array shape (H, W, 3) in linear sRGB."""
    with open(path, 'rb') as f:
        magic = f.readline().rstrip()
        assert magic == b'PF', f"expected PF, got {magic!r}"
        w, h = (int(x) for x in f.readline().split())
        scale = float(f.readline())
        # negative scale = little-endian
        data = np.frombuffer(f.read(), dtype='<f4' if scale < 0 else '>f4')
        # PFM stores rows bottom-up
        img = data.reshape(h, w, 3)[::-1, :, :]
        return img.copy()

def load_png_u8(path):
    """Load 8-bit RGB PNG."""
    from PIL import Image
    im = Image.open(path).convert('RGB')
    return np.asarray(im, dtype=np.uint8)

def load_npy_jxlrs(path):
    """jxl-rs --data-type f32 npy. Standard NumPy format."""
    return np.load(path)

def region_stats(arr_a, arr_b, name, region=None):
    """Per-channel L2 and L∞ in a region."""
    if region is not None:
        y0, y1, x0, x1 = region
        a = arr_a[y0:y1, x0:x1]
        b = arr_b[y0:y1, x0:x1]
    else:
        a, b = arr_a, arr_b
    diff = (a.astype(np.float64) - b.astype(np.float64))
    # Per-channel
    ch_l2 = [float(np.sqrt(np.mean(diff[..., c]**2))) for c in range(diff.shape[-1])]
    ch_max = [float(np.max(np.abs(diff[..., c]))) for c in range(diff.shape[-1])]
    ch_mean = [float(np.mean(diff[..., c])) for c in range(diff.shape[-1])]
    print(f"  {name:24s}: shape={a.shape}, dtype={a.dtype}")
    print(f"    L2  per ch [R,G,B]: {ch_l2}")
    print(f"    Lmax per ch [R,G,B]: {ch_max}")
    print(f"    bias per ch [R,G,B]: {ch_mean}")

def main():
    print("=== SA-E DECODE PIPELINE PIXEL DIVERGENCE ===")
    print("Cell: clic_22ea12 e9 d=4")
    print()

    # PFM = linear sRGB f32 from djxl
    print("Loading PFM (linear sRGB f32) from djxl...")
    ours_lin = load_pfm('/tmp/sa_e/ours_linear.pfm')
    cjxl_lin = load_pfm('/tmp/sa_e/cjxl_linear.pfm')
    print(f"  ours_lin shape: {ours_lin.shape}, dtype: {ours_lin.dtype}, range: [{ours_lin.min():.4f}, {ours_lin.max():.4f}]")
    print(f"  cjxl_lin shape: {cjxl_lin.shape}, dtype: {cjxl_lin.dtype}, range: [{cjxl_lin.min():.4f}, {cjxl_lin.max():.4f}]")
    print()

    # PNG = sRGB-encoded u8 from djxl
    print("Loading PNG (sRGB u8) from djxl...")
    ours_srgb = load_png_u8('/tmp/sa_e/ours_srgb.png')
    cjxl_srgb = load_png_u8('/tmp/sa_e/cjxl_srgb.png')
    print(f"  ours_srgb shape: {ours_srgb.shape}, dtype: {ours_srgb.dtype}")
    print()

    # NPY = jxl-rs default (likely gamma f32 in 0..1)
    print("Loading NPY (jxl-rs f32) ...")
    ours_jxlrs = load_npy_jxlrs('/tmp/sa_e/ours_jxlrs.npy')
    cjxl_jxlrs = load_npy_jxlrs('/tmp/sa_e/cjxl_jxlrs.npy')
    print(f"  ours_jxlrs shape: {ours_jxlrs.shape}, dtype: {ours_jxlrs.dtype}, range: [{ours_jxlrs.min():.4f}, {ours_jxlrs.max():.4f}]")
    print(f"  cjxl_jxlrs shape: {cjxl_jxlrs.shape}, dtype: {cjxl_jxlrs.dtype}, range: [{cjxl_jxlrs.min():.4f}, {cjxl_jxlrs.max():.4f}]")
    print()

    # Decoder-to-decoder consistency check first:
    print("=" * 60)
    print("DECODER-TO-DECODER consistency check (ours.jxl through djxl vs jxl-rs)")
    print("=" * 60)
    # If jxl-rs npy is gamma sRGB f32 in [0,1] and djxl PNG is sRGB u8, compare PNG/255 to jxl-rs npy
    if ours_jxlrs.ndim == 3 and ours_jxlrs.shape == ours_srgb.shape:
        region_stats(ours_jxlrs * 255.0, ours_srgb.astype(np.float32), "ours: jxl-rs vs djxl sRGB")
        region_stats(cjxl_jxlrs * 255.0, cjxl_srgb.astype(np.float32), "cjxl: jxl-rs vs djxl sRGB")
    print()

    print("=" * 60)
    print("OURS-vs-CJXL ABSOLUTE DIVERGENCE")
    print("=" * 60)

    # full-image
    print("Full image (1024x1024):")
    region_stats(ours_lin, cjxl_lin, "PFM linear sRGB f32")
    region_stats(ours_srgb.astype(np.float32), cjxl_srgb.astype(np.float32), "PNG sRGB u8 (raw)")
    region_stats(ours_jxlrs, cjxl_jxlrs, "NPY jxl-rs f32")
    print()

    # Top-right region: bx∈[80,128), by∈[0,40) at DC-block (8x8) granularity = pixel [0,320), [640,1024)
    tr = (0, 320, 640, 1024)
    print(f"Top-right region (pixel y=[{tr[0]},{tr[1]}), x=[{tr[2]},{tr[3]})):")
    region_stats(ours_lin, cjxl_lin, "PFM linear sRGB f32 (TR)", region=tr)
    region_stats(ours_srgb.astype(np.float32), cjxl_srgb.astype(np.float32), "PNG sRGB u8 (TR)", region=tr)
    print()

    # Other regions for context
    tl = (0, 320, 0, 384)
    print(f"Top-left region (pixel y=[{tl[0]},{tl[1]}), x=[{tl[2]},{tl[3]})):")
    region_stats(ours_lin, cjxl_lin, "PFM linear sRGB f32 (TL)", region=tl)
    print()

    bl = (640, 1024, 0, 384)
    print(f"Bottom-left region (pixel y=[{bl[0]},{bl[1]}), x=[{bl[2]},{bl[3]})):")
    region_stats(ours_lin, cjxl_lin, "PFM linear sRGB f32 (BL)", region=bl)
    print()

    # Per-row max-diff distribution to see if certain rows dominate
    print("=" * 60)
    print("Per-row L∞ DIFF distribution (linear sRGB):")
    print("=" * 60)
    diff = np.abs(ours_lin - cjxl_lin)  # (1024, 1024, 3)
    row_max = np.max(diff, axis=(1, 2))  # (1024,)
    # 32-row buckets
    buckets = row_max.reshape(32, 32).max(axis=1)
    for i, v in enumerate(buckets):
        bar = '#' * int(v * 200)
        print(f"  rows [{i*32:4d},{i*32+32:4d}): max={v:.5f}  {bar}")
    print()

    # Per-column max-diff distribution
    print("Per-32px column-band L∞ DIFF (linear sRGB):")
    col_max = np.max(diff, axis=(0, 2))  # (1024,)
    cbuckets = col_max.reshape(32, 32).max(axis=1)
    for i, v in enumerate(cbuckets):
        bar = '#' * int(v * 200)
        print(f"  cols [{i*32:4d},{i*32+32:4d}): max={v:.5f}  {bar}")
    print()

    # Where is the max diff in linear sRGB?
    idx = np.unravel_index(np.argmax(diff), diff.shape)
    print(f"Argmax of |ours_lin - cjxl_lin|: pixel y={idx[0]}, x={idx[1]}, ch={idx[2]}")
    print(f"  ours: {ours_lin[idx[0], idx[1]]}")
    print(f"  cjxl: {cjxl_lin[idx[0], idx[1]]}")
    print()

    # Sample patch at the W44-AUDIT-8 Phase 4 top-right divergence corner:
    print("Sample 8x8 pixel patch at top-right corner (y=0..8, x=1016..1024):")
    print("  OURS linear sRGB:")
    print(ours_lin[0:8, 1016:1024, :].round(4))
    print("  CJXL linear sRGB:")
    print(cjxl_lin[0:8, 1016:1024, :].round(4))
    print()

    # Mid-top-right (y=160, x=720) — middle of dSsim2=-9.68 region
    y0, x0 = 160, 720
    print(f"Sample 8x8 patch at top-right middle (y={y0}..{y0+8}, x={x0}..{x0+8}):")
    print("  OURS linear sRGB:")
    print(ours_lin[y0:y0+8, x0:x0+8, :].round(4))
    print("  CJXL linear sRGB:")
    print(cjxl_lin[y0:y0+8, x0:x0+8, :].round(4))
    print("  DIFF:")
    print((ours_lin[y0:y0+8, x0:x0+8, :] - cjxl_lin[y0:y0+8, x0:x0+8, :]).round(5))


if __name__ == '__main__':
    main()
