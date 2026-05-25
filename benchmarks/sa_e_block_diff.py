#!/usr/bin/env python3
"""Identify pixel-divergent blocks at 8x8 and 64x64 granularity."""
import numpy as np

def load_pfm(path):
    with open(path, 'rb') as f:
        magic = f.readline().rstrip()
        w, h = (int(x) for x in f.readline().split())
        scale = float(f.readline())
        data = np.frombuffer(f.read(), dtype='<f4' if scale < 0 else '>f4')
        return data.reshape(h, w, 3)[::-1, :, :].copy().astype(np.float32)

ours = load_pfm('/tmp/sa_e/ours_linear.pfm')
cjxl = load_pfm('/tmp/sa_e/cjxl_linear.pfm')

# 8x8 block max-diff in linear sRGB
diff = np.abs(ours - cjxl)
# Reshape to 128x8x128x8x3
H, W = 1024, 1024
b8 = diff.reshape(H//8, 8, W//8, 8, 3).max(axis=(1, 3, 4))  # (128, 128)
print(f"8x8 block max-diff stats: max={b8.max():.4f}, mean={b8.mean():.4f}, median={np.median(b8):.4f}")
print(f"Blocks with max-diff > 0.05: {(b8 > 0.05).sum()} / {b8.size}")
print(f"Blocks with max-diff > 0.10: {(b8 > 0.10).sum()} / {b8.size}")
print(f"Blocks with max-diff > 0.20: {(b8 > 0.20).sum()} / {b8.size}")

# Top-20 worst 8x8 blocks
flat = b8.flatten()
idx = np.argsort(flat)[::-1][:20]
print("\nTop-20 worst 8x8 blocks (by, bx, max_diff):")
for i in idx:
    by, bx = i // 128, i % 128
    print(f"  by={by:3d} bx={bx:3d}  max_diff={flat[i]:.4f}  pixel y={by*8}..{by*8+8} x={bx*8}..{bx*8+8}")

# 64x64 (DC-block) granularity
b64 = diff.reshape(16, 64, 16, 64, 3).max(axis=(1, 3, 4))
print(f"\n64x64 block max-diff stats: max={b64.max():.4f}, mean={b64.mean():.4f}")
# Find biggest 64x64 hotspots
for tier in [0.30, 0.20, 0.10]:
    hits = np.argwhere(b64 > tier)
    print(f"  64x64 blocks > {tier}: {len(hits)}")
    for hy, hx in hits[:5]:
        print(f"    DC y={hy} x={hx}  pixel y={hy*64}..{hy*64+64} x={hx*64}..{hx*64+64}  max_diff={b64[hy,hx]:.4f}")

# Mean per channel in different regions (R, G, B):
print("\nPer-region bias (ours - cjxl) for sanity (no sign-flip expected — look for systematic bias):")
for name, (y0, y1, x0, x1) in [
    ("full", (0, 1024, 0, 1024)),
    ("top-right (>SSIM2 hit zone)", (0, 320, 640, 1024)),
    ("top-left", (0, 320, 0, 384)),
    ("middle", (300, 700, 300, 700)),
    ("middle-cols-480-608-rows-256-288 (worst row band)", (256, 288, 480, 608)),
]:
    bias = (ours[y0:y1, x0:x1, :] - cjxl[y0:y1, x0:x1, :]).mean(axis=(0,1))
    rms = np.sqrt(((ours[y0:y1, x0:x1, :] - cjxl[y0:y1, x0:x1, :])**2).mean(axis=(0,1)))
    print(f"  {name}: bias R/G/B = {bias.round(5)}, RMS R/G/B = {rms.round(5)}")
