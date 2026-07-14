#!/usr/bin/env python3
"""Lossless-graphics gap probe vs cjxl (issue #74 "win or match always").

Records where our default lossless modular encoder (e7) stands vs cjxl on
graphics/document content, correlated with background dominance (top-color
fraction). Reproducible against the SHIPPED encoder — no env hooks.

Two robustness points this harness exists to enforce:
  1. cjxl v0.11.2 FAILS to read some Display-P3 / Android-EXIF PNGs
     ("Getting pixel data failed") and produces NO output. A naive sweep
     that stat()s a reused temp path silently compares our size against the
     PREVIOUS image's cjxl output (observed: an 8025 mobile screenshot
     "cjxl=186745" that was actually 8248's size -> a bogus +265% "loss").
     Here a cjxl failure is recorded as cjxl_fail, never compared.
  2. The gap is HETEROGENEOUS and correlates with background dominance:
     high top-color% (sparse docs) is where we tend to lose; low top-color%
     and low-unique-color (palette-friendly) is where we win big.

Columns: image, class, w, h, unique_colors, top_color_pct, cjxl_bytes,
         ours_bytes, ours_vs_cjxl_pct

Provenance: companion .meta file.
"""
import subprocess, os
from PIL import Image

CLI = "/home/lilith/work/zen/jxl-encoder/target/release/cjxl-rs"
CJXL = "/home/lilith/work/libjxl-build/tools/cjxl"
TMP = os.environ.get("PROBE_TMP", "/tmp")
CORPUS = "/home/lilith/work/codec-corpus/imazen-26"

# Stratified: NOAA documents (sparse, background-dominant), web screenshots
# (mixed), CLIC photos (dense control). Extend freely.
IMAGES = [
    ("doc", "5300-noaa-hurricane-documents/5334_noaa_nhc-al132023-lee_p21_2550x3300.png"),
    ("doc", "5300-noaa-hurricane-documents/5324_noaa_nhc-al092022-ian_p25_2550x3300.png"),
    ("doc", "5300-noaa-hurricane-documents/5336_noaa_nhc-al132024-leslie_p05_2550x3300.png"),
    ("doc", "5300-noaa-hurricane-documents/5340_noaa_nhc-al172022-nicole_p40_2550x3300.png"),
    ("doc", "5300-noaa-hurricane-documents/5342_noaa_nhc-al182024-rafael_p10_2550x3300.png"),
    ("screenshot", "8100-lilith-web-screenshots/2880x1800/8222_web-screenshots_bls-employment-situation_dpr2_page1_2880x1800.png"),
    ("screenshot", "8100-lilith-web-screenshots/other/8432_web-screenshots_nih-recipes-keep-the-beat_dpr1_page1_1280x800.png"),
    ("screenshot", "8100-lilith-web-screenshots/1920x1080/8174_web-screenshots_loc-free-to-use_dpr1_page1_1920x1080.png"),
    ("photo", "../clic2025-1024/11f2b039b293758398b1a7a8afa64bb2.png"),
    ("photo", "../clic2025-1024/2a2420f929cd47122eae364a2bc27710.png"),
]

def enc(cmd, dst):
    if os.path.exists(dst):
        os.remove(dst)
    subprocess.run(cmd, capture_output=True)
    return os.path.getsize(dst) if os.path.exists(dst) else None

print("image\tclass\tw\th\tunique_colors\ttop_color_pct\tcjxl_bytes\tours_bytes\tours_vs_cjxl_pct")
for cls, rel in IMAGES:
    src = os.path.join(CORPUS, rel)
    name = os.path.basename(src)
    im = Image.open(src).convert("RGB")
    w, h = im.size
    cols = im.getcolors(maxcolors=2_000_000)
    nc = len(cols) if cols else -1
    top = (100 * max(cols)[0] / (w * h)) if cols else -1.0
    cj = enc([CJXL, src, f"{TMP}/probe_c.jxl", "-d", "0", "-e", "7"], f"{TMP}/probe_c.jxl")
    ours = enc([CLI, src, f"{TMP}/probe_o.jxl", "--lossless", "-e", "7"], f"{TMP}/probe_o.jxl")
    if cj and ours:
        vs = f"{100*(ours-cj)/cj:+.1f}"
        cj_s = str(cj)
    else:
        vs = "cjxl_fail" if not cj else "ours_fail"
        cj_s = "FAIL" if not cj else str(cj)
    print(f"{name}\t{cls}\t{w}\t{h}\t{nc}\t{top:.1f}\t{cj_s}\t{ours}\t{vs}", flush=True)
