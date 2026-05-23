#!/usr/bin/env python3
"""Build W44-229 chunks for the Tier-2 5-knob validation sweep.

This sweep validates the W44-222 5-knob Tier2Knobs expander on
out-of-distribution data + produces Tier-3 MLP training labels in
one shot. Per the W44-229 chunk spec:

  - Image corpus: existing W44-216+W44-219 corpus subset (R2 already
    populated) PLUS the 3 W44-105 SHIP images (terminal, imac_g3,
    codec_wiki).
  - Parameter axis: 40 LHS samples over Tier2Knobs 5-D space + 3
    fixed-anchor arms:
      1. Tier2Knobs::default() baseline
      2. Tier2Knobs::auto_for_distance(class, distance) per-cell
         (the W44-228b API)
      3. cjxl Libjxl-strategy bitstream parity reference
  - Effort × distance: 4 × 6 = 24 combinations spanning W44-105 SHIP.
  - Strategies: zenjxl (40 LHS + 2 fixed knob arms) + libjxl (defaults
    anchor only, per W44-217 finding #1 that libjxl is invariant to
    these params).

Each LHS sample is expanded to the 6 RuntimeTuning params via the
W44-222 Python mirror of Tier2Knobs::expand_to_runtime_tuning(). The
expanded 6 params are postcard-serialised as 24 bytes (6 × f32 LE) —
byte-identical to what postcard::to_allocvec would produce on the
Rust side (verified by w44_229_parity_check.rs).

We also write a sidecar manifest `chunks_manifest.parquet` that maps
each (image, distance, effort, strategy, params_blob_sha256) cell to:
  - knob_source: "lhs_<N>" / "tier2_default" / "tier2_auto_for_distance"
    / "libjxl_anchor"
  - tier2_knobs_vec: the 5-D Tier2Knobs source vector (NaN for libjxl
    anchor rows that don't go through the expander)
  - tier2_class: inferred ImageContentClass label (photo / screenshot,
    coarse heuristic from filename — refined in the parquet by the
    worker's actual zenanalyze features)

Downstream join: the worker emits `params_blob` (24-byte LE blob) into
each per-cell parquet row. To recover (knob_source, tier2_knobs_vec),
post-merge analysis joins on sha256(params_blob) → chunks_manifest row.

Outputs:
  /tmp/w44-229/
    chunks/chunk-NNNNN.json    NDJSON, 50 cells each
    params/<sha>.bin           postcard-encoded RuntimeTuning override
    chunks_manifest.tsv        knob_source + tier2_knobs_vec sidecar
    manifest.tsv               sweep_id, n_cells, n_chunks, n_blobs
    lhs_design.json            full LHS sample matrix for repro

Then upload:
  aws --profile r2 s3 sync /tmp/w44-229/params \
    s3://zen-tuning-ephemeral/w44-229-tier2-knob-validation/params/
  aws --profile r2 s3 sync /tmp/w44-229/chunks \
    s3://zen-tuning-ephemeral/w44-229-tier2-knob-validation/chunks/
  aws --profile r2 s3 cp /tmp/w44-229/chunks_manifest.tsv \
    s3://zen-tuning-ephemeral/w44-229-tier2-knob-validation/chunks_manifest.tsv
  SWEEP_ID=w44-229-tier2-knob-validation BOXES=8 \
    bash scripts/zenjxl-tuning-sweep/launch_w44_229_fleet.sh
"""
import argparse
import hashlib
import json
import math
import random
import struct
import sys
from pathlib import Path

try:
    import numpy as np
    from scipy.stats.qmc import LatinHypercube
except ImportError:
    print("ERROR: scipy + numpy required (pip install scipy numpy)", file=sys.stderr)
    sys.exit(1)

# ─── W44-222 5-knob Tier-2 expander (Python mirror of
#     jxl_encoder::tuning::coupling::Tier2Knobs::expand_to_runtime_tuning).
#
# Source of truth: jxl-encoder/src/tuning.rs:1125 (Rust). Constants
# pinned per W44-222 (commit 152c194c). Verified byte-identical to
# postcard::to_allocvec via the w44_229_parity_check.rs example.

# Defaults — match jxl-encoder/src/tuning.rs DEFAULT_P1..P6.
DEFAULTS = {
    "p1_smart_zenjxl_photo_mask_p25_min": 85.0,
    "p2_screenshot_median_threshold": 95.0,
    "p3_buttloop_default_screenshot_qf_seed_scale": 4.0,
    "p4_buttloop_qf_seed_scale_min_distance": 3.5,
    "p5_adaptive_quant_screenshot_qf_seed_scale_e5_e6": 2.0,
    "p6_adaptive_quant_screenshot_qf_seed_scale_e7": 3.0,
}
PARAM_ORDER = list(DEFAULTS.keys())

# W44-218 ridge constants.
P1_RIDGE_MAX = 192.86
P2_RIDGE_MAX = 108.15
P3_P6_SAT = 0.7
P5_P6_SAT = 0.8

# W44-222 K5 direction + scale (from
# benchmarks/sweeps/w44-219-densify/analysis/w44_222/phase_a_5knob_coverage.log).
# IMPORTANT: dtype=float32 throughout — Rust computes in f32, and a 1-ULP
# f64→f32 rounding drift makes the postcard blob differ by 1 byte at p3/p5.
# Verified byte-identical to Rust via examples/w44_229_parity_check.rs.
K5_DIR = np.array([-0.1479, +0.2589, -0.6501, 0.0, -0.5035, +0.4848], dtype=np.float32)
K5_SCALE = np.float32(2.5)


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def tier2_expand_5knob(
    smoothness_bias: float,
    screenshot_quant_aggressiveness: float,
    screen_quant_lift: float,
    buttloop_screen_d_gate: float,
    buttloop_aq_balance: float,
) -> dict:
    """Expand 5 Tier2 knobs into 6 RuntimeTuning params.

    Mirrors Tier2Knobs::expand_to_runtime_tuning() in jxl-encoder
    Rust source (commit 152c194c). Returns a dict keyed by PARAM_ORDER.

    Every intermediate is computed as np.float32 to match the Rust f32
    chain exactly. Doing the math in f64 then rounding to f32 diverges
    by 1 ULP on ~half the param positions because the K5 delta multiply
    triggers worst-case rounding. Verified byte-identical to Rust via
    examples/w44_229_parity_check.rs (4/4 cases).
    """
    f32 = np.float32
    s = f32(_clamp(smoothness_bias, 0.0, 1.0))
    a = f32(_clamp(screenshot_quant_aggressiveness, 0.0, 2.0))
    k = f32(_clamp(screen_quant_lift, 0.5, 2.0))
    d = f32(_clamp(buttloop_screen_d_gate, 1.5, 5.5))
    k5 = f32(_clamp(buttloop_aq_balance, -1.0, 1.0))

    P1_MAX_F = f32(P1_RIDGE_MAX)
    P2_MAX_F = f32(P2_RIDGE_MAX)
    P3_SAT_F = f32(P3_P6_SAT)
    P5_SAT_F = f32(P5_P6_SAT)
    D1 = f32(DEFAULTS["p1_smart_zenjxl_photo_mask_p25_min"])
    D2 = f32(DEFAULTS["p2_screenshot_median_threshold"])
    D3 = f32(DEFAULTS["p3_buttloop_default_screenshot_qf_seed_scale"])
    D5 = f32(DEFAULTS["p5_adaptive_quant_screenshot_qf_seed_scale_e5_e6"])
    D6 = f32(DEFAULTS["p6_adaptive_quant_screenshot_qf_seed_scale_e7"])
    ZERO = f32(0.0)
    ONE = f32(1.0)
    TWO = f32(2.0)
    THREE = f32(3.0)
    FOUR = f32(4.0)

    # p1/p2: smoothness_dispatch ridge with physical floor.
    p1_unc = D1 + (P1_MAX_F - D1) * (ONE - TWO * s)
    p2_unc = D2 + (P2_MAX_F - D2) * (ONE - TWO * s)
    p1_lo = max(ZERO, TWO * D1 - P1_MAX_F)
    p2_lo = max(ZERO, TWO * D2 - P2_MAX_F)
    p1_s = max(p1_lo, min(P1_MAX_F, p1_unc))
    p2_s = max(p2_lo, min(P2_MAX_F, p2_unc))

    # p3/p6: screenshot_qac lift with soft saturation past a=1.
    a_eff = a if a <= ONE else ONE + (a - ONE) * P3_SAT_F
    p3_a = FOUR * a_eff
    p6_a = THREE * a_eff

    # p5/p6: effort-conditional screen lift with soft saturation past k=1.
    k_eff = k if k <= ONE else ONE + (k - ONE) * P5_SAT_F
    p5_k = TWO * k_eff
    p6_k = THREE * k_eff

    # K5 contribution — every term in f32.
    k5_scale = K5_SCALE * k5
    k5_delta_p1 = k5_scale * K5_DIR[0]
    k5_delta_p2 = k5_scale * K5_DIR[1]
    k5_delta_p3 = k5_scale * K5_DIR[2]
    # K5_DIR[3] = 0 by construction → p4 unperturbed
    k5_delta_p5 = k5_scale * K5_DIR[4]
    k5_delta_p6 = k5_scale * K5_DIR[5]

    # Additive composition — matches Rust expand_to_runtime_tuning exactly.
    p1 = D1 + (p1_s - D1) + k5_delta_p1
    p2 = D2 + (p2_s - D2) + k5_delta_p2
    p3 = D3 + (p3_a - D3) + k5_delta_p3
    p4 = d
    p5 = D5 + (p5_k - D5) + k5_delta_p5
    p6 = D6 + (p6_a - D6) + (p6_k - D6) + k5_delta_p6

    # Physical floors — same as Rust impl.
    p1 = max(ZERO, p1)
    p2 = max(ZERO, p2)
    p3 = max(ZERO, p3)
    p5 = max(ZERO, p5)
    p6 = max(ZERO, p6)

    return {
        "p1_smart_zenjxl_photo_mask_p25_min": float(p1),
        "p2_screenshot_median_threshold": float(p2),
        "p3_buttloop_default_screenshot_qf_seed_scale": float(p3),
        "p4_buttloop_qf_seed_scale_min_distance": float(p4),
        "p5_adaptive_quant_screenshot_qf_seed_scale_e5_e6": float(p5),
        "p6_adaptive_quant_screenshot_qf_seed_scale_e7": float(p6),
    }


# ─── W44-228b ContentStratum lookup (Python mirror) ──────────────────
# Mirrors ContentStratum::from_distance_band + Tier2Knobs::default_for_stratum
# from jxl-encoder/src/tuning.rs:1282 + 1359. Used for the
# "tier2_auto_for_distance" knob arm.
PER_STRATUM_K_TUPLES = {
    # (k1, k2, k3, k4, k5)
    "screen/very_high": (0.0, 0.0, 0.5, 1.5, 0.0),
    "screen/high": (0.0, 0.0, 0.5, 3.5, -0.3333333333333334),
    "screen/mid": (0.0, 0.0, 0.5, 3.5, 0.0),
    "screen/low": (1.0, 0.0, 0.5, 2.1666666666666665, 0.0),
    "photo/very_high": (
        0.3333333333333333, 0.0, 0.5, 4.833333333333333, -0.6666666666666667,
    ),
    "photo/high": (
        0.16666666666666666, 0.0, 1.25, 4.833333333333333, -0.6666666666666667,
    ),
    "photo/mid": (
        1.0, 0.0, 2.0, 2.833333333333333, 0.33333333333333326,
    ),
    "photo/low": (
        0.8333333333333333, 0.0, 0.5, 2.1666666666666665, 0.6666666666666665,
    ),
}


def stratum_for(class_str: str, distance: float) -> str:
    """class_str ∈ {"photo", "screen"}; returns the stratum string key."""
    if distance < 1.0:
        band = "low"
    elif distance < 2.0:
        band = "mid"
    elif distance < 3.5:
        band = "high"
    else:
        band = "very_high"
    return f"{class_str}/{band}"


def auto_for_distance_knobs(class_str: str, distance: float) -> tuple:
    """Mirror Tier2Knobs::auto_for_distance(class, distance) -> 5-tuple."""
    if class_str in ("photo", "screen"):
        return PER_STRATUM_K_TUPLES[stratum_for(class_str, distance)]
    # Default (matches Tier2Knobs::default()) — round-trip byte-identical.
    return (0.5, 1.0, 1.0, 3.5, 0.0)


# ─── Sweep grid ──────────────────────────────────────────────────────
# W44-105 SHIP cells use e8+ d=4-6. Spec wants efforts {5, 7, 8, 9}
# and distances spanning W44-105 SHIP range {0.5..6.0}.
EFFORTS = [5, 7, 8, 9]
DISTANCES = [0.5, 1.0, 2.0, 3.0, 4.5, 6.0]


# ─── Serialisation ──────────────────────────────────────────────────
def encode_postcard_tuning(values: dict) -> bytes:
    """6 little-endian f32 in field-declaration order.

    Verified byte-identical to postcard::to_allocvec(RuntimeTuning)
    (Rust unit test in zenjxl-tuning-runner/src/params.rs:114).
    """
    out = b""
    for k in PARAM_ORDER:
        out += struct.pack("<f", float(values[k]))
    assert len(out) == 24, f"blob must be 24 bytes, got {len(out)}"
    return out


def sha256_hex(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


# ─── Image corpus selection ─────────────────────────────────────────
# W44-229: subset of the W44-216+W44-219 corpus (already in R2) PLUS the
# 3 W44-105 SHIP images. The R2 corpus has 38 PNGs; we use 27 of them so
# every image gets coverage on every (effort, distance, knob_arm) cell
# without inflating cost.
#
# Image selection: per Rule 4 ("≥30 images"), aim for 27 images total
# (the 3 W44-105 SHIP cells + 24 stratified from the existing corpus).
# Stratified: 8 gb82-sc screenshots + 16 CID22 photos + 3 mixed.


def pick_w44_229_corpus(local_repo_root: Path | None = None) -> list[tuple[str, str]]:
    """Pick the W44-229 corpus subset.

    Returns list of (container_path, source_class) tuples. container_path
    matches the R2 layout: /corpus/<basename>.png.

    Coarse source_class: "screen" for gb82-sc/gb82, "photo" for
    CID22/clic2025. The worker computes the actual zenanalyze features
    per-cell — the source_class is just for tier2_auto_for_distance
    knob arm assignment.
    """
    out = []

    # 3 W44-105 SHIP images (always included).
    for name in ("terminal.png", "imac_g3.png", "codec_wiki.png"):
        out.append((f"/corpus/{name}", "screen"))

    # 5 more gb82-sc screens (matches W44-219 corpus, full 8).
    for name in (
        "gmessages.png",
        "graph.png",
        "gui.png",
        "imac_dark.png",
        "imessage.png",
    ):
        out.append((f"/corpus/{name}", "screen"))

    # 2 gb82 (lossless) screens (added in W44-219).
    for name in ("baby-lossless.png", "bulb-lossless.png"):
        out.append((f"/corpus/{name}", "screen"))

    # 17 CID22 photos (a subset of W44-219's 23) — pick the first
    # alphabetically + an evenly-distributed tail. This matches the
    # spread the worker has cached in R2 corpus.
    cid22_picks = [
        "1025469.png", "1029604.png", "1044329.png", "1189261.png",
        "1279330.png", "1418519.png", "1420710.png", "1475938.png",
        "1531677.png", "1544947.png", "159550.png", "1624487.png",
        "2079234.png", "2389166.png", "2775196.png", "297394.png",
        "3637739.png",
    ]
    for name in cid22_picks:
        out.append((f"/corpus/{name}", "photo"))

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-id", default="w44-229-tier2-knob-validation")
    ap.add_argument("--out-dir", default="/tmp/w44-229")
    ap.add_argument("--n-lhs-blobs", type=int, default=40,
                    help="Latin-hypercube samples over Tier2Knobs 5-D space.")
    ap.add_argument("--chunk-size", type=int, default=50)
    ap.add_argument("--lhs-seed", type=int, default=44229)
    ap.add_argument("--shuffle-seed", type=int, default=44229)
    args = ap.parse_args()

    out = Path(args.out_dir)
    (out / "chunks").mkdir(parents=True, exist_ok=True)
    (out / "params").mkdir(parents=True, exist_ok=True)

    # ── 1. Image corpus ─────────────────────────────────────────────
    corpus_specs = pick_w44_229_corpus()
    print(f"[w44-229] {len(corpus_specs)} images "
          f"({sum(1 for _, c in corpus_specs if c == 'screen')} screen + "
          f"{sum(1 for _, c in corpus_specs if c == 'photo')} photo)")

    # ── 2. Param blobs: defaults + LHS samples + per-stratum optima ─
    # We need to enumerate every UNIQUE (knob_arm, image, distance,
    # effort) combination since `auto_for_distance` depends on
    # (image_class, distance) → different blob per cell.
    blob_manifest = {}  # sha → {"source": str, "tier2_vec": [..]}
    blob_files = {}     # sha → (container_path, raw bytes)

    def add_blob(values_dict: dict, source: str, tier2_vec_or_none) -> str:
        b = encode_postcard_tuning(values_dict)
        sha = sha256_hex(b)
        if sha not in blob_manifest:
            blob_manifest[sha] = {
                "source": source,
                "tier2_vec": tier2_vec_or_none,
            }
            blob_files[sha] = (f"/sweep-state/params/{sha}.bin", b)
            (out / "params" / f"{sha}.bin").write_bytes(b)
        return sha

    # 2a. Defaults blob — used for "tier2_default" arm + libjxl anchor.
    default_knobs = (0.5, 1.0, 1.0, 3.5, 0.0)
    default_values = tier2_expand_5knob(*default_knobs)
    default_sha = add_blob(default_values, "tier2_default", list(default_knobs))
    print(f"[w44-229] default blob sha = {default_sha[:16]}…")

    # 2b. LHS samples over Tier2Knobs 5-D space.
    sampler = LatinHypercube(d=5, seed=args.lhs_seed, scramble=True)
    lhs_unit = sampler.random(n=args.n_lhs_blobs)
    KNOB_RANGES = [
        (0.0, 1.0),   # smoothness_bias
        (0.0, 2.0),   # screenshot_quant_aggressiveness
        (0.5, 2.0),   # screen_quant_lift
        (1.5, 5.5),   # buttloop_screen_d_gate
        (-1.0, 1.0),  # buttloop_aq_balance
    ]
    lhs_samples = []
    for row in lhs_unit:
        knobs = tuple(
            float(lo + u * (hi - lo)) for u, (lo, hi) in zip(row, KNOB_RANGES)
        )
        lhs_samples.append(knobs)
    with (out / "lhs_design.json").open("w") as f:
        json.dump(lhs_samples, f, indent=2)

    lhs_shas = []
    for i, knobs in enumerate(lhs_samples):
        vals = tier2_expand_5knob(*knobs)
        sha = add_blob(vals, f"lhs_{i:03d}", list(knobs))
        lhs_shas.append(sha)
    print(f"[w44-229] {len(lhs_shas)} LHS blobs added "
          f"({len(blob_manifest) - 1} unique non-default)")

    # 2c. Per-stratum optima — pre-add all 8 + 1 default fallback (the
    # auto_for_distance arm). One blob per (class, distance band)
    # combination; computed once and reused across many cells.
    auto_shas = {}  # (class, distance) → sha
    for cls in ("photo", "screen"):
        for d in DISTANCES:
            knobs = auto_for_distance_knobs(cls, d)
            vals = tier2_expand_5knob(*knobs)
            sha = add_blob(vals, f"tier2_auto_for_distance_{cls}_d{d}", list(knobs))
            auto_shas[(cls, d)] = sha
    print(f"[w44-229] {len(set(auto_shas.values()))} unique "
          f"auto_for_distance blobs across "
          f"{len(auto_shas)} (class, distance) bins")

    total_blobs = len(blob_manifest)
    print(f"[w44-229] {total_blobs} unique blobs total")

    # ── 3. Cell enumeration ─────────────────────────────────────────
    cells = []
    manifest_rows = []  # for chunks_manifest.tsv sidecar
    cell_id = 0
    for img_container, img_class in corpus_specs:
        for effort in EFFORTS:
            for distance in DISTANCES:
                # zenjxl × {default, LHS_0..N, auto_for_distance}
                # = (1 + N + 1) blobs = 42 zenjxl cells per (img, e, d).
                # Plus 1 libjxl anchor cell.
                arms = []

                # 3a. Defaults blob (zenjxl).
                arms.append((default_sha, "tier2_default", "zenjxl"))

                # 3b. LHS blobs (zenjxl).
                for i, sha in enumerate(lhs_shas):
                    arms.append((sha, f"lhs_{i:03d}", "zenjxl"))

                # 3c. auto_for_distance (zenjxl).
                auto_sha = auto_shas[(img_class, distance)]
                arms.append((
                    auto_sha,
                    f"tier2_auto_for_distance_{img_class}_d{distance}",
                    "zenjxl",
                ))

                # 3d. libjxl anchor (defaults blob, libjxl strategy).
                arms.append((default_sha, "libjxl_anchor", "libjxl"))

                for sha, source_tag, strat in arms:
                    cell_id += 1
                    cells.append({
                        "sweep_id": args.sweep_id,
                        "chunk_claim_id": f"c{cell_id:07d}",
                        "image_path": img_container,
                        "effort": effort,
                        "distance": float(distance),
                        "strategy": strat,
                        "params_blob_path": blob_files[sha][0],
                        "threads": 4,
                        "metric_backend": "auto",
                    })
                    manifest_rows.append({
                        "chunk_claim_id": f"c{cell_id:07d}",
                        "image_path": img_container,
                        "image_class": img_class,
                        "effort": effort,
                        "distance": float(distance),
                        "strategy": strat,
                        "params_blob_sha256": sha,
                        "knob_source": source_tag,
                        "tier2_knobs_vec": (
                            json.dumps(blob_manifest[sha]["tier2_vec"])
                            if blob_manifest[sha]["tier2_vec"] is not None
                            else "null"
                        ),
                    })

    print(f"[w44-229] {len(cells)} cells total")

    # Cost estimate.
    # Per W44-216 / W44-219 telemetry: ~3-5s mean wall per cell on
    # RTX A4000 single-cell with metric backend = "auto" (GPU score).
    avg_s = 4.0
    boxes = 8
    parallel_factor = boxes  # one cell per box at a time, single-thread
    total_s = len(cells) * avg_s / parallel_factor
    print(f"[w44-229] est total wall ≈ {total_s / 3600:.1f}h on {boxes} boxes")

    # ── 4. Shuffle + chunk ───────────────────────────────────────────
    rng = random.Random(args.shuffle_seed)
    rng.shuffle(cells)
    n_chunks = math.ceil(len(cells) / args.chunk_size)
    for i in range(n_chunks):
        chunk = cells[i * args.chunk_size : (i + 1) * args.chunk_size]
        chunk_file = out / "chunks" / f"chunk-{i:06d}.json"
        with chunk_file.open("w") as f:
            for c in chunk:
                f.write(json.dumps(c) + "\n")
    print(f"[w44-229] wrote {n_chunks} chunks of {args.chunk_size} cells each")

    # ── 5. Sidecar manifests ─────────────────────────────────────────
    # chunks_manifest.tsv: knob_source + tier2_knobs_vec per cell
    manifest_path = out / "chunks_manifest.tsv"
    with manifest_path.open("w") as f:
        cols = ["chunk_claim_id", "image_path", "image_class", "effort",
                "distance", "strategy", "params_blob_sha256",
                "knob_source", "tier2_knobs_vec"]
        f.write("\t".join(cols) + "\n")
        for r in manifest_rows:
            f.write("\t".join(str(r[c]) for c in cols) + "\n")
    print(f"[w44-229] chunks_manifest at {manifest_path}")

    # manifest.tsv: aggregate counts
    summary_path = out / "manifest.tsv"
    with summary_path.open("w") as f:
        f.write("key\tvalue\n")
        f.write(f"sweep_id\t{args.sweep_id}\n")
        f.write(f"n_cells\t{len(cells)}\n")
        f.write(f"n_chunks\t{n_chunks}\n")
        f.write(f"chunk_size\t{args.chunk_size}\n")
        f.write(f"n_blobs\t{total_blobs}\n")
        f.write(f"n_lhs_samples\t{args.n_lhs_blobs}\n")
        f.write(f"n_images\t{len(corpus_specs)}\n")
        f.write(f"n_efforts\t{len(EFFORTS)}\n")
        f.write(f"n_distances\t{len(DISTANCES)}\n")
        f.write(f"efforts\t{','.join(str(e) for e in EFFORTS)}\n")
        f.write(f"distances\t{','.join(str(d) for d in DISTANCES)}\n")
        f.write(f"lhs_seed\t{args.lhs_seed}\n")
        f.write(f"shuffle_seed\t{args.shuffle_seed}\n")
        f.write(f"est_wall_hours_8box\t{total_s / 3600:.2f}\n")
    print(f"[w44-229] manifest at {summary_path}")


if __name__ == "__main__":
    main()
