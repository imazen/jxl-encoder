#!/usr/bin/env python3
"""Build W44-PHASE4-S2-c2-validate chunks — VALIDATE the S2-refit-c2
per-stratum lookup (a7da5a7a) on the post-W44-RECON-DEEP encoder under
a TIGHT $10 budget. Forked from build_w44_phase4_s1_chunks.py with a
1/3-sized corpus (9 images) to fit.

PHASE-4 HYPOTHESIS REVISIT:
  S1+M1 confirmed the S2-refit lookup shifts per-stratum optima vs
  W44-229. S2-refit-c1 honest-stopped on screen/very_high k2. S2-refit-c2
  (a7da5a7a) floored k1+k2 on screen/very_high + screen/high. This sweep
  validates that the c2 floor change holds across a representative 9-image
  subset on the latest encoder (bdd5f4fb post-RECON-DEEP + S2-refit stack
  + B5b iter-0 divergence detector).

DESIGN — forks S1 with corpus cut to 9 images for $10 budget:

  1. SAME LHS SEED (441) as S1 — direct A/B comparability per cell
  2. NEW SWEEP_ID — w44-phase4-s2-c2-validate (separate R2 prefix)
  3. NEW DOCKER IMAGE TAG — built from origin/main bdd5f4fb (post-S2-refit
     stack + B5b detector). S1 ran on 53b7655b which lacks S2-refit-c2.
  4. CUT corpus to 9 images: 3 W44-105 SHIP screens + 6 representative
     CID22 photos covering low/mid/high/very_high distance strata.
  5. KEPT 4 efforts + 6 distances + 43 arms — same per-cell shape as S1.

PER METHODOLOGY RULE 1 (ablation-first kitchen-sink): the sweep produces
a kitchen-sink GBR test R² target (≥0.85 on every outcome) to confirm
the joint surface is fittable on the new encoder; if R² drops materially
that's a finding of its own (encoder structural changes broke the
6-param coupling assumption).

PER METHODOLOGY RULE 4 (bigger sweeps, fewer of them): targets ~9.3K
cells. Under the $10 cap; using --bid_price keeps cost projected ~$1-3.

PER METHODOLOGY RULE 9 (interruptible default): launcher defaults to
--interruptible with bid_price tuned for ~$0.10/hr offers.

Outputs (mirrors S1 layout):
  /tmp/w44-phase4-s2/
    chunks/chunk-NNNNN.json    NDJSON, 50 cells each
    params/<sha>.bin           postcard-encoded RuntimeTuning override
    chunks_manifest.tsv        knob_source + tier2_knobs_vec sidecar
    manifest.tsv               sweep_id, n_cells, n_chunks, n_blobs
    lhs_design.json            full LHS sample matrix for repro

Then upload:
  aws --profile r2 s3 sync /tmp/w44-phase4-s2/params \
    s3://zen-tuning-ephemeral/w44-phase4-s2-c2-validate/params/
  aws --profile r2 s3 sync /tmp/w44-phase4-s2/chunks \
    s3://zen-tuning-ephemeral/w44-phase4-s2-c2-validate/chunks/
  aws --profile r2 s3 cp /tmp/w44-phase4-s2/chunks_manifest.tsv \
    s3://zen-tuning-ephemeral/w44-phase4-s2-c2-validate/chunks_manifest.tsv
  SWEEP_ID=w44-phase4-s2-c2-validate BOXES=5 BID_PRICE=0.08 \
    LABEL_PREFIX=claude-w44-phase4-s2 \
    IMAGE=ghcr.io/lilith/zenjxl-tuning-sweep:v3-schema-v2-bdd5f4fb \
    bash scripts/zenjxl-tuning-sweep/launch_w44_phase4_s1_fleet.sh

CORPUS DEPENDENCY (W44-PHASE4-S1h, 2026-05-24):
  This script picks images by basename and produces image_path values
  like "/corpus/<basename>.png" but does NOT stage the image bytes
  itself. Every image referenced in `pick_w44_phase4_s1_corpus()` MUST
  already exist in `s3://zen-tuning-ephemeral/corpus/<basename>.png`
  BEFORE the launcher runs — otherwise workers will silently mark every
  cell for that image as `image_fetch_failed` and you'll lose ~1k cells
  per missing image. The W44-PHASE4-S1 sweep lost 4 × 1,032 = 4,128
  cells this way (images 1029604/2775196/297394/3637739).

  Mitigation: `launch_w44_phase4_s1_fleet.sh` runs a pre-flight
  manifest-vs-corpus check and FAILS LOUD before creating any vast.ai
  boxes. If your sweep adds new images, upload them with:
    AWS_PROFILE=r2 aws s3 cp --endpoint-url=... \
      /path/to/<basename>.png \
      s3://zen-tuning-ephemeral/corpus/<basename>.png
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

# ─── W44-222 5-knob Tier-2 expander — Python mirror.
# IMPORTANT: this MUST stay byte-identical to the W44-229 mirror
# (build_w44_229_chunks.py) until/unless W44-222 expander changes in
# Rust. The same w44_229_parity_check.rs covers this expander; no need
# for a per-phase parity check unless this file diverges.

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

# W44-222 K5 direction + scale. dtype=float32 throughout to match the
# Rust f32 chain exactly.
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
    """Expand 5 Tier2 knobs into 6 RuntimeTuning params (W44-222 mirror).

    Every intermediate is computed as np.float32 to match the Rust f32
    chain exactly. See w44_229_parity_check.rs for verification.
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

    p1_unc = D1 + (P1_MAX_F - D1) * (ONE - TWO * s)
    p2_unc = D2 + (P2_MAX_F - D2) * (ONE - TWO * s)
    p1_lo = max(ZERO, TWO * D1 - P1_MAX_F)
    p2_lo = max(ZERO, TWO * D2 - P2_MAX_F)
    p1_s = max(p1_lo, min(P1_MAX_F, p1_unc))
    p2_s = max(p2_lo, min(P2_MAX_F, p2_unc))

    a_eff = a if a <= ONE else ONE + (a - ONE) * P3_SAT_F
    p3_a = FOUR * a_eff
    p6_a = THREE * a_eff

    k_eff = k if k <= ONE else ONE + (k - ONE) * P5_SAT_F
    p5_k = TWO * k_eff
    p6_k = THREE * k_eff

    k5_scale = K5_SCALE * k5
    k5_delta_p1 = k5_scale * K5_DIR[0]
    k5_delta_p2 = k5_scale * K5_DIR[1]
    k5_delta_p3 = k5_scale * K5_DIR[2]
    k5_delta_p5 = k5_scale * K5_DIR[4]
    k5_delta_p6 = k5_scale * K5_DIR[5]

    p1 = D1 + (p1_s - D1) + k5_delta_p1
    p2 = D2 + (p2_s - D2) + k5_delta_p2
    p3 = D3 + (p3_a - D3) + k5_delta_p3
    p4 = d
    p5 = D5 + (p5_k - D5) + k5_delta_p5
    p6 = D6 + (p6_a - D6) + (p6_k - D6) + k5_delta_p6

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
PER_STRATUM_K_TUPLES = {
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
    if class_str in ("photo", "screen"):
        return PER_STRATUM_K_TUPLES[stratum_for(class_str, distance)]
    return (0.5, 1.0, 1.0, 3.5, 0.0)


# ─── Sweep grid ──────────────────────────────────────────────────────
# Matches W44-229 for direct A/B comparability vs the
# pre-W44-RECON-DEEP baseline.
EFFORTS = [5, 7, 8, 9]
DISTANCES = [0.5, 1.0, 2.0, 3.0, 4.5, 6.0]


# ─── Serialisation ──────────────────────────────────────────────────
def encode_postcard_tuning(values: dict) -> bytes:
    out = b""
    for k in PARAM_ORDER:
        out += struct.pack("<f", float(values[k]))
    assert len(out) == 24, f"blob must be 24 bytes, got {len(out)}"
    return out


def sha256_hex(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


# ─── Image corpus selection ─────────────────────────────────────────
# S2-c2-validate: 1/3 of S1's corpus (9 of 27 images) for $10 budget.
# Picked to span the per-stratum lookup splits (screen × {very_high,
# high, mid, low}, photo × {very_high, high, mid, low}). Every image
# referenced here must exist in s3://zen-tuning-ephemeral/corpus/.
def pick_w44_phase4_s1_corpus() -> list[tuple[str, str]]:
    """Pick the W44-PHASE4-S2-c2-validate corpus subset (9 images).

    Returns list of (container_path, source_class) tuples. container_path
    matches the R2 layout: /corpus/<basename>.png.
    """
    out = []

    # 3 W44-105 SHIP screens — primary screen-stratum coverage.
    for name in ("terminal.png", "imac_g3.png", "codec_wiki.png"):
        out.append((f"/corpus/{name}", "screen"))

    # 6 CID22 photos sampled from S1's 17 (every-3rd index modulo
    # cluster — picks 1 from each rough content cluster S1 had).
    # Spans low-m3 (1531677), mid (1420710), high-m3 photos so the
    # photo strata see varied input.
    cid22_picks = [
        "1025469.png",   # photo
        "1189261.png",   # photo (W44-105 cluster #2)
        "1418519.png",   # photo (W44-105 cluster #2)
        "1420710.png",   # photo (S2-refit-c2 high-m3 reference)
        "1531677.png",   # photo (S2-refit-c2 low-m3 reference)
        "3637739.png",   # photo (W44-202 cluster #1 anchor)
    ]
    for name in cid22_picks:
        out.append((f"/corpus/{name}", "photo"))

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-id", default="w44-phase4-s2-c2-validate")
    ap.add_argument("--out-dir", default="/tmp/w44-phase4-s2")
    ap.add_argument("--n-lhs-blobs", type=int, default=40,
                    help="Latin-hypercube samples over Tier2Knobs 5-D space.")
    ap.add_argument("--chunk-size", type=int, default=50)
    # SAME SEED as S1 (441) so the 40 LHS samples are identical and
    # per-cell A/B comparison vs S1 is straightforward.
    ap.add_argument("--lhs-seed", type=int, default=441)
    ap.add_argument("--shuffle-seed", type=int, default=441)
    args = ap.parse_args()

    out = Path(args.out_dir)
    (out / "chunks").mkdir(parents=True, exist_ok=True)
    (out / "params").mkdir(parents=True, exist_ok=True)

    # ── 1. Image corpus ─────────────────────────────────────────────
    corpus_specs = pick_w44_phase4_s1_corpus()
    print(f"[w44-phase4-s2] {len(corpus_specs)} images "
          f"({sum(1 for _, c in corpus_specs if c == 'screen')} screen + "
          f"{sum(1 for _, c in corpus_specs if c == 'photo')} photo)")

    # ── 2. Param blobs: defaults + LHS samples + per-stratum optima ─
    blob_manifest = {}
    blob_files = {}

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

    # 2a. Defaults blob.
    default_knobs = (0.5, 1.0, 1.0, 3.5, 0.0)
    default_values = tier2_expand_5knob(*default_knobs)
    default_sha = add_blob(default_values, "tier2_default", list(default_knobs))
    print(f"[w44-phase4-s2] default blob sha = {default_sha[:16]}…")

    # 2b. LHS samples over Tier2Knobs 5-D space.
    sampler = LatinHypercube(d=5, seed=args.lhs_seed, scramble=True)
    lhs_unit = sampler.random(n=args.n_lhs_blobs)
    KNOB_RANGES = [
        (0.0, 1.0),
        (0.0, 2.0),
        (0.5, 2.0),
        (1.5, 5.5),
        (-1.0, 1.0),
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
    print(f"[w44-phase4-s2] {len(lhs_shas)} LHS blobs added "
          f"({len(blob_manifest) - 1} unique non-default)")

    # 2c. Per-stratum optima — same as W44-229.
    auto_shas = {}
    for cls in ("photo", "screen"):
        for d in DISTANCES:
            knobs = auto_for_distance_knobs(cls, d)
            vals = tier2_expand_5knob(*knobs)
            sha = add_blob(vals, f"tier2_auto_for_distance_{cls}_d{d}", list(knobs))
            auto_shas[(cls, d)] = sha
    print(f"[w44-phase4-s2] {len(set(auto_shas.values()))} unique "
          f"auto_for_distance blobs across "
          f"{len(auto_shas)} (class, distance) bins")

    total_blobs = len(blob_manifest)
    print(f"[w44-phase4-s2] {total_blobs} unique blobs total")

    # ── 3. Cell enumeration ─────────────────────────────────────────
    cells = []
    manifest_rows = []
    cell_id = 0
    for img_container, img_class in corpus_specs:
        for effort in EFFORTS:
            for distance in DISTANCES:
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

    print(f"[w44-phase4-s2] {len(cells)} cells total")

    # Cost estimate.
    avg_s = 4.0
    boxes = 5
    parallel_factor = boxes
    total_s = len(cells) * avg_s / parallel_factor
    print(f"[w44-phase4-s2] est total wall ≈ {total_s / 3600:.1f}h on {boxes} boxes")

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
    print(f"[w44-phase4-s2] wrote {n_chunks} chunks of {args.chunk_size} cells each")

    # ── 5. Sidecar manifests ─────────────────────────────────────────
    manifest_path = out / "chunks_manifest.tsv"
    with manifest_path.open("w") as f:
        cols = ["chunk_claim_id", "image_path", "image_class", "effort",
                "distance", "strategy", "params_blob_sha256",
                "knob_source", "tier2_knobs_vec"]
        f.write("\t".join(cols) + "\n")
        for r in manifest_rows:
            f.write("\t".join(str(r[c]) for c in cols) + "\n")
    print(f"[w44-phase4-s2] chunks_manifest at {manifest_path}")

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
        f.write(f"est_wall_hours_5box\t{total_s / 3600:.2f}\n")
        f.write(f"comparable_to\tw44-phase4-s1-recon-deep-revalidate\n")
        f.write(f"encoder_state\tpost-S2-refit-c2 main bdd5f4fb\n")
    print(f"[w44-phase4-s2] manifest at {summary_path}")


if __name__ == "__main__":
    main()
