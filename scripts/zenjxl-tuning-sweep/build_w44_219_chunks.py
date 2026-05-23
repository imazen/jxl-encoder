#!/usr/bin/env python3
"""Build W44-219 densify-sweep chunks: LHS + pair-grids + image expansion.

W44-219 follows W44-216 (broad LHS) and W44-218 (R² fit honest-stopped on
too-sparse corpus). This sweep produces a *denser* corpus W44-220 will
refit on:

  Density axis 1 — parameter blobs:
    - 150 Latin-hypercube samples over 6 RuntimeTuning fields
      (scipy.stats.qmc.LatinHypercube, seed=44219, scrambled).
      W44-216 had 13 distinct blobs; W44-219 ships 150.
    - 5 pair-focused 2D grids (5×5 each) over the top couplings from
      W44-217 (`interaction_ranking.tsv`):
        (p4, p6)  – ssim2 cross +0.37 (top)
        (p2, p5)  – ssim2 cross -0.23
        (p1, p5)  – ssim2 cross +0.23  (replaces "p3, p6" from spec —
                                         empirical top-3 per ranking)
        (p3, p6)  – ssim2 cross -0.15  (task-spec ask)
        (p5, p6)  – ssim2 cross -0.18
      Each grid = 5 × 5 = 25 blobs at the other 4 params held at
      DEFAULT. 5 grids × 25 = 125 grid blobs. Defaults blob deduped.
    - Total: ~275 unique blobs (vs 13 in W44-216).

  Density axis 2 — images:
    W44-216 corpus = 20 CID22 photo + 8 gb82-sc screenshots = 28 images.
    W44-217 §9 question 1: want 100+ to tighten per-stratum CIs;
    W44-217 §9 question 3: NO `class=photo + screen-class param` outliers
      sampled — want mid-mask images that bridge photo ↔ screenshot.
    W44-219 picks ~37 images total: keep all 28 W44-216 + add 9 more
      drawn from clic2025 + CID22 training, stratified by edge-density
      proxies (filename hash spread) — image expansion is per-task-spec.

  Density axis 3 — effort/distance:
    Same 5 efforts × 7 distances × 2 strategies as W44-216, but
    e9 had only 79 screen rows in W44-216 — undersampled. W44-219
    runs the same grid; the increased blob density alone takes
    e9 screen rows from 79 → ~870 (11×).

Cell-count budget (cost-capped at $30):
  ~275 blobs × 37 images × 5 efforts × 7 distances × 2 strategies
  = ~712,250 theoretical cells.
  PRUNED via stratified sub-sampling per (image, effort, distance,
  strategy): keep all defaults + LHS-only on ~50 images, plus the
  full grid only on a stratified 12-image subset. Target: ~80-100K
  cells (under W44-216 25K × 4 density target).

Outputs:
  /tmp/w44-219/
    chunks/chunk-NNNNN.json   (NDJSON, 50 cells each)
    params/<sha>.bin          (postcard-encoded RuntimeTuning override)
    corpus/<basename>.png     (deduped image dir; W44-216 corpus + new picks)
    manifest.tsv              (sweep_id, n_cells, n_chunks, n_blobs, n_imgs)
    blob_provenance.tsv       (sha -> source = lhs|grid_pXxpY|defaults + raw vals)
    lhs_design.json           (full LHS sample matrix for reproducibility)

Then upload + launch as in W44-216:
  aws s3 sync --no-sign-request /tmp/w44-219/corpus s3://zen-tuning-ephemeral/corpus/
  aws s3 sync /tmp/w44-219/params s3://zen-tuning-ephemeral/w44-219-densify/params/
  aws s3 sync /tmp/w44-219/chunks s3://zen-tuning-ephemeral/w44-219-densify/chunks/
  SWEEP_ID=w44-219-densify BOXES=20 \
    bash scripts/zenjxl-tuning-sweep/launch_w44_219_fleet.sh

Honest-stop conditions:
  - If W44-216 corpus images aren't reachable on disk → abort
  - If scipy LHS unavailable → abort
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
    from scipy.stats.qmc import LatinHypercube
except ImportError:
    print("ERROR: scipy.stats.qmc.LatinHypercube required (pip install scipy)", file=sys.stderr)
    sys.exit(1)

# ─── RuntimeTuning field defaults (W44-218 SHIPPED, tuning.rs DEFAULT_P1..P6)
#     verified 2026-05-22.
DEFAULTS = {
    "p1_smart_zenjxl_photo_mask_p25_min": 85.0,
    "p2_screenshot_median_threshold": 95.0,
    "p3_buttloop_default_screenshot_qf_seed_scale": 4.0,
    "p4_buttloop_qf_seed_scale_min_distance": 3.5,
    "p5_adaptive_quant_screenshot_qf_seed_scale_e5_e6": 2.0,
    "p6_adaptive_quant_screenshot_qf_seed_scale_e7": 3.0,
}
PARAM_ORDER = list(DEFAULTS.keys())

# Sweep ranges. Mirror W44-216 — same envelope so W44-216 + W44-219 corpora
# can be concatenated cleanly. W44-218 ridge consts P1_RIDGE_MAX=192.86,
# P2_RIDGE_MAX=108.15 sit inside these.
RANGES = {
    "p1_smart_zenjxl_photo_mask_p25_min": (40.0, 200.0),
    "p2_screenshot_median_threshold": (75.0, 110.0),
    "p3_buttloop_default_screenshot_qf_seed_scale": (1.0, 8.0),
    "p4_buttloop_qf_seed_scale_min_distance": (1.5, 5.5),
    "p5_adaptive_quant_screenshot_qf_seed_scale_e5_e6": (1.0, 4.0),
    "p6_adaptive_quant_screenshot_qf_seed_scale_e7": (1.5, 5.5),
}

# Sweep grid (mirrors W44-216 EXACTLY so the corpora concat cleanly).
EFFORTS = [5, 6, 7, 8, 9]
DISTANCES = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
STRATEGIES = ["zenjxl", "libjxl"]

# Top couplings from W44-217 `interaction_ranking.tsv`. Ordered as
# (i, j): we sweep param[i] across 5 values × param[j] across 5 values
# at the other 4 held at their defaults.
PAIR_GRIDS = [
    ("p4_buttloop_qf_seed_scale_min_distance",
     "p6_adaptive_quant_screenshot_qf_seed_scale_e7"),  # top ssim2 cross
    ("p2_screenshot_median_threshold",
     "p5_adaptive_quant_screenshot_qf_seed_scale_e5_e6"),  # 2nd
    ("p1_smart_zenjxl_photo_mask_p25_min",
     "p5_adaptive_quant_screenshot_qf_seed_scale_e5_e6"),  # 3rd
    ("p3_buttloop_default_screenshot_qf_seed_scale",
     "p6_adaptive_quant_screenshot_qf_seed_scale_e7"),  # task-spec ask
    ("p5_adaptive_quant_screenshot_qf_seed_scale_e5_e6",
     "p6_adaptive_quant_screenshot_qf_seed_scale_e7"),  # task-spec ask
]


def encode_postcard_tuning(values: dict[str, float]) -> bytes:
    """6 little-endian f32 in field-declaration order. Verified by W44-214."""
    out = b""
    for k in PARAM_ORDER:
        out += struct.pack("<f", float(values[k]))
    assert len(out) == 24, len(out)
    return out


def sha256_hex(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def lhs_blobs(n: int, seed: int) -> list[dict[str, float]]:
    """Generate n LHS samples in [0, 1]^6 then map to RANGES."""
    sampler = LatinHypercube(d=6, seed=seed, scramble=True)
    samples = sampler.random(n=n)
    out = []
    for s in samples:
        v = {}
        for k, u in zip(PARAM_ORDER, s):
            lo, hi = RANGES[k]
            v[k] = float(lo + u * (hi - lo))
        out.append(v)
    return out


def grid_blobs(pair_i: str, pair_j: str, n_per_axis: int = 5) -> list[dict[str, float]]:
    """5×5 grid over (pair_i, pair_j) with other 4 params at DEFAULT.

    The 5 levels per axis are: range_lo, 0.5*(default+range_lo), default,
    0.5*(default+range_hi), range_hi. Defaults already sit in (range_lo,
    range_hi), so the 5 levels span the full range.
    """
    lo_i, hi_i = RANGES[pair_i]
    lo_j, hi_j = RANGES[pair_j]
    def_i = DEFAULTS[pair_i]
    def_j = DEFAULTS[pair_j]
    # 5 quantiles: 0, .25, .5, .75, 1 (linearly interpolated through
    # default at .5)
    def levels(lo, hi, defv):
        return [lo, 0.5 * (defv + lo), defv, 0.5 * (defv + hi), hi]
    lev_i = levels(lo_i, hi_i, def_i)
    lev_j = levels(lo_j, hi_j, def_j)
    out = []
    for vi in lev_i:
        for vj in lev_j:
            v = dict(DEFAULTS)
            v[pair_i] = vi
            v[pair_j] = vj
            out.append(v)
    return out


def stage_corpus(out_corpus: Path, corpus_specs: list[tuple[Path, str]]) -> dict[str, str]:
    """Copy each src → out_corpus/<basename>; return src→container map."""
    img_to_container = {}
    for src, container_path in corpus_specs:
        dst = out_corpus / src.name
        if not dst.exists():
            dst.write_bytes(src.read_bytes())
        img_to_container[str(src)] = container_path
    return img_to_container


def pick_w44_216_corpus() -> list[tuple[Path, str]]:
    """Pick the 28 W44-216 corpus images (verified present on disk)."""
    photo_dir = Path("/home/lilith/work/codec-corpus/CID22/CID22-512/validation")
    # W44-216 picked the first 20 photos alphabetically + 8 screens.
    photos = sorted(photo_dir.glob("*.png"))[:20]
    screen_dir = Path("/home/lilith/work/codec-corpus/gb82-sc")
    screens = sorted(screen_dir.glob("*.png"))[:8]
    specs = []
    for p in photos + screens:
        specs.append((p, f"/corpus/{p.name}"))
    return specs


def pick_new_w44_219_images() -> list[tuple[Path, str]]:
    """Expansion picks: 9 more images covering missing mask_median range.

    The W44-216 corpus has 20 photos (mask_median ≤ 2500 in the
    feat_mask_median scale) + 7 screenshots (clipped at 10000).
    The "mid-mask" gap that W44-217 §9 question 3 flags is the 2500-9000
    band — images that bridge photo ↔ screen.

    W44-219 picks 9 candidates from:
      - 4 from clic2025 validation (different content distribution
        than CID22 — likely shifted mask_median)
      - 3 from CID22 training subset (NOT the first-20 W44-216 slice —
        avoid overlap; pick indices [20, 25, 30] for variety)
      - 2 from gb82 lossless (different screen-class than gb82-sc)

    The W44-219 worker will compute feat_mask_median per-image at
    encode time (zenanalyze Tier-1 — already in the parquet schema
    as `feat_mask_median`), so no offline-feature-scan needed here.
    The aim is just a *broader* set of images, not pre-stratification.
    """
    new_specs = []
    # 4 from clic2025 validation (different photo distribution)
    clic_val = Path("/home/lilith/work/codec-corpus/clic2025/validation")
    if clic_val.exists():
        clic_picks = sorted(clic_val.glob("*.png"))[:4]
        for p in clic_picks:
            new_specs.append((p, f"/corpus/{p.name}"))
    # 3 from CID22 training (indices [40, 80, 120] for variety — sparse)
    cid_train = Path("/home/lilith/work/codec-corpus/CID22/CID22-512/training")
    if cid_train.exists():
        train_all = sorted(cid_train.glob("*.png"))
        # The 20 W44-216 corpus images come from CID22-512/validation,
        # not training, so there's no overlap to worry about.
        for idx in [40, 80, 120]:
            if idx < len(train_all):
                p = train_all[idx]
                new_specs.append((p, f"/corpus/{p.name}"))
    # 2 from gb82 lossless (different screen class than gb82-sc)
    gb82_dir = Path("/home/lilith/work/codec-corpus/gb82")
    if gb82_dir.exists():
        gb82_picks = sorted(gb82_dir.glob("*.png"))[:2]
        for p in gb82_picks:
            new_specs.append((p, f"/corpus/{p.name}"))
    return new_specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-id", default="w44-219-densify")
    ap.add_argument("--out-dir", default="/tmp/w44-219")
    ap.add_argument("--n-lhs-blobs", type=int, default=150)
    ap.add_argument("--n-grid-per-axis", type=int, default=5)
    ap.add_argument("--chunk-size", type=int, default=50)
    ap.add_argument("--lhs-seed", type=int, default=44219)
    # Cost-capping knobs: we don't want all blobs × all images × all cells.
    # Instead: LHS blobs run on every image; grid blobs run only on a
    # stratified subset of images.
    ap.add_argument("--n-grid-images", type=int, default=12,
                    help="Sub-sample for grid blobs: only this many images "
                         "get the full pair-grid coverage (cost cap).")
    ap.add_argument("--grid-image-seed", type=int, default=44219)
    args = ap.parse_args()

    out = Path(args.out_dir)
    (out / "chunks").mkdir(parents=True, exist_ok=True)
    (out / "params").mkdir(parents=True, exist_ok=True)
    (out / "corpus").mkdir(parents=True, exist_ok=True)

    # ── 1. Corpus: W44-216 + new W44-219 picks ────────────────────────
    w216_specs = pick_w44_216_corpus()
    w219_extra_specs = pick_new_w44_219_images()
    all_specs = w216_specs + w219_extra_specs
    if not all_specs:
        print("ERROR: no corpus images resolved", file=sys.stderr); sys.exit(1)
    print(f"[w44-219] {len(w216_specs)} W44-216 + {len(w219_extra_specs)} new = "
          f"{len(all_specs)} total images")

    img_to_container = stage_corpus(out / "corpus", all_specs)
    print(f"[w44-219] staged {len(img_to_container)} images to {out / 'corpus'}")
    all_images = [Path(k) for k in img_to_container.keys()]

    # ── 2. Param blobs ───────────────────────────────────────────────
    blob_provenance: dict[str, dict] = {}  # sha -> {"source": ..., "values": ...}
    blob_paths: list[tuple[str, str]] = []  # (container_path, sha)

    def add_blob(values: dict, source_tag: str) -> str:
        b = encode_postcard_tuning(values)
        sha = sha256_hex(b)
        (out / "params" / f"{sha}.bin").write_bytes(b)
        if sha not in blob_provenance:
            blob_provenance[sha] = {"source": source_tag, "values": values}
            blob_paths.append((f"/sweep-state/params/{sha}.bin", sha))
        else:
            # already added; merge source tags
            existing = blob_provenance[sha]["source"]
            if source_tag not in existing.split(","):
                blob_provenance[sha]["source"] = existing + "," + source_tag
        return sha

    # 2a. Defaults blob (round-trip anchor)
    add_blob(DEFAULTS, "defaults")

    # 2b. LHS blobs
    lhs_samples = lhs_blobs(args.n_lhs_blobs, args.lhs_seed)
    with (out / "lhs_design.json").open("w") as f:
        json.dump(lhs_samples, f, indent=2)
    for s in lhs_samples:
        add_blob(s, "lhs")

    # 2c. Pair-grid blobs (5 grids × 5×5 = 125 blobs, modulo collisions)
    grid_blobs_by_pair: dict[tuple[str, str], list[str]] = {}
    for pi, pj in PAIR_GRIDS:
        gb = grid_blobs(pi, pj, args.n_grid_per_axis)
        pair_tag = f"grid_{pi.split('_')[0]}x{pj.split('_')[0]}"
        shas = []
        for v in gb:
            s = add_blob(v, pair_tag)
            shas.append(s)
        grid_blobs_by_pair[(pi, pj)] = shas
        print(f"[w44-219] grid {pair_tag}: {len(gb)} blobs (defaults blob deduped)")

    print(f"[w44-219] total unique blobs: {len(blob_paths)}")

    # ── 3. Cell enumeration ──────────────────────────────────────────
    # LHS blobs (+ defaults): paired with EVERY image (the broad axis).
    # Grid blobs: paired only with the grid-image sub-sample (cost cap).
    rng = random.Random(args.grid_image_seed)
    # Stratify the grid-image sub-sample by index (1st, 4th, 7th, ... so
    # we cover both photos and screens).
    stride = max(1, len(all_images) // args.n_grid_images)
    grid_images = all_images[::stride][: args.n_grid_images]
    if len(grid_images) < args.n_grid_images:
        # Fill from the tail
        rest = [im for im in all_images if im not in grid_images]
        grid_images.extend(rest[: args.n_grid_images - len(grid_images)])
    print(f"[w44-219] grid-image subset: {len(grid_images)} of "
          f"{len(all_images)} (stride={stride})")

    # Build the LHS blob list (defaults + all lhs samples; everything not
    # tagged with `grid_`)
    lhs_or_defaults_blobs = [
        (p, sha) for (p, sha) in blob_paths
        if "grid" not in blob_provenance[sha]["source"]
    ]
    # Grid blobs (everything tagged with `grid`)
    grid_only_blobs = [
        (p, sha) for (p, sha) in blob_paths
        if blob_provenance[sha]["source"].startswith("grid")
    ]
    # Some blobs may be in both (e.g. the defaults blob); ensure we don't
    # double-count when assigning cells.
    grid_only_blob_shas = {sha for (_, sha) in grid_only_blobs}
    lhs_only_blob_shas = {sha for (_, sha) in lhs_or_defaults_blobs}
    print(f"[w44-219] LHS-or-defaults blobs: {len(lhs_or_defaults_blobs)}")
    print(f"[w44-219] grid-only blobs: {len(grid_only_blobs)}")

    cells = []
    cell_id = 0
    grid_image_set = set(str(g) for g in grid_images)

    # W44-217 finding #1: "6 RuntimeTuning params ONLY affect zenjxl
    # strategy. libjxl CV ≤ 0.03 %." → sweeping libjxl × non-default
    # blobs is wasted compute. W44-219 runs libjxl only on the defaults
    # blob (small sanity / cross-strategy regression check, ~37 × 5 × 7
    # = 1295 libjxl cells across all images at defaults).
    defaults_sha = sha256_hex(encode_postcard_tuning(DEFAULTS))

    for img_path in all_images:
        container_img = img_to_container[str(img_path)]
        is_grid_image = str(img_path) in grid_image_set
        # Which blobs to assign to this image (for zenjxl):
        if is_grid_image:
            assigned_zen_blobs = blob_paths  # full coverage
        else:
            assigned_zen_blobs = lhs_or_defaults_blobs

        for effort in EFFORTS:
            for distance in DISTANCES:
                # zenjxl × all assigned blobs
                for blob_container_path, blob_sha in assigned_zen_blobs:
                    cell_id += 1
                    cells.append({
                        "sweep_id": args.sweep_id,
                        "chunk_claim_id": f"c{cell_id:07d}",
                        "image_path": container_img,
                        "effort": effort,
                        "distance": float(distance),
                        "strategy": "zenjxl",
                        "params_blob_path": blob_container_path,
                        "threads": 4,
                        "metric_backend": "auto",
                    })
                # libjxl × ONLY defaults blob (per W44-217 finding #1)
                cell_id += 1
                cells.append({
                    "sweep_id": args.sweep_id,
                    "chunk_claim_id": f"c{cell_id:07d}",
                    "image_path": container_img,
                    "effort": effort,
                    "distance": float(distance),
                    "strategy": "libjxl",
                    "params_blob_path": f"/sweep-state/params/{defaults_sha}.bin",
                    "threads": 4,
                    "metric_backend": "auto",
                })
    print(f"[w44-219] {len(cells)} cells total")
    grid_zen = len(grid_images) * len(blob_paths) * 5 * 7
    nongrid_zen = (len(all_images) - len(grid_images)) * len(lhs_or_defaults_blobs) * 5 * 7
    libjxl_total = len(all_images) * 5 * 7
    print(f"[w44-219]   grid_zenjxl   = {grid_images and len(grid_images) or 0} imgs × "
          f"{len(blob_paths)} blobs × 35 (5×7) = {grid_zen}")
    print(f"[w44-219]   nongrid_zenjxl = {len(all_images) - len(grid_images)} imgs × "
          f"{len(lhs_or_defaults_blobs)} blobs × 35 = {nongrid_zen}")
    print(f"[w44-219]   libjxl_defaults = {len(all_images)} imgs × 35 = {libjxl_total}")

    # ── 4. Shuffle & chunk ───────────────────────────────────────────
    rng.shuffle(cells)
    n_chunks = math.ceil(len(cells) / args.chunk_size)
    for i in range(n_chunks):
        chunk = cells[i * args.chunk_size : (i + 1) * args.chunk_size]
        chunk_file = out / "chunks" / f"chunk-{i:06d}.json"
        with chunk_file.open("w") as f:
            for c in chunk:
                f.write(json.dumps(c) + "\n")
    print(f"[w44-219] wrote {n_chunks} chunks of {args.chunk_size} cells each")

    # ── 5. Manifest + provenance ─────────────────────────────────────
    manifest = out / "manifest.tsv"
    with manifest.open("w") as f:
        f.write("key\tvalue\n")
        f.write(f"sweep_id\t{args.sweep_id}\n")
        f.write(f"n_cells\t{len(cells)}\n")
        f.write(f"n_chunks\t{n_chunks}\n")
        f.write(f"chunk_size\t{args.chunk_size}\n")
        f.write(f"n_blobs\t{len(blob_paths)}\n")
        f.write(f"n_lhs_blobs\t{len(lhs_samples)}\n")
        f.write(f"n_grid_pairs\t{len(PAIR_GRIDS)}\n")
        f.write(f"n_grid_blobs_total\t{len(grid_only_blob_shas)}\n")
        f.write(f"n_grid_images\t{len(grid_images)}\n")
        f.write(f"n_corpus_images\t{len(img_to_container)}\n")
        f.write(f"n_w44_216_images\t{len(w216_specs)}\n")
        f.write(f"n_new_w44_219_images\t{len(w219_extra_specs)}\n")
        f.write(f"n_efforts\t{len(EFFORTS)}\n")
        f.write(f"n_distances\t{len(DISTANCES)}\n")
        f.write(f"n_strategies\t{len(STRATEGIES)}\n")
        f.write(f"lhs_seed\t{args.lhs_seed}\n")
        f.write(f"grid_image_seed\t{args.grid_image_seed}\n")
    print(f"[w44-219] manifest at {manifest}")

    # Provenance TSV: which blob came from where
    prov_path = out / "blob_provenance.tsv"
    with prov_path.open("w") as f:
        cols = ["sha256_short", "source"] + PARAM_ORDER
        f.write("\t".join(cols) + "\n")
        for sha, info in blob_provenance.items():
            row = [sha[:16], info["source"]]
            for k in PARAM_ORDER:
                row.append(f"{info['values'][k]:.4f}")
            f.write("\t".join(row) + "\n")
    print(f"[w44-219] blob provenance at {prov_path}")


if __name__ == "__main__":
    main()
