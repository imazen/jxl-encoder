#!/usr/bin/env python3
"""Build Stage B chunk specs + params blobs for the W44-215 full-grid sweep.

Sweep design:
  - N_IMAGES from 2 corpora (photos: CID22; screenshots: gb82-sc).
  - EFFORTS x DISTANCES x STRATEGIES per image = base grid.
  - PARAMS_VARIANTS per base cell = Latin-hypercube samples over the
    6 W44-211 RuntimeTuning fields.
  - cells grouped into CHUNK_SIZE-cell chunks (default 50) for
    parallel claim by fleet workers.

Outputs (under OUT_DIR, default /tmp/w44-215-stage-b):
  - chunks/chunk-NNNNN.json (NDJSON, CHUNK_SIZE cells each)
  - params/<blob-sha256>.bin (postcard-encoded RuntimeTuning override)
  - corpus/<basename>.png (deduped images, ready to upload to R2 corpus/)
  - manifest.tsv (sweep_id, n_cells, n_chunks, n_params_blobs, n_corpus_images)

Then upload to R2:
  aws s3 cp --recursive OUT_DIR/corpus/  s3://zen-tuning-ephemeral/corpus/
  aws s3 cp --recursive OUT_DIR/params/  s3://zen-tuning-ephemeral/$SWEEP_ID/params/
  aws s3 cp --recursive OUT_DIR/chunks/  s3://zen-tuning-ephemeral/$SWEEP_ID/chunks/

Then launch the fleet via launch_w44_215_fleet.sh.
"""
import argparse
import hashlib
import json
import math
import random
import struct
import sys
from pathlib import Path

# ── RuntimeTuning field defaults (must match jxl_encoder::tuning::runtime
#    production source-of-truth consts. Verified 2026-05-22 against
#    jxl-encoder/src/tuning.rs + vardct/butteraugli_loop.rs).
DEFAULTS = {
    "smart_zenjxl_photo_mask_p25_min": 85.0,
    "screenshot_median_threshold": 95.0,
    "buttloop_default_screenshot_qf_seed_scale": 4.0,
    "buttloop_qf_seed_scale_min_distance": 3.5,
    "adaptive_quant_screenshot_qf_seed_scale_e5_e6": 2.0,
    "adaptive_quant_screenshot_qf_seed_scale_e7": 3.0,
}

# Sweep ranges per field. Latin-hypercube samples within these bounds.
# Bounds chosen to bracket the production defaults 2-4x in each direction
# so the MLP can learn the local response surface.
RANGES = {
    "smart_zenjxl_photo_mask_p25_min": (40.0, 200.0),       # default 85
    "screenshot_median_threshold": (75.0, 110.0),           # default 95
    "buttloop_default_screenshot_qf_seed_scale": (1.0, 8.0),   # default 4
    "buttloop_qf_seed_scale_min_distance": (1.5, 5.5),       # default 3.5
    "adaptive_quant_screenshot_qf_seed_scale_e5_e6": (1.0, 4.0),  # default 2
    "adaptive_quant_screenshot_qf_seed_scale_e7": (1.5, 5.5),     # default 3
}

# Sweep grid (per-image)
EFFORTS = [5, 6, 7, 8]
DISTANCES = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
STRATEGIES = ["zenjxl"]  # libjxl/aggressive are spot-checks not full grid


def latin_hypercube(n_samples: int, n_dims: int, seed: int) -> list[list[float]]:
    """Standard LHS in [0, 1]^n_dims."""
    rng = random.Random(seed)
    samples = [[0.0] * n_dims for _ in range(n_samples)]
    for d in range(n_dims):
        perm = list(range(n_samples))
        rng.shuffle(perm)
        for i in range(n_samples):
            samples[i][d] = (perm[i] + rng.random()) / n_samples
    return samples


def encode_postcard_tuning(values: dict[str, float]) -> bytes:
    """Encode a RuntimeTuning struct as postcard.

    RuntimeTuning postcard layout is just 6 little-endian f32 in field-
    declaration order (no length prefix, no tag — postcard struct
    serialisation is field-concatenation by default).
    Verified by W44-214 smoke (blobs were 24 bytes = 6 * 4).
    """
    order = [
        "smart_zenjxl_photo_mask_p25_min",
        "screenshot_median_threshold",
        "buttloop_default_screenshot_qf_seed_scale",
        "buttloop_qf_seed_scale_min_distance",
        "adaptive_quant_screenshot_qf_seed_scale_e5_e6",
        "adaptive_quant_screenshot_qf_seed_scale_e7",
    ]
    out = b""
    for k in order:
        out += struct.pack("<f", float(values[k]))
    assert len(out) == 24, len(out)
    return out


def sha256_hex(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-id", required=True, help="e.g. w44-215-fullgrid-2026-05-22")
    ap.add_argument("--out-dir", default="/tmp/w44-215-stage-b")
    ap.add_argument("--corpus-photo-dir", default="/home/lilith/work/codec-corpus/CID22/CID22-512/validation")
    ap.add_argument("--corpus-screen-dir", default="/home/lilith/work/codec-corpus/gb82-sc")
    ap.add_argument("--n-photos", type=int, default=8)
    ap.add_argument("--n-screens", type=int, default=4)
    ap.add_argument("--params-variants", type=int, default=8,
                    help="Latin-hypercube param-variant count per base cell. 0 = default only")
    ap.add_argument("--include-default-params", action="store_true",
                    help="Add a no-override variant (defaults) alongside the LHS samples")
    ap.add_argument("--chunk-size", type=int, default=20, help="cells per chunk file")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out_dir)
    (out / "chunks").mkdir(parents=True, exist_ok=True)
    (out / "params").mkdir(parents=True, exist_ok=True)
    (out / "corpus").mkdir(parents=True, exist_ok=True)

    # ── pick images ─────────────────────────────────────────────────
    photo_dir = Path(args.corpus_photo_dir)
    screen_dir = Path(args.corpus_screen_dir)
    photos = sorted(p for p in photo_dir.glob("*.png"))[: args.n_photos]
    screens = sorted(p for p in screen_dir.glob("*.png"))[: args.n_screens]
    if not photos:
        print(f"ERROR: no photos in {photo_dir}", file=sys.stderr); sys.exit(1)
    if not screens:
        print(f"ERROR: no screens in {screen_dir}", file=sys.stderr); sys.exit(1)
    images = list(photos) + list(screens)
    print(f"[stage-b] {len(images)} images ({len(photos)} photo + {len(screens)} screen)")

    # Stage corpus (copy to out/corpus/ with deduped basenames)
    img_to_container = {}
    for src in images:
        dst = out / "corpus" / src.name
        if not dst.exists():
            dst.write_bytes(src.read_bytes())
        img_to_container[str(src)] = f"/corpus/{src.name}"
    print(f"[stage-b] staged {len(img_to_container)} images to {out / 'corpus'}")

    # ── generate params blobs ───────────────────────────────────────
    rng = random.Random(args.seed)
    blob_paths = []  # list of (container_path, sha)
    if args.include_default_params:
        b = encode_postcard_tuning(DEFAULTS)
        sha = sha256_hex(b)
        (out / "params" / f"{sha}.bin").write_bytes(b)
        blob_paths.append((f"/sweep-state/params/{sha}.bin", sha))
    if args.params_variants > 0:
        lhs = latin_hypercube(args.params_variants, len(RANGES), args.seed)
        order = list(RANGES.keys())
        for sample in lhs:
            values = {}
            for k, u in zip(order, sample):
                lo, hi = RANGES[k]
                values[k] = lo + u * (hi - lo)
            b = encode_postcard_tuning(values)
            sha = sha256_hex(b)
            (out / "params" / f"{sha}.bin").write_bytes(b)
            blob_paths.append((f"/sweep-state/params/{sha}.bin", sha))
    # de-dup (LHS can collide on tiny variant counts)
    blob_paths = sorted(set(blob_paths))
    print(f"[stage-b] {len(blob_paths)} unique params blobs (incl default={args.include_default_params})")

    # ── enumerate cells ─────────────────────────────────────────────
    cells = []
    cell_id = 0
    for img_path in images:
        container_img = img_to_container[str(img_path)]
        for effort in EFFORTS:
            for distance in DISTANCES:
                for strategy in STRATEGIES:
                    for blob_container_path, blob_sha in blob_paths:
                        cell_id += 1
                        cells.append({
                            "sweep_id": args.sweep_id,
                            "chunk_claim_id": f"c{cell_id:06d}",
                            "image_path": container_img,
                            "effort": effort,
                            "distance": float(distance),
                            "strategy": strategy,
                            "params_blob_path": blob_container_path,
                            "threads": 4,
                            "metric_backend": "auto",
                        })
    print(f"[stage-b] {len(cells)} cells total")

    # shuffle so chunks are heterogeneous (avoids one chunk being all
    # expensive screenshot+e8 cells)
    rng.shuffle(cells)

    # ── write chunks ────────────────────────────────────────────────
    n_chunks = math.ceil(len(cells) / args.chunk_size)
    for i in range(n_chunks):
        chunk = cells[i * args.chunk_size : (i + 1) * args.chunk_size]
        chunk_file = out / "chunks" / f"chunk-{i:05d}.json"
        with chunk_file.open("w") as f:
            for c in chunk:
                f.write(json.dumps(c) + "\n")
    print(f"[stage-b] wrote {n_chunks} chunks of {args.chunk_size} cells each")

    # ── manifest ────────────────────────────────────────────────────
    manifest = out / "manifest.tsv"
    with manifest.open("w") as f:
        f.write("key\tvalue\n")
        f.write(f"sweep_id\t{args.sweep_id}\n")
        f.write(f"n_cells\t{len(cells)}\n")
        f.write(f"n_chunks\t{n_chunks}\n")
        f.write(f"chunk_size\t{args.chunk_size}\n")
        f.write(f"n_params_blobs\t{len(blob_paths)}\n")
        f.write(f"n_corpus_images\t{len(img_to_container)}\n")
        f.write(f"n_efforts\t{len(EFFORTS)}\n")
        f.write(f"n_distances\t{len(DISTANCES)}\n")
        f.write(f"include_default_params\t{args.include_default_params}\n")
        f.write(f"seed\t{args.seed}\n")
    print(f"[stage-b] manifest at {manifest}")


if __name__ == "__main__":
    main()
