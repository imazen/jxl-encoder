# W44-212 zenjxl-tuning-runner fleet

Fleet infrastructure for the W44-212 sweep runner. Companion to the
`zenjxl-tuning-runner` crate under `../../zenjxl-tuning-runner/`.

## Files

- `Dockerfile.zenjxl-tuning-sweep.v1` — vast.ai-targeted image,
  bakes the runner + zen-metrics binary + s5cmd. See header comment
  for the layer plan and build command.
- `onstart.sh` — container ENTRYPOINT. Hydrates env from
  /proc/1/environ, verifies baked tools, loops over chunks from R2.
- `worker.sh` — invoked by onstart per chunk. Reads NDJSON, runs
  the runner per cell, uploads Parquet to R2.
- `launch_fleet.sh` — host-side launcher. Spawns N vast.ai
  instances, each consuming the chunk queue atomically.
- `make_chunks.py` — host-side helper to slice a sweep TSV into
  NDJSON chunk files and push them to R2.

## R2 bucket layout

```
s3://zen-tuning-ephemeral/                   # ephemeral, 14d lifecycle
└── <sweep-id>/
    ├── chunks/<chunk-id>.json               # NDJSON cell specs, claim queue
    ├── chunks-in-flight/<chunk-id>.json     # atomic-claimed (move target)
    ├── chunks-done/<chunk-id>.json          # post-process audit trail
    ├── cells/<worker>-<chunk-claim>.parquet # per-cell output rows
    ├── corpus/<image-relpath>.png           # per-sweep source staging
    ├── params/<params-name>.postcard        # RuntimeTuning override blobs
    ├── logs/<worker>-<chunk-id>.ndjson      # per-chunk runner summary
    └── heartbeat/<host>-<ts>.txt            # worker liveness

s3://zen-corpus/                             # PERMANENT (no lifecycle)
└── <corpus-name>/<image>.png                # shared image corpus
```

## Bucket setup (one-off)

```bash
# Create the ephemeral bucket with 14-day lifecycle (Cloudflare R2):
wrangler r2 bucket create zen-tuning-ephemeral
wrangler r2 bucket lifecycle add zen-tuning-ephemeral \
    --rule-id "auto-delete-after-14d" \
    --prefix "" \
    --expiration-days 14

# Verify lifecycle is in place:
wrangler r2 bucket lifecycle list zen-tuning-ephemeral

# OR via the AWS CLI against the R2 S3 API:
aws s3api put-bucket-lifecycle-configuration \
    --endpoint-url "https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com" \
    --bucket zen-tuning-ephemeral \
    --lifecycle-configuration '{
        "Rules": [{
            "ID": "auto-delete-after-14d",
            "Status": "Enabled",
            "Filter": {"Prefix": ""},
            "Expiration": {"Days": 14}
        }]
    }'
```

The corpus bucket has NO lifecycle — it's the permanent shared
image store, mirrored from `~/work/codec-corpus/` once.

## End-to-end sweep workflow

1. **Generate the sweep cell list** (Python / Rust harness):

   ```python
   # scripts/zenjxl-tuning-sweep/example_sweep.py
   import json, hashlib
   from pathlib import Path
   from itertools import product

   SWEEP_ID = "W44-XYZ-buttloop-scan"
   IMAGES = sorted(Path("~/work/codec-corpus/CID22/CID22-512/training").expanduser().glob("*.png"))[:20]
   EFFORTS = [5, 7]
   DISTANCES = [0.5, 1.0, 2.0, 4.0]
   PARAMS_BLOBS = None  # or a list of postcard paths

   cells = []
   for img, e, d in product(IMAGES, EFFORTS, DISTANCES):
       sha = hashlib.sha256(img.read_bytes()).hexdigest()[:16]
       cells.append({
           "sweep_id": SWEEP_ID,
           "chunk_claim_id": f"{img.stem}-e{e}-d{d}-{sha}",
           "image_path": f"/corpus/cid22/{img.name}",
           "image_sha256": hashlib.sha256(img.read_bytes()).hexdigest(),
           "effort": e,
           "distance": d,
           "strategy": "zenjxl",
           "metric_backend": "auto",
       })

   # Chunk into NDJSON files of 50 cells each
   for i in range(0, len(cells), 50):
       chunk = cells[i:i+50]
       Path(f"/tmp/{SWEEP_ID}-{i//50:04d}.json").write_text(
           "\n".join(json.dumps(c) for c in chunk)
       )
   ```

2. **Push corpus + chunks + params to R2**:

   ```bash
   # Corpus (once, permanent)
   s5cmd sync ~/work/codec-corpus/CID22/CID22-512/training/ \
       s3://zen-corpus/cid22/

   # Chunks (per-sweep)
   s5cmd sync /tmp/W44-XYZ-buttloop-scan-*.json \
       s3://zen-tuning-ephemeral/W44-XYZ-buttloop-scan/chunks/

   # Params (if running with RuntimeTuning override)
   s5cmd sync /tmp/W44-XYZ-params/ \
       s3://zen-tuning-ephemeral/W44-XYZ-buttloop-scan/params/
   ```

3. **Launch fleet**:

   ```bash
   ./launch_fleet.sh \
       --sweep-id W44-XYZ-buttloop-scan \
       --num-instances 8 \
       --gpu-type "RTX 3090"
   ```

4. **Monitor progress**:

   ```bash
   # In-flight chunks (workers still processing)
   s5cmd ls s3://zen-tuning-ephemeral/W44-XYZ-buttloop-scan/chunks-in-flight/ | wc -l

   # Completed cells
   s5cmd ls s3://zen-tuning-ephemeral/W44-XYZ-buttloop-scan/cells/ | wc -l

   # Per-chunk summary logs
   s5cmd cat s3://zen-tuning-ephemeral/W44-XYZ-buttloop-scan/logs/'*.ndjson' | jq -c .status | sort | uniq -c
   ```

5. **Merge cells into one Parquet** (W44-213 coordinator, not in
   this chunk):

   ```bash
   # Download all per-cell Parquet files
   s5cmd sync 's3://zen-tuning-ephemeral/W44-XYZ-buttloop-scan/cells/*.parquet' /tmp/W44-XYZ-cells/

   # Merge via pyarrow into one canonical file
   python3 -c "
   import pyarrow.parquet as pq
   import pyarrow as pa
   from pathlib import Path
   tables = [pq.read_table(str(p)) for p in Path('/tmp/W44-XYZ-cells').glob('*.parquet')]
   combined = pa.concat_tables(tables)
   pq.write_table(combined, '/tmp/W44-XYZ-merged.parquet', compression='zstd', compression_level=5)
   "

   # Push merged to canonical zentrain bucket
   s5cmd cp /tmp/W44-XYZ-merged.parquet \
       s3://zentrain/zenjxl-tuning/2026-05-22/W44-XYZ-buttloop-scan.parquet
   ```

## Local smoke test

Skip the fleet and run a single cell on your dev box:

```bash
mkdir -p /tmp/w44-212-smoke/output
cp ~/work/codec-corpus/CID22/CID22-512/training/144200.png /tmp/w44-212-smoke/test.png

zenjxl-tuning-runner \
    --cell '{"sweep_id":"smoke","chunk_claim_id":"c1",
             "image_path":"/tmp/w44-212-smoke/test.png",
             "effort":7,"distance":1.0,"strategy":"zenjxl",
             "metric_backend":"skip"}' \
    --output /tmp/w44-212-smoke/output/c1.parquet \
    --verbose

# Inspect:
python3 -c "
import pyarrow.parquet as pq
t = pq.read_table('/tmp/w44-212-smoke/output/c1.parquet')
for col in t.column_names:
    print(f'{col}: {t.column(col)[0].as_py()}')
"
```

## Schema

See `../../zenjxl-tuning-runner/src/parquet_writer.rs::build_schema`
for the canonical 43-column Arrow schema. Bumped via
`SCHEMA_VERSION` const.

Column groups:
- **Identity** (8): schema_version, sweep_id, chunk_claim_id,
  image_sha256, image_path, image_w, image_h
- **Inputs** (4): effort, distance, strategy, params_blob
- **Features** (12): m3_colourfulness, fcbr, edge_density, luma_var,
  mask_p25/median/p75, luma_mean, n_pixels, aspect, bpp_source,
  byte_entropy_bits
- **Output** (1): encoded_bytes
- **Quality** (6): ssim2 + ssim2_backend, butter_norm3 +
  butter_norm3_backend, cvvdp + cvvdp_backend
- **Cost** (9): encode_ms, encode_user_ms, encode_sys_ms,
  encode_peak_rss_mb, encode_threads, decode_ms,
  decode_peak_rss_mb, gpu_peak_vram_mb, gpu_kernel_ms
- **Provenance** (4): runner_host, gpu_model, commit_sha, runner_version

## Known limitations

1. **RuntimeTuning override is a no-op at the encoder** (W44-211
   shipped scaffolding only). The `params_blob` column captures the
   postcard payload the runner intended to apply; until W44-213+
   wires consumer sites, MLP training MUST treat per-cell tuning
   variance for W44-211 fields as zero. The other axes
   (effort, distance, strategy, image features) ARE actively varied.

2. **GPU CVVDP via `zen-metrics score --metric cvvdp`** is only
   available when zen-metrics is built with `--features gpu-cvvdp`
   AND a CUDA runtime is present. The CPU fallback path skips CVVDP
   entirely (column = null, backend = "cpu-unsupported").

3. **CPU metrics use the gamma-encoded sRGB pixel values** without
   linearisation in the butteraugli call. This is approximate; the
   `backend = "cpu-butteraugli-norm3"` column documents the
   approximation. GPU path uses correct linearisation via the
   zen-metrics CLI.

4. **R2 ephemeral bucket lifecycle is 14 days.** Critical results
   MUST be merged + pushed to the canonical zentrain bucket before
   the lifecycle deletes them.
