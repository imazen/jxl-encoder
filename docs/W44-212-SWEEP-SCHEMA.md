# W44-212 sweep cell schema

Canonical Parquet column schema emitted by
[`zenjxl-tuning-runner`](../zenjxl-tuning-runner/) per cell.

Source of truth: `zenjxl-tuning-runner/src/parquet_writer.rs::build_schema`.
This doc tracks the schema for human review + downstream MLP training
contracts.

## Versioning

`schema_version: u32` (first column). Bumped on any additive or
breaking column change. Downstream merge tools assert the version.

| version | date       | change                                              |
|---------|------------|-----------------------------------------------------|
| 1       | 2026-05-22 | Initial W44-212 schema (43 cols)                    |

## Columns (v1)

### Identity / provenance

| col              | type | null  | source                                          |
|---               |---   |---    |---                                              |
| `schema_version` | u32  | false | const `SCHEMA_VERSION` in `parquet_writer.rs`   |
| `sweep_id`       | utf8 | false | from cell spec                                  |
| `chunk_claim_id` | utf8 | false | from cell spec (worker-assigned atomic claim)   |
| `image_sha256`   | utf8 | false | hex sha256 of RGB pixel bytes (NOT PNG bytes)   |
| `image_path`     | utf8 | false | absolute disk path on worker                    |
| `image_w`        | u32  | false | from PNG header                                 |
| `image_h`        | u32  | false | from PNG header                                 |

### Inputs

| col            | type   | null  | source                                                       |
|---             |---     |---    |---                                                           |
| `effort`       | u8     | false | from cell spec (1..=12)                                      |
| `distance`     | f32    | false | from cell spec (> 0.0; lossless = separate runner)           |
| `strategy`     | utf8   | false | from cell spec: zenjxl / libjxl / lean-faster / aggressive   |
| `params_blob`  | binary | false | postcard bytes the runner intended to install (RuntimeTuning) |

### Features

Computed from sRGB u8 RGB pixels via
`jxl_encoder::__pre_quantized::ZenanalyzeProxies::compute_srgb_u8` +
the extended-features helper in `zenjxl-tuning-runner/src/features.rs`.

All `f32`, non-null.

| col                       | source / shape                                                                    |
|---                        |---                                                                                |
| `feat_m3_colourfulness`   | Hasler-Süsstrunk M3 (W44-91 / W44-98 / W44-99 discriminator)                      |
| `feat_fcbr`               | Flat-color-block-ratio (W44-91 / W44-96 fcbr discriminator)                       |
| `feat_edge_density`       | Sobel \|∇Y\| > 30 ratio (W44-96 discriminator)                                    |
| `feat_luma_var`           | BT.601 luma variance (W44-176 terminal-class discriminator)                       |
| `feat_mask_p25`           | 25th-percentile of 8×8-block mean of `1/(log1p(|ΔY|) + 0.01)` shape               |
| `feat_mask_median`        | 50th-percentile of same; W22-1 / W44-29 / W44-65 / W44-168 discriminator          |
| `feat_mask_p75`           | 75th-percentile (tail diagnostic)                                                 |
| `feat_luma_mean`          | Mean BT.601 luma                                                                   |
| `feat_n_pixels`           | width × height (f32 for column dtype stability)                                   |
| `feat_aspect`             | width / height                                                                     |
| `feat_bpp_source`         | bytes-per-pixel of source (3.0 for now; future Rgba8 → 4.0)                       |
| `feat_byte_entropy_bits`  | per-byte Shannon entropy on the source pixel buffer                                |

### Output

| col             | type | null  | source                                                  |
|---              |---   |---    |---                                                      |
| `encoded_bytes` | u32  | false | length of the JXL bytes produced                        |

### Quality

| col                   | type | null  | source                                                  |
|---                    |---   |---    |---                                                      |
| `ssim2`               | f32  | true  | SSIMULACRA2; null if backend skipped                    |
| `ssim2_backend`       | utf8 | false | `gpu-cli-ssim2` / `cpu-ssimulacra2` / `skip` / `error`  |
| `butter_norm3`        | f32  | true  | Butteraugli pnorm-3                                     |
| `butter_norm3_backend`| utf8 | false | `gpu-cli-butteraugli-pnorm3` / `cpu-butteraugli-norm3` / `skip` |
| `cvvdp`               | f32  | true  | CVVDP; null on CPU-only path (no Rust CVVDP yet)        |
| `cvvdp_backend`       | utf8 | false | `gpu-cli-cvvdp` / `cpu-unsupported` / `skip`            |

### CPU cost

| col                  | type | null  | source                                          |
|---                   |---   |---    |---                                              |
| `encode_ms`          | f64  | false | wall time via `Instant::elapsed()`              |
| `encode_user_ms`     | u64  | false | `getrusage(RUSAGE_SELF)::ru_utime` delta        |
| `encode_sys_ms`      | u64  | false | `getrusage(RUSAGE_SELF)::ru_stime` delta        |
| `encode_peak_rss_mb` | u32  | false | `ru_maxrss` post-encode (Linux KiB → MiB)       |
| `encode_threads`     | u8   | false | `std::thread::available_parallelism()`          |
| `decode_ms`          | f64  | false | wall time of jxl-rs decode                      |
| `decode_peak_rss_mb` | u32  | false | `ru_maxrss` post-decode                         |

### GPU cost

| col                | type | null  | source                                            |
|---                 |---   |---    |---                                                |
| `gpu_peak_vram_mb` | u32  | false | sum of per-metric `gpu_peak_vram_mb` from zen-metrics |
| `gpu_kernel_ms`    | f64  | false | sum of per-metric `gpu_kernel_ms` from zen-metrics  |

### Provenance

| col              | type | null  | source                                                 |
|---               |---   |---    |---                                                     |
| `runner_host`    | utf8 | false | `$HOSTNAME`                                            |
| `gpu_model`      | utf8 | false | `$W44_212_GPU_MODEL` (set by onstart script)           |
| `commit_sha`     | utf8 | false | `option_env!("GIT_COMMIT")` at build time              |
| `runner_version` | utf8 | false | `env!("CARGO_PKG_VERSION")`                            |

## Downstream join keys

Per CLAUDE.md "Separate concerns" rule, the ideal downstream layout is:

- **encode_table**: `(image_sha256, effort, distance, strategy,
  params_blob_hash) → encoded_bytes + encode_ms + ...` (what
  W44-212 emits)
- **metric_table**: `(image_sha256, encode_sha256) → ssim2 +
  butter_norm3 + cvvdp` (separate sidecar; reusable across sweeps)
- **feature_table**: `(image_sha256) → feat_*` (one row per image,
  joined to many encode rows)

W44-212 currently emits a wide table (all in one row per cell) for
simplicity. W44-213+ can split into sidecars once we have multiple
metric backends per cell.

## Tuning-axis variance

When training a per-image picker, the relevant axes are:
- `effort, distance, strategy` — always varied
- `params_blob` — varied IFF the cell spec sets `params_blob_path`.
  Note the W44-211 RuntimeTuning override is a NO-OP at the
  production encoder until W44-213+ wires consumers — see
  `crate::SCAFFOLDING_NOTE`. Until then, treat `params_blob`
  variance as zero for the W44-211 fields (the bytes are captured
  for join purposes but don't influence encoded output).
- `feat_*` — joined per image; the picker learns from these.

## Failure-row format

If a cell errors during encode or decode, the runner exits non-zero
WITHOUT emitting a Parquet row. The chunk worker logs a JSON line to
the per-chunk summary log:

```
{"status":"err","sweep_id":"...","chunk_claim_id":"...","error":"encode: ..."}
```

The downstream merge step should consult this NDJSON alongside the
Parquet rows to track which cells failed and why.
