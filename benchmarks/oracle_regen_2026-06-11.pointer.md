# Oracle regeneration 2026-06-11 — pointer (issue #25 follow-on A prep)

Per the STALE-DATA RULE (every zen encoder changed → the 2026-04-30 oracle
TSVs are dead for training), the lossy picker oracle was regenerated
overnight on the two Hetzner aarch64 boxes with current-main binaries
(`81d28e16` archive + the rotted-example fix shipped in `dd6b393f`).

Harness: `examples/lossy_pareto_calibrate.rs` (`--features __expert`,
on-box `[[example]]` + sibling-zenanalyze wiring), manifest
`picker-train/manifest_v1_100.tsv` (100 stratified images), distances
{0.25,0.5,0.75,1.0,1.25,1.5,2.0,2.5,3.0,3.5,4.0,5.0,6.5,8.0} (low-q-dense
per the sweep discipline), 8 categorical cells × (1 anchor + 5 scalar
samples).

## Data

- **arm-zen, `--sizes 256` axis: COMPLETE** — 134,400 rows + 100-row
  feature sidecar.
  - `/mnt/v/output/jxl-encoder/oracle-2026-06-11/arm-zen-256/`
    `lossy_pareto_256_2026-06-11.tsv` (21,710,921 B,
    sha256 f6f7ea3e1a057857…), `lossy_pareto_features_256_2026-06-11.tsv`.
- **arm-big, `--sizes native` axis: IN FLIGHT** at collection time
  (40/100 images, ETA ~08:20 local) — lands at
  `arm-big:~/oracle/lossy_pareto_native_2026-06-11.tsv`; collect with
  `rsync -a arm-big:oracle/ /mnt/v/output/jxl-encoder/oracle-2026-06-11/arm-big-native/`.

## Caveats (binding)

- `encode_ms` columns: self-contended ARM boxes — never feed a wall model
  (REVISIT_QUEUE_2026-06-11.md #5). Bytes + butteraugli + ssim2 are
  deterministic and sound.
- aarch64 bytes may differ ±tiny from x86_64 on clustering near-ties
  (issue #70): keep A/B comparisons within-box; rank-based training is
  unaffected at the measured 6-byte scale.
- Tower mirror queued (rsync to /mnt/tower/output/jxl-encoder/ before any
  cleanup of the /mnt/v copies).
