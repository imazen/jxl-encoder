# libjxl v0.12 distance semantics, 2026-09-08

The proposed universal delivered/requested Butteraugli band 1.0–1.2 is not
libjxl v0.12 behavior: 214 of 297 reference cells fall outside it. The measured
range is 0.544–1.425. These are comparisons on normalized RGB8 input with
full-resolution linear decoded output, not comparisons between differently
tagged PNG files. All cells fully rendered in jxl-rs, djxl v0.12 and jxl-oxide.
This does not establish a new universal tolerance or an exact targeting promise.

Grid: nine sources from [the input manifest](qfseed_targeting_inputs_2026-09-08.tsv),
1024-pixel center crop cap without upscaling, efforts 5/7/8, distances
2/2.5/3/3.4/3.5/3.6/4/4.5/5/6/8. Host: mac, ARM64. Probe build
`a7cf6d8d4f06674559a755d5be9aac75acca342b`, preserved on
[the reference probe branch](https://github.com/imazen/jxl-encoder/tree/research/issue103-reference-probe-build).
The reference arm invokes cjxl v0.12; its timing includes process startup and
file I/O and must not be compared with the Rust arm's encode-only timing.

- [Per-image summary](cjxl_target_semantics_2026-09-08.tsv).
- [Raw result hashes](cjxl_target_semantics_2026-09-08.manifest.tsv).
- Local results: `/Users/lilith/tmp/jxl103/cjxl-target-results-2026-09-08/`.
- Local encoded bytes, diffmaps and all p-norms:
  `/Users/lilith/tmp/jxl103/cjxl-target-2026-09-08/`.
- Backup destination: `s3://zen-tuning-ephemeral/jxl-encoder/issue103-cjxl-reference-2026-09-08/`.
- Tower destination: `/mnt/tower/output/jxl-encoder/issue103-cjxl-reference-2026-09-08/`.
  All 1,622 object sizes matched R2; three samples from each of the three
  directories passed SHA256 checks after downloading from R2 and reading
  Tower. [Verification record](cjxl_target_semantics_2026-09-08.remote.json).

Reproduce the summary with
`python3 scripts/qfseed_lift_ab_analyze.py <result-directory> --reference-summary`.
The result directory's `meta.json` records the full grid and executable SHA256;
per-cell metadata identifies the normalized source and reference tool.

The separate [low-distance filter boundary comparison](cjxl_filter_boundary_2026-09-08.tsv)
also reproduces a size increase as distance crosses 0.5 in C++.
Global byte monotonicity across those reference filter changes is not a parity
expectation. That inherited behavior does not excuse the additional seed-lift
cliff at distance 3.5: [the four-policy ablation](qfseed_unlifted_2026-09-08.pointer.md)
removes every measured >10% increase on steps starting at distance 2 or above.
