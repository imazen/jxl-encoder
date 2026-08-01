# zensim-loop targeting studies — runners

Measurement runners for the zensim qf-targeting loop studies (committed so
the tooling outlives scratch dirs):

- `run69.sh` — the #69 loop-steering matrix (controller × redistribution
  arms). Study doc: `benchmarks/zensim_attr_loop69_2026-07-29.md`. Rescued
  from `~/tmp/attrmap-69/` on 2026-07-31.
- `run_eff.sh` — the loop-EFFICIENCY study (iterations-to-tolerance, byte
  cost of tolerance, judged calibration, bytes targeting). Pre-registered
  protocol: `benchmarks/zensim_diffmap_efficiency_2026-07-31.md`. Phases
  `r0`..`r5` (or `all`); `r0` is a hard identity gate and `r1` embeds the
  arm-engagement gate (unknown `JXL_ZENSIM_MODEL_MAP` values silently fall
  through to baseline — always prove engagement in-run).

Both drive `jxl-encoder/examples/zensim_diffmap_rd.rs` (build command in
`run_eff.sh`'s header). Binary path override: `ZDR_BIN`. Heavy work is
nice'd; logs land in the runner's `$OUT`.
