# libjxl byte-parity sweep — 2026-09-17 baseline

Baseline for the `strict-libjxl-bitexact` work: `EncoderStrategy::Libjxl`
(`cjxl-rs --strategy libjxl`) vs cjxl **v0.12.0** (`4279d48`, AVX2 build).
Taken AFTER the two Section A gate corrections in the same commit:

- `epf_dynamic_sharpness` libjxl threshold 0 → 6 (`speed_tier <= kWombat`,
  `enc_heuristics.cc:905`). The old "libjxl has no effort gate" claim was
  wrong; parity mode ran a searched sharpness map at e1–e5 where libjxl
  emits uniform 4.
- NEW `cfl_pass1` gate: pass-1 `compute_cfl_map` now gated to e7+ under
  Libjxl (`speed_tier <= kSquirrel`, `enc_heuristics.cc:1170`); below that
  the zero-initialised cmap is emitted. Byte-identical at e5/6 (pass-2 LS
  refills every tile) and a parity fix at e1–e4.

## Result

Grid: 7 fixtures × e{3,5,7} × d{0.5,1.0,4.0} = 63 cells
(`libjxl_parity_2026-09-17.tsv`, `.meta`).

- **Envelope (signature → SizeHeader → ImageMetadata → FrameHeader → TOC):
  byte-identical on all 63 cells.** `header_and_toc` delta = 0 everywhere.
- Aggregate section payload: ours +3.6 % bytes (HfGroup +2.6 %, LfGroup
  +15.3 %, HfGlobal +26.1 %, LfGlobal +38.6 % — LF/global structures are
  the residual, not the AC payload).
- 1 cell (noise_48x48 e5 d1.0) is same-size but not byte-identical —
  envelope parses identical, divergence is inside the section payload.
- Field-level divergences (42 rows across 3 of 7 fixtures, all 512²):
  - `lfglobal.block_ctx_map.qf_thresholds` — ctx-map clustering picks
    different QF thresholds (n=1 vs 0, 14 vs 13, 5 vs 4).
  - `hfglobal.used_orders[p0]` — cjxl emits more custom coefficient
    orders (e.g. 0x0003/0x0019/0x0059) than we do (0x0000/0x0001/0x0008).
  - `*.ans_starts_at_bit` deltas are downstream of the two above.

So "bit-exact" is currently: envelope ✓, section payload ✗. The remaining
work to close is in ctx-map clustering, coefficient-order selection, and
whatever drives the LfGroup/LfGlobal payload deltas — all data-dependent
algorithm details, not gate coverage.

## Reproduce

    CJXL=/home/lilith/tmp/libjxl-v012-build/tools/cjxl \
    OURS=target/release/cjxl-rs \
    scripts/libjxl_parity_sweep.sh <outdir>

    # per-cell field diff:
    python3 scripts/jxl_bitstream_diff.py diff cjxl.jxl ours.jxl

    # parity-mode timing (new `--ours-flags` passthrough):
    python3 scripts/ladder_vs_cjxl.py --ours target/release/cjxl-rs \
        --cjxl <v0.12 cjxl> --out parity_ladder.tsv \
        --ours-flags '--strategy libjxl' <pngs...>

Note: `cjxl` on PATH (`/usr/bin/cjxl`) is v0.11.1 — NOT the reference.
