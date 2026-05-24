# e9 perf profile recovery — 1418519 d=1.0 e9 (2026-05-23)

Single-cell recovery profiling after a prior chunk hit rate-limit.

**Cell**: `1418519.png d=1.0 e9 --strategy zenjxl` (CID22 512×512 smooth photo,
4 buttloop iters at e9). Wall 246ms multi-thread vs cjxl 437ms single-thread.

**Method**: `perf record -F 1999` (no call-graph; FP unwinding fails in
release rayon thread stacks) across 12 sequential runs → 37,179 samples →
`perf report --no-children --percent-limit 0.5`.

## Files

- `perf_1418519_d1_e9.txt` — flat top-symbol ranking, percent-limit 0.5 (46 lines)
- `perf_1418519_d1_e9_with_callers.txt` — FP call-graph attempt; mostly unresolved
  rayon worker addresses (preserved for posterity, not a useful artifact)

## Top-3 hot families (>10% combined)

| family | % cycles |
|---|---|
| rayon / crossbeam coordination | ~30% |
| butteraugli (buttloop comparator: blur/malta/opsin) | ~13% |
| AC strategy search (`find_best_16x16_transform` etc.) | ~7% |

## Top-3 next-chunk candidates

1. **rayon coordination** (~30%) — confirm with `RAYON_NUM_THREADS=1` A/B
   before any code change. Per-block `par_iter` may have too-fine granularity.
2. **butteraugli comparator** (~13%) — already at libjxl-parity per W44-105/172;
   limited EV without risking SSIM2 wedges.
3. **AC strategy `find_best_16x16`** (~7%) — tight-scope; check if W44-170
   step=1 widening at e9 inflated invocation count vs e7.

Full memo: `~/.claude/projects/-home-lilith-work-zen-jxl-encoder/memory/e9_perf_profile_recovery_2026-05-23.md`.

## Reproducer

```bash
IMAGE=$HOME/work/codec-corpus/CID22/CID22-512/validation/1418519.png
BIN=$HOME/work/zen/jxl-encoder-shared-target/release/cjxl-rs
CARGO_TARGET_DIR=$HOME/work/zen/jxl-encoder-shared-target \
  cargo build --release --bin cjxl-rs \
  --features 'butteraugli-loop ssim2-loop parallel'
perf record -F 1999 -o /tmp/perf_e9.data -- bash -c '
  for i in $(seq 12); do
    '"$BIN"' '"$IMAGE"' /tmp/out.jxl -e 9 -d 1.0 --strategy zenjxl >/dev/null 2>&1
  done'
perf report -i /tmp/perf_e9.data --stdio --no-children --percent-limit 0.5 -g none
```
