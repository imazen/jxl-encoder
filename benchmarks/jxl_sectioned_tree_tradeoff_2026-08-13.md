# Sectioned (local-tree) lossless: measured bytes/memory tradeoff at 4K

De-risks the C-parity architecture for lossless memory: encode the image as
independent tiles (each with its own MA tree + histograms) and measure total
bytes + worst per-tile peak_live vs the whole-image global-tree encode.
Zero encoder changes — tiles cut from the raw bin, each fed to
zenjxl `mem_probe_encode` lossless e7/e9, threads=1, jxl-encoder 3e237d11+.
Tile sums approximate a single-frame encode whose PassGroup modular streams
set `use_global_tree = false` per cluster (per-tile container overhead makes
these byte totals slightly PESSIMISTIC).

| cell | whole-image | 4 tiles (~1920x1080) | 15 tiles (~768x720) |
|---|---|---|---|
| photo e7 bytes | 4,436,537 | 4,446,623 (+0.23%) | 4,369,545 (**-1.51%**) |
| photo e7 peak_live | 834 MB | **216 MB** | 63 MB |
| photo e9 bytes | 4,246,312 | 4,231,781 (**-0.34%**) | — |
| photo e9 peak_live | 1130 MB | **391 MB** | — |
| screen e7 bytes | 305,873 | 337,077 (+10.2%) | 371,608 (+21.5%) |
| screen e7 peak_live | 894 MB | **219 MB** | 62 MB |

Reference points: cjxl 0.12 (streaming default) at the same cells — photo e7
4,478,351 B / 304 MB RSS; photo e9 4,392,315 / 326; screen e7 478,918 / 268.

Findings:

1. **~1080p clusters reach C-parity memory** (216-391 MB live, at or under
   cjxl's RSS) while REMAINING SMALLER than cjxl's output on every cell —
   photo tiled 4,446,623 < cjxl 4,478,351; screen tiled 337,077 < 478,918.
2. On photos, local trees FIT BETTER than one global tree: 15 tiles code
   1.5% fewer bytes; e9 4-tile is 0.34% smaller. The global tree is not
   free lunch even on bytes.
3. Screens pay +10-21% vs our own global tree (global patterns matter) —
   still far ahead of cjxl. A cluster-size knob (or content-adaptive
   clustering) belongs in the design.
4. Per-tile WP/context resets are included in these numbers (tiles are
   fully independent) — the in-frame implementation can only do better.

Implementation shape (single-frame, spec-conformant): per-PassGroup modular
streams signal `use_global_tree = false` and carry their own tree +
histograms — the machinery the single-group path already uses — grouped
into ~8x4-group clusters sharing one learned tree. Gates: byte-accounting
vs this table, jxl-rs + djxl roundtrip on every cell, hash-locks untouched
(mode off by default; auto-engages under Limits memory budgets).
