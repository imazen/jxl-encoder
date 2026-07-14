# IQA-stat Python migration to zenstats (2026-05-26)

Per the cross-repo dedup pass that landed `zenstats` (`imazen/zenmetrics@36d71ca33711`) +
the `panel` Rust binary (`imazen/zensim@032b0119`), every inline
Python reimplementation of SROCC / PLCC / KROCC / OR / PWRC / Z-RMSE
must route through the canonical Rust panel via the vendored
`scripts/lib/zen_stats.py` shim.

User directive (verbatim): "the veric should be use zenstats" — every
hand-rolled IQA stat function in Python migrates; `scipy.stats`
wrappers (proven-equivalent reference impls) stay as-is.

## Inventory (scanned 2026-05-26)

Patterns searched (`grep -rln`):

- `^def srocc | ^def spearman | ^def pearson | ^def kendall | ^def pwrc | ^def outlier_ratio | ^def z_rmse`
- `def .*spearman | def .*pearson | def .*kendall | def .*srocc | def .*plcc | def .*krocc | def .*pwrc | def .*outlier | def .*z_rmse`
- `spearmanr | pearsonr | kendalltau | from scipy`
- `rank_corr | spearman_rho | kendall_tau | monotonicity | outlier.*ratio | z.rmse | pwrc`

## Verdicts

| File | Verdict | Reason |
|---|---|---|
| `scripts/analyze_smart_fanout.py` | **MIGRATE** | Hand-rolled `def spearman(xs, ys)` at line 80 returning `(rho, n)`. Used for feature-vs-speedup rank correlation per effort. |
| `benchmarks/scripts/w44_170_analyze.py` | leave | No IQA stat code; `def per_*`, `def chart_*`, `def fmt_*` etc. are pandas-based summarizers. Broad regex false-matched. |
| `benchmarks/scripts/w44_179_quick_stats.py` | leave | No IQA stat code; `def headline`, `def per_class`, `def top_outliers` are pandas-based summarizers. |
| `benchmarks/sweeps/w44-219-densify/analysis/scripts/w44_221_phase2_basis.py` | leave | `from scipy.stats import qmc` — Latin hypercube sampling, NOT an IQA stat reimpl. |
| `benchmarks/sweeps/w44-219-densify/analysis/scripts/w44_220_fit_p1p2_best.py` | leave | `from scipy.optimize import curve_fit` — nonlinear LSQ, not an IQA stat. |
| `benchmarks/sweeps/w44-216-stage-b/analysis/scripts/w44_218_fit_couplings.py` | leave | `scipy.optimize.curve_fit` — same as above. |
| `scripts/zenjxl-tuning-sweep/build_w44_219_chunks.py` | leave | `scipy.stats.qmc.LatinHypercube` — sampling, not stats. |
| `scripts/zenjxl-tuning-sweep/build_w44_phase4_s1_chunks.py` | leave | Same. |
| `scripts/zenjxl-tuning-sweep/build_w44_229_chunks.py` | leave | Same. |
| `scripts/zenjxl-tuning-sweep/build_w44_phase4_s2_chunks.py` | leave | Same. |

Net: one file to migrate.

## Polarity convention

`zen_stats.panel` (and the underlying Rust `panel`) reports SROCC /
KROCC / PWRC as `abs(...)`, treating polarity as a nuisance per the
IQA convention. `analyze_smart_fanout.py` previously reported the
signed Spearman ρ so the analyst could see direction. Post-migration
the sign is dropped — the script's `abs(rho)`-sorted ranking is
unchanged, and the printed ρ becomes non-negative. Documented in a
comment at the migration site.

## Steps

1. Vendor `scripts/lib/zen_stats.py` (copy of
   `~/work/zen/zensim/scripts/lib/zen_stats.py`) with `_find_panel_bin`
   extended to also look in `~/work/zen/zensim/target/release/panel`
   so the binary is found from this repo's tree.
2. Migrate `scripts/analyze_smart_fanout.py`:
   - delete inline `def spearman(xs, ys)`,
   - `from scripts.lib.zen_stats import panel`,
   - call `panel(xs, ys)` and unpack `srocc` + `n`,
   - add a polarity-convention comment.
3. Smoke-test: `python3 -c "import ast; ast.parse(open(PATH).read())"`.
4. Surgical commits per step.

## Cargo build for panel binary

`cargo build --release -p zensim-validate --bin panel` in
`~/work/zen/zensim`. Verified built (5.7 MB binary present
2026-05-26).
