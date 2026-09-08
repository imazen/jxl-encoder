# Native JXL finite-block interventions — September 8, 2026

D’s strong pixel-restoration coherence does **not** establish reliable native marginal allocation. This fixed training-family screen finds mixed associations between complete-model attribution and the benefit of local native quantizer changes. No encoder policy or model was changed, and no model is qualified.

## Instrument and chronology

The [registration](../docs/zensim-native-interventions-2026-09-08.md) followed the completed native JXL/AVIF targeting screens and JPEG/WebP binding controls. The current `zensim_diffmap_rd --native-interventions` example reuses native cached precomputation/encoding, primary jxl-rs decode, complete Rust `BakeScorer` and production quantizer capture. The existing zensim RD analyzer adds `--interventions`; its older analysis functions and behavior remain unchanged. No public library API or feature was added.

Four canonical imazen-26 training families (2010 photo, 6068 document, 7066 graphic, 8206 screen), original existing longest-side-256 variants, distances 1 and 3; no selection or terminal family. Each cell freezes source XYB, CfL and transform strategies with effort-8 Reference settings. Baseline, exact neutral repeat, and ±10% raw quantizer interventions on 16 raster-stratified transform regions cost 34 complete encodes per cell. Integer quantizers move by at least one when possible; source-clipped region integrals use bin 8. This native cached path differs from the Zenjxl H3 loop; no byte parity with that loop is asserted.

The zero-encode prototype failed an extra, unregistered transfer-function comparison (3.10e-6 maximum difference). Its log and binary remain preserved. The measured runs use the existing exact decoder sRGB transfer explicitly. The final driver reproduces all 272 original probe bytes, scores, quantizer fields, region maps and both independent judges exactly, excluding timing.

After the first screen, a separately registered engineering packet uses existing 512-long-edge variants of those same four families: 410×512, 393×512, 512×512, 512×288. This satisfies multi-group roundtrip coverage without adding independent families or changing the primary experiment.

## Primary findings

All eight neutral repeats match bytes, decoded pixels, requested/actual quantizers and scalar scores. None of the 256 interventions changes a captured quantizer outside its selected transform region; **222 change decoded pixels outside it**. Median absolute D score change is 0.003162; maximum is 0.118782. These are small integer actuator responses, not full-restoration changes.

Among 242 interventions with nonzero captured mean log-quantizer change, D quality moves in the expected direction in 123, the opposite direction in 95, and is flat within 1e-5 in 24. SSIMULACRA2 counts are 127/91/24 and negated Butteraugli 166/49/27 (judge epsilon 1e-6). All 14 final-quantizer-inert interventions stay in raw results; **7 still change decoded pixels**.

The source explains why final quantizer identity is insufficient: `vardct/transform.rs` computes `AdjustQuantBlockAC` thresholds from the incoming quantizer, then captures the adjusted integer field. Equal adjusted fields need not imply equal thresholds or coefficients. CfL is frozen here, so this screen does not establish a CfL cause. The measured secants below include the native threshold/quantization response; they are not derivatives of an isolated quantizer-only function. Zero captured-quantizer spans cannot normalize a secant and remain explicitly excluded from those summaries.

| Origin/class | Distance | Map mass vs D secant | Map density vs D gain/byte |
|---|---:|---:|---:|
| 2010 / photo | 1 | -0.259 | -0.532 |
| 2010 / photo | 3 | 0.262 | 0.068 |
| 6068 / document | 1 | 0.528 | 0.575 |
| 6068 / document | 3 | 0.230 | 0.151 |
| 7066 / graphic | 1 | -0.085 | 0.091 |
| 7066 / graphic | 3 | 0.618 | 0.692 |
| 8206 / screen | 1 | 0.362 | 0.867 |
| 8206 / screen | 3 | 0.676 | 0.083 |

Spearman associations are within each image/distance. Gain/byte includes only positive central byte changes; 34 nonpositive rate responses remain explicit. No correlation threshold was fitted after seeing results. The 512 packet also passes integrity/neutral/region controls and has mixed associations (mass versus D secant −0.074 to 0.624). Geometry changes the descriptive results; it does not establish a general ranking or RD gain.

## Checks, cost and reproducibility

Both independent CPU judges cover every exact reference/distorted pair: 544 comparisons per packet. Each packet also has 24 libjxl **0.12** compatibility decodes, with matching dimensions and maximum RGB8 difference of 1 versus jxl-rs. Compatibility pixels are not training labels or the scoring ground truth.

Final primary packet: 272 full encodes/independent decodes/ordinary scalar comparisons, eight additional complete scored maps, zero internal reconstructions; 1.910 seconds instrument elapsed and 34,580 KiB process peak RSS. Multi-group packet: the same counts, 6.061 seconds and 80,748 KiB. These small cached-precompute experiments include artifact writes and compatibility work; neither timing is a product latency or speedup claim. Including the first successful reproduction source packet, total work is 816 full encodes, 816 ordinary decoded scalar comparisons, 24 scored maps, 48 compatibility decodes and 1,632 independent judge comparisons. The failed prototype produced zero encodes.

Local verification: release build, exact six-feature example Clippy, both CI workspace/all-target Clippy routes, scoped rustfmt, zensim CI-exact Clippy and 605-script lint pass. Six CLI controls reject wrong/missing formula, wrong split/source, missing bake and conflicting modes before creating output. The analyzer accepts the real matrix, rejects 19 damaged/invalid variants, then accepts restored data. Existing analyzer functions are AST-identical (apart from dispatch to the new mode), and its four prior reachable-target tests pass. No CI wait.

Pinned identities:

- Driver: `e8d093e39f3760178b8f7b3f7de755b42614198fb2809926286bfb224430b463`.
- Complete D model: `cd1098b450ef6941b6925b24bcbd129715b6f07c4fe84838a92e13ab364ddea6`.
- Independent judge: `6cb4744c0f1baa178c46c307db3f7a40c679dba2bf3d1ecb083a9d21f1e8cb6f`.
- libjxl decoder: `41b3531133282018310e65fc4fb4aaaba439cfce38f351bcf6a78743113f83bf`.

Full commands, source/source-manifest hashes, original and multi-group registrations, raw outputs, compatibility records, analysis, negative controls and logs are retained under `/mnt/v/output/zensim/jxl-native-interventions-2026-09-08/`. `RESULT_COMPLETE.json` supersedes the immutable launch-time “judges pending” text. [Compact machine-readable results](zensim_native_interventions_2026-09-08.json) retain per-cell independent metrics and counts.

```sh
cargo build --locked --release -p jxl-encoder \
  --features __expert,zensim-loop,ssim2-loop,parallel,__pre_quantized,__internal_recon_hook \
  --example zensim_diffmap_rd
ZENSIM_FORMULA_REV=1 RAYON_NUM_THREADS=8 target/release/examples/zensim_diffmap_rd \
  --native-interventions "$TRAIN_SOURCE_MANIFEST" --bake "$BAKE" --out-dir "$FRESH_OUTPUT"
# Run the pinned CPU zenmetrics batch judge commands retained in JUDGE_COMMANDS.json.
python3 ../zensim/scripts/v_next/rd_probe_analyze_2026-07-18.py --interventions "$FRESH_OUTPUT"
```

## Consequence for the next step

Do not select a model or tune H3 gain from restoration coherence alone. The next allocation hypothesis needs a scale and response appropriate to native quantization, plus measured rate cost; preregister a bounded coarse-region intervention before changing the policy. A successful mechanism screen must still lead to active/neutral/scalar matched-RD testing and actual train-calibrated targeting. JPEG/WebP attained bounds and 1/2/3-shot validation, broader model qualification, corruption and SDR/HDR/color/alpha qualification remain unfinished. No new model training or holdout evaluation occurred in this packet.
