# TODO Features — jxl-encoder-rs

## Patches (Dictionary-Based Repeated Patterns)

**Difficulty: 8/10 | ~3,600 LOC | Est. 2-3 weeks | Photo impact: NONE**

Patches detect repeated rectangular regions (UI elements, text glyphs, icons) and encode
unique patterns once in a reference frame. Each occurrence stores only position + reference
coordinates. 40-60% size wins on screenshots/UI, negligible on photos.

### What's involved

Five subsystems stitched together:

1. **Screenshot detection** — scan for 4x4 flat-color blocks to identify UI/text content
2. **Pattern extraction** — flood-fill segmentation, connected components, bounding boxes
   (max 32x32 per patch via `kMaxPatchSize`)
3. **Deduplication** — quantize patterns to 3-channel int8, sort/compare, merge near-duplicates
   (`kVerySimilarThreshold=0.03`, `kHasSimilarThreshold=0.03`)
4. **Bin-packing** — 2D rectangle packing into a reference frame, which is itself a full
   JXL frame (stored at reference ID 3, `kPatchFrameReferenceId`). Recursive encoding.
5. **Subtraction** — remove patch regions from main image before normal VarDCT encoding

Decoder needs:
- Interval tree for Y-coordinate spatial lookup (O(log n) per row)
- 8 blend modes: None, Replace, Add, Mul, BlendAbove/Below, AlphaWeightedAdd
- Alpha channel handling with per-channel blend control and clamping

### Key libjxl source files

- `lib/jxl/enc_patch_dictionary.{cc,h}` — 889+110 LOC (encoder: detection, extraction, packing)
- `lib/jxl/dec_patch_dictionary.{cc,h}` — 357+167 LOC (decoder: parse, interval tree)
- `lib/jxl/render_pipeline/stage_patches.{cc,h}` — 62+25 LOC (blend stage)
- `lib/jxl/patch_dictionary_internal.h` — 28 LOC

jxl-rs decoder: `jxl/src/features/patches.rs` — 2,034 LOC (includes tests)

### Pipeline integration

Called from `enc_heuristics.cc:1057-1064`, effort 7+ (Squirrel):
1. Detect screenshot-like content (parallel 4x4 grid scan)
2. Flood-fill to find text/UI regions
3. Extract, quantize, deduplicate patterns
4. Bin-pack into reference frame, encode as full JXL frame
5. Subtract patch regions from opsin
6. Continue normal VarDCT encoding

### Content that benefits

- Screenshots, UI, text documents (huge wins)
- Comics/manga (repeated elements)
- Presentations (logos, icons, chrome)

Does NOT help: photographs, natural scenes, anything without repeated rectangles.

### Verdict

Low priority. Won't close the 22-26% RD gap at d>=2.0 on photos. Worth doing
eventually for full e7 coverage and screenshot/document use cases.

---

## Splines (Parametric Curve Encoding)

**Difficulty: 6/10 infrastructure, 9/10 detection | ~1,100 LOC encoder | Photo impact: NICHE**

Splines encode thin smooth curves (power lines, horizons, wire fences) as centripetal
Catmull-Rom curves with DCT-encoded color/thickness parameters. Gaussian splatting renders
each point along the curve. Extremely efficient for the right content (test file: 2048x2048
image encoded as 81 bytes).

### Critical blocker: no auto-detection exists

**libjxl's `FindSplines()` is a stub that returns empty.** After years of development,
they never implemented auto-detection. The encoder only supports manually-provided splines
via `cparams.custom_splines`. This means:

- The rendering/decoding infrastructure works and is well-tested
- The encoding infrastructure (quantization, entropy coding) works
- Detection — the entire value proposition — doesn't exist anywhere
- Would need to be invented: edge detection, line tracing, curve fitting, cost model

### What's involved (excluding detection)

1. **Data structures** — control points (x,y), 4x DCT32 arrays (X,Y,B color + sigma
   thickness), quantization adjustment parameter, double-delta encoding for control points
2. **Curve generation** — Catmull-Rom with 16 interpolation points between control points,
   arc-length parameterization, walk at kDesiredRenderingDistance=1.0 pixel spacing
3. **Rendering** — precompute segments indexed by Y, per-pixel Gaussian splatting:
   `intensity = (erf((d+0.5)/sigma) - erf((d-0.5)/sigma))^2`, apply color * intensity
4. **Quantization** — channel weights [0.0042, 0.075, 0.07, 0.3333] for X/Y/B/sigma
5. **Entropy coding** — 6 contexts for ANS/Huffman, HybridUint for positions and DCT coefficients

### Key libjxl source files

- `lib/jxl/splines.{cc,h}` — 745+157 LOC (core: quantization, curve gen, rendering)
- `lib/jxl/enc_splines.{cc,h}` — 109+30 LOC (encoder: quantize + encode, stub detection)
- `lib/jxl/render_pipeline/stage_splines.{cc,h}` — 72 LOC (render stage)

jxl-rs decoder: `jxl/src/features/spline.rs` — 1,963 LOC (includes tests)

### Pipeline integration

Called from `enc_heuristics.cc:~1050`, effort 7+ (Squirrel):
1. Check for custom_splines (manual) or call FindSplines (stub, returns empty)
2. InitializeDrawCache — precompute curve segments
3. SubtractFrom — render splines and subtract from opsin
4. Encode spline data in frame bitstream (flag `FrameHeader::kSplines`)

### Content that benefits

- Power lines and cables (thin diagonals)
- Horizons (smooth sky/land boundaries)
- Wire fences (regular thin line patterns)
- Thin specular highlights

Does NOT help: textured content, repeated patterns (patches better), sharp corners,
noise, irregular edges. Very niche.

### Verdict

The infrastructure port (rendering, quantization, encoding) is moderate work. But without
auto-detection, users would need to manually specify splines — severely limiting utility.
Inventing a good detection algorithm is a research problem, not an engineering task.

Skip unless targeting specific thin-line-heavy content with manual spline specification,
or willing to invest in detection research.

---

## Priority Summary

Neither feature closes the photo RD gap. For photo quality parity with cjxl e7:

| Feature | Photo impact | Effort | Priority |
|---------|-------------|--------|----------|
| Cost model tuning (kFavor2X2, thresholds) | Medium | Days | **High** |
| Full histogram clustering (e8+) | 2-5% universal | ~1 week | **Medium** |
| Patches | 0% photos, 40-60% screenshots | 2-3 weeks | Low |
| Splines | 0% most photos, huge on niche | 1 week + research | Low |
| Dots detection | Negligible | ~1 week | Very low |

Patches and splines become relevant when targeting screenshot/document compression
or full spec coverage. For photo-focused encoding, they can wait.
