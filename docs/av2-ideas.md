# AV2 Ideas for JXL Encoder

Research date: 2026-02-12.

## Applicable Encoder-Only Ideas

### VarDCT Block Size Selection (from SDP)

AV2's Semi-Decoupled Partitioning research shows chroma benefits from larger transform sizes than luma. JXL VarDCT already supports per-component block sizes — but the encoder heuristics may not fully exploit this. Bias chroma toward larger DCT sizes when the chroma signal is smooth relative to luma.

### Adaptive Quantization Field (from AV2 unified QStep)

AV2 replaces 6 lookup tables with a clean exponential: QStep doubles every 24 q_index steps. This provides consistent R-D behavior across the quality range. JXL's per-block quantization multiplier field could benefit from tuning the quality-vs-multiplier curve shape using similar perceptual research.

### Trellis Coefficient Optimization

Viterbi search for optimal DCT coefficient levels, jointly optimizing coefficient values and zero-run structure. AV2 achieved 52% SIMD speedup for their trellis search — those vectorization techniques apply to any DCT-based trellis implementation.

### Transform Search Pruning (from MDTX)

AV2 maps (block_size, prediction_mode) → 7-candidate transform set. For JXL VarDCT, the equivalent is: given the local image characteristics and chosen DCT size, which transform variant is most likely optimal? Can prune the search space using similar data-driven lookup tables.

## Already Covered by JXL

- XYB color space handles luma-chroma decorrelation (CCTX concept)
- Noise synthesis tool exists (film grain)
- Flexible transform sizes and types
- Modular mode for lossless (better than AV2's lossless in many cases)

## Lower Priority

- AV2's CCSO insight (cross-component correlation) — XYB already decorrelates well enough that the marginal gain from chroma offset correction may be small. Worth measuring.
