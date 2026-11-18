# Conditional modular correctness controls

Issue 144 E4 and E6 remain conditional. Native public controls below validate
inputs and oracle interpretation; they do not observe a winning encoder tree
or palette delta.

## E4 reference-property arithmetic

`reference-property-native-controls` creates 64×32 linear RGB binary32 samples.
The first channel's integer bit patterns alternate between signed ±0x60000000;
both corresponding floating-point values are finite. A second channel labels
a reference-gradient residual as zero/nonzero. The integer construction has
1,032 sites where the mathematical absolute residual exceeds i32::MAX. That
is a source-arithmetic observation, not a decoded-tree observation.

Unmodified libjxl v0.12 encodes this exact source and djxl reproduces every
binary32 word. Input packing uses an explicit little-endian PFM; output reading
honors its declared endianness and bottom-up rows. The canonical source-word
SHA256 is `e09dffe36aeaa8a5209d539420bd91ca4cff8210dbbc9c9a2ca84548f6e6c888`;
the native JXL SHA256 is
`3ae98eb6b21e049635f86b40e228a8cba6364c2b3ff99e993672c0e1f4935367`.
The reference uses effort nine, identity RCT and an explicit previous-channel
property budget. This does not prove that its learned tree selects property 18.

The current encoder's prepared test-only observer records actual successful
MA-tree writes. Its uncompiled probe requires a reachable threshold at which
the narrow and wide properties select different branches, plus complete primary
and native source-word checks. A tree containing property 18 alone earns no
reachability claim. A failed selection assertion means the conditional lead
has not been established. No E4 production correction is made.

## E6 delta-palette interpretation

`palette-native-controls` retains 32×32 and 259×17 RGB integer ramps with
opposing horizontal/vertical slopes. Explicit native palette-off controls
reproduce the source codes exactly. Explicit near-lossless palette controls
change 2,218 and 9,639 RGB8 samples respectively, with maximum code differences
18 and 20. These are observed decoded values, not a proposed tolerance.
Near-lossless palette intent must be checked against its selected quantized
result; exact source recovery is not its contract.

The native control uses identity RCT, Zero stream prediction, no channel
compaction and no explicit palette colors. It does not parse the actual
palette transform, its declared predictor or a winning delta index. E6 still
needs those observations in the Rust production path and an independent
same-payload predictor control before any wrong-pixel claim or repair.

Both tools require explicitly supplied v0.12.0 a7a9c78 executables and retain
input, output, full logs, executable/source hashes and decoded-sample results.
Pillow reads only explicitly selected RGB8 PNG output in the palette control;
there is no binary16, HDR, alpha or complete mode/entry-path qualification.
