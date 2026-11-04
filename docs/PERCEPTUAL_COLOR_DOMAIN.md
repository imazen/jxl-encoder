# Perceptual-loop RGB domain

Metric references and reconstructions use normalized, linear-light sRGB.
The source can arrive in other primaries, and forward XYB multiplies its
linear RGB by `intensity_target / 255` before applying the opsin transform.
Both transformations must be accounted for before comparing pixels.

The source primaries conversion already used by forward XYB is also applied
to the metric reference. It happens after forward XYB so the source is not
converted twice. Owned still-image storage is reused; borrowed storage,
including animation frame input, receives a separately budgeted fallible
copy. The copy's reservation lives as long as the converted source.

Inverse XYB returns the intensity-scaled working values. Every metric-loop
reconstruction is normalized by `255 / intensity_target` before scoring,
including the Butteraugli tightening probe. Strided buffers are processed by
visible row width, leaving padding untouched. At 255 nits this operation
returns immediately, preserving the existing arithmetic.

libjxl implements the inverse normalization in its
[v0.12 inverse opsin matrix](https://github.com/libjxl/libjxl/blob/v0.12.0/lib/jxl/opsin_params.cc#L35).
This encoder keeps its existing inverse kernel and applies the normalization
afterward. No new public API, gate, threshold, or metric choice is introduced.

## Evidence

Before the repair, `metric_color_domain` fails on both independently
observed operand mismatches:

- Intensity 1000: reference `[0.1, 0.2, 0.3]`, reconstruction
  `[0.39222655, 0.78433686, 1.17631]`.
- BT.2020 at intensity 255: reference `[0.1, 0.2, 0.3]`, reconstruction
  `[0.02659699, 0.21163999, 0.313677]`.

Both callback checks pass after the repair. The independent-decoder test
fully renders the same streams in jxl-rs and djxl v0.12, checks agreement
in the native output encoding, and compares the metric reconstruction with
djxl forced to linear sRGB. It covers 1000-nit linear sRGB and 10000-nit PQ
with BT.2020 primaries. This selected, flat-color oracle is not broad HDR
quality qualification.

The Linux CI job runs `metric_color_domain` explicitly with both decoder tools.

Run `just metric-domain-check <pinned-manifest>` with verified `CJXL_PATH`,
`DJXL_PATH`, `TMPDIR=$HOME/tmp` and the required resource wrapper.
Logs: `~/tmp/jxl138-metric-domain-before2.log` (both fail),
`~/tmp/jxl138-metric-oracles.log` (all three pass).

Default PQ/HLG currently disables its perceptual loop. The repair affects
explicit loop use and other loop-enabled non-sRGB/intensity configurations;
it does not resolve the ten remaining default-path issue-138 quality cells.
