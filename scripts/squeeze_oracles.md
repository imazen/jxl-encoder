# Pinned squeeze component controls

These wrappers link the unmodified objects from libjxl v0.12.0 a7a9c78.
The builder checks the reference tool's version and refuses an existing output.
No forward-squeeze arithmetic is copied into the wrappers.

Build with `python3 scripts/frame_header_oracle.py --component squeeze_params
SOURCE BUILD NEW_EXECUTABLE`, or select `lf_squeeze_color` for the coefficient
control. `SOURCE` must be the source used by that existing build.

The parameter executable supports:

- `--defaults`: `DefaultSqueezeParameters`, including one/two-channel controls and the 259-channel case
  at 16x16. Columns are width, height, channel count, operation index,
  horizontal, in-place, begin channel, and channel count for that operation.
- `--wire`: the actual bundle writer's count, bit length, and hex bytes.
- `--transform-counts` and `--begin-fields`: the actual bundle writer's
  success/refusal at the selected field boundaries.

The small text vectors live in `jxl-encoder/src/modular/squeeze_*_v012.tsv`.
They distinguish explicit field limits from implicit defaults (Part 1 H.6.2.1).

The LF control calls the actual `FwdSqueeze` on two identical images with
zero extra placeholders. One schedule addresses the full original prefix;
the other addresses only the three color components in those operations.
It compares every ordered nonempty color band, its shape/shifts/component,
and every coefficient, and requires all extra placeholders to remain zero.
The retained controls agree at 19/20/21 channels on the listed small,
degenerate-axis, and 257-square shapes. This establishes color-transform
separability on those controls, not complete encoder byte parity.

The `squeeze_gray_geometry` component observes the actual full default and
forward-transform band order for one/two-channel 2x16449 images and their
transposes. `squeeze_gray_global_v012.tsv` retains those dimensions/shifts
and the ordered global-prefix boundary. High shifts in globally coded bands
are legal; the remaining bands all retain positive group tiles. This control
separates that case from E3's invalid remaining-band geometry.
