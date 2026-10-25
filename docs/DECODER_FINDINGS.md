# Decoder findings ledger

Problems in JPEG XL **decoders** (and reference tools) that our testing runs
into: libjxl (`djxl`/`cjxl` v0.12), upstream jxl-rs (crates.io `jxl`), our fork
zenjxl-decoder, and jxl-oxide (crates.io and the imazen fork). Encoder bugs of
our own go to `CLAUDE.md` (Known Bugs / Resolved Bugs), not here.

Every entry says which decoder and version, what was observed, how to
reproduce it, how we know it is the decoder (which other decoders agree), and
its upstream status. A disagreement is a finding, not a vote: settle it
against the spec before reporting. Reports to projects outside the imazen and
lilith GitHub orgs need the owner's approval of the full text first; until
then the status is **not reported**.

Most entries come from the weekly conformance sweep
(`tools/conformance-sweep`, `just conformance-sweep`). Its findings are
deterministic: `conformance-sweep --seed S --only N` with the same image and
ICC inputs and the same tool commit rebuilds the exact stream.

Status values: `triage` (not yet known whether it is the decoder), `confirmed`
(decoder at fault, repro in hand), `not reported`, `reported <where>`,
`fixed in <version>`, `not a bug` (kept so nobody re-investigates).

## Upstream jxl-rs (crates.io `jxl`)

| # | Version | Finding | Evidence | Status |
|---|---|---|---|---|
| R1 | 0.7.0 to 0.8.0 | Lossless squeeze on images wider than 2048 px with a partial last group row decodes with samples off by one or more near the corner of the last full group (e.g. 83 samples in x 2036..2047, y 252..256 of a 2049x257 image). | Minimal repro made by cjxl v0.12 itself: flat grey 2049x257 with a 5x5 noise corner patch, `-d 0 -e 3 --responsive=1`, 1,259 bytes; djxl v0.12 and libjxl main exact, jxl-rs main `d7a831f` wrong (scalar, SSE4.2 and AVX alike). Bisected to `c066ee2` "Use border buffers for HSqueeze and VSqueeze" (its parent `2d0b720` does not build); `ae13a2d` is good. Found by the sweep (seed 1 cells 12, 49). | confirmed; issue draft and a regression test for `jxl/src/tests/api.rs` ready, awaiting owner approval; no upstream issue or PR (checked 2026-10-25); not reported |
| R2 | 0.8.0 | Lossy files from the PQ, HLG and BT.709 float layouts decode up to 0.93 away from djxl in f32 output. | Verified 2026-10-25: inside [0, 1] the two agree within 0.011 (PQ), 0.0005 (HLG), 0.0002 (BT.709); the rest is jxl-rs clamping out-of-range output (to 0 for PQ, to about -0.074 for HLG) where djxl passes it through. The out-of-range values are expected: cjxl v0.12 output of the same PQ pixels decodes the same way (d1 mean error 0.0156 vs ours 0.0157, down to -0.65). The sweep now compares these layouts in the clamped signal range. | not a bug |
| R3 | 0.4.3 | Encoded ICC streams over 1 MiB are refused (`IccTooLarge`). | `tests/it/jbrd_spec_validity.rs::large_icc_profile_declares_level_10`. jxl-rs 0.8 makes the cap a decoder option. | fixed in a later release; this repo's tests still pin 0.4 |
| R4 | 0.4.3 | RGB images tagged with a CMYK, Lab or XYZ ICC profile stop with "eof hdr" (djxl decodes them). | `~/tmp/icc128-real` probe, 2026-10-25. The probe paired RGB pixels with non-RGB profiles, so the input itself is malformed. | not a bug |

## zenjxl-decoder (our jxl-rs fork)

| # | Version | Finding | Evidence | Status |
|---|---|---|---|---|
| Z1 | 0.4.0 (local), and jxl-rs 0.8.0 | Lossy HLG float output: jxl-rs and zenjxl-decoder agree with each other but differ from djxl by up to 0.34 on some pixels, even in the clamped [0, 1] signal range (seed 3 run, 2026-10-25: 46 of 2,766 HLG cells over the 0.02 tolerance). All three stay within the source error bound, so this is an inverse-OOTF or tone-mapping difference, not yet attributed. PQ and BT.709 agree once clamped. | Next step: compare the three in linear light (djxl `--color_space`, jxl-rs output profile) before attributing. The sweep skips the cross-decoder comparison for HLG until then. | triage |

## jxl-oxide

| # | Version | Finding | Evidence | Status |
|---|---|---|---|---|
| O1 | imazen fork `08395e61` | JPEG reconstruction matches DQT tables to components by DQT identifier instead of array position (`jxl-jbr/src/reconstruct.rs:638`), so swapped-DQT files rebuild with the wrong tables. libjxl uses the position, as the format specifies. | `tests/it/jbrd_spec_validity.rs` (djxl is the oracle for those cells). | confirmed; still on upstream main (line 638 unchanged since 2025-04); no upstream issue or PR (checked 2026-10-25); not reported |
| O2 | imazen fork `08395e61` | Underflow on an all-zero (empty-DHT) Huffman table (`reconstruct.rs:480`). | Same test file. | confirmed; upstream PR tirr-c/jxl-oxide #511 (another contributor, open) fixes it |
| O3 | imazen fork `08395e61` | Panics reconstructing 4:2:0 progressive JPEG transcodes, libjxl's included (single-component scan geometry in `jxl-jbr`); djxl is exact. | `benchmarks/jpeg_transcode_render_vs_jpeg_2026-10-07.md`. | confirmed; our upstream PR tirr-c/jxl-oxide #508 (open since 2026-10-07) |
| O4 | crates.io 0.12 | Mis-reads some multi-group modular frames that jxl-rs and djxl accept. | 0.3.x aux-frame probe, 2026-10-24 (CLAUDE.md, lossless property bucket entry). | confirmed on 0.12.5; may be the empty-section fix shipped in 0.12.6 (unverified); no upstream issue found (2026-10-25) |
| O5 | 0.12.x, and jxl-rs | Reject ftyp minor-version-1 files with out-of-order `jxlp` boxes, which libjxl v0.12 writes with `--output_mode 2` (both decoders ignore the minor version). | Part 2 audit, 2026-10-22 (CLAUDE.md). | confirmed; no upstream issue found (2026-10-25); not reported |

## libjxl v0.12

| # | Finding | Evidence | Status |
|---|---|---|---|
| L1 | `cjxl` writes ICC streams that its own `djxl` rejects: the ICC reader fails with "Corrupted stream" when the profile expands past 256 times the bytes read at a 65536-symbol checkpoint (`icc_codec.cc:403`), and the writer has no matching bound. | #128; a 1x1 PNG whose sRGB profile carries a 1 MiB zero private tag gives a 294-byte file djxl rejects (v0.12 and main). No real profile examined (138) comes near the ratio. | confirmed; issue draft ready, awaiting owner approval; no libjxl issue or PR found (2026-10-25); not reported |
| L2 | The `jxli` writer: `JXL_ENC_FRAME_INDEX_BOX = 0` still indexes the frame (`encode.cc:2071`), the tick unit is fixed at 1/1000 s, `T_i` lags by one record, and `F_i` counts frames that are not displayed. Offsets are correct (an earlier draft said otherwise: `codestream_bytes_written_end_of_frame` holds the frame start when `AddFrame` runs). | 4-frame 24 tps animation, v0.12 and main; semantics as documented in libjxl's own `encode_internal.h`. | confirmed; issue draft ready, awaiting owner approval; no upstream issue (2026-10-25); not reported |
| L3 | The decoder never parses `jxll`. | Part 2 audit, 2026-10-22. Ignoring it is allowed, and apps receive the box through the decoder's box API (`decode.h` documents it); only `jxlinfo` prints nothing for it. | not a bug (cosmetic `jxlinfo` gap only) |
| L4 | Refuses a jbrd with four quantization tables (`jpeg_data.cc:147`), although the field allows 1 to 4. | Part 2 audit, 2026-10-22; 190-byte repro (3-component JPEG with an unused fourth DQT equal to the third, encoded with the check relaxed), djxl v0.12 and main fall back to pixels; with the check relaxed on both sides the JPEG rebuilds byte for byte. Related but distinct: libjxl issue 5055 (the encoder rejects unused or duplicate DQT tables). | confirmed; issue draft ready, awaiting owner approval; no upstream issue for the decoder side (2026-10-25); not reported |
| L5 | `djxl --bits_per_sample=16` is ignored for PAM and PPM output (MAXVAL 255) while PNG honours it. The PNM reset in `djxl_main.cc:500-504` (added in 9ab6e9d4) tests `opt_jpeg_quality_id` where the help text implies `--bits_per_sample`; adding `--jpeg_quality=90` makes the PAM 16-bit. | Three-format check, djxl v0.12.0 a7a9c78 and main. | confirmed; issue draft ready, awaiting owner approval; no upstream issue (2026-10-25); not reported |
| L6 | `ComputeBestTree` (enc_ma.cc) does not bound MA tree depth, so libjxl can in principle emit Level 5 codestreams with trees deeper than 64 (our learner had the same gap, fixed 2026-10-25). | Source reading; cjxl e9 on noise, stripes and frymire reached depth 21 at most. | triage; not reproduced; do not report without a trigger |
