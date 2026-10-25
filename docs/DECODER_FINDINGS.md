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
| R1 | 0.8.0 | Lossless images with an axis past 2048 px and squeeze decode with up to 663 samples off by one (e.g. 28 vs 27). | Sweep 2026-10-25 seed 1, cells 12 (Bgr8 2049x300 e2 squeeze, no ANS, streamed) and 49 (Rgb8 2049x300 e7 squeeze, AdobeCompat-v4 ICC); seed 2 verification run: 331 cells, every layout. djxl v0.12 and zenjxl-decoder 0.4 reproduce every sample. | confirmed; minimal repro pending; not reported |
| R2 | 0.8.0 | Lossy files from the PQ, HLG and BT.709 float layouts decode up to 0.93 away from djxl (f32 output in the embedded transfer). | Sweep 2026-10-25: seed 1 (20 cells, e.g. 10, 16, 45, 53, 125), seed 2 verification run (5,855 of 118,205 cells; for BT.709 zenjxl-decoder agrees with djxl). May be an output-transfer or tone-mapping default (`desired_intensity_target`) rather than a decoding bug. The sweep no longer compares PQ/HLG pixels across decoders (see Z1). | triage |
| R3 | 0.4.3 | Encoded ICC streams over 1 MiB are refused (`IccTooLarge`). | `tests/it/jbrd_spec_validity.rs::large_icc_profile_declares_level_10`. jxl-rs 0.8 makes the cap a decoder option. | fixed in a later release; this repo's tests still pin 0.4 |
| R4 | 0.4.3 | RGB images tagged with a CMYK, Lab or XYZ ICC profile stop with "eof hdr" (djxl decodes them). | `~/tmp/icc128-real` probe, 2026-10-25. The probe paired RGB pixels with non-RGB profiles, so the input itself is malformed. | not a bug |

## zenjxl-decoder (our jxl-rs fork)

| # | Version | Finding | Evidence | Status |
|---|---|---|---|---|
| Z1 | 0.4.0 (local) | Lossy PQ/HLG float output differs from djxl: median 0.075, max 0.44 (sRGB-range f32 in the embedded transfer); upstream jxl-rs differs more (median 0.28). All three decoders disagree, so this is an HDR rendering (OOTF / tone mapping) question, not yet a bug. | Sweep 2026-10-25 seed 2 verification run: 104 cells (HLG 86, PQ 18); seed 1 cell 47. Next step: compare all three in linear light (djxl `--color_space`, jxl-rs output profile) before attributing. | triage |

## jxl-oxide

| # | Version | Finding | Evidence | Status |
|---|---|---|---|---|
| O1 | imazen fork `08395e61` | JPEG reconstruction matches DQT tables to components by DQT identifier instead of array position (`jxl-jbr/src/reconstruct.rs:638`), so swapped-DQT files rebuild with the wrong tables. libjxl uses the position, as the format specifies. | `tests/it/jbrd_spec_validity.rs` (djxl is the oracle for those cells). | confirmed; not reported |
| O2 | imazen fork `08395e61` | Underflow on an all-zero (empty-DHT) Huffman table (`reconstruct.rs:480`). | Same test file. | confirmed; not reported |
| O3 | imazen fork `08395e61` | Panics reconstructing 4:2:0 progressive JPEG transcodes, libjxl's included (single-component scan geometry in `jxl-jbr`); djxl is exact. | `benchmarks/jpeg_transcode_render_vs_jpeg_2026-10-07.md`. | confirmed; not reported |
| O4 | crates.io 0.12 | Mis-reads some multi-group modular frames that jxl-rs and djxl accept. | 0.3.x aux-frame probe, 2026-10-24 (CLAUDE.md, lossless property bucket entry). | confirmed; minimal repro pending; not reported |
| O5 | 0.12.x, and jxl-rs | Reject ftyp minor-version-1 files with out-of-order `jxlp` boxes, which libjxl v0.12 writes with `--output_mode 2` (both decoders ignore the minor version). | Part 2 audit, 2026-10-22 (CLAUDE.md). | confirmed; not reported |

## libjxl v0.12

| # | Finding | Evidence | Status |
|---|---|---|---|
| L1 | `cjxl` writes ICC streams that its own `djxl` rejects: the ICC reader refuses streams compressed better than 256:1 (`icc_codec.cc:396-403`), and the encoder applies LZ77 without checking that bound. | #128; repeated or zero tag payloads of 1 MB or more. | confirmed; not reported |
| L2 | The `jxli` (frame index) writer has wrong `T_i`, tick-unit and `F_i` semantics. | Part 2 audit, 2026-10-22. | confirmed; not reported |
| L3 | The decoder never parses `jxll`. | Part 2 audit, 2026-10-22. | confirmed; not reported |
| L4 | Refuses a jbrd with four quantization tables (`jpeg_data.cc`), although the field allows 1 to 4. | Part 2 audit, 2026-10-22. | confirmed; not reported |
| L5 | `djxl` ignores `--bits_per_sample=16` for PAM output from 8-bit images (writes MAXVAL 255, dithered). | Sweep harness bring-up, 2026-10-25. | triage (check djxl's documented semantics for the flag before calling it a bug); harness no longer passes it |
| L6 | `ComputeBestTree` (enc_ma.cc) does not bound MA tree depth, so libjxl can in principle emit Level 5 codestreams with trees deeper than 64 (our learner had the same gap, fixed 2026-10-25). | Source reading; not yet reproduced with cjxl. | triage |
