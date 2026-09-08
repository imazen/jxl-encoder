# Float-sample packing oracle (libjxl v0.12.0)

Generates the golden vectors that
[`jxl-encoder/src/modular/float_pack.rs`](../../jxl-encoder/src/modular/float_pack.rs)
tests against, for lossless float modular support (imazen/jxl-encoder#109).

`ref.c` carries libjxl's `float_to_int` (`lib/jxl/enc_modular.cc:157`) and
`int_to_float` (`lib/jxl/dec_modular.cc:128`) **transcribed verbatim**, with
only the `JXL_FAILURE` / `JXL_ENSURE` plumbing replaced by return codes so the
rejection cases are observable. It is deliberately standalone rather than a
call into our Rust port — a differential that shares code with the thing under
test cannot see a bug the two share.

Both source files are byte-identical between `v0.12.0` (`a7a9c787`) and
`d089091` (verified 2026-09-08 with `git diff v0.12.0 d089091 --`), so the
transcription is valid across that range.

## Regenerate

```sh
cc -O2 -o ref ref.c -lm && ./ref > goldens_$(date +%F).tsv
```

Then re-embed the table in `float_pack.rs` (`GOLDENS`), remembering that
packed values are written `0xNNNNNNNNu32 as i32` — several are negative as
`i32` because the float sign bit lands in bit 31.

## Columns

`name`, `bits`, `exp_bits`, `in_f32_bits`, `pack_rc`, `packed`,
`unpack_rc`, `out_f32_bits`. `pack_rc`: 0 = ok, 1 = precision loss,
2 = exponent too large, 3 = exponent too small. Rejected cells print `-`
for the remaining columns.

## Two behaviours worth knowing before reading the table

- **`bits == 32, exp_bits == 8` is a bit copy**, so it round-trips every
  pattern exactly — NaN payloads and subnormals included.
- **Narrow formats are lossy for NaN payloads.** The reference's NaN/infinity
  branch does `mantissa >> mant_shift` with no loss check, so
  `0x7f800001` (signalling NaN, payload 1) packs to f16 `0x7c00` and comes
  back as `0x7f800000` — `+inf`. That is reference behaviour we match on
  purpose; tests must not assert bit-exact round-trip for narrow NaN payloads.
