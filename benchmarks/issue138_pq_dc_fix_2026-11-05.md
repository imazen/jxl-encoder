# PQ DC quantization candidate, 2026-11-05

Status: verified repair; the owner approved the single PQ byte fingerprint
update on November 5. Quality bounds and ignore annotations are unchanged.
The product implementation matches source snapshot `99a4da7f`; later changes
add retained artifacts and share the existing test decoder once per test binary.
Baseline is orientation-integrated main `1989fb37`. All checks use the same
frozen sibling closure; the native decoder pin remains `515f2c8a`.

## Result and scope

PQ pixel input outside `dc_encode_libjxl_parity` uses three DC fractional bits
and nearest rounding. The existing prediction-relative deadzone/snap-to-even
rule can amplify dark-channel errors after wide-gamut chroma cancellation.
Still/streaming and animation share the private policy. AC quantization,
entropy predictors, requested distance, resampling and HDR loop defaults are
unchanged. Strict Libjxl and non-PQ encoding retain their existing policy.
No public API or dependency changes.

- [All 21 original records](issue138_pq_fixed_replay_2026-11-05.tsv) pass with
  the checksum-pinned static djxl and both unchanged Rust decoders, including
  all ten PQ quality failures. This also changes the stream in 2965/28968;
  decoder issue 73 remains valid for its older encoded file.
- A functional regression checks 40 streams / 56 displayed frames through
  Rust and djxl, plus 24 streaming identities, using integer and float inputs,
  both sizes, coarse distance and faster decoding. The reported saturated
  color fails in 20/40 streams before the change and none afterward.
- [76 additional paired cells](issue138_pq_global_control_2026-11-05.tsv)
  reproduce every lossy unprofiled PQ case from retained seed 2966 indices
  0–1999: 68 encoded cases and eight matching expected input refusals.
  Neither arm has failures. All source hashes match. Three explicit strict
  controls retain identical encoded hashes; 53 of 68 encoded pairs change.
- At the recorded six-decimal error precision, 42 pairs improve, 18 tie and
  eight worsen; all remain within the existing bounds. The largest increase
  is 0.001510 to 0.016099 on a 1x1 cell (2966/1466). This is selected
  regression coverage, not universal RD or perceptual-quality qualification.
- Total bytes over the 68 encoded pairs are 6,201,537 before and 6,224,656
  after. Median per-file increase is 0.9499708014104491%; the largest relative
  increase is a 3x2 progressive-DC file, 83 to 132 bytes. Timing is not a
  performance comparison.

## The approved fingerprint

`hash_lock_features::lossy_mg_rgb16_pq_512x512_ramp_e7_d1` changes from
3189 to 5638 bytes. Its file-header fingerprint stays
`0x739c20a36de9e959`; the new frame fingerprint is `0x51a9a261a072a71c`
(previously `0x5f169307b2ba0537`). Only this sidecar expectation changes,
with owner approval.
The complete hash run reports 68 other passes, this one failure and one
pre-existing ignore. All five strict byte-lock tests and seven divergence
checks pass separately. This was the expected encoding-policy change presented for
review. Final validation after the approved update is recorded below.

[Independent ramp measurement](issue138_pq_ramp_quality_2026-11-05.json)
uses the exact same retained 512x512 RGB16 source, SHA256
`11de60028b27db9a5f17a9d4b515d4314ac50b162e6ea15648d79c35d775f98e`.
Clamped mean PQ signal error:

| Decoder | Before | After |
|---|---:|---:|
| djxl v0.12 static | 0.0010578811215463703 | 0.0009311982179735402 |
| Primary Rust decoder | 0.0010578811746965518 | 0.0009311966368096112 |

The ramp gains signal fidelity for 2449 additional bytes. This is an
explicit size/quality trade, and does not establish matched-quality RD benefit.
A [block-gamut restriction](issue138_pq_gamut_control_2026-11-05.md) was tested
and rejected as a complete repair because two original quality failures remain.

## Validation and reproduction

Default library: 1668 pass, 29 existing ignores. Animation contracts: 10 pass;
HDR intensity/metadata: five pass; packed-reference/explicit-HDR/VDP2 checks:
20 pass. The seven wide-integer decoder-support checks, workspace all-target Clippy,
strict minimal-feature Clippy, scoped formatting and three parser regressions
pass after sharing the existing test decoder once per binary.

Use the frozen exporter and unchanged pins, then run under
`flock ~/tmp/heavy.lock` plus `run-heavy --mem 12G --jobs 4`:
`just issue138-retained-replay <binary> <new-out> <corpus> <icc>`,
`just pq-dc-validation`, and
`just issue138-pq-pairs <baseline> <candidate> <prior-results.tsv> <new-out> <corpus> <icc>`.
The original held-fingerprint failure and subsequent independent gate results
remain retained; the approved expectation permits the complete validation recipe.
`just pq-ramp-quality <verified-djxl> <retained-before> <retained-after> <new-out>`
recomputes the ramp comparison (NumPy required).

The retained candidate sweep binary SHA256 is
`e41f7f3a54cc605382f9dd2477bf01501e5a4017cc806f44150472ea9fc4e70c`.
All input/encoded/reference artifacts remain on r7900x beneath
`~/tmp/jxl-hdr-r7900x-20261104/`: `pq-final-21-20261105/`,
`pq-paired-nearest-76-20261105/`, `pq-ramp-before/`, `pq-ramp-after/`, and
`pq-ramp-independent-quality-20261105/`. Reference executable identity is
in the JSON. The advertised reference is v0.12.0 `4128790`, not the
historical dynamic `a7a9c78` build.

Resource receipts: functional/paired check rc0, 81s, peak RSS 0.88 GiB,
minimum available 18410 MiB (`~/tmp/jxl-pq-dc-nearest-after-20261105.log`).
Full replay/library/HDR/hash check rc101 at the held fingerprint, 112s,
peak RSS 1.63 GiB, minimum available 17560 MiB
(`~/tmp/jxl-pq-final-checks-20261105.log`). The first remaining-gates run
passes strict/drift/HDR controls but catches duplicate test-module inclusion
at workspace Clippy; the shared test module repairs that orchestration issue
without changing its decoder implementation. Its original failed log is
`~/tmp/jxl-pq-remaining-checks-20261105.log` (rc101, 110s, peak RSS 1.62 GiB).
The independent ramp comparison reports rc0, peak RSS 0.06 GiB.

Final shared-helper/lint log `~/tmp/jxl-pq-final-lint-20261105.log`: rc0,
49s, peak RSS 1.00 GiB, minimum available 12910 MiB. This last change is
test-only; product Rust sources match the all-21 passing `99a4da7f` snapshot.
