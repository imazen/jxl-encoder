# Native ship-gate adapter — 2026-10-16

The existing `zensim_diffmap_rd` example now admits explicit Rev5 experiments.
Requested map retention is fail-closed: geometry/nonfinite data, failed writes,
and changed existing hash-addressed objects produce an error. Candidate density,
applied tile fields and butteraugli pixel maps retain their geometry and f32 bits.
Every complete native encode is retained, including target calibration/bound probes.
The explicit `--canonical-png` distance-ladder option uses the existing native PNG owner.

Validation: strict release Clippy passes for the complete CPU native feature set;
198 existing SIMD reference/parity tests pass; the CAS test verifies exact bits,
deduplication and tamper/geometry/nonfinite/IO refusals. Constructor migration was
also checked as exact alias substitution after whitespace normalization.

The artifact-first TRAIN smoke used origin2010 at distance1, effort8, two map
updates. All eight bitstreams decode in independent libjxl v0.12 a7a9c787;
all retained map hashes match. Byte counts at this nominal distance are plumbing
evidence and do not imply equal-quality RD savings:

| Arm | Bytes | Saved map records |
|---|---:|---:|
| W-C active | 20345 | 6 |
| diff-frag-820 active | 20411 | 6 |
| diff-410 active | 20539 | 6 |
| Each candidate neutral | 20683 | 6 |
| Butteraugli map | 20399 | 3 |
| No steering | 20683 | 0 |

Every neutral arm is byte-identical to no steering. Full targeting and
independent-judge RD qualification are pending. Source/driver/lock pins and full
private evidence on dev: `/mnt/v/output/zensim/shipgates-2026-10-16/`.

```text
CAS test: run-heavy rc=0 26s peak-RSS2.17GiB min-avail40441MiB peak-load14.73
SIMD tests: run-heavy rc=0 13s peak-RSS0.64GiB min-avail41665MiB peak-load14.84
Native build: run-heavy rc=0 23s peak-RSS1.72GiB min-avail40219MiB peak-load15.30
Smoke: run-heavy rc=0 8s peak-RSS0.03GiB min-avail43032MiB peak-load10.98
```
