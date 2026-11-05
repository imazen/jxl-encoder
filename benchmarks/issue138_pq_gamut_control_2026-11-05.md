# PQ block-gamut attribution control, 2026-11-05

A diagnostic restricts precision-three/nearest DC quantization to PQ images
with an 8x8 average outside the internal linear-sRGB gamut. It passes 18/21
reported records. Two original quality failures remain unchanged:
2965/70595 (mean error 0.067663 versus cjxl 0.027896) and 2965/77723
(0.079336 versus 0.011808). The historical 2965/28968 decoder disagreement
also remains. All source options, thresholds and decoder pins are unchanged.

This shows that the gamut predicate does not cover the full PQ DC problem.
It is not adopted. Source snapshot `dd659112e56a1bb03c8da3b6b354a4e124f7d40f`;
all streams and complete verdicts are retained on r7900x beneath
`~/tmp/jxl-hdr-r7900x-20261104/pq-gamut-final-21-20261105/`.
Full log `~/tmp/jxl-pq-gamut-final-checks-20261105.log`: rc1, 24 seconds,
peak RSS 0.87 GiB, minimum available 18461 MiB, peak load 0.65.
The later gates in that command did not run after the replay failed.

The unrestricted precision-three/nearest control passes all 21 records and
all 76 additional paired PQ cells, but changes one established neutral-ramp
fingerprint. Its quality/rate evidence and any expectation migration require
separate review; this insufficient content predicate cannot substitute for it.
