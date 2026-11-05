# PQ DC precision control, 2026-11-05

An isolated extra-DC-precision=3 control passes all ten original PQ quality
failures, with the decoder pin, AC field and loop budgets unchanged. Every
source byte matches the retained original replay. This source is an experiment,
not the final shared still/animation policy or broad RD qualification.

[Ten retained rows](issue138_pq_dc3_control_2026-11-05.tsv) record both
encoders' errors and sizes and source/stream hashes. The difficult 2966/66247
cell improves from mean PQ signal error 0.038127 to 0.031026, versus cjxl
0.014096. Unlike the VDP2 control, it does not worsen that cell. Tolerances,
expected bytes and the decoder closure are unchanged.

Source snapshot `0493fc1e01b087b0dba4082e466ac257b210ab0e`, based on main
`c8c01cc8`; pinned siblings and static v0.12 tools as in the earlier replay.
Binary SHA256 `0607cf7d54ee154bcbbc8b7e0549d4e5acf45bd9cdb78e43a61b036c54863f47`.
Artifacts and preserved binary on r7900x beneath
`~/tmp/jxl-hdr-r7900x-20261104/`, directory `dc3-only-20261105/` and binary
`pq-dc3-only-0607cf7d`. Full log `~/tmp/jxl-pq-dc3-only-20261105.log`.

Reproduction uses `scripts/ablate_issue138.py --arms original` with the
retained original replay, 41 CID22 plus 10 gb82-sc inputs, and unchanged ICC
branch input. Both build and replay held `flock ~/tmp/heavy.lock` and used
`run-heavy --mem 12G --jobs 4`. Receipt: rc0, 23 seconds, peak RSS 0.87 GiB,
minimum available 18151 MiB, peak load 2.48. These are diagnostic resource
observations, not a codec speed claim.
