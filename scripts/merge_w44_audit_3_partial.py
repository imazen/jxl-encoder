#!/usr/bin/env python3
"""W44-AUDIT-3 merge: splice the 3-cell re-validation TSV
(codec_wiki d=4 e5/e7/e9) into the canonical 36-row W44-AUDIT-1 TSV.

The W44-AUDIT-2 EPF fix only touches the previously-OOM e9 d=4 row.
e5 d=4 and e7 d=4 should be byte-identical to the prior bench — we use
them as sanity guards. If they differ, log a warning and abort.
Only the e9 d=4 row is actually replaced.

Usage:
    python3 scripts/merge_w44_audit_3_partial.py \\
        benchmarks/cjxl_parity_2026-05-24_post_w44_205_s2_refit_c2.tsv \\
        benchmarks/cjxl_parity_2026-05-24_post_w44_audit_2_partial.tsv \\
        benchmarks/cjxl_parity_2026-05-24_post_w44_audit_2.tsv
"""
import csv
import sys
from pathlib import Path


def main():
    if len(sys.argv) != 4:
        print(__doc__, file=sys.stderr)
        sys.exit(2)

    canonical_path = Path(sys.argv[1])
    partial_path = Path(sys.argv[2])
    out_path = Path(sys.argv[3])

    # Load partial: dict keyed on (image_id, effort, distance)
    partial = {}
    with partial_path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            key = (r["image_id"], int(r["effort"]), float(r["distance"]))
            partial[key] = r
    print(f"[merge] loaded {len(partial)} partial rows from {partial_path.name}")

    # Read canonical, replace matching rows
    out_rows = []
    fieldnames = None
    replaced = 0
    sanity_byte_ident = 0
    sanity_mismatch_warn = 0
    with canonical_path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        fieldnames = reader.fieldnames
        for r in reader:
            key = (r["image_id"], int(r["effort"]), float(r["distance"]))
            if key in partial:
                new = partial[key]
                # Sanity check: e5 d=4 and e7 d=4 should byte-match prior
                # (only e9 d=4 was the OOM). cjxl bytes deterministic.
                prior_zen = int(r["zenjxl_bytes"])
                prior_cjxl = int(r["cjxl_bytes"])
                prior_lib = int(r["libjxl_strat_bytes"])
                new_zen = int(new["zenjxl_bytes"])
                new_cjxl = int(new["cjxl_bytes"])
                new_lib = int(new["libjxl_strat_bytes"])
                if prior_zen == 0:
                    # Previously-FAIL row — must be e9 d=4
                    assert key == ("codec_wiki", 9, 4.0), \
                        f"Unexpected previously-zero row {key}"
                    print(f"[merge] REPLACING previously-FAIL row {key}: "
                          f"zen={new_zen}B (was 0), libjxl={new_lib}B (was 0), "
                          f"cjxl={new_cjxl}B (was {prior_cjxl})")
                else:
                    # Sanity check
                    if (prior_zen, prior_lib) == (new_zen, new_lib):
                        sanity_byte_ident += 1
                        print(f"[merge] SANITY-OK row {key}: bytes byte-identical "
                              f"(zen={new_zen}, libjxl={new_lib})")
                    else:
                        sanity_mismatch_warn += 1
                        print(f"[merge] WARN: row {key} differs from prior! "
                              f"zen: {prior_zen}→{new_zen}, "
                              f"libjxl: {prior_lib}→{new_lib}", file=sys.stderr)
                    # Replace anyway (use freshest measurement)
                replaced += 1
                out_rows.append(new)
            else:
                out_rows.append(r)

    # Write merged TSV
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t",
                                lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"[merge] wrote {len(out_rows)} rows to {out_path.name}")
    print(f"[merge] replaced {replaced} rows ({sanity_byte_ident} sanity-OK, "
          f"{sanity_mismatch_warn} sanity-MISMATCH)")
    if sanity_mismatch_warn > 0:
        print("[merge] WARN: sanity rows mismatched — please review", file=sys.stderr)


if __name__ == "__main__":
    main()
