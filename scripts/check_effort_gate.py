#!/usr/bin/env python3
"""Reject incomplete effort-gate inputs and retain a valid before/after control."""
import argparse
import csv
import os
from pathlib import Path
import subprocess


def run(binary, corpus, output, options, expected, log):
    with log.open("x") as stream:
        result = subprocess.run([str(binary), str(corpus), str(output), *options],
                                stdout=stream, stderr=subprocess.STDOUT)
    assert (result.returncode == 0) == expected, (log, result.returncode)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("source", type=Path, help="known 512px lossless inversion source")
    parser.add_argument("before", type=Path, help="original six-cell TSV")
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    source_name = args.source.name
    binary, source = args.binary.resolve(strict=True), args.source.resolve(strict=True)
    args.out.mkdir(parents=True, exist_ok=False)
    empty = args.out / "empty"
    empty.mkdir()
    corpus = args.out / "valid"
    corpus.mkdir()
    (corpus / source_name).symlink_to(source)
    base = ["--images", "1", "--size", "512", "--efforts", "3,5",
            "--distances", "1,4", "--reps", "1"]
    cases = [("empty", empty, []), ("missing", args.out / "missing", []),
             ("zero-images", corpus, ["--images", "0"]),
             ("zero-reps", corpus, ["--reps", "0"]),
             ("duplicate-effort", corpus, ["--efforts", "3,3"]),
             ("duplicate-distance", corpus, ["--distances", "1,1"]),
             ("nonfinite-distance", corpus, ["--distances", "nan"]),
             ("oversize", corpus, ["--size", "999999"])]
    # Options use first-match parsing, so put invalid overrides first.
    for name, input_path, options in cases:
        run(binary, input_path, args.out / f"{name}.tsv", options + base,
            False, args.out / f"{name}.log")
    corrupt = args.out / "corrupt"
    corrupt.mkdir()
    (corrupt / "corrupt.png").write_text("not a PNG")
    run(binary, corrupt, args.out / "corrupt.tsv", base, False, args.out / "corrupt.log")
    existing = args.out / "existing.tsv"
    existing.write_text("preserve this output\n")
    run(binary, corpus, existing, base, False, args.out / "existing.log")
    assert existing.read_text() == "preserve this output\n"
    fake = args.out / "djxl-fails"
    fake.write_text('#!/bin/sh\nif [ "$1" = --version ]; then echo "djxl v0.12.0"; exit 0; fi\necho injected-decoder-failure >&2\nexit 37\n')
    fake.chmod(0o755)
    failed_output = args.out / "decoder-failure.tsv"
    with (args.out / "decoder-failure.log").open("x") as log:
        result = subprocess.run([str(binary), str(corpus), str(failed_output), *base],
                                stdout=log, stderr=subprocess.STDOUT,
                                env={**os.environ, "DJXL_PATH": str(fake.resolve())})
    assert result.returncode != 0 and not failed_output.exists()
    assert "djxl failed: injected-decoder-failure" in (args.out / "decoder-failure.log").read_text()
    # The real inversion must remain a failure with the original bytes and scores.
    output = args.out / "known.tsv"
    run(binary, corpus, output, base, False, args.out / "known.log")
    with args.before.open() as stream:
        old = list(csv.DictReader(stream, delimiter="\t"))
    with output.open() as stream:
        new = list(csv.DictReader(stream, delimiter="\t"))
    assert len(old) == len(new) == 6
    for a, b in zip(old, new):
        assert {k: a[k] for k in a if k != "ms"} == {k: b[k] for k in a if k != "ms"}
        artifact = Path(str(output) + ".artifacts") / (b["encoded_sha256"] + ".jxl")
        assert artifact.stat().st_size == int(b["bytes"])
    print("10 invalid-input/output controls and an injected decoder failure reject; all six real cell values remain unchanged")


if __name__ == "__main__":
    main()
