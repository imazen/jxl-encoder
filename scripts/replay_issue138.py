#!/usr/bin/env python3
"""Replay every encoder failure reported in issue 138 through November 2.

Use the same CID22 validation, gb82-sc and ICC inputs as the weekly workflow.
Each cell retains its own complete output. A failed cell does not prevent the
remaining reported cells from running; the driver exits nonzero if any fail.
"""
import argparse
import subprocess
from pathlib import Path


CASES = {
    2964: [317, 400, 419, 931, 1041, 1495, 1915],
    2965: [28968, 49033, 70595, 77723],
    2966: [14694, 38645, 54486, 57574, 65418, 66247, 92940, 107803, 110894, 118545],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("icc", type=Path)
    parser.add_argument("--retain", action="store_true",
                        help="retain all encoded/source/reference artifacts and study rows")
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    photos = args.corpus / "CID22/CID22-512/validation"
    screens = args.corpus / "gb82-sc"
    for directory in [photos, screens, args.icc]:
        if not directory.is_dir():
            parser.error(f"missing input directory: {directory}")
    args.out.mkdir(parents=True, exist_ok=False)
    failed = []
    with (args.out / "replays.tsv").open("x") as results:
        results.write("seed\tindex\texit\n")
        for seed, indices in CASES.items():
            for index in indices:
                directory = args.out / f"{seed}-{index}"
                command = [str(binary), "--images", str(photos), "--images", str(screens),
                           "--icc", str(args.icc), "--out", str(directory),
                           "--seed", str(seed), "--only", str(index), "--threads", "1"]
                if args.retain:
                    command.extend(["--retain", "--reference-study", "--study-max-pixels", "100000000"])
                with (args.out / f"{seed}-{index}.log").open("x") as log:
                    log.write(f"command: {command!r}\n")
                    log.flush()
                    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
                results.write(f"{seed}\t{index}\t{result.returncode}\n")
                results.flush()
                print(f"{seed}/{index}: exit {result.returncode}", flush=True)
                if result.returncode:
                    failed.append((seed, index))
    raise SystemExit(bool(failed))


if __name__ == "__main__":
    main()
