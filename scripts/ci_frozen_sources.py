#!/usr/bin/env python3
"""Materialize pinned sibling archives in a fresh CI reproduction layout.

The project checkout goes in ROOT/work/jxl-encoder. Sibling source directories
are immutable inputs; an existing directory is never overwritten.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=False)
    archives = args.root / "archives"
    archives.mkdir()
    pins = list(csv.DictReader(args.manifest.open(), delimiter="\t"))
    receipts = []
    for pin in pins:
        repository, revision = pin["repository"], pin["revision"]
        assert len(revision) == 40 and all(c in "0123456789abcdef" for c in revision)
        name = repository.split("/")[1]
        parent = args.root if name in ["butteraugli", "dssim"] else args.root / "work"
        parent.mkdir(exist_ok=True)
        archive = archives / (name + ".tar.gz")
        with archive.open("xb") as stream:
            subprocess.run(["gh", "api", f"repos/{repository}/tarball/{revision}",
                            "--allow-escape-sequences"], stdout=stream, check=True)
        staging = args.root / (name + "-unpack")
        staging.mkdir()
        with tarfile.open(archive) as package:
            package.extractall(staging, filter="data")
        roots = list(staging.iterdir())
        assert len(roots) == 1 and roots[0].is_dir()
        destination = parent / name
        assert not destination.exists()
        roots[0].rename(destination)
        staging.rmdir()
        with archive.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        receipt = dict(repository=repository, revision=revision, archive_sha256=digest,
                       path=str(destination))
        (destination / ".ci-source-pin.json").write_text(json.dumps(receipt, indent=2)+"\n")
        receipts.append(receipt)
        (args.root / "SOURCES.json").write_text(json.dumps(receipts, indent=2)+"\n")
        print(f"Materialized {repository} {revision}", flush=True)


if __name__ == "__main__":
    main()
