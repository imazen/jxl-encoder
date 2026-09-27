#!/usr/bin/env python3
"""Build immutable upstream revisions for the #110 regression comparison.

Run under nice. Sources, dependencies, logs and binaries stay in the supplied
artifact directory; existing checkouts and the v0.12 reference are untouched.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile
import urllib.request

REVISIONS = {
    "parent": "196a43d996aa6ed33ebf98812a7c6d43b2b6d01b",
    "rewrite": "e8ff09762481785938d8e4e01333ed3917571161",
    "current": "b87738951c1254cd8cccaa6d47712ba735da56d8",
}


def fetch(root, project, revision):
    dest = root / revision
    archive = root / (revision + ".tar.gz")
    if not archive.exists():
        url = f"https://codeload.github.com/{project}/tar.gz/{revision}"
        print(f"download {url}", flush=True)
        with urllib.request.urlopen(url) as response:
            data = response.read()
        archive.write_bytes(data)
    if not dest.exists():
        dest.mkdir()
        with tarfile.open(archive) as tar:
            members = tar.getmembers()
            prefix = members[0].name.split('/')[0] + '/'
            for member in members:
                if member.name.startswith(prefix):
                    member.name = member.name[len(prefix):]
                    if member.name:
                        tar.extract(member, dest, filter="data")
    return dest, hashlib.sha256(archive.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    cache = root / "sources"
    cache.mkdir(exist_ok=True)
    records = {}
    for arm, revision in REVISIONS.items():
        src, sha = fetch(cache, "libjxl/libjxl", revision)
        deps = {}
        script = (src / "deps.sh").read_text()
        for name, project in [("brotli", "google/brotli"),
                              ("highway", "google/highway"),
                              ("skcms", "google/skcms")]:
            rev = re.search(rf'THIRD_PARTY_{name.upper()}="([0-9a-f]+)"', script)[1]
            dep, dep_sha = fetch(cache, project, rev)
            target = src / "third_party" / name
            if not target.is_symlink():
                target.rmdir()  # only remove the archive's empty submodule directory
                target.symlink_to(dep, target_is_directory=True)
            deps[name] = {"revision": rev, "archive_sha256": dep_sha}
        build = root / (arm + "-build")
        command = ["cmake", "-S", str(src), "-B", str(build), "-G", "Ninja",
                   "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_TESTING=OFF",
                   "-DJPEGXL_ENABLE_OPENEXR=OFF", "-DJPEGXL_ENABLE_SJPEG=OFF",
                   "-DJPEGXL_ENABLE_SKCMS=ON", "-DJPEGXL_ENABLE_JNI=OFF",
                   "-DJPEGXL_ENABLE_EXAMPLES=OFF", "-DJPEGXL_ENABLE_BENCHMARK=OFF",
                   "-DJPEGXL_ENABLE_MANPAGES=OFF", "-DJPEGXL_ENABLE_DOXYGEN=OFF",
                   "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
                   "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache"]
        env = dict(os.environ, CCACHE_BASEDIR=str(src), TMPDIR=str(Path.home() / "tmp"))
        with (root / (arm + "-build.log")).open("a") as log:
            print(f"configure {arm}", flush=True)
            subprocess.run(command, env=env, stdout=log, stderr=log, check=True)
            print(f"build {arm}; log: {log.name}", flush=True)
            subprocess.run(["cmake", "--build", str(build), "--target", "cjxl",
                            "--parallel", str(args.jobs)], env=env,
                           stdout=log, stderr=log, check=True)
        binary = build / "tools/cjxl"
        records[arm] = {"revision": revision, "archive_sha256": sha,
                        "dependencies": deps, "configure": command,
                        "binary": str(binary),
                        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                        "library_sha256": {
                            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted((build / "lib").glob("*.dylib"))}}
        (root / "builds.json").write_text(json.dumps(records, indent=2) + "\n")
        print(f"completed {arm}: {records[arm]['binary_sha256']}", flush=True)


if __name__ == "__main__":
    main()
