#!/usr/bin/env python3
"""Build a component oracle from an existing unmodified libjxl v0.12 build."""
import argparse
from pathlib import Path
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("source", type=Path, help="source exported from v0.12.0")
p.add_argument("build", type=Path, help="existing Ninja build of v0.12.0")
p.add_argument("output", type=Path, help="new executable path")
a = p.parse_args()
if a.output.exists():
    p.error("refusing to overwrite an oracle executable")
version = subprocess.run([str(a.build / "tools/cjxl"), "--version"],
                         text=True, capture_output=True, check=True)
if "v0.12.0 a7a9c78" not in version.stdout + version.stderr:
    p.error("the existing build must be libjxl v0.12.0 a7a9c78")
line = next(line for line in (a.build / "build.ninja").read_text().splitlines()
            if line.startswith("build lib/libjxl.so.0.12.0:"))
objects = [str(a.build / word) for word in line.split() if word.endswith(".o")]
command = ["c++", "-std=c++17", "-O2", "-fno-rtti", "-DNDEBUG",
           "-I" + str(a.source), "-I" + str(a.source / "lib/include"),
           "-I" + str(a.build / "lib/include"),
           str(Path(__file__).with_suffix(".cc")), *objects,
           "-L" + str(a.build / "lib"), "-Wl,-rpath," + str(a.build / "lib"),
           "-ljxl_cms", "-lhwy", "-lbrotlidec", "-lbrotlicommon", "-lbrotlienc",
           "-pthread", "-o", str(a.output)]
subprocess.run(command, check=True)
subprocess.run([str(a.output)], check=True)
