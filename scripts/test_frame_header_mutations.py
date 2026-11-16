#!/usr/bin/env python3
"""Prove each header regression detects its original fault in an owned export."""
import argparse
from pathlib import Path
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("manifest", type=Path)
p.add_argument("logs", type=Path)
a = p.parse_args()
root = a.manifest.resolve().parent
if (root / ".git").exists() or not (root.parent.parent / "provenance.json").is_file():
    p.error("mutations require a prepare_clean_source.py export, never a working checkout")
a.logs.mkdir(parents=True, exist_ok=False)
header = root / "jxl-encoder/src/headers/frame_header.rs"
original = header.read_text()
start = original.index("    fn covers_canvas(&self) -> bool {")
end = original.index("\n    }", start) + len("\n    }")
old_coverage = """    fn covers_canvas(&self) -> bool {
        self.x0 == 0 && self.y0 == 0 && self.width == 0 && self.height == 0
    }"""
guard = "if normal_frame {\n            for (i, &mode)"
assert original.count(guard) == 1
mutations = [
    ("reference-blends", original.replace(guard, "if true {\n            for (i, &mode)"),
     "reference_only_rgba_then_displayed_frame"),
    ("covering-crop", original[:start] + old_coverage + original[end:],
     "identical_eight_square_lossy_frames"),
]
for label, source, test in mutations:
    try:
        header.write_text(source)
        with (a.logs / f"{label}.log").open("x") as log:
            result = subprocess.run(["cargo", "test", "--locked", "--manifest-path", str(a.manifest),
                "-p", "jxl-encoder", "--features", "__expert", "--test", "frame_header_contract",
                test, "--", "--nocapture"], stdout=log, stderr=subprocess.STDOUT)
        if result.returncode != 101:
            raise RuntimeError(f"{label}: expected a failing regression, got {result.returncode}")
        text = (a.logs / f"{label}.log").read_text()
        if "test result:" not in text or "FAILED" not in text:
            raise RuntimeError(f"{label}: compile/tool failure is not a mutation verdict")
        print(f"{label}: production regression detects original fault", flush=True)
    finally:
        header.write_text(original)
