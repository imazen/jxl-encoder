#!/usr/bin/env python3
"""Exercise original color grammar and missing-ICC faults in an owned export."""
import argparse
from pathlib import Path
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("manifest", type=Path)
p.add_argument("logs", type=Path)
a = p.parse_args()
root = a.manifest.resolve().parent
if (root / ".git").exists() or not (root.parent.parent / "provenance.json").is_file():
    p.error("mutations require an owned prepare_clean_source.py export")
a.logs.mkdir(parents=True, exist_ok=False)
header = root / "jxl-encoder/src/headers/color_encoding.rs"
api = root / "jxl-encoder/src/api.rs"
original = {path: path.read_text() for path in [header, api]}
s = original[header]
xyb_guard = "if self.color_space != ColorSpace::Xyb {"
assert s.count(xyb_guard) == 2
calls = """        validate_icc_presence(
            self.color_encoding.as_ref(),
            self.metadata.and_then(|metadata| metadata.icc_profile),
        )?;
"""
assert original[api].count(calls) == 2
stream = "        validate_icc_presence(self.color_encoding.as_ref(), self.icc_profile.as_deref())?;\n"
assert original[api].count(stream) == 1
mutations = [
    ("raw-color-enum", header, s.replace("writer.write_enum_default(self.color_space as u32)?;", "writer.write(2, self.color_space as u64)?;"), "--lib", "color_grammar_matches_unmodified_libjxl_v012"),
    ("xyb-white-point", header, s.replace(xyb_guard, "if true {", 1), "--lib", "color_grammar_matches_unmodified_libjxl_v012"),
    ("xyb-transfer", header, s[:s.rindex(xyb_guard)] + s[s.rindex(xyb_guard):].replace(xyb_guard, "if true {", 1), "--lib", "color_grammar_matches_unmodified_libjxl_v012"),
    ("unknown-primaries", header, s.replace("matches!(self.color_space, ColorSpace::Rgb | ColorSpace::Unknown)", "self.color_space == ColorSpace::Rgb"), "--lib", "unknown_description_carries_rgb_primaries"),
    ("animation-icc-admission", api, original[api].replace(calls, "", 1), "--test", "explicit_icc_flag"),
    ("still-icc-admission", api, original[api][:original[api].rindex(calls)] + original[api][original[api].rindex(calls):].replace(calls, "", 1), "--test", "explicit_icc_flag"),
    ("streaming-icc-admission", api, original[api].replace(stream, ""), "--test", "explicit_icc_flag"),
]
for label, path, changed, kind, test in mutations:
    assert changed != original[path], label
    try:
        path.write_text(changed)
        command = ["cargo", "test", "--locked", "--manifest-path", str(a.manifest), "-p", "jxl-encoder", "--features", "__expert", kind]
        if kind == "--test":
            command.append("frame_header_contract")
        command += [test, "--", "--nocapture"]
        with (a.logs / f"{label}.log").open("x") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        output = (a.logs / f"{label}.log").read_text()
        if result.returncode != 101 or "test result: FAILED" not in output:
            raise RuntimeError(f"{label}: expected regression failure, rc={result.returncode}")
        print(f"{label}: regression detects original fault", flush=True)
    finally:
        path.write_text(original[path])
