"""Artifact, ordering and failure controls for the process-wall A/B driver."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class DriverTests(unittest.TestCase):
    def test_retains_outputs_and_alternates(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / "tmp") as tmp:
            root = Path(tmp)
            trace = root / "trace.jsonl"
            binaries = []
            for name in ["base", "ours"]:
                binary = root / name
                binary.write_text(
                    f"#!{sys.executable}\n"
                    "import json, pathlib, sys\n"
                    f"with open({str(trace)!r}, 'a') as f: f.write(json.dumps(sys.argv) + '\\n')\n"
                    "pathlib.Path(sys.argv[2]).write_bytes(b'encoded')\n"
                    "print('diagnostic', file=sys.stderr)\n"
                )
                binary.chmod(0o755)
                binaries.append(binary)
            source = root / "source.png"
            source.write_bytes(b"input")
            out = root / "out.tsv"
            result = subprocess.run([
                sys.executable, str(Path(__file__).with_name("bench_lossless_ab.py")),
                "--base", str(binaries[0]), "--ours", str(binaries[1]),
                "--iters", "2", "--out", str(out), "--cell", f"smoke:{source}:7:1",
                "--lossy-distance", "4", "--strategy", "libjxl",
            ], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            calls = [json.loads(line) for line in trace.read_text().splitlines()]
            self.assertEqual([Path(c[0]).name for c in calls],
                             ["base", "ours", "base", "ours", "ours", "base"])
            for call in calls:
                self.assertNotIn("--lossless", call)
                self.assertEqual(call[call.index("-d") + 1], "4.0")
                self.assertEqual(call[call.index("--strategy") + 1], "libjxl")
            sha = hashlib.sha256(b"encoded").hexdigest()
            self.assertEqual((root / "out.artifacts" / f"{sha}.jxl").read_bytes(), b"encoded")
            self.assertIn(sha, out.read_text())
            self.assertIn("diagnostic", (root / "out.artifacts/smoke-1-base.jxl.log").read_text())
            self.assertEqual(json.loads(Path(str(out) + ".meta.json").read_text())["host"],
                             os.uname().nodename)

    def test_failed_encoder_is_not_a_successful_measurement(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / "tmp") as tmp:
            root = Path(tmp)
            binary = root / "failure"
            binary.write_text("#!/bin/sh\necho deliberate-failure >&2\nexit 7\n")
            binary.chmod(0o755)
            source = root / "source.png"
            source.write_bytes(b"input")
            result = subprocess.run([
                sys.executable, str(Path(__file__).with_name("bench_lossless_ab.py")),
                "--base", str(binary), "--ours", str(binary), "--iters", "1",
                "--out", str(root / "fail.tsv"), "--cell", f"smoke:{source}:7:1",
            ], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((root / "fail.tsv").exists())
            self.assertIn("deliberate-failure", (root / "fail.artifacts/smoke-warm-base.jxl.log").read_text())


if __name__ == "__main__":
    unittest.main()
