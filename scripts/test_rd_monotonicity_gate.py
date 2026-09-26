"""Exercise the real distance gate's failure paths and retained artifacts."""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class DistanceGateTest(unittest.TestCase):
    def setUp(self):
        scratch = Path.home() / "tmp"
        scratch.mkdir(exist_ok=True)
        self.root = Path(tempfile.mkdtemp(prefix="rd-gate-test-", dir=scratch))
        self.binary = Path(os.environ["RD_MONOTONICITY_PROBE"]).resolve()
        self.image = ROOT / "jxl-encoder/tests/images/frymire-srgb.png"
        self.out = self.root / "result.tsv"
        self.env = dict(os.environ)
        print(f"evidence: {self.root}", flush=True)

    def run_gate(self, source=None, size=64, extra=()):
        result = subprocess.run([str(self.binary), str(source or self.image), str(self.out),
            "--images", "1", "--size", str(size), "--efforts", "3", "--distances", "1,2",
            "--baseline", "/dev/null", *extra], cwd=ROOT, env=self.env, capture_output=True, text=True)
        (self.root / "process.log").write_text(result.stdout + result.stderr)
        return result

    def assert_failure(self, result, message):
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(message, result.stderr)
        self.assertNotIn("IQA MONOTONICITY: clean", result.stdout)

    def fake_tool(self, name, behavior):
        path = self.root / name
        path.write_text(f'#!/bin/sh\nif [ "$1" = --version ]; then echo "{name} v0.12.0"; exit 0; fi\n' + behavior)
        path.chmod(0o755)
        self.env[name.upper() + "_PATH"] = str(path)

    def test_missing_directory_fails(self):
        self.assert_failure(self.run_gate(self.root / "missing"), "read corpus directory")

    def test_empty_directory_fails(self):
        source = self.root / "empty"
        source.mkdir()
        self.assert_failure(self.run_gate(source), "no PNG inputs selected")

    def test_corrupt_png_fails(self):
        source = self.root / "bad.png"
        source.write_bytes(b"not a PNG")
        self.assert_failure(self.run_gate(source), "bad.png")

    def test_undersized_input_fails(self):
        self.assert_failure(self.run_gate(size=9999), "smaller than requested crop")

    def test_existing_output_is_preserved(self):
        self.out.write_text("previous evidence\n")
        self.assert_failure(self.run_gate(), "refusing existing output")
        self.assertEqual(self.out.read_text(), "previous evidence\n")

    def test_reference_failure_is_not_zero_bytes(self):
        self.fake_tool("cjxl", "echo injected-reference-failure >&2\nexit 37\n")
        self.assert_failure(self.run_gate(), "cjxl failed: injected-reference-failure")
        self.assertEqual(len(self.out.read_text().splitlines()), 1)

    def test_decoder_failure_is_fatal(self):
        self.fake_tool("djxl", "echo injected-decoder-failure >&2\nexit 37\n")
        self.assert_failure(self.run_gate(), "djxl failed: injected-decoder-failure")
        self.assertEqual(len(self.out.read_text().splitlines()), 1)

    def test_corrupt_reference_is_fatal(self):
        self.fake_tool("cjxl", 'printf broken > "$2"\n')
        self.assert_failure(self.run_gate(), "jxl-rs image header")

    def test_real_single_and_multigroup_artifacts(self):
        for size in [64, 259]:
            self.out = self.root / f"size-{size}.tsv"
            result = self.run_gate(size=size)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            rows = list(csv.DictReader(self.out.read_text().splitlines(), delimiter="\t"))
            self.assertEqual(len(rows), 2)
            artifacts = Path(str(self.out) + ".artifacts")
            metadata = json.loads((artifacts / "manifest.json").read_text())
            self.assertEqual(metadata["binary_sha256"], hashlib.sha256(self.binary.read_bytes()).hexdigest())
            for row in rows:
                self.assertEqual(row["source_sha256"], hashlib.sha256(self.image.read_bytes()).hexdigest())
                for field, ext in [("encoded_sha256", "jxl"), ("cjxl_sha256", "jxl"), ("diffmap_sha256", "bfmap")]:
                    data = (artifacts / f"{row[field]}.{ext}").read_bytes()
                    self.assertEqual(hashlib.sha256(data).hexdigest(), row[field])
                    if ext == "bfmap":
                        self.assertEqual(len(data), 16 + size * size * 4)
                self.assertTrue((artifacts / f"{row['encoded_sha256']}.djxl.log").is_file())
                self.assertTrue((artifacts / f"{row['cjxl_sha256']}.djxl.log").is_file())


if __name__ == "__main__":
    unittest.main()
