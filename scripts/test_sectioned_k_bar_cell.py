"""Failure propagation for the local sectioned comparison driver."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).with_name("sectioned_k_bar_cell.sh").resolve()


class BarDriverTest(unittest.TestCase):
    def setUp(self):
        scratch = Path.home() / "tmp"
        scratch.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(prefix="sectioned-driver-", dir=scratch)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.image = self.root / "source.png"
        self.image.write_bytes(b"fixture")
        self.out = self.root / "result.tsv"
        self.cjxl = self.root / "cjxl"
        self.cjxl.write_text("#!/bin/sh\nif [ \"$1\" = --version ]; then echo test-reference; exit 0; fi\nexit 37\n")
        self.cjxl.chmod(0o755)
        self.probe = self.root / "probe"
        self.probe.write_text("#!/bin/sh\nif [ \"$1\" = reference-tools ]; then echo \"$CJXL_PATH\"; exit 0; fi\nexit 41\n")
        self.probe.chmod(0o755)
        self.env = dict(os.environ, SECTIONED_K_PROBE=str(self.probe), CJXL_PATH=str(self.cjxl), ARTIFACT_DIR=str(self.root / "artifacts"))

    def run_driver(self):
        return subprocess.run(["bash", str(SCRIPT), str(self.out), str(self.image), "1"], env=self.env, capture_output=True, text=True)

    def test_reference_failure_preserves_log_and_stops_before_result(self):
        result = self.run_driver()
        self.assertEqual(result.returncode, 37, result.stderr)
        self.assertEqual(len(self.out.read_text().splitlines()), 1)
        self.assertTrue((self.root / "result.logs/cjxl-e7-t1-r1.log").is_file())

    def test_existing_results_are_never_overwritten(self):
        self.out.write_text("existing measurement\n")
        result = self.run_driver()
        self.assertEqual(result.returncode, 2)
        self.assertEqual(self.out.read_text(), "existing measurement\n")

    def test_probe_failure_stops_after_reference(self):
        self.cjxl.write_text("#!/bin/sh\nif [ \"$1\" = --version ]; then echo test-reference; exit 0; fi\nfor arg; do output=$arg; done\nprintf encoded > \"$output\"\n")
        result = self.run_driver()
        self.assertEqual(result.returncode, 41, result.stderr)
        lines = self.out.read_text().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[1].startswith("cjxl\t"))
        self.assertFalse(any(line.startswith("ours\t") for line in lines))


if __name__ == "__main__":
    unittest.main()
