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

    def run_driver(self, reps=1, efforts="7 9", threads="1 8"):
        return subprocess.run(["bash", str(SCRIPT), str(self.out), str(self.image), str(reps), efforts, threads], env=self.env, capture_output=True, text=True)

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

    def success_stubs(self):
        self.cjxl.write_text("#!/bin/sh\nif [ \"$1\" = --version ]; then echo test-reference; exit 0; fi\nfor arg; do output=$arg; done\nprintf encoded > \"$output\"\n")
        self.probe.write_text(
            "#!/bin/sh\nif [ \"$1\" = reference-tools ]; then echo \"$CJXL_PATH\"; exit 0; fi\n"
            "shift 4\nfor arm; do\n"
            "  echo \"== $arm: 10 bytes, 1.0 ms total\"\n"
            "  echo \"artifact $arm " + "a" * 64 + " warmup " + "a" * 64 + "\"\ndone\n"
        )
        baseline = self.root / "baseline"
        baseline.write_text(self.probe.read_text())
        baseline.chmod(0o755)
        self.env["SECTIONED_K_BASELINE_PROBE"] = str(baseline)
        return baseline

    def test_binary_comparison_alternates_order(self):
        self.success_stubs()
        result = self.run_driver(reps=2)
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = [line.split("\t") for line in self.out.read_text().splitlines()[1:]]
        self.assertEqual(len(rows), 24)
        for effort in ("7", "9"):
            for threads in ("1", "8"):
                for rep, expected in (("1", ["cjxl", "baseline", "ours"]),
                                      ("2", ["ours", "baseline", "cjxl"])):
                    selected = [r for r in rows if (r[1], r[2], r[4]) == (effort, threads, rep)]
                    self.assertEqual([r[0] for r in selected], expected)
                    self.assertTrue(all(r[3] == "default" for r in selected if r[0] != "cjxl"))
        self.assertIn("baseline_probe=", Path(str(self.out) + ".meta").read_text())

    def test_baseline_failure_stops_before_candidate(self):
        baseline = self.success_stubs()
        baseline.write_text("#!/bin/sh\nexit 43\n")
        result = self.run_driver()
        self.assertEqual(result.returncode, 43, result.stderr)
        self.assertEqual(len(self.out.read_text().splitlines()), 2)
        self.assertFalse((self.root / "result.logs/ours-e7-t1-r1.log").exists())

    def test_selected_cell_and_invalid_grid(self):
        self.success_stubs()
        result = self.run_driver(efforts="9", threads="8")
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = [line.split("\t") for line in self.out.read_text().splitlines()[1:]]
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(r[1:3] == ["9", "8"] for r in rows))
        for bad in ("9 x", "9 0", "0", "-1"):
            result = self.run_driver(efforts=bad)
            self.assertEqual(result.returncode, 2, result.stderr)


if __name__ == "__main__":
    unittest.main()
