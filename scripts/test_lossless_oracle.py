"""Exercise the real oracle CLI, including persisted output and refusal paths.

Caller supplies LOSSLESS_ORACLE_PROBE. Full artifacts remain under ~/tmp.
"""

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class OracleCli(unittest.TestCase):
    def setUp(self):
        self.binary = Path(os.environ["LOSSLESS_ORACLE_PROBE"]).resolve()
        scratch = Path.home() / "tmp"
        scratch.mkdir(exist_ok=True)
        self.root = Path(tempfile.mkdtemp(prefix="lossless-oracle-cli-", dir=scratch))
        self.source = (Path(__file__).resolve().parents[1]
                       / "jxl-encoder/tests/images/frymire-srgb.png")
        self.manifest = self.root / "inputs.tsv"
        self.write_manifest(hashlib.sha256(self.source.read_bytes()).hexdigest())

    def write_manifest(self, sha):
        self.manifest.write_text(
            "sha256\tsplit\tcontent_class\twidth\theight\tpath\n"
            f"{sha}\ttrain\tgraphics\t0\t0\t{self.source}\n")

    def run_probe(self, *extra):
        result = subprocess.run([
            str(self.binary), "--manifest", str(self.manifest),
            "--output", str(self.root / "rows.tsv"),
            "--features-output", str(self.root / "features.tsv"),
            "--sizes", "16", "--samples-per-cell", "0", *extra,
        ], capture_output=True, text=True)
        with (self.root / "process.log").open("a") as log:
            log.write(result.stdout + result.stderr)
        return result

    def test_all_anchor_rows_reference_verified_artifacts_and_refuse_overwrite(self):
        result = self.run_probe()
        self.assertEqual(result.returncode, 0, result.stderr)
        rows_path = self.root / "rows.tsv"
        original = rows_path.read_bytes()
        with rows_path.open() as file:
            rows = list(csv.DictReader(file, delimiter="\t"))
        self.assertEqual({int(row["cell_id"]) for row in rows}, set(range(16)))
        self.assertEqual(len(rows), 16)
        for row in rows:
            encoded = (self.root / "rows.artifacts"
                       / (row["encoded_sha256"] + ".jxl")).read_bytes()
            self.assertEqual(hashlib.sha256(encoded).hexdigest(), row["encoded_sha256"])
            self.assertEqual(len(encoded), int(row["bytes"]))
            self.assertGreater(float(row["encode_ms"]), 0)
        metadata = json.loads((self.root / "rows.artifacts/_MANIFEST.json").read_text())
        self.assertEqual(len(metadata["build_commit"]), 40)
        self.assertEqual(metadata["binary_sha256"], hashlib.sha256(self.binary.read_bytes()).hexdigest())
        self.assertEqual(metadata["manifest_sha256"], hashlib.sha256(self.manifest.read_bytes()).hexdigest())
        self.assertNotEqual(self.run_probe().returncode, 0)
        self.assertEqual(rows_path.read_bytes(), original)

    def test_changed_source_fails_without_encode_rows(self):
        self.write_manifest("0" * 64)
        result = self.run_probe()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("source-file hash mismatch", result.stderr)
        self.assertEqual(len((self.root / "rows.tsv").read_text().splitlines()), 1)
        self.assertEqual(list((self.root / "rows.artifacts").glob("*.jxl")), [])

    def test_missing_source_fails(self):
        self.source = self.root / "missing.png"
        self.write_manifest("0" * 64)
        self.assertNotEqual(self.run_probe().returncode, 0)

    def test_custom_sizes_have_distinct_feature_join_keys(self):
        result = self.run_probe("--features-only", "--sizes", "24")
        self.assertEqual(result.returncode, 0, result.stderr)
        with (self.root / "features.tsv").open() as file:
            rows = list(csv.DictReader(file, delimiter="\t"))
        self.assertEqual({row["size_class"] for row in rows}, {"max16", "max24"})
        self.assertEqual(len(rows), 2)

    def test_empty_selection_fails(self):
        result = self.run_probe("--split", "absent")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("selected no images", result.stderr)


if __name__ == "__main__":
    unittest.main()
