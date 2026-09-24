"""Regression gates for the LZ77 comparison's evidence checks."""

import csv
import hashlib
from pathlib import Path
import tempfile
import unittest

import lz77_hash_ab_join as join


class ComparisonTests(unittest.TestCase):
    def test_equal_sizes_do_not_prove_byte_identity(self):
        left = {"cell": {"bytes": "2", "encoded_sha256": "ab"}}
        right = {"cell": {"bytes": "2", "encoded_sha256": "cd"}}
        self.assertEqual(join.byte_identity(left, right, ["cell"]), (0, 1))

    def test_legacy_size_only_records_have_no_identity_evidence(self):
        records = {"cell": {"bytes": "2"}}
        self.assertEqual(join.byte_identity(records, records, ["cell"]), (0, 0))

    def test_missing_cells_are_not_silently_dropped(self):
        with self.assertRaisesRegex(ValueError, "unpaired cells"):
            join.paired_keys({"a": {}, "b": {}}, {"a": {}})

    def test_exact_hashes_are_counted(self):
        records = {"cell": {"encoded_sha256": "ab"}}
        self.assertEqual(join.byte_identity(records, records, ["cell"]), (1, 1))

    def test_tables_reject_duplicates_and_empty_inputs(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / "tmp") as scratch:
            path = Path(scratch) / "cells.tsv"
            fields = ["image", "path_kind", "depth", "effort"]
            row = dict(zip(fields, ["photo.png", "lossless", "u8", "8"]))
            with path.open("w") as stream:
                writer = csv.DictWriter(stream, fields, delimiter="\t")
                writer.writeheader()
            with self.assertRaisesRegex(ValueError, "empty"):
                join.load(path)
            with path.open("a") as stream:
                writer = csv.DictWriter(stream, fields, delimiter="\t")
                writer.writerows([row, row])
            with self.assertRaisesRegex(ValueError, "duplicate"):
                join.load(path)

    def test_artifact_bytes_and_size_must_match_record(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / "tmp") as scratch:
            path = Path(scratch) / "encoded.jxl"
            path.write_bytes(b"abc")
            rows = {"cell": {"artifact": str(path), "bytes": "3",
                             "encoded_sha256": hashlib.sha256(b"abc").hexdigest()}}
            join.verify_artifacts(rows)
            path.write_bytes(b"abd")
            with self.assertRaisesRegex(ValueError, "differs"):
                join.verify_artifacts(rows)
            path.write_bytes(b"abc")
            rows["cell"]["bytes"] = "4"
            with self.assertRaisesRegex(ValueError, "differs"):
                join.verify_artifacts(rows)


if __name__ == "__main__":
    unittest.main()
