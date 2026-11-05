"""A replay must never silently accept an empty or duplicated selection."""
from pathlib import Path
import tempfile
import unittest

from replay_issue138 import CASES, load_cases


class CaseSelectionTests(unittest.TestCase):
    def read(self, content):
        scratch = Path.home() / "tmp"
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            path = Path(directory) / "cases.tsv"
            path.write_text(content)
            return load_cases(path)

    def test_empty_or_incomplete_selection_is_refused(self):
        for content in ["", "seed\tindex\n", "seed\n2966\n", "seed\tindex\n2966\t\n"]:
            with self.subTest(content=content), self.assertRaises(ValueError):
                self.read(content)

    def test_duplicate_and_negative_cells_are_refused(self):
        for rows in ["2966\t213\n2966\t213\n", "2966\t-1\n", "-1\t213\n"]:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.read("seed\tindex\n" + rows)

    def test_seed_groups_keep_every_selected_cell(self):
        selected = self.read("seed\tindex\n2966\t213\n2965\t28968\n2966\t253\n")
        self.assertEqual(selected, {2966: [213, 253], 2965: [28968]})
        self.assertEqual(sum(map(len, CASES.values())), 21)


if __name__ == "__main__":
    unittest.main()
