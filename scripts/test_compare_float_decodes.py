"""Check byte order, row order and strict failure behavior of float evidence reads."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from compare_float_decodes import read_pixels


class FloatEvidenceTests(unittest.TestCase):
    def test_formats_preserve_samples(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / "tmp") as folder:
            root = Path(folder)
            original = np.array([[[0.1, 0.2, -0.3], [0.4, 0.5, 0.6]],
                                 [[0.7, 0.8, 0.9], [1.1, -1.2, 0.0]]], dtype="<f4")
            for name, dtype, scale in [("little", "<f4", -1), ("big", ">f4", 1)]:
                file = root / f"{name}.pfm"
                file.write_bytes(f"PF\n2 2\n{scale}\n".encode() + original[::-1].astype(dtype).tobytes())
                np.testing.assert_array_equal(read_pixels(file), original)
            file = root / "pixels.f32le"
            file.write_bytes(original.tobytes())
            file.with_suffix(".json").write_text(json.dumps(dict(format="f32le", width=2, height=2, channels=3, samples=12)))
            np.testing.assert_array_equal(read_pixels(file), original)
            np.save(root / "pixels.npy", original[None])
            np.testing.assert_array_equal(read_pixels(root / "pixels.npy"), original)

    def test_nonfinite_and_incomplete_outputs_fail(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / "tmp") as folder:
            root = Path(folder)
            np.save(root / "nan.npy", np.full((2, 2, 3), np.nan, dtype="f4"))
            with self.assertRaisesRegex(AssertionError, "non-finite"):
                read_pixels(root / "nan.npy")
            file = root / "short.pfm"
            file.write_bytes(b"PF\n2 2\n-1\n" + bytes(4))
            with self.assertRaises(ValueError):
                read_pixels(file)


if __name__ == "__main__":
    unittest.main()
