"""Regression checks for the independent codestream envelope parser."""
import unittest

from jxl_bitstream_diff import BitReader, Parser


class ReferenceFrameHeaderTest(unittest.TestCase):
    def test_reference_only_has_no_blending_mode(self):
        # First 20 bytes of the real imac_g3 d1/e7 codestream, SHA256
        # a2949c8da146943f087be29291e64813c15d8431855019b5530bdff5b61c4a60.
        # jxl-oxide and djxl v0.12 independently decode this reference frame.
        data = bytes.fromhex("ff0aba3b686f8184e2460c8e0111e00000068000")
        parser = Parser(data)
        parser.br = BitReader(data, 2)
        width, height = parser.size_header()
        parser.image_metadata()
        parser.custom_transform_data()
        parser.br.jump_to_byte_boundary()
        parser.frame_header(width, height)
        fields = {name: value for _, name, value in parser.fields}
        self.assertEqual((width, height), (2940, 1912))
        self.assertEqual(parser.frame["frame_type"], 2)
        self.assertEqual((parser.frame["fx"], parser.frame["fy"]), (268, 260))
        self.assertEqual(parser.frame["group_size_shift"], 2)
        self.assertEqual(fields["frame.save_as_reference"], 3)
        self.assertEqual(fields["frame.save_before_color_transform"], 1)
        self.assertNotIn("frame.blending.mode", fields)


if __name__ == "__main__":
    unittest.main()
