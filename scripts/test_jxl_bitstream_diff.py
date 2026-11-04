"""Regression checks for the independent codestream envelope parser."""
import unittest

from jxl_bitstream_diff import BitReader, Parser


class ReferenceFrameHeaderTest(unittest.TestCase):
    def test_float_metadata_keeps_the_following_fields_aligned(self):
        # Real PQ F32 stream decoded by jxl-rs and djxl v0.12, SHA256
        # 09c4f34077e7fe011a82a0b0b3c0683470b2eeb4880953798c742b851a1ec2f4.
        data = bytes.fromhex(
            "ff0af007000ae410f2e811870300000040e01b14080024011d30812c4c00716315"
            "00000a8d7d72dbaaaafa7f00ff0f600925c62df339399394d0a0015c1a340c"
        )
        parser = Parser(data)
        parser.br = BitReader(data, 2)
        self.assertEqual(parser.size_header(), (257, 255))
        parser.image_metadata()
        fields = {name: value for _, name, value in parser.fields}
        self.assertEqual(fields["metadata.bit_depth.bits_per_sample"], 32)
        self.assertEqual(fields["metadata.bit_depth.exponent_bits"], 8)
        self.assertEqual(fields["metadata.xyb_encoded"], 1)
        self.assertEqual(fields["metadata.color_encoding.transfer_function"], "kPQ")
        self.assertEqual(fields["metadata.tone_mapping.intensity_target"], 10000.0)
        self.assertEqual(fields["metadata.tone_mapping.min_nits"], 0.0)

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
