"""Regression checks for the independent codestream envelope parser."""
import unittest

from jxl_bitstream_diff import BitReader, Parser


class ReferenceFrameHeaderTest(unittest.TestCase):
    def walk_frame_envelopes(self, data):
        class Envelopes(Parser):
            def lf_global_prefix(self, *args):
                pass
            def hf_global_prefix(self):
                pass
        parser = Envelopes(data)
        parser.br = BitReader(data, 2)
        width, height = parser.size_header()
        parser.image_metadata()
        parser.custom_transform_data()
        parser.br.jump_to_byte_boundary()
        frames = []
        while parser.br.pos < len(data) * 8:
            parser.fields = []
            parser.toc = None
            start = parser.br.pos // 8
            parser.frame_header(width, height)
            parser.read_toc(width, height)
            end = parser.toc['payload_start'] + sum(parser.toc['entries'])
            self.assertTrue(start < end <= len(data), (start, end, len(data)))
            fields = {name: value for _, name, value in parser.fields}
            frames.append(fields)
            parser.br.pos = end * 8
            if fields.get('frame.all_default') or fields.get('frame.is_last'):
                break
        self.assertEqual(parser.br.pos, len(data) * 8)
        return frames

    def test_unsaved_positive_duration_omits_save_before_ct(self):
        # Actual stream fully rendered by primary Rust and djxl v0.12.
        # SHA256 6326e6d28a2abab268b0ebfc46202bfdeec1cb28b78064f05c464b08a11476f3
        data = bytes.fromhex(
            "ff0a410401880908022a00003c004b12a114845dd6ff00034ae2ab26fe08022a04003c004b12a114"
            "845dd6ff00034ae2ab26fe"
        )
        frames = self.walk_frame_envelopes(data)
        self.assertEqual(len(frames), 2)
        self.assertEqual(frames[0]['frame.save_as_reference'], 0)
        self.assertEqual(frames[0]['frame.animation.duration'], 10)
        self.assertNotIn('frame.save_before_color_transform', frames[0])

    def test_animation_timecodes_keep_following_toc_aligned(self):
        # Actual stream fully rendered by primary Rust and djxl v0.12.
        # SHA256 3755881f61d18066e5ff5557e30782fb1a97ba3090f48edabaa1dc645e46cc78
        data = bytes.fromhex(
            "ff0ac70581880908022a0800000008000028004b120839c00000c004000802520c00000004004400"
            "4b12a114845dfeff0003405205008539fe"
        )
        frames = self.walk_frame_envelopes(data)
        self.assertEqual(len(frames), 2)
        self.assertEqual([f['frame.animation.timecode'] for f in frames], [2, 3])
        self.assertEqual([f['frame.animation.duration'] for f in frames], [10, 20])

    def test_xyb_color_grammar_matches_pinned_component_writer(self):
        # Unmodified v0.12 ColorEncoding output: no white/primaries/TF fields.
        parser = Parser(bytes.fromhex("0800"))
        parser.br = BitReader(bytes.fromhex("0800"))
        parser.color_encoding()
        self.assertEqual(parser.br.pos, 10)
        fields = {name: value for _, name, value in parser.fields}
        self.assertEqual(fields["color_encoding.color_space"], "kXYB")
        self.assertEqual(fields["color_encoding.rendering_intent"], "kPerceptual")
        self.assertNotIn("color_encoding.have_gamma", fields)

    def test_custom_xy_matches_all_native_selector_widths(self):
        # Unmodified v0.12 ColorEncoding Bundle writer, custom_xy_oracle.cc.
        # Includes positive/negative coordinates and all four U32 selectors.
        data = bytes.fromhex("2000d43000352c047c92006ad88fd005e6aa013ec9bfc68d1b")
        parser = Parser(data)
        parser.br = BitReader(data)
        parser.color_encoding()
        self.assertEqual(parser.br.pos, 198)
        fields = {name: value for _, name, value in parser.fields}
        self.assertEqual(fields["color_encoding.white.xy"], (100000, 200000))
        self.assertEqual(fields["color_encoding.red.xy"], (300000, 400000))
        self.assertEqual(fields["color_encoding.green.xy"], (-500000, 700000))
        self.assertEqual(fields["color_encoding.blue.xy"], (1200000, -1500000))
        self.assertEqual(fields["color_encoding.transfer_function"], "kSRGB")
        self.assertEqual(fields["color_encoding.rendering_intent"], "kRelative")

    def test_unknown_description_has_primaries(self):
        parser = Parser(bytes.fromhex("18c505"))
        parser.br = BitReader(bytes.fromhex("18c505"))
        parser.color_encoding()
        self.assertEqual(parser.br.pos, 21)
        fields = {name: value for _, name, value in parser.fields}
        self.assertEqual(fields["color_encoding.color_space"], "kUnknown")
        self.assertEqual(fields["color_encoding.primaries"], "kSRGB")
        self.assertEqual(fields["color_encoding.transfer_function"], "kSRGB")

    def test_float_alpha_keeps_pq_metadata_aligned(self):
        # Real issue-138 cell 2965/49033, rendered by jxl-rs and djxl.
        # Full stream SHA256 e5ba20293607ebbad07de42001bc95e617b1037bbc6941a64b409a3eb0482d42.
        data = bytes.fromhex(
            "ff0a0008fc09e4841c0879f488c30100000020e01b5080000024019f3c40a65c047"
            "1631500000a8d7d72dbaaaafa7f00ff0f600925c62df339399394d0a0015c1a340cd3300d6f43"
        )
        parser = Parser(data)
        parser.br = BitReader(data, 2)
        self.assertEqual(parser.size_header(), (255, 257))
        parser.image_metadata()
        fields = {name: value for _, name, value in parser.fields}
        self.assertEqual(fields["metadata.extra_channel[0].bit_depth.bits_per_sample"], 32)
        self.assertEqual(fields["metadata.extra_channel[0].alpha_associated"], 0)
        self.assertEqual(fields["metadata.xyb_encoded"], 1)
        self.assertEqual(fields["metadata.color_encoding.transfer_function"], "kPQ")
        self.assertEqual(fields["metadata.tone_mapping.intensity_target"], 10000.0)
        self.assertEqual(fields["metadata.tone_mapping.min_nits"], 0.0)

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
