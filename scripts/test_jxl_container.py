"""Byte-preservation and refusal controls for the shared box extractor."""
import unittest
from jxl_container import codestream


def box(tag, data):
    return (len(data) + 8).to_bytes(4, 'big') + tag + data


class CodestreamTest(unittest.TestCase):
    def test_bare_and_single_box_preserve_bytes_and_ranges(self):
        raw = b'\xff\x0a\x01\x02\x03'
        self.assertEqual(codestream(raw), (raw, [(0, 5)]))
        data = box(b'JXL ', b'\r\n\x87\n') + box(b'jxlc', raw)
        self.assertEqual(codestream(data), (raw, [(20, 5)]))

    def test_fragments_preserve_bytes_and_ranges(self):
        data = box(b'JXL ', b'\r\n\x87\n') + box(b'jxlp', b'\0\0\0\0\xff\x0a')
        data += box(b'jxlp', b'\x80\0\0\x01\x01\x02\x03')
        self.assertEqual(codestream(data), (b'\xff\x0a\x01\x02\x03', [(24, 2), (38, 3)]))

    def test_invalid_box_combinations_and_sizes_refuse(self):
        signature = box(b'JXL ', b'\r\n\x87\n')
        for data in [signature, signature + box(b'jxlp', b'\0\0\0\0x'),
                     signature + box(b'jxlp', b'\x80\0\0\x01x'),
                     signature + box(b'jxlc', b'x') + box(b'jxlc', b'y'),
                     signature + b'\0\0\0\x07jxlc', signature + b'\0\0\0\xffjxlc']:
            with self.subTest(data=data), self.assertRaises(AssertionError):
                codestream(data)
