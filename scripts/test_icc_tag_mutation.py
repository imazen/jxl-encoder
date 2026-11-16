#!/usr/bin/env python3
"""Restore the original ICC tag-list decision in an owned pinned export."""
import argparse
from pathlib import Path
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('manifest', type=Path)
p.add_argument('log', type=Path)
a = p.parse_args()
root = a.manifest.resolve().parent
if (root / '.git').exists() or not (root.parent.parent / 'provenance.json').is_file():
    p.error('requires an owned prepare_clean_source.py export')
source = root / 'jxl-encoder/src/icc.rs'
original = source.read_text()
needle = '    if can_predict_tag_list(icc) {'
assert original.count(needle) == 1
try:
    source.write_text(original.replace(needle, '    if pos + 4 <= size {', 1))
    with a.log.open('x') as log:
        result = subprocess.run(['cargo', 'test', '--locked', '--manifest-path', str(a.manifest),
            '-p', 'jxl-encoder', '--features', '__expert,jpeg-reencoding,chroma-subsampling',
            '--test', 'icc_tag_contract'], stdout=log, stderr=subprocess.STDOUT)
    text = a.log.read_text()
    assert result.returncode == 101 and 'test result: FAILED' in text, result.returncode
    for test in ['truncated_icc_tag_table_uses_opaque_prediction',
                 'raw_and_parsed_jpeg_preserve_opaque_icc_and_reconstruction',
                 'subsampled_requests_preserve_opaque_icc_tag_counts']:
        assert f'{test} ... FAILED' in text, test
    print('original ICC tag-list decision is detected by pixel, JPEG and subsampling paths')
finally:
    source.write_text(original)
