#!/usr/bin/env python3
"""Fault injection in an owned CI-pinned export; restores every source on exit."""
import argparse
from pathlib import Path
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('manifest', type=Path)
p.add_argument('logs', type=Path)
a = p.parse_args()
root = a.manifest.resolve().parent
if (root / '.git').exists() or not (root.parent.parent / 'provenance.json').is_file():
    p.error('requires an owned prepare_clean_source.py export')
a.logs.mkdir(exist_ok=False)
mutations = [
    ('presence', 'jxl-encoder/src/headers/file_header.rs',
     'if self.metadata.color_encoding.want_icc != icc.is_some() {',
     'if false {', '--lib', 'headers::file_header::tests::icc_pair_validation_precedes_header_writes'),
    ('model', 'jxl-encoder/src/headers/file_header.rs',
     'if gray != (icc.get(16..20) == Some(b"GRAY")) {',
     'if gray != gray {', '--test', 'color_profile_contract', 'jpeg_icc_channel_model_mismatches_are_refused'),
    ('canonical', 'jxl-encoder/src/api.rs',
     'color.want_icc = metadata.icc_profile.is_some();',
     'color.want_icc = color.want_icc;', '--test', 'color_profile_contract', 'canonicalization_keeps_icc_presence_and_channel_model_consistent'),
    ('gray-animation', 'jxl-encoder/src/api/animate.rs',
     'enc.is_grayscale = layout.is_grayscale();',
     'enc.is_grayscale = false;', '--test', 'color_profile_contract', 'lossy_gray_profiles_match_animation_and_still_output_models'),
]
for name, relative, needle, replacement, *selection in mutations:
    source = root / relative
    original = source.read_text()
    assert original.count(needle) == 1, name
    try:
        source.write_text(original.replace(needle, replacement, 1))
        log = a.logs / (name + '.log')
        with log.open('x') as sink:
            result = subprocess.run(['cargo', 'test', '--locked', '--manifest-path', str(a.manifest),
                '-p', 'jxl-encoder', '--features', '__expert,jpeg-reencoding,chroma-subsampling,__pre_quantized',
                *selection], stdout=sink, stderr=subprocess.STDOUT)
        text = log.read_text()
        assert result.returncode == 101 and 'test result: FAILED' in text, (name, result.returncode)
        print(name + ': regression detected', flush=True)
    finally:
        source.write_text(original)
