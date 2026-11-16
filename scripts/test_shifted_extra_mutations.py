#!/usr/bin/env python3
"""Restore either issue-144 modular routing fault in an owned source export."""
import argparse
from pathlib import Path
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('manifest', type=Path)
p.add_argument('logs', type=Path)
p.add_argument('--only', choices=['clipped', 'normal-lf', 'palette-lf', 'clipped-cli', 'gray'])
p.add_argument('--instrumented', action='store_true')
a = p.parse_args()
root = a.manifest.resolve().parent
if (root / '.git').exists() or not (root.parent.parent / 'provenance.json').is_file():
    p.error('requires an owned prepare_clean_source.py export')
a.logs.mkdir(exist_ok=False)
source = root / 'jxl-encoder/src/modular/frame.rs'
mutations = [
    ('clipped',
     'image.extract_region(x_start, y_start, x_start + group_dim, y_start + group_dim)',
     'image.extract_region(x_start, y_start, (x_start + group_dim).min(self.width), (y_start + group_dim).min(self.height))',
     '--lib', 'shifted_group_regions_match_unmodified_v012_rects'),
    ('normal-lf',
     'let (source_image_owned, lf_images) = self.split_lf_groups(source_image_owned)?;',
     'let (source_image_owned, lf_images) = (source_image_owned, Vec::new());',
     '--test', 'shifted_extra_edges', 'strict_lf_producers_preserve_the_native_grid'),
    ('palette-lf',
     'let (spatial_image, lf_images) = self.split_lf_groups(spatial_image)?;',
     'let (spatial_image, lf_images) = (spatial_image, Vec::new());',
     '--test', 'shifted_extra_edges', 'strict_lf_producers_preserve_the_native_grid'),
]
mutations.append(('clipped-cli', mutations[0][1], mutations[0][2],
    '--test', 'shifted_alpha', 'downsampled_alpha_survives_odd_groups_in_every_cli_mode'))
mutations.extend([
    ('gray-frame',
     'let has_rct = !self.options.skip_rct\n            && !image.is_grayscale\n',
     'let has_rct = !self.options.skip_rct\n',
     '--test', 'shifted_extra_edges', 'grayscale_extras_do_not_become_color_transform_channels'),
    ('gray-squeeze-fixed',
     'let has_rct = !transformed.is_grayscale && transformed.channels.len() >= 3;',
     'let has_rct = transformed.channels.len() >= 3;',
     '--test', 'shifted_extra_edges', 'grayscale_extras_do_not_become_color_transform_channels'),
    ('gray-squeeze-tree',
     '// Step 1: Apply RCT (YCoCg) before squeeze for RGB images\n    let mut transformed = image.clone();\n    let has_rct = !transformed.is_grayscale && transformed.channels.len() >= 3;',
     '// Step 1: Apply RCT (YCoCg) before squeeze for RGB images\n    let mut transformed = image.clone();\n    let has_rct = transformed.channels.len() >= 3;',
     '--test', 'shifted_extra_edges', 'grayscale_extras_do_not_become_color_transform_channels'),
])
for name, needle, replacement, *selection in mutations:
    if a.only is not None and name != a.only and not (a.only == 'gray' and name.startswith('gray-')):
        continue
    package = 'jxl-encoder-cli' if name.endswith('-cli') else 'jxl-encoder'
    features = 'jxl_encoder/__expert' if name.endswith('-cli') else '__expert'
    source = root / ('jxl-encoder/src/modular/encode.rs' if name.startswith('gray-squeeze') else 'jxl-encoder/src/modular/frame.rs')
    original = source.read_text()
    assert original.count(needle) == (2 if name == 'gray-squeeze-fixed' else 1), name
    try:
        source.write_text(original.replace(needle, replacement, 1))
        log = a.logs / (name + '.log')
        with log.open('x') as sink:
            n_selector = 1 if selection[0] == '--lib' else 2
            command = ['cargo', 'llvm-cov' if a.instrumented else 'test', '--locked',
                '--manifest-path', str(a.manifest), '-p', package, '--features', features,
                *selection[:n_selector]]
            if a.instrumented:
                command.extend(['--no-report', '--'])
            command.extend(selection[n_selector:])
            result = subprocess.run(command, stdout=sink, stderr=subprocess.STDOUT)
        assert result.returncode == 101 and 'test result: FAILED' in log.read_text(), (name, result.returncode)
        print(name + ': regression detected', flush=True)
    finally:
        source.write_text(original)
