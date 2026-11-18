#!/usr/bin/env python3
"""Restore representative gain-color faults in an owned closure; require runtime regressions."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('manifest', type=Path)
p.add_argument('output', type=Path)
p.add_argument('--case', action='append', default=[])
a = p.parse_args()
root = a.manifest.resolve().parent
assert not (root / '.git').exists() and (root.parent.parent / 'provenance.json').is_file()
a.output.mkdir(exist_ok=False)
request = 'jxl-encoder/src/hdr/from_sdr.rs'
color = 'jxl-encoder/src/hdr/from_sdr_color.rs'
faults = [
    ('base-gain-gamut-flag', request, 'metadata.use_base_color_space = false;', 'metadata.use_base_color_space = true;', 'strict_gain_map_reconstructs'),
    ('missing-alternate-tag', request, '.with_color_encoding(hdr_ce)', '', 'strict_gain_map_reconstructs'),
    ('missing-base-conversion', color, 'if color.primaries == target.primaries {', 'if true {', 'strict_gain_map_reconstructs'),
    ('integer-pq-srgb-fallback', color,
     'TransferFunction::Pq => ultrahdr_core::color::transfer::pq_eotf(*value),',
     'TransferFunction::Pq if layout != HdrPixelLayout::RgbaF32 => ultrahdr_core::color::transfer::srgb_eotf(*value),\n                    TransferFunction::Pq => ultrahdr_core::color::transfer::pq_eotf(*value),', 'nonlinear_hdr_layouts'),
    ('alpha-disagreement', request, 'color::matching_alpha_row(sdr, hdr)?;', 'let _ = (sdr, hdr);', 'gain_map_input_contract'),
    ('source-override-disagreement', color, 'if supplied.color_space != expected.color_space', 'if false && supplied.color_space != expected.color_space', 'gain_map_input_contract'),
    ('hlg-channel-ootf', color, 'TransferFunction::Hlg => crate::api::hlg_to_linear_f(*value),',
     'TransferFunction::Hlg => ultrahdr_core::color::transfer::hlg_eotf(*value, 1000.0) / 1000.0,', 'gain_map_hlg_color'),
    ('pq-base-white-units', color, 'TransferFunction::Pq => 10_000.0 / target.white_nits,', 'TransferFunction::Pq => 1.0,', 'gain_map_pq_absolute_light'),
    ('hlg-base-white-units', color, 'TransferFunction::Hlg => 1_000.0 / target.white_nits,', 'TransferFunction::Hlg => 1.0,', 'gain_map_ordered_transfer_pairs'),
    ('headroom-reference-white', request, 'f64::from(self.hdr_intensity_target) / f64::from(base_white)', 'f64::from(self.hdr_intensity_target) / 203.0', 'gain_map_headroom_override'),
    ('float-srgb-base', request, 'let mut prepared_sdr = if self.sdr_layout == HdrPixelLayout::RgbaF32',
     'let mut prepared_sdr = if false && self.sdr_layout == HdrPixelLayout::RgbaF32', 'gain_map_source_transfer'),
    ('float-pq-base', request, 'UhdrTransfer::Pq => PixelLayout::RgbaPqF32,',
     'UhdrTransfer::Pq => PixelLayout::RgbaLinearF32,', 'gain_map_source_transfer'),
    ('float-hlg-base', request, 'UhdrTransfer::Hlg => PixelLayout::RgbaHlgF32,',
     'UhdrTransfer::Hlg => PixelLayout::RgbaLinearF32,', 'gain_map_source_transfer'),
]
# Parenthesize the whole comparison to disable every source-description check.
faults[5] = ('source-override-disagreement', color,
    'if supplied.color_space != expected.color_space\n',
    'if false && (supplied.color_space != expected.color_space\n', 'gain_map_input_contract')
if a.case:
    assert set(a.case) <= {row[0] for row in faults}
    faults = [row for row in faults if row[0] in a.case]
original = {root / relative: (root / relative).read_bytes() for _, relative, *_ in faults}
sha = lambda b: hashlib.sha256(b).hexdigest()
rows = []
try:
    for label, relative, before, after, selection in faults:
        path = root / relative
        fixed = original[path]
        text = fixed.decode()
        assert text.count(before) == 1, (label, 'source anchor changed')
        fault = text.replace(before, after, 1)
        if label == 'hlg-channel-ootf':
            anchor='if let Some((luminances, gamma)) = hlg {'
            assert fault.count(anchor)==1
            fault=fault.replace(anchor,'if let Some((luminances, gamma)) = hlg.filter(|_| false) {',1)
        if label == 'source-override-disagreement':
            anchor = '|| supplied.custom_primaries != expected.custom_primaries\n'
            assert fault.count(anchor) == 1
            fault = fault.replace(anchor, '|| supplied.custom_primaries != expected.custom_primaries)\n', 1)
        fault = fault.encode()
        out = a.output / label
        out.mkdir()
        (out / 'fixed.rs').write_bytes(fixed)
        (out / 'fault.rs').write_bytes(fault)
        path.write_bytes(fault)
        command = [sys.executable, str(Path(__file__).with_name('run_selected_test_evidence.py')),
                   str(a.manifest), str(out / 'runtime'), selection,
                   '--features', '__expert,parallel,hdr-gainmap', '--test-target', 'gain_map_color_contract',
                   '--source', request, '--source', color]
        with (out / 'runner.log').open('x') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        identity = json.loads((out / 'runtime/identity.json').read_text())
        assert result.returncode == 101 and identity['exit_code'] == 101, (label, result.returncode)
        assert 'test result: FAILED' in (out / 'runtime/test.log').read_text()
        path.write_bytes(fixed)
        assert path.read_bytes() == fixed
        rows.append(dict(fault=label, source=relative, filter=selection, exit_code=101,
                         binary_sha256=identity['binary_sha256'], fixed_sha256=sha(fixed), fault_sha256=sha(fault)))
        (a.output / 'results.partial.json').write_text(json.dumps(rows, indent=2) + '\n')
        print(label + ': runtime failure detected; exact source restored', flush=True)
finally:
    for path, fixed in original.items():
        path.write_bytes(fixed)
        assert path.read_bytes() == fixed
    (a.output / 'restored.json').write_text(json.dumps(dict(restored=True, files=len(original))) + '\n')
(a.output / 'results.json').write_text(json.dumps(rows, indent=2) + '\n')
