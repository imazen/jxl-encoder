#!/usr/bin/env python3
"""Require runtime regressions for count-admission faults in an owned source export."""
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
api = 'jxl-encoder/src/api.rs'
header = 'jxl-encoder/src/headers/file_header.rs'
info = 'jxl-encoder/src/headers/extra_channels.rs'
encoder = 'jxl-encoder/src/vardct/encoder.rs'
faults = [
    ('supported-count-cap', info, 'if count > crate::container::MAX_EXTRA_CHANNELS_LEVEL10 as usize {',
     'if false && count > crate::container::MAX_EXTRA_CHANNELS_LEVEL10 as usize {',
     'extra_metadata_count_admission_accounts', None),
    ('wire-count-guard', header, 'if meta.extra_channels.len() > 4096 {',
     'if false && meta.extra_channels.len() > 4096 {',
     'extra_channel_count_wire_bound_precedes_header_writes', None),
    ('early-supplied-count', api, 'validate::validate_extra_channel_count(self.extra_channels.len())?;',
     '', 'extra_metadata_count_canonicalized_admission_precedes_budget', None),
    ('implicit-alpha-count', api, '.saturating_add(usize::from(self.layout.has_alpha()))',
     '', 'extra_metadata_count_admission_accounts', None),
    ('implicit-black-count', api, '.saturating_add(usize::from(self.layout.is_cmyk()))',
     '', 'extra_metadata_count_admission_accounts', None),
    ('canonicalized-count-order', api, 'if !may_remove_alpha {',
     'if true || !may_remove_alpha {',
     'extra_metadata_count_canonicalization_can_remove_layout_alpha', None),
]
anchor = 'crate::headers::extra_channels::ExtraChannelInfo::validate_supported_count(extras.len())?;'
faults.append((
    'prepared-layout-count', api,
    '        // Canonicalization can remove layout alpha. Count implicit channels\n'
    '        // on the prepared layout, while retaining every caller-supplied extra.\n'
    '        validate::validate_extra_channel_count(\n'
    '            self.extra_channels\n'
    '                .len()\n'
    '                .saturating_add(usize::from(self.layout.has_alpha()))\n'
    '                .saturating_add(usize::from(self.layout.is_cmyk())),\n'
    '        )?;\n',
    '', 'extra_metadata_count_canonicalization_retains_nonopaque_alpha_in_count', None,
))
faults += [(name, encoder, anchor, '', 'extra_metadata_prepared_entries_validate_before_working_pixels', index)
           for index, name in enumerate(['direct-count-admission', 'prepared-count-admission', 'prequantized-count-admission'])]
if a.case:
    assert set(a.case) <= {row[0] for row in faults}
    faults = [row for row in faults if row[0] in a.case]
original = {root / relative: (root / relative).read_bytes() for _, relative, *_ in faults}
digest = lambda data: hashlib.sha256(data).hexdigest()
rows = []
try:
    for label, relative, before, after, selection, occurrence in faults:
        path = root / relative
        fixed = original[path]
        source = fixed.decode()
        expected = 3 if occurrence is not None else 2 if label in ['implicit-alpha-count', 'implicit-black-count'] else 1
        assert source.count(before) == expected, (label, source.count(before))
        if occurrence is None:
            fault = source.replace(before, after).encode()
        else:
            offset = 0
            for _ in range(occurrence + 1):
                position = source.index(before, offset)
                offset = position + len(before)
            fault = (source[:position] + after + source[offset:]).encode()
        out = a.output / label
        out.mkdir()
        (out / 'fixed.rs').write_bytes(fixed)
        (out / 'fault.rs').write_bytes(fault)
        path.write_bytes(fault)
        command = [sys.executable, str(Path(__file__).with_name('run_selected_test_evidence.py')),
                   str(a.manifest), str(out / 'runtime'), selection,
                   '--features', '__expert,__internals,__pre_quantized,parallel,rate-control',
                   '--source', relative]
        if label != 'wire-count-guard':
            command += ['--test-target', 'extra_metadata_contract']
        with (out / 'runner.log').open('x') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        identity_path = out / 'runtime/identity.json'
        assert identity_path.is_file(), (label, 'no runtime identity')
        identity = json.loads(identity_path.read_text())
        assert result.returncode == 101 and identity['exit_code'] == 101, (label, result.returncode)
        assert 'test result: FAILED' in (out / 'runtime/test.log').read_text()
        path.write_bytes(fixed)
        assert path.read_bytes() == fixed
        rows.append(dict(fault=label, source=relative, filter=selection, occurrence=occurrence,
                         exit_code=101, fixed_sha256=digest(fixed), fault_sha256=digest(fault),
                         binary_sha256=identity['binary_sha256'], profile_sha256=identity['profile_sha256']))
        (a.output / 'results.partial.json').write_text(json.dumps(rows, indent=2) + '\n')
        print(label + ': actual runtime failure; exact source restored', flush=True)
finally:
    for path, fixed in original.items():
        path.write_bytes(fixed)
        assert path.read_bytes() == fixed
    (a.output / 'restored.json').write_text(json.dumps(dict(restored=True, files=len(original))) + '\n')
(a.output / 'results.json').write_text(json.dumps(rows, indent=2) + '\n')
