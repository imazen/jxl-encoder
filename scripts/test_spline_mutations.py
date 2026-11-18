#!/usr/bin/env python3
"""Demonstrate spline admission regressions at runtime, restoring exact owned source."""
import argparse
import hashlib
import json
from pathlib import Path
import re
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
sha = lambda b: hashlib.sha256(b).hexdigest()

def span(text, name):
    lines = text.splitlines(keepends=True)
    starts = [i for i, line in enumerate(lines) if re.search(r'\bfn ' + name + r'\(', line)]
    assert len(starts) == 1, (name, starts)
    start = starts[0]
    indent = lines[start][:len(lines[start]) - len(lines[start].lstrip())]
    end = next(i for i in range(start + 1, len(lines)) if lines[i].rstrip() == indent + '}')
    return lines, start, end

def replace_body(text, name, body):
    lines, start, end = span(text, name)
    assert lines[start].rstrip().endswith('{'), name
    return ''.join(lines[:start + 1]) + body + ''.join(lines[end:])

def remove_guard(text, name):
    lines, start, end = span(text, name)
    low = next(i for i in range(start, end) if 'if let Some(splines) = &self.splines {' in lines[i])
    high = next(i for i in range(low + 1, end) if lines[i].rstrip() == '        }')
    return ''.join(lines[:low] + lines[high + 1:])

spline = 'jxl-encoder/src/vardct/splines.rs'
encoder = 'jxl-encoder/src/vardct/encoder.rs'
faults = [
    ('geometry', spline, lambda s: replace_body(s, 'validate_splines', '    let _ = splines;\n    Ok(())\n'), 'strict_manual_spline_admission', True),
    ('quantization-cast', spline, lambda s: replace_body(s, 'quantized_coefficient', '    Ok(value.round() as i32)\n'), 'numeric_boundaries_and_empty_section_are_checked', False),
    ('streaming-admission', 'jxl-encoder/src/api.rs', lambda s: s.replace('            validate_lossy_manual_splines(&self.cfg)?;\n', '', 1), 'strict_spline_geometry_refused_on_first_push', True),
    ('animation-admission', 'jxl-encoder/src/api/animate.rs', lambda s: s.replace('if cfg\n        .splines', 'if false && cfg\n        .splines', 1), 'strict_animation_manual_splines_refuse_unsupported_payload', True),
    ('precomputed-refusal', encoder, lambda s: remove_guard(s, 'encode_from_precomputed_with_extras'), 'strict_prepared_manual_splines', True),
    ('prequantized-refusal', encoder, lambda s: remove_guard(s, 'encode_from_pre_quantized_ac_with_extras'), 'strict_prepared_manual_splines', True),
    ('subsampling-refusal', 'jxl-encoder/src/api.rs', lambda s: s.replace('if cfg\n            .splines', 'if false && cfg\n            .splines', 1), 'subsampled_manual_spline', True),
    ('frame-writer-refusal', 'jxl-encoder/src/vardct/bitstream.rs', lambda s: remove_guard(s, 'encode_frame_to_writer'), 'frame_writer_refuses_unimplemented_splines_before_writing', False),
    ('section-shape', spline, lambda s: s.replace('if data.splines.is_empty() || data.splines.len() != data.quantized.len() {', 'if false {', 1), 'malformed_spline_sections_preserve_the_existing_writer', False),
]
if a.case:
    assert set(a.case) <= {f[0] for f in faults}
    faults = [f for f in faults if f[0] in a.case]
original = {root / relative: (root / relative).read_bytes() for _, relative, *_ in faults}
rows = []
try:
    for label, relative, mutate, selection, integration in faults:
        path = root / relative
        fixed = original[path]
        assert path.read_bytes() == fixed
        fault = mutate(fixed.decode()).encode()
        assert fault != fixed, (label, 'source shape changed')
        out = a.output / label
        out.mkdir()
        (out / 'fixed.rs').write_bytes(fixed)
        (out / 'fault.rs').write_bytes(fault)
        path.write_bytes(fault)
        command = [sys.executable, str(Path(__file__).with_name('run_selected_test_evidence.py')), str(a.manifest), str(out / 'runtime'), selection, '--features', '__expert,__internals,__pre_quantized,parallel,chroma-subsampling,jpeg-reencoding', '--source', relative]
        if integration:
            command += ['--test-target', 'spline_admission_contract']
        else:
            command += ['--source', 'jxl-encoder/src/vardct/spline_admission_tests.rs']
        with (out / 'runner.log').open('x') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        identity = json.loads((out / 'runtime/identity.json').read_text())
        assert result.returncode == 101 and identity['exit_code'] == 101, (label, result.returncode, identity)
        assert 'test result: FAILED' in (out / 'runtime/test.log').read_text()
        path.write_bytes(fixed)
        assert path.read_bytes() == fixed
        row = dict(fault=label, source=relative, filter=selection, exit_code=101, binary_sha256=identity['binary_sha256'], fixed_sha256=sha(fixed), fault_sha256=sha(fault))
        rows.append(row)
        (a.output / 'results.partial.json').write_text(json.dumps(rows, indent=2) + '\n')
        print(label + ': runtime failure detected; exact source restored', flush=True)
finally:
    for path, fixed in original.items():
        path.write_bytes(fixed)
        assert path.read_bytes() == fixed
    (a.output / 'restored.json').write_text(json.dumps(dict(restored=True, files=len(original))) + '\n')
(a.output / 'results.json').write_text(json.dumps(rows, indent=2) + '\n')
