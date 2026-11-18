#!/usr/bin/env python3
"""Retain source and actual executable identity for the caller-profile probe."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('manifest', type=Path)
p.add_argument('profile', type=Path)
p.add_argument('output', type=Path)
p.add_argument('--source-control', type=Path)
p.add_argument('--physical-control', type=Path)
p.add_argument('--explicit-replace', action='store_true')
a = p.parse_args()
root = a.manifest.resolve().parent
assert not (root/'.git').exists() and (root.parent.parent/'provenance.json').is_file()
a.output.mkdir(exist_ok=False)
rows = []
for relative in ['Cargo.lock', 'jxl-encoder/Cargo.toml', 'jxl-encoder/src/api/cmyk.rs',
                 'jxl-encoder/src/api/validate.rs', 'jxl-encoder/src/headers/file_header.rs',
                 'jxl-encoder/examples/progressive_ac_cmyk.rs', 'jxl-encoder/src/api.rs',
                 'jxl-encoder/src/api/animate.rs', 'jxl-encoder/src/api/ingest.rs',
                 'jxl-encoder/src/vardct/xyb.rs', 'jxl-encoder/tests/images/frymire.png',
                 'jxl-encoder/tests/support/decoded_extras.rs',
                 'jxl-encoder/tests/support/encoder_policies.rs', 'jxl-encoder/tests/support/pfm.rs']:
    source = root/relative
    target = a.output/'source'/relative
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,target)
    rows.append(dict(path=relative,sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
command = ['cargo','build','--locked','--manifest-path',str(a.manifest),'-p','jxl-encoder',
           '--features','__expert,parallel','--example','progressive_ac_cmyk','--message-format=json']
with (a.output/'build.jsonl').open('x') as out, (a.output/'build.log').open('x') as err:
    subprocess.run(command,stdout=out,stderr=err,check=True)
artifacts=[json.loads(line) for line in (a.output/'build.jsonl').read_text().splitlines()]
executables=[r['executable'] for r in artifacts if r.get('reason')=='compiler-artifact'
             and r.get('target',{}).get('name')=='progressive_ac_cmyk' and r.get('executable')]
executable,=executables
retained=a.output/'example-binary'
shutil.copy2(executable,retained)
digest=hashlib.sha256(retained.read_bytes()).hexdigest()
with (a.output/'probe.log').open('x') as log:
    assert not (a.source_control and a.physical_control)
    arguments = ['--physical-control', str(a.physical_control)] if a.physical_control else (['--source-control', str(a.source_control)] if a.source_control else (['--animation-replace'] if a.explicit_replace else ['--animation']))
    child=subprocess.Popen([executable,str(a.profile),str(a.output/'run'),*arguments],stdout=log,stderr=subprocess.STDOUT)
    identity=dict(command=command, executable=executable,binary_sha256=digest,pid=child.pid,sources=rows,
                  profile_sha256=hashlib.sha256(a.profile.read_bytes()).hexdigest())
    if a.source_control:
        references = sorted(a.source_control.glob('*.jxl'))
        assert len(references) == 12, references
        identity['reference_streams'] = [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                                         for path in references]
    if a.physical_control:
        identity['native_lcms_control'] = dict(path=str(a.physical_control),sha256=hashlib.sha256(a.physical_control.read_bytes()).hexdigest())
    (a.output/'identity.json').write_text(json.dumps(identity,indent=2))
    code=child.wait()
assert hashlib.sha256(Path(executable).read_bytes()).hexdigest()==digest
assert (a.output/'run/binary.sha256').read_text()==digest
identity['exit_code']=code
(a.output/'identity.json').write_text(json.dumps(identity,indent=2))
print('CMYK '+('physical source' if a.physical_control else ('source' if a.source_control else 'animation'))+' probe: rc='+str(code),flush=True)
sys.exit(code)
