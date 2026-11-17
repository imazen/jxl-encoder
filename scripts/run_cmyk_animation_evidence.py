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
a = p.parse_args()
root = a.manifest.resolve().parent
assert not (root/'.git').exists() and (root.parent.parent/'provenance.json').is_file()
a.output.mkdir(exist_ok=False)
rows = []
for relative in ['jxl-encoder/examples/progressive_ac_cmyk.rs', 'jxl-encoder/src/api.rs',
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
    child=subprocess.Popen([executable,str(a.profile),str(a.output/'run'),'--animation'],stdout=log,stderr=subprocess.STDOUT)
    identity=dict(command=command, executable=executable,binary_sha256=digest,pid=child.pid,sources=rows,
                  profile_sha256=hashlib.sha256(a.profile.read_bytes()).hexdigest())
    (a.output/'identity.json').write_text(json.dumps(identity,indent=2))
    code=child.wait()
assert hashlib.sha256(Path(executable).read_bytes()).hexdigest()==digest
assert (a.output/'run/binary.sha256').read_text()==digest
identity['exit_code']=code
(a.output/'identity.json').write_text(json.dumps(identity,indent=2))
print('CMYK animation probe: rc='+str(code),flush=True)
sys.exit(code)
