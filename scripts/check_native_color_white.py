#!/usr/bin/env python3
"""Render gamma endpoint white controls emitted by the pinned native oracle."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('log',type=Path);p.add_argument('djxl',type=Path);p.add_argument('output',type=Path)
a=p.parse_args()
version=subprocess.run([str(a.djxl),'--version'],capture_output=True,text=True,check=True)
assert 'v0.12.0 a7a9c78' in version.stdout+version.stderr
a.output.mkdir(exist_ok=False)
rows=[]
for line in a.log.read_text().splitlines():
    if not line.startswith('production\tgamma-') or '-white-' not in line: continue
    _,label,added,finished,hex_data=line.split('\t')
    assert added==finished=='1',label
    out=a.output/label;out.mkdir()
    data=bytes.fromhex(hex_data);(out/'native.jxl').write_bytes(data)
    result=subprocess.run([str(a.djxl),str(out/'native.jxl'),str(out/'native.pfm'),
        '--num_threads=1','--color_space=RGB_D65_SRG_Rel_Lin'],capture_output=True)
    (out/'decode.log').write_bytes(result.stdout+result.stderr);assert result.returncode==0,label
    # PFM endianness follows its signed scale, as in check_animation_alpha_native.py.
    with (out/'native.pfm').open('rb') as stream:
        assert stream.readline().strip()==b'PF'
        assert stream.readline().strip()==b'16 17'
        scale=float(stream.readline());raw=stream.read()
    assert len(raw)==16*17*3*4
    values=struct.unpack(('<' if scale<0 else '>')+str(len(raw)//4)+'f',raw)
    assert all(math.isfinite(v) for v in values),label
    rows.append(dict(label=label,stream_sha256=hashlib.sha256(data).hexdigest(),
        first_rgb=values[:3],maximum_white_error=max(abs(v-1) for v in values)))
assert len(rows)==4
(a.output/'observed.json').write_text(json.dumps(dict(
    native_log_sha256=hashlib.sha256(a.log.read_bytes()).hexdigest(),observations=rows),indent=2))
for row in rows:print(json.dumps(row))
