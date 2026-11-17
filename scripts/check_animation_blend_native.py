#!/usr/bin/env python3
"""Check native full-canvas Add and zero-duration background layer samples."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('log',type=Path);p.add_argument('djxl',type=Path);p.add_argument('output',type=Path)
a=p.parse_args()
version=subprocess.run([str(a.djxl),'--version'],capture_output=True,text=True,check=True)
assert 'v0.12.0 a7a9c78' in version.stdout+version.stderr
a.output.mkdir(exist_ok=False)
rows=[line.split('\t') for line in a.log.read_text().splitlines() if line.startswith(('lossless-','lossy-'))]
assert {r[0] for r in rows}=={'lossless-canvas','lossless-hidden','lossy-canvas','lossy-hidden'}
records=[]
for label,hex_data in rows:
    out=a.output/label;out.mkdir();data=bytes.fromhex(hex_data);stream=out/'native.jxl';stream.write_bytes(data)
    result=subprocess.run([str(a.djxl),str(stream),str(out/'frame.pam'),'--output_frames','--bits_per_sample=16','--num_threads=1'],capture_output=True)
    (out/'decode.log').write_bytes(result.stdout+result.stderr)
    assert result.returncode==0,label
    frames=sorted(out.glob('*.pam'));assert len(frames)==2,(label,frames)
    raw=frames[-1].read_bytes();end=raw.index(b'ENDHDR\n')+7
    header=raw[:end].decode();assert 'WIDTH 16\n' in header and 'HEIGHT 16\n' in header
    channels=4 if 'DEPTH 4\n' in header else 3;assert f'DEPTH {channels}\n' in header
    maximum=int(next(line.split()[1] for line in header.splitlines() if line.startswith('MAXVAL ')))
    assert maximum in (255,65535)
    values=tuple(raw[end:]) if maximum==255 else struct.unpack('>'+str(256*channels)+'H',raw[end:])
    assert len(values)==256*channels
    expected=(96 if label.endswith('hidden') else 128)*(maximum//255)
    error=max(abs(v-expected) for i,v in enumerate(values) if i%channels<3)
    assert error <= (0 if label.startswith('lossless') else max(1,round(maximum*0.002))),(label,error)
    records.append(dict(label=label,stream_sha256=hashlib.sha256(data).hexdigest(),maximum_sample_code=maximum,max_code_error=error))
(a.output/'identity.json').write_text(json.dumps(dict(decoder=version.stdout+version.stderr,controls=records,source_log_sha256=hashlib.sha256(a.log.read_bytes()).hexdigest()),indent=2))
print('Native full-canvas Add and hidden regular-layer controls recover selected sums')
