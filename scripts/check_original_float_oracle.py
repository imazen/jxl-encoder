#!/usr/bin/env python3
"""Check float source-field output against exact nonuniform native lossless inputs."""
import argparse
import json
from pathlib import Path
import struct
import subprocess
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('cjxl',type=Path);p.add_argument('oracle',type=Path);p.add_argument('output',type=Path)
a=p.parse_args();a.output.mkdir(exist_ok=False)
v=subprocess.run([str(a.cjxl),'--version'],capture_output=True,text=True,check=True)
assert 'v0.12.0 a7a9c78' in v.stdout+v.stderr
observed=[]
for channels in [1,3]:
    values=bytes((i*37+17)%256 for i in range(2*3*channels))
    source=a.output/f'input-{channels}.pnm';source.write_bytes((b'P5\n' if channels==1 else b'P6\n')+b'2 3\n255\n'+values)
    encoded=source.with_suffix('.jxl')
    result=subprocess.run([str(a.cjxl),str(source),str(encoded),'-d','0','-e','3','--num_threads=1'],capture_output=True)
    source.with_suffix('.encode.log').write_bytes(result.stdout+result.stderr);assert result.returncode==0
    prefix=a.output/f'decoded-{channels}'
    result=subprocess.run([str(a.oracle),str(encoded),str(prefix),'--source-fields-float'],capture_output=True)
    source.with_suffix('.decode.log').write_bytes(result.stdout+result.stderr);assert result.returncode==0
    with Path(str(prefix)+'-0.pfm').open('rb') as stream:
        assert stream.readline().strip()==(b'Pf' if channels==1 else b'PF')
        assert stream.readline().strip()==b'2 3';scale=float(stream.readline());raw=stream.read()
        pixels=struct.unpack(('<'if scale<0 else'>')+str(len(raw)//4)+'f',raw)
    assert len(pixels)==len(values)
    top_down=[v for y in range(2,-1,-1) for v in pixels[y*2*channels:(y+1)*2*channels]]
    error=max(abs(a-b/255)for a,b in zip(top_down,values))
    assert error<0.000001,(channels,error,top_down,values)
    observed.append(dict(channels=channels,max_error=error,source_codes=list(values),observed=top_down))
(a.output/'observed.json').write_text(json.dumps(observed,indent=2))
print('Native gray/RGB source-profile float output preserves every selected lossless code')
