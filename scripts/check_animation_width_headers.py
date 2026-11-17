#!/usr/bin/env python3
"""Check animation buffer promises and container levels with the independent parser."""
import argparse
import hashlib
import json
from pathlib import Path
from jxl_bitstream_diff import Parser, BitReader
from jxl_container import codestream
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('manifest',type=Path);p.add_argument('output',type=Path)
a=p.parse_args();assert not a.output.exists()
observed=[]
for row in json.loads(a.manifest.read_text()):
    data=Path(row['path']).read_bytes();stream,_=codestream(data)
    parser=Parser(stream);parser.br=BitReader(stream,2);parser.size_header();parser.image_metadata()
    fields={name:value for _,name,value in parser.fields}
    narrow=bool(fields.get('metadata.modular_16bit_buffers',True))
    if row.get('narrow') is not None: assert narrow==row['narrow'],(row,fields)
    level=5;boxes=[]
    if not data.startswith(b'\xff\x0a'):
        pos=0;levels=[]
        while pos<len(data):
            size=int.from_bytes(data[pos:pos+4],'big');tag=data[pos+4:pos+8];header=8
            if size==1: size=int.from_bytes(data[pos+8:pos+16],'big');header=16
            if size==0:size=len(data)-pos
            assert header<=size<=len(data)-pos
            boxes.append(tag.decode('ascii'))
            if tag==b'jxll':
                assert size-header==1
                levels.append(data[pos+header])
            pos+=size
        assert len(levels)<=1
        level=levels[0]if levels else 5
        if level==10:assert boxes[2]=='jxll',boxes
    # These fixtures have small dimensions/profiles and no Black channel.
    assert level==(5 if narrow else 10),(row,narrow,level)
    observed.append(dict(**row,sha256=hashlib.sha256(data).hexdigest(),narrow_observed=narrow,level=level,boxes=boxes))
assert observed
assert any(r['narrow_observed']for r in observed)
assert any(not r['narrow_observed']for r in observed)
a.output.write_text(json.dumps(observed,indent=2))
print('Actual animation metadata and level declarations agree in every selected control')
