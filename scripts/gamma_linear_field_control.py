#!/usr/bin/env python3
"""Change only gamma or the modular-buffer-width bit for decoder attribution.

This is a diagnostic reinterpretation of the same coefficients, not an encoder
output or a claim that its displayed pixels must match the original profile.
"""
import argparse
import hashlib
import json
from pathlib import Path
from jxl_bitstream_diff import Parser, BitReader
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('input',type=Path);p.add_argument('output',type=Path)
p.add_argument('--field', choices=['gamma','modular-width'],default='gamma')
a=p.parse_args()
data=a.input.read_bytes();assert data[:2]==b'\xff\x0a','requires bare codestream'
def fields(data):
    parser=Parser(data);parser.br=BitReader(data,2)
    parser.size_header();parser.image_metadata()
    return parser.fields
if a.field=='gamma':
    suffix='.gamma';bits=24;expected=1221;replacement=10000000;filename='gamma-one.jxl'
    note='Same payload interpreted with linear transfer; not an original encoder output'
else:
    suffix='.modular_16bit_buffers';bits=1;expected=1;replacement=0;filename='modular32.jxl'
    note='Same payload with wider modular buffers; diagnostic raw stream, not an emitted level declaration'
matches=[(end,name,value)for end,name,value in fields(data)if name.endswith(suffix)]
assert len(matches)==1,matches
end,name,old=matches[0];assert old==expected,(name,old)
start=end-bits;modified=bytearray(data)
for bit in range(bits):
    pos=start+bit;mask=1<<(pos%8)
    modified[pos//8]=(modified[pos//8]&~mask)|(((replacement>>bit)&1)<<(pos%8))
assert len(modified)==len(data)
for bit in range(len(data)*8):
    if not start<=bit<end:
        assert ((data[bit//8]^modified[bit//8])>>(bit%8))&1==0
original_fields=fields(data);new_fields=fields(modified)
assert [(pos,key,value if key!=name else replacement)for pos,key,value in original_fields]==new_fields
a.output.mkdir(exist_ok=False)
(a.output/filename).write_bytes(modified)
(a.output/'field-control.json').write_text(json.dumps(dict(input=str(a.input),
    input_sha256=hashlib.sha256(data).hexdigest(),output_sha256=hashlib.sha256(modified).hexdigest(),
    field=name,start_bit=start,end_bit=end,before=old,after=replacement,
    note=note),indent=2))
