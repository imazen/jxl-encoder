#!/usr/bin/env python3
"""Observe every animation envelope using the existing independent field parser."""
import argparse
import hashlib
import json
from pathlib import Path
from jxl_bitstream_diff import Parser, BitReader
from jxl_container import codestream

class Envelopes(Parser):
    # Only the envelope and TOC lengths are needed to reach the next header.
    def lf_global_prefix(self, *args):
        pass
    def hf_global_prefix(self):
        pass

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('manifest',type=Path)
p.add_argument('output',type=Path)
a=p.parse_args()
assert not a.output.exists()
observed=[]
for row in json.loads(a.manifest.read_text()):
    container=Path(row['path']).read_bytes();data,ranges=codestream(container);assert data[:2]==b'\xff\x0a'
    parser=Envelopes(data);parser.br=BitReader(data,2)
    width,height=parser.size_header();parser.image_metadata();parser.custom_transform_data()
    assert not getattr(parser,'want_icc',False),'This non-ICC fixture requires a complete envelope parse'
    parser.br.jump_to_byte_boundary();frames=[]
    while parser.br.pos<len(data)*8:
        parser.fields=[];parser.toc=None
        start=parser.br.pos//8;parser.frame_header(width,height);parser.read_toc(width,height)
        assert parser.toc is not None,'permuted TOC outside this fixture'
        next_byte=parser.toc['payload_start']+sum(parser.toc['entries'])
        assert start<next_byte<=len(data),(start,next_byte,len(data))
        fields={key:value for _,key,value in parser.fields}
        frames.append(dict(start=start,end=next_byte,fields=fields))
        parser.br.pos=next_byte*8
        if fields.get('frame.all_default') or fields.get('frame.is_last'):
            break
    assert parser.br.pos==len(data)*8,'Every byte must belong to the parsed frame sequence'
    assert len(frames)==row.get('frames',3 if row['reference'] else 2),(row,len(frames))
    last=frames[-1]['fields'];assert bool(last.get('frame.custom_size_or_origin',False))==row.get('partial',False),(row,last)
    if row['reference']:
        hidden=frames[row.get('hidden_index',1)]['fields']
        assert hidden['frame.frame_type']=='kReferenceOnly',(row,hidden)
        assert hidden['frame.save_as_reference']==row.get('slot',2)
        assert bool(hidden['frame.save_before_color_transform'])==(row['blend']=='Replace' and not row.get('partial',False)),(row,hidden)
    if row.get('dictionary',False):
        dictionary=frames[row.get('dictionary_index',2)]['fields']
        assert dictionary['frame.frame_type']=='kReferenceOnly'
        assert dictionary['frame.save_as_reference']==3
        assert dictionary['frame.save_before_color_transform']==1
    observed.append(dict(path=row['path'],sha256=hashlib.sha256(container).hexdigest(),codestream_sha256=hashlib.sha256(data).hexdigest(),codestream_ranges=ranges,frames=frames))
a.output.write_text(json.dumps(observed,indent=2))
print('Every selected frame envelope and complete TOC extent verified')
