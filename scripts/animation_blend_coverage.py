#!/usr/bin/env python3
"""Verify passing blend archives and report exact affected function/window regions."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('llvm',type=Path);p.add_argument('output',type=Path)
p.add_argument('archives',nargs='+',type=Path)
p.add_argument('--reservation', action='append', type=Path, default=[])
a=p.parse_args();a.output.mkdir(exist_ok=False)
scripts=Path(__file__).resolve().parent
reports=[];pairs=[];identities=[]
archives=a.archives+a.reservation
for index,archive in enumerate(archives):
    identity=json.loads((archive/'identity.json').read_text());assert identity['exit_code']==0,archive
    for row in identity['sources']:
        assert hashlib.sha256((archive/'source'/row['path']).read_bytes()).hexdigest()==row['sha256'],(archive,row['path'])
    target=a.output/f'input-{index}'
    subprocess.run([sys.executable,str(scripts/'coverage_from_archive.py'),str(a.llvm),str(archive),str(target)],check=True)
    report=target/'selected.json';reports.append(json.loads(report.read_text()));pairs+=['--pair',str(archive),str(report)]
    identities.append(dict(archive=str(archive),exit_code=0,binary_sha256=identity['binary_sha256'],profile_sha256=identity['profile_sha256']))
subprocess.run([sys.executable,str(scripts/'alpha_squeeze_coverage_summary.py'),str(a.output/'reference-plan.json'),'--scope','blend-plan',*pairs],check=True)
rows=[]
def observe(relative,function,first,last,indices=None):
    indices=list(range(len(archives))) if indices is None else indices
    spans=[];bodies=[];declarations=[]
    for index in indices:
        archive=archives[index]
        lines=(archive/'source'/relative).read_text().splitlines()
        starts=[i for i,line in enumerate(lines)if re.search(r'\bfn '+re.escape(function)+r'(?:<[^>]*>)?\(',line)]
        assert len(starts)==1,(archive,function,starts)
        start=starts[0];indent=lines[start][:len(lines[start])-len(lines[start].lstrip())]
        end=next(i for i in range(start+1,len(lines))if lines[i]==indent+'}')
        bodies.append('\n'.join(lines[start:end+1]))
        declarations.append(start+1)
        low=next(i for i in range(start,end+1)if first in lines[i])
        high=next(i for i in range(low,end+1)if last in lines[i])
        spans.append((low+1,high+1))
    assert all(b==bodies[0]for b in bodies),('function bodies differ',function)
    assert all((lo-decl,hi-decl)==(spans[0][0]-declarations[0],spans[0][1]-declarations[0]) for (lo,hi),decl in zip(spans,declarations)),('window relative offsets differ',function)
    counts={}
    for index,(low,high),declaration in zip(indices,spans,declarations):
        for unit in reports[index]['data']:
            for mapped in unit['functions']:
                if str(len(function))+function not in mapped['name']:
                    continue
                for region in mapped['regions']:
                    if region[7]!=0 or not mapped['filenames'][region[5]].endswith(relative) or region[0]<low or region[2]>high:
                        continue
                    key=(region[0]-declaration,region[1],region[2]-declaration,region[3])
                    counts[key]=max(counts.get(key,0),region[4])
    assert counts,('no mapped regions',function)
    rows.append(dict(source=relative,function=function,body_sha256=hashlib.sha256(bodies[0].encode()).hexdigest(),
                     input_archives=[str(archives[i]) for i in indices],source_spans=spans,declarations=declarations,
                     covered=sum(v>0 for v in counts.values()),total=len(counts),
                     uncovered_relative_regions=[key for key,value in sorted(counts.items()) if not value],
                     source_comparison='Exact function bodies; relative declaration offsets'))
    print(json.dumps(rows[-1]))
for function in ['encode_animation_lossless','encode_animation_lossy']:
    observe('jxl-encoder/src/api/animate.rs',function,'        let crop = if frame.reference_only || frame.blend_mode.is_some() {','        } else if let Some(prev) = prev_pixels {')
    observe('jxl-encoder/src/api/animate.rs',function,'reference_only_after_ct: reference_plan[i] & REFERENCE_AFTER_COLOR_TRANSFORM != 0,','reference_only_after_ct: reference_plan[i] & REFERENCE_AFTER_COLOR_TRANSFORM != 0,')
observe('jxl-encoder/src/api/animate.rs','encode_animation_lossy','preserve_color_reference_three: reference_plan[i] & RESERVE_COLOR_SLOT_THREE != 0,','preserve_color_reference_three: reference_plan[i] & RESERVE_COLOR_SLOT_THREE != 0,')
observe('jxl-encoder/src/modular/frame.rs','apply_animation_to_header','        if self.options.reference_only {','            fh.save_before_ct = !self.options.reference_only_after_ct;')
observe('jxl-encoder/src/vardct/bitstream.rs','encode_two_pass_to_writer','                if opts.reference_only {','                    fh.save_before_ct = !opts.reference_only_after_ct;')
if a.reservation:
    indices=list(range(len(a.archives),len(archives)))
    observe('jxl-encoder/src/vardct/bitstream.rs','encode_frame_to_writer','        let mut patches_data =','            super::patches::find_and_build_with_per_patch_gate(',indices)
    subprocess.run([sys.executable,str(scripts/'alpha_squeeze_coverage_summary.py'),str(a.output/'reservation.json'),'--scope','blend-reservation',*[item for i in indices for item in ['--pair',str(archives[i]),str(a.output/f'input-{i}'/'selected.json')]]],check=True)
(a.output/'windows.json').write_text(json.dumps(dict(inputs=identities,windows=rows),indent=2))
