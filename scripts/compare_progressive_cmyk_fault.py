#!/usr/bin/env python3
"""Compare both complete retained decoder outputs after a restored V1 fault."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
from lz77_upstream_compare import read_pfm

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('run', type=Path)
p.add_argument('output', type=Path)
a = p.parse_args()
assert not a.output.exists()
base = a.run / 'libjxl-32x17-solidfalse-widefalse-Single-still'
changed = a.run / 'libjxl-32x17-solidfalse-widefalse-QuantizedAcFullAc-still'
rows = []
for decoder in ['primary', 'native']:
    for channel in ['color', 'black']:
        name = f'primary-0-{channel}.f32le' if decoder == 'primary' else 'native-0' + ('-black' if channel == 'black' else '') + '.pfm'
        paths = [base/name, changed/name]
        buffers = [path.read_bytes() if decoder == 'primary' else read_pfm(path, channels=3 if channel == "color" else 1)[2] for path in paths]
        assert len(buffers[0]) == len(buffers[1]) == 32*17*(3 if channel=='color' else 1)*4
        words = [struct.unpack('<' + str(len(data)//4) + 'I', data) for data in buffers]
        differences = [i for i,(before,after) in enumerate(zip(*words)) if before != after]
        assert bool(differences) == (channel == 'color'), (decoder,channel)
        first = differences[0] if differences else None
        rows.append(dict(decoder=decoder, channel=channel, samples=len(words[0]), changed_samples=len(differences),
                         first_sample=first, first_bits=[v[first] for v in words] if first is not None else None,
                         files=[dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in paths]))
a.output.write_text(json.dumps(dict(run=str(a.run), observations=rows, note='Observed complete decoder output, compared to each decoder own Single stream; not modeled reconstruction.'),indent=2)+'\n')
print(json.dumps(rows))
