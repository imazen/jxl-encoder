#!/usr/bin/env python3
"""Run the production LZ77 harness serially; summarize paired encode timers.

Input TSV: name, input, size, depths, efforts, mode, threads. Every input must
exist and fit the crop. No upscaling or missing-input skips. All raw timings,
commands, source hashes and encoded artifacts remain in the output directory.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import struct
import subprocess


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('binary', type=Path)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    assert args.repeats >= 3
    args.output.mkdir(parents=True, exist_ok=False)
    cells = list(csv.DictReader(args.manifest.open(), delimiter='\t'))
    assert cells
    provenance = {'binary_sha256': sha(args.binary), 'manifest_sha256': sha(args.manifest),
                  'repeats': args.repeats, 'timing_scope': 'encode only; paired alternating order; rep 0 warmup',
                  'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  'git_diff_sha256': hashlib.sha256(subprocess.check_output(['git', 'diff'])).hexdigest(),
                  'host': subprocess.check_output(['uname', '-a'], text=True).strip(), 'commands': []}
    with (args.output / 'summary.tsv').open('w') as out:
        writer = csv.writer(out, delimiter='\t', lineterminator='\n')
        writer.writerow(['name', 'size', 'depth', 'effort', 'mode', 'threads', 'off_ms', 'on_ms',
                         'paired_ratio', 'off_bytes', 'on_bytes', 'off_sha256', 'on_sha256', 'source_sha256'])
        for idx, cell in enumerate(cells):
            source = Path(cell['input']).expanduser().resolve(strict=True)
            header = source.read_bytes()[:24]
            assert header[:8] == b'\x89PNG\r\n\x1a\n'
            width, height = struct.unpack('>II', header[16:24])
            assert min(width, height) >= int(cell['size']), cell
            inputs = args.output / f'input-{idx}'
            inputs.mkdir()
            (inputs / source.name).symlink_to(source)
            table = args.output / f'{idx}.tsv'
            command = [str(args.binary.resolve()), str(inputs), str(table), '--images', '1',
                       '--size', cell['size'], '--depths', cell['depths'], '--efforts', cell['efforts'],
                       '--mode', cell['mode'], '--threads', cell['threads'], '--lossy', '0',
                       '--keep-best-repeats', str(args.repeats), '--artifacts', str(args.output / 'artifacts')]
            provenance['commands'].append(command)
            (args.output / 'meta.json').write_text(json.dumps(provenance, indent=2) + '\n')
            print(f"START {idx + 1}/{len(cells)} {cell}", flush=True)
            with (args.output / f'{idx}.log').open('w') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            grouped = {}
            for row in csv.DictReader(table.open(), delimiter='\t'):
                arm, rep = row['arm'].removeprefix('keepbest_').split('_r')
                key = row['depth'], row['effort']
                target = grouped.setdefault(key, {})
                assert (arm, int(rep)) not in target
                target[arm, int(rep)] = row
                artifact = Path(row['artifact'])
                assert artifact.stat().st_size == int(row['bytes']) and sha(artifact) == row['encoded_sha256']
            assert set(grouped) == {(d, e) for d in cell['depths'].split(',') for e in cell['efforts'].split(',')}
            for (depth, effort), group in grouped.items():
                assert set(group) == {(a, r) for a in ['off', 'on'] for r in range(args.repeats + 1)}
                off = [float(group['off', r]['ms']) for r in range(1, args.repeats + 1)]
                on = [float(group['on', r]['ms']) for r in range(1, args.repeats + 1)]
                for arm in ['off', 'on']:
                    assert len({group[arm, r]['encoded_sha256'] for r in range(args.repeats + 1)}) == 1
                left, right = group['off', 0], group['on', 0]
                assert int(right['bytes']) <= int(left['bytes'])
                ratio = statistics.median(b / a for a, b in zip(off, on))
                writer.writerow([cell['name'], cell['size'], depth, effort, cell['mode'], cell['threads'],
                                 f'{statistics.median(off):.3f}', f'{statistics.median(on):.3f}', f'{ratio:.5f}',
                                 left['bytes'], right['bytes'], left['encoded_sha256'], right['encoded_sha256'], sha(source)])
                out.flush()
                print(f"RESULT {cell['name']} {cell['size']} {depth} e{effort}: {ratio:.3f}x, {left['bytes']} -> {right['bytes']} bytes", flush=True)


if __name__ == '__main__':
    main()
