#!/usr/bin/env python3
"""Constrain this owned transient resource scope to one verified CPU class."""
import argparse
import json
import os
from pathlib import Path
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('output', type=Path)
p.add_argument('command', nargs=argparse.REMAINDER)
a = p.parse_args()
command = a.command[1:] if a.command[:1] == ['--'] else a.command
assert command
a.output.mkdir(exist_ok=False)
group, = [line.removeprefix('0::') for line in Path('/proc/self/cgroup').read_text().splitlines()
          if line.startswith('0::')]
unit = Path(group).name
assert unit.startswith('run-') and unit.endswith('.scope'), 'requires this job\'s transient run-heavy scope'
properties = subprocess.run(['systemctl', '--user', 'show', unit, '-p', 'Transient', '-p', 'MemoryMax',
                             '-p', 'MemorySwapMax'], check=True, stdout=subprocess.PIPE, text=True).stdout
values = dict(line.split('=', 1) for line in properties.splitlines())
assert values['Transient'] == 'yes' and values['MemoryMax'].isdecimal()
assert 0 < int(values['MemoryMax']) <= 16 * 1024**3 and values['MemorySwapMax'] == '0'
before = subprocess.run(['zenbench', 'cpu-info'], check=True, stdout=subprocess.PIPE).stdout
(a.output / 'before.json').write_bytes(before)
classes = json.loads(before)['topology']['classes']
rank = max(row['performance_rank'] for row in classes)
selected, = [row for row in classes if row['performance_rank'] == rank]
cpus = sorted(set(selected['cpus']))
assert cpus and all(isinstance(cpu, int) and cpu >= 0 for cpu in cpus)
# systemd accepts a space-separated CPU list and applies an actual cgroup
# cpuset, preserving this same scope's existing memory/swap limits.
cpu_list = ' '.join(map(str, cpus))
subprocess.run(['systemctl', '--user', 'set-property', '--runtime', unit,
                'AllowedCPUs=' + cpu_list], check=True)
after = subprocess.run(['zenbench', 'cpu-info'], check=True, stdout=subprocess.PIPE).stdout
(a.output / 'after.json').write_bytes(after)
assert 'Ok' in json.loads(after)['placement'], 'benchmark placement guard must verify the constrained scope'
(a.output / 'launch.json').write_text(json.dumps(dict(scope=unit, memory_properties=values,
    selected_class=selected, command=command), indent=2) + '\n')
os.execvp(command[0], command)
