#!/usr/bin/env python3
"""Freeze an owned replay scope during one serialized local command; always thaw."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('host')
p.add_argument('pid', type=int)
p.add_argument('scope')
p.add_argument('binary_sha256')
p.add_argument('filter')
p.add_argument('output', type=Path)
p.add_argument('--queue-output', type=Path, help='recover an interrupted qualifier recorded by this queue')
p.add_argument('command', nargs=argparse.REMAINDER)
a = p.parse_args()
command = a.command[1:] if a.command[:1] == ['--'] else a.command
assert command and a.scope.startswith('run-') and a.scope.endswith('.scope')
a.output.mkdir(exist_ok=False)
remote = r'''
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
pid,scope,digest,selection,action=sys.argv[1:]
proc=Path('/proc')/pid
assert proc.stat().st_uid==os.getuid()
group,=[l.removeprefix('0::') for l in (proc/'cgroup').read_text().splitlines() if l.startswith('0::')]
assert Path(group).name==scope
args=(proc/'cmdline').read_bytes().split(b'\0')
assert selection.encode()in args
binary,=[Path(os.fsdecode(v))for v in args if v.endswith(b'/test-binary')]
assert hashlib.sha256(binary.read_bytes()).hexdigest()==digest
props=subprocess.check_output(['systemctl','--user','show',scope,'-p','Transient','-p','MemoryMax','-p','MemorySwapMax'],text=True)
assert 'Transient=yes' in props and 'MemorySwapMax=0' in props
subprocess.run(['systemctl','--user',action,scope],check=True)
events=Path('/sys/fs/cgroup')/group.lstrip('/')/'cgroup.events'
expected='1'if action=='freeze'else'0'
for _ in range(100):
 values=dict(l.split()for l in events.read_text().splitlines())
 if values.get('frozen')==expected:break
 time.sleep(0.05)
else:raise RuntimeError('scope did not reach requested freezer state')
print(json.dumps(dict(pid=int(pid),scope=scope,binary_sha256=digest,action=action,events=values,properties=props)))
'''
rows = []
def transition(action):
    args = ['python3', '-c', remote, str(a.pid), a.scope, a.binary_sha256, a.filter, action]
    result = subprocess.run(['ssh', a.host, shlex.join(args)], check=True, capture_output=True, text=True)
    row = json.loads(result.stdout)
    row['utc'] = datetime.now(timezone.utc).isoformat()
    rows.append(row)
    (a.output / 'transitions.json').write_text(json.dumps(rows, indent=2) + '\n')
    print(action + ' verified for owned replay scope', flush=True)
code = 1
try:
    transition('freeze')
    (a.output / 'local-command.json').write_text(json.dumps(command, indent=2) + '\n')
    code = subprocess.run(command).returncode
finally:
    try:
        if a.queue_output and (a.queue_output / 'active.json').is_file():
            active = json.loads((a.queue_output / 'active.json').read_text())
            args = active['command']
            if len(args) > 5 and Path(args[1]).name == 'qualify_selected_modular_source.py':
                archive = Path(args[5])
                if ((archive / 'overlay.json').is_file()
                        and not (archive / 'restored.json').is_file()
                        and not (archive / 'recovery-restored.json').is_file()):
                    subprocess.run([sys.executable,
                        str(Path(__file__).with_name('recover_selected_modular_overlay.py')),
                        args[2], str(archive)], check=True)
    finally:
        transition('thaw')
(a.output / 'verdict.json').write_text(json.dumps(dict(exit_code=code,
    note='Remote replay elapsed wall includes this intermission; no latency claim.'), indent=2) + '\n')
sys.exit(code)
