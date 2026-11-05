#!/usr/bin/env python3
"""Record live process identity and completed journal rows for a retained run."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    journal = root / 'full/results.partial.tsv'
    counts = Counter()
    completed = 0
    last_index = None
    if journal.exists():
        with journal.open() as stream:
            for line in stream:
                if not line.endswith('\n') or line.startswith(('#', 'index\t')):
                    continue
                fields = line.rstrip('\n').split('\t')
                assert len(fields) == 6, f'malformed journal row: {fields!r}'
                assert fields[1] in {'OK', 'REJECTED', 'FAIL', 'DECODER'}, fields
                last_index = int(fields[0])
                counts[fields[1]] += 1
                completed += 1
    pid_file = root / 'sweep.pid'
    pid = int(pid_file.read_text()) if pid_file.exists() else None
    alive = False
    if pid is not None:
        try:
            os.kill(pid, 0)
            alive = (Path('/proc') / str(pid) / 'exe').samefile(root / 'conformance-sweep')
        except (ProcessLookupError, FileNotFoundError):
            pass
    lock_rows = subprocess.run(['lslocks', '-n', '-o', 'PID,PATH'], check=True,
                               capture_output=True, text=True).stdout.splitlines()
    lock_path = str(Path.home() / 'tmp/heavy.lock')
    holders = [int(row.split()[0]) for row in lock_rows if row.split()[-1:] == [lock_path]]
    progress = {
        'checked_utc': datetime.now(timezone.utc).isoformat(),
        'stage': (root / 'stage').read_text().strip(),
        'source': json.loads((root / 'source.json').read_text()),
        'pid': pid, 'verified_live_binary': alive, 'lock_holders': holders,
        'completed_cells': completed, 'counts': dict(counts),
        'last_completed_index': last_index,
        'exit': int((root / 'sweep.exit').read_text()) if (root / 'sweep.exit').exists() else None,
    }
    (root / 'progress.json').write_text(json.dumps(progress, indent=2) + '\n')
    print(json.dumps(progress))


if __name__ == '__main__':
    main()
