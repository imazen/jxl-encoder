#!/usr/bin/env python3
"""Export the CI-pinned source closure without changing any sibling checkout."""
import argparse
import csv
import io
import json
import os
import re
from pathlib import Path
import subprocess
import tarfile
import urllib.error
import urllib.request

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('destination', type=Path)
parser.add_argument('revision', help='Committed encoder revision to export')
parser.add_argument('--pins-file', type=Path, help='Use an exported revision table and immutable GitHub archives')
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
dest = args.destination.expanduser().resolve()
if dest.exists():
    raise SystemExit(f'Refusing to replace existing stage: {dest}')
if root == dest or root in dest.parents:
    raise SystemExit('Stage must be outside the checkout')
os.environ['TMPDIR'] = str(Path.home() / 'tmp')
if args.pins_file:
    revision = args.revision
    if not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise SystemExit('Archive-only export requires a full commit SHA')
    pins = args.pins_file.read_text()
else:
    revision = subprocess.check_output(['git', '-C', str(root), 'rev-parse', args.revision + '^{commit}'], text=True).strip()
    pins = subprocess.check_output(['git', '-C', str(root), 'show', revision + ':.github/sibling-revisions.tsv'], text=True)
dest.mkdir(parents=True)
receipt = []

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *unused, **kwargs):
        return None


def stage_filter(member, destination):
    # CI preserves repository-relative license links, some of which reach the
    # enclosing source closure. Permit only links inside this newly owned stage.
    if member.issym():
        link = (Path(destination) / member.name).parent / member.linkname
        resolved = link.resolve()
        if resolved != dest and dest not in resolved.parents:
            raise tarfile.FilterError(f'Link escapes source closure: {member.name}')
        return tarfile.tar_filter(member, destination)
    return tarfile.data_filter(member, destination)


def export(repository, sha, target, checkout=None):
    target.mkdir(parents=True)
    if checkout and subprocess.run(['git', '-C', str(checkout), 'cat-file', '-e', sha + '^{commit}'], capture_output=True).returncode == 0:
        with subprocess.Popen(['git', '-C', str(checkout), 'archive', sha], stdout=subprocess.PIPE) as process:
            with tarfile.open(fileobj=process.stdout, mode='r|') as archive:
                archive.extractall(target, filter=stage_filter)
            if process.wait():
                raise RuntimeError(f'git archive failed for {repository}')
        method = 'local committed objects'
    else:
        token = subprocess.check_output(['gh', 'auth', 'token'], text=True).strip()
        request = urllib.request.Request(f'https://api.github.com/repos/{repository}/tarball/{sha}', headers={'Authorization': 'Bearer ' + token})
        try:
            response = urllib.request.build_opener(NoRedirect).open(request)
        except urllib.error.HTTPError as error:
            if error.code != 302:
                raise
            response = urllib.request.urlopen(error.headers['Location'])
        with response, tarfile.open(fileobj=response, mode='r|gz') as archive:
            for member in archive:
                parts = Path(member.name).parts[1:]
                if not parts:
                    continue
                member.name = str(Path(*parts))
                # Repository tarballs have a top-level directory; symlink
                # targets are repository-relative and need no rewriting.
                archive.extract(member, target, filter=stage_filter)
        method = 'immutable GitHub source archive'
    receipt.append(dict(repository=repository, revision=sha, path=str(target), method=method))
    (dest / 'SOURCE.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(repository, sha, method, flush=True)

export('imazen/jxl-encoder', revision, dest / 'zen/jxl-encoder', None if args.pins_file else root)
for row in csv.DictReader(io.StringIO(pins), delimiter='\t'):
    repo, sha = row['repository'], row['revision']
    name = repo.split('/')[-1]
    choices = [Path.home() / 'work/zen' / name, Path.home() / 'work' / name]
    local = None if args.pins_file else next((p for p in choices if (p / '.git').exists()), None)
    location = dest / name if name in {'butteraugli', 'dssim'} else dest / 'zen' / name
    export(repo, sha, location, local)
print('CI workspace:', dest / 'zen/jxl-encoder')
