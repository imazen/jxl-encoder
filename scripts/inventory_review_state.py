#!/usr/bin/env python3
"""Save live branch, issue, PR and TODO evidence outside the checkout."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('output', type=Path, help='Private output directory outside this repository')
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
out = args.output.expanduser().resolve()
if out == root or root in out.parents:
    raise SystemExit('Inventory evidence must be stored outside the checkout')
out.mkdir(parents=True, exist_ok=True)
os.environ['TMPDIR'] = str(Path.home() / 'tmp')

def run(*argv, check=True):
    return subprocess.run(argv, cwd=root, text=True, capture_output=True, check=check)

def json_command(*argv):
    return json.loads(run(*argv).stdout)

origin = run('git', 'remote', 'get-url', 'origin').stdout.strip()
match = re.search(r'github.com[:/]([^/]+/[^/]+?)(?:\.git)?$', origin)
if not match:
    raise SystemExit('Expected a GitHub origin')
repo = match[1]
pages = json_command('gh', 'api', '--paginate', '--slurp', f'repos/{repo}/branches?per_page=100')
branches = [branch for page in pages for branch in page]
issues = json_command('gh', 'issue', 'list', '-R', repo, '--state', 'open', '--limit', '1000', '--json', 'number,title,url,state,labels,body,updatedAt')
prs = json_command('gh', 'pr', 'list', '-R', repo, '--state', 'all', '--limit', '1000', '--json', 'number,title,url,headRefName,headRefOid,state,mergedAt,isDraft,updatedAt,baseRefName')
main = next(b['commit']['sha'] for b in branches if b['name'] == 'main')

def graph(sha):
    if run('git', 'cat-file', '-e', sha + '^{commit}', check=False).returncode:
        # The live endpoint can advance after the local fetch. Report that
        # explicitly; never classify an unavailable object as merged.
        return {'object_available': False}
    counts = run('git', 'rev-list', '--left-right', '--count', main + '...' + sha).stdout.split()
    date, title = run('git', 'show', '-s', '--format=%cI%n%s', sha).stdout.splitlines()[:2]
    return dict(object_available=True, behind=int(counts[0]), ahead=int(counts[1]),
                tip_in_main=run('git', 'merge-base', '--is-ancestor', sha, main, check=False).returncode == 0,
                commit_date=date, title=title)

remote = []
for b in branches:
    name, sha = b['name'], b['commit']['sha']
    remote.append(dict(name=name, sha=sha, protected=b['protected'], **graph(sha),
                       prs=[p for p in prs if p['headRefName'] == name]))
local = []
for line in run('git', 'for-each-ref', '--format=%(refname)%09%(objectname)', 'refs/heads', 'refs/remotes').stdout.splitlines():
    name, sha = line.split('\t')
    local.append(dict(name=name, sha=sha, **graph(sha)))
markers, guidance, unreadable = [], [], []
for name in run('git', 'ls-files', '-z').stdout.split('\0'):
    if not name or name.endswith('inventory_review_state.py'):
        continue
    path = root / name
    if path.suffix not in {'.rs', '.md', '.sh', '.py', '.toml', '.yml', '.yaml'} and path.name != 'justfile':
        continue
    try:
        lines = path.read_text().splitlines()
    except (OSError, UnicodeError) as error:
        unreadable.append({'path': name, 'error': str(error)})
        continue
    for number, text in enumerate(lines, 1):
        kind = 'historical' if any(part in name for part in ['archive/', 'CODE-HISTORY', 'CHANGELOG']) else 'current'
        row = dict(path=name, line=number, text=text, kind=kind)
        if re.search(r'\b(?:TODO|FIXME)\b', text):
            markers.append(row)
        if path.suffix == '.md' and re.search(r'from_token|token\(\)|token-based|token-taking|summon.*once|concrete.*variant|Requires a token|Never call.*arcane', text):
            guidance.append(row)
report = dict(observed_at=datetime.now(timezone.utc).isoformat(), repo=repo, main=main,
              remote_branches=remote, raw_remote_branches=branches, local_refs=local, open_issues=issues, prs=prs,
              todo_fixme=markers, token_guidance=guidance, unreadable=unreadable)
(out / 'inventory.json').write_text(json.dumps(report, indent=2) + '\n')
links = {}
for label, url in re.findall(r'\[([^\]]+)\]\((https?://[^)]+)\)', json.dumps(report)):
    links[(label, url)] = {'text': label, 'href': url}
for url in re.findall(r'https?://[^\s\"<>]+', json.dumps(report)):
    links.setdefault(('', url), {'text': '', 'href': url})
(out / 'links.jsonl').write_text(''.join(json.dumps(link) + '\n' for link in links.values()))
(out / 'jj-bookmarks.txt').write_text(run('jj', '--ignore-working-copy', 'bookmark', 'list', '--all-remotes', '--color', 'never').stdout)
summary = [f'# {repo} private inventory', f'Observed: {report["observed_at"]}', f'Main: {main}',
           f'Remote branches: {len(remote)}; local refs: {len(local)}; open issues: {len(issues)}; TODO/FIXME matches: {len(markers)}.',
           '\n## Live remote branches\n', '| Branch | SHA | Behind | Ahead | Tip in main | PRs |', '|---|---|---:|---:|---|---|']
for b in remote:
    links = ', '.join(f'[{p["number"]} {p["state"]}]({p["url"]})' for p in b['prs'])
    summary.append(f'| {b["name"]} | {b["sha"]} | {b.get("behind", "unavailable")} | {b.get("ahead", "unavailable")} | {b.get("tip_in_main", "unavailable")} | {links} |')
summary += ['\n## Local refs\n'] + [f'- {b["name"]}: {b["sha"]}; ahead={b.get("ahead", "unavailable")}, behind={b.get("behind", "unavailable")}, tip_in_main={b.get("tip_in_main", "unavailable")}' for b in local]
summary += ['\n## Open issues\n'] + [f'- [{i["number"]}: {i["title"]}]({i["url"]})' for i in issues]
summary += ['\n## TODO/FIXME\n'] + [f'- {m["path"]}:{m["line"]} [{m["kind"]}] {m["text"]}' for m in markers]
summary += ['\n## Token guidance matches\n'] + [f'- {m["path"]}:{m["line"]} {m["text"]}' for m in guidance]
(out / 'inventory.md').write_text('\n'.join(summary) + '\n')
print(f'{repo}: {len(remote)} live branches, {len(local)} local refs, {len(issues)} issues, {len(markers)} TODO/FIXME matches; evidence {out}')
