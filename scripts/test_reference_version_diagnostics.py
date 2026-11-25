#!/usr/bin/env python3
"""Exercise only the reference action's version loop with explicit shell fixtures."""
import argparse
import os
from pathlib import Path
import subprocess
import textwrap

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('output', type=Path)
a = p.parse_args()
a.output.mkdir(exist_ok=False)
action = Path(__file__).resolve().parents[1] / '.github/actions/setup-libjxl/action.yml'
text = action.read_text()
start = text.index('    - name: Verify versions and export reference paths\n')
start = text.index('      run: |\n', start) + len('      run: |\n')
script = textwrap.dedent(text[start:])
# The remaining action covers oracle and temporary-directory setup separately.
script = script[:script.index('for binding in jxl-original-profile:')]
for name, version, status in [('success', 'cjxl v0.13.0 fixture', 0),
                              ('native-error', 'fixture native error detail', 23),
                              ('wrong-version', 'cjxl v0.12.0 fixture', 0),
                              ('version-prefix', 'cjxl v0.13.00 fixture', 0)]:
    root = a.output / name
    tools = root / '.ci-libjxl/tools'
    tools.mkdir(parents=True)
    for tool in ['cjxl', 'djxl']:
        banner = version.replace('cjxl', tool)
        path = tools / tool
        path.write_text('#!/bin/sh\nprintf "%s\\n" "' + banner + '" >&2\nexit ' + str(status) + '\n')
        path.chmod(0o755)
    envfile = root.resolve() / 'exports.env'
    result = subprocess.run(['bash', '-c', script], cwd=root,
                            env=dict(os.environ, RUNNER_OS='Linux', RUNNER_ARCH='X64',
                                     GITHUB_ENV=str(envfile)),
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (root / 'verification.log').write_text(result.stdout)
    assert 'Verifying cjxl at ' in result.stdout, name
    if status:
        assert result.returncode == status, (name, result.returncode)
        assert version in result.stdout and 'exit status 23' in result.stdout, name
        assert not envfile.exists(), name
    elif name in ('wrong-version', 'version-prefix'):
        assert result.returncode == 1 and 'Incorrect libjxl version' in result.stdout, name
        assert not envfile.exists(), name
    else:
        assert result.returncode == 0, result.stdout
        exports = envfile.read_text()
        assert 'CJXL_PATH=' in exports and 'DJXL_PATH=' in exports, exports
        assert 'Verifying djxl at ' in result.stdout, name
    print(name + ': diagnostic contract passed')
