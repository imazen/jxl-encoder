#!/usr/bin/env python3
"""Read selected ZIP members by HTTP range; preserve their hashes and ZIP CRCs."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import urllib.error
import urllib.request
import zipfile

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('repository')
p.add_argument('artifact', type=int)
p.add_argument('output', type=Path)
p.add_argument('--member', action='append', required=True, help='regular expression on ZIP member paths')
a = p.parse_args()
a.output.mkdir(exist_ok=False)

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None

endpoint = f'https://api.github.com/repos/{a.repository}/actions/artifacts/{a.artifact}/zip'
token = subprocess.check_output(['gh', 'auth', 'token'], text=True).strip()
request = urllib.request.Request(endpoint, headers={'Authorization': 'Bearer ' + token, 'Accept': 'application/vnd.github+json'})
try:
    urllib.request.build_opener(NoRedirect).open(request)
    raise RuntimeError('artifact endpoint did not redirect')
except urllib.error.HTTPError as error:
    if error.code != 302:
        raise
    signed = error.headers['Location']

class Remote(io.RawIOBase):
    def __init__(self, url):
        self.url = url
        response = urllib.request.urlopen(urllib.request.Request(url, headers={'Range': 'bytes=0-0'}))
        assert response.status == 206
        self.size = int(response.headers['Content-Range'].split('/')[-1])
        assert len(response.read()) == 1
        self.position = 0
        self.fetched = 1

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset + (0 if whence == 0 else self.position if whence == 1 else self.size)
        assert 0 <= self.position <= self.size
        return self.position

    def read(self, length=-1):
        length = self.size - self.position if length < 0 else min(length, self.size - self.position)
        if not length:
            return b''
        end = self.position + length - 1
        response = urllib.request.urlopen(urllib.request.Request(self.url, headers={'Range': f'bytes={self.position}-{end}'}))
        assert response.status == 206
        assert response.headers['Content-Range'].startswith(f'bytes {self.position}-{end}/')
        data = response.read()
        assert len(data) == length
        self.position += length
        self.fetched += length
        return data

remote = Remote(signed)
rows = []
with zipfile.ZipFile(remote) as archive:
    listing = [dict(path=item.filename, size=item.file_size, compressed=item.compress_size, crc=item.CRC) for item in archive.infolist()]
    (a.output / 'members.json').write_text(json.dumps(listing, indent=2) + '\n')
    for item in archive.infolist():
        if not any(re.search(pattern, item.filename) for pattern in a.member) or item.is_dir():
            continue
        relative = Path(item.filename)
        assert not relative.is_absolute() and '..' not in relative.parts
        dest = a.output / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        count = 0
        with archive.open(item) as source, dest.open('xb') as out:
            while chunk := source.read(1024 * 1024):
                out.write(chunk)
                digest.update(chunk)
                count += len(chunk)
        assert count == item.file_size
        rows.append(dict(path=item.filename, size=count, crc=item.CRC, sha256=digest.hexdigest()))
        (a.output / 'extracted.partial.json').write_text(json.dumps(rows, indent=2) + '\n')
assert rows, 'no selected members'
(a.output / 'identity.json').write_text(json.dumps(dict(repository=a.repository, artifact=a.artifact, patterns=a.member, archive_bytes=remote.size, fetched_bytes=remote.fetched, files=rows), indent=2) + '\n')
print(json.dumps(dict(files=len(rows), fetched_bytes=remote.fetched, archive_bytes=remote.size)))
