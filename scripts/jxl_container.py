"""Extract a codestream and its byte ranges without changing payload bytes."""

def codestream(data):
    if data.startswith(b"\xff\x0a"):
        return data, [(0, len(data))]
    assert data.startswith(b"\x00\x00\x00\x0cJXL \x0d\x0a\x87\x0a"), "unknown container"
    chunks, kind, fragment = [], None, 0
    last = False
    pos = 0
    while pos < len(data):
        assert pos + 8 <= len(data), "truncated box"
        size = int.from_bytes(data[pos:pos + 4], "big")
        tag = data[pos + 4:pos + 8]
        header = 8
        if size == 1:
            assert pos + 16 <= len(data), "truncated large box"
            size = int.from_bytes(data[pos + 8:pos + 16], "big")
            header = 16
        if size == 0:
            size = len(data) - pos
        assert header <= size <= len(data) - pos, "invalid box size"
        start = pos + header
        if tag in [b"jxlc", b"jxlp"]:
            assert kind in [None, tag], "mixed codestream boxes"
            kind = tag
            if tag == b"jxlc":
                assert not chunks, "multiple jxlc boxes"
            else:
                assert not last and start + 4 <= pos + size, "invalid fragment"
                counter = int.from_bytes(data[start:start + 4], "big")
                assert counter & 0x7fffffff == fragment, "unordered fragments"
                last = bool(counter & 0x80000000)
                fragment += 1
                start += 4
            chunks.append((start, pos + size - start))
        pos += size
    assert chunks and (kind == b"jxlc" or last), "incomplete codestream"
    return b"".join(data[start:start + size] for start, size in chunks), chunks
