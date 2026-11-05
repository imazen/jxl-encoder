#!/usr/bin/env python3
"""Attribute a retained PQ discrepancy using a separate linear-tagged diagnostic copy.

Only the six-bit PQ transfer enum changes. The original stream, encoder,
decoder pins and agreement gate are untouched. This is not a repaired image.
"""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from compare_float_decodes import digest, read_pixels
from jxl_bitstream_diff import BitReader, Parser


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


def linear_control(data):
    raw, chunks = codestream(data)
    before = Parser(raw)
    before.parse()
    fields = [f for f in before.fields if f[1] == "metadata.color_encoding.transfer_function"]
    assert len(fields) == 1 and fields[0][2] == "kPQ", "requires explicit PQ header"
    end = fields[0][0]
    reader = BitReader(raw)
    reader.pos = end - 6
    assert reader.enum() == 16 and reader.pos == end, "requires six-bit PQ enum"
    copied = bytearray(data)
    # Both PQ=16 and Linear=8 use selector 2 and four payload bits (value-2).
    for bit in range(6):
        offset = end - 6 + bit
        byte_offset = offset // 8
        for start, size in chunks:
            if byte_offset < size:
                absolute = start + byte_offset
                break
            byte_offset -= size
        else:
            raise AssertionError("header outside codestream")
        copied[absolute] = ((copied[absolute] & ~(1 << (offset % 8)))
                            | (((2 | ((8 - 2) << 2)) >> bit & 1) << (offset % 8)))
    assert sum((a ^ b).bit_count() for a, b in zip(data, copied)) == 1, "unexpected mutation"
    after = Parser(codestream(bytes(copied))[0])
    after.parse()
    assert len(before.fields) == len(after.fields)
    changed = [(a, b) for a, b in zip(before.fields, after.fields) if a != b]
    assert changed == [(fields[0], (end, fields[0][1], "kLinear"))], changed
    return bytes(copied), end - 6


def to_pq(linear):
    # Signed double-precision inverse EOTF, intensity_target=10000 nits.
    value = linear.astype(np.float64)
    xp = np.abs(value) ** (2610.0 / 16384.0)
    signal = ((3424.0 / 4096.0 + xp * (2413.0 / 128.0)) /
              (1.0 + xp * (2392.0 / 128.0))) ** (2523.0 / 32.0)
    return np.where(value == 0, 0, np.copysign(signal, value)).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--djxl", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--static-pq", type=Path, required=True)
    parser.add_argument("--native-pq", type=Path, required=True)
    parser.add_argument("--xyc", type=int, nargs=3, required=True)
    args = parser.parse_args()
    data = args.input.read_bytes()
    header = Parser(codestream(data)[0])
    header.parse()
    peaks = [f for f in header.fields if f[1].endswith("intensity_target")]
    assert len(peaks) == 1 and peaks[0][2] == 10000.0, peaks
    control, offset = linear_control(data)
    args.out.mkdir(parents=True, exist_ok=False)
    copied = args.out / "diagnostic-linear.jxl"
    copied.write_bytes(control)
    static = args.out / "static-linear.pfm"
    native = args.out / "native-linear.npy"
    for name, command in [
        ("static", [str(args.djxl), str(copied), str(static), "--num_threads=1"]),
        ("native", [str(args.native), str(copied), str(native), "--data-type", "f32",
                    "--num-threads", "1", "--no-cms", "--allow-level10"])]:
        with (args.out / f"{name}.log").open("x") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    x, y, channel = args.xyc
    a, b = read_pixels(static)[:, :, :3], read_pixels(native)[:, :, :3]
    assert a.shape == b.shape
    report = {"source_sha256": digest(args.input), "control_sha256": digest(copied),
              "transfer_enum_bit_offset": offset, "xyc": args.xyc,
              "static_linear": float(a[y, x, channel]),
              "native_linear": float(b[y, x, channel]),
              "linear_difference_at_coordinate": float(abs(float(a[y, x, channel]) - float(b[y, x, channel]))),
              "maximum_linear_difference": float(np.abs(a.astype(float) - b.astype(float)).max()),
              "decoders": {"static": digest(args.djxl), "native": digest(args.native)}}
    for name, linear, original in [("static", a, args.static_pq), ("native", b, args.native_pq)]:
        original = read_pixels(original)[:, :, :3]
        assert original.shape == linear.shape, "PQ/linear dimensions differ"
        mapped = to_pq(linear)
        report[name + "_double_pq_at_coordinate"] = float(mapped[y, x, channel])
        report[name + "_original_pq_at_coordinate"] = float(original[y, x, channel])
        report[name + "_maximum_pq_reapplication_difference"] = float(np.abs(mapped.clip(0, 1).astype(float) - original.clip(0, 1).astype(float)).max())
    (args.out / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
