# /// script
# requires-python = ">=3.12"
# dependencies = ["Pillow>=11.0.0"]
# ///
"""Compare pinned libjxl revisions on the retained #110 float regression inputs.

No timings are reported. Every input and output is retained, and djxl must
reproduce the exact sample bits. The TSV is also consumed by the Rust decoder
verification mode in the lossless_float_parity example.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import struct
import subprocess

from jxl_bitstream_diff import Parser


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_pfm(path):
    stream = io.BytesIO(path.read_bytes())
    assert stream.readline().strip() == b"PF", path
    width, height = map(int, stream.readline().split())
    scale = float(stream.readline())
    assert abs(scale) == 1, (path, scale)
    body = stream.read()
    assert len(body) == width * height * 12, path
    if scale > 0:
        body = b"".join(body[i:i+4][::-1] for i in range(0, len(body), 4))
    row = width * 12
    return width, height, b"".join(body[y*row:(y+1)*row] for y in reversed(range(height)))


def run(command, log):
    with log.open("x") as f:
        subprocess.run(list(map(str, command)), stdout=f, stderr=f, check=True)


def grouping(data):
    """Parse actual frame grouping; accept bare streams or one jxlc box."""
    if not data.startswith(b"\xff\x0a"):
        offset = 0
        streams = []
        while offset < len(data):
            size = int.from_bytes(data[offset:offset+4], "big")
            kind = data[offset+4:offset+8]
            assert 8 <= size <= len(data)-offset, "unsupported/truncated box"
            assert kind != b"jxlp", "fragmented codestream not supported by this harness"
            if kind == b"jxlc":
                streams.append(data[offset+8:offset+size])
            offset += size
        assert len(streams) == 1, "expected one codestream box"
        data = streams[0]
    parsed = Parser(data)
    parsed.parse()
    return parsed.frame["group_size_shift"], parsed.toc["num_groups"]


def main():
    from PIL import Image

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("screen", type=Path)
    parser.add_argument("djxl", type=Path)
    parser.add_argument("--sizes", default="64,256,259")
    parser.add_argument("--group-size-shift", type=int, choices=range(4))
    args = parser.parse_args()
    root, screen = args.root.resolve(), args.screen.resolve()
    builds = json.loads((root / "builds.json").read_text())
    assert set(builds) == {"parent", "rewrite", "current"}, "incomplete builds"
    for build in builds.values():
        for path, expected in build["library_sha256"].items():
            assert sha(Path(path).read_bytes()) == expected, path
    inputs = root / "inputs"
    inputs.mkdir()
    outputs = root / "outputs"
    outputs.mkdir()
    repo = Path(__file__).resolve().parents[1]
    tables = [repo / "benchmarks/lz77_bucket_screen_2026-09-24.tsv",
              repo / "benchmarks/lz77_bucket_shapes_2026-09-24.tsv"]
    historical = []
    for table in tables:
        historical.extend(csv.DictReader(table.open(), delimiter="\t"))
    for row in historical:
        if "arm" not in row:
            row["arm"] = row["implementation"]
    source_hashes = {r["image"]: r["source_sha256"] for r in historical}
    columns = ["arm", "revision", "image", "size", "group_size_shift", "num_groups", "effort", "bytes",
               "source_sha256", "input_sha256", "encoded_sha256", "jxl", "raw", "djxl_exact"]
    with (root / "results.tsv").open("x") as f:
        writer = csv.DictWriter(f, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        for image, expected_sha in sorted(source_hashes.items()):
            source = screen / "input" / image
            assert sha(source.read_bytes()) == expected_sha, source
            original = Image.open(source).convert("RGB")
            for size in map(int, args.sizes.split(',')):
                x, y = (original.width-size)//2, (original.height-size)//2
                assert x >= 0 and y >= 0
                rgb = original.crop((x, y, x+size, y+size)).tobytes()
                raw = b"".join(struct.pack("<f", (v << 8) / 65535.0) for v in rgb)
                label = f"{source.stem}-{size}"
                raw_path = inputs / (label + ".f32le")
                raw_path.write_bytes(raw)
                pfm = inputs / (label + ".pfm")
                row = size * 12
                pfm.write_bytes(f"PF\n{size} {size}\n-1.0\n".encode() +
                                b"".join(raw[y*row:(y+1)*row] for y in reversed(range(size))))
                # Independently bind generated samples to the retained Rust baseline.
                old = [r for r in historical if r["arm"] == "chain" and r["depth"] == "f32"
                       and r["image"] == image and int(r["size"]) == size]
                assert len(old) == 1, (image, size, old)
                old = old[0]
                artifact = screen / old["artifact"]
                assert sha(artifact.read_bytes()) == old["encoded_sha256"], artifact
                check = inputs / (label + "-retained.pfm")
                run([args.djxl, artifact, check, "--quiet"], inputs / (label + "-retained.log"))
                assert read_pfm(check) == (size, size, raw), (image, size, "historical input mismatch")
                for effort in [8, 9]:
                    for arm, build in builds.items():
                        binary = Path(build["binary"])
                        assert sha(binary.read_bytes()) == build["binary_sha256"]
                        stem = outputs / f"{label}-e{effort}-{arm}"
                        encoded = stem.with_suffix(".jxl")
                        grouping = ([] if args.group_size_shift is None else
                                    ["--modular_group_size", str(args.group_size_shift)])
                        run([binary, pfm, encoded, "-d", "0", "-e", str(effort),
                             "--num_threads=1", "-x", "color_space=RGB_D65_SRG_Rel_Lin", *grouping],
                            stem.with_suffix(".encode.log"))
                        decoded = stem.with_suffix(".pfm")
                        run([args.djxl, encoded, decoded, "--quiet"], stem.with_suffix(".decode.log"))
                        assert read_pfm(decoded) == (size, size, raw), encoded
                        data = encoded.read_bytes()
                        shift, groups = grouping(data)
                        if args.group_size_shift is not None:
                            assert shift == args.group_size_shift
                        assert groups == ((size + (128 << shift) - 1) // (128 << shift)) ** 2
                        writer.writerow(dict(arm=arm, revision=build["revision"], image=image,
                                             size=size, group_size_shift=shift, num_groups=groups,
                                             effort=effort, bytes=len(data),
                                             source_sha256=expected_sha, input_sha256=sha(raw),
                                             encoded_sha256=sha(data), jxl=str(encoded), raw=str(raw_path),
                                             djxl_exact=True))
                        f.flush()
                        print(f"{label} e{effort} {arm}: {len(data)} bytes, exact", flush=True)


if __name__ == "__main__":
    main()
