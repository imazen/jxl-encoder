#!/usr/bin/env python3

"""Encode an SDR RGB8 photograph with a quality-gated JPEG XL decode profile."""

import argparse
import json
import os
from pathlib import Path
import shutil
import struct
import tempfile
from types import SimpleNamespace

from PIL import Image

from measure_lossy import PHOTO_FILTER_POLICIES, POLICIES, Study, digest, photo_filter_matches


BYTE_GROWTH = {"balanced": 3, "throughput": 10}


def photo_input(source):
    # Inspect the native depth and pre-IDAT color signals. Pillow presents
    # 16-bit RGB PNG as RGB and does not expose every PNG color chunk.
    non_srgb = False
    srgb_cicp = False
    with source.open("rb") as stream:
        signature = stream.read(8)
        header = stream.read(21)
        if (signature != b"\x89PNG\r\n\x1a\n" or len(header) != 21
                or header[:8] != b"\x00\x00\x00\x0dIHDR"):
            raise ValueError("profile input must be a PNG")
        rgb8 = header[16:18] == bytes((8, 2))
        stream.read(4)
        while len(chunk := stream.read(8)) == 8:
            size, kind = struct.unpack(">I4s", chunk)
            if kind == b"IDAT":
                break
            if kind == b"cICP" and size == 4:
                # BT.709/sRGB primaries, sRGB transfer, RGB, full range.
                srgb_cicp = stream.read(4) == bytes((1, 13, 0, 1))
                non_srgb |= not srgb_cicp
                stream.read(4)
                continue
            non_srgb |= kind in (b"iCCP", b"cICP")
            stream.seek(size + 4, os.SEEK_CUR)
    with Image.open(source) as png:
        width, height = png.size
        if (not rgb8 or png.mode != "RGB" or png.n_frames != 1
                or "transparency" in png.info):
            return png.size, "pixel layout outside the opaque RGB8 profile"
        if non_srgb:
            return png.size, "color encoding outside the sRGB scoring domain"
        if not (srgb_cicp or "srgb" in png.info) and (
                "gamma" in png.info or "chromaticity" in png.info):
            return png.size, "explicit PNG gamma or chromaticity without an sRGB tag"
        if min(width, height) < 8:
            return png.size, "dimensions below the metric minimum"
        return png.size, None


def candidates(band):
    if band == 1:
        return [
            ("base", 1),
            *(("tier0-epf0-gaboff", d) for d in (1, 0.98, 0.95)),
            *(("tier1-epf0-gaboff", d) for d in (1, 0.98, 0.955)),
            ("tier0-epf0", 1), ("tier0-epf0", 0.98),
        ]
    return [
        ("base", 4),
        *(("tier1-epf0", d) for d in (4.10, 4.08, 4)),
        ("tier1-epf2", 4.08), ("tier1-epf2", 4),
    ]


def tier_choices(study, cases, bands, name):
    return photo_filter_matches(
        study, cases, bands, "filters", max_growth=BYTE_GROWTH[name],
        fast_dc_growth=BYTE_GROWTH["balanced"] if name == "throughput" else None,
    )


def publish(source, output):
    # Keep the named destination absent until the final, verified bytes are ready.
    with tempfile.NamedTemporaryFile(dir=output.parent, prefix=".photo-jxl-") as stream:
        with source.open("rb") as encoded:
            shutil.copyfileobj(encoded, stream)
        stream.flush()
        os.link(stream.name, output)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--distance", type=int, choices=(1, 4), required=True)
    parser.add_argument("--profile", choices=BYTE_GROWTH, required=True)
    parser.add_argument("--effort", type=int, choices=range(1, 12), default=7)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--evidence", type=Path)
    for name in ("encoder", "decoder", "jxl-rs", "djxl", "scorer"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.threads < 0:
        parser.error("--threads must be non-negative")
    if os.path.lexists(args.output):
        parser.error(f"output exists: {args.output}")
    if not args.output.parent.is_dir():
        parser.error(f"output directory does not exist: {args.output.parent}")
    if args.evidence is None:
        args.evidence = args.output.with_suffix(args.output.suffix + ".fast-decode")
    return args


def main():
    args = parse_args()
    source = args.source.resolve(strict=True)
    (width, height), fallback = photo_input(source)
    source_sha256 = digest(source)

    POLICIES.update(PHOTO_FILTER_POLICIES)
    study = Study(SimpleNamespace(
        output=args.evidence, stage="encode-photo", blocks=3,
        encoder=args.encoder.resolve(strict=True), decoder=args.decoder.resolve(strict=True),
        baseline_decoder=args.decoder.resolve(strict=True), jxl_rs=args.jxl_rs.resolve(strict=True),
        djxl=args.djxl.resolve(strict=True), scorer=args.scorer.resolve(strict=True),
    ), cohort={"photo": {"path": str(source), "sha256": source_sha256}},
        effort=args.effort, threads=args.threads, affinity=False)
    band = args.distance
    max_growth = BYTE_GROWTH[args.profile]
    study.manifest["profile"] = args.profile
    study.manifest["max_byte_growth_percent"] = max_growth
    if fallback is not None:
        stream = study.streams / f"photo-native-base-d{band}.jxl"
        study.encode(source, stream, band, "base")
        selected = {
            "policy": "base", "distance": band, "file": stream.name,
            "bytes": stream.stat().st_size, "stream_sha256": digest(stream),
            "fallback": fallback,
        }
        study.manifest["inputs"]["photo-native"] = {
            "path": str(source), "source_sha256": source_sha256,
            "crop": [0, 0, width, height],
        }
        (study.out / "manifest.json").write_text(json.dumps(study.manifest, indent=2) + "\n")
    else:
        cases = [("photo", "native", candidates(band))]
        study.measure_quality(cases, check_target=True)
        matches = tier_choices(study, cases, (band,), args.profile)
        selected = matches["photo", "native", band][1][1].copy()
        if selected["policy"] == "base":
            selected["fallback"] = "no candidate meets rendered quality and byte limits"
        stream = study.streams / selected["file"]

    study.run([study.args.djxl, stream, "--disable_output", "--num_threads=0"])
    if digest(stream) != selected["stream_sha256"]:
        raise ValueError("selected file changed during verification")
    result = {
        "profile": args.profile, "reference_distance": band,
        "selected": selected, "output": str(args.output),
    }
    (study.out / "selection.json").write_text(json.dumps(result, indent=2) + "\n")
    publish(stream, args.output)
    print(f'{selected["policy"]} d{selected["distance"]:g}, {selected["bytes"]} bytes')


if __name__ == "__main__":
    main()
