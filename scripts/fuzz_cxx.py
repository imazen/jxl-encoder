#!/usr/bin/env python3
"""Keep cc's automatic ASan flag off the pinned libFuzzer engine itself.

cc 1.9 started forwarding Rust's address sanitizer to C/C++ build scripts.
The fuzz target and every other dependency retain their sanitizer flags.
The engine sources and all Rust coverage settings remain unchanged.
"""
import os
from pathlib import Path
import sys


def compiler_command(arguments, package, version):
    arguments = list(arguments)
    engine_source = any(
        Path(arg).parent.name == "libfuzzer"
        and Path(arg).name.startswith("Fuzzer")
        and Path(arg).suffix == ".cpp"
        for arg in arguments
    )
    if package == "libfuzzer-sys" and version == "0.4.13" and engine_source:
        arguments.append("-fno-sanitize=address")
    return ["clang++", *arguments]


if __name__ == "__main__":
    command = compiler_command(
        sys.argv[1:], os.environ.get("CARGO_PKG_NAME"), os.environ.get("CARGO_PKG_VERSION")
    )
    os.execvp(command[0], command)
