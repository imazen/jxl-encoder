"""The runtime compiler must retain sanitizer flags outside the pinned engine."""
import unittest

from fuzz_cxx import compiler_command


class FuzzCompilerTests(unittest.TestCase):
    def test_pinned_engine_overrides_only_its_automatic_asan_flag(self):
        args = ["-O3", "-fsanitize=address", "-mpopcnt", "-c", "libfuzzer/FuzzerTracePC.cpp"]
        self.assertEqual(
            compiler_command(args, "libfuzzer-sys", "0.4.13"),
            ["clang++", *args, "-fno-sanitize=address"],
        )

    def test_other_packages_keep_every_flag_even_with_similar_source_names(self):
        args = ["-fsanitize=address", "-c", "libfuzzer/FuzzerTracePC.cpp"]
        for package, version in [("codec", "0.4.13"), ("libfuzzer-sys", "0.4.14"), (None, None)]:
            with self.subTest(package=package, version=version):
                self.assertEqual(compiler_command(args, package, version), ["clang++", *args])

    def test_compiler_probes_and_nonengine_sources_keep_every_flag(self):
        for args in [["--version"], ["-E", "probe.c"],
                     ["-fsanitize=address", "-c", "codec/FuzzerDecoder.cpp"],
                     ["-fsanitize=address", "-c", "libfuzzer/decoder.cpp"]]:
            with self.subTest(args=args):
                self.assertEqual(compiler_command(args, "libfuzzer-sys", "0.4.13"),
                                 ["clang++", *args])


if __name__ == "__main__":
    unittest.main()
