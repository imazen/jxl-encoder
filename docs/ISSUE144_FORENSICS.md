# Issue 144 retained investigation sources

This document accompanies the `experiment/issue144-forensics` archive. It
preserves investigation sources that were not selected for the maintained
encoder implementation. The owning contracts and tracked issue contain each
accepted change, its actual runtime verdict and its remaining gaps.

The extra scripts cover native CMYK/gain-map observers, exact temporary source
overlays and restoration, selected LLVM mapping, CLI child identity capture,
serialized replay intermissions, dependency-pin staging and retained-object
identity. They preserve the commands behind earlier evidence. Individual
runtime logs and source identities remain authoritative; membership in this
archive does not qualify a script, example or test.

The expanded `progressive_ac_cmyk` example and conditional-modular/decoder
document drafts preserve intermediate attribution work. Their statements may
predate subsequent fixes. Current maintained contracts take precedence.

`tests/frame_level_auxiliary_contract/cli.rs` is an uncompiled prototype that
assumed the large CLI would accept a 16384-square input. The actual CLI rejects
that shape at its default memory limit and has no trusted-limit override.
The executed CLI admission probe preserves the existing destination; the
trusted-limit library replay separately supplies the large frame-area witness.
The prototype is not registered in the maintained test target and must not be
treated as a passing regression or enabled without correcting its premise.

No generated images, encoded streams, native binaries or coverage profiles are
committed here. Retained data lives with the checksum-bound runtime artifacts.
The archive has no release or all-target build qualification. Its purpose is
to preserve exact investigation sources while completed implementation and
maintained tests proceed through their separately verified commits.
