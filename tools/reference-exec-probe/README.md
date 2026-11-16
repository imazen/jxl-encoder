# Reference execution probe

This dependency-free program runs a version probe with Rust's subprocess API
from the same target runtime as the cross tests. It reports file metadata,
spawn errors, exit status, stdout and stderr. `DJXL_PATH` must select a verified
libjxl v0.12 decoder.

Run from this directory with the normal resource wrapper:

```sh
cross run --locked --target i686-unknown-linux-gnu
```

The included Cross configuration matches the encoder's source/scratch mounts.
Supply the same path variables as the CI actions; the reference executable
must be inside a mounted directory. This is an execution diagnostic, not a
pixel correctness test.

The checksum-pinned Linux v0.12 release decoder runs through this probe in
the local cross 0.2.5 i686 image. That local control does not explain CI run
43870680134's reference-execution failure; failed version probes now retain
their actual spawn/exit diagnostics rather than only saying “does not run”.

Run 43874324031 identifies `Function not implemented (os error 38)` from
Rust subprocess spawning. The same image digest and Rust 1.100.0 succeed in
a local release test worker. The image's `/linux-runner i686` defaults to
native execution, so calling that local control QEMU execution was incorrect.
CI now records normal and seccomp-disabled diagnostic controls before the
unchanged full library gate. Neither control establishes pixel correctness.

Run 43877137348 executes the paired control on the hosted runner:
normal container `normal_rc=101` with ENOSYS, identical binary/image with
seccomp disabled `unconfined_rc=0`. The following full library step still
used the normal policy and failed reference launches. CI therefore scopes
`--security-opt seccomp=unconfined` to its i686 probe and full library
execution steps. The full suite's remote confirmation remains separate.
`just reference-cross-diagnostics` retains the paired diagnostic recipe.
