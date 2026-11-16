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
