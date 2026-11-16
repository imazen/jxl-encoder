#!/usr/bin/env bash
# Retain both runtime controls; the existing full library step remains the gate.
set -uo pipefail
manifest=tools/reference-exec-probe/Cargo.toml
printf 'Normal container subprocess control\n'
cross test --locked --verbose --manifest-path "$manifest" --release --target i686-unknown-linux-gnu -- --nocapture
normal_rc=$?
printf 'normal_rc=%s\n' "$normal_rc"
printf 'Container subprocess control with seccomp disabled\n'
CROSS_CONTAINER_OPTS="${CROSS_CONTAINER_OPTS:-} --security-opt seccomp=unconfined" \
    cross test --locked --verbose --manifest-path "$manifest" --release --target i686-unknown-linux-gnu -- --nocapture
unconfined_rc=$?
printf 'unconfined_rc=%s\n' "$unconfined_rc"
# A failure stays visible as a failed diagnostic step (continue-on-error in CI).
# Do not suppress or bypass the subsequent full library test.
(( normal_rc == 0 && unconfined_rc == 0 ))
