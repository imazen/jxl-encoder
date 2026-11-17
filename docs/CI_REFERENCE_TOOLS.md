# Reference-tool setup evidence

Run [44044868568](https://github.com/imazen/jxl-encoder/actions/runs/44044868568),
Windows ARM job 133908156355, exits with -2147483644 during the cached
reference version-export step. Rust is not installed and no encoder test
executes in that job. The captured log has no version output, so attribution
to a particular command or executable remains open. Full raw and cleaned
job logs are retained locally as `~/tmp/jxl144/ci-job-133908156355-*.log`.
GitHub refuses the job rerun while the enclosing workflow remains active.

The version loop previously used a captured command substitution under
`set -e`; a tool failure could terminate it before printing the captured
error. It now prints the selected path first and prints captured output and
status on failure. Version and failure checks retain their existing behavior.
This diagnostic change does not establish the cause of the Windows failure.

`python3 scripts/test_reference_version_diagnostics.py <NEW_OUTPUT_DIRECTORY>`
extracts the actual action's version loop and tests three explicit shell
fixtures: successful v0.12 export, rejected v0.11 output, and a tool printing
an error then exiting 23. All pass locally. The fixtures test orchestration;
they are not reference binaries or encoder conformance controls. Action YAML
and extracted shell syntax also validate. A subsequent remote job still needs
an actual platform verdict.


Expert job 133923235773 in run 44049340538 completes the issue-144 production
contracts, then fails the new wire-check command with `just: command not found`
(exit 127). The workflow now calls the Python checker directly; the exact
folded YAML command passes locally against the retained nine-cell native/encoder
controls and complete section accounting. No field expectation changes.
The local replay is `~/tmp/jxl144/ci-alpha-header-replay/`. A subsequent remote
job remains required. Windows ARM job 133923235733 separately completes
reference setup and starts its Rust build; that is not an encoder-test pass
or an attribution of the original setup exception. Resolve job state with
`gh api repos/imazen/jxl-encoder/actions/jobs/<job-id>`.
