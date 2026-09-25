# Historical correction evidence

These records were produced during the September 10–11 correctness audit.
They include deliberately failing reproductions, intermediate experiments,
superseded checks, successful focused checks, and interrupted matrix attempts.
The directory as a whole is **not** a passing test run.

`CORRECTNESS_AUDIT.md` at the repository root identifies the retained results
and their limitations. The September 25 matrix and packaging evidence is in
`audit/2026-09-25/validation/`.

The old `parallel_matrix_*.json` records with `status: running`, incomplete
`full_*.log` files, and process-ID files are historical snapshots of interrupted
runs. They are not active jobs or completed verification. Do not use their old
process IDs to control current processes. Temporary paths in these historical
records may no longer exist.

Files under `superseded_*` and `preliminary_*` preserve earlier evidence and
are excluded from the final passing matrix. The extreme-scaling probes retain
known subnormal-adjoint limitations; they are diagnostic outputs rather than
all-passing derivative tests.
