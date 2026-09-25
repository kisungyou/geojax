# September 25 verification evidence

The current conclusions are in [the correctness report](../../CORRECTNESS_AUDIT.md).
All ten selected final-source matrix runs passed; their independent checker
records 11,465 passed executions and 15 expected skips. Earlier failures and
interrupted attempts are preserved. A file named `final`
inside an earlier attempt is not automatically evidence for the current source.

## Selected final-source evidence

- `validation/matrix_final_pilot/`: complete installed-wheel minimum-JAX
  float32 run, started first because it exposed the mean-solver failure.
- `validation/matrix_final_a/` and `validation/matrix_final_b/`: the other
  nine pinned combinations, in two sequential batches with at most two
  numerical workers active overall.
- `validation/final_matrix_verification.json`: independent reconciliation of
  source/package hashes, collected and executed tests, JUnit results, skips,
  dependency versions and branch coverage for the ten selected runs.
- `validation/final/`: rebuilt distributions, clean installations,
  documentation execution/rendering, and current-dependency supplementary checks.
- `validation/release_source_guard/`: real release-recipe fixtures proving that
  regenerated tracked figures block packaging, plus separate release-contract
  replays in all ten environments against the final build files.
- `diagnostics/approximate_wolfe_validation.json` and
  `diagnostics/default_mean_consumers/`: focused line-search and downstream
  regressions preceding the matrix.
- `validation/frechet_default_stress/`: twenty analytic checks on other
  geometry families using the corrected default solver.
- `validation/shape_solver_fix_final/float32_default_stress.json` and
  `validation/shape_solver_fix_reference_copy/float64_default_stress.json`:
  the 24-case Kendall stress checks in each precision.

Each result file retains its own status and provenance; an unfinished entry
does not count as passing evidence. The final matrix checker requires exact
agreement with the current library, tests, runner and pinned configuration.

## Historical attempts

`validation/matrix/` predates the traceback-watchdog correction.
`validation/matrix_after_watchdog_fix/` exposed the separate float32 mean-solver
rounding failure; two subsequent runs were deliberately interrupted before
the library correction. Their statuses have not been relabeled as successes.
The root-level package/render records under `validation/` also predate the
mean correction; their replacements are under `validation/final/`.

`validation/shape_min_float32/` preserves the actual failure, independent
gradient/objective calculations, and the insufficient half-step-only experiment.
`validation/shape_solver_fix_final/` also retains a failed float64 *reference
probe*: NumPy returned a read-only view of a JAX array. The corrected reference
makes an explicit copy; no library or numerical-test tolerance changed.

Recorded process IDs are historical diagnostic data, not live status indicators.
Raw coverage databases are ignored by Git; the final coverage JSON reports,
collection manifests, JUnit records, logs and hash manifests are retained.

## Reproduction

The supported entry point is `make test-matrix-parallel`. The audit-only
`run_matrix.py` reuses already prepared tox environments and records a combined
summary. `verify_final_matrix.py --expected-collected 1147` independently checks
the selected final runs. The per-environment runner keeps the full test suite,
uses fresh processes, disables background traceback timers, and enforces its
external 600-second deadline and the unchanged coverage thresholds.
