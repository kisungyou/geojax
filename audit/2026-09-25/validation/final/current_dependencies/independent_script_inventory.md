# Independent checks and final-mean relevance

The supplemental harness passed all 70 selected tests in each precision, with
zero skips. Both independent script replays also passed after that suite.
Installed metadata confirms Python 3.13.11, JAX/jaxlib 0.11.2,
NumPy 2.5.3, SciPy 1.18.1, OTT 0.6.0, pytest 9.0.3, pytest-cov 7.1.0,
and coverage 7.16.1. All 67 installed package files match the frozen source.

The following existing independent scripts were replayed unchanged with the
final default Fréchet-mean solver:

- `audit/2026-09-10/statistical_crosscheck.py` compares Euclidean FANOVA with
  an independent NumPy calculation and a SciPy chi-square tail. Its sample
  and pooled means exercise the changed solver. It also enumerates all 70
  conditional label partitions for Biswas–Ghosh and energy statistics.
- `audit/2026-09-10/reproduce_limitations.py` preserves 11 original analytic
  audit checks, including the fixed-rank Bures and Kendall Fréchet means.
  Those two cases directly exercise the new default. The full script exits
  nonzero if any original finding is reproduced.

Both scripts explicitly enable float64, have no source-path insertion, and
ran using the current installed-wheel interpreter with `-I` from a temporary
directory. Their source hashes, JSON output, dependency metadata, exit status,
and installed-module locations were checked and preserved in `independent/`.
FANOVA matched its independent reference (statistic 9.17874383405389 versus
9.178743834053881), both exact permutation calibrations passed at all four
thresholds, and all 11 original analytic audit checks passed. The scripts took
8.267 and 22.754 seconds respectively. Source, wheel, environment, and original
script bytes were unchanged throughout.

The other independent scripts do not exercise the modified mean solver:

- `first_variation_crosscheck.py` checks the squared-distance first variation
  against logarithms on 18 geometries. Its geometry code is unchanged by
  this correction; no optimizer-driven repeat is required.
- `recheck_extreme_scaling.py` prints endpoint SPD arithmetic and derivative
  diagnostics. It explicitly inserts the checkout into `sys.path`, so an
  unchanged run would not establish installed-wheel behavior even with `-I`.
  It has no pass/fail assertions and concerns the already documented extreme
  floating-point range limitations, not the mean correction.
