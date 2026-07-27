# Testing

GeoJAX uses deterministic numerical tests, executable documentation, and clean
package installation checks. A release should pass the complete matrix rather
than only the maintainer's active Python environment.

## Current environment

Run both JAX precision modes before committing:

```bash
make test
make test-float32
```

Both commands execute the full test suite with branch coverage. Coverage below
85 percent fails the run. `GEOJAX_TEST_X64` is set by these targets so tests do
not silently inherit a developer's JAX configuration.

Invariant assertions keep their original strict float64 tolerances. In
float32, tests add a small dtype-aware allowance for decomposition roundoff
and avoid perturbations below machine resolution. This preserves the
mathematical assertion instead of expecting float32 arithmetic to reproduce a
float64 residual.

## Numerical stability

Every public geometry is exercised with unbatched, nested-batch, and
broadcast-base inputs. The stability suite differentiates exponential maps at
zero tangents, logarithms and squared distances at coincident points, and SPD
matrix functions at repeated eigenvalues. Directional finite differences
independently check representative JVPs away from cut loci.

Contract tests additionally project zero, noisy, indefinite, and rank-deficient
ambient inputs and require the repaired value to pass membership. Malformed
event dimensions must fail before an operation can silently run on a different
manifold. Float32 tests explicitly check open-ball margins and the active
spectral floors used by SPD and fixed-rank repairs.

Second-order tests verify known Riemannian Hessian identities, including the
sphere shape-operator term, and require unsupported automatic conversions to
fail with an explicit request for `rhess_vec`.

Genuine singularities have separate regression tests. In particular, a
spherical antipode retains a finite distance but an explicitly nonfinite
logarithm and transport.

The learning release contract compiles complete deterministic-autoencoder
steps with spherical and hyperbolic latents. It requires decreasing
reconstruction loss, finite encoder gradients, and valid manifold points after
training. Learning-helper tests also combine vector-valued interpolation times
with batched endpoints and reject proxy distances or logarithms where the
public helper promises an exact geodesic operation.

## Supported matrix

Install the development dependencies and run:

```bash
python -m pip install -e ".[dev]"
make test-matrix
```

The first run downloads managed CPython interpreters through `tox-uv`.
Subsequent runs reuse those interpreters and isolated tox environments. Tox
uses managed interpreters even when a matching conda or system Python happens
to be active, making the matrix independent of a developer's base
environment. Missing versions are errors; they are never silently skipped.
The supported minor versions also live in `.python-versions`, so
`uv python install` can prepare all of them directly.

For a faster laptop run with bounded and coverage-safe concurrency:

```bash
make test-matrix-parallel
```

The default is two workers. Override it only when the machine has enough CPU
and memory, for example `make test-matrix-parallel TOX_PARALLEL=3`.

The release-blocking matrix is pinned and covers:

| Python | Dependencies | JAX precision |
|---|---|---|
| 3.11 | JAX 0.6.0 and NumPy 1.26.4 | float32, float64 |
| 3.11 | JAX 0.10.2 and NumPy 2.4.4 | float32, float64 |
| 3.12 | JAX 0.11.0 and NumPy 2.4.4 | float32, float64 |
| 3.13 | JAX 0.11.0 and NumPy 2.4.4 | float32, float64 |
| 3.14 | JAX 0.11.0 and NumPy 2.4.4 | float32, float64 |

The lower-bound environments guard the oldest runtime versions promised by
`pyproject.toml`. Stable environments pin JAX, JAXlib, NumPy, SciPy,
`ml-dtypes`, `opt-einsum`, pytest, and coverage tooling, preventing a new
upstream release from changing an otherwise identical release check. Update
these pins deliberately, then run the complete matrix before merging the
change. Tox also fixes the Python hash seed and stores coverage data inside
each environment, so sequential and modestly parallel runs have the same
isolation guarantees.

## Documentation

```bash
make website
```

This executes every MyST Markdown tutorial from a clean Sphinx environment,
treats warnings as errors, and audits rendered mathematics and local
references. `jupyter_execute/` and `.jupyter_cache/` are transient: the build
deletes both after the HTML audit succeeds. The `.md` tutorial is always the
maintained source.

## Release candidate

```bash
make release-check
```

The release target requires the complete tox matrix, rebuilds every tutorial,
creates clean wheel and source archives, and applies Twine's strict metadata
validation. It does not upload or publish anything.
