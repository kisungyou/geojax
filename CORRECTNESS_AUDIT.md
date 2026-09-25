# GeoJAX correctness audit and correction record

**Correction status, 25 September 2026: the five original open findings and the additional mean-solver rounding failure are corrected and covered by analytic regressions. All ten complete installed-wheel compatibility environments passed. Independent mathematical and statistical checks, current-dependency tests, documentation execution, package validation and release-source checks also passed. Numerical and platform limits are documented below.**

This report updates the [initial audit](audit/2026-09-10/INITIAL_AUDIT.md). The work started from commit `488890e`; the corrections and reproducible evidence are included in the accompanying commit. The initial failure evidence is retained rather than overwritten. This is an empirical and mathematical software audit, not a proof of every possible execution or statistical assumption.

## Resolution of the five open findings

| Finding | Correction | Independent check |
|---|---|---|
| O1: fixed-rank Bures derivatives and means | Differentiate invariant support projectors, rank-restricted Sylvester equations, and complete quotient quantities. Factor gauges and repeated eigenvectors are confined to primal calculations. Normalize intermediate covariance products before differentiation. | Rank-one scalar gradient `1−sqrt(2)`, Hessian `1/sqrt(2)`, and mean `((1+sqrt(2))/2)^2 P`; repeated positive/null spectra, rotated supports, batched adjoints, and large/small unit changes. |
| O2: Kendall-shape derivatives and means | Implicit orientation-preserving Procrustes derivative and horizontal projection, plus a smooth squared spherical angle. Determinant signs use `slogdet` to avoid underflow. | Balanced shapes and regular rank `(m−1)` configurations recover the exact first and second geodesic derivatives. Ordinary default-solver means recover their midpoints. Truly nonunique alignments remain excluded. |
| O3: Grassmann logarithmic features | Differentiate the arctangent matrix function through the inverse tangent-polar equation. Use a reflection logarithm when the inverse-overlap chart becomes numerically unstable. Squared distances use first variation. | Repeated nonzero/zero principal angles; forward/reverse derivatives; exact log-feature and distance Hessians; rotated unequal angles close to the cut in both precisions. |
| O4: SPD higher derivatives | Implicit Sylvester rules for square roots; inverse-Dexp rules for logarithms; differentiable matrix absolute value for regular repairs. Preserve broadcasting and isolate genuinely singular boundary items. | All three original distance Hessians at `I` versus `2I`; analytic spectral trace Hessians; third log derivative; mixed repair batches; compiled large-scale inputs and nearly repeated eigenvalues. |
| O5: rotation-group higher derivatives | Invert the exponential differential on skew tangents, augmented by an identity on the symmetric complement. Refine the spectral inverse against the defining operator. | SO(2) geodesic second derivative `4`; independent noncommuting SO(3) Hessian; repeated angles in larger groups; JIT/vmap/adjoints and near-pi configurations. |

The corrections use [JAX implicit linear solves](https://docs.jax.dev/en/latest/_autosummary/jax.lax.custom_linear_solve.html): derivatives act on the defining matrix equation rather than a choice of eigenvectors. The solve callbacks are true inverses on their augmented matrix spaces; symmetry and transpose rules were checked separately. Eigenvalue jitter is not used to conceal degeneracy.

For a square root `R²=P`, its directional derivative satisfies `R dR + dR R = dP`. For the logarithm, `Dexp_log(P)[Dlog_P(E)] = E`. These equations remain meaningful at repeated positive eigenvalues. Analogous support and quotient equations handle structural zero eigenvalues on fixed-rank strata.

## Additional corrections found during the recheck

- **Bures exponential domain:** the complete horizontal factor segment is checked for rank loss. The original audit's regular path beyond positive overlap now succeeds; paths that leave and reenter the stratum are rejected. Numerical rank tolerances remain explicit.
- **Related low-rank retractions:** shared PSD, trace-normalized, correlation-normalized, and rectangular fixed-rank projections have invariant derivatives on regular strata. Rectangular projection uses a thin implicit solve; a `10000×3` primal/JVP test prevents an accidental ambient-size eigendecomposition.
- **Optimizer cancellation:** Hestenes–Stiefel directions can cancel to roundoff while the gradient remains large. Conjugate gradient now restarts when meaningful descent is lost relative to the preconditioned gradient. This fixed the default shape-mean stall uncovered after its geometry derivatives were repaired.
- **SPD range and batching:** nearly equal logarithmic eigenvalues use a relative-gap `log1p` calculation, preserving accuracy under large unit changes. Frechet maps and Sylvester solves explicitly broadcast their operands. Mixed repair batches no longer introduce eigenvector NaNs into unrelated valid items.
- **Compiled arithmetic:** split power-of-two factors and optimization barriers preserve safe arithmetic order. The selected derivative of JAX `ldexp` at a zero coordinate is avoided by multiplying constant factors instead. Projection eigendecompositions are rescaled before extreme positive/negative inputs reach the backend.
- **Learning and optional transport:** pairwise kernels correctly handle captured constants and adapted traced data. Quadratic Sinkhorn costs use `squared_dist`, preserving the coincident Hessian. Real OTT-JAX tests verify translation costs, compiled gradients, and zero-weight support points.
- **Installed-package test isolation:** documentation tests load their checker by file path. They no longer require the checkout on `sys.path`, which would defeat verification of the installed wheel.
- **Final range recheck:** SPD and fixed-rank PSD kernels now preserve the smallest normal covariance values as well as values near the finite maximum. A shared symmetric-part kernel uses a constant block linear equation with range-safe arithmetic in its solve callback; its AD rule remains linear. Membership tests use scaled spectra, and Sylvester/logarithm equations are normalized at small scales as well as large ones.
- **Rotation projection:** the nearest proper polar factor now reuses the implicit Procrustes derivative. Analytic first/second derivatives cover repeated positive singular values, unique reflected inputs, rank-$(n-1)$ points, batching and adjoints. Nonunique reflected ties remain excluded.
- **Dictionary objective:** a Gram quadratic could cancel into a negative reported squared error near an exact reconstruction. The objective and its history now use the weighted tangent residual norm plus a separately preserved ridge term. The solver's valid incomplete-status outcome is checked through its convergence certificate instead of a restricted list of descriptive messages.
- **Repeated compilation:** caching the SPD projection and its conditional derivative removed a documented covariance-tutorial timeout. The unchanged computation was measured above 300 seconds before the repair and below 80 seconds with the cached kernel; the full strict documentation execution is checked separately below. This follows [JAX's compilation-cache guidance](https://docs.jax.dev/en/latest/jit-compilation.html).
- **SPD exponential differential:** the native large-argument exponential JVP lost about 0.3% accuracy for a float32 covariance near `1e-12`. Evaluate the differential with a fixed Padé approximant at a small argument and recover it with `Dexp_(2A)(E) = sym(exp(A) Dexp_A(E))`, using accurate symmetric exponential factors. Precompute those nonlinear factors outside the linear recovery loop so compiled reverse differentiation preserves their custom derivative rules. Independent divided-difference, second/third-derivative, adjoint and batched checks cover broad spectra, tiny tangents, and the smallest normal exponential values.

## Earlier verified corrections retained

| ID | Verified failure before correction | Correction and validation |
|---|---|---|
| F1 | Squared-distance Hessians at coincident points were too small or zero. For example, Euclidean and spherical geodesic tests returned 1 instead of 2; the tested oblique case returned 0.5 instead of 4. | Removed unnecessary `maximum(quadratic, 0)` operations and norm-then-square compositions where a smooth squared quantity exists. JAX assigns a half derivative to `maximum` at equality. Eleven geometry/product cases now satisfy the metric-Hessian identity under compiled second differentiation. The separate spectral defects are now addressed by O1–O5 above. |
| F2 | Spherical distance rounded a resolvable angle of `1e-9` to zero in float64. Dot-product/arccos formulas discard small-angle information. | Added a two-chord half-angle formula and a squared-ratio series near coincidence; used it for spheres, oblique columns, and the simplex square-root embedding. Distances and gradients are checked down to `1e-9`. |
| F3 | ECM/LEC correlation squared-distance gradients at the identity were NaN, although the chart is smooth there. | Reused an SPD projection derivative that is the identity inside the positive-definite cone, avoiding differentiation of repeated-eigenvalue eigenvectors. Checked the exact two-dimensional chart gradient and coincident Hessian. |
| F4 | Torus wrapping erased angular displacements below machine epsilon through addition/subtraction of pi. | Preserve already canonical angles; compute squared distance directly from squared wrapped differences. A `1e-9` displacement survives in both precisions. |
| F5 | Hyperboloid exponential acceleration at the origin was zero in its time coordinate instead of one. | Compute the upper-sheet coordinate as the norm of the augmented vector `[1, spatial]`, avoiding a nested zero-norm derivative. Checked the exact second derivative of the geodesic. |
| F6 | `LeastSquares.cost_and_grad` ignored an explicitly supplied adjoint Jacobian. An opaque residual with valid derivative callbacks made Gauss–Newton stop at the wrong initial point. | Honor the callback in the combined evaluation and validate its tangent output. Gauss–Newton and Levenberg–Marquardt solve the opaque-residual example correctly. |
| F7 | Trust-region, adaptive cubic, and Levenberg–Marquardt solvers aborted when an oversized trial left a retraction's valid domain. | Route trials through existing domain-aware validation so a failed trial is rejected and the radius/regularization/damping can adapt. All three recover on an independently bounded retraction and reach the known solution. |
| F8 | A zero-weight antipodal observation could make an otherwise trivial mean or median fail through `0 * NaN` derivatives. | Remove zero-mass observations before the affected objectives/logarithms. Updated intrinsic means/medians, scalable diagnostics and updates, clustering diagnostics, geodesic regression, spatial depth, and robust final stationarity checks. Eight regression cases cover ordinary and Product-valued mean/median/scalable summaries. This is not a claim that every weighted API and every cut-locus configuration is now supported. |
| F9 | Transportation-simplex initialization and pivot removal could discard positive mass smaller than the optimization tolerance. | Treat exhaustion and leaving-edge ties using the actual transported masses. Checked an explicit `0.0004` residual-mass case and 12 independent randomized problems against SciPy/HiGHS, including costs and both marginals. |
| F10 | Gaussian affinities/MMD could compute `inf / inf` even when distance divided by bandwidth was representable. | Divide by bandwidth before squaring; similarly separate the self-tuning scales. The MMD statistic and Gram matrix are invariant under large unit changes (`1e200` in float64, `1e20` in float32). |
| F11 | The existing simplex conversion test failed for equal weights at the largest finite value. Backend reciprocal underflow defeated its overflow fallback. | Added power-of-two rescaling with a split exponent. The split also avoids an intermediate-power limitation in JAX 0.6.0. The pre-existing failure now passes in both tested JAX generations. |
| F12 | General sample-weight normalization rejected valid weights at the floating-point limit. | Use the same exponent rescaling before summation. Checked a nonuniform `2:1:0` ratio at the largest finite scale. |
| F13 | A spherical extrinsic mean with very large, equal positive weights returned NaN instead of the known mean direction. | Normalize rescaled weights before forming the ambient mean. Checked two orthogonal unit vectors with maximum finite weights. |
| F14 | Biswas–Ghosh inference returned a p-value even when its statistic overflowed to infinity. | Require a finite observed statistic and finite permutation statistics; otherwise raise an explicit rescaling error. Tested both precisions. |
| F15 | In float32, local-linear regression could stall and raise nonconvergence on the exact affine relation `y=2+x`, queried at `x=1.5`. This was also reproduced from the untouched initial commit. | Disable direction normalization in the signed local-linear solve, consistent with the intrinsic mean fitting path. The example reaches the analytic answer in one step; added affine-extrapolation checks at response scales `1e-5`, `1`, and `1e5`. |


## Final verification

The principal CPU matrix contains ten pinned combinations: Python 3.11 with JAX 0.6.0 and 0.10.2, and Python 3.12, 3.13, and 3.14 with JAX 0.11.0, each in float32 and float64. All ten complete runs passed using the same final wheel without changing their pinned dependencies. Every run verifies its import location and module hashes and executes the complete tests with coverage. The [test runner](scripts/run_test_suite.py) starts a fresh process for each test module and combines the coverage data before applying the unchanged 85% global and 95% learning thresholds. It records every collected test and its execution outcome, per-module timings, diagnostic stack traces for long-running tests, and source hashes before and after the run. Process isolation changes test orchestration; it does not remove tests or weaken numerical expectations.

The 25 September diagnostics completed the two previously slow areas in isolated processes without library changes. The shape audit file passed all 22 tests in 154.49 seconds under Python 3.11/JAX 0.6.0/float64; its wrapper recorded 166.31 seconds including process startup and reporting. The numerical-stability file passed all 108 tests in 184.32 seconds under Python 3.12/JAX 0.11.0/float64. Both diagnostics enabled coverage. These results show that both files finish independently. They are consistent with accumulated compilation or other process resources contributing to the earlier long full-suite runs, but do not establish a precise cause. Logs and timings are retained in [the September 25 diagnostics](audit/2026-09-25/diagnostics/).

<!-- MATRIX_RESULTS -->

All ten selected runs passed: **11,465 passed test executions and 15 expected skips** across the matrix. Each environment completed all 41 modules and collected the same 1,147 test IDs. The absent optional OTT module contributes an additional collection-level skip; float32 also skips the one explicitly float64-only metric-learning case. The real OTT backend is tested separately below.

| Python | JAX | Precision | Passed / skipped | Global coverage | Learning coverage |
|---|---|---|---:|---:|---:|
| 3.11.15 | 0.6.0 | float32 | 1,146 / 2 | 91.8185% | 96.8968% |
| 3.11.15 | 0.6.0 | float64 | 1,147 / 1 | 91.8114% | 96.7866% |
| 3.11.15 | 0.10.2 | float32 | 1,146 / 2 | 91.7899% | 96.8234% |
| 3.11.15 | 0.10.2 | float64 | 1,147 / 1 | 91.8114% | 96.7866% |
| 3.12.13 | 0.11.0 | float32 | 1,146 / 2 | 91.6827% | 96.8234% |
| 3.12.13 | 0.11.0 | float64 | 1,147 / 1 | 91.7042% | 96.7866% |
| 3.13.14 | 0.11.0 | float32 | 1,146 / 2 | 91.6827% | 96.8234% |
| 3.13.14 | 0.11.0 | float64 | 1,147 / 1 | 91.7042% | 96.7866% |
| 3.14.6 | 0.11.0 | float32 | 1,146 / 2 | 91.6827% | 96.8234% |
| 3.14.6 | 0.11.0 | float64 | 1,147 / 1 | 91.7042% | 96.7866% |

Coverage includes branches and exceeds the unchanged 85% global and 95% learning thresholds in every environment. The independent evidence checker reconciled every collected ID, runtime phase and JUnit case, all nine dependency pins, and all 67 installed-file hashes against the final source. Its own checks rejected all 20 deliberately corrupted evidence cases. [Complete matrix evidence](audit/2026-09-25/validation/final_matrix_verification.json), [evidence-checker selftests](audit/2026-09-25/validation/final_matrix_validator_selftest.json).

The first verification batch completed three environments using the initial isolated runner. A
separate minimum-stack float32 run hit a diagnostic failure: the clustering
test printed `PASSED`, then pytest waited indefinitely for CPython's timed
traceback watchdog to finish. A native stack sample placed the main thread in
watchdog cancellation and the diagnostic thread in `dump_frame`, while JAX
workers were idle. The parent deadline terminated that incomplete run. The
entire 28-test module then passed in 103.11 seconds with the watchdog unable to
fire. This identifies the stalled subsystem; the exact interpreter race is
not established by the sample alone.

The corrected runner disables background traceback timers, retains the
external 600-second deadline, and requests a snapshot only after that deadline
has failed. Fifteen runner/release checks passed in each of the three
previously completed environments. Library code and numerical tests were
unchanged across that diagnostic-only transition. The later mean-solver correction made those complete runs historical
evidence; the successful final-source matrix above reran the entire suite
after that correction.
[Diagnosis and primary-source references](audit/2026-09-25/validation/runner_checks/watchdog_diagnosis.md),
[bounded replay](audit/2026-09-25/validation/watchdog_stall/replay_without_timer/result.json).

Retained verification from 11 September 2026:

- **Dependency follow-up passed:** 252 regression, transformation, and real-OTT tests in each precision, using the then-built installed wheel with JAX/JAXlib 0.11.1, NumPy 2.5.3, and SciPy 1.18.1. Both runs exclude checkout imports and verify all 66 installed module hashes. These version numbers describe the tested September 11 environment, rather than a claim about today's newest releases.
- **Independent formulas passed:** all 11 original failure reproductions and all 18 first-variation identities. FANOVA agrees with the independently evaluated formula and SciPy survival function; the two exhaustive 70-labeling permutation checks passed. Both-precision range probes were replayed separately and record the extreme-scale derivative limitation described below.
- **Documentation passed:** all 25 notebooks and 97 code cells executed under the unchanged 300-second cell timeout. The strict Sphinx build and HTML audit passed, covering 45 content pages, 551 math nodes, and 5,307 local references. Output review found no execution errors or flagged numerical output. All 162 original input files remained unchanged.
- **Historical artifacts passed:** the September 11 wheel and source archive passed strict metadata checks and each installed successfully in a fresh environment with dependency checks. All 66 Python module contents matched the frozen source in both artifacts. Those temporary archives are no longer present; their recorded hashes and logs remain historical evidence. The rebuilt artifacts are checked separately below.
- **Source quality passed:** lint, formatting, and whitespace checks passed on September 11. The new test orchestration is checked separately below.

When verification resumed on 25 September, all 107 package/test files matched the September 11 test manifest, all 66 package modules matched the historical artifact manifest, and all 162 documentation inputs matched the executed-documentation manifest. The retained checks describe those recorded inputs. Subsequent edits include test orchestration, the mean solver and line search, their regressions, and documentation; the September 11 documentation pass describes the earlier input snapshot. The retained results are not presented as a completed run of the new compatibility matrix.

### Mean-solver rounding found by the compatibility rerun

Python 3.11/JAX 0.6.0/float32 reproduced a mean that stopped with gradient
`4.0301e-5`, above its unchanged `2.4414e-5` tolerance, for two regular Kendall
shapes separated by 0.1 radians. Independent angle calculations and first
variation agreed with the computed gradient. The evaluated objective rounded
below its true minimum, so strict Armijo rejected a half-gradient step whose
true cost and gradient improved. A half-step alone passed 23 of 24 geometric
stress cases but failed a second regular rank-three shape in dimension four.
These failed runs are preserved and are not counted as passing verification.

The correction enables a bounded approximate-Wolfe safeguard for the default
mean solver and starts with a half-gradient step, matching the local quadratic
curvature of weighted squared distance. The safeguard uses the actual
retraction-curve derivative, the Hager–Zhang slope inequalities, and the
strong-curvature bound. Its cost discrepancy is bounded by
`8 * eps * abs(starting_cost)` for each line search; it introduces no unit-scale
absolute floor.
The acceptance reason distinguishes the safeguard from strict sufficient
decrease, and the final gradient convergence threshold is unchanged. Custom
retractions that raise the specific JAX Python/NumPy conversion errors during
scalar-step tracing retain ordinary Armijo; unrelated failures propagate.
Explicitly supplied solvers retain their settings.
This is a finite-precision stationarity safeguard, not a certificate of exact
monotonicity below objective rounding or of global optimality.
[Primary mathematical reference, Section 4](https://www.math.lsu.edu/~hozhang/papers/cg.pdf).

The original midpoint regression is unchanged. Added regressions cover the
second failing midpoint and nonuniform means of three and five observations,
whose exact answers are weighted geodesic-angle means and variances.
[Independent diagnostics and stress evidence](audit/2026-09-25/validation/shape_min_float32/),
[line-search regressions](tests/test_linesearch.py),
[shape-mean regressions](tests/test_shape_audit_fixes.py).
Final default-solver checks passed all 24 Kendall cases in each precision and
all 20 other-geometry cases without changing point or gradient tolerances.
[Float32 Kendall results](audit/2026-09-25/validation/shape_solver_fix_final/float32_default_stress.json),
[float64 Kendall results](audit/2026-09-25/validation/shape_solver_fix_reference_copy/float64_default_stress.json),
[other-geometry results](audit/2026-09-25/validation/frechet_default_stress/results.json).

<!-- ADDITIONAL_RESULTS -->

Additional verification on 25 September 2026:

- **Current-dependency supplement passed:** 70 tests in each precision (140 total, no skips) against the final installed wheel with JAX/JAXlib 0.11.2, NumPy 2.5.3, SciPy 1.18.1 and OTT-JAX 0.6.0. The checks cover the real optional transport backend, all line-search regressions and all shape-audit cases. Every installed module hash, dependency version and source input remained unchanged. [Complete results](audit/2026-09-25/validation/final/current_dependencies/results.json).
- **Independent final-source replays passed:** all 11 original analytic reproductions passed with the installed wheel. FANOVA agreed with the independent Euclidean formula and SciPy survival function (statistics `9.17874383405389` and `9.178743834053881`), and both exhaustive 70-labeling permutation calibrations passed all four checked thresholds. The original scripts and numerical expectations were unchanged. [Replay evidence](audit/2026-09-25/validation/final/current_dependencies/independent/results.json).
- **Final artifacts passed:** the final wheel and source archive passed strict metadata checks, fresh isolated installations and dependency checks. All 66 package modules and `py.typed` match the final checkout in both artifacts and in a wheel independently rebuilt from the source archive. All 220 included checkout files are byte-identical; the archive includes 42 test files and four verification scripts. This exact wheel is installed, without changing dependency versions, into all ten compatibility environments. The distributions are retained in `dist/audit-2026-09-25-final/`. Fresh installation resolved JAX/JAXlib 0.11.2 and NumPy 2.5.3; installation success alone is not presented as numerical verification. [Final artifact provenance](audit/2026-09-25/validation/final/package_provenance.json), [fresh installations](audit/2026-09-25/validation/final/artifact_smoke.json), [source-archive rebuild](audit/2026-09-25/validation/final/sdist_rebuilt_wheel.json).
- **Runner checks passed:** the original 13 focused runner/release checks passed; after the watchdog correction, all 15 updated checks passed in each of the three previously completed environments. A separate synthetic run exercised actual subprocess collection, explicit module skips and coverage combination, and correctly rejected deliberately insufficient coverage. [Runner evidence](audit/2026-09-25/validation/runner_checks/README.md).
- **Final documentation passed:** all 25 notebooks and 97 code cells executed against the corrected package in 495.27 seconds, with the unchanged 300-second cell timeout. Strict Sphinx completed with no warnings; the HTML audit checked 45 content pages, 551 math nodes and 5,309 local references. Output review found no notebook errors or nonfinite text. One nonfatal plotting-layout warning was retained and its actual figure visually checked for complete labels and panels. A later one-sentence clarification of custom-retraction fallback was separately rendered with strict checks (45 pages, 551 math nodes, 5,273 local references); package and tutorial inputs are byte-identical to the executed snapshot. Seven of the 162 historical inputs differed from September 11 at execution. The final release-guard change makes eight; its provenance record confirms that the website recipe, package and tutorials did not change. [Combined execution, output and prose-delta evidence](audit/2026-09-25/validation/final/docs/summary.json), [release-guard input delta](audit/2026-09-25/validation/final/docs/release_guard_input_delta.json).
- **Release-source guard passed:** tutorial execution can regenerate four tracked figures after the initial tag check. The release recipe now checks clean tagged inputs again immediately before packaging, and the guide explains reviewing generated figures before tagging. Temporary Git fixtures using the actual recipe and source verifier reproduce the former dirty-artifact path, allow clean packaging, and reject changed tracked figures before packaging. All six unchanged release-contract tests passed separately in all ten environments after the final build-file edits. Those 60 repeated checks are supplemental, not additional unique matrix tests. [Release-guard evidence](audit/2026-09-25/validation/release_source_guard/result.json).
- **Source quality passed:** lint and formatting passed for all 115 package, test, documentation and script Python files, along with whitespace checks. The final matrix checker confirmed unchanged package, test and runner inputs throughout all ten runs and exact agreement with the final source; the staged source also passes whitespace checks.

The first attempt at the installed-package matrix exposed the documentation-checker import problem before tests could execute. A later partial run was deliberately stopped when the additional low-rank unit-scaling issue was confirmed. Those logs are retained as preliminary evidence and are not counted as final passing runs. Documentation execution also required permission for Jupyter's local kernel ports; the denied sandbox attempt is retained separately.

The next partial matrix and its successful 224-test latest-dependency checks were also superseded when the final independent review found the range/projection defects and dictionary cancellation above. Their evidence remains under [the superseded September 11 checks](audit/2026-09-10/corrections/superseded_before_final_review/) and is not counted as verification of the final source.

Reproducible evidence and tests:

- [Initial audit reproductions](audit/2026-09-10/reproduce_limitations.py), with unchanged analytic inputs and expectations.
- [Original audit regressions](tests/test_correctness_audit.py).
- [SPD derivatives](tests/test_spd_audit_fixes.py), [spectral repair](tests/test_spd_projection_audit.py), and [exponential scaling and higher derivatives](tests/test_spd_exponential_audit.py).
- [Low-rank/Bures](tests/test_low_rank_audit_fixes.py), [Kendall shape](tests/test_shape_audit_fixes.py), and [Grassmann/SO](tests/test_logarithm_audit_fixes.py).
- [Optimizer cancellation](tests/test_optimizer_audit_fixes.py), [learning compilation](tests/test_learning_geometry_audit.py), and [real optional transport](tests/test_optional_transport.py).
- [All correction evidence](audit/2026-09-10/corrections/), including source manifests, package hashes, complete logs and numerical outputs.

## Scope and remaining limitations

No mathematical derivative is promised at a genuine cut locus, a zero-eigenvalue repair discontinuity, or a nonunique rank-truncation boundary. Numerically regular points are distinguished from those boundaries using dtype-aware tolerances. Low-rank nonpositive-spectrum repair retains a first-order selection without a higher-order guarantee outside its regular positive neighborhood.

Automatic Riemannian Hessian conversion remains guarded on geometries that require a supplied Hessian. Correct matrix-function derivatives do not by themselves derive every geometry's connection formula. Use `squared_dist` for smooth squared-distance objectives; differentiating `dist(...)**2` through a selected zero-norm derivative is not an equivalent Hessian contract.

The near-cut Grassmann fallback uses an ambient-size matrix logarithm, with cubic work and quadratic storage; vectorization can execute both conditional branches. Ordinary thin-chart operations and rectangular low-rank operations retain their smaller-matrix algorithms. Large-data performance and accelerator behavior require separate measurement.

General Fréchet means certify local stationarity, not uniqueness or global optimality. FANOVA asymptotics require regularity; permutation validity requires exchangeability; energy/MMD interpretation requires the stated metric/kernel assumptions. The independent statistical checks verify controlled identities and conditional calibration, not every data-generating distribution.

Validation here is on macOS ARM64 CPU. GPU/TPU execution has not been checked. Passing these tests closes the reproduced defects and materially strengthens confidence; it is not an assurance that no undiscovered technical issue can exist.

Representability applies to intermediate derivatives as well as final values. For example, logarithm and Sylvester calculations at a covariance near the largest finite value have subnormal ambient adjoints; the tested CPU backend can flush them to zero before a surrounding change of variables rescales them. The retained September 11 installed-package probes record that limitation in both precisions: the scaled logarithm's first and second derivatives return zero instead of `2` and `−2`, and the scaled Sylvester trace derivative returns zero instead of `−1`. The endpoint primal checks and small-scale implicit-derivative checks do not promise recovery of every subnormal intermediate. Use suitable input coordinates/units for differentiated models. Evidence: [float64 range evidence](audit/2026-09-10/corrections/recheck_extreme_scaling_float64.log) and [float32 range evidence](audit/2026-09-10/corrections/recheck_extreme_scaling_float32.log); these are diagnostic probes, not all-passing derivative assertions.
