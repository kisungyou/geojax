# GeoJAX correctness audit — 10 September 2026

**Assessment: substantial, verified problems were found and corrected, but the package still has five reproducible differentiation defects. It should not yet be described as free of technical issues.** Two of the remaining defects break ordinary Fréchet-mean fitting on valid, simple inputs. An additional conservative exponential-map domain restriction is documented below as a limitation rather than an incorrect local formula.

The audit started from clean commit `488890e` (`Harden GeoJAX numerical and release contracts`). Changes are left in the working tree for review. No release, commit, or publication was made.

## Evidence and scope

The work combined source review across geometry, optimization, and learning; the existing test suite; independent analytic derivative identities; adversarial numerical inputs; comparisons with SciPy linear programming and distribution functions; and artifact build/installation checks. This is a targeted mathematical and computational audit, not a formal proof of every execution path.

The principal environment was Python 3.12.13, JAX/JAXlib 0.11.0, NumPy 2.4.4, macOS 26.6.2 on ARM64 CPU. Both float32 and float64 were exercised. Focused regressions also ran with the supported JAX 0.6.0 floor under Python 3.11. GPU/TPU execution, every Python/dependency combination, large-data performance, and the optional `ott-jax` backend were not tested in this audit. `ott-jax` was not installed in the test environment.

Reproduction artifacts:

- [New regression tests](tests/test_correctness_audit.py): 41 cases for the verified fixes and independent transport comparisons.
- [Remaining-defect reproducer](audit/2026-09-10/reproduce_limitations.py): prints actual and analytic expected values, and deliberately exits nonzero while findings remain open. These failures are not hidden as passing or expected-failure tests in the main suite.
- [Statistical cross-check](audit/2026-09-10/statistical_crosscheck.py): independent Euclidean FANOVA calculation and exhaustive conditional permutation checks.
- [First-variation cross-check](audit/2026-09-10/first_variation_crosscheck.py): the squared-distance first-variation identity at noncoincident regular pairs on 18 geometries.
- [Captured evidence](audit/2026-09-10/): final logs and numerical results.

Run the standalone diagnostic scripts from the repository with `PYTHONPATH=. python audit/2026-09-10/<script>.py`. The statistical script additionally requires SciPy.

## Corrections made

The table groups related failures by cause, rather than counting every affected method as a separate issue.

| ID | Verified failure before correction | Correction and validation |
|---|---|---|
| F1 | Squared-distance Hessians at coincident points were too small or zero. For example, Euclidean and spherical geodesic tests returned 1 instead of 2; the tested oblique case returned 0.5 instead of 4. | Removed unnecessary `maximum(quadratic, 0)` operations and norm-then-square compositions where a smooth squared quantity exists. JAX assigns a half derivative to `maximum` at equality. Eleven geometry/product cases now satisfy the metric-Hessian identity under compiled second differentiation. This does **not** resolve the separate spectral defects below. |
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

For a tangent vector `u`, the main independent geometry oracle was

`d(p, Exp_p(t u))² = t² g_p(u,u)`

inside a normal neighborhood. Therefore the second derivative at zero must be exactly `2 g_p(u,u)`. Checking only that derivatives are finite would miss several of these failures.

At generic noncoincident pairs, all 18 additional geometry cases also satisfy the first-variation identity `D[d(Exp_p(tu),q)²]_(t=0) = -2 g_p(Log_p(q),u)`. These passing generic examples help localize the remaining failures to special, valid configurations; they do not remove the need to support those configurations.

## Remaining differentiation defects

These examples use valid points and smooth local mathematical maps. Their failures are not explained by an antipode, a cut locus, or a rank-changing perturbation. They remain unresolved in the working tree.

### O1 — High priority: low-rank Bures first derivatives and mean fitting

Location: `geojax/geometry/low_rank.py`, especially `_eigen_factors`, `project`, `sylvester`, and the factor/logarithm path used by squared distance.

For `RankKPSDBuresWasserstein((3,3), rank=1)`, let `P = diag(1,0,0)` and `Q = 2P`. Along the fixed-support curve `sP`,

`d(sP,Q)² = (sqrt(s) - sqrt(2))²`.

At `s=1`, the derivative must be `1 - sqrt(2) = -0.4142135623730951`. GeoJAX returns **NaN**. The repeated zero eigenvalues are structural for this valid rank-one geometry. Raw eigenvector differentiation introduces a singularity that is absent from the quotient quantity.

`frechet_mean(M, stack([P,Q]))` also raises `FloatingPointError: Riemannian gradient must contain only finite values`. Its known answer is `((1+sqrt(2))/2)² P`, approximately `1.4571067811865475 P`.

**Required resolution:** basis-independent derivatives of support/projector/factor-derived quantities, with fixed-rank tests that include repeated null and positive spectra. Perturbing eigenvalues to make them distinct would change the geometry and is not an adequate general correction.

### O2 — High priority: Kendall-shape gradients and mean fitting

Location: `geojax/geometry/shape.py`, particularly `_alignment`, `_vertical_generator`, and `squared_dist`.

The reproducer constructs a centered, full-rank, unit pre-shape `X` with `X.T @ X = 0.5 I`, and a unit horizontal tangent `U`. For the nearby regular curve `Y(t)=cos(t)X+sin(t)U`, the optimal alignment is unique and `d(X,Y(t))²=t²` locally.

At `t=0.1`, the derivative must be **0.2**, but GeoJAX returns **NaN**. Fitting the mean of `X` and `Y(0.1)` also raises a nonfinite-gradient error. The expected midpoint is `Y(0.05)`.

**Required resolution:** stable derivatives of the orientation-preserving Procrustes solution and horizontal projection on the regular stratum; smooth squared-distance treatment at coincidence. Test balanced shapes as well as generic random landmarks.

### O3 — High priority for differentiated log features: Grassmann logarithm

Location: `geojax/geometry/grassmann.py`, `_matrix_arctan_polar_single` and `Grassmann.log`.

Take `X = eye(4)[:, :2]`, `Z = eye(4)[:, 2:]`, and `Y(t)=cos(t)X+sin(t)Z`. Both principal angles equal `t`, and at `t=0.2` the points are well away from the cut locus. The exact logarithm is `tZ`, so its derivative is **Z**. GeoJAX's Jacobian is **all NaN**.

The near-zero polynomial branch does not protect a nonzero repeated singular value. The spectral branch differentiates SVD vectors even though the complete matrix function is smooth.

**Required resolution:** differentiate the matrix function as a whole, including repeated nonzero singular values. Test both forward and reverse differentiation of logarithmic features. This finding concerns log-map differentiation; it does not imply every Grassmann distance or first-order fit fails.

### O4 — Medium priority: SPD higher-order autodiff

Location: `geojax/geometry/spd.py`, the custom spectral JVPs and Sylvester derivative path.

For `P=I`, `Q=2I`, and any symmetric direction `U`, consider the ambient Hessian action of `squared_dist(P,Q)` with respect to `P`.

| Geometry | Analytic Hessian action at I | Actual |
|---|---|---|
| Log-Euclidean | `2(1+log(2)) U` | all NaN |
| Affine-invariant | `2(1+log(2)) U` | all NaN |
| Bures–Wasserstein | `U / sqrt(2)` | all NaN |

The custom first-order rules avoid eigenvector differentiation for one derivative but their own differentiation reintroduces it. These geometries already require a supplied Riemannian Hessian for solver paths that cannot construct one; that guard remains useful. It does not make nested JAX differentiation of the distance safe.

**Required resolution:** second-order-safe spectral derivatives or explicitly scoped higher-order support with analytic alternatives. Validate numerical values at repeated spectra, not merely successful first gradients.

### O5 — Medium priority: rotation-group second derivatives

Location: `geojax/geometry/lie_groups.py`, `_principal_orthogonal_log_jvp`.

For `SO(2)` with the Frobenius metric, let `Omega=[[0,-1],[1,0]]`. Then

`d(I, Exp_I(t Omega))² = 2t²`

for small `t`. At `t=0.2`, the second derivative should be **4**. The tested nested JAX differentiation raises **NotImplementedError** for derivatives of nonsymmetric eigenvectors.

**Required resolution:** a higher-order-safe matrix logarithm derivative. Merely opting into eigenvector derivatives is not sufficient evidence of correctness at repeated spectra. Check geodesic distance objectives under Newton/Hessian workflows, separately from the geometry's Hessian-conversion formula.

## Domain and statistical interpretation

**Fixed-rank Bures exponential domain.** The positive-overlap certificate is sufficient, not necessary, for remaining on the regular stratum. For rank one, `P=diag(1,0)` and `U=[[-4,1],[1,0]]` lift to the straight horizontal path `b(t)=[1-2t,t]`. Its squared norm is `5(t-0.4)²+0.2`, so it never loses rank. Its endpoint covariance is `[[1,-1],[-1,1]]`, but `exp(P,U)` returns NaN because the overlap is negative. Documentation calls the map local, so this audit records a conservative domain restriction rather than an incorrect local expression. Clarify that boundary for callers, or expand the certificate if the full exponential domain is intended.

**Squared distances versus squared norms returned by `dist`.** Use the explicit `squared_dist` method for smooth squared-distance objectives. A selected derivative for a norm at zero does not in general recover the correct second derivative of `dist(...)**2` through composition. The audit regression identity intentionally tests `squared_dist`.

**Fréchet means.** General-manifold mean fitting certifies a local stationary solution under its convergence criterion, not uniqueness or a global minimizer. Existing documentation states this. Downstream inference still requires the mathematical regularity assumptions; a solver convergence flag cannot establish them.

**FANOVA.** The independently calculated unequal-group Euclidean statistic was `9.178743834053881`; the package returned `9.17874383405389`. The chi-square survival probabilities agreed with SciPy to floating-point precision (`0.0101592372138528`). The normalized implementation agrees with the group-proportion scaling in the [Dubey–Mueller paper](https://anson.ucdavis.edu/~mueller/ro14.pdf). This check verifies the formula on a controlled case, not uniform finite-sample calibration. Degenerate variance estimates and nonunique means do not acquire an asymptotic guarantee merely because a numerical variance floor produces finite output.

**Permutation inference.** Exhaustive enumeration of all 70 labelings of a fixed eight-observation Euclidean sample produced super-uniform exact conditional upper-tail p-values for the BG and energy statistics, including ties. At nominal 5%, each rejected for 2/70 labelings. The implementation uses the appropriate Monte Carlo plus-one correction. Conditional permutation validity requires exchangeability; these checks do not establish power against every alternative. Energy distribution identification and MMD interpretation also need their metric/kernel assumptions, already distinguished in the package documentation.

**Transport.** The corrected dependency-free solver agreed with independent SciPy linear programs on 12 small randomized cases. This validates more than matching two code paths that share the same transport implementation. It is not a numerical certificate for all sizes and marginal dynamic ranges.

**Computational scale.** Several learning methods intentionally construct dense pairwise matrices; classical/spectral decompositions and Floyd–Warshall can require cubic work. JAX use alone does not make these eager Python-controlled algorithms scalable or end-to-end differentiable. No large-n or accelerator performance guarantee was established here.

## Validation record

| Check | Result |
|---|---|
| Original complete float64 suite | 938 passed, 1 failed; 748.08 seconds. Simplex overflow failure corrected. |
| Broad audit float64 suite, collected before final small fixes | 974 passed; 1161.76 seconds. |
| Broad audit float32 suite, collected before final small fixes | 972 passed, 1 skipped, 1 failed; 970.67 seconds. Sole failure: the pre-existing local-linear regression stall, subsequently corrected. |
| Final 41 audit regressions, JAX 0.11.0 float64 | 41 passed; 49.23 seconds. |
| Final audit regressions plus the entire edge-contract file, JAX 0.11.0 float32 | 49 passed; 63.19 seconds. Includes the previously failing local-linear regression test. |
| Follow-up data, transport/inference, sphere, and initial 38 audit regressions, float64 | 167 passed; 155.53 seconds. |
| JAX 0.6.0 float64: initial 38 regressions plus original simplex case | 39 passed; 51.88 seconds. |
| JAX 0.6.0 float32: initial 38 regressions plus original simplex case | 39 passed; 52.07 seconds. |
| JAX 0.6.0 final three affine-extrapolation regressions | 3 passed in each precision, in 7.93 seconds (float32) and 8.66 seconds (float64). |
| Generic first-variation identities | 18/18 geometries passed. |
| Independent transport linear programs | 12/12 costs and marginals matched. |
| Statistical formula/calibration checks | FANOVA and SciPy survival function matched; exact 70-labeling BG/energy checks passed. |
| Ruff checks, configured format checks, and diff whitespace checks | Passed. |
| Packaging | Final sdist and wheel built; strict metadata checks passed; all 66 module contents matched source; installed-wheel numerical checks passed in both precisions. |

The original float64 full suite had **938 passing tests and one failure** before fixes. The failure was the already-existing maximum-finite simplex conversion test, rather than a newly invented requirement. The float32 local-linear regression failure was subsequently reproduced in an isolated extraction of the untouched initial commit as well.

Later full-suite runs and focused follow-up runs are recorded separately because additional overflow and affine-extrapolation regressions and their fixes were added after the full runs had collected their tests. This distinction prevents an earlier snapshot from being presented as an untouched final-tree run. Coverage from a run overlapping source edits is not used as a final-tree correctness claim.

The sdist and wheel build successfully, and strict package metadata checks pass. An installed-wheel smoke test is performed outside the source tree so a successful local source import cannot disguise missing packaged files.

## Recommended next engineering work

1. Repair O1 and O2 before relying on general low-rank Bures or Kendall-shape statistical fitting.
2. Repair O3 before differentiating Grassmann logarithmic feature pipelines.
3. Complete higher-order spectral derivative support for O4/O5 before advertising broad Hessian or nested-autodiff reliability.
4. Convert each open diagnostic into a normal passing regression only after the analytic expected result is recovered. Preserve the repeated-spectrum inputs rather than perturbing away the failure.
5. Run the final full release matrix, optional transport backend, documentation examples, and accelerator checks after those changes. The existing release-source gate requires a clean annotated version tag; no release gate was bypassed in this audit.

A green test suite is evidence about its covered inputs. The open reproducers demonstrate why it would presently be misleading to turn that evidence into a blanket mathematical, statistical, or computational correctness claim.
