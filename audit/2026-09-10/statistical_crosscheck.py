"""Independent Euclidean FANOVA calculation and exact permutation calibration."""

from __future__ import annotations

import itertools
import json

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import chi2

from geojax.geometry import Euclidean
from geojax.learning import frechet_anova
from geojax.learning._inference import _bg_statistic
from geojax.learning._uncertainty import _energy_statistic

jax.config.update("jax_enable_x64", True)
rng = np.random.default_rng(260910)
samples = [
    rng.normal(loc, scale, size=(n, 2))
    for n, loc, scale in [(9, 0.0, 0.8), (13, 0.5, 1.3), (17, -0.2, 0.9)]
]
sizes = np.array([len(group) for group in samples])
pooled = np.concatenate(samples)
labels = np.repeat(np.arange(3), sizes)
proportions = sizes / sizes.sum()
within_squared = [np.sum((group - group.mean(axis=0)) ** 2, axis=1) for group in samples]
variances = np.array([s.mean() for s in within_squared])
sigma_squared = np.array([np.mean((s - s.mean()) ** 2) for s in within_squared])
pooled_variance = np.sum((pooled - pooled.mean(axis=0)) ** 2, axis=1).mean()
f_component = pooled_variance - proportions @ variances
u_component = sum(
    proportions[i]
    * proportions[j]
    * (variances[i] - variances[j]) ** 2
    / (sigma_squared[i] * sigma_squared[j])
    for i, j in itertools.combinations(range(3), 2)
)
reference = sizes.sum() * (
    u_component / np.sum(proportions / sigma_squared)
    + f_component**2 / np.sum(proportions**2 * sigma_squared)
)
fanova = frechet_anova(Euclidean(2), pooled, labels, tol=1e-10)
np.testing.assert_allclose(fanova.statistic, reference, rtol=1e-8)
np.testing.assert_allclose(fanova.pvalue, chi2.sf(reference, 2), rtol=1e-8)

# Conditional null distribution: all C(8,4)=70 equally likely labelings of
# a fixed dataset. For each possible observed labeling, exact upper-tail
# permutation p-values must be super-uniform, including tied statistics.
values = jnp.asarray([-2.0, -1.0, -0.5, -0.5, 0.0, 0.4, 1.0, 3.0])
distances = jnp.abs(values[:, None] - values[None, :])
left_indices = list(itertools.combinations(range(8), 4))
calibration = []
for name, statistic in [("Biswas-Ghosh", _bg_statistic), ("energy", _energy_statistic)]:
    statistics = []
    for left in left_indices:
        right = tuple(index for index in range(8) if index not in left)
        statistics.append(float(statistic(distances, jnp.asarray(left), jnp.asarray(right))))
    statistics = np.asarray(statistics)
    pvalues = np.mean(statistics[None, :] >= statistics[:, None], axis=1)
    rejection = {str(alpha): float(np.mean(pvalues <= alpha)) for alpha in [0.01, 0.05, 0.1, 0.2]}
    assert all(rate <= float(alpha) + 1e-14 for alpha, rate in rejection.items())
    calibration.append(
        {
            "statistic": name,
            "partitions": len(statistics),
            "exact_conditional_rejection_rates": rejection,
        }
    )
print(
    json.dumps(
        {
            "fanova": {
                "package": float(fanova.statistic),
                "independent_reference": reference,
                "package_pvalue": float(fanova.pvalue),
                "scipy_pvalue": float(chi2.sf(reference, 2)),
            },
            "permutation_checks": calibration,
        },
        indent=2,
    )
)
