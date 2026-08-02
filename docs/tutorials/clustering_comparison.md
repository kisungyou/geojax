---
title: Comparing Clustering Methods on the Circle
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Comparing Clustering Methods on the Circle

Clustering manifold-valued data is not one algorithm with interchangeable
names. Intrinsic k-means minimizes squared distances to Fréchet centers;
k-medoids restricts representatives to observed points; hierarchical and
spectral methods use only the distance matrix; mean shift searches for density
modes; and competitive quantization updates prototypes online. Their outputs
can differ even when every method uses the same geodesic distance.

We compare these objectives on three noisy arcs of $S^1$. The experiment also
constructs a lightweight coreset, a weighted sample intended to approximate
the full k-means objective. The methods follow the classical constructions of
{cite:t}`lloyd1982least`, {cite:t}`kaufman1990finding`,
{cite:t}`zelnik2005self`, {cite:t}`comaniciu2002mean`, and the Riemannian
quantization method of {cite:t}`lebrigant2019quantization`. The coreset
sampling heuristic is adapted from {cite:t}`bachem2018scalable`.

```{code-cell} python
from dataclasses import replace
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from geojax.geometry import Sphere
from geojax.learning import (
    agglomerative_clustering,
    competitive_quantization,
    kmeans,
    kmedoids,
    lightweight_coreset,
    mean_shift,
    pairwise_distances,
    spectral_clustering,
)

plt.rcParams.update({
    "figure.dpi": 220,
    "savefig.dpi": 320,
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

M = Sphere(size=2)
cluster_angles = jnp.array([-2.10, 0.02, 2.08])
cluster_scales = jnp.array([0.16, 0.24, 0.18])
keys = jax.random.split(jax.random.key(831), 3)
angles = jnp.concatenate([
    center + scale * jax.random.normal(key, (12,))
    for center, scale, key in zip(cluster_angles, cluster_scales, keys)
])
points = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)

full_kmeans = kmeans(
    M, points, n_clusters=3, key=jax.random.key(1), n_init=2,
    maxiter=35, center_maxiter=25,
)
medoids = kmedoids(
    M, points, n_clusters=3, key=jax.random.key(2), maxiter=35
)
hierarchy = agglomerative_clustering(
    M, points, n_clusters=3, linkage="average"
)
spectral = spectral_clustering(
    M,
    points,
    n_clusters=3,
    key=jax.random.key(3),
    affinity="self_tuning",
    n_neighbors=5,
    maxiter=40,
)
modes = mean_shift(
    M, points, bandwidth=0.34, merge_tol=0.42, maxiter=40
)
quantized = competitive_quantization(
    M,
    points,
    n_clusters=3,
    key=jax.random.key(4),
    epochs=10,
    initial_gain=0.45,
)

coreset = lightweight_coreset(
    M, points, size=15, key=jax.random.key(5)
)
coreset_kmeans = kmeans(
    M,
    coreset.points,
    n_clusters=3,
    sample_weight=coreset.weights,
    key=jax.random.key(6),
    n_init=2,
    maxiter=35,
    center_maxiter=25,
)
coreset_labels = jnp.argmin(
    pairwise_distances(M, points, coreset_kmeans.centers), axis=1
)
coreset_kmeans = replace(coreset_kmeans, labels=coreset_labels)

results = {
    "Intrinsic k-means": full_kmeans,
    "k-medoids": medoids,
    "Average linkage": hierarchy,
    "Spectral": spectral,
    "Mean shift": modes,
    "Competitive quantization": quantized,
    "Coreset k-means": coreset_kmeans,
}

for name, result in results.items():
    n_found = int(jnp.unique(result.labels).size)
    print(
        f"{name:25s} clusters={n_found:2d}  "
        f"iterations={result.iterations:3d}  objective={float(result.objective):.5f}"
    )
```

The reported objectives are method-specific: a linkage sum is not directly
comparable with squared quantization error or a medoid distance. The visual
partitions and representative points are the meaningful comparison here.

## Visual report

The first panel shows which observations were sampled by the coreset; larger
rings indicate repeated selections. Each remaining panel displays one fitted
partition. Stars mark representatives when the method returns intrinsic
centers or medoids.

```{code-cell} python
palette = np.array(["#E45756", "#009E8E", "#7C3AED", "#F59E0B", "#2563EB"])
circle = np.linspace(-np.pi, np.pi, 500)


def setup_circle(axis, title):
    axis.plot(np.cos(circle), np.sin(circle), color="0.82", linewidth=1.0)
    axis.set(aspect="equal", xlim=(-1.12, 1.12), ylim=(-1.12, 1.12), title=title)
    axis.set_xticks([])
    axis.set_yticks([])


def draw_partition(axis, result, title):
    setup_circle(axis, f"{title}\n$k={len(np.unique(np.asarray(result.labels)))}$")
    labels = np.asarray(result.labels)
    axis.scatter(
        np.asarray(points[:, 0]), np.asarray(points[:, 1]),
        c=palette[labels % len(palette)], s=30, edgecolor="white", linewidth=0.35,
    )
    centers = getattr(result, "centers", None)
    if centers is not None:
        centers = np.asarray(centers)
        axis.scatter(
            centers[:, 0], centers[:, 1], marker="*", s=165,
            color="#111827", edgecolor="white", linewidth=0.6, zorder=5,
        )


fig, axes = plt.subplots(2, 4, figsize=(12.8, 6.5), constrained_layout=True)
axes = axes.ravel()

setup_circle(axes[0], "Lightweight coreset")
axes[0].scatter(
    np.asarray(points[:, 0]), np.asarray(points[:, 1]),
    color="#94A3B8", s=22, alpha=0.55,
)
unique_indices, selection_counts = np.unique(
    np.asarray(coreset.indices), return_counts=True
)
axes[0].scatter(
    np.asarray(points[unique_indices, 0]), np.asarray(points[unique_indices, 1]),
    facecolors="none", edgecolors="#E45756",
    s=45 + 30 * selection_counts, linewidth=1.3,
)

for axis, (name, result) in zip(axes[1:], results.items()):
    draw_partition(axis, result, name)

output = next(
    path for path in (
        Path("../_static/tutorials/clustering-comparison.png"),
        Path("docs/_static/tutorials/clustering-comparison.png"),
        Path("_static/tutorials/clustering-comparison.png"),
    )
    if path.parent.exists()
)
output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output, bbox_inches="tight")
plt.show()
```

K-means and competitive quantization require exact logarithmic and exponential
maps because they update manifold centers. Medoids, hierarchy, and spectral
clustering can operate with exact distances alone. This capability difference
matters when the chosen geometry exposes only a retraction or a numerical-local
logarithm: GeoJAX rejects unsupported combinations explicitly.

## References

```{bibliography}
:filter: docname in docnames
```
