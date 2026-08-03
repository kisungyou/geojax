---
title: Comparing Clustering Methods on a Flat Torus
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Comparing clustering methods on a flat torus

Clustering manifold-valued data is not one algorithm with interchangeable
names. Intrinsic k-means minimizes squared distances to Frechet centers;
k-medoids restricts representatives to observed points; hierarchical and
spectral methods use only the distance matrix; mean shift searches for density
modes; and competitive quantization updates prototypes online. Their outputs
can differ even when every method uses the same geodesic distance.

We compare these objectives on the flat two-torus
$T^2=(\mathbb R/2\pi\mathbb Z)^2$. GeoJAX represents a point by wrapped angles
$\theta=(\theta_1,\theta_2)\in[-\pi,\pi)^2$, with

$$
\operatorname{Log}_{\theta}(\varphi)
=\operatorname{wrap}(\varphi-\theta),
\qquad
d(\theta,\varphi)
=\left\|\operatorname{wrap}(\varphi-\theta)\right\|_2.
$$

Thus opposite edges of the displayed square are identified. Several groups
below cross those edges, making ordinary Euclidean clustering of the angle
table inappropriate. The methods follow {cite:t}`lloyd1982least`,
{cite:t}`kaufman1990finding`, {cite:t}`zelnik2005self`,
{cite:t}`comaniciu2002mean`, and the Riemannian quantization method of
{cite:t}`lebrigant2019quantization`. The coreset heuristic is adapted from
{cite:t}`bachem2018scalable`.

```{code-cell} python
from dataclasses import replace
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from geojax.geometry import Torus
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

M = Torus(size=2)
cluster_centers = jnp.array([
    [-3.05, 1.80],
    [0.05, -0.18],
    [1.80, 3.02],
])
cluster_scales = jnp.array([
    [0.18, 0.16],
    [0.23, 0.18],
    [0.16, 0.19],
])
keys = jax.random.split(jax.random.key(831), 3)
points = M.project(jnp.concatenate([
    center + scale * jax.random.normal(key, shape=(14, 2))
    for center, scale, key in zip(cluster_centers, cluster_scales, keys)
]))

full_kmeans = kmeans(
    M, points, n_clusters=3, key=jax.random.key(1), n_init=3,
    maxiter=35, center_maxiter=25,
)
medoids = kmedoids(
    M, points, n_clusters=3, key=jax.random.key(2), maxiter=35,
)
hierarchy = agglomerative_clustering(
    M, points, n_clusters=3, linkage="average",
)
spectral = spectral_clustering(
    M,
    points,
    n_clusters=3,
    key=jax.random.key(3),
    affinity="self_tuning",
    n_neighbors=6,
    maxiter=40,
)
modes = mean_shift(
    M, points, bandwidth=0.42, merge_tol=0.50, maxiter=40,
)
quantized = competitive_quantization(
    M,
    points,
    n_clusters=3,
    key=jax.random.key(4),
    epochs=12,
    initial_gain=0.45,
)

coreset = lightweight_coreset(
    M, points, size=16, key=jax.random.key(5),
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
    pairwise_distances(M, points, coreset_kmeans.centers), axis=1,
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

The dashed boundary is a coordinate cut, not a geometric boundary. A cluster
split between the top and bottom or left and right edges remains contiguous on
$T^2$. The first panel shows the coreset; larger rings indicate repeated
selections. Stars mark representatives when a method returns centers or
medoids.

```{code-cell} python
palette = np.array(["#E45756", "#009E8E", "#7C3AED", "#F59E0B", "#2563EB"])


def setup_torus_chart(axis, title):
    axis.set(
        aspect="equal",
        xlim=(-np.pi - 0.18, np.pi + 0.18),
        ylim=(-np.pi - 0.18, np.pi + 0.18),
        title=title,
        xlabel=r"$\theta_1$",
        ylabel=r"$\theta_2$",
        xticks=[-np.pi, 0.0, np.pi],
        yticks=[-np.pi, 0.0, np.pi],
        xticklabels=[r"$-\pi$", "$0$", r"$\pi$"],
        yticklabels=[r"$-\pi$", "$0$", r"$\pi$"],
    )
    axis.plot(
        [-np.pi, np.pi, np.pi, -np.pi, -np.pi],
        [-np.pi, -np.pi, np.pi, np.pi, -np.pi],
        color="#475569", linestyle="--", linewidth=1.0,
    )
    axis.grid(alpha=0.14)


def draw_partition(axis, result, title):
    labels = np.asarray(result.labels)
    setup_torus_chart(
        axis, f"{title}\n$k={len(np.unique(labels))}$",
    )
    axis.scatter(
        np.asarray(points[:, 0]), np.asarray(points[:, 1]),
        c=palette[labels % len(palette)], s=31,
        edgecolor="white", linewidth=0.35,
    )
    centers = getattr(result, "centers", None)
    if centers is not None:
        centers = np.asarray(centers)
        axis.scatter(
            centers[:, 0], centers[:, 1], marker="*", s=165,
            color="#111827", edgecolor="white", linewidth=0.6, zorder=5,
        )


fig, axes = plt.subplots(2, 4, figsize=(13.0, 7.0), constrained_layout=True)
axes = axes.ravel()

setup_torus_chart(axes[0], "Lightweight coreset")
axes[0].scatter(
    np.asarray(points[:, 0]), np.asarray(points[:, 1]),
    color="#94A3B8", s=22, alpha=0.55,
)
unique_indices, selection_counts = np.unique(
    np.asarray(coreset.indices), return_counts=True,
)
axes[0].scatter(
    np.asarray(points[unique_indices, 0]),
    np.asarray(points[unique_indices, 1]),
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

K-means, mean shift, and competitive quantization require logarithmic and
exponential maps because they update manifold representatives. Medoids,
hierarchy, and spectral clustering need only exact distances. The torus makes
the distinction visible: methods must respect edge identification either
through the intrinsic updates or through their distance matrix.

## References

```{bibliography}
:filter: docname in docnames
```
