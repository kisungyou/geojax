---
title: Supervised Metric Learning on the Sphere
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Supervised Metric Learning on the Sphere

An intrinsic distance is fixed by a geometry, but a supervised task may value
embedded directions differently. Riemannian manifold metric learning (RMML)
starts from an equivariant embedding $\phi(x)$ and learns a positive-definite
matrix $A$ so that

$$
d_A(x,y)^2
=\bigl(\phi(x)-\phi(y)\bigr)^\top
A\bigl(\phi(x)-\phi(y)\bigr).
$$

GeoJAX uses the regularized closed-form construction of
{cite:t}`zhu2018generalized`. The learned distance is task-specific and should
not be confused with the sphere's geodesic metric. Here `SphereExtrinsic`
provides the identity equivariant embedding into $\mathbb R^3$.

```{code-cell} python
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from geojax.geometry import SphereExtrinsic
from geojax.learning import (
    classical_mds,
    pairwise_distances,
    riemannian_metric_learning,
)

plt.rcParams.update({
    "figure.dpi": 220,
    "savefig.dpi": 320,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

M = SphereExtrinsic(size=3)
keys = jax.random.split(jax.random.key(510), 6)
n_per_class = 28

def make_class(sign, key_offset):
    signal = sign * 0.24 + 0.07 * jax.random.normal(keys[key_offset], (n_per_class,))
    nuisance = 0.72 * jax.random.normal(keys[key_offset + 1], (n_per_class,))
    height = 1.0 + 0.08 * jax.random.normal(keys[key_offset + 2], (n_per_class,))
    ambient = jnp.stack([signal, nuisance, height], axis=-1)
    return ambient / jnp.linalg.norm(ambient, axis=1, keepdims=True)

x0 = make_class(-1.0, 0)
x1 = make_class(1.0, 3)
points = jnp.concatenate([x0, x1], axis=0)
labels = jnp.concatenate([
    jnp.zeros(n_per_class, dtype=int),
    jnp.ones(n_per_class, dtype=int),
])

model = riemannian_metric_learning(
    M,
    points,
    labels,
    regularization=0.25,
    balance=0.5,
)

intrinsic_distances = pairwise_distances(M, points)
learned_distances = model.pairwise_distances(points)

def leave_one_out_accuracy(distances):
    masked = distances.at[jnp.diag_indices(len(points))].set(jnp.inf)
    neighbor = jnp.argmin(masked, axis=1)
    return jnp.mean(labels[neighbor] == labels)

baseline_accuracy = leave_one_out_accuracy(intrinsic_distances)
learned_accuracy = leave_one_out_accuracy(learned_distances)

print("metric eigenvalues:", np.round(np.linalg.eigvalsh(np.asarray(model.metric)), 4))
print(f"intrinsic 1-NN accuracy: {float(baseline_accuracy):.3f}")
print(f"learned   1-NN accuracy: {float(learned_accuracy):.3f}")
print("similar pairs:", model.diagnostics["similar_pairs"])
print("dissimilar pairs:", model.diagnostics["dissimilar_pairs"])
```

Leave-one-out nearest-neighbor accuracy is used only as an interpretable
diagnostic. It is evaluated on the same labeled observations used to learn
$A$, so it is not an estimate of generalization performance.

## Compare intrinsic and learned representations

Classical MDS visualizes the original geodesic distances. For the learned
metric, `model.transform` maps observations through a Cholesky factor of $A$;
we center those coordinates and retain their leading two singular directions
for display.

```{code-cell} python
intrinsic_embedding = classical_mds(M, points, n_components=2)
transformed = model.transform(points)
centered = transformed - jnp.mean(transformed, axis=0, keepdims=True)
left_vectors, singular_values, _ = jnp.linalg.svd(centered, full_matrices=False)
learned_embedding = left_vectors[:, :2] * singular_values[:2]

same_class = labels[:, None] == labels[None, :]
upper = jnp.triu(jnp.ones((len(points), len(points)), dtype=bool), k=1)
within_mask = upper & same_class
between_mask = upper & ~same_class

distance_sets = {
    "intrinsic within": np.asarray(intrinsic_distances[within_mask]),
    "intrinsic between": np.asarray(intrinsic_distances[between_mask]),
    "learned within": np.asarray(learned_distances[within_mask]),
    "learned between": np.asarray(learned_distances[between_mask]),
}
```

## Visual report

The nuisance direction creates substantial within-class geodesic spread. RMML
does not move the spherical observations; it changes their supervised embedded
comparison by contracting similar-pair scatter and emphasizing dissimilar-pair
scatter.

```{code-cell} python
fig = plt.figure(figsize=(14.0, 3.9), constrained_layout=True)
colors = np.array(["#2563EB", "#E45756"])

axis3d = fig.add_subplot(1, 4, 1, projection="3d")
u = np.linspace(0, 2 * np.pi, 32)
v = np.linspace(0, np.pi, 18)
axis3d.plot_wireframe(
    np.outer(np.cos(u), np.sin(v)),
    np.outer(np.sin(u), np.sin(v)),
    np.outer(np.ones_like(u), np.cos(v)),
    color="0.84", linewidth=0.3, rstride=3, cstride=3,
)
for label in (0, 1):
    selected = np.asarray(labels == label)
    axis3d.scatter(
        *np.asarray(points[selected]).T,
        color=colors[label], s=25, depthshade=False, label=f"class {label}",
    )
axis3d.set(title="Observations on $S^2$", xlabel="$x_1$", ylabel="$x_2$", zlabel="$x_3$")
axis3d.set_box_aspect((1, 1, 1))
axis3d.view_init(elev=18, azim=38)
axis3d.legend(frameon=False)

for position, coordinates, title in (
    (2, intrinsic_embedding.coordinates, "Intrinsic metric MDS"),
    (3, learned_embedding, "Learned embedded metric"),
):
    axis = fig.add_subplot(1, 4, position)
    coordinates = np.asarray(coordinates)
    for label in (0, 1):
        selected = np.asarray(labels == label)
        axis.scatter(
            coordinates[selected, 0], coordinates[selected, 1],
            color=colors[label], s=28, edgecolor="white", linewidth=0.35,
        )
    axis.set(title=title, xlabel="coordinate 1", ylabel="coordinate 2")
    axis.grid(alpha=0.18)

distribution_axis = fig.add_subplot(1, 4, 4)
bins = np.linspace(
    0.0,
    max(np.max(values) for values in distance_sets.values()),
    22,
)
styles = {
    "intrinsic within": ("#64748B", "--"),
    "intrinsic between": ("#111827", "--"),
    "learned within": ("#009E8E", "-"),
    "learned between": ("#7C3AED", "-"),
}
for name, values in distance_sets.items():
    color, linestyle = styles[name]
    histogram, edges = np.histogram(values, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    distribution_axis.plot(centers, histogram, color=color, linestyle=linestyle, label=name)
distribution_axis.set(title="Pair-distance distributions", xlabel="distance", ylabel="density")
distribution_axis.grid(alpha=0.18)
distribution_axis.legend(frameon=False, fontsize=8)

output = next(
    path for path in (
        Path("../_static/tutorials/spherical-metric-learning.png"),
        Path("docs/_static/tutorials/spherical-metric-learning.png"),
        Path("_static/tutorials/spherical-metric-learning.png"),
    )
    if path.parent.exists()
)
output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output, bbox_inches="tight")
plt.show()
```

RMML needs an equivariant embedding, not merely a distance function. When a
geometry does not provide one, `riemannian_metric_learning` requires an
explicit callable. Product geometries are supported when every factor has a
compatible embedding.

## References

```{bibliography}
:filter: docname in docnames
```
