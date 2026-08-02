---
title: Comparing Manifold Embeddings
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Comparing Manifold Embeddings

Dimension-reduction methods preserve different structures. Classical MDS and
Sammon mapping target pairwise distances with different stress functions;
kernel PCA diagonalizes a distance kernel; Isomap replaces direct distances by
shortest paths in a neighbor graph; t-SNE emphasizes local probability
neighborhoods; and PHATE embeds diffusion-potential distances. GeoJAX builds
all six from the same validated manifold dataset.

We use a winding trajectory on $S^2$. MDS is the global metric baseline,
Sammon reweights stress toward nearby pairs {cite:p}`sammon1969nonlinear`,
Isomap follows the sampled one-dimensional path {cite:p}`tenenbaum2000global`,
t-SNE emphasizes probabilistic neighborhoods
{cite:p}`vandermaaten2008visualizing`, and PHATE emphasizes multiscale
transitions {cite:p}`moon2019phate`.

```{code-cell} python
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from geojax.geometry import Sphere
from geojax.learning import (
    classical_mds,
    isomap,
    kernel_pca,
    phate,
    sammon_mapping,
    tsne,
)

plt.rcParams.update({
    "figure.dpi": 220,
    "savefig.dpi": 320,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

M = Sphere(3)
parameter = jnp.linspace(-1.0, 1.0, 46)
longitude = 2.35 * parameter
latitude = 0.58 * jnp.sin(2.2 * parameter)
points = jnp.stack([
    jnp.cos(latitude) * jnp.cos(longitude),
    jnp.cos(latitude) * jnp.sin(longitude),
    jnp.sin(latitude),
], axis=-1)

mds = classical_mds(M, points, n_components=2)
kpca = kernel_pca(M, points, n_components=2)
iso = isomap(M, points, n_components=2, n_neighbors=5, mutual=False)
sammon = sammon_mapping(M, points, n_components=2, maxiter=140, tol=1e-6)
stochastic = tsne(
    M,
    points,
    n_components=2,
    perplexity=8.0,
    key=jax.random.key(72),
    maxiter=400,
    learning_rate=35.0,
    early_exaggeration=4.0,
    exaggeration_iterations=100,
)
diffusion = phate(
    M, points, n_components=2, n_neighbors=5,
    diffusion_time=5, max_diffusion_time=10,
)

results = {
    "Classical MDS": mds,
    "Kernel PCA": kpca,
    "Isomap": iso,
    "Sammon mapping": sammon,
    "t-SNE": stochastic,
    "PHATE": diffusion,
}

for name, result in results.items():
    print(f"{name:16s} objective={float(result.objective):.5f}  reason={result.reason}")
```

## Visual report

Coordinates can rotate, reflect, or rescale without changing the scientific
content. We therefore compare continuity and ordering visually rather than
matching raw axes. Color records the hidden trajectory parameter.

```{code-cell} python
fig = plt.figure(figsize=(14.8, 7.2), constrained_layout=True)
axis3d = fig.add_subplot(2, 4, 1, projection="3d")
u = np.linspace(0, 2 * np.pi, 36)
v = np.linspace(0, np.pi, 20)
sx = np.outer(np.cos(u), np.sin(v))
sy = np.outer(np.sin(u), np.sin(v))
sz = np.outer(np.ones_like(u), np.cos(v))
axis3d.plot_wireframe(sx, sy, sz, color="0.82", linewidth=0.35, rstride=3, cstride=3)
axis3d.scatter(*np.asarray(points).T, c=np.asarray(parameter), cmap="viridis", s=24, depthshade=False)
axis3d.set(title="Data on $S^2$", xlabel="$x_1$", ylabel="$x_2$", zlabel="$x_3$")
axis3d.set_box_aspect((1, 1, 1))
axis3d.view_init(elev=20, azim=35)

for position, (name, result) in enumerate(results.items(), start=2):
    axis = fig.add_subplot(2, 4, position)
    coordinates = np.asarray(result.coordinates)
    axis.plot(coordinates[:, 0], coordinates[:, 1], color="0.80", linewidth=1)
    axis.scatter(coordinates[:, 0], coordinates[:, 1], c=np.asarray(parameter), cmap="viridis", s=27, edgecolor="white", linewidth=0.25)
    axis.set(title=name, xlabel="coordinate 1", ylabel="coordinate 2")
    axis.grid(alpha=0.17)

legend_axis = fig.add_subplot(2, 4, 8)
legend_axis.set_axis_off()
gradient = np.linspace(float(parameter.min()), float(parameter.max()), 256)[:, None]
legend_axis.imshow(gradient, cmap="viridis", aspect="auto", extent=(0.15, 0.35, -1, 1))
legend_axis.text(0.42, 0.96, "trajectory parameter", va="top", fontsize=11)
legend_axis.text(0.42, 0.70, "yellow: path end", color="#374151")
legend_axis.text(0.42, -0.70, "purple: path start", color="#374151")
legend_axis.set(xlim=(0, 1), ylim=(-1.1, 1.1))

output = next(
    path for path in (
        Path("../_static/tutorials/dimension-reduction.png"),
        Path("docs/_static/tutorials/dimension-reduction.png"),
        Path("_static/tutorials/dimension-reduction.png"),
    )
    if path.parent.exists()
)
output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output, bbox_inches="tight")
plt.show()
```

The implementations are dense: storing pairwise distances costs $O(n^2)$ and
Floyd--Warshall Isomap costs $O(n^3)$ work. Sammon and t-SNE add iterative
optimization, while the eigendecomposition and diffusion methods are
deterministic for fixed input. Raw axes are not compared across panels because
all embeddings are identifiable only up to transformations appropriate to
their objective.

## References

```{bibliography}
:filter: docname in docnames
```
