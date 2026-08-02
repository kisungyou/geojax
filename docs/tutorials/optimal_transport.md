---
title: Exact Transport on the Circle
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Exact Transport on the Circle

Pointwise distances compare observations; optimal transport compares weighted
empirical distributions. For supports $x_i,y_j$ on a manifold and probability
weights $a_i,b_j$, GeoJAX solves

$$
W_p(\mu,\nu)^p=
\min_{\pi\mathbf 1=a,\,\pi^\top\mathbf 1=b}
\sum_{i,j}\pi_{ij}d(x_i,y_j)^p.
$$

This tutorial uses exact geodesic costs on $S^1$. The finite linear program is
solved by GeoJAX's deterministic transportation simplex. Entropic Sinkhorn
regularization is available separately through the optional OTT-JAX integration
{cite:p}`cuturi2013sinkhorn`.

```{code-cell} python
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from geojax.geometry import Sphere
from geojax.learning import empirical_wasserstein_distance

plt.rcParams.update({
    "figure.dpi": 220,
    "savefig.dpi": 320,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

M = Sphere(2)
angles_x = jnp.array([-2.75, -2.30, -1.75, -0.45, 0.20, 1.10, 2.55])
angles_y = jnp.array([-2.45, -1.15, -0.10, 0.75, 1.55, 2.20])
x = jnp.stack([jnp.cos(angles_x), jnp.sin(angles_x)], axis=-1)
y = jnp.stack([jnp.cos(angles_y), jnp.sin(angles_y)], axis=-1)
a = jnp.array([0.08, 0.17, 0.12, 0.20, 0.18, 0.15, 0.10])
b = jnp.array([0.14, 0.18, 0.20, 0.16, 0.17, 0.15])

transport = empirical_wasserstein_distance(
    M, x, y, p=2.0, weights_x=a, weights_y=b
)

print("W2 distance:", f"{float(transport.distance):.6f}")
print("simplex pivots:", transport.iterations)
print("termination:", transport.reason)
print("largest row residual:", f"{float(transport.diagnostics['row_residual']):.2e}")
print("largest column residual:", f"{float(transport.diagnostics['column_residual']):.2e}")
print("duality gap:", f"{float(transport.diagnostics['duality_gap']):.2e}")
```

## Visual report

Line opacity and width encode transported mass. The plan heat map exposes the
same coupling numerically; its row and column sums equal the two supplied
weight vectors.

```{code-cell} python
plan = np.asarray(transport.plan)
circle_angle = np.linspace(-np.pi, np.pi, 500)
circle = np.column_stack([np.cos(circle_angle), np.sin(circle_angle)])

fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4), constrained_layout=True)
axes[0].plot(circle[:, 0], circle[:, 1], color="0.78", linewidth=1.2)
for row, column in zip(*np.nonzero(plan > 1e-10)):
    mass = plan[row, column]
    axes[0].plot(
        [float(x[row, 0]), float(y[column, 0])],
        [float(x[row, 1]), float(y[column, 1])],
        color="#64748B",
        linewidth=0.7 + 15.0 * mass,
        alpha=0.25 + 2.2 * mass,
        zorder=1,
    )
axes[0].scatter(np.asarray(x[:, 0]), np.asarray(x[:, 1]), s=500 * np.asarray(a), color="#009A8E", edgecolor="white", label=r"$\mu$", zorder=3)
axes[0].scatter(np.asarray(y[:, 0]), np.asarray(y[:, 1]), s=500 * np.asarray(b), color="#FF5A5F", marker="s", edgecolor="white", label=r"$\nu$", zorder=3)
axes[0].set(aspect="equal", xlim=(-1.18, 1.18), ylim=(-1.18, 1.18), title="Optimal coupling on $S^1$")
axes[0].legend(frameon=False, loc="center")
axes[0].set_xlabel("$x_1$")
axes[0].set_ylabel("$x_2$")

image = axes[1].imshow(plan, cmap="viridis", origin="lower", aspect="auto")
axes[1].set(xlabel=r"$\nu$ support index", ylabel=r"$\mu$ support index", title="Exact transport plan")
fig.colorbar(image, ax=axes[1], shrink=0.82, label="transported mass")

output = next(
    path for path in (
        Path("../_static/tutorials/optimal-transport.png"),
        Path("docs/_static/tutorials/optimal-transport.png"),
        Path("_static/tutorials/optimal-transport.png"),
    )
    if path.parent.exists()
)
output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output, bbox_inches="tight")
plt.show()
```

The exact plan and its cluster or permutation uses are nondifferentiable by
contract. For gradient-based models, use `sinkhorn_divergence` and install the
`ot` extra.

## References

```{bibliography}
:filter: docname in docnames
```
