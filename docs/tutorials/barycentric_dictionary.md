---
title: Barycentric Dictionary Learning on the Circle
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Barycentric Dictionary Learning on the Circle

A Euclidean dictionary reconstructs data by linear combinations of atoms.
There is no global addition operation on a curved manifold. Intrinsic
barycentric coding instead asks for simplex weights $w\in\Delta^{m-1}$ that
approximately satisfy the Karcher equation at an observation $x$:

$$
\sum_{j=1}^m w_j\operatorname{Log}_x(D_j)\approx 0.
$$

GeoJAX minimizes the squared norm of this residual plus an $\ell_2$ ridge
term, then reconstructs $x$ as the weighted Fréchet mean of the atoms.
Dictionary learning alternates code estimation and a Product-manifold atom
optimization {cite:p}`ho2013dictionary`.

```{code-cell} python
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from geojax.geometry import Sphere
from geojax.learning import (
    geodesic_barycentric_coding,
    manifold_dictionary_learning,
)

plt.rcParams.update({
    "figure.dpi": 220,
    "savefig.dpi": 320,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

M = Sphere(size=2)
latent = jnp.linspace(-1.15, 1.15, 21)
angles = latent + 0.045 * jax.random.normal(jax.random.key(530), latent.shape)

def on_circle(theta):
    return jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=-1)

data = on_circle(angles)
initial_atoms = on_circle(jnp.array([-0.95, 0.12, 0.95]))

initial_codes = geodesic_barycentric_coding(
    M,
    data,
    initial_atoms,
    ridge=1e-5,
    maxiter=100,
    reconstruction_maxiter=40,
)
learned = manifold_dictionary_learning(
    M,
    data,
    n_atoms=3,
    initial_atoms=initial_atoms,
    ridge=1e-5,
    maxiter=4,
    coding_maxiter=80,
    center_maxiter=35,
    tol=1e-5,
)

print(f"initial reconstruction objective: {float(initial_codes.objective):.7f}")
print(f"learned reconstruction objective: {float(learned.objective):.7f}")
print("maximum simplex-sum error:", float(jnp.max(jnp.abs(jnp.sum(learned.codes, axis=1) - 1.0))))
```

## Inspect the representation

The codes are nonnegative and sum to one. They are barycentric, not lasso-
sparse: the $\ell_1$ norm is constant over the simplex. A row can nevertheless
concentrate on one or two atoms when that best satisfies the intrinsic
residual equation.

```{code-cell} python
initial_errors = M.dist(initial_codes.reconstructions, data)
learned_errors = M.dist(learned.reconstructions, data)
history = np.asarray(learned.diagnostics["objective_history"])

fig, axes = plt.subplots(1, 3, figsize=(13.3, 4.0), constrained_layout=True)
circle = np.linspace(-np.pi, np.pi, 600)
axes[0].plot(np.cos(circle), np.sin(circle), color="0.82", linewidth=1.1)
axes[0].scatter(
    np.asarray(data[:, 0]), np.asarray(data[:, 1]),
    c=np.asarray(latent), cmap="viridis", s=34, edgecolor="white", linewidth=0.35,
    label="observations",
)
axes[0].scatter(
    np.asarray(initial_atoms[:, 0]), np.asarray(initial_atoms[:, 1]),
    marker="x", color="#E45756", s=75, linewidth=2.0, label="initial atoms",
)
axes[0].scatter(
    np.asarray(learned.atoms[:, 0]), np.asarray(learned.atoms[:, 1]),
    marker="*", color="#009E8E", edgecolor="white", linewidth=0.5,
    s=210, label="learned atoms",
)
axes[0].set(aspect="equal", xlim=(-1.1, 1.1), ylim=(-1.1, 1.1), title="Intrinsic dictionary atoms")
axes[0].set_xticks([])
axes[0].set_yticks([])
axes[0].legend(frameon=False, fontsize=8, loc="center")

image = axes[1].imshow(
    np.asarray(learned.codes).T,
    aspect="auto", origin="lower", cmap="magma", interpolation="nearest",
    extent=(float(latent[0]), float(latent[-1]), -0.5, 2.5),
)
axes[1].set(
    title="Barycentric code matrix",
    xlabel="latent position of observation",
    ylabel="atom",
    yticks=[0, 1, 2],
)
fig.colorbar(image, ax=axes[1], shrink=0.78, label="weight")

axes[2].plot(
    np.asarray(latent), np.asarray(initial_errors),
    color="#E45756", linestyle="--", linewidth=1.8, label="initial dictionary",
)
axes[2].plot(
    np.asarray(latent), np.asarray(learned_errors),
    color="#009E8E", linewidth=2.0, label="learned dictionary",
)
if history.size > 1:
    inset = axes[2].inset_axes([0.56, 0.53, 0.4, 0.4])
    inset.plot(np.arange(history.size), history, marker="o", color="#7C3AED")
    inset.set_title("objective", fontsize=8)
    inset.tick_params(labelsize=7)
axes[2].set(
    title="Geodesic reconstruction errors",
    xlabel="latent position",
    ylabel="distance",
)
axes[2].grid(alpha=0.18)
axes[2].legend(frameon=False, fontsize=8, loc="upper left")

output = next(
    path for path in (
        Path("../_static/tutorials/barycentric-dictionary.png"),
        Path("docs/_static/tutorials/barycentric-dictionary.png"),
        Path("_static/tutorials/barycentric-dictionary.png"),
    )
    if path.parent.exists()
)
output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output, bbox_inches="tight")
plt.show()
```

The coding equation is local to each observation and is undefined where the
selected logarithm is undefined. Dictionary learning is therefore most
reliable when observations and atoms remain in a common regular geodesic
region.

## References

```{bibliography}
:filter: docname in docnames
```
