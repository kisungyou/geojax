---
title: Hand Poses in Kendall Shape Space
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Hand Poses in Kendall Shape Space

This tutorial analyzes real skeletal hand poses using Kendall shape space. A
hand is represented by $22$ joints in $\mathbb R^3$. We want to compare the
shape of a pose without letting its location, physical scale, or global
orientation dominate the result. Kendall's quotient construction supplies the
statistical shape space used here {cite:p}`kendall1984shape`.

The data are a 52-pose subset of the SHREC'17 3D hand-gesture dataset,
distributed by the [Geomstats dataset
module](https://github.com/geomstats/geomstats/tree/master/geomstats/datasets/data/hands).
There are 25 **Grab** poses and 27 **Expand** poses. The source study is De
Smedt et al.'s SHREC'17 hand-gesture track
{cite:p}`desmedt2017shrec`.

GeoJAX vendors the two small source text files with this page, so the
documentation remains executable without downloading data at build time. See
the accompanying [license and provenance
notice](../_static/data/hands/NOTICE.txt) for the upstream source and
redistribution terms.

## Shape-space model

For $m$ landmarks in $\mathbb R^d$, a pre-shape is an $m\times d$ matrix $X$
satisfying

$$
\mathbf 1^\top X=0,
\qquad
\lVert X\rVert_F=1.
$$

Centering removes translation and normalization removes scale. Kendall shape
space additionally identifies $X$ and $XR$ for every $R\in SO(d)$, thereby
removing global orientation. Its distance is the spherical distance after
orientation-preserving Procrustes alignment:

$$
d([X],[Y])
=\arccos\left(\max_{R\in SO(d)}\langle X,YR\rangle_F\right).
$$

We will compute a Fréchet mean for each pose class and classify held-out hands
by their nearest intrinsic mean.

```{code-cell} python
import jax
jax.config.update("jax_enable_x64", True)

from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from geojax.geometry import KendallShape
from geojax.learning import (
    frechet_anova,
    nearest_centroid_classifier,
    pairwise_distances,
)

plt.rcParams.update({
    "figure.dpi": 200,
    "savefig.dpi": 240,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
```

## Load the landmark data

The files are the original whitespace-separated Geomstats data. Their first
rows are numeric column headers, matching the upstream loader, so we skip one
row before reshaping each pose to $22\times3$.

```{code-cell} python
candidate_directories = [
    Path("../_static/data/hands"),
    Path("docs/_static/data/hands"),
]
data_directory = next(path for path in candidate_directories if path.exists())

raw_hands = np.loadtxt(data_directory / "hands.txt", skiprows=1).reshape(-1, 22, 3)
labels = np.loadtxt(data_directory / "labels.txt", skiprows=1, dtype=int)
label_names = {0: "Grab", 1: "Expand"}

bones = np.array(
    [
        [0, 1], [0, 2], [2, 3], [3, 4], [4, 5],
        [1, 6], [6, 7], [7, 8], [8, 9],
        [1, 10], [10, 11], [11, 12], [12, 13],
        [1, 14], [14, 15], [15, 16], [16, 17],
        [1, 18], [18, 19], [19, 20], [20, 21],
    ]
)

print("landmark array:", raw_hands.shape)
for label, name in label_names.items():
    print(f"{name:6s}: {int(np.sum(labels == label))} poses")
```

## Remove translation and scale

`KendallShape.project` centers every configuration and normalizes its
Frobenius norm. Rotation is not removed by choosing one permanent alignment;
it is handled by the quotient operations whenever distances and logarithms are
evaluated.

```{code-cell} python
M = KendallShape(size=(22, 3))
hands = M.project(jnp.asarray(raw_hands))

centroid_norms = jnp.linalg.norm(jnp.mean(hands, axis=1), axis=1)
preshape_norms = jnp.linalg.norm(hands, axis=(1, 2))

print("All projected hands are regular pre-shapes:", bool(jnp.all(M.belongs(hands))))
print("Largest centroid norm:", f"{float(jnp.max(centroid_norms)):.3e}")
print("Largest unit-norm error:", f"{float(jnp.max(jnp.abs(preshape_norms - 1))):.3e}")
print("Intrinsic dimension:", M.dim)
```

## Compute intrinsic group means

At a current estimate $\mu$, the Fréchet objective and its gradient are

$$
F(\mu)=\frac1N\sum_{i=1}^N d(\mu,X_i)^2,
\qquad
\operatorname{grad}F(\mu)
=-\frac{2}{N}\sum_{i=1}^N\operatorname{Log}_{\mu}(X_i).
$$

GeoJAX's public `nearest_centroid_classifier` computes one weighted Fréchet
mean per class with deterministic medoid initialization. This avoids giving
the tutorial a second, subtly different implementation of a core learning
routine.

Every fourth observation is held out before estimating the two means. This is
a deterministic illustrative split, not a claim of benchmark performance.

```{code-cell} python
sample_ids = np.arange(len(labels))
train_mask = sample_ids % 4 != 0
test_mask = ~train_mask

classifier = nearest_centroid_classifier(
    M,
    hands[jnp.asarray(train_mask)],
    labels[train_mask],
    maxiter=60,
    tol=1e-7,
)
means = classifier.centers
mean_histories = []
for label, result in enumerate(classifier.diagnostics["class_fits"]):
    group_size = int(np.sum(train_mask & (labels == label)))
    mean_histories.append(np.asarray([
        entry.gradnorm for entry in result.diagnostics["history"]
    ]))
    print(
        f"{label_names[label]:6s}: {group_size:2d} training poses, "
        f"{result.iterations:2d} iterations, final norm "
        f"{float(result.gradient_norm):.3e} ({result.reason})"
    )

print("Distance between group means:", f"{float(M.dist(means[0], means[1])):.4f}")
```

## Classify by the nearest mean

For each hand $X$, we compute the two intrinsic distances
$d(X,\mu_{\mathrm{Grab}})$ and $d(X,\mu_{\mathrm{Expand}})$. The smaller one
determines the predicted pose.

```{code-cell} python
distances = pairwise_distances(M, hands, means)
predictions = np.asarray(classifier.predict(hands))
test_accuracy = np.mean(predictions[test_mask] == labels[test_mask])

print(f"Held-out poses: {int(np.sum(test_mask))}")
print(f"Correct predictions: {int(np.sum(predictions[test_mask] == labels[test_mask]))}")
print(f"Held-out accuracy: {test_accuracy:.1%}")
print("\nHeld-out confusion counts (truth -> prediction)")
for truth in (0, 1):
    counts = [
        int(np.sum(test_mask & (labels == truth) & (predictions == prediction)))
        for prediction in (0, 1)
    ]
    print(f"{label_names[truth]:6s}: Grab={counts[0]:2d}, Expand={counts[1]:2d}")
```

## Test the two pose populations

Classification asks whether a rule predicts held-out labels. Fréchet ANOVA asks
a different question: whether the two object-valued populations have equal
location and variation in the metric space. The test combines between-group
mean separation with differences in within-group Fréchet variance
{cite:p}`dubey2019frechet`.

```{code-cell} python
fanova = frechet_anova(
    M,
    hands,
    jnp.asarray(labels),
    method="asymptotic",
    maxiter=60,
    tol=1e-7,
)

print("Fréchet ANOVA statistic:", f"{float(fanova.statistic):.4f}")
print("asymptotic p-value:", f"{float(fanova.pvalue):.4g}")
print(
    "group Fréchet variances:",
    np.round(np.asarray(fanova.diagnostics["group_variances"]), 5),
)
```

## Visualize poses, means, and separation

For display only, the Expand mean is aligned to the Grab mean so both use a
common orientation. The distance scatter is already quotient-invariant: points
below the diagonal are closer to Grab, and points above it are closer to
Expand. Stars mark the held-out observations.

```{code-cell} python
colors = {0: "#2563EB", 1: "#D97706"}


def draw_hand(ax, hand, color, title):
    hand = np.asarray(hand)
    for start, end in bones:
        segment = hand[[start, end]]
        ax.plot(segment[:, 0], segment[:, 1], segment[:, 2], color=color, linewidth=2.0)
    ax.scatter(hand[:, 0], hand[:, 1], hand[:, 2], color=color, s=18, depthshade=False)
    center = np.mean(hand, axis=0)
    radius = 0.55 * np.max(np.ptp(hand, axis=0))
    for setter, coordinate in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center):
        setter(coordinate - radius, coordinate + radius)
    ax.set_title(title)
    ax.set_box_aspect((1, 1, 1), zoom=1.35)
    ax.set_axis_off()
    ax.view_init(elev=24, azim=-58)


grab_index = int(np.flatnonzero(labels == 0)[0])
expand_index = int(np.flatnonzero(labels == 1)[0])
grab_example = M.align(hands[grab_index], means[0])[0]
expand_example = M.align(hands[expand_index], means[1])[0]
aligned_expand_mean = M.align(means[1], means[0])[0]

fig = plt.figure(figsize=(12.0, 7.2), constrained_layout=True)
grid = fig.add_gridspec(2, 3)

draw_hand(fig.add_subplot(grid[0, 0], projection="3d"), grab_example, colors[0], "Grab example")
draw_hand(
    fig.add_subplot(grid[0, 1], projection="3d"),
    expand_example,
    colors[1],
    "Expand example",
)
draw_hand(fig.add_subplot(grid[1, 0], projection="3d"), means[0], colors[0], "Grab Fréchet mean")
draw_hand(
    fig.add_subplot(grid[1, 1], projection="3d"),
    aligned_expand_mean,
    colors[1],
    "Expand Fréchet mean",
)

distance_axis = fig.add_subplot(grid[0, 2])
distance_array = np.asarray(distances)
for label in (0, 1):
    training = train_mask & (labels == label)
    held_out = test_mask & (labels == label)
    distance_axis.scatter(
        distance_array[training, 0],
        distance_array[training, 1],
        color=colors[label],
        alpha=0.65,
        s=28,
        label=f"{label_names[label]} train",
    )
    distance_axis.scatter(
        distance_array[held_out, 0],
        distance_array[held_out, 1],
        color=colors[label],
        edgecolor="white",
        marker="*",
        s=105,
        linewidth=0.7,
        label=f"{label_names[label]} held out",
    )
limit = 1.04 * float(jnp.max(distances))
distance_axis.plot([0, limit], [0, limit], color="0.35", linestyle="--", linewidth=1.0)
distance_axis.set(
    xlim=(0, limit),
    ylim=(0, limit),
    xlabel="distance to Grab mean",
    ylabel="distance to Expand mean",
    title="Intrinsic nearest-mean rule",
)
distance_axis.set_aspect("equal")
distance_axis.grid(alpha=0.2)
distance_axis.legend(frameon=False, fontsize=9)

convergence_axis = fig.add_subplot(grid[1, 2])
for label, history in enumerate(mean_histories):
    convergence_axis.semilogy(
        np.arange(len(history)),
        history,
        marker="o",
        color=colors[label],
        label=label_names[label],
    )
convergence_axis.set(
    xlabel="solver iteration",
    ylabel="Riemannian gradient norm",
    title="Fréchet-mean convergence",
)
convergence_axis.grid(alpha=0.2)
convergence_axis.legend(frameon=False)

plt.show()
```

The group means summarize the characteristic joint configuration after
translation, scale, and orientation have been removed. The classifier is
deliberately simple: it demonstrates that intrinsic distances can expose pose
information, but a serious evaluation would use repeated subject-aware splits,
uncertainty estimates, and comparisons with tangent-space or skeletal models.

## References

```{bibliography}
:filter: docname in docnames
```
