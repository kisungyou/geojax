"""Largest eigenvalue/eigenvector problem on the sphere.

Run from the directory that contains the ``geojax`` package folder:

    python -m geojax.examples.largest_eigenvector

Mathematical problem
--------------------
For a symmetric matrix A, the largest eigenvector solves

    maximize    x^T A x
    subject to  ||x|| = 1.

Since the steepest descent solver minimizes, we use the cost

    f(x) = -x^T A x,

on the unit sphere S^{n-1}.  Its minimizer is a dominant eigenvector of A,
and -f(x) is the Rayleigh quotient.
"""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp

from geojax.geometry import Sphere
from geojax.optimization import Minimize, SteepestDescent

jax.config.update("jax_enable_x64", True)


def make_problem(n: int, seed: int) -> tuple[jax.Array, jax.Array, Sphere, jax.Array]:
    key = jax.random.key(seed)
    key_matrix, key_init = jax.random.split(key)
    B = jax.random.normal(key_matrix, shape=(n, n))
    A = 0.5 * (B + B.T)  # Symmetrize.
    M = Sphere(size=n)  # S^{n-1} embedded in R^n.
    x0 = M.random_point(key_init)
    return A, x0, M, key


def solve_largest_eigenvector(
    n: int = 8,
    seed: int = 123,
    *,
    maxiter: int = 500,
    verbosity: int = 2,
) -> tuple[jax.Array, float, list[object], jax.Array, Sphere, jax.Array]:
    A, x0, M, _ = make_problem(n, seed)

    def cost(x: jax.Array) -> jax.Array:
        return -jnp.dot(x, A @ x)

    solver = SteepestDescent(
        tolgradnorm=1e-7,
        maxiter=maxiter,
        minstepsize=1e-14,
        verbosity=verbosity,
        ls_initial_stepsize=1.0,
    )

    problem = Minimize(M=M, cost=cost, x0=x0, solver=solver)
    x, final_cost, info = problem.solve()
    return x, final_cost, info, A, M, x0


def maybe_make_plot(
    A: jax.Array,
    x0: jax.Array,
    x: jax.Array,
    path: str,
) -> None:
    """Save a simple visualization for the 3D case."""
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"Could not import matplotlib; skipping plot. ({exc})")
        return

    if A.shape[0] != 3:
        print("Plotting is only supported for n=3.")
        return

    eigvals, eigvecs = jnp.linalg.eigh(A)
    principal = eigvecs[:, -1]
    t = jnp.linspace(0.0, 2.0 * jnp.pi, 240)
    u = jnp.linspace(0.0, jnp.pi, 120)
    xs = jnp.outer(jnp.cos(t), jnp.sin(u))
    ys = jnp.outer(jnp.sin(t), jnp.sin(u))
    zs = jnp.outer(jnp.ones_like(t), jnp.cos(u))

    fig = plt.figure(figsize=(6.8, 6.2))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(xs, ys, zs, color="#c9d7e8", alpha=0.22, linewidth=0.0, shade=False)
    ax.quiver(0, 0, 0, *principal, color="#cc4b37", linewidth=2.5, label="dominant eigenvector")
    ax.quiver(0, 0, 0, *x0, color="#3d7ea6", linewidth=2.0, label="initial point")
    ax.quiver(0, 0, 0, *x, color="#2f9e44", linewidth=2.6, label="optimizer solution")
    ax.scatter(*x0, color="#3d7ea6", s=44)
    ax.scatter(*x, color="#2f9e44", s=54)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("x3")
    ax.set_title("Largest eigenvector on the sphere")
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=26, azim=36)
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Largest eigenvector optimization on the sphere.")
    parser.add_argument("--size", type=int, default=8, help="Ambient sphere dimension n.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    parser.add_argument("--maxiter", type=int, default=500, help="Maximum number of iterations.")
    parser.add_argument("--plot", action="store_true", help="Save a figure for the n=3 case.")
    parser.add_argument("--plot-path", type=str, default="largest_eigenvector.png")
    args = parser.parse_args()

    x, final_cost, info, A, M, x0 = solve_largest_eigenvector(
        n=args.size,
        seed=args.seed,
        maxiter=args.maxiter,
        verbosity=2,
    )

    eigvals, eigvecs = jnp.linalg.eigh(A)
    true_largest_eigval = eigvals[-1]
    true_largest_eigvec = eigvecs[:, -1]

    rayleigh = jnp.dot(x, A @ x)
    alignment = jnp.abs(jnp.dot(x, true_largest_eigvec))

    print("\nSummary")
    print("-------")
    print(f"iterations:             {info[-1].iter}")
    print(f"final cost:             {final_cost:.12f}")
    print(f"Rayleigh quotient:      {rayleigh:.12f}")
    print(f"true largest eigenval:  {true_largest_eigval:.12f}")
    print(f"|<x, v_max>|:           {alignment:.12f}")
    print(f"final grad norm:        {info[-1].gradnorm:.3e}")

    if args.plot:
        maybe_make_plot(A, x0, x, args.plot_path)
        print(f"plot saved to:          {args.plot_path}")


if __name__ == "__main__":
    main()
