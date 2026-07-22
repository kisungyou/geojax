"""SPD Fréchet mean simulation with GeoJAX.

Run from the directory containing the ``geojax`` package folder:

    python -m geojax.examples.spd_frechet_mean

The experiment generates synthetic SPD matrices near a ground-truth center P0,
then estimates the Fréchet mean under two SPD geometries:

    1. SPDLogEuclidean
    2. SPDAffineInvariant

For each geometry, it compares two solvers:

    1. SteepestDescent
    2. ConjugateGradient

The optimization problem is

    minimize_P  F(P) = (1 / 2N) sum_i d(P, P_i)^2,
    subject to  P in SPD(n).

For the log-Euclidean geometry, the Fréchet mean has the closed form

    P_LE = exp( (1/N) sum_i log(P_i) ),

which is used as an implementation check.
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import time
from typing import Any, Callable

import jax
import jax.numpy as jnp

from geojax.geometry import SPDAffineInvariant, SPDLogEuclidean
from geojax.optimization import Minimize, SteepestDescent, ConjugateGradient

jax.config.update("jax_enable_x64", True)

Array = Any


def sym(A: Array) -> Array:
    """Symmetrize a matrix or batch of matrices."""
    return 0.5 * (A + jnp.swapaxes(A, -1, -2))


def sym_funcm(A: Array, f: Callable[[Array], Array], eps: float = 1e-10) -> Array:
    """Apply a scalar function to the eigenvalues of a symmetric matrix."""
    vals, vecs = jnp.linalg.eigh(sym(A))
    vals = jnp.maximum(vals, eps)
    return (vecs * f(vals)[..., None, :]) @ jnp.swapaxes(vecs, -1, -2)


def expm_sym(A: Array) -> Array:
    """Matrix exponential for symmetric matrices."""
    vals, vecs = jnp.linalg.eigh(sym(A))
    return (vecs * jnp.exp(vals)[..., None, :]) @ jnp.swapaxes(vecs, -1, -2)


def logm_spd(P: Array, eps: float = 1e-10) -> Array:
    """Matrix logarithm for SPD matrices."""
    return sym_funcm(P, jnp.log, eps=eps)


def sqrtm_spd(P: Array, eps: float = 1e-10) -> Array:
    """Matrix square-root for SPD matrices."""
    return sym_funcm(P, jnp.sqrt, eps=eps)


def make_true_spd_center(key: Array, n: int) -> Array:
    """Construct a nontrivial SPD ground-truth center P0."""
    A = jax.random.normal(key, shape=(n, n))
    Q, _ = jnp.linalg.qr(A)

    # A moderate range of eigenvalues avoids a nearly spherical example while
    # keeping the simulation numerically stable.
    eigs = jnp.exp(jnp.linspace(1.0, -1.0, n))
    return sym(Q @ jnp.diag(eigs) @ Q.T)


def sample_spd_affine_cloud(key: Array, P0: Array, n_samples: int, sigma: float) -> Array:
    """Generate SPD samples near P0 using affine-invariant local noise.

    Samples are generated as

        P_i = P0^{1/2} exp(sigma S_i) P0^{1/2},

    where S_i is a random symmetric matrix. This creates a cloud centered near
    P0 in the affine-invariant geometry.
    """
    n = P0.shape[0]
    P0_sqrt = sqrtm_spd(P0)
    keys = jax.random.split(key, n_samples)

    def one_sample(k: Array) -> Array:
        Z = jax.random.normal(k, shape=(n, n))
        S = sym(Z) / jnp.sqrt(float(n))
        return sym(P0_sqrt @ expm_sym(sigma * S) @ P0_sqrt)

    return jax.vmap(one_sample)(keys)


def random_spd_near(key: Array, P0: Array, scale: float = 0.75) -> Array:
    """Generate one SPD initial point near P0 in log coordinates."""
    n = P0.shape[0]
    A0 = logm_spd(P0)
    Z = jax.random.normal(key, shape=(n, n))
    S = sym(Z) / jnp.sqrt(float(n))
    return expm_sym(A0 + scale * S)


def make_frechet_cost(M: Any, samples: Array) -> Callable[[Array], Array]:
    """Return F(P) = (1 / 2N) sum_i d_M(P, P_i)^2."""

    def cost(P: Array) -> Array:
        dists = jax.vmap(lambda Q: M.dist(P, Q))(samples)
        return 0.5 * jnp.mean(dists**2)

    return cost


@dataclass
class SolverOutput:
    geometry: str
    method: str
    sol: Array
    final_cost: float
    info: list[Any]
    time_sec: float


def run_solver(
    M: Any,
    geometry_name: str,
    cost: Callable[[Array], Array],
    solver: Any,
    x0: Array,
    method_name: str,
) -> SolverOutput:
    """Build a Minimize problem, solve it and time execution."""
    problem = Minimize(
        M=M,
        cost=cost,
        solver=solver,
        x0=x0,
    )

    tic = time.perf_counter()
    sol, final_cost, info = problem.solve()
    jax.block_until_ready(sol)
    elapsed = time.perf_counter() - tic

    return SolverOutput(
        geometry=geometry_name,
        method=method_name,
        sol=sol,
        final_cost=float(final_cost),
        info=info,
        time_sec=elapsed,
    )


def summarize_output(M: Any, out: SolverOutput, P0: Array) -> dict[str, Any]:
    eigvals = jnp.linalg.eigvalsh(out.sol)
    return {
        "geometry": out.geometry,
        "method": out.method,
        "iterations": out.info[-1].iter,
        "time_sec": out.time_sec,
        "final_cost": out.final_cost,
        "final_gradnorm": out.info[-1].gradnorm,
        "dist_to_true_P0": float(M.dist(out.sol, P0)),
        "min_eig_sol": float(jnp.min(eigvals)),
        "max_eig_sol": float(jnp.max(eigvals)),
        "reason": out.info[-1].reason,
    }


def print_table(rows: list[dict[str, Any]]) -> None:
    """Print a compact text table without requiring pandas."""
    columns = [
        "geometry",
        "method",
        "iterations",
        "time_sec",
        "final_cost",
        "final_gradnorm",
        "dist_to_true_P0",
        "min_eig_sol",
        "max_eig_sol",
    ]
    widths = {col: max(len(col), *(len(format_cell(row[col])) for row in rows)) for col in columns}
    header = "  ".join(col.ljust(widths[col]) for col in columns)
    print(header)
    print("-" * len(header))
    for row in rows:
        print("  ".join(format_cell(row[col]).ljust(widths[col]) for col in columns))


def format_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def maybe_make_plot(outputs: list[SolverOutput], path: str) -> None:
    """Save a convergence plot if matplotlib is available."""
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"Could not import matplotlib; skipping plot. ({exc})")
        return

    plt.figure(figsize=(7, 4.5))
    for out in outputs:
        iters = [h.iter for h in out.info]
        gradnorms = [h.gradnorm for h in out.info]
        plt.semilogy(iters, gradnorms, label=f"{out.geometry} + {out.method}")
    plt.xlabel("Iteration")
    plt.ylabel("Riemannian gradient norm")
    plt.title("SPD Fréchet mean solver comparison")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    print(f"Saved convergence plot to: {path}")


def maybe_make_ellipse_plot(
    samples: Array,
    P0: Array,
    outputs: list[SolverOutput],
    path: str,
) -> None:
    """Visualize 2x2 SPD matrices as covariance ellipses."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse
    except Exception as exc:  # pragma: no cover
        print(f"Could not import matplotlib; skipping ellipse plot. ({exc})")
        return

    if samples.shape[-2:] != (2, 2):
        print("Ellipse plotting is only supported for 2x2 SPD matrices.")
        return

    def add_ellipse(
        ax, P: Array, color: str, *, lw: float = 1.5, alpha: float = 1.0, label: str | None = None
    ) -> None:
        vals, vecs = jnp.linalg.eigh(P)
        order = jnp.argsort(vals)[::-1]
        vals = vals[order]
        vecs = vecs[:, order]
        angle = float(jnp.degrees(jnp.arctan2(vecs[1, 0], vecs[0, 0])))
        width = 2.0 * float(jnp.sqrt(vals[0]))
        height = 2.0 * float(jnp.sqrt(vals[1]))
        ell = Ellipse(
            (0.0, 0.0),
            width=width,
            height=height,
            angle=angle,
            fill=False,
            lw=lw,
            alpha=alpha,
            color=color,
            label=label,
        )
        ax.add_patch(ell)

    fig, ax = plt.subplots(figsize=(6.8, 6.0))
    for sample in samples:
        add_ellipse(ax, sample, "#adb5bd", lw=1.0, alpha=0.6)
    add_ellipse(ax, P0, "#cc4b37", lw=2.8, label="true center")

    palette = {
        ("LogEuclidean", "SteepestDescent"): "#1f77b4",
        ("LogEuclidean", "ConjugateGradient"): "#2ca02c",
        ("AffineInvariant", "SteepestDescent"): "#9467bd",
        ("AffineInvariant", "ConjugateGradient"): "#ff7f0e",
    }
    for out in outputs:
        label = f"{out.geometry} + {out.method}"
        add_ellipse(ax, out.sol, palette[(out.geometry, out.method)], lw=2.0, label=label)

    ax.axhline(0.0, color="#dee2e6", lw=0.8)
    ax.axvline(0.0, color="#dee2e6", lw=0.8)
    ax.set_aspect("equal")
    ax.set_xlim(-2.8, 2.8)
    ax.set_ylim(-2.8, 2.8)
    ax.set_title("2x2 SPD matrices as covariance ellipses")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close(fig)
    print(f"Saved ellipse plot to: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="SPD Fréchet mean simulation with GeoJAX.")
    parser.add_argument("--matrix-size", type=int, default=3, help="SPD matrix dimension n.")
    parser.add_argument("--n-samples", type=int, default=50, help="Number of SPD samples.")
    parser.add_argument(
        "--sigma", type=float, default=0.25, help="Noise scale around the true SPD center."
    )
    parser.add_argument(
        "--x0-scale",
        type=float,
        default=0.75,
        help="Scale of random perturbation for initial point.",
    )
    parser.add_argument(
        "--maxiter", type=int, default=200, help="Maximum iterations for each solver."
    )
    parser.add_argument(
        "--tolgradnorm", type=float, default=1e-8, help="Gradient norm stopping tolerance."
    )
    parser.add_argument(
        "--minstepsize", type=float, default=1e-12, help="Minimum step size stopping threshold."
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--cg-beta-type", type=str, default="P-R", help="Conjugate-gradient beta rule."
    )
    parser.add_argument("--plot", action="store_true", help="Save a convergence plot.")
    parser.add_argument("--plot-path", type=str, default="spd_frechet_mean_convergence.png")
    parser.add_argument(
        "--ellipse-plot", action="store_true", help="Save a 2x2 ellipse visualization."
    )
    parser.add_argument("--ellipse-plot-path", type=str, default="spd_frechet_mean_ellipses.png")
    args = parser.parse_args()

    n = args.matrix_size
    N = args.n_samples

    key = jax.random.key(args.seed)
    key_center, key_data, key_x0 = jax.random.split(key, 3)

    P0 = make_true_spd_center(key_center, n)
    samples = sample_spd_affine_cloud(key_data, P0, N, args.sigma)
    x0 = random_spd_near(key_x0, P0, scale=args.x0_scale)

    print("SPD Fréchet mean simulation")
    print(f"  matrix size:     {n} x {n}")
    print(f"  sample count:    {N}")
    print(f"  sigma:           {args.sigma}")
    print(f"  x0 min eig:      {float(jnp.min(jnp.linalg.eigvalsh(x0))):.6g}")
    print(f"  first sample eig:{jnp.asarray(jnp.linalg.eigvalsh(samples[0]))}")
    print()

    solvers = {
        "SteepestDescent": SteepestDescent(
            maxiter=args.maxiter,
            tolgradnorm=args.tolgradnorm,
            minstepsize=args.minstepsize,
            verbosity=0,
        ),
        "ConjugateGradient": ConjugateGradient(
            beta_type=args.cg_beta_type,
            maxiter=args.maxiter,
            tolgradnorm=args.tolgradnorm,
            minstepsize=args.minstepsize,
            verbosity=0,
        ),
    }

    geometries = {
        "LogEuclidean": SPDLogEuclidean(size=(n, n)),
        "AffineInvariant": SPDAffineInvariant(size=(n, n)),
    }

    outputs: list[SolverOutput] = []
    rows: list[dict[str, Any]] = []

    for geometry_name, M in geometries.items():
        cost = make_frechet_cost(M, samples)
        for method_name, solver in solvers.items():
            out = run_solver(M, geometry_name, cost, solver, x0, method_name)
            outputs.append(out)
            rows.append(summarize_output(M, out, P0))

    print("Solver comparison:")
    print_table(rows)
    print()

    # Closed-form check for log-Euclidean mean.
    M_le = geometries["LogEuclidean"]
    P_le_closed = expm_sym(jnp.mean(jax.vmap(logm_spd)(samples), axis=0))
    print("Log-Euclidean closed-form check:")
    for out in outputs:
        if out.geometry == "LogEuclidean":
            print(
                f"  {out.method:18s} dist(solution, closed-form) = "
                f"{float(M_le.dist(out.sol, P_le_closed)):.6e}"
            )
    print(f"  closed-form dist_to_true_P0 = {float(M_le.dist(P_le_closed, P0)):.6e}")

    if args.plot:
        maybe_make_plot(outputs, args.plot_path)
    if args.ellipse_plot:
        maybe_make_ellipse_plot(samples, P0, outputs, args.ellipse_plot_path)


if __name__ == "__main__":
    main()
