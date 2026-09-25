"""Recheck floating-range endpoints; expose backend subnormal-AD limits too."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import jax

jax.config.update("jax_enable_x64", os.environ.get("GEOJAX_TEST_X64", "1") == "1")
import jax.numpy as jnp
import numpy as np
from geojax.geometry import SPDLogEuclidean, SPDAffineInvariant, SPDBuresWasserstein
from geojax.geometry.spd import _solve_spd_sylvester, _spd_sqrtm

m = SPDLogEuclidean((2, 2))
dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
for name, scale in [("tiny", np.finfo(dtype).tiny), ("large", np.finfo(dtype).max * 0.75)]:
    p = jnp.asarray(scale, dtype=dtype) * jnp.eye(2, dtype=dtype)
    print(name, "project/scale", np.asarray(jax.jit(m.project)(p)) / scale, flush=True)
    print(
        "belongs",
        [
            bool(M((2, 2)).belongs(p))
            for M in [SPDLogEuclidean, SPDAffineInvariant, SPDBuresWasserstein]
        ],
        flush=True,
    )
    print("log", np.asarray(jax.jit(m.logm)(p)), flush=True)
    print(
        "sqrt/expected",
        np.asarray(jax.jit(lambda a: _spd_sqrtm(a, 1e-10))(p)) / np.sqrt(scale),
        flush=True,
    )
    print("sylvester", np.asarray(jax.jit(_solve_spd_sylvester)(p, p)), flush=True)

    def f(t):
        return jnp.trace(_solve_spd_sylvester(t * p, p))

    print("sylvester derivative", np.asarray(jax.jit(jax.grad(f))(1.0)), flush=True)

    def g(t):
        return jnp.trace(m.logm(t * p))

    print("logderiv", np.asarray(jax.jit(jax.grad(g))(1.0)), flush=True)
    print("loghess", np.asarray(jax.jit(jax.grad(jax.grad(g)))(1.0)), flush=True)
