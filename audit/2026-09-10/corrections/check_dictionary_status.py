import json
import os

import jax
jax.config.update("jax_enable_x64", os.environ.get("GEOJAX_TEST_X64") == "1")
import jax.numpy as jnp
import numpy as np
from geojax.geometry import Euclidean
from geojax.learning import manifold_dictionary_learning

values = jnp.array([[-2.], [-1.], [1.], [2.]])
result = manifold_dictionary_learning(Euclidean(1), values, n_atoms=2,
    initial_atoms=values[jnp.array([0, 3])], maxiter=1,
    coding_maxiter=20, center_maxiter=10, tol=0.)
history = result.diagnostics["objective_history"]
atom_history = result.diagnostics["atom_optimizer_histories"][-1]
atom_tol = result.diagnostics["atom_optimizer_tolerances"][-1]
atoms = np.asarray(result.atoms, dtype=np.float64)
codes = np.asarray(result.codes, dtype=np.float64)
original = np.asarray(values, dtype=np.float64)
oracle = np.mean(np.sum((codes @ atoms - original)**2, axis=1) + 1e-6 * np.sum(codes**2, axis=1))
summary = dict(jax=jax.__version__, x64=jax.config.x64_enabled,
    reason=result.reason, converged=result.converged,
    objective=float(result.objective), independent_objective=oracle,
    objective_history=np.asarray(history).tolist(),
    coding_converged=result.diagnostics["coding_result"].converged,
    atom_converged=bool(result.diagnostics["atom_optimizer_converged"][-1]),
    atom_final_gradnorm=atom_history[-1].gradnorm, atom_tol=float(atom_tol),
    atom_reason=atom_history[-1].reason,
    simplex_residual=float(jnp.max(jnp.abs(jnp.sum(result.codes, axis=1)-1.))),
    max_reconstruction_error=float(jnp.max(jnp.abs(result.reconstructions-values))))
print(json.dumps(summary, indent=2))
