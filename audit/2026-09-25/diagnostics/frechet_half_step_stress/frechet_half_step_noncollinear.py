import json,time
from pathlib import Path
import numpy as np
import jax
jax.config.update('jax_enable_x64',False)
import jax.numpy as jnp
from geojax.geometry import Sphere,SPDAffineInvariant
from geojax.learning import frechet_mean
from geojax.learning._statistics import _gradient_tolerances
from geojax.optimization import ConjugateGradient,AdaptiveArmijo
points=[];theta=.6
for axis in [1,2]:
 for sign in [-1,1]:
  p=np.array([np.cos(theta),0.,0.]);p[axis]=sign*np.sin(theta);points.append(p)
cases=[('Sphere symmetric four points',Sphere(3),np.array(points),np.array([1.,0.,0.]))]
matrices=[]
for tangent in [np.array([[.3,.18],[.18,-.1]]),np.array([[-.2,.12],[.12,.25]])]:
 for sign in [-1,1]:
  values,vectors=np.linalg.eigh(sign*tangent);matrices.append((vectors*np.exp(values))@vectors.T)
cases.append(('AffineSPD noncommuting symmetric logarithms',SPDAffineInvariant((2,2)),np.array(matrices),np.eye(2)))
records=[]
for name,M,array,reference in cases:
 start=time.monotonic();data=jnp.asarray(array);pair=M.squared_dist(data[:,None],data[None,:]);index=int(jnp.argmin(jnp.mean(pair,axis=1)));x0=data[index];radius=jnp.sqrt(jnp.mean(M.squared_dist(x0,data)));_,tol=_gradient_tolerances(x0,radius,1e-7)
 solver=ConjugateGradient(maxiter=200,tolgradnorm=tol,verbosity=0,line_search=AdaptiveArmijo(initial_stepsize=.5,normalize_step=False))
 fit=frechet_mean(M,data,solver=solver)
 record={'geometry':name,'converged':fit.converged,'gradient_norm':float(fit.gradient_norm),'effective_tolerance':fit.diagnostics['effective_tolerance'],'reference_error':float(M.dist(fit.point,jnp.asarray(reference))),'iterations':fit.iterations,'reason':fit.reason,'seconds':time.monotonic()-start,'history':[{'iter':r.iter,'cost':r.cost,'gradnorm':r.gradnorm,'alpha':None if r.linesearch is None else r.linesearch.alpha} for r in fit.diagnostics['history']]}
 records.append(record);print(json.dumps(record),flush=True)
 Path('/private/tmp/geojax-validation-sep25/frechet_half_step_noncollinear.json').write_text(json.dumps({'jax':jax.__version__,'x64':jax.config.jax_enable_x64,'cases':records},indent=2)+'\n')
