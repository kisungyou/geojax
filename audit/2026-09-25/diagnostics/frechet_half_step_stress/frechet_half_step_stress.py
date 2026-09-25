import json, time, traceback
from pathlib import Path
import numpy as np
import jax
jax.config.update('jax_enable_x64',False)
import jax.numpy as jnp
from geojax.geometry import Euclidean,Sphere,SPDAffineInvariant,SPDLogEuclidean,SPDBuresWasserstein,RankKPSDBuresWasserstein
from geojax.learning import frechet_mean
from geojax.learning._statistics import _gradient_tolerances
from geojax.learning._utils import normalize_weights
from geojax.optimization import ConjugateGradient,AdaptiveArmijo
output=Path('/private/tmp/geojax-validation-sep25/frechet_half_step_stress.json')
records=[];started=time.monotonic()
weights=jnp.array([.3,.7]);normalized=normalize_weights(2,weights);w=np.asarray(normalized,dtype=float);w=w/w.sum()
cases=[]
for scale in [1e-5,1.,1e5]:
 a=scale*np.array([1.,-2.]);b=scale*np.array([1.4,-1.7]);cases.append(('Euclidean',scale,Euclidean((2,)),a,b,'euclidean'))
for angle in [.1,.8,1.6]:
 a=np.array([1.,0.,0.]);b=np.array([np.cos(angle),np.sin(angle),0.]);cases.append(('Sphere',angle,Sphere(3),a,b,'sphere'))
for cls in [SPDAffineInvariant,SPDLogEuclidean,SPDBuresWasserstein]:
 for scale in [1e-6,1.,1e6]:
  a=scale*np.diag([1.,2.]);b=scale*np.diag([1.5,2.4]);cases.append((cls.__name__,scale,cls((2,2)),a,b,'bures' if cls is SPDBuresWasserstein else 'log'))
for scale in [1e-6,1.,1e6]:
 a=scale*np.diag([1.,2.,0.]);b=scale*np.diag([1.5,2.4,0.]);cases.append(('RankKPSDBuresWasserstein',scale,RankKPSDBuresWasserstein((3,3),rank=2),a,b,'rank_bures'))
for index,(name,scale,M,a,b,kind) in enumerate(cases,1):
 start=time.monotonic();row={'geometry':name,'scale_or_angle':scale,'weights':w.tolist()}
 try:
  a=jnp.asarray(a);b=jnp.asarray(b);data=jnp.stack([a,b]);x0=b
  radius=jnp.sqrt(jnp.sum(normalized*M.squared_dist(x0,data)))
  requested,tol=_gradient_tolerances(x0,radius,1e-7)
  solver=ConjugateGradient(maxiter=200,tolgradnorm=tol,verbosity=0,line_search=AdaptiveArmijo(initial_stepsize=.5,normalize_step=False))
  fit=frechet_mean(M,data,sample_weight=weights,solver=solver)
  af=np.asarray(a,dtype=float);bf=np.asarray(b,dtype=float);p=np.asarray(fit.point,dtype=float)
  if kind=='euclidean':
   ref=w[0]*af+w[1]*bf;error=np.linalg.norm(p-ref);distance=np.linalg.norm(af-bf)
  elif kind=='sphere':
   theta=np.arctan2(bf[1],bf[0]);ref=np.array([np.cos(w[1]*theta),np.sin(w[1]*theta),0]);error=abs(np.arctan2(p[1],p[0])-w[1]*theta);distance=theta
  elif kind=='log':
   la=np.log(np.diag(af));lb=np.log(np.diag(bf));lr=w[0]*la+w[1]*lb;ref=np.diag(np.exp(lr));error=np.linalg.norm(np.log(np.diag(p))-lr);distance=np.linalg.norm(la-lb)
  else:
   indices=slice(None,-1) if kind=='rank_bures' else slice(None)
   sa=np.sqrt(np.diag(af)[indices]);sb=np.sqrt(np.diag(bf)[indices]);sr=w[0]*sa+w[1]*sb
   ref=np.diag(np.r_[sr**2,0.] if kind=='rank_bures' else sr**2);error=np.linalg.norm(np.sqrt(np.diag(p)[indices])-sr);distance=np.linalg.norm(sa-sb)
  expected=w.prod()*distance**2
  row.update(converged=bool(fit.converged),reason=fit.reason,iterations=fit.iterations,gradient_norm=float(fit.gradient_norm),effective_tolerance=fit.diagnostics['effective_tolerance'],custom_tolerance=tol,geodesic_reference_error=float(error),reference_error_over_effective_tolerance=float(error/tol),objective=float(fit.objective),analytic_minimum=float(expected),objective_relative_error=float((float(fit.objective)-expected)/max(expected,np.finfo(float).tiny)),history=[{'iter':r.iter,'cost':r.cost,'gradnorm':r.gradnorm,'line_search_accepted':None if r.linesearch is None else r.linesearch.accepted,'alpha':None if r.linesearch is None else r.linesearch.alpha} for r in fit.diagnostics['history']])
 except Exception as error:
  row.update(converged=False,error=repr(error),traceback=traceback.format_exc())
 row['seconds']=time.monotonic()-start;records.append(row)
 summary={'jax':jax.__version__,'x64':jax.config.jax_enable_x64,'initial_stepsize':.5,'planned_cases':len(cases),'completed_cases':len(records),'seconds':time.monotonic()-started,'cases':records}
 output.write_text(json.dumps(summary,indent=2)+'\n')
 print(index,name,scale,'converged',row['converged'],'grad',row.get('gradient_norm'),'tol',row.get('effective_tolerance'),'seconds',round(row['seconds'],2),flush=True)
