"""Analytic short-geodesic stress check of the final default Fréchet mean."""
from pathlib import Path
import dataclasses,gc,json,os,platform,sys,time,traceback
import jax
jax.config.update('jax_enable_x64',os.environ.get('GEOJAX_TEST_X64','1')=='1')
import jax.numpy as jnp
import numpy as np
from geojax.geometry import KendallShape
from geojax.learning import frechet_mean
from geojax.learning._data import as_manifold_data
from geojax.learning._statistics import _weighted_medoid,_gradient_tolerances
from geojax.learning._utils import normalize_weights

output=Path(sys.argv[1]);start=time.monotonic();rows=[];point_tolerance=5e-7 if jax.config.jax_enable_x64 else 4e-5

def normalize(a):
 a=np.asarray(a,dtype=np.float64);a=a-a.mean(axis=-2,keepdims=True);return a/np.linalg.norm(a,axis=(-2,-1),keepdims=True)
def align(a,b):
 l,_,rt=np.linalg.svd(a.T@b,full_matrices=False);l[:,-1]*=np.linalg.det(l@rt);return a@(l@rt)
def distance(a,b):
 a=normalize(a);b=align(normalize(b),a);return 2*np.arctan2(np.linalg.norm(a-b),np.linalg.norm(a+b))
def log(a,b):
 a=normalize(a);b=align(normalize(b),a);c=np.sum(a*b);v=b-c*a;s=np.linalg.norm(v);return np.zeros_like(v) if s==0 else np.arctan2(s,c)/s*v

def save():
 d={'python':platform.python_version(),'jax':jax.__version__,'numpy':np.__version__,'precision':'float64' if jax.config.jax_enable_x64 else 'float32','solver':'default frechet_mean','package_path':__import__('geojax').__file__,'point_tolerance':point_tolerance,'cases':rows,'elapsed_seconds':time.monotonic()-start,'completed':len(rows),'passed':sum(r.get('passed',False) for r in rows),'failed':sum(not r.get('passed',False) for r in rows)}
 output.write_text(json.dumps(d,indent=2,default=float)+'\n')

for m,rank in [(2,2),(3,2),(4,3)]:
 pairs=jnp.concatenate([jnp.eye(m),-jnp.eye(m)]);zeros=jnp.zeros_like(pairs);weights_x=jnp.arange(m)<rank
 x=jnp.concatenate([pairs,zeros])*weights_x/jnp.sqrt(float(2*rank));u=jnp.concatenate([zeros,pairs])/jnp.sqrt(float(2*m));M=KendallShape(x.shape)
 specs=[(n,sep,False) for n,sep in zip([2,3,5,2,5],[.001,.02,.1,.3,.7])]+[(n,.1,True) for n in [2,3,5]]
 for n,separation,uniform in specs:
  row={'ambient_dim':m,'base_rank':rank,'samples':n,'separation':separation,'uniform_weights':uniform};case_start=time.monotonic()
  try:
   angles=jnp.linspace(0,separation,n);values=jnp.cos(angles)[:,None,None]*x+jnp.sin(angles)[:,None,None]*u
   raw_weights=jnp.ones(n) if uniform else jnp.asarray({2:[1.,4.],3:[1.,2.,7.],5:[1.,2.,3.,5.,8.]}[n]);weights=normalize_weights(n,raw_weights)
   adapted=as_manifold_data(M,values);x0=_weighted_medoid(M,adapted,weights,squared=True)
   radius=jnp.sqrt(jnp.maximum(jnp.sum(weights*M.squared_dist(x0,values)),0.));_,tol=_gradient_tolerances(x0,radius,1e-7)
   unique=M._alignment(values[:,None],values[None,:])[2];row['all_alignment_pairs_unique']=bool(jnp.all(unique));row['all_samples_belong']=bool(jnp.all(M.belongs(values)));assert row['all_alignment_pairs_unique'] and row['all_samples_belong']
   fitted=frechet_mean(M,values,sample_weight=raw_weights);assert fitted.diagnostics['effective_tolerance']==tol
   npweights=np.array(weights,dtype=np.float64,copy=True);npweights/=npweights.sum();mean_angle=float(np.dot(npweights,np.asarray(angles,dtype=np.float64)))
   expected=jnp.cos(mean_angle)*x+jnp.sin(mean_angle)*u
   point_distance=float(M.dist(fitted.point,expected));refgrad=-2*sum(w*log(fitted.point,v) for w,v in zip(npweights,np.asarray(values)))
   refobj=sum(w*distance(fitted.point,v)**2 for w,v in zip(npweights,np.asarray(values)))
   row.update(weights=npweights.tolist(),mean_angle=mean_angle,converged=bool(fitted.converged),reason=fitted.reason,gradient_norm=float(fitted.gradient_norm),effective_tolerance=tol,point_distance=point_distance,reference_point_distance=distance(fitted.point,expected),reference_gradient_norm=float(np.linalg.norm(refgrad)),objective=float(fitted.objective),reference_objective=refobj,iterations=fitted.iterations,passed=bool(fitted.converged) and float(fitted.gradient_norm)<=tol and point_distance<=point_tolerance)
   if not row['passed']:
    row['point']=np.asarray(fitted.point).tolist();row['history']=[dataclasses.asdict(h) for h in fitted.diagnostics['history']]
  except Exception as exc:row.update(passed=False,error=repr(exc),traceback=traceback.format_exc())
  row['wall_seconds']=time.monotonic()-case_start;rows.append(row);save();print(json.dumps({k:v for k,v in row.items() if k not in ['point','history','traceback']},default=float),flush=True)
 jax.clear_caches();gc.collect()
print('SUMMARY '+json.dumps({'completed':len(rows),'passed':sum(r['passed'] for r in rows),'wall_seconds':time.monotonic()-start}),flush=True)
