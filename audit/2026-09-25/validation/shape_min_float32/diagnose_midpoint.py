"""Read-only comparison with an independent NumPy float64 quotient formula."""
import dataclasses,json,platform,sys,time
from pathlib import Path
import jax
jax.config.update('jax_enable_x64',False)
import jax.numpy as jnp
import numpy as np
from geojax.geometry import KendallShape
from geojax.learning import frechet_mean

start=time.monotonic();m=3
weights=jnp.array([1.,1.,0.]);pairs=jnp.concatenate([jnp.eye(m),-jnp.eye(m)]);zero=jnp.zeros_like(pairs)
x=jnp.concatenate([pairs,zero])*weights/jnp.sqrt(4.)
u=jnp.concatenate([zero,pairs])/jnp.sqrt(6.)
y=jnp.cos(.1)*x+jnp.sin(.1)*u
M=KendallShape(x.shape);data=jnp.stack([x,y]);objective=lambda p:jnp.sum(.5*M.squared_dist(p,data))
fit=frechet_mean(M,data);p=fit.point;expected=jnp.cos(.05)*x+jnp.sin(.05)*u

def normalize(a):
 a=np.asarray(a,dtype=np.float64);a=a-np.mean(a,axis=-2,keepdims=True);return a/np.linalg.norm(a,axis=(-2,-1),keepdims=True)
def aligned(a,b):
 l,_,rt=np.linalg.svd(a.T@b,full_matrices=False);q=l@rt;l[:,-1]*=np.linalg.det(q);return a@(l@rt)
def dist(a,b):
 a,b=normalize(a),normalize(b);b=aligned(b,a);return 2*np.arctan2(np.linalg.norm(a-b),np.linalg.norm(a+b))
def ref_log(a,b):
 a,b=normalize(a),normalize(b);b=aligned(b,a);c=np.sum(a*b);v=b-c*a;theta=np.arctan2(np.linalg.norm(v),c);return theta/np.linalg.norm(v)*v
def ref(a):
 a=normalize(a);vals=[dist(a,b)**2 for b in np.asarray(data)];g=-(ref_log(a,data[0])+ref_log(a,data[1]));return {'objective':float(np.mean(vals)),'distances_squared':vals,'gradient_norm':float(np.linalg.norm(g)),'gradient':g.tolist()}
refx=normalize(x);refy=aligned(normalize(y),refx);mid=normalize(refx+refy);dxy=dist(refx,refy)

def describe(name,a):
 raw=float(objective(a));value,eg=jax.value_and_grad(objective)(a);g=M.egrad_to_rgrad(a,eg)
 aligned_data,rotation=M.align(M.project(data),M.project(a))
 return {'name':name,'point':np.asarray(a).tolist(),'raw_cost':raw,'value_and_grad_cost':float(value),'value_minus_raw':float(value)-raw,'gradient_norm':float(M.norm(a,g)),'reference':ref(a),'distance_to_reference_midpoint':dist(a,mid),'normalization_max_change':float(jnp.max(jnp.abs(M.project(a)-a))),'rotation_max_error_from_identity':float(jnp.max(jnp.abs(rotation-jnp.eye(m))))}
value,eg=jax.value_and_grad(objective)(p);g=M.egrad_to_rgrad(p,eg)
points=[describe('fitted',p),describe('analytic_trig_midpoint',expected),describe('reference_midpoint_cast_float32',jnp.asarray(mid))]
trials=[]
for alpha in [2.,1.,.75,.5,.25,.125,.0625,.01,.001,0.]:
 q=M.retr(p,-g,alpha);d=describe('alpha='+str(alpha),q);d['alpha']=alpha;d['predicted_first_order_change']=-alpha*float(M.inner(p,g,g));trials.append(d)
result={'python':platform.python_version(),'jax':jax.__version__,'numpy':np.__version__,'package_path':__import__('geojax').__file__,'converged':fit.converged,'reason':fit.reason,'stored_objective':float(fit.objective),'stored_gradient_norm':float(fit.gradient_norm),'effective_tolerance':fit.diagnostics['effective_tolerance'],'exact_numpy_data_distance':dxy,'exact_numpy_minimum_objective':dxy*dxy/4,'points':points,'trials':trials,'history':[dataclasses.asdict(h) for h in fit.diagnostics['history']],'wall_seconds':time.monotonic()-start}
path=Path(sys.argv[1]);path.write_text(json.dumps(result,indent=2,default=float)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['points','trials','history']},indent=2),flush=True)
for d in points+trials:print(d['name'],'raw',d['raw_cost'],'vg',d['value_and_grad_cost'],'reference',d['reference']['objective'],'gnorm',d['gradient_norm'],'refgnorm',d['reference']['gradient_norm'],'distance_mid',d['distance_to_reference_midpoint'],flush=True)
