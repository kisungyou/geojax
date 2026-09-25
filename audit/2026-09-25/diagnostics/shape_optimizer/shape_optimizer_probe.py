import json
import jax
jax.config.update('jax_enable_x64', False)
import jax.numpy as jnp
from geojax.geometry import KendallShape
from geojax.learning import frechet_mean
from geojax.optimization import ConjugateGradient,AdaptiveArmijo,Minimize
m=3
pairs=jnp.concatenate([jnp.eye(m),-jnp.eye(m)]);zeros=jnp.zeros_like(pairs)
x=jnp.concatenate([pairs,zeros])*jnp.array([1.,1.,0.])/jnp.sqrt(4.)
u=jnp.concatenate([zeros,pairs])/jnp.sqrt(6.)
M=KendallShape(x.shape);y=jnp.cos(.1)*x+jnp.sin(.1)*u;data=jnp.stack([x,y]);objective=lambda p:jnp.mean(M.squared_dist(p,data))
grad=lambda p:M.egrad_to_rgrad(p,jax.grad(objective)(p))
loggrad=lambda p:-jnp.mean(M.log(p,data),axis=0)*2
fit=frechet_mean(M,data);p=fit.point;g=grad(p)
def record(p):
 angle=jnp.arctan2(jnp.sum(p*u),jnp.sum(p*x))
 ag=grad(p);lg=loggrad(p)
 return {'cost':float(objective(p)),'angle':float(angle),'analytic_cost':float((angle**2+(angle-.1)**2)/2),'analytic_gradient_norm':float(2*jnp.abs(angle-.05)),'ad_gradient_norm':float(M.norm(p,ag)),'log_gradient_norm':float(M.norm(p,lg)),'ad_log_difference':float(M.norm(p,ag-lg))}
output={'default':record(p),'default_reason':fit.reason,'default_history':[{'iter':r.iter,'cost':r.cost,'gradnorm':r.gradnorm,'alpha':None if r.linesearch is None else r.linesearch.alpha} for r in fit.diagnostics['history']]}
output['trials']={str(a):record(M.retr(p,-a*g)) for a in [1.,.5,.25,.125]}
for step in [.5,.25]:
 custom=ConjugateGradient(maxiter=200,tolgradnorm=fit.diagnostics['effective_tolerance'],verbosity=0,line_search=AdaptiveArmijo(normalize_step=False,initial_stepsize=step))
 result=frechet_mean(M,data,solver=custom)
 output['initial_step_'+str(step)]={'converged':result.converged,'iterations':result.iterations,'reason':result.reason,**record(result.point)}
print(json.dumps(output,indent=2))
