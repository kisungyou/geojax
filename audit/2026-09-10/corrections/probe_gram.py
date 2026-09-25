import jax.numpy as j
from geojax.geometry import Euclidean
from geojax.learning import geodesic_barycentric_coding
for a,b,p in [(-1.4142135,1.7320508,.3),(-2.1,3.3,.8),(-3.7,1.3,.1),(-1.3,2.8,1.4)]:
    result=geodesic_barycentric_coding(Euclidean(1),j.array([[p]]),j.array([[a],[b]]),ridge=0.,maxiter=200,reconstruction_maxiter=20)
    code=result.codes[0]
    residual=code[0]*(a-p)+code[1]*(b-p)
    print(a,b,p,'code',code,'objective',float(result.objective),'direct',float(residual**2),'converged',result.converged)
