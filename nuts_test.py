import numpy as np

from nuts_badgrad import NUTS
#from nuts_goodgrad import NUTS
from scipy.stats import wishart
np.random.seed(50)
import matplotlib.pyplot as plt

#np.seterr(all='raise')

# TODO: FIX FUCKING GRADIENT EVALS

# 250 dimensional MVN as in the original paper
dim = 250
A = wishart.rvs(df=dim, scale=np.eye(dim))
# this means that the log-likelihood is proportional to -1/2 * theta^T A theta

def logp(theta):
    logprob = -.5 * theta.T @ A @ theta
    grad = -A@theta
    return logprob, grad

lp, grad = logp(np.random.rand(dim))
print(grad.shape)

sampler = NUTS(logp, 10000, 100, np.random.rand(dim), debug=False)



sampler.sample()
xp, yp = list(zip(*sampler.samples))
"""
print(np.linalg.inv(A))
print(np.cov(xp, yp))
plt.scatter(xp, yp)
xg, yg = list(zip(*np.random.multivariate_normal([0,0], cov = np.linalg.inv(A), size=50000)))

plt.scatter(xg, yg, c='orange')
plt.show()
"""



