import numpy as np
import sys
#from nuts_badgrad import NUTS
from nuts_goodgrad import NUTS
from scipy.stats import wishart
np.random.seed(20)
import matplotlib.pyplot as plt

#np.seterr(all='raise')


if len(sys.argv) < 3:
    sys.exit("Usage: python %s num_iterations M_adapt" % sys.argv[0])

num_samples = int(sys.argv[1])
M_adapt = int(sys.argv[2])
# 250 dimensional MVN as in the original paper
dim = 2
A = wishart.rvs(df=dim, scale=np.eye(dim))
# this means that the log-likelihood is proportional to -1/2 * theta^T A theta

def logp(theta):
    logprob = -.5 * theta.T @ A @ theta
    grad = -A@theta
    return logprob, grad

sampler = NUTS(logp, num_samples, M_adapt, np.random.multivariate_normal([0,0], cov=np.linalg.inv(A)), debug=False)



sampler.sample()
xp, yp = list(zip(*sampler.samples))
print(len(xp))

print(np.linalg.inv(A))
print(np.cov(xp, yp))


plt.scatter(xp, yp, alpha=0.3, label="NUTS")
xg, yg = list(zip(*np.random.multivariate_normal([0,0], cov = np.linalg.inv(A), size=num_samples-M_adapt)))

plt.scatter(xg, yg, c='orange', alpha=0.3, label="Built-in")
plt.legend()
plt.show()



