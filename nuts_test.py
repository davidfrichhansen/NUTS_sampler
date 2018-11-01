import numpy as np
import sys
#from nuts_badgrad import NUTS
from nuts_goodgrad import NUTS
from scipy.stats import wishart
import matplotlib.pyplot as plt

#np.seterr(all='raise')



if len(sys.argv) < 3:
    sys.exit("Usage: python %s num_iterations M_adapt" % sys.argv[0])

num_samples = int(sys.argv[1])
M_adapt = int(sys.argv[2])


dim = 2
num_tests=15
np.random.seed(20)
A = wishart.rvs(df=dim, scale=np.eye(dim))
# this means that the log-likelihood is proportional to -1/2 * theta^T A theta
def logp(theta):
    logprob = -.5 * theta.T @ A @ theta
    grad = -A@theta
    return logprob, grad
# do different tests
x1diff = []
x2diff = []
x1x2diff = []
invA= np.linalg.inv(A)

init = np.random.multivariate_normal([0,0], cov=invA)
for i in range(1, num_tests):
    sampler = NUTS(logp, i*500, M_adapt, init, debug=False, delta=0.25)
    sampler.sample()
    xp, yp = list(zip(*sampler.samples))
    cov_NUTS = np.cov(xp, yp)
    diff = cov_NUTS - invA
    print(diff)

    x1diff.append(diff[0,0])
    x2diff.append(diff[1,1])
    x1x2diff.append(diff[0,1])

print(invA)
itsp = [i*1500 for i in range(num_tests-1)]

plt.plot(itsp, x1diff, 'b*-', label="Variance of x1")
plt.plot(itsp, x2diff, 'r*-', label="Variance of x2")
plt.plot(itsp, x1x2diff, 'g*-', label="Cov x1 x2")
plt.plot(itsp, np.zeros_like(itsp), 'k--')
plt.legend()
plt.show()







"""
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


"""
