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
num_tests=20
np.random.seed(10)
A = wishart.rvs(df=dim, scale=np.eye(dim))

#A = np.linalg.inv(np.array([[20, 2], [2, 16]]))

# this means that the log-likelihood is proportional to -1/2 * theta^T A theta
mu = np.array([-1,1])
def logp(theta):
    logprob = -.5 * (theta-mu).T @ A @ (theta-mu)
    grad = -A@(theta-mu)
    return logprob, grad
# do different tests
x1diff = []
x2diff = []
x1x2diff = []
meanx1 = []
meanx2 = []
invA= np.linalg.inv(A)
inc = 500
init = np.random.multivariate_normal([0,0], cov=invA)
for i in range(1, num_tests):
    sampler = NUTS(logp, i*inc, int(0.5*inc*i), init, debug=False, delta=0.65) # delta is set from paper
    sampler.sample()
    xp, yp = list(zip(*sampler.samples))
    cov_NUTS = np.cov(xp, yp)
    diff = cov_NUTS - invA
    print(diff)
    x1diff.append(diff[0,0])
    x2diff.append(diff[1,1])
    x1x2diff.append(diff[0,1])
    meanx1.append(np.mean(xp))
    meanx2.append(np.mean(yp))

print(invA)
print(cov_NUTS)
itsp = [i*inc for i in range(num_tests-1)]

plt.plot(itsp, x1diff, 'b*-', label="Variance of x1")
plt.plot(itsp, x2diff, 'r*-', label="Variance of x2")
plt.plot(itsp, x1x2diff, 'g*-', label="Cov x1 x2")
plt.plot(itsp, meanx1, 'k*-', label="Mean x1")
plt.plot(itsp, meanx2, 'c*-', label="Mean x2")
plt.xlabel("Iterations run")
plt.ylabel("Difference from theretical value")
plt.plot(itsp, np.zeros_like(itsp), 'k--')
plt.legend()
plt.show()



plt.scatter(xp, yp, c="orange", alpha=0.5, label="NUTS")
xg, yg = zip(*np.random.multivariate_normal(mu, invA, size=itsp[-1] - int(0.5*itsp[-1])))
plt.scatter(xg,yg, c="blue", alpha=0.5, label="Numpy")
plt.legend()
plt.show()
