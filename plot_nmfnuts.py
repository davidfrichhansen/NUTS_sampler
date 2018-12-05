import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from aux_funcs import link_rectgauss, link_exp_to_gauss, loglik, get_2d_exp_kernel, rbf
from scipy.spatial.distance import euclidean

# dimensions

mat = loadmat('./data/newdata.mat')
a = mat['a']
vp = mat['vp']
X = mat['X']
print(X.shape)

K, L = X.shape

M = 2
nr_D = K*M
nr_H = L*M
print(nr_D)
print(nr_H)
sigma_N = 5
beta_D = 2.5*10
beta_H = 2.5*10

# traces
samples = np.load('smooth_prior.npy')
print("Sample shape")
print(samples.shape)

dim = int(np.sqrt(len(X)))
cov_D_2d = get_2d_exp_kernel(beta_D, (dim, dim))
cov_D_2d = cov_D_2d + 1e-5 * np.eye(K)

cov_H = np.zeros((L, L))
for i in range(L):
    for j in range(L):
        cov_H[i, j] = rbf(beta_H, i + 1, j + 1)

cov_H = cov_H + 1e-5 * np.eye(L)

cholD = np.linalg.cholesky(cov_D_2d)
cholH = np.linalg.cholesky(cov_H)

print(samples.shape)
eta = samples[:, :nr_H]
delta = samples[:, nr_H:]
print("Shape eta")
print(eta.shape)
print("Shape delta")
print(delta.shape)

# get H with link_exp_to_gauss
H_space_samples = np.zeros((M,L,samples.shape[0]))
D_space_samples = np.zeros((M,K, samples.shape[0]))

for i in range(samples.shape[0]):
    H_space_samples[:,:,i] = link_exp_to_gauss((eta[i,:].reshape(M,L))@cholH.T, pars=[1,1])[0]
    D_space_samples[:,:,i] = link_exp_to_gauss((delta[i,:].reshape(M,K))@cholD.T, pars=[1,1])[0]

































































    #_, grad = loglik(samples[i,:], sigma_N, X, link_rectgauss, link_exp_to_gauss, M, cholD, cholH)
    #gradnorms[i] = np.linalg.norm(grad, 'inf')
    #gradnorms[i] = np.amax(np.abs(grad))

"""
plt.plot(gradnorms)
plt.show()
maxdist = 0
dists = np.zeros(samples.shape[0])
for i in range(1,samples.shape[0]):
    cur_sample = samples[i,:]
    dist = euclidean(cur_sample, samples[0,:])
    if dist > maxdist:
        maxdist = dist
        print(maxdist)
    dists[i] = dist

plt.plot(dists)
plt.show()

H_space_samples_re = H_space_samples.reshape((eta.shape[0],M,nr_H // M))
D_space_samples_re = D_space_samples.reshape(delta.shape[0],M,nr_D // M)
H_try = H_space_samples_re[eta.shape[0] - 100,:,:]
D_try = D_space_samples_re[eta.shape[0] - 100,:,:]
plt.plot(H_space_samples_re[eta.shape[0] - 100, 1, :])
plt.show()
X_re = D_try.T@H_try

print(X.shape)
plt.matshow(X)
plt.axis('tight')
plt.show()

#plt.plot(H_space_samples[1400, 1, :])
#plt.show()
plt.matshow(H_space_samples)
plt.axis('tight')
plt.show()
#plt.plot(H_space_samples[7500, :, 0])
#plt.show()
#plt.matshow(D_space_samples)
#plt.axis('tight')
#plt.show()
#plt.plot(D_space_samples[7500, :])
#plt.show()

#plt.plot(H_space_samples[2500,:].reshape(50,2)[:,0], 'b-')
#plt.show()





# Mean and standard deviation
D_mean = np.mean(delta, axis=0).reshape(M, int(nr_D / M))
H_mean = np.mean(eta, axis=0).reshape(M, int(nr_H / M))

D_sd = np.std(delta, axis=0)
H_sd = np.std(eta, axis=0)
print(D_sd.shape)
D_sd = D_sd.reshape(M, int(nr_D / M))
H_sd = H_sd.reshape(M, int(nr_H / M))

# Confidence intervals in original domain
H_li = np.zeros(H_mean.shape)
H_ui = np.zeros(H_mean.shape)
H_m = np.zeros(H_mean.shape)
D_li = np.zeros(D_mean.shape)
D_ui = np.zeros(D_mean.shape)
D_m = np.zeros(D_mean.shape)

for i in range(M):
    H_li[i, :] = link_exp_to_gauss(H_mean[i, :] - 2 * H_sd[i, :], pars=[1, 1])[0]
    H_ui[i, :] = link_exp_to_gauss(H_mean[i, :] + 2 * H_sd[i, :], pars=[1, 1])[0]
    H_m[i, :] = link_exp_to_gauss(H_mean[i, :], pars=[1, 1])[0]
    D_m[i,:] = link_rectgauss(D_mean[i, :], pars=[1, 1])[0]



# True spectrum
spectra = a.T @ vp

# Plots
plt.figure()
plt.subplot(2,1,1)
plt.plot(spectra[0,:], color='green')
#plt.gca().axes.get_yaxis().set_visible(False)
plt.legend(["True Spectrum"], loc = 2)
plt.subplot(2,1,2)
#plt.fill_between(np.arange(len(H_mean[0,:])), H_li[0,:], H_ui[0,:],color='blue', alpha=.5)
plt.plot(D_m[1, :], color='blue')
#plt.gca().axes.get_yaxis().set_visible(False)
plt.legend(["Mean","95% Confidence Int."], loc = 2)
plt.show()


"""





