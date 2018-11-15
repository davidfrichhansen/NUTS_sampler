import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from aux_funcs import link_rectgauss, link_exp_to_gauss

# dimensions
nr_D = 800
nr_H = 100
M = 2


mat = loadmat('./data/sampling.mat')
a = mat['a']
vp = mat['vp']

# traces
samples = np.load('inverse_sign5k.npy')

eta = samples[:, :nr_H]
delta = samples[:, nr_H:]



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

for i in range(M):
    H_li[i, :] = link_exp_to_gauss(H_mean[i, :] - 2 * H_sd[i, :], pars=[1, 1])[0].flatten()
    H_ui[i, :] = link_exp_to_gauss(H_mean[i, :] + 2 * H_sd[i, :], pars=[1, 1])[0].flatten()
    H_m[i, :] = link_exp_to_gauss(H_mean[i, :], pars=[1, 1])[0].flatten()

# True spectrum
spectra = a.T @ vp

# Plots
plt.figure()
plt.subplot(2,1,1)
plt.plot(spectra[0,:], color='green')
#plt.gca().axes.get_yaxis().set_visible(False)
plt.legend(["True Spectrum"], loc = 2)
plt.subplot(2,1,2)
plt.fill_between(np.arange(len(H_mean[1,:])), H_li[1,:], H_ui[1,:],color='blue', alpha=.5)
plt.plot(H_m[1, :], color='blue')
#plt.gca().axes.get_yaxis().set_visible(False)
plt.legend(["Mean","95% Confidence Int."], loc = 2)
plt.show()








