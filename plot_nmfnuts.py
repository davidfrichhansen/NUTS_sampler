import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from aux_funcs import link_rectgauss, link_exp_to_gauss

# dimensions
nr_D = 1250
nr_H = 100
M = 2


mat = loadmat('./data/newdata.mat')
a = mat['a']
vp = mat['vp']
X = mat['X']
print(X.shape)

# traces
samples = np.load('pls.npy')

print(samples.shape)
eta = samples[:, :nr_H]
delta = samples[:, nr_H:]
print("Shape eta")
print(eta.shape)
print("Shape delta")
print(delta.shape)

# get H with link_exp_to_gauss
H_space_samples = np.zeros_like(eta)
D_space_samples = np.zeros_like(delta)
for i in range(samples.shape[0]):
    H_space_samples[i,:] = link_exp_to_gauss(eta[i,:], pars=[np.ones(100),1])[0]
    D_space_samples[i,:] = link_rectgauss(delta[i,:], pars=[1,1])[0]
plt.plot(delta[2400,:])
plt.show()
#print(D_space_samples.shape)
#orig_space = np.concatenate((H_space_samples, D_space_samples), axis=0)

#H_space_samples = H_space_samples.reshape((8000,50,2))
plt.matshow(H_space_samples)
plt.axis('tight')
plt.show()
#plt.plot(H_space_samples[7500, :, 0])
#plt.show()
plt.matshow(D_space_samples)
plt.axis('tight')
plt.show()
plt.plot(D_space_samples[7500, :])
plt.show()

#plt.plot(H_space_samples[2500,:].reshape(50,2)[:,0], 'b-')
#plt.show()




"""
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
    H_li[i, :] = link_exp_to_gauss(H_mean[i, :] - 2 * H_sd[i, :], pars=[np.ones(50), 1])[0]
    H_ui[i, :] = link_exp_to_gauss(H_mean[i, :] + 2 * H_sd[i, :], pars=[np.ones(50), 1])[0]
    H_m[i, :] = link_exp_to_gauss(H_mean[i, :], pars=[np.ones(50), 1])[0]
    D_m[i,:] = link_rectgauss(D_mean[i, :], pars=[np.ones(400), 1])[0]



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
plt.plot(H_m[0, :], color='blue')
#plt.gca().axes.get_yaxis().set_visible(False)
plt.legend(["Mean","95% Confidence Int."], loc = 2)
plt.show()

"""






