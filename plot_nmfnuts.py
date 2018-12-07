import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from aux_funcs import link_rectgauss, link_exp_to_gauss, get_2d_exp_kernel, rbf




### LOAD DATA
mat = loadmat('./data/K1_25x25x50_2hot.mat')
a = mat['a']
vp = mat['vp']
X = mat['X']
loading = mat['gendata']['A'].ravel()[0].ravel()
print(X.shape)

K, L = X.shape

M = 2
nr_D = K*M
nr_H = L*M
print(nr_D)
print(nr_H)
sigma_N = 5


### LOAD SAMPLES
samples = np.load('npys/K1_2hot/K1_25x25x50_2hot_un_smooth.npy')
maps = np.load("npys/K1_2hot/K1_25x25x50_2hot_un_smooth_map.npz")
D_map = maps['D_map']
H_map = maps['H_map']
cholH = maps['cholH']
cholD = maps['cholD']
print("Sample shape")
print(samples.shape)

num_samples = samples.shape[0]

### SETUP COVARIANCES
dim = int(np.sqrt(len(X)))

print(samples.shape)
#eta = samples[:, :nr_H]
#delta = samples[:, nr_H:]
delta = samples[:,:M*K]
eta = samples[:,M*K:]
print("Shape eta")
print(eta.shape)
print("Shape delta")
print(delta.shape)

### TRANSFORM SAMPLES TO RIGHT SPACE
# get H with link_exp_to_gauss
H_samples = np.zeros((M,L,num_samples))
D_samples = np.zeros((M,K, num_samples))

for i in range(samples.shape[0]):
    H_samples[:,:,i] = link_exp_to_gauss((eta[i,:].reshape(M,L))@cholH.T, pars=[1,1])[0]
    D_samples[:,:,i] = link_exp_to_gauss((delta[i,:].reshape(M,K))@cholD.T, pars=[1,1])[0]


### PLOT MAP ESTIMATE AND TRUE SPECTRUM (ie. H)
plt.subplot(2,1,1)
plt.plot(H_map[1,:])
plt.title('First component of H MAP')

plt.subplot(2,1,2)
plt.plot((a.T@vp).ravel())
plt.title('True spectrum')

plt.tight_layout()

plt.show()


### PLOT HOTSPOT MAP AND TRUE HOTSPOT (ie. D)
plt.subplot(1,2,1)
plt.imshow(D_map[1,:].reshape(dim,dim))
plt.title("D MAP")

plt.subplot(1,2,2)
plt.imshow(loading.reshape(dim,dim))
plt.title('True loading')

plt.tight_layout()

plt.show()


### PLOT 4 EVENLY SPACED SAMPLES OF H WITH MAP ESTIMATE

H_plots = H_samples[:,:,-100::100 // 4]
legend_names = ['First','Second','Third','Fourth','Fifth']
colornames = ['g','r','c','m','k']
plt.subplot(1,2,1)
plt.title("Map estimate and one component")
# plot MAP
plt.plot(H_map[1,:], label='MAP')
plt.xlabel("Index")
plt.ylabel("Intensity")
for i in range(4):
    plt.plot(H_plots[0,:,i],c=colornames[i])
plt.legend()
plt.subplot(1,2,2)
plt.xlabel("Index")
plt.ylabel("Intensity")
plt.plot(H_map[0,:], label='MAP')
plt.title("Map estimate and one component")
for i in range(4):
    plt.plot(H_plots[1,:,i], c=colornames[i])
plt.legend()

plt.tight_layout()
plt.show()



### PLOT 4 EXAMPLES OF D WITH MAP ESTIMATE
D_plots = D_samples[:,:,-100::100//4]

ax1 = plt.subplot2grid((4,4),(1,0), rowspan=2,colspan=2)
plt.imshow(D_map[1,:].reshape(dim,dim))
plt.title('MAP estimate')
plt.axis('tight')


ax2 = plt.subplot2grid((4,4), (1,2))
plt.imshow(D_plots[0,:,0].reshape(dim,dim))
plt.axis('tight')

ax3 = plt.subplot2grid((4,4), (1,3))
plt.imshow(D_plots[0,:,1].reshape(dim,dim))
plt.axis('tight')

ax4 = plt.subplot2grid((4,4), (2,2))
plt.imshow(D_plots[0,:,2].reshape(dim,dim))
plt.axis('tight')

ax5 = plt.subplot2grid((4,4), (2,3))
plt.imshow(D_plots[0,:,3].reshape(dim,dim))
plt.axis('tight')

plt.tight_layout()
plt.show()

## Second component
ax1 = plt.subplot2grid((4,4),(1,0), rowspan=2,colspan=2)
plt.imshow(D_map[0,:].reshape(dim,dim))
plt.title('MAP estimate')
plt.axis('tight')


ax2 = plt.subplot2grid((4,4), (1,2))
plt.imshow(D_plots[1,:,0].reshape(dim,dim))
plt.axis('tight')

ax3 = plt.subplot2grid((4,4), (1,3))
plt.imshow(D_plots[1,:,1].reshape(dim,dim))
plt.axis('tight')

ax4 = plt.subplot2grid((4,4), (2,2))
plt.imshow(D_plots[1,:,2].reshape(dim,dim))
plt.axis('tight')

ax5 = plt.subplot2grid((4,4), (2,3))
plt.imshow(D_plots[1,:,3].reshape(dim,dim))
plt.axis('tight')

plt.tight_layout()
plt.show()