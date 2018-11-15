import numpy as np
from nuts_goodgrad import NUTS
import aux_funcs as fs
from scipy.special import erf, erfinv
from scipy.io import loadmat
import time
import matplotlib.pyplot as plt


if __name__ == "__main__":
    show_cov = 0
    t0 = time.time()
    print("Loading data")
    # load data
    """
    file_name1 = '50x50_nw250_noise12_nhs2_k4_21'
    mat1 = loadmat('./data/'+file_name1+'.mat')
    X1 = mat1["X"]

    file_name2 = '50x50_nw250_noise12_nhs2_k4_22'
    mat2 = loadmat('./data/'+file_name2+'.mat')
    X2 = mat2["X"]
    print("Done after %.2f s!" % (time.time() - t0))
    # additive spectra
    X = X1 + X2
    # delete objects from memory
    del mat1, mat2
    """
    mat = loadmat('./data/sampling.mat')
    X = mat['X']
    K, L = X.shape
    print("Shape of data is (%d, %d)" % (K,L))
    
    # setup covariances
    print("\n\nSetting up covariances")
    M = 2
    beta_H = 2.5
    beta_D = 2.5
    sigma_N = 5
    
    dim = int(np.sqrt(len(X)))
    cov_D_2d = fs.get_2d_exp_kernel(beta_D, (dim,dim))
    cov_D_2d = cov_D_2d + 1e-5 * np.eye(K)

    cov_H = np.zeros((L, L))
    for i in range(L):
        for j in range(L):
            cov_H[i, j] = fs.rbf(beta_H, i + 1, j + 1)

    cov_H = cov_H + 1e-5 * np.eye(L)
    
    print("Done after %.2f s!\n\n" % (time.time() - t0))
    if show_cov:
        print("Plotting covariance of D\n")
        plt.imshow(cov_D_2d)
        plt.show()

    print("Doing additional setup")
    loglik = fs.loglik
    delta = np.random.randn(M*K)
    eta = np.random.randn(M*L)
    etadelta = np.concatenate((eta, delta), axis=0)
    cholD = np.linalg.cholesky(cov_D_2d)
    cholH = np.linalg.cholesky(cov_H)
    # 'wrapper' - may be inefficient
    def loglik_w(etadelta):
        return list(map(lambda x : -x, loglik(etadelta, sigma_N, X, fs.link_rectgauss, fs.link_exp_to_gauss,M,cholD, cholH,
                      (np.diag(cov_D_2d), 1), (np.diag(cov_H), 1))))
                      
    print("Done with additional setup after %f s!\n" % (time.time() - t0))
    
    
    num_samples = 5000
    num_burnin = int(0.2*num_samples)
    #num_burnin = 4900
    print("Setting up sampler wíth %d samples and %d adaptive\n" % (num_samples, num_burnin))
    sampler = NUTS(loglik_w, num_samples, num_burnin, etadelta, delta=0.8)
    print("Beginning sampling")
    sampler.sample()
    print("Done sampling after %f s!\n" % (time.time() - t0))
    out_filename = input("Please input filename for saving\n")
    out_path = out_filename
    print("\nSaving into %s" % out_path)
    np.save(out_filename, sampler.samples)
    print("Finished in %f s!" % (time.time() - t0))