import numpy as np
from nuts import NUTS
import aux_funcs as fs
from scipy.io import loadmat, savemat
import time
import matplotlib.pyplot as plt


if __name__ == "__main__":
    show_cov = 0        # plot covariances
    t0 = time.time()    # start timer
    print("Loading data")
    # load data
    mat = loadmat('./data/K2_25x25x50_2hot.mat')
    X = mat['X']
    K, L = X.shape
    print("Shape of data is (%d, %d)" % (K,L))
    
    # setup covariances
    print("\n\nSetting up covariances")
    M = 3
    beta_H = 2.5*3
    beta_D = 2.5*3

    # Variance of noise
    sigma_N = 5

    # size of Raman map - assumed square
    dim = int(np.sqrt(len(X)))
    print(dim)
    # 2D Exponential kernel on D
    cov_D_2d = fs.get_2d_exp_kernel(beta_D, (dim, dim))
    cov_D_2d = cov_D_2d + 1e-5 * np.eye(K)

    # regular exponential squared kernel on H
    cov_H = np.zeros((L, L))
    for i in range(L):
        for j in range(L):
            cov_H[i, j] = fs.rbf(beta_H, i + 1, j + 1)

    cov_H = cov_H + 1e-5 * np.eye(L)
    
    print("Done after %.2f s!\n\n" % (time.time() - t0))
    if show_cov:
        plt.imshow(cov_D_2d)
        plt.show()
        plt.imshow(cov_H)
        plt.show()

    print("Doing additional setup")
    # setup likelihood
    logprob = fs.logprob

    #eta = np.random.randn(M*L)
    #delta = np.random.randn(M * K)

    # For rotation of d and h to delta and eta
    cholD = np.linalg.cholesky(cov_D_2d)
    cholH = np.linalg.cholesky(cov_H)

    # compute MAP estimate for starting point
    D_map,H_map,delta,eta = fs.nmf_gpp_map(X, M, MaxIter=1000, covH=cov_H, covD=cov_D_2d, linkH=fs.link_exp_to_gauss,
                                           argsH=[1,1], linkD=fs.link_exp_to_gauss, argsD=[1,1], sigma_N=sigma_N)
    # show MAP estimate
    X_re = D_map.T@H_map
    plt.subplot(1,2,1)
    plt.imshow(X_re)
    plt.axis('tight')
    plt.subplot(1,2,2)
    plt.imshow(X)
    plt.axis('tight')
    plt.show()

    # Starting point after rotation and link functions
    plt.subplot(2,1,1)
    plt.plot(eta)
    plt.title('Initial eta')
    plt.subplot(2,1,2)
    plt.title('Initial delta')
    plt.plot(delta)

    plt.show()
    etadelta = np.concatenate((delta,eta))


    # 'wrapper' - may be inefficient
    #### This may be wrong. Something with sign and step size determination doesn't converge!!!
    def logprob_w(etadelta):
        return logprob(etadelta, X, M, fs.link_exp_to_gauss, fs.link_exp_to_gauss, cholD, cholH, sigma_N)


    print("Done with additional setup after %f s!\n" % (time.time() - t0))
    init_logp, init_grad = logprob_w(etadelta)
    print("Initial loglik: %f" % init_logp)
    plt.plot(init_grad)
    plt.show()
    out_filename = input("Please input filename for saving\n")
    ### Sampling part
    num_samples = 3000
    num_burnin = num_samples - 1500
    # setup sampler
    sampler = NUTS(logprob_w, num_samples, num_burnin, etadelta, delta=0.65)
    # do sampling
    sampler.sample()

    out_path = out_filename
    print("\nSaving into %s" % out_path)
    np.save(out_path, sampler.samples)
    np.save(out_path + "logp_trace", sampler.logparr)
    np.savez(out_path+'_map', D_map=D_map, H_map=H_map, cholH=cholH, cholD=cholD)
    print("Finished in %f s!" % (time.time() - t0))

