import numpy as np
from nuts_goodgrad import NUTS
import aux_funcs as fs
from scipy.io import loadmat
import time
import matplotlib.pyplot as plt
#np.seterr('raise')


if __name__ == "__main__":
    show_cov = 1
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
    mat = loadmat('./data/newdata.mat')
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
    print(dim)
    cov_D_2d = fs.get_2d_exp_kernel(beta_D, (dim, dim))
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
        plt.imshow(cov_H)
        plt.show()

    print("Doing additional setup")
    loglik = fs.loglik

    #eta = np.random.randn(M*L)
    #delta = np.random.randn(M * K)

    ### Starting in right solution
    a = mat['a'].ravel()
    vp = mat['vp'].ravel()
    true_spec = (a.T*vp).ravel()
    #true_spec = vp
    H = np.zeros((M,L))
    h0 = true_spec
    H[0,:] = h0
    #H[1,:] = np.random.randn(L)
    h1 = np.random.multivariate_normal(np.zeros(L), cov_H)
    #plt.plot(h1)
    #plt.show()
    H[1,:,] = fs.link_rectgauss(h1, (1,1))[0].reshape(L,)
    #plt.plot(H[1,:])
    #plt.show()

    plt.plot(H.reshape(M*L))
    plt.show()
    #H[H < 0] = 1e-12

    gendata = mat['gendata']
    true_load = gendata['A'][0][0].ravel()
    D = np.zeros((M,K))
    D[0,:] = true_load

    #D[1,:] = np.zeros((K,)) + 1e-12
    #D[1,:] = np.random.randn(K)
    #D[D < 0] = 1e-12
    d1 = np.random.multivariate_normal(np.zeros(K), cov_D_2d) / sigma_N
    #plt.plot(d1)
    #plt.show()
    D[1,:] = fs.link_exp_to_gauss(d1, (1,1))[0].reshape(K)
    plt.plot(D.reshape(M*K))
    plt.show()
    plt.subplot(1,2,1)
    plt.imshow(D.T@H)
    plt.axis('tight')
    plt.subplot(1,2,2)
    plt.imshow(X)
    plt.axis('tight')
    plt.tight_layout()
    plt.show()

    #D_reshape = D.reshape((M*K, ))
    #H_reshape = H.reshape((M*L, ))

    cholD = np.linalg.cholesky(cov_D_2d)
    cholH = np.linalg.cholesky(cov_H)

    eta = (fs.forward_exp_to_gauss(H, (1,1))@np.linalg.inv(cholH.T)).reshape(M*L)
    delta = (fs.forward_exp_to_gauss(D,(1,1))@np.linalg.inv(cholD.T)).reshape(M*K)
    plt.plot(eta)
    plt.title("Initial eta")
    plt.show()
    plt.plot(delta)
    plt.title("inital delta")
    plt.show()


    #etadelta = np.concatenate((eta, delta), axis=0)

    # 'wrapper' - may be inefficient
    #### This may be wrong. Something with sign and step size determination doesn't converge!!!
    def loglik_w(etadelta):
        return loglik(etadelta, X, M, fs.link_exp_to_gauss, fs.link_exp_to_gauss, cholD, cholH, sigma_N)


    print("Done with additional setup after %f s!\n" % (time.time() - t0))
    #print("Initial loglik: %f" %loglik_w(etadelta)[0])






































"""







    num_samples = 3000
    #num_burnin = int(0.2*num_samples)
    num_burnin = 1500
    print("Setting up sampler wíth %d samples and %d adaptive\n" % (num_samples, num_burnin))
    sampler = NUTS(loglik_w, num_samples, num_burnin, etadelta, delta=0.6)
    print("Beginning sampling")
    sampler.sample()
    print("Done sampling after %f s!\n" % (time.time() - t0))
    #out_filename = input("Please input filename for saving\n")
    #out_path = out_filename

    out_path = input("Enter filename:")
    print("\nSaving into %s" % out_path)
    np.save(out_path, sampler.samples)
    np.save(out_path + "logp_trace", sampler.logparr)
    print("Finished in %f s!" % (time.time() - t0))
"""