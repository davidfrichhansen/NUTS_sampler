import numpy as np
from scipy.special import erf, erfinv
from scipy.spatial.distance import euclidean
from numpy.matlib import repmat, repeat
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from tqdm import tqdm

########################
#### LINK FUNCTIONS ####
########################


def forward_exp_to_gauss(fh, pars):
    sigma = pars[0]
    lamb = pars[1]
    inner = -2*np.exp(-lamb*fh) + 1
    ei = erfinv(inner)
    h = np.sqrt(2)*sigma*ei
    return h

def forward_rectgauss(fh, pars):
    sigma = pars[0]
    s = pars[1]

    inner = 2 * erf(fh / (np.sqrt(2)*s) - 1)
    h = np.sqrt(2) * sigma * erfinv(inner)

    return h


def link_exp_to_gauss(h, pars):
    sigma = pars[0]
    lamb = pars[1]  # inverse scale
    # actual inverse link value
    inner = 1e-12 + .5 - .5*erf(h / (np.sqrt(2) * sigma))
    #print(inner)
    val = np.maximum((-1.0 / lamb) * np.log(inner), 0)
    val = np.nan_to_num(val)
    #print(np.sum(np.isnan(val)))
    # elementwise derivative of inverse link
    #grad = (np.sqrt(2 * np.pi) * sigma * lamb) ** (-1) * np.exp(lamb * val - h ** 2 / (2 * sigma ** 2))
    #tmp = 1e-12 + np.exp(-(h**2) / (2*sigma**2))
    #grad = 1.0 / (sigma*lamb*np.sqrt(2*np.pi)) * (tmp /inner)
    grad = 1 / (np.sqrt(2*np.pi) * sigma * lamb) * np.exp(lamb*val - h*h / (2*sigma*sigma))
    return val, grad


def link_rectgauss(h, pars):
    sigma = pars[0]  # diag of Sigma_h
    s = pars[1]  # "width" parameter
    # value of inverse link
    inner = .5 + .5 * erf(h / (np.sqrt(2) * sigma))
    val = np.sqrt(2) * s * erfinv(inner)

    # elementwise derivative of inverse link
    grad = (s / (2 * sigma)) * np.exp(val ** 2 / (2 * s * s) - h ** 2 / (2 * sigma ** 2))

    return val, grad


########################
######## kernels #######
########################

def get_2d_exp_kernel(beta, shape_plate):
    dummy = np.ones(shape_plate)
    dummy = np.argwhere(dummy)
    distances = calcDistanceMatrixFastEuclidean(dummy)
    final = np.exp((-distances) / (beta ** 2))

    return final

def rbf(beta, i, j):
    return np.exp(- ((i - j) ** 2 / (beta ** 2)))


def calcDistanceMatrixFastEuclidean(points):
    """
    Just a memory efficient way to calculate euclidian distances

    http://code.activestate.com/recipes/498246-calculate-the-distance-matrix-for-n-dimensional-po/

    :param points: List of coordinates
    :return: Distance matrix
    """
    numPoints = len(points)
    distMat = np.sqrt(np.sum((repmat(points, numPoints, 1) - repeat(points, numPoints, axis=0))**2, axis=1))
    return distMat.reshape((numPoints,numPoints))

########################
#### Log likelihood ####
########################
def loglik(etadelta, X, M, linkD, linkH, cholD, cholH, sigma):
    K,L = X.shape

    #eta = etadelta[:M*L]
    #delta = etadelta[M*L:]
    #print(delta.shape)
    delta = etadelta[:M*K]
    eta = etadelta[M*K:]
    D_args = (1,1)

    D, D_prime = linkD((delta.reshape(M, K) @ cholD.T), D_args)

    H_args = (1,1)

    H, H_prime = linkH((eta.reshape(M, L) @ cholH.T), H_args)

    # cost itself - eq. (22)
    first = sigma ** (-2) * np.sum(np.sum((X - D.T @ H) * (X - D.T @ H)))
    second = (delta.T @ delta) + (eta.T @ eta)
    cost_val = .5 * (first + second)

    X_re = D.T @ H
    # gradient of cost - eq. (23)
    inner1 = ((D @ (X_re - X)) * H_prime.reshape(M, L))
    grad1 = sigma ** (-2) * (inner1 @ cholH).ravel() + eta
    inner2 = ((H @ (X_re - X).T) * D_prime.reshape(M, K))
    grad2 = sigma ** (-2) * (inner2 @ cholD).ravel() + delta
    grad = -np.concatenate((grad2,grad1))
    return -cost_val, grad




def nmf_gpp_map(X, M, **kwargs):
    """
    Computes the MAP estimate of Non-negative matrix factorization with Gaussian Process Priors.
    :param X:       Data matrix, K x L
    :param M:       Number of factors.
    :param kwargs:  Options:
                    'MaxIter'   : Iterations for optimization
                    'covH'      : Covariance of GPP determining H, M*L x M*L
                    'covD'      : Covariance of GPP determining D, K*M x K*M
                    'linkH'     : Link function for H, callable. Inverse link
                    'argsH'     : Extra arguments for linkH. Should be a list.
                    'linkD'     : Link function for D, callable. Inverse link
                    'argsD'     : Extra arguments for linkD. Should be a list.
                    'sigma_N'   : Variance of Gaussian noise.

    :return D,H:    NN Matrix Factors
    """
    # parse arguments
    try:
        maxIter = kwargs['MaxIter']
        covH = kwargs['covH']
        covD = kwargs['covD']
        linkH = kwargs['linkH']
        argsH = kwargs['argsH']
        linkD = kwargs['linkD']
        argsD = kwargs['argsD']
        sigma_N = kwargs['sigma_N']
    except KeyError:
        print("Missing parameter(s). All parameters must be passed.")
        return None
    K, L = X.shape
    # initial value of delta, eta
    delta = np.random.randn(M*K)
    eta = np.random.randn(M*L)

    # compute D and H from delta and eta
    def cost(opti, other, sigma, X, ret_eta, linkD, linkH, M, cholD, cholH, *linkArgs):
        # linkD and linkH should be callable and return f^-1(h) and f^-1(d) resp.
        # furthermore it should return the derivative of these!
        # linkArgs is a nested list where the first list is a list of extra args to linkD
        # and second list is a list of extra args to linkD
        K, L = X.shape
        if ret_eta:
            eta = opti
            delta = other
        else:
            delta = opti
            eta = other
        # convert 'greeks' to matrices D and H - eq. (17)
        D_args = linkArgs[0]
        # link_rectgauss
        D, D_prime = linkD((delta.reshape(M, K) @ cholD.T), D_args)

        H_args = linkArgs[1]
        # link_exp_to_gauss
        H, H_prime = linkH((eta.reshape(M, L) @ cholH.T), H_args)

        # cost itself - eq. (22)
        first = sigma ** (-2) * np.sum(np.sum((X - D.T @ H) * (X - D.T @ H)))
        second = (delta.T @ delta) + (eta.T @ eta)
        cost_val = .5 * (first + second)

        X_re = D.T @ H
        # gradient of cost - eq. (23)
        if ret_eta:
            inner = ((D @ (X_re - X)) * H_prime.reshape(M, L))
            grad = sigma ** (-2) * (inner @ cholH).ravel() + eta
        else:
            inner = ((H @ (X_re - X).T) * D_prime.reshape(M, K))
            grad = sigma ** (-2) * (inner @ cholD).ravel() + delta
        return cost_val, grad

    # iteratively optimize over eta and delta
    cholD = np.linalg.cholesky(covD)
    cholH = np.linalg.cholesky(covH)

    for _ in tqdm(range(maxIter)):

        # optimize delta
        delta_old = delta
        args = (eta, sigma_N, X, False, linkD, linkH, M, cholD, cholH, argsD, argsH)
        delta = minimize(cost, delta_old, args, 'L-BFGS-B', jac=True).x


        # optimize eta
        eta_old = eta
        args = (delta, sigma_N, X, True, linkD, linkH, M, cholD, cholH, argsD, argsH)
        eta_min = minimize(cost, eta_old, args, 'L-BFGS-B', jac=True)
        eta = eta_min.x

    print(eta_min.fun)
    #  convert to D and H
    D = linkD(delta.reshape(M, K) @ cholD.T, argsD)[0]
    H = linkH(eta.reshape(M, L) @ cholH.T, argsH)[0]

    return D, H, delta, eta




