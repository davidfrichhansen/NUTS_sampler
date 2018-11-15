import numpy as np
from scipy.special import erf, erfinv 
from numpy.matlib import repmat, repeat
from numba import jit

########################
#### LINK FUNCTIONS ####
########################

def link_exp_to_gauss(h, pars):
    sigma = pars[0]
    lamb = pars[1]  # inverse scale
    # actual inverse link value
    inner = .5 - .5*erf(h / (np.sqrt(2) * sigma)) + 1e-12
    val = np.maximum(-1 / lamb * np.log(inner), 0)
    val = np.nan_to_num(val)
    # elementwise derivative of inverse link
    grad = (np.sqrt(2 * np.pi) * sigma * lamb) ** (-1) * np.exp(lamb * val - h ** 2 / (2 * sigma ** 2))
    return val, grad


def link_rectgauss(h, pars):
    sigma = pars[0]  # diag of Sigma_h
    s = pars[1]  # "width" parameter
    # value of inverse link
    inner = .5 + .5 * erf(h / (np.sqrt(2) * sigma))
    val = np.sqrt(2) * s * erfinv(inner)

    # elementwise derivative of inverse link
    grad = s / (2 * sigma) * np.exp(val ** 2 / (2 * s ** 2) - h ** 2 / (2 * sigma ** 2))

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
#@jit(nopython=False)
def loglik(etadelta, sigma, X, linkD, linkH, M, cholD, cholH, *linkArgs):
    # linkD and linkH should be callable and return f^-1(h) and f^-1(d) resp.
    # furthermore it should return the derivative of these!
    # linkArgs is a nested list where the first list is a list of extra args to linkD
    # and second list is a list of extra args to linkD
    
    K, L = X.shape
    eta = etadelta[:M*L]
    delta = etadelta[M*L:]



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
    inner1 = ((D @ (X_re - X)) * H_prime.reshape(M, L))
    grad1 = sigma ** (-2) * (inner1 @ cholH).ravel() + eta
    inner2 = ((H @ (X_re - X).T) * D_prime.reshape(M, K))
    grad2 = sigma ** (-2) * (inner2 @ cholD).ravel() + delta
    grad = np.concatenate((grad1, grad2), axis=0)
    #print(grad.shape)
    return cost_val, grad


