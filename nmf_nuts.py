import numpy as np
from nuts_goodgrad import NUTS
import aux_funcs as fs
from scipy.special import erf, erfinv
from scipy.io import loadmat


if __name__ == "__main__":
    # load data
    mats = loadmat("data/sampling.mat")
    # data matrix
    X = mats['X']
    
    

