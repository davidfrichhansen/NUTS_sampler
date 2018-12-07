# NUTS_sampler

This is a working repo for a special course in MCMC for latent factor models  

We implement the No-U-Turn Sampler (NUTS) and use it for Non-negative Matrix Factorization with Gaussian Process Priors (NMF-GPP)

 ```nuts.py``` contains the implementation of NUTS as a class. It accepts the following parameters
  * Log probability, ```logp``` such that ```log_prob, grad = logp(theta)``` - ie. it should return the log-probability of the parameter value and the gradient of the log probability evaluated in that point.
  * ```M```, the number of samples
  * ```M_adapt```, the number of samples to be used as burn in (dual averaging)
  * ```theta0```, initial value of parameters
  * ```delta```, desired acceptance rate
  * ```debug```, debugging flag (not up to date)
  * ```delta_max```, parameter for stop criterion
  
 It has the method ```sample``` that handles the sampling and every method needed to run NUTS.
   
 ```nuts_test.py``` runs a simple 2D Gaussian example with known precision and plots the relative difference between the true covariance and the estimated covariance as a function of the number of samples drawn. 
 The experiment is run 10 times for each amount of samples and the plotted result is the average over those runs.
 
 ```aux_funcs.py``` contains functions relevant to run the NMF-GPP model with NUTS such as Link functions, functions setting up covariances and log-posterior.
 
 ```nmf_nuts.py``` runs NUTS for NMF-GPP. It loads in the data as a ```.mat``` file and runs NUTS with dual averaging. It prompts the user for a filename and then saves the samples in a ```.npy```-file. Everything is setup in this script, so it is here change should be made for different problems.  
 The sampler always starts in the MAP estimate, which is achieved through direct optimization of the log-posterior
 
 ```plot_nmfnuts.py``` evaluates samples from NUTS by comparing them to the MAP estimate and plotting 4 evenly spaced samples from the last 100 iterations alongside the MAP estimate. The MAP estimate is plotted beside the ground truth.
  