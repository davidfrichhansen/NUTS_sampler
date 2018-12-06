# NUTS_sampler

This is a working repo for a special course in MCMC for latent factor models  

We implement the No-U-Turn Sampler (NUTS) and use it for Non-negative Matrix Factorization with Gaussian Process Priors (NMF-GPP)

 ```nuts.py``` contains the implementation of NUTS. It accepts the following parameters
  * Log probability, ```logp``` such that ```log_prob, grad = logp(theta)``` - ie. it should return the log-probability of the parameter value and the gradient of the log probability evaluated in that point.
  * ```M```, the number of samples
  * ```M_adapt```, the number of samples to be used as burn in (dual averaging)
  * ```theta0```, initial value of parameters
  * ```delta```, desired acceptance rate
  * ```debug```, debugging flag (not up to date)
  * ```delta_max```, parameter for stop criterion
  
 It has the method ```sample``` that handles the sampling