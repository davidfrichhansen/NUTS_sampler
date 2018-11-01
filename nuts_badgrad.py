import numpy as np
import tqdm



class NUTS:
    """
    Implements the efficient NUTS sampler with dual averaging (algorithm 6) from Hoffman and Gelman (2011)
    """
    def __init__(self, logp, M, M_adapt, theta0, delta=0.63, debug=False, delta_max=1000.0):
        """
        Initialize
        :param logp:        Callable that takes parameters of target distribution and return log probability and
                            the gradient in that point in the form
                            logprob, grad = f(theta)
        :param M:           Number of desired samples
        :param M_adapt:     Amount of samples in which the dual averaging for estimating step size, espilon, should be run
        :param theta0:      Initial parameter value
        :param delta:       Desired acceptance rate for the sampler, used in the dual averaging
        :param debug:       Boolean determining verbosity - for debugging purposes
        :param delta_max:   Parameter determining precision. Default: 1000 as recommended by Hoffman and Gelman
        """
        self.logp = logp
        self.M = M
        self.M_adapt = M_adapt
        self.theta0 = theta0
        self.delta = delta
        self.debug = debug
        self.delta_max = delta_max

        # maybe do some smart stuff, pickling, sqlite etc.
        self.samples = np.zeros((M, len(theta0)))
        self.samples[0,:] = theta0

    def leapfrog(self, theta, r, epsilon):
        """
        Leapfrog integration
        :param theta:   initial parameter value
        :param r:       momentum parameter
        :param epsilon: step length
        :param f:       function returning log p and gradient
        """
        #if self.debug:
        #    print("Enter leapfrog")
        f = self.logp
        _, grad = f(theta)
        r_bar = r + 0.5*epsilon*grad
        theta_bar = theta + epsilon*r_bar
        # recompute gradient
        _, grad_new = f(theta_bar)
        r_bar = r_bar + 0.5*epsilon*grad_new

        #if self.debug:
        #    print("Leave leapfrog")
        return theta_bar, r_bar

    def epsilon_heuristic(self, theta):
        """
        Heuristic for finding initial value of step length epsilon
        :param theta:   Initial parameter value
        :return:        Reasonable value for initial epsilon
        """
        if self.debug:
            print("Enter reasonable epsilon")
        dim = len(theta)
        epsilon = 1
        r = np.random.multivariate_normal(np.zeros((dim, )), np.eye(dim))
        f = self.logp
        # initial leapfrog
        theta_prime, r_prime = self.leapfrog(theta, r, epsilon)
        old_logp, _ = f(theta)
        new_logp, _ = f(theta_prime)
        logp_ratio = new_logp - old_logp - 0.5*r_prime.T@r_prime + 0.5*r.T@r

        criteria = logp_ratio > np.log(.5)

        a = 1.0 if criteria else -1.0

        while a*logp_ratio > -a*np.log(2.0):
            epsilon *= 2.0**a
            theta_prime, r_prime = self.leapfrog(theta, r, epsilon)
            new_logp, _ = f(theta_prime)
            logp_ratio = new_logp - old_logp - 0.5*r_prime.T@r_prime + 0.5*r.T@r

            # shouldn't be updated every iteration -
            # solid hour spent debugging this... :(
            # criteria = logp_ratio > np.log(.5)
            # a = 1.0 if criteria else -1.0

        # debugging
        if self.debug:
            print("Find reasonable epsilon: %.4lf\n" % epsilon)

        return epsilon

    def build_tree(self, theta, r, log_u, v, j, epsilon, theta0, r0):
        """
        Implicitly builds balanced search tree recursively of visited (theta, r)-positions
        This is one of the main enhancements from regular HMC
        :param theta:       Latest parameter value
        :param r:           Latest momentum value
        :param u:           Slice variable
        :param v:           Direction (-1 or 1)
        :param j:           height of tree
        :param epsilon:     step length
        :param r0:          Initial momentum value
        :return:
        """
        #theta0 = self.theta0
        f = self.logp
        delta_max = self.delta_max
        old_logp, _ = f(theta0)

        if self.debug:
            print("Recursion, j = %d" % j)

        # base case - tree height is 0
        if j == 0:
            #print("Base case")
            theta_prime, r_prime = self.leapfrog(theta, r, v*epsilon)
            new_logp, _ = f(theta_prime)

            #n_prime = 1 if log_u < new_logp - 0.5 * r_prime.T @ r_prime else 0
            n_prime = int(log_u <= (new_logp - 0.5 * r_prime.T @ r_prime))
            #print(n_prime)
            s_prime = 1 if log_u < delta_max + new_logp - 0.5 * r_prime.T @ r_prime else 0

            return (theta_prime, r_prime, theta_prime, r_prime, theta_prime, n_prime, s_prime,
                    min(1, np.exp(new_logp - 0.5*r_prime.T@r_prime - old_logp + 0.5 * r0.T @ r0)),
                    1)
        else:
            # main recursion
            (theta_m, r_m, theta_p, r_p,
            theta_prime, n_prime, s_prime, alpha_prime, n_alpha) = self.build_tree(theta, r, log_u, v, j-1, epsilon, theta0, r0)
            #print(s_prime)
            if s_prime == 1:
                if v == -1:
                    (theta_m, r_m, _, _, theta_pp, n_pp,
                     s_pp, alpha_pp, n_alphapp) = self.build_tree(theta_m, r_m, log_u, v, j-1, epsilon, theta0, r0)
                else:
                    (_,_, theta_p, r_p, theta_pp, n_pp,
                     s_pp, alpha_pp, n_alphapp) = self.build_tree(theta_p, r_p, log_u, v, j-1, epsilon,theta0,  r0)

                # draw bernoulli sample
                #print("npp %d" % n_pp)
                #bern = np.random.binomial(1, n_pp / max((n_prime + n_pp), 1))
                try:
                    bern = np.random.rand() < n_pp / (n_prime + n_pp)
                except ZeroDivisionError:
                    bern = False
                #print(n_pp / max((n_prime + n_pp), 1))
                #bern = np.random.binomial(1, n_pp / (n_prime + n_pp))
                if bern:
                    theta_prime = theta_pp
                alpha_prime += alpha_pp
                n_alpha += n_alphapp

                s_prime = s_pp if ((theta_p - theta_m) @ r_m >= 0 and (theta_p - theta_m) @ r_p >= 0) else 0

                if self.debug:
                    print("s' = %d" % s_prime)

                n_prime += n_pp

            return theta_m, r_m, theta_p, r_p, theta_prime, n_prime, s_prime, alpha_prime, n_alpha

    def sample(self, kappa=0.75, t0=10):
        """
        Run the NUTS sampling algorithm with dual averaging
        :return:
        """
        f = self.logp
        M = self.M
        M_adapt = self.M_adapt
        theta0 = self.theta0

        epsilon = self.epsilon_heuristic(theta0)
        mu = np.log(10*epsilon)
        logeps_bar = 0
        H_bar = 0
        gamma = 0.05
        for m in tqdm.trange(1,M, unit_scale=True, desc="Sample"):
            # tqdm.trange is a specialised instance of range that is optimised for progess bar output
            r0 = np.random.multivariate_normal(np.zeros_like(theta0), np.eye(len(theta0)))
            logp, _ = f(self.samples[m-1, :])
            joint = logp - 0.5*r0.T@r0
            #u = np.random.uniform(0, np.exp(logp - 0.5*r0.T@r0))
            log_u = joint - np.random.exponential(1, size=1)
            # initialize
            theta_m = self.samples[m-1,:]
            theta_p = self.samples[m-1,:]
            r_m = r0
            r_p = r0
            j = 0
            # proposed value of parameters
            theta_prop = self.samples[m-1,:]
            n = s = 1

            while s:
                # choose direction uniformly
                v = np.random.choice([-1,1])
                if v == -1:
                    (theta_m, r_m, _, _, theta_prime, n_prime,
                     s_prime, alpha, n_alpha) = self.build_tree(theta_m, r_m, log_u, v, j, epsilon, self.samples[m-1,:], r0)
                else:
                    (_, _, theta_p, r_p, theta_prime, n_prime,
                     s_prime, alpha, n_alpha) = self.build_tree(theta_p, r_p, log_u, v, j, epsilon,
                                                                self.samples[m-1,:], r0)
                if s_prime:
                    # accept sample
                    bern = np.random.binomial(1, min(1, n_prime / n))
                    if bern:
                        theta_prop = theta_prime

                n += n_prime
                s = s_prime if (theta_p - theta_m) @ r_m >= 0 and (theta_p - theta_m) @ r_p >= 0 else 0
                j = j + 1

            # dual averaging
            if m <= M_adapt:
                H_bar = (1 - 1.0 / (m + t0))*H_bar + 1 / (m+t0) * (self.delta - alpha / n_alpha)
                #print(n_alpha)
                logeps = mu - np.sqrt(m) / gamma * H_bar
                epsilon = np.exp(logeps)
                logeps_bar = m**(-kappa) * logeps + (1-m**(-kappa))*logeps_bar
            else:
                epsilon = np.exp(logeps_bar)

            if self.debug:
                print(epsilon)
                print("Epsilon for m = %d is %.4lf" % (m, epsilon[m]))


            # add proposal to sample list
            self.samples[m, :] = theta_prop
        print("Sampling finished!")

















