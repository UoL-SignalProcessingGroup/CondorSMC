from typing import Tuple

import autograd.numpy as np  # type: ignore


class SMCBase:
    def __init__(
        self,
        K: int,
        N: int,
        dim: int,
        target,
        forward_kernel,
        sample_proposal,
        lkernel: None,
        recycling: None,
        verbose: bool = False,
        seed: int = 0,
    ):
        self.K = K  # Number of iterations
        self.N = N  # Number of particles
        self.dim = dim  # Dimensionality of the target
        self.target = target  # Target distribution
        self.forward_kernel = forward_kernel  # Forward kernel distribution
        self.sample_proposal = sample_proposal  # Initial sample proposal distribution
        self.lkernel = lkernel  # L-kernel distribution
        self.recycling = recycling  # Recycling scheme
        self.verbose = verbose  # Show stdout
        self.seed = seed  # Random seed

        # Hold etimated quantities and diagnostic metrics
        if hasattr(self.target, "constrained_dim"):
            self.mean_estimate = np.zeros([self.K, self.target.constrained_dim])
            self.recycled_mean_estimate = np.zeros([self.K, self.target.constrained_dim])
            self.variance_estimate = np.zeros([self.K, self.target.constrained_dim])
            self.recycled_variance_estimate = np.zeros([self.K, self.target.constrained_dim])
        else:
            self.mean_estimate = np.zeros([self.K, self.target.dim])
            self.recycled_mean_estimate = np.zeros([self.K, self.target.dim])
            self.variance_estimate = np.zeros([self.K, self.target.dim])
            self.recycled_variance_estimate = np.zeros([self.K, self.target.dim])
        self.ess = np.zeros(self.K)
        self.log_likelihood = np.zeros(self.K)
        self.recycling_constant = np.zeros(self.K)
