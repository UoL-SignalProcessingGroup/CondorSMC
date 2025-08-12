import logging
from time import time
from tqdm import tqdm

import autograd.numpy as np  # type: ignore

from . import importance_sampling
from .smc_base import SMCBase


class HMCSMCSampler(SMCBase):
    """Hamiltonian Monte Carlo SMC Sampler

    An SMC sampler that uses Hamiltonian Monte Carlo (HMC) methods to sample
    from the target distribution of interest.

    Attributes:
        K: Number of iterations.
        N: Number of particles.
        D: Dimensionality of the target distribution.
        target: Target distribution of interest.
        forward_kernel: Forward kernel used to propagate samples.
        sample_proposal: Distribution to draw the initial samples from (q0).
        lkernel: Approximation method for the optimum L-kernel.
        recycling: Particle recycling method.
    """

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
        SMCBase.__init__(
            self,
            K,
            N,
            dim,
            target,
            forward_kernel,
            sample_proposal,
            lkernel,
            recycling,
            verbose,
            seed,
        )

    def sample(self, prof=None, save_samples=False):
        """
        Sample from the target distribution using an SMC sampler.
        """

        logging.info(f"Propagating {self.N} samples for {self.K} iterations")
        start_time = time()

        # Tensors to hold samples drawn at each iteration
        if save_samples:
            self.x_saved = np.zeros([self.K, self.N, self.dim])
            self.logw_saved = np.zeros([self.K, self.N])

        # Draw initial samples from the sample proposal distribution
        x = self.sample_proposal.rvs(self.N)
        x_new = np.zeros([self.N, self.dim])

        # Tensors to hold velocities, gradients and number of leapfrog steps
        v_new = np.zeros([self.N, self.dim])
        grad_x = np.zeros([self.N, self.dim])
        num_steps = np.zeros(self.N)

        # Calculate the initial weights
        logw = np.zeros(self.N)
        logw_new = np.zeros(self.N)
        p_logpdf_x = self.target.logpdf(x)
        q0_logpdf_x = self.sample_proposal.logpdf(x)
        logw = p_logpdf_x - q0_logpdf_x

        # Main sampling loop
        for k in tqdm(range(self.K)):
            logging.info(f"Iteration {k+1} of {self.K}")

            # Normalise importance weights and calculate the log likelihood
            wn, self.log_likelihood[k] = importance_sampling.normalise_weights(logw)

            if self.recycling:
                # Calculate the recycling constant
                self.recycling_constant[k] = self.recycling.constant(logw, wn)

            # Estimate the mean and variance of the target distribution
            (
                self.mean_estimate[k],
                self.variance_estimate[k],
            ) = importance_sampling.estimate(x, wn, self.target)

            # Calculate the effective sample size and resample if necessary
            self.ess[k] = importance_sampling.calculate_ess(wn)
            if self.ess[k] < self.N / 2:
                x, logw = importance_sampling.resample(x, wn, self.log_likelihood[k])

            # Propogate particles through the forward kernel
            v = self.forward_kernel.momentum_proposal.rvs(self.N)

            grad_x = self.target.logpdfgrad(x)
            x_new, v_new, num_steps = self.forward_kernel.rvs(x, v, grad_x)

            # Evaluate the target distribution, l kernel and forward kernel
            p_logpdf_x = self.target.logpdf(x)
            p_logpdf_xnew = self.target.logpdf(x_new)
            if self.lkernel:
                lkernel_logpdf = self.lkernel.calculate_hmc(x, x_new, v, v_new)
                q_logpdf = self.forward_kernel.logpdf(v)

            # Calculate new weights
            if self.lkernel:
                logw_new = logw + p_logpdf_xnew - p_logpdf_x + lkernel_logpdf - q_logpdf
            else:
                logw_new = logw + p_logpdf_xnew - p_logpdf_x

            # Update x and logw
            x = x_new.copy()
            logw = logw_new.copy()

            if save_samples:
                self.x_saved[k] = x_new.copy()
                self.logw_saved[k] = logw_new.copy()

        if self.recycling:
            # Recycle the mean and variance estimates
            self.recycled_mean_estimate = self.recycling.recycle_mean(
                self.mean_estimate, self.recycling_constant
            )
            self.recycled_variance_estimate = self.recycling.recycle_variance(
                self.variance_estimate,
                self.mean_estimate,
                self.recycled_mean_estimate,
                self.recycling_constant,
            )

        time_taken = time() - start_time
        logging.info(f"Finished sampling in {time_taken:5f} seconds")
