import autograd.numpy as np

from .base import LKernelBase


class ForwardLKernel(LKernelBase):
    """Forward Kernel L Kernel

    The forward kernel approximation of the optimal L-kernel is
    presented in [1].

    [1] Devlin, L., Carter, M. Horridge, P., Green, P. and Maskell, S (2024). The 
    No-U-Turn Sampler as a Proposal Distribution in a Sequential Monte Carlo Sampler
    Without Accept/Reject. IEEE Signal Processing Letters, Vol. 31.

    Attributes:
        forward_kernel: The forward kernel approximation of the optimal L-kernel.
    """

    def __init__(self, forward_kernel):
        self.forward_kernel = forward_kernel

    def calculate_rw(self, smc_state):
        """
        Description:
            Calculate the Forward Kernel approximation of the optimal L-kernel
            for a random walk proposal.

        Args:
            smc_state: The current state of the SMC sampler.

        Returns:
            log_pdf: The forward kernel approximation of the optimal L-kernel.
        """

        raise NotImplementedError("Random walk proposal not implemented for Forward Kernel L Kernel")

    def calculate_hmc(self, smc_state):
        """
        Description:
            Calculate the Forward Kernel approximation of the optimal L-kernel
            for a Hamiltonian Monte Carlo (HMC) proposal.

        Args:
            smc_state: The current state of the SMC sampler.

        Returns:
            log_pdf: The forward kernel approximation of the optimal L-kernel.
        """

        return self.forward_kernel.momentum_proposal.logpdf(np.multiply(-1, smc_state.momenta))
