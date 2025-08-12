import numpy as np  # type: ignore
from autograd.scipy import stats as AutoStats  # type: ignore


class ForwardLKernel:
    """Forward Kernel L Kernel

    The forward kernel approximation of the optimal L-kernel is
    presented in [1].

    [1] Devlin, L., Horridge, P., Green, P. and Maskell, S (2021). The
    No-U-Turn Sampler as a Proposal Distribution in a Sequential Monte
    Carlo Sampler with a Near-Optimal L Kernel.

    Attributes:
        N: Number of particles
        D: Dimensionality of the target distribution
    """

    def __init__(self, dim: int):
        self.dim = dim

    def calculate_rw(self, x, x_new):
        """
        Description:
            Calculate the Forward Kernel approximation of the optimal L-kernel
            for a random walk proposal.

        Args:
            x: Current particle positions.
            x_new: New particle positions.

        Returns:
            log_pdf: The forward kernel approximation of the optimal L-kernel.
        """

        raise Exception("Forward Kernel L-Kernel not implemented for random walk")

    def calculate_hmc(self, x, x_new, v, v_new):
        """
        Description:
            Calculate the Forward Kernel approximation of the optimal L-kernel
            for a Hamiltonian Monte Carlo (HMC) proposal.

        Args:
            x: Current particle positions.
            x_new: New particle positions.
            v: Current particle velocities.
            v_new: New particle velocities.

        Returns:
            log_pdf: The forward kernel approximation of the optimal L-kernel.
        """

        return AutoStats.multivariate_normal.logpdf(
            -v_new, mean=np.zeros(self.dim), cov=np.eye(self.dim)
        )
