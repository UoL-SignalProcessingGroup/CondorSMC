import autograd.numpy as np  # type: ignore
from scipy.stats import multivariate_normal  # type: ignore


class RandomWalkProposal:
    """Random Walk Proposal

    Propagate samples using a Gaussian random walk proposal.

    Attributes:
        D: Dimensionality of the system.
        target: Target distribution of interest.
    """

    def __init__(
        self,
        dim: int,
        cov: float = 1.0,
    ):
        self.dim = dim  # Dimensionality of the target
        self.cov = cov  # Covariance of the Gaussian noise

        cov_matrix = np.multiply(np.eye(self.dim), self.cov)
        self.dist = multivariate_normal(np.zeros(self.dim), cov_matrix)

    def rvs(self, x_cond, N: int):
        """
        Description:
            Propogate a set of samples using Gaussian random walk.

        Args:
            x_cond: Current particle positions.
            N: Number of particles to sample.

        Returns:
            x_prime: Updated particle positions.
        """

        return x_cond + self.dist.rvs(N)

    def logpdf(self, x, x_new):
        """
        Description:
            Calculate the log probability of the forward kernel.

        Args:
            x: Current particle positions.
            x_new: New particle positions.

        Returns:
            log_prob: Log probability of the forward kernel.
        """

        return self.dist.lpdf(x_new - x)
