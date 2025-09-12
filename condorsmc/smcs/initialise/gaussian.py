import autograd.numpy as np

from scipy.stats import multivariate_normal

from .base import InitialiserBase
from ..state import SMCState
from ..importance_sampling import normalise_weights


class GaussianInitialiser(InitialiserBase):
    def init(self, num_particles) -> SMCState:
        smc_state = SMCState(self.target, num_particles, self.executor)
        smc_state.positions = multivariate_normal.rvs(mean=np.zeros(self.target.dim), cov=np.eye(self.target.dim), size=smc_state.num_particles_local, random_state=self.rng)
        smc_state.momenta = np.zeros_like(smc_state.positions)
        smc_state.log_pdfs = self.target.logpdf(smc_state.positions)
        smc_state.log_pdf_grads = self.target.logpdfgrad(smc_state.positions)
        smc_state.log_weights = smc_state.log_pdfs - multivariate_normal.logpdf(smc_state.positions, mean=np.zeros(self.target.dim), cov=np.eye(self.target.dim))
        smc_state.normalised_weights, smc_state.log_normalising_constant = normalise_weights(smc_state.log_weights, self.executor)

        return smc_state
