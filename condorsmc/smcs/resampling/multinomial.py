import warnings
import autograd.numpy as np

from .base import ResamplingBase
from ..state import SMCState
from ..executor.sequential import SequentialExecutor


class MultinomialResampling(ResamplingBase):
    def _resample(self, smc_state):
        indices = np.linspace(0, smc_state.num_particles - 1, smc_state.num_particles, dtype=int)
        resampled_indices = self.rng.choice(indices, size=smc_state.num_particles, replace=True, p=smc_state.normalised_weights)

        smc_state.positions = smc_state.positions[resampled_indices]
        smc_state.momenta = smc_state.momenta[resampled_indices]
        smc_state.log_pdfs = smc_state.log_pdfs[resampled_indices]
        smc_state.log_pdf_grads = smc_state.log_pdf_grads[resampled_indices]
        smc_state.log_weights = np.log(np.ones(smc_state.num_particles_local)) - np.log(smc_state.num_particles)
        smc_state.log_normalising_constant = 1.0
        smc_state.normalised_weights = np.full(smc_state.num_particles_local, 1 / smc_state.num_particles)

        return smc_state
