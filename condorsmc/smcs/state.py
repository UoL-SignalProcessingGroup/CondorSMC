import copy
import autograd.numpy as np

from scipy.special import logsumexp
from .executor.sequential import SequentialExecutor
from .importance_sampling import estimate_moments


class SMCState:
    """Class to store the particles and their associated quantities."""

    def __init__(self, target, num_particles, executor=SequentialExecutor()):
        self.target = target
        self.num_particles = num_particles
        self.num_iters = 0
        self.executor = executor
        self.num_particles_local = self.num_particles // self.executor.num_procs

        if self.executor.num_procs > 1:
            assert self.num_particles & (self.num_particles - 1) == 0, "Number of samples must be a power of 2."

        # Current state
        self.k = 0
        self.phi = 1.0
        self.recycling_constant = 1.0
        self.positions = np.empty((self.num_particles_local, target.dim))
        self.momenta = np.empty((self.num_particles_local, target.dim))
        self.log_pdfs = np.empty(self.num_particles_local)
        self.log_pdf_grads = np.empty((self.num_particles_local, target.dim))
        self.log_weights = np.empty(self.num_particles_local)
        self.log_normalising_constant = np.empty(1)
        self.normalised_weights = np.empty(self.num_particles_local)

        # Old state
        self.phi_old = 1.0
        self.positions_old = np.empty((self.num_particles_local, target.dim))
        self.momenta_old = np.empty((self.num_particles_local, target.dim))
        self.log_pdfs_old = np.empty(self.num_particles_local)
        self.log_pdf_grads_old = np.empty((self.num_particles_local, target.dim))
        self.normalised_weights_old = np.empty(self.num_particles_local)

        # Adaption information
        self.acceptance_rate = np.zeros(num_particles)

    @property
    def ess(self):
        """Effective Sample Size (ESS) computed from normalised weights."""
        return 1 / self.executor.reduce(np.sum(np.square(self.normalised_weights)))

    def update_samples(self, positions, momenta):
        self.positions_old = self.positions
        self.positions = positions
        self.momenta_old = self.momenta
        self.momenta = momenta
        self.log_pdfs_old = self.log_pdfs
        self.log_pdfs = self.target.logpdf(positions, phi=self.phi)
        self.log_pdf_grads_old = self.log_pdf_grads
        self.log_pdf_grads = self.target.logpdfgrad(positions, phi=self.phi)

    def update_weights(self, log_weights, log_normalising_constant, normalised_weights):
        self.log_weights = log_weights
        self.log_normalising_constant = log_normalising_constant
        self.normalised_weights_old = self.normalised_weights
        self.normalised_weights = normalised_weights

    def copy(self):
        return copy.copy(self)
