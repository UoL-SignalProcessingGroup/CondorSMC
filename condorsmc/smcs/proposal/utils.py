import autograd.numpy as np
from scipy.stats import multivariate_normal


def hmc_accept_reject(smc_state, rng=np.random.default_rng()):
    # Calculate the Hamiltonian
    H1 = smc_state.target.logpdf(smc_state.positions, phi=smc_state.phi) - (0.5 * np.einsum('ij,ij->i', smc_state.momenta, smc_state.momenta))
    H0 = smc_state.target.logpdf(smc_state.positions_old, phi=smc_state.phi_old) - (0.5 * np.einsum('ij,ij->i', smc_state.momenta_old, smc_state.momenta_old))

    # Calculate the acceptanace rate and probability
    acceptance_ratio = np.exp(H1 - H0)
    acceptance_probability = np.minimum(1., acceptance_ratio)

    # Accept or reject the proposed samples
    accepted = rng.uniform(size=smc_state.num_particles_local) <  acceptance_probability

    return accepted
