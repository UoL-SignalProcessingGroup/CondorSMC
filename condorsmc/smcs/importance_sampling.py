import autograd.numpy as np

from .executor.sequential import SequentialExecutor
from .recycling.none import NoRecycling


def normalise_weights(log_weights, executor=SequentialExecutor()):
    """
    Description:
        Normalise the log weights to avoid numerical underflow.

    Args:
        log_weights: Unnormalised log weights.

    Returns:
        normalised_log_weights: Normalised log weights.
    """

    indices = np.isfinite(log_weights)
    log_normalising_constant = executor.logsumexp(log_weights[indices])
    normalised_weights = np.exp(log_weights - log_normalising_constant)

    return normalised_weights, log_normalising_constant


def estimate_moments(smc_state, executor=SequentialExecutor()):
    """
    Description:
        Importance sampling estimate of the mean and variance of the
        target distribution.

    Args:
        x: Particle positions.
        wn: Normalised importance weights.

    Returns:
        mean_estimate: Estimated mean of the target distribution.
        variance_estimate: Estimated variance of the target distribution.
    """

    if hasattr(smc_state.target, "constrain"):
        x_constrained = smc_state.target.constrain(smc_state.positions)
    else:
        x_constrained = smc_state.positions.copy()

    mean = executor.weighted_sum(x_constrained, smc_state.normalised_weights)
    var = executor.weighted_sum(np.square(x_constrained - mean), smc_state.normalised_weights)

    return mean, var


def estimate_moments_tempered(smc_states, smc_statistics, recycling=NoRecycling(), executor=SequentialExecutor()):
    """ Calculate adjusted weights, form estimates and recycle all past simulated samples.

    This function calculates the adjusted importance weights `ess_logw` for all samples. The
    weights are defined as \pi(x) / \pi(x, \phi_k) where \pi(x) is the target density and
    \pi(x, \phi_k) is the density of the kth proposal. The adjusted weights are then used to
    form estimates of the mean and variance of the target density, calculate recycling constants
    and recycle all past simulated samples. This scheme is outlined in [1].

    [1] Nguyen, T., Septier, F., Peters, G. and Delignon, Y. (2014). Improving
    SMC sampler estimate by recycling all past simulated samples.
    """

    for k, smc_state in enumerate(smc_states):
        # Using weights calculated in the the sampler draw a set a set of samples
        z = np.linspace(0, smc_state.num_particles - 1, smc_state.num_particles, dtype=int)
        z_new = np.random.choice(z, smc_state.num_particles, p=smc_state.normalised_weights)
        x = smc_state.positions[z_new]

        # Calculate importance weights and normalise
        smc_state.logw = smc_state.target.logpdf(smc_state.positions) - smc_state.target.logpdf(smc_state.positions, phi=smc_state.phi)
        smc_state.normalised_weights, _ = normalise_weights(smc_state.logw)

        # Calculate mean and variance estimates, as well as recycling coefficients
        smc_statistics.mean_estimate[k], smc_statistics.variance_estimate[k] = estimate_moments(smc_state, executor=executor)
        smc_statistics.recycling_constants[k] = recycling.calculate_constant(smc_state).recycling_constant

    # Recycle mean and variance estimates using recycling coefficients
    smc_statistics = recycling.recycle_mean(smc_statistics)
    smc_statistics = recycling.recycle_variance(smc_statistics)

    return smc_states, smc_statistics
