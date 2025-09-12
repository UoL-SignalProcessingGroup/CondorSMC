import autograd.numpy as np

from condorsmc.smcs.executor.sequential import SequentialExecutor


class RecyclingBase:
    """ Base class for recycling schemes.
    """

    def __init__(self, executor=SequentialExecutor()):
        self.executor = executor

    def calculate_constant(self, smc_state):
        """
        Description:
            Calculate the CESS recycling constant.

        Args:
            wn: Normalised importance weights.
            wn_old: Normalised importance weights from the previous iteration.

        Returns:
            ess_constant: Constant used in the ESS recycling scheme.
        """

        return smc_state

    def recycle_mean(self, smc_statistics):
        """
        Description:
            Recycle the mean estimates.

        Args:
            mean_estimates: Mean estimates.
            constants: Constants used in the ESS recycling scheme.

        Returns:
            recycled_mean: Recycled mean estimates.
        """

        recycled_estimates = np.zeros([smc_statistics.K, smc_statistics.dim])
        for k in range(smc_statistics.K):
            ld = np.zeros((k + 1,))
            _sum = np.sum(smc_statistics.recycling_constants[: k + 1])
            for i in range(k + 1):
                ld[i] = smc_statistics.recycling_constants[i] / _sum
            recycled_estimates[k] = ld @ smc_statistics.mean_estimate[: k + 1, :]

        smc_statistics.recycled_mean_estimate = recycled_estimates

        return smc_statistics

    def recycle_variance(self, smc_statistics):
        """
        Description:
            Recycle the variance estimates.

        Args:
            var_estimates: Variance estimates.
            mean_estimates: Mean estimates.
            recycled_mean: Recycled mean estimates.
            constants: Constants used in the ESS recycling scheme.

        Returns:
            recycled_var: Recycled variance estimates.
        """

        recycled_estimates = np.zeros([smc_statistics.K, smc_statistics.dim])
        for k in range(smc_statistics.K):
            ld = np.zeros((k + 1,))
            _sum = np.sum(smc_statistics.recycling_constants[: k + 1])
            correction = np.zeros([smc_statistics.K, smc_statistics.dim])
            for i in range(0, k + 1):
                ld[i] = smc_statistics.recycling_constants[i] / _sum
                correction[i] = np.square(smc_statistics.recycled_mean_estimate[k] - smc_statistics.mean_estimate[i])
            recycled_estimates[k] = ld @ (
                smc_statistics.variance_estimate[: k + 1, :] + correction[: k + 1, :]
            )

        smc_statistics.recycled_variance_estimate = recycled_estimates
        return smc_statistics
