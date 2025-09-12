import numpy as np

from condorsmc.smcs.recycling.base import RecyclingBase


class ESSRecycling(RecyclingBase):
    """Effective Sample Size Recycling

    Recycle samples generated at all iterations of the SMC sampler using
    the effective sample size (ESS) recycling scheme. See [1] for details.

    [1] Nguyen, T., Septier, F., Peters, G. and Delignon, Y. (2014). Improving
    SMC sampler estimate by recycling all past simulated samples.
    """

    def calculate_constant(self, smc_state):
        """
        Description:
            Calculate the ESS recycling constant.

        Args:
            wn: Normalised importance weights.

        Returns:
            ess_constant: Constant used in the ESS recycling scheme.
        """

        smc_state.recycling_constant = (
            1 / self.executor.reduce(
                np.sum(np.square(smc_state.normalised_weights))
            )
        )

        return smc_state