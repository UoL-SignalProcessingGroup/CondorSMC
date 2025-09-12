import autograd.numpy as np

from ..state import SMCState
from ..executor.sequential import SequentialExecutor


class ResamplingBase:
    def __init__(self, resampling_threshold=0.5, executor=SequentialExecutor(), rng=np.random.default_rng()):
        self.resampling_threshold = resampling_threshold
        self.executor = executor
        self.rng = rng

    def resample_if_required(self, smc_state) -> SMCState:
        """Resample if the ESS is below a certain threshold."""
        if smc_state.ess < 0.5 * smc_state.num_particles:
            return self._resample(smc_state)

        return smc_state

    def _resample(self, smc_state):
        pass