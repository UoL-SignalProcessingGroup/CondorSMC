from condorsmc.smcs.recycling.base import RecyclingBase


class NoRecycling(RecyclingBase):
    def calculate_constant(self, smc_state):
        return smc_state

    def recycle_mean(self, smc_statistics):
        return smc_statistics

    def recycle_variance(self, smc_statistics):
        return smc_statistics
