import autograd.numpy as np

from .executor.sequential import SequentialExecutor
from .importance_sampling import normalise_weights
from .lkernel.forward_lkernel import ForwardLKernel


class WeightUpdater:
    def __init__(self, target, proposal, lkernel, executor=SequentialExecutor()):
        self.target = target
        self.proposal = proposal
        self.lkernel = lkernel
        self.executor = executor

    def update(self, smc_state):
        lkernel_lpdf = self.lkernel.calculate_hmc(smc_state)
        proposal_lpdf = self.proposal.logpdf(smc_state)

        new_log_weights = (
            smc_state.log_weights
            + smc_state.log_pdfs
            - smc_state.log_pdfs_old
            + lkernel_lpdf
            - proposal_lpdf
        )

        normalised_weights, log_normalising_constant = normalise_weights(new_log_weights, self.executor)
        smc_state.update_weights(new_log_weights, log_normalising_constant, normalised_weights)

        return smc_state
