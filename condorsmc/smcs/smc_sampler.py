import autograd.numpy as np

from abc import ABC, abstractmethod
from tqdm import tqdm
from time import time

from .executor.sequential import SequentialExecutor
from .resampling.multinomial import MultinomialResampling
from .weight_updater import WeightUpdater
from .recycling.none import NoRecycling
from .importance_sampling import estimate_moments


import numpy as np

class SMCStatistics:
    def __init__(self, N, dim, executor):
        self.N = N
        self.K = 0
        self.dim = dim
        self.executor = executor
        self.mean_estimate = np.zeros((0, dim))
        self.recycled_mean_estimate = np.zeros((0, dim))
        self.variance_estimate = np.zeros((0, dim))
        self.recycled_variance_estimate = np.zeros((0, dim))
        self.log_normalising_constant = np.zeros(0)
        self.resampled = np.zeros(0, dtype=bool)
        self.ess = np.zeros(0)
        self.recycling_constants = np.zeros(0)
        self.acceptance_rate = np.zeros(0)
        self.run_time = None

    def update(self, smc_state):
        mean, variance = estimate_moments(smc_state, self.executor)

        self.mean_estimate = np.concatenate([self.mean_estimate, mean[np.newaxis, :]], axis=0)
        self.variance_estimate = np.concatenate([self.variance_estimate, variance[np.newaxis, :]], axis=0)
        self.ess = np.concatenate([self.ess, [smc_state.ess]])
        self.log_normalising_constant = np.concatenate([self.log_normalising_constant, [smc_state.log_normalising_constant]])
        self.recycling_constants = np.concatenate([self.recycling_constants, [smc_state.recycling_constant]])
        self.acceptance_rate = np.concatenate([self.acceptance_rate, [1.0]])
        self.resampled = np.concatenate([self.resampled, [False]])
        self.K += 1


class SMCSampler:
    def __init__(self, target, forward_kernel, lkernel, recycling=NoRecycling(), executor=SequentialExecutor(), rng=np.random.default_rng()):
        self.target = target  # Target distribution
        self.forward_kernel = forward_kernel  # Forward kernel distribution
        self.lkernel = lkernel  # L-kernel distribution
        self.recycling = recycling  # Recycling scheme
        self.executor = executor # Executor (Sequential, MPI)
        self.rng = rng

        self.resample_method = MultinomialResampling(executor=self.executor, rng=self.rng)
        self.weight_updater = WeightUpdater(self.target, self.forward_kernel, self.lkernel, self.executor)

    def sample(self, smc_state, smc_statistics, num_iters, record_states=False):
        start_time = time()

        iterator = range(num_iters)
        if self.executor.rank == 0:
            iterator = tqdm(iterator, desc="Sampling")

        smc_states = []
        for k in iterator:
            smc_state, smc_statistics = self.step(k, smc_statistics, smc_state)

            if record_states:
                smc_states.append(smc_state.copy())

        # Recycle mean and variance estimates
        smc_statistics = self.recycling.recycle_mean(smc_statistics)
        smc_statistics = self.recycling.recycle_variance(smc_statistics)

        smc_statistics.run_time = time() - start_time

        return smc_states if record_states else smc_state, smc_statistics

    def step(self, k, smc_statistics, smc_state):
        # Resample if needed
        smc_state = self.resample_method.resample_if_required(smc_state)

        # Propose new particles
        smc_state = self.forward_kernel.propose(smc_state)

        # Update weights
        smc_state = self.weight_updater.update(smc_state)

        # Calculate recycling constant
        smc_state = self.recycling.calculate_constant(smc_state)

        # Estimate moments and diagnostics
        smc_statistics.update(smc_state)

        return smc_state, smc_statistics
