import numpy as np

from scipy.special import logsumexp

from .base import ExecutorBase


class SequentialExecutor(ExecutorBase):
    def max(self, x):
        return np.max(x)

    def sum(self, x):
        return np.sum(x)

    def weighted_sum(self, x, w):
        return w.T @ x

    def gather(self, x):
        return x

    def reduce(self, x):
        return x

    def bcast(self, x):
        return x

    def logsumexp(self, x):
        return logsumexp(x)

    def cumsum(self, x):
        return np.cumsum(x)

    def weighted_mean_covar(self, X, w):
        mu = np.average(X, axis=0, weights=w)
        cov = np.cov(X, rowvar=False, aweights=w)

        return mu, cov
    
    def barrier(self):
        pass
