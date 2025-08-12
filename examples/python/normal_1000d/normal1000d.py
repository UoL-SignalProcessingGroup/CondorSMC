import autograd.numpy as np  # type: ignore
from autograd import elementwise_grad as egrad  # type: ignore
from autograd.scipy import stats as AutoStats  # type: ignore
from condorsmc.smcs.target_base import TargetBase


class Target(TargetBase):
    def __init__(self, data={}):
        self.dim = 1000
        self.mean = np.ones(self.dim) * 8
        self.cov = np.eye(self.dim)

    def logpdf(self, x, phi=1.0):
        return AutoStats.multivariate_normal.logpdf(x, mean=self.mean, cov=self.cov)

    def logpdfgrad(self, x, phi=1.0):
        grad = egrad(self.logpdf)
        return grad(x)
