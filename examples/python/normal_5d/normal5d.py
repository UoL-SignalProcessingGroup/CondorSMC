import autograd.numpy as np  # type: ignore
from autograd import elementwise_grad as egrad  # type: ignore
from autograd.scipy import stats as AutoStats  # type: ignore
from condorsmc.smcs.target_base import TargetBase


class Target(TargetBase):
    def __init__(self, data={}):
        self.dim = 5
        self.mean = np.array([-4, 2, 0, 2, 4])
        self.cov = np.eye(5)

    def logpdf(self, x, phi=1.0):
        return AutoStats.multivariate_normal.logpdf(x, mean=self.mean, cov=self.cov)

    def logpdfgrad(self, x, phi=1.0):
        grad = egrad(self.logpdf)
        return grad(x)
