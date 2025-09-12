import os
import json
import warnings
import bridgestan as bs
import autograd.numpy as np

from .base import TargetBase
from ..executor.sequential import SequentialExecutor


def load_ground_truth(ground_truth_path):
    true_mean = []
    true_var = []
    with open(ground_truth_path, "r") as f:
        for line in f:
            mean = line.split()[1]
            variance = line.split()[2]
            true_mean.append(float(mean))
            true_var.append(float(variance))
    
    return np.array(true_mean), np.array(true_var)


class StanModel(TargetBase):
    def __init__(self, model_name, model_path, data_path=None, executor=SequentialExecutor()):
        self.model_name = model_name
        self.model_path = model_path
        self.data_path = data_path
        self.executor = executor
        self.stan_model = bs.StanModel.from_stan_file(model_path, data_path if data_path else None)
        self.dim = self.stan_model.param_unc_num()
        self.constrained_dim = self.stan_model.param_num(include_tp=True, include_gq=True)
        self.param_names = self.stan_model.param_names(include_tp=True, include_gq=True)
        self.last_phi = 1.0

    def logpdf(self, x, phi=1.0, propto=True, jacobian=True):
        """
        Calculate the log density of the target distribution at x

        Args:
            x: Unconstrained parameter values to evaluate the log density at
            phi: The temperature of the target distribution
            propto: Whether to drop constant terms from the log density
            jacobian: Whether to include terms for constrained parameters in the log density

        
        Returns:
            The log density of the target distribution at x (or an array of log densities if x is a 2D array of samples)
        """

        if x.ndim == 1:
            try:
                return self.stan_model.log_density(x, propto=propto, jacobian=jacobian)
            except:
                return -np.inf
        else:
            N = x.shape[0]
            p_logpdf_x_new = np.zeros(N)
            for i in range(N):
                try:
                    p_logpdf_x_new[i] = self.stan_model.log_density(x[i], propto=propto, jacobian=jacobian)
                except:
                    p_logpdf_x_new[i] = -np.inf

            return np.array(p_logpdf_x_new)
    
    def logpdfgrad(self, x, phi=1.0, propto=True, jacobian=True):
        """
        Calculate the gradient of the log density of the target distribution at x

        Args:
            x: Unconstrained parameter values to evaluate the gradient of the log density at
            phi: The temperature of the target distribution
            propto: Whether to drop constant terms from the log density
            jacobian: Whether to include terms for constrained parameters in the log density

        Returns:
            The gradient of the log density of the target distribution at x (or a 2D array of gradients if x is a 2D array of samples)
        """

        if x.ndim == 1:
            grad_x = np.zeros(self.dim)
            try:
                self.stan_model.log_density_gradient(x, out=grad_x, propto=propto, jacobian=jacobian)
            except:
                grad_x = np.full(self.dim, -np.inf)

            return np.array(grad_x)
        else:
            N = x.shape[0]
            p_logpdf_x_new = np.zeros((N, self.dim))
            for i in range(N):
                try:
                    self.stan_model.log_density_gradient(x[i], out=p_logpdf_x_new[i], propto=propto, jacobian=jacobian)
                except:
                    p_logpdf_x_new[i] = np.full(self.dim, -np.inf)

            return np.array(p_logpdf_x_new)

    def constrain(self, x, include_tparams=True, include_gqs=True):
        """
        Constrain the parameters to the support of the target distribution

        Args:
            x: Unconstrained parameter values to constrain
            include_tparams: Whether to include the transformed parameters
            include_gqs: Whether to include the generated quantities

        Returns:
            The constrained parameter values (or a 2D array of constrained parameters if x is a 2D array of samples)
        """
        
        stan_rng = self.stan_model.new_rng(seed=0)
        if x.ndim == 1:
            try:
                constrained_sample = self.stan_model.param_constrain(x, include_tp=include_tparams, include_gq=include_gqs, rng=stan_rng)
                return np.array(constrained_sample)
            except:
                return np.full(self.constrained_dim, 0.0)
        else:
            constrained_samples = np.zeros([x.shape[0], self.constrained_dim])
            for i in range(x.shape[0]):
                try:
                    constrained_samples[i] = self.stan_model.param_constrain(x[i], include_tp=include_tparams, include_gq=include_gqs, rng=stan_rng)
                except:
                    constrained_samples[i] = np.full(self.constrained_dim, 0.0)
            return np.array(constrained_samples)
