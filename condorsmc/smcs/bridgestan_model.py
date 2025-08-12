import os
import json
import bridgestan as bs
import autograd.numpy as np

from .target_base import TargetBase


class StanModel(TargetBase):
    def __init__(self, model_name, model_path, data_path=None):
        self.model_name = model_name
        self.model_path = model_path
        self.data_path = data_path

        self.stan_model = bs.StanModel.from_stan_file(model_path, data_path if data_path else None)

        self.dim = self.stan_model.param_unc_num()
        self.constrained_dim = self.stan_model.param_num(include_tp=True, include_gq=True)
        self.param_names = self.stan_model.param_names()

    def logpdf(self, upar, phi=1.0, adjust_transform=True):
        # If upar is a 1D array, calculate the logpdf of upar
        if upar.ndim == 1:
            try:
                return self.stan_model.log_density(upar)
            except Exception as e:
                # print(f"Inf logpdf: {upar}, {e}")
                return -np.inf
        else:
            N = upar.shape[0]
            p_logpdf_x_new = np.zeros(N)
            for i in range(N):
                try:
                    p_logpdf_x_new[i] = self.stan_model.log_density(upar[i])
                except Exception as e:
                    # print(f"Inf logpdf: {upar[i]}, {e}")
                    p_logpdf_x_new[i] = -np.inf
            return np.array(p_logpdf_x_new)
    
    def logpdfgrad(self, upar, phi=1.0, adjust_transform=True):
        if upar.ndim == 1:
            grad_x = np.zeros(self.dim)
            try:
                self.stan_model.log_density_gradient(upar, out=grad_x)
            except Exception as e:
                # print(f"Inf gradient: {upar}, {e}")
                grad_x = np.full(self.dim, -np.inf)
            return np.array(grad_x)
        else:
            N = upar.shape[0]
            p_logpdf_x_new = np.zeros((N, self.dim))
            for i in range(N):
                try:
                    self.stan_model.log_density_gradient(upar[i], out=p_logpdf_x_new[i])
                except Exception as e:
                    # print(f"Inf gradient: {upar[i]}, {e}")
                    p_logpdf_x_new[i] = np.full(self.dim, -np.inf)
            return np.array(p_logpdf_x_new)

    def constrain(self, upar, include_tparams=True, include_gqs=True):
        stan_rng = self.stan_model.new_rng(seed=0)
        if upar.ndim == 1:
            try:
                cpar = self.stan_model.param_constrain(upar, include_tp=include_tparams, include_gq=include_gqs, rng=stan_rng)
                return np.array(cpar)
            except Exception as e:
                # print(f"Inf constrain: {upar}, {e}")
                return np.full(self.constrained_dim, 0.0)
        else:
            constrained_samples = np.zeros([upar.shape[0], self.constrained_dim])
            for i in range(upar.shape[0]):
                try:
                    constrained_samples[i] = self.stan_model.param_constrain(upar[i], include_tp=include_tparams, include_gq=include_gqs, rng=stan_rng)
                except Exception as e:
                    # print(f"Inf constrain: {upar[i]}, {e}")
                    constrained_samples[i] = np.full(self.constrained_dim, 0.0)
            return np.array(constrained_samples)
